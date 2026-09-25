/*
 * WinMoE Inference Engine — Full Token Generation
 *
 * Generates tokens from Qwen3.5 MoE models using:
 * - Custom GGUF parser (no mmap)
 * - Explicit unbuffered I/O for expert weights
 * - Q4_K / Q5_K dequant kernels with AVX2 + OpenMP
 * - GQA attention with KV cache
 * - Within-layer expert streaming
 *
 * Usage: winmoe.exe --model <gguf> --prompt "Hello" --tokens 50
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../platform/runtime.h"
#include <immintrin.h>
#include <float.h>

/* Quant tables and kernels */
#include "iq2xxs_grid_def.h"
#include "iq2_dequant.h"
#include "iq2_dequant_fast.h"
#include "q8k_quant.h"
#include "q4k_dequant.h"
#include "q5k_dequant.h"
#include "q6k_dequant.h"

/* ---- Layer-bisect trace dump (matches llama-trace-dump TSV format) ---- */
static FILE* g_trace_out = NULL;
static int   g_trace_tok = -1;

/* ---- Parity sprint: full-logit binary dump per position ---- */
static FILE* g_all_logits_bin = NULL;
static int   g_all_logits_rows = 0;

/* §1.8 diagnosis: WINMOE_LM_SCALAR=1 forces scalar q6k_dot_block in q6k_matvec
 * (LM head + any other q6k matvecs) so we can isolate AVX2 path as fault. */
int g_q6k_use_scalar = 0;

/* §1.8 diagnosis: WINMOE_DUMP_NORMED=path writes per-position normed[H=4096]
 * binary so we can compare element-wise vs llama's result_norm. */
static FILE* g_normed_bin = NULL;
static int   g_normed_rows = 0;
static int   g_normed_dim = 0;

/* Phase 4.1: WINMOE_EXPERT_TRACE=path enables a (layer × expert) access counter
 * dumped at exit. Used to identify the hot/cold expert split for selective IQ2
 * cold-tier quantization. Bounded at 60 layers × 512 experts = 30720 slots. */
#define EXPERT_TRACE_MAX_LAYERS 64
#define EXPERT_TRACE_MAX_EXPERTS 1024
static int g_expert_hits[EXPERT_TRACE_MAX_LAYERS * EXPERT_TRACE_MAX_EXPERTS] = {0};
static int g_expert_trace_enabled = 0;
static const char* g_expert_trace_path = NULL;

/* Phase 4.3: WINMOE_MOE_SKIP_EVERY=N skips the routed-expert MoE FFN on every
 * Nth MoE layer (i.e., layer % N == N-1). N=2 skips half, N=3 skips a third.
 * The shared expert still runs (cheap; ~1/4 of routed expert cost). Routing
 * still runs (cheap; ~25 ms/token). Routed expert IO + compute are skipped.
 * Quality impact must be verified via Phase 1 parity gate before promoting. */
static int g_moe_skip_every = 0;

static void trace_dump(const char* name, int layer, int tok, const float* buf, int n) {
    if (!g_trace_out || tok != g_trace_tok || !buf || n <= 0) return;
    double sum = 0.0, sqsum = 0.0, mn = 1e30, mx = -1e30;
    /* Per-head rms for h=0..7 (if n is divisible by 128 = DN_HEAD_DIM) */
    double head_ss[8] = {0};
    int head_n[8] = {0};
    for (int i = 0; i < n; i++) {
        double v = (double)buf[i];
        sum += v; sqsum += v*v;
        if (v < mn) mn = v; if (v > mx) mx = v;
        int h = i / 128;
        if (h < 8) { head_ss[h] += v*v; head_n[h]++; }
    }
    double rms  = sqrt(sqsum / (double)n);
    double mean = sum / (double)n;
    /* NODE name-L type ne n rms mean min max first10 */
    fprintf(g_trace_out, "NODE\t%s-%d\tf32\t%d,1,1,1\t%d\t%.8g\t%.8g\t%.8g\t%.8g\t",
            name, layer, n, n, rms, mean, mn, mx);
    int show = n < 10 ? n : 10;
    for (int k = 0; k < show; k++) fprintf(g_trace_out, "%s%.8g", k == 0 ? "" : ",", (double)buf[k]);
    /* Append per-head rms (h=0..7) when tensor is per-head-shaped (multiple of 128) */
    if (n % 128 == 0 && n >= 1024) {
        fprintf(g_trace_out, "\tHEAD_RMS:");
        for (int h = 0; h < 8; h++) {
            double hr = head_n[h] > 0 ? sqrt(head_ss[h] / head_n[h]) : 0;
            fprintf(g_trace_out, "%s%.6g", h == 0 ? "" : ",", hr);
        }
        /* Also sample 4 elements from head 1 (offset 128) and head 10 (offset 1280) */
        if (n >= 1408) {
            fprintf(g_trace_out, "\tH1_OFF128:%.8g,%.8g,%.8g,%.8g\tH10_OFF1280:%.8g,%.8g,%.8g,%.8g",
                (double)buf[128], (double)buf[129], (double)buf[130], (double)buf[131],
                (double)buf[1280], (double)buf[1281], (double)buf[1282], (double)buf[1283]);
            /* Additional samples at offsets 256, 512, 2048, 4000, 7000 to check layout */
            if (n >= 7100) {
                fprintf(g_trace_out, "\tOFF256:%.6g,%.6g,%.6g\tOFF512:%.6g,%.6g,%.6g\tOFF2048:%.6g,%.6g,%.6g\tOFF4000:%.6g,%.6g,%.6g\tOFF7000:%.6g,%.6g,%.6g",
                    (double)buf[256],(double)buf[257],(double)buf[258],
                    (double)buf[512],(double)buf[513],(double)buf[514],
                    (double)buf[2048],(double)buf[2049],(double)buf[2050],
                    (double)buf[4000],(double)buf[4001],(double)buf[4002],
                    (double)buf[7000],(double)buf[7001],(double)buf[7002]);
            }
        }
    }
    fprintf(g_trace_out, "\n");
    fflush(g_trace_out);
}
static void trace_dump_scalar(const char* name, int layer, int tok, float v) {
    if (!g_trace_out || tok != g_trace_tok) return;
    fprintf(g_trace_out, "NODE\t%s-%d\tf32\t1,1,1,1\t1\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\n",
            name, layer, (double)v, (double)v, (double)v, (double)v, (double)v);
    fflush(g_trace_out);
}
/* --------------------------------------------------------------------- */
#include "q8_dequant.h"
#include "gguf_parser.h"
#include "transformer.h"
#include "deltanet_impl.h"
#include "gpu_offload.h"

#define MAX_SEQ 512
#define ALIGN 65536

/* Keep OS storage semantics outside inference scheduling. */
#ifdef _WIN32
#include "runtime_io_win32.h"
#else
#include "runtime_io_posix.h"
#endif

/* Undo GGUF +1 norm weight convention: stored as (weight + 1), subtract 1 */
static void undo_norm_plus1(float* w, int n) {
    for (int i = 0; i < n; i++) w[i] -= 1.0f;
}

/* Load tensor from correct shard using persistent handles */
static void* load_tensor_data(GGUFModel* model, TensorInfo* tensor, int* out_size) {
    int shard = tensor->shard;
    *out_size = (int)tensor->data_size;
    return read_tensor_from_handle(g_shard_handles[shard],
                                   model->shard_data_starts[shard], tensor);
}

/*
 * Load shared weights (attention, norms, router) into RAM at startup
 * These are small and always needed — loaded once.
 */
static int load_shared_weights(const char* gguf_path, GGUFModel* model,
                                LayerWeights* layers) {
    int l;
    char name[256];

    /* For large models (397B): only load norms and router (small).
       Attention weights loaded on-demand per layer during inference.
       For small models (35B): load everything. */
    int lazy_attn = (model->hidden_dim >= 4096); /* 397B has hidden=4096 */
    printf("Loading shared weights for %d layers (lazy_attn=%d)...\n",
           model->num_layers, lazy_attn);

    for (l = 0; l < model->num_layers; l++) {
        int size;

        /* Attention norms (FP32) */
        snprintf(name, 256, "blk.%d.attn_norm.weight", l);
        TensorInfo* t = find_tensor(model, name);
        if (t) {
            layers[l].attn_norm = (float*)load_tensor_data(model, t, &size);
            /* GGUF stores w+1 for (1+w)-style RMSNorm. llama.cpp uses w+1 directly in standard RMSNorm.
               Do NOT subtract 1 — the stored values are already correct for standard RMSNorm. */
        }

        snprintf(name, 256, "blk.%d.ffn_norm.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].ffn_norm = (float*)load_tensor_data(model, t, &size);
        }

        /* Detect layer type: DeltaNet (has ssm_a) vs full attention (has attn_q) */
        snprintf(name, 256, "blk.%d.ssm_a", l);
        int is_deltanet = (find_tensor(model, name) != NULL);

        if (is_deltanet) {
            layers[l].is_deltanet = 1;

            if (!lazy_attn) {
                /* Small model: load attention weights upfront */
                snprintf(name, 256, "blk.%d.attn_qkv.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_qkv = load_tensor_data(model, t, &size); layers[l].w_qkv_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_gate.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_attn_gate = load_tensor_data(model, t, &size); layers[l].w_attn_gate_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_out.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_ssm_out = load_tensor_data(model, t, &size); layers[l].w_ssm_out_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_alpha.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_alpha = load_tensor_data(model, t, &size); layers[l].w_alpha_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_beta.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_beta = load_tensor_data(model, t, &size); layers[l].w_beta_type = t->type; }
            } else {
                /* Large model: store tensor types for on-demand loading */
                snprintf(name, 256, "blk.%d.attn_qkv.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_qkv_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_gate.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_attn_gate_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_out.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_ssm_out_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_alpha.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_alpha_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_beta.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_beta_type = t->type; }
            }

            /* Small SSM params (always load — tiny) */
            snprintf(name, 256, "blk.%d.ssm_a", l);
            t = find_tensor(model, name);
            if (t) { layers[l].ssm_a = (float*)load_tensor_data(model, t, &size); }

            snprintf(name, 256, "blk.%d.ssm_dt.bias", l);
            t = find_tensor(model, name);
            if (t) { layers[l].ssm_dt_bias = (float*)load_tensor_data(model, t, &size); }

            snprintf(name, 256, "blk.%d.ssm_norm.weight", l);
            t = find_tensor(model, name);
            if (t) {
                layers[l].ssm_norm_w = (float*)load_tensor_data(model, t, &size);
                if (l == 0 && layers[l].ssm_norm_w) {
                    int nn = (int)(t->dims[0] ? t->dims[0] : 0);
                    double ss = 0; float mn = 1e30f, mx = -1e30f;
                    for (int k = 0; k < nn; k++) {
                        float v = layers[l].ssm_norm_w[k];
                        ss += (double)v*v; if (v < mn) mn = v; if (v > mx) mx = v;
                    }
                    fprintf(stderr, "  SSM_NORM_W L0: dim0=%llu dim1=%llu type=%d n=%d rms=%.6f min=%.6f max=%.6f\n",
                        (unsigned long long)t->dims[0], (unsigned long long)t->dims[1], t->type, nn,
                        sqrtf((float)(ss/nn)), mn, mx);
                    fprintf(stderr, "  SSM_NORM_W L0 [0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                        layers[l].ssm_norm_w[0], layers[l].ssm_norm_w[1], layers[l].ssm_norm_w[2], layers[l].ssm_norm_w[3],
                        layers[l].ssm_norm_w[4], layers[l].ssm_norm_w[5], layers[l].ssm_norm_w[6], layers[l].ssm_norm_w[7]);
                }
            }

            snprintf(name, 256, "blk.%d.ssm_conv1d.weight", l);
            t = find_tensor(model, name);
            if (t) { layers[l].ssm_conv1d_w = (float*)load_tensor_data(model, t, &size); }
        } else {
            layers[l].is_deltanet = 0;
            /* Standard attention layer (every 4th) */
            if (!lazy_attn) {
                snprintf(name, 256, "blk.%d.attn_q.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wq = load_tensor_data(model, t, &size); layers[l].wq_type = t->type; layers[l].wq_rows = (int)t->dims[1]; }

                snprintf(name, 256, "blk.%d.attn_k.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wk = load_tensor_data(model, t, &size); layers[l].wk_type = t->type; layers[l].wk_rows = (int)t->dims[1]; }

                snprintf(name, 256, "blk.%d.attn_v.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wv = load_tensor_data(model, t, &size); layers[l].wv_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_output.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wo = load_tensor_data(model, t, &size); layers[l].wo_type = t->type; }
            } else {
                /* Lazy load for 397B — load Q/K/V/O + detect head_dim */
                snprintf(name, 256, "blk.%d.attn_q.weight", l);
                t = find_tensor(model, name);
                if (t) {
                    layers[l].wq = load_tensor_data(model, t, &size);
                    layers[l].wq_type = t->type;
                    layers[l].wq_rows = (int)t->dims[1];
                    if (l == 3) /* first standard layer */
                        fprintf(stderr, "Std attn L%d: Q dims=[%llu,%llu] type=%d size=%llu\n",
                            /* uint64_t is unsigned long on Linux, unlike Win64. */
                            l, (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
                            t->type, (unsigned long long)t->data_size);
                }

                snprintf(name, 256, "blk.%d.attn_k.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wk = load_tensor_data(model, t, &size); layers[l].wk_type = t->type; layers[l].wk_rows = (int)t->dims[1]; }

                snprintf(name, 256, "blk.%d.attn_v.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wv = load_tensor_data(model, t, &size); layers[l].wv_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_output.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wo = load_tensor_data(model, t, &size); layers[l].wo_type = t->type; }
            }

            /* QK RMSNorm weights (Qwen3.5 specific) */
            snprintf(name, 256, "blk.%d.attn_q_norm.weight", l);
            t = find_tensor(model, name);
            if (t) { layers[l].q_norm = (float*)load_tensor_data(model, t, &size); }

            snprintf(name, 256, "blk.%d.attn_k_norm.weight", l);
            t = find_tensor(model, name);
            if (t) { layers[l].k_norm = (float*)load_tensor_data(model, t, &size); }
        }

        /* Post-attention norm (both layer types) */
        snprintf(name, 256, "blk.%d.post_attention_norm.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].post_attn_norm = (float*)load_tensor_data(model, t, &size);
        }

        /* Router gate */
        snprintf(name, 256, "blk.%d.ffn_gate_inp.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].gate_inp = load_tensor_data(model, t, &size);
            layers[l].gate_type = t->type;
        }

        /* Expert tensor offsets (don't load — read on demand) */
        snprintf(name, 256, "blk.%d.ffn_gate_exps.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].gate_exps_offset = t->offset;
            layers[l].gate_per_expert = t->data_size / model->num_experts;
            layers[l].gate_exps_type = t->type;
            layers[l].gate_exps_shard = t->shard;
            if (l <= 2) fprintf(stderr, "  L%d gate_exps: shard=%d offset=%llu per_exp=%llu type=%d data_size=%llu\n",
                l, t->shard, (unsigned long long)t->offset,
                (unsigned long long)layers[l].gate_per_expert, t->type, (unsigned long long)t->data_size);
        }

        snprintf(name, 256, "blk.%d.ffn_up_exps.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].up_exps_offset = t->offset;
            layers[l].up_per_expert = t->data_size / model->num_experts;
            layers[l].up_exps_type = t->type;
            layers[l].up_exps_shard = t->shard;
        }

        snprintf(name, 256, "blk.%d.ffn_down_exps.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].down_exps_offset = t->offset;
            layers[l].down_per_expert = t->data_size / model->num_experts;
            layers[l].down_exps_type = t->type;
            layers[l].down_exps_shard = t->shard;
        }

        /* Shared expert weights (always active alongside routed experts) */
        snprintf(name, 256, "blk.%d.ffn_gate_shexp.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].shexp_gate = load_tensor_data(model, t, &size); layers[l].shexp_gate_type = t->type;
            if (l == 0) fprintf(stderr, "  SharedExp gate: dims=[%llu,%llu] type=%d size=%d\n",
                (unsigned long long)t->dims[0], (unsigned long long)t->dims[1], t->type, size);
        }

        snprintf(name, 256, "blk.%d.ffn_up_shexp.weight", l);
        t = find_tensor(model, name);
        if (t) { layers[l].shexp_up = load_tensor_data(model, t, &size); layers[l].shexp_up_type = t->type; }

        snprintf(name, 256, "blk.%d.ffn_down_shexp.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].shexp_down = load_tensor_data(model, t, &size); layers[l].shexp_down_type = t->type;
            if (l == 0) fprintf(stderr, "  SharedExp down: dims=[%llu,%llu] type=%d size=%d\n",
                (unsigned long long)t->dims[0], (unsigned long long)t->dims[1], t->type, size);
        }

        /* Shared expert sigmoid gate weight [hidden] FP32 */
        snprintf(name, 256, "blk.%d.ffn_gate_inp_shexp.weight", l);
        t = find_tensor(model, name);
        if (t) { layers[l].shexp_gate_inp = (float*)load_tensor_data(model, t, &size); }

        if (l % 10 == 0) { fprintf(stderr, "  Layer %d/%d loaded\n", l, model->num_layers); fflush(stderr); }
    }

    if (layers[0].shexp_gate) fprintf(stderr, "Shared expert loaded for all layers\n");
    printf("Shared weights loaded.\n\n");
    return 0;
}

int main(int argc, char** argv) {
    /* Require an explicit model; no machine-specific Windows default. */
    const char* gguf_path = NULL;
    int num_tokens = 10;
    const char* prompt_tokens_arg = NULL;  /* comma-separated token IDs (overrides default "Hello" prompt) */

    /* Reject unknown/missing arguments before opening files or allocating weights.
     * --tokens counts total forward positions, matching the inherited engine. */
    int i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            printf("Usage: linmoe --model FILE.gguf [--tokens 1..512] [--prompt-tokens ID,ID,...]\n"
                   "CPU baseline requires AVX512F/BW/DQ/VL, AVX2, FMA and F16C.\n"
                   "--tokens is total forward positions, including the prompt.\n");
            return 0;
        }
        if (!strcmp(argv[i], "--model") && i + 1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            char* end;
            errno = 0;
            long n = strtol(argv[++i], &end, 10);
            if (errno || !*argv[i] || *end || n < 1 || n > MAX_SEQ) {
                fprintf(stderr, "LinMoE: --tokens must be 1..%d\n", MAX_SEQ);
                return 2;
            }
            num_tokens = (int)n;
        } else if (!strcmp(argv[i], "--prompt-tokens") && i + 1 < argc) {
            prompt_tokens_arg = argv[++i];
        } else {
            fprintf(stderr, "LinMoE: unknown/incomplete argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (!gguf_path || !*gguf_path) {
        fprintf(stderr, "LinMoE: --model FILE.gguf is required (see --help)\n");
        return 2;
    }

    fprintf(stderr, "=== LinMoE Inference Engine ===\n");
    fprintf(stderr, "Model: %s\n", gguf_path);

    /* Layer-bisect trace output (for diff against llama-trace-dump). g_trace_tok
     * is set later in main, AFTER prompt_len is parsed from --prompt-tokens. */
    {
        const char* trace_path = getenv("WINMOE_TRACE_OUT");
        if (trace_path && *trace_path) {
            g_trace_out = fopen(trace_path, "w");
            if (!g_trace_out) {
                fprintf(stderr, "WARN: failed to open WINMOE_TRACE_OUT=%s\n", trace_path);
            }
        }
    }

    /* Parity-sprint binary logit dump — one float32[vocab] row per generated
     * position. Sidecar .json written at exit with metadata. */
    {
        const char* path = getenv("WINMOE_DUMP_ALL_LOGITS");
        if (path && *path) {
            g_all_logits_bin = fopen(path, "wb");
            if (!g_all_logits_bin) {
                fprintf(stderr, "WARN: failed to open WINMOE_DUMP_ALL_LOGITS=%s\n", path);
            } else {
                fprintf(stderr, "Dumping per-position logits to %s\n", path);
            }
        }
    }

    /* §1.8 diagnosis: scalar q6k path override (slow, only for parity diagnosis) */
    {
        const char* sc = getenv("WINMOE_LM_SCALAR");
        if (sc && *sc && *sc != '0') {
            g_q6k_use_scalar = 1;
            fprintf(stderr, "WINMOE_LM_SCALAR=1 — q6k_matvec uses scalar q6k_dot_block (no AVX2)\n");
        }
    }

    /* §1.8 diagnosis: per-position normed (post-final-RMSNorm, pre-LM-head) binary dump */
    {
        const char* path = getenv("WINMOE_DUMP_NORMED");
        if (path && *path) {
            g_normed_bin = fopen(path, "wb");
            if (!g_normed_bin) {
                fprintf(stderr, "WARN: failed to open WINMOE_DUMP_NORMED=%s\n", path);
            } else {
                fprintf(stderr, "Dumping per-position normed (post-final-norm) to %s\n", path);
            }
        }
    }

    /* Phase 4.1: expert access trace */
    {
        const char* path = getenv("WINMOE_EXPERT_TRACE");
        if (path && *path) {
            g_expert_trace_enabled = 1;
            g_expert_trace_path = path;
            fprintf(stderr, "Expert access tracing enabled, dump at exit to %s\n", path);
        }
    }

    /* Phase 4.3: MoE skip-every-N */
    {
        const char* env = getenv("WINMOE_MOE_SKIP_EVERY");
        if (env && *env) {
            int n = atoi(env);
            if (n >= 2 && n <= 60) {
                g_moe_skip_every = n;
                fprintf(stderr, "WINMOE_MOE_SKIP_EVERY=%d (routed MoE skipped on layers where layer%%%d == %d)\n",
                        n, n, n - 1);
            }
        }
    }

    fflush(stderr);
    printf("=== LinMoE Inference Engine ===\n\n");

    /* Parse GGUF — heap-allocate because GGUFModel is >1MB (stack overflow on Windows) */
    GGUFModel* pmodel = (GGUFModel*)calloc(1, sizeof(GGUFModel));
    if (!pmodel) { fprintf(stderr, "ERROR: cannot alloc GGUFModel\n"); return 1; }
    if (parse_gguf_split(gguf_path, pmodel) != 0) {
        printf("ERROR: Cannot parse GGUF: %s\n", gguf_path);
        free(pmodel); return 1;
    }
    /* Use macro so all existing `model.` references work with heap allocation */
    #define model (*pmodel)
    fprintf(stderr, "Parse done. Shards: %d, Tensors: %llu\n", model.num_shards, (unsigned long long)model.n_tensors);
    fflush(stderr);
    printf("Shards: %d, Total tensors: %llu\n", model.num_shards, (unsigned long long)model.n_tensors);

    /* Existing expert arrays have 16 entries and Q5_K down activations have
     * four 256-value blocks. Reject dimensions outside those implementation
     * bounds before divisions, buffer sizing or kernel dispatch. */
    if (model.hidden_dim <= 0 || model.hidden_dim > 8192 || model.hidden_dim % 256 ||
        model.expert_intermediate <= 0 || model.expert_intermediate > 1024 || model.expert_intermediate % 256 ||
        model.num_layers <= 0 || model.num_layers > 64 ||
        model.num_experts <= 0 || model.num_experts > 1024 ||
        model.expert_used_count <= 0 || model.expert_used_count > 16 ||
        model.expert_used_count > model.num_experts || model.head_count_kv <= 0 ||
        model.head_count_kv > 64 || model.ssm_state_size != 128 ||
        model.ssm_inner_size <= 0 || model.ssm_inner_size > 8192 || model.ssm_inner_size % 128 ||
        model.ssm_group_count <= 0 || model.ssm_group_count > 16 ||
        (model.ssm_inner_size / 128) % model.ssm_group_count) {
        fprintf(stderr, "LinMoE: model dimensions exceed the current Qwen3.5 runtime bounds\n");
        free(pmodel);
        return 1;
    }

    ModelConfig cfg;
    cfg.hidden_dim = model.hidden_dim;
    cfg.intermediate = model.expert_intermediate;
    cfg.num_layers = model.num_layers;
    cfg.num_experts = model.num_experts;
    cfg.expert_k = model.expert_used_count > 0 ? model.expert_used_count : 10;
    fprintf(stderr, "GGUF expert_used_count=%d\n", model.expert_used_count);
    cfg.rope_theta = model.rope_theta > 0 ? model.rope_theta : 10000000.0f;
    cfg.max_seq_len = MAX_SEQ;
    cfg.num_kv_heads = model.head_count_kv > 0 ? model.head_count_kv : 2;
    cfg.head_dim = model.ssm_state_size > 0 ? model.ssm_state_size : 128;
    cfg.num_q_heads = (model.ssm_inner_size > 0) ?
        (model.ssm_inner_size / cfg.head_dim) : (cfg.hidden_dim / cfg.head_dim);

    /* Initialize DeltaNet dimensions from GGUF metadata */
    {
        int ssm_inner = model.ssm_inner_size > 0 ? model.ssm_inner_size : 8192;
        int ssm_groups = model.ssm_group_count > 0 ? model.ssm_group_count : 16;
        int ssm_hd = model.ssm_state_size > 0 ? model.ssm_state_size : 128;
        dn_init_dims(ssm_inner, ssm_groups, ssm_hd);
    }

    /* Reviewer's Phase-1 check: assert known model dimensions */
    fprintf(stderr, "\n=== MODEL DIMS (from GGUF) ===\n");
    fprintf(stderr, "  hidden=%d num_layers=%d experts=%d top_k=%d\n",
        cfg.hidden_dim, cfg.num_layers, cfg.num_experts, cfg.expert_k);
    fprintf(stderr, "  moe_intermediate=%d shared_intermediate=%d\n",
        model.expert_intermediate, model.feed_forward_length);
    fprintf(stderr, "  head_count=%d head_count_kv=%d key_length/head_dim=%d\n",
        model.head_count, model.head_count_kv, cfg.head_dim);
    fprintf(stderr, "  ssm.inner=%d ssm.group_count=%d ssm.state_size=%d\n",
        model.ssm_inner_size, model.ssm_group_count, model.ssm_state_size);
    /* Sanity check: 397B should be (4096, 60, 512, 10, 1024, 1024, 32, 2, 256, 8192, 16, 128)
     *               35B  should be (2048, 40, 256, 8,  512,  512,  16, 2, 256, 4096, 16, 128) */
    if (cfg.num_experts == 512 && cfg.num_layers == 60) {
        fprintf(stderr, "  MATCH: Qwen3.5-397B config\n");
    } else if (cfg.num_experts == 256 && cfg.num_layers == 40) {
        fprintf(stderr, "  MATCH: Qwen3.5-35B config\n");
    } else {
        fprintf(stderr, "  WARNING: unexpected config!\n");
    }
    fprintf(stderr, "===\n\n");

    printf("Model: %s\n", gguf_path);
    printf("Config: hidden=%d, intermediate=%d, layers=%d\n",
           cfg.hidden_dim, cfg.intermediate, cfg.num_layers);
    fprintf(stderr, "GGUF: expert_feed_forward_length=%d, feed_forward_length=%d, routed_scaling_factor=%.4f\n",
            model.expert_intermediate, model.feed_forward_length, model.routed_scaling_factor);
    printf("Attention: Q=%d heads, KV=%d heads, head_dim=%d\n",
           cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim);
    printf("SSM: inner=%d state=%d head_count=%d\n",
           model.ssm_inner_size, model.ssm_state_size, model.head_count);
    printf("MoE: %d experts, K=%d active\n", cfg.num_experts, cfg.expert_k);
    printf("Tokens to generate: %d\n\n", num_tokens);

    /* DEBUG: scan for conv and other tensors */
    {
        int ti;
        for (ti = 0; ti < (int)model.n_tensors; ti++) {
            if ((strstr(model.tensors[ti].name, "blk.0.") &&
                (strstr(model.tensors[ti].name, "conv") || strstr(model.tensors[ti].name, "ssm_dt") ||
                 strstr(model.tensors[ti].name, "ssm_D"))) ||
                0) {
                fprintf(stderr, "TENSOR: %s  dims=[%llu,%llu] type=%d\n",
                    model.tensors[ti].name,
                    (unsigned long long)model.tensors[ti].dims[0],
                    (unsigned long long)model.tensors[ti].dims[1],
                    model.tensors[ti].type);
            }
        }
    }

    /* Open persistent shard handles for all reads */
    open_shard_handles(&model);

    /* Load shared weights */
    LayerWeights* layers = (LayerWeights*)calloc(cfg.num_layers, sizeof(LayerWeights));
    load_shared_weights(gguf_path, &model, layers);

    /* Check what we got */
    int loaded_layers = 0;
    for (i = 0; i < cfg.num_layers; i++) {
        if (layers[i].attn_norm && layers[i].gate_inp) loaded_layers++;
    }
    printf("Loaded %d/%d layers with shared weights\n\n", loaded_layers, cfg.num_layers);

    /* Critical struct size check */
    fprintf(stderr, "STRUCT SIZES: block_q4_K=%d(expect 144) block_q5_K=%d(expect 176) block_q6_K=%d(expect 210)\n",
        (int)sizeof(block_q4_K), (int)sizeof(block_q5_K), (int)sizeof(block_q6_K));
    if (sizeof(block_q6_K) != 210 || sizeof(block_q4_K) != 144 || sizeof(block_q5_K) != 176) {
        fprintf(stderr, "FATAL: Struct padding detected! This will corrupt weight addressing.\n");
    }

    /* Weight audit: check all layers have required tensors */
    {
        int missing = 0;
        for (i = 0; i < cfg.num_layers; i++) {
            if (!layers[i].attn_norm) { fprintf(stderr, "AUDIT: L%d missing attn_norm!\n", i); missing++; }
            if (!layers[i].post_attn_norm && !layers[i].ffn_norm) { fprintf(stderr, "AUDIT: L%d missing post_attn_norm AND ffn_norm!\n", i); missing++; }
            if (!layers[i].gate_inp) { fprintf(stderr, "AUDIT: L%d missing gate_inp (router)!\n", i); missing++; }
            if (layers[i].is_deltanet && !layers[i].ssm_a) { fprintf(stderr, "AUDIT: L%d (DN) missing ssm_a!\n", i); missing++; }
            if (!layers[i].is_deltanet && !layers[i].wq) { fprintf(stderr, "AUDIT: L%d (GQA) missing wq!\n", i); missing++; }
        }
        if (missing == 0) fprintf(stderr, "AUDIT: All %d layers have required weights.\n", cfg.num_layers);
        else {
            /* Inference cannot recover from a missing required layer weight. */
            fprintf(stderr, "AUDIT: %d MISSING weights found!\n", missing);
            exit(EXIT_FAILURE);
        }
        /* Dump norm magnitudes for a few layers */
        int HD_AUDIT = cfg.hidden_dim;
        for (i = 0; i < cfg.num_layers; i += 10) {
            float an=0, pan=0;
            for (int j = 0; j < HD_AUDIT; j++) {
                if (layers[i].attn_norm) an += layers[i].attn_norm[j] * layers[i].attn_norm[j];
                if (layers[i].post_attn_norm) pan += layers[i].post_attn_norm[j] * layers[i].post_attn_norm[j];
            }
            fprintf(stderr, "NORM L%d: attn_norm_rms=%.4f post_attn_norm_rms=%.4f",
                i, sqrtf(an/HD_AUDIT), sqrtf(pan/HD_AUDIT));
            if (layers[i].q_norm) {
                float qn = 0; int qd = cfg.head_dim;
                for (int j = 0; j < qd; j++) qn += layers[i].q_norm[j] * layers[i].q_norm[j];
                fprintf(stderr, " q_norm_rms=%.4f", sqrtf(qn/qd));
            }
            if (layers[i].k_norm) {
                float kn = 0; int qd = cfg.head_dim;
                for (int j = 0; j < qd; j++) kn += layers[i].k_norm[j] * layers[i].k_norm[j];
                fprintf(stderr, " k_norm_rms=%.4f", sqrtf(kn/qd));
            }
            fprintf(stderr, "\n");
        }
        /* final_norm printed later, after loading */
    }

    /* Load embedding and LM head */
    TensorInfo* tok_embd = find_tensor(&model, "token_embd.weight");
    TensorInfo* output_norm = find_tensor(&model, "output_norm.weight");
    TensorInfo* output_weight = find_tensor(&model, "output.weight");

    void* embd_data = NULL;
    float* final_norm = NULL;
    void* lm_head = NULL;
    int lm_head_type = 0;
    int vocab_size = 0;
    int sz;

    if (tok_embd) {
        /* Only Q8_0 has an on-demand row decoder in this runtime. Reject
         * other large/unsupported embeddings instead of generating zeros. */
        if (tok_embd->n_dims != 2 || tok_embd->dims[0] != (uint64_t)cfg.hidden_dim ||
            tok_embd->dims[1] > INT_MAX ||
            (tok_embd->type != GGML_TYPE_F32 && tok_embd->type != GGML_TYPE_F16 && tok_embd->type != GGML_TYPE_Q8_0) ||
            (tok_embd->data_size > 100000000ULL && tok_embd->type != GGML_TYPE_Q8_0)) {
            fprintf(stderr, "LinMoE: unsupported embedding shape/type\n");
            exit(EXIT_FAILURE);
        }
        vocab_size = (int)tok_embd->dims[1];
        fprintf(stderr, "Embeddings: %llu x %llu, type=%d (%llu bytes)\n",
                (unsigned long long)tok_embd->dims[0], (unsigned long long)tok_embd->dims[1],
                tok_embd->type, (unsigned long long)tok_embd->data_size);
        /* For large models: DON'T load entire embedding table.
           Instead, read one row on-demand per token. Store offset info. */
        if (tok_embd->data_size > 100000000ULL) { /* >100 MB = too large */
            fprintf(stderr, "  Embedding too large for bulk load — will read per-token\n");
            embd_data = NULL; /* flag for per-token loading */
        } else {
            embd_data = load_tensor_data(&model, tok_embd, &sz);
        }
    }
    if (output_norm) {
        final_norm = (float*)load_tensor_data(&model, output_norm, &sz);
    }
    if (output_weight) {
        lm_head_type = output_weight->type;
        fprintf(stderr, "Loading LM head: type=%d, vocab=%d, size=%.0f MB...\n",
                lm_head_type, vocab_size, output_weight->data_size / (1024.0 * 1024.0));
        fflush(stderr);
        lm_head = load_tensor_data(&model, output_weight, &sz);
        if (lm_head) fprintf(stderr, "LM head loaded (%d MB)\n", sz / (1024*1024));
        else fprintf(stderr, "WARNING: LM head failed to load\n");
    }

    /* Missing output/embedding tensors cannot produce meaningful logits. */
    if (!tok_embd || !final_norm || !lm_head || vocab_size <= 0) {
        fprintf(stderr, "LinMoE: missing embedding, final norm, or output weights\n");
        exit(EXIT_FAILURE);
    }
    cfg.vocab_size = vocab_size;

    /* Allocate working buffers */
    int H = cfg.hidden_dim;
    int I = cfg.intermediate;
    int K = cfg.expert_k;
    /* Default K=8 — Pareto improvement over GGUF's top_k=10:
     * - K=8 fib produces identical Python output as K=10 (same first 40 tokens)
     * - K=8 chat produces identical "Hello! How can I help you today?<|im_end|>"
     * - K=8 chat steady-state +9% (0.359 -> 0.390 tok/s) over K=10
     * - K=6 BREAKS code generation (stuck _cached loop)
     * Override via WINMOE_TOPK env (e.g., =10 to restore GGUF default). */
    if (K > 8) K = 8;
    {
        const char* k_env = getenv("WINMOE_TOPK");
        if (k_env) {
            int new_k = atoi(k_env);
            if (new_k >= 1 && new_k <= cfg.expert_k) {
                fprintf(stderr, "WINMOE_TOPK override: K=%d (was %d)\n", new_k, K);
                K = new_k;
            }
        }
    }
    fprintf(stderr, "MoE top_k = %d (GGUF says %d)\n", K, cfg.expert_k);
    int qkv_dim = cfg.num_q_heads * cfg.head_dim;
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;
    /* GQA reuses q but its head geometry is independent of DeltaNet's.
     * Derive maxima from loaded projections; otherwise mixed architectures
     * can overwrite q even when every tensor individually fits its file. */
    for (i = 0; i < cfg.num_layers; ++i) {
        if (!layers[i].is_deltanet) {
            int qrows = layers[i].wq_rows, krows = layers[i].wk_rows;
            if (krows <= 0 || krows > 16384 || krows % cfg.num_kv_heads ||
                qrows <= 0 || qrows > 32768 || qrows % (2 * krows)) {
                fprintf(stderr, "LinMoE: unsupported GQA projection geometry at layer %d\n", i);
                exit(EXIT_FAILURE);
            }
            if (qrows / 2 > qkv_dim) qkv_dim = qrows / 2;
            if (krows > kv_dim) kv_dim = krows;
        }
    }


    float* hidden = (float*)lm_aligned_alloc(H * sizeof(float), 32);
    float* residual = (float*)lm_aligned_alloc(H * sizeof(float), 32);
    float* normed = (float*)lm_aligned_alloc(H * sizeof(float), 32);
    float* q = (float*)lm_aligned_alloc(qkv_dim * sizeof(float), 32);
    float* k_cur = (float*)lm_aligned_alloc(kv_dim * sizeof(float), 32);
    float* v_cur = (float*)lm_aligned_alloc(kv_dim * sizeof(float), 32);
    float* attn_out = (float*)lm_aligned_alloc(qkv_dim * sizeof(float), 32);
    float* o_out = (float*)lm_aligned_alloc(H * sizeof(float), 32);
    float* gate_buf = (float*)lm_aligned_alloc(I * sizeof(float), 32);
    float* up_buf = (float*)lm_aligned_alloc(I * sizeof(float), 32);
    float* act_buf = (float*)lm_aligned_alloc(I * sizeof(float), 32);
    float* expert_out = (float*)lm_aligned_alloc(H * sizeof(float), 32);
    float* moe_out = (float*)lm_aligned_alloc(H * sizeof(float), 32);
    float* logits = (float*)malloc(vocab_size * sizeof(float));

    /* Expert weight read buffer */
    int max_expert_size = 1024 * 1024;  /* will be set properly */
    if (loaded_layers > 0) {
        uint64_t es = layers[0].gate_per_expert;
        if (es > (uint64_t)max_expert_size) max_expert_size = (int)es;
        es = layers[0].up_per_expert;
        if (es > (uint64_t)max_expert_size) max_expert_size = (int)es;
        es = layers[0].down_per_expert;
        if (es > (uint64_t)max_expert_size) max_expert_size = (int)es;
    }
    int expert_buf_size = ((max_expert_size + ALIGN * 2) / ALIGN + 1) * ALIGN;
    void* expert_buf = lm_aligned_alloc(expert_buf_size, ALIGN);

    /* DeltaNet states + KV caches per layer */
    DeltaNetState* dn_states = (DeltaNetState*)calloc(cfg.num_layers, sizeof(DeltaNetState));
    KVCache* kv_caches = (KVCache*)calloc(cfg.num_layers, sizeof(KVCache));
    for (i = 0; i < cfg.num_layers; i++) {
        if (layers[i].is_deltanet) {
            dn_state_init(&dn_states[i]);
        } else {
            /* Standard attention: detect head_dim from K weight dims */
            int std_hd = layers[i].wk_rows > 0 ? layers[i].wk_rows / cfg.num_kv_heads : 256;
            kv_cache_init(&kv_caches[i], MAX_SEQ, cfg.num_kv_heads, std_hd);
        }
    }
    /* === GPU INITIALIZATION === */
    int use_gpu = (gpu_init() == 0);
    if (use_gpu) {
        /* Configure GPU expert cache limit (default 200, env-overridable) */
        const char* gpu_exp_env = getenv("WINMOE_GPU_EXPERTS");
        int gpu_exp_limit = gpu_exp_env ? atoi(gpu_exp_env) : 200;
        gpu_set_expert_limit(gpu_exp_limit);

        fprintf(stderr, "Uploading DeltaNet weights to GPU...\n");
        fflush(stderr);
        for (i = 0; i < cfg.num_layers; i++) {
            if (layers[i].is_deltanet) {
                /* Load weights from disk if not already in RAM (lazy mode) */
                char tname[256]; int tsz;
                if (!layers[i].w_qkv) {
                    snprintf(tname, 256, "blk.%d.attn_qkv.weight", i);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) {
                        if (i == 0) fprintf(stderr, "  L0 QKV LOAD: shard=%d offset=%llu data_start=%llu abs=%llu datasize=%llu\n",
                            tt->shard, (unsigned long long)tt->offset,
                            (unsigned long long)model.shard_data_starts[tt->shard],
                            (unsigned long long)(model.shard_data_starts[tt->shard] + tt->offset),
                            (unsigned long long)tt->data_size);
                        layers[i].w_qkv = load_tensor_data(&model, tt, &tsz); layers[i].w_qkv_type = tt->type;
                    }
                }
                if (!layers[i].w_attn_gate) {
                    snprintf(tname, 256, "blk.%d.attn_gate.weight", i);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { layers[i].w_attn_gate = load_tensor_data(&model, tt, &tsz); layers[i].w_attn_gate_type = tt->type; }
                }
                if (!layers[i].w_ssm_out) {
                    snprintf(tname, 256, "blk.%d.ssm_out.weight", i);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { layers[i].w_ssm_out = load_tensor_data(&model, tt, &tsz); layers[i].w_ssm_out_type = tt->type; }
                }
                if (i % 10 == 0) { fprintf(stderr, "  Loading layer %d/%d...\n", i, cfg.num_layers); fflush(stderr); }
            }
            if (layers[i].is_deltanet && layers[i].w_qkv) {
                if (i == 0) {
                    char tn[256];
                    snprintf(tn, 256, "blk.0.attn_qkv.weight");
                    TensorInfo* tqkv = find_tensor(&model, tn);
                    if (tqkv) fprintf(stderr, "  DeltaNet QKV dims=[%llu,%llu] type=%d\n",
                        (unsigned long long)tqkv->dims[0], (unsigned long long)tqkv->dims[1], tqkv->type);
                    snprintf(tn, 256, "blk.0.attn_gate.weight");
                    TensorInfo* tg = find_tensor(&model, tn);
                    if (tg) fprintf(stderr, "  DeltaNet Gate dims=[%llu,%llu]\n",
                        (unsigned long long)tg->dims[0], (unsigned long long)tg->dims[1]);
                    snprintf(tn, 256, "blk.0.ssm_alpha.weight");
                    TensorInfo* ta = find_tensor(&model, tn);
                    if (ta) fprintf(stderr, "  DeltaNet Alpha dims=[%llu,%llu]\n",
                        (unsigned long long)ta->dims[0], (unsigned long long)ta->dims[1]);
                    snprintf(tn, 256, "blk.0.ssm_a");
                    TensorInfo* tssa = find_tensor(&model, tn);
                    if (tssa) fprintf(stderr, "  DeltaNet ssm_a dims=[%llu,%llu]\n",
                        (unsigned long long)tssa->dims[0], (unsigned long long)tssa->dims[1]);
                    snprintf(tn, 256, "blk.0.ssm_out.weight");
                    TensorInfo* tso = find_tensor(&model, tn);
                    if (tso) fprintf(stderr, "  DeltaNet SSM_Out dims=[%llu,%llu]\n",
                        (unsigned long long)tso->dims[0], (unsigned long long)tso->dims[1]);
                    snprintf(tn, 256, "blk.0.ssm_norm.weight");
                    TensorInfo* tsn = find_tensor(&model, tn);
                    if (tsn) fprintf(stderr, "  DeltaNet SSM_Norm dims=[%llu]\n",
                        (unsigned long long)tsn->dims[0]);
                }
                /* Upload Q8_0 weights → GPU
                 * QKV: [hidden, 12288] = Q(2048) + K(2048) + V(8192)
                 * Gate: [hidden, 8192] = Z gate
                 * SSM Out: [8192, hidden] = output projection
                 */
                gpu_upload_deltanet_weights(i,
                    layers[i].w_qkv, H, DN_QKV_DIM,      /* QKV: [4096, 12288] */
                    layers[i].w_attn_gate, H, DN_INNER,   /* Gate: [4096, 8192] */
                    layers[i].w_ssm_out, DN_INNER, H);    /* SSM Out: [8192, 4096] */
            }
        }
        fprintf(stderr, "GPU VRAM used: %.0f MB (DeltaNet)\n", gpu_vram_used_mb());

        /* Upload standard GQA weights to GPU (Q8_0, ~840 MB for 15 layers) */
        {
            int gqa_count = 0;
            for (i = 0; i < cfg.num_layers; i++) {
                if (!layers[i].is_deltanet && layers[i].wq && layers[i].wk && layers[i].wv && layers[i].wo) {
                    /* Q: [H, nqh*hd*2], K: [H, nkvh*hd], V: [H, nkvh*hd], O: [nqh*hd, H] */
                    int std_nqh = 32, std_hd = 256;
                    int q_cols = std_nqh * std_hd * 2;  /* 16384 */
                    int kv_cols = cfg.num_kv_heads * std_hd; /* 512 */
                    int o_rows = std_nqh * std_hd;  /* 8192 */
                    if (gpu_upload_gqa_weights(i,
                        layers[i].wq, H, q_cols,
                        layers[i].wk, H, kv_cols,
                        layers[i].wv, H, kv_cols,
                        layers[i].wo, o_rows, H) == 0)
                        gqa_count++;
                }
            }
            if (gqa_count > 0)
                fprintf(stderr, "GPU GQA: %d layers uploaded, VRAM=%.0f MB\n", gqa_count, gpu_vram_used_mb());
        }
        /* Router stays on CPU — 480MB VRAM better spent on expert cache (v8.7, v9.5 confirmed) */
    } else {
        fprintf(stderr, "GPU init failed — running CPU-only\n");
    }

    fprintf(stderr, "DeltaNet states: %d layers, KV caches: %d layers\n",
            (int)(cfg.num_layers * 3 / 4), (int)(cfg.num_layers / 4));

    printf("\nBuffers allocated. Starting generation...\n\n");

    /* === EXPERT RAM CACHE === */
    /* Simple array-based cache: (layer * num_experts + expert_id) → cached data */
    /* Budget: ~20 GB = fits ~1400 full experts (gate+up+down ~14 MB each at Q4_K 397B) */
    int expert_total_size = (int)(layers[0].gate_per_expert + layers[0].up_per_expert + layers[0].down_per_expert);
    long long cache_budget = 20LL * 1024 * 1024 * 1024; /* default 20 GB */
    {
        const char* env = getenv("WINMOE_CACHE_GB");
        if (env) {
            long long gb = atoll(env);
            if (gb >= 1 && gb <= 32) cache_budget = gb * 1024LL * 1024 * 1024;
        }
    }
    /* All cache slots use the same layout; reject heterogeneous experts
     * rather than copying a later layer into a layer-zero-sized allocation. */
    if (expert_total_size <= 0 || expert_total_size > cache_budget) {
        fprintf(stderr, "LinMoE: invalid expert size/cache budget\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < cfg.num_layers; ++i) {
        if (!layers[i].gate_per_expert ||
            layers[i].gate_per_expert != layers[0].gate_per_expert ||
            layers[i].up_per_expert != layers[0].up_per_expert ||
            layers[i].down_per_expert != layers[0].down_per_expert) {
            fprintf(stderr, "LinMoE: incompatible expert layout at layer %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    int max_cached = (int)(cache_budget / expert_total_size);
    if (max_cached > cfg.num_layers * cfg.num_experts) max_cached = cfg.num_layers * cfg.num_experts;

    int total_slots = cfg.num_layers * cfg.num_experts;
    void** expert_cache = (void**)calloc(total_slots, sizeof(void*));
    /* Reverse map for LRU: cached_keys[i] = the cache_key stored at LRU slot i (or -1) */
    int* cached_keys = (int*)malloc(max_cached * sizeof(int));
    for (int q = 0; q < max_cached; q++) cached_keys[q] = -1;
    int evict_ptr = 0;  /* round-robin eviction pointer (legacy) */
    int cache_hits = 0, cache_misses = 0, cache_stored = 0;

    /* Phase 4.2: LFU cache eviction. WINMOE_CACHE_POLICY=lfu enables hit-frequency
     * based eviction. Per-slot hit_count, evict argmin(hit_count) on full-cache miss.
     * O(1) hit (via key_to_slot index), O(max_cached) miss eviction. */
    const char* cache_policy_env = getenv("WINMOE_CACHE_POLICY");
    int cache_policy_lfu = (cache_policy_env && (!strcmp(cache_policy_env, "lfu") || !strcmp(cache_policy_env, "LFU")));
    int* cache_hit_counts = (int*)calloc(max_cached, sizeof(int));
    int* key_to_slot = (int*)malloc(total_slots * sizeof(int));
    for (int q = 0; q < total_slots; q++) key_to_slot[q] = -1;

    fprintf(stderr, "Expert cache: %d slots, budget %.1f GB, expert_size=%d bytes, policy=%s\n",
            max_cached, (double)cache_budget / (1024*1024*1024), expert_total_size,
            cache_policy_lfu ? "LFU" : "round-robin");

    /* Reusable IO buffer pool: 16 sets × 3 aligned buffers (gate/up/down).
     * 16 covers the max K=10 with headroom. Allocated once, reused across all
     * miss reads — eliminates per-miss aligned_malloc churn. */
    void* io_pool[16][3] = {0};
    {
        int got_all = 1;
        for (int q = 0; q < 16; q++) {
            io_pool[q][0] = lm_aligned_alloc(expert_buf_size, ALIGN);
            io_pool[q][1] = lm_aligned_alloc(expert_buf_size, ALIGN);
            io_pool[q][2] = lm_aligned_alloc(expert_buf_size, ALIGN);
            if (!io_pool[q][0] || !io_pool[q][1] || !io_pool[q][2]) got_all = 0;
        }
        fprintf(stderr, "IO buffer pool: %d slots × 3 × %d bytes = %.1f MB %s\n",
            16, expert_buf_size, 16.0 * 3.0 * expert_buf_size / (1024.0*1024.0),
            got_all ? "OK" : "PARTIAL");
    }

    /* === TOKEN GENERATION LOOP === */
    /* Monotonic ticks exclude wall-clock adjustments from timings. */
    int64_t freq, gen_start, gen_end;
    freq = lm_clock_frequency();

    /* Profiling accumulators (per token, reset each token) */
    double prof_attn_ms, prof_expert_io_ms, prof_expert_compute_ms;
    double prof_router_ms, prof_norm_ms, prof_embed_ms;
    int64_t prof_t0, prof_t1;

    /* Default prompt (chat "Hello"): <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n */
    int prompt_tokens_default[] = {248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198};
    int prompt_tokens_buf[1024];
    int* prompt_tokens = prompt_tokens_default;
    int prompt_len = 9;
    if (prompt_tokens_arg) {
        prompt_len = 0;
        const char* p = prompt_tokens_arg;
        /* Token IDs are indices, not arbitrary integers. Refuse truncation,
         * trailing garbage and empty fields before accessing embeddings/logits. */
        while (*p) {
            char* end = NULL;
            errno = 0;
            long v = strtol(p, &end, 10);
            if (errno || end == p || v < 0 || v >= vocab_size ||
                prompt_len >= MAX_SEQ || (*end && *end != ',') ||
                (*end == ',' && !end[1])) {
                fprintf(stderr, "LinMoE: invalid --prompt-tokens near %s\n", p);
                exit(EXIT_FAILURE);
            }
            prompt_tokens_buf[prompt_len++] = (int)v;
            p = *end ? end + 1 : end;
        }
        if (!prompt_len) {
            fprintf(stderr, "LinMoE: --prompt-tokens must not be empty\n");
            exit(EXIT_FAILURE);
        }
        prompt_tokens = prompt_tokens_buf;
    }
    for (i = 0; i < prompt_len; ++i) {
        if (prompt_tokens[i] < 0 || prompt_tokens[i] >= vocab_size) {
            fprintf(stderr, "LinMoE: prompt token %d exceeds vocabulary %d\n", prompt_tokens[i], vocab_size);
            exit(EXIT_FAILURE);
        }
    }
    int cur_token = prompt_tokens[0];
    int tokens_generated = 0;

    /* Now that prompt_len is known, set the trace target to the last prompt position */
    if (g_trace_out) {
        g_trace_tok = prompt_len - 1;
        fprintf(g_trace_out, "# winmoe trace: tok=%d (last prompt of %d-token prompt)\n",
                g_trace_tok, prompt_len);
        fprintf(stderr, "Trace target: tok=%d (last of %d prompt tokens)\n",
                g_trace_tok, prompt_len);
    }

    gen_start = lm_clock_now();

    for (int tok = 0; tok < num_tokens; tok++) {
        int64_t tok_start, tok_end;
        tok_start = lm_clock_now();
        prof_attn_ms = prof_expert_io_ms = prof_expert_compute_ms = 0;
        prof_router_ms = prof_norm_ms = prof_embed_ms = 0;

        /* DEBUG: dump raw Q8_0 embedding bytes for first token */
        if (tok == 0) {
            fprintf(stderr, "DEBUG: loading embedding for token %d\n", cur_token);
        }
        /* 1. Get embedding for current token */
        if (embd_data) {
            /* Small model — embedding table already in RAM */
            if (tok_embd->type == GGML_TYPE_F32) {
                const float* emb = (const float*)embd_data;
                memcpy(hidden, emb + cur_token * H, H * sizeof(float));
            } else if (tok_embd->type == GGML_TYPE_F16) {
                const uint16_t* emb = (const uint16_t*)embd_data;
                for (i = 0; i < H; i++) hidden[i] = fp16_to_fp32(emb[cur_token * H + i]);
            } else if (tok_embd->type == 8) {
                int blocks_per_row = H / Q8_QK;
                int row_bytes = blocks_per_row * Q8_BLOCK_SIZE;
                const void* row_data = (const char*)embd_data + (long long)cur_token * row_bytes;
                q8_dequant_row(hidden, row_data, H);
            }
        } else if (tok_embd && tok_embd->type == 8) {
            /* Large model — read one embedding row from disk on demand */
            int blocks_per_row = H / Q8_QK;
            int row_bytes = blocks_per_row * Q8_BLOCK_SIZE;
            uint64_t embd_abs = model.shard_data_starts[tok_embd->shard] +
                                tok_embd->offset + (uint64_t)cur_token * row_bytes;
            uint64_t embd_aligned = (embd_abs / ALIGN) * ALIGN;
            int embd_sub = (int)(embd_abs - embd_aligned);
            int embd_read_sz = ((row_bytes + embd_sub + ALIGN) / ALIGN) * ALIGN;
            void* embd_buf = lm_aligned_alloc(embd_read_sz, ALIGN);
            if (embd_buf) {
                /* The storage boundary validates the row; no shared seek position. */
                read_bytes_from_shard(tok_embd->shard, embd_abs, NULL,
                                      row_bytes, embd_buf, embd_read_sz);
                const unsigned char* raw = (const unsigned char*)embd_buf + embd_sub;
                if (tok == 0) {
                    /* Dump first Q8_0 block: 2 bytes FP16 scale + 32 int8 values */
                    fprintf(stderr, "  Q8_0 block0 hex: ");
                    for (int bi = 0; bi < 34; bi++) fprintf(stderr, "%02x ", raw[bi]);
                    fprintf(stderr, "\n");
                    uint16_t d16; memcpy(&d16, raw, 2);
                    float d_val = 0;
                    /* FP16 to float */
                    int sign = (d16 >> 15) & 1;
                    int exp = (d16 >> 10) & 0x1F;
                    int mant = d16 & 0x3FF;
                    if (exp == 0) d_val = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (1.0f / 16384.0f);
                    else if (exp == 31) d_val = sign ? -INFINITY : INFINITY;
                    else d_val = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15);
                    fprintf(stderr, "  d16=0x%04x d_float=%.6f qs[0..3]=%d %d %d %d\n",
                        d16, d_val, (int)(signed char)raw[2], (int)(signed char)raw[3],
                        (int)(signed char)raw[4], (int)(signed char)raw[5]);
                    fprintf(stderr, "  expected hidden[0]=%.6f (d*qs[0]=%.6f)\n",
                        d_val * (float)(signed char)raw[2], d_val * (float)(signed char)raw[2]);
                }
                q8_dequant_row(hidden, raw, H);
                lm_aligned_free(embd_buf);
            } else {
                memset(hidden, 0, H * sizeof(float));
            }
        } else {
            memset(hidden, 0, H * sizeof(float));
        }

        /* Debug: trace signal through layers */
        if (tok == 0) {
            fprintf(stderr, "Embedding[0..3]: %.6f %.6f %.6f %.6f\n",
                    hidden[0], hidden[1], hidden[2], hidden[3]);
        }

        /* 2. Run through all layers */
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            LayerWeights* lw = &layers[layer];
            memcpy(residual, hidden, H * sizeof(float));

            /* 2a. Attention norm */
            if (lw->attn_norm) {
                rmsnorm(normed, hidden, lw->attn_norm, H);
            } else {
                if (tok == 0) fprintf(stderr, "WARNING: L%d has NO attn_norm! Using raw hidden.\n", layer);
                memcpy(normed, hidden, H * sizeof(float));
            }
            trace_dump("layer_input", layer, tok, hidden, H);
            trace_dump("attn_norm",   layer, tok, normed, H);

            /* 2b-g. Attention — dispatch DeltaNet or standard GQA */
            prof_t0 = lm_clock_now();
            memset(o_out, 0, H * sizeof(float));

            if (lw->is_deltanet) {
                /* On-demand loading for lazy mode (397B) */
                char tname[256];
                int tsz;
                if (!lw->w_qkv) {
                    snprintf(tname, 256, "blk.%d.attn_qkv.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) {
                        if (layer == 0) fprintf(stderr, "  L0 QKV: shard=%d offset=%llu data_start=%llu abs=%llu\n",
                            tt->shard, (unsigned long long)tt->offset,
                            (unsigned long long)model.shard_data_starts[tt->shard],
                            (unsigned long long)(model.shard_data_starts[tt->shard] + tt->offset));
                        lw->w_qkv = load_tensor_data(&model, tt, &tsz); lw->w_qkv_type = tt->type;
                    }
                }
                if (!lw->w_attn_gate) {
                    snprintf(tname, 256, "blk.%d.attn_gate.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_attn_gate = load_tensor_data(&model, tt, &tsz); lw->w_attn_gate_type = tt->type; }
                }
                if (!lw->w_ssm_out) {
                    snprintf(tname, 256, "blk.%d.ssm_out.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_ssm_out = load_tensor_data(&model, tt, &tsz); lw->w_ssm_out_type = tt->type; }
                }
                if (!lw->w_alpha) {
                    snprintf(tname, 256, "blk.%d.ssm_alpha.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_alpha = load_tensor_data(&model, tt, &tsz); lw->w_alpha_type = tt->type; }
                }
                if (!lw->w_beta) {
                    snprintf(tname, 256, "blk.%d.ssm_beta.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_beta = load_tensor_data(&model, tt, &tsz); lw->w_beta_type = tt->type; }
                }

                if (use_gpu && lw->w_qkv) {
                    /* === GPU DELTANET (Phase 3: async pipeline) === */
                    int slot_idx = layer % 2;

                    /* ASYNC: Launch QKV+Gate on GPU (returns immediately) */
                    gpu_launch_qkv_gate(layer, slot_idx, normed, H, DN_QKV_DIM, DN_INNER);

                    /* CPU: Alpha + Beta projections WHILE GPU computes (overlap!) */
                    float* alpha_raw = dn_states[layer].tmp_alpha;
                    float* beta_raw = dn_states[layer].tmp_beta;
                    if (lw->w_alpha) quant_matvec(alpha_raw, lw->w_alpha, normed, DN_NUM_GATES, H, lw->w_alpha_type);
                    if (lw->w_beta) quant_matvec(beta_raw, lw->w_beta, normed, DN_NUM_GATES, H, lw->w_beta_type);

                    /* SYNC: Wait for GPU QKV+Gate results */
                    gpu_wait_qkv_gate(slot_idx);
                    float* gpu_qkv = gpu_get_qkv_out(slot_idx);
                    float* gpu_gate = gpu_get_gate_out(slot_idx);

                    /* DIAG: optionally override GPU output with CPU matmul to test GPU kernel correctness */
                    static int dn_force_cpu = -1;
                    if (dn_force_cpu < 0) dn_force_cpu = getenv("WINMOE_DN_CPU") ? 1 : 0;
                    if (dn_force_cpu) {
                        if (lw->w_qkv) quant_matvec(gpu_qkv, lw->w_qkv, normed, DN_QKV_DIM, H, lw->w_qkv_type);
                        if (lw->w_attn_gate) quant_matvec(gpu_gate, lw->w_attn_gate, normed, DN_INNER, H, lw->w_attn_gate_type);
                    }

                    /* L0 debug: print values at each stage */
                    if (tok == 0 && layer == 0 && lw->ssm_conv1d_w) {
                        /* Test both weight layouts to determine which is correct */
                        fprintf(stderr, "  conv1d_w layout A [ch,k]: ch0=[%.6f %.6f %.6f %.6f] ch1=[%.6f %.6f %.6f %.6f]\n",
                            lw->ssm_conv1d_w[0*4+0], lw->ssm_conv1d_w[0*4+1], lw->ssm_conv1d_w[0*4+2], lw->ssm_conv1d_w[0*4+3],
                            lw->ssm_conv1d_w[1*4+0], lw->ssm_conv1d_w[1*4+1], lw->ssm_conv1d_w[1*4+2], lw->ssm_conv1d_w[1*4+3]);
                        fprintf(stderr, "  conv1d_w layout B [k,ch]: ch0=[%.6f %.6f %.6f %.6f] ch1=[%.6f %.6f %.6f %.6f]\n",
                            lw->ssm_conv1d_w[0*12288+0], lw->ssm_conv1d_w[1*12288+0], lw->ssm_conv1d_w[2*12288+0], lw->ssm_conv1d_w[3*12288+0],
                            lw->ssm_conv1d_w[0*12288+1], lw->ssm_conv1d_w[1*12288+1], lw->ssm_conv1d_w[2*12288+1], lw->ssm_conv1d_w[3*12288+1]);
                    }
                    if (tok == 0 && layer == 0) {
                        fprintf(stderr, "=== L0 VALIDATION (tok=0) ===\n");
                        fprintf(stderr, "  normed[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                            normed[0], normed[1], normed[2], normed[3],
                            normed[4], normed[5], normed[6], normed[7]);
                        fprintf(stderr, "  qkv_pre_conv[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                            gpu_qkv[0], gpu_qkv[1], gpu_qkv[2], gpu_qkv[3],
                            gpu_qkv[4], gpu_qkv[5], gpu_qkv[6], gpu_qkv[7]);
                        fprintf(stderr, "  gate[0..3]: %.6f %.6f %.6f %.6f\n",
                            gpu_gate[0], gpu_gate[1], gpu_gate[2], gpu_gate[3]);
                    }

                    /* Apply conv1d to QKV before state recurrence */
                    if (lw->ssm_conv1d_w) {
                        DeltaNetState* ds = &dn_states[layer];
                        int conv_slot = ds->conv_pos % DN_CONV_WIDTH;
                        memcpy(ds->conv_buf + conv_slot * DN_QKV_DIM, gpu_qkv, DN_QKV_DIM * sizeof(float));
                        ds->conv_pos++;
                        for (i = 0; i < DN_QKV_DIM; i++) {
                            float sum = 0.0f;
                            for (int k = 0; k < DN_CONV_WIDTH; k++) {
                                int hs = ((ds->conv_pos - DN_CONV_WIDTH + k) % DN_CONV_WIDTH + DN_CONV_WIDTH) % DN_CONV_WIDTH;
                                /* ggml convention: k=0 → oldest, k=d_conv-1 → newest */
                                sum += lw->ssm_conv1d_w[i * DN_CONV_WIDTH + k] * ds->conv_buf[hs * DN_QKV_DIM + i];
                            }
                            /* Stable SiLU: x * sigmoid(x), sigmoid_stable avoids overflow */
                            gpu_qkv[i] = sum * sigmoid_stable(sum);
                        }
                    }

                    if (tok == 0 && layer == 0) {
                        fprintf(stderr, "  qkv_post_conv[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                            gpu_qkv[0], gpu_qkv[1], gpu_qkv[2], gpu_qkv[3],
                            gpu_qkv[4], gpu_qkv[5], gpu_qkv[6], gpu_qkv[7]);
                        /* Also print Q/K/V split points */
                        fprintf(stderr, "  Q[0..3]: %.6f %.6f %.6f %.6f\n",
                            gpu_qkv[0], gpu_qkv[1], gpu_qkv[2], gpu_qkv[3]);
                        fprintf(stderr, "  K[0..3]: %.6f %.6f %.6f %.6f\n",
                            gpu_qkv[DN_KEY_DIM], gpu_qkv[DN_KEY_DIM+1], gpu_qkv[DN_KEY_DIM+2], gpu_qkv[DN_KEY_DIM+3]);
                        fprintf(stderr, "  V[0..3]: %.6f %.6f %.6f %.6f\n",
                            gpu_qkv[2*DN_KEY_DIM], gpu_qkv[2*DN_KEY_DIM+1], gpu_qkv[2*DN_KEY_DIM+2], gpu_qkv[2*DN_KEY_DIM+3]);
                        fprintf(stderr, "  alpha[0..3]: %.6f %.6f %.6f %.6f\n",
                            alpha_raw[0], alpha_raw[1], alpha_raw[2], alpha_raw[3]);
                        fprintf(stderr, "  ssm_a[0..3]: %.6f %.6f %.6f %.6f\n",
                            lw->ssm_a ? lw->ssm_a[0] : 0, lw->ssm_a ? lw->ssm_a[1] : 0,
                            lw->ssm_a ? lw->ssm_a[2] : 0, lw->ssm_a ? lw->ssm_a[3] : 0);
                    }

                    /* CPU: DeltaNet state recurrence using GPU's QKV output */
                    /* QKV split: Q[2048] + K[2048] + V[8192] = 12288 */
                    float* Q = gpu_qkv;
                    float* K_ptr = gpu_qkv + DN_KEY_DIM;
                    float* V_ptr = gpu_qkv + DN_KEY_DIM + DN_KEY_DIM;

                    /* Compute gate parameters */
                    float* gate_decay = (float*)lm_temp_alloc(DN_NUM_GATES * sizeof(float));
                    float* beta_vals = (float*)lm_temp_alloc(DN_NUM_GATES * sizeof(float));
                    int hi;
                    for (hi = 0; hi < DN_NUM_GATES; hi++) {
                        /* Stable softplus: log1p(exp(-|x|)) + max(x,0) avoids overflow at large x */
                        float a_val = (lw->ssm_a && lw->ssm_dt_bias) ?
                            expf(lw->ssm_a[hi] * softplus_stable(alpha_raw[hi] + lw->ssm_dt_bias[hi])) : 0.99f;
                        if (a_val > 1.0f) a_val = 1.0f;
                        if (a_val < 0.0f) a_val = 0.0f;
                        if (isnan(a_val)) a_val = 0.99f;
                        gate_decay[hi] = a_val;
                        beta_vals[hi] = sigmoid_stable(beta_raw[hi]);
                    }

                    /* L2 normalize Q (16 key heads) and K (16 key heads) */
                    for (hi = 0; hi < DN_NUM_KV_GROUPS; hi++) dn_l2norm(Q + hi * DN_HEAD_DIM, DN_HEAD_DIM);
                    for (hi = 0; hi < DN_NUM_KV_GROUPS; hi++) dn_l2norm(K_ptr + hi * DN_HEAD_DIM, DN_HEAD_DIM);

                    /* State recurrence (AVX2 vectorized).
                     * OMP over heads — each head's state matrix S is independent, no race.
                     * At OMP=2 expect ~1.8x on the per-head work. */
                    float* head_output = dn_states[layer].tmp_head_out;
                    memset(head_output, 0, DN_INNER * sizeof(float));
                    float scale = 1.0f / sqrtf((float)DN_HEAD_DIM);
                    int h_idx;

                    #pragma omp parallel for schedule(static)
                    for (h_idx = 0; h_idx < DN_NUM_Q_HEADS; h_idx++) {
                        int g_idx, d_idx, d2_idx;  /* per-thread locals */
                        g_idx = h_idx % DN_NUM_KV_GROUPS; /* INTERLEAVED — matches llama.cpp ggml_repeat_4d */
                        float* S = dn_states[layer].S + h_idx * DN_HEAD_DIM * DN_HEAD_DIM;
                        float* k = K_ptr + g_idx * DN_HEAD_DIM;  /* shared K from key group */
                        float* v = V_ptr + h_idx * DN_HEAD_DIM;  /* per-value-head V */

                        __m256 vdecay = _mm256_set1_ps(gate_decay[h_idx]);
                        __m256 ret_acc[16];
                        for (d_idx = 0; d_idx < 16; d_idx++) ret_acc[d_idx] = _mm256_setzero_ps();

                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx++) {
                            float* S_row = S + d_idx * DN_HEAD_DIM;
                            __m256 vk = _mm256_set1_ps(k[d_idx]);
                            for (d2_idx = 0; d2_idx < DN_HEAD_DIM; d2_idx += 8) {
                                __m256 vs = _mm256_loadu_ps(S_row + d2_idx);
                                vs = _mm256_mul_ps(vs, vdecay);
                                _mm256_storeu_ps(S_row + d2_idx, vs);
                                ret_acc[d2_idx / 8] = _mm256_fmadd_ps(vk, vs, ret_acc[d2_idx / 8]);
                            }
                        }

                        float retrieved[128], delta[128];
                        for (d_idx = 0; d_idx < 16; d_idx++) _mm256_storeu_ps(retrieved + d_idx * 8, ret_acc[d_idx]);

                        __m256 vbeta = _mm256_set1_ps(beta_vals[h_idx]);
                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx += 8) {
                            __m256 vv = _mm256_loadu_ps(v + d_idx);
                            __m256 vr = _mm256_loadu_ps(retrieved + d_idx);
                            _mm256_storeu_ps(delta + d_idx, _mm256_mul_ps(vbeta, _mm256_sub_ps(vv, vr)));
                        }

                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx++) {
                            float* S_row = S + d_idx * DN_HEAD_DIM;
                            __m256 vk2 = _mm256_set1_ps(k[d_idx]);
                            for (d2_idx = 0; d2_idx < DN_HEAD_DIM; d2_idx += 8) {
                                __m256 vdelta = _mm256_loadu_ps(delta + d2_idx);
                                __m256 vs = _mm256_loadu_ps(S_row + d2_idx);
                                _mm256_storeu_ps(S_row + d2_idx, _mm256_fmadd_ps(vk2, vdelta, vs));
                            }
                        }

                        float* q_head = Q + g_idx * DN_HEAD_DIM; /* Q is per key group */
                        float* o = head_output + h_idx * DN_HEAD_DIM;
                        __m256 out_acc[16];
                        for (d_idx = 0; d_idx < 16; d_idx++) out_acc[d_idx] = _mm256_setzero_ps();
                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx++) {
                            float* S_row = S + d_idx * DN_HEAD_DIM;
                            __m256 vq = _mm256_set1_ps(q_head[d_idx]);
                            for (d2_idx = 0; d2_idx < DN_HEAD_DIM; d2_idx += 8)
                                out_acc[d2_idx / 8] = _mm256_fmadd_ps(vq, _mm256_loadu_ps(S_row + d2_idx), out_acc[d2_idx / 8]);
                        }
                        __m256 vscale = _mm256_set1_ps(scale);
                        for (d_idx = 0; d_idx < 16; d_idx++)
                            _mm256_storeu_ps(o + d_idx * 8, _mm256_mul_ps(out_acc[d_idx], vscale));
                    }

                    /* RMS norm + SiLU gating */
                    float* normed_out = dn_states[layer].tmp_normed;
                    /* T9 DEBUG: state matrix stats at midpoint */
                    if ((tok == 0 || tok == prompt_len - 1) && layer == 0) {
                        float* S0 = dn_states[layer].S;
                        double srms = 0; float smax = 0;
                        for (int si = 0; si < DN_HEAD_DIM * DN_HEAD_DIM; si++) {
                            srms += (double)S0[si] * S0[si];
                            if (fabsf(S0[si]) > smax) smax = fabsf(S0[si]);
                        }
                        fprintf(stderr, "  STATE L%d head0: rms=%.6f max=%.6f\n",
                            layer, sqrtf((float)(srms / (DN_HEAD_DIM * DN_HEAD_DIM))), smax);
                        fprintf(stderr, "  head_out rms=%.6f normed rms=%.6f gated rms=%.6f\n",
                            sqrtf(head_output[0]*head_output[0]), 0.0f, 0.0f);
                    }

                    /* Trace: attn_output (head_output before per-head rmsnorm, llama.cpp's attn_output) */
                    trace_dump("attn_output", layer, tok, head_output, DN_INNER);
                    trace_dump("z", layer, tok, gpu_gate, DN_INNER);

                    float* gated_out = dn_states[layer].tmp_gated;
                    for (hi = 0; hi < DN_NUM_Q_HEADS; hi++) {
                        if (lw->ssm_norm_w)
                            dn_rmsnorm_weighted(normed_out + hi * DN_HEAD_DIM,
                                               head_output + hi * DN_HEAD_DIM, lw->ssm_norm_w, DN_HEAD_DIM);
                        else
                            dn_rmsnorm_simple(normed_out + hi * DN_HEAD_DIM, head_output + hi * DN_HEAD_DIM, DN_HEAD_DIM);
                    }
                    /* Trace: head_output after per-head RMSNorm (pre-silu-gate) */
                    trace_dump("attn_out_normed", layer, tok, normed_out, DN_INNER);
                    for (hi = 0; hi < DN_INNER; hi++) {
                        float zv = gpu_gate[hi];
                        if (zv > 88.0f) zv = 88.0f;
                        if (zv < -88.0f) zv = -88.0f;
                        gated_out[hi] = normed_out[hi] * (zv / (1.0f + expf(-zv)));
                    }
                    /* Trace: final_output = normed * silu(z), input to ssm_out projection */
                    trace_dump("final_output", layer, tok, gated_out, DN_INNER);

                    if (tok == 0 && layer == 0) {
                        float hr = 0; for (i = 0; i < DN_INNER; i++) hr += head_output[i]*head_output[i];
                        float nr = 0; for (i = 0; i < DN_INNER; i++) nr += normed_out[i]*normed_out[i];
                        float gr = 0; for (i = 0; i < DN_INNER; i++) gr += gated_out[i]*gated_out[i];
                        fprintf(stderr, "  head_output rms=%.6f normed rms=%.6f gated rms=%.6f\n",
                            sqrtf(hr/DN_INNER), sqrtf(nr/DN_INNER), sqrtf(gr/DN_INNER));
                        fprintf(stderr, "  head_out[0..3]: %.6f %.6f %.6f %.6f\n",
                            head_output[0], head_output[1], head_output[2], head_output[3]);
                        fprintf(stderr, "  gated[0..3]: %.6f %.6f %.6f %.6f\n",
                            gated_out[0], gated_out[1], gated_out[2], gated_out[3]);
                    }

                    /* ASYNC: Launch SSM Out on GPU */
                    gpu_launch_ssm_out(layer, slot_idx, gated_out, DN_INNER, H);
                    /* SYNC: Wait for SSM Out result */
                    gpu_wait_ssm_out(slot_idx);
                    memcpy(o_out, gpu_get_ssm_out_buf(slot_idx), H * sizeof(float));
                    if (tok == 0 && layer == 0) {
                        fprintf(stderr, "  o_out[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                            o_out[0], o_out[1], o_out[2], o_out[3],
                            o_out[4], o_out[5], o_out[6], o_out[7]);
                        fprintf(stderr, "=== END L0 VALIDATION ===\n");
                    }

                    /* Recurrence temporaries are owned by this layer invocation. */
                    free(gate_decay);
                    free(beta_vals);
                } else if (lw->w_qkv) {
                    /* === CPU FALLBACK: GATED DELTANET === */
                    deltanet_forward(
                        o_out, normed, &dn_states[layer],
                        lw->w_qkv, lw->w_qkv_type,
                        lw->w_attn_gate, lw->w_attn_gate_type,
                        lw->w_alpha, lw->w_alpha_type,
                        lw->w_beta, lw->w_beta_type,
                        lw->w_ssm_out, lw->w_ssm_out_type,
                        lw->ssm_conv1d_w,
                        lw->ssm_a, lw->ssm_dt_bias, lw->ssm_norm_w,
                        H
                    );
                }
            } else if (lw->wq && lw->wk && lw->wv && lw->wo) {
                /* === STANDARD GQA ATTENTION (Qwen3.5: gated + QK norm + partial RoPE) === */
                /* Detect nqh and hd from Q weight dims: Q_out = nqh * hd * 2 (doubled for gate) */
                int nkvh = cfg.num_kv_heads; /* 2 */
                int q_out_dim = lw->wq_rows; /* Total Q+gate output dim from weight tensor */
                /* hd from GGUF key_length, or derive: K_out = nkvh * hd */
                int hd = lw->wk_rows / nkvh; /* wk_rows = K output dim = nkvh * hd */
                int nqh = q_out_dim / (hd * 2); /* Q+gate = nqh * hd * 2 */
                if (hd <= 0 || nqh <= 0) { hd = 256; nqh = 32; q_out_dim = nqh * hd * 2; } /* fallback */
                int kv_out_dim = nkvh * hd;   /* 512 */
                int attn_dim = nqh * hd;      /* 8192 */

                /* Allocate temp buffers */
                float* q_gate_buf = (float*)lm_temp_alloc(q_out_dim * sizeof(float));
                float* k_buf = (float*)lm_temp_alloc(kv_out_dim * sizeof(float));
                float* v_buf = (float*)lm_temp_alloc(kv_out_dim * sizeof(float));
                float* attn_buf = (float*)lm_temp_alloc(attn_dim * sizeof(float));
                float* gate_buf_std = (float*)lm_temp_alloc(attn_dim * sizeof(float));

                if (q_gate_buf && k_buf && v_buf && attn_buf && gate_buf_std) {
                    /* 1. Q+Gate, K, V projections — GPU if available, else CPU */
                    /* DIAG: force CPU path via env var to isolate GPU kernel issues */
                    int force_cpu_gqa = getenv("WINMOE_GQA_CPU") != NULL;
                    int gpu_proj = (use_gpu && !force_cpu_gqa) ? gpu_gqa_projections(layer, normed, H,
                        q_gate_buf, q_out_dim, k_buf, kv_out_dim, v_buf, kv_out_dim) : -1;
                    if (gpu_proj != 0) {
                        /* CPU fallback */
                        quant_matvec(q_gate_buf, lw->wq, normed, q_out_dim, H, lw->wq_type);
                        quant_matvec(k_buf, lw->wk, normed, kv_out_dim, H, lw->wk_type);
                        quant_matvec(v_buf, lw->wv, normed, kv_out_dim, H, lw->wv_type);
                    }

                    /* Trace: raw Q+Gate projection output before deinterleave */
                    trace_dump("Qcur_full", layer, tok, q_gate_buf, q_out_dim);

                    /* 2. Deinterleave Q and Gate from doubled output */
                    /* Layout: [Q_h0(hd), Gate_h0(hd), Q_h1(hd), Gate_h1(hd), ...] */
                    float* q_std = q;  /* reuse pre-allocated q buffer */
                    for (int h = 0; h < nqh; h++) {
                        memcpy(q_std + h * hd, q_gate_buf + h * hd * 2, hd * sizeof(float));
                        memcpy(gate_buf_std + h * hd, q_gate_buf + h * hd * 2 + hd, hd * sizeof(float));
                    }
                    trace_dump("Qcur_reshaped", layer, tok, q_std, attn_dim);
                    trace_dump("gate_reshaped", layer, tok, gate_buf_std, attn_dim);

                    /* 3. QK RMSNorm (per-head, before RoPE) */
                    if (lw->q_norm) {
                        for (int h = 0; h < nqh; h++)
                            rmsnorm(q_std + h * hd, q_std + h * hd, lw->q_norm, hd);
                    }
                    if (lw->k_norm) {
                        for (int h = 0; h < nkvh; h++)
                            rmsnorm(k_buf + h * hd, k_buf + h * hd, lw->k_norm, hd);
                    }

                    /* 4. RoPE — IMROPE = NEOX split-halves for text-only models.
                     * GGML rotates pairs (i, i+n_dims/2) for i in 0..n_dims/2-1
                     * Frequency for pair i: 1/theta^(2i/n_dims)
                     * NOTE: full rope_dim, not just rope_dim/4! Verify n_rot for Qwen3.5 */
                    {
                        int rope_dim = hd / 4; /* 64 — partial rotary 0.25 */
                        int half = rope_dim / 2; /* 32 */
                        float theta = cfg.rope_theta;
                        for (int h = 0; h < nqh; h++) {
                            float* qh = q_std + h * hd;
                            for (int j = 0; j < half; j++) {
                                float freq = 1.0f / powf(theta, (float)(2 * j) / (float)rope_dim);
                                float val = (float)tok * freq;
                                float cv = cosf(val), sv = sinf(val);
                                float r0 = qh[j], r1 = qh[j + half];
                                qh[j]        = r0 * cv - r1 * sv;
                                qh[j + half] = r0 * sv + r1 * cv;
                            }
                        }
                        for (int h = 0; h < nkvh; h++) {
                            float* kh = k_buf + h * hd;
                            for (int j = 0; j < half; j++) {
                                float freq = 1.0f / powf(theta, (float)(2 * j) / (float)rope_dim);
                                float val = (float)tok * freq;
                                float cv = cosf(val), sv = sinf(val);
                                float r0 = kh[j], r1 = kh[j + half];
                                kh[j]        = r0 * cv - r1 * sv;
                                kh[j + half] = r0 * sv + r1 * cv;
                            }
                        }
                    }

                    /* 5. KV cache append */
                    kv_cache_append(&kv_caches[layer], k_buf, v_buf);

                    /* 6. GQA attention */
                    gqa_attention(attn_buf, q_std, &kv_caches[layer], nqh, nkvh, hd);

                    /* Trace: attn_pregate = attention output BEFORE sigmoid gate mul */
                    trace_dump("attn_pregate", layer, tok, attn_buf, attn_dim);

                    /* DEBUG: GQA intermediate magnitudes — dump at tok=0 AND tok=7 (late prompt) */
                    if ((tok == 0 || tok == 7) && layer == 3) {
                        float arms = 0; for (i = 0; i < attn_dim; i++) arms += attn_buf[i]*attn_buf[i];
                        arms = sqrtf(arms / attn_dim);
                        float grms = 0; for (i = 0; i < attn_dim; i++) grms += gate_buf_std[i]*gate_buf_std[i];
                        grms = sqrtf(grms / attn_dim);
                        float krms = 0; for (i = 0; i < kv_out_dim; i++) krms += k_buf[i]*k_buf[i];
                        krms = sqrtf(krms / kv_out_dim);
                        float vrms = 0; for (i = 0; i < kv_out_dim; i++) vrms += v_buf[i]*v_buf[i];
                        vrms = sqrtf(vrms / kv_out_dim);
                        float nrms = 0; for (i = 0; i < H; i++) nrms += normed[i]*normed[i];
                        nrms = sqrtf(nrms / H);
                        fprintf(stderr, "  GQA L3: input_rms=%.4f k_rms=%.4f v_rms=%.4f q+gate_rms=%.4f gate_rms=%.4f attn_pre_gate=%.4f\n",
                            nrms, krms, vrms, sqrtf(arms), grms, arms);
                        fprintf(stderr, "  GQA L3: dims: q_out=%d kv_out=%d attn=%d hd=%d nqh=%d nkvh=%d\n",
                            q_out_dim, kv_out_dim, attn_dim, hd, nqh, nkvh);
                    }

                    /* 7. Sigmoid output gating: attn_out *= sigmoid(gate) — stable */
                    /* Compute gate_sigmoid into a temp buffer for tracing */
                    {
                        /* Emit gate_sigmoid as a tensor too (only if tracing) */
                        if (g_trace_out && tok == g_trace_tok) {
                            float* gs_buf = (float*)lm_temp_alloc(attn_dim * sizeof(float));
                            if (gs_buf) {
                                for (i = 0; i < attn_dim; i++) gs_buf[i] = sigmoid_stable(gate_buf_std[i]);
                                trace_dump("gate_sigmoid", layer, tok, gs_buf, attn_dim);
                                free(gs_buf);
                            }
                        }
                    }
                    for (i = 0; i < attn_dim; i++) {
                        float g = gate_buf_std[i];
                        float sig = sigmoid_stable(g);
                        attn_buf[i] *= sig;
                    }
                    /* Trace: attn_gated = attn_pregate * gate_sigmoid */
                    trace_dump("attn_gated", layer, tok, attn_buf, attn_dim);

                    if ((tok == 0 || tok == 7) && layer == 3) {
                        float arms2 = 0; for (i = 0; i < attn_dim; i++) arms2 += attn_buf[i]*attn_buf[i];
                        arms2 = sqrtf(arms2 / attn_dim);
                        fprintf(stderr, "  GQA L3[tok=%d]: attn_post_gate=%.4f seq_len=%d\n", tok, arms2, kv_caches[layer].len);
                        /* For tok=7: check attention peakedness by computing scores for head 0 */
                        if (tok == 7) {
                            int hd2 = hd;
                            float sc = 1.0f / sqrtf((float)hd2);
                            float* q_head0 = q_std;
                            float max_score = -1e30f, min_score = 1e30f;
                            float scores_dump[20];
                            int seq_len = kv_caches[layer].len;
                            for (int t = 0; t < seq_len && t < 20; t++) {
                                const float* kt = kv_caches[layer].keys + t * nkvh * hd + 0 * hd;
                                double dot = 0.0;
                                for (int d = 0; d < hd2; d++) dot += (double)q_head0[d] * (double)kt[d];
                                scores_dump[t] = (float)(dot * sc);
                                if (scores_dump[t] > max_score) max_score = scores_dump[t];
                                if (scores_dump[t] < min_score) min_score = scores_dump[t];
                            }
                            fprintf(stderr, "  GQA L3 head0 scores[0..%d]: ", seq_len-1);
                            for (int t = 0; t < seq_len && t < 8; t++) fprintf(stderr, "%.3f ", scores_dump[t]);
                            fprintf(stderr, "\n  GQA L3 score range: [%.3f, %.3f] spread=%.3f\n", min_score, max_score, max_score - min_score);
                        }
                    }

                    /* 8. O projection: [attn_dim → hidden_dim] — GPU if available */
                    if (!use_gpu || force_cpu_gqa || gpu_gqa_output(layer, attn_buf, attn_dim, o_out, H) != 0)
                        quant_matvec(o_out, lw->wo, attn_buf, H, attn_dim, lw->wo_type);

                    if (tok == 0 && layer == 3) {
                        float orms = 0; for (i = 0; i < H; i++) orms += o_out[i]*o_out[i];
                        orms = sqrtf(orms / H);
                        fprintf(stderr, "  GQA L3: o_out_rms=%.4f\n", orms);
                    }
                }

                if (q_gate_buf) free(q_gate_buf);
                if (k_buf) free(k_buf);
                if (v_buf) free(v_buf);
                if (attn_buf) free(attn_buf);
                if (gate_buf_std) free(gate_buf_std);
            } else {
                /* Standard attention layer but weights not loaded — zero output */
            }

            /* TEST: zero DeltaNet attention output to isolate */
            static int DISABLE_DELTANET = 0; /* reverted */
            if (DISABLE_DELTANET && lw->is_deltanet) {
                memset(o_out, 0, H * sizeof(float));
            }

            /* Trace: attn_output (= linear_attn_out for DN, attn_output for GQA) */
            trace_dump(lw->is_deltanet ? "linear_attn_out" : "attn_output", layer, tok, o_out, H);

            /* Residual add */
            for (i = 0; i < H; i++) hidden[i] = residual[i] + o_out[i];
            memcpy(residual, hidden, H * sizeof(float));
            trace_dump("attn_residual", layer, tok, hidden, H);
            prof_t1 = lm_clock_now();
            prof_attn_ms += (double)(prof_t1 - prof_t0) / freq * 1000.0;

            /* Post-attention norm (before MoE) */
            if (lw->post_attn_norm) {
                rmsnorm(normed, hidden, lw->post_attn_norm, H);
            } else if (lw->ffn_norm) {
                rmsnorm(normed, hidden, lw->ffn_norm, H);
            } else {
                if (tok == 0) fprintf(stderr, "WARNING: L%d has NO post-attn norm! Using raw hidden.\n", layer);
                memcpy(normed, hidden, H * sizeof(float));
            }
            trace_dump("attn_post_norm", layer, tok, normed, H);

            /* Debug: after attention */
            if (tok == 0 && layer == 0) {
                int nan_attn = 0;
                for (i = 0; i < H; i++) if (hidden[i] != hidden[i]) nan_attn++;
                fprintf(stderr, "  L0 after attn: h[0..3]=%.4f %.4f %.4f %.4f  nan=%d\n",
                        hidden[0], hidden[1], hidden[2], hidden[3], nan_attn);
                fprintf(stderr, "  L0 q[0..3]=%.4f %.4f  k[0..3]=%.4f %.4f\n",
                        q[0], q[1], k_cur[0], k_cur[1]);
                fprintf(stderr, "  L0 attn_out[0..3]=%.4f %.4f  o_out[0..3]=%.4f %.4f\n",
                        attn_out[0], attn_out[1], o_out[0], o_out[1]);
            }

            /* (post_attn_norm already applied above, normed is ready for MoE) */

            /* 2i. Router — find top-K experts */
            prof_t0 = lm_clock_now();
            /* Unused entries stay defined for inherited top-three diagnostics. */
            int expert_ids[16] = {0};
            float expert_weights[16] = {0};
            if (lw->gate_inp) {
                router_topk(normed, lw->gate_inp, lw->gate_type,
                           H, cfg.num_experts, K, expert_ids, expert_weights);
            } else {
                /* Fallback: use first K experts */
                for (i = 0; i < K; i++) { expert_ids[i] = i; expert_weights[i] = 1.0f / K; }
            }

            prof_t1 = lm_clock_now();
            prof_router_ms += (double)(prof_t1 - prof_t0) / freq * 1000.0;

            /* Pre-quantize normed activation to Q8_K once for all experts */
            int q8k_blocks = H / Q8K_QK;
            block_q8_K* normed_q8k = (block_q8_K*)lm_temp_alloc(q8k_blocks * sizeof(block_q8_K));
            if (normed_q8k) quantize_row_q8_K(normed_q8k, normed, H);

            /* T27: Dump first expert's FFN output for layer 0 tok 0 */
            if (tok == 0 && layer == 0) {
                /* Dump MoE input (normed) for Python verification */
                { FILE* fn = fopen("moe_normed_L0.bin","wb"); if(fn){fwrite(normed,sizeof(float),H,fn);fclose(fn);} }
                fprintf(stderr, "  MoE L0: experts=[%d,%d,%d] weights=[%.4f,%.4f,%.4f] per_expert=[%llu,%llu,%llu]\n",
                    expert_ids[0], expert_ids[1], expert_ids[2],
                    expert_weights[0], expert_weights[1], expert_weights[2],
                    (unsigned long long)lw->gate_per_expert,
                    (unsigned long long)lw->up_per_expert,
                    (unsigned long long)lw->down_per_expert);
                { float nrms2=0; for(int ii=0;ii<H;ii++) nrms2+=normed[ii]*normed[ii];
                fprintf(stderr, "  MoE L0: normed_rms=%.4f normed[0..3]=%.4f %.4f %.4f %.4f\n",
                    sqrtf(nrms2/H), normed[0], normed[1], normed[2], normed[3]); }
            }

            /* 2j. Expert FFN for each selected expert */
            memset(moe_out, 0, H * sizeof(float));

            /* Phase 4.3: optionally skip the routed-expert MoE on this layer */
            int skip_routed_moe = (g_moe_skip_every >= 2 &&
                                   (layer % g_moe_skip_every) == (g_moe_skip_every - 1));
            if (skip_routed_moe) {
                if (normed_q8k) free(normed_q8k);
                goto skip_routed_moe_label;
            }

            /* === Cross-expert async prefetch ===
             * Phase 1: classify all K experts (GPU/RAM/miss) and issue all
             *          miss reads concurrently so the SSD queue is deep.
             * Phase 2: walk experts in order; for misses, wait for that
             *          expert's IO to complete then compute.
             * Compute time on expert N can overlap SSD IO for expert N+1. */
            typedef struct {
                AsyncRead ar_gate, ar_up, ar_down;
                void* slot;
                void* gate_io_buf;
                void* up_io_buf;
                void* down_io_buf;
                int issued;       /* 1 if 3 reads were issued */
            } PendingRead;
            PendingRead pending[16];
            for (int ek = 0; ek < K; ek++) {
                pending[ek].issued = 0;
                pending[ek].slot = NULL;
                pending[ek].gate_io_buf = NULL;
                pending[ek].up_io_buf = NULL;
                pending[ek].down_io_buf = NULL;
                int eid = expert_ids[ek];
                int cache_key = layer * cfg.num_experts + eid;
                /* Phase 4.1: count per-(layer,expert) accesses */
                if (g_expert_trace_enabled
                    && layer < EXPERT_TRACE_MAX_LAYERS
                    && eid >= 0 && eid < EXPERT_TRACE_MAX_EXPERTS) {
                    g_expert_hits[layer * EXPERT_TRACE_MAX_EXPERTS + eid]++;
                }
                /* Skip if GPU-cached or RAM-cached — no read needed */
                int gpu_idx_pre = use_gpu ? gpu_find_cached_expert(layer, eid) : -1;
                if (gpu_idx_pre >= 0) continue;
                if (expert_cache[cache_key]) continue;
                /* Need to read — allocate slot (will become cache entry); reuse pooled IO buffers */
                pending[ek].slot = malloc(expert_total_size);
                pending[ek].gate_io_buf = io_pool[ek][0];
                pending[ek].up_io_buf   = io_pool[ek][1];
                pending[ek].down_io_buf = io_pool[ek][2];
                if (pending[ek].slot && pending[ek].gate_io_buf &&
                    pending[ek].up_io_buf && pending[ek].down_io_buf) {
                    uint64_t g_off = model.shard_data_starts[lw->gate_exps_shard] + lw->gate_exps_offset +
                                     (uint64_t)eid * lw->gate_per_expert;
                    uint64_t u_off = model.shard_data_starts[lw->up_exps_shard] + lw->up_exps_offset +
                                     (uint64_t)eid * lw->up_per_expert;
                    uint64_t d_off = model.shard_data_starts[lw->down_exps_shard] + lw->down_exps_offset +
                                     (uint64_t)eid * lw->down_per_expert;
                    async_read_start(&pending[ek].ar_gate, lw->gate_exps_shard, g_off,
                                     pending[ek].slot, (int)lw->gate_per_expert,
                                     pending[ek].gate_io_buf, expert_buf_size);
                    async_read_start(&pending[ek].ar_up, lw->up_exps_shard, u_off,
                                     (char*)pending[ek].slot + lw->gate_per_expert,
                                     (int)lw->up_per_expert,
                                     pending[ek].up_io_buf, expert_buf_size);
                    async_read_start(&pending[ek].ar_down, lw->down_exps_shard, d_off,
                                     (char*)pending[ek].slot + lw->gate_per_expert + lw->up_per_expert,
                                     (int)lw->down_per_expert,
                                     pending[ek].down_io_buf, expert_buf_size);
                    pending[ek].issued = 1;
                }
            }

            for (int ek = 0; ek < K; ek++) {
                int eid = expert_ids[ek];

                /* === Expert I/O + Compute with cache === */
                prof_t0 = lm_clock_now();

                int cache_key = layer * cfg.num_experts + eid;

                /* Check GPU expert cache first (256 GB/s vs 38 GB/s) */
                int gpu_idx = use_gpu ? gpu_find_cached_expert(layer, eid) : -1;
                if (gpu_idx >= 0) {
                    /* GPU EXPERT HIT — fully fused: gate+up+swiglu+down all on GPU */
                    gpu_expert_ffn_fused(gpu_idx, normed, H, I, expert_out);

                    prof_t1 = lm_clock_now();
                    prof_expert_io_ms += (double)(prof_t1 - prof_t0) / freq * 1000.0;

                    /* Accumulate */
                    __m256 vw = _mm256_set1_ps(expert_weights[ek]);
                    for (i = 0; i + 7 < H; i += 8) {
                        __m256 vm = _mm256_loadu_ps(moe_out + i);
                        __m256 ve = _mm256_loadu_ps(expert_out + i);
                        _mm256_storeu_ps(moe_out + i, _mm256_fmadd_ps(vw, ve, vm));
                    }
                    for (; i < H; i++) moe_out[i] += expert_weights[ek] * expert_out[i];
                    continue; /* skip CPU path */
                }

                void* cached = expert_cache[cache_key];
                void* gate_data; void* up_data; void* down_data;

                if (cached) {
                    /* RAM CACHE HIT */
                    gate_data = cached;

                    /* Try to promote to GPU cache. Limit + LRU eviction are handled
                     * inside gpu_cache_expert itself (see gpu_set_expert_limit in
                     * main() which reads WINMOE_GPU_EXPERTS env). */
                    if (use_gpu) {
                        gpu_cache_expert(layer, eid,
                            cached, (int)lw->gate_per_expert,
                            (char*)cached + lw->gate_per_expert, (int)lw->up_per_expert,
                            (char*)cached + lw->gate_per_expert + lw->up_per_expert, (int)lw->down_per_expert);
                    }
                    up_data = (char*)cached + lw->gate_per_expert;
                    down_data = (char*)cached + lw->gate_per_expert + lw->up_per_expert;
                    cache_hits++;
                    /* Phase 4.2: bump LFU hit counter via key_to_slot reverse map */
                    if (cache_policy_lfu) {
                        int slot = key_to_slot[cache_key];
                        if (slot >= 0 && slot < max_cached) cache_hit_counts[slot]++;
                    }
                } else {
                    /* CACHE MISS — reads were issued in Phase 1, just wait for completion */
                    if (pending[ek].issued) {
                        async_read_wait(&pending[ek].ar_gate);
                        async_read_wait(&pending[ek].ar_up);
                        async_read_wait(&pending[ek].ar_down);
                    }
                    /* IO buffers are pooled (reused next iter) — DO NOT free */

                    if (pending[ek].slot && pending[ek].issued) {
                        gate_data = pending[ek].slot;
                        up_data = (char*)pending[ek].slot + lw->gate_per_expert;
                        down_data = (char*)pending[ek].slot + lw->gate_per_expert + lw->up_per_expert;
                        if (cache_stored < max_cached) {
                            int slot = cache_stored;
                            expert_cache[cache_key] = pending[ek].slot;
                            cached_keys[slot] = cache_key;
                            if (cache_policy_lfu) {
                                key_to_slot[cache_key] = slot;
                                cache_hit_counts[slot] = 1;
                            }
                            cache_stored++;
                        } else {
                            int evict_slot;
                            int min_hits_for_seed = 1;
                            if (cache_policy_lfu) {
                                /* Pick slot with min hit_count; tiebreak on first encountered */
                                evict_slot = 0;
                                int min_hits = cache_hit_counts[0];
                                for (int s = 1; s < max_cached; s++) {
                                    if (cache_hit_counts[s] < min_hits) {
                                        min_hits = cache_hit_counts[s];
                                        evict_slot = s;
                                    }
                                }
                                /* LFU-DA: seed new arrivals at (min+1) so they survive the
                                 * next eviction round instead of being trivially the min.
                                 * Without this, every new miss evicts itself before being hit. */
                                min_hits_for_seed = min_hits + 1;
                            } else {
                                evict_slot = evict_ptr;
                                evict_ptr = (evict_ptr + 1) % max_cached;
                            }
                            int victim_key = cached_keys[evict_slot];
                            if (victim_key >= 0 && expert_cache[victim_key]) {
                                free(expert_cache[victim_key]);
                                expert_cache[victim_key] = NULL;
                                if (cache_policy_lfu) key_to_slot[victim_key] = -1;
                            }
                            expert_cache[cache_key] = pending[ek].slot;
                            cached_keys[evict_slot] = cache_key;
                            if (cache_policy_lfu) {
                                key_to_slot[cache_key] = evict_slot;
                                cache_hit_counts[evict_slot] = min_hits_for_seed;
                            }
                        }
                    } else {
                        /* A missing read is fatal: the scratch buffer is not weights. */
                        fprintf(stderr, "LinMoE: expert allocation/read failed layer=%d expert=%d\n", layer, eid);
                        exit(EXIT_FAILURE);
                    }
                    cache_misses++;
                }

                /* Fused gate+up matmul — single OMP region, shared activation */
                if (normed_q8k && lw->gate_exps_type == GGML_TYPE_Q4_K
                              && lw->up_exps_type == GGML_TYPE_Q4_K) {
                    /* Fused gate+up with prefetch */
                    int bpr = H / Q4K_QK;
                    const block_q4_K* gw = (const block_q4_K*)gate_data;
                    const block_q4_K* uw = (const block_q4_K*)up_data;
                    int frow;
                    #pragma omp parallel for schedule(static)
                    for (frow = 0; frow < I; frow++) {
                        /* Prefetch next row's first blocks */
                        if (frow + 1 < I) {
                            _mm_prefetch((const char*)&gw[(frow+1) * bpr], _MM_HINT_T0);
                            _mm_prefetch((const char*)&uw[(frow+1) * bpr], _MM_HINT_T0);
                        }
                        float gsum = 0.0f, usum = 0.0f;
                        int fb;
                        for (fb = 0; fb < bpr; fb++) {
                            gsum += q4k_dot_q8k(&gw[frow * bpr + fb], &normed_q8k[fb]);
                            usum += q4k_dot_q8k(&uw[frow * bpr + fb], &normed_q8k[fb]);
                        }
                        gate_buf[frow] = gsum;
                        up_buf[frow] = usum;
                    }
                } else {
                    quant_matvec(gate_buf, gate_data, normed, I, H, lw->gate_exps_type);
                    quant_matvec(up_buf, up_data, normed, I, H, lw->up_exps_type);
                }

                /* SwiGLU: act = silu(gate) * up — stable sigmoid */
                for (i = 0; i < I; i++) {
                    float g = gate_buf[i];
                    act_buf[i] = g * sigmoid_stable(g) * up_buf[i];
                }

                /* T27: dump first CPU expert's intermediate values */
                if (tok == 0 && layer == 0 && ek == 0) {
                    float grms=0,urms=0,arms=0;
                    for (i=0;i<I;i++) {grms+=gate_buf[i]*gate_buf[i]; urms+=up_buf[i]*up_buf[i]; arms+=act_buf[i]*act_buf[i];}
                    fprintf(stderr, "  Expert[%d] CPU: gate_rms=%.4f up_rms=%.4f act_rms=%.4f\n",
                        eid, sqrtf(grms/I), sqrtf(urms/I), sqrtf(arms/I));
                    fprintf(stderr, "  Expert[%d] CPU: gate[0..3]=%.4f %.4f %.4f %.4f\n",
                        eid, gate_buf[0], gate_buf[1], gate_buf[2], gate_buf[3]);
                    /* Verify Q4_K block header */
                    const unsigned char* graw = (const unsigned char*)gate_data;
                    uint16_t gd16; memcpy(&gd16, graw, 2);
                    uint16_t gdm16; memcpy(&gdm16, graw+2, 2);
                    fprintf(stderr, "  Expert[%d] Q4K block0: d=0x%04x dmin=0x%04x scales[0..3]=%d,%d,%d,%d\n",
                        eid, gd16, gdm16, graw[4], graw[5], graw[6], graw[7]);
                }

                /* Down matmul — pre-quantize act_buf for Q5_K integer path */
                if (lw->down_exps_type == GGML_TYPE_Q5_K) {
                    int act_q8k_blocks = I / Q8K_QK;
                    block_q8_K act_q8k[4]; /* I=1024 → 4 blocks, stack-safe */
                    quantize_row_q8_K(act_q8k, act_buf, I);
                    q5k_matvec_q8k(expert_out, down_data, act_q8k, H, I);
                } else {
                    quant_matvec(expert_out, down_data, act_buf, H, I, lw->down_exps_type);
                }

                prof_t1 = lm_clock_now();
                prof_expert_io_ms += (double)(prof_t1 - prof_t0) / freq * 1000.0;

                /* Accumulate weighted expert output — AVX2 */
                {
                    __m256 vw = _mm256_set1_ps(expert_weights[ek]);
                    for (i = 0; i + 7 < H; i += 8) {
                        __m256 vm = _mm256_loadu_ps(moe_out + i);
                        __m256 ve = _mm256_loadu_ps(expert_out + i);
                        _mm256_storeu_ps(moe_out + i, _mm256_fmadd_ps(vw, ve, vm));
                    }
                    for (; i < H; i++) moe_out[i] += expert_weights[ek] * expert_out[i];
                }
            }

            if (normed_q8k) free(normed_q8k);

        skip_routed_moe_label:
            /* Trace: ffn_moe_out = routed-experts sum only (BEFORE shared expert) */
            trace_dump("ffn_moe_out", layer, tok, moe_out, H);

            /* 2j-shared. Shared expert FFN (always active, added to moe_out) */
            if (lw->shexp_gate && lw->shexp_up && lw->shexp_down) {
                /* Gate + Up projections: [hidden → intermediate] */
                quant_matvec(gate_buf, lw->shexp_gate, normed, I, H, lw->shexp_gate_type);
                quant_matvec(up_buf, lw->shexp_up, normed, I, H, lw->shexp_up_type);

                /* SwiGLU activation — stable sigmoid */
                for (i = 0; i < I; i++) {
                    float g = gate_buf[i];
                    act_buf[i] = g * sigmoid_stable(g) * up_buf[i];
                }

                /* Down projection: [intermediate → hidden] */
                quant_matvec(expert_out, lw->shexp_down, act_buf, H, I, lw->shexp_down_type);

                /* Apply sigmoid gating to shared expert output — stable */
                if (lw->shexp_gate_inp) {
                    double gate_scalar = 0.0;
                    for (i = 0; i < H; i++) gate_scalar += (double)lw->shexp_gate_inp[i] * (double)normed[i];
                    float sig_gate = sigmoid_stable((float)gate_scalar);
                    /* Dump shared expert gate distribution for diagnostics */
                    if (tok == 0 && (layer == 0 || layer == 10 || layer == 30 || layer == 59)) {
                        fprintf(stderr, "  SHEXP_GATE L%d: scalar=%.4f sigmoid=%.4f\n",
                            layer, (float)gate_scalar, sig_gate);
                    }
                    trace_dump_scalar("shared_expert_gate",         layer, tok, (float)gate_scalar);
                    trace_dump_scalar("shared_expert_gate_sigmoid", layer, tok, sig_gate);
                    for (i = 0; i < H; i++) expert_out[i] *= sig_gate;
                    trace_dump("ffn_shexp_gated", layer, tok, expert_out, H);
                }

                /* Add gated shared expert output to moe_out */
                for (i = 0; i + 7 < H; i += 8) {
                    __m256 vm = _mm256_loadu_ps(moe_out + i);
                    __m256 ve = _mm256_loadu_ps(expert_out + i);
                    _mm256_storeu_ps(moe_out + i, _mm256_add_ps(vm, ve));
                }
                for (; i < H; i++) moe_out[i] += expert_out[i];
            }

            /* Dump MoE output for layer 0 tok 0 */
            if (tok == 0 && layer == 0) {
                FILE* fm = fopen("moe_out_L0.bin","wb"); if(fm){fwrite(moe_out,sizeof(float),H,fm);fclose(fm);}
                float mrms=0; for(i=0;i<H;i++) mrms+=moe_out[i]*moe_out[i];
                fprintf(stderr, "  MoE output L0: rms=%.4f [0..3]=%.4f %.4f %.4f %.4f\n",
                    sqrtf(mrms/H), moe_out[0], moe_out[1], moe_out[2], moe_out[3]);
            }

            /* Trace: ffn_out = routed + shared (final FFN output) */
            trace_dump("ffn_out", layer, tok, moe_out, H);

            /* 2k. Residual add */
            for (i = 0; i < H; i++) hidden[i] = residual[i] + moe_out[i];
            trace_dump("l_out", layer, tok, hidden, H);

            /* DEBUG: track hidden state magnitude per layer */
            if (tok == 7 || (tok == 0 && layer <= 5)) { /* tok 7 = last prompt token */
                float hmag = 0; for (i = 0; i < H; i++) hmag += hidden[i] * hidden[i];
                hmag = sqrtf(hmag / H);
                float omag = 0; for (i = 0; i < H; i++) omag += o_out[i] * o_out[i];
                omag = sqrtf(omag / H);
                float mmag = 0; for (i = 0; i < H; i++) mmag += moe_out[i] * moe_out[i];
                mmag = sqrtf(mmag / H);
                fprintf(stderr, "  L%d: h_rms=%.2f attn_rms=%.4f moe_rms=%.4f%s\n",
                    layer, hmag, omag, mmag, layers[layer].is_deltanet ? " [DN]" : " [GQA]");
            }

            /* Dump hidden state trajectory for tok=0 ALL layers */
            if (tok == 0) {
                char fname[64];
                snprintf(fname, 64, "hidden_L%d.bin", layer);
                FILE* fh = fopen(fname, "wb");
                if (fh) { fwrite(hidden, sizeof(float), H, fh); fclose(fh); }
                float hrms2 = 0; for (i = 0; i < H; i++) hrms2 += hidden[i]*hidden[i];
                fprintf(stderr, "  DUMP L%d: h[0..3]=%.6f %.6f %.6f %.6f rms=%.4f\n",
                    layer, hidden[0], hidden[1], hidden[2], hidden[3], sqrtf(hrms2/H));
            }

            /* Debug: compact per-layer trace for last prompt token */
            if (tok == prompt_len - 1 && layer % 4 == 0) {
                float hrms = 0; for (i = 0; i < H; i++) hrms += hidden[i]*hidden[i];
                fprintf(stderr, "  [T%d L%d] h[0]=%.3f h_rms=%.2f\n", tok, layer, hidden[0], sqrtf(hrms/H));
            }

            /* Debug: trace per layer */
            if (tok == 0 && (layer == 0 || layer == cfg.num_layers - 1)) {
                float hmax = 0;
                int has_nan_h = 0;
                for (i = 0; i < H; i++) {
                    if (hidden[i] != hidden[i]) has_nan_h = 1;
                    if (fabsf(hidden[i]) > hmax) hmax = fabsf(hidden[i]);
                }
                fprintf(stderr, "  L%d after MoE: h[0..3]=%.4f %.4f %.4f %.4f  max=%.4f nan=%d  experts=%d,%d,%d\n",
                        layer, hidden[0], hidden[1], hidden[2], hidden[3],
                        hmax, has_nan_h, expert_ids[0], expert_ids[1], expert_ids[2]);
            }
        }

        /* Debug: pre-norm hidden state statistics */
        if (tok == prompt_len) {
            float hmin=1e30f, hmax=-1e30f, hsum=0, hsumsq=0;
            for (int ii=0; ii<H; ii++) {
                if (hidden[ii]<hmin) hmin=hidden[ii];
                if (hidden[ii]>hmax) hmax=hidden[ii];
                hsum+=hidden[ii]; hsumsq+=hidden[ii]*hidden[ii];
            }
            fprintf(stderr, "PRE-NORM hidden: min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
                hmin, hmax, hsum/H, sqrtf(hsumsq/H));
        }

        /* 3. Final norm */
        if (final_norm) {
            rmsnorm(normed, hidden, final_norm, H);
        } else {
            memcpy(normed, hidden, H * sizeof(float));
        }
        trace_dump("result_norm", -1, tok, normed, H);

        /* §1.8 diagnosis: dump full normed vector for element-wise diff vs llama */
        if (g_normed_bin) {
            fwrite(normed, sizeof(float), (size_t)H, g_normed_bin);
            g_normed_rows++;
            g_normed_dim = H;
        }

        /* Debug: hidden state before LM head */
        if (tok <= 12) {
            float hmax = 0; for (i = 0; i < H; i++) if (fabsf(normed[i]) > hmax) hmax = fabsf(normed[i]);
            fprintf(stderr, "  Pre-LM[%d]: normed[0..3]=%.4f %.4f %.4f %.4f  max=%.4f\n",
                    tok, normed[0], normed[1], normed[2], normed[3], hmax);
        }

        /* DEBUG: check Q6_K scale factor for LM head row 0 */
        if (tok == prompt_len && lm_head && lm_head_type == 14) {
            const unsigned char* lm_raw = (const unsigned char*)lm_head;
            /* Q6_K block: ql[128] + qh[64] + scales[16] + d(2) */
            /* d is at offset 128+64+16 = 208 */
            uint16_t d16;
            memcpy(&d16, lm_raw + 208, 2);
            fprintf(stderr, "LM head Q6K block0: d16=0x%04x scales[0]=%d scales[1]=%d\n",
                d16, (int)(signed char)lm_raw[192], (int)(signed char)lm_raw[193]);
        }

        /* 4. LM head — project to vocab */
        if (lm_head) {
            /* T26 DEBUG: verify Q6_K + check LM head row norms */
            if (tok == 0 && lm_head_type == 14) {
                int blocks_per_row = H / 256;
                const block_q6_K* lm_blocks = (const block_q6_K*)lm_head;
                int test_tokens[] = {32, 53, 872, 198, 13, 7294, 2245, 154378};
                for (int tt = 0; tt < 8; tt++) {
                    int tid = test_tokens[tt];
                    /* Diagnostic IDs belong to Qwen's vocabulary, not every fixture. */
                    if (tid >= vocab_size) continue;
                    double dot = 0.0;
                    double row_norm_sq = 0.0;
                    /* Dequant entire row and compute both dot product and L2 norm */
                    for (int b = 0; b < blocks_per_row; b++) {
                        const block_q6_K* blk = &lm_blocks[tid * blocks_per_row + b];
                        float dequant[256];
                        /* Simple Q6_K dequant for norm computation */
                        float d_val = fp16_to_fp32(blk->d);
                        for (int sub = 0; sub < 2; sub++) {
                            for (int grp = 0; grp < 4; grp++) {
                                int sc_idx = sub * 8 + grp * 2;
                                float sc = d_val * (float)(blk->scales[sc_idx] - 32);
                                for (int j = 0; j < 32; j++) {
                                    int idx = sub * 128 + grp * 32 + j;
                                    int ql_byte = blk->ql[sub*64 + grp*16 + (j < 16 ? j : j-16)];
                                    int ql_val = (j < 16) ? (ql_byte & 0xF) : (ql_byte >> 4);
                                    int qh_byte = blk->qh[sub*32 + grp*8 + j/4];
                                    int qh_val = (qh_byte >> ((j%4)*2)) & 3;
                                    int q = ql_val | (qh_val << 4);
                                    dequant[idx] = sc * (float)(q - 32);
                                    row_norm_sq += dequant[idx] * dequant[idx];
                                }
                            }
                        }
                        dot += q6k_dot_block(blk, &normed[b * 256]);
                    }
                    fprintf(stderr, "  LM[%d]: dot=%.4f row_norm=%.4f\n", tid, (float)dot, (float)sqrt(row_norm_sq));
                }
            }

            int64_t lm_t0, lm_t1;
            lm_t0 = lm_clock_now();
            quant_matvec(logits, lm_head, normed, vocab_size, H, lm_head_type);
            lm_t1 = lm_clock_now();
            double lm_ms = (double)(lm_t1 - lm_t0) / freq * 1000.0;
            if (tok >= prompt_len - 1) fprintf(stderr, "  LM_HEAD t%d: %.0f ms\n", tok, lm_ms);
        } else {
            /* LM head not loaded (too large) — use normed[0] as dummy logit */
            for (i = 0; i < vocab_size; i++) logits[i] = normed[i % H];
        }

        /* Parity sprint: dump THIS position's logits vector to binary (float32). */
        if (g_all_logits_bin) {
            fwrite(logits, sizeof(float), (size_t)vocab_size, g_all_logits_bin);
            g_all_logits_rows++;
        }

        /* Debug: logit statistics */
        if (tok == prompt_len - 1 || tok == prompt_len) {
            float lmin=1e30f, lmax=-1e30f, lsum=0, lsumsq=0;
            for (int ii=0; ii<vocab_size; ii++) {
                if (logits[ii]<lmin) lmin=logits[ii];
                if (logits[ii]>lmax) lmax=logits[ii];
                lsum+=logits[ii]; lsumsq+=logits[ii]*logits[ii];
            }
            fprintf(stderr, "LOGITS[tok=%d]: min=%.4f max=%.4f mean=%.4f std=%.4f\n",
                tok, lmin, lmax, lsum/vocab_size, sqrtf(lsumsq/vocab_size - (lsum/vocab_size)*(lsum/vocab_size)));

            /* Trace: full logits vector stats + sentinel logits */
            if (g_trace_out && tok == g_trace_tok) {
                trace_dump("result_output", -1, tok, logits, vocab_size);
                int sentinels[] = {198, 13, 846, 74455, 248045, 248046, 248068};
                int nsent = sizeof(sentinels)/sizeof(sentinels[0]);
                fprintf(g_trace_out, "\n# sentinel logits (tok=%d)\n", tok);
                for (int s = 0; s < nsent; s++)
                    if (sentinels[s] < vocab_size)
                        fprintf(g_trace_out, "SENTINEL\t%d\t%.8g\n", sentinels[s], logits[sentinels[s]]);
                /* top-20 */
                int top_ids[20]; float top_vals[20];
                for (int ii = 0; ii < 20; ii++) { top_ids[ii] = 0; top_vals[ii] = -1e30f; }
                for (int ii = 0; ii < vocab_size; ii++) {
                    if (logits[ii] > top_vals[19]) {
                        top_vals[19] = logits[ii]; top_ids[19] = ii;
                        for (int j = 18; j >= 0; j--) {
                            if (top_vals[j+1] > top_vals[j]) {
                                float tv = top_vals[j]; top_vals[j] = top_vals[j+1]; top_vals[j+1] = tv;
                                int ti = top_ids[j]; top_ids[j] = top_ids[j+1]; top_ids[j+1] = ti;
                            }
                        }
                    }
                }
                fprintf(g_trace_out, "\n# top-20 logits\n");
                for (int k = 0; k < 20; k++)
                    fprintf(g_trace_out, "TOP\t%d\t%d\t%.8g\n", k, top_ids[k], top_vals[k]);
                fflush(g_trace_out);
            }
        }

        /* Debug: top-10 logits + check for expected token */
        if (tok == prompt_len - 1 || tok == prompt_len) {
            /* Find top-10 */
            int top_ids[10] = {0}; float top_vals[10];
            for (i = 0; i < 10; i++) top_vals[i] = -1e30f;
            for (i = 0; i < vocab_size; i++) {
                if (logits[i] > top_vals[9]) {
                    top_vals[9] = logits[i]; top_ids[9] = i;
                    /* Bubble up */
                    for (int j = 8; j >= 0; j--) {
                        if (top_vals[j+1] > top_vals[j]) {
                            float tv = top_vals[j]; top_vals[j] = top_vals[j+1]; top_vals[j+1] = tv;
                            int ti = top_ids[j]; top_ids[j] = top_ids[j+1]; top_ids[j+1] = ti;
                        }
                    }
                }
            }
            fprintf(stderr, "Top-10 logits [tok=%d]:\n", tok);
            for (i = 0; i < 10; i++)
                fprintf(stderr, "  #%d: id=%d logit=%.4f\n", i+1, top_ids[i], top_vals[i]);
            /* Only inspect sentinels present in this model's vocabulary. */
            const int diagnostic_ids[] = {248045, 248068, 248069, 198, 9419};
            for (size_t di = 0; di < sizeof(diagnostic_ids)/sizeof(diagnostic_ids[0]); ++di) {
                int id = diagnostic_ids[di];
                if (id < vocab_size) fprintf(stderr, "  Token %d logit=%.4f\n", id, logits[id]);
            }
            int nan_count = 0;
            for (i = 0; i < vocab_size; i++) if (logits[i] != logits[i]) nan_count++;
            if (nan_count > 0) fprintf(stderr, "WARNING: %d NaN logits!\n", nan_count);
        }

        /* 5. Greedy sample */
        int best_token = 0;
        float best_logit = logits[0];
        for (i = 1; i < vocab_size; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_token = i;
            }
        }

        tok_end = lm_clock_now();
        double tok_ms = (double)(tok_end - tok_start) / freq * 1000.0;

        /* Track logit confidence during prompt (correct token logit + margin) */
        if (tok + 1 < prompt_len) {
            int correct_next = prompt_tokens[tok + 1];
            float correct_logit = logits[correct_next];
            /* Find second-best logit for margin */
            float second_best = -1e30f;
            for (i = 0; i < vocab_size; i++)
                if (i != best_token && logits[i] > second_best) second_best = logits[i];
            fprintf(stderr, "  [CONF t%d] top1=%d(%.2f) correct=%d(%.2f) margin=%.2f max=%.2f\n",
                tok, best_token, best_logit, correct_next, correct_logit,
                best_logit - second_best, best_logit);
        }

        /* During prompt: use next prompt token. After prompt: use generated token. */
        if (tok + 1 < prompt_len) {
            cur_token = prompt_tokens[tok + 1];
            best_token = cur_token;
        } else {
            cur_token = best_token;
        }
        tokens_generated++;

        fprintf(stderr, "Token %d: id=%d (%.1f ms) [attn=%.0f expert=%.0f router=%.0f]%s\n",
                tok, best_token, tok_ms,
                prof_attn_ms, prof_expert_io_ms, prof_router_ms,
                tok < prompt_len ? " [prompt]" : "");
        /* Print token IDs to stdout for output capture */
        printf("%d ", best_token);
        fflush(stdout);

        /* EOS detection: stop after generating <|im_end|> (248046) post-prompt */
        if (tok >= prompt_len && (best_token == 248046 || best_token == 248044)) {
            fprintf(stderr, "EOS token (%d) generated — stopping.\n", best_token);
            break;
        }
    }

    gen_end = lm_clock_now();
    double total_ms = (double)(gen_end - gen_start) / freq * 1000.0;
    double tps = tokens_generated / (total_ms / 1000.0);

    fprintf(stderr, "\n=== Generation Complete ===\n");
    fprintf(stderr, "Tokens: %d in %.1f ms = %.2f tok/s\n", tokens_generated, total_ms, tps);
    fprintf(stderr, "Expert cache: %d hits, %d misses, %d stored (%.1f%% hit rate)\n",
            cache_hits, cache_misses, cache_stored,
            cache_hits + cache_misses > 0 ? 100.0 * cache_hits / (cache_hits + cache_misses) : 0);

    /* JSON output for benchmark */
    printf("{\"tok_s\": %.2f, \"tokens\": %d, \"total_ms\": %.1f, \"status\": \"ok\"}\n",
           tps, tokens_generated, total_ms);

    /* All synchronous CPU reads/compute are complete. Drain the GPU before
     * releasing any host tensors that a backend could still reference. */
    if (use_gpu) gpu_shutdown();
    for (i = 0; i < cfg.num_layers; i++) {
        kv_cache_free(&kv_caches[i]);
        dn_state_free(&dn_states[i]);
        LayerWeights* lw = &layers[i];
        /* Each weight load owns a distinct malloc allocation. */
        free(lw->wq); free(lw->wk); free(lw->wv); free(lw->wo);
        free(lw->attn_norm); free(lw->ffn_norm); free(lw->q_norm); free(lw->k_norm);
        free(lw->gate_inp); free(lw->post_attn_norm);
        free(lw->w_qkv); free(lw->w_attn_gate); free(lw->w_alpha); free(lw->w_beta); free(lw->w_ssm_out);
        free(lw->ssm_a); free(lw->ssm_dt_bias); free(lw->ssm_norm_w); free(lw->ssm_conv1d_w);
        free(lw->shexp_gate); free(lw->shexp_up); free(lw->shexp_down); free(lw->shexp_gate_inp);
    }
    free(dn_states);
    for (i = 0; i < total_slots; ++i) free(expert_cache[i]);
    free(expert_cache); free(cached_keys); free(cache_hit_counts); free(key_to_slot);
    for (i = 0; i < 16; ++i)
        for (int j = 0; j < 3; ++j) lm_aligned_free(io_pool[i][j]);
    free(embd_data); free(final_norm); free(lm_head);
    free(kv_caches);
    lm_aligned_free(hidden); lm_aligned_free(residual); lm_aligned_free(normed);
    lm_aligned_free(q); lm_aligned_free(k_cur); lm_aligned_free(v_cur);
    lm_aligned_free(attn_out); lm_aligned_free(o_out);
    lm_aligned_free(gate_buf); lm_aligned_free(up_buf); lm_aligned_free(act_buf);
    lm_aligned_free(expert_out); lm_aligned_free(moe_out);
    lm_aligned_free(expert_buf);
    free(logits);
    free(layers);
    close_shard_handles(&model);
    if (g_trace_out) fclose(g_trace_out);

    /* Phase 4.1: dump per-(layer,expert) access counter for hot/cold analysis */
    if (g_expert_trace_enabled && g_expert_trace_path) {
        FILE* ef = fopen(g_expert_trace_path, "w");
        if (ef) {
            fprintf(ef, "# layer\texpert\thits\n");
            long total = 0;
            for (int l = 0; l < EXPERT_TRACE_MAX_LAYERS; l++) {
                for (int e = 0; e < EXPERT_TRACE_MAX_EXPERTS; e++) {
                    int h = g_expert_hits[l * EXPERT_TRACE_MAX_EXPERTS + e];
                    if (h > 0) { fprintf(ef, "%d\t%d\t%d\n", l, e, h); total += h; }
                }
            }
            fclose(ef);
            fprintf(stderr, "Expert trace dumped: %ld accesses to %s\n", total, g_expert_trace_path);
        } else {
            fprintf(stderr, "WARN: failed to open %s for expert trace dump\n", g_expert_trace_path);
        }
    }

    /* §1.8 diagnosis: close normed dump */
    if (g_normed_bin) {
        fclose(g_normed_bin);
        g_normed_bin = NULL;
        fprintf(stderr, "Dumped %d normed rows (dim=%d)\n", g_normed_rows, g_normed_dim);
    }

    /* Parity sprint: close binary logit dump + write sidecar JSON */
    if (g_all_logits_bin) {
        fclose(g_all_logits_bin);
        g_all_logits_bin = NULL;
        const char* path = getenv("WINMOE_DUMP_ALL_LOGITS");
        if (path && *path) {
            char sidecar[1024];
            snprintf(sidecar, sizeof(sidecar), "%s.json", path);
            FILE* jf = fopen(sidecar, "w");
            if (jf) {
                fprintf(jf, "{\n");
                fprintf(jf, "  \"n_positions\": %d,\n", g_all_logits_rows);
                fprintf(jf, "  \"vocab_size\": %d,\n", cfg.vocab_size);
                fprintf(jf, "  \"dtype\": \"float32\",\n");
                fprintf(jf, "  \"row_bytes\": %d,\n", cfg.vocab_size * 4);
                fprintf(jf, "  \"prompt_ids\": [");
                for (int k = 0; k < prompt_len; k++)
                    fprintf(jf, "%s%d", k == 0 ? "" : ",", prompt_tokens[k]);
                fprintf(jf, "],\n");
                fprintf(jf, "  \"note\": \"WinMoE-emitted; row i = logits AFTER processing input position i\"\n");
                fprintf(jf, "}\n");
                fclose(jf);
                fprintf(stderr, "Dumped %d logit rows + %s\n", g_all_logits_rows, sidecar);
            }
        }
    }

    #undef model
    free(pmodel);
    return 0;
}
