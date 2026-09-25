#pragma once
/*
 * Transformer Layer — Attention + MoE FFN
 *
 * Per layer:
 *   1. x = rmsnorm(residual)
 *   2. q,k,v = attention projections
 *   3. apply RoPE
 *   4. kv cache append
 *   5. attention output
 *   6. o_proj
 *   7. residual += attention_out
 *   8. x = rmsnorm(residual)
 *   9. router → topK experts
 *  10. expert FFN × K
 *  11. weighted combine
 *  12. residual += moe_out
 */

#include "attention.h"

/* Model configuration — auto-detected from GGUF */
typedef struct {
    int hidden_dim;
    int intermediate;
    int num_layers;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int num_experts;
    int expert_k;
    int vocab_size;
    float rope_theta;
    int max_seq_len;
} ModelConfig;

/* Per-layer weight pointers */
typedef struct {
    /* Attention weights (Q4_K or Q5_K) */
    void* wq;      /* [hidden, num_q_heads * head_dim] */
    void* wk;      /* [hidden, num_kv_heads * head_dim] */
    void* wv;      /* [hidden, num_kv_heads * head_dim] */
    void* wo;      /* [num_q_heads * head_dim, hidden] */
    int wq_type, wk_type, wv_type, wo_type;
    int wq_rows, wk_rows; /* output dims for detecting GQA head config */

    /* Norms (FP32) */
    float* attn_norm;  /* [hidden] */
    float* ffn_norm;   /* [hidden] */
    float* q_norm;     /* [head_dim] — Qwen3.5 per-head QK RMSNorm */
    float* k_norm;     /* [head_dim] */

    /* Router (FP32 or FP16) */
    void* gate_inp;    /* [hidden, num_experts] */
    int gate_type;

    /* Expert weights — these come from slab/GGUF via direct I/O */
    /* Not stored here — loaded on demand per expert */

    /* Post-attention norm */
    float* post_attn_norm;

    /* DeltaNet-specific weights */
    int is_deltanet;       /* 1 if DeltaNet layer, 0 if standard attention */
    void* w_qkv;           /* attn_qkv.weight [hidden, 8192] */
    int w_qkv_type;
    void* w_attn_gate;     /* attn_gate.weight [hidden, 4096] */
    int w_attn_gate_type;
    void* w_alpha;         /* ssm_alpha.weight [hidden, 32] */
    int w_alpha_type;
    void* w_beta;          /* ssm_beta.weight [hidden, 32] */
    int w_beta_type;
    void* w_ssm_out;       /* ssm_out.weight [4096, hidden] */
    int w_ssm_out_type;
    float* ssm_a;          /* ssm_a [32] */
    float* ssm_dt_bias;    /* ssm_dt.bias [32] */
    float* ssm_norm_w;     /* ssm_norm.weight [128] */
    float* ssm_conv1d_w;   /* ssm_conv1d.weight [4, 12288] FP32 */

    /* Shared expert (sigmoid-gated, added to routed experts) */
    void* shexp_gate;   /* [hidden, intermediate] Q8_0 */
    void* shexp_up;     /* [hidden, intermediate] Q8_0 */
    void* shexp_down;   /* [intermediate, hidden] Q8_0 */
    int shexp_gate_type, shexp_up_type, shexp_down_type;
    float* shexp_gate_inp; /* [hidden] FP32 — sigmoid gate for shared expert */

    /* Expert tensor info for loading */
    uint64_t gate_exps_offset;  /* offset in GGUF shard */
    uint64_t up_exps_offset;
    uint64_t down_exps_offset;
    int gate_exps_shard;        /* which shard file */
    int up_exps_shard;
    int down_exps_shard;
    uint64_t gate_per_expert;   /* bytes per expert */
    uint64_t up_per_expert;
    uint64_t down_per_expert;
    int gate_exps_type;
    int up_exps_type;
    int down_exps_type;
} LayerWeights;

/* Forward declaration */
static void quant_matvec(float* out, const void* W, const float* x,
                         int out_dim, int in_dim, int type);

/* Router: find top-K experts */
static void router_topk(
    const float* hidden,
    const void* gate_weights,
    int gate_type,
    int hidden_dim,
    int num_experts,
    int K,
    int* expert_ids,
    float* expert_weights
) {
    /* Compute gate logits: hidden @ gate_weights */
    /* Use stack allocation for small expert counts, heap for large */
    float logits_stack[512];
    float* logits = (num_experts <= 512) ? logits_stack : (float*)malloc(num_experts * sizeof(float));

    /* Gate projection — dispatch by type */
    quant_matvec(logits, gate_weights, hidden, num_experts, hidden_dim, gate_type);

    /* TopK selection — find K largest WITHOUT full softmax first */
    /* O(N*K) partial selection is faster than O(N) softmax + O(N*K) scan */
    int k, e;
    for (k = 0; k < K; k++) {
        float best = -1e30f;
        int best_idx = 0;
        for (e = 0; e < num_experts; e++) {
            int already = 0;
            int j;
            for (j = 0; j < k; j++) if (expert_ids[j] == e) { already = 1; break; }
            if (!already && logits[e] > best) { best = logits[e]; best_idx = e; }
        }
        expert_ids[k] = best_idx;
        expert_weights[k] = logits[best_idx];
    }

    /* Softmax only on selected K weights */
    {
        float wmax = expert_weights[0];
        float wsum;
        for (k = 1; k < K; k++) if (expert_weights[k] > wmax) wmax = expert_weights[k];
        wsum = 0.0f;
        for (k = 0; k < K; k++) { expert_weights[k] = expf(expert_weights[k] - wmax); wsum += expert_weights[k]; }
        if (wsum > 0) for (k = 0; k < K; k++) expert_weights[k] /= wsum;
    }

    if (num_experts > 512) free(logits);
}

/*
 * Generic quantized matmul dispatch based on tensor type
 */
static void quant_matvec(float* out, const void* W, const float* x,
                         int out_dim, int in_dim, int type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            q4k_matvec(out, W, x, out_dim, in_dim);
            break;
        case GGML_TYPE_Q5_K:
            q5k_matvec(out, W, x, out_dim, in_dim);
            break;
        case GGML_TYPE_Q6_K:
            q6k_matvec(out, W, x, out_dim, in_dim);
            break;
        case 8: /* Q8_0 */
            q8_matvec(out, W, x, out_dim, in_dim);
            break;
        case GGML_TYPE_F32:
            /* FP32 matmul with AVX-512 + OpenMP */
            {
                const float* wf = (const float*)W;
                int r;
                #pragma omp parallel for schedule(static) if(out_dim >= 64)
                for (r = 0; r < out_dim; r++) {
                    const float* row = wf + r * in_dim;
                    __m512 acc512 = _mm512_setzero_ps();
                    int c;
                    for (c = 0; c + 15 < in_dim; c += 16) {
                        __m512 vw = _mm512_loadu_ps(row + c);
                        __m512 vx = _mm512_loadu_ps(x + c);
                        acc512 = _mm512_fmadd_ps(vw, vx, acc512);
                    }
                    float s = _mm512_reduce_add_ps(acc512);
                    for (; c < in_dim; c++) s += row[c] * x[c];
                    out[r] = s;
                }
            }
            break;
        default:
            /* Parser support does not imply a matvec implementation. Treating
             * an unknown packed type as FP32 reads wrong values/out of bounds. */
            fprintf(stderr, "LinMoE: unsupported matvec type=%d rows=%d cols=%d\n",
                    type, out_dim, in_dim);
            exit(EXIT_FAILURE);
    }
}
