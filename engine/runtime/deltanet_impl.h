#pragma once
/* Portable heap temporaries must be released before returning. */
#include "../platform/runtime.h"
/*
 * Gated DeltaNet Implementation — The Critical Attention Layer
 *
 * Qwen3.5 uses this for 75% of layers (layers 0,1,2, skip 3, 4,5,6, skip 7, ...)
 *
 * State: 16 groups × [128, 128] per layer = ~1 MB per layer
 * Per token: O(1) state update (no KV cache growth!)
 *
 * References:
 *   - NVlabs/GatedDeltaNet (ICLR 2025)
 *   - justinchuby/qwen3.5-gated-deltanet-analysis
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* DeltaNet config — initialized from GGUF metadata at startup.
 * 397B: 64 value heads, 16 key heads, 128 head dim
 * 35B:  32 value heads, 16 key heads, 128 head dim */
static int DN_NUM_Q_HEADS = 64;     /* num_value_heads — output & state heads */
static int DN_NUM_KV_GROUPS = 16;   /* num_key_heads — K groups */
static int DN_HEADS_PER_GROUP = 4;  /* NUM_Q_HEADS / NUM_KV_GROUPS */
static int DN_HEAD_DIM = 128;       /* both key_head_dim and value_head_dim */
static int DN_INNER = 8192;         /* NUM_Q_HEADS * HEAD_DIM */
static int DN_KEY_DIM = 2048;       /* NUM_KV_GROUPS * HEAD_DIM */
static int DN_QKV_DIM = 12288;      /* KEY_DIM*2 + INNER */
static int DN_GATE_DIM = 8192;      /* same as INNER */
static int DN_NUM_GATES = 64;       /* same as NUM_Q_HEADS */

static void dn_init_dims(int ssm_inner_size, int ssm_group_count, int ssm_head_dim) {
    DN_HEAD_DIM = ssm_head_dim > 0 ? ssm_head_dim : 128;
    DN_NUM_KV_GROUPS = ssm_group_count > 0 ? ssm_group_count : 16;
    DN_INNER = ssm_inner_size > 0 ? ssm_inner_size : 8192;
    DN_NUM_Q_HEADS = DN_INNER / DN_HEAD_DIM;
    DN_HEADS_PER_GROUP = DN_NUM_Q_HEADS / DN_NUM_KV_GROUPS;
    DN_KEY_DIM = DN_NUM_KV_GROUPS * DN_HEAD_DIM;
    DN_QKV_DIM = DN_KEY_DIM * 2 + DN_INNER;
    DN_GATE_DIM = DN_INNER;
    DN_NUM_GATES = DN_NUM_Q_HEADS;
    fprintf(stderr, "DeltaNet dims: heads=%d, kv_groups=%d, head_dim=%d, inner=%d, qkv=%d\n",
        DN_NUM_Q_HEADS, DN_NUM_KV_GROUPS, DN_HEAD_DIM, DN_INNER, DN_QKV_DIM);
}
#define DN_CONV_WIDTH 4

/* Per-layer DeltaNet state */
typedef struct {
    float* S;           /* [num_q_heads, 128, 128] state matrices */
    float* conv_buf;    /* [4, 8192] conv1d sliding window */
    int conv_pos;       /* current position in conv buffer (circular) */
    /* Pre-allocated temp buffers (avoid malloc per forward call) */
    float* tmp_qkv;     /* [DN_QKV_DIM] */
    float* tmp_z;       /* [DN_GATE_DIM] */
    float* tmp_alpha;   /* [DN_NUM_GATES] */
    float* tmp_beta;    /* [DN_NUM_GATES] */
    float* tmp_head_out;/* [DN_INNER] */
    float* tmp_normed;  /* [DN_INNER] */
    float* tmp_gated;   /* [DN_INNER] */
} DeltaNetState;

static void dn_state_init(DeltaNetState* s) {
    /* Per Q-head state (32 matrices), not per group (16). Each head has own decay. */
    s->S = (float*)calloc(DN_NUM_Q_HEADS * DN_HEAD_DIM * DN_HEAD_DIM, sizeof(float));
    s->conv_buf = (float*)calloc(DN_CONV_WIDTH * DN_QKV_DIM, sizeof(float));
    s->conv_pos = 0;
    s->tmp_qkv = (float*)malloc(DN_QKV_DIM * sizeof(float));
    s->tmp_z = (float*)malloc(DN_GATE_DIM * sizeof(float));
    s->tmp_alpha = (float*)malloc(DN_NUM_GATES * sizeof(float));
    s->tmp_beta = (float*)malloc(DN_NUM_GATES * sizeof(float));
    s->tmp_head_out = (float*)calloc(DN_INNER, sizeof(float));
    s->tmp_normed = (float*)malloc(DN_INNER * sizeof(float));
    s->tmp_gated = (float*)malloc(DN_INNER * sizeof(float));
}

static void dn_state_free(DeltaNetState* s) {
    free(s->S);
    free(s->conv_buf);
    free(s->tmp_qkv); free(s->tmp_z);
    free(s->tmp_alpha); free(s->tmp_beta);
    free(s->tmp_head_out); free(s->tmp_normed); free(s->tmp_gated);
}

/* L2 normalize in-place */
static void dn_l2norm(float* x, int n) {
    double ss = 0.0;
    int i;
    for (i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / (sqrtf((float)ss) + 1e-6f);
    for (i = 0; i < n; i++) x[i] *= inv;
}

/* RMS norm (no learned weight version) */
static void dn_rmsnorm_simple(float* out, const float* x, int n) {
    double ss = 0.0;
    int i;
    for (i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (i = 0; i < n; i++) out[i] = x[i] * inv;
}

/* Numerically stable softplus: log(1+exp(x)) for all x */
static inline float softplus_stable(float x) {
    return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

/* Numerically stable sigmoid: 1/(1+exp(-x)) for all x */
static inline float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

/* RMS norm with learned weight */
static void dn_rmsnorm_weighted(float* out, const float* x, const float* w, int n) {
    double ss = 0.0;
    int i;
    for (i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

/*
 * Full Gated DeltaNet forward pass for one token
 *
 * Inputs:
 *   x: [hidden_dim] normalized hidden state
 *   state: per-layer recurrent state
 *
 * Weight data (loaded from GGUF):
 *   w_qkv: attn_qkv.weight [hidden, 8192]
 *   w_gate: attn_gate.weight [hidden, 4096]
 *   w_alpha: ssm_alpha.weight [hidden, 32]
 *   w_beta: ssm_beta.weight [hidden, 32]
 *   w_out: ssm_out.weight [4096, hidden]
 *   a_log: ssm_a [32]
 *   dt_bias: ssm_dt.bias [32]
 *   head_norm: ssm_norm.weight [128]
 *
 * Output: [hidden_dim]
 */
static void deltanet_forward(
    float* output,
    const float* x,
    DeltaNetState* state,
    /* Projection weights */
    const void* w_qkv, int w_qkv_type,
    const void* w_gate, int w_gate_type,
    const void* w_alpha, int w_alpha_type,
    const void* w_beta, int w_beta_type,
    const void* w_out, int w_out_type,
    /* Conv1d weight [4, 12288] FP32 */
    const float* conv1d_w,
    /* Learned parameters */
    const float* a_log,
    const float* dt_bias,
    const float* head_norm,
    int hidden_dim
) {
    int i, g, h, d, d2;

    /* 1. Projections — use pre-allocated buffers */
    float* qkv = state->tmp_qkv;
    float* z = state->tmp_z;
    float* alpha_raw = state->tmp_alpha;
    float* beta_raw = state->tmp_beta;

    /* DEBUG: dump first Q8_0 block of QKV weight */
    {
        static int qkv_dump = 0;
        if (qkv_dump == 0 && w_qkv) {
            const unsigned char* raw = (const unsigned char*)w_qkv;
            fprintf(stderr, "QKV weight block0 hex: ");
            for (int bi = 0; bi < 34; bi++) fprintf(stderr, "%02x ", raw[bi]);
            fprintf(stderr, "\n");
            /* Also dequant and print first few weight values */
            uint16_t d16; memcpy(&d16, raw, 2);
            fprintf(stderr, "QKV w_row0 d16=0x%04x qs[0..7]=%d %d %d %d %d %d %d %d\n",
                d16, (int)(signed char)raw[2], (int)(signed char)raw[3],
                (int)(signed char)raw[4], (int)(signed char)raw[5],
                (int)(signed char)raw[6], (int)(signed char)raw[7],
                (int)(signed char)raw[8], (int)(signed char)raw[9]);
            qkv_dump++;
        }
    }
    quant_matvec(qkv, w_qkv, x, DN_QKV_DIM, hidden_dim, w_qkv_type);
    quant_matvec(z, w_gate, x, DN_GATE_DIM, hidden_dim, w_gate_type);

    if (w_alpha) quant_matvec(alpha_raw, w_alpha, x, DN_NUM_GATES, hidden_dim, w_alpha_type);
    else memset(alpha_raw, 0, DN_NUM_GATES * sizeof(float));

    if (w_beta) quant_matvec(beta_raw, w_beta, x, DN_NUM_GATES, hidden_dim, w_beta_type);
    else memset(beta_raw, 0, DN_NUM_GATES * sizeof(float));

    /* DEBUG: print RMS + normed values for verification */
    {
        static int cpu_dn_call = 0;
        if (cpu_dn_call == 0) {
            /* What input x does the matmul see? x is the normed hidden state */
            double sum_sq = 0;
            for (i = 0; i < hidden_dim; i++) sum_sq += (double)x[i] * x[i];
            float x_rms = sqrtf((float)(sum_sq / hidden_dim));
            fprintf(stderr, "CPU L0: input_x[0..3]=%.6f %.6f %.6f %.6f x_rms=%.6f\n",
                x[0], x[1], x[2], x[3], x_rms);
            fprintf(stderr, "CPU L0: QKV pre-conv[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                qkv[0], qkv[1], qkv[2], qkv[3], qkv[4], qkv[5], qkv[6], qkv[7]);
            cpu_dn_call++;
        }
    }

    /* 1b. Conv1d: depthwise causal convolution on QKV [kernel=4, channels=12288]
     * For autoregressive: sliding window of last 4 QKV vectors.
     * conv_out[ch] = sum_k(conv_weight[k][ch] * qkv_history[(pos-k) % 4][ch])
     * conv_weight is [4, 12288] FP32. */
    if (state->conv_buf && conv1d_w) {
        /* Store current QKV in circular buffer */
        int conv_slot = state->conv_pos % DN_CONV_WIDTH;
        memcpy(state->conv_buf + conv_slot * DN_QKV_DIM, qkv, DN_QKV_DIM * sizeof(float));
        state->conv_pos++;

        /* Apply depthwise conv1d with SiLU activation
         * ggml convention: k=0 → oldest, k=d_conv-1 → newest */
        for (i = 0; i < DN_QKV_DIM; i++) {
            float sum = 0.0f;
            for (int k = 0; k < DN_CONV_WIDTH; k++) {
                int hist_slot = ((state->conv_pos - DN_CONV_WIDTH + k) % DN_CONV_WIDTH + DN_CONV_WIDTH) % DN_CONV_WIDTH;
                sum += conv1d_w[i * DN_CONV_WIDTH + k] * state->conv_buf[hist_slot * DN_QKV_DIM + i];
            }
            /* SiLU activation: x * sigmoid(x) */
            float sig = 1.0f / (1.0f + expf(-sum));
            qkv[i] = sum * sig;
        }
    }

    /* 2. Split QKV: Q[16×128=2048] + K[16×128=2048] + V[64×128=8192] */
    float* Q = qkv;                          /* [2048] = 16 key heads × 128 */
    float* K = qkv + DN_KEY_DIM;             /* [2048] = 16 key heads × 128 */
    float* V = qkv + DN_KEY_DIM + DN_KEY_DIM; /* [8192] = 64 value heads × 128 */

    /* 3. Compute gate parameters per head */
    float* gate_decay = (float*)lm_temp_alloc(DN_NUM_GATES * sizeof(float));
    float* beta = (float*)lm_temp_alloc(DN_NUM_GATES * sizeof(float));

    for (h = 0; h < DN_NUM_GATES; h++) {
        /* Decay: g = exp(ssm_a * softplus(alpha + dt_bias))
         * Stable softplus: log1p(exp(-|x|)) + max(x, 0) */
        float a = (a_log && dt_bias) ?
            expf(a_log[h] * softplus_stable(alpha_raw[h] + dt_bias[h])) :
            0.99f;
        /* Clamp to (0, 1] */
        if (a > 1.0f) a = 1.0f;
        if (a < 0.0f) a = 0.0f;
        if (isnan(a)) a = 0.99f;
        gate_decay[h] = a;

        /* Update rate: stable sigmoid */
        beta[h] = sigmoid_stable(beta_raw[h]);
    }

    /* 4. L2 normalize Q per key-head and K per key-head */
    for (g = 0; g < DN_NUM_KV_GROUPS; g++) {
        dn_l2norm(Q + g * DN_HEAD_DIM, DN_HEAD_DIM);
        dn_l2norm(K + g * DN_HEAD_DIM, DN_HEAD_DIM);
    }

    /* 5. Per-value-head recurrence (64 heads, each with own state + decay) */
    /* Q/K are broadcast from 16 key groups (4 value-heads per key group) */
    float* head_output = state->tmp_head_out;
    memset(head_output, 0, DN_INNER * sizeof(float));
    float scale = 1.0f / sqrtf((float)DN_HEAD_DIM);

    /* Note: do NOT use omp parallel here — the matvec projections above
       already use OpenMP. Nested parallelism causes severe thread contention.
       The recurrence itself is fast enough (~30ms) that parallelizing heads
       within it gives diminishing returns vs the matvec cost (~400ms). */
    for (h = 0; h < DN_NUM_Q_HEADS; h++) {
        g = h / DN_HEADS_PER_GROUP;  /* KV group index — GROUPED order (this GGUF uses original HF order) */
        float* S = state->S + h * DN_HEAD_DIM * DN_HEAD_DIM; /* per-value-head state [128,128] */
        float* k = K + g * DN_HEAD_DIM;   /* shared K from key group */
        float* v = V + h * DN_HEAD_DIM;   /* per-value-head V */
        float g_decay = gate_decay[h];     /* per-head gate */
        float b = beta[h];                 /* per-head beta */

        /* === FUSED AVX2 state operations (row-sequential access) === */
        /* Pass 1: Decay + Retrieve + Update — all in one sweep over S rows */
        float retrieved[128];
        float delta[128];
        __m256 vdecay = _mm256_set1_ps(g_decay);

        /* Initialize retrieve accumulator */
        __m256 ret_acc[16]; /* 16 × 8 = 128 floats */
        for (d = 0; d < 16; d++) ret_acc[d] = _mm256_setzero_ps();

        /* Row-by-row: decay S, accumulate k[row] * S_row into retrieved */
        for (d = 0; d < DN_HEAD_DIM; d++) {
            float* S_row = S + d * DN_HEAD_DIM;
            __m256 vk = _mm256_set1_ps(k[d]);
            for (d2 = 0; d2 < DN_HEAD_DIM; d2 += 8) {
                __m256 vs = _mm256_loadu_ps(S_row + d2);
                vs = _mm256_mul_ps(vs, vdecay);           /* decay */
                _mm256_storeu_ps(S_row + d2, vs);          /* store decayed */
                ret_acc[d2 / 8] = _mm256_fmadd_ps(vk, vs, ret_acc[d2 / 8]); /* retrieve */
            }
        }
        /* Store retrieved result */
        for (d = 0; d < 16; d++) _mm256_storeu_ps(retrieved + d * 8, ret_acc[d]);

        /* Delta = beta * (v - retrieved) */
        __m256 vbeta = _mm256_set1_ps(b);
        for (d = 0; d < DN_HEAD_DIM; d += 8) {
            __m256 vv = _mm256_loadu_ps(v + d);
            __m256 vr = _mm256_loadu_ps(retrieved + d);
            _mm256_storeu_ps(delta + d, _mm256_mul_ps(vbeta, _mm256_sub_ps(vv, vr)));
        }

        /* Update state: S += outer(k, delta) — row-sequential */
        for (d = 0; d < DN_HEAD_DIM; d++) {
            float* S_row = S + d * DN_HEAD_DIM;
            __m256 vk = _mm256_set1_ps(k[d]);
            for (d2 = 0; d2 < DN_HEAD_DIM; d2 += 8) {
                __m256 vdelta = _mm256_loadu_ps(delta + d2);
                __m256 vs = _mm256_loadu_ps(S_row + d2);
                _mm256_storeu_ps(S_row + d2, _mm256_fmadd_ps(vk, vdelta, vs));
            }
        }

        /* Pass 2: Output read — o = S^T @ q (row-broadcast FMA pattern) */
        float* q_head = Q + g * DN_HEAD_DIM; /* Q is per key group, not per value head */
        float* o = head_output + h * DN_HEAD_DIM;
        __m256 out_acc[16];
        for (d = 0; d < 16; d++) out_acc[d] = _mm256_setzero_ps();

        for (d = 0; d < DN_HEAD_DIM; d++) {
            float* S_row = S + d * DN_HEAD_DIM;
            __m256 vq = _mm256_set1_ps(q_head[d]);
            for (d2 = 0; d2 < DN_HEAD_DIM; d2 += 8) {
                __m256 vs = _mm256_loadu_ps(S_row + d2);
                out_acc[d2 / 8] = _mm256_fmadd_ps(vq, vs, out_acc[d2 / 8]);
            }
        }
        __m256 vscale = _mm256_set1_ps(scale);
        for (d = 0; d < 16; d++) {
            _mm256_storeu_ps(o + d * 8, _mm256_mul_ps(out_acc[d], vscale));
        }
    }

    /* 6. Per-head RMS norm */
    float* normed_output = state->tmp_normed;
    for (h = 0; h < DN_NUM_Q_HEADS; h++) {
        if (head_norm) {
            dn_rmsnorm_weighted(normed_output + h * DN_HEAD_DIM,
                               head_output + h * DN_HEAD_DIM,
                               head_norm, DN_HEAD_DIM);
        } else {
            dn_rmsnorm_simple(normed_output + h * DN_HEAD_DIM,
                             head_output + h * DN_HEAD_DIM, DN_HEAD_DIM);
        }
    }

    /* 7. Output gating: normed * silu(z) */
    float* gated = state->tmp_gated;
    for (i = 0; i < DN_INNER; i++) {
        float zv = z[i];
        /* Clamp to prevent overflow in silu */
        if (zv > 88.0f) zv = 88.0f;
        if (zv < -88.0f) zv = -88.0f;
        gated[i] = normed_output[i] * (zv / (1.0f + expf(-zv)));
    }

    /* 8. Output projection: gated[4096] → output[hidden_dim] */
    quant_matvec(output, w_out, gated, hidden_dim, DN_INNER, w_out_type);

    /* State buffers are retained; recurrence temporaries belong to this call. */
    free(gate_decay);
    free(beta);
}
