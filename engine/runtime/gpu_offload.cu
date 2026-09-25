/*
 * GPU Offload v2 — Phase 3 Pipeline Architecture
 *
 * 3 CUDA streams: compute, H2D, D2H
 * Double-buffered pinned memory slots
 * Custom Q8_0 matvec kernel (weights stay in Q8_0 on GPU)
 * Async API: launch → do CPU work → wait
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_offload.h"

#define MAX_LAYERS 64
#define MATVEC_THREADS 256
#define NUM_PIPELINE_SLOTS 2

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* Per-layer GPU weight pointers (Q8_0 on device) */
typedef struct {
    /* DeltaNet weights */
    void* d_qkv;
    void* d_gate;
    void* d_ssm_out;
    int qkv_rows, qkv_cols;
    int gate_rows, gate_cols;
    int ssm_rows, ssm_cols;
    int loaded;
    /* Standard GQA weights (Q8_0) */
    void* d_wq;     /* [hidden, nqh*hd*2] — doubled for gate */
    void* d_wk;     /* [hidden, nkvh*hd] */
    void* d_wv;     /* [hidden, nkvh*hd] */
    void* d_wo;     /* [nqh*hd, hidden] */
    int wq_rows, wq_cols;
    int wk_rows, wk_cols;
    int wv_rows, wv_cols;
    int wo_rows, wo_cols;
    int gqa_loaded;
} LayerGPU;

/* Double-buffered pipeline slot */
typedef struct {
    /* Pinned host memory */
    float* h_normed;
    float* h_qkv_out;
    float* h_gate_out;
    float* h_gated_in;
    float* h_ssm_out;
    /* Device memory */
    float* d_normed;
    float* d_qkv_out;
    float* d_gate_out;
    float* d_gated_in;
    float* d_ssm_out;
    /* CUDA events */
    cudaEvent_t evt_h2d_normed_done;
    cudaEvent_t evt_qkv_gate_done;
    cudaEvent_t evt_d2h_qkv_gate_done;
    cudaEvent_t evt_h2d_gated_done;
    cudaEvent_t evt_ssm_out_done;
    cudaEvent_t evt_d2h_ssm_out_done;
} PipelineSlot;

static cublasHandle_t g_cublas = NULL;
static LayerGPU g_layers[MAX_LAYERS];
static PipelineSlot g_slots[NUM_PIPELINE_SLOTS];
static cudaStream_t g_stream_compute = NULL;
static cudaStream_t g_stream_h2d = NULL;
static cudaStream_t g_stream_d2h = NULL;
static cudaStream_t g_stream_expert = NULL;  /* Dedicated stream for expert FFN */
static int g_initialized = 0;
static float g_vram_used = 0;
static int g_hidden_dim = 4096;
static int g_qkv_dim = 8192;
static int g_gate_dim = 4096;

/* GGML Q4_K/Q5_K scale decode helper (matches get_scale_min_k4) */
__device__ __forceinline__ void gpu_get_scale_min(int sub, const unsigned char* sc, int* scale, int* mn) {
    if (sub < 4) {
        *scale = sc[sub] & 63;
        *mn = sc[sub + 4] & 63;
    } else {
        *scale = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        *mn = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }
}

/* Single-thread per row kernel — absolute minimum correctness reference. */
__global__ void q8_matvec_single_kernel(float* output, const unsigned char* weights,
                                         const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    int blocks_per_row = in_dim / 32;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 34;
    float total = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = row_data + b * 34;
        half d_h;
        memcpy(&d_h, block, 2);
        float d = __half2float(d_h);
        const signed char* qs = (const signed char*)(block + 2);
        const float* xp = input + b * 32;
        float bs = 0.0f;
        for (int j = 0; j < 32; j++) bs += (float)qs[j] * xp[j];
        total += d * bs;
    }
    output[row] = total;
}

/* Simple Q8_0 matvec kernel — one warp per row, NO shared memory caching.
 * Used as a correctness reference to validate the optimized kernel. */
__global__ void q8_matvec_simple_kernel(float* output, const unsigned char* weights,
                                         const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 32;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 34;

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += 32) {
        const unsigned char* block = row_data + b * 34;
        half d_h;
        memcpy(&d_h, block, 2);
        float d = __half2float(d_h);
        const signed char* qs = (const signed char*)(block + 2);
        const float* xp = input + b * 32;
        float bs = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) bs += (float)qs[j] * xp[j];
        partial += d * bs;
    }
    /* Warp reduction (32 threads) */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    if (tid == 0) output[row] = partial;
}

/* Q8_0 matvec kernel — shared memory input caching + vectorized loads */
__global__ void q8_matvec_kernel(float* output, const unsigned char* weights,
                                  const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 32;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 34;

    /* Cache input vector in shared memory (all threads collaborate to load) */
    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += MATVEC_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += MATVEC_THREADS) {
        const unsigned char* block = row_data + b * 34;
        half d_h;
        memcpy(&d_h, block, 2);
        float d = __half2float(d_h);
        const signed char* qs = (const signed char*)(block + 2);
        const float* xp = s_input + b * 32; /* read from shared memory */

        float block_sum = 0.0f;
        for (int j = 0; j < 32; j += 4) {
            block_sum += (float)qs[j] * xp[j] + (float)qs[j+1] * xp[j+1] +
                         (float)qs[j+2] * xp[j+2] + (float)qs[j+3] * xp[j+3];
        }
        partial += d * block_sum;
    }

    /* Warp reduction */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

    /* Block reduction via shared memory */
    __shared__ float shared[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

/*
 * Q4_K matvec CUDA kernel — nibble unpacking + dot product
 * Q4_K block: 144 bytes per 256 weights
 *   d(2) + dmin(2) + scales(12) + qs(128)
 */
__global__ void q4k_matvec_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 144;

    /* Cache input in shared memory */
    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += MATVEC_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += MATVEC_THREADS) {
        const unsigned char* blk = row_data + b * 144;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        /* Decode 6-bit scales/mins from packed 12 bytes */
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;

        float block_sum = 0.0f;
        /* GGML Q4_K layout: 32 bytes per 64-weight chunk.
         * Low nibbles of 32 bytes = sub-block A (32 weights, scale[2*chunk])
         * High nibbles of 32 bytes = sub-block B (next 32 weights, scale[2*chunk+1]) */
        for (int chunk = 0; chunk < 4; chunk++) {
            int scale_lo, min_lo, scale_hi, min_hi;
            gpu_get_scale_min(chunk*2,   sc, &scale_lo, &min_lo);
            gpu_get_scale_min(chunk*2+1, sc, &scale_hi, &min_hi);
            float sc_lo = d * (float)scale_lo, mn_lo = dmin * (float)min_lo;
            float sc_hi = d * (float)scale_hi, mn_hi = dmin * (float)min_hi;

            for (int j = 0; j < 32; j++) {
                unsigned char packed = qs[chunk * 32 + j];
                float w_lo = sc_lo * (float)(packed & 0x0F) - mn_lo;  /* sub-block A */
                float w_hi = sc_hi * (float)(packed >> 4) - mn_hi;    /* sub-block B */
                block_sum += w_lo * s_input[b * 256 + chunk * 64 + j]
                           + w_hi * s_input[b * 256 + chunk * 64 + 32 + j];
            }
        }
        partial += block_sum;
    }

    /* Warp + block reduction (same as Q8_0) */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    __shared__ float shared[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

/* Q5_K matvec CUDA kernel — Q4_K + high bit per weight */
__global__ void q5k_matvec_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 176;

    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += MATVEC_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += MATVEC_THREADS) {
        const unsigned char* blk = row_data + b * 176;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        const unsigned char* sc = blk + 4;
        const unsigned char* qh = blk + 16;  /* 32 bytes of high bits */
        const unsigned char* qs = blk + 48;  /* 128 bytes of low nibbles */

        /* GGML Q5_K: 32 bytes per 64-weight chunk */
        float block_sum = 0.0f;
        for (int chunk = 0; chunk < 4; chunk++) {
            int scale_lo, min_lo, scale_hi, min_hi;
            gpu_get_scale_min(chunk*2,   sc, &scale_lo, &min_lo);
            gpu_get_scale_min(chunk*2+1, sc, &scale_hi, &min_hi);
            float sc_lo = d * (float)scale_lo, mn_lo = dmin * (float)min_lo;
            float sc_hi = d * (float)scale_hi, mn_hi = dmin * (float)min_hi;

            for (int j = 0; j < 32; j++) {
                unsigned char packed = qs[chunk * 32 + j];
                int q_lo = (packed & 0x0F) | (((qh[j] >> (chunk*2)) & 1) << 4);
                int q_hi = (packed >> 4)   | (((qh[j] >> (chunk*2+1)) & 1) << 4);
                float w_lo = sc_lo * (float)q_lo - mn_lo;
                float w_hi = sc_hi * (float)q_hi - mn_hi;
                block_sum += w_lo * s_input[b * 256 + chunk * 64 + j]
                           + w_hi * s_input[b * 256 + chunk * 64 + 32 + j];
            }
        }
        partial += block_sum;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    __shared__ float shared_q5[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) shared_q5[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared_q5[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

/*
 * Expert-optimized Q4_K kernel — 32 threads, sub-block work distribution
 * For small in_dim (e.g. 1024-4096), the standard kernel wastes 94% of threads.
 * This kernel distributes work at sub-block granularity (blocks_per_row * 8 units)
 * and uses warp-only reduction (no block reduction needed for 32 threads).
 */
#define EXPERT_THREADS 32

__global__ void q4k_expert_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 144;

    /* Cache input in shared memory — 32 threads load cooperatively */
    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += EXPERT_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    /* Distribute at chunk level: blocks_per_row * 4 chunks (64 weights each) */
    float partial = 0.0f;
    int total_chunks = blocks_per_row * 4;

    for (int ci = tid; ci < total_chunks; ci += EXPERT_THREADS) {
        int b = ci / 4;
        int chunk = ci % 4;

        const unsigned char* blk = row_data + b * 144;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;

        /* GGML Q4_K: 32 bytes per chunk, low nibbles = sub-block A, high = sub-block B */
        int scale_lo, min_lo, scale_hi, min_hi;
        gpu_get_scale_min(chunk*2,   sc, &scale_lo, &min_lo);
        gpu_get_scale_min(chunk*2+1, sc, &scale_hi, &min_hi);
        float sc_lo = d * (float)scale_lo, mn_lo = dmin * (float)min_lo;
        float sc_hi = d * (float)scale_hi, mn_hi = dmin * (float)min_hi;

        float chunk_sum = 0.0f;
        const unsigned char* qp = qs + chunk * 32;
        int base_idx = b * 256 + chunk * 64;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unsigned char packed = __ldg(qp + j);
            float w_lo = sc_lo * (float)(packed & 0x0F) - mn_lo;
            float w_hi = sc_hi * (float)(packed >> 4) - mn_hi;
            chunk_sum += w_lo * s_input[base_idx + j] + w_hi * s_input[base_idx + 32 + j];
        }
        partial += chunk_sum;
    }

    /* Warp-only reduction (32 threads = 1 warp, no block reduction needed) */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    if (tid == 0) output[row] = partial;
}

/* Expert-optimized Q5_K kernel — same pattern as Q4_K but with high-bit extraction */
__global__ void q5k_expert_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 176;

    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += EXPERT_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    /* GGML Q5_K: 32 bytes per 64-weight chunk, low nibbles = sub-block A, high = sub-block B */
    float partial = 0.0f;
    int total_chunks = blocks_per_row * 4;

    for (int ci = tid; ci < total_chunks; ci += EXPERT_THREADS) {
        int b = ci / 4;
        int chunk = ci % 4;

        const unsigned char* blk = row_data + b * 176;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        const unsigned char* sc = blk + 4;
        const unsigned char* qh = blk + 16;
        const unsigned char* qs = blk + 48;

        int scale_lo, min_lo, scale_hi, min_hi;
        gpu_get_scale_min(chunk*2,   sc, &scale_lo, &min_lo);
        gpu_get_scale_min(chunk*2+1, sc, &scale_hi, &min_hi);
        float sc_lo = d * (float)scale_lo, mn_lo = dmin * (float)min_lo;
        float sc_hi = d * (float)scale_hi, mn_hi = dmin * (float)min_hi;

        float chunk_sum = 0.0f;
        const unsigned char* qp = qs + chunk * 32;
        int base_idx = b * 256 + chunk * 64;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unsigned char packed = __ldg(qp + j);
            /* Q5_K high bits: bit (chunk*2) for low nibbles, bit (chunk*2+1) for high */
            int q_lo = (packed & 0x0F) | (((qh[j] >> (chunk*2)) & 1) << 4);
            int q_hi = (packed >> 4)   | (((qh[j] >> (chunk*2+1)) & 1) << 4);
            float w_lo = sc_lo * (float)q_lo - mn_lo;
            float w_hi = sc_hi * (float)q_hi - mn_hi;
            chunk_sum += w_lo * s_input[base_idx + j] + w_hi * s_input[base_idx + 32 + j];
        }
        partial += chunk_sum;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    if (tid == 0) output[row] = partial;
}

/* SwiGLU kernel — gate * sigmoid(gate) * up, all on GPU */
__global__ void swiglu_kernel(float* output, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g)); /* exact sigmoid — matches CPU */
        output[i] = g * sig * up[i];
    }
}

/* Weighted accumulation: moe_out[i] += weight * expert_out[i] */
__global__ void weighted_add_kernel(float* moe_out, const float* expert_out, float weight, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) moe_out[i] += weight * expert_out[i];
}

/* Zero buffer kernel */
__global__ void zero_kernel(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.0f;
}

/* Forward declarations for io buffers */
static float* d_input_fp32 = NULL;
static float* d_output_fp32 = NULL;
static int d_io_size = 0;

/* Pre-allocated expert FFN buffers (eliminates cudaMalloc/cudaFree per call) */
static float* d_expert_gate = NULL;
static float* d_expert_up = NULL;
static float* d_expert_act = NULL;
static float* d_expert_input = NULL;   /* Dedicated device input for expert stream */
static float* d_expert_output = NULL;  /* Dedicated device output for expert stream */
static float* d_moe_out = NULL;        /* GPU accumulation buffer for batched experts */
static int d_moe_out_size = 0;
static float* h_expert_input = NULL;   /* PINNED host input for truly async H2D */
static float* h_expert_output = NULL;  /* PINNED host output for truly async D2H */
static int d_expert_buf_size = 0;
static int d_expert_io_size = 0;

static int ensure_io_buffers(int max_dim) {
    if (max_dim <= d_io_size) return 0;
    if (d_input_fp32) cudaFree(d_input_fp32);
    if (d_output_fp32) cudaFree(d_output_fp32);
    cudaMalloc(&d_input_fp32, max_dim * sizeof(float));
    cudaMalloc(&d_output_fp32, max_dim * sizeof(float));
    d_io_size = max_dim;
    return 0;
}

static int ensure_expert_buffers(int intermediate) {
    if (intermediate <= d_expert_buf_size) return 0;
    if (d_expert_gate) cudaFree(d_expert_gate);
    if (d_expert_up) cudaFree(d_expert_up);
    if (d_expert_act) cudaFree(d_expert_act);
    cudaMalloc(&d_expert_gate, intermediate * sizeof(float));
    cudaMalloc(&d_expert_up, intermediate * sizeof(float));
    cudaMalloc(&d_expert_act, intermediate * sizeof(float));
    d_expert_buf_size = intermediate;
    return 0;
}

static int ensure_expert_io(int dim) {
    if (dim <= d_expert_io_size) return 0;
    if (d_expert_input) cudaFree(d_expert_input);
    if (d_expert_output) cudaFree(d_expert_output);
    if (h_expert_input) cudaFreeHost(h_expert_input);
    if (h_expert_output) cudaFreeHost(h_expert_output);
    cudaMalloc(&d_expert_input, dim * sizeof(float));
    cudaMalloc(&d_expert_output, dim * sizeof(float));
    cudaMallocHost(&h_expert_input, dim * sizeof(float));   /* Pinned for async */
    cudaMallocHost(&h_expert_output, dim * sizeof(float));  /* Pinned for async */
    d_expert_io_size = dim;
    return 0;
}

/* === GPU Expert Cache === */
/* v10.50 experiment: tried bumping to 500. RESULT: no benefit — VRAM ceiling
 * (DeltaNet 5355 MB + GQA 1594 MB = 6949 MB on RTX 3070 8GB) limits actual
 * caching to ~135 experts regardless of MAX. To grow GPU cache, must first
 * shrink DN/GQA VRAM (via IQ2 cold quant, smaller DN state, or fewer GQA on GPU). */
#define MAX_GPU_EXPERTS 300
typedef struct {
    void* d_gate;    /* Q4_K gate weights on GPU */
    void* d_up;      /* Q4_K up weights on GPU */
    void* d_down;    /* Q5_K down weights on GPU */
    int layer;
    int expert_id;
    int loaded;
    int gate_size;   /* for VRAM accounting on evict */
    int up_size;
    int down_size;
} GPUExpert;

static GPUExpert g_gpu_experts[MAX_GPU_EXPERTS];
static int g_gpu_expert_count = 0;
/* Round-robin eviction pointer (LRU approximation) — mirrors CPU expert cache.
 * When count hits limit, we evict at evict_ptr, replace, increment. */
static int g_gpu_evict_ptr = 0;
/* O(1) lookup table: layer*max_experts + eid -> slot index (-1 if not cached).
 * Allocated on first gpu_cache_expert call once we know the dimensions. */
static int* g_gpu_lookup = NULL;
static int g_gpu_lookup_layers = 0;
static int g_gpu_lookup_experts = 0;

static void gpu_lookup_init(int num_layers, int num_experts) {
    if (g_gpu_lookup) return;  /* already init */
    g_gpu_lookup_layers = num_layers;
    g_gpu_lookup_experts = num_experts;
    int n = num_layers * num_experts;
    g_gpu_lookup = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) g_gpu_lookup[i] = -1;
}

/* Effective GPU expert cache limit. Can be lowered via WINMOE_GPU_EXPERTS env
 * (see winmoe_inference.c); clamp so we never exceed MAX_GPU_EXPERTS. */
static int g_gpu_expert_limit = MAX_GPU_EXPERTS;

/* VRAM safety threshold — don't allow gpu_cache_expert to exceed this (MB).
 * 8192 MB on RTX 3070 Laptop - ~290 MB leaves headroom for kernels + driver.
 * Env override via WINMOE_VRAM_CEILING_MB if tighter machines need it. */
static float g_gpu_vram_ceiling_mb = 7900.0f;

extern "C" void gpu_set_expert_limit(int limit) {
    if (limit < 1) limit = 1;
    if (limit > MAX_GPU_EXPERTS) limit = MAX_GPU_EXPERTS;
    g_gpu_expert_limit = limit;
    fprintf(stderr, "GPU expert cache: limit set to %d (MAX_GPU_EXPERTS=%d)\n",
            g_gpu_expert_limit, MAX_GPU_EXPERTS);
}

extern "C" int gpu_cache_expert(int layer, int expert_id,
    const void* gate_data, int gate_size,
    const void* up_data, int up_size,
    const void* down_data, int down_size)
{
    /* Lazy-init lookup table on first call. Use generous bounds so we don't
     * need to re-allocate: 80 layers × 1024 experts × 4 bytes = 320 KB. */
    if (!g_gpu_lookup) {
        gpu_lookup_init(80, 1024);
    }
    /* Clamp against layer/expert dims */
    if (layer >= g_gpu_lookup_layers || expert_id >= g_gpu_lookup_experts) return -1;

    /* VRAM safety: refuse to add if it would push VRAM over ceiling */
    float new_mb = (gate_size + up_size + down_size) / (1024.0f * 1024.0f);
    if (g_vram_used + new_mb > g_gpu_vram_ceiling_mb) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "GPU VRAM ceiling hit (%.0f MB used + %.0f MB new > %.0f MB)\n",
                    g_vram_used, new_mb, g_gpu_vram_ceiling_mb);
            warned = 1;
        }
        return -1;
    }

    int slot;
    if (g_gpu_expert_count < g_gpu_expert_limit) {
        /* Still growing — use next free slot */
        slot = g_gpu_expert_count;
        g_gpu_expert_count++;
    } else {
        /* Cache full — evict round-robin */
        slot = g_gpu_evict_ptr;
        g_gpu_evict_ptr = (g_gpu_evict_ptr + 1) % g_gpu_expert_limit;

        GPUExpert* old = &g_gpu_experts[slot];
        /* Clear old lookup entry */
        if (old->layer >= 0 && old->expert_id >= 0 &&
            old->layer < g_gpu_lookup_layers && old->expert_id < g_gpu_lookup_experts) {
            g_gpu_lookup[old->layer * g_gpu_lookup_experts + old->expert_id] = -1;
        }
        /* Free old GPU buffers */
        if (old->d_gate) cudaFree(old->d_gate);
        if (old->d_up) cudaFree(old->d_up);
        if (old->d_down) cudaFree(old->d_down);
        g_vram_used -= (old->gate_size + old->up_size + old->down_size) / (1024.0f * 1024.0f);
    }

    GPUExpert* ge = &g_gpu_experts[slot];

    /* Stream-matched async H2D — same g_stream_compute as kernels to avoid
     * cross-stream races. Single sync at end covers all three transfers. */
    CHECK_CUDA(cudaMalloc(&ge->d_gate, gate_size));
    CHECK_CUDA(cudaMemcpyAsync(ge->d_gate, gate_data, gate_size, cudaMemcpyHostToDevice, g_stream_compute));
    CHECK_CUDA(cudaMalloc(&ge->d_up, up_size));
    CHECK_CUDA(cudaMemcpyAsync(ge->d_up, up_data, up_size, cudaMemcpyHostToDevice, g_stream_compute));
    CHECK_CUDA(cudaMalloc(&ge->d_down, down_size));
    CHECK_CUDA(cudaMemcpyAsync(ge->d_down, down_data, down_size, cudaMemcpyHostToDevice, g_stream_compute));
    cudaStreamSynchronize(g_stream_compute);

    ge->layer = layer;
    ge->expert_id = expert_id;
    ge->loaded = 1;
    ge->gate_size = gate_size;
    ge->up_size = up_size;
    ge->down_size = down_size;
    g_gpu_lookup[layer * g_gpu_lookup_experts + expert_id] = slot;
    g_vram_used += new_mb;
    return slot;
}

extern "C" int gpu_find_cached_expert(int layer, int expert_id) {
    /* O(1) lookup via 2D index table (vs former O(n) linear scan). */
    if (!g_gpu_lookup) return -1;
    if (layer >= g_gpu_lookup_layers || expert_id >= g_gpu_lookup_experts) return -1;
    return g_gpu_lookup[layer * g_gpu_lookup_experts + expert_id];
#if 0
    for (int i = 0; i < g_gpu_expert_count; i++) {
        if (g_gpu_experts[i].layer == layer && g_gpu_experts[i].expert_id == expert_id)
            return i;
    }
    return -1;
#endif
}

extern "C" int gpu_expert_ffn(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* gate_out, float* up_out, float* expert_out,
    int gate_type, int down_type)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_io_buffers(hidden_dim > intermediate ? hidden_dim : intermediate);
    ensure_expert_buffers(intermediate);

    /* Upload input once */
    CHECK_CUDA(cudaMemcpy(d_input_fp32, input, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    /* Gate matmul — pre-allocated buffer */
    q4k_matvec_kernel<<<intermediate, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_gate, (const unsigned char*)ge->d_gate, d_input_fp32, hidden_dim, intermediate);

    /* Up matmul — pre-allocated buffer */
    q4k_matvec_kernel<<<intermediate, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_up, (const unsigned char*)ge->d_up, d_input_fp32, hidden_dim, intermediate);

    /* Download gate+up for CPU SwiGLU (legacy path) */
    CHECK_CUDA(cudaMemcpy(gate_out, d_expert_gate, intermediate * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(up_out, d_expert_up, intermediate * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int gpu_expert_down(int cache_idx, const float* act, int intermediate,
    float* output, int hidden_dim)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_io_buffers(hidden_dim > intermediate ? hidden_dim : intermediate);

    CHECK_CUDA(cudaMemcpy(d_input_fp32, act, intermediate * sizeof(float), cudaMemcpyHostToDevice));

    q5k_matvec_kernel<<<hidden_dim, MATVEC_THREADS, intermediate * sizeof(float), g_stream_compute>>>(
        d_output_fp32, (const unsigned char*)ge->d_down, d_input_fp32, intermediate, hidden_dim);

    CHECK_CUDA(cudaMemcpy(output, d_output_fp32, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

/* Fully fused GPU expert: gate+up+swiglu+down all on GPU, single upload/download
 * Runs on g_stream_compute — serialized with DeltaNet gives best throughput
 * (separate streams cause SM contention on single GPU, tested v9.1/v9.2) */
extern "C" int gpu_expert_ffn_fused(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* expert_out)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_expert_buffers(intermediate);
    ensure_expert_io(hidden_dim > intermediate ? hidden_dim : intermediate);

    /* Stream-matched H2D — same stream as the kernels guarantees ordering */
    CHECK_CUDA(cudaMemcpyAsync(d_expert_input, input, hidden_dim * sizeof(float),
                               cudaMemcpyHostToDevice, g_stream_compute));

    /* Gate matmul — expert-optimized kernel (32 threads, full utilization) */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_gate, (const unsigned char*)ge->d_gate, d_expert_input, hidden_dim, intermediate);

    /* Up matmul */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_up, (const unsigned char*)ge->d_up, d_expert_input, hidden_dim, intermediate);

    /* SwiGLU on GPU */
    int swiglu_blocks = (intermediate + EXPERT_THREADS - 1) / EXPERT_THREADS;
    swiglu_kernel<<<swiglu_blocks, EXPERT_THREADS, 0, g_stream_compute>>>(
        d_expert_act, d_expert_gate, d_expert_up, intermediate);

    /* Down matmul — expert-optimized Q5_K */
    q5k_expert_kernel<<<hidden_dim, EXPERT_THREADS, intermediate * sizeof(float), g_stream_compute>>>(
        d_expert_output, (const unsigned char*)ge->d_down, d_expert_act, intermediate, hidden_dim);

    /* Download output on the same stream + sync — guarantees correct ordering */
    CHECK_CUDA(cudaMemcpyAsync(expert_out, d_expert_output, hidden_dim * sizeof(float),
                               cudaMemcpyDeviceToHost, g_stream_compute));
    cudaStreamSynchronize(g_stream_compute);
    return 0;
}

/* === Batched Expert API — single upload/download per layer === */

/* Start a batched expert block: upload input once, zero moe_out on GPU */
extern "C" int gpu_expert_batch_start(const float* normed, int hidden_dim) {
    ensure_expert_buffers(hidden_dim);  /* reuse intermediate buffers */
    ensure_expert_io(hidden_dim);
    if (hidden_dim > d_moe_out_size) {
        if (d_moe_out) cudaFree(d_moe_out);
        cudaMalloc(&d_moe_out, hidden_dim * sizeof(float));
        d_moe_out_size = hidden_dim;
    }
    /* Upload normed ONCE — stream-matched with kernels on g_stream_compute */
    CHECK_CUDA(cudaMemcpyAsync(d_expert_input, normed, hidden_dim * sizeof(float),
                               cudaMemcpyHostToDevice, g_stream_compute));
    /* Zero moe_out accumulator on GPU */
    int zblocks = (hidden_dim + 255) / 256;
    zero_kernel<<<zblocks, 256, 0, g_stream_compute>>>(d_moe_out, hidden_dim);
    return 0;
}

/* Add one expert to the batch — all kernels queued on g_stream_compute, no sync */
extern "C" int gpu_expert_batch_add(int cache_idx, int hidden_dim,
    int intermediate, float weight)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_expert_buffers(intermediate);

    /* Gate matmul — input already on GPU from batch_start */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_gate, (const unsigned char*)ge->d_gate, d_expert_input, hidden_dim, intermediate);

    /* Up matmul */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_up, (const unsigned char*)ge->d_up, d_expert_input, hidden_dim, intermediate);

    /* SwiGLU on GPU */
    int swiglu_blocks = (intermediate + EXPERT_THREADS - 1) / EXPERT_THREADS;
    swiglu_kernel<<<swiglu_blocks, EXPERT_THREADS, 0, g_stream_compute>>>(
        d_expert_act, d_expert_gate, d_expert_up, intermediate);

    /* Down matmul */
    q5k_expert_kernel<<<hidden_dim, EXPERT_THREADS, intermediate * sizeof(float), g_stream_compute>>>(
        d_expert_output, (const unsigned char*)ge->d_down, d_expert_act, intermediate, hidden_dim);

    /* Weighted accumulation on GPU — no download needed */
    int ablocks = (hidden_dim + 255) / 256;
    weighted_add_kernel<<<ablocks, 256, 0, g_stream_compute>>>(
        d_moe_out, d_expert_output, weight, hidden_dim);

    return 0;
}

/* Finish batch: download accumulated moe_out */
extern "C" int gpu_expert_batch_finish(float* moe_out, int hidden_dim) {
    CHECK_CUDA(cudaMemcpy(moe_out, d_moe_out, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int gpu_expert_cache_count(void) { return g_gpu_expert_count; }

/* === GPU Standard Attention (GQA) === */

extern "C" int gpu_upload_gqa_weights(int layer,
    const void* wq_q8, int wq_rows, int wq_cols,
    const void* wk_q8, int wk_rows, int wk_cols,
    const void* wv_q8, int wv_rows, int wv_cols,
    const void* wo_q8, int wo_rows, int wo_cols)
{
    if (layer >= MAX_LAYERS) return -1;
    LayerGPU* lg = &g_layers[layer];

    /* Q8_0: (rows * cols / 32) * 34 bytes */
    int wq_size = (wq_rows * wq_cols / 32) * 34;
    int wk_size = (wk_rows * wk_cols / 32) * 34;
    int wv_size = (wv_rows * wv_cols / 32) * 34;
    int wo_size = (wo_rows * wo_cols / 32) * 34;

    CHECK_CUDA(cudaMalloc(&lg->d_wq, wq_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wq, wq_q8, wq_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_wk, wk_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wk, wk_q8, wk_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_wv, wv_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wv, wv_q8, wv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_wo, wo_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wo, wo_q8, wo_size, cudaMemcpyHostToDevice));

    lg->wq_rows = wq_rows; lg->wq_cols = wq_cols;
    lg->wk_rows = wk_rows; lg->wk_cols = wk_cols;
    lg->wv_rows = wv_rows; lg->wv_cols = wv_cols;
    lg->wo_rows = wo_rows; lg->wo_cols = wo_cols;
    lg->gqa_loaded = 1;

    g_vram_used += (wq_size + wk_size + wv_size + wo_size) / (1024.0f * 1024.0f);
    return 0;
}

/* GPU GQA projections: Q+gate, K, V all on GPU, download results */
extern "C" int gpu_gqa_projections(int layer,
    const float* normed, int hidden_dim,
    float* q_gate_out, int q_gate_dim,
    float* k_out, int k_dim,
    float* v_out, int v_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->gqa_loaded) return -1;
    ensure_io_buffers(q_gate_dim > hidden_dim ? q_gate_dim : hidden_dim);

    /* Upload normed input on the same stream as the kernel */
    CHECK_CUDA(cudaMemcpyAsync(d_input_fp32, normed, hidden_dim * sizeof(float),
                               cudaMemcpyHostToDevice, g_stream_compute));

    /* Pick kernel: 0=optimized, 1=simple (32t/row), 2=single (1t/row) */
    static int kernel_mode = -1;
    if (kernel_mode < 0) {
        const char* m = getenv("WINMOE_GPU_KERNEL");
        kernel_mode = m ? atoi(m) : 0;
        fprintf(stderr, "GQA kernel mode: %d (%s)\n", kernel_mode,
            kernel_mode == 0 ? "OPTIMIZED" : kernel_mode == 1 ? "SIMPLE" : "SINGLE");
    }

    /* Q+Gate projection */
    if (kernel_mode == 2) {
        int blocks = (q_gate_dim + 31) / 32;
        q8_matvec_single_kernel<<<blocks, 32, 0, g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wq, d_input_fp32, hidden_dim, q_gate_dim);
    } else if (kernel_mode == 1) {
        q8_matvec_simple_kernel<<<q_gate_dim, 32, 0, g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wq, d_input_fp32, hidden_dim, q_gate_dim);
    } else {
        q8_matvec_kernel<<<q_gate_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wq, d_input_fp32, hidden_dim, q_gate_dim);
    }
    /* Use cudaMemcpyAsync on the SAME stream as the kernel — guarantees ordering */
    CHECK_CUDA(cudaMemcpyAsync(q_gate_out, d_output_fp32, q_gate_dim * sizeof(float),
                               cudaMemcpyDeviceToHost, g_stream_compute));
    cudaStreamSynchronize(g_stream_compute);

    /* K projection */
    if (kernel_mode == 2) {
        int blocks = (k_dim + 31) / 32;
        q8_matvec_single_kernel<<<blocks, 32, 0, g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wk, d_input_fp32, hidden_dim, k_dim);
    } else if (kernel_mode == 1) {
        q8_matvec_simple_kernel<<<k_dim, 32, 0, g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wk, d_input_fp32, hidden_dim, k_dim);
    } else {
        q8_matvec_kernel<<<k_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wk, d_input_fp32, hidden_dim, k_dim);
    }
    CHECK_CUDA(cudaMemcpyAsync(k_out, d_output_fp32, k_dim * sizeof(float),
                               cudaMemcpyDeviceToHost, g_stream_compute));
    cudaStreamSynchronize(g_stream_compute);

    /* V projection */
    if (kernel_mode == 2) {
        int blocks = (v_dim + 31) / 32;
        q8_matvec_single_kernel<<<blocks, 32, 0, g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wv, d_input_fp32, hidden_dim, v_dim);
    } else if (kernel_mode == 1) {
        q8_matvec_simple_kernel<<<v_dim, 32, 0, g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wv, d_input_fp32, hidden_dim, v_dim);
    } else {
        q8_matvec_kernel<<<v_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
            d_output_fp32, (const unsigned char*)lg->d_wv, d_input_fp32, hidden_dim, v_dim);
    }
    CHECK_CUDA(cudaMemcpyAsync(v_out, d_output_fp32, v_dim * sizeof(float),
                               cudaMemcpyDeviceToHost, g_stream_compute));
    cudaStreamSynchronize(g_stream_compute);

    return 0;
}

/* GPU O projection: [attn_dim → hidden_dim] */
extern "C" int gpu_gqa_output(int layer,
    const float* attn_out, int attn_dim,
    float* output, int hidden_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->gqa_loaded) return -1;
    ensure_io_buffers(attn_dim > hidden_dim ? attn_dim : hidden_dim);

    /* Stream-matched: H2D, kernel, D2H all on g_stream_compute, then sync */
    CHECK_CUDA(cudaMemcpyAsync(d_input_fp32, attn_out, attn_dim * sizeof(float),
                               cudaMemcpyHostToDevice, g_stream_compute));
    q8_matvec_kernel<<<hidden_dim, MATVEC_THREADS, attn_dim * sizeof(float), g_stream_compute>>>(
        d_output_fp32, (const unsigned char*)lg->d_wo, d_input_fp32, attn_dim, hidden_dim);
    CHECK_CUDA(cudaMemcpyAsync(output, d_output_fp32, hidden_dim * sizeof(float),
                               cudaMemcpyDeviceToHost, g_stream_compute));
    cudaStreamSynchronize(g_stream_compute);
    return 0;
}

/* === GPU Router === */
static float* d_router_weights[MAX_LAYERS] = {0};
static int d_router_loaded[MAX_LAYERS] = {0};

extern "C" int gpu_upload_router(int layer, const float* weights, int hidden_dim, int num_experts) {
    if (layer >= MAX_LAYERS) return -1;
    int size = hidden_dim * num_experts * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_router_weights[layer], size));
    CHECK_CUDA(cudaMemcpy(d_router_weights[layer], weights, size, cudaMemcpyHostToDevice));
    d_router_loaded[layer] = 1;
    g_vram_used += size / (1024.0f * 1024.0f);
    return 0;
}

/* FP32 matvec kernel for router (same pattern as Q8_0 but simpler — no dequant) */
__global__ void fp32_matvec_kernel(float* output, const float* weights,
                                    const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    const float* row_data = weights + (long long)row * in_dim;

    float partial = 0.0f;
    for (int j = tid; j < in_dim; j += MATVEC_THREADS)
        partial += row_data[j] * input[j];

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    __shared__ float shared[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

extern "C" int gpu_router(int layer, const float* normed, int hidden_dim,
                           float* logits_out, int num_experts) {
    if (!d_router_loaded[layer]) return -1;
    ensure_io_buffers(hidden_dim > num_experts ? hidden_dim : num_experts);

    CHECK_CUDA(cudaMemcpy(d_input_fp32, normed, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    fp32_matvec_kernel<<<num_experts, MATVEC_THREADS, 0, g_stream_compute>>>(
        d_output_fp32, d_router_weights[layer], d_input_fp32, hidden_dim, num_experts);
    CHECK_CUDA(cudaMemcpy(logits_out, d_output_fp32, num_experts * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int gpu_init(void) {
    if (g_initialized) return 0;
    CHECK_CUDA(cudaSetDevice(0));
    cublasCreate(&g_cublas);
    memset(g_layers, 0, sizeof(g_layers));

    /* Create 3 non-blocking streams */
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_h2d, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_d2h, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_expert, cudaStreamNonBlocking));

    /* Allocate double-buffered pipeline slots */
    for (int s = 0; s < NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];
        CHECK_CUDA(cudaMallocHost(&slot->h_normed, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_qkv_out, 16384 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_gate_out, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_gated_in, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_ssm_out, 8192 * sizeof(float)));

        CHECK_CUDA(cudaMalloc(&slot->d_normed, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_qkv_out, 16384 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gate_out, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gated_in, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_ssm_out, 8192 * sizeof(float)));

        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_normed_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_gated_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_ssm_out_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_ssm_out_done, cudaEventDisableTiming));
    }

    g_initialized = 1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU initialized: %s, %.0f MB VRAM, compute %d.%d, 4 streams\n",
            prop.name, prop.totalGlobalMem / (1024.0f * 1024.0f), prop.major, prop.minor);
    fflush(stderr);
    return 0;
}

extern "C" int gpu_upload_deltanet_weights(int layer,
    const void* qkv_q8, int qkv_rows, int qkv_cols,
    const void* gate_q8, int gate_rows, int gate_cols,
    const void* ssm_out_q8, int ssm_rows, int ssm_cols)
{
    if (layer >= MAX_LAYERS) return -1;
    LayerGPU* lg = &g_layers[layer];
    lg->qkv_rows = qkv_rows; lg->qkv_cols = qkv_cols;
    lg->gate_rows = gate_rows; lg->gate_cols = gate_cols;
    lg->ssm_rows = ssm_rows; lg->ssm_cols = ssm_cols;

    int qkv_q8_size = (qkv_rows * qkv_cols / 32) * 34;
    int gate_q8_size = (gate_rows * gate_cols / 32) * 34;
    int ssm_q8_size = (ssm_rows * ssm_cols / 32) * 34;

    CHECK_CUDA(cudaMalloc(&lg->d_qkv, qkv_q8_size));
    CHECK_CUDA(cudaMemcpy(lg->d_qkv, qkv_q8, qkv_q8_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_gate, gate_q8_size));
    CHECK_CUDA(cudaMemcpy(lg->d_gate, gate_q8, gate_q8_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_ssm_out, ssm_q8_size));
    CHECK_CUDA(cudaMemcpy(lg->d_ssm_out, ssm_out_q8, ssm_q8_size, cudaMemcpyHostToDevice));

    g_vram_used += (qkv_q8_size + gate_q8_size + ssm_q8_size) / (1024.0f * 1024.0f);
    lg->loaded = 1;

    g_hidden_dim = qkv_rows;
    g_qkv_dim = qkv_cols;
    g_gate_dim = gate_cols;
    return 0;
}

/* === ASYNC API (Phase 3 Pipeline) === */

extern "C" int gpu_launch_qkv_gate(int layer, int slot_idx,
    const float* normed_src, int hidden_dim, int qkv_dim, int gate_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];

    /* Copy normed into pinned buffer */
    memcpy(slot->h_normed, normed_src, hidden_dim * sizeof(float));

    /* Async H2D */
    cudaMemcpyAsync(slot->d_normed, slot->h_normed,
        hidden_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d);
    cudaEventRecord(slot->evt_h2d_normed_done, g_stream_h2d);

    /* Compute waits for H2D */
    cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_normed_done, 0);

    /* Launch QKV + Gate kernels */
    q8_matvec_kernel<<<qkv_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        slot->d_qkv_out, (const unsigned char*)lg->d_qkv,
        slot->d_normed, hidden_dim, qkv_dim);
    q8_matvec_kernel<<<gate_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        slot->d_gate_out, (const unsigned char*)lg->d_gate,
        slot->d_normed, hidden_dim, gate_dim);
    cudaEventRecord(slot->evt_qkv_gate_done, g_stream_compute);

    /* D2H waits for compute, copies results */
    cudaStreamWaitEvent(g_stream_d2h, slot->evt_qkv_gate_done, 0);
    cudaMemcpyAsync(slot->h_qkv_out, slot->d_qkv_out,
        qkv_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h);
    cudaMemcpyAsync(slot->h_gate_out, slot->d_gate_out,
        gate_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h);
    cudaEventRecord(slot->evt_d2h_qkv_gate_done, g_stream_d2h);

    return 0; /* Returns immediately */
}

extern "C" int gpu_wait_qkv_gate(int slot_idx) {
    cudaEventSynchronize(g_slots[slot_idx].evt_d2h_qkv_gate_done);
    return 0;
}

extern "C" float* gpu_get_qkv_out(int slot_idx) { return g_slots[slot_idx].h_qkv_out; }
extern "C" float* gpu_get_gate_out(int slot_idx) { return g_slots[slot_idx].h_gate_out; }
extern "C" float* gpu_get_ssm_out_buf(int slot_idx) { return g_slots[slot_idx].h_ssm_out; }

extern "C" int gpu_launch_ssm_out(int layer, int slot_idx,
    const float* gated_src, int gated_dim, int output_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];

    memcpy(slot->h_gated_in, gated_src, gated_dim * sizeof(float));

    cudaMemcpyAsync(slot->d_gated_in, slot->h_gated_in,
        gated_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d);
    cudaEventRecord(slot->evt_h2d_gated_done, g_stream_h2d);

    cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_gated_done, 0);
    q8_matvec_kernel<<<output_dim, MATVEC_THREADS, gated_dim * sizeof(float), g_stream_compute>>>(
        slot->d_ssm_out, (const unsigned char*)lg->d_ssm_out,
        slot->d_gated_in, gated_dim, output_dim);
    cudaEventRecord(slot->evt_ssm_out_done, g_stream_compute);

    cudaStreamWaitEvent(g_stream_d2h, slot->evt_ssm_out_done, 0);
    cudaMemcpyAsync(slot->h_ssm_out, slot->d_ssm_out,
        output_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h);
    cudaEventRecord(slot->evt_d2h_ssm_out_done, g_stream_d2h);

    return 0;
}

extern "C" int gpu_wait_ssm_out(int slot_idx) {
    cudaEventSynchronize(g_slots[slot_idx].evt_d2h_ssm_out_done);
    return 0;
}

/* Legacy sync API (calls async then waits — for backward compat) */
extern "C" int gpu_deltanet_projections(int layer,
    const float* normed, int hidden_dim,
    float* qkv_out, int qkv_dim,
    float* gate_out, int gate_dim)
{
    int slot = 0;
    gpu_launch_qkv_gate(layer, slot, normed, hidden_dim, qkv_dim, gate_dim);
    gpu_wait_qkv_gate(slot);
    memcpy(qkv_out, g_slots[slot].h_qkv_out, qkv_dim * sizeof(float));
    memcpy(gate_out, g_slots[slot].h_gate_out, gate_dim * sizeof(float));
    return 0;
}

extern "C" int gpu_ssm_out_projection(int layer,
    const float* gated, int gated_dim,
    float* output, int output_dim)
{
    int slot = 0;
    gpu_launch_ssm_out(layer, slot, gated, gated_dim, output_dim);
    gpu_wait_ssm_out(slot);
    memcpy(output, g_slots[slot].h_ssm_out, output_dim * sizeof(float));
    return 0;
}

extern "C" void gpu_shutdown(void) {
    if (!g_initialized) return;
    cudaDeviceSynchronize();
    for (int i = 0; i < MAX_LAYERS; i++) {
        if (g_layers[i].loaded) {
            cudaFree(g_layers[i].d_qkv);
            cudaFree(g_layers[i].d_gate);
            cudaFree(g_layers[i].d_ssm_out);
        }
    }
    for (int s = 0; s < NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];
        cudaFreeHost(slot->h_normed); cudaFreeHost(slot->h_qkv_out);
        cudaFreeHost(slot->h_gate_out); cudaFreeHost(slot->h_gated_in);
        cudaFreeHost(slot->h_ssm_out);
        cudaFree(slot->d_normed); cudaFree(slot->d_qkv_out);
        cudaFree(slot->d_gate_out); cudaFree(slot->d_gated_in);
        cudaFree(slot->d_ssm_out);
        cudaEventDestroy(slot->evt_h2d_normed_done);
        cudaEventDestroy(slot->evt_qkv_gate_done);
        cudaEventDestroy(slot->evt_d2h_qkv_gate_done);
        cudaEventDestroy(slot->evt_h2d_gated_done);
        cudaEventDestroy(slot->evt_ssm_out_done);
        cudaEventDestroy(slot->evt_d2h_ssm_out_done);
    }
    if (g_stream_compute) cudaStreamDestroy(g_stream_compute);
    if (g_stream_h2d) cudaStreamDestroy(g_stream_h2d);
    if (g_stream_d2h) cudaStreamDestroy(g_stream_d2h);
    if (g_stream_expert) cudaStreamDestroy(g_stream_expert);
    if (g_cublas) cublasDestroy(g_cublas);
    g_initialized = 0;
}

extern "C" int gpu_is_initialized(void) { return g_initialized; }
/* The inherited CUDA backend implements the complete high-level API expected by
 * winmoe_inference.c; this capability distinguishes it from staged backends. */
extern "C" int gpu_supports_full_inference(void) { return 1; }
extern "C" float gpu_vram_used_mb(void) { return g_vram_used; }

/* Q/K/V capability is independent of the full legacy inference pipeline. */
extern "C" int gpu_supports_gqa_projections(void) { return 1; }
