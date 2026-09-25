/*
 * LinMoE staged AMD/HIP backend.
 *
 * This milestone intentionally implements runtime ownership plus isolated Q8_0
 * validation and persistent GQA Q/K/V projections, not the complete CUDA feature
 * set. Full-inference capability stays false so DeltaNet and experts stay on CPU.
 * GQA uses a separate capability and an explicit validation opt-in.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "gpu_offload.h"
#include "gpu_offload_hip_test.h"

namespace {

constexpr int kQ8Values = 32;
constexpr size_t kQ8BlockBytes = 34;
constexpr int kOptimizedThreads = 256;

struct HipBackendState {
    int initialized;
    int device;
    int runtime_version;
    hipDeviceProp_t prop;
    hipStream_t compute;
    hipStream_t h2d;
    hipStream_t d2h;
    size_t allocated_bytes;
};

HipBackendState g_hip = {};

static int report_hip_error(const char* operation, hipError_t error,
                            size_t bytes = 0, int device = -1) {
    fprintf(stderr, "LinMoE HIP: %s failed: code=%d (%s)",
            operation, (int)error, hipGetErrorString(error));
    if (bytes) fprintf(stderr, " bytes=%zu", bytes);
    if (device >= 0) fprintf(stderr, " device=%d", device);
    fputc('\n', stderr);
    return -1;
}

static int check_hip(const char* operation, hipError_t error,
                     size_t bytes = 0, int device = -1) {
    return error == hipSuccess ? 0 : report_hip_error(operation, error, bytes, device);
}

static void destroy_stream_checked(hipStream_t* stream, const char* name) {
    if (!*stream) return;
    hipError_t error = hipStreamDestroy(*stream);
    if (error != hipSuccess) report_hip_error(name, error, 0, g_hip.device);
    *stream = nullptr;
}

/* Initialization failure is transactional: no caller can observe initialized=1
 * until all required properties and streams are valid. */
static void cleanup_partial_init(void) {
    destroy_stream_checked(&g_hip.d2h, "hipStreamDestroy(d2h)");
    destroy_stream_checked(&g_hip.h2d, "hipStreamDestroy(h2d)");
    destroy_stream_checked(&g_hip.compute, "hipStreamDestroy(compute)");
    g_hip = {};
}

static int hip_alloc_tracked(void** ptr, size_t bytes, const char* purpose) {
    if (!g_hip.initialized || !ptr || bytes == 0 || bytes > SIZE_MAX - g_hip.allocated_bytes) {
        fprintf(stderr, "LinMoE HIP: invalid allocation request purpose=%s bytes=%zu\n",
                purpose, bytes);
        return -1;
    }
    hipError_t error = hipMalloc(ptr, bytes);
    if (error != hipSuccess)
        return report_hip_error(purpose, error, bytes, g_hip.device);
    g_hip.allocated_bytes += bytes;
    return 0;
}

static int hip_free_tracked(void* ptr, size_t bytes, const char* purpose) {
    if (!ptr) return 0;
    hipError_t error = hipFree(ptr);
    if (error != hipSuccess)
        return report_hip_error(purpose, error, bytes, g_hip.device);
    if (bytes <= g_hip.allocated_bytes) g_hip.allocated_bytes -= bytes;
    else g_hip.allocated_bytes = 0;
    return 0;
}

static int copy_async_checked(void* dst, const void* src, size_t bytes,
                              hipMemcpyKind kind, hipStream_t stream,
                              const char* operation) {
    hipError_t error = hipMemcpyAsync(dst, src, bytes, kind, stream);
    return check_hip(operation, error, bytes, g_hip.device);
}

static int sync_stream_checked(hipStream_t stream, const char* operation) {
    return check_hip(operation, hipStreamSynchronize(stream), 0, g_hip.device);
}

/* Q8_0 stores a binary16 scale at byte offsets 0..1 of every 34-byte block.
 * Consecutive blocks are therefore not naturally aligned for uint16_t/__half.
 * Assemble the little-endian bits bytewise, then reinterpret the bits through
 * HIP's supported __ushort_as_half intrinsic; no unaligned load or device
 * memcpy is involved. */
__device__ __forceinline__ float q8_scale(const unsigned char* block) {
    unsigned short bits = (unsigned short)block[0] |
                          ((unsigned short)block[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

__global__ void q8_matvec_single_kernel(float* output,
                                         const unsigned char* weights,
                                         const float* input,
                                         int in_dim, int out_dim) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= out_dim) return;

    int blocks_per_row = in_dim / kQ8Values;
    const unsigned char* row_data =
        weights + (size_t)row * (size_t)blocks_per_row * kQ8BlockBytes;
    float total = 0.0f;

    /* One thread computes one complete row. This intentionally slow kernel is
     * the GPU-side reference and avoids every wave-reduction assumption. */
    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_data + (size_t)b * kQ8BlockBytes;
        const signed char* qs = (const signed char*)(block + 2);
        const float* xp = input + (size_t)b * kQ8Values;
        float block_sum = 0.0f;
        for (int j = 0; j < kQ8Values; ++j)
            block_sum += (float)qs[j] * xp[j];
        total += q8_scale(block) * block_sum;
    }
    output[row] = total;
}

__global__ void q8_matvec_simple_kernel(float* output,
                                         const unsigned char* weights,
                                         const float* input,
                                         int in_dim, int out_dim) {
    int row = (int)blockIdx.x;
    if (row >= out_dim) return;

    int lane = (int)threadIdx.x;
    int blocks_per_row = in_dim / kQ8Values;
    const unsigned char* row_data =
        weights + (size_t)row * (size_t)blocks_per_row * kQ8BlockBytes;
    float partial = 0.0f;

    /* The first supported execution model is one 32-lane RDNA2 wave per row.
     * Runtime initialization rejects any device whose reported wave size is not
     * 32, so the block and reduction geometry cannot silently miscompute. */
    for (int b = lane; b < blocks_per_row; b += 32) {
        const unsigned char* block = row_data + (size_t)b * kQ8BlockBytes;
        const signed char* qs = (const signed char*)(block + 2);
        const float* xp = input + (size_t)b * kQ8Values;
        float block_sum = 0.0f;
        for (int j = 0; j < kQ8Values; ++j)
            block_sum += (float)qs[j] * xp[j];
        partial += q8_scale(block) * block_sum;
    }

    /* HIP's full-wave __shfl_down is sufficient here; unlike the CUDA source,
     * no CUDA-specific active-lane mask is carried into the AMD implementation. */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, (unsigned)offset, 32);
    if (lane == 0) output[row] = partial;
}

__global__ void q8_matvec_optimized_kernel(float* output,
                                            const unsigned char* weights,
                                            const float* input,
                                            int in_dim, int out_dim) {
    int row = (int)blockIdx.x;
    if (row >= out_dim) return;

    int tid = (int)threadIdx.x;
    int blocks_per_row = in_dim / kQ8Values;
    const unsigned char* row_data =
        weights + (size_t)row * (size_t)blocks_per_row * kQ8BlockBytes;

    extern __shared__ float shared_input[];
    for (int i = tid; i < in_dim; i += kOptimizedThreads)
        shared_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += kOptimizedThreads) {
        const unsigned char* block = row_data + (size_t)b * kQ8BlockBytes;
        const signed char* qs = (const signed char*)(block + 2);
        const float* xp = shared_input + (size_t)b * kQ8Values;
        float block_sum = 0.0f;
        for (int j = 0; j < kQ8Values; j += 4) {
            block_sum += (float)qs[j] * xp[j]
                       + (float)qs[j + 1] * xp[j + 1]
                       + (float)qs[j + 2] * xp[j + 2]
                       + (float)qs[j + 3] * xp[j + 3];
        }
        partial += q8_scale(block) * block_sum;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, (unsigned)offset, 32);

    __shared__ float wave_sums[kOptimizedThreads / 32];
    int wave = tid / 32;
    int lane = tid % 32;
    if (lane == 0) wave_sums[wave] = partial;
    __syncthreads();

    /* Only eight first-wave lanes are live here. width=8 creates an explicit
     * eight-lane subgroup, replacing CUDA's 0xFF participation mask. */
    if (tid < kOptimizedThreads / 32) {
        float value = wave_sums[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            value += __shfl_down(value, (unsigned)offset, 8);
        if (tid == 0) output[row] = value;
    }
}

__global__ void smoke_kernel(int* values, int count) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) values[i] = values[i] * 3 + 7;
}

static int checked_q8_sizes(int out_dim, int in_dim,
                            size_t* weights_bytes, size_t* input_bytes,
                            size_t* output_bytes) {
    if (out_dim <= 0 || in_dim <= 0 || (in_dim % kQ8Values) != 0) {
        fprintf(stderr, "LinMoE HIP: invalid Q8 dimensions out=%d in=%d (in must be a multiple of 32)\n",
                out_dim, in_dim);
        return -1;
    }

    size_t blocks = (size_t)in_dim / kQ8Values;
    if (blocks > SIZE_MAX / kQ8BlockBytes ||
        (size_t)out_dim > SIZE_MAX / (blocks * kQ8BlockBytes) ||
        (size_t)in_dim > SIZE_MAX / sizeof(float) ||
        (size_t)out_dim > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "LinMoE HIP: Q8 size overflow out=%d in=%d\n", out_dim, in_dim);
        return -1;
    }

    *weights_bytes = (size_t)out_dim * blocks * kQ8BlockBytes;
    *input_bytes = (size_t)in_dim * sizeof(float);
    *output_bytes = (size_t)out_dim * sizeof(float);
    return 0;
}

/* Shared device-pointer launch boundary: fixtures and resident GQA weights use
 * exactly the same kernels. The caller owns ordering and synchronization. */
static int launch_q8(float* d_output, const unsigned char* d_weights,
                     const float* d_input, int out_dim, int in_dim,
                     LmHipQ8Kernel kernel, const char* context) {
    size_t input_bytes = (size_t)in_dim * sizeof(float);
    switch (kernel) {
        case LM_HIP_Q8_SINGLE:
            hipLaunchKernelGGL(q8_matvec_single_kernel,
                               dim3((out_dim + 127) / 128), dim3(128),
                               0, g_hip.compute,
                               d_output, d_weights, d_input, in_dim, out_dim);
            break;
        case LM_HIP_Q8_SIMPLE:
            hipLaunchKernelGGL(q8_matvec_simple_kernel,
                               dim3(out_dim), dim3(32),
                               0, g_hip.compute,
                               d_output, d_weights, d_input, in_dim, out_dim);
            break;
        case LM_HIP_Q8_OPTIMIZED:
            hipLaunchKernelGGL(q8_matvec_optimized_kernel,
                               dim3(out_dim), dim3(kOptimizedThreads),
                               input_bytes, g_hip.compute,
                               d_output, d_weights, d_input, in_dim, out_dim);
            break;
        default:
            return -1;
    }

    return check_hip(context, hipGetLastError(), 0, g_hip.device);
}

/* The layer bound matches the legacy GPU API. Each layer owns its three packed
 * matrices; uploads borrow host pointers only until their stream is drained.
 * Re-upload builds a complete replacement before retiring the old allocation.
 * This backend, like the existing inference loop, is single-host-thread only. */
constexpr int kGqaLayers = 64;
struct GqaMatrix {
    unsigned char* data;
    size_t bytes;
    int input_dim, output_dim;
};
struct GqaLayer { GqaMatrix matrix[3]; bool loaded; };
GqaLayer g_gqa[kGqaLayers] = {};

/* One reusable allocation holds input then Q, K and V. It grows transactionally
 * across models/layers. Host staging publishes all three results only on success. */
void* g_gqa_scratch = nullptr;
float* g_gqa_host = nullptr;
size_t g_gqa_scratch_bytes = 0;
bool g_gqa_faulted = false;
/* Test-only deterministic OOM injection exercises partial replacement rollback
 * without exhausting a real GPU. It is disabled during normal execution. */
int g_gqa_fail_alloc_after = -1;

static void gqa_context(char* text, size_t count, const char* op, int layer,
                        const char* matrix, int input_dim, int output_dim) {
    snprintf(text, count, "%s GQA layer=%d matrix=%s input=%d output=%d",
             op, layer, matrix, input_dim, output_dim);
}

static void free_gqa_layer(GqaLayer& storage, int layer) {
    const char* names[] = {"Wq", "Wk", "Wv"};
    for (int i = 0; i < 3; ++i) {
        GqaMatrix& m = storage.matrix[i];
        char context[160];
        gqa_context(context, sizeof(context), "hipFree", layer, names[i],
                    m.input_dim, m.output_dim);
        /* A failed device free cannot be reported as successful replacement or
         * shutdown. Stop rather than discard an allocation's ownership record. */
        if (hip_free_tracked(m.data, m.bytes, context) != 0) abort();
        m = {};
    }
    storage.loaded = false;
}

static int reserve_gqa_scratch(size_t bytes, const char* context) {
    if (bytes <= g_gqa_scratch_bytes) return 0;
    float* host = (float*)malloc(bytes);
    if (!host) {
        fprintf(stderr, "LinMoE HIP: %s host staging allocation failed bytes=%zu\n", context, bytes);
        return -1;
    }
    void* device = nullptr;
    if (hip_alloc_tracked(&device, bytes, context) != 0) { free(host); return -1; }
    if (hip_free_tracked(g_gqa_scratch, g_gqa_scratch_bytes,
                         "hipFree(GQA scratch resize)") != 0) abort();
    free(g_gqa_host);
    g_gqa_host = host;
    g_gqa_scratch = device;
    g_gqa_scratch_bytes = bytes;
    return 0;
}

/* Even an enqueue failure can follow successful async copies. Drain them before
 * releasing temporary weights or returning borrowed host memory to the caller. */
static int gqa_failed(const char* context) {
    g_gqa_faulted = true;
    if (sync_stream_checked(g_hip.compute, context) != 0) {
        /* An undrainable stream cannot safely return host buffer ownership. */
        fprintf(stderr, "LinMoE HIP: GQA stream could not be drained; stopping\n");
        abort();
    }
    return -1;
}

static int foundation_unavailable(const char* operation) {
    fprintf(stderr,
            "LinMoE HIP: %s is not implemented by the foundation backend; using it for full inference is a bug\n",
            operation);
    return -1;
}

}  // namespace

extern "C" int gpu_init(void) {
    if (g_hip.initialized) return 0;
    g_hip = {};
    g_hip.device = -1;

    if (check_hip("hipInit", hipInit(0)) != 0) {
        cleanup_partial_init();
        return -1;
    }

    int count = 0;
    if (check_hip("hipGetDeviceCount", hipGetDeviceCount(&count)) != 0 || count <= 0) {
        if (count <= 0) fprintf(stderr, "LinMoE HIP: no HIP devices found\n");
        cleanup_partial_init();
        return -1;
    }

    fprintf(stderr, "LinMoE backend: HIP (%d device%s)\n", count, count == 1 ? "" : "s");
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t prop = {};
        if (check_hip("hipGetDeviceProperties(enumerate)",
                      hipGetDeviceProperties(&prop, device), 0, device) != 0) {
            cleanup_partial_init();
            return -1;
        }
        fprintf(stderr,
                "  HIP device %d: %s arch=%s VRAM=%.0f MiB wave=%d\n",
                device, prop.name, prop.gcnArchName,
                prop.totalGlobalMem / (1024.0 * 1024.0), prop.warpSize);
        if (device == 0) g_hip.prop = prop;
    }

    /* Device 0 is deterministic and matches the inherited CUDA selection rule.
     * Multi-device policy belongs in a later milestone. */
    g_hip.device = 0;
    if (check_hip("hipSetDevice", hipSetDevice(g_hip.device), 0, g_hip.device) != 0) {
        cleanup_partial_init();
        return -1;
    }

    int queried_wave = 0;
    if (check_hip("hipDeviceGetAttribute(warpSize)",
                  hipDeviceGetAttribute(&queried_wave, hipDeviceAttributeWarpSize,
                                        g_hip.device),
                  0, g_hip.device) != 0) {
        cleanup_partial_init();
        return -1;
    }
    if (queried_wave != g_hip.prop.warpSize || queried_wave != 32) {
        fprintf(stderr,
                "LinMoE HIP: unsupported wave size: properties=%d attribute=%d; "
                "this milestone requires wave32\n",
                g_hip.prop.warpSize, queried_wave);
        cleanup_partial_init();
        return -1;
    }

    if (check_hip("hipRuntimeGetVersion",
                  hipRuntimeGetVersion(&g_hip.runtime_version),
                  0, g_hip.device) != 0) {
        cleanup_partial_init();
        return -1;
    }

    if (check_hip("hipStreamCreateWithFlags(compute)",
                  hipStreamCreateWithFlags(&g_hip.compute, hipStreamNonBlocking),
                  0, g_hip.device) != 0 ||
        check_hip("hipStreamCreateWithFlags(h2d)",
                  hipStreamCreateWithFlags(&g_hip.h2d, hipStreamNonBlocking),
                  0, g_hip.device) != 0 ||
        check_hip("hipStreamCreateWithFlags(d2h)",
                  hipStreamCreateWithFlags(&g_hip.d2h, hipStreamNonBlocking),
                  0, g_hip.device) != 0) {
        cleanup_partial_init();
        return -1;
    }

    g_hip.initialized = 1;
    fprintf(stderr,
            "LinMoE HIP selected device=0 name=%s arch=%s runtime=%d "
            "VRAM=%.0f MiB wave=%d backend=hip full_inference=no\n",
            g_hip.prop.name, g_hip.prop.gcnArchName, g_hip.runtime_version,
            g_hip.prop.totalGlobalMem / (1024.0 * 1024.0), g_hip.prop.warpSize);
    return 0;
}

extern "C" void gpu_shutdown(void) {
    if (!g_hip.initialized) return;

    hipError_t sync_error = hipDeviceSynchronize();
    if (sync_error != hipSuccess)
        report_hip_error("hipDeviceSynchronize(shutdown)", sync_error, 0, g_hip.device);

    for (int layer = 0; layer < kGqaLayers; ++layer)
        free_gqa_layer(g_gqa[layer], layer);
    if (hip_free_tracked(g_gqa_scratch, g_gqa_scratch_bytes,
                         "hipFree(GQA scratch shutdown)") != 0) abort();
    free(g_gqa_host);
    g_gqa_host = nullptr;
    g_gqa_scratch = nullptr;
    g_gqa_scratch_bytes = 0;
    g_gqa_faulted = false;
    g_gqa_fail_alloc_after = -1;

    if (g_hip.allocated_bytes != 0)
        fprintf(stderr, "LinMoE HIP: shutdown with %zu tracked device bytes still allocated\n",
                g_hip.allocated_bytes);

    destroy_stream_checked(&g_hip.d2h, "hipStreamDestroy(d2h)");
    destroy_stream_checked(&g_hip.h2d, "hipStreamDestroy(h2d)");
    destroy_stream_checked(&g_hip.compute, "hipStreamDestroy(compute)");
    g_hip = {};
}

extern "C" int gpu_is_initialized(void) {
    return g_hip.initialized;
}

extern "C" int gpu_supports_full_inference(void) {
    return 0;
}

extern "C" float gpu_vram_used_mb(void) {
    return (float)((double)g_hip.allocated_bytes / (1024.0 * 1024.0));
}

extern "C" int gpu_hip_test_smoke(void) {
    if (!g_hip.initialized) {
        fprintf(stderr, "LinMoE HIP smoke: backend is not initialized\n");
        return -1;
    }

    constexpr int count = 257;
    int input[count];
    int output[count];
    for (int i = 0; i < count; ++i) input[i] = i - 129;
    memset(output, 0, sizeof(output));

    size_t bytes = sizeof(input);
    size_t before = g_hip.allocated_bytes;
    int* device_values = nullptr;
    int result = -1;

    if (hip_alloc_tracked((void**)&device_values, bytes, "hipMalloc(smoke)") != 0)
        goto cleanup;
    if (copy_async_checked(device_values, input, bytes, hipMemcpyHostToDevice,
                           g_hip.h2d, "hipMemcpyAsync(smoke H2D)") != 0 ||
        sync_stream_checked(g_hip.h2d, "hipStreamSynchronize(smoke H2D)") != 0)
        goto cleanup;

    hipLaunchKernelGGL(smoke_kernel, dim3((count + 127) / 128), dim3(128),
                       0, g_hip.compute, device_values, count);
    if (check_hip("hipGetLastError(smoke kernel)", hipGetLastError(),
                  0, g_hip.device) != 0 ||
        sync_stream_checked(g_hip.compute,
                            "hipStreamSynchronize(smoke kernel)") != 0)
        goto cleanup;

    if (copy_async_checked(output, device_values, bytes, hipMemcpyDeviceToHost,
                           g_hip.d2h, "hipMemcpyAsync(smoke D2H)") != 0 ||
        sync_stream_checked(g_hip.d2h, "hipStreamSynchronize(smoke D2H)") != 0)
        goto cleanup;

    for (int i = 0; i < count; ++i) {
        int expected = input[i] * 3 + 7;
        if (output[i] != expected) {
            fprintf(stderr,
                    "LinMoE HIP smoke: result mismatch index=%d got=%d expected=%d\n",
                    i, output[i], expected);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (device_values &&
        hip_free_tracked(device_values, bytes, "hipFree(smoke)") != 0)
        result = -1;
    if (g_hip.allocated_bytes != before) {
        fprintf(stderr,
                "LinMoE HIP smoke: tracked VRAM mismatch before=%zu after=%zu\n",
                before, g_hip.allocated_bytes);
        result = -1;
    }
    return result;
}

extern "C" int gpu_hip_test_q8_matvec(const void* weights,
                                       const float* input,
                                       float* output,
                                       int out_dim, int in_dim,
                                       LmHipQ8Kernel kernel) {
    if (!g_hip.initialized || !weights || !input || !output) {
        fprintf(stderr, "LinMoE HIP Q8: invalid state or null buffer\n");
        return -1;
    }

    size_t weights_bytes = 0, input_bytes = 0, output_bytes = 0;
    if (checked_q8_sizes(out_dim, in_dim, &weights_bytes,
                         &input_bytes, &output_bytes) != 0)
        return -1;

    if (kernel == LM_HIP_Q8_OPTIMIZED) {
        size_t shared_bytes = input_bytes;
        if (g_hip.prop.maxThreadsPerBlock < kOptimizedThreads ||
            shared_bytes > g_hip.prop.sharedMemPerBlock) {
            fprintf(stderr,
                    "LinMoE HIP Q8 optimized: launch unsupported threads=%d/%d "
                    "shared=%zu/%zu\n",
                    kOptimizedThreads, g_hip.prop.maxThreadsPerBlock,
                    shared_bytes, g_hip.prop.sharedMemPerBlock);
            return -1;
        }
    } else if (kernel != LM_HIP_Q8_SINGLE && kernel != LM_HIP_Q8_SIMPLE) {
        fprintf(stderr, "LinMoE HIP Q8: unknown kernel selector=%d\n", (int)kernel);
        return -1;
    }

    unsigned char* d_weights = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;
    int result = -1;

    if (hip_alloc_tracked((void**)&d_weights, weights_bytes,
                          "hipMalloc(Q8 weights)") != 0 ||
        hip_alloc_tracked((void**)&d_input, input_bytes,
                          "hipMalloc(Q8 input)") != 0 ||
        hip_alloc_tracked((void**)&d_output, output_bytes,
                          "hipMalloc(Q8 output)") != 0)
        goto cleanup;

    /* Both H2D copies share one stream; synchronizing it establishes visibility
     * before the compute stream starts. Event-based overlap comes later. */
    if (copy_async_checked(d_weights, weights, weights_bytes,
                           hipMemcpyHostToDevice, g_hip.h2d,
                           "hipMemcpyAsync(Q8 weights H2D)") != 0 ||
        copy_async_checked(d_input, input, input_bytes,
                           hipMemcpyHostToDevice, g_hip.h2d,
                           "hipMemcpyAsync(Q8 input H2D)") != 0 ||
        sync_stream_checked(g_hip.h2d,
                            "hipStreamSynchronize(Q8 H2D)") != 0)
        goto cleanup;

    if (launch_q8(d_output, d_weights, d_input, out_dim, in_dim, kernel,
                  "hipGetLastError(Q8 kernel)") != 0) goto cleanup;

    if (sync_stream_checked(g_hip.compute,
                            "hipStreamSynchronize(Q8 kernel)") != 0)
        goto cleanup;

    if (copy_async_checked(output, d_output, output_bytes,
                           hipMemcpyDeviceToHost, g_hip.d2h,
                           "hipMemcpyAsync(Q8 output D2H)") != 0 ||
        sync_stream_checked(g_hip.d2h,
                            "hipStreamSynchronize(Q8 D2H)") != 0)
        goto cleanup;

    result = 0;

cleanup:
    if (d_output &&
        hip_free_tracked(d_output, output_bytes, "hipFree(Q8 output)") != 0)
        result = -1;
    if (d_input &&
        hip_free_tracked(d_input, input_bytes, "hipFree(Q8 input)") != 0)
        result = -1;
    if (d_weights &&
        hip_free_tracked(d_weights, weights_bytes, "hipFree(Q8 weights)") != 0)
        result = -1;
    return result;
}

/* Full inference is intentionally not exposed yet. These stubs make accidental
 * use diagnostic instead of silently executing a partial or CPU implementation. */
extern "C" int gpu_upload_deltanet_weights(int layer,
    const void* qkv_q8, int qkv_rows, int qkv_cols,
    const void* gate_q8, int gate_rows, int gate_cols,
    const void* ssm_out_q8, int ssm_rows, int ssm_cols) {
    (void)layer; (void)qkv_q8; (void)qkv_rows; (void)qkv_cols;
    (void)gate_q8; (void)gate_rows; (void)gate_cols;
    (void)ssm_out_q8; (void)ssm_rows; (void)ssm_cols;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_deltanet_projections(int layer,
    const float* normed, int hidden_dim,
    float* qkv_out, int qkv_dim, float* gate_out, int gate_dim) {
    (void)layer; (void)normed; (void)hidden_dim; (void)qkv_out;
    (void)qkv_dim; (void)gate_out; (void)gate_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_ssm_out_projection(int layer,
    const float* gated, int gated_dim, float* output, int output_dim) {
    (void)layer; (void)gated; (void)gated_dim; (void)output; (void)output_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_launch_qkv_gate(int layer, int slot,
    const float* normed, int hidden_dim, int qkv_dim, int gate_dim) {
    (void)layer; (void)slot; (void)normed; (void)hidden_dim;
    (void)qkv_dim; (void)gate_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_wait_qkv_gate(int slot) {
    (void)slot;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_launch_ssm_out(int layer, int slot,
    const float* gated, int gated_dim, int output_dim) {
    (void)layer; (void)slot; (void)gated; (void)gated_dim; (void)output_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_wait_ssm_out(int slot) {
    (void)slot;
    return foundation_unavailable(__func__);
}

extern "C" float* gpu_get_qkv_out(int slot) {
    (void)slot;
    return nullptr;
}

extern "C" float* gpu_get_gate_out(int slot) {
    (void)slot;
    return nullptr;
}

extern "C" float* gpu_get_ssm_out_buf(int slot) {
    (void)slot;
    return nullptr;
}

extern "C" int gpu_supports_gqa_projections(void) { return 1; }

extern "C" void gpu_hip_test_gqa_fail_alloc_after(int count) {
    g_gqa_fail_alloc_after = count;
}

extern "C" int gpu_upload_gqa_weights(int layer,
    const void* wq_q8, int wq_rows, int wq_cols,
    const void* wk_q8, int wk_rows, int wk_cols,
    const void* wv_q8, int wv_rows, int wv_cols,
    const void* wo_q8, int wo_rows, int wo_cols) {
    /* Preserve the inherited ABI: rows is GGUF dims[0] (input width), cols is
     * dims[1] (output count), despite their misleading mathematical names.
     * Wo is deliberately neither read nor retained: output remains on CPU. */
    (void)wo_q8; (void)wo_rows; (void)wo_cols;
    if (!g_hip.initialized || g_gqa_faulted || layer < 0 || layer >= kGqaLayers ||
        !wq_q8 || !wk_q8 || !wv_q8 || wq_rows != wk_rows || wq_rows != wv_rows ||
        wk_cols != wv_cols || wk_cols <= 0 || wq_cols <= 0 ||
        (int64_t)wq_cols % (2LL * wk_cols) != 0) {
        fprintf(stderr, "LinMoE HIP: invalid GQA upload layer=%d Wq=%dx%d Wk=%dx%d Wv=%dx%d\n",
                layer, wq_rows, wq_cols, wk_rows, wk_cols, wv_rows, wv_cols);
        return -1;
    }
    const void* host[] = {wq_q8, wk_q8, wv_q8};
    const char* names[] = {"Wq", "Wk", "Wv"};
    int outputs[] = {wq_cols, wk_cols, wv_cols};
    GqaLayer replacement = {};
    /* Validate every packed size before any allocation or host read. Q8_0 is
     * output_count * (input_width / 32) * 34 bytes, never rows * cols bytes. */
    for (int i = 0; i < 3; ++i) {
        GqaMatrix& m = replacement.matrix[i];
        size_t input_bytes, output_bytes;
        m.input_dim = wq_rows;
        m.output_dim = outputs[i];
        if (checked_q8_sizes(m.output_dim, m.input_dim, &m.bytes,
                             &input_bytes, &output_bytes) != 0) {
            fprintf(stderr, "LinMoE HIP: invalid packed size layer=%d matrix=%s\n", layer, names[i]);
            return -1;
        }
    }
    for (int i = 0; i < 3; ++i) {
        GqaMatrix& m = replacement.matrix[i];
        char context[160];
        gqa_context(context, sizeof(context), "hipMalloc/upload", layer, names[i],
                    m.input_dim, m.output_dim);
        int allocated;
        if (g_gqa_fail_alloc_after == 0) {
            g_gqa_fail_alloc_after = -1;
            allocated = report_hip_error(context, hipErrorOutOfMemory, m.bytes, g_hip.device);
        } else {
            allocated = hip_alloc_tracked((void**)&m.data, m.bytes, context);
            if (allocated == 0 && g_gqa_fail_alloc_after > 0) --g_gqa_fail_alloc_after;
        }
        if (allocated != 0) {
            /* Ordinary OOM is recoverable: drain earlier copies, discard only
             * the replacement, and leave the previously uploaded layer usable. */
            if (sync_stream_checked(g_hip.compute, context) != 0) gqa_failed(context);
            free_gqa_layer(replacement, layer);
            return -1;
        }
        if (copy_async_checked(m.data, host[i], m.bytes, hipMemcpyHostToDevice,
                                g_hip.compute, context) != 0) {
            gqa_failed(context);
            free_gqa_layer(replacement, layer);
            return -1;
        }
    }
    char context[160];
    gqa_context(context, sizeof(context), "hipStreamSynchronize(upload)", layer,
                "Wq/Wk/Wv", wq_rows, wq_cols);
    if (sync_stream_checked(g_hip.compute, context) != 0) {
        gqa_failed(context);
        free_gqa_layer(replacement, layer);
        return -1;
    }
    free_gqa_layer(g_gqa[layer], layer);
    replacement.loaded = true;
    g_gqa[layer] = replacement;
    return 0;
}

extern "C" int gpu_gqa_projections(int layer,
    const float* normed, int hidden_dim,
    float* q_gate_out, int q_gate_dim,
    float* k_out, int k_dim, float* v_out, int v_dim) {
    if (!g_hip.initialized || g_gqa_faulted || layer < 0 || layer >= kGqaLayers ||
        !normed || !q_gate_out || !k_out || !v_out || !g_gqa[layer].loaded) {
        fprintf(stderr, "LinMoE HIP: invalid GQA projection state/buffers layer=%d\n", layer);
        return -1;
    }
    GqaLayer& weights = g_gqa[layer];
    int outputs[] = {q_gate_dim, k_dim, v_dim};
    size_t bytes[3] = {};
    size_t total = (size_t)weights.matrix[0].input_dim * sizeof(float);
    for (int i = 0; i < 3; ++i) {
        if (hidden_dim != weights.matrix[i].input_dim || outputs[i] != weights.matrix[i].output_dim) {
            fprintf(stderr, "LinMoE HIP: GQA shape mismatch layer=%d matrix=%d input=%d/%d output=%d/%d\n",
                    layer, i, hidden_dim, weights.matrix[i].input_dim, outputs[i], weights.matrix[i].output_dim);
            return -1;
        }
        bytes[i] = (size_t)outputs[i] * sizeof(float);
        if (bytes[i] > SIZE_MAX - total) return -1;
        total += bytes[i];
    }
    char context[160];
    gqa_context(context, sizeof(context), "hipMalloc(scratch)", layer, "Q/K/V", hidden_dim, q_gate_dim);
    if (reserve_gqa_scratch(total, context) != 0) return -1;
    float* input = (float*)g_gqa_scratch;
    float* device[3] = {input + hidden_dim, input + hidden_dim + q_gate_dim,
                        input + hidden_dim + (size_t)q_gate_dim + k_dim};
    float* staged[3] = {g_gqa_host, g_gqa_host + q_gate_dim, g_gqa_host + (size_t)q_gate_dim + k_dim};
    const char* names[] = {"Wq", "Wk", "Wv"};
    gqa_context(context, sizeof(context), "hipMemcpyAsync(input H2D)", layer, "Q/K/V", hidden_dim, q_gate_dim);
    /* A single compute stream orders the sole H2D input copy, all three tested
     * simple kernels, and all D2H copies. No host attention overlaps this work. */
    if (copy_async_checked(input, normed, (size_t)hidden_dim * sizeof(float),
                            hipMemcpyHostToDevice, g_hip.compute, context) != 0)
        return gqa_failed(context);
    for (int i = 0; i < 3; ++i) {
        gqa_context(context, sizeof(context), "kernel launch", layer, names[i], hidden_dim, outputs[i]);
        if (launch_q8(device[i], weights.matrix[i].data, input, outputs[i], hidden_dim,
                       LM_HIP_Q8_SIMPLE, context) != 0) return gqa_failed(context);
    }
    for (int i = 0; i < 3; ++i) {
        gqa_context(context, sizeof(context), "hipMemcpyAsync(D2H)", layer, names[i], hidden_dim, outputs[i]);
        if (copy_async_checked(staged[i], device[i], bytes[i], hipMemcpyDeviceToHost,
                                g_hip.compute, context) != 0) return gqa_failed(context);
    }
    gqa_context(context, sizeof(context), "hipStreamSynchronize(projections)", layer, "Q/K/V", hidden_dim, q_gate_dim);
    if (sync_stream_checked(g_hip.compute, context) != 0) return gqa_failed(context);
    /* This is the host-attention boundary. Publish the complete interleaved
     * [Q_head0, gate_head0, Q_head1, gate_head1, ...] buffer unchanged. On any
     * earlier failure the caller's Q/K/V buffers remain untouched. */
    memcpy(q_gate_out, staged[0], bytes[0]);
    memcpy(k_out, staged[1], bytes[1]);
    memcpy(v_out, staged[2], bytes[2]);
    return 0;
}

extern "C" int gpu_gqa_output(int layer,
    const float* attn_out, int attn_dim,
    float* output, int hidden_dim) {
    (void)layer; (void)attn_out; (void)attn_dim; (void)output; (void)hidden_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_upload_router(int layer, const float* weights,
                                  int hidden_dim, int num_experts) {
    (void)layer; (void)weights; (void)hidden_dim; (void)num_experts;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_router(int layer, const float* normed, int hidden_dim,
                           float* logits_out, int num_experts) {
    (void)layer; (void)normed; (void)hidden_dim; (void)logits_out; (void)num_experts;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_cache_expert(int layer, int expert_id,
    const void* gate_data, int gate_size,
    const void* up_data, int up_size,
    const void* down_data, int down_size) {
    (void)layer; (void)expert_id; (void)gate_data; (void)gate_size;
    (void)up_data; (void)up_size; (void)down_data; (void)down_size;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_find_cached_expert(int layer, int expert_id) {
    (void)layer; (void)expert_id;
    return -1;
}

extern "C" int gpu_expert_ffn(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* gate_out, float* up_out, float* expert_out,
    int gate_type, int down_type) {
    (void)cache_idx; (void)input; (void)hidden_dim; (void)intermediate;
    (void)gate_out; (void)up_out; (void)expert_out; (void)gate_type; (void)down_type;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_expert_down(int cache_idx, const float* act, int intermediate,
    float* output, int hidden_dim) {
    (void)cache_idx; (void)act; (void)intermediate; (void)output; (void)hidden_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_expert_ffn_fused(int cache_idx, const float* input,
    int hidden_dim, int intermediate, float* expert_out) {
    (void)cache_idx; (void)input; (void)hidden_dim; (void)intermediate; (void)expert_out;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_expert_batch_start(const float* normed, int hidden_dim) {
    (void)normed; (void)hidden_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_expert_batch_add(int cache_idx, int hidden_dim,
    int intermediate, float weight) {
    (void)cache_idx; (void)hidden_dim; (void)intermediate; (void)weight;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_expert_batch_finish(float* moe_out, int hidden_dim) {
    (void)moe_out; (void)hidden_dim;
    return foundation_unavailable(__func__);
}

extern "C" int gpu_expert_cache_count(void) {
    return 0;
}

extern "C" void gpu_set_expert_limit(int limit) {
    (void)limit;
}
