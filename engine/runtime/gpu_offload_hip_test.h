#pragma once
/* Test-only entry points for the staged HIP backend. They deliberately stay
 * outside gpu_offload.h so production inference does not depend on validation
 * machinery or mistake isolated Q8 support for full backend capability. */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LM_HIP_Q8_SINGLE = 0,
    LM_HIP_Q8_SIMPLE = 1,
    LM_HIP_Q8_OPTIMIZED = 2
} LmHipQ8Kernel;

/* Exercise allocation, H2D copy, a tiny kernel, D2H copy, synchronization and
 * cleanup using the backend-owned streams. Returns zero only after validating
 * every returned element and restoring tracked VRAM usage. */
int gpu_hip_test_smoke(void);

/* Simulate an allocation failure after count successful GQA weight allocations.
 * A negative count disables injection. No actual device exhaustion is required. */
void gpu_hip_test_gqa_fail_alloc_after(int count);

/* Run one raw GGML Q8_0 matrix-vector multiplication. The matrix contains
 * out_dim rows, each with in_dim/32 packed 34-byte blocks. */
int gpu_hip_test_q8_matvec(const void* weights, const float* input, float* output,
                           int out_dim, int in_dim, LmHipQ8Kernel kernel);

#ifdef __cplusplus
}
#endif
