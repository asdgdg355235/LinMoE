/*
 * Device-required smoke test for the HIP foundation. Keeping this separate from
 * make check lets CPU-only development remain independent of ROCm availability.
 */
#include <stdio.h>

#include "../engine/runtime/gpu_offload.h"
#include "../engine/runtime/gpu_offload_hip_test.h"

int main(void) {
    if (gpu_init() != 0) {
        fprintf(stderr, "HIP smoke FAIL: gpu_init\n");
        return 1;
    }
    if (!gpu_is_initialized()) {
        fprintf(stderr, "HIP smoke FAIL: initialized flag\n");
        gpu_shutdown();
        return 1;
    }
    if (gpu_supports_full_inference()) {
        fprintf(stderr, "HIP smoke FAIL: foundation backend advertised full inference\n");
        gpu_shutdown();
        return 1;
    }
    int result = gpu_hip_test_smoke();
    float remaining_mb = gpu_vram_used_mb();
    gpu_shutdown();
    if (result != 0 || remaining_mb != 0.0f) {
        fprintf(stderr, "HIP smoke FAIL: operation=%d tracked_vram=%.6f MiB\n",
                result, remaining_mb);
        return 1;
    }
    puts("HIP smoke PASS (init/properties/alloc/H2D/kernel/D2H/free)");
    return 0;
}
