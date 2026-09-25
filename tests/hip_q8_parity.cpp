/*
 * Independent CPU-vs-HIP Q8_0 matvec parity.
 *
 * The CPU oracle decodes binary16 mathematically and accumulates in double; it
 * does not call another GPU kernel or LinMoE's SIMD Q8 path. Tolerances are
 * fixed here before target-device results are observed:
 *   |gpu-ref| <= 5e-2 + 5e-5 * |ref|
 * The absolute term covers cancellation near zero; the relative term covers
 * reduction-order/FMA differences at large magnitudes.
 */
#include "hip_q8_fixture.h"

int main(void) {
    /* Coverage includes zeros, sign-homogeneous and alternating data, full int8
     * extrema, tiny binary16 scales, many blocks, multiple rows and a 4096-wide
     * projection representative of LinMoE hidden-state matvecs. */
    const Fixture fixtures[] = {
        {"zeros",             1,   32, 0, 0, 0x3c00u},
        {"positive",          3,   64, 1, 1, 0x3800u},
        {"negative",          4,  256, 2, 2, 0x3c00u},
        {"alternating",       7, 1024, 3, 3, 0x3c00u},
        {"min-normal-scale",  5,  256, 4, 4, 0x0400u},
        {"min-subnormal",     5,  256, 4, 1, 0x0001u},
        {"random-4096",      17, 4096, 4, 4, 0x3c00u},
    };
    constexpr int case_count = (int)(sizeof(fixtures) / sizeof(fixtures[0]));

    if (gpu_init() != 0) {
        fprintf(stderr, "HIP Q8 parity FAIL: gpu_init\n");
        return 1;
    }

    Metrics cpu_metrics[3];
    Metrics cross_single_simple;
    Metrics cross_single_optimized;
    int status = 0;
    int completed_cases = 0;

    for (const Fixture& fixture : fixtures) {
        std::vector<unsigned char> weights;
        std::vector<float> input;
        std::vector<float> reference;
        std::vector<float> gpu[3];

        build_fixture(fixture, weights, input);
        cpu_reference(weights, input, fixture.rows, fixture.cols, reference);

        for (int impl = 0; impl < 3; ++impl) {
            gpu[impl].assign(fixture.rows, 0.0f);
            if (gpu_hip_test_q8_matvec(weights.data(), input.data(),
                                       gpu[impl].data(), fixture.rows,
                                       fixture.cols, (LmHipQ8Kernel)impl) != 0) {
                fprintf(stderr, "HIP Q8 parity FAIL: case=%s implementation=%d launch/copy error\n",
                        fixture.name, impl);
                status = 1;
                break;
            }
            for (int row = 0; row < fixture.rows; ++row)
                update_metrics(cpu_metrics[impl], fixture.name, row,
                               gpu[impl][row], reference[row]);
        }
        if (status) break;

        for (int row = 0; row < fixture.rows; ++row) {
            update_metrics(cross_single_simple, fixture.name, row,
                           gpu[1][row], gpu[0][row]);
            update_metrics(cross_single_optimized, fixture.name, row,
                           gpu[2][row], gpu[0][row]);
        }
        ++completed_cases;
    }

    if (completed_cases != case_count) status = 1;
    print_metrics("CPU vs HIP single", completed_cases, cpu_metrics[0]);
    print_metrics("CPU vs HIP simple", completed_cases, cpu_metrics[1]);
    print_metrics("CPU vs HIP optimized", completed_cases, cpu_metrics[2]);
    print_metrics("HIP single vs simple", completed_cases, cross_single_simple);
    print_metrics("HIP single vs optimized", completed_cases, cross_single_optimized);
    printf("tolerance: abs<=%.3g + %.3g*|reference|; relative metric floor=%.1e\n",
           kAbsTol, kRelTol, kRelativeFloor);

    for (const Metrics& m : cpu_metrics) if (!m.pass) status = 1;
    if (!cross_single_simple.pass || !cross_single_optimized.pass) status = 1;

    float remaining_mb = gpu_vram_used_mb();
    if (remaining_mb != 0.0f) {
        fprintf(stderr, "HIP Q8 parity FAIL: tracked VRAM after tests=%.6f MiB\n",
                remaining_mb);
        status = 1;
    }
    gpu_shutdown();

    if (status == 0)
        puts("HIP Q8 parity PASS");
    return status;
}
