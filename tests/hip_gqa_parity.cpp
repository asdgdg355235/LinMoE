/* Exercise the real persistent GQA API, independently of full inference.
 * Reuse the validated Q8 suite's double-accumulating CPU oracle and tolerances.
 * All Q rows are checked, including every interleaved gate segment. */
#include "hip_q8_fixture.h"
#include <algorithm>
#include <climits>

namespace {
struct Shape { const char* name; int hidden, q_gate, kv; uint16_t scale; };
struct GqaFixture {
    Shape shape;
    std::vector<unsigned char> weights[3];
    std::vector<float> input;
    GqaFixture(Shape s, int variant) : shape(s) {
        for (int m = 0; m < 3; ++m) {
            Fixture f = {s.name, m == 0 ? s.q_gate : s.kv, s.hidden, 4, 4, s.scale};
            build_fixture(f, weights[m], input);
            /* Distinct matrices, zero rows and extrema catch swapped K/V,
             * ignored blocks, and lost gate rows, even for equal K/V shapes. */
            for (size_t b = 0; b < weights[m].size() / 34; ++b) {
                unsigned char* block = weights[m].data() + b * 34;
                for (int j = 0; j < 32; ++j) {
                    int value = (int)block[2+j] - 128;
                    if (b < (size_t)s.hidden / 32 || (b + m + variant) % 11 == 0) value = 0;
                    else if ((b + j + m + variant) % 7 == 0) value = -128;
                    else if ((b + j + 2*m + variant) % 9 == 0) value = 127;
                    block[2+j] = (unsigned char)(int8_t)value;
                }
            }
        }
    }
    size_t bytes() const { return weights[0].size()+weights[1].size()+weights[2].size(); }
    int upload(int layer) const {
        return gpu_upload_gqa_weights(layer, weights[0].data(), shape.hidden, shape.q_gate,
            weights[1].data(), shape.hidden, shape.kv, weights[2].data(), shape.hidden, shape.kv,
            nullptr, 0, 0); // HIP must not need or retain Wo.
    }
};

static bool usage(size_t expected) {
    double actual = gpu_vram_used_mb();
    double target = (double)expected / (1024*1024);
    if (std::fabs(actual-target) > 1e-5) {
        fprintf(stderr, "GQA VRAM FAIL: got=%.9g expected=%.9g MiB\n", actual, target);
        return false;
    }
    return true;
}

static bool project(GqaFixture& f, int layer, Metrics (&metrics)[3], int call) {
    std::vector<float> reference, output[3];
    /* Change inputs between calls, keeping them bounded and reproducible so
     * stale input or result buffers cannot pass repeated-execution coverage. */
    for (int j = 0; j < f.shape.hidden; ++j)
        f.input[j] = (float)(((j * 71 + call * 137) % 1009) - 504) / 503.0f;
    int rows[] = {f.shape.q_gate, f.shape.kv, f.shape.kv};
    constexpr float guard = 1234567.0f;
    for (int m = 0; m < 3; ++m) output[m].assign(rows[m]+2, guard);
    if (gpu_gqa_projections(layer, f.input.data(), f.shape.hidden,
            output[0].data()+1, rows[0], output[1].data()+1, rows[1],
            output[2].data()+1, rows[2]) != 0) return false;
    for (int m = 0; m < 3; ++m) {
        if (output[m].front() != guard || output[m].back() != guard) return false;
        cpu_reference(f.weights[m], f.input, rows[m], f.shape.hidden, reference);
        for (int r = 0; r < rows[m]; ++r) {
            update_metrics(metrics[m], f.shape.name, r, output[m][r+1], reference[r]);
            /* The inherited absolute bound scales linearly with uniformly tiny
             * weights. Tighten it accordingly so a zero result cannot pass the
             * subnormal fixture merely because its signal is below 0.05. */
            double scale = f.shape.scale < 0x0400 ? fp16_reference(f.shape.scale) : 1.;
            double allowed = kAbsTol*scale + kRelTol*std::fabs(reference[r]);
            if (std::fabs((double)output[m][r+1]-reference[r]) > allowed) metrics[m].pass = false;
        }
    }
    return true;
}

static bool invalid_requests(GqaFixture& f) {
    float output[128];
    std::fill(output, output+128, 777.0f);
    // Invalid calls must reject before enqueueing copies or publishing outputs.
    for (int layer : {-1, 64, INT_MAX}) {
        if (f.upload(layer) == 0 || gpu_gqa_projections(layer, f.input.data(), f.shape.hidden,
                output, f.shape.q_gate, output, f.shape.kv, output, f.shape.kv) == 0) return false;
    }
    if (gpu_gqa_projections(63, f.input.data(), f.shape.hidden,
             output, f.shape.q_gate, output, f.shape.kv, output, f.shape.kv) == 0) return false;
    if (gpu_gqa_projections(0, f.input.data(), f.shape.hidden,
             output, f.shape.q_gate-1, output, f.shape.kv, output, f.shape.kv) == 0) return false;
    if (gpu_gqa_projections(0, nullptr, f.shape.hidden,
             output, f.shape.q_gate, output, f.shape.kv, output, f.shape.kv) == 0) return false;
    for (int h : {0, -32, 33, INT_MAX}) {
        if (gpu_upload_gqa_weights(0, f.weights[0].data(), h, f.shape.q_gate,
            f.weights[1].data(), h, f.shape.kv, f.weights[2].data(), h, f.shape.kv,
            nullptr, 0, 0) == 0) return false;
    }
    if (gpu_upload_gqa_weights(0, nullptr, f.shape.hidden, f.shape.q_gate,
        f.weights[1].data(), f.shape.hidden, f.shape.kv,
        f.weights[2].data(), f.shape.hidden, f.shape.kv, nullptr, 0, 0) == 0) return false;
    if (gpu_upload_gqa_weights(0, f.weights[0].data(), f.shape.hidden, f.shape.q_gate,
        f.weights[1].data(), f.shape.hidden, f.shape.kv,
        f.weights[2].data(), f.shape.hidden+32, f.shape.kv, nullptr, 0, 0) == 0) return false;
    return std::all_of(output, output+128, [](float v) { return v == 777.0f; });
}
} // namespace

int main() {
    Metrics metrics[3];
    int calls = 0;
    bool pass = true;
    const Shape shapes[] = {
        {"small", 64, 128, 32, 0x3c00},
        {"non-256-width", 96, 96, 24, 0x3800},
        {"subnormal", 256, 256, 64, 0x0001},
        {"397B-geometry", 4096, 16384, 512, 0x3c00},
    };
    // Two complete lifetimes cover shutdown with live weights and reinitialization.
    for (int cycle = 0; cycle < 2 && pass; ++cycle) {
        if (gpu_init() != 0) return 1;
        GqaFixture anchor(shapes[0], 1);
        pass = anchor.upload(0) == 0 && usage(anchor.bytes()) && invalid_requests(anchor);
        size_t scratch = 0;
        // Failure on Wq, Wk or Wv must preserve both old weights and VRAM.
        GqaFixture failed_replacement(shapes[0], 42);
        for (int fail_after = 0; fail_after < 3 && pass; ++fail_after) {
            gpu_hip_test_gqa_fail_alloc_after(fail_after);
            pass = failed_replacement.upload(0) != 0 && usage(anchor.bytes()+scratch);
            pass = pass && project(anchor, 0, metrics, 20+fail_after); ++calls;
            scratch = (size_t)(anchor.shape.hidden+anchor.shape.q_gate+2*anchor.shape.kv)*sizeof(float);
            pass = pass && usage(anchor.bytes()+scratch);
        }
        for (int c = 0; c < (cycle == 0 ? 4 : 2) && pass; ++c) {
            GqaFixture f(shapes[c], c+3);
            pass = f.upload(7) == 0 && usage(anchor.bytes()+f.bytes()+scratch);
            printf("GQA %s: weight_bytes=%zu weight_MiB=%.6f\n", f.shape.name, f.bytes(), f.bytes()/(1024.*1024.));
            for (int repeat = 0; repeat < 3 && pass; ++repeat) {
                pass = project(f, 7, metrics, repeat);
                ++calls;
                scratch = std::max(scratch, (size_t)(f.shape.hidden+f.shape.q_gate+2*f.shape.kv)*sizeof(float));
                pass = pass && usage(anchor.bytes()+f.bytes()+scratch);
            }
            // Same-size replacement must change results without increasing VRAM.
            GqaFixture changed(shapes[c], c+9);
            pass = pass && changed.upload(7) == 0 && usage(anchor.bytes()+changed.bytes()+scratch);
            pass = pass && project(changed, 7, metrics, 5); ++calls;
            // Another layer's weights must survive every replacement and resize.
            pass = pass && project(anchor, 0, metrics, c+10); ++calls;
            scratch = std::max(scratch, (size_t)(anchor.shape.hidden+anchor.shape.q_gate+2*anchor.shape.kv)*sizeof(float));
            pass = pass && usage(anchor.bytes()+changed.bytes()+scratch);
        }
        gpu_shutdown();
        gpu_shutdown(); // idempotence must not double-free streams or allocations.
        pass = pass && !gpu_is_initialized() && usage(0);
    }
    const char* names[] = {"GQA Q+gate", "GQA K", "GQA V"};
    for (int m = 0; m < 3; ++m) {
        print_metrics(names[m], calls, metrics[m]);
        pass = pass && metrics[m].pass && metrics[m].values != 0;
    }
    printf("GQA tolerance: abs<=%.3g + %.3g*|reference|; subnormal absolute term scaled by FP16 scale\n", kAbsTol, kRelTol);
    printf("HIP GQA parity %s (repeated calls, two layers, re-upload, resize, shutdown)\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
