/* Validate resident GQA Wo through the production gpu_gqa_output() API.
 * The established Q/K/V fixture remains unchanged; this test adds Wo parity and
 * verifies that four-matrix replacement is atomic, including failure at Wo. */
#include "hip_q8_fixture.h"
#include <algorithm>
#include <climits>

namespace {

struct Shape {
    const char* name;
    int hidden, q_gate, kv;
    uint16_t scale;
    int wo_weight_mode, wo_input_mode;
};

struct FullGqaFixture {
    Shape shape;
    std::vector<unsigned char> weights[4];
    std::vector<float> normed;
    std::vector<float> attn;

    FullGqaFixture(Shape s, int variant) : shape(s) {
        for (int m = 0; m < 3; ++m) {
            Fixture f = {s.name, m == 0 ? s.q_gate : s.kv, s.hidden,
                         4, 4, s.scale};
            build_fixture(f, weights[m], normed);
            /* Keep Q/K/V distinct so rollback checks cannot pass after a matrix
             * swap or partial replacement. */
            for (size_t b = 0; b < weights[m].size() / kBlockBytes; ++b) {
                unsigned char* block = weights[m].data() + b * kBlockBytes;
                for (int j = 0; j < kQK; ++j) {
                    int value = (int)(int8_t)block[2 + j];
                    if ((b + j + m + variant) % 17 == 0) value = -128;
                    else if ((b + 2 * j + m + variant) % 19 == 0) value = 127;
                    block[2 + j] = (unsigned char)(int8_t)value;
                }
            }
        }

        /* Wo is mathematically hidden_dim outputs by q_gate_dim/2 inputs. */
        Fixture wo = {s.name, s.hidden, s.q_gate / 2,
                      s.wo_weight_mode, s.wo_input_mode, s.scale};
        build_fixture(wo, weights[3], attn);
        if (s.wo_weight_mode == 4) {
            for (size_t b = 0; b < weights[3].size() / kBlockBytes; ++b) {
                unsigned char* block = weights[3].data() + b * kBlockBytes;
                for (int j = 0; j < kQK; ++j) {
                    int value = (int)(int8_t)block[2 + j];
                    if ((b + j + variant) % 23 == 0) value = -128;
                    else if ((b + 3 * j + variant) % 29 == 0) value = 127;
                    block[2 + j] = (unsigned char)(int8_t)value;
                }
            }
        }
    }

    size_t qkv_bytes() const {
        return weights[0].size() + weights[1].size() + weights[2].size();
    }
    size_t bytes() const { return qkv_bytes() + weights[3].size(); }
    size_t scratch_bytes() const {
        size_t qkv = (size_t)(shape.hidden + shape.q_gate + 2 * shape.kv) *
                     sizeof(float);
        size_t wo = (size_t)(shape.q_gate / 2 + shape.hidden) * sizeof(float);
        return std::max(qkv, wo);
    }

    int upload(int layer) const {
        return gpu_upload_gqa_weights(
            layer,
            weights[0].data(), shape.hidden, shape.q_gate,
            weights[1].data(), shape.hidden, shape.kv,
            weights[2].data(), shape.hidden, shape.kv,
            weights[3].data(), shape.q_gate / 2, shape.hidden);
    }
};

static bool usage(size_t expected) {
    double actual = gpu_vram_used_mb();
    double target = (double)expected / (1024.0 * 1024.0);
    if (std::fabs(actual - target) > 1e-5) {
        fprintf(stderr, "GQA Wo VRAM FAIL: got=%.9g expected=%.9g MiB\n",
                actual, target);
        return false;
    }
    return true;
}

static void apply_bound(Metrics& metrics, const Shape& shape,
                        float actual, float reference) {
    double scale = shape.scale < 0x0400 ? fp16_reference(shape.scale) : 1.0;
    double allowed = kAbsTol * scale + kRelTol * std::fabs(reference);
    if (std::fabs((double)actual - reference) > allowed) metrics.pass = false;
}

static bool check_all(FullGqaFixture& f, int layer, Metrics (&metrics)[4],
                      int call) {
    /* Vary ordinary inputs across repeated calls; preserve intentionally
     * structured zero/positive/negative/alternating Wo fixtures. */
    for (int j = 0; j < f.shape.hidden; ++j)
        f.normed[j] = (float)(((j * 71 + call * 137) % 1009) - 504) / 503.0f;
    if (f.shape.wo_input_mode == 4) {
        for (int j = 0; j < f.shape.q_gate / 2; ++j)
            f.attn[j] = (float)(((j * 43 + call * 89) % 997) - 498) / 497.0f;
    }

    int rows[3] = {f.shape.q_gate, f.shape.kv, f.shape.kv};
    std::vector<float> qkv[3], reference;
    constexpr float guard = 1234567.0f;
    for (int m = 0; m < 3; ++m) qkv[m].assign(rows[m] + 2, guard);

    if (gpu_gqa_projections(layer, f.normed.data(), f.shape.hidden,
            qkv[0].data() + 1, rows[0],
            qkv[1].data() + 1, rows[1],
            qkv[2].data() + 1, rows[2]) != 0)
        return false;

    for (int m = 0; m < 3; ++m) {
        if (qkv[m].front() != guard || qkv[m].back() != guard) return false;
        cpu_reference(f.weights[m], f.normed, rows[m], f.shape.hidden, reference);
        for (int r = 0; r < rows[m]; ++r) {
            update_metrics(metrics[m], f.shape.name, r, qkv[m][r + 1], reference[r]);
            apply_bound(metrics[m], f.shape, qkv[m][r + 1], reference[r]);
        }
    }

    std::vector<float> wo(f.shape.hidden + 2, guard);
    if (gpu_gqa_output(layer, f.attn.data(), f.shape.q_gate / 2,
                       wo.data() + 1, f.shape.hidden) != 0)
        return false;
    if (wo.front() != guard || wo.back() != guard) return false;

    cpu_reference(f.weights[3], f.attn, f.shape.hidden,
                  f.shape.q_gate / 2, reference);
    for (int r = 0; r < f.shape.hidden; ++r) {
        update_metrics(metrics[3], f.shape.name, r, wo[r + 1], reference[r]);
        apply_bound(metrics[3], f.shape, wo[r + 1], reference[r]);
    }
    return true;
}

static bool invalid_output_requests(FullGqaFixture& f) {
    std::vector<float> output((size_t)f.shape.hidden + 2, 777.0f);
    float* guarded = output.data() + 1;
    int attn_dim = f.shape.q_gate / 2;

    for (int layer : {-1, 64, INT_MAX}) {
        if (gpu_gqa_output(layer, f.attn.data(), attn_dim,
                           guarded, f.shape.hidden) == 0)
            return false;
    }
    if (gpu_gqa_output(0, nullptr, attn_dim, guarded, f.shape.hidden) == 0 ||
        gpu_gqa_output(0, f.attn.data(), attn_dim, nullptr, f.shape.hidden) == 0 ||
        gpu_gqa_output(0, f.attn.data(), attn_dim - 32,
                       guarded, f.shape.hidden) == 0 ||
        gpu_gqa_output(0, f.attn.data(), attn_dim,
                       guarded, f.shape.hidden + 32) == 0)
        return false;

    /* Partial/incorrect Wo specification must fail before touching the old
     * resident layer. */
    if (gpu_upload_gqa_weights(
            0,
            f.weights[0].data(), f.shape.hidden, f.shape.q_gate,
            f.weights[1].data(), f.shape.hidden, f.shape.kv,
            f.weights[2].data(), f.shape.hidden, f.shape.kv,
            nullptr, attn_dim, f.shape.hidden) == 0)
        return false;
    if (gpu_upload_gqa_weights(
            0,
            f.weights[0].data(), f.shape.hidden, f.shape.q_gate,
            f.weights[1].data(), f.shape.hidden, f.shape.kv,
            f.weights[2].data(), f.shape.hidden, f.shape.kv,
            f.weights[3].data(), attn_dim + 32, f.shape.hidden) == 0)
        return false;

    return std::all_of(output.begin(), output.end(),
                       [](float v) { return v == 777.0f; });
}

}  // namespace

int main() {
    const Shape shapes[] = {
        {"zero",        64,  128,   32, 0x3c00, 0, 0},
        {"positive",    96,  192,   32, 0x3800, 1, 1},
        {"negative",   128,  256,   64, 0x3c00, 2, 2},
        {"alternating",256,  512,   64, 0x3c00, 3, 3},
        {"subnormal",  256,  512,   64, 0x0001, 4, 1},
        {"397B-Wo",   4096,16384,  512, 0x3c00, 4, 4},
    };

    Metrics metrics[4];
    bool pass = true;
    int calls = 0;
    size_t scratch = 0;

    if (gpu_init() != 0) return 1;

    FullGqaFixture anchor(shapes[0], 1);
    pass = anchor.upload(0) == 0 && usage(anchor.bytes());
    if (pass) {
        pass = check_all(anchor, 0, metrics, 1);
        ++calls;
        scratch = anchor.scratch_bytes();
        pass = pass && usage(anchor.bytes() + scratch);
    }
    pass = pass && invalid_output_requests(anchor);

    /* Replacement allocation may fail at Wq/Wk/Wv/Wo. The old layer must
     * remain complete and executable after every recoverable failure. */
    FullGqaFixture replacement(shapes[0], 42);
    for (int fail_after = 0; fail_after < 4 && pass; ++fail_after) {
        gpu_hip_test_gqa_fail_alloc_after(fail_after);
        pass = replacement.upload(0) != 0 && usage(anchor.bytes() + scratch);
        pass = pass && check_all(anchor, 0, metrics, 10 + fail_after);
        ++calls;
        pass = pass && usage(anchor.bytes() + scratch);
    }

    /* Three successful copies followed by a pre-enqueue failure targets Wo
     * specifically and verifies four-matrix rollback after queued Q/K/V work. */
    if (pass) {
        gpu_hip_test_gqa_fail_copy_after(3);
        pass = replacement.upload(0) != 0 && usage(anchor.bytes() + scratch);
        pass = pass && check_all(anchor, 0, metrics, 20);
        ++calls;
        pass = pass && usage(anchor.bytes() + scratch);
    }

    /* Exercise repeated execution, independent layer ownership, replacement,
     * smaller->larger growth and larger->smaller reuse without scratch leaks. */
    for (size_t c = 0; c < sizeof(shapes) / sizeof(shapes[0]) && pass; ++c) {
        FullGqaFixture f(shapes[c], (int)c + 3);
        pass = f.upload(7) == 0;
        if (!pass) break;
        pass = usage(anchor.bytes() + f.bytes() + scratch);
        printf("GQA %s: four_weight_bytes=%zu four_weight_MiB=%.6f Wo_bytes=%zu\n",
               f.shape.name, f.bytes(), f.bytes() / (1024.0 * 1024.0),
               f.weights[3].size());

        for (int repeat = 0; repeat < 2 && pass; ++repeat) {
            pass = check_all(f, 7, metrics, repeat);
            ++calls;
            scratch = std::max(scratch, f.scratch_bytes());
            pass = pass && usage(anchor.bytes() + f.bytes() + scratch);
        }

        FullGqaFixture changed(shapes[c], (int)c + 17);
        pass = pass && changed.upload(7) == 0;
        pass = pass && usage(anchor.bytes() + changed.bytes() + scratch);
        pass = pass && check_all(changed, 7, metrics, 30 + (int)c);
        ++calls;

        /* Return to the small layer after every size, including the 397B case,
         * to cover larger->smaller scratch reuse and independent residency. */
        pass = pass && check_all(anchor, 0, metrics, 40 + (int)c);
        ++calls;
        pass = pass && usage(anchor.bytes() + changed.bytes() + scratch);
    }

    gpu_shutdown();
    gpu_shutdown();
    pass = pass && !gpu_is_initialized() && usage(0);

    const char* names[] = {"GQA Q+gate", "GQA K", "GQA V", "GQA Wo"};
    for (int m = 0; m < 4; ++m) {
        print_metrics(names[m], calls, metrics[m]);
        pass = pass && metrics[m].pass && metrics[m].values != 0;
    }
    printf("GQA/Wo tolerance: abs<=%.3g + %.3g*|reference|; "
           "subnormal absolute term scaled by FP16 scale\n",
           kAbsTol, kRelTol);
    printf("HIP GQA Wo parity %s (four-matrix rollback, repeat, resize, shutdown)\n",
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
