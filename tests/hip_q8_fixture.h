#pragma once
/* Shared independent host oracle and fixed tolerances from the validated Q8
 * suite. No device code is used to generate expected values. */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../engine/runtime/gpu_offload.h"
#include "../engine/runtime/gpu_offload_hip_test.h"

namespace {

constexpr int kQK = 32;
constexpr int kBlockBytes = 34;
constexpr double kAbsTol = 5e-2;
constexpr double kRelTol = 5e-5;
constexpr double kRelativeFloor = 1e-6;

struct Fixture {
    const char* name;
    int rows;
    int cols;
    int weight_mode;
    int input_mode;
    uint16_t scale;
};

struct Metrics {
    size_t values = 0;
    double max_abs = 0.0;
    double sum_sq = 0.0;
    double max_rel = 0.0;
    std::string worst_case;
    int worst_row = -1;
    bool pass = true;
};

static float fp16_reference(uint16_t bits) {
    int sign = (bits & 0x8000u) ? -1 : 1;
    unsigned exp = (bits >> 10) & 0x1fu;
    unsigned mant = bits & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) return sign < 0 ? -0.0f : 0.0f;
        return (float)(sign * std::ldexp((double)mant, -24));
    }
    if (exp == 31) {
        if (mant == 0)
            return sign < 0 ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::quiet_NaN();
    }
    return (float)(sign * std::ldexp(1.0 + (double)mant / 1024.0,
                                     (int)exp - 15));
}

static uint32_t rng_next(uint32_t& state) {
    /* Fixed xorshift state makes every parity failure reproducible. */
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static int8_t make_weight(int mode, int row, int index, uint32_t& rng) {
    switch (mode) {
        case 0: return 0;
        case 1: return 127;
        case 2: return -128;
        case 3: return (index & 1) ? 127 : -128;
        default: {
            uint32_t v = rng_next(rng);
            /* Full signed range is intentional, including both -128 and +127. */
            int value = (int)(v & 0xffu) - 128;
            if (((row + index) % 97) == 0) value = -128;
            if (((row + index) % 89) == 0) value = 127;
            return (int8_t)value;
        }
    }
}

static float make_input(int mode, int index, uint32_t& rng) {
    switch (mode) {
        case 0: return 0.0f;
        case 1: return 0.125f + (float)(index % 17) * 0.03125f;
        case 2: return -(0.125f + (float)(index % 13) * 0.046875f);
        case 3: return (index & 1) ? 0.75f : -0.5f;
        default: {
            int centered = (int)(rng_next(rng) % 20001u) - 10000;
            return (float)centered / 10000.0f;
        }
    }
}

static void build_fixture(const Fixture& fixture,
                          std::vector<unsigned char>& weights,
                          std::vector<float>& input) {
    int blocks_per_row = fixture.cols / kQK;
    weights.assign((size_t)fixture.rows * blocks_per_row * kBlockBytes, 0);
    input.resize(fixture.cols);
    uint32_t wrng = 0x12345678u ^ (uint32_t)fixture.rows ^ ((uint32_t)fixture.cols << 8);
    uint32_t irng = 0x9e3779b9u ^ (uint32_t)fixture.cols;

    for (int i = 0; i < fixture.cols; ++i)
        input[i] = make_input(fixture.input_mode, i, irng);

    for (int row = 0; row < fixture.rows; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            unsigned char* block = weights.data() +
                ((size_t)row * blocks_per_row + b) * kBlockBytes;

            /* Vary ordinary-scale fixtures by block while retaining explicit
             * tiny-scale fixtures exactly as requested. */
            uint16_t scale = fixture.scale;
            if (fixture.scale == 0x3c00u && (b % 4) == 1) scale = 0x3800u;
            if (fixture.scale == 0x3c00u && (b % 4) == 2) scale = 0x4000u;
            if (fixture.scale == 0x3c00u && (b % 4) == 3) scale = 0x3400u;
            block[0] = (unsigned char)(scale & 0xffu);
            block[1] = (unsigned char)(scale >> 8);

            for (int j = 0; j < kQK; ++j) {
                int index = b * kQK + j;
                block[2 + j] = (unsigned char)make_weight(
                    fixture.weight_mode, row, index, wrng);
            }
        }
    }
}

static void cpu_reference(const std::vector<unsigned char>& weights,
                          const std::vector<float>& input,
                          int rows, int cols,
                          std::vector<float>& output) {
    int blocks_per_row = cols / kQK;
    output.assign(rows, 0.0f);
    for (int row = 0; row < rows; ++row) {
        double total = 0.0;
        for (int b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = weights.data() +
                ((size_t)row * blocks_per_row + b) * kBlockBytes;
            uint16_t scale_bits = (uint16_t)block[0] |
                                  ((uint16_t)block[1] << 8);
            double scale = fp16_reference(scale_bits);
            double dot = 0.0;
            for (int j = 0; j < kQK; ++j) {
                int8_t q = (int8_t)block[2 + j];
                dot += (double)q * input[b * kQK + j];
            }
            total += scale * dot;
        }
        output[row] = (float)total;
    }
}

static void update_metrics(Metrics& metrics, const char* case_name, int row,
                           float actual, float reference) {
    double abs_error = std::fabs((double)actual - reference);
    double ref_abs = std::fabs((double)reference);
    double rel_error = ref_abs > kRelativeFloor ? abs_error / ref_abs : 0.0;

    ++metrics.values;
    metrics.sum_sq += abs_error * abs_error;
    if (abs_error > metrics.max_abs) {
        metrics.max_abs = abs_error;
        metrics.worst_case = case_name;
        metrics.worst_row = row;
    }
    if (rel_error > metrics.max_rel) metrics.max_rel = rel_error;

    double allowed = kAbsTol + kRelTol * ref_abs;
    if (!std::isfinite(actual) || !std::isfinite(reference) ||
        abs_error > allowed)
        metrics.pass = false;
}

static void print_metrics(const char* name, int cases, const Metrics& metrics) {
    double rmse = metrics.values ? std::sqrt(metrics.sum_sq / metrics.values) : 0.0;
    printf("%s: cases=%d values=%zu max_abs=%.9g rmse=%.9g max_rel=%.9g "
           "worst=%s/row%d %s\n",
           name, cases, metrics.values, metrics.max_abs, rmse, metrics.max_rel,
           metrics.worst_case.empty() ? "<none>" : metrics.worst_case.c_str(),
           metrics.worst_row, metrics.pass ? "PASS" : "FAIL");
}

}  // namespace
