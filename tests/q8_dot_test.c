/* Independently check signed integer dot products, including the -128 corner
 * that cannot be negated in an int8 VNNI sign-adjustment trick. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
static inline float fp16_to_fp32(uint16_t h) { return _cvtsh_ss(h); }
#include "../engine/runtime/q8k_quant.h"
#include "../engine/runtime/q8_dequant.h"

int main(void) {
    block_q8_0 w = {0};
    block_q8_K a = {0};
    w.d = 0x3c00; /* FP16 1: integer expectations remain exactly representable. */
    a.d = 1;
    for (int wi = -128; wi <= 127; ++wi) {
        for (int ai = -128; ai <= 127; ++ai) {
            for (int lane = 0; lane < 32; ++lane) {
                w.qs[lane] = (int8_t)wi;
                a.qs[lane] = (int8_t)ai;
            }
            float actual = q8_dot_q8k(&w, &a);
            float expected = (float)(32 * wi * ai);
            if (actual != expected) {
                fprintf(stderr, "Q8 mismatch w=%d a=%d got=%g expected=%g\n", wi, ai, actual, expected);
                return 1;
            }
        }
    }
    /* Differing lane values catch reductions that omit or duplicate lanes. */
    int reference = 0;
    for (int i = 0; i < 32; ++i) {
        w.qs[i] = (int8_t)(i * 7 - 128);
        a.qs[i] = (int8_t)(127 - i * 5);
        reference += w.qs[i] * a.qs[i];
    }
    if (q8_dot_q8k(&w, &a) != (float)reference) return 1;
    puts("Q8 signed dot tests PASS (65536 pairs and mixed lanes)");
    return 0;
}
