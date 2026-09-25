#pragma once
/* Portable heap temporaries must be released before returning. */
#include "../platform/runtime.h"
/*
 * Q8_0 Dequantization — simplest quantized format
 * 32 weights per block: 1 FP16 scale + 32 int8 values
 * Block size: 34 bytes per 32 weights
 * Dequant: weight = scale * int8_value
 */

#include <stdint.h>
#include <immintrin.h>

#define Q8_BLOCK_SIZE 34
#define Q8_QK 32

typedef struct {
    uint16_t d;       /* FP16 scale */
    int8_t qs[32];    /* quantized values */
} block_q8_0;

/* Q8_0 dot with float activations — AVX-512 */
static inline float q8_dot_block(const block_q8_0* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    const int8_t* qs = block->qs;

    __m512 acc = _mm512_setzero_ps();
    __m128i q8_lo = _mm_loadu_si128((const __m128i*)qs);
    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q8_lo)), _mm512_loadu_ps(y), acc);
    __m128i q8_hi = _mm_loadu_si128((const __m128i*)(qs + 16));
    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q8_hi)), _mm512_loadu_ps(y + 16), acc);

    return d * _mm512_reduce_add_ps(acc);
}

/* Q8_0 dot with Q8_K pre-quantized activations — exact AVX2 integer accumulation */
static inline float q8_dot_q8k(const block_q8_0* w, const block_q8_K* a) {
    float d = fp16_to_fp32(w->d) * a->d;

    /* Load 32 weight int8 and 32 activation int8 */
    __m256i qw = _mm256_loadu_si256((const __m256i*)w->qs);
    __m256i qa = _mm256_loadu_si256((const __m256i*)a->qs);

    /* Widen signed bytes before multiplying. This AVX2 baseline needs no
     * VNNI and handles -128 exactly: negating an int8 activation for the old
     * unsigned/signed trick overflows for that value. Pair sums fit int32. */
    __m256i wlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qw));
    __m256i whi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qw, 1));
    __m256i alo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qa));
    __m256i ahi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qa, 1));
    __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(wlo, alo),
                                  _mm256_madd_epi16(whi, ahi));

    /* Horizontal sum of 8 int32 values */
    __m256 vsum = _mm256_cvtepi32_ps(acc);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float result;
    _mm_store_ss(&result, s4);
    return d * result;
}

static inline void q8_matvec(
    float* out, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q8_QK;
    const block_q8_0* blocks = (const block_q8_0*)W;

    /* Pre-quantize activation to Q8_K for integer dot product */
    /* Q8_0 has 32 values/block, Q8_K has 256 values/block.
       We need Q8_K blocks aligned with Q8_0 blocks.
       Since Q8_0 block = 32 values and Q8_K block = 256 values = 8 Q8_0 blocks,
       we quantize per Q8_K block (256 values). */
    int q8k_blocks = in_dim / Q8K_QK;
    block_q8_K* x_q8k = NULL;
    if (q8k_blocks > 0 && out_dim >= 128) {
        x_q8k = (block_q8_K*)lm_temp_alloc(q8k_blocks * sizeof(block_q8_K));
        if (x_q8k) quantize_row_q8_K(x_q8k, x, q8k_blocks * Q8K_QK);
    }

    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_q8_0* row_blocks = &blocks[row * blocks_per_row];
        if (x_q8k) {
            /* Integer path: process 8 Q8_0 blocks per Q8_K block */
            for (int qb = 0; qb < q8k_blocks; qb++) {
                /* Each Q8_K block covers 8 consecutive Q8_0 blocks (256/32) */
                /* Use per-Q8_0-block dot product with the Q8_K activation */
                /* For now: use float path per Q8_0 block but with Q8_K scale */
                /* TODO: fuse 8 Q8_0 blocks into one Q8_K integer dot */
                for (int sb = 0; sb < 8; sb++) {
                    int bidx = qb * 8 + sb;
                    if (bidx < blocks_per_row) {
                        /* Create a mini Q8_K for this 32-value sub-block */
                        block_q8_K mini;
                        mini.d = x_q8k[qb].d;
                        memcpy(mini.qs, x_q8k[qb].qs + sb * 32, 32);
                        sum += q8_dot_q8k(&row_blocks[bidx], &mini);
                    }
                }
            }
        } else {
            int b;
            for (b = 0; b < blocks_per_row; b++)
                sum += q8_dot_block(&row_blocks[b], &x[b * Q8_QK]);
        }
        out[row] = sum;
    }
    if (x_q8k) free(x_q8k);
}

/* Dequantize a full Q8_0 row to FP32 (for embedding lookup) */
static inline void q8_dequant_row(float* out, const void* data, int n) {
    const block_q8_0* blocks = (const block_q8_0*)data;
    int num_blocks = n / Q8_QK;
    int b, j;
    for (b = 0; b < num_blocks; b++) {
        float d = fp16_to_fp32(blocks[b].d);
        for (j = 0; j < Q8_QK; j++) {
            out[b * Q8_QK + j] = d * (float)blocks[b].qs[j];
        }
    }
}
