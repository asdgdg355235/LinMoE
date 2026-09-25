#pragma once
/* Portable heap temporaries must be released before returning. */
#include "../platform/runtime.h"
/*
 * Q5_K Dequantization — AVX-512 + Integer Accumulation
 *
 * Two paths:
 * 1. Float path (q5k_dot_block): AVX-512, used when no Q8_K activations available
 * 2. Integer path (q5k_dot_q8k): vpmaddubsw with pre-quantized Q8_K activations
 */

#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <immintrin.h>
#include "q8k_quant.h"

#define Q5K_BLOCK_SIZE 176
#define Q5K_QK 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} block_q5_K;

/* AVX-512 float path (fallback) */
static inline float q5k_dot_block(const block_q5_K* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    float dmin = fp16_to_fp32(block->dmin);
    uint8_t scales[8], mins[8];
    decode_q4k_scales(block->scales, scales, mins);
    const uint8_t* qs = block->qs;
    const uint8_t* qh = block->qh;
    __m512 acc = _mm512_setzero_ps();

    for (int sub = 0; sub < 8; sub++) {
        __m512 vsc = _mm512_set1_ps(d * (float)scales[sub]);
        __m512 vmn = _mm512_set1_ps(dmin * (float)mins[sub]);
        const float* yp = y + sub * 32;
        uint32_t hb;
        memcpy(&hb, qh + sub * 4, 4);

        /* First 16 weights */
        __m128i raw = _mm_loadl_epi64((const __m128i*)(qs + sub * 16));
        __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        __m128i interleaved = _mm_unpacklo_epi8(lo, hi_nib);
        __m512i q4 = _mm512_cvtepu8_epi32(interleaved);
        uint16_t hb16 = (uint16_t)(hb & 0xFFFF);
        __m512i vhb = _mm512_set_epi32(
            (hb16>>15)&1,(hb16>>14)&1,(hb16>>13)&1,(hb16>>12)&1,
            (hb16>>11)&1,(hb16>>10)&1,(hb16>>9)&1,(hb16>>8)&1,
            (hb16>>7)&1,(hb16>>6)&1,(hb16>>5)&1,(hb16>>4)&1,
            (hb16>>3)&1,(hb16>>2)&1,(hb16>>1)&1,hb16&1);
        __m512i q5 = _mm512_or_epi32(q4, _mm512_slli_epi32(vhb, 4));
        __m512 vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q5)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp), acc);

        /* Second 16 weights */
        raw = _mm_loadl_epi64((const __m128i*)(qs + sub * 16 + 8));
        lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        interleaved = _mm_unpacklo_epi8(lo, hi_nib);
        q4 = _mm512_cvtepu8_epi32(interleaved);
        hb16 = (uint16_t)(hb >> 16);
        vhb = _mm512_set_epi32(
            (hb16>>15)&1,(hb16>>14)&1,(hb16>>13)&1,(hb16>>12)&1,
            (hb16>>11)&1,(hb16>>10)&1,(hb16>>9)&1,(hb16>>8)&1,
            (hb16>>7)&1,(hb16>>6)&1,(hb16>>5)&1,(hb16>>4)&1,
            (hb16>>3)&1,(hb16>>2)&1,(hb16>>1)&1,hb16&1);
        q5 = _mm512_or_epi32(q4, _mm512_slli_epi32(vhb, 4));
        vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q5)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp + 16), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

/* Integer accumulation path with Q8_K activations */
static inline float q5k_dot_q8k(const block_q5_K* w, const block_q8_K* a) {
    float d = fp16_to_fp32(w->d) * a->d;
    float dmin = fp16_to_fp32(w->dmin) * a->d;
    uint8_t scales[8], mins_arr[8];
    decode_q4k_scales(w->scales, scales, mins_arr);

    const uint8_t* qs = w->qs;
    const uint8_t* qh = w->qh;
    const int8_t* q8 = a->qs;

    /* Min contribution via bsums */
    int32_t min_sum = 0;
    int sub;
    for (sub = 0; sub < 8; sub++)
        min_sum += (int32_t)mins_arr[sub] * (a->bsums[sub*2] + a->bsums[sub*2+1]);

    /* Load all 256 high bits */
    __m256i hbits = _mm256_loadu_si256((const __m256i*)qh);
    __m256i hmask = _mm256_set1_epi8(1);

    __m256i sumi = _mm256_setzero_si256();
    __m256i m4 = _mm256_set1_epi8(0x0F);

    /* GGML Q5_K layout: 32 bytes per 64-weight chunk, same as Q4_K.
     * Low nibbles of 32 bytes = sub-block A (32 weights), high nibbles = sub-block B.
     * High bits: qh[l] & (1 << chunk*2) for sub-block A, qh[l] & (1 << chunk*2+1) for B. */
    for (int chunk = 0; chunk < 4; chunk++) {
        int sb_lo = chunk * 2;
        int sb_hi = chunk * 2 + 1;
        __m256i vscale_lo = _mm256_set1_epi16((int16_t)scales[sb_lo]);
        __m256i vscale_hi = _mm256_set1_epi16((int16_t)scales[sb_hi]);

        /* Load 32 bytes of qs */
        __m256i q5bits = _mm256_loadu_si256((const __m256i*)(qs + chunk * 32));
        __m256i q5_lo = _mm256_and_si256(q5bits, m4);                           /* low nibbles → sub-block A */
        __m256i q5_hi = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);     /* high nibbles → sub-block B */

        /* High bits for sub-block A: bit (chunk*2) of each qh byte */
        __m256i hmask_lo = _mm256_set1_epi8(1 << (chunk * 2));
        __m256i qh_lo = _mm256_and_si256(hbits, hmask_lo);
        __m256i qh_lo_shifted = _mm256_slli_epi16(qh_lo, 4);
        /* Need to normalize: qh_lo has bit at position chunk*2, need at position 4 */
        /* Actually: shift right by chunk*2 then left by 4, net = left by (4-chunk*2) */
        /* Simpler: cmpeq to get 0xFF or 0, then AND with 16 */
        __m256i qh_lo_flag = _mm256_cmpeq_epi8(qh_lo, hmask_lo); /* 0xFF if bit set, 0 if not */
        __m256i q5v_lo = _mm256_or_si256(q5_lo, _mm256_and_si256(qh_lo_flag, _mm256_set1_epi8(16)));

        /* High bits for sub-block B: bit (chunk*2+1) */
        __m256i hmask_hi = _mm256_set1_epi8(1 << (chunk * 2 + 1));
        __m256i qh_hi = _mm256_and_si256(hbits, hmask_hi);
        __m256i qh_hi_flag = _mm256_cmpeq_epi8(qh_hi, hmask_hi);
        __m256i q5v_hi = _mm256_or_si256(q5_hi, _mm256_and_si256(qh_hi_flag, _mm256_set1_epi8(16)));

        /* Sub-block A × Q8 */
        __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8 + chunk * 64));
        __m256i p16l = _mm256_maddubs_epi16(q5v_lo, q8l);
        __m256i p32l = _mm256_madd_epi16(vscale_lo, p16l);
        sumi = _mm256_add_epi32(sumi, p32l);

        /* Sub-block B × Q8 */
        __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8 + chunk * 64 + 32));
        __m256i p16h = _mm256_maddubs_epi16(q5v_hi, q8h);
        __m256i p32h = _mm256_madd_epi16(vscale_hi, p16h);
        sumi = _mm256_add_epi32(sumi, p32h);
    }

    /* Convert to float */
    __m256 vd = _mm256_set1_ps(d);
    __m256 vsum = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi));
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float result;
    _mm_store_ss(&result, s4);
    return result - dmin * (float)min_sum;
}

/* Matvec with automatic pre-quantization */
static inline void q5k_matvec(
    float* out, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q5K_QK;
    const block_q5_K* weights = (const block_q5_K*)W;

    /* Pre-quantize activation to Q8_K */
    int num_blocks = in_dim / Q8K_QK;
    block_q8_K* x_q8 = (block_q8_K*)lm_temp_alloc(num_blocks * sizeof(block_q8_K));
    if (!x_q8) {
        /* Float fallback */
        int row;
        #pragma omp parallel for schedule(static) if(out_dim >= 128)
        for (row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const block_q5_K* rb = &weights[row * blocks_per_row];
            for (int b = 0; b < blocks_per_row; b++)
                sum += q5k_dot_block(&rb[b], &x[b * Q5K_QK]);
            out[row] = sum;
        }
        return;
    }
    quantize_row_q8_K(x_q8, x, in_dim);

    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_q5_K* rb = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++)
            sum += q5k_dot_q8k(&rb[b], &x_q8[b]);
        out[row] = sum;
    }
    free(x_q8);
}

/* Q5_K matvec with externally pre-quantized Q8_K activations */
static inline void q5k_matvec_q8k(
    float* out, const void* W, const block_q8_K* x_q8,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q5K_QK;
    const block_q5_K* weights = (const block_q5_K*)W;
    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_q5_K* rb = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++)
            sum += q5k_dot_q8k(&rb[b], &x_q8[b]);
        out[row] = sum;
    }
}
