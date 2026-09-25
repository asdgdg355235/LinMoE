#pragma once
/* Portable heap temporaries must be released before returning. */
#include "../platform/runtime.h"
/*
 * Q4_K_M Dequantization — Standard GGML format
 *
 * Much simpler than IQ2_XXS: no lookup tables, just linear scale+min.
 *
 * Block structure (144 bytes per 256 weights):
 *   d:      FP16 super-block scale
 *   dmin:   FP16 super-block min
 *   scales: uint8[12] — packed 6-bit scales and mins for 8 sub-blocks
 *   qs:     uint8[128] — packed 4-bit quantized values (2 per byte)
 *
 * Dequant: weight = d * scale * (q4_value) - dmin * min_value
 *
 * For Q4_K, each 256-weight block has 8 sub-blocks of 32 weights.
 * Each sub-block has its own 6-bit scale and 6-bit min.
 */

#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <immintrin.h>
#include "q8k_quant.h"

#define Q4K_BLOCK_SIZE 144
#define Q4K_QK 256

typedef struct {
    uint16_t d;         /* FP16 super-block scale */
    uint16_t dmin;      /* FP16 super-block min */
    uint8_t scales[12]; /* packed 6-bit scales and mins */
    uint8_t qs[128];    /* packed 4-bit weights (256 values, 2 per byte) */
} block_q4_K;

/*
 * Decode 6-bit scales and mins from packed format.
 * 12 bytes encode 8 scales + 8 mins, each 6 bits.
 * Layout (from GGML):
 *   bytes 0-3:  low 4 bits of scales[0..7] (packed as nibbles)
 *   bytes 4-7:  low 4 bits of mins[0..7]
 *   bytes 8-11: high 2 bits of scales[0..7] and mins[0..7]
 */
static inline void decode_q4k_scales(const uint8_t* q, uint8_t* scales, uint8_t* mins) {
    /* GGML Q4_K scale layout (get_scale_min_k4):
     * bytes 0-3: low 6 bits of scales[0..3]
     * bytes 4-7: low 6 bits of mins[0..3]
     * bytes 8-11: high bits for sub-blocks 4-7 + remaining 2 bits from 0-3 */
    scales[0] = q[0] & 63;  mins[0] = q[4] & 63;
    scales[1] = q[1] & 63;  mins[1] = q[5] & 63;
    scales[2] = q[2] & 63;  mins[2] = q[6] & 63;
    scales[3] = q[3] & 63;  mins[3] = q[7] & 63;
    scales[4] = (q[8]  & 0xF) | ((q[0] >> 6) << 4);
    scales[5] = (q[9]  & 0xF) | ((q[1] >> 6) << 4);
    scales[6] = (q[10] & 0xF) | ((q[2] >> 6) << 4);
    scales[7] = (q[11] & 0xF) | ((q[3] >> 6) << 4);
    mins[4]   = (q[8]  >> 4)  | ((q[4] >> 6) << 4);
    mins[5]   = (q[9]  >> 4)  | ((q[5] >> 6) << 4);
    mins[6]   = (q[10] >> 4)  | ((q[6] >> 6) << 4);
    mins[7]   = (q[11] >> 4)  | ((q[7] >> 6) << 4);
}

/*
 * AVX-512 / AVX2 fused dequant + dot product for one Q4_K block
 *
 * AVX-512 path: processes 16 weights per iteration using unpacklo nibble interleave
 * AVX2 fallback: processes 8 weights per iteration
 * Both eliminate the scalar _mm256_set_epi32 antipattern
 */
static inline float q4k_dot_block(const block_q4_K* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    float dmin = fp16_to_fp32(block->dmin);

    uint8_t scales[8], mins[8];
    decode_q4k_scales(block->scales, scales, mins);

    const uint8_t* qs = block->qs;
    __m512 acc = _mm512_setzero_ps();

    for (int sub = 0; sub < 8; sub++) {
        __m512 vsc = _mm512_set1_ps(d * (float)scales[sub]);
        __m512 vmn = _mm512_set1_ps(dmin * (float)mins[sub]);
        const uint8_t* qp = qs + sub * 16;
        const float* yp = y + sub * 32;

        /* First 8 bytes → 16 nibbles → 16 weights */
        __m128i raw = _mm_loadl_epi64((const __m128i*)qp);
        __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        __m128i interleaved = _mm_unpacklo_epi8(lo, hi);
        __m512i q = _mm512_cvtepu8_epi32(interleaved);
        __m512 vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp), acc);

        /* Second 8 bytes → next 16 weights */
        raw = _mm_loadl_epi64((const __m128i*)(qp + 8));
        lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        interleaved = _mm_unpacklo_epi8(lo, hi);
        q = _mm512_cvtepu8_epi32(interleaved);
        vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp + 16), acc);
    }

    return _mm512_reduce_add_ps(acc);
}

/*
 * Q4_K dot product with Q8_K pre-quantized activations (integer accumulation)
 * Uses vpmaddubsw for u8*s8 → s16 multiply, vpmaddwd for scale, stays in int32
 * Only converts to float ONCE per 256-weight block
 */
static inline float q4k_dot_q8k(const block_q4_K* w, const block_q8_K* a) {
    float d = fp16_to_fp32(w->d) * a->d;
    float dmin = fp16_to_fp32(w->dmin) * a->d;

    uint8_t scales[8], mins_arr[8];
    decode_q4k_scales(w->scales, scales, mins_arr);

    const uint8_t* qs = w->qs;
    const int8_t* q8 = a->qs;

    /* Compute min contribution using bsums */
    int32_t min_sum = 0;
    int sub;
    for (sub = 0; sub < 8; sub++) {
        min_sum += (int32_t)mins_arr[sub] * (a->bsums[sub*2] + a->bsums[sub*2+1]);
    }

    /* AVX2 integer dot product — matches GGML ggml_vec_dot_q4_K_q8_K layout.
     * Q4_K stores 32 bytes per 64-weight chunk: low nibbles = sub-block A (32 weights),
     * high nibbles = sub-block B (next 32 weights, DIFFERENT scale). */
    __m256i sumi = _mm256_setzero_si256();
    __m256i m4 = _mm256_set1_epi8(0x0F);

    for (int chunk = 0; chunk < 4; chunk++) { /* 4 chunks of 64 weights = 256 total */
        int sb_lo = chunk * 2;     /* sub-block index for low nibbles */
        int sb_hi = chunk * 2 + 1; /* sub-block index for high nibbles */

        /* Load 32 bytes of Q4 data */
        __m256i q4bits = _mm256_loadu_si256((const __m256i*)(qs + chunk * 32));
        __m256i q4l = _mm256_and_si256(q4bits, m4);                              /* low nibbles → sub-block A */
        __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);        /* high nibbles → sub-block B */

        /* Low nibbles × first 32 Q8_K values (sub-block A) */
        __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8 + chunk * 64));
        __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
        __m256i p32l = _mm256_madd_epi16(_mm256_set1_epi16((int16_t)scales[sb_lo]), p16l);
        sumi = _mm256_add_epi32(sumi, p32l);

        /* High nibbles × next 32 Q8_K values (sub-block B) */
        __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8 + chunk * 64 + 32));
        __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
        __m256i p32h = _mm256_madd_epi16(_mm256_set1_epi16((int16_t)scales[sb_hi]), p16h);
        sumi = _mm256_add_epi32(sumi, p32h);
    }

    /* Convert accumulated int32 to float */
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

/*
 * Q4_K matrix-vector multiply
 * Automatically pre-quantizes activation to Q8_K for integer dot products
 */
static inline void q4k_matvec(
    float* out, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q4K_QK;
    const block_q4_K* weights = (const block_q4_K*)W;

    /* Pre-quantize activation vector to Q8_K (once, reused for all rows) */
    int num_blocks = in_dim / Q8K_QK;
    block_q8_K* x_q8 = (block_q8_K*)lm_temp_alloc(num_blocks * sizeof(block_q8_K));
    if (!x_q8) {
        /* Fallback to float path */
        int row;
        #pragma omp parallel for schedule(static) if(out_dim >= 128)
        for (row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const block_q4_K* row_blocks = &weights[row * blocks_per_row];
            for (int b = 0; b < blocks_per_row; b++)
                sum += q4k_dot_block(&row_blocks[b], &x[b * Q4K_QK]);
            out[row] = sum;
        }
        return;
    }
    quantize_row_q8_K(x_q8, x, in_dim);

    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_q4_K* row_blocks = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++) {
            sum += q4k_dot_q8k(&row_blocks[b], &x_q8[b]);
        }
        out[row] = sum;
    }
    free(x_q8);
}

/* Q4_K matvec with externally pre-quantized Q8_K activations
   Software prefetching for next row's weight data */
static inline void q4k_matvec_q8k(
    float* out, const void* W, const block_q8_K* x_q8,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q4K_QK;
    const block_q4_K* weights = (const block_q4_K*)W;
    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        /* Prefetch next row's first blocks while computing current */
        if (row + 1 < out_dim) {
            _mm_prefetch((const char*)&weights[(row+1) * blocks_per_row], _MM_HINT_T0);
            _mm_prefetch((const char*)&weights[(row+1) * blocks_per_row] + 64, _MM_HINT_T0);
        }
        float sum = 0.0f;
        const block_q4_K* rb = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++)
            sum += q4k_dot_q8k(&rb[b], &x_q8[b]);
        out[row] = sum;
    }
}
