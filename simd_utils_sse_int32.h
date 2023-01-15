/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <stdint.h>
#ifndef ARM
#include <immintrin.h>
#else
#include "sse2neon_wrapper.h"
#endif

static inline void add128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_store_si128((__m128i *) (dst + i), _mm_add_epi32(_mm_load_si128((__m128i *) (src1 + i)),
                                                                 _mm_load_si128((__m128i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_storeu_si128((__m128i *) (dst + i), _mm_add_epi32(_mm_loadu_si128((__m128i *) (src1 + i)),
                                                                  _mm_loadu_si128((__m128i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

// Works only for Integers stored on 32bits smaller than 16bits
static inline void mul128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src1_tmp = _mm_load_si128((__m128i *) (src1 + i));
            v4si src2_tmp = _mm_load_si128((__m128i *) (src2 + i));
            v4si src1_tmp2 = _mm_load_si128((__m128i *) (src1 + i + SSE_LEN_INT32));
            v4si src2_tmp2 = _mm_load_si128((__m128i *) (src2 + i + SSE_LEN_INT32));
            v4si tmp = _mm_mullo_epi32(src1_tmp, src2_tmp);
            v4si tmp2 = _mm_mullo_epi32(src1_tmp2, src2_tmp2);
            _mm_store_si128((__m128i *) (dst + i), tmp);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src1_tmp = _mm_loadu_si128((__m128i *) (src1 + i));
            v4si src2_tmp = _mm_loadu_si128((__m128i *) (src2 + i));
            v4si src1_tmp2 = _mm_loadu_si128((__m128i *) (src1 + i + SSE_LEN_INT32));
            v4si src2_tmp2 = _mm_loadu_si128((__m128i *) (src2 + i + SSE_LEN_INT32));
            v4si tmp = _mm_mullo_epi32(src1_tmp, src2_tmp);
            v4si tmp2 = _mm_mullo_epi32(src1_tmp2, src2_tmp2);
            _mm_storeu_si128((__m128i *) (dst + i), tmp);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_store_si128((__m128i *) (dst + i), _mm_sub_epi32(_mm_load_si128((__m128i *) (src1 + i)),
                                                                 _mm_load_si128((__m128i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_storeu_si128((__m128i *) (dst + i), _mm_sub_epi32(_mm_loadu_si128((__m128i *) (src1 + i)),
                                                                  _mm_loadu_si128((__m128i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void addc128s(int32_t *src, int32_t value, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

    const v4si tmp = _mm_set1_epi32(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_store_si128((__m128i *) (dst + i), _mm_add_epi32(tmp, _mm_load_si128((__m128i *) (src + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_storeu_si128((__m128i *) (dst + i), _mm_add_epi32(tmp, _mm_loadu_si128((__m128i *) (src + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void vectorSlope128s(int *dst, int len, int offset, int slope)
{
    v4si coef = _mm_set_epi32(3 * slope, 2 * slope, slope, 0);
    v4si slope8_vec = _mm_set1_epi32(8 * slope);
    v4si curVal = _mm_add_epi32(_mm_set1_epi32(offset), coef);
    v4si curVal2 = _mm_add_epi32(_mm_set1_epi32(offset), coef);
    curVal2 = _mm_add_epi32(curVal2, _mm_set1_epi32(4 * slope));

    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
        _mm_store_si128((__m128i *) dst, curVal);
        _mm_store_si128((__m128i *) (dst + SSE_LEN_INT32), curVal2);
    } else {
        _mm_storeu_si128((__m128i *) dst, curVal);
        _mm_storeu_si128((__m128i *) (dst + SSE_LEN_INT32), curVal2);
    }

    if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 2 * SSE_LEN_INT32; i < stop_len; i += 2 * SSE_LEN_INT32) {
            curVal = _mm_add_epi32(curVal, slope8_vec);
            _mm_store_si128((__m128i *) (dst + i), curVal);
            curVal2 = _mm_add_epi32(curVal2, slope8_vec);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), curVal2);
        }
    } else {
        for (int i = 2 * SSE_LEN_INT32; i < stop_len; i += 2 * SSE_LEN_INT32) {
            curVal = _mm_add_epi32(curVal, slope8_vec);
            _mm_storeu_si128((__m128i *) (dst + i), curVal);
            curVal2 = _mm_add_epi32(curVal2, slope8_vec);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * i;
    }
}

static inline void sum128s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    __attribute__((aligned(SSE_LEN_BYTES))) int32_t accumulate[SSE_LEN_INT32] = {0, 0, 0, 0};
    int32_t tmp_acc = 0;
    v4si vec_acc1 = _mm_setzero_si128();  // initialize the vector accumulator
    v4si vec_acc2 = _mm_setzero_si128();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si vec_tmp1 = _mm_load_si128((__m128i *) (src + i));
            vec_acc1 = _mm_add_epi32(vec_acc1, vec_tmp1);
            v4si vec_tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
            vec_acc2 = _mm_add_epi32(vec_acc2, vec_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si vec_tmp1 = _mm_loadu_si128((__m128i *) (src + i));
            vec_acc1 = _mm_add_epi32(vec_acc1, vec_tmp1);
            v4si vec_tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
            vec_acc2 = _mm_add_epi32(vec_acc2, vec_tmp2);
        }
    }
    vec_acc1 = _mm_add_epi32(vec_acc1, vec_acc2);
    _mm_store_si128((__m128i *) accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

    *dst = tmp_acc;
}

// Experimental

static inline void copy128s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
        _mm_store_si128((__m128i *) (dst + i), _mm_load_si128((__m128i *) (src + i)));
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void copy128s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
        __m128i tmp1 = _mm_load_si128((__m128i *) (src + i));
        __m128i tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
        _mm_store_si128((__m128i *) (dst + i), tmp1);
        _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), tmp2);
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy128s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
        _mm_stream_si128((__m128i *) (dst + i), _mm_stream_load_si128((__m128i *) (src + i)));
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}


static inline void fast_copy128s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
        __m128i tmp1 = _mm_stream_load_si128((__m128i *) (src + i));
        __m128i tmp2 = _mm_stream_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
        _mm_stream_si128((__m128i *) (dst + i), tmp1);
        _mm_stream_si128((__m128i *) (dst + i + SSE_LEN_INT32), tmp2);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy128s_4(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (4 * SSE_LEN_INT32);
    stop_len *= (4 * SSE_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 4 * SSE_LEN_INT32) {
        __m128i tmp1 = _mm_stream_load_si128((__m128i *) (src + i));
        __m128i tmp2 = _mm_stream_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
        __m128i tmp3 = _mm_stream_load_si128((__m128i *) (src + i + 2 * SSE_LEN_INT32));
        __m128i tmp4 = _mm_stream_load_si128((__m128i *) (src + i + 3 * SSE_LEN_INT32));
        _mm_stream_si128((__m128i *) (dst + i), tmp1);
        _mm_stream_si128((__m128i *) (dst + i + SSE_LEN_INT32), tmp2);
        _mm_stream_si128((__m128i *) (dst + i + 2 * SSE_LEN_INT32), tmp3);
        _mm_stream_si128((__m128i *) (dst + i + 3 * SSE_LEN_INT32), tmp4);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}


// Adapted from NEON2SSE (does not exists for X86)
static inline __m128i _mm_absdiff_epi16(__m128i a, __m128i b)
{
#ifndef ARM
    __m128i cmp, difab, difba;
    cmp = _mm_cmpgt_epi16(a, b);
    difab = _mm_sub_epi16(a, b);
    difba = _mm_sub_epi16(b, a);
#if 1  // should be faster
    return _mm_blendv_epi8(difba, difab, cmp);
#else
    difab = _mm_and_si128(cmp, difab);
    difba = _mm_andnot_si128(cmp, difba);
    return _mm_or_si128(difab, difba);
#endif

#else
    return vreinterpretq_m128i_s16(vabdq_s16(vreinterpretq_s16_m128i(a), vreinterpretq_s16_m128i(b)));
#endif
}

// Adapted from NEON2SSE (does not exists for X86)
static inline __m128i _mm_absdiff_epi32(__m128i a, __m128i b)
{
#ifndef ARM
    __m128i cmp, difab, difba;
    cmp = _mm_cmpgt_epi32(a, b);
    difab = _mm_sub_epi32(a, b);
    difba = _mm_sub_epi32(b, a);
#if 1  // should be faster
    return _mm_blendv_epi8(difba, difab, cmp);
#else
    difab = _mm_and_si128(cmp, difab);
    difba = _mm_andnot_si128(cmp, difba);
    return _mm_or_si128(difab, difba);
#endif

#else
    return vreinterpretq_m128i_s32(vabdq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)));
#endif
}

static inline __m128i _mm_absdiff_epi8(__m128i a, __m128i b)
{
#ifndef ARM
    __m128i cmp, difab, difba;
    cmp = _mm_cmpgt_epi8(a, b);
    difab = _mm_sub_epi8(a, b);
    difba = _mm_sub_epi8(b, a);
#if 1  // should be faster
    return _mm_blendv_epi8(difba, difab, cmp);
#else
    difab = _mm_and_si128(cmp, difab);
    difba = _mm_andnot_si128(cmp, difba);
    return _mm_or_si128(difab, difba);
#endif

#else
    return vreinterpretq_m128i_s8(vabdq_s8(vreinterpretq_s8_m128i(a), vreinterpretq_s8_m128i(b)));
#endif
}

static inline void absdiff16s_128s(int16_t *src1, int16_t *src2, int16_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT16;
    stop_len *= SSE_LEN_INT16;


    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT16) {
            __m128i a = _mm_load_si128((__m128i *) (src1 + i));
            __m128i b = _mm_load_si128((__m128i *) (src2 + i));
            _mm_store_si128((__m128i *) (dst + i), _mm_absdiff_epi16(a, b));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT16) {
            __m128i a = _mm_loadu_si128((__m128i *) (src1 + i));
            __m128i b = _mm_loadu_si128((__m128i *) (src2 + i));
            _mm_storeu_si128((__m128i *) (dst + i), _mm_absdiff_epi16(a, b));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = abs(src1[i] - src2[i]);
    }
}

/*
static inline void print8i(__m128i v)
{
    int16_t *p = (int16_t *) &v;
#ifndef __SSE2__
    _mm_empty();
#endif
    printf("[%d, %d, %d, %d,%d, %d, %d, %d]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}*/

static inline void powerspect16s_128s_interleaved(complex16s_t *src, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            __m128i reim = _mm_load_si128((__m128i *) ((const int16_t *) src + j));
            // print8i(reim); printf("\n");
            _mm_store_si128((__m128i *) (dst + i), _mm_madd_epi16(reim, reim));
            j += SSE_LEN_INT16;
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            __m128i reim = _mm_loadu_si128((__m128i *) ((const int16_t *) src + j));
            _mm_storeu_si128((__m128i *) (dst + i), _mm_madd_epi16(reim, reim));
            j += SSE_LEN_INT16;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (int32_t) src[i].re * (int32_t) src[i].re + (int32_t) src[i].im * (int32_t) src[i].im;
    }
}

// Works with positive scale_factor (divides final value)
static inline void sum16s32s128(int16_t *src, int len, int32_t *dst, int scale_factor)
{
    int stop_len = len / (4 * SSE_LEN_INT16);
    stop_len *= (4 * SSE_LEN_INT16);

    __attribute__((aligned(SSE_LEN_BYTES))) int32_t accumulate[SSE_LEN_INT32];
    int32_t tmp_acc = 0;
    int16_t scale = 1 << scale_factor;
    v4si one = _mm_set1_epi16(1);
    v4si vec_acc1 = _mm_setzero_si128();  // initialize the vector accumulator
    v4si vec_acc2 = _mm_setzero_si128();  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_INT16) {
            v4si vec_src_tmp = _mm_load_si128((__m128i *) ((const int16_t *) src + i));
            v4si vec_src_tmp2 = _mm_load_si128((__m128i *) ((const int16_t *) src + i + SSE_LEN_INT16));
            v4si vec_src_tmp3 = _mm_load_si128((__m128i *) ((const int16_t *) src + i + 2 * SSE_LEN_INT16));
            v4si vec_src_tmp4 = _mm_load_si128((__m128i *) ((const int16_t *) src + i + 3 * SSE_LEN_INT16));
            vec_src_tmp = _mm_madd_epi16(vec_src_tmp, one);
            vec_src_tmp2 = _mm_madd_epi16(vec_src_tmp2, one);
            vec_src_tmp3 = _mm_madd_epi16(vec_src_tmp3, one);
            vec_src_tmp4 = _mm_madd_epi16(vec_src_tmp4, one);
            vec_src_tmp = _mm_add_epi32(vec_src_tmp, vec_src_tmp2);
            vec_src_tmp3 = _mm_add_epi32(vec_src_tmp3, vec_src_tmp4);
            vec_acc1 = _mm_add_epi32(vec_src_tmp, vec_acc1);
            vec_acc2 = _mm_add_epi32(vec_src_tmp3, vec_acc2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_INT16) {
            v4si vec_src_tmp = _mm_loadu_si128((__m128i *) ((const int16_t *) src + i));
            v4si vec_src_tmp2 = _mm_loadu_si128((__m128i *) ((const int16_t *) src + i + SSE_LEN_INT16));
            v4si vec_src_tmp3 = _mm_loadu_si128((__m128i *) ((const int16_t *) src + i + 2 * SSE_LEN_INT16));
            v4si vec_src_tmp4 = _mm_loadu_si128((__m128i *) ((const int16_t *) src + i + 3 * SSE_LEN_INT16));
            vec_src_tmp = _mm_madd_epi16(vec_src_tmp, one);
            vec_src_tmp2 = _mm_madd_epi16(vec_src_tmp2, one);
            vec_src_tmp3 = _mm_madd_epi16(vec_src_tmp3, one);
            vec_src_tmp4 = _mm_madd_epi16(vec_src_tmp4, one);
            vec_src_tmp = _mm_add_epi32(vec_src_tmp, vec_src_tmp2);
            vec_src_tmp3 = _mm_add_epi32(vec_src_tmp3, vec_src_tmp4);
            vec_acc1 = _mm_add_epi32(vec_src_tmp, vec_acc1);
            vec_acc2 = _mm_add_epi32(vec_src_tmp3, vec_acc2);
        }
    }

    vec_acc1 = _mm_add_epi32(vec_acc1, vec_acc2);
    _mm_store_si128((__m128i *) accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += (int32_t) src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

    tmp_acc /= scale;
    *dst = tmp_acc;
}

static inline void flip128s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    int mini = ((len - 1) < (2 * SSE_LEN_INT32)) ? (len - 1) : (2 * SSE_LEN_INT32);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst + len - SSE_LEN_INT32), SSE_LEN_BYTES)) {
        for (int i = 2 * SSE_LEN_INT32; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_load_si128((__m128i *) ((const int32_t *) src + i));  // load a,b,c,d
            v4si src_tmp2 = _mm_load_si128((__m128i *) ((const int32_t *) src + i + SSE_LEN_INT32));
            v4si src_tmp_slip = _mm_shuffle_epi32(src_tmp, IMM8_FLIP_VEC);  // rotate vec from abcd to bcba
            v4si src_tmp_slip2 = _mm_shuffle_epi32(src_tmp2, IMM8_FLIP_VEC);
            _mm_store_si128((__m128i *) (dst + len - i - SSE_LEN_INT32), src_tmp_slip);  // store the flipped vector
            _mm_store_si128((__m128i *) (dst + len - i - 2 * SSE_LEN_INT32), src_tmp_slip2);
        }
    } else {
        for (int i = 2 * SSE_LEN_INT32; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_loadu_si128((__m128i *) (src + i));  // load a,b,c,d
            v4si src_tmp2 = _mm_loadu_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si src_tmp_slip = _mm_shuffle_epi32(src_tmp, IMM8_FLIP_VEC);  // rotate vec from abcd to bcba
            v4si src_tmp_slip2 = _mm_shuffle_epi32(src_tmp2, IMM8_FLIP_VEC);
            _mm_storeu_si128((__m128i *) (dst + len - i - SSE_LEN_INT32), src_tmp_slip);  // store the flipped vector
            _mm_storeu_si128((__m128i *) (dst + len - i - 2 * SSE_LEN_INT32), src_tmp_slip2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void maxevery128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src1_tmp = _mm_load_si128((__m128i *) ((const int32_t *) src1 + i));
            v4si src2_tmp = _mm_load_si128((__m128i *) ((const int32_t *) src2 + i));
            v4si src1_tmp2 = _mm_load_si128((__m128i *) ((const int32_t *) src1 + i + SSE_LEN_INT32));
            v4si src2_tmp2 = _mm_load_si128((__m128i *) ((const int32_t *) src2 + i + SSE_LEN_INT32));
            v4si max1 = _mm_max_epi32(src1_tmp, src2_tmp);
            v4si max2 = _mm_max_epi32(src1_tmp2, src2_tmp2);
            _mm_store_si128((__m128i *) (dst + i), max1);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), max2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src1_tmp = _mm_loadu_si128((__m128i *) (src1 + i));
            v4si src2_tmp = _mm_loadu_si128((__m128i *) (src2 + i));
            v4si src1_tmp2 = _mm_loadu_si128((__m128i *) (src1 + i + SSE_LEN_INT32));
            v4si src2_tmp2 = _mm_loadu_si128((__m128i *) (src2 + i + SSE_LEN_INT32));
            v4si max1 = _mm_max_epi32(src1_tmp, src2_tmp);
            v4si max2 = _mm_max_epi32(src1_tmp2, src2_tmp2);
            _mm_storeu_si128((__m128i *) (dst + i), max1);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), max2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src1_tmp = _mm_load_si128((__m128i *) ((const int32_t *) src1 + i));
            v4si src2_tmp = _mm_load_si128((__m128i *) ((const int32_t *) src2 + i));
            v4si src1_tmp2 = _mm_load_si128((__m128i *) ((const int32_t *) src1 + i + SSE_LEN_INT32));
            v4si src2_tmp2 = _mm_load_si128((__m128i *) ((const int32_t *) src2 + i + SSE_LEN_INT32));
            v4si min1 = _mm_min_epi32(src1_tmp, src2_tmp);
            v4si min2 = _mm_min_epi32(src1_tmp2, src2_tmp2);
            _mm_store_si128((__m128i *) (dst + i), min1);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), min2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src1_tmp = _mm_loadu_si128((__m128i *) (src1 + i));
            v4si src2_tmp = _mm_loadu_si128((__m128i *) (src2 + i));
            v4si src1_tmp2 = _mm_loadu_si128((__m128i *) (src1 + i + SSE_LEN_INT32));
            v4si src2_tmp2 = _mm_loadu_si128((__m128i *) (src2 + i + SSE_LEN_INT32));
            v4si min1 = _mm_min_epi32(src1_tmp, src2_tmp);
            v4si min2 = _mm_min_epi32(src1_tmp2, src2_tmp2);
            _mm_storeu_si128((__m128i *) (dst + i), min1);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), min2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void minmax128s(int32_t *src, int len, int32_t *min_value, int32_t *max_value)
{
    int stop_len = (len - SSE_LEN_INT32) / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    int32_t min_s[SSE_LEN_INT32] __attribute__((aligned(SSE_LEN_BYTES)));
    int32_t max_s[SSE_LEN_INT32] __attribute__((aligned(SSE_LEN_BYTES)));
    v4si max_v, min_v, max_v2, min_v2;
    v4si src_tmp, src_tmp2;

    int32_t min_tmp = src[0];
    int32_t max_tmp = src[0];

    if (len >= SSE_LEN_INT32) {
        if (isAligned((uintptr_t) (src), SSE_LEN_BYTES)) {
            src_tmp = _mm_load_si128((__m128i *) ((const int32_t *) src + 0));
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = SSE_LEN_INT32; i < stop_len; i += 2 * SSE_LEN_INT32) {
                src_tmp = _mm_load_si128((__m128i *) ((const int32_t *) src + i));
                src_tmp2 = _mm_load_si128((__m128i *) ((const int32_t *) src + i + SSE_LEN_INT32));
                max_v = _mm_max_epi32(max_v, src_tmp);
                min_v = _mm_min_epi32(min_v, src_tmp);
                max_v2 = _mm_max_epi32(max_v2, src_tmp2);
                min_v2 = _mm_min_epi32(min_v2, src_tmp2);
            }
        } else {
            src_tmp = _mm_loadu_si128((__m128i *) (src + 0));
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = SSE_LEN_INT32; i < stop_len; i += 2 * SSE_LEN_INT32) {
                src_tmp = _mm_loadu_si128((__m128i *) ((const int32_t *) src + i));
                src_tmp2 = _mm_loadu_si128((__m128i *) ((const int32_t *) src + i + SSE_LEN_INT32));
                max_v = _mm_max_epi32(max_v, src_tmp);
                min_v = _mm_min_epi32(min_v, src_tmp);
                max_v2 = _mm_max_epi32(max_v2, src_tmp2);
                min_v2 = _mm_min_epi32(min_v2, src_tmp2);
            }
        }

        max_v = _mm_max_epi32(max_v, max_v2);
        min_v = _mm_min_epi32(min_v, min_v2);

        _mm_store_si128((__m128i *) (max_s), max_v);
        _mm_store_si128((__m128i *) (min_s), min_v);

        max_tmp = max_s[0];
        max_tmp = max_tmp > max_s[1] ? max_tmp : max_s[1];
        max_tmp = max_tmp > max_s[2] ? max_tmp : max_s[2];
        max_tmp = max_tmp > max_s[3] ? max_tmp : max_s[3];

        min_tmp = min_s[0];
        min_tmp = min_tmp < min_s[1] ? min_tmp : min_s[1];
        min_tmp = min_tmp < min_s[2] ? min_tmp : min_s[2];
        min_tmp = min_tmp < min_s[3] ? min_tmp : min_s[3];
    }

    for (int i = stop_len; i < len; i++) {
        max_tmp = max_tmp > src[i] ? max_tmp : src[i];
        min_tmp = min_tmp < src[i] ? min_tmp : src[i];
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

static inline void threshold128_gt_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si tmp = _mm_set1_epi32(value);

    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_load_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si dst_tmp = _mm_min_epi32(src_tmp, tmp);
            v4si dst_tmp2 = _mm_min_epi32(src_tmp2, tmp);
            _mm_store_si128((__m128i *) (dst + i), dst_tmp);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_loadu_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_loadu_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si dst_tmp = _mm_min_epi32(src_tmp, tmp);
            v4si dst_tmp2 = _mm_min_epi32(src_tmp2, tmp);
            _mm_storeu_si128((__m128i *) (dst + i), dst_tmp);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}


static inline void threshold128_gtabs_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si pval = _mm_set1_epi32(value);
    const v4si mval = _mm_set1_epi32(-value);

    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_load_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si src_abs = _mm_abs_epi32(src_tmp);
            v4si src_abs2 = _mm_abs_epi32(src_tmp2);
            v4si eqmask = _mm_cmpeq_epi32(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4si eqmask2 = _mm_cmpeq_epi32(src_abs2, src_tmp2);
            v4si min = _mm_min_epi32(src_tmp, pval);
            v4si min2 = _mm_min_epi32(src_tmp2, pval);
            v4si max = _mm_max_epi32(src_tmp, mval);
            v4si max2 = _mm_max_epi32(src_tmp2, mval);
            v4si dst_tmp = _mm_blendv_epi8(max, min, eqmask);
            v4si dst_tmp2 = _mm_blendv_epi8(max2, min2, eqmask2);
            _mm_store_si128((__m128i *) (dst + i), dst_tmp);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_loadu_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_loadu_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si src_abs = _mm_abs_epi32(src_tmp);
            v4si src_abs2 = _mm_abs_epi32(src_tmp2);
            v4si eqmask = _mm_cmpeq_epi32(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4si eqmask2 = _mm_cmpeq_epi32(src_abs2, src_tmp2);
            v4si min = _mm_min_epi32(src_tmp, pval);
            v4si min2 = _mm_min_epi32(src_tmp2, pval);
            v4si max = _mm_max_epi32(src_tmp, mval);
            v4si max2 = _mm_max_epi32(src_tmp2, mval);
            v4si dst_tmp = _mm_blendv_epi8(max, min, eqmask);
            v4si dst_tmp2 = _mm_blendv_epi8(max2, min2, eqmask2);
            _mm_storeu_si128((__m128i *) (dst + i), dst_tmp);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] >= 0) {
            dst[i] = src[i] > value ? value : src[i];
        } else {
            dst[i] = src[i] < (-value) ? (-value) : src[i];
        }
    }
}

static inline void threshold128_lt_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si tmp = _mm_set1_epi32(value);

    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_load_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si dst_tmp = _mm_max_epi32(src_tmp, tmp);
            v4si dst_tmp2 = _mm_max_epi32(src_tmp2, tmp);
            _mm_store_si128((__m128i *) (dst + i), dst_tmp);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_loadu_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_loadu_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si dst_tmp = _mm_max_epi32(src_tmp, tmp);
            v4si dst_tmp2 = _mm_max_epi32(src_tmp2, tmp);
            _mm_storeu_si128((__m128i *) (dst + i), dst_tmp);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

static inline void threshold128_ltabs_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si pval = _mm_set1_epi32(value);
    const v4si mval = _mm_set1_epi32(-value);

    int stop_len = len / (2 * SSE_LEN_INT32);
    stop_len *= (2 * SSE_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_load_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si src_abs = _mm_abs_epi32(src_tmp);
            v4si src_abs2 = _mm_abs_epi32(src_tmp2);
            v4si eqmask = _mm_cmpeq_epi32(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4si eqmask2 = _mm_cmpeq_epi32(src_abs2, src_tmp2);
            v4si max = _mm_max_epi32(src_tmp, pval);
            v4si max2 = _mm_max_epi32(src_tmp2, pval);
            v4si min = _mm_min_epi32(src_tmp, mval);
            v4si min2 = _mm_min_epi32(src_tmp2, mval);
            v4si dst_tmp = _mm_blendv_epi8(min, max, eqmask);
            v4si dst_tmp2 = _mm_blendv_epi8(min2, max2, eqmask2);
            _mm_store_si128((__m128i *) (dst + i), dst_tmp);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_INT32) {
            v4si src_tmp = _mm_loadu_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_loadu_si128((__m128i *) (src + i + SSE_LEN_INT32));
            v4si src_abs = _mm_abs_epi32(src_tmp);
            v4si src_abs2 = _mm_abs_epi32(src_tmp2);
            v4si eqmask = _mm_cmpeq_epi32(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4si eqmask2 = _mm_cmpeq_epi32(src_abs2, src_tmp2);
            v4si max = _mm_max_epi32(src_tmp, pval);
            v4si max2 = _mm_max_epi32(src_tmp2, pval);
            v4si min = _mm_min_epi32(src_tmp, mval);
            v4si min2 = _mm_min_epi32(src_tmp2, mval);
            v4si dst_tmp = _mm_blendv_epi8(min, max, eqmask);
            v4si dst_tmp2 = _mm_blendv_epi8(min2, max2, eqmask2);
            _mm_storeu_si128((__m128i *) (dst + i), dst_tmp);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] >= 0) {
            dst[i] = src[i] < value ? value : src[i];
        } else {
            dst[i] = src[i] > (-value) ? (-value) : src[i];
        }
    }
}

static inline void threshold128_ltval_gtval_s(int32_t *src, int32_t *dst, int len, int32_t ltlevel, int32_t ltvalue, int32_t gtlevel, int32_t gtvalue)
{
    const v4si ltlevel_v = _mm_set1_epi32(ltlevel);
    const v4si ltvalue_v = _mm_set1_epi32(ltvalue);
    const v4si gtlevel_v = _mm_set1_epi32(gtlevel);
    const v4si gtvalue_v = _mm_set1_epi32(gtvalue);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4si src_tmp = _mm_load_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_load_si128((__m128i *) (src + i + SSE_LEN_FLOAT));
            v4si lt_mask = _mm_cmplt_epi32(src_tmp, ltlevel_v);
            v4si gt_mask = _mm_cmpgt_epi32(src_tmp, gtlevel_v);
            v4si dst_tmp = _mm_blendv_epi8(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm_blendv_epi8(dst_tmp, gtvalue_v, gt_mask);
            _mm_store_si128((__m128i *) (dst + i), dst_tmp);
            v4si lt_mask2 = _mm_cmplt_epi32(src_tmp2, ltlevel_v);
            v4si gt_mask2 = _mm_cmpgt_epi32(src_tmp2, gtlevel_v);
            v4si dst_tmp2 = _mm_blendv_epi8(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm_blendv_epi8(dst_tmp2, gtvalue_v, gt_mask2);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_FLOAT), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4si src_tmp = _mm_loadu_si128((__m128i *) (src + i));
            v4si src_tmp2 = _mm_loadu_si128((__m128i *) (src + i + SSE_LEN_FLOAT));
            v4si lt_mask = _mm_cmplt_epi32(src_tmp, ltlevel_v);
            v4si gt_mask = _mm_cmpgt_epi32(src_tmp, gtlevel_v);
            v4si dst_tmp = _mm_blendv_epi8(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm_blendv_epi8(dst_tmp, gtvalue_v, gt_mask);
            _mm_storeu_si128((__m128i *) (dst + i), dst_tmp);
            v4si lt_mask2 = _mm_cmplt_epi32(src_tmp2, ltlevel_v);
            v4si gt_mask2 = _mm_cmpgt_epi32(src_tmp2, gtlevel_v);
            v4si dst_tmp2 = _mm_blendv_epi8(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm_blendv_epi8(dst_tmp2, gtvalue_v, gt_mask2);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_FLOAT), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}
