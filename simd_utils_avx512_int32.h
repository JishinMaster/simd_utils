/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#include "immintrin.h"

static inline void add512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_store_si512((__m512i *) (dst + i), _mm512_add_epi32(_mm512_load_si512((__m512i *) (src1 + i)),
                                                                       _mm512_load_si512((__m512i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512((__m512i *) (dst + i), _mm512_add_epi32(_mm512_loadu_si512((__m512i *) (src1 + i)),
                                                                        _mm512_loadu_si512((__m512i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

// Works only for Integers stored on 32bits smaller than 16bits
static inline void mul512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src1_tmp = _mm512_load_si512((__m512i *) (src1 + i));
            v16si src2_tmp = _mm512_load_si512((__m512i *) (src2 + i));
            v16si src1_tmp2 = _mm512_load_si512((__m512i *) (src1 + i + AVX512_LEN_INT32));
            v16si src2_tmp2 = _mm512_load_si512((__m512i *) (src2 + i + AVX512_LEN_INT32));
            v16si tmp = _mm512_mullo_epi32(src1_tmp, src2_tmp);
            v16si tmp2 = _mm512_mullo_epi32(src1_tmp2, src2_tmp2);
            _mm512_store_si512((__m512i *) (dst + i), tmp);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src1_tmp = _mm512_loadu_si512((__m512i *) (src1 + i));
            v16si src2_tmp = _mm512_loadu_si512((__m512i *) (src2 + i));
            v16si src1_tmp2 = _mm512_loadu_si512((__m512i *) (src1 + i + AVX512_LEN_INT32));
            v16si src2_tmp2 = _mm512_loadu_si512((__m512i *) (src2 + i + AVX512_LEN_INT32));
            v16si tmp = _mm512_mullo_epi32(src1_tmp, src2_tmp);
            v16si tmp2 = _mm512_mullo_epi32(src1_tmp2, src2_tmp2);
            _mm512_storeu_si512((__m512i *) (dst + i), tmp);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_store_si512((__m512i *) (dst + i), _mm512_sub_epi32(_mm512_load_si512((__m512i *) (src1 + i)),
                                                                       _mm512_load_si512((__m512i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512((__m512i *) (dst + i), _mm512_sub_epi32(_mm512_loadu_si512((__m512i *) (src1 + i)),
                                                                        _mm512_loadu_si512((__m512i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void addc512s(int32_t *src, int32_t value, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    const v16si tmp = _mm512_set1_epi32(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_store_si512((__m512i *) (dst + i), _mm512_add_epi32(tmp, _mm512_load_si512((__m512i *) (src + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512((__m512i *) (dst + i), _mm512_add_epi32(tmp, _mm512_loadu_si512((__m512i *) (src + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void vectorSlope512s(int *dst, int len, int offset, int slope)
{
    v16si coef = _mm512_set_epi32(15 * slope, 14 * slope, 13 * slope, 12 * slope,
                                  11 * slope, 10 * slope, 9 * slope, 8 * slope,
                                  7 * slope, 6 * slope, 5 * slope, 4 * slope,
                                  3 * slope, 2 * slope, slope, 0);
    v16si slope32_vec = _mm512_set1_epi32(32 * slope);
    v16si curVal = _mm512_add_epi32(_mm512_set1_epi32(offset), coef);
    v16si curVal2 = _mm512_add_epi32(_mm512_set1_epi32(offset), coef);
    curVal2 = _mm512_add_epi32(curVal2, _mm512_set1_epi32(16 * slope));

    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
        _mm512_store_si512((__m512i *) (dst + 0), curVal);
        _mm512_store_si512((__m512i *) (dst + AVX512_LEN_INT32), curVal2);
    } else {
        _mm512_storeu_si512((__m512i *) (dst + 0), curVal);
        _mm512_storeu_si512((__m512i *) (dst + AVX512_LEN_INT32), curVal2);
    }

    if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 2 * AVX512_LEN_INT32; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            curVal = _mm512_add_epi32(curVal, slope32_vec);
            _mm512_store_si512((__m512i *) (dst + i), curVal);
            curVal2 = _mm512_add_epi32(curVal2, slope32_vec);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), curVal2);
        }
    } else {
        for (int i = 2 * AVX512_LEN_INT32; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            curVal = _mm512_add_epi32(curVal, slope32_vec);
            _mm512_storeu_si512((__m512i *) (dst + i), curVal);
            curVal2 = _mm512_add_epi32(curVal2, slope32_vec);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * i;
    }
}

// Experimental
static inline void copy512s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_store_si512((__m512i *) (dst + i), _mm512_load_si512((__m512i *) (src + i)));
        }
    } else {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512((__m512i *) (dst + i), _mm512_loadu_si512((__m512i *) (src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void copy512s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);


    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            __m512i tmp1 = _mm512_load_si512((__m512i *) (src + i));
            __m512i tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            _mm512_store_si512((__m512i *) (dst + i), tmp1);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), tmp2);
        }
    } else {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            __m512i tmp1 = _mm512_loadu_si512((__m512i *) (src + i));
            __m512i tmp2 = _mm512_loadu_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            _mm512_storeu_si512((__m512i *) (dst + i), tmp1);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), tmp2);
        }
    }
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif


    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy512s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
        _mm512_stream_si512((__m512i *) (dst + i), _mm512_stream_load_si512((__m512i *) (src + i)));
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}


static inline void fast_copy512s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
        __m512i tmp1 = _mm512_stream_load_si512((__m512i *) (src + i));
        __m512i tmp2 = _mm512_stream_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
        _mm512_stream_si512((__m512i *) (dst + i), tmp1);
        _mm512_stream_si512((__m512i *) (dst + i + AVX512_LEN_INT32), tmp2);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy512s_4(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (4 * AVX512_LEN_INT32);
    stop_len *= (4 * AVX512_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_INT32) {
        __m512i tmp1 = _mm512_stream_load_si512((__m512i *) (src + i));
        __m512i tmp2 = _mm512_stream_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
        __m512i tmp3 = _mm512_stream_load_si512((__m512i *) (src + i + 2 * AVX512_LEN_INT32));
        __m512i tmp4 = _mm512_stream_load_si512((__m512i *) (src + i + 3 * AVX512_LEN_INT32));
        _mm512_stream_si512((__m512i *) (dst + i), tmp1);
        _mm512_stream_si512((__m512i *) (dst + i + AVX512_LEN_INT32), tmp2);
        _mm512_stream_si512((__m512i *) (dst + i + 2 * AVX512_LEN_INT32), tmp3);
        _mm512_stream_si512((__m512i *) (dst + i + 3 * AVX512_LEN_INT32), tmp4);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline __m512i _mm512_absdiff_epi16(__m512i a, __m512i b)
{
    __m512i cmp, difab, difba;
    __mmask32 cmp_mask = _mm512_cmpgt_epi16_mask(a, b);

    difab = _mm512_sub_epi16(a, b);
    difba = _mm512_sub_epi16(b, a);
#if 1  // should be faster
    return _mm512_mask_blend_epi16(cmp_mask, difba, difab);
#else
    __m512i zero = _mm512_setzero_epi32();
    cmp = _mm512_mask_set1_epi16(zero, cmp_mask, 0xFFFF);
    difab = _mm512_and_si512(cmp, difab);
    difba = _mm512_andnot_si512(cmp, difba);
    return _mm512_or_si512(difab, difba);
#endif
}

static inline __m512i _mm512_absdiff_epi32(__m512i a, __m512i b)
{
    __m512i cmp, difab, difba;
    __mmask16 cmp_mask = _mm512_cmpgt_epi32_mask(a, b);

    difab = _mm512_sub_epi32(a, b);
    difba = _mm512_sub_epi32(b, a);
#if 1  // should be faster
    return _mm512_mask_blend_epi32(cmp_mask, difba, difab);
#else
    __m512i zero = _mm512_setzero_epi32();
    cmp = _mm512_mask_set1_epi32(zero, cmp_mask, 0xFFFFFFFF);
    difab = _mm512_and_si512(cmp, difab);
    difba = _mm512_andnot_si512(cmp, difba);
    return _mm512_or_si512(difab, difba);
#endif
}

static inline __m512i _mm512_absdiff_epi8(__m512i a, __m512i b)
{
    __m512i cmp, difab, difba;
    __mmask64 cmp_mask = _mm512_cmpgt_epi8_mask(a, b);

    difab = _mm512_sub_epi8(a, b);
    difba = _mm512_sub_epi8(b, a);
#if 1  // should be faster
    return _mm512_mask_blend_epi32(cmp_mask, difba, difab);
#else
    __m512i zero = _mm512_setzero_epi32();
    cmp = _mm512_mask_set1_epi8(zero, cmp_mask, 0xFF);
    difab = _mm512_and_si512(cmp, difab);
    difba = _mm512_andnot_si512(cmp, difba);
    return _mm512_or_si512(difab, difba);
#endif
}

static inline void absdiff16s_512s(int16_t *src1, int16_t *src2, int16_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT16;
    stop_len *= AVX512_LEN_INT16;


    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT16) {
            __m512i a = _mm512_load_si512((__m512i *) (src1 + i));
            __m512i b = _mm512_load_si512((__m512i *) (src2 + i));
            _mm512_store_si512((__m512i *) (dst + i), _mm512_absdiff_epi16(a, b));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT16) {
            __m512i a = _mm512_loadu_si512((__m512i *) (src1 + i));
            __m512i b = _mm512_loadu_si512((__m512i *) (src2 + i));
            _mm512_storeu_si512((__m512i *) (dst + i), _mm512_absdiff_epi16(a, b));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = abs(src1[i] - src2[i]);
    }
}

static inline void powerspect16s_512s_interleaved(complex16s_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            __m512i reim = _mm512_load_si512((__m512i *) ((const int16_t *) src + j));
            _mm512_store_si512((__m512i *) (dst + i), _mm512_madd_epi16(reim, reim));
            j += AVX512_LEN_INT16;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            __m512i reim = _mm512_loadu_si512((__m512i *) ((const int16_t *) src + j));
            _mm512_storeu_si512((__m512i *) (dst + i), _mm512_madd_epi16(reim, reim));
            j += AVX512_LEN_INT16;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (int32_t) src[i].re * (int32_t) src[i].re + (int32_t) src[i].im * (int32_t) src[i].im;
    }
}

static inline void sum16s32s512(int16_t *src, int len, int32_t *dst, int scale_factor)
{
    int stop_len = len / (4 * AVX512_LEN_INT16);
    stop_len *= (4 * AVX512_LEN_INT16);

    __attribute__((aligned(AVX512_LEN_BYTES))) int32_t accumulate[AVX512_LEN_INT32] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };
    int32_t tmp_acc = 0;
    int16_t scale = 1 << scale_factor;
    v16si one = _mm512_set1_epi16(1);
    v16si vec_acc1 = _mm512_setzero_si512();  // initialize the vector accumulator
    v16si vec_acc2 = _mm512_setzero_si512();  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_INT16) {
            v16si vec_src_tmp = _mm512_load_si512((__m512i *) ((const int16_t *) src + i));
            v16si vec_src_tmp2 = _mm512_load_si512((__m512i *) ((const int16_t *) src + i + AVX512_LEN_INT16));
            v16si vec_src_tmp3 = _mm512_load_si512((__m512i *) ((const int16_t *) src + i + 2 * AVX512_LEN_INT16));
            v16si vec_src_tmp4 = _mm512_load_si512((__m512i *) ((const int16_t *) src + i + 3 * AVX512_LEN_INT16));
            vec_src_tmp = _mm512_madd_epi16(vec_src_tmp, one);
            vec_src_tmp2 = _mm512_madd_epi16(vec_src_tmp2, one);
            vec_src_tmp3 = _mm512_madd_epi16(vec_src_tmp3, one);
            vec_src_tmp4 = _mm512_madd_epi16(vec_src_tmp4, one);
            vec_src_tmp = _mm512_add_epi32(vec_src_tmp, vec_src_tmp2);
            vec_src_tmp3 = _mm512_add_epi32(vec_src_tmp3, vec_src_tmp4);
            vec_acc1 = _mm512_add_epi32(vec_src_tmp, vec_acc1);
            vec_acc2 = _mm512_add_epi32(vec_src_tmp3, vec_acc2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_INT16) {
            v16si vec_src_tmp = _mm512_loadu_si512((__m512i *) ((const int16_t *) src + i));
            v16si vec_src_tmp2 = _mm512_loadu_si512((__m512i *) ((const int16_t *) src + i + AVX512_LEN_INT16));
            v16si vec_src_tmp3 = _mm512_loadu_si512((__m512i *) ((const int16_t *) src + i + 2 * AVX512_LEN_INT16));
            v16si vec_src_tmp4 = _mm512_loadu_si512((__m512i *) ((const int16_t *) src + i + 3 * AVX512_LEN_INT16));
            vec_src_tmp = _mm512_madd_epi16(vec_src_tmp, one);
            vec_src_tmp2 = _mm512_madd_epi16(vec_src_tmp2, one);
            vec_src_tmp3 = _mm512_madd_epi16(vec_src_tmp3, one);
            vec_src_tmp4 = _mm512_madd_epi16(vec_src_tmp4, one);
            vec_src_tmp = _mm512_add_epi32(vec_src_tmp, vec_src_tmp2);
            vec_src_tmp3 = _mm512_add_epi32(vec_src_tmp3, vec_src_tmp4);
            vec_acc1 = _mm512_add_epi32(vec_src_tmp, vec_acc1);
            vec_acc2 = _mm512_add_epi32(vec_src_tmp3, vec_acc2);
        }
    }

    vec_acc1 = _mm512_add_epi32(vec_acc1, vec_acc2);
    _mm512_store_si512((v16si *) accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += (int32_t) src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] +
              accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7] +
              accumulate[8] + accumulate[9] + accumulate[10] + accumulate[11] +
              accumulate[12] + accumulate[13] + accumulate[14] + accumulate[15];
    tmp_acc /= scale;
    *dst = tmp_acc;
}

static inline void flip512s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);
    v16si flip_idx = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    int mini = ((len - 1) < (2 * AVX512_LEN_INT32)) ? (len - 1) : (2 * AVX512_LEN_INT32);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst + len - AVX512_LEN_INT32), AVX512_LEN_BYTES)) {
        for (int i = 2 * AVX512_LEN_INT32; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_load_si512((__m512i *) (src + i));  // load a,b,c,d,e,f,g,h
            v16si src_tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si src_tmp_flip = _mm512_permutex2var_epi32(src_tmp, flip_idx, src_tmp);
            v16si src_tmp_flip2 = _mm512_permutex2var_epi32(src_tmp2, flip_idx, src_tmp2);
            _mm512_store_si512((__m512i *) (dst + len - i - AVX512_LEN_INT32), src_tmp_flip);
            _mm512_store_si512((__m512i *) (dst + len - i - 2 * AVX512_LEN_INT32), src_tmp_flip2);
        }
    } else {
        for (int i = 2 * AVX512_LEN_INT32; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_loadu_si512((__m512i *) (src + i));  // load a,b,c,d,e,f,g,h
            v16si src_tmp2 = _mm512_loadu_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si src_tmp_flip = _mm512_permutex2var_epi32(src_tmp, flip_idx, src_tmp);
            v16si src_tmp_flip2 = _mm512_permutex2var_epi32(src_tmp2, flip_idx, src_tmp2);
            _mm512_storeu_si512((__m512i *) (dst + len - i - AVX512_LEN_INT32), src_tmp_flip);
            _mm512_storeu_si512((__m512i *) (dst + len - i - 2 * AVX512_LEN_INT32), src_tmp_flip2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void maxevery512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src1_tmp = _mm512_load_si512((__m512i *) ((const int32_t *) src1 + i));
            v16si src2_tmp = _mm512_load_si512((__m512i *) ((const int32_t *) src2 + i));
            v16si src1_tmp2 = _mm512_load_si512((__m512i *) ((const int32_t *) src1 + i + AVX512_LEN_INT32));
            v16si src2_tmp2 = _mm512_load_si512((__m512i *) ((const int32_t *) src2 + i + AVX512_LEN_INT32));
            v16si max1 = _mm512_max_epi32(src1_tmp, src2_tmp);
            v16si max2 = _mm512_max_epi32(src1_tmp2, src2_tmp2);
            _mm512_store_si512((__m512i *) (dst + i), max1);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), max2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src1_tmp = _mm512_loadu_si512((__m512i *) (src1 + i));
            v16si src2_tmp = _mm512_loadu_si512((__m512i *) (src2 + i));
            v16si src1_tmp2 = _mm512_loadu_si512((__m512i *) (src1 + i + AVX512_LEN_INT32));
            v16si src2_tmp2 = _mm512_loadu_si512((__m512i *) (src2 + i + AVX512_LEN_INT32));
            v16si max1 = _mm512_max_epi32(src1_tmp, src2_tmp);
            v16si max2 = _mm512_max_epi32(src1_tmp2, src2_tmp2);
            _mm512_storeu_si512((__m512i *) (dst + i), max1);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), max2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src1_tmp = _mm512_load_si512((__m512i *) ((const int32_t *) src1 + i));
            v16si src2_tmp = _mm512_load_si512((__m512i *) ((const int32_t *) src2 + i));
            v16si src1_tmp2 = _mm512_load_si512((__m512i *) ((const int32_t *) src1 + i + AVX512_LEN_INT32));
            v16si src2_tmp2 = _mm512_load_si512((__m512i *) ((const int32_t *) src2 + i + AVX512_LEN_INT32));
            v16si min1 = _mm512_min_epi32(src1_tmp, src2_tmp);
            v16si min2 = _mm512_min_epi32(src1_tmp2, src2_tmp2);
            _mm512_store_si512((__m512i *) (dst + i), min1);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), min2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src1_tmp = _mm512_loadu_si512((__m512i *) (src1 + i));
            v16si src2_tmp = _mm512_loadu_si512((__m512i *) (src2 + i));
            v16si src1_tmp2 = _mm512_loadu_si512((__m512i *) (src1 + i + AVX512_LEN_INT32));
            v16si src2_tmp2 = _mm512_loadu_si512((__m512i *) (src2 + i + AVX512_LEN_INT32));
            v16si min1 = _mm512_min_epi32(src1_tmp, src2_tmp);
            v16si min2 = _mm512_min_epi32(src1_tmp2, src2_tmp2);
            _mm512_storeu_si512((__m512i *) (dst + i), min1);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), min2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void minmax512s(int32_t *src, int len, int32_t *min_value, int32_t *max_value)
{
    int stop_len = (len - AVX512_LEN_INT32) / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    v16si max_v, min_v, max_v2, min_v2;
    v16si src_tmp, src_tmp2;

    int32_t min_tmp = src[0];
    int32_t max_tmp = src[0];

    __attribute__((aligned(SSE_LEN_BYTES))) int32_t max_f[SSE_LEN_INT32];
    __attribute__((aligned(SSE_LEN_BYTES))) int32_t min_f[SSE_LEN_INT32];

    if (len >= AVX512_LEN_INT32) {
        if (isAligned((uintptr_t) (src), AVX512_LEN_BYTES)) {
            src_tmp = _mm512_load_si512((__m512i *) (src + 0));
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = AVX512_LEN_INT32; i < stop_len; i += 2 * AVX512_LEN_INT32) {
                src_tmp = _mm512_load_si512((__m512i *) (src + i));
                src_tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
                max_v = _mm512_max_epi32(max_v, src_tmp);
                min_v = _mm512_min_epi32(min_v, src_tmp);
                max_v2 = _mm512_max_epi32(max_v2, src_tmp2);
                min_v2 = _mm512_min_epi32(min_v2, src_tmp2);
            }
        } else {
            src_tmp = _mm512_loadu_si512((__m512i *) (src + 0));
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = AVX512_LEN_INT32; i < stop_len; i += 2 * AVX512_LEN_INT32) {
                src_tmp = _mm512_loadu_si512((__m512i *) (src + i));
                src_tmp2 = _mm512_loadu_si512((__m512i *) (src + i + AVX512_LEN_INT32));
                max_v = _mm512_max_epi32(max_v, src_tmp);
                min_v = _mm512_min_epi32(min_v, src_tmp);
                max_v2 = _mm512_max_epi32(max_v2, src_tmp2);
                min_v2 = _mm512_min_epi32(min_v2, src_tmp2);
            }
        }

        max_v = _mm512_max_epi32(max_v, max_v2);
        min_v = _mm512_min_epi32(min_v, min_v2);

        v8si max1 = _mm512_castsi512_si256(max_v);
        v8si min1 = _mm512_castsi512_si256(min_v);
        v8si max2 = _mm512_extracti32x8_epi32(max_v, 1);
        v8si min2 = _mm512_extracti32x8_epi32(min_v, 1);
        max2 = _mm256_max_epi32(max1, max2);
        min2 = _mm256_min_epi32(min1, min2);
        v4si max3 = _mm256_castsi256_si128(max2);
        v4si min3 = _mm256_castsi256_si128(min2);
        v4si max4 = _mm256_extracti32x4_epi32(max2, 1);
        v4si min4 = _mm256_extracti32x4_epi32(min2, 1);
        max4 = _mm_max_epi32(max3, max4);
        min4 = _mm_min_epi32(min3, min4);

        _mm_store_si128((__m128i *) (max_f), max4);
        _mm_store_si128((__m128i *) (min_f), min4);

        max_tmp = max_f[0];
        max_tmp = max_tmp > max_f[1] ? max_tmp : max_f[1];
        max_tmp = max_tmp > max_f[2] ? max_tmp : max_f[2];
        max_tmp = max_tmp > max_f[3] ? max_tmp : max_f[3];

        min_tmp = min_f[0];
        min_tmp = min_tmp < min_f[1] ? min_tmp : min_f[1];
        min_tmp = min_tmp < min_f[2] ? min_tmp : min_f[2];
        min_tmp = min_tmp < min_f[3] ? min_tmp : min_f[3];
    }

    for (int i = stop_len; i < len; i++) {
        max_tmp = max_tmp > src[i] ? max_tmp : src[i];
        min_tmp = min_tmp < src[i] ? min_tmp : src[i];
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

static inline void threshold512_gt_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v16si tmp = _mm512_set1_epi32(value);

    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_load_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si dst_tmp = _mm512_min_epi32(src_tmp, tmp);
            v16si dst_tmp2 = _mm512_min_epi32(src_tmp2, tmp);
            _mm512_store_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_loadu_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_loadu_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si dst_tmp = _mm512_min_epi32(src_tmp, tmp);
            v16si dst_tmp2 = _mm512_min_epi32(src_tmp2, tmp);
            _mm512_storeu_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}


static inline void threshold512_gtabs_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v16si pval = _mm512_set1_epi32(value);
    const v16si mval = _mm512_set1_epi32(-value);

    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_load_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si src_abs = _mm512_abs_epi32(src_tmp);
            v16si src_abs2 = _mm512_abs_epi32(src_tmp2);
            __mmask16 eqmask = _mm512_cmp_epi32_mask(src_abs, src_tmp, _MM_CMPINT_EQ);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_epi32_mask(src_abs2, src_tmp2, _MM_CMPINT_EQ);
            v16si max = _mm512_min_epi32(src_tmp, pval);
            v16si max2 = _mm512_min_epi32(src_tmp2, pval);
            v16si min = _mm512_max_epi32(src_tmp, mval);
            v16si min2 = _mm512_max_epi32(src_tmp2, mval);
            v16si dst_tmp = _mm512_mask_blend_epi32(eqmask, min, max);
            v16si dst_tmp2 = _mm512_mask_blend_epi32(eqmask2, min2, max2);
            _mm512_store_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_loadu_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_loadu_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si src_abs = _mm512_abs_epi32(src_tmp);
            v16si src_abs2 = _mm512_abs_epi32(src_tmp2);
            __mmask16 eqmask = _mm512_cmp_epi32_mask(src_abs, src_tmp, _MM_CMPINT_EQ);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_epi32_mask(src_abs2, src_tmp2, _MM_CMPINT_EQ);
            v16si max = _mm512_min_epi32(src_tmp, pval);
            v16si max2 = _mm512_min_epi32(src_tmp2, pval);
            v16si min = _mm512_max_epi32(src_tmp, mval);
            v16si min2 = _mm512_max_epi32(src_tmp2, mval);
            v16si dst_tmp = _mm512_mask_blend_epi32(eqmask, min, max);
            v16si dst_tmp2 = _mm512_mask_blend_epi32(eqmask2, min2, max2);
            _mm512_storeu_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
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

static inline void threshold512_lt_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v16si tmp = _mm512_set1_epi32(value);

    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_load_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si dst_tmp = _mm512_max_epi32(src_tmp, tmp);
            v16si dst_tmp2 = _mm512_max_epi32(src_tmp2, tmp);
            _mm512_store_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_loadu_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_loadu_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si dst_tmp = _mm512_max_epi32(src_tmp, tmp);
            v16si dst_tmp2 = _mm512_max_epi32(src_tmp2, tmp);
            _mm512_storeu_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

static inline void threshold512_ltabs_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v16si pval = _mm512_set1_epi32(value);
    const v16si mval = _mm512_set1_epi32(-value);

    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_load_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si src_abs = _mm512_abs_epi32(src_tmp);
            v16si src_abs2 = _mm512_abs_epi32(src_tmp2);
            __mmask16 eqmask = _mm512_cmp_epi32_mask(src_abs, src_tmp, _MM_CMPINT_EQ);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_epi32_mask(src_abs2, src_tmp2, _MM_CMPINT_EQ);
            v16si max = _mm512_max_epi32(src_tmp, pval);
            v16si max2 = _mm512_max_epi32(src_tmp2, pval);
            v16si min = _mm512_min_epi32(src_tmp, mval);
            v16si min2 = _mm512_min_epi32(src_tmp2, mval);
            v16si dst_tmp = _mm512_mask_blend_epi32(eqmask, min, max);
            v16si dst_tmp2 = _mm512_mask_blend_epi32(eqmask2, min2, max2);
            _mm512_store_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_loadu_si512((__m512i *) (src + i));
            v16si src_tmp2 = _mm512_loadu_si512((__m512i *) (src + i + AVX512_LEN_INT32));
            v16si src_abs = _mm512_abs_epi32(src_tmp);
            v16si src_abs2 = _mm512_abs_epi32(src_tmp2);
            __mmask16 eqmask = _mm512_cmp_epi32_mask(src_abs, src_tmp, _MM_CMPINT_EQ);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_epi32_mask(src_abs2, src_tmp2, _MM_CMPINT_EQ);
            v16si max = _mm512_max_epi32(src_tmp, pval);
            v16si max2 = _mm512_max_epi32(src_tmp2, pval);
            v16si min = _mm512_min_epi32(src_tmp, mval);
            v16si min2 = _mm512_min_epi32(src_tmp2, mval);
            v16si dst_tmp = _mm512_mask_blend_epi32(eqmask, min, max);
            v16si dst_tmp2 = _mm512_mask_blend_epi32(eqmask2, min2, max2);
            _mm512_storeu_si512((__m512i *) (dst + i), dst_tmp);
            _mm512_storeu_si512((__m512i *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
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

static inline void threshold512_ltval_gtval_s(int32_t *src, int32_t *dst, int len, int32_t ltlevel, int32_t ltvalue, int32_t gtlevel, int32_t gtvalue)
{
    const v16si ltlevel_v = _mm512_set1_epi32(ltlevel);
    const v16si ltvalue_v = _mm512_set1_epi32(ltvalue);
    const v16si gtlevel_v = _mm512_set1_epi32(gtlevel);
    const v16si gtvalue_v = _mm512_set1_epi32(gtvalue);

    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_load_si512((v16si *) (src + i));
            v16si src_tmp2 = _mm512_load_si512((v16si *) (src + i + AVX512_LEN_INT32));
            __mmask16 lt_mask = _mm512_cmp_epi32_mask(src_tmp, ltlevel_v, _MM_CMPINT_LT);
            __mmask16 gt_mask = _mm512_cmp_epi32_mask(src_tmp, gtlevel_v, _MM_CMPINT_NLE);
            v16si dst_tmp = _mm512_mask_blend_epi32(lt_mask, src_tmp, ltvalue_v);
            dst_tmp = _mm512_mask_blend_epi32(gt_mask, dst_tmp, gtvalue_v);
            _mm512_store_si512((v16si *) (dst + i), dst_tmp);
            __mmask16 lt_mask2 = _mm512_cmp_epi32_mask(src_tmp2, ltlevel_v, _MM_CMPINT_LT);
            __mmask16 gt_mask2 = _mm512_cmp_epi32_mask(src_tmp2, gtlevel_v, _MM_CMPINT_NLE);
            v16si dst_tmp2 = _mm512_mask_blend_epi32(lt_mask2, src_tmp2, ltvalue_v);
            dst_tmp2 = _mm512_mask_blend_epi32(gt_mask2, dst_tmp2, gtvalue_v);
            _mm512_store_si512((v16si *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
            v16si src_tmp = _mm512_loadu_si512((v16si *) (src + i));
            v16si src_tmp2 = _mm512_loadu_si512((v16si *) (src + i + AVX512_LEN_INT32));
            __mmask16 lt_mask = _mm512_cmp_epi32_mask(src_tmp, ltlevel_v, _MM_CMPINT_LT);
            __mmask16 gt_mask = _mm512_cmp_epi32_mask(src_tmp, gtlevel_v, _MM_CMPINT_NLE);
            v16si dst_tmp = _mm512_mask_blend_epi32(lt_mask, src_tmp, ltvalue_v);
            dst_tmp = _mm512_mask_blend_epi32(gt_mask, dst_tmp, gtvalue_v);
            _mm512_storeu_si512((v16si *) (dst + i), dst_tmp);
            __mmask16 lt_mask2 = _mm512_cmp_epi32_mask(src_tmp2, ltlevel_v, _MM_CMPINT_LT);
            __mmask16 gt_mask2 = _mm512_cmp_epi32_mask(src_tmp2, gtlevel_v, _MM_CMPINT_NLE);
            v16si dst_tmp2 = _mm512_mask_blend_epi32(lt_mask2, src_tmp2, ltvalue_v);
            dst_tmp2 = _mm512_mask_blend_epi32(gt_mask2, dst_tmp2, gtvalue_v);
            _mm512_storeu_si512((v16si *) (dst + i + AVX512_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}

// is it useful to unroll?
static inline void gatheri_512s(int32_t *src, int32_t *dst, int stride, int offset, int len)
{
    int stop_len = len / (AVX512_LEN_INT32);
    stop_len *= (AVX512_LEN_INT32);

    v16si vindex = _mm512_setr_epi32(offset, stride + offset, 2 * stride + offset, 3 * stride + offset,
                                     4 * stride + offset, 5 * stride + offset, 6 * stride + offset, 7 * stride + offset,
                                     8 * stride + offset, 9 * stride + offset, 10 * stride + offset, 11 * stride + offset,
                                     12 * stride + offset, 13 * stride + offset, 14 * stride + offset, 15 * stride + offset);

    if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            v16si tmp = _mm512_i32gather_epi32(vindex, (const int *) (src + i * AVX512_LEN_INT32), 1);
            _mm512_store_si512((v16si *) (dst + i * AVX512_LEN_INT32), tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            v16si tmp = _mm512_i32gather_epi32(vindex, (const int *) (src + i * AVX512_LEN_INT32), 1);
            _mm512_storeu_si512((v16si *) (dst + i * AVX512_LEN_INT32), tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i * stride + offset];
    }
}

// is it useful to unroll?
static inline void scatteri_512s(int32_t *src, int32_t *dst, int stride, int offset, int len)
{
    int stop_len = len / (AVX512_LEN_INT32);
    stop_len *= (AVX512_LEN_INT32);

    v16si vindex = _mm512_setr_epi32(offset, stride + offset, 2 * stride + offset, 3 * stride + offset,
                                     4 * stride + offset, 5 * stride + offset, 6 * stride + offset, 7 * stride + offset,
                                     8 * stride + offset, 9 * stride + offset, 10 * stride + offset, 11 * stride + offset,
                                     12 * stride + offset, 13 * stride + offset, 14 * stride + offset, 15 * stride + offset);

    if (isAligned((uintptr_t) (src), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            v16si tmp = _mm512_load_si512((const int *) (src + i * AVX512_LEN_INT32));
            _mm512_i32scatter_epi32((int *) (dst + i * AVX512_LEN_INT32), vindex, tmp, 1);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            v16si tmp = _mm512_loadu_si512((const int *) (src + i * AVX512_LEN_INT32));
            _mm512_i32scatter_epi32((int *) (dst + i * AVX512_LEN_INT32), vindex, tmp, 1);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i * stride + offset];
    }
}
