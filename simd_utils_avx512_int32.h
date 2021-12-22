/*
 * Project : SIMD_Utils
 * Version : 0.2.1
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

// Work in progress
#if 0 
static inline void mul512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_store_si512(dst + i, _mm512_mul_epi32(_mm512_load_si512(src1 + i), _mm512_load_si512(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512(dst + i, _mm512_mul_epi32(_mm512_loadu_si512(src1 + i), _mm512_loadu_si512(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}
#endif

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

// to be improved?
static inline __m512i _mm512_absdiff_epi16(__m512i a, __m512i b)
{
    __m512i cmp, difab, difba;
    __m512i zero = _mm512_setzero_epi32();
    __mmask64 cmp_mask = _mm512_cmpgt_epi16_mask(a, b);
    cmp = _mm512_mask_set1_epi16(zero, cmp_mask, 0xFFFF);
    difab = _mm512_sub_epi16(a, b);
    difba = _mm512_sub_epi16(b, a);
    difab = _mm512_and_si512(cmp, difab);
    difba = _mm512_andnot_si512(cmp, difba);
    return _mm512_or_si512(difab, difba);
}

static inline __m512i _mm512_absdiff_epi32(__m512i a, __m512i b)
{
    __m512i cmp, difab, difba;
    __m512i zero = _mm512_setzero_epi32();
    __mmask64 cmp_mask = _mm512_cmpgt_epi32_mask(a, b);
    cmp = _mm512_mask_set1_epi32(zero, cmp_mask, 0xFFFFFFFF);
    difab = _mm512_sub_epi32(a, b);
    difba = _mm512_sub_epi32(b, a);
    difab = _mm512_and_si512(cmp, difab);
    difba = _mm512_andnot_si512(cmp, difba);
    return _mm512_or_si512(difab, difba);
}

static inline __m512i _mm512_absdiff_epi8(__m512i a, __m512i b)
{
    __m512i cmp, difab, difba;
    __m512i zero = _mm512_setzero_epi32();
    __mmask64 cmp_mask = _mm512_cmpgt_epi8_mask(a, b);
    cmp = _mm512_mask_set1_epi8(zero, cmp_mask, 0xFF);
    difab = _mm512_sub_epi8(a, b);
    difba = _mm512_sub_epi8(b, a);
    difab = _mm512_and_si512(cmp, difab);
    difba = _mm512_andnot_si512(cmp, difba);
    return _mm512_or_si512(difab, difba);
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
