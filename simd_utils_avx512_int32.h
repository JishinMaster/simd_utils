/*
 * Project : SIMD_Utils
 * Version : 0.1.12
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
            _mm512_store_si512(dst + i, _mm512_add_epi32(_mm512_load_si512(src1 + i), _mm512_load_si512(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512(dst + i, _mm512_add_epi32(_mm512_loadu_si512(src1 + i), _mm512_loadu_si512(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void mul512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

static inline void sub512s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_INT32;
    stop_len *= AVX512_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_store_si512(dst + i, _mm512_sub_epi32(_mm512_load_si512(src1 + i), _mm512_load_si512(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512(dst + i, _mm512_sub_epi32(_mm512_loadu_si512(src1 + i), _mm512_loadu_si512(src2 + i)));
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
            _mm512_store_si512(dst + i, _mm512_add_epi32(tmp, _mm512_load_si512(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
            _mm512_storeu_si512(dst + i, _mm512_add_epi32(tmp, _mm512_loadu_si512(src + i)));
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

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += AVX512_LEN_INT32) {
        _mm512_store_si512((__m512i *) (dst + i), _mm512_load_si512((__m512i *) (src + i)));
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void copy512s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_INT32);
    stop_len *= (2 * AVX512_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_INT32) {
        __m512i tmp1 = _mm512_load_si512((__m512i *) (src + i));
        __m512i tmp2 = _mm512_load_si512((__m512i *) (src + i + AVX512_LEN_INT32));
        _mm512_store_si512((__m512i *) (dst + i), tmp1);
        _mm512_store_si512((__m512i *) (dst + i + AVX512_LEN_INT32), tmp2);
    }

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
