/*
 * Project : SIMD_Utils
 * Version : 0.2.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#include "immintrin.h"

#ifdef __AVX2__
static inline void add256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_add_epi32(_mm256_load_si256((__m256i *) (src1 + i)),
                                                                       _mm256_load_si256((__m256i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_add_epi32(_mm256_loadu_si256((__m256i *) (src1 + i)),
                                                                        _mm256_loadu_si256((__m256i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

#if 0
//Work in progress
static inline void mul256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_mul_epi32(_mm256_load_si256((__m256i *) (src1 + i)), _mm256_load_si256((__m256i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_mul_epi32(_mm256_loadu_si256((__m256i *) (src1 + i)), _mm256_loadu_si256((__m256i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}
#endif

static inline void sub256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_sub_epi32(_mm256_load_si256((__m256i *) (src1 + i)),
                                                                       _mm256_load_si256((__m256i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_sub_epi32(_mm256_loadu_si256((__m256i *) (src1 + i)),
                                                                        _mm256_loadu_si256((__m256i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void addc256s(int32_t *src, int32_t value, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    const v8si tmp = _mm256_set1_epi32(value);

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_add_epi32(tmp, _mm256_load_si256((__m256i *) (src + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_add_epi32(tmp, _mm256_loadu_si256((__m256i *) (src + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

// Experimental

static inline void copy256s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
        _mm256_store_si256((__m256i *) (dst + i), _mm256_load_si256((__m256i *) (src + i)));
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void copy256s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_INT32);
    stop_len *= (2 * AVX_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 2 * AVX_LEN_INT32) {
        __m256i tmp1 = _mm256_load_si256((__m256i *) (src + i));
        __m256i tmp2 = _mm256_load_si256((__m256i *) (src + i + AVX_LEN_INT32));
        _mm256_store_si256((__m256i *) (dst + i), tmp1);
        _mm256_store_si256((__m256i *) (dst + i + AVX_LEN_INT32), tmp2);
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy256s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
        _mm256_stream_si256((__m256i *) (dst + i), _mm256_stream_load_si256((__m256i *) (src + i)));
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}


static inline void fast_copy256s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_INT32);
    stop_len *= (2 * AVX_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 2 * AVX_LEN_INT32) {
        __m256i tmp1 = _mm256_stream_load_si256((__m256i *) (src + i));
        __m256i tmp2 = _mm256_stream_load_si256((__m256i *) (src + i + AVX_LEN_INT32));
        _mm256_stream_si256((__m256i *) (dst + i), tmp1);
        _mm256_stream_si256((__m256i *) (dst + i + AVX_LEN_INT32), tmp2);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy256s_4(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (4 * AVX_LEN_INT32);
    stop_len *= (4 * AVX_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 4 * AVX_LEN_INT32) {
        __m256i tmp1 = _mm256_stream_load_si256((__m256i *) (src + i));
        __m256i tmp2 = _mm256_stream_load_si256((__m256i *) (src + i + AVX_LEN_INT32));
        __m256i tmp3 = _mm256_stream_load_si256((__m256i *) (src + i + 2 * AVX_LEN_INT32));
        __m256i tmp4 = _mm256_stream_load_si256((__m256i *) (src + i + 3 * AVX_LEN_INT32));
        _mm256_stream_si256((__m256i *) (dst + i), tmp1);
        _mm256_stream_si256((__m256i *) (dst + i + AVX_LEN_INT32), tmp2);
        _mm256_stream_si256((__m256i *) (dst + i + 2 * AVX_LEN_INT32), tmp3);
        _mm256_stream_si256((__m256i *) (dst + i + 3 * AVX_LEN_INT32), tmp4);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}
#endif

static inline __m256i _mm256_absdiff_epi16(__m256i a, __m256i b)
{
    __m256i cmp, difab, difba;
    cmp = _mm256_cmpgt_epi16(a, b);
    difab = _mm256_sub_epi16(a, b);
    difba = _mm256_sub_epi16(b, a);
    difab = _mm256_and_si256(cmp, difab);
    difba = _mm256_andnot_si256(cmp, difba);
    return _mm256_or_si256(difab, difba);
}

static inline __m256i _mm256_absdiff_epi32(__m256i a, __m256i b)
{
    __m256i cmp, difab, difba;
    cmp = _mm256_cmpgt_epi32(a, b);
    difab = _mm256_sub_epi32(a, b);
    difba = _mm256_sub_epi32(b, a);
    difab = _mm256_and_si256(cmp, difab);
    difba = _mm256_andnot_si256(cmp, difba);
    return _mm256_or_si256(difab, difba);
}

static inline __m256i _mm256_absdiff_epi8(__m256i a, __m256i b)
{
    __m256i cmp, difab, difba;
    cmp = _mm256_cmpgt_epi8(a, b);
    difab = _mm256_sub_epi8(a, b);
    difba = _mm256_sub_epi8(b, a);
    difab = _mm256_and_si256(cmp, difab);
    difba = _mm256_andnot_si256(cmp, difba);
    return _mm256_or_si256(difab, difba);
}

static inline void absdiff16s_256s(int16_t *src1, int16_t *src2, int16_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT16;
    stop_len *= AVX_LEN_INT16;


    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT16) {
            __m256i a = _mm256_load_si256((__m256i *) (src1 + i));
            __m256i b = _mm256_load_si256((__m256i *) (src2 + i));
            _mm256_store_si256((__m256i *) (dst + i), _mm256_absdiff_epi16(a, b));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT16) {
            __m256i a = _mm256_loadu_si256((__m256i *) (src1 + i));
            __m256i b = _mm256_loadu_si256((__m256i *) (src2 + i));
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_absdiff_epi16(a, b));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = abs(src1[i] - src2[i]);
    }
}

static inline void powerspect16s_256s_interleaved(complex16s_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    int j = 0;
    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            __m256i reim = _mm256_load_si256((__m256i *) ((const int16_t *) src + j));
            // print8i(reim); printf("\n");
            _mm256_store_si256((__m256i *) (dst + i), _mm256_madd_epi16(reim, reim));
            j += AVX_LEN_INT16;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            __m256i reim = _mm256_loadu_si256((__m256i *) ((const int16_t *) src + j));
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_madd_epi16(reim, reim));
            j += AVX_LEN_INT16;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (int32_t) src[i].re * (int32_t) src[i].re + (int32_t) src[i].im * (int32_t) src[i].im;
    }
}
