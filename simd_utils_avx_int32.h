/*
 * Project : SIMD_Utils
 * Version : 0.1.3
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

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256(dst + i, _mm256_add_epi32(_mm256_load_si256(src1 + i), _mm256_load_si256(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256(dst + i, _mm256_add_epi32(_mm256_loadu_si256(src1 + i), _mm256_loadu_si256(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void mul256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256(dst + i, _mm256_mul_epi32(_mm256_load_si256(src1 + i), _mm256_load_si256(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256(dst + i, _mm256_mul_epi32(_mm256_loadu_si256(src1 + i), _mm256_loadu_si256(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256(dst + i, _mm256_sub_epi32(_mm256_load_si256(src1 + i), _mm256_load_si256(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256(dst + i, _mm256_sub_epi32(_mm256_loadu_si256(src1 + i), _mm256_loadu_si256(src2 + i)));
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

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256(dst + i, _mm256_add_epi32(tmp, _mm256_load_si256(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256(dst + i, _mm256_add_epi32(tmp, _mm256_loadu_si256(src + i)));
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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
