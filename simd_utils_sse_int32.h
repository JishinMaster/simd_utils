/*
 * Project : SIMD_Utils
 * Version : 0.1.4
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_store_si128((__m128i *) dst + i, _mm_add_epi32(_mm_load_si128((__m128i *) (src1 + i)), _mm_load_si128((__m128i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_storeu_si128((__m128i *) dst + i, _mm_add_epi32(_mm_loadu_si128((__m128i *) (src1 + i)), _mm_loadu_si128((__m128i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void mul128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_store_si128((__m128i *) dst + i, _mm_mul_epi32(_mm_load_si128((__m128i *) (src1 + i)), _mm_load_si128((__m128i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_storeu_si128((__m128i *) dst + i, _mm_mul_epi32(_mm_loadu_si128((__m128i *) (src1 + i)), _mm_loadu_si128((__m128i *) (src2 + i))));
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_store_si128((__m128i *) dst + i, _mm_sub_epi32(_mm_load_si128((__m128i *) (src1 + i)), _mm_load_si128((__m128i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_storeu_si128((__m128i *) dst + i, _mm_sub_epi32(_mm_loadu_si128((__m128i *) (src1 + i)), _mm_loadu_si128((__m128i *) (src2 + i))));
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_store_si128((__m128i *) dst + i, _mm_add_epi32(tmp, _mm_load_si128((__m128i *) (src + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_INT32) {
            _mm_storeu_si128((__m128i *) dst + i, _mm_add_epi32(tmp, _mm_loadu_si128((__m128i *) (src + i))));
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
    v4si curVal  = _mm_add_epi32(_mm_set1_epi32(offset), coef);
	v4si curVal2 = _mm_add_epi32(_mm_set1_epi32(offset), coef);
	curVal2		 =  _mm_add_epi32(curVal2, _mm_set1_epi32(4 * slope));
	
    int stop_len = len / (2*SSE_LEN_INT32);
    stop_len *= (2*SSE_LEN_INT32);

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        _mm_store_si128((__m128i *)dst + 0, curVal);
		_mm_store_si128((__m128i *)dst + SSE_LEN_INT32, curVal2);
    } else {
        _mm_storeu_si128((__m128i *)dst + 0, curVal);
		_mm_storeu_si128((__m128i *)dst + SSE_LEN_INT32, curVal2);
    }

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        for (int i = 2*SSE_LEN_FLOAT; i < stop_len; i += 2*SSE_LEN_INT32) {
            curVal = _mm_add_epi32(curVal, slope8_vec);
            _mm_store_si128((__m128i *)dst + i, curVal);
			 curVal2 = _mm_add_epi32(curVal2, slope8_vec);
            _mm_store_si128((__m128i *)dst + i + SSE_LEN_INT32, curVal2);
        }
    } else {
        for (int i = 2*SSE_LEN_FLOAT; i < stop_len; i += 2*SSE_LEN_INT32) {
            curVal = _mm_add_epi32(curVal, slope8_vec);
            _mm_storeu_si128((__m128i *)dst + i, curVal);
			curVal2 = _mm_add_epi32(curVal2, slope8_vec);
            _mm_storeu_si128((__m128i *)dst + i + SSE_LEN_INT32, curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * i;
    }
}

// Experimental

static inline void copy128s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_INT32;
    stop_len *= SSE_LEN_INT32;

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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

#pragma omp parallel for schedule(auto) num_threads(NBTHREADS)
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
