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


typedef __m128d v2sd;  // vector of 2 double (sse)
typedef __m128i v2si;  // vector of 2 int 64 (sse)

static inline void set128d(double *src, double value, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % SSE_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = value;
    }
}

static inline void zero128d(double *src, int len)
{
    const v2sd tmp = _mm_setzero_pd();

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % SSE_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = 0.0;
    }
}

static inline void copy128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_load_pd(src + i));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_loadu_pd(src + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void sqrt128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_sqrt_pd(_mm_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_sqrt_pd(_mm_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrt(src[i]);
    }
}

static inline void add128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_add_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_add_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void mul128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_mul_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_mul_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_sub_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_sub_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void div128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_div_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_div_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

//TODO : "Immediate add/mul?"
static inline void addc128d(double *src, double value, double *dst, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_add_pd(tmp, _mm_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_add_pd(tmp, _mm_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc128d(double *src, double value, double *dst, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_mul_pd(tmp, _mm_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_add_pd(tmp, _mm_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void round128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = round(src[i]);
    }
}

static inline void ceil128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceil(src[i]);
    }
}

static inline void floor128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floor(src[i]);
    }
}



static inline void vectorSlope128d(double *dst, int len, double offset, double slope)
{
    v2sd coef = _mm_set_pd(slope, 0.0);
    v2sd slope2_vec = _mm_set1_pd(2.0 * slope);
    v2sd curVal = _mm_add_pd(_mm_set1_pd(offset), coef);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        _mm_store_pd(dst + 0, curVal);
    } else {
        _mm_storeu_pd(dst + 0, curVal);
    }

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        for (int i = SSE_LEN_DOUBLE; i < stop_len; i += SSE_LEN_DOUBLE) {
            curVal = _mm_add_pd(curVal, slope2_vec);
            _mm_store_pd(dst + i, curVal);
        }
    } else {
        for (int i = SSE_LEN_DOUBLE; i < stop_len; i += SSE_LEN_DOUBLE) {
            curVal = _mm_add_pd(curVal, slope2_vec);
            _mm_storeu_pd(dst + i, curVal);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (double) i;
    }
}
