/*
 * Project : SIMD_Utils
 * Version : 0.1.6
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <stdint.h>
#include "immintrin.h"

typedef __m256d v4sd;  // vector of 4 double (avx)

static inline void set256d(double *src, double value, int len)
{
    const v4sd tmp = _mm256_set1_pd(value);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = value;
    }
}

static inline void zero256d(double *src, int len)
{
    const v4sd tmp = _mm256_setzero_pd();

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = 0.0;
    }
}

static inline void copy256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_load_pd(src + i));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void sqrt256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_sqrt_pd(_mm256_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_sqrt_pd(_mm256_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrt(src[i]);
    }
}

static inline void add256d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_add_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_add_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void mul256d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_mul_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub256d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_sub_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_sub_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void div256d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_div_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_div_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

//TODO : "Immediate add/mul?"
static inline void addc256d(double *src, double value, double *dst, int len)
{
    const v4sd tmp = _mm256_set1_pd(value);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_add_pd(tmp, _mm256_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_add_pd(tmp, _mm256_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc256d(double *src, double value, double *dst, int len)
{
    const v4sd tmp = _mm256_set1_pd(value);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, _mm256_mul_pd(tmp, _mm256_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, _mm256_mul_pd(tmp, _mm256_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd256d(double *_a, double *_b, double *_c, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(_b), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t)(_c), (uintptr_t)(dst), AVX_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_load_pd(_a + i);
            v4sd b = _mm256_load_pd(_b + i);
            v4sd c = _mm256_load_pd(_c + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_loadu_pd(_a + i);
            v4sd b = _mm256_loadu_pd(_b + i);
            v4sd c = _mm256_loadu_pd(_c + i);
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcadd256d(double *_a, double _b, double *_c, double *dst, int len)
{
    v4sd b = _mm256_set1_pd(_b);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_c), (uintptr_t)(dst), AVX_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_load_pd(_a + i);
            v4sd c = _mm256_load_pd(_c + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_loadu_pd(_a + i);
            v4sd c = _mm256_loadu_pd(_c + i);
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddc256d(double *_a, double _b, double _c, double *dst, int len)
{
    v4sd b = _mm256_set1_pd(_b);
    v4sd c = _mm256_set1_pd(_c);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(dst), AVX_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_loadu_pd(_a + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_loadu_pd(_a + i);
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
    }
}

static inline void muladdc256d(double *_a, double *_b, double _c, double *dst, int len)
{
    v4sd c = _mm256_set1_pd(_c);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_b), (uintptr_t)(dst), AVX_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_load_pd(_a + i);
            v4sd b = _mm256_load_pd(_b + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_loadu_pd(_a + i);
            v4sd b = _mm256_loadu_pd(_b + i);
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

static inline void round256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = round(src[i]);
    }
}

static inline void ceil256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceil(src[i]);
    }
}

static inline void floor256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floor(src[i]);
    }
}

static inline void vectorSlope256d(double *dst, int len, double offset, double slope)
{
    v4sd coef = _mm256_set_pd(3.0 * slope, 2.0 * slope, slope, 0.0);
    v4sd slope4_vec = _mm256_set1_pd(4.0 * slope);
    v4sd curVal = _mm256_add_pd(_mm256_set1_pd(offset), coef);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (dst) % AVX_LEN_BYTES) == 0) {
        _mm256_store_pd(dst + 0, curVal);
    } else {
        _mm256_storeu_pd(dst + 0, curVal);
    }

    if (((uintptr_t)(const void *) (dst) % AVX_LEN_BYTES) == 0) {
        for (int i = AVX_LEN_DOUBLE; i < stop_len; i += AVX_LEN_DOUBLE) {
            curVal = _mm256_add_pd(curVal, slope4_vec);
            _mm256_store_pd(dst + i, curVal);
        }
    } else {
        for (int i = AVX_LEN_DOUBLE; i < stop_len; i += AVX_LEN_DOUBLE) {
            curVal = _mm256_add_pd(curVal, slope4_vec);
            _mm256_storeu_pd(dst + i, curVal);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (double) i;
    }
}
