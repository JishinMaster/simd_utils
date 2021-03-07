/*
 * Project : SIMD_Utils
 * Version : 0.1.7
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <stdint.h>
#include "immintrin.h"

typedef __m512d v8sd;  // vector of 8 double (avx512)


static inline void set512d(double *src, double value, int len)
{
    const v8sd tmp = _mm512_set1_pd(value);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = value;
    }
}

static inline void zero512d(double *src, int len)
{
    const v8sd tmp = _mm512_setzero_pd();

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = 0.0;
    }
}

static inline void copy512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_load_pd(src + i));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_loadu_pd(src + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void sqrt512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_sqrt_pd(_mm512_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_sqrt_pd(_mm512_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrt(src[i]);
    }
}

static inline void add512d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_add_pd(_mm512_load_pd(src1 + i), _mm512_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_add_pd(_mm512_loadu_pd(src1 + i), _mm512_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void mul512d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_mul_pd(_mm512_load_pd(src1 + i), _mm512_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_mul_pd(_mm512_loadu_pd(src1 + i), _mm512_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub512d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_sub_pd(_mm512_load_pd(src1 + i), _mm512_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_sub_pd(_mm512_loadu_pd(src1 + i), _mm512_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void div512d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_div_pd(_mm512_load_pd(src1 + i), _mm512_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_div_pd(_mm512_loadu_pd(src1 + i), _mm512_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

//TODO : "Immediate add/mul?"
static inline void addc512d(double *src, double value, double *dst, int len)
{
    const v8sd tmp = _mm512_set1_pd(value);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_add_pd(tmp, _mm512_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_add_pd(tmp, _mm512_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc512d(double *src, double value, double *dst, int len)
{
    const v8sd tmp = _mm512_set1_pd(value);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, _mm512_mul_pd(tmp, _mm512_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, _mm512_mul_pd(tmp, _mm512_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd512d(double *_a, double *_b, double *_c, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(_b), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t)(_c), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_load_pd(_a + i);
            v8sd b = _mm512_load_pd(_b + i);
            v8sd c = _mm512_load_pd(_c + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_loadu_pd(_a + i);
            v8sd b = _mm512_loadu_pd(_b + i);
            v8sd c = _mm512_loadu_pd(_c + i);
            _mm512_storeu_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcadd512d(double *_a, double _b, double *_c, double *dst, int len)
{
    v8sd b = _mm512_set1_pd(_b);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_c), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_load_pd(_a + i);
            v8sd c = _mm512_load_pd(_c + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_loadu_pd(_a + i);
            v8sd c = _mm512_loadu_pd(_c + i);
            _mm512_storeu_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddc512d(double *_a, double _b, double _c, double *dst, int len)
{
    v8sd b = _mm512_set1_pd(_b);
    v8sd c = _mm512_set1_pd(_c);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_loadu_pd(_a + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_loadu_pd(_a + i);
            _mm512_storeu_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
    }
}

static inline void muladdc512d(double *_a, double *_b, double _c, double *dst, int len)
{
    v8sd c = _mm512_set1_pd(_c);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_b), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_load_pd(_a + i);
            v8sd b = _mm512_load_pd(_b + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_loadu_pd(_a + i);
            v8sd b = _mm512_loadu_pd(_b + i);
            _mm512_storeu_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

static inline void round512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = round(src[i]);
    }
}

static inline void ceil512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceil(src[i]);
    }
}

static inline void floor512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floor(src[i]);
    }
}

static inline void trunc512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, _mm512_roundscale_pd(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = trunc(src[i]);
    }
}

static inline void vectorSlope512d(double *dst, int len, double offset, double slope)
{
    v8sd coef = _mm512_set_pd(7.0 * slope, 6.0 * slope, 5.0 * slope, 4.0 * slope, 3.0 * slope, 2.0 * slope, slope, 0.0);
    v8sd slope16_vec = _mm512_set1_pd(16.0f * slope);
    v8sd curVal = _mm512_add_pd(_mm512_set1_pd(offset), coef);
    v8sd curVal2 = _mm512_add_pd(_mm512_set1_pd(offset), coef);
    curVal2 = _mm512_add_pd(curVal2, _mm512_set1_pd(8.0f * slope));
    int stop_len = len / (2 * AVX512_LEN_DOUBLE);
    stop_len *= (2 * AVX512_LEN_DOUBLE);

    if (((uintptr_t)(const void *) (dst) % AVX512_LEN_BYTES) == 0) {
        _mm512_store_pd(dst + 0, curVal);
        _mm512_store_pd(dst + AVX512_LEN_DOUBLE, curVal2);
    } else {
        _mm512_storeu_pd(dst + 0, curVal);
        _mm512_storeu_pd(dst + AVX512_LEN_DOUBLE, curVal2);
    }

    if (((uintptr_t)(const void *) (dst) % AVX512_LEN_BYTES) == 0) {
        for (int i = 2 * AVX512_LEN_DOUBLE; i < stop_len; i += 2 * AVX512_LEN_DOUBLE) {
            curVal = _mm512_add_pd(curVal, slope16_vec);
            _mm512_store_pd(dst + i, curVal);
            curVal2 = _mm512_add_pd(curVal2, slope16_vec);
            _mm512_store_pd(dst + i + AVX512_LEN_DOUBLE, curVal2);
        }
    } else {
        for (int i = 2 * AVX512_LEN_DOUBLE; i < stop_len; i += 2 * AVX512_LEN_DOUBLE) {
            curVal = _mm512_add_pd(curVal, slope16_vec);
            _mm512_storeu_pd(dst + i, curVal);
            curVal2 = _mm512_add_pd(curVal2, slope16_vec);
            _mm512_storeu_pd(dst + i + AVX512_LEN_DOUBLE, curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (double) i;
    }
}
