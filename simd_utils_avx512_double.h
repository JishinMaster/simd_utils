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


static inline v8sd asin512_pd(v8sd x)
{
    v8sd a, z, z_tmp;
    __mmask8 sign;
    __mmask8 ainfem8, asup0p625;

    // first branch, a > 0.625
    v8sd zz_first_branch;
    v8sd p;
    v8sd z_first_branch;
    v8sd tmp_first_branch;

    // second branch a <= 0.625
    v8sd zz_second_branch;
    v8sd z_second_branch;
    v8sd tmp_second_branch;

    a = _mm512_and_pd(*(v8sd *) _pd512_positive_mask, x);           //fabs(x)
    sign = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_LT_OS);  //0xFFFFFFFF if x < 0.0

    ainfem8 = _mm512_cmp_pd_mask(a, _mm512_set1_pd(1.0e-8), _CMP_LT_OS);  //if( a < 1.0e-8)
    asup0p625 = _mm512_cmp_pd_mask(a, _mm512_set1_pd(0.625), _CMP_GT_OS);

    // fist branch
    zz_first_branch = _mm512_sub_pd(_mm512_set1_pd(1.0), a);
    p = _mm512_fmadd_pd_custom(*(v8sd *) _pd512_ASIN_R0, zz_first_branch, *(v8sd *) _pd512_ASIN_R1);
    p = _mm512_fmadd_pd_custom(p, zz_first_branch, *(v8sd *) _pd512_ASIN_R2);
    p = _mm512_fmadd_pd_custom(p, zz_first_branch, *(v8sd *) _pd512_ASIN_R3);
    p = _mm512_fmadd_pd_custom(p, zz_first_branch, *(v8sd *) _pd512_ASIN_R4);
    p = _mm512_mul_pd(p, zz_first_branch);

    tmp_first_branch = _mm512_add_pd(zz_first_branch, *(v8sd *) _pd512_ASIN_S0);
    tmp_first_branch = _mm512_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v8sd *) _pd512_ASIN_S1);
    tmp_first_branch = _mm512_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v8sd *) _pd512_ASIN_S2);
    tmp_first_branch = _mm512_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v8sd *) _pd512_ASIN_S3);
    p = _mm512_div_pd(p, tmp_first_branch);

    zz_first_branch = _mm512_sqrt_pd(_mm512_add_pd(zz_first_branch, zz_first_branch));
    z_first_branch = _mm512_sub_pd(*(v8sd *) _pd512_PIO4, zz_first_branch);
    zz_first_branch = _mm512_fmadd_pd_custom(zz_first_branch, p, *(v8sd *) _pd512_minMOREBITS);
    z_first_branch = _mm512_sub_pd(z_first_branch, zz_first_branch);
    z_first_branch = _mm512_add_pd(z_first_branch, *(v8sd *) _pd512_PIO4);

    //second branch
    zz_second_branch = _mm512_mul_pd(a, a);
    z_second_branch = _mm512_fmadd_pd_custom(*(v8sd *) _pd512_ASIN_P0, zz_second_branch, *(v8sd *) _pd512_ASIN_P1);
    z_second_branch = _mm512_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_P2);
    z_second_branch = _mm512_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_P3);
    z_second_branch = _mm512_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_P4);
    z_second_branch = _mm512_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_P5);
    z_second_branch = _mm512_mul_pd(z_second_branch, zz_second_branch);

    tmp_second_branch = _mm512_add_pd(zz_second_branch, *(v8sd *) _pd512_ASIN_Q0);
    tmp_second_branch = _mm512_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_Q1);
    tmp_second_branch = _mm512_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_Q2);
    tmp_second_branch = _mm512_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_Q3);
    tmp_second_branch = _mm512_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v8sd *) _pd512_ASIN_Q4);

    z_second_branch = _mm512_div_pd(z_second_branch, tmp_second_branch);
    z_second_branch = _mm512_fmadd_pd_custom(a, z_second_branch, a);



    z = _mm512_mask_blend_pd(asup0p625, z_second_branch, z_first_branch);
    z = _mm512_mask_blend_pd(sign, z, _mm512_xor_pd(*(v8sd *) _pd512_negative_mask, z));
    z = _mm512_mask_blend_pd(ainfem8, z, x);

    // if (x > 1.0) then return 0.0
    z = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, *(v8sd *) _pd512_1, _CMP_GT_OS), z, _mm512_setzero_pd());

    return (z);
}

static inline void asin512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, asin512_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, asin512_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}


static inline v8sd atan512_pd(v8sd xx)
{
    v8sd x, y, z;
    __mmask8 sign;
    __mmask8 suptan3pi8, inftan3pi8inf0p66;  // > T3PI8 or (< T3PI8 and > 0.66)
    v8sd tmp, tmp2;
    v8sd zerop66 = _mm512_set1_pd(0.66);
    v8sd flag = _mm512_setzero_pd();  // flag = 0

    x = _mm512_and_pd(*(v8sd *) _pd512_positive_mask, xx);           // x = fabs(xx)
    sign = _mm512_cmp_pd_mask(xx, _mm512_setzero_pd(), _CMP_LT_OS);  //0xFFFFFFFFFFFFFFFF if x < 0.0, sign = -1

    /* range reduction */

    y = _mm512_setzero_pd();
    suptan3pi8 = _mm512_cmp_pd_mask(x, *(v8sd *) _pd512_TAN3PI8, _CMP_GT_OS);          // if( x > tan 3pi/8 )
    x = _mm512_mask_blend_pd(suptan3pi8, x, _mm512_div_pd(*(v8sd *) _pd512_min1, x));  // if( x > tan 3pi/8 ) then x = -1.0/x
    y = _mm512_mask_blend_pd(suptan3pi8, y, *(v8sd *) _pd512_PIO2);                    // if( x > tan 3pi/8 ) then y = PI/2
    flag = _mm512_mask_blend_pd(suptan3pi8, flag, *(v8sd *) _pd512_1);                 // if( x > tan 3pi/8 ) then flag = 1

    inftan3pi8inf0p66 = _kand_mask8(_mm512_cmp_pd_mask(x, *(v8sd *) _pd512_TAN3PI8, _CMP_LE_OS), _mm512_cmp_pd_mask(x, zerop66, _CMP_LE_OS));  // if( x <= tan 3pi/8 ) && (x <= 0.66)
    y = _mm512_mask_blend_pd(inftan3pi8inf0p66, *(v8sd *) _pd512_PIO4, y);                                                                     // y = 0 or PIO4
    x = _mm512_mask_blend_pd(inftan3pi8inf0p66, _mm512_div_pd(_mm512_sub_pd(x, *(v8sd *) _pd512_1), _mm512_add_pd(x, *(v8sd *) _pd512_1)), x);
    flag = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(*(v8sd *) _pd512_PIO4, y, _CMP_EQ_OS), flag, *(v8sd *) _pd512_2);  // if y = PIO4 then flag = 2

    z = _mm512_mul_pd(x, x);  // z = x*x

    //z = z * polevl(z, P_, 4)
    tmp = _mm512_fmadd_pd_custom(*(v8sd *) _pd512_ATAN_P0, z, *(v8sd *) _pd512_ATAN_P1);
    tmp = _mm512_fmadd_pd_custom(tmp, z, *(v8sd *) _pd512_ATAN_P2);
    tmp = _mm512_fmadd_pd_custom(tmp, z, *(v8sd *) _pd512_ATAN_P3);
    tmp = _mm512_fmadd_pd_custom(tmp, z, *(v8sd *) _pd512_ATAN_P4);
    tmp = _mm512_mul_pd(z, tmp);

    // z = z / p1evl(z, Q_, 5);
    tmp2 = _mm512_add_pd(z, *(v8sd *) _pd512_ATAN_Q0);
    tmp2 = _mm512_fmadd_pd_custom(tmp2, z, *(v8sd *) _pd512_ATAN_Q1);
    tmp2 = _mm512_fmadd_pd_custom(tmp2, z, *(v8sd *) _pd512_ATAN_Q2);
    tmp2 = _mm512_fmadd_pd_custom(tmp2, z, *(v8sd *) _pd512_ATAN_Q3);
    tmp2 = _mm512_fmadd_pd_custom(tmp2, z, *(v8sd *) _pd512_ATAN_Q4);
    z = _mm512_div_pd(tmp, tmp2);

    // z = x * z + x
    z = _mm512_fmadd_pd_custom(x, z, x);

    z = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(flag, *(v8sd *) _pd512_2, _CMP_EQ_OS), z, _mm512_fmadd_pd_custom(*(v8sd *) _pd512_0p5, *(v8sd *) _pd512_MOREBITS, z));  // if (flag == 2) then z += 0.5 * MOREBITS
    z = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(flag, *(v8sd *) _pd512_1, _CMP_EQ_OS), z, _mm512_add_pd(z, *(v8sd *) _pd512_MOREBITS));                                 // if (flag == 1) then z +=  MOREBITS

    y = _mm512_add_pd(y, z);
    y = _mm512_mask_blend_pd(sign, y, _mm512_xor_pd(*(v8sd *) _pd512_negative_mask, y));

    y = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OS), y, xx);  // if (xx == 0) then return xx (x is fabs(xx))
    return (y);
}

static inline void atan512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, atan512_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, atan512_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan(src[i]);
    }
}
