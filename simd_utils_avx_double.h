/*
 * Project : SIMD_Utils
 * Version : 0.2.2
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <stdint.h>
#include "immintrin.h"

static inline void set256d(double *dst, double value, int len)
{
    const v4sd tmp = _mm256_set1_pd(value);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (isAligned((uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zero256d(double *dst, int len)
{
    const v4sd tmp = _mm256_setzero_pd();

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (isAligned((uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_store_pd(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            _mm256_storeu_pd(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 0.0;
    }
}

static inline void copy256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

// TODO : "Immediate add/mul?"
static inline void addc256d(double *src, double value, double *dst, int len)
{
    const v4sd tmp = _mm256_set1_pd(value);

    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_load_pd(_a + i);
            v4sd b = _mm256_load_pd(_b + i);
            v4sd c = _mm256_load_pd(_c + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
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
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_load_pd(_a + i);
            v4sd c = _mm256_load_pd(_c + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
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
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_loadu_pd(_a + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
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
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd a = _mm256_load_pd(_a + i);
            v4sd b = _mm256_load_pd(_b + i);
            _mm256_store_pd(dst + i, _mm256_fmadd_pd_custom(a, b, c));
        }
    } else {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

static inline void trunc256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = trunc(src[i]);
    }
}

static inline void vectorSlope256d(double *dst, int len, double offset, double slope)
{
    int stop_len = len / (2 * AVX_LEN_DOUBLE);
    stop_len *= (2 * AVX_LEN_DOUBLE);

    v4sd coef = _mm256_set_pd(3.0 * slope, 2.0 * slope, slope, 0.0);
    v4sd slope8_vec = _mm256_set1_pd(8.0 * slope);
    v4sd curVal = _mm256_add_pd(_mm256_set1_pd(offset), coef);
    v4sd curVal2 = _mm256_add_pd(_mm256_set1_pd(offset), coef);
    curVal2 = _mm256_add_pd(curVal2, _mm256_set1_pd(4.0 * slope));


    if (isAligned((uintptr_t)(dst), AVX_LEN_BYTES)) {
        _mm256_store_pd(dst + 0, curVal);
        _mm256_store_pd(dst + AVX_LEN_DOUBLE, curVal2);
    } else {
        _mm256_storeu_pd(dst + 0, curVal);
        _mm256_storeu_pd(dst + AVX_LEN_DOUBLE, curVal2);
    }

    if (isAligned((uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 2 * AVX_LEN_DOUBLE; i < stop_len; i += 2 * AVX_LEN_DOUBLE) {
            curVal = _mm256_add_pd(curVal, slope8_vec);
            _mm256_store_pd(dst + i, curVal);
            curVal2 = _mm256_add_pd(curVal2, slope8_vec);
            _mm256_store_pd(dst + i + AVX_LEN_DOUBLE, curVal2);
        }
    } else {
        for (int i = 2 * AVX_LEN_DOUBLE; i < stop_len; i += 2 * AVX_LEN_DOUBLE) {
            curVal = _mm256_add_pd(curVal, slope8_vec);
            _mm256_storeu_pd(dst + i, curVal);
            curVal2 = _mm256_add_pd(curVal2, slope8_vec);
            _mm256_storeu_pd(dst + i + AVX_LEN_DOUBLE, curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (double) i;
    }
}

static inline v4sd asin256_pd(v4sd x)
{
    v4sd a, z;
    v4sd sign;
    v4sd ainfem8, asup0p625;

    // first branch, a > 0.625
    v4sd zz_first_branch;
    v4sd p;
    v4sd z_first_branch;
    v4sd tmp_first_branch;

    // second branch a <= 0.625
    v4sd zz_second_branch;
    v4sd z_second_branch;
    v4sd tmp_second_branch;

    a = _mm256_and_pd(*(v4sd *) _pd256_positive_mask, x);      // fabs(x)
    sign = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OS);  // 0xFFFFFFFF if x < 0.0

    ainfem8 = _mm256_cmp_pd(a, _mm256_set1_pd(1.0e-8), _CMP_LT_OS);  // if( a < 1.0e-8)
    asup0p625 = _mm256_cmp_pd(a, _mm256_set1_pd(0.625), _CMP_GT_OS);

    // fist branch
    zz_first_branch = _mm256_sub_pd(_mm256_set1_pd(1.0), a);
    p = _mm256_fmadd_pd_custom(*(v4sd *) _pd256_ASIN_R0, zz_first_branch, *(v4sd *) _pd256_ASIN_R1);
    p = _mm256_fmadd_pd_custom(p, zz_first_branch, *(v4sd *) _pd256_ASIN_R2);
    p = _mm256_fmadd_pd_custom(p, zz_first_branch, *(v4sd *) _pd256_ASIN_R3);
    p = _mm256_fmadd_pd_custom(p, zz_first_branch, *(v4sd *) _pd256_ASIN_R4);
    p = _mm256_mul_pd(p, zz_first_branch);

    tmp_first_branch = _mm256_add_pd(zz_first_branch, *(v4sd *) _pd256_ASIN_S0);
    tmp_first_branch = _mm256_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v4sd *) _pd256_ASIN_S1);
    tmp_first_branch = _mm256_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v4sd *) _pd256_ASIN_S2);
    tmp_first_branch = _mm256_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v4sd *) _pd256_ASIN_S3);
    p = _mm256_div_pd(p, tmp_first_branch);

    zz_first_branch = _mm256_sqrt_pd(_mm256_add_pd(zz_first_branch, zz_first_branch));
    z_first_branch = _mm256_sub_pd(*(v4sd *) _pd256_PIO4, zz_first_branch);
    zz_first_branch = _mm256_fmadd_pd_custom(zz_first_branch, p, *(v4sd *) _pd256_minMOREBITS);
    z_first_branch = _mm256_sub_pd(z_first_branch, zz_first_branch);
    z_first_branch = _mm256_add_pd(z_first_branch, *(v4sd *) _pd256_PIO4);

    // second branch
    zz_second_branch = _mm256_mul_pd(a, a);
    z_second_branch = _mm256_fmadd_pd_custom(*(v4sd *) _pd256_ASIN_P0, zz_second_branch, *(v4sd *) _pd256_ASIN_P1);
    z_second_branch = _mm256_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_P2);
    z_second_branch = _mm256_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_P3);
    z_second_branch = _mm256_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_P4);
    z_second_branch = _mm256_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_P5);
    z_second_branch = _mm256_mul_pd(z_second_branch, zz_second_branch);

    tmp_second_branch = _mm256_add_pd(zz_second_branch, *(v4sd *) _pd256_ASIN_Q0);
    tmp_second_branch = _mm256_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_Q1);
    tmp_second_branch = _mm256_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_Q2);
    tmp_second_branch = _mm256_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_Q3);
    tmp_second_branch = _mm256_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v4sd *) _pd256_ASIN_Q4);

    z_second_branch = _mm256_div_pd(z_second_branch, tmp_second_branch);
    z_second_branch = _mm256_fmadd_pd_custom(a, z_second_branch, a);



    z = _mm256_blendv_pd(z_second_branch, z_first_branch, asup0p625);
    z = _mm256_blendv_pd(z, _mm256_xor_pd(*(v4sd *) _pd256_negative_mask, z), sign);
    z = _mm256_blendv_pd(z, x, ainfem8);

    // if (x > 1.0) then return 0.0
    z = _mm256_blendv_pd(z, _mm256_setzero_pd(), _mm256_cmp_pd(x, *(v4sd *) _pd256_1, _CMP_GT_OS));

    return (z);
}

static inline void asin256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, asin256_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, asin256_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}


static inline v4sd atan256_pd(v4sd xx)
{
    v4sd x, y, z;
    v4sd sign;
    v4sd suptan3pi8, inftan3pi8inf0p66;  // > T3PI8 or (< T3PI8 and > 0.66)
    v4sd tmp, tmp2;
    v4sd zerop66 = _mm256_set1_pd(0.66);
    v4sd flag = _mm256_setzero_pd();  // flag = 0

    x = _mm256_and_pd(*(v4sd *) _pd256_positive_mask, xx);      // x = fabs(xx)
    sign = _mm256_cmp_pd(xx, _mm256_setzero_pd(), _CMP_LT_OS);  // 0xFFFFFFFFFFFFFFFF if x < 0.0, sign = -1

    /* range reduction */

    y = _mm256_setzero_pd();
    suptan3pi8 = _mm256_cmp_pd(x, *(v4sd *) _pd256_TAN3PI8, _CMP_GT_OS);           // if( x > tan 3pi/8 )
    x = _mm256_blendv_pd(x, _mm256_div_pd(*(v4sd *) _pd256_min1, x), suptan3pi8);  // if( x > tan 3pi/8 ) then x = -1.0/x
    y = _mm256_blendv_pd(y, *(v4sd *) _pd256_PIO2, suptan3pi8);                    // if( x > tan 3pi/8 ) then y = PI/2
    flag = _mm256_blendv_pd(flag, *(v4sd *) _pd256_1, suptan3pi8);                 // if( x > tan 3pi/8 ) then flag = 1

    inftan3pi8inf0p66 = _mm256_and_pd(_mm256_cmp_pd(x, *(v4sd *) _pd256_TAN3PI8, _CMP_LE_OS), _mm256_cmp_pd(x, zerop66, _CMP_LE_OS));  // if( x <= tan 3pi/8 ) && (x <= 0.66)
    y = _mm256_blendv_pd(*(v4sd *) _pd256_PIO4, y, inftan3pi8inf0p66);                                                                 // y = 0 or PIO4
    x = _mm256_blendv_pd(_mm256_div_pd(_mm256_sub_pd(x, *(v4sd *) _pd256_1), _mm256_add_pd(x, *(v4sd *) _pd256_1)), x, inftan3pi8inf0p66);
    flag = _mm256_blendv_pd(flag, *(v4sd *) _pd256_2, _mm256_cmp_pd(*(v4sd *) _pd256_PIO4, y, _CMP_EQ_OS));  // if y = PIO4 then flag = 2

    z = _mm256_mul_pd(x, x);  // z = x*x

    // z = z * polevl(z, P_, 4)
    tmp = _mm256_fmadd_pd_custom(*(v4sd *) _pd256_ATAN_P0, z, *(v4sd *) _pd256_ATAN_P1);
    tmp = _mm256_fmadd_pd_custom(tmp, z, *(v4sd *) _pd256_ATAN_P2);
    tmp = _mm256_fmadd_pd_custom(tmp, z, *(v4sd *) _pd256_ATAN_P3);
    tmp = _mm256_fmadd_pd_custom(tmp, z, *(v4sd *) _pd256_ATAN_P4);
    tmp = _mm256_mul_pd(z, tmp);

    // z = z / p1evl(z, Q_, 5);
    tmp2 = _mm256_add_pd(z, *(v4sd *) _pd256_ATAN_Q0);
    tmp2 = _mm256_fmadd_pd_custom(tmp2, z, *(v4sd *) _pd256_ATAN_Q1);
    tmp2 = _mm256_fmadd_pd_custom(tmp2, z, *(v4sd *) _pd256_ATAN_Q2);
    tmp2 = _mm256_fmadd_pd_custom(tmp2, z, *(v4sd *) _pd256_ATAN_Q3);
    tmp2 = _mm256_fmadd_pd_custom(tmp2, z, *(v4sd *) _pd256_ATAN_Q4);
    z = _mm256_div_pd(tmp, tmp2);

    // z = x * z + x
    z = _mm256_fmadd_pd_custom(x, z, x);

    z = _mm256_blendv_pd(z, _mm256_fmadd_pd_custom(*(v4sd *) _pd256_0p5, *(v4sd *) _pd256_MOREBITS, z),
                         _mm256_cmp_pd(flag, *(v4sd *) _pd256_2, _CMP_EQ_OS));  // if (flag == 2) then z += 0.5 * MOREBITS
    z = _mm256_blendv_pd(z, _mm256_add_pd(z, *(v4sd *) _pd256_MOREBITS),
                         _mm256_cmp_pd(flag, *(v4sd *) _pd256_1, _CMP_EQ_OS));  // if (flag == 1) then z +=  MOREBITS

    y = _mm256_add_pd(y, z);
    y = _mm256_blendv_pd(y, _mm256_xor_pd(*(v4sd *) _pd256_negative_mask, y), sign);

    y = _mm256_blendv_pd(y, xx, _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OS));  // if (xx == 0) then return xx (x is fabs(xx))
    return (y);
}

static inline void atan256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, atan256_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, atan256_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan(src[i]);
    }
}

#ifdef __AVX2__

static inline v4sid _mm256_cvttpd_epi64_custom(v4sd x)
{
    x = _mm256_add_pd(x, _mm256_set1_pd(0x0010000000000000));
    return _mm256_xor_si256(
        _mm256_castpd_si256(x),
        _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
}

static inline v4sd _mm256_cvtepi64_pd_custom(v4sid x)
{
    x = _mm256_or_si256(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0010000000000000));
}

static inline void sincos256_pd(v4sd x, v4sd *s, v4sd *c)
{
    v4sd xmm1, xmm2, xmm3 = _mm256_setzero_pd(), sign_bit_sin, y;

    v4sid emm0, emm2, emm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_pd(x, *(v4sd *) _pd256_inv_sign_mask);

    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_pd(sign_bit_sin, *(v4sd *) _pd256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_pd(x, *(v4sd *) _pd256_cephes_FOPI);

    /* store the integer part of y in emm2 */
    emm2 = _mm256_cvttpd_epi64_custom(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm256_add_epi64(emm2, *(v4sid *) _pi256_64_1);

    emm2 = _mm256_and_si256(emm2, *(v4sid *) _pi256_64_inv1);
    y = _mm256_cvtepi64_pd_custom(emm2);
    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0 = _mm256_and_si256(emm2, *(v4sid *) _pi256_64_4);
    // print2i(emm0);
    emm0 = _mm256_slli_epi64(emm0, 61);
    // print2i(emm0);
    v4sd swap_sign_bit_sin = _mm256_castsi256_pd(emm0);

    /* get the polynom selection mask for the sine*/
    emm2 = _mm256_and_si256(emm2, *(v4sid *) _pi256_64_2);
    emm2 = _mm256_cmpeq_epi64(emm2, _mm256_setzero_si256());
    v4sd poly_mask = _mm256_castsi256_pd(emm2);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm256_fmadd_pd_custom(y, *(v4sd *) _pd256_minus_cephes_DP1, x);
    x = _mm256_fmadd_pd_custom(y, *(v4sd *) _pd256_minus_cephes_DP2, x);
    x = _mm256_fmadd_pd_custom(y, *(v4sd *) _pd256_minus_cephes_DP3, x);

    emm4 = _mm256_sub_epi64(emm4, *(v4sid *) _pi256_64_2);
    emm4 = _mm256_andnot_si256(emm4, *(v4sid *) _pi256_64_4);
    emm4 = _mm256_slli_epi64(emm4, 61);
    v4sd sign_bit_cos = _mm256_castsi256_pd(emm4);

    sign_bit_sin = _mm256_xor_pd(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v4sd z = _mm256_mul_pd(x, x);

    y = _mm256_fmadd_pd_custom(*(v4sd *) _pd256_coscof_p0, z, *(v4sd *) _pd256_coscof_p1);
    y = _mm256_fmadd_pd_custom(y, z, *(v4sd *) _pd256_coscof_p2);
    y = _mm256_fmadd_pd_custom(y, z, *(v4sd *) _pd256_coscof_p3);
    y = _mm256_fmadd_pd_custom(y, z, *(v4sd *) _pd256_coscof_p4);
    y = _mm256_fmadd_pd_custom(y, z, *(v4sd *) _pd256_coscof_p5);
    y = _mm256_mul_pd(y, z);
    y = _mm256_mul_pd(y, z);
    y = _mm256_fnmadd_pd_custom(z, *(v4sd *) _pd256_0p5, y);
    y = _mm256_add_pd(y, *(v4sd *) _pd256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    v4sd y2 = _mm256_fmadd_pd_custom(*(v4sd *) _pd256_sincof_p0, z, *(v4sd *) _pd256_sincof_p1);
    y2 = _mm256_fmadd_pd_custom(y2, z, *(v4sd *) _pd256_sincof_p2);
    y2 = _mm256_fmadd_pd_custom(y2, z, *(v4sd *) _pd256_sincof_p3);
    y2 = _mm256_fmadd_pd_custom(y2, z, *(v4sd *) _pd256_sincof_p4);
    y2 = _mm256_fmadd_pd_custom(y2, z, *(v4sd *) _pd256_sincof_p5);
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_fmadd_pd_custom(y2, x, x);


    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    v4sd ysin2 = _mm256_and_pd(xmm3, y2);
    v4sd ysin1 = _mm256_andnot_pd(xmm3, y);
    y2 = _mm256_sub_pd(y2, ysin2);
    y = _mm256_sub_pd(y, ysin1);
    xmm1 = _mm256_add_pd(ysin1, ysin2);
    xmm2 = _mm256_add_pd(y, y2);

    /* update the sign */
    *s = _mm256_xor_pd(xmm1, sign_bit_sin);
    *c = _mm256_xor_pd(xmm2, sign_bit_cos);
}

static inline void sincos256d(double *src, double *dst_sin, double *dst_cos, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(src), (uintptr_t)(dst_sin), (uintptr_t)(dst_cos), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            v4sd dst_sin_tmp;
            v4sd dst_cos_tmp;
            sincos256_pd(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm256_store_pd(dst_sin + i, dst_sin_tmp);
            _mm256_store_pd(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            v4sd dst_sin_tmp;
            v4sd dst_cos_tmp;
            sincos256_pd(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm256_storeu_pd(dst_sin + i, dst_sin_tmp);
            _mm256_storeu_pd(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst_sin[i] = sin(src[i]);
        dst_cos[i] = cos(src[i]);
    }
}

#endif
