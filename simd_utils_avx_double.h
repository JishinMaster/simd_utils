/*
 * Project : SIMD_Utils
 * Version : 0.1.7
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

static inline void trunc256d(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
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


static inline v4sd asin256_pd(v4sd x)
{
    v4sd a, z, z_tmp;
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

    a = _mm256_and_pd(*(v4sd *) _pd256_positive_mask, x);      //fabs(x)
    sign = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OS);  //0xFFFFFFFF if x < 0.0

    ainfem8 = _mm256_cmp_pd(a, _mm256_set1_pd(1.0e-8), _CMP_LT_OS);  //if( a < 1.0e-8)
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

    //second branch
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
