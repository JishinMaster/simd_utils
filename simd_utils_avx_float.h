/*
 * Project : SIMD_Utils
 * Version : 0.1.12
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#include "immintrin.h"

#if 0
/* Compare */
_CMP_EQ_OQ    0x00 /* Equal (ordered, non-signaling)  */
_CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
_CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
_CMP_UNORD_Q  0x03 /* Unordered (non-signaling)  */
_CMP_NEQ_UQ   0x04 /* Not-equal (unordered, non-signaling)  */
_CMP_NLT_US   0x05 /* Not-less-than (unordered, signaling)  */
_CMP_NLE_US   0x06 /* Not-less-than-or-equal (unordered, signaling)  */
_CMP_ORD_Q    0x07 /* Ordered (nonsignaling)   */
_CMP_EQ_UQ    0x08 /* Equal (unordered, non-signaling)  */
_CMP_NGE_US   0x09 /* Not-greater-than-or-equal (unord, signaling)  */
_CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */
_CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
_CMP_NEQ_OQ   0x0c /* Not-equal (ordered, non-signaling)  */
_CMP_GE_OS    0x0d /* Greater-than-or-equal (ordered, signaling)  */
_CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
_CMP_TRUE_UQ  0x0f /* True (unordered, non-signaling)  */
_CMP_EQ_OS    0x10 /* Equal (ordered, signaling)  */
_CMP_LT_OQ    0x11 /* Less-than (ordered, non-signaling)  */
_CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
_CMP_UNORD_S  0x13 /* Unordered (signaling)  */
_CMP_NEQ_US   0x14 /* Not-equal (unordered, signaling)  */
_CMP_NLT_UQ   0x15 /* Not-less-than (unordered, non-signaling)  */
_CMP_NLE_UQ   0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
_CMP_ORD_S    0x17 /* Ordered (signaling)  */
_CMP_EQ_US    0x18 /* Equal (unordered, signaling)  */
_CMP_NGE_UQ   0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
_CMP_NGT_UQ   0x1a /* Not-greater-than (unordered, non-signaling)  */
_CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
_CMP_NEQ_OS   0x1c /* Not-equal (ordered, signaling)  */
_CMP_GE_OQ    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
_CMP_GT_OQ    0x1e /* Greater-than (ordered, non-signaling)  */
_CMP_TRUE_US  0x1f /* True (unordered, signaling)  */
#endif

_PS256_CONST(min1, -1.0f);
_PS256_CONST(min2, -2.0f);
_PS256_CONST(min0p5, -0.5f);

// For tanf
_PS256_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS256_CONST(DP1, -0.78515625);
_PS256_CONST(DP2, -2.4187564849853515625e-4);
_PS256_CONST(DP3, -3.77489497744594108e-8);

_PS256_CONST(FOPI, 1.27323954473516); /* 4/pi */
_PS256_CONST(TAN_P0, 9.38540185543E-3);
_PS256_CONST(TAN_P1, 3.11992232697E-3);
_PS256_CONST(TAN_P2, 2.44301354525E-2);
_PS256_CONST(TAN_P3, 5.34112807005E-2);
_PS256_CONST(TAN_P4, 1.33387994085E-1);
_PS256_CONST(TAN_P5, 3.33331568548E-1);

_PS256_CONST(ASIN_P0, 4.2163199048E-2);
_PS256_CONST(ASIN_P1, 2.4181311049E-2);
_PS256_CONST(ASIN_P2, 4.5470025998E-2);
_PS256_CONST(ASIN_P3, 7.4953002686E-2);
_PS256_CONST(ASIN_P4, 1.6666752422E-1);

_PS256_CONST(PIF, 3.14159265358979323846);      // PI
_PS256_CONST(mPIF, -3.14159265358979323846);    // -PI
_PS256_CONST(PIO2F, 1.57079632679489661923);    // PI/2 1.570796326794896619
_PS256_CONST(mPIO2F, -1.57079632679489661923);  // -PI/2 1.570796326794896619
_PS256_CONST(PIO4F, 0.785398163397448309615);   // PI/4 0.7853981633974483096

_PS256_CONST(TANPI8F, 0.414213562373095048802);   // tan(pi/8) => 0.4142135623730950
_PS256_CONST(TAN3PI8F, 2.414213562373095048802);  // tan(3*pi/8) => 2.414213562373095

_PS256_CONST(ATAN_P0, 8.05374449538e-2);
_PS256_CONST(ATAN_P1, -1.38776856032E-1);
_PS256_CONST(ATAN_P2, 1.99777106478E-1);
_PS256_CONST(ATAN_P3, -3.33329491539E-1);

_PS256_CONST_TYPE(pos_sign_mask, int, (int) 0x7FFFFFFF);
_PS256_CONST_TYPE(neg_sign_mask, int, (int) ~0x7FFFFFFF);

_PS256_CONST(MAXLOGF, 88.72283905206835f);
_PS256_CONST(MAXLOGFDIV2, 44.361419526034176f);
_PS256_CONST(MINLOGF, -103.278929903431851103f);
_PS256_CONST(cephes_exp_minC1, -0.693359375f);
_PS256_CONST(cephes_exp_minC2, 2.12194440e-4f);
_PS256_CONST(0p625, 0.625f);

_PS256_CONST(TANH_P0, -5.70498872745E-3f);
_PS256_CONST(TANH_P1, 2.06390887954E-2f);
_PS256_CONST(TANH_P2, -5.37397155531E-2f);
_PS256_CONST(TANH_P3, 1.33314422036E-1f);
_PS256_CONST(TANH_P4, -3.33332819422E-1f);

_PS256_CONST(MAXNUMF, 3.4028234663852885981170418348451692544e38f);
_PS256_CONST(minMAXNUMF, -3.4028234663852885981170418348451692544e38f);
_PS256_CONST(SINH_P0, 2.03721912945E-4f);
_PS256_CONST(SINH_P1, 8.33028376239E-3f);
_PS256_CONST(SINH_P2, 1.66667160211E-1f);

_PS256_CONST(1emin4, 1e-4f);
_PS256_CONST(ATANH_P0, 1.81740078349E-1f);
_PS256_CONST(ATANH_P1, 8.24370301058E-2f);
_PS256_CONST(ATANH_P2, 1.46691431730E-1f);
_PS256_CONST(ATANH_P3, 1.99782164500E-1f);
_PS256_CONST(ATANH_P4, 3.33337300303E-1f);

_PS256_CONST(1500, 1500.0f);
_PS256_CONST(LOGE2F, 0.693147180559945309f);
_PS256_CONST(ASINH_P0, 2.0122003309E-2f);
_PS256_CONST(ASINH_P1, -4.2699340972E-2f);
_PS256_CONST(ASINH_P2, 7.4847586088E-2f);
_PS256_CONST(ASINH_P3, -1.6666288134E-1f);

_PS256_CONST(ACOSH_P0, 1.7596881071E-3f);
_PS256_CONST(ACOSH_P1, -7.5272886713E-3f);
_PS256_CONST(ACOSH_P2, 2.6454905019E-2f);
_PS256_CONST(ACOSH_P3, -1.1784741703E-1f);
_PS256_CONST(ACOSH_P4, 1.4142135263E0f);

static inline void log10_256f(float *src, float *dst, int len)
{
    const v8sf invln10f = _mm256_set1_ps((float) INVLN10);  //_mm256_broadcast_ss(&invln10f_mask);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log256_ps(_mm256_load_ps(src + i));
            _mm256_store_ps(dst + i, _mm256_mul_ps(src_tmp, invln10f));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log256_ps(_mm256_loadu_ps(src + i));
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(src_tmp, invln10f));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void log2_256f(float *src, float *dst, int len)
{
    const v8sf invln2f = _mm256_set1_ps((float) INVLN2);  //_mm256_broadcast_ss(&invln10f_mask);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log256_ps(_mm256_load_ps(src + i));
            _mm256_store_ps(dst + i, _mm256_mul_ps(src_tmp, invln2f));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log256_ps(_mm256_loadu_ps(src + i));
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(src_tmp, invln2f));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void ln_256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, log256_ps(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, log256_ps(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void exp_256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, exp256_ps(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, exp256_ps(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

//rewritten alternate version which properly returns MAXNUMF or 0.0 outside of boundaries
static inline v8sf exp256_ps_alternate(v8sf x)
{
    v8sf z_tmp, z, fx;
    v8si n;
    v8sf xsupmaxlogf, xinfminglogf;

    xsupmaxlogf = _mm256_cmp_ps(x, *(v8sf *) _ps256_MAXLOGF, _CMP_GT_OS);
    xinfminglogf = _mm256_cmp_ps(x, *(v8sf *) _ps256_MINLOGF, _CMP_LT_OS);

    /* Express e**x = e**g 2**n
   *   = e**g e**( n loge(2) )
   *   = e**( g + n loge(2) )
   */
    fx = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_cephes_LOG2EF, x, *(v8sf *) _ps256_0p5);
    z = _mm256_round_ps(fx, _MM_FROUND_FLOOR);

    x = _mm256_fmadd_ps_custom(z, *(v8sf *) _ps256_cephes_exp_minC1, x);
    x = _mm256_fmadd_ps_custom(z, *(v8sf *) _ps256_cephes_exp_minC2, x);

    n = _mm256_cvttps_epi32(z);
    n = _mm256_add_epi32(n, *(v8si *) _pi32_256_0x7f);
    n = _mm256_slli_epi32(n, 23);
    v8sf pow2n = _mm256_castsi256_ps(n);

    z = _mm256_mul_ps(x, x);

    z_tmp = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_cephes_exp_p0, x, *(v8sf *) _ps256_cephes_exp_p1);
    z_tmp = _mm256_fmadd_ps_custom(z_tmp, x, *(v8sf *) _ps256_cephes_exp_p2);
    z_tmp = _mm256_fmadd_ps_custom(z_tmp, x, *(v8sf *) _ps256_cephes_exp_p3);
    z_tmp = _mm256_fmadd_ps_custom(z_tmp, x, *(v8sf *) _ps256_cephes_exp_p4);
    z_tmp = _mm256_fmadd_ps_custom(z_tmp, x, *(v8sf *) _ps256_cephes_exp_p5);
    z_tmp = _mm256_fmadd_ps_custom(z_tmp, z, x);
    z_tmp = _mm256_add_ps(z_tmp, *(v8sf *) _ps256_1);

    /* build 2^n */
    z_tmp = _mm256_mul_ps(z_tmp, pow2n);

    z = _mm256_blendv_ps(z_tmp, *(v8sf *) _ps256_MAXNUMF, xsupmaxlogf);
    z = _mm256_blendv_ps(z, _mm256_setzero_ps(), xinfminglogf);

    return z;
}

static inline void fabs256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void set256f(float *src, float value, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = value;
    }
}

static inline void zero256f(float *src, int len)
{
    const v8sf tmp = _mm256_setzero_ps();

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = 0.0f;
    }
}


static inline void copy256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_load_ps(src + i));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_add_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}


static inline void mul256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_mul_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_sub_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}


static inline void addc256f(float *src, float value, float *dst, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_add_ps(tmp, _mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_add_ps(tmp, _mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc256f(float *src, float value, float *dst, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_mul_ps(tmp, _mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(tmp, _mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd256f(float *_a, float *_b, float *_c, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(_b), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t)(_c), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(_a + i);
            v8sf b = _mm256_load_ps(_b + i);
            v8sf c = _mm256_load_ps(_c + i);
            _mm256_store_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(_a + i);
            v8sf b = _mm256_loadu_ps(_b + i);
            v8sf c = _mm256_loadu_ps(_c + i);
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcadd256f(float *_a, float _b, float *_c, float *dst, int len)
{
    v8sf b = _mm256_set1_ps(_b);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_c), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(_a + i);
            v8sf c = _mm256_load_ps(_c + i);
            _mm256_store_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(_a + i);
            v8sf c = _mm256_loadu_ps(_c + i);
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddc256f(float *_a, float _b, float _c, float *dst, int len)
{
    v8sf b = _mm256_set1_ps(_b);
    v8sf c = _mm256_set1_ps(_c);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(_a + i);
            _mm256_store_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(_a + i);
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
    }
}

static inline void muladdc256f(float *_a, float *_b, float _c, float *dst, int len)
{
    v8sf c = _mm256_set1_ps(_c);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_b), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(_a + i);
            v8sf b = _mm256_load_ps(_b + i);
            _mm256_store_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(_a + i);
            v8sf b = _mm256_loadu_ps(_b + i);
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

static inline void div256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_div_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_div_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

// TODO : remove previous index dependency to become Out Of Order
//TODO: intel targets can do 2 mul/FMA per cycle but only one add => replace add_ps(a,b) by fmadd_ps(1.0f,a,b)
static inline void vectorSlope256f(float *dst, int len, float offset, float slope)
{
    v8sf coef = _mm256_set_ps(7.0f * slope, 6.0f * slope, 5.0f * slope, 4.0f * slope, 3.0f * slope, 2.0f * slope, slope, 0.0f);
    v8sf slope16_vec = _mm256_set1_ps(16.0f * slope);
    v8sf curVal = _mm256_add_ps(_mm256_set1_ps(offset), coef);
    v8sf curVal2 = _mm256_add_ps(_mm256_set1_ps(offset), coef);
    curVal2 = _mm256_add_ps(curVal2, _mm256_set1_ps(8.0f * slope));
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (((uintptr_t)(const void *) (dst) % AVX_LEN_BYTES) == 0) {
        _mm256_store_ps(dst + 0, curVal);
        _mm256_store_ps(dst + AVX_LEN_FLOAT, curVal2);
    } else {
        _mm256_storeu_ps(dst + 0, curVal);
        _mm256_storeu_ps(dst + AVX_LEN_FLOAT, curVal2);
    }

    if (((uintptr_t)(const void *) (dst) % AVX_LEN_BYTES) == 0) {
        for (int i = 2 * AVX_LEN_FLOAT; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            curVal = _mm256_add_ps(curVal, slope16_vec);
            _mm256_store_ps(dst + i, curVal);
            curVal2 = _mm256_add_ps(curVal2, slope16_vec);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, curVal2);
        }
    } else {
        for (int i = 2 * AVX_LEN_FLOAT; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            curVal = _mm256_add_ps(curVal, slope16_vec);
            _mm256_storeu_ps(dst + i, curVal);
            curVal2 = _mm256_add_ps(curVal2, slope16_vec);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

static inline void print8(__m256 v)
{
    float *p = (float *) &v;
#ifndef __SSE2__
    _mm_empty();
#endif
    printf("[%3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}

// converts 32bits complex float to two arrays real and im
//Work in progress
static inline void cplxtoreal256f(float *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (AVX_LEN_FLOAT);
    stop_len *= AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned3((uintptr_t)(src), (uintptr_t)(dstRe), (uintptr_t)(dstIm), AVX_LEN_FLOAT)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf vec1 = _mm256_load_ps(src + i);                                                       //load 0 1 2 3 4 5 6 7
            v8sf vec2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);                                       // load 8 9 10 11 12 13 14 15
            v8sf vec1_permute = _mm256_permute2f128_ps(vec1, vec1, IMM8_PERMUTE_128BITS_LANES);        // reverse v1 4 5 6 7 0 1 2 3
            v8sf vec2_permute = _mm256_permute2f128_ps(vec2, vec1, IMM8_PERMUTE_128BITS_LANES);        // reverse v2 12 13 14 15 8 9 10 11
            v8sf vec1_even = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(2, 0, 2, 0));           // 0 2 4 6 0 2 4 6
            v8sf vec1_odd = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(3, 1, 3, 1));            // 1 3 5 7 1 3 5 7
            v8sf vec2_even = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(2, 0, 2, 0));           // 8 10 12 14
            v8sf vec2_odd = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(3, 1, 3, 1));            // 9 11 13 15
            v8sf tmp1permute = _mm256_insertf128_ps(vec1_even, _mm256_castps256_ps128(vec2_even), 1);  // 0 2 4 6 8 10 12 14
            v8sf tmp2permute = _mm256_insertf128_ps(vec1_odd, _mm256_castps256_ps128(vec2_odd), 1);    //1 3 5 7 9 11 13 15

            _mm256_store_ps(dstRe + j, tmp1permute);
            _mm256_store_ps(dstIm + j, tmp2permute);
            j += AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf vec1 = _mm256_loadu_ps(src + i);
            v8sf vec2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf vec1_permute = _mm256_permute2f128_ps(vec1, vec1, IMM8_PERMUTE_128BITS_LANES);
            v8sf vec2_permute = _mm256_permute2f128_ps(vec2, vec1, IMM8_PERMUTE_128BITS_LANES);
            v8sf vec1_even = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(2, 0, 2, 0));
            v8sf vec1_odd = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(3, 1, 3, 1));
            v8sf vec2_even = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(2, 0, 2, 0));
            v8sf vec2_odd = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(3, 1, 3, 1));
            v8sf tmp1permute = _mm256_insertf128_ps(vec1_even, _mm256_castps256_ps128(vec2_even), 1);
            v8sf tmp2permute = _mm256_insertf128_ps(vec1_odd, _mm256_castps256_ps128(vec2_odd), 1);

            _mm256_storeu_ps(dstRe + j, tmp1permute);
            _mm256_storeu_ps(dstIm + j, tmp2permute);
            j += AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < 2 * len; i += 2) {
        dstRe[j] = src[i];
        dstIm[j] = src[i + 1];
        j++;
    }
}


static inline void realtocplx256f(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);
    stop_len *= AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned3((uintptr_t)(srcRe), (uintptr_t)(srcIm), (uintptr_t)(dst), AVX_LEN_FLOAT)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf re = _mm256_load_ps(srcRe + i);
            v8sf im = _mm256_load_ps(srcIm + i);

            //printf("Re : "); print8(re); printf("\n");
            //printf("Im : "); print8(im); printf("\n");
            v8sf cplx0 = _mm256_unpacklo_ps(re, im);
            //printf("cplx0 : "); print8(cplx0); printf("\n");
            v8sf cplx1 = _mm256_unpackhi_ps(re, im);
            v8sf perm0 = _mm256_permute2f128_ps(cplx0, cplx1, 0x20);  //permute mask [cplx1(127:0],cplx0[127:0])
            v8sf perm1 = _mm256_permute2f128_ps(cplx0, cplx1, 0x31);  //permute mask [cplx1(255:128],cplx0[255:128])

            //printf("perm0 : "); print8(perm0); printf("\n");
            //printf("perm1 : "); print8(perm1); printf("\n");
            _mm256_store_ps(dst + j, perm0);
            _mm256_store_ps(dst + j + AVX_LEN_FLOAT, perm1);
            j += 2 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf re = _mm256_loadu_ps(srcRe + i);
            v8sf im = _mm256_loadu_ps(srcIm + i);
            v8sf cplx0 = _mm256_unpacklo_ps(re, im);
            v8sf cplx1 = _mm256_unpackhi_ps(re, im);
            v8sf perm0 = _mm256_permute2f128_ps(cplx0, cplx1, 0x20);  //permute mask [cplx1(127:0],cplx0[127:0])
            v8sf perm1 = _mm256_permute2f128_ps(cplx0, cplx1, 0x31);  //permute mask [cplx1(255:128],cplx0[255:128])

            //printf("perm0 : "); print8(perm0); printf("\n");
            //printf("perm1 : "); print8(perm1); printf("\n");
            _mm256_storeu_ps(dst + j, perm0);
            _mm256_storeu_ps(dst + j + AVX_LEN_FLOAT, perm1);
            j += 2 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[j] = srcRe[i];
        dst[j + 1] = srcIm[i];
        j += 2;
    }
}

static inline void convert256_64f32f(double *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            __m128 src_lo = _mm256_cvtpd_ps(_mm256_load_pd(src + i));
            __m128 src_hi = _mm256_cvtpd_ps(_mm256_load_pd(src + i + 4));
            _mm256_store_ps(dst + i, _mm256_set_m128(src_hi, src_lo));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            __m128 src_lo = _mm256_cvtpd_ps(_mm256_loadu_pd(src + i));
            __m128 src_hi = _mm256_cvtpd_ps(_mm256_loadu_pd(src + i + 4));
            _mm256_storeu_ps(dst + i, _mm256_set_m128(src_hi, src_lo));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i];
    }
}

static inline void convert256_32f64f(float *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            __m128 src_tmp = _mm_load_ps(src + i);               //load a,b,c,d
            _mm256_store_pd(dst + i, _mm256_cvtps_pd(src_tmp));  //store the abcd converted in 64bits
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            __m128 src_tmp = _mm_loadu_ps(src + i);               //load a,b,c,d
            _mm256_storeu_pd(dst + i, _mm256_cvtps_pd(src_tmp));  //store the c and d converted in 64bits
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (double) src[i];
    }
}

static inline void flip256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    for (int i = 0; i < AVX_LEN_FLOAT; i++) {
        dst[len - i - 1] = src[i];
    }

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = AVX_LEN_FLOAT; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);                                                           //load a,b,c,d,e,f,g,h
            v8sf src_tmp_flip = _mm256_permute2f128_ps(src_tmp, src_tmp, IMM8_PERMUTE_128BITS_LANES);         // reverse lanes abcdefgh to efghabcd
            _mm256_storeu_ps(dst + len - i - AVX_LEN_FLOAT, _mm256_permute_ps(src_tmp_flip, IMM8_FLIP_VEC));  //store the flipped vector
        }
    } else {
        /*for(int i = AVX_LEN_FLOAT; i < stop_len; i+= AVX_LEN_FLOAT){
			__m128 src_tmp = _mm256_loadu_ps(src + i); //load a,b,c,d,e,f,g,h
			__m128 src_tmp_flip = _mm256_shuffle_ps (src_tmp, src_tmp, IMM8_FLIP_VEC);// rotate vec from abcd to bcba
			_mm256_storeu_ps(dst + len -i - AVX_LEN_FLOAT, src_tmp_flip); //store the flipped vector
		}*/
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void maxevery256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_max_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_max_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_min_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_min_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void minmax256f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    v8sf max_v, min_v;
    v8sf src_tmp;
    float min_f[AVX_LEN_FLOAT] __attribute__((aligned(AVX_LEN_BYTES)));
    float max_f[AVX_LEN_FLOAT] __attribute__((aligned(AVX_LEN_BYTES)));
    float min_tmp;
    float max_tmp;

    if (isAligned((uintptr_t)(src), AVX_LEN_BYTES)) {
        src_tmp = _mm256_load_ps(src + 0);
        max_v = src_tmp;
        min_v = src_tmp;
        for (int i = AVX_LEN_FLOAT; i < stop_len; i += AVX_LEN_FLOAT) {
            src_tmp = _mm256_load_ps(src + i);
            max_v = _mm256_max_ps(max_v, src_tmp);
            min_v = _mm256_min_ps(min_v, src_tmp);
        }
    } else {
        src_tmp = _mm256_loadu_ps(src + 0);
        max_v = src_tmp;
        min_v = src_tmp;
        for (int i = AVX_LEN_FLOAT; i < stop_len; i += AVX_LEN_FLOAT) {
            src_tmp = _mm256_loadu_ps(src + i);
            max_v = _mm256_max_ps(max_v, src_tmp);
            min_v = _mm256_min_ps(min_v, src_tmp);
        }
    }

    _mm256_store_ps(max_f, max_v);
    _mm256_store_ps(min_f, min_v);

    max_tmp = max_f[0];
    max_tmp = max_tmp > max_f[1] ? max_tmp : max_f[1];
    max_tmp = max_tmp > max_f[2] ? max_tmp : max_f[2];
    max_tmp = max_tmp > max_f[3] ? max_tmp : max_f[3];
    max_tmp = max_tmp > max_f[4] ? max_tmp : max_f[4];
    max_tmp = max_tmp > max_f[5] ? max_tmp : max_f[5];
    max_tmp = max_tmp > max_f[6] ? max_tmp : max_f[6];
    max_tmp = max_tmp > max_f[7] ? max_tmp : max_f[7];


    min_tmp = min_f[0];
    min_tmp = min_tmp < min_f[1] ? min_tmp : min_f[1];
    min_tmp = min_tmp < min_f[2] ? min_tmp : min_f[2];
    min_tmp = min_tmp < min_f[3] ? min_tmp : min_f[3];
    min_tmp = min_tmp < min_f[4] ? min_tmp : min_f[4];
    min_tmp = min_tmp < min_f[5] ? min_tmp : min_f[5];
    min_tmp = min_tmp < min_f[6] ? min_tmp : min_f[6];
    min_tmp = min_tmp < min_f[7] ? min_tmp : min_f[7];

    for (int i = stop_len; i < len; i++) {
        max_tmp = max_tmp > src[i] ? max_tmp : src[i];
        min_tmp = min_tmp < src[i] ? min_tmp : src[i];
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

static inline void threshold256_gt_f(float *src, float *dst, int len, float value)
{
    v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_min_ps(src_tmp, tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_min_ps(src_tmp, tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold256_gtabs_f(float *src, float *dst, int len, float value)
{
    const v8sf pval = _mm256_set1_ps(value);
    const v8sf mval = _mm256_set1_ps(-value);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf gtmask = _mm256_cmp_ps(src_abs, pval, _CMP_GT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);           //if A >= 0 value, else -value
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, gtmask);     // either A or sval (+- value)
            _mm256_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf gtmask = _mm256_cmp_ps(src_abs, pval, _CMP_GT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);           //if A >= 0 value, else -value
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, gtmask);     // either A or sval (+- value)
            _mm256_storeu_ps(dst + i, dst_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] >= 0.0f) {
            dst[i] = src[i] > value ? value : src[i];
        } else {
            dst[i] = src[i] < (-value) ? (-value) : src[i];
        }
    }
}

static inline void threshold256_lt_f(float *src, float *dst, int len, float value)
{
    v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_max_ps(src_tmp, tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_max_ps(src_tmp, tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold256_ltabs_f(float *src, float *dst, int len, float value)
{
    const v8sf pval = _mm256_set1_ps(value);
    const v8sf mval = _mm256_set1_ps(-value);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf gtmask = _mm256_cmp_ps(src_abs, pval, _CMP_LT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);           //if A >= 0 value, else -value
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, gtmask);     // either A or sval (+- value)
            _mm256_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf gtmask = _mm256_cmp_ps(src_abs, pval, _CMP_LT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);           //if A >= 0 value, else -value
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, gtmask);     // either A or sval (+- value)
            _mm256_storeu_ps(dst + i, dst_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] >= 0.0f) {
            dst[i] = src[i] < value ? value : src[i];
        } else {
            dst[i] = src[i] > (-value) ? (-value) : src[i];
        }
    }
}

static inline void threshold256_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v8sf ltlevel_v = _mm256_set1_ps(ltlevel);
    const v8sf ltvalue_v = _mm256_set1_ps(ltvalue);
    const v8sf gtlevel_v = _mm256_set1_ps(gtlevel);
    const v8sf gtvalue_v = _mm256_set1_ps(gtvalue);

    int stop_len = len / AVX_LEN_BYTES;
    stop_len *= AVX_LEN_BYTES;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf lt_mask = _mm256_cmp_ps(src_tmp, ltlevel_v, _CMP_LT_OS);
            v8sf gt_mask = _mm256_cmp_ps(src_tmp, gtlevel_v, _CMP_GT_OS);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm256_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm256_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf lt_mask = _mm256_cmp_ps(src_tmp, ltlevel_v, _CMP_LT_OS);
            v8sf gt_mask = _mm256_cmp_ps(src_tmp, gtlevel_v, _CMP_GT_OS);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm256_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm256_storeu_ps(dst + i, dst_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}

static inline void sin256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, sin256_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, sin256_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, cos256_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, cos256_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos256f(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf dst_sin_tmp;
            v8sf dst_cos_tmp;
            sincos256_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm256_store_ps(dst_sin + i, dst_sin_tmp);
            _mm256_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf dst_sin_tmp;
            v8sf dst_cos_tmp;
            sincos256_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm256_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm256_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline v8sf acosh256f_ps(v8sf x)
{
    v8sf z, z_first_branch, z_second_branch;
    v8sf xsup1500, zinf0p5, xinf1;

    xsup1500 = _mm256_cmp_ps(x, *(v8sf *) _ps256_1500, _CMP_GT_OS);  // return  (logf(x) + LOGE2F)
    xinf1 = _mm256_cmp_ps(x, *(v8sf *) _ps256_1, _CMP_LT_OS);        // return 0

    z = _mm256_sub_ps(x, *(v8sf *) _ps256_1);

    zinf0p5 = _mm256_cmp_ps(z, *(v8sf *) _ps256_0p5, _CMP_LT_OS);  // first and second branch

    //First Branch (z < 0.5)
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ACOSH_P0, z, *(v8sf *) _ps256_ACOSH_P1);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P2);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P3);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P4);
    z_first_branch = _mm256_mul_ps(z_first_branch, _mm256_sqrt_ps(z));

    //Second Branch
    z_second_branch = _mm256_sqrt_ps(_mm256_fmadd_ps_custom(z, x, z));
    z_second_branch = log256_ps(_mm256_add_ps(x, z_second_branch));

    z = _mm256_blendv_ps(z_second_branch, z_first_branch, zinf0p5);
    z = _mm256_blendv_ps(z, _mm256_add_ps(log256_ps(x), *(v8sf *) _ps256_LOGE2F), xsup1500);
    z = _mm256_blendv_ps(z, _mm256_setzero_ps(), xinf1);

    return z;
}

static inline void acosh256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, acosh256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, acosh256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline v8sf asinh256f_ps(v8sf xx)
{
    v8sf x, tmp, z, z_first_branch, z_second_branch;
    v8sf xxinf0, xsup1500, xinf0p5;

    x = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, xx);
    xsup1500 = _mm256_cmp_ps(x, *(v8sf *) _ps256_1500, _CMP_GT_OS);
    xinf0p5 = _mm256_cmp_ps(x, *(v8sf *) _ps256_0p5, _CMP_LT_OS);

    xxinf0 = _mm256_cmp_ps(xx, _mm256_setzero_ps(), _CMP_LT_OS);

    tmp = _mm256_mul_ps(x, x);
    //First Branch (x < 0.5)
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ASINH_P0, tmp, *(v8sf *) _ps256_ASINH_P1);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ASINH_P2);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ASINH_P3);
    z_first_branch = _mm256_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, x, x);

    //Second Branch
    z_second_branch = _mm256_sqrt_ps(_mm256_add_ps(tmp, *(v8sf *) _ps256_1));
    z_second_branch = log256_ps(_mm256_add_ps(z_second_branch, x));

    z = _mm256_blendv_ps(z_second_branch, z_first_branch, xinf0p5);
    z = _mm256_blendv_ps(z, _mm256_add_ps(log256_ps(x), *(v8sf *) _ps256_LOGE2F), xsup1500);
    z = _mm256_blendv_ps(z, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, z), xxinf0);

    return z;
}

static inline void asinh256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, asinh256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, asinh256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline v8sf atanh256f_ps(v8sf x)
{
    v8sf z, tmp, tmp2, z_first_branch, z_second_branch;
    v8sf xsup1, xinfmin1, zinf1emin4, zinf0p5;

    z = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, x);

    xsup1 = _mm256_cmp_ps(x, *(v8sf *) _ps256_1, _CMP_GE_OS);
    xinfmin1 = _mm256_cmp_ps(x, *(v8sf *) _ps256_min1, _CMP_LE_OS);
    zinf1emin4 = _mm256_cmp_ps(z, *(v8sf *) _ps256_1emin4, _CMP_LT_OS);
    zinf0p5 = _mm256_cmp_ps(z, *(v8sf *) _ps256_0p5, _CMP_LT_OS);

    //First branch
    tmp = _mm256_mul_ps(x, x);
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ATANH_P0, tmp, *(v8sf *) _ps256_ATANH_P1);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ATANH_P2);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ATANH_P3);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ATANH_P4);
    z_first_branch = _mm256_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, x, x);

    //Second branch
    tmp = _mm256_sub_ps(*(v8sf *) _ps256_1, x);
    tmp2 = _mm256_rcp_ps(tmp);
    tmp = _mm256_fmadd_ps_custom(tmp2, x, tmp2);
    z_second_branch = log256_ps(tmp);
    z_second_branch = _mm256_mul_ps(*(v8sf *) _ps256_0p5, z_second_branch);

    z = _mm256_blendv_ps(z_second_branch, z_first_branch, zinf0p5);
    z = _mm256_blendv_ps(z, x, zinf1emin4);
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_MAXNUMF, xsup1);
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_minMAXNUMF, xinfmin1);

    return (z);
}

static inline void atanh256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, atanh256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, atanh256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline v8sf cosh256f_ps(v8sf xx)
{
    v8sf x, y, tmp;
    v8sf xsupmaxlogf;

    x = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, xx);
    xsupmaxlogf = _mm256_cmp_ps(x, *(v8sf *) _ps256_MAXLOGF, _CMP_GT_OS);

    y = exp256_ps(x);
    tmp = _mm256_div_ps(*(v8sf *) _ps256_0p5, y);
    y = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_0p5, y, tmp);
    y = _mm256_blendv_ps(y, *(v8sf *) _ps256_MAXNUMF, xsupmaxlogf);

    return y;
}

static inline void cosh256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, cosh256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, cosh256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline v8sf sinh256f_ps(v8sf x)
{
    v8sf z, z_first_branch, z_second_branch, tmp;
    v8sf xsupmaxlogf, zsup1, xinf0;

    // x = xx; if x < 0, z = -x, else x
    z = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, x);

    xsupmaxlogf = _mm256_cmp_ps(z, *(v8sf *) _ps256_MAXLOGF, _CMP_GT_OS);

    // First branch
    zsup1 = _mm256_cmp_ps(z, *(v8sf *) _ps256_1, _CMP_GT_OS);
    xinf0 = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OS);
    z_first_branch = exp256_ps(z);
    tmp = _mm256_div_ps(*(v8sf *) _ps256_min0p5, z_first_branch);
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_0p5, z_first_branch, tmp);

    z_first_branch = _mm256_blendv_ps(z_first_branch, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, z_first_branch), xinf0);

    // Second branch
    tmp = _mm256_mul_ps(x, x);
    z_second_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_SINH_P0, tmp, *(v8sf *) _ps256_SINH_P1);
    z_second_branch = _mm256_fmadd_ps_custom(z_second_branch, tmp, *(v8sf *) _ps256_SINH_P2);
    z_second_branch = _mm256_mul_ps(z_second_branch, tmp);
    z_second_branch = _mm256_fmadd_ps_custom(z_second_branch, x, x);

    // Choose between first and second branch
    z = _mm256_blendv_ps(z_second_branch, z_first_branch, zsup1);

    // Set value to MAXNUMF if abs(x) > MAGLOGF
    // Set value to -MAXNUMF if abs(x) > MAGLOGF and x < 0
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_MAXNUMF, xsupmaxlogf);
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_minMAXNUMF, _mm256_and_ps(xinf0, xsupmaxlogf));

    return (z);
}

static inline void sinh256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, sinh256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, sinh256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}

#ifndef __AVX2__  // Needs AVX2 to  get _mm256_cmpgt_epi32
#warning "Using SSE2 to perform AVX2 integer ops"
AVX2_INTOP_USING_SSE2(cmpgt_epi32)
#endif

static inline v8sf atan256f_ps(v8sf xx)
{
    v8sf x, y, z;
    v8sf sign2;
    v8sf suptan3pi8, inftan3pi8suppi8;
    v8sf tmp;

    x = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, xx);
    sign2 = _mm256_cmp_ps(xx, _mm256_setzero_ps(), _CMP_LT_OS);  //0xFFFFFFFF if x < 0.0, sign = -1
    /* range reduction */

    y = _mm256_setzero_ps();
    suptan3pi8 = _mm256_cmp_ps(x, *(v8sf *) _ps256_TAN3PI8F, _CMP_GT_OS);  // if( x > tan 3pi/8 )
    x = _mm256_blendv_ps(x, _mm256_div_ps(*(v8sf *) _ps256_min1, x), suptan3pi8);
    y = _mm256_blendv_ps(y, *(v8sf *) _ps256_PIO2F, suptan3pi8);


    inftan3pi8suppi8 = _mm256_and_ps(_mm256_cmp_ps(x, *(v8sf *) _ps256_TAN3PI8F, _CMP_LT_OS), _mm256_cmp_ps(x, *(v8sf *) _ps256_TANPI8F, _CMP_GT_OS));  // if( x > tan 3pi/8 )
    x = _mm256_blendv_ps(x, _mm256_div_ps(_mm256_sub_ps(x, *(v8sf *) _ps256_1), _mm256_add_ps(x, *(v8sf *) _ps256_1)), inftan3pi8suppi8);
    y = _mm256_blendv_ps(y, *(v8sf *) _ps256_PIO4F, inftan3pi8suppi8);

    z = _mm256_mul_ps(x, x);
    tmp = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ATAN_P0, z, *(v8sf *) _ps256_ATAN_P1);
    tmp = _mm256_fmadd_ps_custom(tmp, z, *(v8sf *) _ps256_ATAN_P2);
    tmp = _mm256_fmadd_ps_custom(tmp, z, *(v8sf *) _ps256_ATAN_P3);
    tmp = _mm256_mul_ps(z, tmp);
    tmp = _mm256_fmadd_ps_custom(tmp, x, x);

    y = _mm256_add_ps(y, tmp);

    y = _mm256_blendv_ps(y, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, y), sign2);

    return (y);
}

static inline void atan256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, atan256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, atan256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline v8sf atan2256f_ps(v8sf y, v8sf x)
{
    v8sf z, w;
    v8sf xinfzero, yinfzero, xeqzero, yeqzero;
    v8sf xeqzeroandyinfzero, yeqzeroandxinfzero;
    v8sf specialcase;

    xinfzero = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OS);  // code =2
    yinfzero = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_LT_OS);  // code = code |1;

    xeqzero = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OS);
    yeqzero = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_EQ_OS);

    z = *(v8sf *) _ps256_PIO2F;

    xeqzeroandyinfzero = _mm256_and_ps(xeqzero, yinfzero);
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_mPIO2F, xeqzeroandyinfzero);
    z = _mm256_blendv_ps(z, _mm256_setzero_ps(), yeqzero);

    yeqzeroandxinfzero = _mm256_and_ps(yeqzero, xinfzero);
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_PIF, yeqzeroandxinfzero);

    specialcase = _mm256_or_ps(xeqzero, yeqzero);

    w = _mm256_setzero_ps();
    w = _mm256_blendv_ps(w, *(v8sf *) _ps256_PIF, _mm256_andnot_ps(yinfzero, xinfzero));  // y >= 0 && x<0
    w = _mm256_blendv_ps(w, *(v8sf *) _ps256_mPIF, _mm256_and_ps(yinfzero, xinfzero));    // y < 0 && x<0

    z = _mm256_blendv_ps(_mm256_add_ps(w, atan256f_ps(_mm256_div_ps(y, x))), z, specialcase);  // atanf(y/x) if not in special case

    return (z);
}

static inline void atan2256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, atan2256f_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, atan2256f_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}

static inline v8sf asin256f_ps(v8sf xx)
{
    v8sf a, x, z, z_tmp;
    v8sf sign;
    v8sf ainfem4, asup0p5;
    v8sf tmp;
    x = xx;
    a = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, x);      //fabs(x)
    sign = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OS);  //0xFFFFFFFF if x < 0.0

    //TODO : vectorize this
    /*if( a > 1.0f )
	{
		return( 0.0f );
	}*/


    ainfem4 = _mm256_cmp_ps(a, _mm256_set1_ps(1.0e-4), _CMP_LT_OS);  //if( a < 1.0e-4f )

    asup0p5 = _mm256_cmp_ps(a, *(v8sf *) _ps256_0p5, _CMP_GT_OS);  //if( a > 0.5f ) flag = 1 else 0
    z_tmp = _mm256_sub_ps(*(v8sf *) _ps256_1, a);
    z_tmp = _mm256_mul_ps(*(v8sf *) _ps256_0p5, z_tmp);
    z = _mm256_blendv_ps(_mm256_mul_ps(a, a), z_tmp, asup0p5);
    x = _mm256_blendv_ps(a, _mm256_sqrt_ps(z), asup0p5);

    tmp = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ASIN_P0, z, *(v8sf *) _ps256_ASIN_P1);
    tmp = _mm256_fmadd_ps_custom(z, tmp, *(v8sf *) _ps256_ASIN_P2);
    tmp = _mm256_fmadd_ps_custom(z, tmp, *(v8sf *) _ps256_ASIN_P3);
    tmp = _mm256_fmadd_ps_custom(z, tmp, *(v8sf *) _ps256_ASIN_P4);
    tmp = _mm256_mul_ps(z, tmp);
    tmp = _mm256_fmadd_ps_custom(x, tmp, x);

    z = tmp;

    //with FMA (fmsub_ps), it could be 1 or 2 cycles faster
    z_tmp = _mm256_add_ps(z, z);
    z_tmp = _mm256_sub_ps(*(v8sf *) _ps256_PIO2F, z_tmp);

    z = _mm256_blendv_ps(z, z_tmp, asup0p5);

    //done:
    z = _mm256_blendv_ps(z, a, ainfem4);
    z = _mm256_blendv_ps(z, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, z), sign);

    return (z);
}

static inline void asin256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, asin256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, asin256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}


static inline v8sf tanh256f_ps(v8sf xx)
{
    v8sf x, z, z_first_branch, z_second_branch;
    v8sf xxsup0, xsupmaxlogfdiv2, xsup0p625;

    xxsup0 = _mm256_cmp_ps(xx, _mm256_setzero_ps(), _CMP_GT_OS);
    xsupmaxlogfdiv2 = _mm256_cmp_ps(xx, *(v8sf *) _ps256_MAXLOGFDIV2, _CMP_GT_OS);

    x = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, xx);

    xsup0p625 = _mm256_cmp_ps(x, *(v8sf *) _ps256_0p625, _CMP_GE_OS);
    x = _mm256_blendv_ps(x, exp256_ps(_mm256_add_ps(x, x)), xsup0p625);

    // z = 1.0 - 2.0 / (x + 1.0);
    z_first_branch = _mm256_add_ps(x, *(v8sf *) _ps256_1);
    z_first_branch = _mm256_div_ps(*(v8sf *) _ps256_min2, z_first_branch);
    z_first_branch = _mm256_add_ps(*(v8sf *) _ps256_1, z_first_branch);
    z_first_branch = _mm256_blendv_ps(_mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, z_first_branch), z_first_branch, xxsup0);

    //z = x * x;
    z = _mm256_mul_ps(x, x);

    z_second_branch = _mm256_fmadd_ps_custom(z, *(v8sf *) _ps256_TANH_P0, *(v8sf *) _ps256_TANH_P1);
    z_second_branch = _mm256_fmadd_ps_custom(z_second_branch, z, *(v8sf *) _ps256_TANH_P2);
    z_second_branch = _mm256_fmadd_ps_custom(z_second_branch, z, *(v8sf *) _ps256_TANH_P3);
    z_second_branch = _mm256_fmadd_ps_custom(z_second_branch, z, *(v8sf *) _ps256_TANH_P4);
    z_second_branch = _mm256_mul_ps(z_second_branch, z);
    z_second_branch = _mm256_fmadd_ps_custom(z_second_branch, xx, xx);

    z = _mm256_blendv_ps(z_second_branch, z_first_branch, xsup0p625);
    // if (x > 0.5 * MAXLOGF), return (xx > 0)? 1.0f: -1.0f
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_min1, xsupmaxlogfdiv2);
    z = _mm256_blendv_ps(z, *(v8sf *) _ps256_1, _mm256_and_ps(xxsup0, xsupmaxlogfdiv2));

    return (z);
}

static inline void tanh256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, tanh256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, tanh256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}


#if 1
static inline v8sf tan256f_ps(v8sf xx)
{
    v8sf x, y, z, zz;
    v8si j;  //long?
    v8sf sign, xsupem4;
    v8sf tmp;
    v8si jandone, jandtwo;

    x = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, xx);  //fabs(xx)

    /* compute x mod PIO4 */

    //TODO : on neg values should be ceil and not floor
    //j = _mm256_cvtps_epi32( _mm256_round_ps(_mm256_mul_ps(*(v8sf*)_ps256_FOPI,x), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC )); /* integer part of x/(PI/4), using floor */
    j = _mm256_cvttps_epi32(_mm256_mul_ps(*(v8sf *) _ps256_FOPI, x));
    y = _mm256_cvtepi32_ps(j);


#ifndef __AVX2__
    v4si andone_gt_0, andone_gt_1;
    v8si andone_gt;
    v4si j_0, j_1;
    COPY_IMM_TO_XMM(j, j_0, j_1);

    //FT: 0 1 and not 1 0?
    andone_gt_0 = _mm_and_si128(j_0, *(v4si *) _pi32avx_1);
    andone_gt_1 = _mm_and_si128(j_1, *(v4si *) _pi32avx_1);
    COPY_XMM_TO_IMM(andone_gt_0, andone_gt_1, andone_gt);
    jandone = _mm256_cmpgt_epi32(andone_gt, _mm256_setzero_si256());
#else
    jandone = _mm256_cmpgt_epi32(_mm256_and_si256(j, *(v8si *) _pi32_256_1), _mm256_setzero_si256());
#endif

    y = _mm256_blendv_ps(y, _mm256_add_ps(y, *(v8sf *) _ps256_1), _mm256_cvtepi32_ps(jandone));
    j = _mm256_cvttps_epi32(y);  // no need to round again

    //z = ((x - y * DP1) - y * DP2) - y * DP3;

#if 1

    z = _mm256_fmadd_ps_custom(y, *(v8sf *) _ps256_DP1, x);
    z = _mm256_fmadd_ps_custom(y, *(v8sf *) _ps256_DP2, z);
    z = _mm256_fmadd_ps_custom(y, *(v8sf *) _ps256_DP3, z);
#else  // faster but less precision
    tmp = _mm256_mul_ps(y, *(v8sf *) _ps256_DP123);
    z = _mm256_sub_ps(x, tmp);
#endif
    zz = _mm256_mul_ps(z, z);  //z*z

    //TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_TAN_P0, zz, *(v8sf *) _ps256_TAN_P1);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P2);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P3);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P4);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P5);
    tmp = _mm256_mul_ps(zz, tmp);
    tmp = _mm256_fmadd_ps_custom(tmp, z, z);

    xsupem4 = _mm256_cmp_ps(x, _mm256_set1_ps(1.0e-4), _CMP_GT_OS);  //if( x > 1.0e-4 )
    y = _mm256_blendv_ps(z, tmp, xsupem4);

#ifndef __AVX2__
    v4si andtwo_gt_0, andtwo_gt_1;
    v8si andtwo_gt;
    COPY_IMM_TO_XMM(j, j_0, j_1);
    andtwo_gt_0 = _mm_and_si128(j_0, *(v4si *) _pi32avx_2);
    andtwo_gt_1 = _mm_and_si128(j_1, *(v4si *) _pi32avx_2);
    COPY_XMM_TO_IMM(andtwo_gt_0, andtwo_gt_1, andtwo_gt);
    jandtwo = _mm256_cmpgt_epi32(andtwo_gt, _mm256_setzero_si256());
#else
    jandtwo = _mm256_cmpgt_epi32(_mm256_and_si256(j, *(v8si *) _pi32_256_2), _mm256_setzero_si256());
#endif

    y = _mm256_blendv_ps(y, _mm256_div_ps(_mm256_set1_ps(-1.0f), y), _mm256_cvtepi32_ps(jandtwo));

    sign = _mm256_cmp_ps(xx, _mm256_setzero_ps(), _CMP_LT_OS);  //0xFFFFFFFF if xx < 0.0
    y = _mm256_blendv_ps(y, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, y), sign);

    return (y);
}

static inline void tan256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, tan256f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, tan256f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

#else

static inline void tan256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_div_ps(sin256_ps(src_tmp), cos256_ps(src_tmp)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_div_ps(sin256_ps(src_tmp), cos256_ps(src_tmp)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}
#endif

static inline void magnitude256f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (srcRe) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_load_ps(srcRe + i);
            v8sf re2 = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf im_tmp = _mm256_load_ps(srcIm + i);
            v8sf im2 = _mm256_mul_ps(im_tmp, im_tmp);
            _mm256_store_ps(dst + i, _mm256_sqrt_ps(_mm256_add_ps(re2, im2)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_loadu_ps(srcRe + i);
            v8sf re2 = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf im_tmp = _mm256_loadu_ps(srcIm + i);
            v8sf im2 = _mm256_mul_ps(im_tmp, im_tmp);
            _mm256_store_ps(dst + i, _mm256_sqrt_ps(_mm256_add_ps(re2, im2)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i]);
    }
}

static inline void powerspect256f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (srcRe) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_load_ps(srcRe + i);
            v8sf re2 = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf im_tmp = _mm256_load_ps(srcIm + i);
            v8sf im2 = _mm256_mul_ps(im_tmp, im_tmp);
            _mm256_store_ps(dst + i, _mm256_add_ps(re2, im2));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_loadu_ps(srcRe + i);
            v8sf re2 = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf im_tmp = _mm256_loadu_ps(srcIm + i);
            v8sf im2 = _mm256_mul_ps(im_tmp, im_tmp);
            _mm256_store_ps(dst + i, _mm256_add_ps(re2, im2));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i];
    }
}

/*static inline void magnitude256f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf cplx01 = _mm256_load_ps((const float *) src + j);
            v8sf cplx23 = _mm256_load_ps((const float *) src + j + AVX_LEN_FLOAT);  // complex is 2 floats
            v8sf cplx01_square = _mm256_mul_ps(cplx01, cplx01);
            v8sf cplx23_square = _mm256_mul_ps(cplx23, cplx23);
            v8sf square_sum_0123 = _mm256_hadd_ps(cplx23_square, cplx01_square);
            _mm256_store_ps(dst + i, _mm256_sqrt_ps(square_sum_0123));
            j += 2 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf cplx01 = _mm256_loadu_ps((const float *) src + j);
            v8sf cplx23 = _mm256_loadu_ps((const float *) src + j + AVX_LEN_FLOAT);  // complex is 2 floats
            v8sf cplx01_square = _mm256_mul_ps(cplx01, cplx01);
            v8sf cplx23_square = _mm256_mul_ps(cplx23, cplx23);
            v8sf square_sum_0123 = _mm256_hadd_ps(cplx01_square, cplx23_square);
            _mm256_storeu_ps(dst + i, _mm256_sqrt_ps(square_sum_0123));
            j += 2 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i].re * src[i].re + src[i].im * src[i].im);
    }
}*/


static inline void subcrev256f(float *src, float value, float *dst, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_sub_ps(tmp, _mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_sub_ps(tmp, _mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value - src[i];
    }
}

static inline void sum256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v8sf vec_acc1 = _mm256_setzero_ps();  //initialize the vector accumulator
    v8sf vec_acc2 = _mm256_setzero_ps();  //initialize the vector accumulator
    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf vec_tmp1 = _mm256_load_ps(src + i);
            vec_acc1 = _mm256_add_ps(vec_acc1, vec_tmp1);
            v8sf vec_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            vec_acc2 = _mm256_add_ps(vec_acc2, vec_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf vec_tmp1 = _mm256_loadu_ps(src + i);
            vec_acc1 = _mm256_add_ps(vec_acc1, vec_tmp1);
            v8sf vec_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            vec_acc2 = _mm256_add_ps(vec_acc2, vec_tmp2);
        }
    }

    vec_acc1 = _mm256_add_ps(vec_acc1, vec_acc2);
    _mm256_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];

    *dst = tmp_acc;
}


static inline void mean256f(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sum256f(src, dst, len);
    *dst *= coeff;
}

static inline void sqrt256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_sqrt_ps(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_sqrt_ps(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static inline void round256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}

static inline void trunc256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

static inline void cplxvecmul256f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX_LEN_FLOAT;   //stop_len << 2;

    int i;
    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_load_ps((float *) (src1) + i);                        // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_load_ps((float *) (src2) + i);                        // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);                                    //a1,a1,a0,a0
            v8sf tmp2 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            v8sf tmp3 = _mm256_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(tmp2, tmp3);
            out = _mm256_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm256_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_loadu_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_loadu_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);                                    //a1,a1,a0,a0
            v8sf tmp2 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            v8sf tmp3 = _mm256_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(tmp2, tmp3);
            out = _mm256_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm256_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + src2[i].re * src1[i].im;
    }
}

static inline void cplxvecmul256f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);
    stop_len = stop_len * AVX_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t)(src1Re), (uintptr_t)(src1Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t)(src2Re), (uintptr_t)(src2Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t)(dstRe), (uintptr_t)(dstIm), AVX_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_load_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_load_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_load_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_load_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            //v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            //v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_store_ps(dstRe + i, _mm256_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  //ac - bd
            _mm256_store_ps(dstIm + i, _mm256_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    } else {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_loadu_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_loadu_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_loadu_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_loadu_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            //v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            //v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_storeu_ps(dstRe + i, _mm256_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  //ac - bd
            _mm256_storeu_ps(dstIm + i, _mm256_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + src2Re[i] * src1Im[i];
    }
}

static inline void cplxconjvecmul256f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX_LEN_FLOAT;   //stop_len << 2;

    int i;
    const v8sf conj_mask = _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_load_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_load_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);                                   //a1,a1,a0,a0
            v8sf tmp2 = _mm256_mul_ps(tmp1, src2_tmp);                                  //a1d1,a1c1,a0d0,a0c0
            src2_tmp = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm256_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(src2_tmp, tmp1);
            out = _mm256_fmadd_ps_custom(conj_mask, tmp2, out);
            _mm256_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_loadu_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_loadu_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);               //a1,a1,a0,a0
            v8sf tmp2 = _mm256_mul_ps(tmp1, src2_tmp);              //a1d1,a1c1,a0d0,a0c0
            tmp2 = _mm256_mul_ps(tmp2, *(v8sf *) _ps256_min1);
            src2_tmp = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm256_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(src2_tmp, tmp1);
            out = _mm256_fmadd_ps_custom(conj_mask, tmp2, out);
            _mm256_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + src1[i].im * src2[i].im;
        dst[i].im = src2[i].re * src1[i].im - src1[i].re * src2[i].im;
    }
}

static inline void cplxconjvecmul256f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);
    stop_len = stop_len * AVX_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t)(src1Re), (uintptr_t)(src1Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t)(src2Re), (uintptr_t)(src2Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t)(dstRe), (uintptr_t)(dstIm), AVX_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_load_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_load_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_load_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_load_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            //v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            //v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_store_ps(dstRe + i, _mm256_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   //ac + bd
            _mm256_store_ps(dstIm + i, _mm256_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    } else {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_loadu_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_loadu_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_loadu_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_loadu_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            //v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            //v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_storeu_ps(dstRe + i, _mm256_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   //ac + bd
            _mm256_storeu_ps(dstIm + i, _mm256_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + src1Im[i] * src2Im[i];
        dstIm[i] = src2Re[i] * src1Im[i] - src1Re[i] * src2Im[i];
    }
}

//prefer using cplxconjvecmulXf if you also need to do a multiply
static inline void cplxconj256f(complex32_t *src, complex32_t *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX_LEN_FLOAT;   //stop_len << 2;

    const v8sf conj_mask = _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

    int i;
    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        //printf("Aligned\n");
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps((float *) (src) + i);
            _mm256_store_ps((float *) (dst) + i, _mm256_mul_ps(src_tmp, conj_mask));
        }
    } else {
        //printf("Unaligned\n");
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps((float *) (src) + i);
            _mm256_storeu_ps((float *) (dst) + i, _mm256_mul_ps(src_tmp, conj_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src[i].re;
        dst[i].im = -src[i].im;
    }
}

static inline void sigmoid256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf tmp = _mm256_add_ps(*(v8sf *) _ps256_1, exp256_ps_alternate(_mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, src_tmp)));
            _mm256_store_ps(dst + i, _mm256_rcp_ps(tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf tmp = _mm256_add_ps(*(v8sf *) _ps256_1, exp256_ps_alternate(_mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, src_tmp)));
            _mm256_storeu_ps(dst + i, _mm256_rcp_ps(tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}

static inline void PRelu256f(float *src, float *dst, float alpha, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    v8sf alpha_vec = _mm256_set1_ps(alpha);
    v8sf zero = _mm256_setzero_ps();

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf tmp = _mm256_mul_ps(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)

            // if x > 0
            _mm256_store_ps(dst + i, _mm256_blendv_ps(tmp, src_tmp, _mm256_cmp_ps(src_tmp, zero, _CMP_GT_OS)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf tmp = _mm256_mul_ps(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)

            // if x > 0
            _mm256_storeu_ps(dst + i, _mm256_blendv_ps(tmp, src_tmp, _mm256_cmp_ps(src_tmp, zero, _CMP_GT_OS)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] > 0.0f)
            dst[i] = src[i];
        else
            dst[i] = alpha * src[i];
    }
}

//to be improved
static inline void softmax256f(float *src, float *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);
    stop_len *= (AVX_LEN_FLOAT);

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f,
                                                                               0.0f, 0.0f, 0.0f, 0.0f};
    float acc = 0.0f;

    v8sf vec_acc1 = _mm256_setzero_ps();  //initialize the vector accumulator

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf dst_tmp = exp256_ps_alternate(src_tmp);
            vec_acc1 = _mm256_add_ps(vec_acc1, dst_tmp);
            _mm256_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf dst_tmp = exp256_ps_alternate(src_tmp);
            vec_acc1 = _mm256_add_ps(vec_acc1, dst_tmp);
            _mm256_storeu_ps(dst + i, dst_tmp);
        }
    }

    _mm256_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
        acc += dst[i];
    }

    acc = acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] +
          accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];
    vec_acc1 = _mm256_set1_ps(acc);

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf dst_tmp = _mm256_load_ps(dst + i);
            _mm256_store_ps(dst + i, _mm256_div_ps(dst_tmp, vec_acc1));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf dst_tmp = _mm256_loadu_ps(dst + i);
            _mm256_storeu_ps(dst + i, _mm256_div_ps(dst_tmp, vec_acc1));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] /= acc;
    }
}
