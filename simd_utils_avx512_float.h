/*
 * Project : SIMD_Utils
 * Version : 0.1.12
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#include "immintrin.h"

_PS512_CONST(min1, -1.0f);
_PS512_CONST(plus1, 1.0f);
_PS512_CONST(min2, -2.0f);
_PS512_CONST(min0p5, -0.5f);

// For tanf
_PS512_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS512_CONST(DP1, -0.78515625);
_PS512_CONST(DP2, -2.4187564849853515625e-4);
_PS512_CONST(DP3, -3.77489497744594108e-8);

_PS512_CONST(FOPI, 1.27323954473516); /* 4/pi */
_PS512_CONST(TAN_P0, 9.38540185543E-3);
_PS512_CONST(TAN_P1, 3.11992232697E-3);
_PS512_CONST(TAN_P2, 2.44301354525E-2);
_PS512_CONST(TAN_P3, 5.34112807005E-2);
_PS512_CONST(TAN_P4, 1.33387994085E-1);
_PS512_CONST(TAN_P5, 3.33331568548E-1);

_PS512_CONST(ASIN_P0, 4.2163199048E-2);
_PS512_CONST(ASIN_P1, 2.4181311049E-2);
_PS512_CONST(ASIN_P2, 4.5470025998E-2);
_PS512_CONST(ASIN_P3, 7.4953002686E-2);
_PS512_CONST(ASIN_P4, 1.6666752422E-1);

_PS512_CONST(PIF, 3.14159265358979323846);      // PI
_PS512_CONST(mPIF, -3.14159265358979323846);    // -PI
_PS512_CONST(PIO2F, 1.57079632679489661923);    // PI/2 1.570796326794896619
_PS512_CONST(mPIO2F, -1.57079632679489661923);  // -PI/2 1.570796326794896619
_PS512_CONST(PIO4F, 0.785398163397448309615);   // PI/4 0.7853981633974483096

_PS512_CONST(TANPI8F, 0.414213562373095048802);   // tan(pi/8) => 0.4142135623730950
_PS512_CONST(TAN3PI8F, 2.414213562373095048802);  // tan(3*pi/8) => 2.414213562373095

_PS512_CONST(ATAN_P0, 8.05374449538e-2);
_PS512_CONST(ATAN_P1, -1.38776856032E-1);
_PS512_CONST(ATAN_P2, 1.99777106478E-1);
_PS512_CONST(ATAN_P3, -3.33329491539E-1);

_PS512_CONST_TYPE(pos_sign_mask, int, (int) 0x7FFFFFFF);
_PS512_CONST_TYPE(neg_sign_mask, int, (int) ~0x7FFFFFFF);

_PS512_CONST(MAXLOGF, 88.72283905206835f);
_PS512_CONST(MAXLOGFDIV2, 44.361419526034176f);
_PS512_CONST(0p625, 0.625f);
_PS512_CONST(TANH_P0, -5.70498872745E-3f);
_PS512_CONST(TANH_P1, 2.06390887954E-2f);
_PS512_CONST(TANH_P2, -5.37397155531E-2f);
_PS512_CONST(TANH_P3, 1.33314422036E-1f);
_PS512_CONST(TANH_P4, -3.33332819422E-1f);

_PS512_CONST(MAXNUMF, 3.4028234663852885981170418348451692544e38f);
_PS512_CONST(minMAXNUMF, -3.4028234663852885981170418348451692544e38f);
_PS512_CONST(SINH_P0, 2.03721912945E-4f);
_PS512_CONST(SINH_P1, 8.33028376239E-3f);
_PS512_CONST(SINH_P2, 1.66667160211E-1f);

_PS512_CONST(1emin4, 1e-4f);
_PS512_CONST(ATANH_P0, 1.81740078349E-1f);
_PS512_CONST(ATANH_P1, 8.24370301058E-2f);
_PS512_CONST(ATANH_P2, 1.46691431730E-1f);
_PS512_CONST(ATANH_P3, 1.99782164500E-1f);
_PS512_CONST(ATANH_P4, 3.33337300303E-1f);

_PS512_CONST(1500, 1500.0f);
_PS512_CONST(LOGE2F, 0.693147180559945309f);
_PS512_CONST(ASINH_P0, 2.0122003309E-2f);
_PS512_CONST(ASINH_P1, -4.2699340972E-2f);
_PS512_CONST(ASINH_P2, 7.4847586088E-2f);
_PS512_CONST(ASINH_P3, -1.6666288134E-1f);

_PS512_CONST(ACOSH_P0, 1.7596881071E-3f);
_PS512_CONST(ACOSH_P1, -7.5272886713E-3f);
_PS512_CONST(ACOSH_P2, 2.6454905019E-2f);
_PS512_CONST(ACOSH_P3, -1.1784741703E-1f);
_PS512_CONST(ACOSH_P4, 1.4142135263E0f);

static inline void log10_512f(float *src, float *dst, int len)
{
    const v16sf invln10f = _mm512_set1_ps((float) INVLN10);  //_mm512_broadcast_ss(&invln10f_mask);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log512_ps(_mm512_load_ps(src + i));
            _mm512_store_ps(dst + i, _mm512_mul_ps(src_tmp, invln10f));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log512_ps(_mm512_loadu_ps(src + i));
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(src_tmp, invln10f));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void log2_512f(float *src, float *dst, int len)
{
    const v16sf invln2f = _mm512_set1_ps((float) INVLN2);  //_mm512_broadcast_ss(&invln10f_mask);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log512_ps(_mm512_load_ps(src + i));
            _mm512_store_ps(dst + i, _mm512_mul_ps(src_tmp, invln2f));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log512_ps(_mm512_loadu_ps(src + i));
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(src_tmp, invln2f));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void ln_512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, log512_ps(_mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, log512_ps(_mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void exp_512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, exp512_ps(_mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, exp512_ps(_mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

static inline void fabs512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void set512f(float *src, float value, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = value;
    }
}

static inline void zero512f(float *src, int len)
{
    const v16sf tmp = _mm512_setzero_ps();

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = 0.0f;
    }
}


static inline void copy512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_load_ps(src + i));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_add_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_add_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}


static inline void mul512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_mul_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_sub_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_sub_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}


static inline void addc512f(float *src, float value, float *dst, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_add_ps(tmp, _mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_add_ps(tmp, _mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc512f(float *src, float value, float *dst, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_mul_ps(tmp, _mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(tmp, _mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd512f(float *_a, float *_b, float *_c, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(_b), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t)(_c), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(_a + i);
            v16sf b = _mm512_load_ps(_b + i);
            v16sf c = _mm512_load_ps(_c + i);
            _mm512_store_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(_a + i);
            v16sf b = _mm512_loadu_ps(_b + i);
            v16sf c = _mm512_loadu_ps(_c + i);
            _mm512_storeu_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcadd512f(float *_a, float _b, float *_c, float *dst, int len)
{
    v16sf b = _mm512_set1_ps(_b);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_c), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(_a + i);
            v16sf c = _mm512_load_ps(_c + i);
            _mm512_store_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(_a + i);
            v16sf c = _mm512_loadu_ps(_c + i);
            _mm512_storeu_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddc512f(float *_a, float _b, float _c, float *dst, int len)
{
    v16sf b = _mm512_set1_ps(_b);
    v16sf c = _mm512_set1_ps(_c);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(_a + i);
            _mm512_store_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(_a + i);
            _mm512_storeu_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
    }
}

static inline void muladdc512f(float *_a, float *_b, float _c, float *dst, int len)
{
    v16sf c = _mm512_set1_ps(_c);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_b), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(_a + i);
            v16sf b = _mm512_load_ps(_b + i);
            _mm512_store_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(_a + i);
            v16sf b = _mm512_loadu_ps(_b + i);
            _mm512_storeu_ps(dst + i, _mm512_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

static inline void div512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_div_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_div_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

static inline void vectorSlope512f(float *dst, int len, float offset, float slope)
{
    v16sf coef = _mm512_set_ps(15.0f * slope, 14.0f * slope, 13.0f * slope, 12.0f * slope,
                               11.0f * slope, 10.0f * slope, 9.0f * slope, 8.0f * slope,
                               7.0f * slope, 6.0f * slope, 5.0f * slope, 4.0f * slope,
                               3.0f * slope, 2.0f * slope, slope, 0.0f);
    v16sf slope16_vec = _mm512_set1_ps(16.0f * slope);
    v16sf curVal = _mm512_add_ps(_mm512_set1_ps(offset), coef);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (dst) % AVX512_LEN_BYTES) == 0) {
        _mm512_store_ps(dst + 0, curVal);
    } else {
        _mm512_storeu_ps(dst + 0, curVal);
    }

    if (((uintptr_t)(const void *) (dst) % AVX512_LEN_BYTES) == 0) {
        for (int i = AVX512_LEN_FLOAT; i < stop_len; i += AVX512_LEN_FLOAT) {
            curVal = _mm512_add_ps(curVal, slope16_vec);
            _mm512_storeu_ps(dst + i, curVal);
        }
    } else {
        for (int i = AVX512_LEN_FLOAT; i < stop_len; i += AVX512_LEN_FLOAT) {
            curVal = _mm512_add_ps(curVal, slope16_vec);
            _mm512_storeu_ps(dst + i, curVal);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

static inline void convert512_32f64f(float *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            __m256 src_tmp = _mm256_load_ps(src + i);            //load a,b,c,d
            _mm512_store_pd(dst + i, _mm512_cvtps_pd(src_tmp));  //store the abcd converted in 64bits
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            __m256 src_tmp = _mm256_loadu_ps(src + i);            //load a,b,c,d
            _mm512_storeu_pd(dst + i, _mm512_cvtps_pd(src_tmp));  //store the c and d converted in 64bits
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (double) src[i];
    }
}


static inline void maxevery512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_max_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_max_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_min_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_min_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void minmax512f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    v16sf max_v, min_v;
    v16sf src_tmp;
    float min_f[AVX512_LEN_FLOAT] __attribute__((aligned(AVX512_LEN_BYTES)));
    float max_f[AVX512_LEN_FLOAT] __attribute__((aligned(AVX512_LEN_BYTES)));
    float min_tmp;
    float max_tmp;

    if (isAligned((uintptr_t)(src), AVX512_LEN_BYTES)) {
        src_tmp = _mm512_load_ps(src + 0);
        max_v = src_tmp;
        min_v = src_tmp;
        for (int i = AVX512_LEN_FLOAT; i < stop_len; i += AVX512_LEN_FLOAT) {
            src_tmp = _mm512_load_ps(src + i);
            max_v = _mm512_max_ps(max_v, src_tmp);
            min_v = _mm512_min_ps(min_v, src_tmp);
        }
    } else {
        src_tmp = _mm512_loadu_ps(src + 0);
        max_v = src_tmp;
        min_v = src_tmp;
        for (int i = AVX512_LEN_FLOAT; i < stop_len; i += AVX512_LEN_FLOAT) {
            src_tmp = _mm512_loadu_ps(src + i);
            max_v = _mm512_max_ps(max_v, src_tmp);
            min_v = _mm512_min_ps(min_v, src_tmp);
        }
    }

    _mm512_store_ps(max_f, max_v);
    _mm512_store_ps(min_f, min_v);

    max_tmp = max_f[0];
    max_tmp = max_tmp > max_f[1] ? max_tmp : max_f[1];
    max_tmp = max_tmp > max_f[2] ? max_tmp : max_f[2];
    max_tmp = max_tmp > max_f[3] ? max_tmp : max_f[3];
    max_tmp = max_tmp > max_f[4] ? max_tmp : max_f[4];
    max_tmp = max_tmp > max_f[5] ? max_tmp : max_f[5];
    max_tmp = max_tmp > max_f[6] ? max_tmp : max_f[6];
    max_tmp = max_tmp > max_f[7] ? max_tmp : max_f[7];
    max_tmp = max_tmp > max_f[8] ? max_tmp : max_f[8];
    max_tmp = max_tmp > max_f[9] ? max_tmp : max_f[9];
    max_tmp = max_tmp > max_f[10] ? max_tmp : max_f[10];
    max_tmp = max_tmp > max_f[11] ? max_tmp : max_f[11];
    max_tmp = max_tmp > max_f[12] ? max_tmp : max_f[12];
    max_tmp = max_tmp > max_f[13] ? max_tmp : max_f[13];
    max_tmp = max_tmp > max_f[14] ? max_tmp : max_f[14];
    max_tmp = max_tmp > max_f[15] ? max_tmp : max_f[15];

    min_tmp = min_f[0];
    min_tmp = min_tmp < min_f[1] ? min_tmp : min_f[1];
    min_tmp = min_tmp < min_f[2] ? min_tmp : min_f[2];
    min_tmp = min_tmp < min_f[3] ? min_tmp : min_f[3];
    min_tmp = min_tmp < min_f[4] ? min_tmp : min_f[4];
    min_tmp = min_tmp < min_f[5] ? min_tmp : min_f[5];
    min_tmp = min_tmp < min_f[6] ? min_tmp : min_f[6];
    min_tmp = min_tmp < min_f[7] ? min_tmp : min_f[7];
    min_tmp = min_tmp < min_f[8] ? min_tmp : min_f[8];
    min_tmp = min_tmp < min_f[9] ? min_tmp : min_f[9];
    min_tmp = min_tmp < min_f[10] ? min_tmp : min_f[10];
    min_tmp = min_tmp < min_f[11] ? min_tmp : min_f[11];
    min_tmp = min_tmp < min_f[12] ? min_tmp : min_f[12];
    min_tmp = min_tmp < min_f[13] ? min_tmp : min_f[13];
    min_tmp = min_tmp < min_f[14] ? min_tmp : min_f[14];
    min_tmp = min_tmp < min_f[15] ? min_tmp : min_f[15];

    for (int i = stop_len; i < len; i++) {
        max_tmp = max_tmp > src[i] ? max_tmp : src[i];
        min_tmp = min_tmp < src[i] ? min_tmp : src[i];
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

static inline void threshold512_gt_f(float *src, float *dst, int len, float value)
{
    v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_min_ps(src_tmp, tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_min_ps(src_tmp, tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold512_gtabs_f(float *src, float *dst, int len, float value)
{
    const v16sf pval = _mm512_set1_ps(value);
    const v16sf mval = _mm512_set1_ps(-value);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 gtmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_GT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);                //if A >= 0 value, else -value
            v16sf dst_tmp = _mm512_mask_blend_ps(gtmask, src_tmp, sval);          // either A or sval (+- value)
            _mm512_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 gtmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_GT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);                //if A >= 0 value, else -value
            v16sf dst_tmp = _mm512_mask_blend_ps(gtmask, src_tmp, sval);          // either A or sval (+- value)
            _mm512_storeu_ps(dst + i, dst_tmp);
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

static inline void threshold512_lt_f(float *src, float *dst, int len, float value)
{
    v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_max_ps(src_tmp, tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_max_ps(src_tmp, tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold512_ltabs_f(float *src, float *dst, int len, float value)
{
    const v16sf pval = _mm512_set1_ps(value);
    const v16sf mval = _mm512_set1_ps(-value);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 gtmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_LT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);                //if A >= 0 value, else -value
            v16sf dst_tmp = _mm512_mask_blend_ps(gtmask, src_tmp, sval);          // either A or sval (+- value)
            _mm512_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 gtmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_LT_OS);     //if abs(A) < value => 0xFFFFFFFF, else 0
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);                //if A >= 0 value, else -value
            v16sf dst_tmp = _mm512_mask_blend_ps(gtmask, src_tmp, sval);          // either A or sval (+- value)
            _mm512_storeu_ps(dst + i, dst_tmp);
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

static inline void threshold512_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v16sf ltlevel_v = _mm512_set1_ps(ltlevel);
    const v16sf ltvalue_v = _mm512_set1_ps(ltvalue);
    const v16sf gtlevel_v = _mm512_set1_ps(gtlevel);
    const v16sf gtvalue_v = _mm512_set1_ps(gtvalue);

    int stop_len = len / AVX512_LEN_BYTES;
    stop_len *= AVX512_LEN_BYTES;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            __mmask16 lt_mask = _mm512_cmp_ps_mask(src_tmp, ltlevel_v, _CMP_LT_OS);
            __mmask16 gt_mask = _mm512_cmp_ps_mask(src_tmp, gtlevel_v, _CMP_GT_OS);
            v16sf dst_tmp = _mm512_mask_blend_ps(lt_mask, src_tmp, ltvalue_v);
            dst_tmp = _mm512_mask_blend_ps(gt_mask, dst_tmp, gtvalue_v);
            _mm512_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            __mmask16 lt_mask = _mm512_cmp_ps_mask(src_tmp, ltlevel_v, _CMP_LT_OS);
            __mmask16 gt_mask = _mm512_cmp_ps_mask(src_tmp, gtlevel_v, _CMP_GT_OS);
            v16sf dst_tmp = _mm512_mask_blend_ps(lt_mask, src_tmp, ltvalue_v);
            dst_tmp = _mm512_mask_blend_ps(gt_mask, dst_tmp, gtvalue_v);
            _mm512_storeu_ps(dst + i, dst_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}

static inline void sin512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, sin512_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, sin512_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, cos512_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, cos512_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos512f(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf dst_sin_tmp;
            v16sf dst_cos_tmp;
            sincos512_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm512_store_ps(dst_sin + i, dst_sin_tmp);
            _mm512_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf dst_sin_tmp;
            v16sf dst_cos_tmp;
            sincos512_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm512_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm512_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline v16sf acosh512f_ps(v16sf x)
{
    v16sf z, z_first_branch, z_second_branch;
    __mmask16 xsup1500, zinf0p5, xinf1;

    xsup1500 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1500, _CMP_GT_OS);  // return  (logf(x) + LOGE2F)
    xinf1 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1, _CMP_LT_OS);        // return 0

    z = _mm512_sub_ps(x, *(v16sf *) _ps512_1);

    zinf0p5 = _mm512_cmp_ps_mask(z, *(v16sf *) _ps512_0p5, _CMP_LT_OS);  // first and second branch

    //First Branch (z < 0.5)
    z_first_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ACOSH_P0, z, *(v16sf *) _ps512_ACOSH_P1);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, z, *(v16sf *) _ps512_ACOSH_P2);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, z, *(v16sf *) _ps512_ACOSH_P3);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, z, *(v16sf *) _ps512_ACOSH_P4);
    z_first_branch = _mm512_mul_ps(z_first_branch, _mm512_sqrt_ps(z));

    //Second Branch
    z_second_branch = _mm512_sqrt_ps(_mm512_fmadd_ps_custom(z, x, z));
    z_second_branch = log512_ps(_mm512_add_ps(x, z_second_branch));

    z = _mm512_mask_blend_ps(zinf0p5, z_second_branch, z_first_branch);
    z = _mm512_mask_blend_ps(xsup1500, z, _mm512_add_ps(log512_ps(x), *(v16sf *) _ps512_LOGE2F));
    z = _mm512_mask_blend_ps(xinf1, z, _mm512_setzero_ps());

    return z;
}

static inline void acosh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, acosh512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, acosh512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline v16sf asinh512f_ps(v16sf xx)
{
    v16sf x, tmp, z, z_first_branch, z_second_branch;
    __mmask16 xxinf0, xsup1500, xinf0p5;

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);
    xsup1500 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1500, _CMP_GT_OS);
    xinf0p5 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_0p5, _CMP_LT_OS);
    xxinf0 = _mm512_cmp_ps_mask(xx, _mm512_setzero_ps(), _CMP_LT_OS);

    tmp = _mm512_mul_ps(x, x);
    //First Branch (x < 0.5)
    z_first_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ASINH_P0, tmp, *(v16sf *) _ps512_ASINH_P1);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ASINH_P2);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ASINH_P3);
    z_first_branch = _mm512_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, x, x);

    //Second Branch
    z_second_branch = _mm512_sqrt_ps(_mm512_add_ps(tmp, *(v16sf *) _ps512_1));
    z_second_branch = log512_ps(_mm512_add_ps(z_second_branch, x));

    z = _mm512_mask_blend_ps(xinf0p5, z_second_branch, z_first_branch);
    z = _mm512_mask_blend_ps(xsup1500, z, _mm512_add_ps(log512_ps(x), *(v16sf *) _ps512_LOGE2F));
    z = _mm512_mask_blend_ps(xxinf0, z, _mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, z));

    return z;
}

static inline void asinh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, asinh512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, asinh512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline v16sf atanh512f_ps(v16sf x)
{
    v16sf z, tmp, z_first_branch, z_second_branch;
    __mmask16 xsup1, xinfmin1, zinf1emin4, zinf0p5;

    z = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, x);

    xsup1 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1, _CMP_GE_OS);
    xinfmin1 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_min1, _CMP_LE_OS);
    zinf1emin4 = _mm512_cmp_ps_mask(z, *(v16sf *) _ps512_1emin4, _CMP_LT_OS);
    zinf0p5 = _mm512_cmp_ps_mask(z, *(v16sf *) _ps512_0p5, _CMP_LT_OS);

    //First branch
    tmp = _mm512_mul_ps(x, x);
    z_first_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ATANH_P0, tmp, *(v16sf *) _ps512_ATANH_P1);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ATANH_P2);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ATANH_P3);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ATANH_P4);
    z_first_branch = _mm512_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, x, x);

    //Second branch
    tmp = _mm512_sub_ps(*(v16sf *) _ps512_1, x);
    tmp = _mm512_div_ps(_mm512_add_ps(*(v16sf *) _ps512_1, x), tmp);
    z_second_branch = log512_ps(tmp);
    z_second_branch = _mm512_mul_ps(*(v16sf *) _ps512_0p5, z_second_branch);

    z = _mm512_mask_blend_ps(zinf0p5, z_second_branch, z_first_branch);
    z = _mm512_mask_blend_ps(zinf1emin4, z, x);
    z = _mm512_mask_blend_ps(xsup1, z, *(v16sf *) _ps512_MAXNUMF);
    z = _mm512_mask_blend_ps(xinfmin1, z, *(v16sf *) _ps512_minMAXNUMF);

    return (z);
}

static inline void atanh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, atanh512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, atanh512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}


static inline v16sf cosh512f_ps(v16sf xx)
{
    v16sf x, y, tmp;
    __mmask16 xsupmaxlogf;

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);
    xsupmaxlogf = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_MAXLOGF, _CMP_GT_OS);

    y = exp512_ps(x);
    tmp = _mm512_div_ps(*(v16sf *) _ps512_0p5, y);
    y = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_0p5, y, tmp);
    y = _mm512_mask_blend_ps(xsupmaxlogf, y, *(v16sf *) _ps512_MAXNUMF);

    return y;
}

static inline void cosh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, cosh512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, cosh512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline v16sf sinh512f_ps(v16sf x)
{
    v16sf z, z_first_branch, z_second_branch, tmp;
    __mmask16 xsupmaxlogf, zsup1, xinf0;

    // x = xx; if x < 0, z = -x, else x
    z = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, x);

    xsupmaxlogf = _mm512_cmp_ps_mask(z, *(v16sf *) _ps512_MAXLOGF, _CMP_GT_OS);

    // First branch
    zsup1 = _mm512_cmp_ps_mask(z, *(v16sf *) _ps512_1, _CMP_GT_OS);
    xinf0 = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OS);
    z_first_branch = exp512_ps(z);
    tmp = _mm512_div_ps(*(v16sf *) _ps512_min0p5, z_first_branch);
    z_first_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_0p5, z_first_branch, tmp);

    z_first_branch = _mm512_mask_blend_ps(xinf0, z_first_branch, _mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, z_first_branch));

    // Second branch
    tmp = _mm512_mul_ps(x, x);
    z_second_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_SINH_P0, tmp, *(v16sf *) _ps512_SINH_P1);
    z_second_branch = _mm512_fmadd_ps_custom(z_second_branch, tmp, *(v16sf *) _ps512_SINH_P2);
    z_second_branch = _mm512_mul_ps(z_second_branch, tmp);
    z_second_branch = _mm512_fmadd_ps_custom(z_second_branch, x, x);

    // Choose between first and second branch
    z = _mm512_mask_blend_ps(zsup1, z_second_branch, z_first_branch);

    // Set value to MAXNUMF if abs(x) > MAGLOGF
    // Set value to -MAXNUMF if abs(x) > MAGLOGF and x < 0
    z = _mm512_mask_blend_ps(xsupmaxlogf, z, *(v16sf *) _ps512_MAXNUMF);
    z = _mm512_mask_blend_ps(_kand_mask64(xinf0, xsupmaxlogf), z, *(v16sf *) _ps512_minMAXNUMF);

    return (z);
}

static inline void sinh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, sinh512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, sinh512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}

static inline v16sf atan512f_ps(v16sf xx)
{
    v16sf x, y, z;
    __mmask16 sign2;
    __mmask16 suptan3pi8, inftan3pi8suppi8;
    v16sf tmp;

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);
    sign2 = _mm512_cmp_ps_mask(xx, _mm512_setzero_ps(), _CMP_LT_OS);  //0xFFFFFFFF if x < 0.0, sign = -1
    /* range reduction */

    y = _mm512_setzero_ps();
    suptan3pi8 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_TAN3PI8F, _CMP_GT_OS);  // if( x > tan 3pi/8 )
    x = _mm512_mask_blend_ps(suptan3pi8, x, _mm512_div_ps(*(v16sf *) _ps512_min1, x));
    y = _mm512_mask_blend_ps(suptan3pi8, y, *(v16sf *) _ps512_PIO2F);


    inftan3pi8suppi8 = _kand_mask64(_mm512_cmp_ps_mask(x, *(v16sf *) _ps512_TAN3PI8F, _CMP_LT_OS), _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_TANPI8F, _CMP_GT_OS));  // if( x > tan 3pi/8 )
    x = _mm512_mask_blend_ps(inftan3pi8suppi8, x, _mm512_div_ps(_mm512_sub_ps(x, *(v16sf *) _ps512_1), _mm512_add_ps(x, *(v16sf *) _ps512_1)));
    y = _mm512_mask_blend_ps(inftan3pi8suppi8, y, *(v16sf *) _ps512_PIO4F);

    z = _mm512_mul_ps(x, x);
    tmp = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ATAN_P0, z, *(v16sf *) _ps512_ATAN_P1);
    tmp = _mm512_fmadd_ps_custom(tmp, z, *(v16sf *) _ps512_ATAN_P2);
    tmp = _mm512_fmadd_ps_custom(tmp, z, *(v16sf *) _ps512_ATAN_P3);
    tmp = _mm512_mul_ps(z, tmp);
    tmp = _mm512_fmadd_ps_custom(tmp, x, x);

    y = _mm512_add_ps(y, tmp);

    y = _mm512_mask_blend_ps(sign2, y, _mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, y));

    return (y);
}

static inline void atan512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, atan512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, atan512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline v16sf atan2512f_ps(v16sf y, v16sf x)
{
    v16sf z, w;
    __mmask16 xinfzero, yinfzero, xeqzero, yeqzero;
    __mmask16 xeqzeroandyinfzero, yeqzeroandxinfzero;
    __mmask16 specialcase;

    xinfzero = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OS);  // code =2
    yinfzero = _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OS);  // code = code |1;

    xeqzero = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_OS);
    yeqzero = _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_EQ_OS);

    z = *(v16sf *) _ps512_PIO2F;

    xeqzeroandyinfzero = _kand_mask16(xeqzero, yinfzero);
    z = _mm512_mask_blend_ps(xeqzeroandyinfzero, z, *(v16sf *) _ps512_mPIO2F);
    z = _mm512_mask_blend_ps(yeqzero, z, _mm512_setzero_ps());

    yeqzeroandxinfzero = _kand_mask16(yeqzero, xinfzero);
    z = _mm512_mask_blend_ps(yeqzeroandxinfzero, z, *(v16sf *) _ps512_PIF);

    specialcase = _kor_mask16(xeqzero, yeqzero);

    w = _mm512_setzero_ps();
    w = _mm512_mask_blend_ps(_kandn_mask16(yinfzero, xinfzero), w, *(v16sf *) _ps512_PIF);  // y >= 0 && x<0
    w = _mm512_mask_blend_ps(_kand_mask16(yinfzero, xinfzero), w, *(v16sf *) _ps512_mPIF);  // y < 0 && x<0

    z = _mm512_mask_blend_ps(specialcase, _mm512_add_ps(w, atan512f_ps(_mm512_div_ps(y, x))), z);  // atanf(y/x) if not in special case

    return (z);
}

static inline void atan2512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src1) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, atan2512f_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, atan2512f_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}

static inline v16sf asin512f_ps(v16sf xx)
{
    v16sf a, x, z, z_tmp;
    __mmask16 sign;
    __mmask16 ainfem4, asup0p5;
    v16sf tmp;
    x = xx;
    a = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, x);          //fabs(x)
    sign = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OS);  //0xFFFFFFFF if x < 0.0

    //TODO : vectorize this
    /*if( a > 1.0f )
	{
		return( 0.0f );
	}*/


    ainfem4 = _mm512_cmp_ps_mask(a, _mm512_set1_ps(1.0e-4), _CMP_LT_OS);  //if( a < 1.0e-4f )

    asup0p5 = _mm512_cmp_ps_mask(a, *(v16sf *) _ps512_0p5, _CMP_GT_OS);  //if( a > 0.5f ) flag = 1 else 0
    z_tmp = _mm512_sub_ps(*(v16sf *) _ps512_1, a);
    z_tmp = _mm512_mul_ps(*(v16sf *) _ps512_0p5, z_tmp);
    z = _mm512_mask_blend_ps(asup0p5, _mm512_mul_ps(a, a), z_tmp);
    x = _mm512_mask_blend_ps(asup0p5, a, _mm512_sqrt_ps(z));

    tmp = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ASIN_P0, z, *(v16sf *) _ps512_ASIN_P1);
    tmp = _mm512_fmadd_ps_custom(z, tmp, *(v16sf *) _ps512_ASIN_P2);
    tmp = _mm512_fmadd_ps_custom(z, tmp, *(v16sf *) _ps512_ASIN_P3);
    tmp = _mm512_fmadd_ps_custom(z, tmp, *(v16sf *) _ps512_ASIN_P4);
    tmp = _mm512_mul_ps(z, tmp);
    tmp = _mm512_fmadd_ps_custom(x, tmp, x);

    z = tmp;

    z_tmp = _mm512_add_ps(z, z);
    z_tmp = _mm512_sub_ps(*(v16sf *) _ps512_PIO2F, z_tmp);
    z = _mm512_mask_blend_ps(asup0p5, z, z_tmp);

    //done:
    z = _mm512_mask_blend_ps(ainfem4, z, a);
    z = _mm512_mask_blend_ps(sign, z, _mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, z));

    return (z);
}

static inline void asin512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, asin512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, asin512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}


static inline v16sf tanh512f_ps(v16sf xx)
{
    v16sf x, z, z_first_branch, z_second_branch;
    __mmask16 xxsup0, xsupmaxlogfdiv2, xsup0p625;

    xxsup0 = _mm512_cmp_ps_mask(xx, _mm512_setzero_ps(), _CMP_GT_OS);
    xsupmaxlogfdiv2 = _mm512_cmp_ps_mask(xx, *(v16sf *) _ps512_MAXLOGFDIV2, _CMP_GT_OS);

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);

    xsup0p625 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_0p625, _CMP_GE_OS);
    x = _mm512_mask_blend_ps(xsup0p625, x, exp512_ps(_mm512_add_ps(x, x)));

    // z = 1.0 - 2.0 / (x + 1.0);
    z_first_branch = _mm512_add_ps(x, *(v16sf *) _ps512_1);
    z_first_branch = _mm512_div_ps(*(v16sf *) _ps512_min2, z_first_branch);
    z_first_branch = _mm512_add_ps(*(v16sf *) _ps512_1, z_first_branch);
    z_first_branch = _mm512_mask_blend_ps(xxsup0, _mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, z_first_branch), z_first_branch);

    //z = x * x;
    z = _mm512_mul_ps(x, x);

    z_second_branch = _mm512_fmadd_ps_custom(z, *(v16sf *) _ps512_TANH_P0, *(v16sf *) _ps512_TANH_P1);
    z_second_branch = _mm512_fmadd_ps_custom(z_second_branch, z, *(v16sf *) _ps512_TANH_P2);
    z_second_branch = _mm512_fmadd_ps_custom(z_second_branch, z, *(v16sf *) _ps512_TANH_P3);
    z_second_branch = _mm512_fmadd_ps_custom(z_second_branch, z, *(v16sf *) _ps512_TANH_P4);
    z_second_branch = _mm512_mul_ps(z_second_branch, z);
    z_second_branch = _mm512_fmadd_ps_custom(z_second_branch, xx, xx);

    z = _mm512_mask_blend_ps(xsup0p625, z_second_branch, z_first_branch);
    // if (x > 0.5 * MAXLOGF), return (xx > 0)? 1.0f: -1.0f
    z = _mm512_mask_blend_ps(xsupmaxlogfdiv2, z, *(v16sf *) _ps512_min1);
    z = _mm512_mask_blend_ps(_kand_mask64(xxsup0, xsupmaxlogfdiv2), z, *(v16sf *) _ps512_1);

    return (z);
}

static inline void tanh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, tanh512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, tanh512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}


#if 1
static inline v16sf tan512f_ps(v16sf xx)
{
    v16sf x, y, z, zz;
    v16si j;  //long?
    __mmask16 sign, xsupem4;
    v16sf tmp;
    __mmask16 jandone, jandtwo;

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);  //fabs(xx)

    /* compute x mod PIO4 */

    //TODO : on neg values should be ceil and not floor
    //j = _mm512_cvtps_epi32( _mm512_roundscale_ps(_mm512_mul_ps(*(v16sf*)_ps512_FOPI,x), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC )); /* integer part of x/(PI/4), using floor */
    j = _mm512_cvttps_epi32(_mm512_mul_ps(*(v16sf *) _ps512_FOPI, x));
    y = _mm512_cvtepi32_ps(j);

    jandone = _mm512_cmpgt_epi32_mask(_mm512_and_si512(j, *(v16si *) _pi32_512_1), _mm512_setzero_si512());
    y = _mm512_mask_blend_ps(jandone, y, _mm512_add_ps(y, *(v16sf *) _ps512_1));
    j = _mm512_cvttps_epi32(y);  // no need to round again

    //z = ((x - y * DP1) - y * DP2) - y * DP3;
    z = _mm512_fmadd_ps_custom(y, *(v16sf *) _ps512_DP1, x);
    z = _mm512_fmadd_ps_custom(y, *(v16sf *) _ps512_DP2, z);
    z = _mm512_fmadd_ps_custom(y, *(v16sf *) _ps512_DP3, z);
    zz = _mm512_mul_ps(z, z);  //z*z

    //TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_TAN_P0, zz, *(v16sf *) _ps512_TAN_P1);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P2);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P3);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P4);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P5);
    tmp = _mm512_mul_ps(zz, tmp);
    tmp = _mm512_fmadd_ps_custom(tmp, z, z);

    xsupem4 = _mm512_cmp_ps_mask(x, _mm512_set1_ps(1.0e-4), _CMP_GT_OS);  //if( x > 1.0e-4 )
    y = _mm512_mask_blend_ps(xsupem4, z, tmp);

    jandtwo = _mm512_cmpgt_epi32_mask(_mm512_and_si512(j, *(v16si *) _pi32_512_2), _mm512_setzero_si512());

    y = _mm512_mask_blend_ps(jandtwo, y, _mm512_div_ps(_mm512_set1_ps(-1.0f), y));

    sign = _mm512_cmp_ps_mask(xx, _mm512_setzero_ps(), _CMP_LT_OS);  //0xFFFFFFFF if xx < 0.0
    y = _mm512_mask_blend_ps(sign, y, _mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, y));

    return (y);
}

static inline void tan512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, tan512f_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, tan512f_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

#else

static inline void tan512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_div_ps(sin512_ps(src_tmp), cos512_ps(src_tmp)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_div_ps(sin512_ps(src_tmp), cos512_ps(src_tmp)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}
#endif

static inline void magnitude512f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (srcRe) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_load_ps(srcRe + i);
            v16sf re2 = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf im_tmp = _mm512_load_ps(srcIm + i);
            v16sf im2 = _mm512_mul_ps(im_tmp, im_tmp);
            _mm512_store_ps(dst + i, _mm512_sqrt_ps(_mm512_add_ps(re2, im2)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_loadu_ps(srcRe + i);
            v16sf re2 = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf im_tmp = _mm512_loadu_ps(srcIm + i);
            v16sf im2 = _mm512_mul_ps(im_tmp, im_tmp);
            _mm512_store_ps(dst + i, _mm512_sqrt_ps(_mm512_add_ps(re2, im2)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i]);
    }
}

static inline void powerspect512f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (srcRe) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_load_ps(srcRe + i);
            v16sf re2 = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf im_tmp = _mm512_load_ps(srcIm + i);
            v16sf im2 = _mm512_mul_ps(im_tmp, im_tmp);
            _mm512_store_ps(dst + i, _mm512_add_ps(re2, im2));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_loadu_ps(srcRe + i);
            v16sf re2 = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf im_tmp = _mm512_loadu_ps(srcIm + i);
            v16sf im2 = _mm512_mul_ps(im_tmp, im_tmp);
            _mm512_store_ps(dst + i, _mm512_add_ps(re2, im2));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i];
    }
}

static inline void subcrev512f(float *src, float value, float *dst, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_sub_ps(tmp, _mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_sub_ps(tmp, _mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value - src[i];
    }
}

static inline void sum512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    __attribute__((aligned(AVX512_LEN_BYTES))) float accumulate[AVX512_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v16sf vec_acc = _mm512_setzero_ps();  //initialize the vector accumulator
    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf vec_tmp = _mm512_load_ps(src + i);
            vec_acc = _mm512_add_ps(vec_acc, vec_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf vec_tmp = _mm512_loadu_ps(src + i);
            vec_acc = _mm512_add_ps(vec_acc, vec_tmp);
        }
    }

    _mm512_store_ps(accumulate, vec_acc);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7] + accumulate[8] + accumulate[9] + accumulate[10] + accumulate[11] + accumulate[12] + accumulate[13] + accumulate[14] + accumulate[15];

    *dst = tmp_acc;
}


static inline void mean512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    __attribute__((aligned(AVX512_LEN_BYTES))) float accumulate[AVX512_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v16sf vec_acc = _mm512_setzero_ps();  //initialize the vector accumulator
    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf vec_tmp = _mm512_load_ps(src + i);
            vec_acc = _mm512_add_ps(vec_acc, vec_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf vec_tmp = _mm512_loadu_ps(src + i);
            vec_acc = _mm512_add_ps(vec_acc, vec_tmp);
        }
    }

    _mm512_store_ps(accumulate, vec_acc);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7] + accumulate[8] + accumulate[9] + accumulate[10] + accumulate[11] + accumulate[12] + accumulate[13] + accumulate[14] + accumulate[15];
    tmp_acc /= (float) len;

    *dst = tmp_acc;
}

static inline void sqrt512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_sqrt_ps(_mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_sqrt_ps(_mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static inline void round512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}

static inline void trunc512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

static inline void cplxvecmul512f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX512_LEN_FLOAT;   //stop_len << 2;

    int i;
    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_load_ps((float *) (src1) + i);                      // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_load_ps((float *) (src2) + i);                      // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);                                  //a1,a1,a0,a0
            v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                 //a1d1,a1c1,a0d0,a0c0
            src2_tmp = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm512_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            v16sf out = _mm512_mul_ps(src2_tmp, tmp1);
            out = _mm512_fmaddsub_ps(*(v16sf *) _ps512_plus1, tmp2, out);
            _mm512_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_loadu_ps((float *) (src1) + i);                     // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_loadu_ps((float *) (src2) + i);                     // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);                                  //a1,a1,a0,a0
            v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                 //a1d1,a1c1,a0d0,a0c0
            src2_tmp = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm512_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            v16sf out = _mm512_mul_ps(src2_tmp, tmp1);
            out = _mm512_fmaddsub_ps(*(v16sf *) _ps512_plus1, tmp2, out);
            _mm512_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + src2[i].re * src1[i].im;
    }
}

static inline void cplxvecmul512f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);
    stop_len = stop_len * AVX512_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t)(src1Re), (uintptr_t)(src1Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t)(src2Re), (uintptr_t)(src2Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t)(dstRe), (uintptr_t)(dstIm), AVX512_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_load_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_load_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_load_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_load_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_store_ps(dstRe + i, _mm512_sub_ps(ac, bd));
            _mm512_store_ps(dstIm + i, _mm512_add_ps(ad, bc));
        }
    } else {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_loadu_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_loadu_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_loadu_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_loadu_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_storeu_ps(dstRe + i, _mm512_sub_ps(ac, bd));
            _mm512_storeu_ps(dstIm + i, _mm512_add_ps(ad, bc));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + src2Re[i] * src1Im[i];
    }
}

static inline void cplxconjvecmul512f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX512_LEN_FLOAT;   //stop_len << 2;

    int i;
    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_load_ps((float *) (src1) + i);                      // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_load_ps((float *) (src2) + i);                      // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);                                  //a1,a1,a0,a0
            v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                 //a1d1,a1c1,a0d0,a0c0
            src2_tmp = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm512_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            tmp1 = _mm512_mul_ps(tmp1, *(v16sf *) _ps512_min1);                         // -b1,-b1,-b0,-b0
            v16sf out = _mm512_mul_ps(src2_tmp, tmp1);
            out = _mm512_fmaddsub_ps(*(v16sf *) _ps512_plus1, tmp2, out);
            _mm512_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_loadu_ps((float *) (src1) + i);                     // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_loadu_ps((float *) (src2) + i);                     // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);                                  //a1,a1,a0,a0
            v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                 //a1d1,a1c1,a0d0,a0c0
            src2_tmp = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm512_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            tmp1 = _mm512_mul_ps(tmp1, *(v16sf *) _ps512_min1);                         // -b1,-b1,-b0,-b0
            v16sf out = _mm512_mul_ps(src2_tmp, tmp1);
            out = _mm512_fmaddsub_ps(*(v16sf *) _ps512_plus1, tmp2, out);
            _mm512_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + src1[i].im * src2[i].im;
        dst[i].im = src2[i].re * src1[i].im - src1[i].re * src2[i].im;
    }
}

static inline void cplxconjvecmul512f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);
    stop_len = stop_len * AVX512_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t)(src1Re), (uintptr_t)(src1Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t)(src2Re), (uintptr_t)(src2Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t)(dstRe), (uintptr_t)(dstIm), AVX512_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_load_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_load_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_load_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_load_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_store_ps(dstRe + i, _mm512_add_ps(ac, bd));
            _mm512_store_ps(dstIm + i, _mm512_sub_ps(bc, ad));
        }
    } else {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_loadu_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_loadu_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_loadu_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_loadu_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_storeu_ps(dstRe + i, _mm512_add_ps(ac, bd));
            _mm512_storeu_ps(dstIm + i, _mm512_sub_ps(bc, ad));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + src1Im[i] * src2Im[i];
        dstIm[i] = src2Re[i] * src1Im[i] - src1Re[i] * src2Im[i];
    }
}

//prefer using cplxconjvecmulXf if you also need to do a multiply
static inline void cplxconj512f(complex32_t *src, complex32_t *dst, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX512_LEN_FLOAT;   //stop_len << 2;

    float mask[AVX512_LEN_FLOAT] __attribute__((aligned(AVX512_LEN_BYTES)));
    mask[0] = 1.0f;
    mask[1] = -1.0f;
    mask[2] = 1.0f;
    mask[3] = -1.0f;
    mask[4] = 1.0f;
    mask[5] = -1.0f;
    mask[6] = 1.0f;
    mask[7] = -1.0f;
    mask[8] = 1.0f;
    mask[9] = -1.0f;
    mask[10] = 1.0f;
    mask[11] = -1.0f;
    mask[12] = 1.0f;
    mask[13] = -1.0f;
    mask[14] = 1.0f;
    mask[15] = -1.0f;
    v16sf *mask_vec = (v16sf *) mask;

    int i;
    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX512_LEN_BYTES)) {
        //printf("Aligned\n");
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps((float *) (src) + i);
            _mm512_store_ps((float *) (dst) + i, _mm512_mul_ps(src_tmp, *mask_vec));
        }
    } else {
        //printf("Unaligned\n");
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps((float *) (src) + i);
            _mm512_storeu_ps((float *) (dst) + i, _mm512_mul_ps(src_tmp, *mask_vec));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src[i].re;
        dst[i].im = -src[i].im;
    }
}
