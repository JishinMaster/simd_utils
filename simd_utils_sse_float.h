/*
 * Project : SIMD_Utils
 * Version : 0.1.6
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <fenv.h>
#include <stdint.h>
#ifndef ARM
#include <immintrin.h>
#else
#include "sse2neon_wrapper.h"
#endif

#include <math.h>
#include <string.h>

_PS_CONST(min1, -1.0f);

// For tanf
_PS_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS_CONST(DP1, -0.78515625f);
_PS_CONST(DP2, -2.4187564849853515625E-4f);
_PS_CONST(DP3, -3.77489497744594108E-8f);

_PS_CONST(FOPI, 1.27323954473516); /* 4/pi */
_PS_CONST(TAN_P0, 9.38540185543E-3);
_PS_CONST(TAN_P1, 3.11992232697E-3);
_PS_CONST(TAN_P2, 2.44301354525E-2);
_PS_CONST(TAN_P3, 5.34112807005E-2);
_PS_CONST(TAN_P4, 1.33387994085E-1);
_PS_CONST(TAN_P5, 3.33331568548E-1);

_PS_CONST(ASIN_P0, 4.2163199048E-2);
_PS_CONST(ASIN_P1, 2.4181311049E-2);
_PS_CONST(ASIN_P2, 4.5470025998E-2);
_PS_CONST(ASIN_P3, 7.4953002686E-2);
_PS_CONST(ASIN_P4, 1.6666752422E-1);

_PS_CONST(PIF, 3.14159265358979323846);      // PI
_PS_CONST(mPIF, -3.14159265358979323846);    // -PI
_PS_CONST(PIO2F, 1.57079632679489661923);    // PI/2 1.570796326794896619
_PS_CONST(mPIO2F, -1.57079632679489661923);  // -PI/2 1.570796326794896619
_PS_CONST(PIO4F, 0.785398163397448309615);   // PI/4 0.7853981633974483096

_PS_CONST(TANPI8F, 0.414213562373095048802);   // tan(pi/8) => 0.4142135623730950
_PS_CONST(TAN3PI8F, 2.414213562373095048802);  // tan(3*pi/8) => 2.414213562373095

_PS_CONST(ATAN_P0, 8.05374449538e-2);
_PS_CONST(ATAN_P1, -1.38776856032E-1);
_PS_CONST(ATAN_P2, 1.99777106478E-1);
_PS_CONST(ATAN_P3, -3.33329491539E-1);

_PS_CONST_TYPE(pos_sign_mask, int, (int) 0x7FFFFFFF);

#define ROUNDTONEAREST (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define ROUNDTOFLOOR (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
#define ROUNDTOCEIL (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
#define ROUNDTOZERO (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)

/*static inline void print4(__m128 v)
{
    float *p = (float *) &v;
#ifndef __SSE2__
    _mm_empty();
#endif
    printf("[%3.24g, %3.24g, %3.24g, %3.24g]", p[0], p[1], p[2], p[3]);
}

static inline void print8i(__m128i v)
{
    int16_t *p = (int16_t *) &v;
#ifndef __SSE2__
    _mm_empty();
#endif
    printf("[%d, %d, %d, %d,%d, %d, %d, %d]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}*/

static inline void log10_128f(float *src, float *dst, int len)
{
    const v4sf invln10f = _mm_set1_ps((float) INVLN10);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log_ps(_mm_load_ps(src + i));
            _mm_store_ps(dst + i, _mm_mul_ps(src_tmp, invln10f));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log_ps(_mm_loadu_ps(src + i));
            _mm_storeu_ps(dst + i, _mm_mul_ps(src_tmp, invln10f));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void ln_128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, log_ps(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, log_ps(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void fabs128f(float *src, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= (SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, src_tmp));
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void set128f(float *src, float value, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (isAligned((uintptr_t)(src), SSE_LEN_BYTES)) {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(src + i, tmp);
        }
    } else {
#pragma unroll(2)
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = value;
    }
}

// Could be better to just use set(0)
static inline void zero128f(float *src, int len)
{
    const v4sf tmp = _mm_setzero_ps();

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (isAligned((uintptr_t)(src), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(src + i, tmp);
            //_mm_stream_si128(src + i, (__m128i)tmp);
        }
        //_mm_sfence();
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = 0.0f;
    }
}

static inline void copy128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_load_ps(src + i));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_loadu_ps(src + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_add_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_add_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}


static inline void mul128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_mul_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_mul_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_sub_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_sub_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

//TODO : "Immediate add/mul?"
// No need for subc, just use addc(-value)
static inline void addc128f(float *src, float value, float *dst, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_add_ps(tmp, _mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_add_ps(tmp, _mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc128f(float *src, float value, float *dst, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_mul_ps(tmp, _mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_mul_ps(tmp, _mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd128f(float *_a, float *_b, float *_c, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(_b), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t)(_c), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(_a + i);
            v4sf b = _mm_load_ps(_b + i);
            v4sf c = _mm_load_ps(_c + i);
            _mm_store_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(_a + i);
            v4sf b = _mm_loadu_ps(_b + i);
            v4sf c = _mm_loadu_ps(_c + i);
            _mm_storeu_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcadd128f(float *_a, float _b, float *_c, float *dst, int len)
{
    v4sf b = _mm_set1_ps(_b);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_c), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(_a + i);
            v4sf c = _mm_load_ps(_c + i);
            _mm_store_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(_a + i);
            v4sf c = _mm_loadu_ps(_c + i);
            _mm_storeu_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddc128f(float *_a, float _b, float _c, float *dst, int len)
{
    v4sf b = _mm_set1_ps(_b);
    v4sf c = _mm_set1_ps(_c);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(_a + i);
            _mm_store_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(_a + i);
            _mm_storeu_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
    }
}

static inline void muladdc128f(float *_a, float *_b, float _c, float *dst, int len)
{
    v4sf c = _mm_set1_ps(_c);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_b), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(_a + i);
            v4sf b = _mm_load_ps(_b + i);
            _mm_store_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(_a + i);
            v4sf b = _mm_loadu_ps(_b + i);
            _mm_storeu_ps(dst + i, _mm_fmadd_ps_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

#warning "src2 should have no 0.0f values!"
static inline void div128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_div_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_div_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

// TODO : remove previous index dependency to become Out Of Order
static inline void vectorSlope128f(float *dst, int len, float offset, float slope)
{
    v4sf coef = _mm_set_ps(3.0f * slope, 2.0f * slope, slope, 0.0f);
    v4sf slope8_vec = _mm_set1_ps(8.0f * slope);
    v4sf curVal = _mm_add_ps(_mm_set1_ps(offset), coef);
    v4sf curVal2 = _mm_add_ps(_mm_set1_ps(offset), coef);
    curVal2 = _mm_add_ps(curVal2, _mm_set1_ps(4.0f * slope));

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        _mm_store_ps(dst + 0, curVal);
        _mm_store_ps(dst + SSE_LEN_FLOAT, curVal2);
    } else {
        _mm_storeu_ps(dst + 0, curVal);
        _mm_storeu_ps(dst + SSE_LEN_FLOAT, curVal2);
    }

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        for (int i = 2 * SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            curVal = _mm_add_ps(curVal, slope8_vec);
            _mm_store_ps(dst + i, curVal);
            curVal2 = _mm_add_ps(curVal2, slope8_vec);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, curVal2);
        }
    } else {
        for (int i = 2 * SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            curVal = _mm_add_ps(curVal, slope8_vec);
            _mm_storeu_ps(dst + i, curVal);
            curVal2 = _mm_add_ps(curVal2, slope8_vec);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

//#ifndef ARM

static inline void convertFloat32ToU8_128(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    // Default bankers rounding => round to nearest even
    if (rounding_mode == RndFinancial) {
        if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
            for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
                v4sf tmp1 = _mm_mul_ps(_mm_load_ps(src + i), scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(_mm_load_ps(src + i + SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(_mm_load_ps(src + i + 2 * SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(_mm_load_ps(src + i + 3 * SSE_LEN_FLOAT), scale_fact_vec);

                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(tmp2), _mm_cvtps_pi16(tmp1));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(tmp4), _mm_cvtps_pi16(tmp3));
                _mm_store_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        } else {
            for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                v4sf tmp1 = _mm_mul_ps(_mm_loadu_ps(src + i), scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(_mm_loadu_ps(src + i + SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(_mm_loadu_ps(src + i + 2 * SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(_mm_loadu_ps(src + i + 3 * SSE_LEN_FLOAT), scale_fact_vec);

                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(tmp2), _mm_cvtps_pi16(tmp1));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(tmp4), _mm_cvtps_pi16(tmp3));
                _mm_storeu_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        }

        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint8_t)(tmp > 255.0f ? 255.0f : tmp);  //round to nearest even with round(x/2)*2
        }
    } else {
        if (rounding_mode == RndZero) {
            _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  //rounding_vec = ROUNDTOZERO;
            fesetround(FE_TOWARDZERO);
        } else {
            _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  //rounding_vec = ROUNDTONEAREST;
            fesetround(FE_TONEAREST);
        }

        if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
            for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
                v4sf tmp1 = _mm_mul_ps(_mm_load_ps(src + i), scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(_mm_load_ps(src + i + SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(_mm_load_ps(src + i + 2 * SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(_mm_load_ps(src + i + 3 * SSE_LEN_FLOAT), scale_fact_vec);

                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp2, _MM_FROUND_CUR_DIRECTION)), _mm_cvtps_pi16(_mm_round_ps(tmp1, _MM_FROUND_CUR_DIRECTION)));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp4, _MM_FROUND_CUR_DIRECTION)), _mm_cvtps_pi16(_mm_round_ps(tmp3, _MM_FROUND_CUR_DIRECTION)));
                _mm_store_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        } else {
            for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                v4sf tmp1 = _mm_mul_ps(_mm_loadu_ps(src + i), scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(_mm_loadu_ps(src + i + SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(_mm_loadu_ps(src + i + 2 * SSE_LEN_FLOAT), scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(_mm_loadu_ps(src + i + 3 * SSE_LEN_FLOAT), scale_fact_vec);

                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp2, _MM_FROUND_CUR_DIRECTION)), _mm_cvtps_pi16(_mm_round_ps(tmp1, _MM_FROUND_CUR_DIRECTION)));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp4, _MM_FROUND_CUR_DIRECTION)), _mm_cvtps_pi16(_mm_round_ps(tmp3, _MM_FROUND_CUR_DIRECTION)));
                _mm_storeu_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        }

        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (uint8_t)(tmp > 255.0f ? 255.0f : tmp);
        }
    }
}

//#endif  //ARM

//TODO : find a way to avoid __m64
typedef union xmm_mm_union_int {
    __m128i xmm;
    __m64 mm[2];
} xmm_mm_union_int;

#define COPY_XMM_TO_MM_INT(xmm_, mm0_, mm1_) \
    {                                        \
        xmm_mm_union_int u;                  \
        u.xmm = xmm_;                        \
        mm0_ = u.mm[0];                      \
        mm1_ = u.mm[1];                      \
    }

static inline void convertInt16ToFloat32_128_old(int16_t *src, float *dst, int len, int scale_factor)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            __m64 shortlo, shorthi;
            COPY_XMM_TO_MM_INT(_mm_load_si128((__m128i *) (src + i)), shortlo, shorthi);

            v4sf floatlo = _mm_mul_ps(_mm_cvtpi16_ps(shortlo), scale_fact_vec);
            v4sf floathi = _mm_mul_ps(_mm_cvtpi16_ps(shorthi), scale_fact_vec);

            _mm_store_ps(dst + i, floatlo);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, floathi);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            __m64 shortlo, shorthi;
            COPY_XMM_TO_MM_INT(_mm_loadu_si128((__m128i *) (src + i)), shortlo, shorthi);

            v4sf floatlo = _mm_mul_ps(_mm_cvtpi16_ps(shortlo), scale_fact_vec);
            v4sf floathi = _mm_mul_ps(_mm_cvtpi16_ps(shorthi), scale_fact_vec);

            _mm_storeu_ps(dst + i, floatlo);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, floathi);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}

//This improved version does not use MMX
static inline void convertInt16ToFloat32_128(int16_t *src, float *dst, int len, int scale_factor)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4si vec = _mm_load_si128((__m128i *) (src + i));  //loads 1 2 3 4 5 6 7 8 8
            v4si low = _mm_unpacklo_epi16(vec, vec);           // low 1 1 2 2 3 3 4 4
            v4si high = _mm_unpackhi_epi16(vec, vec);          // high 5 5 6 6 7 7 8 8
            low = _mm_srai_epi32(low, 0x10);                   //make low 1 -1 2 -1 3 -1 4 -4
            high = _mm_srai_epi32(high, 0x10);                 // make high 5 -1 6 -1 7 -1 8 -1

            //convert the vector to float and scale it
            v4sf floatlo = _mm_mul_ps(_mm_cvtepi32_ps(low), scale_fact_vec);
            v4sf floathi = _mm_mul_ps(_mm_cvtepi32_ps(high), scale_fact_vec);

            _mm_store_ps(dst + i, floatlo);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, floathi);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4si vec = _mm_loadu_si128((__m128i *) (src + i));  //loads 1 2 3 4 5 6 7 8 8
            v4si low = _mm_unpacklo_epi16(vec, vec);            // low 1 1 2 2 3 3 4 4
            v4si high = _mm_unpackhi_epi16(vec, vec);           // high 5 5 6 6 7 7 8 8
            low = _mm_srai_epi32(low, 0x10);                    //make low 1 -1 2 -1 3 -1 4 -4
            high = _mm_srai_epi32(high, 0x10);                  // make high 5 -1 6 -1 7 -1 8 -1

            //convert the vector to float and scale it
            v4sf floatlo = _mm_mul_ps(_mm_cvtepi32_ps(low), scale_fact_vec);
            v4sf floathi = _mm_mul_ps(_mm_cvtepi32_ps(high), scale_fact_vec);

            _mm_storeu_ps(dst + i, floatlo);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, floathi);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}

#ifndef ARM
// converts 32bits complex float to two arrays real and im
static inline void cplxtoreal128f(float *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (SSE_LEN_FLOAT);
    stop_len *= SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned3((uintptr_t)(src), (uintptr_t)(dstRe), (uintptr_t)(dstIm), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf vec1 = _mm_load_ps(src + i);
            v4sf vec2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            _mm_store_ps(dstRe + j, _mm_shuffle_ps(vec1, vec2, _MM_SHUFFLE(2, 0, 2, 0)));
            _mm_store_ps(dstIm + j, _mm_shuffle_ps(vec1, vec2, _MM_SHUFFLE(3, 1, 3, 1)));
            j += SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf vec1 = _mm_loadu_ps(src + i);
            v4sf vec2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            _mm_storeu_ps(dstRe + j, _mm_shuffle_ps(vec1, vec2, _MM_SHUFFLE(2, 0, 2, 0)));
            _mm_storeu_ps(dstIm + j, _mm_shuffle_ps(vec1, vec2, _MM_SHUFFLE(3, 1, 3, 1)));
            j += SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < 2 * len; i += 2) {
        dstRe[j] = src[i];
        dstIm[j] = src[i + 1];
        j++;
    }
}

static inline void realtocplx128f(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned3((uintptr_t)(srcRe), (uintptr_t)(srcIm), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf re = _mm_load_ps(srcRe + i);
            v4sf im = _mm_load_ps(srcIm + i);
            v4sf cplx0 = _mm_unpacklo_ps(re, im);
            v4sf cplx1 = _mm_unpackhi_ps(re, im);
            _mm_store_ps(dst + j, cplx0);
            _mm_store_ps(dst + j + SSE_LEN_FLOAT, cplx1);
            j += 2 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf re = _mm_loadu_ps(srcRe + i);
            v4sf im = _mm_loadu_ps(srcIm + i);
            v4sf cplx0 = _mm_unpacklo_ps(re, im);
            v4sf cplx1 = _mm_unpackhi_ps(re, im);
            _mm_storeu_ps(dst + j, cplx0);
            _mm_storeu_ps(dst + j + SSE_LEN_FLOAT, cplx1);
            j += 2 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[j] = srcRe[i];
        dst[j + 1] = srcIm[i];
        j += 2;
    }
}
#else  /* ARM */
// ARM Neon optimized version
static inline void cplxtoreal128f(float *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (SSE_LEN_FLOAT);
    stop_len *= SSE_LEN_FLOAT;

    int j = 0;
    for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
        float32x4x2_t values = vld2q_f32(src + i);
        vst1q_f32(dstRe + j, values.val[0]);
        vst1q_f32(dstIm + j, values.val[1]);
        j += SSE_LEN_FLOAT;
    }

    for (int i = stop_len; i < 2 * len; i += 2) {
        dstRe[j] = src[i];
        dstIm[j] = src[i + 1];
        j++;
    }
}

static inline void realtocplx128f(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= SSE_LEN_FLOAT;

    int j = 0;
    for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
        v4sf re = vld1q_f32(srcRe + i);
        v4sf im = vld1q_f32(srcIm + i);
        float32x4x2_t reim = {re, im};
        vst2q_f32(dst + j, reim);
        j += 2 * SSE_LEN_FLOAT;
    }

    for (int i = stop_len; i < len; i++) {
        dst[j] = srcRe[i];
        dst[j + 1] = srcIm[i];
        j += 2;
    }
}
#endif /* ARM */


static inline void convert128_64f32f(double *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            __m128d src_lo = _mm_load_pd(src + i);
            __m128d src_hi = _mm_load_pd(src + i + 2);
            v4sf tmp = _mm_movelh_ps(_mm_cvtpd_ps(src_lo), _mm_cvtpd_ps(src_hi));
            _mm_store_ps(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            __m128d src_lo = _mm_loadu_pd(src + i);
            __m128d src_hi = _mm_loadu_pd(src + i + 2);
            v4sf tmp = _mm_movelh_ps(_mm_cvtpd_ps(src_lo), _mm_cvtpd_ps(src_hi));
            _mm_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i];
    }
}

//#ifndef ARM
static inline void convert128_32f64f(float *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);                                          //load a,b,c,d
            v4sf src_tmp_hi = _mm_shuffle_ps(src_tmp, src_tmp, _MM_SHUFFLE(1, 0, 3, 2));  // rotate vec from abcd to cdab
            _mm_store_pd(dst + i, _mm_cvtps_pd(src_tmp));                                 //store the c and d converted in 64bits
            _mm_store_pd(dst + i + 2, _mm_cvtps_pd(src_tmp_hi));                          //store the a and b converted in 64bits
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);                                         //load a,b,c,d
            v4sf src_tmp_hi = _mm_shuffle_ps(src_tmp, src_tmp, _MM_SHUFFLE(1, 0, 3, 2));  // rotate vec from abcd to cdab
            _mm_storeu_pd(dst + i, _mm_cvtps_pd(src_tmp));                                //store the c and d converted in 64bits
            _mm_storeu_pd(dst + i + 2, _mm_cvtps_pd(src_tmp_hi));                         //store the a and b converted in 64bits
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (double) src[i];
    }
}
//#endif

//TODO : find a better way to work on aligned data
static inline void flip128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    for (int i = 0; i < SSE_LEN_FLOAT; i++) {
        dst[len - i - 1] = src[i];
    }

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = SSE_LEN_FLOAT; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);                                  //load a,b,c,d
            v4sf src_tmp_flip = _mm_shuffle_ps(src_tmp, src_tmp, IMM8_FLIP_VEC);  // rotate vec from abcd to bcba
            _mm_storeu_ps(dst + len - i - SSE_LEN_FLOAT, src_tmp_flip);           //store the flipped vector
        }
    } else {
        for (int i = SSE_LEN_FLOAT; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);                                 //load a,b,c,d
            v4sf src_tmp_flip = _mm_shuffle_ps(src_tmp, src_tmp, IMM8_FLIP_VEC);  // rotate vec from abcd to bcba
            _mm_storeu_ps(dst + len - i - SSE_LEN_FLOAT, src_tmp_flip);           //store the flipped vector
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void maxevery128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_max_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_max_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_min_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_min_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}


static inline void minmax128f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    v4sf max_v, min_v;
    v4sf src_tmp;
    float min_f[SSE_LEN_FLOAT] __attribute__((aligned(SSE_LEN_BYTES)));
    float max_f[SSE_LEN_FLOAT] __attribute__((aligned(SSE_LEN_BYTES)));
    float min_tmp;
    float max_tmp;

    if (isAligned((uintptr_t)(src), SSE_LEN_BYTES)) {
        src_tmp = _mm_load_ps(src + 0);
        max_v = src_tmp;
        min_v = src_tmp;
        for (int i = SSE_LEN_FLOAT; i < stop_len; i += SSE_LEN_FLOAT) {
            src_tmp = _mm_load_ps(src + i);
            max_v = _mm_max_ps(max_v, src_tmp);
            min_v = _mm_min_ps(min_v, src_tmp);
        }
    } else {
        src_tmp = _mm_loadu_ps(src + 0);
        max_v = src_tmp;
        min_v = src_tmp;
        for (int i = SSE_LEN_FLOAT; i < stop_len; i += SSE_LEN_FLOAT) {
            src_tmp = _mm_loadu_ps(src + i);
            max_v = _mm_max_ps(max_v, src_tmp);
            min_v = _mm_min_ps(min_v, src_tmp);
        }
    }

    _mm_store_ps(max_f, max_v);
    _mm_store_ps(min_f, min_v);

    max_tmp = max_f[0];
    max_tmp = max_tmp > max_f[1] ? max_tmp : max_f[1];
    max_tmp = max_tmp > max_f[2] ? max_tmp : max_f[2];
    max_tmp = max_tmp > max_f[3] ? max_tmp : max_f[3];

    min_tmp = min_f[0];
    min_tmp = min_tmp < min_f[1] ? min_tmp : min_f[1];
    min_tmp = min_tmp < min_f[2] ? min_tmp : min_f[2];
    min_tmp = min_tmp < min_f[3] ? min_tmp : min_f[3];

    for (int i = stop_len; i < len; i++) {
        max_tmp = max_tmp > src[i] ? max_tmp : src[i];
        min_tmp = min_tmp < src[i] ? min_tmp : src[i];
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}


static inline void threshold128_gt_f(float *src, float *dst, int len, float value)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_min_ps(src_tmp, tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_min_ps(src_tmp, tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold128_gtabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = _mm_set1_ps(value);
    const v4sf mval = _mm_set1_ps(-value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);         //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf gtmask = _mm_cmpgt_ps(src_abs, pval);            //if abs(A) > value => 0xFFFFFFFF, else 0
            v4sf sval = _mm_blendv_ps(mval, pval, eqmask);        //if A >= 0 value, else -value
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            _mm_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);         //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf gtmask = _mm_cmpgt_ps(src_abs, pval);            //if abs(A) > value => 0xFFFFFFFF, else 0
            v4sf sval = _mm_blendv_ps(mval, pval, eqmask);        //if A >= 0 value, else -value
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            _mm_storeu_ps(dst + i, dst_tmp);
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

static inline void threshold128_lt_f(float *src, float *dst, int len, float value)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_max_ps(src_tmp, tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_max_ps(src_tmp, tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

static inline void threshold128_ltabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = _mm_set1_ps(value);
    const v4sf mval = _mm_set1_ps(-value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);         //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf gtmask = _mm_cmplt_ps(src_abs, pval);            //if abs(A) > value => 0xFFFFFFFF, else 0
            v4sf sval = _mm_blendv_ps(mval, pval, eqmask);        //if A >= 0 value, else -value
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            _mm_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);         //if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf gtmask = _mm_cmplt_ps(src_abs, pval);            //if abs(A) > value => 0xFFFFFFFF, else 0
            v4sf sval = _mm_blendv_ps(mval, pval, eqmask);        //if A >= 0 value, else -value
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            _mm_storeu_ps(dst + i, dst_tmp);
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

static inline void threshold128_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v4sf ltlevel_v = _mm_set1_ps(ltlevel);
    const v4sf ltvalue_v = _mm_set1_ps(ltvalue);
    const v4sf gtlevel_v = _mm_set1_ps(gtlevel);
    const v4sf gtvalue_v = _mm_set1_ps(gtvalue);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf lt_mask = _mm_cmplt_ps(src_tmp, ltlevel_v);
            v4sf gt_mask = _mm_cmpgt_ps(src_tmp, gtlevel_v);
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf lt_mask = _mm_cmplt_ps(src_tmp, ltlevel_v);
            v4sf gt_mask = _mm_cmpgt_ps(src_tmp, gtlevel_v);
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm_storeu_ps(dst + i, dst_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}

static inline void sin128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, sin_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, sin_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, cos_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, cos_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos128f(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src), (uintptr_t)(dst_sin), (uintptr_t)(dst_cos), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm_store_ps(dst_sin + i, dst_sin_tmp);
            _mm_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline v4sf atanf_ps(v4sf xx, const v4sf positive_mask, const v4sf negative_mask)
{
    v4sf x, y, z;
    v4sf sign;
    v4sf suptan3pi8, inftan3pi8suppi8;
    v4sf tmp;

    x = _mm_and_ps(positive_mask, xx);
    sign = _mm_cmplt_ps(xx, _mm_setzero_ps());  //0xFFFFFFFF if x < 0.0, sign = -1

    /* range reduction */

    y = _mm_setzero_ps();
    suptan3pi8 = _mm_cmpgt_ps(x, *(v4sf *) _ps_TAN3PI8F);  // if( x > tan 3pi/8 )
    x = _mm_blendv_ps(x, _mm_div_ps(*(v4sf *) _ps_min1, x), suptan3pi8);
    y = _mm_blendv_ps(y, *(v4sf *) _ps_PIO2F, suptan3pi8);


    inftan3pi8suppi8 = _mm_and_ps(_mm_cmplt_ps(x, *(v4sf *) _ps_TAN3PI8F), _mm_cmpgt_ps(x, *(v4sf *) _ps_TANPI8F));  // if( x > tan 3pi/8 )
    x = _mm_blendv_ps(x, _mm_div_ps(_mm_sub_ps(x, *(v4sf *) _ps_1), _mm_add_ps(x, *(v4sf *) _ps_1)), inftan3pi8suppi8);
    y = _mm_blendv_ps(y, *(v4sf *) _ps_PIO4F, inftan3pi8suppi8);

    z = _mm_mul_ps(x, x);

    tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_ATAN_P0, z, *(v4sf *) _ps_ATAN_P1);
    tmp = _mm_fmadd_ps_custom(tmp, z, *(v4sf *) _ps_ATAN_P2);
    tmp = _mm_fmadd_ps_custom(tmp, z, *(v4sf *) _ps_ATAN_P3);
    tmp = _mm_mul_ps(z, tmp);
    tmp = _mm_fmadd_ps_custom(tmp, x, x);

    y = _mm_add_ps(y, tmp);

    y = _mm_blendv_ps(y, _mm_xor_ps(negative_mask, y), sign);

    return (y);
}

static inline void atan128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    const v4sf positive_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    const v4sf negative_mask = _mm_castsi128_ps(_mm_set1_epi32(~0x7FFFFFFF));

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, atanf_ps(src_tmp, positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, atanf_ps(src_tmp, positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline v4sf atan2f_ps(v4sf y, v4sf x, const v4sf positive_mask, const v4sf negative_mask)
{
    v4sf z, w;
    v4sf xinfzero, yinfzero, xeqzero, yeqzero;
    v4sf xeqzeroandyinfzero, yeqzeroandxinfzero;
    v4sf specialcase;

    xinfzero = _mm_cmplt_ps(x, _mm_setzero_ps());  // code =2
    yinfzero = _mm_cmplt_ps(y, _mm_setzero_ps());  // code = code |1;

    xeqzero = _mm_cmpeq_ps(x, _mm_setzero_ps());
    yeqzero = _mm_cmpeq_ps(y, _mm_setzero_ps());

    z = *(v4sf *) _ps_PIO2F;
    xeqzeroandyinfzero = _mm_and_ps(xeqzero, yinfzero);
    z = _mm_blendv_ps(z, *(v4sf *) _ps_mPIO2F, xeqzeroandyinfzero);
    z = _mm_blendv_ps(z, _mm_setzero_ps(), yeqzero);
    yeqzeroandxinfzero = _mm_and_ps(yeqzero, xinfzero);
    z = _mm_blendv_ps(z, *(v4sf *) _ps_PIF, yeqzeroandxinfzero);
    specialcase = _mm_or_ps(xeqzero, yeqzero);
    w = _mm_setzero_ps();
    w = _mm_blendv_ps(w, *(v4sf *) _ps_PIF, _mm_andnot_ps(yinfzero, xinfzero));                                  // y >= 0 && x<0
    w = _mm_blendv_ps(w, *(v4sf *) _ps_mPIF, _mm_and_ps(yinfzero, xinfzero));                                    // y < 0 && x<0
    z = _mm_blendv_ps(_mm_add_ps(w, atanf_ps(_mm_div_ps(y, x), positive_mask, negative_mask)), z, specialcase);  // atanf(y/x) if not in special case
    return (z);
}

static inline void atan2128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    const v4sf positive_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    const v4sf negative_mask = _mm_castsi128_ps(_mm_set1_epi32(~0x7FFFFFFF));

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, atan2f_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i), positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, atan2f_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i), positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}


static inline v4sf asinf_ps(v4sf xx, const v4sf positive_mask, const v4sf negative_mask)
{
    v4sf a, x, z, z_tmp;
    v4sf sign;
    v4sf ainfem4, asup0p5;
    v4sf tmp;
    x = xx;
    a = _mm_and_ps(positive_mask, x);          //fabs(x)
    sign = _mm_cmplt_ps(x, _mm_setzero_ps());  //0xFFFFFFFF if x < 0.0

    //TODO : vectorize this
    /*if( a > 1.0f )
	{
		return( 0.0f );
	}*/


    ainfem4 = _mm_cmplt_ps(a, _mm_set1_ps(1.0e-4));  //if( a < 1.0e-4f )

    asup0p5 = _mm_cmpgt_ps(a, *(v4sf *) _ps_0p5);  //if( a > 0.5f ) flag = 1 else 0
    z_tmp = _mm_sub_ps(*(v4sf *) _ps_1, a);
    z_tmp = _mm_mul_ps(*(v4sf *) _ps_0p5, z_tmp);
    z = _mm_blendv_ps(_mm_mul_ps(a, a), z_tmp, asup0p5);
    x = _mm_blendv_ps(a, _mm_sqrt_ps(z), asup0p5);

    tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_ASIN_P0, z, *(v4sf *) _ps_ASIN_P1);
    tmp = _mm_fmadd_ps_custom(z, tmp, *(v4sf *) _ps_ASIN_P2);
    tmp = _mm_fmadd_ps_custom(z, tmp, *(v4sf *) _ps_ASIN_P3);
    tmp = _mm_fmadd_ps_custom(z, tmp, *(v4sf *) _ps_ASIN_P4);
    tmp = _mm_mul_ps(z, tmp);
    tmp = _mm_fmadd_ps_custom(x, tmp, x);

    z = tmp;

    z_tmp = _mm_add_ps(z, z);
    z_tmp = _mm_sub_ps(*(v4sf *) _ps_PIO2F, z_tmp);
    z = _mm_blendv_ps(z, z_tmp, asup0p5);

    //done:
    z = _mm_blendv_ps(z, a, ainfem4);
    z = _mm_blendv_ps(z, _mm_xor_ps(negative_mask, z), sign);

    return (z);
}

static inline void asin128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    const v4sf positive_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    const v4sf negative_mask = _mm_castsi128_ps(_mm_set1_epi32(~0x7FFFFFFF));

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, asinf_ps(src_tmp, positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, asinf_ps(src_tmp, positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}


static inline v4sf tanf_ps(v4sf xx, const v4sf positive_mask, const v4sf negative_mask)
{
    v4sf x, y, z, zz;
    v4si j;  //long?
    v4sf sign, xsupem4;
    v4sf tmp;
    v4si jandone, jandtwo;

    x = _mm_and_ps(positive_mask, xx);  //fabs(xx) //OK

    /* compute x mod PIO4 */

    //TODO : on neg values should be ceil and not floor
    //j = _mm_cvtps_epi32( _mm_round_ps(_mm_mul_ps(*(v4sf*)_ps_FOPI,x), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC )); /* integer part of x/(PI/4), using floor */
    j = _mm_cvttps_epi32(_mm_mul_ps(*(v4sf *) _ps_FOPI, x));
    y = _mm_cvtepi32_ps(j);

    jandone = _mm_cmpgt_epi32(_mm_and_si128(j, *(v4si *) _pi32_1), _mm_setzero_si128());  //Ok?

    y = _mm_blendv_ps(y, _mm_add_ps(y, *(v4sf *) _ps_1), (v4sf) jandone);

    j = _mm_cvttps_epi32(y);  // no need to round again

#if 1
    //z = ((x - y * DP1) - y * DP2) - y * DP3;
    /*tmp = _mm_mul_ps(y, *(v4sf*)_ps_DP1);
	z   = _mm_add_ps(x, tmp);
	tmp = _mm_mul_ps(y, *(v4sf*)_ps_DP2);
	z   = _mm_add_ps(z, tmp);
	tmp = _mm_mul_ps(y, *(v4sf*)_ps_DP3);
	z   = _mm_add_ps(z, tmp);*/

    z = _mm_fmadd_ps_custom(y, *(v4sf *) _ps_DP1, x);
    z = _mm_fmadd_ps_custom(y, *(v4sf *) _ps_DP2, z);
    z = _mm_fmadd_ps_custom(y, *(v4sf *) _ps_DP3, z);
    //print4(*(v4sf*)_ps_DP1);print4(*(v4sf*)_ps_DP2);print4(*(v4sf*)_ps_DP3);printf("\n");
    //print4(y);printf("\n");


    zz = _mm_mul_ps(z, z);  //z*z

    //TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_TAN_P0, zz, *(v4sf *) _ps_TAN_P1);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P2);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P3);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P4);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P5);
    tmp = _mm_mul_ps(zz, tmp);
    tmp = _mm_fmadd_ps_custom(tmp, z, z);
#endif

    xsupem4 = _mm_cmpgt_ps(x, _mm_set1_ps(1.0e-4));  //if( x > 1.0e-4 )
    y = _mm_blendv_ps(z, tmp, xsupem4);

    jandtwo = _mm_cmpgt_epi32(_mm_and_si128(j, *(v4si *) _pi32_2), _mm_setzero_si128());

    y = _mm_blendv_ps(y, _mm_div_ps(_mm_set1_ps(-1.0f), y), (v4sf)(jandtwo));

    sign = _mm_cmplt_ps(xx, _mm_setzero_ps());  //0xFFFFFFFF if xx < 0.0
    y = _mm_blendv_ps(y, _mm_xor_ps(negative_mask, y), sign);

    return (y);
}

static inline void tan128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    const v4sf positive_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    const v4sf negative_mask = _mm_castsi128_ps(_mm_set1_epi32(~0x7FFFFFFF));

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, tanf_ps(src_tmp, positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, tanf_ps(src_tmp, positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void tan128f_naive(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf sin_tmp, cos_tmp;
            sincos_ps(src_tmp, &sin_tmp, &cos_tmp);
            _mm_store_ps(dst + i, _mm_div_ps(sin_tmp, cos_tmp));
            //_mm_store_ps(dst + i, _mm_div_ps(sin_ps(src_tmp),cos_ps(src_tmp)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_div_ps(sin_ps(src_tmp), cos_ps(src_tmp)));
            //_mm_storeu_ps(dst + i, _mm_div_ps(sin_ps(src_tmp),cos_ps(src_tmp)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void magnitude128f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(srcRe), (uintptr_t)(srcIm), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_load_ps(srcRe + i);
            v4sf re2 = _mm_mul_ps(re_tmp, re_tmp);
            v4sf im_tmp = _mm_load_ps(srcIm + i);
            v4sf im2 = _mm_mul_ps(im_tmp, im_tmp);
            _mm_store_ps(dst + i, _mm_sqrt_ps(_mm_add_ps(re2, im2)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_loadu_ps(srcRe + i);
            v4sf re2 = _mm_mul_ps(re_tmp, re_tmp);
            v4sf im_tmp = _mm_loadu_ps(srcIm + i);
            v4sf im2 = _mm_mul_ps(im_tmp, im_tmp);
            _mm_storeu_ps(dst + i, _mm_sqrt_ps(_mm_add_ps(re2, im2)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i]);
    }
}

static inline void powerspect128f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(srcRe), (uintptr_t)(srcIm), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_load_ps(srcRe + i);
            v4sf re2 = _mm_mul_ps(re_tmp, re_tmp);
            v4sf im_tmp = _mm_load_ps(srcIm + i);
            v4sf im2 = _mm_mul_ps(im_tmp, im_tmp);
            _mm_store_ps(dst + i, _mm_add_ps(re2, im2));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_loadu_ps(srcRe + i);
            v4sf re2 = _mm_mul_ps(re_tmp, re_tmp);
            v4sf im_tmp = _mm_loadu_ps(srcIm + i);
            v4sf im2 = _mm_mul_ps(im_tmp, im_tmp);
            _mm_storeu_ps(dst + i, _mm_add_ps(re2, im2));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i];
    }
}


static inline void magnitude128f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf cplx01 = _mm_load_ps((const float *) src + j);
            v4sf cplx23 = _mm_load_ps((const float *) src + j + SSE_LEN_FLOAT);  // complex is 2 floats
            v4sf cplx01_square = _mm_mul_ps(cplx01, cplx01);
            v4sf cplx23_square = _mm_mul_ps(cplx23, cplx23);
            v4sf square_sum_0123 = _mm_hadd_ps(cplx01_square, cplx23_square);
            _mm_store_ps(dst + i, _mm_sqrt_ps(square_sum_0123));
            j += 2 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf cplx01 = _mm_loadu_ps((const float *) src + j);
            v4sf cplx23 = _mm_loadu_ps((const float *) src + j + SSE_LEN_FLOAT);  // complex is 2 floats
            v4sf cplx01_square = _mm_mul_ps(cplx01, cplx01);
            v4sf cplx23_square = _mm_mul_ps(cplx23, cplx23);
            v4sf square_sum_0123 = _mm_hadd_ps(cplx01_square, cplx23_square);
            _mm_storeu_ps(dst + i, _mm_sqrt_ps(square_sum_0123));
            j += 2 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i].re * src[i].re + src[i].im * src[i].im);
    }
}

static inline void subcrev128f(float *src, float value, float *dst, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_sub_ps(tmp, _mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_sub_ps(tmp, _mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void sum128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v4sf vec_acc1 = _mm_setzero_ps();  //initialize the vector accumulator
    v4sf vec_acc2 = _mm_setzero_ps();  //initialize the vector accumulator
    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf vec_tmp1 = _mm_load_ps(src + i);
            vec_acc1 = _mm_add_ps(vec_acc1, vec_tmp1);
            v4sf vec_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            vec_acc2 = _mm_add_ps(vec_acc2, vec_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf vec_tmp1 = _mm_loadu_ps(src + i);
            vec_acc1 = _mm_add_ps(vec_acc1, vec_tmp1);
            v4sf vec_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            vec_acc2 = _mm_add_ps(vec_acc2, vec_tmp2);
        }
    }
    vec_acc1 = _mm_add_ps(vec_acc1, vec_acc2);
    _mm_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

    *dst = tmp_acc;
}

static inline void mean128f(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sum128f(src, dst, len);
    *dst *= coeff;
}

static inline void sqrt128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_sqrt_ps(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_sqrt_ps(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}


static inline void round128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_round_ps(src_tmp, ROUNDTONEAREST));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_round_ps(src_tmp, ROUNDTONEAREST));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_round_ps(src_tmp, ROUNDTOCEIL));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_round_ps(src_tmp, ROUNDTOCEIL));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_round_ps(src_tmp, ROUNDTOFLOOR));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_round_ps(src_tmp, ROUNDTOFLOOR));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}



static inline void cplxvecmul128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * SSE_LEN_FLOAT;   //stop_len << 2;

    int i;
    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        //printf("Aligned\n");
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_load_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_load_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                   //a1,a1,a0,a0
            v4sf tmp2 = _mm_mul_ps(tmp1, src2_tmp);                                  //a1d1,a1c1,a0d0,a0c0
            src2_tmp = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            v4sf out = _mm_mul_ps(src2_tmp, tmp1);
            out = _mm_addsub_ps(tmp2, out);
            _mm_store_ps((float *) (dst) + i, out);
        }
    } else {
        //printf("Unaligned\n");
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_loadu_ps((float *) (src1) + i);                      // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_loadu_ps((float *) (src2) + i);                      // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                   //a1,a1,a0,a0
            v4sf tmp2 = _mm_mul_ps(tmp1, src2_tmp);                                  //a1d1,a1c1,a0d0,a0c0
            src2_tmp = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            tmp1 = _mm_movehdup_ps(src1_tmp);                                        //b1,b1,b0,b0
            v4sf out = _mm_mul_ps(src2_tmp, tmp1);
            out = _mm_addsub_ps(tmp2, out);
            _mm_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + src2[i].re * src1[i].im;
    }
}


static inline void cplxvecmul128f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len = stop_len * SSE_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t)(src1Re), (uintptr_t)(src1Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t)(src2Re), (uintptr_t)(src2Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t)(dstRe), (uintptr_t)(dstIm), SSE_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_load_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_load_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_load_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_load_ps((float *) (src2Im) + i);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);
            v4sf bd = _mm_mul_ps(src1Im_tmp, src2Im_tmp);
            v4sf ad = _mm_mul_ps(src1Re_tmp, src2Im_tmp);
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm_store_ps(dstRe + i, _mm_sub_ps(ac, bd));
            _mm_store_ps(dstIm + i, _mm_add_ps(ad, bc));
        }
    } else {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_loadu_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_loadu_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_loadu_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_loadu_ps((float *) (src2Im) + i);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);
            v4sf bd = _mm_mul_ps(src1Im_tmp, src2Im_tmp);
            v4sf ad = _mm_mul_ps(src1Re_tmp, src2Im_tmp);
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm_storeu_ps(dstRe + i, _mm_sub_ps(ac, bd));
            _mm_storeu_ps(dstIm + i, _mm_add_ps(ad, bc));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + src2Re[i] * src1Im[i];
    }
}
