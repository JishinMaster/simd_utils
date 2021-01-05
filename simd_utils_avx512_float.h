/*
 * Project : SIMD_Utils
 * Version : 0.1.4
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#include "immintrin.h"

_PS512_CONST(min1, -1.0f);
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


static inline void fabs512f(float *src, float *dst, int len)
{
    const v16sf mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_and_ps(mask, src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_and_ps(mask, src_tmp));
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

static inline void threshold512_gt_f(float *src, float *dst, float value, int len)
{
    v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
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

static inline void threshold512_lt_f(float *src, float *dst, float value, int len)
{
    v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
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
        sincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

v16sf atan512f_ps(v16sf xx, const v16sf positive_mask, const v16sf negative_mask)
{
    v16sf x, y, z;
    __mmask16 sign2;
    __mmask16 suptan3pi8, inftan3pi8suppi8;
    v16sf tmp;

    x = _mm512_and_ps(positive_mask, xx);
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

    y = _mm512_mask_blend_ps(sign2, y, _mm512_xor_ps(negative_mask, y));

    return (y);
}

static inline void atan512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    const v16sf positive_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const v16sf negative_mask = _mm512_castsi512_ps(_mm512_set1_epi32(~0x7FFFFFFF));

    if (((uintptr_t)(const void *) (src) % AVX512_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, atan512f_ps(src_tmp, positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, atan512f_ps(src_tmp, positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}
