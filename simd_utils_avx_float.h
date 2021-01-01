/*
 * Project : SIMD_Utils
 * Version : 0.1.4
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

static inline void ln_256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
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

static inline void fabs256f(float *src, float *dst, int len)
{
    const v8sf mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_and_ps(mask, src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_and_ps(mask, src_tmp));
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
static inline void vectorSlope256f(float *dst, int len, float offset, float slope)
{
    v8sf coef = _mm256_set_ps(7.0f * slope, 6.0f * slope, 5.0f * slope, 4.0f * slope, 3.0f * slope, 2.0f * slope, slope, 0.0f);
    v8sf slope8_vec = _mm256_set1_ps(8.0f * slope);
    v8sf curVal = _mm256_add_ps(_mm256_set1_ps(offset), coef);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (dst) % AVX_LEN_BYTES) == 0) {
        _mm256_store_ps(dst + 0, curVal);
    } else {
        _mm256_storeu_ps(dst + 0, curVal);
    }

    if (((uintptr_t)(const void *) (dst) % AVX_LEN_BYTES) == 0) {
        for (int i = AVX_LEN_FLOAT; i < stop_len; i += AVX_LEN_FLOAT) {
            curVal = _mm256_add_ps(curVal, slope8_vec);
            _mm256_storeu_ps(dst + i, curVal);
        }
    } else {
        for (int i = AVX_LEN_FLOAT; i < stop_len; i += AVX_LEN_FLOAT) {
            curVal = _mm256_add_ps(curVal, slope8_vec);
            _mm256_storeu_ps(dst + i, curVal);
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
void cplxtoreal256f(float *src, float *dstRe, float *dstIm, int len)
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
            _mm256_storeu_ps(dst + j, cplx0);
            _mm256_storeu_ps(dst + j + AVX_LEN_FLOAT, cplx1);
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

static inline void threshold256_gt_f(float *src, float *dst, float value, int len)
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

static inline void threshold256_lt_f(float *src, float *dst, float value, int len)
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
        sincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

#ifndef __AVX2__  // Needs AVX2 to  get _mm256_cmpgt_epi32
#warning "Using SSE2 to perform AVX2 integer ops"
AVX2_INTOP_USING_SSE2(cmpgt_epi32)
#endif

v8sf atan256f_ps(v8sf xx, const v8sf positive_mask, const v8sf negative_mask)
{
    v8sf x, y, z;
    v8sf sign2;
    v8sf suptan3pi8, inftan3pi8suppi8;
    v8sf tmp;

    x = _mm256_and_ps(positive_mask, xx);
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

    y = _mm256_blendv_ps(y, _mm256_xor_ps(negative_mask, y), sign2);

    return (y);
}

static inline void atan256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    const v8sf positive_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const v8sf negative_mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7FFFFFFF));

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, atan256f_ps(src_tmp, positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, atan256f_ps(src_tmp, positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

v8sf atan2256f_ps(v8sf y, v8sf x, const v8sf positive_mask, const v8sf negative_mask)
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

    z = _mm256_blendv_ps(_mm256_add_ps(w, atan256f_ps(_mm256_div_ps(y, x), positive_mask, negative_mask)), z, specialcase);  // atanf(y/x) if not in special case

    return (z);
}

static inline void atan2256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    const v8sf positive_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const v8sf negative_mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7FFFFFFF));

    if (((uintptr_t)(const void *) (src1) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, atan2256f_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i), positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, atan2256f_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i), positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}

v8sf asin256f_ps(v8sf xx, const v8sf positive_mask, const v8sf negative_mask)
{
    v8sf a, x, z, z_tmp;
    v8sf sign;
    v8sf ainfem4, asup0p5;
    v8sf tmp;
    x = xx;
    a = _mm256_and_ps(positive_mask, x);                       //fabs(x)
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

    z_tmp = _mm256_add_ps(z, z);
    z_tmp = _mm256_sub_ps(*(v8sf *) _ps256_PIO2F, z_tmp);
    z = _mm256_blendv_ps(z, z_tmp, asup0p5);

    //done:
    z = _mm256_blendv_ps(z, a, ainfem4);
    z = _mm256_blendv_ps(z, _mm256_xor_ps(negative_mask, z), sign);

    return (z);
}

static inline void asin256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    const v8sf positive_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const v8sf negative_mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7FFFFFFF));

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, asin256f_ps(src_tmp, positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, asin256f_ps(src_tmp, positive_mask, negative_mask));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}


#if 1
v8sf tan256f_ps(v8sf xx, const v8sf positive_mask, const v8sf negative_mask)
{
    v8sf x, y, z, zz;
    v8si j;  //long?
    v8sf sign, xsupem4;
    v8sf tmp;
    v8si jandone, jandtwo;

    x = _mm256_and_ps(positive_mask, xx);  //fabs(xx)

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
    y = _mm256_blendv_ps(y, _mm256_xor_ps(negative_mask, y), sign);

    return (y);
}

static inline void tan256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    const v8sf positive_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const v8sf negative_mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7FFFFFFF));

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, tan256f_ps(src_tmp, positive_mask, negative_mask));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, tan256f_ps(src_tmp, positive_mask, negative_mask));
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
        dst[i] = src[i] + value;
    }
}

static inline void sum256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v8sf vec_acc = _mm256_setzero_ps();  //initialize the vector accumulator
    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf vec_tmp = _mm256_load_ps(src + i);
            vec_acc = _mm256_add_ps(vec_acc, vec_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf vec_tmp = _mm256_loadu_ps(src + i);
            vec_acc = _mm256_add_ps(vec_acc, vec_tmp);
        }
    }

    _mm256_store_ps(accumulate, vec_acc);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];

    *dst = tmp_acc;
}


static inline void mean256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v8sf vec_acc = _mm256_setzero_ps();  //initialize the vector accumulator
    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf vec_tmp = _mm256_load_ps(src + i);
            vec_acc = _mm256_add_ps(vec_acc, vec_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf vec_tmp = _mm256_loadu_ps(src + i);
            vec_acc = _mm256_add_ps(vec_acc, vec_tmp);
        }
    }

    _mm256_store_ps(accumulate, vec_acc);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];
    tmp_acc /= (float) len;

    *dst = tmp_acc;
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
