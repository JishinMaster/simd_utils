/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <stdint.h>
#include "immintrin.h"

static inline void set512d(double *dst, double value, int len)
{
    const v8sd tmp = _mm512_set1_pd(value);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zero512d(double *dst, int len)
{
    const v8sd tmp = _mm512_setzero_pd();

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 0.0;
    }
}

static inline void copy512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

// TODO : "Immediate add/mul?"
static inline void addc512d(double *src, double value, double *dst, int len)
{
    const v8sd tmp = _mm512_set1_pd(value);

    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (_b), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (_c), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_load_pd(_a + i);
            v8sd b = _mm512_load_pd(_b + i);
            v8sd c = _mm512_load_pd(_c + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
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

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_c), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_load_pd(_a + i);
            v8sd c = _mm512_load_pd(_c + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
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

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_loadu_pd(_a + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
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

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_b), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd a = _mm512_load_pd(_a + i);
            v8sd b = _mm512_load_pd(_b + i);
            _mm512_store_pd(dst + i, _mm512_fmadd_pd_custom(a, b, c));
        }
    } else {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    v8sd slope16_vec = _mm512_set1_pd(16.0 * slope);
    v8sd curVal = _mm512_add_pd(_mm512_set1_pd(offset), coef);
    v8sd curVal2 = _mm512_add_pd(_mm512_set1_pd(offset), coef);
    curVal2 = _mm512_add_pd(curVal2, _mm512_set1_pd(8.0 * slope));
    int stop_len = len / (2 * AVX512_LEN_DOUBLE);
    stop_len *= (2 * AVX512_LEN_DOUBLE);

    if (len >= AVX512_LEN_DOUBLE) {
        if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
            _mm512_store_pd(dst + 0, curVal);
            _mm512_store_pd(dst + AVX512_LEN_DOUBLE, curVal2);
        } else {
            _mm512_storeu_pd(dst + 0, curVal);
            _mm512_storeu_pd(dst + AVX512_LEN_DOUBLE, curVal2);
        }

        if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (double) i;
    }
}

static inline void cplxtoreal512d(complex64_t *src, double *dstRe, double *dstIm, int len)
{
    int stop_len = 2 * len / (4 * AVX512_LEN_DOUBLE);
    stop_len *= 4 * AVX512_LEN_DOUBLE;

    int j = 0;
    if (areAligned3((uintptr_t) (src), (uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_DOUBLE) {
            v8sdx2 vec1 = _mm512_load2_pd((double const *) (src) + i);
            v8sdx2 vec2 = _mm512_load2_pd((double const *) (src) + i + 2 * AVX512_LEN_DOUBLE);
            _mm512_store_pd(dstRe + j, vec1.val[0]);
            _mm512_store_pd(dstIm + j, vec1.val[1]);
            _mm512_store_pd(dstRe + j + AVX512_LEN_DOUBLE, vec2.val[0]);
            _mm512_store_pd(dstIm + j + AVX512_LEN_DOUBLE, vec2.val[1]);
            j += 2 * AVX512_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_DOUBLE) {
            v8sdx2 vec1 = _mm512_load2u_pd((double const *) (src) + i);
            v8sdx2 vec2 = _mm512_load2u_pd((double const *) (src) + i + 2 * AVX512_LEN_DOUBLE);
            _mm512_storeu_pd(dstRe + j, vec1.val[0]);
            _mm512_storeu_pd(dstIm + j, vec1.val[1]);
            _mm512_storeu_pd(dstRe + j + AVX512_LEN_DOUBLE, vec2.val[0]);
            _mm512_storeu_pd(dstIm + j + AVX512_LEN_DOUBLE, vec2.val[1]);
            j += 2 * AVX512_LEN_DOUBLE;
        }
    }

    for (int i = j; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
}

static inline void realtocplx512d(double *srcRe, double *srcIm, complex64_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_DOUBLE);
    stop_len *= 2 * AVX512_LEN_DOUBLE;

    int j = 0;
    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_DOUBLE) {
            v8sd re = _mm512_load_pd(srcRe + i);
            v8sd im = _mm512_load_pd(srcIm + i);
            v8sd re2 = _mm512_load_pd(srcRe + i + AVX512_LEN_DOUBLE);
            v8sd im2 = _mm512_load_pd(srcIm + i + AVX512_LEN_DOUBLE);
            v8sdx2 reim = {{re, im}};
            v8sdx2 reim2 = {{re2, im2}};
            _mm512_store2_pd((double *) (dst) + j, reim);
            _mm512_store2_pd((double *) (dst) + j + 2 * AVX512_LEN_DOUBLE, reim2);
            j += 4 * AVX512_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_DOUBLE) {
            v8sd re = _mm512_loadu_pd(srcRe + i);
            v8sd im = _mm512_loadu_pd(srcIm + i);
            v8sd re2 = _mm512_loadu_pd(srcRe + i + AVX512_LEN_DOUBLE);
            v8sd im2 = _mm512_loadu_pd(srcIm + i + AVX512_LEN_DOUBLE);
            v8sdx2 reim = {{re, im}};
            v8sdx2 reim2 = {{re2, im2}};
            _mm512_store2u_pd((double *) (dst) + j, reim);
            _mm512_store2u_pd((double *) (dst) + j + 2 * AVX512_LEN_DOUBLE, reim2);
            j += 4 * AVX512_LEN_DOUBLE;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
}

static inline v8sd asin512_pd(v8sd x)
{
    v8sd a, z;
    //__mmask8 sign;
    v8sd sign;
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

    a = _mm512_and_pd(*(v8sd *) _pd512_positive_mask, x);  // fabs(x)
    // sign = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_LT_OS);  // 0xFFFFFFFF if x < 0.0
    sign = _mm512_and_pd(x, *(v8sd *) _pd512_sign_mask);

    ainfem8 = _mm512_cmp_pd_mask(a, _mm512_set1_pd(1.0e-8), _CMP_LT_OS);  // if( a < 1.0e-8)
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

    // second branch
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

    z = _mm512_mask_add_pd(z_second_branch, asup0p625, z_first_branch, *(v8sd *) _pd512_PIO4);
    z = _mm512_xor_pd(z, sign);
    z = _mm512_mask_blend_pd(ainfem8, z, x);

    // if (x > 1.0) then return 0.0
    z = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, *(v8sd *) _pd512_1, _CMP_GT_OS), z, _mm512_setzero_pd());

    return (z);
}

static inline void asin512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    //__mmask8 sign;
    v8sd sign;
    __mmask8 suptan3pi8, inftan3pi8inf0p66;  // > T3PI8 or (< T3PI8 and > 0.66)
    __mmask8 flagand1, flagand2;
    v8sd tmp, tmp2, tmp3;
    v8sd zerop66 = _mm512_set1_pd(0.66);
    v8sd flag = _mm512_setzero_pd();  // flag = 0

    x = _mm512_and_pd(*(v8sd *) _pd512_positive_mask, xx);  // x = fabs(xx)
    // sign = _mm512_cmp_pd_mask(xx, _mm512_setzero_pd(), _CMP_LT_OS);  // 0xFFFFFFFFFFFFFFFF if x < 0.0, sign = -1
    sign = _mm512_and_pd(xx, *(v8sd *) _pd512_sign_mask);

    /* range reduction */

    y = _mm512_setzero_pd();
    suptan3pi8 = _mm512_cmp_pd_mask(x, *(v8sd *) _pd512_TAN3PI8, _CMP_GT_OS);  // if( x > tan 3pi/8 )
    x = _mm512_mask_div_pd(x, suptan3pi8, *(v8sd *) _pd512_min1, x);           // if( x > tan 3pi/8 ) then x = -1.0/x
    y = _mm512_mask_blend_pd(suptan3pi8, y, *(v8sd *) _pd512_PIO2);            // if( x > tan 3pi/8 ) then y = PI/2
    flag = _mm512_mask_blend_pd(suptan3pi8, flag, *(v8sd *) _pd512_1);         // if( x > tan 3pi/8 ) then flag = 1

    inftan3pi8inf0p66 = _kand_mask8(_mm512_cmp_pd_mask(x, *(v8sd *) _pd512_TAN3PI8, _CMP_LE_OS), _mm512_cmp_pd_mask(x, zerop66, _CMP_LE_OS));  // if( x <= tan 3pi/8 ) && (x <= 0.66)
    y = _mm512_mask_blend_pd(inftan3pi8inf0p66, *(v8sd *) _pd512_PIO4, y);                                                                     // y = 0 or PIO4
    tmp2 = _mm512_add_pd(x, *(v8sd *) _pd512_1);
    tmp3 = _mm512_sub_pd(x, *(v8sd *) _pd512_1);
    x = _mm512_mask_blend_pd(inftan3pi8inf0p66, _mm512_div_pd(tmp3, tmp2), x);
    flag = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(*(v8sd *) _pd512_PIO4, y, _CMP_EQ_OS), flag, *(v8sd *) _pd512_2);  // if y = PIO4 then flag = 2

    z = _mm512_mul_pd(x, x);  // z = x*x

    // z = z * polevl(z, P_, 4)
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
    flagand1 = _mm512_cmp_pd_mask(flag, *(v8sd *) _pd512_1, _CMP_EQ_OS);
    flagand2 = _mm512_cmp_pd_mask(flag, *(v8sd *) _pd512_2, _CMP_EQ_OS);

    // TODO : could we compute 0.5*MOREBITS?
    tmp = _mm512_fmadd_pd_custom(*(v8sd *) _pd512_0p5, *(v8sd *) _pd512_MOREBITS, z);
    z = _mm512_mask_blend_pd(flagand2, z, tmp);                         // if (flag == 2) then z += 0.5 * MOREBITS
    z = _mm512_mask_add_pd(z, flagand1, z, *(v8sd *) _pd512_MOREBITS);  // if (flag == 1) then z +=  MOREBITS

    y = _mm512_add_pd(y, z);
    y = _mm512_xor_pd(y, sign);
    y = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OS), y, xx);  // if (xx == 0) then return xx (x is fabs(xx))
    return (y);
}

static inline v8sd atan2512_pd(v8sd y, v8sd x)
{
    v8sd z, w, tmp;
    __mmask8 xinfzero, yinfzero, xeqzero, yeqzero;
    __mmask8 xeqzeroandyinfzero, yeqzeroandxinfzero;
    __mmask8 specialcase;

    xinfzero = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_LT_OS);  // code =2
    yinfzero = _mm512_cmp_pd_mask(y, _mm512_setzero_pd(), _CMP_LT_OS);  // code = code |1;

    xeqzero = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OS);
    yeqzero = _mm512_cmp_pd_mask(y, _mm512_setzero_pd(), _CMP_EQ_OS);

    z = *(v8sd *) _pd512_PIO2F;

    xeqzeroandyinfzero = _kand_mask8(xeqzero, yinfzero);
    z = _mm512_mask_blend_pd(xeqzeroandyinfzero, z, *(v8sd *) _pd512_mPIO2F);
    z = _mm512_mask_blend_pd(yeqzero, z, _mm512_setzero_pd());

    yeqzeroandxinfzero = _kand_mask8(yeqzero, xinfzero);
    z = _mm512_mask_blend_pd(yeqzeroandxinfzero, z, *(v8sd *) _pd512_PIF);

    specialcase = _kor_mask8(xeqzero, yeqzero);

    w = _mm512_setzero_pd();
    w = _mm512_mask_blend_pd(_kandn_mask8(yinfzero, xinfzero), w, *(v8sd *) _pd512_PIF);  // y >= 0 && x<0
    w = _mm512_mask_blend_pd(_kand_mask8(yinfzero, xinfzero), w, *(v8sd *) _pd512_mPIF);  // y < 0 && x<0

    tmp = atan512_pd(_mm512_div_pd(y, x));
    z = _mm512_mask_blend_pd(specialcase, _mm512_add_pd(w, tmp), z);  // atanf(y/x) if not in special case

    return (z);
}

static inline void atan512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

static inline void atan2512d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_store_pd(dst + i, atan2512_pd(_mm512_load_pd(src1 + i), _mm512_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            _mm512_storeu_pd(dst + i, atan2512_pd(_mm512_loadu_pd(src1 + i), _mm512_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2(src1[i], src2[i]);
    }
}

static inline void atan2512d_interleaved(complex64_t *src, double *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_DOUBLE);
    stop_len *= 2 * AVX512_LEN_DOUBLE;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_DOUBLE) {
            v8sdx2 src_split = _mm512_load2_pd((double *) (src) + j);
            v8sdx2 src_split2 = _mm512_load2_pd((double *) (src) + j + 2 * AVX512_LEN_DOUBLE);
            _mm512_store_pd(dst + i, atan2512_pd(src_split.val[1], src_split.val[0]));
            _mm512_store_pd(dst + i + AVX512_LEN_DOUBLE, atan2512_pd(src_split2.val[1], src_split2.val[0]));
            j += 4 * AVX512_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_DOUBLE) {
            v8sdx2 src_split = _mm512_load2u_pd((double *) (src) + j);
            v8sdx2 src_split2 = _mm512_load2u_pd((double *) (src) + j + 2 * AVX512_LEN_DOUBLE);
            _mm512_storeu_pd(dst + i, atan2512_pd(src_split.val[1], src_split.val[0]));
            _mm512_storeu_pd(dst + i + AVX512_LEN_DOUBLE, atan2512_pd(src_split2.val[1], src_split2.val[0]));
            j += 4 * AVX512_LEN_DOUBLE;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2(src[i].im, src[i].re);
    }
}

//_mm512_cvttpd_epi64 gives wrong result??
static inline v8sid _mm512_cvttpd_epi64_custom(v8sd x)
{
    x = _mm512_add_pd(x, *(v8sd *) _pd512_PDEPI64U);
    return _mm512_xor_si512(
        _mm512_castpd_si512(x),
        _mm512_castpd_si512(*(v8sd *) _pd512_PDEPI64U));
}

static inline v8sd _mm512_cvtepi64_pd_custom(v8sid x)
{
    x = _mm512_or_si512(x, _mm512_castpd_si512(*(v8sd *) _pd512_PDEPI64U));
    return _mm512_sub_pd(_mm512_castsi512_pd(x), *(v8sd *) _pd512_PDEPI64U);
}

static inline void sincos512_pd(v8sd x, v8sd *s, v8sd *c)
{
    v8sd xmm1, xmm2, xmm3 = _mm512_setzero_pd(), sign_bit_sin, y;
    v8sid emm0, emm2, emm4;
    __mmask8 cmpeq_mask;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm512_and_pd(x, *(v8sd *) _pd512_inv_sign_mask);

    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm512_and_pd(sign_bit_sin, *(v8sd *) _pd512_sign_mask);

    /* scale by 4/Pi */
    y = _mm512_mul_pd(x, *(v8sd *) _pd512_cephes_FOPI);
    y = _mm512_roundscale_pd(y, ROUNDTOFLOOR);

    /* store the integer part of y in emm2 */
    emm2 = _mm512_cvttpd_epi64_custom(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm512_add_epi64(emm2, *(v8sid *) _pi512_64_1);
    emm2 = _mm512_and_si512(emm2, *(v8sid *) _pi512_64_inv1);
    y = _mm512_cvtepi64_pd_custom(emm2);
    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0 = _mm512_and_si512(emm2, *(v8sid *) _pi512_64_4);
    emm0 = _mm512_slli_epi64(emm0, 61);
    v8sd swap_sign_bit_sin = _mm512_castsi512_pd(emm0);
    /* get the polynom selection mask for the sine*/
    emm2 = _mm512_and_si512(emm2, *(v8sid *) _pi512_64_2);
    cmpeq_mask = _mm512_cmpeq_epi64_mask(emm2, _mm512_setzero_si512());
    v8sid ffff = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    emm2 = _mm512_mask_blend_epi64(cmpeq_mask, _mm512_setzero_si512(), ffff);

#if 1
    // Cast integer 0000 FFFF (negative int) to mmask type. Is there a better way?
    __mmask8 poly_mask = _mm512_cmplt_epi64_mask(emm2, _mm512_setzero_si512());
#else
    v8sd poly_mask = _mm512_castsi512_pd(emm2);
#endif



    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm512_fmadd_pd_custom(y, *(v8sd *) _pd512_minus_cephes_DP1, x);
    x = _mm512_fmadd_pd_custom(y, *(v8sd *) _pd512_minus_cephes_DP2, x);
    x = _mm512_fmadd_pd_custom(y, *(v8sd *) _pd512_minus_cephes_DP3, x);

    emm4 = _mm512_sub_epi64(emm4, *(v8sid *) _pi512_64_2);
    emm4 = _mm512_andnot_si512(emm4, *(v8sid *) _pi512_64_4);
    emm4 = _mm512_slli_epi64(emm4, 61);
    v8sd sign_bit_cos = _mm512_castsi512_pd(emm4);

    sign_bit_sin = _mm512_xor_pd(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v8sd z = _mm512_mul_pd(x, x);
    y = _mm512_fmadd_pd_custom(*(v8sd *) _pd512_coscof_p0, z, *(v8sd *) _pd512_coscof_p1);
    y = _mm512_fmadd_pd_custom(y, z, *(v8sd *) _pd512_coscof_p2);
    y = _mm512_fmadd_pd_custom(y, z, *(v8sd *) _pd512_coscof_p3);
    y = _mm512_fmadd_pd_custom(y, z, *(v8sd *) _pd512_coscof_p4);
    y = _mm512_fmadd_pd_custom(y, z, *(v8sd *) _pd512_coscof_p5);
    y = _mm512_mul_pd(y, z);
    y = _mm512_mul_pd(y, z);
    y = _mm512_fnmadd_pd_custom(z, *(v8sd *) _pd512_0p5, y);
    y = _mm512_add_pd(y, *(v8sd *) _pd512_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    v8sd y2 = _mm512_fmadd_pd_custom(*(v8sd *) _pd512_sincof_p0, z, *(v8sd *) _pd512_sincof_p1);
    y2 = _mm512_fmadd_pd_custom(y2, z, *(v8sd *) _pd512_sincof_p2);
    y2 = _mm512_fmadd_pd_custom(y2, z, *(v8sd *) _pd512_sincof_p3);
    y2 = _mm512_fmadd_pd_custom(y2, z, *(v8sd *) _pd512_sincof_p4);
    y2 = _mm512_fmadd_pd_custom(y2, z, *(v8sd *) _pd512_sincof_p5);
    y2 = _mm512_mul_pd(y2, z);
    y2 = _mm512_fmadd_pd_custom(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    xmm1 = _mm512_mask_blend_pd(poly_mask, y, y2);
    xmm2 = _mm512_mask_blend_pd(poly_mask, y2, y);
#else
    v8sd ysin2 = _mm512_and_pd(poly_mask, y2);
    v8sd ysin1 = _mm512_andnot_pd(poly_mask, y);
    y2 = _mm512_sub_pd(y2, ysin2);
    y = _mm512_sub_pd(y, ysin1);
    xmm1 = _mm512_add_pd(ysin1, ysin2);
    xmm2 = _mm512_add_pd(y, y2);
#endif

    /* update the sign */
    *s = _mm512_xor_pd(xmm1, sign_bit_sin);
    *c = _mm512_xor_pd(xmm2, sign_bit_cos);
}

static inline void sincos512d(double *src, double *dst_sin, double *dst_cos, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            v8sd dst_sin_tmp;
            v8sd dst_cos_tmp;
            sincos512_pd(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm512_store_pd(dst_sin + i, dst_sin_tmp);
            _mm512_store_pd(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            v8sd dst_sin_tmp;
            v8sd dst_cos_tmp;
            sincos512_pd(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm512_storeu_pd(dst_sin + i, dst_sin_tmp);
            _mm512_storeu_pd(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst_sin[i] = sin(src[i]);
        dst_cos[i] = cos(src[i]);
    }
}

static inline void sincos512d_interleaved(double *src, complex64_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            v8sdx2 dst_tmp;
            sincos512_pd(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm512_store2_pd((double *) dst + j, dst_tmp);
            j += 2 * AVX512_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            v8sdx2 dst_tmp;
            sincos512_pd(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm512_store2u_pd((double *) dst + j, dst_tmp);
            j += 2 * AVX512_LEN_DOUBLE;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].im = sin(src[i]);
        dst[i].re = cos(src[i]);
    }
}

static inline void pol2cart2D512f_precise(float *r, float *theta, float *x, float *y, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf r_tmpf = _mm256_load_ps(r + i);
            v8sf theta_tmpf = _mm256_load_ps(theta + i);
            v8sd r_tmp = _mm512_cvtps_pd(r_tmpf);
            v8sd theta_tmp = _mm512_cvtps_pd(theta_tmpf);
            v8sd sin_tmp;
            v8sd cos_tmp;
            sincos512_pd(theta_tmp, &sin_tmp, &cos_tmp);
            v8sd x_tmpd = _mm512_mul_pd(r_tmp, cos_tmp);
            v8sd y_tmpd = _mm512_mul_pd(r_tmp, sin_tmp);
            v8sf x_tmp = _mm512_cvtpd_ps(x_tmpd);
            v8sf y_tmp = _mm512_cvtpd_ps(y_tmpd);
            _mm256_store_ps(x + i, x_tmp);
            _mm256_store_ps(y + i, y_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf r_tmpf = _mm256_loadu_ps(r + i);
            v8sf theta_tmpf = _mm256_loadu_ps(theta + i);
            v8sd r_tmp = _mm512_cvtps_pd(r_tmpf);
            v8sd theta_tmp = _mm512_cvtps_pd(theta_tmpf);
            v8sd sin_tmp;
            v8sd cos_tmp;
            sincos512_pd(theta_tmp, &sin_tmp, &cos_tmp);
            v8sd x_tmpd = _mm512_mul_pd(r_tmp, cos_tmp);
            v8sd y_tmpd = _mm512_mul_pd(r_tmp, sin_tmp);
            v8sf x_tmp = _mm512_cvtpd_ps(x_tmpd);
            v8sf y_tmp = _mm512_cvtpd_ps(y_tmpd);
            _mm256_storeu_ps(x + i, x_tmp);
            _mm256_storeu_ps(y + i, y_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        double sin_tmp, cos_tmp;
        double theta_double = (double) theta[i];
        double r_double = (double) r[i];
        sin_tmp = sin(theta_double);
        cos_tmp = cos(theta_double);
        x[i] = (float) (r_double * cos_tmp);
        y[i] = (float) (r_double * sin_tmp);
    }
}

static inline void cart2pol2D512f_precise(float *x, float *y, float *r, float *theta, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf x_tmpf = _mm256_load_ps(x + i);
            v8sf y_tmpf = _mm256_load_ps(y + i);
            v8sd x_tmp = _mm512_cvtps_pd(x_tmpf);
            v8sd y_tmp = _mm512_cvtps_pd(y_tmpf);
            v8sd y_square = _mm512_mul_pd(y_tmp, y_tmp);
            v8sd r_tmpd = _mm512_fmadd_pd_custom(x_tmp, x_tmp, y_square);
            r_tmpd = _mm512_sqrt_pd(r_tmpd);
            v8sd theta_tmpd = atan2512_pd(y_tmp, x_tmp);

            v8sf r_tmp = _mm512_cvtpd_ps(r_tmpd);
            v8sf theta_tmp = _mm512_cvtpd_ps(theta_tmpd);
            _mm256_store_ps(r + i, r_tmp);
            _mm256_store_ps(theta + i, theta_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf x_tmpf = _mm256_loadu_ps(x + i);
            v8sf y_tmpf = _mm256_loadu_ps(y + i);
            v8sd x_tmp = _mm512_cvtps_pd(x_tmpf);
            v8sd y_tmp = _mm512_cvtps_pd(y_tmpf);
            v8sd y_square = _mm512_mul_pd(y_tmp, y_tmp);
            v8sd r_tmpd = _mm512_fmadd_pd_custom(x_tmp, x_tmp, y_square);
            r_tmpd = _mm512_sqrt_pd(r_tmpd);
            v8sd theta_tmpd = atan2512_pd(y_tmp, x_tmp);

            v8sf r_tmp = _mm512_cvtpd_ps(r_tmpd);
            v8sf theta_tmp = _mm512_cvtpd_ps(theta_tmpd);
            _mm256_storeu_ps(r + i, r_tmp);
            _mm256_storeu_ps(theta + i, theta_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        double y_double = (double) y[i];
        double x_double = (double) x[i];
        double y_square = y_double * y_double;
        r[i] = (float) sqrt(x_double * x_double + y_square);
        theta[i] = (float) atan2(y_double, x_double);
    }
}

static inline v8sd tan512_pd(v8sd xx)
{
    __mmask8 xxeqzero, zzsup1m14;
    v8sd ysup1m14;
    v8sd tmp, tmp2;

    xxeqzero = _mm512_cmp_pd_mask(xx, _mm512_setzero_pd(), _CMP_EQ_OS);

    v8sd x, y, z, zz;
    v8sid j;
    __mmask8 jandone, jandtwo;
    v8sd sign;

    /* make argument positive but save the sign */
    x = xx;
    x = _mm512_and_pd(x, *(v8sd *) _pd512_inv_sign_mask);
    sign = _mm512_and_pd(xx, *(v8sd *) _pd512_sign_mask);
#ifdef LOSSTH
    __mmask8 xsuplossth = _mm512_cmp_pd_mask(x, *(v8sd *) _pd512_tanlossth, _CMP_GT_OS);
#endif

    /* compute x mod PIO4 */
    y = _mm512_mul_pd(x, *(v8sd *) _pd512_cephes_FOPI);
    // useful?
    y = _mm512_roundscale_pd(y, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    /* strip high bits of integer part */
    z = _mm512_mul_pd(y, *(v8sd *) _pd512_0p125);
    // useful?
    z = _mm512_roundscale_pd(z, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    z = _mm512_fmadd_pd_custom(z, *(v8sd *) _pd512_min8, y);

    /* integer and fractional part modulo one octant */
    j = _mm512_cvttpd_epi64_custom(z);

    /* map zeros and singularities to origin */
    jandone = _mm512_cmpgt_epi64_mask(_mm512_and_si512(j, *(v8sid *) _pi512_64_1), _mm512_setzero_si512());
    j = _mm512_mask_add_epi64(j, jandone, j, *(v8sid *) _pi512_64_1);
    y = _mm512_mask_add_pd(y, jandone, y, *(v8sd *) _pd512_1);
    jandtwo = _mm512_cmpgt_epi64_mask(_mm512_and_si512(j, *(v8sid *) _pi512_64_2), _mm512_setzero_si512());

    z = _mm512_fmadd_pd_custom(y, *(v8sd *) _pd512_TAN_mDP1, x);
    z = _mm512_fmadd_pd_custom(y, *(v8sd *) _pd512_TAN_mDP2, z);
    z = _mm512_fmadd_pd_custom(y, *(v8sd *) _pd512_TAN_mDP3, z);
    zz = _mm512_mul_pd(z, z);

    zzsup1m14 = _mm512_cmp_pd_mask(zz, *(v8sd *) _pd512_1m14, _CMP_GT_OS);
    tmp = _mm512_fmadd_pd_custom(zz, *(v8sd *) _pd512_TAN_P0, *(v8sd *) _pd512_TAN_P1);
    tmp = _mm512_fmadd_pd_custom(zz, tmp, *(v8sd *) _pd512_TAN_P2);
    tmp2 = _mm512_add_pd(zz, *(v8sd *) _pd512_TAN_Q0);
    tmp2 = _mm512_fmadd_pd_custom(zz, tmp2, *(v8sd *) _pd512_TAN_Q1);
    tmp2 = _mm512_fmadd_pd_custom(zz, tmp2, *(v8sd *) _pd512_TAN_Q2);
    tmp2 = _mm512_fmadd_pd_custom(zz, tmp2, *(v8sd *) _pd512_TAN_Q3);
    tmp2 = _mm512_div_pd(tmp, tmp2);
    tmp2 = _mm512_mul_pd(zz, tmp2);
    ysup1m14 = _mm512_fmadd_pd_custom(z, tmp2, z);
    y = _mm512_mask_blend_pd(zzsup1m14, z, ysup1m14);

    y = _mm512_mask_div_pd(y, jandtwo, *(v8sd *) _pd512_min1, y);
    y = _mm512_xor_pd(y, sign);

#ifdef LOSSTH
    y = _mm512_mask_blend_pd(xsuplossth, y, _mm512_setzero_pd());
#endif
    y = _mm512_mask_blend_pd(xxeqzero, y, xx);
    return y;
}

static inline void tan512d(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, tan512_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, tan512_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tan(src[i]);
    }
}
