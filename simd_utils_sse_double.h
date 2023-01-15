/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#ifndef ARM
#include <immintrin.h>
#else
#include "sse2neon_wrapper.h"
#endif

static inline void set128d(double *dst, double value, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zero128d(double *dst, int len)
{
    const v2sd tmp = _mm_setzero_pd();

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 0.0;
    }
}

static inline void copy128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_load_pd(src + i));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_loadu_pd(src + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void sqrt128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_sqrt_pd(_mm_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_sqrt_pd(_mm_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrt(src[i]);
    }
}

static inline void add128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_add_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_add_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void mul128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_mul_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_mul_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_sub_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_sub_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void div128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_div_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_div_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

// TODO : "Immediate add/mul?"
static inline void addc128d(double *src, double value, double *dst, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_add_pd(tmp, _mm_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_add_pd(tmp, _mm_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc128d(double *src, double value, double *dst, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, _mm_mul_pd(tmp, _mm_load_pd(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, _mm_mul_pd(tmp, _mm_loadu_pd(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd128d(double *_a, double *_b, double *_c, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (_b), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (_c), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_load_pd(_a + i);
            v2sd b = _mm_load_pd(_b + i);
            v2sd c = _mm_load_pd(_c + i);
            _mm_store_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_loadu_pd(_a + i);
            v2sd b = _mm_loadu_pd(_b + i);
            v2sd c = _mm_loadu_pd(_c + i);
            _mm_storeu_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcadd128d(double *_a, double _b, double *_c, double *dst, int len)
{
    v2sd b = _mm_set1_pd(_b);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_c), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_load_pd(_a + i);
            v2sd c = _mm_load_pd(_c + i);
            _mm_store_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_loadu_pd(_a + i);
            v2sd c = _mm_loadu_pd(_c + i);
            _mm_storeu_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddc128d(double *_a, double _b, double _c, double *dst, int len)
{
    v2sd b = _mm_set1_pd(_b);
    v2sd c = _mm_set1_pd(_c);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_load_pd(_a + i);
            _mm_store_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_loadu_pd(_a + i);
            _mm_storeu_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
    }
}

static inline void muladdc128d(double *_a, double *_b, double _c, double *dst, int len)
{
    v2sd c = _mm_set1_pd(_c);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_b), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_load_pd(_a + i);
            v2sd b = _mm_load_pd(_b + i);
            _mm_store_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd a = _mm_loadu_pd(_a + i);
            v2sd b = _mm_loadu_pd(_b + i);
            _mm_storeu_pd(dst + i, _mm_fmadd_pd_custom(a, b, c));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

static inline void round128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = round(src[i]);
    }
}

static inline void ceil128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceil(src[i]);
    }
}

static inline void floor128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floor(src[i]);
    }
}

static inline void trunc128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = trunc(src[i]);
    }
}

static inline void vectorSlope128d(double *dst, int len, double offset, double slope)
{
    int stop_len = len / (2 * SSE_LEN_DOUBLE);
    stop_len *= (2 * SSE_LEN_DOUBLE);

    v2sd coef = _mm_set_pd(slope, 0.0);
    v2sd slope4_vec = _mm_set1_pd(4.0 * slope);
    v2sd curVal = _mm_add_pd(_mm_set1_pd(offset), coef);
    v2sd curVal2 = _mm_add_pd(_mm_set1_pd(offset), coef);
    curVal2 = _mm_add_pd(curVal2, _mm_set1_pd(2.0 * slope));

    if (len >= SSE_LEN_DOUBLE) {
        if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
            _mm_store_pd(dst + 0, curVal);
            _mm_store_pd(dst + SSE_LEN_DOUBLE, curVal2);
        } else {
            _mm_storeu_pd(dst + 0, curVal);
            _mm_storeu_pd(dst + SSE_LEN_DOUBLE, curVal2);
        }

        if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
            for (int i = 2 * SSE_LEN_DOUBLE; i < stop_len; i += 2 * SSE_LEN_DOUBLE) {
                curVal = _mm_add_pd(curVal, slope4_vec);
                _mm_store_pd(dst + i, curVal);
                curVal2 = _mm_add_pd(curVal2, slope4_vec);
                _mm_store_pd(dst + i + SSE_LEN_DOUBLE, curVal2);
            }
        } else {
            for (int i = 2 * SSE_LEN_DOUBLE; i < stop_len; i += 2 * SSE_LEN_DOUBLE) {
                curVal = _mm_add_pd(curVal, slope4_vec);
                _mm_storeu_pd(dst + i, curVal);
                curVal2 = _mm_add_pd(curVal2, slope4_vec);
                _mm_storeu_pd(dst + i + SSE_LEN_DOUBLE, curVal2);
            }
        }
    }
    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (double) i;
    }
}

static inline void cplxtoreal128d(complex64_t *src, double *dstRe, double *dstIm, int len)
{
    int stop_len = 2 * len / (4 * SSE_LEN_DOUBLE);
    stop_len *= 4 * SSE_LEN_DOUBLE;

    int j = 0;
    if (areAligned3((uintptr_t) (src), (uintptr_t) (dstRe), (uintptr_t) (dstIm), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_DOUBLE) {
            v2sdx2 vec1 = _mm_load2_pd((double const *) (src) + i);
            v2sdx2 vec2 = _mm_load2_pd((double const *) (src) + i + 2 * SSE_LEN_DOUBLE);
            _mm_store_pd(dstRe + j, vec1.val[0]);
            _mm_store_pd(dstIm + j, vec1.val[1]);
            _mm_store_pd(dstRe + j + SSE_LEN_DOUBLE, vec2.val[0]);
            _mm_store_pd(dstIm + j + SSE_LEN_DOUBLE, vec2.val[1]);
            j += 2 * SSE_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_DOUBLE) {
            v2sdx2 vec1 = _mm_load2u_pd((double const *) (src) + i);
            v2sdx2 vec2 = _mm_load2u_pd((double const *) (src) + i + 2 * SSE_LEN_DOUBLE);
            _mm_storeu_pd(dstRe + j, vec1.val[0]);
            _mm_storeu_pd(dstIm + j, vec1.val[1]);
            _mm_storeu_pd(dstRe + j + SSE_LEN_DOUBLE, vec2.val[0]);
            _mm_storeu_pd(dstIm + j + SSE_LEN_DOUBLE, vec2.val[1]);
            j += 2 * SSE_LEN_DOUBLE;
        }
    }

    for (int i = j; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
}

static inline void realtocplx128d(double *srcRe, double *srcIm, complex64_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_DOUBLE);
    stop_len *= 2 * SSE_LEN_DOUBLE;

    int j = 0;
    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_DOUBLE) {
            v2sd re = _mm_load_pd(srcRe + i);
            v2sd im = _mm_load_pd(srcIm + i);
            v2sd re2 = _mm_load_pd(srcRe + i + SSE_LEN_DOUBLE);
            v2sd im2 = _mm_load_pd(srcIm + i + SSE_LEN_DOUBLE);
            v2sdx2 reim = {{re, im}};
            v2sdx2 reim2 = {{re2, im2}};
            _mm_store2_pd((double *) (dst) + j, reim);
            _mm_store2_pd((double *) (dst) + j + 2 * SSE_LEN_DOUBLE, reim2);
            j += 4 * SSE_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_DOUBLE) {
            v2sd re = _mm_loadu_pd(srcRe + i);
            v2sd im = _mm_loadu_pd(srcIm + i);
            v2sd re2 = _mm_loadu_pd(srcRe + i + SSE_LEN_DOUBLE);
            v2sd im2 = _mm_loadu_pd(srcIm + i + SSE_LEN_DOUBLE);
            v2sdx2 reim = {{re, im}};
            v2sdx2 reim2 = {{re2, im2}};
            _mm_store2u_pd((double *) (dst) + j, reim);
            _mm_store2u_pd((double *) (dst) + j + 2 * SSE_LEN_DOUBLE, reim2);
            j += 4 * SSE_LEN_DOUBLE;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
}


// Work in progress
// in SSE, missing _mm_cvtepi64_pd, _mm_cvttpd_epi64
// See : https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx

static inline v2sd _mm_cvtepi64_pd_custom(v2sid x)
{
#if 0
    //Signed
    x = _mm_add_epi64(x, _mm_castpd_si128(_mm_set1_pd(0x0018000000000000)));
    return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0018000000000000));
#else
    // unsigned
    x = _mm_or_si128(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
    return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0010000000000000));
#endif
}

static inline v2sid _mm_cvttpd_epi64_custom(v2sd x)
{
    // Signed
#if 0
   x = _mm_add_pd(x, _mm_set1_pd(0x0018000000000000));
    return _mm_sub_epi64(
        _mm_castpd_si128(x),
        _mm_castpd_si128(_mm_set1_pd(0x0018000000000000))
    );
#else
    // Unsigned
    x = _mm_add_pd(x, *(v2sd *) _pd_PDEPI64U);  //_mm_set1_pd(0x0010000000000000));
    return _mm_xor_si128(
        _mm_castpd_si128(x),
        _mm_castpd_si128(*(v2sd *) _pd_PDEPI64U));
#endif
}

static inline void sincos_pd(v2sd x, v2sd *s, v2sd *c)
{
    v2sd xmm1, xmm2, sign_bit_sin, y;

    v2sid emm0, emm2, emm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm_and_pd(x, *(v2sd *) _pd_inv_sign_mask);

    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm_and_pd(sign_bit_sin, *(v2sd *) _pd_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_pd(x, *(v2sd *) _pd_cephes_FOPI);
    y = _mm_round_pd(y, ROUNDTOFLOOR);
    /* strip high bits of integer part to prevent integer overflow */
    // v2sd ldexpmin4 = _mm_set1_pd(1.0/(16.0));
    // v2sd ldexp4 = _mm_set1_pd(16.0);
    v2sd z;
    /*z = _mm_mul_pd( y,  ldexpmin4);
    z = _mm_round_pd(z, ROUNDTOFLOOR);
    z = _mm_mul_pd(z, ldexp4);
    z = _mm_sub_pd(y, z); */

    /* store the integer part of y in emm2 */
    emm2 = _mm_cvttpd_epi64_custom(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi64(emm2, *(v2sid *) _pi64_1);

    emm2 = _mm_and_si128(emm2, *(v2sid *) _pi64_inv1);
    y = _mm_cvtepi64_pd_custom(emm2);
    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0 = _mm_and_si128(emm2, *(v2sid *) _pi64_4);
    // print2i(emm0);
    emm0 = _mm_slli_epi64(emm0, 61);
    // print2i(emm0);
    v2sd swap_sign_bit_sin = _mm_castsi128_pd(emm0);

    /* get the polynom selection mask for the sine*/
    emm2 = _mm_and_si128(emm2, *(v2sid *) _pi64_2);
    // SSE3
    emm2 = _mm_cmpeq_epi64(emm2, _mm_setzero_si128());
    v2sd poly_mask = _mm_castsi128_pd(emm2);
    // print2i(emm2);
    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_minus_cephes_DP1, x);
    x = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_minus_cephes_DP2, x);
    x = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_minus_cephes_DP3, x);

    emm4 = _mm_sub_epi64(emm4, *(v2sid *) _pi64_2);
    emm4 = _mm_andnot_si128(emm4, *(v2sid *) _pi64_4);
    emm4 = _mm_slli_epi64(emm4, 61);
    v2sd sign_bit_cos = _mm_castsi128_pd(emm4);

    sign_bit_sin = _mm_xor_pd(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    z = _mm_mul_pd(x, x);

    y = _mm_fmadd_pd_custom(*(v2sd *) _pd_coscof_p0, z, *(v2sd *) _pd_coscof_p1);
    y = _mm_fmadd_pd_custom(y, z, *(v2sd *) _pd_coscof_p2);
    y = _mm_fmadd_pd_custom(y, z, *(v2sd *) _pd_coscof_p3);
    y = _mm_fmadd_pd_custom(y, z, *(v2sd *) _pd_coscof_p4);
    y = _mm_fmadd_pd_custom(y, z, *(v2sd *) _pd_coscof_p5);
    y = _mm_mul_pd(y, z);
    y = _mm_mul_pd(y, z);
    y = _mm_fnmadd_pd_custom(z, *(v2sd *) _pd_0p5, y);
    y = _mm_add_pd(y, *(v2sd *) _pd_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    v2sd y2 = _mm_fmadd_pd_custom(*(v2sd *) _pd_sincof_p0, z, *(v2sd *) _pd_sincof_p1);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p2);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p3);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p4);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p5);
    y2 = _mm_mul_pd(y2, z);
    y2 = _mm_fmadd_pd_custom(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    xmm1 = _mm_blendv_pd(y, y2, poly_mask);
    xmm2 = _mm_blendv_pd(y2, y, poly_mask);
#else
    v2sd ysin2 = _mm_and_pd(poly_mask, y2);
    v2sd ysin1 = _mm_andnot_pd(poly_mask, y);
    y2 = _mm_sub_pd(y2, ysin2);
    y = _mm_sub_pd(y, ysin1);
    xmm1 = _mm_add_pd(ysin1, ysin2);
    xmm2 = _mm_add_pd(y, y2);
#endif

    /* update the sign */
    *s = _mm_xor_pd(xmm1, sign_bit_sin);
    *c = _mm_xor_pd(xmm2, sign_bit_cos);
}

static inline void sincos128d(double *src, double *dst_sin, double *dst_cos, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            v2sd dst_sin_tmp;
            v2sd dst_cos_tmp;
            sincos_pd(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm_store_pd(dst_sin + i, dst_sin_tmp);
            _mm_store_pd(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            v2sd dst_sin_tmp;
            v2sd dst_cos_tmp;
            sincos_pd(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            _mm_storeu_pd(dst_sin + i, dst_sin_tmp);
            _mm_storeu_pd(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst_sin[i] = sin(src[i]);
        dst_cos[i] = cos(src[i]);
    }
}

static inline void sincos128d_interleaved(double *src, complex64_t *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            v2sdx2 dst_tmp;
            sincos_pd(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm_store2_pd((double *) dst + j, dst_tmp);
            j += 2 * SSE_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            v2sdx2 dst_tmp;
            sincos_pd(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm_store2u_pd((double *) dst + j, dst_tmp);
            j += 2 * SSE_LEN_DOUBLE;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].im = sin(src[i]);
        dst[i].re = cos(src[i]);
    }
}

static inline v2sd asin_pd(v2sd x)
{
    v2sd a, z;
    v2sd sign;
    v2sd ainfem8, asup0p625;

    // first branch, a > 0.625
    v2sd zz_first_branch;
    v2sd p;
    v2sd z_first_branch;
    v2sd tmp_first_branch;

    // second branch a <= 0.625
    v2sd zz_second_branch;
    v2sd z_second_branch;
    v2sd tmp_second_branch;
    v2sd tmp;

    a = _mm_and_pd(*(v2sd *) _pd_positive_mask, x);  // fabs(x)
    // sign = _mm_cmplt_pd(x, _mm_setzero_pd());        // 0xFFFFFFFF if x < 0.0
    sign = _mm_and_pd(x, *(v2sd *) _pd_sign_mask);

    ainfem8 = _mm_cmplt_pd(a, *(v2sd *) _pd_1em8);  // if( a < 1.0e-8)
    asup0p625 = _mm_cmpgt_pd(a, *(v2sd *) _pd_0p625);

    // fist branch
    zz_first_branch = _mm_sub_pd(*(v2sd *) _pd_1, a);
    p = _mm_fmadd_pd_custom(*(v2sd *) _pd_ASIN_R0, zz_first_branch, *(v2sd *) _pd_ASIN_R1);
    p = _mm_fmadd_pd_custom(p, zz_first_branch, *(v2sd *) _pd_ASIN_R2);
    p = _mm_fmadd_pd_custom(p, zz_first_branch, *(v2sd *) _pd_ASIN_R3);
    p = _mm_fmadd_pd_custom(p, zz_first_branch, *(v2sd *) _pd_ASIN_R4);
    p = _mm_mul_pd(p, zz_first_branch);

    tmp_first_branch = _mm_add_pd(zz_first_branch, *(v2sd *) _pd_ASIN_S0);
    tmp_first_branch = _mm_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v2sd *) _pd_ASIN_S1);
    tmp_first_branch = _mm_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v2sd *) _pd_ASIN_S2);
    tmp_first_branch = _mm_fmadd_pd_custom(tmp_first_branch, zz_first_branch, *(v2sd *) _pd_ASIN_S3);
    p = _mm_div_pd(p, tmp_first_branch);

    zz_first_branch = _mm_sqrt_pd(_mm_add_pd(zz_first_branch, zz_first_branch));
    z_first_branch = _mm_sub_pd(*(v2sd *) _pd_PIO4, zz_first_branch);
    zz_first_branch = _mm_fmadd_pd_custom(zz_first_branch, p, *(v2sd *) _pd_minMOREBITS);
    z_first_branch = _mm_sub_pd(z_first_branch, zz_first_branch);
    z_first_branch = _mm_add_pd(z_first_branch, *(v2sd *) _pd_PIO4);

    // second branch
    zz_second_branch = _mm_mul_pd(a, a);
    z_second_branch = _mm_fmadd_pd_custom(*(v2sd *) _pd_ASIN_P0, zz_second_branch, *(v2sd *) _pd_ASIN_P1);
    z_second_branch = _mm_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_P2);
    z_second_branch = _mm_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_P3);
    z_second_branch = _mm_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_P4);
    z_second_branch = _mm_fmadd_pd_custom(z_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_P5);
    z_second_branch = _mm_mul_pd(z_second_branch, zz_second_branch);

    tmp_second_branch = _mm_add_pd(zz_second_branch, *(v2sd *) _pd_ASIN_Q0);
    tmp_second_branch = _mm_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_Q1);
    tmp_second_branch = _mm_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_Q2);
    tmp_second_branch = _mm_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_Q3);
    tmp_second_branch = _mm_fmadd_pd_custom(tmp_second_branch, zz_second_branch, *(v2sd *) _pd_ASIN_Q4);

    z_second_branch = _mm_div_pd(z_second_branch, tmp_second_branch);
    z_second_branch = _mm_fmadd_pd_custom(a, z_second_branch, a);



    z = _mm_blendv_pd(z_second_branch, z_first_branch, asup0p625);
    // z = _mm_blendv_pd(z, _mm_xor_pd(*(v2sd *) _pd_negative_mask, z), sign);
    z = _mm_xor_pd(z, sign);
    z = _mm_blendv_pd(z, x, ainfem8);

    // if (x > 1.0) then return 0.0
    tmp = _mm_cmpgt_pd(x, *(v2sd *) _pd_1);
#if 1
    z = _mm_andnot_pd(tmp, z);
#else
    z = _mm_blendv_pd(z, _mm_setzero_pd(), tmp);
#endif
    return (z);
}

static inline v2sd atan_pd(v2sd xx)
{
    v2sd x, y, z;
    v2sd sign;
    v2sd suptan3pi8, inftan3pi8inf0p66;  // > T3PI8 or (< T3PI8 and > 0.66)
    v2sd xeqzero;
    v2sd tmp, tmp2;
    v2sd flag;  // flag = 0

    x = _mm_and_pd(*(v2sd *) _pd_positive_mask, xx);  // x = fabs(xx)
    // sign = _mm_cmplt_pd(xx, _mm_setzero_pd());        // 0xFFFFFFFFFFFFFFFF if x < 0.0, sign = -1
    sign = _mm_and_pd(xx, *(v2sd *) _pd_sign_mask);

    /* range reduction */
    suptan3pi8 = _mm_cmpgt_pd(x, *(v2sd *) _pd_TAN3PI8);  // if( x > tan 3pi/8 )
    tmp = _mm_div_pd(*(v2sd *) _pd_min1, x);
    x = _mm_blendv_pd(x, tmp, suptan3pi8);  // if( x > tan 3pi/8 ) then x = -1.0/x
    tmp = _mm_cmple_pd(x, *(v2sd *) _pd_TAN3PI8);
    tmp2 = _mm_cmple_pd(x, *(v2sd *) _pd_0p66);
    inftan3pi8inf0p66 = _mm_and_pd(tmp, tmp2);  // if( x <= tan 3pi/8 ) && (x <= 0.66)

#if 1
    y = _mm_and_pd(suptan3pi8, *(v2sd *) _pd_PIO2);  // if( x > tan 3pi/8 ) then y = PI/2, else 0.0
    flag = _mm_and_pd(suptan3pi8, *(v2sd *) _pd_1);  // if( x > tan 3pi/8 ) then flag = 1 else 0
#else
    y = _mm_blendv_pd(_mm_setzero_pd(), *(v2sd *) _pd_PIO2, suptan3pi8);   // if( x > tan 3pi/8 ) then y = PI/2
    flag = _mm_blendv_pd(_mm_setzero_pd(), *(v2sd *) _pd_1, suptan3pi8);   // if( x > tan 3pi/8 ) then flag = 1
#endif
    // one _mm_blendv_pd vs 2 _mm_and_pd and 1 _mm_add_pd?
    y = _mm_blendv_pd(*(v2sd *) _pd_PIO4, y, inftan3pi8inf0p66);

    tmp = _mm_sub_pd(x, *(v2sd *) _pd_1);
    tmp2 = _mm_add_pd(x, *(v2sd *) _pd_1);
    tmp = _mm_div_pd(tmp, tmp2);
    x = _mm_blendv_pd(tmp, x, inftan3pi8inf0p66);
    xeqzero = _mm_cmpeq_pd(x, _mm_setzero_pd());

    tmp2 = _mm_cmpeq_pd(*(v2sd *) _pd_PIO4, y);
    flag = _mm_blendv_pd(flag, *(v2sd *) _pd_2, tmp2);  // if y = PIO4 then flag = 2


    z = _mm_mul_pd(x, x);  // z = x*x

    // z = z * polevl(z, P_, 4)
    tmp = _mm_fmadd_pd_custom(*(v2sd *) _pd_ATAN_P0, z, *(v2sd *) _pd_ATAN_P1);
    tmp = _mm_fmadd_pd_custom(tmp, z, *(v2sd *) _pd_ATAN_P2);
    tmp = _mm_fmadd_pd_custom(tmp, z, *(v2sd *) _pd_ATAN_P3);
    tmp = _mm_fmadd_pd_custom(tmp, z, *(v2sd *) _pd_ATAN_P4);
    tmp = _mm_mul_pd(z, tmp);

    // z = z / p1evl(z, Q_, 5);
    tmp2 = _mm_add_pd(z, *(v2sd *) _pd_ATAN_Q0);
    tmp2 = _mm_fmadd_pd_custom(tmp2, z, *(v2sd *) _pd_ATAN_Q1);
    tmp2 = _mm_fmadd_pd_custom(tmp2, z, *(v2sd *) _pd_ATAN_Q2);
    tmp2 = _mm_fmadd_pd_custom(tmp2, z, *(v2sd *) _pd_ATAN_Q3);
    tmp2 = _mm_fmadd_pd_custom(tmp2, z, *(v2sd *) _pd_ATAN_Q4);
    z = _mm_div_pd(tmp, tmp2);

    // z = x * z + x
    z = _mm_fmadd_pd_custom(x, z, x);

    tmp = _mm_cmpeq_pd(flag, *(v2sd *) _pd_2);
    tmp2 = _mm_cmpeq_pd(flag, *(v2sd *) _pd_1);

#if 1
    tmp = _mm_and_pd(tmp, *(v2sd *) _pd_0p5xMOREBITS);
    tmp2 = _mm_and_pd(tmp2, *(v2sd *) _pd_MOREBITS);
    z = _mm_add_pd(z, tmp);
    z = _mm_add_pd(z, tmp2);
#else
    z = _mm_blendv_pd(z, _mm_add_pd(z, *(v2sd *) _pd_0p5xMOREBITS), tmp);  // if (flag == 2) then z += 0.5 * MOREBITS
    z = _mm_blendv_pd(z, _mm_add_pd(z, *(v2sd *) _pd_MOREBITS), tmp2);     // if (flag == 1) then z +=  MOREBITS
#endif

    y = _mm_add_pd(y, z);
    // y = _mm_blendv_pd(y, _mm_xor_pd(*(v2sd *) _pd_negative_mask, y), sign);
    y = _mm_xor_pd(y, sign);
    y = _mm_blendv_pd(y, xx, xeqzero);  // if (xx == 0) then return xx (x is fabs(xx))
    return (y);
}

static inline v2sd atan2_pd(v2sd y, v2sd x)
{
    v2sd z, w;
    v2sd xinfzero, yinfzero, xeqzero, yeqzero;
    v2sd xeqzeroandyinfzero, yeqzeroandxinfzero;
    v2sd specialcase;
    v2sd tmp, tmp2;

    xinfzero = _mm_cmplt_pd(x, _mm_setzero_pd());  // code =2
    yinfzero = _mm_cmplt_pd(y, _mm_setzero_pd());  // code = code |1;

    xeqzero = _mm_cmpeq_pd(x, _mm_setzero_pd());
    yeqzero = _mm_cmpeq_pd(y, _mm_setzero_pd());

    xeqzeroandyinfzero = _mm_and_pd(xeqzero, yinfzero);
    yeqzeroandxinfzero = _mm_and_pd(yeqzero, xinfzero);
#if 1
    xeqzeroandyinfzero = _mm_and_pd(xeqzeroandyinfzero, *(v2sd *) _pd_sign_mask);
    tmp = _mm_xor_pd(*(v2sd *) _pd_PIO2F, xeqzeroandyinfzero);  // either PI or -PI
    z = _mm_andnot_pd(yeqzero, tmp);                            // not(yeqzero) and tmp => 0, PI/2, -PI/2
#else
    z = *(v2sd *) _pd_PIO2F;
    z = _mm_blendv_pd(z, *(v2sd *) _pd_mPIO2F, xeqzeroandyinfzero);
    z = _mm_blendv_pd(z, _mm_setzero_pd(), yeqzero);
#endif
    z = _mm_blendv_pd(z, *(v2sd *) _pd_PIF, yeqzeroandxinfzero);

    specialcase = _mm_or_pd(xeqzero, yeqzero);

#if 1
    tmp = _mm_and_pd(*(v2sd *) _pd_PIF, _mm_andnot_pd(yinfzero, xinfzero));
    tmp2 = _mm_and_pd(*(v2sd *) _pd_mPIF, _mm_and_pd(yinfzero, xinfzero));
    w = _mm_add_pd(tmp, tmp2);
#else
    w = _mm_setzero_pd();
    w = _mm_blendv_pd(w, *(v2sd *) _pd_PIF, _mm_andnot_pd(yinfzero, xinfzero));  // y >= 0 && x<0
    w = _mm_blendv_pd(w, *(v2sd *) _pd_mPIF, _mm_and_pd(yinfzero, xinfzero));    // y < 0 && x<0
#endif

    tmp = _mm_div_pd(y, x);
    tmp = atan_pd(tmp);
    tmp = _mm_add_pd(w, tmp);
    z = _mm_blendv_pd(tmp, z, specialcase);  // atanf(y/x) if not in special case

    return (z);
}

static inline void atan128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, atan_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, atan_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan(src[i]);
    }
}

static inline void atan2128d(double *src1, double *src2, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(dst + i, atan2_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(dst + i, atan2_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2(src1[i], src2[i]);
    }
}

static inline void atan2128d_interleaved(complex64_t *src, double *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_DOUBLE);
    stop_len *= 2 * SSE_LEN_DOUBLE;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_DOUBLE) {
            v2sdx2 src_split = _mm_load2_pd((double *) (src) + j);
            v2sdx2 src_split2 = _mm_load2_pd((double *) (src) + j + 2 * SSE_LEN_DOUBLE);
            _mm_store_pd(dst + i, atan2_pd(src_split.val[1], src_split.val[0]));
            _mm_store_pd(dst + i + SSE_LEN_DOUBLE, atan2_pd(src_split2.val[1], src_split2.val[0]));
            j += 4 * SSE_LEN_DOUBLE;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_DOUBLE) {
            v2sdx2 src_split = _mm_load2u_pd((double *) (src) + j);
            v2sdx2 src_split2 = _mm_load2u_pd((double *) (src) + j + 2 * SSE_LEN_DOUBLE);
            _mm_storeu_pd(dst + i, atan2_pd(src_split.val[1], src_split.val[0]));
            _mm_storeu_pd(dst + i + SSE_LEN_DOUBLE, atan2_pd(src_split2.val[1], src_split2.val[0]));
            j += 4 * SSE_LEN_DOUBLE;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2(src[i].im, src[i].re);
    }
}

static inline void asin128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, asin_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, asin_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}

// Work in progress
// from atan
// asin(x) = atan( x / sqrt( 1 - x*x ) )
static inline void asin128d_(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            v2sd tmp = _mm_fnmadd_pd_custom(src_tmp, src_tmp, *(v2sd *) _pd_1);  // 1 - x*x
            tmp = _mm_sqrt_pd(tmp);
            _mm_store_pd(dst + i, atan_pd(_mm_div_pd(src_tmp, tmp)));  // atan( x / sqrt( 1 - x*x ) )
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            v2sd tmp = _mm_fnmadd_pd_custom(src_tmp, src_tmp, *(v2sd *) _pd_1);  // 1 - x*x
            tmp = _mm_sqrt_pd(tmp);
            _mm_storeu_pd(dst + i, atan_pd(_mm_div_pd(src_tmp, tmp)));  // atan( x / sqrt( 1 - x*x ) )
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}

static inline void pol2cart2D128f_precise(float *r, float *theta, float *x, float *y, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf r_tmpf = _mm_load_ps(r + i);
            v4sf theta_tmpf = _mm_load_ps(theta + i);
            v2sd r_tmp0 = _mm_cvtps_pd(r_tmpf);
            v2sd r_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(r_tmpf, r_tmpf));
            v2sd theta_tmp0 = _mm_cvtps_pd(theta_tmpf);
            v2sd theta_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(theta_tmpf, theta_tmpf));
            v2sd sin_tmp0, sin_tmp1;
            v2sd cos_tmp0, cos_tmp1;
            sincos_pd(theta_tmp0, &sin_tmp0, &cos_tmp0);
            sincos_pd(theta_tmp1, &sin_tmp1, &cos_tmp1);
            v2sd x_tmpd0 = _mm_mul_pd(r_tmp0, cos_tmp0);
            v2sd y_tmpd0 = _mm_mul_pd(r_tmp0, sin_tmp0);
            v2sd x_tmpd1 = _mm_mul_pd(r_tmp1, cos_tmp1);
            v2sd y_tmpd1 = _mm_mul_pd(r_tmp1, sin_tmp1);

            v4sf x_tmp = _mm_movelh_ps(_mm_cvtpd_ps(x_tmpd0), _mm_cvtpd_ps(x_tmpd1));
            v4sf y_tmp = _mm_movelh_ps(_mm_cvtpd_ps(y_tmpd0), _mm_cvtpd_ps(y_tmpd1));
            _mm_store_ps(x + i, x_tmp);
            _mm_store_ps(y + i, y_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf r_tmpf = _mm_loadu_ps(r + i);
            v4sf theta_tmpf = _mm_loadu_ps(theta + i);
            v2sd r_tmp0 = _mm_cvtps_pd(r_tmpf);
            v2sd r_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(r_tmpf, r_tmpf));
            v2sd theta_tmp0 = _mm_cvtps_pd(theta_tmpf);
            v2sd theta_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(theta_tmpf, theta_tmpf));
            v2sd sin_tmp0, sin_tmp1;
            v2sd cos_tmp0, cos_tmp1;
            sincos_pd(theta_tmp0, &sin_tmp0, &cos_tmp0);
            sincos_pd(theta_tmp1, &sin_tmp1, &cos_tmp1);
            v2sd x_tmpd0 = _mm_mul_pd(r_tmp0, cos_tmp0);
            v2sd y_tmpd0 = _mm_mul_pd(r_tmp0, sin_tmp0);
            v2sd x_tmpd1 = _mm_mul_pd(r_tmp1, cos_tmp1);
            v2sd y_tmpd1 = _mm_mul_pd(r_tmp1, sin_tmp1);

            v4sf x_tmp = _mm_movelh_ps(_mm_cvtpd_ps(x_tmpd0), _mm_cvtpd_ps(x_tmpd1));
            v4sf y_tmp = _mm_movelh_ps(_mm_cvtpd_ps(y_tmpd0), _mm_cvtpd_ps(y_tmpd1));
            _mm_storeu_ps(x + i, x_tmp);
            _mm_storeu_ps(y + i, y_tmp);
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

static inline void cart2pol2D128f_precise(float *x, float *y, float *r, float *theta, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x_tmpf = _mm_load_ps(x + i);
            v4sf y_tmpf = _mm_load_ps(y + i);
            v2sd x_tmp0 = _mm_cvtps_pd(x_tmpf);
            v2sd x_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(x_tmpf, x_tmpf));
            v2sd y_tmp0 = _mm_cvtps_pd(y_tmpf);
            v2sd y_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(y_tmpf, y_tmpf));
            v2sd y_square0 = _mm_mul_pd(y_tmp0, y_tmp0);
            v2sd y_square1 = _mm_mul_pd(y_tmp1, y_tmp1);
            v2sd r_tmpd0 = _mm_fmadd_pd_custom(x_tmp0, x_tmp0, y_square0);
            v2sd r_tmpd1 = _mm_fmadd_pd_custom(x_tmp1, x_tmp1, y_square1);
            r_tmpd0 = _mm_sqrt_pd(r_tmpd0);
            r_tmpd1 = _mm_sqrt_pd(r_tmpd1);
            v2sd theta_tmpd0 = atan2_pd(y_tmp0, x_tmp0);
            v2sd theta_tmpd1 = atan2_pd(y_tmp1, x_tmp1);
            v4sf r_tmp = _mm_movelh_ps(_mm_cvtpd_ps(r_tmpd0), _mm_cvtpd_ps(r_tmpd1));
            v4sf theta_tmp = _mm_movelh_ps(_mm_cvtpd_ps(theta_tmpd0), _mm_cvtpd_ps(theta_tmpd1));
            _mm_store_ps(r + i, r_tmp);
            _mm_store_ps(theta + i, theta_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x_tmpf = _mm_loadu_ps(x + i);
            v4sf y_tmpf = _mm_loadu_ps(y + i);
            v2sd x_tmp0 = _mm_cvtps_pd(x_tmpf);
            v2sd x_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(x_tmpf, x_tmpf));
            v2sd y_tmp0 = _mm_cvtps_pd(y_tmpf);
            v2sd y_tmp1 = _mm_cvtps_pd(_mm_movehl_ps(y_tmpf, y_tmpf));
            v2sd y_square0 = _mm_mul_pd(y_tmp0, y_tmp0);
            v2sd y_square1 = _mm_mul_pd(y_tmp1, y_tmp1);
            v2sd r_tmpd0 = _mm_fmadd_pd_custom(x_tmp0, x_tmp0, y_square0);
            v2sd r_tmpd1 = _mm_fmadd_pd_custom(x_tmp1, x_tmp1, y_square1);
            r_tmpd0 = _mm_sqrt_pd(r_tmpd0);
            r_tmpd1 = _mm_sqrt_pd(r_tmpd1);
            v2sd theta_tmpd0 = atan2_pd(y_tmp0, x_tmp0);
            v2sd theta_tmpd1 = atan2_pd(y_tmp1, x_tmp1);
            v4sf r_tmp = _mm_movelh_ps(_mm_cvtpd_ps(r_tmpd0), _mm_cvtpd_ps(r_tmpd1));
            v4sf theta_tmp = _mm_movelh_ps(_mm_cvtpd_ps(theta_tmpd0), _mm_cvtpd_ps(theta_tmpd1));
            _mm_storeu_ps(r + i, r_tmp);
            _mm_storeu_ps(theta + i, theta_tmp);
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

// To be tested
#if 0
static inline v2sd exp_pd(v2sd x)
{
    v2sd tmp = _mm_setzero_pd(), fx;

    v2sid emm0;

    v2sd one = *(v2sd *) _pd_1;
    v2sd two = *(v2sd *) _pd_2;

    x = _mm_min_pd(x, *(v2sd *) _pd_exp_hi);
    x = _mm_max_pd(x, *(v2sd *) _pd_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm_fmadd_pd_custom(x, *(v2sd *) _pd_cephes_LOG2EF, *(v2sd *) _pd_0p5);

    /* how to perform a floorf with SSE: just below */
    //emm0 = _mm_cvttpd_epi64_custom(fx);
    //tmp = _mm_cvtepi64_pd_custom(emm0);
    tmp = _mm_round_pd(fx, ROUNDTOFLOOR);

    /* if greater, substract 1 */
    v2sd mask = _mm_cmpgt_pd(tmp, fx);
    mask = _mm_and_pd(mask, one);
    fx = _mm_sub_pd(tmp, mask);

    x = _mm_fnmadd_pd_custom(fx, *(v2sd *) _pd_cephes_exp_C1, x);
    x = _mm_fnmadd_pd_custom(fx, *(v2sd *) _pd_cephes_exp_C2, x);

    v2sd z = _mm_mul_pd(x, x);

    v2sd y = _mm_fmadd_pd_custom(*(v2sd *) _pd_cephes_exp_p0, z, *(v2sd *) _pd_cephes_exp_p1);
    y = _mm_fmadd_pd_custom(y, z, *(v2sd *) _pd_cephes_exp_p2);
    y = _mm_mul_pd(y,x);
    v2sd k = _mm_fmadd_pd_custom(*(v2sd *) _pd_cephes_exp_q0, z, *(v2sd *) _pd_cephes_exp_q1);
    k = _mm_fmadd_pd_custom(k, z, *(v2sd *) _pd_cephes_exp_q2);
    k = _mm_fmadd_pd_custom(k, z, *(v2sd *) _pd_cephes_exp_q3);
    k = _mm_sub_pd(k,y);
    k = _mm_div_pd(y,k);
    k = _mm_fmadd_pd_custom(k, two, one);

    /* build 2^n */
    emm0 = _mm_cvttpd_epi64_custom(fx);
    emm0 = _mm_add_epi64(emm0, *(v2sid *) _pi64_0x7f);
    emm0 = _mm_slli_epi64(emm0, 52);
    v2sd pow2n = _mm_castsi128_pd(emm0);

    k = _mm_mul_pd(k, pow2n);
    return y;
}

static inline v2sd log_pd(v2sd x)
{
    v2sid emm0;
    v2sd one = *(v2sd *) _pd_1;

    v2sd invalid_mask = _mm_cmple_pd(x, _mm_setzero_pd());

    x = _mm_max_pd(x, *(v2sd *) _pd_min_norm_pos); /* cut off denormalized stuff */

    emm0 = _mm_srli_epi64(_mm_castpd_si128(x), 55);

    /* keep only the fractional part */
    x = _mm_and_pd(x, *(v2sd *) _pd_inv_mant_mask);
    x = _mm_or_pd(x, *(v2sd *) _pd_0p5);

    emm0 = _mm_sub_epi64(emm0, *(v2sd *) _pi64_0x7f);
    v2sd e = _mm_cvtepi64_pd_custom_ps(emm0);

    e = _mm_add_pd(e, one);

    /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
	 */
    v2sd mask = _mm_cmplt_pd(x, *(v2sd *) _pd_cephes_SQRTHF);
    v2sd tmp = _mm_and_pd(x, mask);
    x = _mm_sub_pd(x, one);
    e = _mm_sub_pd(e, _mm_and_pd(one, mask));
    x = _mm_add_pd(x, tmp);


    v4sf z = _mm_mul_pd(x, x);

    v2sd y = _mm_fmadd_pd_custom(*(v2sd *) _pd_cephes_log_p0, x, *(v4sf *) _pd_cephes_log_p1);
    y = _mm_fmadd_pd_custom(y, x, *(v2sd *) _pd_cephes_log_p2);
    y = _mm_fmadd_pd_custom(y, x, *(v2sd *) _pd_cephes_log_p3);
    y = _mm_fmadd_pd_custom(y, x, *(v2sd *) _pd_cephes_log_p4);
    y = _mm_fmadd_pd_custom(y, x, *(v2sd *) _pd_cephes_log_p5);
    y = _mm_fmadd_pd_custom(y, x, *(v2sd *) _pd_cephes_log_p6);
    y = _mm_fmadd_pd_custom(y, x, *(v2sd *) _pd_cephes_log_p7);
    y = _mm_fmadd_pd_custom(y, x, *(v2sd *) _pd_cephes_log_p8);
    y = _mm_mul_pd(y, x);

    y = _mm_mul_pd(y, z);

    y = _mm_fmadd_pd_custom(e, *(v2sd *) _pd_cephes_log_q1, y);
    y = _mm_fnmadd_pd_custom(z, *(v2sd *) _pd_0p5, y);

    tmp = _mm_fmadd_pd_custom(e, *(v4sf *) _pd_cephes_log_q2, y);
    x = _mm_add_pd(x, tmp);
    x = _mm_or_pd(x, invalid_mask);  // negative arg will be NAN
    return x;
}
#endif

static inline v2sd tan_pd(v2sd xx)
{
    v2sd xxeqzero, zzsup1m14, ysup1m14;
    v2sd tmp, tmp2;

    xxeqzero = _mm_cmpeq_pd(xx, _mm_setzero_pd());

    v2sd x, y, z, zz;
    v2sid j, jandone, jandtwo, tmpi;
    v2sd sign;

    /* make argument positive but save the sign */
    x = xx;
    x = _mm_and_pd(x, *(v2sd *) _pd_inv_sign_mask);
    // sign = _mm_cmplt_pd(xx, _mm_setzero_pd());
    sign = _mm_and_pd(xx, *(v2sd *) _pd_sign_mask);
#ifdef LOSSTH
    v2sd xsuplossth = _mm_cmpgt_pd(x, *(v2sd *) _pd_tanlossth);
#endif

    /* compute x mod PIO4 */
    y = _mm_mul_pd(x, *(v2sd *) _pd_cephes_FOPI);

    // useful?
    y = _mm_round_pd(y, ROUNDTOFLOOR);

    /* strip high bits of integer part */
    z = _mm_mul_pd(y, *(v2sd *) _pd_0p125);
    // useful?
    z = _mm_round_pd(z, ROUNDTOFLOOR);
    z = _mm_fmadd_pd_custom(z, *(v2sd *) _pd_min8, y);

    /* integer and fractional part modulo one octant */
    j = _mm_cvttpd_epi64_custom(z);

    /* map zeros and singularities to origin */
    jandone = _mm_cmpgt_epi64(_mm_and_si128(j, *(v2sid *) _pi64_1), _mm_setzero_si128());

#if 1
    tmp = _mm_and_pd(*(v2sd *) _pd_1, _mm_castsi128_pd(jandone));
    tmpi = _mm_and_si128(*(v2sid *) _pi64_1, jandone);
    j = _mm_add_epi64(j, tmpi);
    y = _mm_add_pd(y, tmp);
#else
    j = _mm_blendv_epi8(j, _mm_add_epi64(j, *(v2sid *) _pi64_1), jandone);
    y = _mm_blendv_pd(y, _mm_add_pd(y, *(v2sd *) _pd_1), (v2sd) jandone);
#endif

    jandtwo = _mm_cmpgt_epi64(_mm_and_si128(j, *(v2sid *) _pi64_2), _mm_setzero_si128());

    z = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_TAN_mDP1, x);
    z = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_TAN_mDP2, z);
    z = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_TAN_mDP3, z);
    zz = _mm_mul_pd(z, z);

    zzsup1m14 = _mm_cmpgt_pd(zz, *(v2sd *) _pd_1m14);
    tmp = _mm_fmadd_pd_custom(zz, *(v2sd *) _pd_TAN_P0, *(v2sd *) _pd_TAN_P1);
    tmp = _mm_fmadd_pd_custom(zz, tmp, *(v2sd *) _pd_TAN_P2);
    tmp2 = _mm_add_pd(zz, *(v2sd *) _pd_TAN_Q0);
    tmp2 = _mm_fmadd_pd_custom(zz, tmp2, *(v2sd *) _pd_TAN_Q1);
    tmp2 = _mm_fmadd_pd_custom(zz, tmp2, *(v2sd *) _pd_TAN_Q2);
    tmp2 = _mm_fmadd_pd_custom(zz, tmp2, *(v2sd *) _pd_TAN_Q3);
    tmp2 = _mm_div_pd(tmp, tmp2);
    tmp2 = _mm_mul_pd(zz, tmp2);

#if 1
    ysup1m14 = _mm_mul_pd(z, tmp2);
    ysup1m14 = _mm_and_pd(ysup1m14, zzsup1m14);
    y = _mm_add_pd(z, ysup1m14);
#else
    ysup1m14 = _mm_fmadd_pd_custom(z, tmp2, z);
    y = _mm_blendv_pd(z, ysup1m14, zzsup1m14);
#endif

    tmp = _mm_div_pd(*(v2sd *) _pd_min1, y);
    y = _mm_blendv_pd(y, tmp, _mm_castsi128_pd(jandtwo));

    // y = _mm_blendv_pd(y,_mm_mul_pd(y,*(v2sd *) _pd_min1), sign);
    y = _mm_xor_pd(y, sign);

#ifdef LOSSTH
    y = _mm_blendv_pd(y, _mm_setzero_pd(), xsuplossth);
#endif
    y = _mm_blendv_pd(y, xx, xxeqzero);
    return y;
}

static inline void tan128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, tan_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, tan_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tan(src[i]);
    }
}
