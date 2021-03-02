/*
 * Project : SIMD_Utils
 * Version : 0.1.6
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

typedef __m128d v2sd;  // vector of 2 double (sse)
typedef __m128i v2si;  // vector of 2 int 64 (sse)

static inline void set128d(double *src, double value, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % SSE_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = value;
    }
}

static inline void zero128d(double *src, int len)
{
    const v2sd tmp = _mm_setzero_pd();

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (src) % SSE_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_store_pd(src + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            _mm_storeu_pd(src + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        src[i] = 0.0;
    }
}

static inline void copy128d(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

//TODO : "Immediate add/mul?"
static inline void addc128d(double *src, double value, double *dst, int len)
{
    const v2sd tmp = _mm_set1_pd(value);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(_b), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t)(_c), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_c), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(_a), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t)(_a), (uintptr_t)(_b), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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


//TODO : add dual accumulator like vectorSlope128f
static inline void vectorSlope128d(double *dst, int len, double offset, double slope)
{
    v2sd coef = _mm_set_pd(slope, 0.0);
    v2sd slope2_vec = _mm_set1_pd(2.0 * slope);
    v2sd curVal = _mm_add_pd(_mm_set1_pd(offset), coef);

    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        _mm_store_pd(dst + 0, curVal);
    } else {
        _mm_storeu_pd(dst + 0, curVal);
    }

    if (((uintptr_t)(const void *) (dst) % SSE_LEN_BYTES) == 0) {
        for (int i = SSE_LEN_DOUBLE; i < stop_len; i += SSE_LEN_DOUBLE) {
            curVal = _mm_add_pd(curVal, slope2_vec);
            _mm_store_pd(dst + i, curVal);
        }
    } else {
        for (int i = SSE_LEN_DOUBLE; i < stop_len; i += SSE_LEN_DOUBLE) {
            curVal = _mm_add_pd(curVal, slope2_vec);
            _mm_storeu_pd(dst + i, curVal);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (double) i;
    }
}



// Work in progress
// in SSE, missing _mm_cvtepi64_pd, _mm_cvttpd_epi64
// See : https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx

static inline v2sd _mm_cvtepi64_pd_custom(v2si x)
{
#if 0
    //Signed
    x = _mm_add_epi64(x, _mm_castpd_si128(_mm_set1_pd(0x0018000000000000)));
    return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0018000000000000));
#else
    //unsigned
    x = _mm_or_si128(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
    return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0010000000000000));
#endif
}

static inline v2si _mm_cvttpd_epi64_custom(v2sd x)
{
    //Signed
#if 0
   x = _mm_add_pd(x, _mm_set1_pd(0x0018000000000000));
    return _mm_sub_epi64(
        _mm_castpd_si128(x),
        _mm_castpd_si128(_mm_set1_pd(0x0018000000000000))
    );
#else
    //Unsigned
    x = _mm_add_pd(x, _mm_set1_pd(0x0010000000000000));
    return _mm_xor_si128(
        _mm_castpd_si128(x),
        _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
#endif
}

static inline void print2(__m128d v)
{
    double *p = (double *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%13.8g, %13.8g]", p[0], p[1]);
}

static inline void print2i(__m128i v)
{
    int64_t *p = (int64_t *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%ld, %ld]", p[0], p[1]);
}

static inline void sincos_pd(v2sd x, v2sd *s, v2sd *c)
{
    v2sd xmm1, xmm2, xmm3 = _mm_setzero_pd(), sign_bit_sin, y;

    v2si emm0, emm2, emm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm_and_pd(x, *(v2sd *) _pd_inv_sign_mask);

    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm_and_pd(sign_bit_sin, *(v2sd *) _pd_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_pd(x, *(v2sd *) _pd_cephes_FOPI);

    /* store the integer part of y in emm2 */
    emm2 = _mm_cvttpd_epi64_custom(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi64(emm2, *(v2si *) _pi64_1);

    emm2 = _mm_and_si128(emm2, *(v2si *) _pi64_inv1);
    y = _mm_cvtepi64_pd_custom(emm2);
    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0 = _mm_and_si128(emm2, *(v2si *) _pi64_4);
    //print2i(emm0);
    emm0 = _mm_slli_epi64(emm0, 61);
    //print2i(emm0);
    v2sd swap_sign_bit_sin = _mm_castsi128_pd(emm0);

    /* get the polynom selection mask for the sine*/
    emm2 = _mm_and_si128(emm2, *(v2si *) _pi64_2);
    emm2 = _mm_cmpeq_epi64(emm2, _mm_setzero_si128());
    v2sd poly_mask = _mm_castsi128_pd(emm2);
    //print2i(emm2);
    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_minus_cephes_DP1, x);
    x = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_minus_cephes_DP2, x);
    x = _mm_fmadd_pd_custom(y, *(v2sd *) _pd_minus_cephes_DP3, x);

    emm4 = _mm_sub_epi64(emm4, *(v2si *) _pi64_2);
    emm4 = _mm_andnot_si128(emm4, *(v2si *) _pi64_4);
    emm4 = _mm_slli_epi64(emm4, 61);
    v2sd sign_bit_cos = _mm_castsi128_pd(emm4);

    sign_bit_sin = _mm_xor_pd(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v2sd z = _mm_mul_pd(x, x);

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
    v2sd y2 = _mm_fmadd_pd_custom(*(v2sd *) _ps_sincof_p0, z, *(v2sd *) _pd_sincof_p1);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p2);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p3);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p4);
    y2 = _mm_fmadd_pd_custom(y2, z, *(v2sd *) _pd_sincof_p5);
    y2 = _mm_mul_pd(y2, z);
    y2 = _mm_fmadd_pd_custom(y2, x, x);


    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    v2sd ysin2 = _mm_and_pd(xmm3, y2);
    v2sd ysin1 = _mm_andnot_pd(xmm3, y);
    y2 = _mm_sub_pd(y2, ysin2);
    y = _mm_sub_pd(y, ysin1);
    xmm1 = _mm_add_pd(ysin1, ysin2);
    xmm2 = _mm_add_pd(y, y2);

    /* update the sign */
    *s = _mm_xor_pd(xmm1, sign_bit_sin);
    *c = _mm_xor_pd(xmm2, sign_bit_cos);
}

static inline void sincos128d(double *src, double *dst_sin, double *dst_cos, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t)(src), (uintptr_t)(dst_sin), (uintptr_t)(dst_cos), SSE_LEN_BYTES)) {
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
        dst_sin[i] = sin(i);
        dst_cos[i] = cos(i);
    }
}

// To be tested
#if 0
static inline v2sd exp_pd(v2sd x)
{
    v2sd tmp = _mm_setzero_pd(), fx;

    v2si emm0;

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
    emm0 = _mm_add_epi64(emm0, *(v2si *) _pi64_0x7f);
    emm0 = _mm_slli_epi64(emm0, 52);
    v2sd pow2n = _mm_castsi128_pd(emm0);

    k = _mm_mul_pd(k, pow2n);
    return y;
}

static inline v2sd log_pd(v2sd x)
{
    v2si emm0;
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
