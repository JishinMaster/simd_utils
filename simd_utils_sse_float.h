/*
 * Project : SIMD_Utils
 * Version : 0.2.5
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

static inline v4sf log10_ps(v4sf x)
{
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN log10_ps" ::
                       : "memory");
#endif

    v4si emm0;
    v4sf one = *(v4sf *) _ps_1;
    v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());
    x = _mm_max_ps(x, *(v4sf *) _ps_min_norm_pos); /* cut off denormalized stuff */
    emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

    /* keep only the fractional part */
    x = _mm_and_ps(x, *(v4sf *) _ps_inv_mant_mask);
    x = _mm_or_ps(x, *(v4sf *) _ps_0p5);
    emm0 = _mm_sub_epi32(emm0, *(v4si *) _pi32_0x7f);
    v4sf e = _mm_cvtepi32_ps(emm0);
    e = _mm_add_ps(e, one);

    v4sf mask = _mm_cmplt_ps(x, *(v4sf *) _ps_cephes_SQRTHF);
    v4sf tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);

    v4sf z = _mm_mul_ps(x, x);
    v4sf y = _mm_fmadd_ps_custom(*(v4sf *) _ps_cephes_log_p0, x, *(v4sf *) _ps_cephes_log_p1);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p2);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p3);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p4);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p5);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p6);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p7);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p8);
    y = _mm_mul_ps(y, x);
    y = _mm_mul_ps(y, z);

    y = _mm_fnmadd_ps_custom(z, *(v4sf *) _ps_0p5, y);

    // Could it be improved with more parallelism or would it worsen precision?
    tmp = _mm_add_ps(x, y);
    z = _mm_mul_ps(tmp, *(v4sf *) _ps_cephes_L10EB);
    z = _mm_fmadd_ps_custom(y, *(v4sf *) _ps_cephes_L10EA, z);
    z = _mm_fmadd_ps_custom(x, *(v4sf *) _ps_cephes_L10EA, z);
    z = _mm_fmadd_ps_custom(e, *(v4sf *) _ps_cephes_L102B, z);
    x = _mm_fmadd_ps_custom(e, *(v4sf *) _ps_cephes_L102A, z);

    x = _mm_or_ps(x, invalid_mask);  // negative arg will be NAN
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END log10_ps" ::
                       : "memory");
#endif
    return x;
}

static inline v4sf log2_ps(v4sf x)
{
    v4si emm0;
    v4sf one = *(v4sf *) _ps_1;
    v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());
    x = _mm_max_ps(x, *(v4sf *) _ps_min_norm_pos); /* cut off denormalized stuff */
    emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

    /* keep only the fractional part */
    x = _mm_and_ps(x, *(v4sf *) _ps_inv_mant_mask);
    x = _mm_or_ps(x, *(v4sf *) _ps_0p5);
    emm0 = _mm_sub_epi32(emm0, *(v4si *) _pi32_0x7f);
    v4sf e = _mm_cvtepi32_ps(emm0);
    e = _mm_add_ps(e, one);

    v4sf mask = _mm_cmplt_ps(x, *(v4sf *) _ps_cephes_SQRTHF);
    v4sf tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);

    v4sf z = _mm_mul_ps(x, x);
    v4sf y = _mm_fmadd_ps_custom(*(v4sf *) _ps_cephes_log_p0, x, *(v4sf *) _ps_cephes_log_p1);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p2);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p3);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p4);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p5);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p6);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p7);
    y = _mm_fmadd_ps_custom(y, x, *(v4sf *) _ps_cephes_log_p8);
    y = _mm_mul_ps(y, x);
    y = _mm_mul_ps(y, z);

    y = _mm_fnmadd_ps_custom(z, *(v4sf *) _ps_0p5, y);

    // Could it be improved with more parallelism or would it worsen precision?
    tmp = _mm_add_ps(y, x);
    z = _mm_mul_ps(y, *(v4sf *) _ps_cephes_LOG2EA);
    z = _mm_fmadd_ps_custom(x, *(v4sf *) _ps_cephes_LOG2EA, z);
    z = _mm_add_ps(z, tmp);
    x = _mm_add_ps(z, e);
    x = _mm_or_ps(x, invalid_mask);  // negative arg will be NAN
    return x;
}


static inline void log10_128f(float *src, float *dst, int len)
{
    const v4sf invln10f = _mm_set1_ps((float) INVLN10);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

static inline void log10_128f_precise(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log10_ps(_mm_load_ps(src + i));
            _mm_store_ps(dst + i, src_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log10_ps(_mm_loadu_ps(src + i));
            _mm_storeu_ps(dst + i, src_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void log2_128f(float *src, float *dst, int len)
{
    const v4sf invln2f = _mm_set1_ps((float) INVLN2);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log_ps(_mm_load_ps(src + i));
            _mm_store_ps(dst + i, _mm_mul_ps(src_tmp, invln2f));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log_ps(_mm_loadu_ps(src + i));
            _mm_storeu_ps(dst + i, _mm_mul_ps(src_tmp, invln2f));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline void log2_128f_precise(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log2_ps(_mm_load_ps(src + i));
            _mm_store_ps(dst + i, src_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = log2_ps(_mm_loadu_ps(src + i));
            _mm_storeu_ps(dst + i, src_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline void ln_128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

static inline void exp_128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, exp_ps(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, exp_ps(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}



// rewritten alternate version which properly returns MAXNUMF or 0.0 outside of boundaries
static inline v4sf exp_ps_alternate(v4sf x)
{
    v4sf z_tmp, z, fx;
    v4si n;
    v4sf xsupmaxlogf, xinfminglogf;

    xsupmaxlogf = _mm_cmpgt_ps(x, *(v4sf *) _ps_MAXLOGF);
    xinfminglogf = _mm_cmplt_ps(x, *(v4sf *) _ps_MINLOGF);

    /* Express e**x = e**g 2**n
     *   = e**g e**( n loge(2) )
     *   = e**( g + n loge(2) )
     */
    fx = _mm_fmadd_ps_custom(*(v4sf *) _ps_cephes_LOG2EF, x, *(v4sf *) _ps_0p5);
    z = _mm_round_ps(fx, _MM_FROUND_TO_NEG_INF);  // round to floor

    x = _mm_fmadd_ps_custom(z, *(v4sf *) _ps_cephes_exp_minC1, x);
    x = _mm_fmadd_ps_custom(z, *(v4sf *) _ps_cephes_exp_minC2, x);

    n = _mm_cvttps_epi32(z);
    n = _mm_add_epi32(n, *(v4si *) _pi32_0x7f);
    n = _mm_slli_epi32(n, 23);
    v4sf pow2n = _mm_castsi128_ps(n);

    z = _mm_mul_ps(x, x);

    z_tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_cephes_exp_p0, x, *(v4sf *) _ps_cephes_exp_p1);
    z_tmp = _mm_fmadd_ps_custom(z_tmp, x, *(v4sf *) _ps_cephes_exp_p2);
    z_tmp = _mm_fmadd_ps_custom(z_tmp, x, *(v4sf *) _ps_cephes_exp_p3);
    z_tmp = _mm_fmadd_ps_custom(z_tmp, x, *(v4sf *) _ps_cephes_exp_p4);
    z_tmp = _mm_fmadd_ps_custom(z_tmp, x, *(v4sf *) _ps_cephes_exp_p5);
    z_tmp = _mm_fmadd_ps_custom(z_tmp, z, x);

    /* build 2^n */
    z_tmp = _mm_fmadd_ps_custom(z_tmp, pow2n, pow2n);

    z = _mm_blendv_ps(z_tmp, *(v4sf *) _ps_MAXNUMF, xsupmaxlogf);

#if 1
    z = _mm_andnot_ps(xinfminglogf, z);
#else
    z = _mm_blendv_ps(z, _mm_setzero_ps(), xinfminglogf);
#endif
    return z;
}

static inline void exp_128f_(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, exp_ps_alternate(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, exp_ps_alternate(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

/////////////// CBRT
// Test, seems okay, see sse_mathfun
#if 0
static inline void frexpf128f(float *src, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= (SSE_LEN_FLOAT);
    int ep[4];
    float fr[4];

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x = _mm_load_ps(src + i);
            // x = _mm_max_ps(x, *(v4sf *) _ps_min_norm_pos);
            v4si emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
            v4sf x_frac = _mm_and_ps(x, *(v4sf *) _ps_inv_mant_mask);
            x_frac = _mm_or_ps(x_frac, *(v4sf *) _ps_0p5);
            emm0 = _mm_sub_epi32(emm0, *(v4si *) _pi32_0x7f);
            v4sf e = _mm_cvtepi32_ps(emm0);
            e = _mm_add_ps(e, *(v4sf *) _ps_1);
            fr[0] = frexpf(src[i], &ep[0]);
            fr[1] = frexpf(src[i + 1], &ep[1]);
            fr[2] = frexpf(src[i + 2], &ep[2]);
            fr[3] = frexpf(src[i + 3], &ep[3]);
            printf("%f %d\n", fr[0], ep[0]);
            printf("%f %d\n", fr[1], ep[1]);
            printf("%f %d\n", fr[2], ep[2]);
            printf("%f %d\n", fr[3], ep[3]);
        }
    }
}

static inline void ldexp128f(float *src, int ex, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= (SSE_LEN_FLOAT);
    int ep[4];
    float fr[4];
    v4sf cst = _mm_set1_ps((float) (1 << ex));
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x = _mm_load_ps(src + i);
            v4sf x_ex = _mm_mul_ps(x, cst);
            print4(x);
            printf("\n");
            print4(x_ex);
            printf("\n");
            fr[0] = ldexpf(src[i], ex);
            fr[1] = ldexpf(src[i + 1], ex);
            fr[2] = ldexpf(src[i + 2], ex);
            fr[3] = ldexpf(src[i + 3], ex);
            printf("%3.24g %3.24g %3.24g %3.24g\n", fr[0], fr[1], fr[2], fr[3]);
        }
    }
}
#endif

// from https://stackoverflow.com/questions/57454416/sse-integer-2n-powers-of-2-for-32-bit-integers-without-avx2
static inline v4sf power_of_twof(v4si b)
{
    v4si exp = _mm_add_epi32(b, _mm_set1_epi32(127));
    v4sf f = _mm_castsi128_ps(_mm_slli_epi32(exp, 23));
    return f;
}

static inline v4sf cbrtf_ps(v4sf xx)
{
    v4sf e, rem, sign;
    v4sf x, z;
    v4sf tmp, tmp2;

    x = xx;
    // sign = _mm_cmpgt_ps(x, _mm_setzero_ps());
    sign = _mm_and_ps(xx, *(v4sf *) _ps_sign_mask);
    x = _mm_and_ps(x, *(v4sf *) _ps_pos_sign_mask);

    z = x;
    /* extract power of 2, leaving
     * mantissa between 0.5 and 1
     */
    // x = frexpf(x, &e);
    // solve problem for zero
    v4si emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
    x = _mm_and_ps(x, *(v4sf *) _ps_inv_mant_mask);
    x = _mm_or_ps(x, *(v4sf *) _ps_0p5);
    emm0 = _mm_sub_epi32(emm0, *(v4si *) _pi32_0x7e);  // -7f + 1
    e = _mm_cvtepi32_ps(emm0);

    /* Approximate cube root of number between .5 and 1,
     * peak relative error = 9.2e-6
     */
    tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_CBRTF_P0, x, *(v4sf *) _ps_CBRTF_P1);
    tmp = _mm_fmadd_ps_custom(x, tmp, *(v4sf *) _ps_CBRTF_P2);
    tmp = _mm_fmadd_ps_custom(x, tmp, *(v4sf *) _ps_CBRTF_P3);
    x = _mm_fmadd_ps_custom(x, tmp, *(v4sf *) _ps_CBRTF_P4);

    /* exponent divided by 3 */
    v4sf e_sign = _mm_cmpge_ps(e, _mm_setzero_ps());
    e = _mm_and_ps(e, *(v4sf *) _ps_pos_sign_mask);

    rem = e;
    e = _mm_mul_ps(e, *(v4sf *) _ps_0p3);
    v4sf e_tmp = _mm_mul_ps(*(v4sf *) _ps_3, _mm_round_ps(e, ROUNDTOZERO));
    rem = _mm_sub_ps(rem, e_tmp);

    v4sf mul1, mul2;
    v4sf mul_cst1 = _mm_blendv_ps(*(v4sf *) _ps_cephes_invCBRT2, *(v4sf *) _ps_cephes_CBRT2, e_sign);
    v4sf mul_cst2 = _mm_blendv_ps(*(v4sf *) _ps_cephes_invCBRT4, *(v4sf *) _ps_cephes_CBRT4, e_sign);
    mul1 = _mm_mul_ps(x, mul_cst1);
    mul2 = _mm_mul_ps(x, mul_cst2);

    v4si remi = _mm_cvtps_epi32(rem);  // rem integer
    v4si rem1 = _mm_cmpeq_epi32(remi, *(v4si *) _pi32_1);
    v4si rem2 = _mm_cmpeq_epi32(remi, *(v4si *) _pi32_2);

    x = _mm_blendv_ps(x, mul1, _mm_castsi128_ps(rem1));  // rem==1
    x = _mm_blendv_ps(x, mul2, _mm_castsi128_ps(rem2));  // rem==2

    /* multiply by power of 2 */
    //  x = ldexpf(x, e);
    // v4sf cst = _mm_srli_epi32()
    // x= x* (1 >> e)
    // AVX2 : _mm256_srlv_epi32 pour shift
    v4sf cst = power_of_twof(_mm_cvtps_epi32(e));
    // blend sign of e
    tmp = _mm_mul_ps(x, cst);
    tmp2 = _mm_div_ps(x, cst);
    x = _mm_blendv_ps(tmp2, tmp, e_sign);

    /* Newton iteration */
    // x -= (x - (z / (x * x))) * 0.333333333333;
    tmp2 = _mm_mul_ps(x, x);
    tmp2 = _mm_div_ps(z, tmp2);
    tmp2 = _mm_sub_ps(x, tmp2);
    tmp2 = _mm_mul_ps(tmp2, *(v4sf *) _ps_0p3);
    x = _mm_sub_ps(x, tmp2);

    // x = _mm_blendv_ps(_mm_mul_ps(x, *(v4sf *) _ps_min1), x, sign);
    x = _mm_xor_ps(x, sign);
    return x;
}

static inline void cbrt128f(float *src, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= (SSE_LEN_FLOAT);
    // float fr[4];
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x = _mm_load_ps(src + i);
            // printf("x : ");print4(x);printf("\n");
            v4sf dst_tmp = cbrtf_ps(x);
            /*printf("out : ");print4(tmp);printf("\n");
            fr[0] = cbrtf(src[i]);
            fr[1] = cbrtf(src[i+1]);
            fr[2] = cbrtf(src[i+2]);
            fr[3] = cbrtf(src[i+3]);
            printf("%3.9g %3.9g %3.9g %3.9g\n",fr[0], fr[1], fr[2], fr[3]);*/
            _mm_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x = _mm_loadu_ps(src + i);
            v4sf dst_tmp = cbrtf_ps(x);
            _mm_storeu_ps(dst + i, dst_tmp);
        }
    }
    for (int i = stop_len; i < len; i++) {
        dst[i] = cbrtf(src[i]);
    }
}
/////

static inline void fabs128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, src_tmp);
            v4sf dst_tmp2 = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, src_tmp2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, src_tmp);
            v4sf dst_tmp2 = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, src_tmp2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void set128f(float *dst, float value, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value;
    }
}

// Could be better to just use set(0)
static inline void zero128f(float *dst, int len)
{
    const v4sf tmp = _mm_setzero_ps();

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, tmp);
            //_mm_stream_si128(src + i, (__m128i)tmp);
        }
        //_mm_sfence();
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 0.0f;
    }
}

static inline void copy128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            _mm_store_ps(dst + i, src_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, src_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            _mm_storeu_ps(dst + i, src_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, src_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(src1 + i);
            v4sf b = _mm_load_ps(src2 + i);
            v4sf a2 = _mm_load_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_load_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_add_ps(a, b);
            v4sf dst_tmp2 = _mm_add_ps(a2, b2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(src1 + i);
            v4sf b = _mm_loadu_ps(src2 + i);
            v4sf a2 = _mm_loadu_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_loadu_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_add_ps(a, b);
            v4sf dst_tmp2 = _mm_add_ps(a2, b2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}


static inline void mul128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(src1 + i);
            v4sf b = _mm_load_ps(src2 + i);
            v4sf a2 = _mm_load_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_load_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_mul_ps(a, b);
            v4sf dst_tmp2 = _mm_mul_ps(a2, b2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(src1 + i);
            v4sf b = _mm_loadu_ps(src2 + i);
            v4sf a2 = _mm_loadu_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_loadu_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_mul_ps(a, b);
            v4sf dst_tmp2 = _mm_mul_ps(a2, b2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(src1 + i);
            v4sf b = _mm_load_ps(src2 + i);
            v4sf a2 = _mm_load_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_load_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_sub_ps(a, b);
            v4sf dst_tmp2 = _mm_sub_ps(a2, b2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(src1 + i);
            v4sf b = _mm_loadu_ps(src2 + i);
            v4sf a2 = _mm_loadu_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_loadu_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_sub_ps(a, b);
            v4sf dst_tmp2 = _mm_sub_ps(a2, b2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

// TODO : "Immediate add/mul?"
//  No need for subc, just use addc(-value)
static inline void addc128f(float *src, float value, float *dst, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(src + i);
            v4sf a2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_add_ps(a, tmp);
            v4sf dst_tmp2 = _mm_add_ps(a2, tmp);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(src + i);
            v4sf a2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_add_ps(a, tmp);
            v4sf dst_tmp2 = _mm_add_ps(a2, tmp);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc128f(float *src, float value, float *dst, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp1 = _mm_mul_ps(tmp, src_tmp1);
            v4sf dst_tmp2 = _mm_mul_ps(tmp, src_tmp2);
            _mm_store_ps(dst + i, dst_tmp1);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp1 = _mm_mul_ps(tmp, src_tmp1);
            v4sf dst_tmp2 = _mm_mul_ps(tmp, src_tmp2);
            _mm_storeu_ps(dst + i, dst_tmp1);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd128f(float *_a, float *_b, float *_c, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (_b), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (_c), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps(_a + i);
            v4sf b = _mm_load_ps(_b + i);
            v4sf c = _mm_load_ps(_c + i);
            v4sf a2 = _mm_load_ps(_a + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_load_ps(_b + i + SSE_LEN_FLOAT);
            v4sf c2 = _mm_load_ps(_c + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_fmadd_ps_custom(a, b, c);
            v4sf dst_tmp2 = _mm_fmadd_ps_custom(a2, b2, c2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps(_a + i);
            v4sf b = _mm_loadu_ps(_b + i);
            v4sf c = _mm_loadu_ps(_c + i);
            v4sf a2 = _mm_loadu_ps(_a + i + SSE_LEN_FLOAT);
            v4sf b2 = _mm_loadu_ps(_b + i + SSE_LEN_FLOAT);
            v4sf c2 = _mm_loadu_ps(_c + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_fmadd_ps_custom(a, b, c);
            v4sf dst_tmp2 = _mm_fmadd_ps_custom(a2, b2, c2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (_a[i] * _b[i]) + _c[i];
    }
}

static inline void mulcadd128f(float *_a, float _b, float *_c, float *dst, int len)
{
    v4sf b = _mm_set1_ps(_b);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_c), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_b), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

// warning: src2 should have no 0.0f values
static inline void div128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

    if (len >= SSE_LEN_BYTES) {
        if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
            _mm_store_ps(dst + 0, curVal);
            _mm_store_ps(dst + SSE_LEN_FLOAT, curVal2);
        } else {
            _mm_storeu_ps(dst + 0, curVal);
            _mm_storeu_ps(dst + SSE_LEN_FLOAT, curVal2);
        }

        if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
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
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

// prefered version
#if 1
static inline void convertFloat32ToU8_128(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= (4 * SSE_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    int _mm_rounding_ori = _MM_GET_ROUNDING_MODE();  // save rounding mode
    int rounding_ori = fegetround();

    if (rounding_mode == RndZero) {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
        fesetround(FE_TOWARDZERO);
    } else if (rounding_mode == RndFinancial) {  // nothing to do, Default bankers rounding => round to nearest even
    } else {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
        fesetround(FE_TONEAREST);
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp3 = _mm_load_ps(src + i + 2 * SSE_LEN_FLOAT);
            v4sf src_tmp4 = _mm_load_ps(src + i + 3 * SSE_LEN_FLOAT);
            v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
            v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
            v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
            v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
            v4si tmp1_int = _mm_cvtps_epi32(tmp1);
            v4si tmp2_int = _mm_cvtps_epi32(tmp2);
            v4si tmp3_int = _mm_cvtps_epi32(tmp3);
            v4si tmp4_int = _mm_cvtps_epi32(tmp4);
            v4si tmp5 = _mm_packs_epi32(tmp1_int, tmp2_int);
            v4si tmp6 = _mm_packs_epi32(tmp3_int, tmp4_int);
            v4si tmp7 = _mm_packus_epi16(tmp5, tmp6);
            _mm_store_si128((__m128i *) (dst + i), tmp7);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp3 = _mm_loadu_ps(src + i + 2 * SSE_LEN_FLOAT);
            v4sf src_tmp4 = _mm_loadu_ps(src + i + 3 * SSE_LEN_FLOAT);
            v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
            v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
            v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
            v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
            v4si tmp1_int = _mm_cvtps_epi32(tmp1);
            v4si tmp2_int = _mm_cvtps_epi32(tmp2);
            v4si tmp3_int = _mm_cvtps_epi32(tmp3);
            v4si tmp4_int = _mm_cvtps_epi32(tmp4);
            v4si tmp5 = _mm_packs_epi32(tmp1_int, tmp2_int);
            v4si tmp6 = _mm_packs_epi32(tmp3_int, tmp4_int);
            v4si tmp7 = _mm_packus_epi16(tmp5, tmp6);
            _mm_storeu_si128((__m128i *) (dst + i), tmp7);
        }
    }

    if (rounding_mode == RndFinancial) {
        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);
        }
        _MM_SET_ROUNDING_MODE(_mm_rounding_ori);  // restore previous rounding mode
        fesetround(rounding_ori);
    }
}
#else
static inline void convertFloat32ToU8_128(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= (4 * SSE_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    // Default bankers rounding => round to nearest even
    if (rounding_mode == RndFinancial) {
        if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
            for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_load_ps(src + i);
                v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
                v4sf src_tmp3 = _mm_load_ps(src + i + 2 * SSE_LEN_FLOAT);
                v4sf src_tmp4 = _mm_load_ps(src + i + 3 * SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(tmp2), _mm_cvtps_pi16(tmp1));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(tmp4), _mm_cvtps_pi16(tmp3));
                _mm_store_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        } else {
            for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_loadu_ps(src + i);
                v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
                v4sf src_tmp3 = _mm_loadu_ps(src + i + 2 * SSE_LEN_FLOAT);
                v4sf src_tmp4 = _mm_loadu_ps(src + i + 3 * SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(tmp2), _mm_cvtps_pi16(tmp1));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(tmp4), _mm_cvtps_pi16(tmp3));
                _mm_storeu_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        }
        _mm_empty();
        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        if (rounding_mode == RndZero) {
            _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
            fesetround(FE_TOWARDZERO);
        } else {
            _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
            fesetround(FE_TONEAREST);
        }

        if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
            for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_load_ps(src + i);
                v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
                v4sf src_tmp3 = _mm_load_ps(src + i + 2 * SSE_LEN_FLOAT);
                v4sf src_tmp4 = _mm_load_ps(src + i + 3 * SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp2, _MM_FROUND_CUR_DIRECTION)),
                                          _mm_cvtps_pi16(_mm_round_ps(tmp1, _MM_FROUND_CUR_DIRECTION)));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp4, _MM_FROUND_CUR_DIRECTION)),
                                          _mm_cvtps_pi16(_mm_round_ps(tmp3, _MM_FROUND_CUR_DIRECTION)));
                _mm_store_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        } else {
            for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_loadu_ps(src + i);
                v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
                v4sf src_tmp3 = _mm_loadu_ps(src + i + 2 * SSE_LEN_FLOAT);
                v4sf src_tmp4 = _mm_loadu_ps(src + i + 3 * SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
                v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
                v4si tmp5 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp2, _MM_FROUND_CUR_DIRECTION)),
                                          _mm_cvtps_pi16(_mm_round_ps(tmp1, _MM_FROUND_CUR_DIRECTION)));
                v4si tmp6 = _mm_set_epi64(_mm_cvtps_pi16(_mm_round_ps(tmp4, _MM_FROUND_CUR_DIRECTION)),
                                          _mm_cvtps_pi16(_mm_round_ps(tmp3, _MM_FROUND_CUR_DIRECTION)));
                _mm_storeu_si128((__m128i *) (dst + i), _mm_packus_epi16(tmp5, tmp6));
            }
        }
        _mm_empty();

        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);
        }
    }
}
#endif

#if 1
static inline void convertFloat32ToI16_128(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= (4 * SSE_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    int _mm_rounding_ori = _MM_GET_ROUNDING_MODE();  // save rounding mode
    int rounding_ori = fegetround();

    if (rounding_mode == RndZero) {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
        fesetround(FE_TOWARDZERO);
    } else if (rounding_mode == RndFinancial) {  // nothing to do, Default bankers rounding => round to nearest even
    } else {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
        fesetround(FE_TONEAREST);
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp3 = _mm_load_ps(src + i + 2 * SSE_LEN_FLOAT);
            v4sf src_tmp4 = _mm_load_ps(src + i + 3 * SSE_LEN_FLOAT);
            v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
            v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
            v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
            v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
            v4si tmp1_int = _mm_cvtps_epi32(tmp1);
            v4si tmp2_int = _mm_cvtps_epi32(tmp2);
            v4si tmp3_int = _mm_cvtps_epi32(tmp3);
            v4si tmp4_int = _mm_cvtps_epi32(tmp4);
            v4si tmp5 = _mm_packs_epi32(tmp1_int, tmp2_int);
            v4si tmp6 = _mm_packs_epi32(tmp3_int, tmp4_int);
            _mm_store_si128((__m128i *) (dst + i), tmp5);
            _mm_store_si128((__m128i *) (dst + i + SSE_LEN_INT16), tmp6);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp3 = _mm_loadu_ps(src + i + 2 * SSE_LEN_FLOAT);
            v4sf src_tmp4 = _mm_loadu_ps(src + i + 3 * SSE_LEN_FLOAT);
            v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
            v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
            v4sf tmp3 = _mm_mul_ps(src_tmp3, scale_fact_vec);
            v4sf tmp4 = _mm_mul_ps(src_tmp4, scale_fact_vec);
            v4si tmp1_int = _mm_cvtps_epi32(tmp1);
            v4si tmp2_int = _mm_cvtps_epi32(tmp2);
            v4si tmp3_int = _mm_cvtps_epi32(tmp3);
            v4si tmp4_int = _mm_cvtps_epi32(tmp4);
            v4si tmp5 = _mm_packs_epi32(tmp1_int, tmp2_int);
            v4si tmp6 = _mm_packs_epi32(tmp3_int, tmp4_int);
            _mm_storeu_si128((__m128i *) (dst + i), tmp5);
            _mm_storeu_si128((__m128i *) (dst + i + SSE_LEN_INT16), tmp6);
        }
    }

    if (rounding_mode == RndFinancial) {
        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (int16_t) (tmp > 32767.0f ? 32767.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (int16_t) (tmp > 32767.0f ? 32767.0f : tmp);
        }
        _MM_SET_ROUNDING_MODE(_mm_rounding_ori);  // restore previous rounding mode
        fesetround(rounding_ori);
    }
}

#else

static inline void convertFloat32ToI16_128(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    //  Default bankers rounding => round to nearest even
    if (rounding_mode == RndFinancial) {
        if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
            for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_load_ps(src + i);
                v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                __m64 cvt1 = _mm_cvtps_pi16(tmp1);
                __m64 cvt2 = _mm_cvtps_pi16(tmp2);
                v4si tmp5 = _mm_set_epi64(cvt2, cvt1);
                _mm_store_si128((__m128i *) (dst + i), tmp5);
            }
        } else {
            for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_loadu_ps(src + i);
                v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                __m64 cvt1 = _mm_cvtps_pi16(tmp1);
                __m64 cvt2 = _mm_cvtps_pi16(tmp2);
                v4si tmp5 = _mm_set_epi64(cvt2, cvt1);
                _mm_storeu_si128((__m128i *) (dst + i), tmp5);
            }
        }
        _mm_empty();

        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (int16_t) (tmp > 32767.0f ? 32767.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        if (rounding_mode == RndZero) {
            _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
            fesetround(FE_TOWARDZERO);
        } else {
            _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
            fesetround(FE_TONEAREST);
        }
        if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
            for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_load_ps(src + i);
                v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                __m64 cvt1 = _mm_cvtps_pi16(_mm_round_ps(tmp1, _MM_FROUND_CUR_DIRECTION));
                __m64 cvt2 = _mm_cvtps_pi16(_mm_round_ps(tmp2, _MM_FROUND_CUR_DIRECTION));
                v4si tmp5 = _mm_set_epi64(cvt2, cvt1);
                _mm_store_si128((__m128i *) (dst + i), tmp5);
            }
        } else {
            for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                v4sf src_tmp1 = _mm_loadu_ps(src + i);
                v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
                v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
                v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
                __m64 cvt1 = _mm_cvtps_pi16(_mm_round_ps(tmp1, _MM_FROUND_CUR_DIRECTION));
                __m64 cvt2 = _mm_cvtps_pi16(_mm_round_ps(tmp2, _MM_FROUND_CUR_DIRECTION));
                v4si tmp5 = _mm_set_epi64(cvt2, cvt1);
                _mm_storeu_si128((__m128i *) (dst + i), tmp5);
            }
        }
        _mm_empty();

        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (int16_t) (tmp > 32767.0f ? 32767.0f : tmp);
        }
    }
}
#endif

static inline void convertFloat32ToU16_128(float *src, uint16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    int _mm_rounding_ori = _MM_GET_ROUNDING_MODE();  // save rounding mode
    int rounding_ori = fegetround();

    if (rounding_mode == RndZero) {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
        fesetround(FE_TOWARDZERO);
    } else if (rounding_mode == RndFinancial) {  // nothing to do, Default bankers rounding => round to nearest even
    } else {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
        fesetround(FE_TONEAREST);
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
            v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
            v4si tmp1_int = _mm_cvtps_epi32(tmp1);
            v4si tmp2_int = _mm_cvtps_epi32(tmp2);
            v4si tmp5 = _mm_packus_epi32(tmp1_int, tmp2_int);
            _mm_store_si128((__m128i *) (dst + i), tmp5);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf tmp1 = _mm_mul_ps(src_tmp1, scale_fact_vec);
            v4sf tmp2 = _mm_mul_ps(src_tmp2, scale_fact_vec);
            v4si tmp1_int = _mm_cvtps_epi32(tmp1);
            v4si tmp2_int = _mm_cvtps_epi32(tmp2);
            v4si tmp5 = _mm_packus_epi32(tmp1_int, tmp2_int);
            _mm_storeu_si128((__m128i *) (dst + i), tmp5);
        }
    }

    if (rounding_mode == RndFinancial) {
        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint16_t) (tmp > 65535.0f ? 65535.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (uint16_t) (tmp > 65535.0f ? 65535.0f : tmp);  // round to nearest even with round(x/2)*2
        }
        _MM_SET_ROUNDING_MODE(_mm_rounding_ori);  // restore previous rounding mode
        fesetround(rounding_ori);
    }
}

static inline void convertInt16ToFloat32_128(int16_t *src, float *dst, int len, int scale_factor)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = _mm_set1_ps(scale_fact_mult);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4si vec = _mm_load_si128((__m128i *) (src + i));  // loads 1 2 3 4 5 6 7 8 8
            v4si low = _mm_unpacklo_epi16(vec, vec);           // low 1 1 2 2 3 3 4 4
            v4si high = _mm_unpackhi_epi16(vec, vec);          // high 5 5 6 6 7 7 8 8
            low = _mm_srai_epi32(low, 0x10);                   // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm_srai_epi32(high, 0x10);                 // make high 5 -1 6 -1 7 -1 8 -1

            // convert the vector to float and scale it
            v4sf floatlo = _mm_mul_ps(_mm_cvtepi32_ps(low), scale_fact_vec);
            v4sf floathi = _mm_mul_ps(_mm_cvtepi32_ps(high), scale_fact_vec);

            _mm_store_ps(dst + i, floatlo);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, floathi);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4si vec = _mm_loadu_si128((__m128i *) (src + i));  // loads 1 2 3 4 5 6 7 8 8
            v4si low = _mm_unpacklo_epi16(vec, vec);            // low 1 1 2 2 3 3 4 4
            v4si high = _mm_unpackhi_epi16(vec, vec);           // high 5 5 6 6 7 7 8 8
            low = _mm_srai_epi32(low, 0x10);                    // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm_srai_epi32(high, 0x10);                  // make high 5 -1 6 -1 7 -1 8 -1

            // convert the vector to float and scale it
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

// converts 32bits complex float to two arrays real and im
static inline void cplxtoreal128f(complex32_t *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (4 * SSE_LEN_FLOAT);
    stop_len *= 4 * SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned3((uintptr_t) (src), (uintptr_t) (dstRe), (uintptr_t) (dstIm), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 vec1 = _mm_load2_ps((float const *) (src) + i);
            v4sfx2 vec2 = _mm_load2_ps((float const *) (src) + i + 2 * SSE_LEN_FLOAT);
            _mm_store_ps(dstRe + j, vec1.val[0]);
            _mm_store_ps(dstIm + j, vec1.val[1]);
            _mm_store_ps(dstRe + j + SSE_LEN_FLOAT, vec2.val[0]);
            _mm_store_ps(dstIm + j + SSE_LEN_FLOAT, vec2.val[1]);
            j += 2 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 vec1 = _mm_load2u_ps((float const *) (src) + i);
            v4sfx2 vec2 = _mm_load2u_ps((float const *) (src) + i + 2 * SSE_LEN_FLOAT);
            _mm_storeu_ps(dstRe + j, vec1.val[0]);
            _mm_storeu_ps(dstIm + j, vec1.val[1]);
            _mm_storeu_ps(dstRe + j + SSE_LEN_FLOAT, vec2.val[0]);
            _mm_storeu_ps(dstIm + j + SSE_LEN_FLOAT, vec2.val[1]);
            j += 2 * SSE_LEN_FLOAT;
        }
    }

    for (int i = j; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
}

static inline void realtocplx128f(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf re = _mm_load_ps(srcRe + i);
            v4sf im = _mm_load_ps(srcIm + i);
            v4sf re2 = _mm_load_ps(srcRe + i + SSE_LEN_FLOAT);
            v4sf im2 = _mm_load_ps(srcIm + i + SSE_LEN_FLOAT);
            v4sfx2 reim = {{re, im}};
            v4sfx2 reim2 = {{re2, im2}};
            _mm_store2_ps((float *) (dst) + j, reim);
            _mm_store2_ps((float *) (dst) + j + 2 * SSE_LEN_FLOAT, reim2);
            j += 4 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf re = _mm_loadu_ps(srcRe + i);
            v4sf im = _mm_loadu_ps(srcIm + i);
            v4sf re2 = _mm_loadu_ps(srcRe + i + SSE_LEN_FLOAT);
            v4sf im2 = _mm_loadu_ps(srcIm + i + SSE_LEN_FLOAT);
            v4sfx2 reim = {{re, im}};
            v4sfx2 reim2 = {{re2, im2}};
            _mm_store2u_ps((float *) (dst) + j, reim);
            _mm_store2u_ps((float *) (dst) + j + 2 * SSE_LEN_FLOAT, reim2);
            j += 4 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
}

static inline void convert128_64f32f(double *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

// prefer 2 unaligned loads instead of shuffles
#if 1
static inline void convert128_32f64f(float *src, double *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp_hi = _mm_loadu_ps(src + i + SSE_LEN_FLOAT / 2);
            v4sf src_tmp_hi2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT + SSE_LEN_FLOAT / 2);
            v2sd dst_tmp = _mm_cvtps_pd(src_tmp);
            v2sd dst_tmp_hi = _mm_cvtps_pd(src_tmp_hi);
            v2sd dst_tmp2 = _mm_cvtps_pd(src_tmp2);
            v2sd dst_tmp_hi2 = _mm_cvtps_pd(src_tmp_hi2);
            _mm_store_pd(dst + i, dst_tmp);                      // store the c and d converted in 64bits
            _mm_store_pd(dst + i + SSE_LEN_DOUBLE, dst_tmp_hi);  // store the a and b converted in 64bits
            _mm_store_pd(dst + i + 2 * SSE_LEN_DOUBLE, dst_tmp2);
            _mm_store_pd(dst + i + 3 * SSE_LEN_DOUBLE, dst_tmp_hi2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp_hi = _mm_loadu_ps(src + i + SSE_LEN_FLOAT / 2);
            v4sf src_tmp_hi2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT + SSE_LEN_FLOAT / 2);
            v2sd dst_tmp = _mm_cvtps_pd(src_tmp);
            v2sd dst_tmp_hi = _mm_cvtps_pd(src_tmp_hi);
            v2sd dst_tmp2 = _mm_cvtps_pd(src_tmp2);
            v2sd dst_tmp_hi2 = _mm_cvtps_pd(src_tmp_hi2);
            _mm_storeu_pd(dst + i, dst_tmp);                      // store the c and d converted in 64bits
            _mm_storeu_pd(dst + i + SSE_LEN_DOUBLE, dst_tmp_hi);  // store the a and b converted in 64bits
            _mm_storeu_pd(dst + i + 2 * SSE_LEN_DOUBLE, dst_tmp2);
            _mm_storeu_pd(dst + i + 3 * SSE_LEN_DOUBLE, dst_tmp_hi2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (double) src[i];
    }
}
#else
static inline void convert128_32f64f(float *src, double *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp_hi = _mm_shuffle_ps(src_tmp, src_tmp, _MM_SHUFFLE(1, 0, 3, 2));  // rotate vec from abcd to cdab
            v4sf src_tmp_hi2 = _mm_shuffle_ps(src_tmp2, src_tmp2, _MM_SHUFFLE(1, 0, 3, 2));
            v2sd dst_tmp = _mm_cvtps_pd(src_tmp);
            v2sd dst_tmp_hi = _mm_cvtps_pd(src_tmp_hi);
            v2sd dst_tmp2 = _mm_cvtps_pd(src_tmp2);
            v2sd dst_tmp_hi2 = _mm_cvtps_pd(src_tmp_hi2);
            _mm_store_pd(dst + i, dst_tmp);                      // store the c and d converted in 64bits
            _mm_store_pd(dst + i + SSE_LEN_DOUBLE, dst_tmp_hi);  // store the a and b converted in 64bits
            _mm_store_pd(dst + i + 2 * SSE_LEN_DOUBLE, dst_tmp2);
            _mm_store_pd(dst + i + 3 * SSE_LEN_DOUBLE, dst_tmp_hi2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp_hi = _mm_shuffle_ps(src_tmp, src_tmp, _MM_SHUFFLE(1, 0, 3, 2));  // rotate vec from abcd to cdab
            v4sf src_tmp_hi2 = _mm_shuffle_ps(src_tmp2, src_tmp2, _MM_SHUFFLE(1, 0, 3, 2));
            v2sd dst_tmp = _mm_cvtps_pd(src_tmp);
            v2sd dst_tmp_hi = _mm_cvtps_pd(src_tmp_hi);
            v2sd dst_tmp2 = _mm_cvtps_pd(src_tmp2);
            v2sd dst_tmp_hi2 = _mm_cvtps_pd(src_tmp_hi2);
            _mm_storeu_pd(dst + i, dst_tmp);                      // store the c and d converted in 64bits
            _mm_storeu_pd(dst + i + SSE_LEN_DOUBLE, dst_tmp_hi);  // store the a and b converted in 64bits
            _mm_storeu_pd(dst + i + 2 * SSE_LEN_DOUBLE, dst_tmp2);
            _mm_storeu_pd(dst + i + 3 * SSE_LEN_DOUBLE, dst_tmp_hi2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (double) src[i];
    }
}
#endif

static inline void flip128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    int mini = ((len - 1) < (2 * SSE_LEN_FLOAT)) ? (len - 1) : (2 * SSE_LEN_FLOAT);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst + len - SSE_LEN_FLOAT), SSE_LEN_BYTES)) {
        for (int i = 2 * SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp_flip = _mm_shuffle_ps(src_tmp, src_tmp, IMM8_FLIP_VEC);  // rotate vec from abcd to bcba
            v4sf src_tmp_flip2 = _mm_shuffle_ps(src_tmp2, src_tmp2, IMM8_FLIP_VEC);
            _mm_store_ps(dst + len - i - SSE_LEN_FLOAT, src_tmp_flip);  // store the flipped vector
            _mm_store_ps(dst + len - i - 2 * SSE_LEN_FLOAT, src_tmp_flip2);
        }
    } else {
        for (int i = 2 * SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_tmp_flip = _mm_shuffle_ps(src_tmp, src_tmp, IMM8_FLIP_VEC);  // rotate vec from abcd to bcba
            v4sf src_tmp_flip2 = _mm_shuffle_ps(src_tmp2, src_tmp2, IMM8_FLIP_VEC);
            _mm_storeu_ps(dst + len - i - SSE_LEN_FLOAT, src_tmp_flip);  // store the flipped vector
            _mm_storeu_ps(dst + len - i - 2 * SSE_LEN_FLOAT, src_tmp_flip2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void maxevery128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_load_ps(src1 + i);
            v4sf src2_tmp = _mm_load_ps(src2 + i);
            v4sf src1_tmp2 = _mm_load_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf src2_tmp2 = _mm_load_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf max1 = _mm_max_ps(src1_tmp, src2_tmp);
            v4sf max2 = _mm_max_ps(src1_tmp2, src2_tmp2);
            _mm_store_ps(dst + i, max1);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, max2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_loadu_ps(src1 + i);
            v4sf src2_tmp = _mm_loadu_ps(src2 + i);
            v4sf src1_tmp2 = _mm_loadu_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf src2_tmp2 = _mm_loadu_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf max1 = _mm_max_ps(src1_tmp, src2_tmp);
            v4sf max2 = _mm_max_ps(src1_tmp2, src2_tmp2);
            _mm_storeu_ps(dst + i, max1);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, max2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_load_ps(src1 + i);
            v4sf src2_tmp = _mm_load_ps(src2 + i);
            v4sf src1_tmp2 = _mm_load_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf src2_tmp2 = _mm_load_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf min1 = _mm_min_ps(src1_tmp, src2_tmp);
            v4sf min2 = _mm_min_ps(src1_tmp2, src2_tmp2);
            _mm_store_ps(dst + i, min1);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, min2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_loadu_ps(src1 + i);
            v4sf src2_tmp = _mm_loadu_ps(src2 + i);
            v4sf src1_tmp2 = _mm_loadu_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf src2_tmp2 = _mm_loadu_ps(src2 + i + SSE_LEN_FLOAT);
            v4sf min1 = _mm_min_ps(src1_tmp, src2_tmp);
            v4sf min2 = _mm_min_ps(src1_tmp2, src2_tmp2);
            _mm_storeu_ps(dst + i, min1);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, min2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}


static inline void minmax128f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = (len - SSE_LEN_FLOAT) / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    float min_f[SSE_LEN_FLOAT] __attribute__((aligned(SSE_LEN_BYTES)));
    float max_f[SSE_LEN_FLOAT] __attribute__((aligned(SSE_LEN_BYTES)));
    v4sf max_v, min_v, max_v2, min_v2;
    v4sf src_tmp, src_tmp2;

    float min_tmp = src[0];
    float max_tmp = src[0];

    if (len >= SSE_LEN_FLOAT) {
        if (isAligned((uintptr_t) (src), SSE_LEN_BYTES)) {
            src_tmp = _mm_load_ps(src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                src_tmp = _mm_load_ps(src + i);
                src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
                max_v = _mm_max_ps(max_v, src_tmp);
                min_v = _mm_min_ps(min_v, src_tmp);
                max_v2 = _mm_max_ps(max_v2, src_tmp2);
                min_v2 = _mm_min_ps(min_v2, src_tmp2);
            }
        } else {
            src_tmp = _mm_loadu_ps(src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                src_tmp = _mm_loadu_ps(src + i);
                src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
                max_v = _mm_max_ps(max_v, src_tmp);
                min_v = _mm_min_ps(min_v, src_tmp);
                max_v2 = _mm_max_ps(max_v2, src_tmp2);
                min_v2 = _mm_min_ps(min_v2, src_tmp2);
            }
        }

        max_v = _mm_max_ps(max_v, max_v2);
        min_v = _mm_min_ps(min_v, min_v2);

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
    }

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

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_min_ps(src_tmp, tmp);
            v4sf dst_tmp2 = _mm_min_ps(src_tmp2, tmp);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_min_ps(src_tmp, tmp);
            v4sf dst_tmp2 = _mm_min_ps(src_tmp2, tmp);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

// Alternate version
#if 1
static inline void threshold128_gtabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = _mm_set1_ps(value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_sign = _mm_and_ps(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = _mm_min_ps(src_abs, pval);
            v4sf dst_tmp2 = _mm_min_ps(src_abs2, pval);
            dst_tmp = _mm_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm_xor_ps(dst_tmp2, src_sign2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_sign = _mm_and_ps(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = _mm_min_ps(src_abs, pval);
            v4sf dst_tmp2 = _mm_min_ps(src_abs2, pval);
            dst_tmp = _mm_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm_xor_ps(dst_tmp2, src_sign2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
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
#else
static inline void threshold128_gtabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = _mm_set1_ps(value);
    const v4sf mval = _mm_set1_ps(-value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf eqmask2 = _mm_cmpeq_ps(src_abs2, src_tmp2);
            v4sf gtmask = _mm_cmpgt_ps(src_abs, pval);  // if abs(A) > value => 0xFFFFFFFF, else 0
            v4sf gtmask2 = _mm_cmpgt_ps(src_abs2, pval);
            v4sf sval = _mm_blendv_ps(mval, pval, eqmask);  // if A >= 0 value, else -value
            v4sf sval2 = _mm_blendv_ps(mval, pval, eqmask2);
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            v4sf dst_tmp2 = _mm_blendv_ps(src_tmp2, sval2, gtmask2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf eqmask2 = _mm_cmpeq_ps(src_abs2, src_tmp2);
            v4sf gtmask = _mm_cmpgt_ps(src_abs, pval);  // if abs(A) > value => 0xFFFFFFFF, else 0
            v4sf gtmask2 = _mm_cmpgt_ps(src_abs2, pval);
            v4sf sval = _mm_blendv_ps(mval, pval, eqmask);  // if A >= 0 value, else -value
            v4sf sval2 = _mm_blendv_ps(mval, pval, eqmask2);
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            v4sf dst_tmp2 = _mm_blendv_ps(src_tmp2, sval2, gtmask2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
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
#endif

static inline void threshold128_lt_f(float *src, float *dst, int len, float value)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_max_ps(src_tmp, tmp);
            v4sf dst_tmp2 = _mm_max_ps(src_tmp2, tmp);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_max_ps(src_tmp, tmp);
            v4sf dst_tmp2 = _mm_max_ps(src_tmp2, tmp);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

// Alternate version
#if 1

static inline void threshold128_ltabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = _mm_set1_ps(value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_sign = _mm_and_ps(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = _mm_max_ps(src_abs, pval);
            v4sf dst_tmp2 = _mm_max_ps(src_abs2, pval);
            dst_tmp = _mm_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm_xor_ps(dst_tmp2, src_sign2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_sign = _mm_and_ps(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = _mm_max_ps(src_abs, pval);
            v4sf dst_tmp2 = _mm_max_ps(src_abs2, pval);
            dst_tmp = _mm_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm_xor_ps(dst_tmp2, src_sign2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
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
#else
static inline void threshold128_ltabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = _mm_set1_ps(value);
    const v4sf mval = _mm_set1_ps(-value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf eqmask2 = _mm_cmpeq_ps(src_abs2, src_tmp2);
            v4sf max = _mm_max_ps(src_tmp, pval);
            v4sf max2 = _mm_max_ps(src_tmp2, pval);
            v4sf min = _mm_min_ps(src_tmp, mval);
            v4sf min2 = _mm_min_ps(src_tmp2, mval);
            v4sf dst_tmp = _mm_blendv_ps(min, max, eqmask);
            v4sf dst_tmp2 = _mm_blendv_ps(min2, max2, eqmask2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);
            v4sf src_abs2 = _mm_and_ps(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf eqmask = _mm_cmpeq_ps(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4sf eqmask2 = _mm_cmpeq_ps(src_abs2, src_tmp2);
            v4sf max = _mm_max_ps(src_tmp, pval);
            v4sf max2 = _mm_max_ps(src_tmp2, pval);
            v4sf min = _mm_min_ps(src_tmp, mval);
            v4sf min2 = _mm_min_ps(src_tmp2, mval);
            v4sf dst_tmp = _mm_blendv_ps(min, max, eqmask);
            v4sf dst_tmp2 = _mm_blendv_ps(min2, max2, eqmask2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
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
#endif

static inline void threshold128_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v4sf ltlevel_v = _mm_set1_ps(ltlevel);
    const v4sf ltvalue_v = _mm_set1_ps(ltvalue);
    const v4sf gtlevel_v = _mm_set1_ps(gtlevel);
    const v4sf gtvalue_v = _mm_set1_ps(gtvalue);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf lt_mask = _mm_cmplt_ps(src_tmp, ltlevel_v);
            v4sf gt_mask = _mm_cmpgt_ps(src_tmp, gtlevel_v);
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm_store_ps(dst + i, dst_tmp);
            v4sf lt_mask2 = _mm_cmplt_ps(src_tmp2, ltlevel_v);
            v4sf gt_mask2 = _mm_cmpgt_ps(src_tmp2, gtlevel_v);
            v4sf dst_tmp2 = _mm_blendv_ps(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm_blendv_ps(dst_tmp2, gtvalue_v, gt_mask2);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf lt_mask = _mm_cmplt_ps(src_tmp, ltlevel_v);
            v4sf gt_mask = _mm_cmpgt_ps(src_tmp, gtlevel_v);
            v4sf dst_tmp = _mm_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm_storeu_ps(dst + i, dst_tmp);
            v4sf lt_mask2 = _mm_cmplt_ps(src_tmp2, ltlevel_v);
            v4sf gt_mask2 = _mm_cmpgt_ps(src_tmp2, gtlevel_v);
            v4sf dst_tmp2 = _mm_blendv_ps(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm_blendv_ps(dst_tmp2, gtvalue_v, gt_mask2);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), SSE_LEN_BYTES)) {
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

// e^ix = cos(x) + i*sin(x)
static inline void sincos128f_interleaved(float *src, complex32_t *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sfx2 dst_tmp;
            sincos_ps(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm_store2_ps((float *) dst + j, dst_tmp);
            j += 2 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sfx2 dst_tmp;
            sincos_ps(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm_store2u_ps((float *) dst + j, dst_tmp);
            j += 2 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], &(dst[i].im), &(dst[i].re));
    }
}

static inline v4sf coshf_ps(v4sf xx)
{
    v4sf x, y, tmp;
    v4sf xsupmaxlogf;

    x = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, xx);
    xsupmaxlogf = _mm_cmpgt_ps(x, *(v4sf *) _ps_MAXLOGF);

    y = exp_ps_alternate(x);
    tmp = _mm_div_ps(*(v4sf *) _ps_0p5, y);  // or 1/(2*y)
    y = _mm_fmadd_ps_custom(*(v4sf *) _ps_0p5, y, tmp);
    y = _mm_blendv_ps(y, *(v4sf *) _ps_MAXNUMF, xsupmaxlogf);

    return y;
}

static inline void cosh128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, coshf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, coshf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline v4sf sinhf_ps(v4sf x)
{
    v4sf z, z_first_branch, z_second_branch, tmp;
    v4sf xsupmaxlogf, zsup1;
    v4sf sign;

    // x = xx; if x < 0, z = -x, else x
    z = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, x);
    sign = _mm_and_ps(x, *(v4sf *) _ps_sign_mask);

    xsupmaxlogf = _mm_cmpgt_ps(z, *(v4sf *) _ps_MAXLOGF);

    // First branch
    zsup1 = _mm_cmpgt_ps(z, *(v4sf *) _ps_1);
    z_first_branch = exp_ps_alternate(z);
    tmp = _mm_div_ps(*(v4sf *) _ps_min0p5, z_first_branch);
    z_first_branch = _mm_fmadd_ps_custom(*(v4sf *) _ps_0p5, z_first_branch, tmp);

#if 1
    z_first_branch = _mm_xor_ps(z_first_branch, sign);
#else
    v4sf xinf0 = _mm_cmplt_ps(x, _mm_setzero_ps());
    z_first_branch = _mm_blendv_ps(z_first_branch, _mm_xor_ps(*(v4sf *) _ps_neg_sign_mask, z_first_branch), xinf0);
#endif

    // Second branch
    tmp = _mm_mul_ps(x, x);
    z_second_branch = _mm_fmadd_ps_custom(*(v4sf *) _ps_SINH_P0, tmp, *(v4sf *) _ps_SINH_P1);
    z_second_branch = _mm_fmadd_ps_custom(z_second_branch, tmp, *(v4sf *) _ps_SINH_P2);
    z_second_branch = _mm_mul_ps(z_second_branch, tmp);
    z_second_branch = _mm_fmadd_ps_custom(z_second_branch, x, x);

    // Choose between first and second branch
    z = _mm_blendv_ps(z_second_branch, z_first_branch, zsup1);

    // Set value to MAXNUMF if abs(x) > MAGLOGF
    // Set value to -MAXNUMF if abs(x) > MAGLOGF and x < 0
    tmp = _mm_xor_ps(*(v4sf *) _ps_MAXNUMF, sign);
    z = _mm_blendv_ps(z, tmp, xsupmaxlogf);

    return (z);
}

static inline void sinh128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, sinhf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, sinhf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}

static inline v4sf acoshf_ps(v4sf x)
{
    v4sf z, z_first_branch, z_second_branch;
    v4sf xsup1500, zinf0p5, xinf1;
    v4sf tmp;
    xsup1500 = _mm_cmpgt_ps(x, *(v4sf *) _ps_1500);  // return  (logf(x) + LOGE2F)
    xinf1 = _mm_cmplt_ps(x, *(v4sf *) _ps_1);        // return 0

    z = _mm_sub_ps(x, *(v4sf *) _ps_1);

    zinf0p5 = _mm_cmplt_ps(z, *(v4sf *) _ps_0p5);  // first and second branch

    // First Branch (z < 0.5)
    z_first_branch = _mm_fmadd_ps_custom(*(v4sf *) _ps_ACOSH_P0, z, *(v4sf *) _ps_ACOSH_P1);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, z, *(v4sf *) _ps_ACOSH_P2);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, z, *(v4sf *) _ps_ACOSH_P3);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, z, *(v4sf *) _ps_ACOSH_P4);
    z_first_branch = _mm_mul_ps(z_first_branch, _mm_sqrt_ps(z));

    // Second Branch
    z_second_branch = _mm_fmadd_ps_custom(z, x, z);
    z_second_branch = _mm_sqrt_ps(z_second_branch);
    z_second_branch = _mm_add_ps(x, z_second_branch);
    z_second_branch = log_ps(z_second_branch);

    z = _mm_blendv_ps(z_second_branch, z_first_branch, zinf0p5);
    tmp = log_ps(x);
    tmp = _mm_add_ps(tmp, *(v4sf *) _ps_LOGE2F);
    z = _mm_blendv_ps(z, tmp, xsup1500);

#if 1
    z = _mm_andnot_ps(xinf1, z);
#else
    z = _mm_blendv_ps(z, _mm_setzero_ps(), xinf1);
#endif

    return z;
}

static inline void acosh128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, acoshf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, acoshf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline v4sf asinhf_ps(v4sf xx)
{
    v4sf x, tmp, z, z_first_branch, z_second_branch;
    v4sf xxinf0, xsup1500, xinf0p5;

    x = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, xx);
    xsup1500 = _mm_cmpgt_ps(x, *(v4sf *) _ps_1500);
    xinf0p5 = _mm_cmplt_ps(x, *(v4sf *) _ps_0p5);

    // xxinf0 = _mm_cmplt_ps(xx, _mm_setzero_ps());
    xxinf0 = _mm_and_ps(xx, *(v4sf *) _ps_sign_mask);

    tmp = _mm_mul_ps(x, x);
    // First Branch (x < 0.5)
    z_first_branch = _mm_fmadd_ps_custom(*(v4sf *) _ps_ASINH_P0, tmp, *(v4sf *) _ps_ASINH_P1);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, tmp, *(v4sf *) _ps_ASINH_P2);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, tmp, *(v4sf *) _ps_ASINH_P3);
    z_first_branch = _mm_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, x, x);

    // Second Branch
    z_second_branch = _mm_sqrt_ps(_mm_add_ps(tmp, *(v4sf *) _ps_1));
    z_second_branch = log_ps(_mm_add_ps(z_second_branch, x));

    z = _mm_blendv_ps(z_second_branch, z_first_branch, xinf0p5);
    tmp = log_ps(x);
    tmp = _mm_add_ps(tmp, *(v4sf *) _ps_LOGE2F);
    z = _mm_blendv_ps(z, tmp, xsup1500);
    // z = _mm_blendv_ps(z, _mm_xor_ps(*(v4sf *) _ps_neg_sign_mask, z), xxinf0);
    z = _mm_xor_ps(z, xxinf0);
    return z;
}

static inline void asinh128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, asinhf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, asinhf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline v4sf atanhf_ps(v4sf x)
{
    v4sf z, tmp, tmp2, z_first_branch, z_second_branch;
    v4sf xsup1, xinfmin1, zinf1emin4, zinf0p5;

    z = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, x);

    xsup1 = _mm_cmpge_ps(x, *(v4sf *) _ps_1);
    xinfmin1 = _mm_cmple_ps(x, *(v4sf *) _ps_min1);
    zinf1emin4 = _mm_cmplt_ps(z, *(v4sf *) _ps_1emin4);
    zinf0p5 = _mm_cmplt_ps(z, *(v4sf *) _ps_0p5);

    // First branch
    tmp = _mm_mul_ps(x, x);
    z_first_branch = _mm_fmadd_ps_custom(*(v4sf *) _ps_ATANH_P0, tmp, *(v4sf *) _ps_ATANH_P1);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, tmp, *(v4sf *) _ps_ATANH_P2);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, tmp, *(v4sf *) _ps_ATANH_P3);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, tmp, *(v4sf *) _ps_ATANH_P4);
    z_first_branch = _mm_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm_fmadd_ps_custom(z_first_branch, x, x);

    // Second branch
    // precision of rcp vs div?
    tmp = _mm_sub_ps(*(v4sf *) _ps_1, x);
    tmp2 = _mm_rcp_ps(tmp);
    tmp = _mm_fmadd_ps_custom(tmp2, x, tmp2);
    z_second_branch = log_ps(tmp);
    z_second_branch = _mm_mul_ps(*(v4sf *) _ps_0p5, z_second_branch);

    z = _mm_blendv_ps(z_second_branch, z_first_branch, zinf0p5);
    z = _mm_blendv_ps(z, x, zinf1emin4);

    z = _mm_blendv_ps(z, *(v4sf *) _ps_MAXNUMF, xsup1);
    z = _mm_blendv_ps(z, *(v4sf *) _ps_minMAXNUMF, xinfmin1);

    return (z);
}

static inline void atanh128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, atanhf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, atanhf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline v4sf atanf_ps(v4sf xx)
{
    v4sf x, y, z;
    v4sf sign;
    v4sf suptan3pi8, inftan3pi8suppi8;
    v4sf tmp;

    x = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, xx);
    // sign = _mm_cmplt_ps(xx, _mm_setzero_ps());  // 0xFFFFFFFF if x < 0.0, sign = -1
    sign = _mm_and_ps(xx, *(v4sf *) _ps_sign_mask);

    /* range reduction */

    y = _mm_setzero_ps();
    suptan3pi8 = _mm_cmpgt_ps(x, *(v4sf *) _ps_TAN3PI8F);  // if( x > tan 3pi/8 )
    x = _mm_blendv_ps(x, _mm_div_ps(*(v4sf *) _ps_min1, x), suptan3pi8);
    y = _mm_blendv_ps(y, *(v4sf *) _ps_PIO2F, suptan3pi8);

    inftan3pi8suppi8 = _mm_and_ps(_mm_cmple_ps(x, *(v4sf *) _ps_TAN3PI8F), _mm_cmpgt_ps(x, *(v4sf *) _ps_TANPI8F));  // if( x > tan 3pi/8 )

    // To be optimised with RCP?
    x = _mm_blendv_ps(x, _mm_div_ps(_mm_sub_ps(x, *(v4sf *) _ps_1), _mm_add_ps(x, *(v4sf *) _ps_1)), inftan3pi8suppi8);
    y = _mm_blendv_ps(y, *(v4sf *) _ps_PIO4F, inftan3pi8suppi8);

    z = _mm_mul_ps(x, x);

    tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_ATAN_P0, z, *(v4sf *) _ps_ATAN_P1);
    tmp = _mm_fmadd_ps_custom(tmp, z, *(v4sf *) _ps_ATAN_P2);
    tmp = _mm_fmadd_ps_custom(tmp, z, *(v4sf *) _ps_ATAN_P3);
    tmp = _mm_mul_ps(z, tmp);
    tmp = _mm_fmadd_ps_custom(tmp, x, x);

    y = _mm_add_ps(y, tmp);

    // y = _mm_blendv_ps(y, _mm_xor_ps(*(v4sf *) _ps_neg_sign_mask, y), sign);
    y = _mm_xor_ps(y, sign);
    return (y);
}

static inline void atan128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, atanf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, atanf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline v4sf atan2f_ps(v4sf y, v4sf x)
{
    v4sf z, w;
    v4sf xinfzero, yinfzero, xeqzero, yeqzero;
    v4sf xeqzeroandyinfzero, yeqzeroandxinfzero;
    v4sf specialcase;
    v4sf tmp, tmp2;

    xinfzero = _mm_cmplt_ps(x, _mm_setzero_ps());  // code =2
    yinfzero = _mm_cmplt_ps(y, _mm_setzero_ps());  // code = code |1;

    xeqzero = _mm_cmpeq_ps(x, _mm_setzero_ps());
    yeqzero = _mm_cmpeq_ps(y, _mm_setzero_ps());

    xeqzeroandyinfzero = _mm_and_ps(xeqzero, yinfzero);
    yeqzeroandxinfzero = _mm_and_ps(yeqzero, xinfzero);

#if 1
    xeqzeroandyinfzero = _mm_and_ps(xeqzeroandyinfzero, *(v4sf *) _ps_sign_mask);
    tmp = _mm_xor_ps(*(v4sf *) _ps_PIO2F, xeqzeroandyinfzero);  // either PI or -PI
    z = _mm_andnot_ps(yeqzero, tmp);                            // not(yeqzero) and tmp => 0, PI/2, -PI/2
#else
    z = *(v4sf *) _ps_PIO2F;
    z = _mm_blendv_ps(z, *(v4sf *) _ps_mPIO2F, xeqzeroandyinfzero);
    z = _mm_blendv_ps(z, _mm_setzero_ps(), yeqzero);
#endif
    z = _mm_blendv_ps(z, *(v4sf *) _ps_PIF, yeqzeroandxinfzero);
    specialcase = _mm_or_ps(xeqzero, yeqzero);

#if 1
    tmp = _mm_and_ps(*(v4sf *) _ps_PIF, _mm_andnot_ps(yinfzero, xinfzero));
    tmp2 = _mm_and_ps(*(v4sf *) _ps_mPIF, _mm_and_ps(yinfzero, xinfzero));
    w = _mm_add_ps(tmp, tmp2);
#else
    w = _mm_setzero_ps();
    w = _mm_blendv_ps(w, *(v4sf *) _ps_PIF, _mm_andnot_ps(yinfzero, xinfzero));  // y >= 0 && x<0
    w = _mm_blendv_ps(w, *(v4sf *) _ps_mPIF, _mm_and_ps(yinfzero, xinfzero));    // y < 0 && x<0
#endif

    tmp = _mm_div_ps(y, x);
    tmp = atanf_ps(tmp);
    tmp = _mm_add_ps(w, tmp);
    z = _mm_blendv_ps(tmp, z, specialcase);  // atanf(y/x) if not in special case
    return (z);
}

static inline void atan2128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, atan2f_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, atan2f_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}


static inline void atan2128f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sfx2 src_split = _mm_load2_ps((float *) (src) + j);
            v4sfx2 src_split2 = _mm_load2_ps((float *) (src) + j + 2 * SSE_LEN_FLOAT);
            _mm_store_ps(dst + i, atan2f_ps(src_split.val[1], src_split.val[0]));
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, atan2f_ps(src_split2.val[1], src_split2.val[0]));
            j += 4 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sfx2 src_split = _mm_load2u_ps((float *) (src) + j);
            v4sfx2 src_split2 = _mm_load2u_ps((float *) (src) + j + 2 * SSE_LEN_FLOAT);
            _mm_storeu_ps(dst + i, atan2f_ps(src_split.val[1], src_split.val[0]));
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, atan2f_ps(src_split2.val[1], src_split2.val[0]));
            j += 4 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src[i].im, src[i].re);
    }
}

static inline v4sf asinf_ps(v4sf xx)
{
    v4sf a, x, z, z_tmp;
    v4sf sign;
    v4sf ainfem4, asup0p5;
    v4sf tmp;
    x = xx;
    a = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, x);  // fabs(x)
    // sign = _mm_cmplt_ps(x, _mm_setzero_ps());        // 0xFFFFFFFF if x < 0.0
    sign = _mm_and_ps(xx, *(v4sf *) _ps_sign_mask);

    ainfem4 = _mm_cmplt_ps(a, _mm_set1_ps(1.0e-4));  // if( a < 1.0e-4f )

    asup0p5 = _mm_cmpgt_ps(a, *(v4sf *) _ps_0p5);  // if( a > 0.5f ) flag = 1 else 0
    // TODO : optimise with fmsub?
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

    // done:
    z = _mm_blendv_ps(z, a, ainfem4);
    // z = _mm_blendv_ps(z, _mm_xor_ps(*(v4sf *) _ps_neg_sign_mask, z), sign);
    z = _mm_xor_ps(z, sign);

    // if (x > 1.0) then return 0.0
    z = _mm_blendv_ps(z, _mm_setzero_ps(), _mm_cmpgt_ps(x, *(v4sf *) _ps_1));
    return (z);
}

static inline void asin128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, asinf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, asinf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}


static inline v4sf tanhf_ps(v4sf xx)
{
    v4sf x, z, z_first_branch, z_second_branch;
    v4sf xxsup0, xsupmaxlogfdiv2, xsup0p625;

    xxsup0 = _mm_cmpgt_ps(xx, _mm_setzero_ps());
    xsupmaxlogfdiv2 = _mm_cmpgt_ps(xx, *(v4sf *) _ps_MAXLOGFDIV2);

    x = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, xx);

    xsup0p625 = _mm_cmpge_ps(x, *(v4sf *) _ps_0p625);
    x = _mm_blendv_ps(x, exp_ps_alternate(_mm_add_ps(x, x)), xsup0p625);

    // z = 1.0 - 2.0 / (x + 1.0);
    z_first_branch = _mm_add_ps(x, *(v4sf *) _ps_1);
    z_first_branch = _mm_div_ps(*(v4sf *) _ps_min2, z_first_branch);
    z_first_branch = _mm_add_ps(*(v4sf *) _ps_1, z_first_branch);
    z_first_branch = _mm_blendv_ps(_mm_xor_ps(*(v4sf *) _ps_neg_sign_mask, z_first_branch), z_first_branch, xxsup0);

    // z = x * x;
    z = _mm_mul_ps(x, x);

    z_second_branch = _mm_fmadd_ps_custom(z, *(v4sf *) _ps_TANH_P0, *(v4sf *) _ps_TANH_P1);
    z_second_branch = _mm_fmadd_ps_custom(z_second_branch, z, *(v4sf *) _ps_TANH_P2);
    z_second_branch = _mm_fmadd_ps_custom(z_second_branch, z, *(v4sf *) _ps_TANH_P3);
    z_second_branch = _mm_fmadd_ps_custom(z_second_branch, z, *(v4sf *) _ps_TANH_P4);
    z_second_branch = _mm_mul_ps(z_second_branch, z);
    z_second_branch = _mm_fmadd_ps_custom(z_second_branch, xx, xx);

    z = _mm_blendv_ps(z_second_branch, z_first_branch, xsup0p625);
    // if (x > 0.5 * MAXLOGF), return (xx > 0)? 1.0f: -1.0f
    z = _mm_blendv_ps(z, *(v4sf *) _ps_min1, xsupmaxlogfdiv2);
    z = _mm_blendv_ps(z, *(v4sf *) _ps_1, _mm_and_ps(xxsup0, xsupmaxlogfdiv2));

    return (z);
}

static inline void tanh128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, tanhf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, tanhf_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}

static inline v4sf tanf_ps(v4sf xx)
{
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN tanf_ps" ::
                       : "memory");
#endif
    v4sf x, y, z, zz;
    v4si j;  // long?
    v4sf sign, xsupem4;
    v4sf tmp;
    v4si tmpi;
    v4si jandone, jandtwo;

    x = _mm_and_ps(*(v4sf *) _ps_pos_sign_mask, xx);  // fabs(xx) //OK
    sign = _mm_and_ps(xx, *(v4sf *) _ps_sign_mask);

    /* compute x mod PIO4 */
    tmp = _mm_mul_ps(*(v4sf *) _ps_FOPI, x);
    j = _mm_cvttps_epi32(tmp);
#if 1  // convert is faster than round on some targets
    y = _mm_cvtepi32_ps(j);
#else
    y = _mm_round_ps(tmp, ROUNDTOZERO);
#endif

    jandone = _mm_cmpgt_epi32(_mm_and_si128(j, *(v4si *) _pi32_1), _mm_setzero_si128());
    tmp = _mm_and_ps(*(v4sf *) _ps_1, _mm_castsi128_ps(jandone));
    y = _mm_add_ps(y, tmp);
    tmpi = _mm_and_si128(*(v4si *) _pi32_1, jandone);
    j = _mm_add_epi32(j, tmpi);

    z = _mm_fmadd_ps_custom(y, *(v4sf *) _ps_DP1, x);
    z = _mm_fmadd_ps_custom(y, *(v4sf *) _ps_DP2, z);
    z = _mm_fmadd_ps_custom(y, *(v4sf *) _ps_DP3, z);
    zz = _mm_mul_ps(z, z);  // z*z

    // TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_TAN_P0, zz, *(v4sf *) _ps_TAN_P1);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P2);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P3);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P4);
    tmp = _mm_fmadd_ps_custom(tmp, zz, *(v4sf *) _ps_TAN_P5);
    tmp = _mm_mul_ps(zz, tmp);

#if 1  // _mm_fmadd_ps_custom(tmp, z, z) has been optimised to tmp*z and the + z is merged after.
       // some targets, with no FMA or slow blendv should see improvements
    tmp = _mm_mul_ps(tmp, z);
    xsupem4 = _mm_cmpgt_ps(x, *(v4sf *) _ps_1emin4);  // if( x > 1.0e-4 )
    tmp = _mm_and_ps(tmp, xsupem4);
    y = _mm_add_ps(z, tmp);
#else
    tmp = _mm_fmadd_ps_custom(tmp, z, z);
    xsupem4 = _mm_cmpgt_ps(x, *(v4sf *) _ps_1emin4);  // if( x > 1.0e-4 )
    y = _mm_blendv_ps(z, tmp, xsupem4);
#endif

    jandtwo = _mm_cmpgt_epi32(_mm_and_si128(j, *(v4si *) _pi32_2), _mm_setzero_si128());

    // xor(rcp(y)) gives not good enough result
    tmp = _mm_div_ps(*(v4sf *) _ps_min1, y);
    y = _mm_blendv_ps(y, tmp, _mm_castsi128_ps(jandtwo));
    y = _mm_xor_ps(y, sign);

#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END tanf_ps" ::
                       : "memory");
#endif
    return (y);
}

static inline void tan128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, tanf_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, tanf_ps(src_tmp));
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
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
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_load_ps(srcRe + i);
            v4sf im_tmp = _mm_load_ps(srcIm + i);
            v4sf re_tmp2 = _mm_load_ps(srcRe + i + SSE_LEN_FLOAT);
            v4sf im_tmp2 = _mm_load_ps(srcIm + i + SSE_LEN_FLOAT);
            v4sf re_square = _mm_mul_ps(re_tmp, re_tmp);
            v4sf re_square2 = _mm_mul_ps(re_tmp2, re_tmp2);
            v4sf dst_tmp = _mm_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v4sf dst_tmp2 = _mm_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            dst_tmp = _mm_sqrt_ps(dst_tmp);
            dst_tmp2 = _mm_sqrt_ps(dst_tmp2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_loadu_ps(srcRe + i);
            v4sf im_tmp = _mm_loadu_ps(srcIm + i);
            v4sf re_tmp2 = _mm_loadu_ps(srcRe + i + SSE_LEN_FLOAT);
            v4sf im_tmp2 = _mm_loadu_ps(srcIm + i + SSE_LEN_FLOAT);
            v4sf re_square = _mm_mul_ps(re_tmp, re_tmp);
            v4sf re_square2 = _mm_mul_ps(re_tmp2, re_tmp2);
            v4sf dst_tmp = _mm_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v4sf dst_tmp2 = _mm_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            dst_tmp = _mm_sqrt_ps(dst_tmp);
            dst_tmp2 = _mm_sqrt_ps(dst_tmp2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]));
    }
}

static inline void powerspect128f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_load_ps(srcRe + i);
            v4sf im_tmp = _mm_load_ps(srcIm + i);
            v4sf re_tmp2 = _mm_load_ps(srcRe + i + SSE_LEN_FLOAT);
            v4sf im_tmp2 = _mm_load_ps(srcIm + i + SSE_LEN_FLOAT);
            v4sf re_square = _mm_mul_ps(re_tmp, re_tmp);
            v4sf re_square2 = _mm_mul_ps(re_tmp2, re_tmp2);
            v4sf dst_tmp = _mm_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v4sf dst_tmp2 = _mm_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf re_tmp = _mm_loadu_ps(srcRe + i);
            v4sf im_tmp = _mm_loadu_ps(srcIm + i);
            v4sf re_tmp2 = _mm_loadu_ps(srcRe + i + SSE_LEN_FLOAT);
            v4sf im_tmp2 = _mm_loadu_ps(srcIm + i + SSE_LEN_FLOAT);
            v4sf re_square = _mm_mul_ps(re_tmp, re_tmp);
            v4sf re_square2 = _mm_mul_ps(re_tmp2, re_tmp2);
            v4sf dst_tmp = _mm_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v4sf dst_tmp2 = _mm_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]);
    }
}

// Old version
#if 0
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
        dst[i] = sqrtf(src[i].re * src[i].re + (src[i].im * src[i].im));
    }
}

static inline void powerspect128f_interleaved(complex32_t *src, float *dst, int len)
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
            _mm_store_ps(dst + i, square_sum_0123);
            j += 2 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf cplx01 = _mm_loadu_ps((const float *) src + j);
            v4sf cplx23 = _mm_loadu_ps((const float *) src + j + SSE_LEN_FLOAT);  // complex is 2 floats
            v4sf cplx01_square = _mm_mul_ps(cplx01, cplx01);
            v4sf cplx23_square = _mm_mul_ps(cplx23, cplx23);
            v4sf square_sum_0123 = _mm_hadd_ps(cplx01_square, cplx23_square);
            _mm_storeu_ps(dst + i, square_sum_0123);
            j += 2 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i].re * src[i].re + (src[i].im * src[i].im);
    }
}

#else

// 10% faster on Atom? And on newer CPU?
// On ARM 32bits, we could call vmla instead of mul + add
static inline void magnitude128f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= 4 * SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src_split = _mm_load2_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src_split2 = _mm_load2_ps((float *) (src) + j + 2 * SSE_LEN_FLOAT);
            v4sfx2 src_split3 = _mm_load2_ps((float *) (src) + j + 4 * SSE_LEN_FLOAT);
            v4sfx2 src_split4 = _mm_load2_ps((float *) (src) + j + 6 * SSE_LEN_FLOAT);
            v4sf split_square0 = _mm_mul_ps(src_split.val[0], src_split.val[0]);
            v4sf split2_square0 = _mm_mul_ps(src_split2.val[0], src_split2.val[0]);
            v4sf split3_square0 = _mm_mul_ps(src_split3.val[0], src_split3.val[0]);
            v4sf split4_square0 = _mm_mul_ps(src_split4.val[0], src_split4.val[0]);
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            dst_split.val[0] = _mm_sqrt_ps(dst_split.val[0]);
            dst_split.val[1] = _mm_sqrt_ps(dst_split.val[1]);
            dst_split2.val[0] = _mm_sqrt_ps(dst_split2.val[0]);
            dst_split2.val[1] = _mm_sqrt_ps(dst_split2.val[1]);

            _mm_store_ps((float *) (dst) + i, dst_split.val[0]);
            _mm_store_ps((float *) (dst) + i + SSE_LEN_FLOAT, dst_split.val[1]);
            _mm_store_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2.val[0]);
            _mm_store_ps((float *) (dst) + i + 3 * SSE_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src_split = _mm_load2u_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src_split2 = _mm_load2u_ps((float *) (src) + j + 2 * SSE_LEN_FLOAT);
            v4sfx2 src_split3 = _mm_load2u_ps((float *) (src) + j + 4 * SSE_LEN_FLOAT);
            v4sfx2 src_split4 = _mm_load2u_ps((float *) (src) + j + 6 * SSE_LEN_FLOAT);
            v4sf split_square0 = _mm_mul_ps(src_split.val[0], src_split.val[0]);
            v4sf split2_square0 = _mm_mul_ps(src_split2.val[0], src_split2.val[0]);
            v4sf split3_square0 = _mm_mul_ps(src_split3.val[0], src_split3.val[0]);
            v4sf split4_square0 = _mm_mul_ps(src_split4.val[0], src_split4.val[0]);
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            dst_split.val[0] = _mm_sqrt_ps(dst_split.val[0]);
            dst_split.val[1] = _mm_sqrt_ps(dst_split.val[1]);
            dst_split2.val[0] = _mm_sqrt_ps(dst_split2.val[0]);
            dst_split2.val[1] = _mm_sqrt_ps(dst_split2.val[1]);

            _mm_storeu_ps((float *) (dst) + i, dst_split.val[0]);
            _mm_storeu_ps((float *) (dst) + i + SSE_LEN_FLOAT, dst_split.val[1]);
            _mm_storeu_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2.val[0]);
            _mm_storeu_ps((float *) (dst) + i + 3 * SSE_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i].re * src[i].re + (src[i].im * src[i].im));
    }
}

static inline void powerspect128f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= 4 * SSE_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src_split = _mm_load2_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src_split2 = _mm_load2_ps((float *) (src) + j + 2 * SSE_LEN_FLOAT);
            v4sfx2 src_split3 = _mm_load2_ps((float *) (src) + j + 4 * SSE_LEN_FLOAT);
            v4sfx2 src_split4 = _mm_load2_ps((float *) (src) + j + 6 * SSE_LEN_FLOAT);
            v4sf split_square0 = _mm_mul_ps(src_split.val[0], src_split.val[0]);
            v4sf split2_square0 = _mm_mul_ps(src_split2.val[0], src_split2.val[0]);
            v4sf split3_square0 = _mm_mul_ps(src_split3.val[0], src_split3.val[0]);
            v4sf split4_square0 = _mm_mul_ps(src_split4.val[0], src_split4.val[0]);
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            _mm_store_ps((dst + i), dst_split.val[0]);
            _mm_store_ps((dst + i + SSE_LEN_FLOAT), dst_split.val[1]);
            _mm_store_ps((dst + i + 2 * SSE_LEN_FLOAT), dst_split2.val[0]);
            _mm_store_ps((dst + i + 3 * SSE_LEN_FLOAT), dst_split2.val[1]);
            j += 8 * SSE_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src_split = _mm_load2u_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src_split2 = _mm_load2u_ps((float *) (src) + j + 2 * SSE_LEN_FLOAT);
            v4sfx2 src_split3 = _mm_load2u_ps((float *) (src) + j + 4 * SSE_LEN_FLOAT);
            v4sfx2 src_split4 = _mm_load2u_ps((float *) (src) + j + 6 * SSE_LEN_FLOAT);
            v4sf split_square0 = _mm_mul_ps(src_split.val[0], src_split.val[0]);
            v4sf split2_square0 = _mm_mul_ps(src_split2.val[0], src_split2.val[0]);
            v4sf split3_square0 = _mm_mul_ps(src_split3.val[0], src_split3.val[0]);
            v4sf split4_square0 = _mm_mul_ps(src_split4.val[0], src_split4.val[0]);
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            _mm_storeu_ps((dst + i), dst_split.val[0]);
            _mm_storeu_ps((dst + i + SSE_LEN_FLOAT), dst_split.val[1]);
            _mm_storeu_ps((dst + i + 2 * SSE_LEN_FLOAT), dst_split2.val[0]);
            _mm_storeu_ps((dst + i + 3 * SSE_LEN_FLOAT), dst_split2.val[1]);
            j += 8 * SSE_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i].re * src[i].re + (src[i].im * src[i].im);
    }
}
#endif

static inline void subcrev128f(float *src, float value, float *dst, int len)
{
    const v4sf tmp = _mm_set1_ps(value);

    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_sub_ps(tmp, _mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_sub_ps(tmp, _mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value - src[i];
    }
}

static inline void sum128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v4sf vec_acc1 = _mm_setzero_ps();  // initialize the vector accumulator
    v4sf vec_acc2 = _mm_setzero_ps();  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), SSE_LEN_BYTES)) {
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


static inline void sumkahan128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f};

    float tmp_acc = 0.0f;
    v4sf vec_acc1 = _mm_setzero_ps();  // initialize the vector accumulator
    v4sf vec_acc2 = _mm_setzero_ps();  // initialize the vector accumulator
    v4sf vec_cor1 = _mm_setzero_ps();  // initialize the vector accumulator
    v4sf vec_cor2 = _mm_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf value1 = _mm_load_ps(src + i);
            v4sf y1 = _mm_sub_ps(value1, vec_cor1);
            v4sf t1 = _mm_add_ps(vec_acc1, y1);
            vec_cor1 = _mm_sub_ps(t1, vec_acc1);
            vec_cor1 = _mm_sub_ps(vec_cor1, y1);
            vec_acc1 = t1;

            v4sf value2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf y2 = _mm_sub_ps(value2, vec_cor2);
            v4sf t2 = _mm_add_ps(vec_acc2, y2);
            vec_cor2 = _mm_sub_ps(t2, vec_acc2);
            vec_cor2 = _mm_sub_ps(vec_cor2, y2);
            vec_acc2 = t2;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf value1 = _mm_loadu_ps(src + i);
            v4sf y1 = _mm_sub_ps(value1, vec_cor1);
            v4sf t1 = _mm_add_ps(vec_acc1, y1);
            vec_cor1 = _mm_sub_ps(t1, vec_acc1);
            vec_cor1 = _mm_sub_ps(vec_cor1, y1);
            vec_acc1 = t1;

            v4sf value2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf y2 = _mm_sub_ps(value2, vec_cor2);
            v4sf t2 = _mm_add_ps(vec_acc2, y2);
            vec_cor2 = _mm_sub_ps(t2, vec_acc2);
            vec_cor2 = _mm_sub_ps(vec_cor2, y2);
            vec_acc2 = t2;
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

static inline void meankahan128f(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sumkahan128f(src, dst, len);
    *dst *= coeff;
}

static inline void dot128f(float *src1, float *src2, int len, float *dst)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT];
    float tmp_acc = 0.0f;
    v4sf vec_acc1 = _mm_setzero_ps();  // initialize the vector accumulator
    v4sf vec_acc2 = _mm_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf vec_src1_tmp = _mm_load_ps(src1 + i);
            v4sf vec_src1_tmp2 = _mm_load_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf vec_src2_tmp = _mm_load_ps(src2 + i);
            v4sf vec_src2_tmp2 = _mm_load_ps(src2 + i + SSE_LEN_FLOAT);
            vec_acc1 = _mm_fmadd_ps_custom(vec_src1_tmp, vec_src2_tmp, vec_acc1);
            vec_acc2 = _mm_fmadd_ps_custom(vec_src1_tmp2, vec_src2_tmp2, vec_acc2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf vec_src1_tmp = _mm_loadu_ps(src1 + i);
            v4sf vec_src1_tmp2 = _mm_loadu_ps(src1 + i + SSE_LEN_FLOAT);
            v4sf vec_src2_tmp = _mm_loadu_ps(src2 + i);
            v4sf vec_src2_tmp2 = _mm_loadu_ps(src2 + i + SSE_LEN_FLOAT);
            vec_acc1 = _mm_fmadd_ps_custom(vec_src1_tmp, vec_src2_tmp, vec_acc1);
            vec_acc2 = _mm_fmadd_ps_custom(vec_src1_tmp2, vec_src2_tmp2, vec_acc2);
        }
    }
    vec_acc1 = _mm_add_ps(vec_acc1, vec_acc2);
    _mm_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src1[i] * src2[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

    *dst = tmp_acc;
}

static inline void dotc128f(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= (4 * SSE_LEN_FLOAT);

    v4sfx2 vec_acc1 = {{_mm_setzero_ps(), _mm_setzero_ps()}};  // initialize the vector accumulator
    v4sfx2 vec_acc2 = {{_mm_setzero_ps(), _mm_setzero_ps()}};  // initialize the vector accumulator

    complex32_t dst_tmp = {{0.0f, 0.0f}};

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulateRe[SSE_LEN_FLOAT];
    __attribute__((aligned(SSE_LEN_BYTES))) float accumulateIm[SSE_LEN_FLOAT];

    //  (ac -bd) + i(ad + bc)
    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), SSE_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = _mm_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 tmp_split;
            v4sfx2 tmp_split2;
            tmp_split.val[0] = _mm_fnmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = _mm_fnmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = _mm_add_ps(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = _mm_add_ps(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = _mm_add_ps(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = _mm_add_ps(vec_acc2.val[1], tmp_split2.val[1]);
        }
    } else {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2u_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2u_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2u_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2u_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = _mm_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 tmp_split;
            v4sfx2 tmp_split2;
            tmp_split.val[0] = _mm_fnmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = _mm_fnmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = _mm_add_ps(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = _mm_add_ps(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = _mm_add_ps(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = _mm_add_ps(vec_acc2.val[1], tmp_split2.val[1]);
        }
    }

    vec_acc1.val[0] = _mm_add_ps(vec_acc1.val[0], vec_acc2.val[0]);
    vec_acc1.val[1] = _mm_add_ps(vec_acc1.val[1], vec_acc2.val[1]);
    _mm_store_ps(accumulateRe, vec_acc1.val[0]);
    _mm_store_ps(accumulateIm, vec_acc1.val[1]);

    for (int i = stop_len; i < len; i++) {
        dst_tmp.re += src1[i].re * src2[i].re - (src1[i].im * src2[i].im);
        dst_tmp.im += src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }

    dst_tmp.re = dst_tmp.re + accumulateRe[0] + accumulateRe[1] + accumulateRe[2] + accumulateRe[3];
    dst_tmp.im = dst_tmp.im + accumulateIm[0] + accumulateIm[1] + accumulateIm[2] + accumulateIm[3];


    dst->re = dst_tmp.re;
    dst->im = dst_tmp.im;
}

static inline void sqrt128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_sqrt_ps(src_tmp);
            v4sf dst_tmp2 = _mm_sqrt_ps(src_tmp2);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_sqrt_ps(src_tmp);
            v4sf dst_tmp2 = _mm_sqrt_ps(src_tmp2);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}


static inline void round128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTONEAREST);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTONEAREST);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTONEAREST);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTONEAREST);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTOCEIL);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOCEIL);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTOCEIL);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOCEIL);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTOFLOOR);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOFLOOR);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTOFLOOR);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOFLOOR);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}

static inline void trunc128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTOZERO);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOZERO);
            _mm_store_ps(dst + i, dst_tmp);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_round_ps(src_tmp, ROUNDTOZERO);
            v4sf dst_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOZERO);
            _mm_storeu_ps(dst + i, dst_tmp);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

// Old version
#if 0 
// (a + ib )/ (c + id) => (ac + bd)/(c2+d2) + i(bc -ad)/(c2+d2)
static inline void cplxvecdiv128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)

{
    int stop_len = len / (SSE_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * SSE_LEN_FLOAT;   //stop_len << 2;

    int i;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_load_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_load_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v4sf c2d2 = _mm_mul_ps(src2_tmp, src2_tmp);
            c2d2 = _mm_hadd_ps(c2d2, c2d2);
            c2d2 = _mm_shuffle_ps(c2d2, c2d2, _MM_SHUFFLE(1, 1, 0, 0));
            //            c2d2 = _mm_rcp_ps(c2d2);
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);  //a1,a1,a0,a0
            tmp1 = _mm_mul_ps(*(v4sf *) _ps_conj_mask, tmp1);
            v4sf tmp2 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            v4sf tmp3 = _mm_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
            v4sf out = _mm_mul_ps(tmp2, tmp3);                                        // c1b1, b1d1, c0b0, d0b0
            out = _mm_fmadd_ps_custom(tmp1, src2_tmp, out);
            out = _mm_div_ps(out, c2d2);
            _mm_store_ps((float *) (dst) + i, out);
        }
    } else {
        v4sf src1_tmp = _mm_load_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
        v4sf src2_tmp = _mm_load_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
        v4sf c2d2 = _mm_mul_ps(src2_tmp, src2_tmp);
        c2d2 = _mm_hadd_ps(c2d2, c2d2);
        c2d2 = _mm_shuffle_ps(c2d2, c2d2, _MM_SHUFFLE(1, 1, 0, 0));
        v4sf tmp1 = _mm_moveldup_ps(src1_tmp);  //a1,a1,a0,a0
        tmp1 = _mm_mul_ps(*(v4sf *) _ps_conj_mask, tmp1);
        v4sf tmp2 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
        v4sf tmp3 = _mm_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
        v4sf out = _mm_mul_ps(tmp2, tmp3);                                        // c1b1, b1d1, c0b0, d0b0
        out = _mm_fmadd_ps_custom(tmp1, src2_tmp, out);
        out = _mm_div_ps(out, c2d2);
        _mm_store_ps((float *) (dst) + i, out);
    }
    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2[i].re * src2[i].re + src2[i].im * src2[i].im;
        dst[i].re = ((src1[i].re * src2[i].re) + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-(src1[i].re * src2[i].im) + (src2[i].re * src1[i].im)) / c2d2;
    }
}

static inline void cplxvecmul128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * SSE_LEN_FLOAT;   //stop_len << 2;

    int i;
    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_load_ps((float *) (src1) + i);                        // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_load_ps((float *) (src2) + i);                        // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                    //a1,a1,a0,a0
            v4sf tmp2 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            v4sf tmp3 = _mm_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
            v4sf out = _mm_mul_ps(tmp2, tmp3);                                        // c1b1, b1d1, c0b0, d0b0
            out = _mm_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_loadu_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_loadu_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                    //a1,a1,a0,a0
            v4sf tmp2 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            v4sf tmp3 = _mm_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
            v4sf out = _mm_mul_ps(tmp2, tmp3);
            out = _mm_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re - src1[i].im * src2[i].im;
        dst[i].im = (src1[i].re * src2[i].im) + src2[i].re * src1[i].im;
    }
}
#else
static inline void cplxvecdiv128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= 4 * SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf c2 = _mm_mul_ps(src2_split.val[0], src2_split.val[0]);
            v4sf c2d2 = _mm_fmadd_ps_custom(src2_split.val[1], src2_split.val[1], c2);
            v4sf c2_ = _mm_mul_ps(src2_split2.val[0], src2_split2.val[0]);
            v4sf c2d2_ = _mm_fmadd_ps_custom(src2_split2.val[1], src2_split2.val[1], c2_);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf bc = _mm_mul_ps(src1_split.val[1], src2_split.val[0]);     // bc
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf bc2 = _mm_mul_ps(src1_split2.val[1], src2_split2.val[0]);  // bc

            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = _mm_fnmadd_ps_custom(src1_split.val[0], src2_split.val[1], bc);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = _mm_fnmadd_ps_custom(src1_split2.val[0], src2_split2.val[1], bc2);

            dst_split.val[0] = _mm_div_ps(dst_split.val[0], c2d2);
            dst_split.val[1] = _mm_div_ps(dst_split.val[1], c2d2);
            dst_split2.val[0] = _mm_div_ps(dst_split2.val[0], c2d2_);
            dst_split2.val[1] = _mm_div_ps(dst_split2.val[1], c2d2_);

            _mm_store2_ps((float *) (dst) + i, dst_split);
            _mm_store2_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2);
        }
    } else {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2u_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2u_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2u_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2u_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf c2 = _mm_mul_ps(src2_split.val[0], src2_split.val[0]);
            v4sf c2d2 = _mm_fmadd_ps_custom(src2_split.val[1], src2_split.val[1], c2);
            v4sf c2_ = _mm_mul_ps(src2_split2.val[0], src2_split2.val[0]);
            v4sf c2d2_ = _mm_fmadd_ps_custom(src2_split2.val[1], src2_split2.val[1], c2_);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf bc = _mm_mul_ps(src1_split.val[1], src2_split.val[0]);     // bc
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf bc2 = _mm_mul_ps(src1_split2.val[1], src2_split2.val[0]);  // bc

            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = _mm_fnmadd_ps_custom(src1_split.val[0], src2_split.val[1], bc);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = _mm_fnmadd_ps_custom(src1_split2.val[0], src2_split2.val[1], bc2);

            dst_split.val[0] = _mm_div_ps(dst_split.val[0], c2d2);
            dst_split.val[1] = _mm_div_ps(dst_split.val[1], c2d2);
            dst_split2.val[0] = _mm_div_ps(dst_split2.val[0], c2d2_);
            dst_split2.val[1] = _mm_div_ps(dst_split2.val[1], c2d2_);

            _mm_store2u_ps((float *) (dst) + i, dst_split);
            _mm_store2u_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2[i].re * src2[i].re + src2[i].im * src2[i].im;
        dst[i].re = ((src1[i].re * src2[i].re) + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-(src1[i].re * src2[i].im) + (src2[i].re * src1[i].im)) / c2d2;
    }
}

static inline void cplxvecmul128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= 4 * SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = _mm_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fnmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[0], ad);
            dst_split2.val[0] = _mm_fnmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[0], ad2);

            _mm_store2_ps((float *) (dst) + i, dst_split);
            _mm_store2_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2);
        }
    } else {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2u_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2u_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2u_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2u_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = _mm_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fnmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[0], ad);
            dst_split2.val[0] = _mm_fnmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[0], ad2);

            _mm_store2u_ps((float *) (dst) + i, dst_split);
            _mm_store2u_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}
#endif

static inline void cplxvecmul128f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len = stop_len * SSE_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), SSE_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_load_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_load_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_load_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_load_ps((float *) (src2Im) + i);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);
            // v4sf bd = _mm_mul_ps(src1Im_tmp, src2Im_tmp);
            // v4sf ad = _mm_mul_ps(src1Re_tmp, src2Im_tmp);
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm_store_ps(dstRe + i, _mm_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  // ac - bd
            _mm_store_ps(dstIm + i, _mm_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    } else {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_loadu_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_loadu_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_loadu_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_loadu_ps((float *) (src2Im) + i);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm_storeu_ps(dstRe + i, _mm_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  // ac - bd
            _mm_storeu_ps(dstIm + i, _mm_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = (src1Re[i] * src2Re[i]) - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i]);
    }
}

static inline void cplxvecdiv128f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1Re), (uintptr_t) (src2Re), (uintptr_t) (src2Re), SSE_LEN_BYTES) &&
        areAligned3((uintptr_t) (src1Im), (uintptr_t) (dstRe), (uintptr_t) (dstIm), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_load_ps((float *) (src1Re) + i);
            v4sf src1Re_tmp2 = _mm_load_ps((float *) (src1Re) + i + SSE_LEN_FLOAT);
            v4sf src1Im_tmp = _mm_load_ps((float *) (src1Im) + i);
            v4sf src1Im_tmp2 = _mm_load_ps((float *) (src1Im) + i + SSE_LEN_FLOAT);
            v4sf src2Re_tmp = _mm_load_ps((float *) (src2Re) + i);
            v4sf src2Re_tmp2 = _mm_load_ps((float *) (src2Re) + i + SSE_LEN_FLOAT);
            v4sf src2Im_tmp = _mm_load_ps((float *) (src2Im) + i);
            v4sf src2Im_tmp2 = _mm_load_ps((float *) (src2Im) + i + SSE_LEN_FLOAT);

            v4sf c2 = _mm_mul_ps(src2Re_tmp, src2Re_tmp);
            v4sf c2d2 = _mm_fmadd_ps_custom(src2Im_tmp, src2Im_tmp, c2);
            v4sf c2_ = _mm_mul_ps(src2Re_tmp2, src2Re_tmp2);
            v4sf c2d2_ = _mm_fmadd_ps_custom(src2Im_tmp2, src2Im_tmp2, c2_);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);     // ac
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);     // bc
            v4sf ac2 = _mm_mul_ps(src1Re_tmp2, src2Re_tmp2);  // ac
            v4sf bc2 = _mm_mul_ps(src1Im_tmp2, src2Re_tmp2);  // bc

            v4sf dstRe_tmp = _mm_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac);
            v4sf dstRe_tmp2 = _mm_fmadd_ps_custom(src1Im_tmp2, src2Im_tmp2, ac2);
            v4sf dstIm_tmp = _mm_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc);
            v4sf dstIm_tmp2 = _mm_fnmadd_ps_custom(src1Re_tmp2, src2Im_tmp2, bc2);

            dstRe_tmp = _mm_div_ps(dstRe_tmp, c2d2);
            dstIm_tmp = _mm_div_ps(dstIm_tmp, c2d2);
            dstRe_tmp2 = _mm_div_ps(dstRe_tmp2, c2d2_);
            dstIm_tmp2 = _mm_div_ps(dstIm_tmp2, c2d2_);

            _mm_store_ps((float *) (dstRe) + i, dstRe_tmp);
            _mm_store_ps((float *) (dstIm) + i, dstIm_tmp);
            _mm_store_ps((float *) (dstRe) + i + SSE_LEN_FLOAT, dstRe_tmp2);
            _mm_store_ps((float *) (dstIm) + i + SSE_LEN_FLOAT, dstIm_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_loadu_ps((float *) (src1Re) + i);
            v4sf src1Re_tmp2 = _mm_loadu_ps((float *) (src1Re) + i + SSE_LEN_FLOAT);
            v4sf src1Im_tmp = _mm_loadu_ps((float *) (src1Im) + i);
            v4sf src1Im_tmp2 = _mm_loadu_ps((float *) (src1Im) + i + SSE_LEN_FLOAT);
            v4sf src2Re_tmp = _mm_loadu_ps((float *) (src2Re) + i);
            v4sf src2Re_tmp2 = _mm_loadu_ps((float *) (src2Re) + i + SSE_LEN_FLOAT);
            v4sf src2Im_tmp = _mm_loadu_ps((float *) (src2Im) + i);
            v4sf src2Im_tmp2 = _mm_loadu_ps((float *) (src2Im) + i + SSE_LEN_FLOAT);

            v4sf c2 = _mm_mul_ps(src2Re_tmp, src2Re_tmp);
            v4sf c2d2 = _mm_fmadd_ps_custom(src2Im_tmp, src2Im_tmp, c2);
            v4sf c2_ = _mm_mul_ps(src2Re_tmp2, src2Re_tmp2);
            v4sf c2d2_ = _mm_fmadd_ps_custom(src2Im_tmp2, src2Im_tmp2, c2_);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);     // ac
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);     // bc
            v4sf ac2 = _mm_mul_ps(src1Re_tmp2, src2Re_tmp2);  // ac
            v4sf bc2 = _mm_mul_ps(src1Im_tmp2, src2Re_tmp2);  // bc

            v4sf dstRe_tmp = _mm_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac);
            v4sf dstRe_tmp2 = _mm_fmadd_ps_custom(src1Im_tmp2, src2Im_tmp2, ac2);
            v4sf dstIm_tmp = _mm_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc);
            v4sf dstIm_tmp2 = _mm_fnmadd_ps_custom(src1Re_tmp2, src2Im_tmp2, bc2);

            dstRe_tmp = _mm_div_ps(dstRe_tmp, c2d2);
            dstIm_tmp = _mm_div_ps(dstIm_tmp, c2d2);
            dstRe_tmp2 = _mm_div_ps(dstRe_tmp2, c2d2_);
            dstIm_tmp2 = _mm_div_ps(dstIm_tmp2, c2d2_);

            _mm_storeu_ps((float *) (dstRe) + i, dstRe_tmp);
            _mm_storeu_ps((float *) (dstIm) + i, dstIm_tmp);
            _mm_storeu_ps((float *) (dstRe) + i + SSE_LEN_FLOAT, dstRe_tmp2);
            _mm_storeu_ps((float *) (dstIm) + i + SSE_LEN_FLOAT, dstIm_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2Re[i] * src2Re[i] + src2Im[i] * src2Im[i];
        dstRe[i] = (src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i])) / c2d2;
        dstIm[i] = (-src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i])) / c2d2;
    }
}

// Old version
#if 0
// out = a * conj(b)
static inline void cplxconjvecmul128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * SSE_LEN_FLOAT;   //stop_len << 2;

    int i;
    //const v4sf conj_mask = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_load_ps((float *) (src1) + i);                        // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_load_ps((float *) (src2) + i);                        // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                    //a1,a1,a0,a0
            v4sf tmp2 = _mm_mul_ps(tmp1, src2_tmp);                                   //a1d1,a1c1,a0d0,a0c0
            v4sf tmp3 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            v4sf tmp4 = _mm_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
            v4sf out = _mm_mul_ps(tmp3, tmp4);                                        // c1b1,b1d1,c0b0,d0b0
            out = _mm_fmadd_ps_custom(*(v4sf *) _ps_conj_mask, tmp2, out);            // c1b1 -a1d1,b1d1 + a1c1,c0b0 -a0d0,d0b0 + a0c0
            _mm_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_loadu_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_loadu_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                    //a1,a1,a0,a0
            v4sf tmp2 = _mm_mul_ps(tmp1, src2_tmp);                                   //a1d1,a1c1,a0d0,a0c0
            v4sf tmp3 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  //c1,d1,c0,d0
            v4sf tmp4 = _mm_movehdup_ps(src1_tmp);                                    //b1,b1,b0,b0
            v4sf out = _mm_mul_ps(tmp3, tmp4);                                        // c1b1,b1d1,c0b0,d0b0
            out = _mm_fmadd_ps_custom(*(v4sf *) _ps_conj_mask, tmp2, out);            // c1b1 -a1d1,b1d1 + a1c1,c0b0 -a0d0,d0b0 + a0c0
            _mm_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + (src1[i].im * src2[i].im);
        dst[i].im = (src2[i].re * src1[i].im) - src1[i].re * src2[i].im;
    }
}

#else

static inline void cplxconjvecmul128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (4 * SSE_LEN_FLOAT);
    stop_len *= (4 * SSE_LEN_FLOAT);

    // vmls(a,b,c) => a -(b*c)
    //  (ac -bd) + i(ad + bc)
    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf bc = _mm_mul_ps(src1_split.val[1], src2_split.val[0]);     // bc
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf bc2 = _mm_mul_ps(src1_split2.val[1], src2_split2.val[0]);  // bc
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = _mm_fnmadd_ps_custom(src1_split.val[0], src2_split.val[1], bc);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = _mm_fnmadd_ps_custom(src1_split2.val[0], src2_split2.val[1], bc2);

            _mm_store2_ps((float *) (dst) + i, dst_split);
            _mm_store2_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2);
        }
    } else {
        for (int i = 0; i < 2 * stop_len; i += 4 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2u_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2u_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = _mm_load2u_ps((float *) (src1) + i + 2 * SSE_LEN_FLOAT);
            v4sfx2 src2_split2 = _mm_load2u_ps((float *) (src2) + i + 2 * SSE_LEN_FLOAT);
            v4sf ac = _mm_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf bc = _mm_mul_ps(src1_split.val[1], src2_split.val[0]);     // bc
            v4sf ac2 = _mm_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf bc2 = _mm_mul_ps(src1_split2.val[1], src2_split2.val[0]);  // bc
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = _mm_fmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = _mm_fnmadd_ps_custom(src1_split.val[0], src2_split.val[1], bc);
            dst_split2.val[0] = _mm_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = _mm_fnmadd_ps_custom(src1_split2.val[0], src2_split2.val[1], bc2);

            _mm_store2u_ps((float *) (dst) + i, dst_split);
            _mm_store2u_ps((float *) (dst) + i + 2 * SSE_LEN_FLOAT, dst_split2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + (src1[i].im * src2[i].im);
        dst[i].im = -src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

#endif

// X = a + ib
// Yconj = c - id
// Z = (ac + bd) + i*(-ad + bc)
// less precise than the _precise (with double) version
static inline void cplxconjvecmul128f_kahan(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * SSE_LEN_FLOAT;   // stop_len << 2;

    int i;
    // const v4sf conj_mask = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_load_ps((float *) (src1) + i);                        // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_load_ps((float *) (src2) + i);                        // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            tmp1 = _mm_mul_ps(*(v4sf *) _ps_conj_mask, tmp1);                         // multiplying by -1 should not induce error?
            v4sf tmp2 = _mm_mul_ps(tmp1, src2_tmp);                                   // a1d1,a1c1,a0d0,a0c0
            v4sf tmp2err = _mm_fnmadd_ps_custom(tmp1, src2_tmp, tmp2);                // error from previous computation
            v4sf tmp3 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v4sf tmp4 = _mm_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v4sf out = _mm_fmadd_ps_custom(tmp3, tmp4, tmp2);                         // c1b1 -a1d1,b1d1 + a1c1,c0b0 -a0d0,d0b0 + a0c0
            out = _mm_sub_ps(out, tmp2err);
            _mm_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1_tmp = _mm_loadu_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v4sf src2_tmp = _mm_loadu_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v4sf tmp1 = _mm_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            tmp1 = _mm_mul_ps(*(v4sf *) _ps_conj_mask, tmp1);                         // multiplying by -1 should not induce error?
            v4sf tmp2 = _mm_mul_ps(tmp1, src2_tmp);                                   // a1d1,a1c1,a0d0,a0c0
            v4sf tmp2err = _mm_fnmadd_ps_custom(tmp1, src2_tmp, tmp2);                // error from previous computation
            v4sf tmp3 = _mm_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v4sf tmp4 = _mm_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v4sf out = _mm_fmadd_ps_custom(tmp3, tmp4, tmp2);                         // c1b1 -a1d1,b1d1 + a1c1,c0b0 -a0d0,d0b0 + a0c0
            out = _mm_sub_ps(out, tmp2err);
            _mm_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + (src1[i].im * src2[i].im);
        dst[i].im = (src2[i].re * src1[i].im) - src1[i].re * src2[i].im;
    }
}


// Switch to double precision
// Work in progress
static inline void cplxconjvecmul128f_precise(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    // vmls(a,b,c) => a -(b*c)
    //  (ac -bd) + i(ad + bc)
    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3

            v2sd src1_split_lowd_0 = _mm_cvtps_pd(src1_split.val[0]);
            v2sd src1_split_lowd_1 = _mm_cvtps_pd(src1_split.val[1]);
            v2sd src2_split_lowd_0 = _mm_cvtps_pd(src2_split.val[0]);
            v2sd src2_split_lowd_1 = _mm_cvtps_pd(src2_split.val[1]);

            v2sd src1_split_highd_0 = _mm_cvtps_pd(_mm_shuffle_ps(src1_split.val[0], src1_split.val[0], _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src1_split_highd_1 = _mm_cvtps_pd(_mm_shuffle_ps(src1_split.val[1], src1_split.val[1], _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2_split_highd_0 = _mm_cvtps_pd(_mm_shuffle_ps(src2_split.val[0], src2_split.val[0], _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2_split_highd_1 = _mm_cvtps_pd(_mm_shuffle_ps(src2_split.val[1], src2_split.val[1], _MM_SHUFFLE(1, 0, 3, 2)));

            v2sd ac_lowd = _mm_mul_pd(src1_split_lowd_0, src2_split_lowd_0);     // ac
            v2sd bc_lowd = _mm_mul_pd(src1_split_lowd_1, src2_split_lowd_0);     // bc
            v2sd ac_highd = _mm_mul_pd(src1_split_highd_0, src2_split_highd_0);  // ac
            v2sd bc_highd = _mm_mul_pd(src1_split_highd_1, src2_split_highd_0);  // bc

            v4sfx2 dst_split;
            v2sd re_lowd = _mm_fmadd_pd_custom(src1_split_lowd_1, src2_split_lowd_1, ac_lowd);
            v2sd re_highd = _mm_fmadd_pd_custom(src1_split_highd_1, src2_split_highd_1, ac_highd);
            v2sd im_lowd = _mm_fnmadd_pd_custom(src1_split_lowd_0, src2_split_lowd_1, bc_lowd);
            v2sd im_highd = _mm_fnmadd_pd_custom(src1_split_highd_0, src2_split_highd_1, bc_highd);

            dst_split.val[0] = _mm_movelh_ps(_mm_cvtpd_ps(re_lowd), _mm_cvtpd_ps(re_highd));
            dst_split.val[1] = _mm_movelh_ps(_mm_cvtpd_ps(im_lowd), _mm_cvtpd_ps(im_highd));

            _mm_store2_ps((float *) (dst) + i, dst_split);
        }
    } else {
        for (int i = 0; i < 2 * stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sfx2 src1_split = _mm_load2u_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = _mm_load2u_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3

            v2sd src1_split_lowd_0 = _mm_cvtps_pd(src1_split.val[0]);
            v2sd src1_split_lowd_1 = _mm_cvtps_pd(src1_split.val[1]);
            v2sd src2_split_lowd_0 = _mm_cvtps_pd(src2_split.val[0]);
            v2sd src2_split_lowd_1 = _mm_cvtps_pd(src2_split.val[1]);

            v2sd src1_split_highd_0 = _mm_cvtps_pd(_mm_shuffle_ps(src1_split.val[0], src1_split.val[0], _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src1_split_highd_1 = _mm_cvtps_pd(_mm_shuffle_ps(src1_split.val[1], src1_split.val[1], _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2_split_highd_0 = _mm_cvtps_pd(_mm_shuffle_ps(src2_split.val[0], src2_split.val[0], _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2_split_highd_1 = _mm_cvtps_pd(_mm_shuffle_ps(src2_split.val[1], src2_split.val[1], _MM_SHUFFLE(1, 0, 3, 2)));

            v2sd ac_lowd = _mm_mul_pd(src1_split_lowd_0, src2_split_lowd_0);     // ac
            v2sd bc_lowd = _mm_mul_pd(src1_split_lowd_1, src2_split_lowd_0);     // bc
            v2sd ac_highd = _mm_mul_pd(src1_split_highd_0, src2_split_highd_0);  // ac
            v2sd bc_highd = _mm_mul_pd(src1_split_highd_1, src2_split_highd_0);  // bc

            v4sfx2 dst_split;
            v2sd re_lowd = _mm_fmadd_pd_custom(src1_split_lowd_1, src2_split_lowd_1, ac_lowd);
            v2sd re_highd = _mm_fmadd_pd_custom(src1_split_highd_1, src2_split_highd_1, ac_highd);
            v2sd im_lowd = _mm_fnmadd_pd_custom(src1_split_lowd_0, src2_split_lowd_1, bc_lowd);
            v2sd im_highd = _mm_fnmadd_pd_custom(src1_split_highd_0, src2_split_highd_1, bc_highd);

            dst_split.val[0] = _mm_movelh_ps(_mm_cvtpd_ps(re_lowd), _mm_cvtpd_ps(re_highd));
            dst_split.val[1] = _mm_movelh_ps(_mm_cvtpd_ps(im_lowd), _mm_cvtpd_ps(im_highd));

            _mm_store2u_ps((float *) (dst) + i, dst_split);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + (src1[i].im * src2[i].im);
        dst[i].im = -src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxconjvecmul128f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len = stop_len * SSE_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), SSE_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_load_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_load_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_load_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_load_ps((float *) (src2Im) + i);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);
            // v4sf bd = _mm_mul_ps(src1Im_tmp, src2Im_tmp);
            // v4sf ad = _mm_mul_ps(src1Re_tmp, src2Im_tmp);
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm_store_ps(dstRe + i, _mm_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   // ac + bd
            _mm_store_ps(dstIm + i, _mm_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    } else {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_loadu_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_loadu_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_loadu_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_loadu_ps((float *) (src2Im) + i);
            v4sf ac = _mm_mul_ps(src1Re_tmp, src2Re_tmp);
            // v4sf bd = _mm_mul_ps(src1Im_tmp, src2Im_tmp);
            // v4sf ad = _mm_mul_ps(src1Re_tmp, src2Im_tmp);
            v4sf bc = _mm_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm_storeu_ps(dstRe + i, _mm_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   // ac + bd
            _mm_storeu_ps(dstIm + i, _mm_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i]);
        dstIm[i] = (src2Re[i] * src1Im[i]) - src1Re[i] * src2Im[i];
    }
}

// Implements the Kahan complex multiply to minimize error
//  X = a + ib
//  Yconj = c - id
//  Z = (ac + bd) + i*(-ad + bc)
// RN = round to nearest
//  Less precise than the 64bit double version
static inline void cplxconjvecmul128f_split_kahan(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len = stop_len * SSE_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), SSE_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_load_ps((float *) (src1Re) + i);  // a
            v4sf b = _mm_load_ps((float *) (src1Im) + i);  // b
            v4sf c = _mm_load_ps((float *) (src2Re) + i);  // c
            v4sf d = _mm_load_ps((float *) (src2Im) + i);  // d

            v4sf p1 = _mm_mul_ps(a, c);                  // RN(ac)
            v4sf p1pbd = _mm_fmadd_ps_custom(b, d, p1);  // RN(p1 + bd)
            p1 = _mm_fnmadd_ps_custom(a, c, p1);         // -ac + p1. How to directly get ac -p1?
            v4sf real = _mm_sub_ps(p1pbd, p1);
            _mm_store_ps(dstRe + i, real);

            v4sf p3 = _mm_mul_ps(b, c);                    // RN(bc)
            v4sf madpp3 = _mm_fnmadd_ps_custom(a, d, p3);  // RN(-ad + p3)
            p3 = _mm_fnmadd_ps_custom(b, c, p3);           // -bc + p3.
            v4sf imag = _mm_sub_ps(madpp3, p3);
            _mm_store_ps(dstIm + i, imag);
        }
    } else {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf a = _mm_loadu_ps((float *) (src1Re) + i);  // a
            v4sf b = _mm_loadu_ps((float *) (src1Im) + i);  // b
            v4sf c = _mm_loadu_ps((float *) (src2Re) + i);  // c
            v4sf d = _mm_loadu_ps((float *) (src2Im) + i);  // d

            v4sf p1 = _mm_mul_ps(a, c);                  // RN(ac)
            v4sf p1pbd = _mm_fmadd_ps_custom(b, d, p1);  // RN(p1 + bd)
            p1 = _mm_fnmadd_ps_custom(a, c, p1);         // -ac + p1. How to directly get ac -p1?
            v4sf real = _mm_sub_ps(p1pbd, p1);
            _mm_storeu_ps(dstRe + i, real);

            v4sf p3 = _mm_mul_ps(b, c);                    // RN(bc)
            v4sf madpp3 = _mm_fnmadd_ps_custom(a, d, p3);  // RN(-ad + p3)
            p3 = _mm_fnmadd_ps_custom(b, c, p3);           // -bc + p3.
            v4sf imag = _mm_sub_ps(madpp3, p3);
            _mm_storeu_ps(dstIm + i, imag);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + src1Im[i] * src2Im[i];
        dstIm[i] = src2Re[i] * src1Im[i] - src1Re[i] * src2Im[i];
    }
}

// switch to double precision for the computation
static inline void cplxconjvecmul128f_split_precise(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len = stop_len * SSE_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), SSE_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_load_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_load_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_load_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_load_ps((float *) (src2Im) + i);

            v2sd src1Re_lowd = _mm_cvtps_pd(src1Re_tmp);
            v2sd src1Im_lowd = _mm_cvtps_pd(src1Im_tmp);
            v2sd src2Re_lowd = _mm_cvtps_pd(src2Re_tmp);
            v2sd src2Im_lowd = _mm_cvtps_pd(src2Im_tmp);

            v2sd src1Re_highd = _mm_cvtps_pd(_mm_shuffle_ps(src1Re_tmp, src1Re_tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src1Im_highd = _mm_cvtps_pd(_mm_shuffle_ps(src1Im_tmp, src1Im_tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2Re_highd = _mm_cvtps_pd(_mm_shuffle_ps(src2Re_tmp, src2Re_tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2Im_highd = _mm_cvtps_pd(_mm_shuffle_ps(src2Im_tmp, src2Im_tmp, _MM_SHUFFLE(1, 0, 3, 2)));

            v2sd ac_lowd = _mm_mul_pd(src1Re_lowd, src2Re_lowd);
            v2sd bc_lowd = _mm_mul_pd(src1Im_lowd, src2Re_lowd);
            v2sd ac_highd = _mm_mul_pd(src1Re_highd, src2Re_highd);
            v2sd bc_highd = _mm_mul_pd(src1Im_highd, src2Re_highd);

            v2sd dstRe_lowd = _mm_fmadd_pd_custom(src1Im_lowd, src2Im_lowd, ac_lowd);
            v2sd dstIm_lowd = _mm_fnmadd_pd_custom(src1Re_lowd, src2Im_lowd, bc_lowd);
            v2sd dstRe_highd = _mm_fmadd_pd_custom(src1Im_highd, src2Im_highd, ac_highd);
            v2sd dstIm_highd = _mm_fnmadd_pd_custom(src1Re_highd, src2Im_highd, bc_highd);

            v4sf dstRe_tmp = _mm_movelh_ps(_mm_cvtpd_ps(dstRe_lowd), _mm_cvtpd_ps(dstRe_highd));
            v4sf dstIm_tmp = _mm_movelh_ps(_mm_cvtpd_ps(dstIm_lowd), _mm_cvtpd_ps(dstIm_highd));
            _mm_store_ps(dstRe + i, dstRe_tmp);  // ac + bd
            _mm_store_ps(dstIm + i, dstIm_tmp);  // bc - ad
        }
    } else {
        for (i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src1Re_tmp = _mm_loadu_ps((float *) (src1Re) + i);
            v4sf src1Im_tmp = _mm_loadu_ps((float *) (src1Im) + i);
            v4sf src2Re_tmp = _mm_loadu_ps((float *) (src2Re) + i);
            v4sf src2Im_tmp = _mm_loadu_ps((float *) (src2Im) + i);

            v2sd src1Re_lowd = _mm_cvtps_pd(src1Re_tmp);
            v2sd src1Im_lowd = _mm_cvtps_pd(src1Im_tmp);
            v2sd src2Re_lowd = _mm_cvtps_pd(src2Re_tmp);
            v2sd src2Im_lowd = _mm_cvtps_pd(src2Im_tmp);

            v2sd src1Re_highd = _mm_cvtps_pd(_mm_shuffle_ps(src1Re_tmp, src1Re_tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src1Im_highd = _mm_cvtps_pd(_mm_shuffle_ps(src1Im_tmp, src1Im_tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2Re_highd = _mm_cvtps_pd(_mm_shuffle_ps(src2Re_tmp, src2Re_tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            v2sd src2Im_highd = _mm_cvtps_pd(_mm_shuffle_ps(src2Im_tmp, src2Im_tmp, _MM_SHUFFLE(1, 0, 3, 2)));

            v2sd ac_lowd = _mm_mul_pd(src1Re_lowd, src2Re_lowd);
            v2sd bc_lowd = _mm_mul_pd(src1Im_lowd, src2Re_lowd);
            v2sd ac_highd = _mm_mul_pd(src1Re_highd, src2Re_highd);
            v2sd bc_highd = _mm_mul_pd(src1Im_highd, src2Re_highd);

            v2sd dstRe_lowd = _mm_fmadd_pd_custom(src1Im_lowd, src2Im_lowd, ac_lowd);
            v2sd dstIm_lowd = _mm_fnmadd_pd_custom(src1Re_lowd, src2Im_lowd, bc_lowd);
            v2sd dstRe_highd = _mm_fmadd_pd_custom(src1Im_highd, src2Im_highd, ac_highd);
            v2sd dstIm_highd = _mm_fnmadd_pd_custom(src1Re_highd, src2Im_highd, bc_highd);

            v4sf dstRe_tmp = _mm_movelh_ps(_mm_cvtpd_ps(dstRe_lowd), _mm_cvtpd_ps(dstRe_highd));
            v4sf dstIm_tmp = _mm_movelh_ps(_mm_cvtpd_ps(dstIm_lowd), _mm_cvtpd_ps(dstIm_highd));
            _mm_storeu_ps(dstRe + i, dstRe_tmp);  // ac + bd
            _mm_storeu_ps(dstIm + i, dstIm_tmp);  // bc - ad
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i]);
        dstIm[i] = (src2Re[i] * src1Im[i]) - src1Re[i] * src2Im[i];
    }
}

// prefer using cplxconjvecmulXf if you also need to do a multiply
static inline void cplxconj128f(complex32_t *src, complex32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len *= 2 * SSE_LEN_FLOAT;             // stop_len << 2;

    // const v4sf conj_mask = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    __attribute__((aligned(SSE_LEN_BYTES))) int32_t conj_mask[SSE_LEN_FLOAT] = {(int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000};
    int i;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps((float *) (src) + i);
            v4sf src_tmp2 = _mm_load_ps((float *) (src) + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_xor_ps(src_tmp, *(v4sf *) &conj_mask);
            v4sf dst_tmp2 = _mm_xor_ps(src_tmp2, *(v4sf *) &conj_mask);
            _mm_store_ps((float *) (dst) + i, dst_tmp);
            _mm_store_ps((float *) (dst) + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps((float *) (src) + i);
            v4sf src_tmp2 = _mm_loadu_ps((float *) (src) + i + SSE_LEN_FLOAT);
            v4sf dst_tmp = _mm_xor_ps(src_tmp, *(v4sf *) &conj_mask);
            v4sf dst_tmp2 = _mm_xor_ps(src_tmp2, *(v4sf *) &conj_mask);
            _mm_storeu_ps((float *) (dst) + i, dst_tmp);
            _mm_storeu_ps((float *) (dst) + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src[i].re;
        dst[i].im = -src[i].im;
    }
}

static inline void sigmoid128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf tmp = _mm_add_ps(*(v4sf *) _ps_1, exp_ps_alternate(_mm_xor_ps(*(v4sf *) _ps_neg_sign_mask, src_tmp)));
            _mm_store_ps(dst + i, _mm_div_ps(*(v4sf *) _ps_1, tmp));  //)_mm_rcp_ps(tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf tmp = _mm_add_ps(*(v4sf *) _ps_1, exp_ps_alternate(_mm_xor_ps(*(v4sf *) _ps_neg_sign_mask, src_tmp)));
            _mm_storeu_ps(dst + i, _mm_div_ps(*(v4sf *) _ps_1, tmp));  //)_mm_rcp_ps(tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}

// Alternate sigmoid version with tanh => slower?
static inline void sigmoid128f_(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_0p5, tanhf_ps(_mm_mul_ps(*(v4sf *) _ps_0p5, src_tmp)), *(v4sf *) _ps_0p5);
            _mm_store_ps(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf tmp = _mm_fmadd_ps_custom(*(v4sf *) _ps_0p5, tanhf_ps(_mm_mul_ps(*(v4sf *) _ps_0p5, src_tmp)), *(v4sf *) _ps_0p5);
            _mm_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}


static inline void PRelu128f(float *src, float *dst, float alpha, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    v4sf alpha_vec = _mm_set1_ps(alpha);
    v4sf zero = _mm_setzero_ps();

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf tmp = _mm_mul_ps(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)

            // if x > 0
            _mm_store_ps(dst + i, _mm_blendv_ps(tmp, src_tmp, _mm_cmpgt_ps(src_tmp, zero)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf tmp = _mm_mul_ps(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)

            // if x > 0
            _mm_storeu_ps(dst + i, _mm_blendv_ps(tmp, src_tmp, _mm_cmpgt_ps(src_tmp, zero)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] > 0.0f)
            dst[i] = src[i];
        else
            dst[i] = alpha * src[i];
    }
}

// to be improved
static inline void softmax128f(float *src, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= (SSE_LEN_FLOAT);

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc = 0.0f;

    v4sf vec_acc1 = _mm_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf dst_tmp = exp_ps_alternate(src_tmp);
            vec_acc1 = _mm_add_ps(vec_acc1, dst_tmp);
            _mm_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf dst_tmp = exp_ps_alternate(src_tmp);
            vec_acc1 = _mm_add_ps(vec_acc1, dst_tmp);
            _mm_storeu_ps(dst + i, dst_tmp);
        }
    }

    _mm_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
        acc += dst[i];
    }

    acc = acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];
    vec_acc1 = _mm_set1_ps(acc);

    if (isAligned((uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf dst_tmp = _mm_load_ps(dst + i);
            _mm_store_ps(dst + i, _mm_div_ps(dst_tmp, vec_acc1));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf dst_tmp = _mm_loadu_ps(dst + i);
            _mm_storeu_ps(dst + i, _mm_div_ps(dst_tmp, vec_acc1));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] /= acc;
    }
}

// to be improved
static inline void softmax128f_dualacc(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc = 0.0f;

    v4sf vec_acc1 = _mm_setzero_ps();  // initialize the vector accumulator
    v4sf vec_acc2 = _mm_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp1 = exp_ps_alternate(src_tmp1);
            v4sf dst_tmp2 = exp_ps_alternate(src_tmp2);
            vec_acc1 = _mm_add_ps(vec_acc1, dst_tmp1);
            vec_acc2 = _mm_add_ps(vec_acc2, dst_tmp2);
            _mm_store_ps(dst + i, dst_tmp1);
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp1 = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf dst_tmp1 = exp_ps_alternate(src_tmp1);
            v4sf dst_tmp2 = exp_ps_alternate(src_tmp2);
            vec_acc1 = _mm_add_ps(vec_acc1, dst_tmp1);
            vec_acc2 = _mm_add_ps(vec_acc2, dst_tmp2);
            _mm_storeu_ps(dst + i, dst_tmp1);
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, dst_tmp2);
        }
    }

    vec_acc1 = _mm_add_ps(vec_acc1, vec_acc2);
    _mm_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
        acc += dst[i];
    }

    acc = acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];
    vec_acc1 = _mm_set1_ps(acc);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf dst_tmp1 = _mm_load_ps(dst + i);
            v4sf dst_tmp2 = _mm_load_ps(dst + i + SSE_LEN_FLOAT);
            _mm_store_ps(dst + i, _mm_div_ps(dst_tmp1, vec_acc1));
            _mm_store_ps(dst + i + SSE_LEN_FLOAT, _mm_div_ps(dst_tmp2, vec_acc1));
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf dst_tmp1 = _mm_loadu_ps(dst + i);
            v4sf dst_tmp2 = _mm_loadu_ps(dst + i + SSE_LEN_FLOAT);
            _mm_storeu_ps(dst + i, _mm_div_ps(dst_tmp1, vec_acc1));
            _mm_storeu_ps(dst + i + SSE_LEN_FLOAT, _mm_div_ps(dst_tmp2, vec_acc1));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] /= acc;
    }
}

static inline void pol2cart2D128f(float *r, float *theta, float *x, float *y, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf r_tmp = _mm_load_ps(r + i);
            v4sf theta_tmp = _mm_load_ps(theta + i);
            v4sf sin_tmp;
            v4sf cos_tmp;
            sincos_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v4sf x_tmp = _mm_mul_ps(r_tmp, cos_tmp);
            v4sf y_tmp = _mm_mul_ps(r_tmp, sin_tmp);
            _mm_store_ps(x + i, x_tmp);
            _mm_store_ps(y + i, y_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf r_tmp = _mm_loadu_ps(r + i);
            v4sf theta_tmp = _mm_loadu_ps(theta + i);
            v4sf sin_tmp;
            v4sf cos_tmp;
            sincos_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v4sf x_tmp = _mm_mul_ps(r_tmp, cos_tmp);
            v4sf y_tmp = _mm_mul_ps(r_tmp, sin_tmp);
            _mm_storeu_ps(x + i, x_tmp);
            _mm_storeu_ps(y + i, y_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        float sin_tmp, cos_tmp;
        mysincosf(theta[i], &sin_tmp, &cos_tmp);
        x[i] = r[i] * cos_tmp;
        y[i] = r[i] * sin_tmp;
    }
}

static inline void cart2pol2D128f(float *x, float *y, float *r, float *theta, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), SSE_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x_tmp = _mm_load_ps(x + i);
            v4sf y_tmp = _mm_load_ps(y + i);
            v4sf y_square = _mm_mul_ps(y_tmp, y_tmp);
            v4sf r_tmp = _mm_fmadd_ps_custom(x_tmp, x_tmp, y_square);
            r_tmp = _mm_sqrt_ps(r_tmp);
            v4sf theta_tmp = atan2f_ps(y_tmp, x_tmp);
            _mm_store_ps(r + i, r_tmp);
            _mm_store_ps(theta + i, theta_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x_tmp = _mm_loadu_ps(x + i);
            v4sf y_tmp = _mm_loadu_ps(y + i);
            v4sf y_square = _mm_mul_ps(y_tmp, y_tmp);
            v4sf r_tmp = _mm_fmadd_ps_custom(x_tmp, x_tmp, y_square);
            r_tmp = _mm_sqrt_ps(r_tmp);
            v4sf theta_tmp = atan2f_ps(y_tmp, x_tmp);
            _mm_storeu_ps(r + i, r_tmp);
            _mm_storeu_ps(theta + i, theta_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        r[i] = sqrtf(x[i] * x[i] + (y[i] * y[i]));
        theta[i] = atan2f(y[i], x[i]);
    }
}

static inline void modf128f(float *src, float *integer, float *remainder, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src), (uintptr_t) (integer), (uintptr_t) (remainder), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sf integer_tmp = _mm_round_ps(src_tmp, ROUNDTOZERO);
            v4sf integer_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOZERO);
            v4sf remainder_tmp = _mm_sub_ps(src_tmp, integer_tmp);
            v4sf remainder_tmp2 = _mm_sub_ps(src_tmp2, integer_tmp2);
            _mm_store_ps(integer + i, integer_tmp);
            _mm_store_ps(integer + i + SSE_LEN_FLOAT, integer_tmp2);
            _mm_store_ps(remainder + i, remainder_tmp);
            _mm_store_ps(remainder + i + SSE_LEN_FLOAT, remainder_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sf integer_tmp = _mm_round_ps(src_tmp, ROUNDTOZERO);
            v4sf integer_tmp2 = _mm_round_ps(src_tmp2, ROUNDTOZERO);
            v4sf remainder_tmp = _mm_sub_ps(src_tmp, integer_tmp);
            v4sf remainder_tmp2 = _mm_sub_ps(src_tmp2, integer_tmp2);
            _mm_storeu_ps(integer + i, integer_tmp);
            _mm_storeu_ps(integer + i + SSE_LEN_FLOAT, integer_tmp2);
            _mm_storeu_ps(remainder + i, remainder_tmp);
            _mm_storeu_ps(remainder + i + SSE_LEN_FLOAT, remainder_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        remainder[i] = modff(src[i], &(integer[i]));
    }
}
