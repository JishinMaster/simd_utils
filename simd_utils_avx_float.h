/*
 * Project : SIMD_Utils
 * Version : 0.2.5
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

static inline v8sf log10256_ps(v8sf x)
{
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN log10256_ps" ::
                       : "memory");
#endif
    v8si imm0;
    v8sf one = *(v8sf *) _ps256_1;

    v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, *(v8sf *) _ps256_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(v8sf *) _ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, *(v8si *) _pi32_256_0x7f);
    v8sf e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    v8sf mask = _mm256_cmp_ps(x, *(v8sf *) _ps256_cephes_SQRTHF, _CMP_LT_OS);
    v8sf tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    v8sf z = _mm256_mul_ps(x, x);

    v8sf y = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_cephes_log_p0, x, *(v8sf *) _ps256_cephes_log_p1);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p2);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p3);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p4);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p5);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p6);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p7);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);

    y = _mm256_fnmadd_ps_custom(z, *(v8sf *) _ps256_0p5, y);

    // Could it be improved with more parallelism or would it worsen precision?
    tmp = _mm256_add_ps(x, y);
    z = _mm256_mul_ps(tmp, *(v8sf *) _ps256_cephes_L10EB);
    z = _mm256_fmadd_ps_custom(y, *(v8sf *) _ps256_cephes_L10EA, z);
    z = _mm256_fmadd_ps_custom(x, *(v8sf *) _ps256_cephes_L10EA, z);
    z = _mm256_fmadd_ps_custom(e, *(v8sf *) _ps256_cephes_L102B, z);
    x = _mm256_fmadd_ps_custom(e, *(v8sf *) _ps256_cephes_L102A, z);

    x = _mm256_or_ps(x, invalid_mask);  // negative arg will be NAN
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END log10256_ps" ::
                       : "memory");
#endif
    return x;
}

static inline v8sf log2256_ps(v8sf x)
{
    v8si imm0;
    v8sf one = *(v8sf *) _ps256_1;

    v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, *(v8sf *) _ps256_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(v8sf *) _ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, *(v8si *) _pi32_256_0x7f);
    v8sf e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    v8sf mask = _mm256_cmp_ps(x, *(v8sf *) _ps256_cephes_SQRTHF, _CMP_LT_OS);
    v8sf tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    v8sf z = _mm256_mul_ps(x, x);

    v8sf y = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_cephes_log_p0, x, *(v8sf *) _ps256_cephes_log_p1);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p2);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p3);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p4);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p5);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p6);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p7);
    y = _mm256_fmadd_ps_custom(y, x, *(v8sf *) _ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);

    y = _mm256_fnmadd_ps_custom(z, *(v8sf *) _ps256_0p5, y);

    // Could it be improved with more parallelism or would it worsen precision?
    tmp = _mm256_add_ps(y, x);
    z = _mm256_mul_ps(y, *(v8sf *) _ps256_cephes_LOG2EA);
    z = _mm256_fmadd_ps_custom(x, *(v8sf *) _ps256_cephes_LOG2EA, z);
    z = _mm256_add_ps(z, tmp);
    x = _mm256_add_ps(z, e);

    x = _mm256_or_ps(x, invalid_mask);  // negative arg will be NAN
    return x;
}


static inline void log10_256f(float *src, float *dst, int len)
{
    const v8sf invln10f = _mm256_set1_ps((float) INVLN10);  //_mm256_broadcast_ss(&invln10f_mask);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

static inline void log10_256f_precise(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log10256_ps(_mm256_load_ps(src + i));
            _mm256_store_ps(dst + i, src_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log10256_ps(_mm256_loadu_ps(src + i));
            _mm256_storeu_ps(dst + i, src_tmp);
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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
        dst[i] = log2f(src[i]);
    }
}

static inline void log2_256f_precise(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log2256_ps(_mm256_load_ps(src + i));
            _mm256_store_ps(dst + i, src_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = log2256_ps(_mm256_loadu_ps(src + i));
            _mm256_storeu_ps(dst + i, src_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline void ln_256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

// rewritten alternate version which properly returns MAXNUMF or 0.0 outside of boundaries
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

#ifdef __AVX2__
// from https://stackoverflow.com/questions/57454416/AVX-integer-2n-powers-of-2-for-32-bit-integers-without-avx2
static inline v8sf power_of_two256f(v8si b)
{
    /*#ifndef __AVX2__
        v8si exp = _mm256_add_epi32(b, _mm256_set1_epi32(127));
        v8sf f = _mm256_castsi256_ps(_mm256_slli_epi32(exp, 23));
        return f;
    #else*/
    return _mm256_cvtepi32_ps(_mm256_sllv_epi32(*(v8si *) _pi32_256_1, b));
    //#endif
}

static inline v8sf cbrt256f_ps(v8sf xx)
{
    v8sf e, rem, sign;
    v8sf x, z;

    x = xx;
    // sign = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OS);
    sign = _mm256_and_ps(xx, *(v8sf *) _ps256_sign_mask);
    x = _mm256_and_ps(x, *(v8sf *) _ps256_pos_sign_mask);

    z = x;
    /* extract power of 2, leaving
     * mantissa between 0.5 and 1
     */
    // x = frexpf(x, &e);
    // solve problem for zero
    v8si emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(v8sf *) _ps256_0p5);
    emm0 = _mm256_sub_epi32(emm0, *(v8si *) _pi32_256_0x7e);
    e = _mm256_cvtepi32_ps(emm0);

    /* Approximate cube root of number between .5 and 1,
     * peak relative error = 9.2e-6
     */
    v8sf tmp;
    tmp = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_CBRTF_P0, x, *(v8sf *) _ps256_CBRTF_P1);
    tmp = _mm256_fmadd_ps_custom(x, tmp, *(v8sf *) _ps256_CBRTF_P2);
    tmp = _mm256_fmadd_ps_custom(x, tmp, *(v8sf *) _ps256_CBRTF_P3);
    x = _mm256_fmadd_ps_custom(x, tmp, *(v8sf *) _ps256_CBRTF_P4);

    /* exponent divided by 3 */
    v8sf e_sign = _mm256_cmp_ps(e, _mm256_setzero_ps(), _CMP_GE_OS);
    e = _mm256_and_ps(e, *(v8sf *) _ps256_pos_sign_mask);

    rem = e;
    e = _mm256_mul_ps(e, *(v8sf *) _ps256_0p3);
    v8sf e_tmp = _mm256_mul_ps(*(v8sf *) _ps256_3, _mm256_round_ps(e, ROUNDTOZERO));
    rem = _mm256_sub_ps(rem, e_tmp);

    v8sf mul1, mul2;
    v8sf mul_cst1 = _mm256_blendv_ps(*(v8sf *) _ps256_cephes_invCBRT2, *(v8sf *) _ps256_cephes_CBRT2, e_sign);
    v8sf mul_cst2 = _mm256_blendv_ps(*(v8sf *) _ps256_cephes_invCBRT4, *(v8sf *) _ps256_cephes_CBRT4, e_sign);
    mul1 = _mm256_mul_ps(x, mul_cst1);
    mul2 = _mm256_mul_ps(x, mul_cst2);

    v8si remi = _mm256_cvtps_epi32(rem);  // rem integer
    v8si rem1 = _mm256_cmpeq_epi32(remi, _mm256_set1_epi32(1));
    v8si rem2 = _mm256_cmpeq_epi32(remi, _mm256_set1_epi32(2));

    x = _mm256_blendv_ps(x, mul1, _mm256_castsi256_ps(rem1));  // rem==1
    x = _mm256_blendv_ps(x, mul2, _mm256_castsi256_ps(rem2));  // rem==2

    /* multiply by power of 2 */
    //  x = ldexpf(x, e);
    // x= x* (1 >> e)
    v8sf cst = power_of_two256f(_mm256_cvtps_epi32(e));
    // blend sign of e
    x = _mm256_blendv_ps(_mm256_div_ps(x, cst), _mm256_mul_ps(x, cst), e_sign);

    /* Newton iteration */
    // x -= (x - (z / (x * x))) * 0.333333333333;
    v8sf tmp2 = _mm256_mul_ps(x, x);
    tmp2 = _mm256_div_ps(z, tmp2);
    tmp2 = _mm256_sub_ps(x, tmp2);
    tmp2 = _mm256_mul_ps(tmp2, *(v8sf *) _ps256_0p3);
    x = _mm256_sub_ps(x, tmp2);

    // x = _mm256_blendv_ps(_mm256_mul_ps(x, *(v8sf *) _ps256_min1), x, sign);
    x = _mm256_xor_ps(x, sign);
    return x;
}

static inline void cbrt256f(float *src, float *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);
    stop_len *= (AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf x = _mm256_load_ps(src + i);
            v8sf dst_tmp = cbrt256f_ps(x);
            _mm256_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf x = _mm256_loadu_ps(src + i);
            v8sf dst_tmp = cbrt256f_ps(x);
            _mm256_storeu_ps(dst + i, dst_tmp);
        }
    }
    for (int i = stop_len; i < len; i++) {
        dst[i] = cbrtf(src[i]);
    }
}
#endif

static inline void fabs256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf fabs1 = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, src_tmp);
            v8sf fabs2 = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, src_tmp2);
            _mm256_store_ps(dst + i, fabs1);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, fabs2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf fabs1 = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, src_tmp);
            v8sf fabs2 = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, src_tmp2);
            _mm256_storeu_ps(dst + i, fabs1);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, fabs2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void set256f(float *dst, float value, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t) (const void *) (dst) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zero256f(float *dst, int len)
{
    const v8sf tmp = _mm256_setzero_ps();

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t) (const void *) (dst) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 0.0f;
    }
}


static inline void copy256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            _mm256_store_ps(dst + i, src_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, src_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            _mm256_storeu_ps(dst + i, src_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, src_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(src1 + i);
            v8sf b = _mm256_load_ps(src2 + i);
            v8sf a2 = _mm256_load_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_load_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_add_ps(a, b);
            v8sf dst_tmp2 = _mm256_add_ps(a2, b2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(src1 + i);
            v8sf b = _mm256_loadu_ps(src2 + i);
            v8sf a2 = _mm256_loadu_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_loadu_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_add_ps(a, b);
            v8sf dst_tmp2 = _mm256_add_ps(a2, b2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}


static inline void mul256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(src1 + i);
            v8sf b = _mm256_load_ps(src2 + i);
            v8sf a2 = _mm256_load_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_load_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_mul_ps(a, b);
            v8sf dst_tmp2 = _mm256_mul_ps(a2, b2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(src1 + i);
            v8sf b = _mm256_loadu_ps(src2 + i);
            v8sf a2 = _mm256_loadu_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_loadu_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_mul_ps(a, b);
            v8sf dst_tmp2 = _mm256_mul_ps(a2, b2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(src1 + i);
            v8sf b = _mm256_load_ps(src2 + i);
            v8sf a2 = _mm256_load_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_load_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_sub_ps(a, b);
            v8sf dst_tmp2 = _mm256_sub_ps(a2, b2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(src1 + i);
            v8sf b = _mm256_loadu_ps(src2 + i);
            v8sf a2 = _mm256_loadu_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_loadu_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_sub_ps(a, b);
            v8sf dst_tmp2 = _mm256_sub_ps(a2, b2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}


static inline void addc256f(float *src, float value, float *dst, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(src + i);
            v8sf a2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_add_ps(a, tmp);
            v8sf dst_tmp2 = _mm256_add_ps(a2, tmp);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(src + i);
            v8sf a2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_add_ps(a, tmp);
            v8sf dst_tmp2 = _mm256_add_ps(a2, tmp);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc256f(float *src, float value, float *dst, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp1 = _mm256_mul_ps(tmp, src_tmp1);
            v8sf dst_tmp2 = _mm256_mul_ps(tmp, src_tmp2);
            _mm256_store_ps(dst + i, dst_tmp1);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp1 = _mm256_mul_ps(tmp, src_tmp1);
            v8sf dst_tmp2 = _mm256_mul_ps(tmp, src_tmp2);
            _mm256_storeu_ps(dst + i, dst_tmp1);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd256f(float *_a, float *_b, float *_c, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (_b), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (_c), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_load_ps(_a + i);
            v8sf b = _mm256_load_ps(_b + i);
            v8sf c = _mm256_load_ps(_c + i);
            v8sf a2 = _mm256_load_ps(_a + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_load_ps(_b + i + AVX_LEN_FLOAT);
            v8sf c2 = _mm256_load_ps(_c + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_fmadd_ps_custom(a, b, c);
            v8sf dst_tmp2 = _mm256_fmadd_ps_custom(a2, b2, c2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf a = _mm256_loadu_ps(_a + i);
            v8sf b = _mm256_loadu_ps(_b + i);
            v8sf c = _mm256_loadu_ps(_c + i);
            v8sf a2 = _mm256_loadu_ps(_a + i + AVX_LEN_FLOAT);
            v8sf b2 = _mm256_loadu_ps(_b + i + AVX_LEN_FLOAT);
            v8sf c2 = _mm256_loadu_ps(_c + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_fmadd_ps_custom(a, b, c);
            v8sf dst_tmp2 = _mm256_fmadd_ps_custom(a2, b2, c2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (_a[i] * _b[i]) + _c[i];
    }
}

static inline void mulcadd256f(float *_a, float _b, float *_c, float *dst, int len)
{
    v8sf b = _mm256_set1_ps(_b);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_c), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_b), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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
// TODO: intel targets can do 2 mul/FMA per cycle but only one add => replace add_ps(a,b) by fmadd_ps(1.0f,a,b)
static inline void vectorSlope256f(float *dst, int len, float offset, float slope)
{
    v8sf coef = _mm256_set_ps(7.0f * slope, 6.0f * slope, 5.0f * slope, 4.0f * slope, 3.0f * slope, 2.0f * slope, slope, 0.0f);
    v8sf slope16_vec = _mm256_set1_ps(16.0f * slope);
    v8sf curVal = _mm256_add_ps(_mm256_set1_ps(offset), coef);
    v8sf curVal2 = _mm256_add_ps(_mm256_set1_ps(offset), coef);
    curVal2 = _mm256_add_ps(curVal2, _mm256_set1_ps(8.0f * slope));
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (len >= AVX_LEN_FLOAT) {
        if (((uintptr_t) (const void *) (dst) % AVX_LEN_BYTES) == 0) {
            _mm256_store_ps(dst + 0, curVal);
            _mm256_store_ps(dst + AVX_LEN_FLOAT, curVal2);
        } else {
            _mm256_storeu_ps(dst + 0, curVal);
            _mm256_storeu_ps(dst + AVX_LEN_FLOAT, curVal2);
        }

        if (((uintptr_t) (const void *) (dst) % AVX_LEN_BYTES) == 0) {
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
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

#ifdef __AVX2__

#if 0
static inline void convertInt16ToFloat32_256(int16_t *src, float *dst, int len, int scale_factor)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v8sf scale_fact_vec = _mm256_set1_ps(scale_fact_mult);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8si vec = _mm256_load_si256((__m256i *) (src + i));                         // loads 1 2 3 4 5 6 7 8 8 9 10 11 12 13 14 15 16
            v8si low_unordered = _mm256_unpacklo_epi16(vec, vec);                        // low 1 1 2 2 3 3 4 4  9 9 10 10 11 11 12 12
            v8si high_unordered = _mm256_unpackhi_epi16(vec, vec);                       // high 5 5 6 6 7 7 8 8 13 13 14 14 15 15 16 16
            v8si low = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x20);   // low 1 1 2 2 3 3 4 45 5 6 6 7 7 8 8
            v8si high = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x31);  // high 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16
            low = _mm256_srai_epi32(low, 0x10);                                          // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm256_srai_epi32(high, 0x10);                                        // make high 5 -1 6 -1 7 -1 8 -1

            // convert the vector to float and scale it
            v8sf floatlo = _mm256_mul_ps(_mm256_cvtepi32_ps(low), scale_fact_vec);
            v8sf floathi = _mm256_mul_ps(_mm256_cvtepi32_ps(high), scale_fact_vec);

            _mm256_store_ps(dst + i, floatlo);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, floathi);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8si vec = _mm256_loadu_si256((__m256i *) (src + i));                        // loads 1 2 3 4 5 6 7 8 8 9 10 11 12 13 14 15 16
            v8si low_unordered = _mm256_unpacklo_epi16(vec, vec);                        // low 1 1 2 2 3 3 4 4  9 9 10 10 11 11 12 12
            v8si high_unordered = _mm256_unpackhi_epi16(vec, vec);                       // high 5 5 6 6 7 7 8 8 13 13 14 14 15 15 16 16
            v8si low = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x20);   // low 1 1 2 2 3 3 4 45 5 6 6 7 7 8 8
            v8si high = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x31);  // high 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16
            low = _mm256_srai_epi32(low, 0x10);                                          // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm256_srai_epi32(high, 0x10);                                        // make high 5 -1 6 -1 7 -1 8 -1

            // convert the vector to float and scale it
            v8sf floatlo = _mm256_mul_ps(_mm256_cvtepi32_ps(low), scale_fact_vec);
            v8sf floathi = _mm256_mul_ps(_mm256_cvtepi32_ps(high), scale_fact_vec);

            _mm256_storeu_ps(dst + i, floatlo);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, floathi);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}
#else

static inline void convertInt16ToFloat32_256(int16_t *src, float *dst, int len, int scale_factor)
{
    int stop_len = len / (4 * AVX_LEN_FLOAT);
    stop_len *= (4 * AVX_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v8sf scale_fact_vec = _mm256_set1_ps(scale_fact_mult);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8si vec = _mm256_load_si256((__m256i *) (src + i));                         // loads 1 2 3 4 5 6 7 8 8 9 10 11 12 13 14 15 16
            v8si low_unordered = _mm256_unpacklo_epi16(vec, vec);                        // low 1 1 2 2 3 3 4 4  9 9 10 10 11 11 12 12
            v8si high_unordered = _mm256_unpackhi_epi16(vec, vec);                       // high 5 5 6 6 7 7 8 8 13 13 14 14 15 15 16 16
            v8si low = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x20);   // low 1 1 2 2 3 3 4 45 5 6 6 7 7 8 8
            v8si high = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x31);  // high 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16
            low = _mm256_srai_epi32(low, 0x10);                                          // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm256_srai_epi32(high, 0x10);                                        // make high 5 -1 6 -1 7 -1 8 -1
            v8sf floatlo = _mm256_cvtepi32_ps(low);                                      // convert the vector to float and scale it
            floatlo = _mm256_mul_ps(floatlo, scale_fact_vec);
            v8sf floathi = _mm256_cvtepi32_ps(high);
            floathi = _mm256_mul_ps(floathi, scale_fact_vec);
            _mm256_store_ps(dst + i, floatlo);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, floathi);

            v8si vec2 = _mm256_load_si256((__m256i *) (src + i + 2 * AVX_LEN_FLOAT));
            v8si low_unordered2 = _mm256_unpacklo_epi16(vec2, vec2);
            v8si high_unordered2 = _mm256_unpackhi_epi16(vec2, vec2);
            v8si low2 = _mm256_permute2f128_si256(low_unordered2, high_unordered2, 0x20);
            v8si high2 = _mm256_permute2f128_si256(low_unordered2, high_unordered2, 0x31);
            low2 = _mm256_srai_epi32(low2, 0x10);
            high2 = _mm256_srai_epi32(high2, 0x10);
            v8sf floatlo2 = _mm256_cvtepi32_ps(low2);
            floatlo2 = _mm256_mul_ps(floatlo2, scale_fact_vec);
            v8sf floathi2 = _mm256_cvtepi32_ps(high2);
            floathi2 = _mm256_mul_ps(floathi2, scale_fact_vec);
            _mm256_store_ps(dst + i + 2 * AVX_LEN_FLOAT, floatlo2);
            _mm256_store_ps(dst + i + 3 * AVX_LEN_FLOAT, floathi2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8si vec = _mm256_loadu_si256((__m256i *) (src + i));                        // loads 1 2 3 4 5 6 7 8 8 9 10 11 12 13 14 15 16
            v8si low_unordered = _mm256_unpacklo_epi16(vec, vec);                        // low 1 1 2 2 3 3 4 4  9 9 10 10 11 11 12 12
            v8si high_unordered = _mm256_unpackhi_epi16(vec, vec);                       // high 5 5 6 6 7 7 8 8 13 13 14 14 15 15 16 16
            v8si low = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x20);   // low 1 1 2 2 3 3 4 45 5 6 6 7 7 8 8
            v8si high = _mm256_permute2f128_si256(low_unordered, high_unordered, 0x31);  // high 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16
            low = _mm256_srai_epi32(low, 0x10);                                          // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm256_srai_epi32(high, 0x10);                                        // make high 5 -1 6 -1 7 -1 8 -1
            v8sf floatlo = _mm256_cvtepi32_ps(low);                                      // convert the vector to float and scale it
            floatlo = _mm256_mul_ps(floatlo, scale_fact_vec);
            v8sf floathi = _mm256_cvtepi32_ps(high);
            floathi = _mm256_mul_ps(floathi, scale_fact_vec);
            _mm256_storeu_ps(dst + i, floatlo);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, floathi);

            v8si vec2 = _mm256_loadu_si256((__m256i *) (src + i + 2 * AVX_LEN_FLOAT));
            v8si low_unordered2 = _mm256_unpacklo_epi16(vec2, vec2);
            v8si high_unordered2 = _mm256_unpackhi_epi16(vec2, vec2);
            v8si low2 = _mm256_permute2f128_si256(low_unordered2, high_unordered2, 0x20);
            v8si high2 = _mm256_permute2f128_si256(low_unordered2, high_unordered2, 0x31);
            low2 = _mm256_srai_epi32(low2, 0x10);
            high2 = _mm256_srai_epi32(high2, 0x10);
            v8sf floatlo2 = _mm256_cvtepi32_ps(low2);
            floatlo2 = _mm256_mul_ps(floatlo2, scale_fact_vec);
            v8sf floathi2 = _mm256_cvtepi32_ps(high2);
            floathi2 = _mm256_mul_ps(floathi2, scale_fact_vec);
            _mm256_storeu_ps(dst + i + 2 * AVX_LEN_FLOAT, floatlo2);
            _mm256_storeu_ps(dst + i + 3 * AVX_LEN_FLOAT, floathi2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}
#endif

static inline void convertFloat32ToU8_256(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * AVX_LEN_FLOAT);
    stop_len *= (4 * AVX_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v8sf scale_fact_vec = _mm256_set1_ps(scale_fact_mult);
    v8si idx = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_tmp3 = _mm256_load_ps(src + i + 2 * AVX_LEN_FLOAT);
            v8sf src_tmp4 = _mm256_load_ps(src + i + 3 * AVX_LEN_FLOAT);
            v8sf tmp1 = _mm256_mul_ps(src_tmp1, scale_fact_vec);
            v8sf tmp2 = _mm256_mul_ps(src_tmp2, scale_fact_vec);
            v8sf tmp3 = _mm256_mul_ps(src_tmp3, scale_fact_vec);
            v8sf tmp4 = _mm256_mul_ps(src_tmp4, scale_fact_vec);
            v8si tmp1_int = _mm256_cvtps_epi32(tmp1);
            v8si tmp2_int = _mm256_cvtps_epi32(tmp2);
            v8si tmp3_int = _mm256_cvtps_epi32(tmp3);
            v8si tmp4_int = _mm256_cvtps_epi32(tmp4);
            v8si tmp5 = _mm256_packs_epi32(tmp1_int, tmp2_int);
            v8si tmp6 = _mm256_packs_epi32(tmp3_int, tmp4_int);
            v8si tmp7 = _mm256_packus_epi16(tmp5, tmp6);
            tmp7 = _mm256_permutevar8x32_epi32(tmp7, idx);
            _mm256_store_si256((__m256i *) (dst + i), tmp7);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_tmp3 = _mm256_loadu_ps(src + i + 2 * AVX_LEN_FLOAT);
            v8sf src_tmp4 = _mm256_loadu_ps(src + i + 3 * AVX_LEN_FLOAT);
            v8sf tmp1 = _mm256_mul_ps(src_tmp1, scale_fact_vec);
            v8sf tmp2 = _mm256_mul_ps(src_tmp2, scale_fact_vec);
            v8sf tmp3 = _mm256_mul_ps(src_tmp3, scale_fact_vec);
            v8sf tmp4 = _mm256_mul_ps(src_tmp4, scale_fact_vec);
            v8si tmp1_int = _mm256_cvtps_epi32(tmp1);
            v8si tmp2_int = _mm256_cvtps_epi32(tmp2);
            v8si tmp3_int = _mm256_cvtps_epi32(tmp3);
            v8si tmp4_int = _mm256_cvtps_epi32(tmp4);
            v8si tmp5 = _mm256_packs_epi32(tmp1_int, tmp2_int);
            v8si tmp6 = _mm256_packs_epi32(tmp3_int, tmp4_int);
            v8si tmp7 = _mm256_packus_epi16(tmp5, tmp6);
            tmp7 = _mm256_permutevar8x32_epi32(tmp7, idx);
            _mm256_storeu_si256((__m256i *) (dst + i), tmp7);
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

// Thanks to :
//  https://coderedirect.com/questions/559388/how-can-i-convert-a-vector-of-float-to-short-int-using-avx-instructions
static inline void convertFloat32ToI16_256(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v8sf scale_fact_vec = _mm256_set1_ps(scale_fact_mult);

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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf tmp1 = _mm256_mul_ps(src_tmp1, scale_fact_vec);
            v8sf tmp2 = _mm256_mul_ps(src_tmp2, scale_fact_vec);
            v8si tmp1_int = _mm256_cvtps_epi32(tmp1);
            v8si tmp2_int = _mm256_cvtps_epi32(tmp2);
            v8si tmp5 = _mm256_packs_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm256_permute4x64_epi64(tmp5, 0xD8);
            _mm256_store_si256((__m256i *) (dst + i), tmp5);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf tmp1 = _mm256_mul_ps(src_tmp1, scale_fact_vec);
            v8sf tmp2 = _mm256_mul_ps(src_tmp2, scale_fact_vec);
            v8si tmp1_int = _mm256_cvtps_epi32(tmp1);
            v8si tmp2_int = _mm256_cvtps_epi32(tmp2);
            v8si tmp5 = _mm256_packs_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm256_permute4x64_epi64(tmp5, 0xD8);
            _mm256_storeu_si256((__m256i *) (dst + i), tmp5);
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

static inline void convertFloat32ToU16_256(float *src, uint16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v8sf scale_fact_vec = _mm256_set1_ps(scale_fact_mult);

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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf tmp1 = _mm256_mul_ps(src_tmp1, scale_fact_vec);
            v8sf tmp2 = _mm256_mul_ps(src_tmp2, scale_fact_vec);
            v8si tmp1_int = _mm256_cvtps_epi32(tmp1);
            v8si tmp2_int = _mm256_cvtps_epi32(tmp2);
            v8si tmp5 = _mm256_packus_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm256_permute4x64_epi64(tmp5, 0xD8);
            _mm256_store_si256((__m256i *) (dst + i), tmp5);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp1 = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf tmp1 = _mm256_mul_ps(src_tmp1, scale_fact_vec);
            v8sf tmp2 = _mm256_mul_ps(src_tmp2, scale_fact_vec);
            v8si tmp1_int = _mm256_cvtps_epi32(tmp1);
            v8si tmp2_int = _mm256_cvtps_epi32(tmp2);
            v8si tmp5 = _mm256_packus_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm256_permute4x64_epi64(tmp5, 0xD8);
            _mm256_storeu_si256((__m256i *) (dst + i), tmp5);
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
#endif

// converts 32bits complex float to two arrays real and im
// Work in progress => could be improved with custom SSE mm_load2_ps
static inline void cplxtoreal256f(complex32_t *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (4 * AVX_LEN_FLOAT);
    stop_len *= 4 * AVX_LEN_FLOAT;

    int j = 0;

#ifdef __AVX2__
    v8si idx = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
#endif

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX_LEN_FLOAT)) {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sf vec1 = _mm256_load_ps((float const *) (src) + i);                  // load 0 1 2 3 4 5 6 7
            v8sf vec2 = _mm256_load_ps((float const *) (src) + i + AVX_LEN_FLOAT);  // load 8 9 10 11 12 13 14 15
            v8sf vec3 = _mm256_load_ps((float const *) (src) + i + 2 * AVX_LEN_FLOAT);
            v8sf vec4 = _mm256_load_ps((float const *) (src) + i + 3 * AVX_LEN_FLOAT);
#ifdef __AVX2__
            v8sf vec1_permute = _mm256_permutevar8x32_ps(vec1, idx);
            v8sf vec2_permute = _mm256_permutevar8x32_ps(vec2, idx);
            v8sf vec3_permute = _mm256_permutevar8x32_ps(vec3, idx);
            v8sf vec4_permute = _mm256_permutevar8x32_ps(vec4, idx);
            v8sf tmp1permute = _mm256_permute2f128_ps(vec1_permute, vec2_permute, 0x20);
            v8sf tmp2permute = _mm256_permute2f128_ps(vec1_permute, vec2_permute, 0x31);
            v8sf tmp3permute = _mm256_permute2f128_ps(vec3_permute, vec4_permute, 0x20);
            v8sf tmp4permute = _mm256_permute2f128_ps(vec3_permute, vec4_permute, 0x31);
#else
            v8sf vec1_permute = _mm256_permute2f128_ps(vec1, vec1, IMM8_PERMUTE_128BITS_LANES);        // reverse v1 4 5 6 7 0 1 2 3
            v8sf vec2_permute = _mm256_permute2f128_ps(vec2, vec1, IMM8_PERMUTE_128BITS_LANES);        // reverse v2 12 13 14 15 8 9 10 11
            v8sf vec1_even = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(2, 0, 2, 0));           // 0.2.5 6 0.2.5 6
            v8sf vec1_odd = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(3, 1, 3, 1));            // 1 3 5 7 1 3 5 7
            v8sf vec2_even = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(2, 0, 2, 0));           // 8 10 12 14
            v8sf vec2_odd = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(3, 1, 3, 1));            // 9 11 13 15
            v8sf tmp1permute = _mm256_insertf128_ps(vec1_even, _mm256_castps256_ps128(vec2_even), 1);  // 0.2.5 6 8 10 12 14
            v8sf tmp2permute = _mm256_insertf128_ps(vec1_odd, _mm256_castps256_ps128(vec2_odd), 1);    // 1 3 5 7 9 11 13 15
            v8sf vec3_permute = _mm256_permute2f128_ps(vec3, vec3, IMM8_PERMUTE_128BITS_LANES);
            v8sf vec4_permute = _mm256_permute2f128_ps(vec4, vec3, IMM8_PERMUTE_128BITS_LANES);
            v8sf vec3_even = _mm256_shuffle_ps(vec3, vec3_permute, _MM_SHUFFLE(2, 0, 2, 0));
            v8sf vec3_odd = _mm256_shuffle_ps(vec3, vec3_permute, _MM_SHUFFLE(3, 1, 3, 1));
            v8sf vec4_even = _mm256_shuffle_ps(vec4, vec4_permute, _MM_SHUFFLE(2, 0, 2, 0));
            v8sf vec4_odd = _mm256_shuffle_ps(vec4, vec4_permute, _MM_SHUFFLE(3, 1, 3, 1));
            v8sf tmp3permute = _mm256_insertf128_ps(vec3_even, _mm256_castps256_ps128(vec4_even), 1);
            v8sf tmp4permute = _mm256_insertf128_ps(vec3_odd, _mm256_castps256_ps128(vec4_odd), 1);
#endif
            _mm256_store_ps(dstRe + j, tmp1permute);
            _mm256_store_ps(dstIm + j, tmp2permute);
            _mm256_store_ps(dstRe + j + AVX_LEN_FLOAT, tmp3permute);
            _mm256_store_ps(dstIm + j + AVX_LEN_FLOAT, tmp4permute);

            j += 2 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sf vec1 = _mm256_loadu_ps((float const *) (src) + i);                  // load 0 1 2 3 4 5 6 7
            v8sf vec2 = _mm256_loadu_ps((float const *) (src) + i + AVX_LEN_FLOAT);  // load 8 9 10 11 12 13 14 15
            v8sf vec3 = _mm256_loadu_ps((float const *) (src) + i + 2 * AVX_LEN_FLOAT);
            v8sf vec4 = _mm256_loadu_ps((float const *) (src) + i + 3 * AVX_LEN_FLOAT);
#ifdef __AVX2__
            v8sf vec1_permute = _mm256_permutevar8x32_ps(vec1, idx);
            v8sf vec2_permute = _mm256_permutevar8x32_ps(vec2, idx);
            v8sf vec3_permute = _mm256_permutevar8x32_ps(vec3, idx);
            v8sf vec4_permute = _mm256_permutevar8x32_ps(vec4, idx);
            v8sf tmp1permute = _mm256_permute2f128_ps(vec1_permute, vec2_permute, 0x20);
            v8sf tmp2permute = _mm256_permute2f128_ps(vec1_permute, vec2_permute, 0x31);
            v8sf tmp3permute = _mm256_permute2f128_ps(vec3_permute, vec4_permute, 0x20);
            v8sf tmp4permute = _mm256_permute2f128_ps(vec3_permute, vec4_permute, 0x31);
#else
            v8sf vec1_permute = _mm256_permute2f128_ps(vec1, vec1, IMM8_PERMUTE_128BITS_LANES);        // reverse v1 4 5 6 7 0 1 2 3
            v8sf vec2_permute = _mm256_permute2f128_ps(vec2, vec1, IMM8_PERMUTE_128BITS_LANES);        // reverse v2 12 13 14 15 8 9 10 11
            v8sf vec1_even = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(2, 0, 2, 0));           // 0.2.5 6 0.2.5 6
            v8sf vec1_odd = _mm256_shuffle_ps(vec1, vec1_permute, _MM_SHUFFLE(3, 1, 3, 1));            // 1 3 5 7 1 3 5 7
            v8sf vec2_even = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(2, 0, 2, 0));           // 8 10 12 14
            v8sf vec2_odd = _mm256_shuffle_ps(vec2, vec2_permute, _MM_SHUFFLE(3, 1, 3, 1));            // 9 11 13 15
            v8sf tmp1permute = _mm256_insertf128_ps(vec1_even, _mm256_castps256_ps128(vec2_even), 1);  // 0.2.5 6 8 10 12 14
            v8sf tmp2permute = _mm256_insertf128_ps(vec1_odd, _mm256_castps256_ps128(vec2_odd), 1);    // 1 3 5 7 9 11 13 15
            v8sf vec3_permute = _mm256_permute2f128_ps(vec3, vec3, IMM8_PERMUTE_128BITS_LANES);
            v8sf vec4_permute = _mm256_permute2f128_ps(vec4, vec3, IMM8_PERMUTE_128BITS_LANES);
            v8sf vec3_even = _mm256_shuffle_ps(vec3, vec3_permute, _MM_SHUFFLE(2, 0, 2, 0));
            v8sf vec3_odd = _mm256_shuffle_ps(vec3, vec3_permute, _MM_SHUFFLE(3, 1, 3, 1));
            v8sf vec4_even = _mm256_shuffle_ps(vec4, vec4_permute, _MM_SHUFFLE(2, 0, 2, 0));
            v8sf vec4_odd = _mm256_shuffle_ps(vec4, vec4_permute, _MM_SHUFFLE(3, 1, 3, 1));
            v8sf tmp3permute = _mm256_insertf128_ps(vec3_even, _mm256_castps256_ps128(vec4_even), 1);
            v8sf tmp4permute = _mm256_insertf128_ps(vec3_odd, _mm256_castps256_ps128(vec4_odd), 1);
#endif
            _mm256_storeu_ps(dstRe + j, tmp1permute);
            _mm256_storeu_ps(dstIm + j, tmp2permute);
            _mm256_storeu_ps(dstRe + j + AVX_LEN_FLOAT, tmp3permute);
            _mm256_storeu_ps(dstIm + j + AVX_LEN_FLOAT, tmp4permute);

            j += 2 * AVX_LEN_FLOAT;
        }
    }

    for (int i = j; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
}

static inline void realtocplx256f(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= 2 * AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf re = _mm256_load_ps(srcRe + i);
            v8sf im = _mm256_load_ps(srcIm + i);
            v8sf re2 = _mm256_load_ps(srcRe + i + AVX_LEN_FLOAT);
            v8sf im2 = _mm256_load_ps(srcIm + i + AVX_LEN_FLOAT);
            v8sf cplx0 = _mm256_unpacklo_ps(re, im);
            v8sf cplx1 = _mm256_unpackhi_ps(re, im);
            v8sf cplx02 = _mm256_unpacklo_ps(re2, im2);
            v8sf cplx12 = _mm256_unpackhi_ps(re2, im2);
            v8sf perm0 = _mm256_permute2f128_ps(cplx0, cplx1, 0x20);     // permute mask [cplx1(127:0],cplx0[127:0])
            v8sf perm1 = _mm256_permute2f128_ps(cplx0, cplx1, 0x31);     // permute mask [cplx1(255:128],cplx0[255:128])
            v8sf perm02 = _mm256_permute2f128_ps(cplx02, cplx12, 0x20);  // permute mask [cplx1(127:0],cplx0[127:0])
            v8sf perm12 = _mm256_permute2f128_ps(cplx02, cplx12, 0x31);  // permute mask [cplx1(255:128],cplx0[255:128])
            _mm256_store_ps((float *) (dst) + j, perm0);
            _mm256_store_ps((float *) (dst) + j + AVX_LEN_FLOAT, perm1);
            _mm256_store_ps((float *) (dst) + j + 2 * AVX_LEN_FLOAT, perm02);
            _mm256_store_ps((float *) (dst) + j + 3 * AVX_LEN_FLOAT, perm12);
            j += 4 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf re = _mm256_loadu_ps(srcRe + i);
            v8sf im = _mm256_loadu_ps(srcIm + i);
            v8sf re2 = _mm256_loadu_ps(srcRe + i + AVX_LEN_FLOAT);
            v8sf im2 = _mm256_loadu_ps(srcIm + i + AVX_LEN_FLOAT);
            v8sf cplx0 = _mm256_unpacklo_ps(re, im);
            v8sf cplx1 = _mm256_unpackhi_ps(re, im);
            v8sf cplx02 = _mm256_unpacklo_ps(re2, im2);
            v8sf cplx12 = _mm256_unpackhi_ps(re2, im2);
            v8sf perm0 = _mm256_permute2f128_ps(cplx0, cplx1, 0x20);     // permute mask [cplx1(127:0],cplx0[127:0])
            v8sf perm1 = _mm256_permute2f128_ps(cplx0, cplx1, 0x31);     // permute mask [cplx1(255:128],cplx0[255:128])
            v8sf perm02 = _mm256_permute2f128_ps(cplx02, cplx12, 0x20);  // permute mask [cplx1(127:0],cplx0[127:0])
            v8sf perm12 = _mm256_permute2f128_ps(cplx02, cplx12, 0x31);  // permute mask [cplx1(255:128],cplx0[255:128])
            _mm256_storeu_ps((float *) (dst) + j, perm0);
            _mm256_storeu_ps((float *) (dst) + j + AVX_LEN_FLOAT, perm1);
            _mm256_storeu_ps((float *) (dst) + j + 2 * AVX_LEN_FLOAT, perm02);
            _mm256_storeu_ps((float *) (dst) + j + 3 * AVX_LEN_FLOAT, perm12);
            j += 4 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
}

static inline void convert256_64f32f(double *src, float *dst, int len)
{
    int stop_len = len / (4 * AVX_LEN_DOUBLE);
    stop_len *= (4 * AVX_LEN_DOUBLE);

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            v4sd src_tmp2 = _mm256_load_pd(src + i + AVX_LEN_DOUBLE);
            v4sd src_tmp3 = _mm256_load_pd(src + i + 2 * AVX_LEN_DOUBLE);
            v4sd src_tmp4 = _mm256_load_pd(src + i + 3 * AVX_LEN_DOUBLE);
            v4sf src_lo = _mm256_cvtpd_ps(src_tmp);
            v4sf src_hi = _mm256_cvtpd_ps(src_tmp2);
            v4sf src_lo2 = _mm256_cvtpd_ps(src_tmp3);
            v4sf src_hi2 = _mm256_cvtpd_ps(src_tmp4);
            v8sf dst_tmp = _mm256_set_m128(src_hi, src_lo);
            v8sf dst_tmp2 = _mm256_set_m128(src_hi2, src_lo2);
            _mm256_store_ps(dst + j, dst_tmp);
            _mm256_store_ps(dst + j + AVX_LEN_FLOAT, dst_tmp2);
            j += 2 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            v4sd src_tmp2 = _mm256_loadu_pd(src + i + AVX_LEN_DOUBLE);
            v4sd src_tmp3 = _mm256_loadu_pd(src + i + 2 * AVX_LEN_DOUBLE);
            v4sd src_tmp4 = _mm256_loadu_pd(src + i + 3 * AVX_LEN_DOUBLE);
            v4sf src_lo = _mm256_cvtpd_ps(src_tmp);
            v4sf src_hi = _mm256_cvtpd_ps(src_tmp2);
            v4sf src_lo2 = _mm256_cvtpd_ps(src_tmp3);
            v4sf src_hi2 = _mm256_cvtpd_ps(src_tmp4);
            v8sf dst_tmp = _mm256_set_m128(src_hi, src_lo);
            v8sf dst_tmp2 = _mm256_set_m128(src_hi2, src_lo2);
            _mm256_storeu_ps(dst + j, dst_tmp);
            _mm256_storeu_ps(dst + j + AVX_LEN_FLOAT, dst_tmp2);
            j += 2 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i];
    }
}

// Should we add more unrolling to improve this?
static inline void convert256_32f64f(float *src, double *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_load_ps(src + i + SSE_LEN_FLOAT);
            v4sd dst_tmp = _mm256_cvtps_pd(src_tmp);
            v4sd dst_tmp2 = _mm256_cvtps_pd(src_tmp2);
            _mm256_store_pd(dst + i, dst_tmp);  // store the abcd converted in 64bits
            _mm256_store_pd(dst + i + AVX_LEN_DOUBLE, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);  // load a,b,c,d
            v4sf src_tmp2 = _mm_loadu_ps(src + i + SSE_LEN_FLOAT);
            v4sd dst_tmp = _mm256_cvtps_pd(src_tmp);
            v4sd dst_tmp2 = _mm256_cvtps_pd(src_tmp2);
            _mm256_storeu_pd(dst + i, dst_tmp);  // store the abcd converted in 64bits
            _mm256_storeu_pd(dst + i + AVX_LEN_DOUBLE, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (double) src[i];
    }
}

static inline void flip256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    int mini = ((len - 1) < (2 * AVX_LEN_FLOAT)) ? (len - 1) : (2 * AVX_LEN_FLOAT);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    // Since we work in reverse we do not know for sure if destination address will be aligned
    // Could it be improved?
#ifdef __AVX2__
    v8si reverse_reg = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
#endif
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst + len - AVX_LEN_FLOAT), AVX_LEN_BYTES)) {
        for (int i = 2 * AVX_LEN_FLOAT; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);                   // load a,b,c,d,e,f,g,h
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);  // load a,b,c,d,e,f,g,h
#ifdef __AVX2__
            v8sf src_tmp_flip = _mm256_permutevar8x32_ps(src_tmp, reverse_reg);    // reverse lanes abcdefgh to hgfedcba
            v8sf src_tmp_flip2 = _mm256_permutevar8x32_ps(src_tmp2, reverse_reg);  // reverse lanes abcdefgh to hgfedcba
#else
            v8sf src_tmp_flip = _mm256_permute2f128_ps(src_tmp, src_tmp, IMM8_PERMUTE_128BITS_LANES);
            v8sf src_tmp_flip2 = _mm256_permute2f128_ps(src_tmp2, src_tmp2, IMM8_PERMUTE_128BITS_LANES);
#endif
            _mm256_storeu_ps(dst + len - i - AVX_LEN_FLOAT, src_tmp_flip);       // store the flipped vector
            _mm256_storeu_ps(dst + len - i - 2 * AVX_LEN_FLOAT, src_tmp_flip2);  // store the flipped vector
        }
    } else {
        for (int i = 2 * AVX_LEN_FLOAT; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);                   // load a,b,c,d,e,f,g,h
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);  // load a,b,c,d,e,f,g,h
#ifdef __AVX2__
            v8sf src_tmp_flip = _mm256_permutevar8x32_ps(src_tmp, reverse_reg);    // reverse lanes abcdefgh to hgfedcba
            v8sf src_tmp_flip2 = _mm256_permutevar8x32_ps(src_tmp2, reverse_reg);  // reverse lanes abcdefgh to hgfedcba
#else
            v8sf src_tmp_flip = _mm256_permute2f128_ps(src_tmp, src_tmp, IMM8_PERMUTE_128BITS_LANES);
            v8sf src_tmp_flip2 = _mm256_permute2f128_ps(src_tmp2, src_tmp2, IMM8_PERMUTE_128BITS_LANES);
#endif
            _mm256_storeu_ps(dst + len - i - AVX_LEN_FLOAT, src_tmp_flip);       // store the flipped vector
            _mm256_storeu_ps(dst + len - i - 2 * AVX_LEN_FLOAT, src_tmp_flip2);  // store the flipped vector
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void maxevery256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_load_ps(src1 + i);
            v8sf src2_tmp = _mm256_load_ps(src2 + i);
            v8sf src1_tmp2 = _mm256_load_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf src2_tmp2 = _mm256_load_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf max1 = _mm256_max_ps(src1_tmp, src2_tmp);
            v8sf max2 = _mm256_max_ps(src1_tmp2, src2_tmp2);
            _mm256_store_ps(dst + i, max1);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, max2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_loadu_ps(src1 + i);
            v8sf src2_tmp = _mm256_loadu_ps(src2 + i);
            v8sf src1_tmp2 = _mm256_loadu_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf src2_tmp2 = _mm256_loadu_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf max1 = _mm256_max_ps(src1_tmp, src2_tmp);
            v8sf max2 = _mm256_max_ps(src1_tmp2, src2_tmp2);
            _mm256_storeu_ps(dst + i, max1);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, max2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery256f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_load_ps(src1 + i);
            v8sf src2_tmp = _mm256_load_ps(src2 + i);
            v8sf src1_tmp2 = _mm256_load_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf src2_tmp2 = _mm256_load_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf min1 = _mm256_min_ps(src1_tmp, src2_tmp);
            v8sf min2 = _mm256_min_ps(src1_tmp2, src2_tmp2);
            _mm256_store_ps(dst + i, min1);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, min2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_loadu_ps(src1 + i);
            v8sf src2_tmp = _mm256_loadu_ps(src2 + i);
            v8sf src1_tmp2 = _mm256_loadu_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf src2_tmp2 = _mm256_loadu_ps(src2 + i + AVX_LEN_FLOAT);
            v8sf min1 = _mm256_min_ps(src1_tmp, src2_tmp);
            v8sf min2 = _mm256_min_ps(src1_tmp2, src2_tmp2);
            _mm256_storeu_ps(dst + i, min1);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, min2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void minmax256f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = (len - AVX_LEN_FLOAT) / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    v8sf max_v, min_v, max_v2, min_v2;
    v8sf src_tmp, src_tmp2;

    float min_tmp = src[0];
    float max_tmp = src[0];

    if (len >= AVX_LEN_FLOAT) {
        if (isAligned((uintptr_t) (src), AVX_LEN_BYTES)) {
            src_tmp = _mm256_load_ps(src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = AVX_LEN_FLOAT; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
                src_tmp = _mm256_load_ps(src + i);
                src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
                max_v = _mm256_max_ps(max_v, src_tmp);
                min_v = _mm256_min_ps(min_v, src_tmp);
                max_v2 = _mm256_max_ps(max_v2, src_tmp2);
                min_v2 = _mm256_min_ps(min_v2, src_tmp2);
            }
        } else {
            src_tmp = _mm256_loadu_ps(src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = AVX_LEN_FLOAT; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
                src_tmp = _mm256_loadu_ps(src + i);
                src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
                max_v = _mm256_max_ps(max_v, src_tmp);
                min_v = _mm256_min_ps(min_v, src_tmp);
                max_v2 = _mm256_max_ps(max_v2, src_tmp2);
                min_v2 = _mm256_min_ps(min_v2, src_tmp2);
            }
        }

        max_v = _mm256_max_ps(max_v, max_v2);
        min_v = _mm256_min_ps(min_v, min_v2);

#if 1
        v4sf max3 = _mm256_castps256_ps128(max_v);
        v4sf min3 = _mm256_castps256_ps128(min_v);
        v4sf max4 = _mm256_extractf128_ps(max_v, 1);
        v4sf min4 = _mm256_extractf128_ps(min_v, 1);
        max4 = _mm_max_ps(max3, max4);
        min4 = _mm_min_ps(min3, min4);
        max3 = _mm_permute_ps(max4, 0x0E);
        min3 = _mm_permute_ps(min4, 0x0E);
        max4 = _mm_max_ps(max3, max4);
        min4 = _mm_min_ps(min3, min4);
        max3 = _mm_permute_ps(max4, 0x01);
        min3 = _mm_permute_ps(min4, 0x01);
        max4 = _mm_max_ps(max3, max4);
        min4 = _mm_min_ps(min3, min4);
        _mm_store_ss(&max_tmp, max4);
        _mm_store_ss(&min_tmp, min4);
#else
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
#endif
    }

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

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_min_ps(src_tmp, tmp);
            v8sf dst_tmp2 = _mm256_min_ps(src_tmp2, tmp);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_min_ps(src_tmp, tmp);
            v8sf dst_tmp2 = _mm256_min_ps(src_tmp2, tmp);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

#if 1
static inline void threshold256_gtabs_f(float *src, float *dst, int len, float value)
{
    const v8sf pval = _mm256_set1_ps(value);

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_sign = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_sign_mask);  // extract sign
            v8sf src_sign2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_sign_mask);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);  // take absolute value
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf dst_tmp = _mm256_min_ps(src_abs, pval);
            v8sf dst_tmp2 = _mm256_min_ps(src_abs2, pval);
            dst_tmp = _mm256_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm256_xor_ps(dst_tmp2, src_sign2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_sign = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_sign_mask);  // extract sign
            v8sf src_sign2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_sign_mask);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);  // take absolute value
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf dst_tmp = _mm256_min_ps(src_abs, pval);
            v8sf dst_tmp2 = _mm256_min_ps(src_abs2, pval);
            dst_tmp = _mm256_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm256_xor_ps(dst_tmp2, src_sign2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
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
static inline void threshold256_gtabs_f(float *src, float *dst, int len, float value)
{
    const v8sf pval = _mm256_set1_ps(value);
    const v8sf mval = _mm256_set1_ps(-value);

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf eqmask2 = _mm256_cmp_ps(src_abs2, src_tmp2, _CMP_EQ_OS);
            v8sf gtmask = _mm256_cmp_ps(src_abs, pval, _CMP_GT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf gtmask2 = _mm256_cmp_ps(src_abs2, pval, _CMP_GT_OS);
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);  // if A >= 0 value, else -value
            v8sf sval2 = _mm256_blendv_ps(mval, pval, eqmask2);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            v8sf dst_tmp2 = _mm256_blendv_ps(src_tmp2, sval2, gtmask2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf eqmask2 = _mm256_cmp_ps(src_abs2, src_tmp2, _CMP_EQ_OS);
            v8sf gtmask = _mm256_cmp_ps(src_abs, pval, _CMP_GT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf gtmask2 = _mm256_cmp_ps(src_abs2, pval, _CMP_GT_OS);
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);  // if A >= 0 value, else -value
            v8sf sval2 = _mm256_blendv_ps(mval, pval, eqmask2);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, gtmask);  // either A or sval (+- value)
            v8sf dst_tmp2 = _mm256_blendv_ps(src_tmp2, sval2, gtmask2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
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

static inline void threshold256_lt_f(float *src, float *dst, int len, float value)
{
    v8sf tmp = _mm256_set1_ps(value);  //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_max_ps(src_tmp, tmp);
            v8sf dst_tmp2 = _mm256_max_ps(src_tmp2, tmp);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_max_ps(src_tmp, tmp);
            v8sf dst_tmp2 = _mm256_max_ps(src_tmp2, tmp);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

#if 1
static inline void threshold256_ltabs_f(float *src, float *dst, int len, float value)
{
    const v8sf pval = _mm256_set1_ps(value);

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_sign = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_sign_mask);  // extract sign
            v8sf src_sign2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_sign_mask);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);  // take absolute value
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf dst_tmp = _mm256_max_ps(src_abs, pval);
            v8sf dst_tmp2 = _mm256_max_ps(src_abs2, pval);
            dst_tmp = _mm256_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm256_xor_ps(dst_tmp2, src_sign2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_sign = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_sign_mask);  // extract sign
            v8sf src_sign2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_sign_mask);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);  // take absolute value
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf dst_tmp = _mm256_max_ps(src_abs, pval);
            v8sf dst_tmp2 = _mm256_max_ps(src_abs2, pval);
            dst_tmp = _mm256_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm256_xor_ps(dst_tmp2, src_sign2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
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
static inline void threshold256_ltabs_f(float *src, float *dst, int len, float value)
{
    const v8sf pval = _mm256_set1_ps(value);
    const v8sf mval = _mm256_set1_ps(-value);

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf eqmask2 = _mm256_cmp_ps(src_abs2, src_tmp2, _CMP_EQ_OS);
            v8sf ltmask = _mm256_cmp_ps(src_abs, pval, _CMP_LT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf ltmask2 = _mm256_cmp_ps(src_abs2, pval, _CMP_LT_OS);
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);  // if A >= 0 value, else -value
            v8sf sval2 = _mm256_blendv_ps(mval, pval, eqmask2);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, ltmask);  // either A or sval (+- value)
            v8sf dst_tmp2 = _mm256_blendv_ps(src_tmp2, sval2, ltmask2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf src_abs = _mm256_and_ps(src_tmp, *(v8sf *) _ps256_pos_sign_mask);
            v8sf src_abs2 = _mm256_and_ps(src_tmp2, *(v8sf *) _ps256_pos_sign_mask);
            v8sf eqmask = _mm256_cmp_ps(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v8sf eqmask2 = _mm256_cmp_ps(src_abs2, src_tmp2, _CMP_EQ_OS);
            v8sf ltmask = _mm256_cmp_ps(src_abs, pval, _CMP_LT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            v8sf ltmask2 = _mm256_cmp_ps(src_abs2, pval, _CMP_LT_OS);
            v8sf sval = _mm256_blendv_ps(mval, pval, eqmask);  // if A >= 0 value, else -value
            v8sf sval2 = _mm256_blendv_ps(mval, pval, eqmask2);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, sval, ltmask);  // either A or sval (+- value)
            v8sf dst_tmp2 = _mm256_blendv_ps(src_tmp2, sval2, ltmask2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
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

static inline void threshold256_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v8sf ltlevel_v = _mm256_set1_ps(ltlevel);
    const v8sf ltvalue_v = _mm256_set1_ps(ltvalue);
    const v8sf gtlevel_v = _mm256_set1_ps(gtlevel);
    const v8sf gtvalue_v = _mm256_set1_ps(gtvalue);

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf lt_mask = _mm256_cmp_ps(src_tmp, ltlevel_v, _CMP_LT_OS);
            v8sf gt_mask = _mm256_cmp_ps(src_tmp, gtlevel_v, _CMP_GT_OS);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm256_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm256_store_ps(dst + i, dst_tmp);
            v8sf lt_mask2 = _mm256_cmp_ps(src_tmp2, ltlevel_v, _CMP_LT_OS);
            v8sf gt_mask2 = _mm256_cmp_ps(src_tmp2, gtlevel_v, _CMP_GT_OS);
            v8sf dst_tmp2 = _mm256_blendv_ps(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm256_blendv_ps(dst_tmp2, gtvalue_v, gt_mask2);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf lt_mask = _mm256_cmp_ps(src_tmp, ltlevel_v, _CMP_LT_OS);
            v8sf gt_mask = _mm256_cmp_ps(src_tmp, gtlevel_v, _CMP_GT_OS);
            v8sf dst_tmp = _mm256_blendv_ps(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm256_blendv_ps(dst_tmp, gtvalue_v, gt_mask);
            _mm256_storeu_ps(dst + i, dst_tmp);
            v8sf lt_mask2 = _mm256_cmp_ps(src_tmp2, ltlevel_v, _CMP_LT_OS);
            v8sf gt_mask2 = _mm256_cmp_ps(src_tmp2, gtlevel_v, _CMP_GT_OS);
            v8sf dst_tmp2 = _mm256_blendv_ps(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm256_blendv_ps(dst_tmp2, gtvalue_v, gt_mask2);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), AVX_LEN_FLOAT)) {
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

// e^ix = cos(x) + i*sin(x)
static inline void sincos256f_interleaved(float *src, complex32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sfx2 dst_tmp;
            sincos256_ps(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm256_store2_ps((float *) dst + j, dst_tmp);
            j += 2 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sfx2 dst_tmp;
            sincos256_ps(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm256_store2u_ps((float *) dst + j, dst_tmp);
            j += 2 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], &(dst[i].im), &(dst[i].re));
    }
}

#ifndef NO_OOB
static inline v8sf acosh256f_ps(v8sf x)
{
    v8sf z, z_first_branch, z_second_branch;
    v8sf xsup1500, zinf0p5, xinf1;

    xsup1500 = _mm256_cmp_ps(x, *(v8sf *) _ps256_1500, _CMP_GT_OS);  // return  (logf(x) + LOGE2F)
    xinf1 = _mm256_cmp_ps(x, *(v8sf *) _ps256_1, _CMP_LT_OS);        // return 0

    z = _mm256_sub_ps(x, *(v8sf *) _ps256_1);

    zinf0p5 = _mm256_cmp_ps(z, *(v8sf *) _ps256_0p5, _CMP_LT_OS);  // first and second branch

    // First Branch (z < 0.5)
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ACOSH_P0, z, *(v8sf *) _ps256_ACOSH_P1);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P2);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P3);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P4);
    z_first_branch = _mm256_mul_ps(z_first_branch, _mm256_sqrt_ps(z));

    // Second Branch
    z_second_branch = _mm256_sqrt_ps(_mm256_fmadd_ps_custom(z, x, z));
    z_second_branch = log256_ps(_mm256_add_ps(x, z_second_branch));

    z = _mm256_blendv_ps(z_second_branch, z_first_branch, zinf0p5);
    z = _mm256_blendv_ps(z, _mm256_add_ps(log256_ps(x), *(v8sf *) _ps256_LOGE2F), xsup1500);
    z = _mm256_blendv_ps(z, _mm256_setzero_ps(), xinf1);

    return z;
}
#else
static inline v8sf acosh256f_ps(v8sf x)
{
    v8sf z, z_first_branch, z_second_branch;
    v8sf xsup1500, zinf0p5, xinf1;

    z = _mm256_sub_ps(x, *(v8sf *) _ps256_1);

    zinf0p5 = _mm256_cmp_ps(z, *(v8sf *) _ps256_0p5, _CMP_LT_OS);  // first and second branch

    // First Branch (z < 0.5)
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ACOSH_P0, z, *(v8sf *) _ps256_ACOSH_P1);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P2);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P3);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, z, *(v8sf *) _ps256_ACOSH_P4);
    z_first_branch = _mm256_mul_ps(z_first_branch, _mm256_sqrt_ps(z));

    // Second Branch
    z_second_branch = _mm256_sqrt_ps(_mm256_fmadd_ps_custom(z, x, z));
    z_second_branch = log256_ps(_mm256_add_ps(x, z_second_branch));

    z = _mm256_blendv_ps(z_second_branch, z_first_branch, zinf0p5);

    return z;
}
#endif

static inline void acosh256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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
    // First Branch (x < 0.5)
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ASINH_P0, tmp, *(v8sf *) _ps256_ASINH_P1);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ASINH_P2);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ASINH_P3);
    z_first_branch = _mm256_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, x, x);

    // Second Branch
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    // First branch
    tmp = _mm256_mul_ps(x, x);
    z_first_branch = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_ATANH_P0, tmp, *(v8sf *) _ps256_ATANH_P1);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ATANH_P2);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ATANH_P3);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, tmp, *(v8sf *) _ps256_ATANH_P4);
    z_first_branch = _mm256_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm256_fmadd_ps_custom(z_first_branch, x, x);

    // Second branch
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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
    v8sf sign;
    v8sf suptan3pi8, inftan3pi8suppi8;
    v8sf tmp;

    x = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, xx);
    // sign = _mm256_cmp_ps(xx, _mm256_setzero_ps(), _CMP_LT_OS);  // 0xFFFFFFFF if x < 0.0, sign = -1
    sign = _mm256_and_ps(xx, *(v8sf *) _ps256_sign_mask);
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

    // y = _mm256_blendv_ps(y, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, y), sign);
    y = _mm256_xor_ps(y, sign);

    return (y);
}

static inline void atan256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

static inline void atan2256f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= 2 * AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sfx2 src1 = _mm256_load2_ps((float *) (src) + j);
            v8sfx2 src2 = _mm256_load2_ps((float *) (src) + j + 2 * AVX_LEN_FLOAT);
            _mm256_store_ps(dst + i, atan2256f_ps(src1.val[1], src1.val[0]));
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, atan2256f_ps(src2.val[1], src2.val[0]));
            j += 4 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sfx2 src1 = _mm256_load2u_ps((float *) (src) + j);
            v8sfx2 src2 = _mm256_load2u_ps((float *) (src) + j + 2 * AVX_LEN_FLOAT);
            _mm256_storeu_ps(dst + i, atan2256f_ps(src1.val[1], src1.val[0]));
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, atan2256f_ps(src2.val[1], src2.val[0]));
            j += 4 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src[i].im, src[i].re);
    }
}


static inline v8sf asin256f_ps(v8sf xx)
{
    v8sf a, x, z, z_tmp;
    v8sf sign;
    v8sf ainfem4, asup0p5;
    v8sf tmp;
    x = xx;
    a = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, x);  // fabs(x)
    // sign = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OS);  // 0xFFFFFFFF if x < 0.0
    sign = _mm256_and_ps(xx, *(v8sf *) _ps256_sign_mask);

    // TODO : vectorize this
    /*if( a > 1.0f )
    {
        return( 0.0f );
    }*/


    ainfem4 = _mm256_cmp_ps(a, _mm256_set1_ps(1.0e-4), _CMP_LT_OS);  // if( a < 1.0e-4f )

    asup0p5 = _mm256_cmp_ps(a, *(v8sf *) _ps256_0p5, _CMP_GT_OS);  // if( a > 0.5f ) flag = 1 else 0
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

    // with FMA (fmsub_ps), it could be 1 or 2 cycles faster
    z_tmp = _mm256_add_ps(z, z);
    z_tmp = _mm256_sub_ps(*(v8sf *) _ps256_PIO2F, z_tmp);

    z = _mm256_blendv_ps(z, z_tmp, asup0p5);

    // done:
    z = _mm256_blendv_ps(z, a, ainfem4);
    // z = _mm256_blendv_ps(z, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, z), sign);
    z = _mm256_xor_ps(z, sign);

    return (z);
}

static inline void asin256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    // z = x * x;
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN tan256f_ps" ::
                       : "memory");
#endif
    v8sf x, y, z, zz;
    v8si j;  // long?
    v8sf sign, xsupem4;
    v8sf tmp;
    v8si jandone, jandtwo;

    x = _mm256_and_ps(*(v8sf *) _ps256_pos_sign_mask, xx);  // fabs(xx)

    /* compute x mod PIO4 */

    // TODO : on neg values should be ceil and not floor
    // j = _mm256_cvtps_epi32( _mm256_round_ps(_mm256_mul_ps(*(v8sf*)_ps256_FOPI,x), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC )); /* integer part of x/(PI/4), using floor */
    j = _mm256_cvttps_epi32(_mm256_mul_ps(*(v8sf *) _ps256_FOPI, x));
    y = _mm256_cvtepi32_ps(j);


#ifndef __AVX2__
    v4si andone_gt_0, andone_gt_1;
    v8si andone_gt;
    v4si j_0, j_1;
    COPY_IMM_TO_XMM(j, j_0, j_1);

    // FT: 0 1 and not 1 0?
    andone_gt_0 = _mm_and_si128(j_0, *(v4si *) _pi32avx_1);
    andone_gt_1 = _mm_and_si128(j_1, *(v4si *) _pi32avx_1);
    COPY_XMM_TO_IMM(andone_gt_0, andone_gt_1, andone_gt);
    jandone = _mm256_cmpgt_epi32(andone_gt, _mm256_setzero_si256());
#else
    jandone = _mm256_cmpgt_epi32(_mm256_and_si256(j, *(v8si *) _pi32_256_1), _mm256_setzero_si256());
#endif

    y = _mm256_blendv_ps(y, _mm256_add_ps(y, *(v8sf *) _ps256_1), _mm256_cvtepi32_ps(jandone));
    j = _mm256_cvttps_epi32(y);  // no need to round again

    // z = ((x - y * DP1) - y * DP2) - y * DP3;

#if 1

    z = _mm256_fmadd_ps_custom(y, *(v8sf *) _ps256_DP1, x);
    z = _mm256_fmadd_ps_custom(y, *(v8sf *) _ps256_DP2, z);
    z = _mm256_fmadd_ps_custom(y, *(v8sf *) _ps256_DP3, z);
#else  // faster but less precision
    tmp = _mm256_mul_ps(y, *(v8sf *) _ps256_DP123);
    z = _mm256_sub_ps(x, tmp);
#endif
    zz = _mm256_mul_ps(z, z);  // z*z

    // TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_TAN_P0, zz, *(v8sf *) _ps256_TAN_P1);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P2);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P3);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P4);
    tmp = _mm256_fmadd_ps_custom(tmp, zz, *(v8sf *) _ps256_TAN_P5);
    tmp = _mm256_mul_ps(zz, tmp);
    tmp = _mm256_fmadd_ps_custom(tmp, z, z);

    xsupem4 = _mm256_cmp_ps(x, _mm256_set1_ps(1.0e-4), _CMP_GT_OS);  // if( x > 1.0e-4 )
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

    sign = _mm256_cmp_ps(xx, _mm256_setzero_ps(), _CMP_LT_OS);  // 0xFFFFFFFF if xx < 0.0
    y = _mm256_blendv_ps(y, _mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, y), sign);
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END tan256f_ps" ::
                       : "memory");
#endif
    return (y);
}

static inline void tan256f(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_load_ps(srcRe + i);
            v8sf im_tmp = _mm256_load_ps(srcIm + i);
            v8sf re_tmp2 = _mm256_load_ps(srcRe + i + AVX_LEN_FLOAT);
            v8sf im_tmp2 = _mm256_load_ps(srcIm + i + AVX_LEN_FLOAT);
            v8sf re_square = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf re_square2 = _mm256_mul_ps(re_tmp2, re_tmp2);
            v8sf dst_tmp = _mm256_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v8sf dst_tmp2 = _mm256_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            dst_tmp = _mm256_sqrt_ps(dst_tmp);
            dst_tmp2 = _mm256_sqrt_ps(dst_tmp2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_loadu_ps(srcRe + i);
            v8sf im_tmp = _mm256_loadu_ps(srcIm + i);
            v8sf re_tmp2 = _mm256_loadu_ps(srcRe + i + AVX_LEN_FLOAT);
            v8sf im_tmp2 = _mm256_loadu_ps(srcIm + i + AVX_LEN_FLOAT);
            v8sf re_square = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf re_square2 = _mm256_mul_ps(re_tmp2, re_tmp2);
            v8sf dst_tmp = _mm256_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v8sf dst_tmp2 = _mm256_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            dst_tmp = _mm256_sqrt_ps(dst_tmp);
            dst_tmp2 = _mm256_sqrt_ps(dst_tmp2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]));
    }
}

static inline void powerspect256f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_load_ps(srcRe + i);
            v8sf im_tmp = _mm256_load_ps(srcIm + i);
            v8sf re_tmp2 = _mm256_load_ps(srcRe + i + AVX_LEN_FLOAT);
            v8sf im_tmp2 = _mm256_load_ps(srcIm + i + AVX_LEN_FLOAT);
            v8sf re_square = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf re_square2 = _mm256_mul_ps(re_tmp2, re_tmp2);
            v8sf dst_tmp = _mm256_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v8sf dst_tmp2 = _mm256_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf re_tmp = _mm256_loadu_ps(srcRe + i);
            v8sf im_tmp = _mm256_loadu_ps(srcIm + i);
            v8sf re_tmp2 = _mm256_loadu_ps(srcRe + i + AVX_LEN_FLOAT);
            v8sf im_tmp2 = _mm256_loadu_ps(srcIm + i + AVX_LEN_FLOAT);
            v8sf re_square = _mm256_mul_ps(re_tmp, re_tmp);
            v8sf re_square2 = _mm256_mul_ps(re_tmp2, re_tmp2);
            v8sf dst_tmp = _mm256_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v8sf dst_tmp2 = _mm256_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]);
    }
}

static inline void magnitude256f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (4 * AVX_LEN_FLOAT);
    stop_len *= 4 * AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sfx2 src_split = _mm256_load2_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v8sfx2 src_split2 = _mm256_load2_ps((float *) (src) + j + 2 * AVX_LEN_FLOAT);
            v8sfx2 src_split3 = _mm256_load2_ps((float *) (src) + j + 4 * AVX_LEN_FLOAT);
            v8sfx2 src_split4 = _mm256_load2_ps((float *) (src) + j + 6 * AVX_LEN_FLOAT);
            v8sf split_square0 = _mm256_mul_ps(src_split.val[0], src_split.val[0]);
            v8sf split2_square0 = _mm256_mul_ps(src_split2.val[0], src_split2.val[0]);
            v8sf split3_square0 = _mm256_mul_ps(src_split3.val[0], src_split3.val[0]);
            v8sf split4_square0 = _mm256_mul_ps(src_split4.val[0], src_split4.val[0]);
            v8sfx2 dst_split;
            v8sfx2 dst_split2;
            dst_split.val[0] = _mm256_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm256_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm256_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm256_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            dst_split.val[0] = _mm256_sqrt_ps(dst_split.val[0]);
            dst_split.val[1] = _mm256_sqrt_ps(dst_split.val[1]);
            dst_split2.val[0] = _mm256_sqrt_ps(dst_split2.val[0]);
            dst_split2.val[1] = _mm256_sqrt_ps(dst_split2.val[1]);

            _mm256_store_ps((float *) (dst) + i, dst_split.val[0]);
            _mm256_store_ps((float *) (dst) + i + AVX_LEN_FLOAT, dst_split.val[1]);
            _mm256_store_ps((float *) (dst) + i + 2 * AVX_LEN_FLOAT, dst_split2.val[0]);
            _mm256_store_ps((float *) (dst) + i + 3 * AVX_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sfx2 src_split = _mm256_load2u_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v8sfx2 src_split2 = _mm256_load2u_ps((float *) (src) + j + 2 * AVX_LEN_FLOAT);
            v8sfx2 src_split3 = _mm256_load2u_ps((float *) (src) + j + 4 * AVX_LEN_FLOAT);
            v8sfx2 src_split4 = _mm256_load2u_ps((float *) (src) + j + 6 * AVX_LEN_FLOAT);
            v8sf split_square0 = _mm256_mul_ps(src_split.val[0], src_split.val[0]);
            v8sf split2_square0 = _mm256_mul_ps(src_split2.val[0], src_split2.val[0]);
            v8sf split3_square0 = _mm256_mul_ps(src_split3.val[0], src_split3.val[0]);
            v8sf split4_square0 = _mm256_mul_ps(src_split4.val[0], src_split4.val[0]);
            v8sfx2 dst_split;
            v8sfx2 dst_split2;
            dst_split.val[0] = _mm256_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm256_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm256_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm256_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            dst_split.val[0] = _mm256_sqrt_ps(dst_split.val[0]);
            dst_split.val[1] = _mm256_sqrt_ps(dst_split.val[1]);
            dst_split2.val[0] = _mm256_sqrt_ps(dst_split2.val[0]);
            dst_split2.val[1] = _mm256_sqrt_ps(dst_split2.val[1]);

            _mm256_storeu_ps((float *) (dst) + i, dst_split.val[0]);
            _mm256_storeu_ps((float *) (dst) + i + AVX_LEN_FLOAT, dst_split.val[1]);
            _mm256_storeu_ps((float *) (dst) + i + 2 * AVX_LEN_FLOAT, dst_split2.val[0]);
            _mm256_storeu_ps((float *) (dst) + i + 3 * AVX_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i].re * src[i].re + (src[i].im * src[i].im));
    }
}

static inline void powerspect256f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (4 * AVX_LEN_FLOAT);
    stop_len *= 4 * AVX_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sfx2 src_split = _mm256_load2_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v8sfx2 src_split2 = _mm256_load2_ps((float *) (src) + j + 2 * AVX_LEN_FLOAT);
            v8sfx2 src_split3 = _mm256_load2_ps((float *) (src) + j + 4 * AVX_LEN_FLOAT);
            v8sfx2 src_split4 = _mm256_load2_ps((float *) (src) + j + 6 * AVX_LEN_FLOAT);
            v8sf split_square0 = _mm256_mul_ps(src_split.val[0], src_split.val[0]);
            v8sf split2_square0 = _mm256_mul_ps(src_split2.val[0], src_split2.val[0]);
            v8sf split3_square0 = _mm256_mul_ps(src_split3.val[0], src_split3.val[0]);
            v8sf split4_square0 = _mm256_mul_ps(src_split4.val[0], src_split4.val[0]);
            v8sfx2 dst_split;
            v8sfx2 dst_split2;
            dst_split.val[0] = _mm256_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm256_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm256_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm256_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            _mm256_store_ps((dst + i), dst_split.val[0]);
            _mm256_store_ps((dst + i + AVX_LEN_FLOAT), dst_split.val[1]);
            _mm256_store_ps((dst + i + 2 * AVX_LEN_FLOAT), dst_split2.val[0]);
            _mm256_store_ps((dst + i + 3 * AVX_LEN_FLOAT), dst_split2.val[1]);
            j += 8 * AVX_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sfx2 src_split = _mm256_load2u_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v8sfx2 src_split2 = _mm256_load2u_ps((float *) (src) + j + 2 * AVX_LEN_FLOAT);
            v8sfx2 src_split3 = _mm256_load2u_ps((float *) (src) + j + 4 * AVX_LEN_FLOAT);
            v8sfx2 src_split4 = _mm256_load2u_ps((float *) (src) + j + 6 * AVX_LEN_FLOAT);
            v8sf split_square0 = _mm256_mul_ps(src_split.val[0], src_split.val[0]);
            v8sf split2_square0 = _mm256_mul_ps(src_split2.val[0], src_split2.val[0]);
            v8sf split3_square0 = _mm256_mul_ps(src_split3.val[0], src_split3.val[0]);
            v8sf split4_square0 = _mm256_mul_ps(src_split4.val[0], src_split4.val[0]);
            v8sfx2 dst_split;
            v8sfx2 dst_split2;
            dst_split.val[0] = _mm256_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm256_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm256_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm256_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            _mm256_storeu_ps((dst + i), dst_split.val[0]);
            _mm256_storeu_ps((dst + i + AVX_LEN_FLOAT), dst_split.val[1]);
            _mm256_storeu_ps((dst + i + 2 * AVX_LEN_FLOAT), dst_split2.val[0]);
            _mm256_storeu_ps((dst + i + 3 * AVX_LEN_FLOAT), dst_split2.val[1]);
            j += 8 * AVX_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i].re * src[i].re + (src[i].im * src[i].im);
    }
}

static inline void subcrev256f(float *src, float value, float *dst, int len)
{
    const v8sf tmp = _mm256_set1_ps(value);

    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT];
    float tmp_acc = 0.0f;
    v8sf vec_acc1 = _mm256_setzero_ps();  // initialize the vector accumulator
    v8sf vec_acc2 = _mm256_setzero_ps();  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), AVX_LEN_BYTES)) {
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

static inline void dot256f(float *src1, float *src2, int len, float *dst)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT];
    float tmp_acc = 0.0f;
    v8sf vec_acc1 = _mm256_setzero_ps();  // initialize the vector accumulator
    v8sf vec_acc2 = _mm256_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf vec_src1_tmp = _mm256_load_ps(src1 + i);
            v8sf vec_src1_tmp2 = _mm256_load_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf vec_src2_tmp = _mm256_load_ps(src2 + i);
            v8sf vec_src2_tmp2 = _mm256_load_ps(src2 + i + AVX_LEN_FLOAT);
            vec_acc1 = _mm256_fmadd_ps_custom(vec_src1_tmp, vec_src2_tmp, vec_acc1);
            vec_acc2 = _mm256_fmadd_ps_custom(vec_src1_tmp2, vec_src2_tmp2, vec_acc2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf vec_src1_tmp = _mm256_loadu_ps(src1 + i);
            v8sf vec_src1_tmp2 = _mm256_loadu_ps(src1 + i + AVX_LEN_FLOAT);
            v8sf vec_src2_tmp = _mm256_loadu_ps(src2 + i);
            v8sf vec_src2_tmp2 = _mm256_loadu_ps(src2 + i + AVX_LEN_FLOAT);
            vec_acc1 = _mm256_fmadd_ps_custom(vec_src1_tmp, vec_src2_tmp, vec_acc1);
            vec_acc2 = _mm256_fmadd_ps_custom(vec_src1_tmp2, vec_src2_tmp2, vec_acc2);
        }
    }
    vec_acc1 = _mm256_add_ps(vec_acc1, vec_acc2);
    _mm256_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src1[i] * src2[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] +
              accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];

    *dst = tmp_acc;
}

static inline void dotc256f(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    int stop_len = len / (4 * AVX_LEN_FLOAT);
    stop_len *= (4 * AVX_LEN_FLOAT);

    v8sfx2 vec_acc1 = {_mm256_setzero_ps(), _mm256_setzero_ps()};  // initialize the vector accumulator
    v8sfx2 vec_acc2 = {_mm256_setzero_ps(), _mm256_setzero_ps()};  // initialize the vector accumulator

    complex32_t dst_tmp = {0.0f, 0.0f};

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulateRe[AVX_LEN_FLOAT];
    __attribute__((aligned(AVX_LEN_BYTES))) float accumulateIm[AVX_LEN_FLOAT];

    //  (ac -bd) + i(ad + bc)
    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), AVX_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sfx2 src1_split = _mm256_load2_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v8sfx2 src2_split = _mm256_load2_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v8sfx2 src1_split2 = _mm256_load2_ps((float *) (src1) + i + 2 * AVX_LEN_FLOAT);
            v8sfx2 src2_split2 = _mm256_load2_ps((float *) (src2) + i + 2 * AVX_LEN_FLOAT);
            v8sf ac = _mm256_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v8sf ad = _mm256_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v8sf ac2 = _mm256_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v8sf ad2 = _mm256_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v8sfx2 tmp_split;
            v8sfx2 tmp_split2;
            tmp_split.val[0] = _mm256_fnmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = _mm256_fmadd_ps_custom(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = _mm256_fnmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = _mm256_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = _mm256_add_ps(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = _mm256_add_ps(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = _mm256_add_ps(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = _mm256_add_ps(vec_acc2.val[1], tmp_split2.val[1]);
        }
    } else {
        for (int i = 0; i < 2 * stop_len; i += 4 * AVX_LEN_FLOAT) {
            v8sfx2 src1_split = _mm256_load2u_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v8sfx2 src2_split = _mm256_load2u_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v8sfx2 src1_split2 = _mm256_load2u_ps((float *) (src1) + i + 2 * AVX_LEN_FLOAT);
            v8sfx2 src2_split2 = _mm256_load2u_ps((float *) (src2) + i + 2 * AVX_LEN_FLOAT);
            v8sf ac = _mm256_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v8sf ad = _mm256_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v8sf ac2 = _mm256_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v8sf ad2 = _mm256_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v8sfx2 tmp_split;
            v8sfx2 tmp_split2;
            tmp_split.val[0] = _mm256_fnmadd_ps_custom(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = _mm256_fmadd_ps_custom(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = _mm256_fnmadd_ps_custom(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = _mm256_fmadd_ps_custom(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = _mm256_add_ps(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = _mm256_add_ps(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = _mm256_add_ps(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = _mm256_add_ps(vec_acc2.val[1], tmp_split2.val[1]);
        }
    }

    vec_acc1.val[0] = _mm256_add_ps(vec_acc1.val[0], vec_acc2.val[0]);
    vec_acc1.val[1] = _mm256_add_ps(vec_acc1.val[1], vec_acc2.val[1]);
    _mm256_store_ps(accumulateRe, vec_acc1.val[0]);
    _mm256_store_ps(accumulateIm, vec_acc1.val[1]);

    for (int i = stop_len; i < len; i++) {
        dst_tmp.re += src1[i].re * src2[i].re - (src1[i].im * src2[i].im);
        dst_tmp.im += src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }

    dst_tmp.re = dst_tmp.re + accumulateRe[0] + accumulateRe[1] + accumulateRe[2] + accumulateRe[3] +
                 accumulateRe[4] + accumulateRe[5] + accumulateRe[6] + accumulateRe[7];
    dst_tmp.im = dst_tmp.im + accumulateIm[0] + accumulateIm[1] + accumulateIm[2] + accumulateIm[3] +
                 accumulateIm[4] + accumulateIm[5] + accumulateIm[6] + accumulateIm[7];


    dst->re = dst_tmp.re;
    dst->im = dst_tmp.im;
}

static inline void sqrt256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_sqrt_ps(src_tmp);
            v8sf dst_tmp2 = _mm256_sqrt_ps(src_tmp2);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_sqrt_ps(src_tmp);
            v8sf dst_tmp2 = _mm256_sqrt_ps(src_tmp2);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static inline void round256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}

static inline void trunc256f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            _mm256_store_ps(dst + i, dst_tmp);
            _mm256_store_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_round_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            v8sf dst_tmp2 = _mm256_round_ps(src_tmp2, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(dst + i, dst_tmp);
            _mm256_storeu_ps(dst + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

static inline void cplxvecdiv256f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)

{
    int stop_len = len / (AVX_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX_LEN_FLOAT;   // stop_len << 2;

    int i;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_load_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_load_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v8sf c2d2 = _mm256_mul_ps(src2_tmp, src2_tmp);
            v8sf c2d2_shuf = _mm256_shuffle_ps(c2d2, c2d2, _MM_SHUFFLE(2, 3, 0, 1));
            c2d2 = _mm256_add_ps(c2d2_shuf, c2d2);
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);  // a1,a1,a0,a0
            tmp1 = _mm256_mul_ps(*(v8sf *) _ps256_conj_mask, tmp1);
            v8sf tmp2 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v8sf tmp3 = _mm256_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(tmp2, tmp3);                                        // c1b1, b1d1, c0b0, d0b0
            out = _mm256_fmadd_ps_custom(tmp1, src2_tmp, out);
            out = _mm256_div_ps(out, c2d2);
            _mm256_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_loadu_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_loadu_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v8sf c2d2 = _mm256_mul_ps(src2_tmp, src2_tmp);
            v8sf c2d2_shuf = _mm256_shuffle_ps(c2d2, c2d2, _MM_SHUFFLE(2, 3, 0, 1));
            c2d2 = _mm256_add_ps(c2d2_shuf, c2d2);
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);  // a1,a1,a0,a0
            tmp1 = _mm256_mul_ps(*(v8sf *) _ps256_conj_mask, tmp1);
            v8sf tmp2 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v8sf tmp3 = _mm256_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(tmp2, tmp3);                                        // c1b1, b1d1, c0b0, d0b0
            out = _mm256_fmadd_ps_custom(tmp1, src2_tmp, out);
            out = _mm256_div_ps(out, c2d2);
            _mm256_storeu_ps((float *) (dst) + i, out);
        }
    }
    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2[i].re * src2[i].re + src2[i].im * src2[i].im;
        dst[i].re = ((src1[i].re * src2[i].re) + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-(src1[i].re * src2[i].im) + (src2[i].re * src1[i].im)) / c2d2;
    }
}

static inline void cplxvecdiv256f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= 2 * AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1Re), (uintptr_t) (src2Re), (uintptr_t) (src2Re), AVX_LEN_BYTES) &&
        areAligned3((uintptr_t) (src1Im), (uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_load_ps((float *) (src1Re) + i);
            v8sf src1Re_tmp2 = _mm256_load_ps((float *) (src1Re) + i + AVX_LEN_FLOAT);
            v8sf src1Im_tmp = _mm256_load_ps((float *) (src1Im) + i);
            v8sf src1Im_tmp2 = _mm256_load_ps((float *) (src1Im) + i + AVX_LEN_FLOAT);
            v8sf src2Re_tmp = _mm256_load_ps((float *) (src2Re) + i);
            v8sf src2Re_tmp2 = _mm256_load_ps((float *) (src2Re) + i + AVX_LEN_FLOAT);
            v8sf src2Im_tmp = _mm256_load_ps((float *) (src2Im) + i);
            v8sf src2Im_tmp2 = _mm256_load_ps((float *) (src2Im) + i + AVX_LEN_FLOAT);

            v8sf c2 = _mm256_mul_ps(src2Re_tmp, src2Re_tmp);
            v8sf c2d2 = _mm256_fmadd_ps_custom(src2Im_tmp, src2Im_tmp, c2);
            v8sf c2_ = _mm256_mul_ps(src2Re_tmp2, src2Re_tmp2);
            v8sf c2d2_ = _mm256_fmadd_ps_custom(src2Im_tmp2, src2Im_tmp2, c2_);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);     // ac
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);     // bc
            v8sf ac2 = _mm256_mul_ps(src1Re_tmp2, src2Re_tmp2);  // ac
            v8sf bc2 = _mm256_mul_ps(src1Im_tmp2, src2Re_tmp2);  // bc

            v8sf dstRe_tmp = _mm256_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac);
            v8sf dstRe_tmp2 = _mm256_fmadd_ps_custom(src1Im_tmp2, src2Im_tmp2, ac2);
            v8sf dstIm_tmp = _mm256_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc);
            v8sf dstIm_tmp2 = _mm256_fnmadd_ps_custom(src1Re_tmp2, src2Im_tmp2, bc2);

            dstRe_tmp = _mm256_div_ps(dstRe_tmp, c2d2);
            dstIm_tmp = _mm256_div_ps(dstIm_tmp, c2d2);
            dstRe_tmp2 = _mm256_div_ps(dstRe_tmp2, c2d2_);
            dstIm_tmp2 = _mm256_div_ps(dstIm_tmp2, c2d2_);

            _mm256_store_ps((float *) (dstRe) + i, dstRe_tmp);
            _mm256_store_ps((float *) (dstIm) + i, dstIm_tmp);
            _mm256_store_ps((float *) (dstRe) + i + AVX_LEN_FLOAT, dstRe_tmp2);
            _mm256_store_ps((float *) (dstIm) + i + AVX_LEN_FLOAT, dstIm_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_loadu_ps((float *) (src1Re) + i);
            v8sf src1Re_tmp2 = _mm256_loadu_ps((float *) (src1Re) + i + AVX_LEN_FLOAT);
            v8sf src1Im_tmp = _mm256_loadu_ps((float *) (src1Im) + i);
            v8sf src1Im_tmp2 = _mm256_loadu_ps((float *) (src1Im) + i + AVX_LEN_FLOAT);
            v8sf src2Re_tmp = _mm256_loadu_ps((float *) (src2Re) + i);
            v8sf src2Re_tmp2 = _mm256_loadu_ps((float *) (src2Re) + i + AVX_LEN_FLOAT);
            v8sf src2Im_tmp = _mm256_loadu_ps((float *) (src2Im) + i);
            v8sf src2Im_tmp2 = _mm256_loadu_ps((float *) (src2Im) + i + AVX_LEN_FLOAT);

            v8sf c2 = _mm256_mul_ps(src2Re_tmp, src2Re_tmp);
            v8sf c2d2 = _mm256_fmadd_ps_custom(src2Im_tmp, src2Im_tmp, c2);
            v8sf c2_ = _mm256_mul_ps(src2Re_tmp2, src2Re_tmp2);
            v8sf c2d2_ = _mm256_fmadd_ps_custom(src2Im_tmp2, src2Im_tmp2, c2_);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);     // ac
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);     // bc
            v8sf ac2 = _mm256_mul_ps(src1Re_tmp2, src2Re_tmp2);  // ac
            v8sf bc2 = _mm256_mul_ps(src1Im_tmp2, src2Re_tmp2);  // bc

            v8sf dstRe_tmp = _mm256_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac);
            v8sf dstRe_tmp2 = _mm256_fmadd_ps_custom(src1Im_tmp2, src2Im_tmp2, ac2);
            v8sf dstIm_tmp = _mm256_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc);
            v8sf dstIm_tmp2 = _mm256_fnmadd_ps_custom(src1Re_tmp2, src2Im_tmp2, bc2);

            dstRe_tmp = _mm256_div_ps(dstRe_tmp, c2d2);
            dstIm_tmp = _mm256_div_ps(dstIm_tmp, c2d2);
            dstRe_tmp2 = _mm256_div_ps(dstRe_tmp2, c2d2_);
            dstIm_tmp2 = _mm256_div_ps(dstIm_tmp2, c2d2_);

            _mm256_storeu_ps((float *) (dstRe) + i, dstRe_tmp);
            _mm256_storeu_ps((float *) (dstIm) + i, dstIm_tmp);
            _mm256_storeu_ps((float *) (dstRe) + i + AVX_LEN_FLOAT, dstRe_tmp2);
            _mm256_storeu_ps((float *) (dstIm) + i + AVX_LEN_FLOAT, dstIm_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2Re[i] * src2Re[i] + src2Im[i] * src2Im[i];
        dstRe[i] = (src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i])) / c2d2;
        dstIm[i] = (-src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i])) / c2d2;
    }
}

static inline void cplxvecmul256f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX_LEN_FLOAT;   // stop_len << 2;

    int i;
    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_load_ps((float *) (src1) + i);                        // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_load_ps((float *) (src2) + i);                        // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            v8sf tmp2 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v8sf tmp3 = _mm256_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(tmp2, tmp3);
            out = _mm256_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm256_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_loadu_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_loadu_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            v8sf tmp2 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v8sf tmp3 = _mm256_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v8sf out = _mm256_mul_ps(tmp2, tmp3);
            out = _mm256_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm256_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxvecmul256f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);
    stop_len = stop_len * AVX_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_load_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_load_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_load_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_load_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            // v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            // v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_store_ps(dstRe + i, _mm256_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  // ac - bd
            _mm256_store_ps(dstIm + i, _mm256_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    } else {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_loadu_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_loadu_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_loadu_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_loadu_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            // v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            // v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_storeu_ps(dstRe + i, _mm256_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  // ac - bd
            _mm256_storeu_ps(dstIm + i, _mm256_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = (src1Re[i] * src2Re[i]) - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i]);
    }
}

static inline void cplxconjvecmul256f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX_LEN_FLOAT;   // stop_len << 2;

    int i;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_load_ps((float *) (src1) + i);                        // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_load_ps((float *) (src2) + i);                        // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            v8sf tmp2 = _mm256_mul_ps(tmp1, src2_tmp);                                   // a1d1,a1c1,a0d0,a0c0
            v8sf tmp3 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v8sf tmp4 = _mm256_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0

#ifndef FMA
            v8sf out = _mm256_mul_ps(tmp3, tmp4);
            out = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_conj_mask, tmp2, out);
#else
            v8sf out = _mm256_fmsubadd_ps(tmp3, tmp4, tmp2);
#endif
            _mm256_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1_tmp = _mm256_loadu_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v8sf src2_tmp = _mm256_loadu_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v8sf tmp1 = _mm256_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            v8sf tmp2 = _mm256_mul_ps(tmp1, src2_tmp);                                   // a1d1,a1c1,a0d0,a0c0
            v8sf tmp3 = _mm256_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v8sf tmp4 = _mm256_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0

#ifndef FMA
            v8sf out = _mm256_mul_ps(tmp3, tmp4);
            out = _mm256_fmadd_ps_custom(*(v8sf *) _ps256_conj_mask, tmp2, out);
#else
            v8sf out = _mm256_fmsubadd_ps(tmp3, tmp4, tmp2);
#endif

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
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_load_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_load_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_load_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_load_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            // v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            // v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_store_ps(dstRe + i, _mm256_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   // ac + bd
            _mm256_store_ps(dstIm + i, _mm256_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    } else {
        for (i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src1Re_tmp = _mm256_loadu_ps((float *) (src1Re) + i);
            v8sf src1Im_tmp = _mm256_loadu_ps((float *) (src1Im) + i);
            v8sf src2Re_tmp = _mm256_loadu_ps((float *) (src2Re) + i);
            v8sf src2Im_tmp = _mm256_loadu_ps((float *) (src2Im) + i);
            v8sf ac = _mm256_mul_ps(src1Re_tmp, src2Re_tmp);
            // v8sf bd = _mm256_mul_ps(src1Im_tmp, src2Im_tmp);
            // v8sf ad = _mm256_mul_ps(src1Re_tmp, src2Im_tmp);
            v8sf bc = _mm256_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm256_storeu_ps(dstRe + i, _mm256_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   // ac + bd
            _mm256_storeu_ps(dstIm + i, _mm256_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + src1Im[i] * src2Im[i];
        dstIm[i] = src2Re[i] * src1Im[i] - src1Re[i] * src2Im[i];
    }
}

// prefer using cplxconjvecmulXf if you also need to do a multiply
static inline void cplxconj256f(complex32_t *src, complex32_t *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len *= 2 * AVX_LEN_FLOAT;             // stop_len << 2;

    __attribute__((aligned(AVX_LEN_BYTES))) int32_t conj_mask[AVX_LEN_FLOAT] = {
        (int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000,
        (int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000};
    int i;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps((float *) (src) + i);
            v8sf src_tmp2 = _mm256_load_ps((float *) (src) + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_xor_ps(src_tmp, *(v8sf *) &conj_mask);
            v8sf dst_tmp2 = _mm256_xor_ps(src_tmp2, *(v8sf *) &conj_mask);
            _mm256_store_ps((float *) (dst) + i, dst_tmp);
            _mm256_store_ps((float *) (dst) + i + AVX_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps((float *) (src) + i);
            v8sf src_tmp2 = _mm256_loadu_ps((float *) (src) + i + AVX_LEN_FLOAT);
            v8sf dst_tmp = _mm256_xor_ps(src_tmp, *(v8sf *) &conj_mask);
            v8sf dst_tmp2 = _mm256_xor_ps(src_tmp2, *(v8sf *) &conj_mask);
            _mm256_storeu_ps((float *) (dst) + i, dst_tmp);
            _mm256_storeu_ps((float *) (dst) + i + AVX_LEN_FLOAT, dst_tmp2);
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf tmp = _mm256_add_ps(*(v8sf *) _ps256_1, exp256_ps_alternate(_mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, src_tmp)));
            _mm256_store_ps(dst + i, _mm256_div_ps(*(v8sf *) _ps256_1, tmp));  //)_mm256_rcp_ps(tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf tmp = _mm256_add_ps(*(v8sf *) _ps256_1, exp256_ps_alternate(_mm256_xor_ps(*(v8sf *) _ps256_neg_sign_mask, src_tmp)));
            _mm256_storeu_ps(dst + i, _mm256_div_ps(*(v8sf *) _ps256_1, tmp));  //)_mm256_rcp_ps(tmp));
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

// to be improved
static inline void softmax256f(float *src, float *dst, int len)
{
    int stop_len = len / (AVX_LEN_FLOAT);
    stop_len *= (AVX_LEN_FLOAT);

    __attribute__((aligned(AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f,
                                                                               0.0f, 0.0f, 0.0f, 0.0f};
    float acc = 0.0f;

    v8sf vec_acc1 = _mm256_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
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

static inline void pol2cart2D256f(float *r, float *theta, float *x, float *y, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf r_tmp = _mm256_load_ps(r + i);
            v8sf theta_tmp = _mm256_load_ps(theta + i);
            v8sf sin_tmp;
            v8sf cos_tmp;
            sincos256_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v8sf x_tmp = _mm256_mul_ps(r_tmp, cos_tmp);
            v8sf y_tmp = _mm256_mul_ps(r_tmp, sin_tmp);
            _mm256_store_ps(x + i, x_tmp);
            _mm256_store_ps(y + i, y_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf r_tmp = _mm256_loadu_ps(r + i);
            v8sf theta_tmp = _mm256_loadu_ps(theta + i);
            v8sf sin_tmp;
            v8sf cos_tmp;
            sincos256_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v8sf x_tmp = _mm256_mul_ps(r_tmp, cos_tmp);
            v8sf y_tmp = _mm256_mul_ps(r_tmp, sin_tmp);
            _mm256_storeu_ps(x + i, x_tmp);
            _mm256_storeu_ps(y + i, y_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        float sin_tmp, cos_tmp;
        mysincosf(theta[i], &sin_tmp, &cos_tmp);
        x[i] = r[i] * cos_tmp;
        y[i] = r[i] * sin_tmp;
    }
}

static inline void cart2pol2D256f(float *x, float *y, float *r, float *theta, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), AVX_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf x_tmp = _mm256_load_ps(x + i);
            v8sf y_tmp = _mm256_load_ps(y + i);
            v8sf y_square = _mm256_mul_ps(y_tmp, y_tmp);
            v8sf r_tmp = _mm256_fmadd_ps_custom(x_tmp, x_tmp, y_square);
            r_tmp = _mm256_sqrt_ps(r_tmp);
            v8sf theta_tmp = atan2256f_ps(y_tmp, x_tmp);
            _mm256_store_ps(r + i, r_tmp);
            _mm256_store_ps(theta + i, theta_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf x_tmp = _mm256_loadu_ps(x + i);
            v8sf y_tmp = _mm256_loadu_ps(y + i);
            v8sf y_square = _mm256_mul_ps(y_tmp, y_tmp);
            v8sf r_tmp = _mm256_fmadd_ps_custom(x_tmp, x_tmp, y_square);
            r_tmp = _mm256_sqrt_ps(r_tmp);
            v8sf theta_tmp = atan2256f_ps(y_tmp, x_tmp);
            _mm256_storeu_ps(r + i, r_tmp);
            _mm256_storeu_ps(theta + i, theta_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        r[i] = sqrtf(x[i] * x[i] + (y[i] * y[i]));
        theta[i] = atan2f(y[i], x[i]);
    }
}

static inline void modf256f(float *src, float *integer, float *remainder, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src), (uintptr_t) (integer), (uintptr_t) (remainder), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sf integer_tmp = _mm256_round_ps(src_tmp, ROUNDTOZERO);
            v8sf integer_tmp2 = _mm256_round_ps(src_tmp2, ROUNDTOZERO);
            v8sf remainder_tmp = _mm256_sub_ps(src_tmp, integer_tmp);
            v8sf remainder_tmp2 = _mm256_sub_ps(src_tmp2, integer_tmp2);
            _mm256_store_ps(integer + i, integer_tmp);
            _mm256_store_ps(integer + i + AVX_LEN_FLOAT, integer_tmp2);
            _mm256_store_ps(remainder + i, remainder_tmp);
            _mm256_store_ps(remainder + i + AVX_LEN_FLOAT, remainder_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sf integer_tmp = _mm256_round_ps(src_tmp, ROUNDTOZERO);
            v8sf integer_tmp2 = _mm256_round_ps(src_tmp2, ROUNDTOZERO);
            v8sf remainder_tmp = _mm256_sub_ps(src_tmp, integer_tmp);
            v8sf remainder_tmp2 = _mm256_sub_ps(src_tmp2, integer_tmp2);
            _mm256_storeu_ps(integer + i, integer_tmp);
            _mm256_storeu_ps(integer + i + AVX_LEN_FLOAT, integer_tmp2);
            _mm256_storeu_ps(remainder + i, remainder_tmp);
            _mm256_storeu_ps(remainder + i + AVX_LEN_FLOAT, remainder_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        remainder[i] = modff(src[i], &(integer[i]));
    }
}
