/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#include "immintrin.h"

static inline v16sf log10512_ps(v16sf x)
{
    v16si imm0;
    v16sf one = *(v16sf *) _ps512_1;

    v16sf invalid_mask = (v16sf) _mm512_movm_epi32(_mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OS));
    x = _mm512_max_ps(x, *(v16sf *) _ps512_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);

    /* keep only the fractional part */
    x = _mm512_and_ps(x, *(v16sf *) _ps512_inv_mant_mask);
    x = _mm512_or_ps(x, *(v16sf *) _ps512_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm512_sub_epi32(imm0, *(v16si *) _pi32_512_0x7f);
    v16sf e = _mm512_cvtepi32_ps(imm0);

    e = _mm512_add_ps(e, one);

    v16sf mask = (v16sf) _mm512_movm_epi32(_mm512_cmp_ps_mask(x, *(v16sf *) _ps512_cephes_SQRTHF, _CMP_LT_OS));

    v16sf tmp = _mm512_and_ps(x, mask);
    x = _mm512_sub_ps(x, one);
    e = _mm512_sub_ps(e, _mm512_and_ps(one, mask));
    x = _mm512_add_ps(x, tmp);

    v16sf z = _mm512_mul_ps(x, x);

    v16sf y = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_cephes_log_p0, x, *(v16sf *) _ps512_cephes_log_p1);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p2);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p3);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p4);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p5);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p6);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p7);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p8);
    y = _mm512_mul_ps(y, x);
    y = _mm512_mul_ps(y, z);

    y = _mm512_fnmadd_ps_custom(z, *(v16sf *) _ps512_0p5, y);

    // Could it be improved with more parallelism or would it worsen precision?
    tmp = _mm512_add_ps(x, y);
    z = _mm512_mul_ps(tmp, *(v16sf *) _ps512_cephes_L10EB);
    z = _mm512_fmadd_ps_custom(y, *(v16sf *) _ps512_cephes_L10EA, z);
    z = _mm512_fmadd_ps_custom(x, *(v16sf *) _ps512_cephes_L10EA, z);
    z = _mm512_fmadd_ps_custom(e, *(v16sf *) _ps512_cephes_L102B, z);
    x = _mm512_fmadd_ps_custom(e, *(v16sf *) _ps512_cephes_L102A, z);

    x = _mm512_or_ps(x, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline v16sf log2512_ps(v16sf x)
{
    v16si imm0;
    v16sf one = *(v16sf *) _ps512_1;

    v16sf invalid_mask = (v16sf) _mm512_movm_epi32(_mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OS));
    x = _mm512_max_ps(x, *(v16sf *) _ps512_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);

    /* keep only the fractional part */
    x = _mm512_and_ps(x, *(v16sf *) _ps512_inv_mant_mask);
    x = _mm512_or_ps(x, *(v16sf *) _ps512_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm512_sub_epi32(imm0, *(v16si *) _pi32_512_0x7f);
    v16sf e = _mm512_cvtepi32_ps(imm0);

    e = _mm512_add_ps(e, one);

    v16sf mask = (v16sf) _mm512_movm_epi32(_mm512_cmp_ps_mask(x, *(v16sf *) _ps512_cephes_SQRTHF, _CMP_LT_OS));

    v16sf tmp = _mm512_and_ps(x, mask);
    x = _mm512_sub_ps(x, one);
    e = _mm512_sub_ps(e, _mm512_and_ps(one, mask));
    x = _mm512_add_ps(x, tmp);

    v16sf z = _mm512_mul_ps(x, x);

    v16sf y = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_cephes_log_p0, x, *(v16sf *) _ps512_cephes_log_p1);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p2);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p3);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p4);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p5);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p6);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p7);
    y = _mm512_fmadd_ps_custom(y, x, *(v16sf *) _ps512_cephes_log_p8);
    y = _mm512_mul_ps(y, x);
    y = _mm512_mul_ps(y, z);

    y = _mm512_fnmadd_ps_custom(z, *(v16sf *) _ps512_0p5, y);

    // Could it be improved with more parallelism or would it worsen precision?
    tmp = _mm512_add_ps(y, x);
    z = _mm512_mul_ps(y, *(v16sf *) _ps512_cephes_LOG2EA);
    z = _mm512_fmadd_ps_custom(x, *(v16sf *) _ps512_cephes_LOG2EA, z);
    z = _mm512_add_ps(z, tmp);
    x = _mm512_add_ps(z, e);

    x = _mm512_or_ps(x, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline void log10_512f(float *src, float *dst, int len)
{
    const v16sf invln10f = _mm512_set1_ps((float) INVLN10);  //_mm512_broadcast_ss(&invln10f_mask);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

static inline void log10_512f_precise(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log10512_ps(_mm512_load_ps(src + i));
            _mm512_store_ps(dst + i, src_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log10512_ps(_mm512_loadu_ps(src + i));
            _mm512_storeu_ps(dst + i, src_tmp);
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
        dst[i] = log2f(src[i]);
    }
}

static inline void log2_512f_precise(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log2512_ps(_mm512_load_ps(src + i));
            _mm512_store_ps(dst + i, src_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = log2512_ps(_mm512_loadu_ps(src + i));
            _mm512_storeu_ps(dst + i, src_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline void ln_512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

static inline v16sf power_of_two512f(v16si b)
{
    return _mm512_cvtepi32_ps(_mm512_sllv_epi32(*(v16si *) _pi32_512_1, b));
}

static inline v16sf cbrt512f_ps(v16sf xx)
{
    v16sf e, rem;
    //__mmask16 sign;
    v16sf x, z;
    v16sf sign;

    x = xx;
    // sign = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GT_OS);
    sign = _mm512_and_ps(xx, *(v16sf *) _ps512_sign_mask);

    x = _mm512_and_ps(x, *(v16sf *) _ps512_pos_sign_mask);

    z = x;
    /* extract power of 2, leaving
     * mantissa between 0.5 and 1
     */
    // x = frexpf(x, &e);
    // solve problem for zero
    v16si emm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);
    x = _mm512_and_ps(x, *(v16sf *) _ps512_inv_mant_mask);
    x = _mm512_or_ps(x, *(v16sf *) _ps512_0p5);
    emm0 = _mm512_sub_epi32(emm0, *(v16si *) _pi32_512_0x7e);
    e = _mm512_cvtepi32_ps(emm0);

    /* Approximate cube root of number between .5 and 1,
     * peak relative error = 9.2e-6
     */
    v16sf tmp;
    tmp = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_CBRTF_P0, x, *(v16sf *) _ps512_CBRTF_P1);
    tmp = _mm512_fmadd_ps_custom(x, tmp, *(v16sf *) _ps512_CBRTF_P2);
    tmp = _mm512_fmadd_ps_custom(x, tmp, *(v16sf *) _ps512_CBRTF_P3);
    x = _mm512_fmadd_ps_custom(x, tmp, *(v16sf *) _ps512_CBRTF_P4);

    /* exponent divided by 3 */
    __mmask16 e_sign = _mm512_cmp_ps_mask(e, _mm512_setzero_ps(), _CMP_GE_OS);
    e = _mm512_and_ps(e, *(v16sf *) _ps512_pos_sign_mask);

    rem = e;
    e = _mm512_mul_ps(e, *(v16sf *) _ps512_0p3);
    v16sf e_tmp = _mm512_mul_ps(*(v16sf *) _ps512_3, _mm512_roundscale_ps(e, ROUNDTOZERO));
    rem = _mm512_sub_ps(rem, e_tmp);

    v16sf mul1, mul2;
    v16sf mul_cst1 = _mm512_mask_blend_ps(e_sign, *(v16sf *) _ps512_cephes_invCBRT2, *(v16sf *) _ps512_cephes_CBRT2);
    v16sf mul_cst2 = _mm512_mask_blend_ps(e_sign, *(v16sf *) _ps512_cephes_invCBRT4, *(v16sf *) _ps512_cephes_CBRT4);

    v16si remi = _mm512_cvtps_epi32(rem);  // rem integer
    __mmask16 rem1 = _mm512_cmpeq_epi32_mask(remi, _mm512_set1_epi32(1));
    __mmask16 rem2 = _mm512_cmpeq_epi32_mask(remi, _mm512_set1_epi32(2));

    x = _mm512_mask_mul_ps(x, rem1, x, mul_cst1);  // rem==1
    x = _mm512_mask_mul_ps(x, rem2, x, mul_cst2);  // rem==2

    /* multiply by power of 2 */
    //  x = ldexpf(x, e);
    v16sf cst = power_of_two512f(_mm512_cvtps_epi32(e));
    // blend sign of e
    tmp = _mm512_div_ps(x, cst);
    x = _mm512_mask_mul_ps(tmp, e_sign, x, cst);

    /* Newton iteration */
    // x -= (x - (z / (x * x))) * 0.333333333333;
    v16sf tmp2 = _mm512_mul_ps(x, x);
    tmp2 = _mm512_div_ps(z, tmp2);
    tmp2 = _mm512_sub_ps(x, tmp2);
    tmp2 = _mm512_mul_ps(tmp2, *(v16sf *) _ps512_0p3);
    x = _mm512_sub_ps(x, tmp2);

    // x = _mm512_mask_blend_ps(sign, _mm512_mul_ps(x, *(v16sf *) _ps512_min1), x);
    x = _mm512_xor_ps(x, sign);

    return x;
}

static inline void cbrt512f(float *src, float *dst, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);
    stop_len *= (AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf x = _mm512_load_ps(src + i);
            v16sf dst_tmp = cbrt512f_ps(x);
            _mm512_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf x = _mm512_loadu_ps(src + i);
            v16sf dst_tmp = cbrt512f_ps(x);
            _mm512_storeu_ps(dst + i, dst_tmp);
        }
    }
    for (int i = stop_len; i < len; i++) {
        dst[i] = cbrtf(src[i]);
    }
}

static inline void fabs512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, src_tmp);
            v16sf dst_tmp2 = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, src_tmp2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, src_tmp);
            v16sf dst_tmp2 = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, src_tmp2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void set512f(float *dst, float value, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zero512f(float *dst, int len)
{
    const v16sf tmp = _mm512_setzero_ps();

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 0.0f;
    }
}


static inline void copy512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            _mm512_store_ps(dst + i, src_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, src_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            _mm512_storeu_ps(dst + i, src_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, src_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(src1 + i);
            v16sf b = _mm512_load_ps(src2 + i);
            v16sf a2 = _mm512_load_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_load_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_add_ps(a, b);
            v16sf dst_tmp2 = _mm512_add_ps(a2, b2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(src1 + i);
            v16sf b = _mm512_loadu_ps(src2 + i);
            v16sf a2 = _mm512_loadu_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_loadu_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_add_ps(a, b);
            v16sf dst_tmp2 = _mm512_add_ps(a2, b2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}


static inline void mul512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(src1 + i);
            v16sf b = _mm512_load_ps(src2 + i);
            v16sf a2 = _mm512_load_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_load_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_mul_ps(a, b);
            v16sf dst_tmp2 = _mm512_mul_ps(a2, b2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(src1 + i);
            v16sf b = _mm512_loadu_ps(src2 + i);
            v16sf a2 = _mm512_loadu_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_loadu_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_mul_ps(a, b);
            v16sf dst_tmp2 = _mm512_mul_ps(a2, b2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void sub512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(src1 + i);
            v16sf b = _mm512_load_ps(src2 + i);
            v16sf a2 = _mm512_load_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_load_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_sub_ps(a, b);
            v16sf dst_tmp2 = _mm512_sub_ps(a2, b2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(src1 + i);
            v16sf b = _mm512_loadu_ps(src2 + i);
            v16sf a2 = _mm512_loadu_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_loadu_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_sub_ps(a, b);
            v16sf dst_tmp2 = _mm512_sub_ps(a2, b2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}


static inline void addc512f(float *src, float value, float *dst, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(src + i);
            v16sf a2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_add_ps(a, tmp);
            v16sf dst_tmp2 = _mm512_add_ps(a2, tmp);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(src + i);
            v16sf a2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_add_ps(a, tmp);
            v16sf dst_tmp2 = _mm512_add_ps(a2, tmp);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulc512f(float *src, float value, float *dst, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp1 = _mm512_mul_ps(tmp, src_tmp1);
            v16sf dst_tmp2 = _mm512_mul_ps(tmp, src_tmp2);
            _mm512_store_ps(dst + i, dst_tmp1);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp1 = _mm512_mul_ps(tmp, src_tmp1);
            v16sf dst_tmp2 = _mm512_mul_ps(tmp, src_tmp2);
            _mm512_storeu_ps(dst + i, dst_tmp1);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladd512f(float *_a, float *_b, float *_c, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (_b), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (_c), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_load_ps(_a + i);
            v16sf b = _mm512_load_ps(_b + i);
            v16sf c = _mm512_load_ps(_c + i);
            v16sf a2 = _mm512_load_ps(_a + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_load_ps(_b + i + AVX512_LEN_FLOAT);
            v16sf c2 = _mm512_load_ps(_c + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_fmadd_ps_custom(a, b, c);
            v16sf dst_tmp2 = _mm512_fmadd_ps_custom(a2, b2, c2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf a = _mm512_loadu_ps(_a + i);
            v16sf b = _mm512_loadu_ps(_b + i);
            v16sf c = _mm512_loadu_ps(_c + i);
            v16sf a2 = _mm512_loadu_ps(_a + i + AVX512_LEN_FLOAT);
            v16sf b2 = _mm512_loadu_ps(_b + i + AVX512_LEN_FLOAT);
            v16sf c2 = _mm512_loadu_ps(_c + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_fmadd_ps_custom(a, b, c);
            v16sf dst_tmp2 = _mm512_fmadd_ps_custom(a2, b2, c2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (_a[i] * _b[i]) + _c[i];
    }
}

static inline void mulcadd512f(float *_a, float _b, float *_c, float *dst, int len)
{
    v16sf b = _mm512_set1_ps(_b);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_c), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (_a), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (_a), (uintptr_t) (_b), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    v16sf slope32_vec = _mm512_set1_ps(32.0f * slope);
    v16sf curVal = _mm512_add_ps(_mm512_set1_ps(offset), coef);
    v16sf curVal2 = _mm512_add_ps(_mm512_set1_ps(offset), coef);
    curVal2 = _mm512_add_ps(curVal2, _mm512_set1_ps(16.0f * slope));

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (len >= AVX512_LEN_FLOAT) {
        if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
            _mm512_store_ps(dst + 0, curVal);
            _mm512_store_ps(dst + AVX512_LEN_FLOAT, curVal2);
        } else {
            _mm512_storeu_ps(dst + 0, curVal);
            _mm512_storeu_ps(dst + AVX512_LEN_FLOAT, curVal2);
        }

        if (isAligned((uintptr_t) (dst), AVX512_LEN_BYTES)) {
            for (int i = 2 * AVX512_LEN_FLOAT; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
                curVal = _mm512_add_ps(curVal, slope32_vec);
                _mm512_store_ps(dst + i, curVal);
                curVal2 = _mm512_add_ps(curVal2, slope32_vec);
                _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, curVal2);
            }
        } else {
            for (int i = 2 * AVX512_LEN_FLOAT; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
                curVal = _mm512_add_ps(curVal, slope32_vec);
                _mm512_storeu_ps(dst + i, curVal);
                curVal2 = _mm512_add_ps(curVal2, slope32_vec);
                _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, curVal2);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

static inline void convertInt16ToFloat32_512(int16_t *src, float *dst, int len, int scale_factor)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v16sf scale_fact_vec = _mm512_set1_ps(scale_fact_mult);

    v16si idx_lo = _mm512_set_epi16(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10,
                                    9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    v16si idx_hi = _mm512_set_epi16(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
                                    23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16);
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16si vec = _mm512_load_si512((__m512i *) (src + i));
            v16si low = _mm512_permutexvar_epi16(idx_lo, vec);   // low 1 1 2 2 3 3 .. 15 15
            v16si high = _mm512_permutexvar_epi16(idx_hi, vec);  // high 16 16 17 17 .. 31 31
            low = _mm512_srai_epi32(low, 0x10);                  // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm512_srai_epi32(high, 0x10);                // make high 5 -1 6 -1 7 -1 8 -1

            // convert the vector to float and scale it
            v16sf floatlo = _mm512_mul_ps(_mm512_cvtepi32_ps(low), scale_fact_vec);
            v16sf floathi = _mm512_mul_ps(_mm512_cvtepi32_ps(high), scale_fact_vec);

            _mm512_store_ps(dst + i, floatlo);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, floathi);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16si vec = _mm512_loadu_si512((__m512i *) (src + i));
            v16si low = _mm512_permutexvar_epi16(idx_lo, vec);   // low 1 1 2 2 3 3 .. 15 15
            v16si high = _mm512_permutexvar_epi16(idx_hi, vec);  // high 16 16 17 17 .. 31 31
            low = _mm512_srai_epi32(low, 0x10);                  // make low 1 -1 2 -1 3 -1 4 -4
            high = _mm512_srai_epi32(high, 0x10);                // make high 5 -1 6 -1 7 -1 8 -1

            // convert the vector to float and scale it
            v16sf floatlo = _mm512_mul_ps(_mm512_cvtepi32_ps(low), scale_fact_vec);
            v16sf floathi = _mm512_mul_ps(_mm512_cvtepi32_ps(high), scale_fact_vec);

            _mm512_storeu_ps(dst + i, floatlo);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, floathi);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}

static inline void convertFloat32ToU8_512(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * AVX512_LEN_FLOAT);
    stop_len *= (4 * AVX512_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v16sf scale_fact_vec = _mm512_set1_ps(scale_fact_mult);

    v16si idx = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_tmp3 = _mm512_load_ps(src + i + 2 * AVX512_LEN_FLOAT);
            v16sf src_tmp4 = _mm512_load_ps(src + i + 3 * AVX512_LEN_FLOAT);
            v16sf tmp1 = _mm512_mul_ps(src_tmp1, scale_fact_vec);
            v16sf tmp2 = _mm512_mul_ps(src_tmp2, scale_fact_vec);
            v16sf tmp3 = _mm512_mul_ps(src_tmp3, scale_fact_vec);
            v16sf tmp4 = _mm512_mul_ps(src_tmp4, scale_fact_vec);
            v16si tmp1_int = _mm512_cvtps_epi32(tmp1);
            v16si tmp2_int = _mm512_cvtps_epi32(tmp2);
            v16si tmp3_int = _mm512_cvtps_epi32(tmp3);
            v16si tmp4_int = _mm512_cvtps_epi32(tmp4);
            v16si tmp5 = _mm512_packs_epi32(tmp1_int, tmp2_int);
            v16si tmp6 = _mm512_packs_epi32(tmp3_int, tmp4_int);
            v16si tmp7 = _mm512_packus_epi16(tmp5, tmp6);
            tmp7 = _mm512_permutexvar_epi32(idx, tmp7);
            _mm512_store_si512((__m512i *) (dst + i), tmp7);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_tmp3 = _mm512_loadu_ps(src + i + 2 * AVX512_LEN_FLOAT);
            v16sf src_tmp4 = _mm512_loadu_ps(src + i + 3 * AVX512_LEN_FLOAT);
            v16sf tmp1 = _mm512_mul_ps(src_tmp1, scale_fact_vec);
            v16sf tmp2 = _mm512_mul_ps(src_tmp2, scale_fact_vec);
            v16sf tmp3 = _mm512_mul_ps(src_tmp3, scale_fact_vec);
            v16sf tmp4 = _mm512_mul_ps(src_tmp4, scale_fact_vec);
            v16si tmp1_int = _mm512_cvtps_epi32(tmp1);
            v16si tmp2_int = _mm512_cvtps_epi32(tmp2);
            v16si tmp3_int = _mm512_cvtps_epi32(tmp3);
            v16si tmp4_int = _mm512_cvtps_epi32(tmp4);
            v16si tmp5 = _mm512_packs_epi32(tmp1_int, tmp2_int);
            v16si tmp6 = _mm512_packs_epi32(tmp3_int, tmp4_int);
            v16si tmp7 = _mm512_packus_epi16(tmp5, tmp6);
            tmp7 = _mm512_permutexvar_epi32(idx, tmp7);
            _mm512_storeu_si512((__m512i *) (dst + i), tmp7);
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

static inline void convertFloat32ToU16_512(float *src, uint16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v16sf scale_fact_vec = _mm512_set1_ps(scale_fact_mult);

    v16si idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf tmp1 = _mm512_mul_ps(src_tmp1, scale_fact_vec);
            v16sf tmp2 = _mm512_mul_ps(src_tmp2, scale_fact_vec);
            v16si tmp1_int = _mm512_cvtps_epi32(tmp1);
            v16si tmp2_int = _mm512_cvtps_epi32(tmp2);
            v16si tmp5 = _mm512_packus_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm512_permutexvar_epi64(idx, tmp5);
            _mm512_store_si512((__m512i *) (dst + i), tmp5);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf tmp1 = _mm512_mul_ps(src_tmp1, scale_fact_vec);
            v16sf tmp2 = _mm512_mul_ps(src_tmp2, scale_fact_vec);
            v16si tmp1_int = _mm512_cvtps_epi32(tmp1);
            v16si tmp2_int = _mm512_cvtps_epi32(tmp2);
            v16si tmp5 = _mm512_packus_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm512_permutexvar_epi64(idx, tmp5);
            _mm512_storeu_si512((__m512i *) (dst + i), tmp5);
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

static inline void convertFloat32ToI16_512(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v16sf scale_fact_vec = _mm512_set1_ps(scale_fact_mult);

    v16si idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf tmp1 = _mm512_mul_ps(src_tmp1, scale_fact_vec);
            v16sf tmp2 = _mm512_mul_ps(src_tmp2, scale_fact_vec);
            v16si tmp1_int = _mm512_cvtps_epi32(tmp1);
            v16si tmp2_int = _mm512_cvtps_epi32(tmp2);
            v16si tmp5 = _mm512_packs_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm512_permutexvar_epi64(idx, tmp5);
            _mm512_store_si512((__m512i *) (dst + i), tmp5);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp1 = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf tmp1 = _mm512_mul_ps(src_tmp1, scale_fact_vec);
            v16sf tmp2 = _mm512_mul_ps(src_tmp2, scale_fact_vec);
            v16si tmp1_int = _mm512_cvtps_epi32(tmp1);
            v16si tmp2_int = _mm512_cvtps_epi32(tmp2);
            v16si tmp5 = _mm512_packs_epi32(tmp1_int, tmp2_int);
            tmp5 = _mm512_permutexvar_epi64(idx, tmp5);
            _mm512_storeu_si512((__m512i *) (dst + i), tmp5);
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

static inline void cplxtoreal512f(complex32_t *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (4 * AVX512_LEN_FLOAT);
    stop_len *= 4 * AVX512_LEN_FLOAT;

    int j = 0;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 tmp = _mm512_load2_ps((float const *) (src) + i);
            v16sfx2 tmp2 = _mm512_load2_ps((float const *) (src) + i + 2 * AVX512_LEN_FLOAT);
            _mm512_store_ps(dstRe + j, tmp.val[0]);
            _mm512_store_ps(dstIm + j, tmp.val[1]);
            _mm512_store_ps(dstRe + j + AVX512_LEN_FLOAT, tmp2.val[0]);
            _mm512_store_ps(dstIm + j + AVX512_LEN_FLOAT, tmp2.val[1]);

            j += 2 * AVX512_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 tmp = _mm512_load2u_ps((float const *) (src) + i);
            v16sfx2 tmp2 = _mm512_load2u_ps((float const *) (src) + i + 2 * AVX512_LEN_FLOAT);
            _mm512_storeu_ps(dstRe + j, tmp.val[0]);
            _mm512_storeu_ps(dstIm + j, tmp.val[1]);
            _mm512_storeu_ps(dstRe + j + AVX512_LEN_FLOAT, tmp2.val[0]);
            _mm512_storeu_ps(dstIm + j + AVX512_LEN_FLOAT, tmp2.val[1]);

            j += 2 * AVX512_LEN_FLOAT;
        }
    }

    for (int i = j; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
}


static inline void realtocplx512f(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= 2 * AVX512_LEN_FLOAT;

    int j = 0;

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sfx2 vec1, vec2;
            vec1.val[0] = _mm512_load_ps(srcRe + i);
            vec1.val[1] = _mm512_load_ps(srcIm + i);
            vec2.val[0] = _mm512_load_ps(srcRe + i + AVX512_LEN_FLOAT);
            vec2.val[1] = _mm512_load_ps(srcIm + i + AVX512_LEN_FLOAT);
            _mm512_store2_ps((float *) (dst) + j, vec1);
            _mm512_store2_ps((float *) (dst) + j + 2 * AVX512_LEN_FLOAT, vec2);

            j += 4 * AVX512_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sfx2 vec1, vec2;
            vec1.val[0] = _mm512_loadu_ps(srcRe + i);
            vec1.val[1] = _mm512_loadu_ps(srcIm + i);
            vec2.val[0] = _mm512_loadu_ps(srcRe + i + AVX512_LEN_FLOAT);
            vec2.val[1] = _mm512_loadu_ps(srcIm + i + AVX512_LEN_FLOAT);
            _mm512_store2u_ps((float *) (dst) + j, vec1);
            _mm512_store2u_ps((float *) (dst) + j + 2 * AVX512_LEN_FLOAT, vec2);

            j += 4 * AVX512_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
}


static inline void convert512_64f32f(double *src, float *dst, int len)
{
    int stop_len = len / (4 * AVX512_LEN_DOUBLE);
    stop_len *= (4 * AVX512_LEN_DOUBLE);

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            v8sd src_tmp2 = _mm512_load_pd(src + i + AVX512_LEN_DOUBLE);
            v8sd src_tmp3 = _mm512_load_pd(src + i + 2 * AVX512_LEN_DOUBLE);
            v8sd src_tmp4 = _mm512_load_pd(src + i + 3 * AVX512_LEN_DOUBLE);
            v8sf src_lo = _mm512_cvtpd_ps(src_tmp);
            v8sf src_hi = _mm512_cvtpd_ps(src_tmp2);
            v8sf src_lo2 = _mm512_cvtpd_ps(src_tmp3);
            v8sf src_hi2 = _mm512_cvtpd_ps(src_tmp4);
            v16sf dst_tmp, dst_tmp2;
            dst_tmp = _mm512_insertf32x8(dst_tmp, src_lo, 0);
            dst_tmp = _mm512_insertf32x8(dst_tmp, src_hi, 1);
            dst_tmp2 = _mm512_insertf32x8(dst_tmp2, src_lo2, 0);
            dst_tmp2 = _mm512_insertf32x8(dst_tmp2, src_hi2, 1);
            _mm512_store_ps(dst + j, dst_tmp);
            _mm512_store_ps(dst + j + AVX512_LEN_FLOAT, dst_tmp2);
            j += 2 * AVX512_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            v8sd src_tmp2 = _mm512_loadu_pd(src + i + AVX512_LEN_DOUBLE);
            v8sd src_tmp3 = _mm512_loadu_pd(src + i + 2 * AVX512_LEN_DOUBLE);
            v8sd src_tmp4 = _mm512_loadu_pd(src + i + 3 * AVX512_LEN_DOUBLE);
            v8sf src_lo = _mm512_cvtpd_ps(src_tmp);
            v8sf src_hi = _mm512_cvtpd_ps(src_tmp2);
            v8sf src_lo2 = _mm512_cvtpd_ps(src_tmp3);
            v8sf src_hi2 = _mm512_cvtpd_ps(src_tmp4);
            v16sf dst_tmp, dst_tmp2;
            dst_tmp = _mm512_insertf32x8(dst_tmp, src_lo, 0);
            dst_tmp = _mm512_insertf32x8(dst_tmp, src_hi, 1);
            dst_tmp2 = _mm512_insertf32x8(dst_tmp2, src_lo2, 0);
            dst_tmp2 = _mm512_insertf32x8(dst_tmp2, src_hi2, 1);
            _mm512_storeu_ps(dst + j, dst_tmp);
            _mm512_storeu_ps(dst + j + AVX512_LEN_FLOAT, dst_tmp2);
            j += 2 * AVX512_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i];
    }
}

static inline void convert512_32f64f(float *src, double *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);  // load a,b,c,d
            v8sf src_tmp2 = _mm256_load_ps(src + i + AVX_LEN_FLOAT);
            v8sd dst_tmp = _mm512_cvtps_pd(src_tmp);
            v8sd dst_tmp2 = _mm512_cvtps_pd(src_tmp2);
            _mm512_store_pd(dst + i, dst_tmp);  // store the abcd converted in 64bits
            _mm512_store_pd(dst + i + AVX512_LEN_DOUBLE, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);  // load a,b,c,d
            v8sf src_tmp2 = _mm256_loadu_ps(src + i + AVX_LEN_FLOAT);
            v8sd dst_tmp = _mm512_cvtps_pd(src_tmp);
            v8sd dst_tmp2 = _mm512_cvtps_pd(src_tmp2);
            _mm512_storeu_pd(dst + i, dst_tmp);  // store the abcd converted in 64bits
            _mm512_storeu_pd(dst + i + AVX512_LEN_DOUBLE, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (double) src[i];
    }
}

static inline void flip512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);
    v16si flip_idx = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    int mini = ((len - 1) < (2 * AVX512_LEN_FLOAT)) ? (len - 1) : (2 * AVX512_LEN_FLOAT);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst + len - AVX512_LEN_FLOAT), AVX512_LEN_BYTES)) {
        for (int i = 2 * AVX512_LEN_FLOAT; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);  // load a,b,c,d,e,f,g,h
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_tmp_flip = _mm512_permutex2var_ps(src_tmp, flip_idx, src_tmp);
            v16sf src_tmp_flip2 = _mm512_permutex2var_ps(src_tmp2, flip_idx, src_tmp2);
            _mm512_store_ps(dst + len - i - AVX512_LEN_FLOAT, src_tmp_flip);
            _mm512_store_ps(dst + len - i - 2 * AVX512_LEN_FLOAT, src_tmp_flip2);
        }
    } else {
        for (int i = 2 * AVX512_LEN_FLOAT; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);  // load a,b,c,d,e,f,g,h
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_tmp_flip = _mm512_permutex2var_ps(src_tmp, flip_idx, src_tmp);
            v16sf src_tmp_flip2 = _mm512_permutex2var_ps(src_tmp2, flip_idx, src_tmp2);
            _mm512_storeu_ps(dst + len - i - AVX512_LEN_FLOAT, src_tmp_flip);
            _mm512_storeu_ps(dst + len - i - 2 * AVX512_LEN_FLOAT, src_tmp_flip2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void maxevery512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_load_ps(src1 + i);
            v16sf src2_tmp = _mm512_load_ps(src2 + i);
            v16sf src1_tmp2 = _mm512_load_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf src2_tmp2 = _mm512_load_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_max_ps(src1_tmp, src2_tmp);
            v16sf dst_tmp2 = _mm512_max_ps(src1_tmp2, src2_tmp2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_loadu_ps(src1 + i);
            v16sf src2_tmp = _mm512_loadu_ps(src2 + i);
            v16sf src1_tmp2 = _mm512_loadu_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf src2_tmp2 = _mm512_loadu_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_max_ps(src1_tmp, src2_tmp);
            v16sf dst_tmp2 = _mm512_max_ps(src1_tmp2, src2_tmp2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void minevery512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_load_ps(src1 + i);
            v16sf src2_tmp = _mm512_load_ps(src2 + i);
            v16sf src1_tmp2 = _mm512_load_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf src2_tmp2 = _mm512_load_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_min_ps(src1_tmp, src2_tmp);
            v16sf dst_tmp2 = _mm512_min_ps(src1_tmp2, src2_tmp2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_loadu_ps(src1 + i);
            v16sf src2_tmp = _mm512_loadu_ps(src2 + i);
            v16sf src1_tmp2 = _mm512_loadu_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf src2_tmp2 = _mm512_loadu_ps(src2 + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_min_ps(src1_tmp, src2_tmp);
            v16sf dst_tmp2 = _mm512_min_ps(src1_tmp2, src2_tmp2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void minmax512f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = (len - AVX512_LEN_FLOAT) / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    v16sf max_v, min_v, max_v2, min_v2;
    v16sf src_tmp, src_tmp2;

    float min_tmp = src[0];
    float max_tmp = src[0];

    if (len >= AVX512_LEN_FLOAT) {
        if (isAligned((uintptr_t) (src), AVX512_LEN_BYTES)) {
            src_tmp = _mm512_load_ps(src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = AVX512_LEN_FLOAT; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
                src_tmp = _mm512_load_ps(src + i);
                src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
                max_v = _mm512_max_ps(max_v, src_tmp);
                min_v = _mm512_min_ps(min_v, src_tmp);
                max_v2 = _mm512_max_ps(max_v2, src_tmp2);
                min_v2 = _mm512_min_ps(min_v2, src_tmp2);
            }
        } else {
            src_tmp = _mm512_loadu_ps(src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = AVX512_LEN_FLOAT; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
                src_tmp = _mm512_loadu_ps(src + i);
                src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
                max_v = _mm512_max_ps(max_v, src_tmp);
                min_v = _mm512_min_ps(min_v, src_tmp);
                max_v2 = _mm512_max_ps(max_v2, src_tmp2);
                min_v2 = _mm512_min_ps(min_v2, src_tmp2);
            }
        }

        max_v = _mm512_max_ps(max_v, max_v2);
        min_v = _mm512_min_ps(min_v, min_v2);

// With SIMD reduction
#if 1
        v8sf max1 = _mm512_castps512_ps256(max_v);
        v8sf min1 = _mm512_castps512_ps256(min_v);
        v8sf max2 = _mm512_extractf32x8_ps(max_v, 1);
        v8sf min2 = _mm512_extractf32x8_ps(min_v, 1);
        max2 = _mm256_max_ps(max1, max2);
        min2 = _mm256_min_ps(min1, min2);
        v4sf max3 = _mm256_castps256_ps128(max2);
        v4sf min3 = _mm256_castps256_ps128(min2);
        v4sf max4 = _mm256_extractf32x4_ps(max2, 1);
        v4sf min4 = _mm256_extractf32x4_ps(min2, 1);
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
#endif
    }

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

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_min_ps(src_tmp, tmp);
            v16sf dst_tmp2 = _mm512_min_ps(src_tmp2, tmp);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_min_ps(src_tmp, tmp);
            v16sf dst_tmp2 = _mm512_min_ps(src_tmp2, tmp);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

#if 1
static inline void threshold512_gtabs_f(float *src, float *dst, int len, float value)
{
    const v16sf pval = _mm512_set1_ps(value);

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_sign = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_sign_mask);  // extract sign
            v16sf src_sign2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_sign_mask);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);  // take absolute value
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            v16sf dst_tmp = _mm512_min_ps(src_abs, pval);
            v16sf dst_tmp2 = _mm512_min_ps(src_abs2, pval);
            dst_tmp = _mm512_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm512_xor_ps(dst_tmp2, src_sign2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_sign = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_sign_mask);  // extract sign
            v16sf src_sign2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_sign_mask);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);  // take absolute value
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            v16sf dst_tmp = _mm512_min_ps(src_abs, pval);
            v16sf dst_tmp2 = _mm512_min_ps(src_abs2, pval);
            dst_tmp = _mm512_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm512_xor_ps(dst_tmp2, src_sign2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
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
static inline void threshold512_gtabs_f(float *src, float *dst, int len, float value)
{
    const v16sf pval = _mm512_set1_ps(value);
    const v16sf mval = _mm512_set1_ps(-value);

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_ps_mask(src_abs2, src_tmp2, _CMP_EQ_OS);
            __mmask16 gtmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_GT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            __mmask16 gtmask2 = _mm512_cmp_ps_mask(src_abs2, pval, _CMP_GT_OS);
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);  // if A >= 0 value, else -value
            v16sf sval2 = _mm512_mask_blend_ps(eqmask2, mval, pval);
            v16sf dst_tmp = _mm512_mask_blend_ps(gtmask, src_tmp, sval);  // either A or sval (+- value)
            v16sf dst_tmp2 = _mm512_mask_blend_ps(gtmask2, src_tmp2, sval2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_ps_mask(src_abs2, src_tmp2, _CMP_EQ_OS);
            __mmask16 gtmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_GT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            __mmask16 gtmask2 = _mm512_cmp_ps_mask(src_abs2, pval, _CMP_GT_OS);
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);  // if A >= 0 value, else -value
            v16sf sval2 = _mm512_mask_blend_ps(eqmask2, mval, pval);
            v16sf dst_tmp = _mm512_mask_blend_ps(gtmask, src_tmp, sval);  // either A or sval (+- value)
            v16sf dst_tmp2 = _mm512_mask_blend_ps(gtmask2, src_tmp2, sval2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
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

static inline void threshold512_lt_f(float *src, float *dst, int len, float value)
{
    v16sf tmp = _mm512_set1_ps(value);  //_mm512_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_max_ps(src_tmp, tmp);
            v16sf dst_tmp2 = _mm512_max_ps(src_tmp2, tmp);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_max_ps(src_tmp, tmp);
            v16sf dst_tmp2 = _mm512_max_ps(src_tmp2, tmp);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

#if 1
static inline void threshold512_ltabs_f(float *src, float *dst, int len, float value)
{
    const v16sf pval = _mm512_set1_ps(value);

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_sign = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_sign_mask);  // extract sign
            v16sf src_sign2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_sign_mask);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);  // take absolute value
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            v16sf dst_tmp = _mm512_max_ps(src_abs, pval);
            v16sf dst_tmp2 = _mm512_max_ps(src_abs2, pval);
            dst_tmp = _mm512_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm512_xor_ps(dst_tmp2, src_sign2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_sign = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_sign_mask);  // extract sign
            v16sf src_sign2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_sign_mask);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);  // take absolute value
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            v16sf dst_tmp = _mm512_max_ps(src_abs, pval);
            v16sf dst_tmp2 = _mm512_max_ps(src_abs2, pval);
            dst_tmp = _mm512_xor_ps(dst_tmp, src_sign);
            dst_tmp2 = _mm512_xor_ps(dst_tmp2, src_sign2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
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
static inline void threshold512_ltabs_f(float *src, float *dst, int len, float value)
{
    const v16sf pval = _mm512_set1_ps(value);
    const v16sf mval = _mm512_set1_ps(-value);

    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_ps_mask(src_abs2, src_tmp2, _CMP_EQ_OS);
            __mmask16 ltmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_LT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            __mmask16 ltmask2 = _mm512_cmp_ps_mask(src_abs2, pval, _CMP_LT_OS);
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);  // if A >= 0 value, else -value
            v16sf sval2 = _mm512_mask_blend_ps(eqmask2, mval, pval);
            v16sf dst_tmp = _mm512_mask_blend_ps(ltmask, src_tmp, sval);  // either A or sval (+- value)
            v16sf dst_tmp2 = _mm512_mask_blend_ps(ltmask2, src_tmp2, sval2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf src_abs = _mm512_and_ps(src_tmp, *(v16sf *) _ps512_pos_sign_mask);
            v16sf src_abs2 = _mm512_and_ps(src_tmp2, *(v16sf *) _ps512_pos_sign_mask);
            __mmask16 eqmask = _mm512_cmp_ps_mask(src_abs, src_tmp, _CMP_EQ_OS);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            __mmask16 eqmask2 = _mm512_cmp_ps_mask(src_abs2, src_tmp2, _CMP_EQ_OS);
            __mmask16 ltmask = _mm512_cmp_ps_mask(src_abs, pval, _CMP_LT_OS);  // if abs(A) < value => 0xFFFFFFFF, else 0
            __mmask16 ltmask2 = _mm512_cmp_ps_mask(src_abs2, pval, _CMP_LT_OS);
            v16sf sval = _mm512_mask_blend_ps(eqmask, mval, pval);  // if A >= 0 value, else -value
            v16sf sval2 = _mm512_mask_blend_ps(eqmask2, mval, pval);
            v16sf dst_tmp = _mm512_mask_blend_ps(ltmask, src_tmp, sval);  // either A or sval (+- value)
            v16sf dst_tmp2 = _mm512_mask_blend_ps(ltmask2, src_tmp2, sval2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
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

static inline void threshold512_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v16sf ltlevel_v = _mm512_set1_ps(ltlevel);
    const v16sf ltvalue_v = _mm512_set1_ps(ltvalue);
    const v16sf gtlevel_v = _mm512_set1_ps(gtlevel);
    const v16sf gtvalue_v = _mm512_set1_ps(gtvalue);

    int stop_len = len / (2 * AVX512_LEN_BYTES);
    stop_len *= (2 * AVX512_LEN_BYTES);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            __mmask16 lt_mask = _mm512_cmp_ps_mask(src_tmp, ltlevel_v, _CMP_LT_OS);
            __mmask16 gt_mask = _mm512_cmp_ps_mask(src_tmp, gtlevel_v, _CMP_GT_OS);
            v16sf dst_tmp = _mm512_mask_blend_ps(lt_mask, src_tmp, ltvalue_v);
            dst_tmp = _mm512_mask_blend_ps(gt_mask, dst_tmp, gtvalue_v);
            _mm512_store_ps(dst + i, dst_tmp);
            __mmask16 lt_mask2 = _mm512_cmp_ps_mask(src_tmp2, ltlevel_v, _CMP_LT_OS);
            __mmask16 gt_mask2 = _mm512_cmp_ps_mask(src_tmp2, gtlevel_v, _CMP_GT_OS);
            v16sf dst_tmp2 = _mm512_mask_blend_ps(lt_mask2, src_tmp2, ltvalue_v);
            dst_tmp2 = _mm512_mask_blend_ps(gt_mask2, dst_tmp2, gtvalue_v);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            __mmask16 lt_mask = _mm512_cmp_ps_mask(src_tmp, ltlevel_v, _CMP_LT_OS);
            __mmask16 gt_mask = _mm512_cmp_ps_mask(src_tmp, gtlevel_v, _CMP_GT_OS);
            v16sf dst_tmp = _mm512_mask_blend_ps(lt_mask, src_tmp, ltvalue_v);
            dst_tmp = _mm512_mask_blend_ps(gt_mask, dst_tmp, gtvalue_v);
            _mm512_storeu_ps(dst + i, dst_tmp);
            __mmask16 lt_mask2 = _mm512_cmp_ps_mask(src_tmp2, ltlevel_v, _CMP_LT_OS);
            __mmask16 gt_mask2 = _mm512_cmp_ps_mask(src_tmp2, gtlevel_v, _CMP_GT_OS);
            v16sf dst_tmp2 = _mm512_mask_blend_ps(lt_mask2, src_tmp2, ltvalue_v);
            dst_tmp2 = _mm512_mask_blend_ps(gt_mask2, dst_tmp2, gtvalue_v);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), AVX512_LEN_BYTES)) {
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

// e^ix = cos(x) + i*sin(x)
static inline void sincos512f_interleaved(float *src, complex32_t *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sfx2 dst_tmp;
            sincos512_ps(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm512_store2_ps((float *) dst + j, dst_tmp);
            j += 2 * AVX512_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sfx2 dst_tmp;
            sincos512_ps(src_tmp, &(dst_tmp.val[1]), &(dst_tmp.val[0]));
            _mm512_store2u_ps((float *) dst + j, dst_tmp);
            j += 2 * AVX512_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], &(dst[i].im), &(dst[i].re));
    }
}

static inline v16sf acosh512f_ps(v16sf x)
{
    v16sf z, z_first_branch, z_second_branch, tmp, tmp2;
    __mmask16 xsup1500, zinf0p5, xinf1;

    xsup1500 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1500, _CMP_GT_OS);  // return  (logf(x) + LOGE2F)
    xinf1 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1, _CMP_LT_OS);        // return 0

    z = _mm512_sub_ps(x, *(v16sf *) _ps512_1);

    zinf0p5 = _mm512_cmp_ps_mask(z, *(v16sf *) _ps512_0p5, _CMP_LT_OS);  // first and second branch

    tmp2 = log512_ps(x);

    // First Branch (z < 0.5)
    z_first_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ACOSH_P0, z, *(v16sf *) _ps512_ACOSH_P1);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, z, *(v16sf *) _ps512_ACOSH_P2);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, z, *(v16sf *) _ps512_ACOSH_P3);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, z, *(v16sf *) _ps512_ACOSH_P4);
    tmp = _mm512_sqrt_ps(z);

    // Second Branch
    z_second_branch = _mm512_sqrt_ps(_mm512_fmadd_ps_custom(z, x, z));
    z_second_branch = log512_ps(_mm512_add_ps(x, z_second_branch));

    z = _mm512_mask_mul_ps(z_second_branch, zinf0p5, z_first_branch, tmp);
    z = _mm512_mask_add_ps(z, xsup1500, tmp2, *(v16sf *) _ps512_LOGE2F);
    z = _mm512_mask_blend_ps(xinf1, z, _mm512_setzero_ps());

    return z;
}

static inline void acosh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    v16sf x, tmp, tmp2, z, z_first_branch, z_second_branch;
    __mmask16 xxinf0, xsup1500, xinf0p5;

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);
    xsup1500 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1500, _CMP_GT_OS);
    xinf0p5 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_0p5, _CMP_LT_OS);
    xxinf0 = _mm512_cmp_ps_mask(xx, _mm512_setzero_ps(), _CMP_LT_OS);

    tmp = _mm512_mul_ps(x, x);
    tmp2 = log512_ps(x);

    // First Branch (x < 0.5)
    z_first_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ASINH_P0, tmp, *(v16sf *) _ps512_ASINH_P1);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ASINH_P2);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ASINH_P3);
    z_first_branch = _mm512_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, x, x);

    // Second Branch
    z_second_branch = _mm512_sqrt_ps(_mm512_add_ps(tmp, *(v16sf *) _ps512_1));
    z_second_branch = log512_ps(_mm512_add_ps(z_second_branch, x));

    z = _mm512_mask_blend_ps(xinf0p5, z_second_branch, z_first_branch);
    z = _mm512_mask_add_ps(z, xsup1500, tmp2, *(v16sf *) _ps512_LOGE2F);
    z = _mm512_mask_xor_ps(z, xxinf0, *(v16sf *) _ps512_neg_sign_mask, z);

    return z;
}

static inline void asinh512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    // First branch
    tmp = _mm512_mul_ps(x, x);
    z_first_branch = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ATANH_P0, tmp, *(v16sf *) _ps512_ATANH_P1);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ATANH_P2);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ATANH_P3);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, tmp, *(v16sf *) _ps512_ATANH_P4);
    z_first_branch = _mm512_mul_ps(z_first_branch, tmp);
    z_first_branch = _mm512_fmadd_ps_custom(z_first_branch, x, x);

    // Second branch
    tmp = _mm512_sub_ps(*(v16sf *) _ps512_1, x);
#if 0
    //NEEDS AVX512ER tmp2 
    v16sf tmp2 = _mm512_rcp28_ps(tmp);
    tmp = _mm512_fmadd_ps_custom(tmp2,x,tmp2);
#else
    tmp = _mm512_div_ps(_mm512_add_ps(*(v16sf *) _ps512_1, x), tmp);
#endif
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    z_first_branch = _mm512_mask_xor_ps(z_first_branch, xinf0, *(v16sf *) _ps512_neg_sign_mask, z_first_branch);

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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    //__mmask16 sign2;
    __mmask16 suptan3pi8, inftan3pi8suppi8;
    v16sf tmp, tmp2, tmp3;
    v16sf sign;

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);
    // sign2 = _mm512_cmp_ps_mask(xx, _mm512_setzero_ps(), _CMP_LT_OS);  // 0xFFFFFFFF if x < 0.0, sign = -1
    sign = _mm512_and_ps(xx, *(v16sf *) _ps512_sign_mask);

    /* range reduction */
    y = _mm512_setzero_ps();
    suptan3pi8 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_TAN3PI8F, _CMP_GT_OS);  // if( x > tan 3pi/8 )
    x = _mm512_mask_div_ps(x, suptan3pi8, *(v16sf *) _ps512_min1, x);
    y = _mm512_mask_blend_ps(suptan3pi8, y, *(v16sf *) _ps512_PIO2F);


    inftan3pi8suppi8 = _kand_mask64(_mm512_cmp_ps_mask(x, *(v16sf *) _ps512_TAN3PI8F, _CMP_LT_OS), _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_TANPI8F, _CMP_GT_OS));  // if( x > tan 3pi/8 )
    tmp2 = _mm512_add_ps(x, *(v16sf *) _ps512_1);
    tmp3 = _mm512_sub_ps(x, *(v16sf *) _ps512_1);
    x = _mm512_mask_div_ps(x, inftan3pi8suppi8, tmp3, tmp2);
    y = _mm512_mask_blend_ps(inftan3pi8suppi8, y, *(v16sf *) _ps512_PIO4F);

    z = _mm512_mul_ps(x, x);
    tmp = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ATAN_P0, z, *(v16sf *) _ps512_ATAN_P1);
    tmp = _mm512_fmadd_ps_custom(tmp, z, *(v16sf *) _ps512_ATAN_P2);
    tmp = _mm512_fmadd_ps_custom(tmp, z, *(v16sf *) _ps512_ATAN_P3);
    tmp = _mm512_mul_ps(z, tmp);
    tmp = _mm512_fmadd_ps_custom(tmp, x, x);

    y = _mm512_add_ps(y, tmp);
    y = _mm512_xor_ps(y, sign);
    return (y);
}

static inline void atan512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    v16sf z, w, tmp;
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

    tmp = _mm512_div_ps(y, x);
    tmp = atan512f_ps(tmp);
    z = _mm512_mask_blend_ps(specialcase, _mm512_add_ps(w, tmp), z);  // atanf(y/x) if not in special case

    return (z);
}

static inline void atan2512f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

static inline void atan2512f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= 2 * AVX512_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sfx2 src1 = _mm512_load2_ps((float *) (src) + j);
            v16sfx2 src2 = _mm512_load2_ps((float *) (src) + j + 2 * AVX512_LEN_FLOAT);
            _mm512_store_ps(dst + i, atan2512f_ps(src1.val[1], src1.val[0]));
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, atan2512f_ps(src2.val[1], src2.val[0]));
            j += 4 * AVX512_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sfx2 src1 = _mm512_load2u_ps((float *) (src) + j);
            v16sfx2 src2 = _mm512_load2u_ps((float *) (src) + j + 2 * AVX512_LEN_FLOAT);
            _mm512_storeu_ps(dst + i, atan2512f_ps(src1.val[1], src1.val[0]));
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, atan2512f_ps(src2.val[1], src2.val[0]));
            j += 4 * AVX512_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src[i].im, src[i].re);
    }
}

static inline v16sf asin512f_ps(v16sf xx)
{
    v16sf a, x, z, z_tmp;
    //__mmask16 sign;
    __mmask16 ainfem4, asup0p5;
    v16sf tmp, sign;

    x = xx;
    a = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, x);  // fabs(x)
    // sign = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OS);  // 0xFFFFFFFF if x < 0.0
    sign = _mm512_and_ps(xx, *(v16sf *) _ps512_sign_mask);

    // TODO : vectorize this
    /*if( a > 1.0f )
    {
        return( 0.0f );
    }*/
    ainfem4 = _mm512_cmp_ps_mask(a, _mm512_set1_ps(1.0e-4), _CMP_LT_OS);  // if( a < 1.0e-4f )

    asup0p5 = _mm512_cmp_ps_mask(a, *(v16sf *) _ps512_0p5, _CMP_GT_OS);  // if( a > 0.5f ) flag = 1 else 0
    z_tmp = _mm512_sub_ps(*(v16sf *) _ps512_1, a);
    z_tmp = _mm512_mul_ps(*(v16sf *) _ps512_0p5, z_tmp);
    z = _mm512_mask_blend_ps(asup0p5, _mm512_mul_ps(a, a), z_tmp);
    x = _mm512_mask_sqrt_ps(a, asup0p5, z);

    tmp = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_ASIN_P0, z, *(v16sf *) _ps512_ASIN_P1);
    tmp = _mm512_fmadd_ps_custom(z, tmp, *(v16sf *) _ps512_ASIN_P2);
    tmp = _mm512_fmadd_ps_custom(z, tmp, *(v16sf *) _ps512_ASIN_P3);
    tmp = _mm512_fmadd_ps_custom(z, tmp, *(v16sf *) _ps512_ASIN_P4);
    tmp = _mm512_mul_ps(z, tmp);
    tmp = _mm512_fmadd_ps_custom(x, tmp, x);

    z = tmp;

    z_tmp = _mm512_add_ps(z, z);
    z = _mm512_mask_sub_ps(z, asup0p5, *(v16sf *) _ps512_PIO2F, z_tmp);

    // done:
    z = _mm512_mask_blend_ps(ainfem4, z, a);
    z = _mm512_xor_ps(z, sign);

    return (z);
}

static inline void asin512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    // z = x * x;
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN tan512f_ps" ::
                       : "memory");
#endif
    v16sf x, y, z, zz;
    v16si j;  // long?
    __mmask16 sign, xsupem4;
    v16sf tmp;
    __mmask16 jandone, jandtwo;

    x = _mm512_and_ps(*(v16sf *) _ps512_pos_sign_mask, xx);  // fabs(xx)

    /* compute x mod PIO4 */

    // TODO : on neg values should be ceil and not floor
    // j = _mm512_cvtps_epi32( _mm512_roundscale_ps(_mm512_mul_ps(*(v16sf*)_ps512_FOPI,x), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC )); /* integer part of x/(PI/4), using floor */
    j = _mm512_cvttps_epi32(_mm512_mul_ps(*(v16sf *) _ps512_FOPI, x));
    y = _mm512_cvtepi32_ps(j);

    jandone = _mm512_cmpgt_epi32_mask(_mm512_and_si512(j, *(v16si *) _pi32_512_1), _mm512_setzero_si512());
    y = _mm512_mask_add_ps(y, jandone, y, *(v16sf *) _ps512_1);
    j = _mm512_cvttps_epi32(y);  // no need to round again

    // z = ((x - y * DP1) - y * DP2) - y * DP3;
    z = _mm512_fmadd_ps_custom(y, *(v16sf *) _ps512_DP1, x);
    z = _mm512_fmadd_ps_custom(y, *(v16sf *) _ps512_DP2, z);
    z = _mm512_fmadd_ps_custom(y, *(v16sf *) _ps512_DP3, z);
    zz = _mm512_mul_ps(z, z);  // z*z

    // TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_TAN_P0, zz, *(v16sf *) _ps512_TAN_P1);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P2);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P3);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P4);
    tmp = _mm512_fmadd_ps_custom(tmp, zz, *(v16sf *) _ps512_TAN_P5);
    tmp = _mm512_mul_ps(zz, tmp);

    xsupem4 = _mm512_cmp_ps_mask(x, *(v16sf *) _ps512_1emin4, _CMP_GT_OS);  // if( x > 1.0e-4 )
    y = _mm512_mask_fmadd_ps(z, xsupem4, tmp, z);

    jandtwo = _mm512_cmpgt_epi32_mask(_mm512_and_si512(j, *(v16si *) _pi32_512_2), _mm512_setzero_si512());

    y = _mm512_mask_div_ps(y, jandtwo, *(v16sf *) _ps512_min1, y);

    sign = _mm512_cmp_ps_mask(xx, _mm512_setzero_ps(), _CMP_LT_OS);  // 0xFFFFFFFF if xx < 0.0
    y = _mm512_mask_xor_ps(y, sign, *(v16sf *) _ps512_neg_sign_mask, y);
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END tan512f_ps" ::
                       : "memory");
#endif
    return (y);
}

static inline void tan512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_load_ps(srcRe + i);
            v16sf im_tmp = _mm512_load_ps(srcIm + i);
            v16sf re_tmp2 = _mm512_load_ps(srcRe + i + AVX512_LEN_FLOAT);
            v16sf im_tmp2 = _mm512_load_ps(srcIm + i + AVX512_LEN_FLOAT);
            v16sf re_square = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf re_square2 = _mm512_mul_ps(re_tmp2, re_tmp2);
            v16sf dst_tmp = _mm512_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v16sf dst_tmp2 = _mm512_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            dst_tmp = _mm512_sqrt_ps(dst_tmp);
            dst_tmp2 = _mm512_sqrt_ps(dst_tmp2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_loadu_ps(srcRe + i);
            v16sf im_tmp = _mm512_loadu_ps(srcIm + i);
            v16sf re_tmp2 = _mm512_loadu_ps(srcRe + i + AVX512_LEN_FLOAT);
            v16sf im_tmp2 = _mm512_loadu_ps(srcIm + i + AVX512_LEN_FLOAT);
            v16sf re_square = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf re_square2 = _mm512_mul_ps(re_tmp2, re_tmp2);
            v16sf dst_tmp = _mm512_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v16sf dst_tmp2 = _mm512_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            dst_tmp = _mm512_sqrt_ps(dst_tmp);
            dst_tmp2 = _mm512_sqrt_ps(dst_tmp2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i]);
    }
}

static inline void powerspect512f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_load_ps(srcRe + i);
            v16sf im_tmp = _mm512_load_ps(srcIm + i);
            v16sf re_tmp2 = _mm512_load_ps(srcRe + i + AVX512_LEN_FLOAT);
            v16sf im_tmp2 = _mm512_load_ps(srcIm + i + AVX512_LEN_FLOAT);
            v16sf re_square = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf re_square2 = _mm512_mul_ps(re_tmp2, re_tmp2);
            v16sf dst_tmp = _mm512_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v16sf dst_tmp2 = _mm512_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf re_tmp = _mm512_loadu_ps(srcRe + i);
            v16sf im_tmp = _mm512_loadu_ps(srcIm + i);
            v16sf re_tmp2 = _mm512_loadu_ps(srcRe + i + AVX512_LEN_FLOAT);
            v16sf im_tmp2 = _mm512_loadu_ps(srcIm + i + AVX512_LEN_FLOAT);
            v16sf re_square = _mm512_mul_ps(re_tmp, re_tmp);
            v16sf re_square2 = _mm512_mul_ps(re_tmp2, re_tmp2);
            v16sf dst_tmp = _mm512_fmadd_ps_custom(im_tmp, im_tmp, re_square);
            v16sf dst_tmp2 = _mm512_fmadd_ps_custom(im_tmp2, im_tmp2, re_square2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i];
    }
}

static inline void magnitude512f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (4 * AVX512_LEN_FLOAT);
    stop_len *= 4 * AVX512_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 src_split = _mm512_load2_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v16sfx2 src_split2 = _mm512_load2_ps((float *) (src) + j + 2 * AVX512_LEN_FLOAT);
            v16sfx2 src_split3 = _mm512_load2_ps((float *) (src) + j + 4 * AVX512_LEN_FLOAT);
            v16sfx2 src_split4 = _mm512_load2_ps((float *) (src) + j + 6 * AVX512_LEN_FLOAT);
            v16sf split_square0 = _mm512_mul_ps(src_split.val[0], src_split.val[0]);
            v16sf split2_square0 = _mm512_mul_ps(src_split2.val[0], src_split2.val[0]);
            v16sf split3_square0 = _mm512_mul_ps(src_split3.val[0], src_split3.val[0]);
            v16sf split4_square0 = _mm512_mul_ps(src_split4.val[0], src_split4.val[0]);
            v16sfx2 dst_split;
            v16sfx2 dst_split2;
            dst_split.val[0] = _mm512_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm512_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm512_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm512_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            dst_split.val[0] = _mm512_sqrt_ps(dst_split.val[0]);
            dst_split.val[1] = _mm512_sqrt_ps(dst_split.val[1]);
            dst_split2.val[0] = _mm512_sqrt_ps(dst_split2.val[0]);
            dst_split2.val[1] = _mm512_sqrt_ps(dst_split2.val[1]);

            _mm512_store_ps((float *) (dst) + i, dst_split.val[0]);
            _mm512_store_ps((float *) (dst) + i + AVX512_LEN_FLOAT, dst_split.val[1]);
            _mm512_store_ps((float *) (dst) + i + 2 * AVX512_LEN_FLOAT, dst_split2.val[0]);
            _mm512_store_ps((float *) (dst) + i + 3 * AVX512_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * AVX512_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 src_split = _mm512_load2u_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v16sfx2 src_split2 = _mm512_load2u_ps((float *) (src) + j + 2 * AVX512_LEN_FLOAT);
            v16sfx2 src_split3 = _mm512_load2u_ps((float *) (src) + j + 4 * AVX512_LEN_FLOAT);
            v16sfx2 src_split4 = _mm512_load2u_ps((float *) (src) + j + 6 * AVX512_LEN_FLOAT);
            v16sf split_square0 = _mm512_mul_ps(src_split.val[0], src_split.val[0]);
            v16sf split2_square0 = _mm512_mul_ps(src_split2.val[0], src_split2.val[0]);
            v16sf split3_square0 = _mm512_mul_ps(src_split3.val[0], src_split3.val[0]);
            v16sf split4_square0 = _mm512_mul_ps(src_split4.val[0], src_split4.val[0]);
            v16sfx2 dst_split;
            v16sfx2 dst_split2;
            dst_split.val[0] = _mm512_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm512_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm512_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm512_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            dst_split.val[0] = _mm512_sqrt_ps(dst_split.val[0]);
            dst_split.val[1] = _mm512_sqrt_ps(dst_split.val[1]);
            dst_split2.val[0] = _mm512_sqrt_ps(dst_split2.val[0]);
            dst_split2.val[1] = _mm512_sqrt_ps(dst_split2.val[1]);

            _mm512_storeu_ps((float *) (dst) + i, dst_split.val[0]);
            _mm512_storeu_ps((float *) (dst) + i + AVX512_LEN_FLOAT, dst_split.val[1]);
            _mm512_storeu_ps((float *) (dst) + i + 2 * AVX512_LEN_FLOAT, dst_split2.val[0]);
            _mm512_storeu_ps((float *) (dst) + i + 3 * AVX512_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * AVX512_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i].re * src[i].re + (src[i].im * src[i].im));
    }
}

static inline void powerspect512f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (4 * AVX512_LEN_FLOAT);
    stop_len *= 4 * AVX512_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 src_split = _mm512_load2_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v16sfx2 src_split2 = _mm512_load2_ps((float *) (src) + j + 2 * AVX512_LEN_FLOAT);
            v16sfx2 src_split3 = _mm512_load2_ps((float *) (src) + j + 4 * AVX512_LEN_FLOAT);
            v16sfx2 src_split4 = _mm512_load2_ps((float *) (src) + j + 6 * AVX512_LEN_FLOAT);
            v16sf split_square0 = _mm512_mul_ps(src_split.val[0], src_split.val[0]);
            v16sf split2_square0 = _mm512_mul_ps(src_split2.val[0], src_split2.val[0]);
            v16sf split3_square0 = _mm512_mul_ps(src_split3.val[0], src_split3.val[0]);
            v16sf split4_square0 = _mm512_mul_ps(src_split4.val[0], src_split4.val[0]);
            v16sfx2 dst_split;
            v16sfx2 dst_split2;
            dst_split.val[0] = _mm512_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm512_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm512_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm512_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            _mm512_store_ps((float *) (dst) + i, dst_split.val[0]);
            _mm512_store_ps((float *) (dst) + i + AVX512_LEN_FLOAT, dst_split.val[1]);
            _mm512_store_ps((float *) (dst) + i + 2 * AVX512_LEN_FLOAT, dst_split2.val[0]);
            _mm512_store_ps((float *) (dst) + i + 3 * AVX512_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * AVX512_LEN_FLOAT;
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 src_split = _mm512_load2u_ps((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
            v16sfx2 src_split2 = _mm512_load2u_ps((float *) (src) + j + 2 * AVX512_LEN_FLOAT);
            v16sfx2 src_split3 = _mm512_load2u_ps((float *) (src) + j + 4 * AVX512_LEN_FLOAT);
            v16sfx2 src_split4 = _mm512_load2u_ps((float *) (src) + j + 6 * AVX512_LEN_FLOAT);
            v16sf split_square0 = _mm512_mul_ps(src_split.val[0], src_split.val[0]);
            v16sf split2_square0 = _mm512_mul_ps(src_split2.val[0], src_split2.val[0]);
            v16sf split3_square0 = _mm512_mul_ps(src_split3.val[0], src_split3.val[0]);
            v16sf split4_square0 = _mm512_mul_ps(src_split4.val[0], src_split4.val[0]);
            v16sfx2 dst_split;
            v16sfx2 dst_split2;
            dst_split.val[0] = _mm512_fmadd_ps_custom(src_split.val[1], src_split.val[1], split_square0);
            dst_split.val[1] = _mm512_fmadd_ps_custom(src_split2.val[1], src_split2.val[1], split2_square0);
            dst_split2.val[0] = _mm512_fmadd_ps_custom(src_split3.val[1], src_split3.val[1], split3_square0);
            dst_split2.val[1] = _mm512_fmadd_ps_custom(src_split4.val[1], src_split4.val[1], split4_square0);

            _mm512_storeu_ps((float *) (dst) + i, dst_split.val[0]);
            _mm512_storeu_ps((float *) (dst) + i + AVX512_LEN_FLOAT, dst_split.val[1]);
            _mm512_storeu_ps((float *) (dst) + i + 2 * AVX512_LEN_FLOAT, dst_split2.val[0]);
            _mm512_storeu_ps((float *) (dst) + i + 3 * AVX512_LEN_FLOAT, dst_split2.val[1]);
            j += 8 * AVX512_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i].re * src[i].re + (src[i].im * src[i].im);
    }
}

static inline void subcrev512f(float *src, float value, float *dst, int len)
{
    const v16sf tmp = _mm512_set1_ps(value);

    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
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
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    __attribute__((aligned(AVX512_LEN_BYTES))) float accumulate[AVX512_LEN_FLOAT];
    float tmp_acc = 0.0f;
    v16sf vec_acc1 = _mm512_setzero_ps();  // initialize the vector accumulator
    v16sf vec_acc2 = _mm512_setzero_ps();  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf vec_tmp1 = _mm512_load_ps(src + i);
            vec_acc1 = _mm512_add_ps(vec_acc1, vec_tmp1);
            v16sf vec_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            vec_acc2 = _mm512_add_ps(vec_acc2, vec_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf vec_tmp1 = _mm512_loadu_ps(src + i);
            vec_acc1 = _mm512_add_ps(vec_acc1, vec_tmp1);
            v16sf vec_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            vec_acc2 = _mm512_add_ps(vec_acc2, vec_tmp2);
        }
    }

    vec_acc1 = _mm512_add_ps(vec_acc1, vec_acc2);
    _mm512_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7] + accumulate[8] + accumulate[9] + accumulate[10] + accumulate[11] + accumulate[12] + accumulate[13] + accumulate[14] + accumulate[15];

    *dst = tmp_acc;
}


static inline void mean512f(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sum512f(src, dst, len);
    *dst *= coeff;
}

static inline void dot512f(float *src1, float *src2, int len, float *dst)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    __attribute__((aligned(AVX512_LEN_BYTES))) float accumulate[AVX512_LEN_FLOAT];
    float tmp_acc = 0.0f;
    v16sf vec_acc1 = _mm512_setzero_ps();  // initialize the vector accumulator
    v16sf vec_acc2 = _mm512_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf vec_src1_tmp = _mm512_load_ps(src1 + i);
            v16sf vec_src1_tmp2 = _mm512_load_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf vec_src2_tmp = _mm512_load_ps(src2 + i);
            v16sf vec_src2_tmp2 = _mm512_load_ps(src2 + i + AVX512_LEN_FLOAT);
            vec_acc1 = _mm512_fmadd_ps(vec_src1_tmp, vec_src2_tmp, vec_acc1);
            vec_acc2 = _mm512_fmadd_ps(vec_src1_tmp2, vec_src2_tmp2, vec_acc2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf vec_src1_tmp = _mm512_loadu_ps(src1 + i);
            v16sf vec_src1_tmp2 = _mm512_loadu_ps(src1 + i + AVX512_LEN_FLOAT);
            v16sf vec_src2_tmp = _mm512_loadu_ps(src2 + i);
            v16sf vec_src2_tmp2 = _mm512_loadu_ps(src2 + i + AVX512_LEN_FLOAT);
            vec_acc1 = _mm512_fmadd_ps(vec_src1_tmp, vec_src2_tmp, vec_acc1);
            vec_acc2 = _mm512_fmadd_ps(vec_src1_tmp2, vec_src2_tmp2, vec_acc2);
        }
    }
    vec_acc1 = _mm512_add_ps(vec_acc1, vec_acc2);
    _mm512_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src1[i] * src2[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7] + accumulate[8] + accumulate[9] + accumulate[10] + accumulate[11] + accumulate[12] + accumulate[13] + accumulate[14] + accumulate[15];

    *dst = tmp_acc;
}

static inline void dotc512f(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    int stop_len = len / (4 * AVX512_LEN_FLOAT);
    stop_len *= (4 * AVX512_LEN_FLOAT);

    v16sfx2 vec_acc1 = {_mm512_setzero_ps(), _mm512_setzero_ps()};  // initialize the vector accumulator
    v16sfx2 vec_acc2 = {_mm512_setzero_ps(), _mm512_setzero_ps()};  // initialize the vector accumulator

    complex32_t dst_tmp = {0.0f, 0.0f};

    __attribute__((aligned(AVX512_LEN_BYTES))) float accumulateRe[AVX512_LEN_FLOAT];
    __attribute__((aligned(AVX512_LEN_BYTES))) float accumulateIm[AVX512_LEN_FLOAT];

    //  (ac -bd) + i(ad + bc)
    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), AVX512_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 src1_split = _mm512_load2_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v16sfx2 src2_split = _mm512_load2_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v16sfx2 src1_split2 = _mm512_load2_ps((float *) (src1) + i + 2 * AVX512_LEN_FLOAT);
            v16sfx2 src2_split2 = _mm512_load2_ps((float *) (src2) + i + 2 * AVX512_LEN_FLOAT);
            v16sf ac = _mm512_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v16sf ad = _mm512_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v16sf ac2 = _mm512_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v16sf ad2 = _mm512_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v16sfx2 tmp_split;
            v16sfx2 tmp_split2;
            tmp_split.val[0] = _mm512_fnmadd_ps(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = _mm512_fmadd_ps(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = _mm512_fnmadd_ps(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = _mm512_fmadd_ps(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = _mm512_add_ps(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = _mm512_add_ps(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = _mm512_add_ps(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = _mm512_add_ps(vec_acc2.val[1], tmp_split2.val[1]);
        }
    } else {
        for (int i = 0; i < 2 * stop_len; i += 4 * AVX512_LEN_FLOAT) {
            v16sfx2 src1_split = _mm512_load2u_ps((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v16sfx2 src2_split = _mm512_load2u_ps((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v16sfx2 src1_split2 = _mm512_load2u_ps((float *) (src1) + i + 2 * AVX512_LEN_FLOAT);
            v16sfx2 src2_split2 = _mm512_load2u_ps((float *) (src2) + i + 2 * AVX512_LEN_FLOAT);
            v16sf ac = _mm512_mul_ps(src1_split.val[0], src2_split.val[0]);     // ac
            v16sf ad = _mm512_mul_ps(src1_split.val[0], src2_split.val[1]);     // ad
            v16sf ac2 = _mm512_mul_ps(src1_split2.val[0], src2_split2.val[0]);  // ac
            v16sf ad2 = _mm512_mul_ps(src1_split2.val[0], src2_split2.val[1]);  // ad
            v16sfx2 tmp_split;
            v16sfx2 tmp_split2;
            tmp_split.val[0] = _mm512_fnmadd_ps(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = _mm512_fmadd_ps(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = _mm512_fnmadd_ps(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = _mm512_fmadd_ps(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = _mm512_add_ps(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = _mm512_add_ps(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = _mm512_add_ps(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = _mm512_add_ps(vec_acc2.val[1], tmp_split2.val[1]);
        }
    }

    vec_acc1.val[0] = _mm512_add_ps(vec_acc1.val[0], vec_acc2.val[0]);
    vec_acc1.val[1] = _mm512_add_ps(vec_acc1.val[1], vec_acc2.val[1]);
    _mm512_store_ps(accumulateRe, vec_acc1.val[0]);
    _mm512_store_ps(accumulateIm, vec_acc1.val[1]);

    for (int i = stop_len; i < len; i++) {
        dst_tmp.re += src1[i].re * src2[i].re - (src1[i].im * src2[i].im);
        dst_tmp.im += src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }

    dst_tmp.re = dst_tmp.re + accumulateRe[0] + accumulateRe[1] + accumulateRe[2] + accumulateRe[3] +
                 accumulateRe[4] + accumulateRe[5] + accumulateRe[6] + accumulateRe[7] +
                 accumulateRe[8] + accumulateRe[9] + accumulateRe[10] + accumulateRe[11] +
                 accumulateRe[12] + accumulateRe[13] + accumulateRe[14] + accumulateRe[15];
    dst_tmp.im = dst_tmp.im + accumulateIm[0] + accumulateIm[1] + accumulateIm[2] + accumulateIm[3] +
                 accumulateIm[4] + accumulateIm[5] + accumulateIm[6] + accumulateIm[7] +
                 accumulateIm[8] + accumulateIm[9] + accumulateIm[10] + accumulateIm[11] +
                 accumulateIm[12] + accumulateIm[13] + accumulateIm[14] + accumulateIm[15];


    dst->re = dst_tmp.re;
    dst->im = dst_tmp.im;
}

static inline void sqrt512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_sqrt_ps(src_tmp);
            v16sf dst_tmp2 = _mm512_sqrt_ps(src_tmp2);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_sqrt_ps(src_tmp);
            v16sf dst_tmp2 = _mm512_sqrt_ps(src_tmp2);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static inline void round512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}

static inline void trunc512f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            _mm512_store_ps(dst + i, dst_tmp);
            _mm512_store_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_roundscale_ps(src_tmp, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            v16sf dst_tmp2 = _mm512_roundscale_ps(src_tmp2, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(dst + i, dst_tmp);
            _mm512_storeu_ps(dst + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

static inline void cplxvecdiv512f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)

{
    int stop_len = len / (AVX512_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX512_LEN_FLOAT;   // stop_len << 2;

    int i;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_load_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_load_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v16sf c2d2 = _mm512_mul_ps(src2_tmp, src2_tmp);
            v16sf c2d2_shuf = _mm512_shuffle_ps(c2d2, c2d2, _MM_SHUFFLE(2, 3, 0, 1));
            c2d2 = _mm512_add_ps(c2d2_shuf, c2d2);
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);  // a1,a1,a0,a0
            tmp1 = _mm512_mul_ps(*(v16sf *) _ps512_conj_mask, tmp1);
            v16sf tmp2 = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v16sf tmp3 = _mm512_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v16sf out = _mm512_mul_ps(tmp2, tmp3);                                        // c1b1, b1d1, c0b0, d0b0
            out = _mm512_fmadd_ps_custom(tmp1, src2_tmp, out);
            out = _mm512_div_ps(out, c2d2);
            _mm512_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_loadu_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_loadu_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v16sf c2d2 = _mm512_mul_ps(src2_tmp, src2_tmp);
            v16sf c2d2_shuf = _mm512_shuffle_ps(c2d2, c2d2, _MM_SHUFFLE(2, 3, 0, 1));
            c2d2 = _mm512_add_ps(c2d2_shuf, c2d2);
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);  // a1,a1,a0,a0
            tmp1 = _mm512_mul_ps(*(v16sf *) _ps512_conj_mask, tmp1);
            v16sf tmp2 = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v16sf tmp3 = _mm512_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v16sf out = _mm512_mul_ps(tmp2, tmp3);                                        // c1b1, b1d1, c0b0, d0b0
            out = _mm512_fmadd_ps_custom(tmp1, src2_tmp, out);
            out = _mm512_div_ps(out, c2d2);
            _mm512_storeu_ps((float *) (dst) + i, out);
        }
    }
    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2[i].re * src2[i].re + src2[i].im * src2[i].im;
        dst[i].re = ((src1[i].re * src2[i].re) + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-(src1[i].re * src2[i].im) + (src2[i].re * src1[i].im)) / c2d2;
    }
}

static inline void cplxvecdiv512f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= 2 * AVX512_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1Re), (uintptr_t) (src2Re), (uintptr_t) (src2Re), AVX512_LEN_BYTES) &&
        areAligned3((uintptr_t) (src1Im), (uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_load_ps((float *) (src1Re) + i);
            v16sf src1Re_tmp2 = _mm512_load_ps((float *) (src1Re) + i + AVX512_LEN_FLOAT);
            v16sf src1Im_tmp = _mm512_load_ps((float *) (src1Im) + i);
            v16sf src1Im_tmp2 = _mm512_load_ps((float *) (src1Im) + i + AVX512_LEN_FLOAT);
            v16sf src2Re_tmp = _mm512_load_ps((float *) (src2Re) + i);
            v16sf src2Re_tmp2 = _mm512_load_ps((float *) (src2Re) + i + AVX512_LEN_FLOAT);
            v16sf src2Im_tmp = _mm512_load_ps((float *) (src2Im) + i);
            v16sf src2Im_tmp2 = _mm512_load_ps((float *) (src2Im) + i + AVX512_LEN_FLOAT);

            v16sf c2 = _mm512_mul_ps(src2Re_tmp, src2Re_tmp);
            v16sf c2d2 = _mm512_fmadd_ps_custom(src2Im_tmp, src2Im_tmp, c2);
            v16sf c2_ = _mm512_mul_ps(src2Re_tmp2, src2Re_tmp2);
            v16sf c2d2_ = _mm512_fmadd_ps_custom(src2Im_tmp2, src2Im_tmp2, c2_);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);     // ac
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);     // bc
            v16sf ac2 = _mm512_mul_ps(src1Re_tmp2, src2Re_tmp2);  // ac
            v16sf bc2 = _mm512_mul_ps(src1Im_tmp2, src2Re_tmp2);  // bc

            v16sf dstRe_tmp = _mm512_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac);
            v16sf dstRe_tmp2 = _mm512_fmadd_ps_custom(src1Im_tmp2, src2Im_tmp2, ac2);
            v16sf dstIm_tmp = _mm512_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc);
            v16sf dstIm_tmp2 = _mm512_fnmadd_ps_custom(src1Re_tmp2, src2Im_tmp2, bc2);

            dstRe_tmp = _mm512_div_ps(dstRe_tmp, c2d2);
            dstIm_tmp = _mm512_div_ps(dstIm_tmp, c2d2);
            dstRe_tmp2 = _mm512_div_ps(dstRe_tmp2, c2d2_);
            dstIm_tmp2 = _mm512_div_ps(dstIm_tmp2, c2d2_);

            _mm512_store_ps((float *) (dstRe) + i, dstRe_tmp);
            _mm512_store_ps((float *) (dstIm) + i, dstIm_tmp);
            _mm512_store_ps((float *) (dstRe) + i + AVX512_LEN_FLOAT, dstRe_tmp2);
            _mm512_store_ps((float *) (dstIm) + i + AVX512_LEN_FLOAT, dstIm_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_loadu_ps((float *) (src1Re) + i);
            v16sf src1Re_tmp2 = _mm512_loadu_ps((float *) (src1Re) + i + AVX512_LEN_FLOAT);
            v16sf src1Im_tmp = _mm512_loadu_ps((float *) (src1Im) + i);
            v16sf src1Im_tmp2 = _mm512_loadu_ps((float *) (src1Im) + i + AVX512_LEN_FLOAT);
            v16sf src2Re_tmp = _mm512_loadu_ps((float *) (src2Re) + i);
            v16sf src2Re_tmp2 = _mm512_loadu_ps((float *) (src2Re) + i + AVX512_LEN_FLOAT);
            v16sf src2Im_tmp = _mm512_loadu_ps((float *) (src2Im) + i);
            v16sf src2Im_tmp2 = _mm512_loadu_ps((float *) (src2Im) + i + AVX512_LEN_FLOAT);

            v16sf c2 = _mm512_mul_ps(src2Re_tmp, src2Re_tmp);
            v16sf c2d2 = _mm512_fmadd_ps_custom(src2Im_tmp, src2Im_tmp, c2);
            v16sf c2_ = _mm512_mul_ps(src2Re_tmp2, src2Re_tmp2);
            v16sf c2d2_ = _mm512_fmadd_ps_custom(src2Im_tmp2, src2Im_tmp2, c2_);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);     // ac
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);     // bc
            v16sf ac2 = _mm512_mul_ps(src1Re_tmp2, src2Re_tmp2);  // ac
            v16sf bc2 = _mm512_mul_ps(src1Im_tmp2, src2Re_tmp2);  // bc

            v16sf dstRe_tmp = _mm512_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac);
            v16sf dstRe_tmp2 = _mm512_fmadd_ps_custom(src1Im_tmp2, src2Im_tmp2, ac2);
            v16sf dstIm_tmp = _mm512_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc);
            v16sf dstIm_tmp2 = _mm512_fnmadd_ps_custom(src1Re_tmp2, src2Im_tmp2, bc2);

            dstRe_tmp = _mm512_div_ps(dstRe_tmp, c2d2);
            dstIm_tmp = _mm512_div_ps(dstIm_tmp, c2d2);
            dstRe_tmp2 = _mm512_div_ps(dstRe_tmp2, c2d2_);
            dstIm_tmp2 = _mm512_div_ps(dstIm_tmp2, c2d2_);

            _mm512_storeu_ps((float *) (dstRe) + i, dstRe_tmp);
            _mm512_storeu_ps((float *) (dstIm) + i, dstIm_tmp);
            _mm512_storeu_ps((float *) (dstRe) + i + AVX512_LEN_FLOAT, dstRe_tmp2);
            _mm512_storeu_ps((float *) (dstIm) + i + AVX512_LEN_FLOAT, dstIm_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2Re[i] * src2Re[i] + src2Im[i] * src2Im[i];
        dstRe[i] = (src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i])) / c2d2;
        dstIm[i] = (-src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i])) / c2d2;
    }
}

static inline void cplxvecmul512f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX512_LEN_FLOAT;   // stop_len << 2;

    int i;
    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_load_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_load_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);              // a1,a1,a0,a0
            // v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                 //a1d1,a1c1,a0d0,a0c0
            v16sf tmp3 = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v16sf tmp4 = _mm512_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v16sf out = _mm512_mul_ps(tmp3, tmp4);
            // out = _mm512_fmaddsub_ps_custom(*(v16sf *) _ps512_plus1, tmp2, out);
            out = _mm512_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm512_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_loadu_ps((float *) (src1) + i);  // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_loadu_ps((float *) (src2) + i);  // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);               // a1,a1,a0,a0
            // v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                 //a1d1,a1c1,a0d0,a0c0
            v16sf tmp3 = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v16sf tmp4 = _mm512_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0
            v16sf out = _mm512_mul_ps(tmp3, tmp4);
            // out = _mm512_fmaddsub_ps_custom(*(v16sf *) _ps512_plus1, tmp2, out);
            out = _mm512_fmaddsub_ps_custom(tmp1, src2_tmp, out);
            _mm512_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxvecmul512f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);
    stop_len = stop_len * AVX512_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX512_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_load_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_load_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_load_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_load_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            // v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            // v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_store_ps(dstRe + i, _mm512_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  // ac - bd
            _mm512_store_ps(dstIm + i, _mm512_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    } else {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_loadu_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_loadu_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_loadu_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_loadu_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            // v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            // v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_storeu_ps(dstRe + i, _mm512_fnmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));  // ac - bd
            _mm512_storeu_ps(dstIm + i, _mm512_fmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));   // ad + bc
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = (src1Re[i] * src2Re[i]) - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i]);
    }
}

static inline void cplxconjvecmul512f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len = stop_len * AVX512_LEN_FLOAT;   // stop_len << 2;

    int i;
    // const v16sf conj_mask = _mm512_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    //                                       -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_load_ps((float *) (src1) + i);                        // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_load_ps((float *) (src2) + i);                        // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                   // a1d1,a1c1,a0d0,a0c0
            v16sf tmp3 = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v16sf tmp4 = _mm512_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0

#ifndef FMA
            v16sf out = _mm512_mul_ps(tmp3, tmp4);
            out = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_conj_mask, tmp2, out);
#else
            v16sf out = _mm512_fmsubadd_ps(tmp3, tmp4, tmp2);
#endif

            _mm512_store_ps((float *) (dst) + i, out);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1_tmp = _mm512_loadu_ps((float *) (src1) + i);                       // src1 = b1,a1,b0,a0 (little endian)
            v16sf src2_tmp = _mm512_loadu_ps((float *) (src2) + i);                       // src2 = d1,c1,d0,c0
            v16sf tmp1 = _mm512_moveldup_ps(src1_tmp);                                    // a1,a1,a0,a0
            v16sf tmp2 = _mm512_mul_ps(tmp1, src2_tmp);                                   // a1d1,a1c1,a0d0,a0c0
            v16sf tmp3 = _mm512_shuffle_ps(src2_tmp, src2_tmp, _MM_SHUFFLE(2, 3, 0, 1));  // c1,d1,c0,d0
            v16sf tmp4 = _mm512_movehdup_ps(src1_tmp);                                    // b1,b1,b0,b0

#ifndef FMA
            v16sf out = _mm512_mul_ps(tmp3, tmp4);
            out = _mm512_fmadd_ps_custom(*(v16sf *) _ps512_conj_mask, tmp2, out);
#else
            v16sf out = _mm512_fmsubadd_ps(tmp3, tmp4, tmp2);
#endif

            _mm512_storeu_ps((float *) (dst) + i, out);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + (src1[i].im * src2[i].im);
        dst[i].im = (src2[i].re * src1[i].im) - src1[i].re * src2[i].im;
    }
}

static inline void cplxconjvecmul512f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);
    stop_len = stop_len * AVX512_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), AVX512_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_load_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_load_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_load_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_load_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            // v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            // v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_store_ps(dstRe + i, _mm512_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   // ac + bd
            _mm512_store_ps(dstIm + i, _mm512_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    } else {
        for (i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src1Re_tmp = _mm512_loadu_ps((float *) (src1Re) + i);
            v16sf src1Im_tmp = _mm512_loadu_ps((float *) (src1Im) + i);
            v16sf src2Re_tmp = _mm512_loadu_ps((float *) (src2Re) + i);
            v16sf src2Im_tmp = _mm512_loadu_ps((float *) (src2Im) + i);
            v16sf ac = _mm512_mul_ps(src1Re_tmp, src2Re_tmp);
            // v16sf bd = _mm512_mul_ps(src1Im_tmp, src2Im_tmp);
            // v16sf ad = _mm512_mul_ps(src1Re_tmp, src2Im_tmp);
            v16sf bc = _mm512_mul_ps(src1Im_tmp, src2Re_tmp);
            _mm512_storeu_ps(dstRe + i, _mm512_fmadd_ps_custom(src1Im_tmp, src2Im_tmp, ac));   // ac + bd
            _mm512_storeu_ps(dstIm + i, _mm512_fnmadd_ps_custom(src1Re_tmp, src2Im_tmp, bc));  // bc - ad
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i]);
        dstIm[i] = (src2Re[i] * src1Im[i]) - src1Re[i] * src2Im[i];
    }
}

// prefer using cplxconjvecmulXf if you also need to do a multiply
static inline void cplxconj512f(complex32_t *src, complex32_t *dst, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len *= 2 * AVX512_LEN_FLOAT;             // stop_len << 2;

    __attribute__((aligned(AVX512_LEN_BYTES))) int32_t conj_mask[AVX512_LEN_FLOAT] = {
        (int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000,
        (int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000,
        (int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000,
        (int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000};
    int i;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), 2 * AVX512_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps((float *) (src) + i);
            v16sf src_tmp2 = _mm512_load_ps((float *) (src) + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_xor_ps(src_tmp, *(v16sf *) &conj_mask);
            v16sf dst_tmp2 = _mm512_xor_ps(src_tmp2, *(v16sf *) &conj_mask);
            _mm512_store_ps((float *) (dst) + i, dst_tmp);
            _mm512_store_ps((float *) (dst) + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    } else {
        for (i = 0; i < 2 * stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps((float *) (src) + i);
            v16sf src_tmp2 = _mm512_loadu_ps((float *) (src) + i + AVX512_LEN_FLOAT);
            v16sf dst_tmp = _mm512_xor_ps(src_tmp, *(v16sf *) &conj_mask);
            v16sf dst_tmp2 = _mm512_xor_ps(src_tmp2, *(v16sf *) &conj_mask);
            _mm512_storeu_ps((float *) (dst) + i, dst_tmp);
            _mm512_storeu_ps((float *) (dst) + i + AVX512_LEN_FLOAT, dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src[i].re;
        dst[i].im = -src[i].im;
    }
}

static inline void sigmoid512f(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf tmp = _mm512_add_ps(*(v16sf *) _ps512_1, exp512_ps(_mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, src_tmp)));
            _mm512_store_ps(dst + i, _mm512_div_ps(*(v16sf *) _ps512_1, tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf tmp = _mm512_add_ps(*(v16sf *) _ps512_1, exp512_ps(_mm512_xor_ps(*(v16sf *) _ps512_neg_sign_mask, src_tmp)));
            _mm512_storeu_ps(dst + i, _mm512_div_ps(*(v16sf *) _ps512_1, tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}

static inline void PRelu512f(float *src, float *dst, float alpha, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    v16sf alpha_vec = _mm512_set1_ps(alpha);
    v16sf zero = _mm512_setzero_ps();

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf tmp = _mm512_mul_ps(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)
            __mmask16 compare = _mm512_cmp_ps_mask(src_tmp, zero, _CMP_GT_OS);
            _mm512_store_ps(dst + i, _mm512_mask_blend_ps(compare, tmp, src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf tmp = _mm512_mul_ps(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)
            __mmask16 compare = _mm512_cmp_ps_mask(src_tmp, zero, _CMP_GT_OS);
            _mm512_storeu_ps(dst + i, _mm512_mask_blend_ps(compare, tmp, src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] > 0.0f)
            dst[i] = src[i];
        else
            dst[i] = alpha * src[i];
    }
}

static inline void softmax512f(float *src, float *dst, int len)
{
    int stop_len = len / (AVX512_LEN_FLOAT);
    stop_len *= (AVX512_LEN_FLOAT);

    __attribute__((aligned(AVX512_LEN_BYTES))) float accumulate[AVX512_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f,
                                                                                     0.0f, 0.0f, 0.0f, 0.0f,
                                                                                     0.0f, 0.0f, 0.0f, 0.0f,
                                                                                     0.0f, 0.0f, 0.0f, 0.0f};
    float acc = 0.0f;

    v16sf vec_acc1 = _mm512_setzero_ps();  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf dst_tmp = exp512_ps(src_tmp);
            vec_acc1 = _mm512_add_ps(vec_acc1, dst_tmp);
            _mm512_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf dst_tmp = exp512_ps(src_tmp);
            vec_acc1 = _mm512_add_ps(vec_acc1, dst_tmp);
            _mm512_storeu_ps(dst + i, dst_tmp);
        }
    }

    _mm512_store_ps(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
        acc += dst[i];
    }

    acc = acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] +
          accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7] +
          accumulate[8] + accumulate[9] + accumulate[10] + accumulate[11] +
          accumulate[12] + accumulate[13] + accumulate[14] + accumulate[15];
    vec_acc1 = _mm512_set1_ps(acc);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf dst_tmp = _mm512_load_ps(dst + i);
            _mm512_store_ps(dst + i, _mm512_div_ps(dst_tmp, vec_acc1));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf dst_tmp = _mm512_loadu_ps(dst + i);
            _mm512_storeu_ps(dst + i, _mm512_div_ps(dst_tmp, vec_acc1));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] /= acc;
    }
}

static inline void pol2cart2D512f(float *r, float *theta, float *x, float *y, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf r_tmp = _mm512_load_ps(r + i);
            v16sf theta_tmp = _mm512_load_ps(theta + i);
            v16sf sin_tmp;
            v16sf cos_tmp;
            sincos512_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v16sf x_tmp = _mm512_mul_ps(r_tmp, cos_tmp);
            v16sf y_tmp = _mm512_mul_ps(r_tmp, sin_tmp);
            _mm512_store_ps(x + i, x_tmp);
            _mm512_store_ps(y + i, y_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf r_tmp = _mm512_loadu_ps(r + i);
            v16sf theta_tmp = _mm512_loadu_ps(theta + i);
            v16sf sin_tmp;
            v16sf cos_tmp;
            sincos512_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v16sf x_tmp = _mm512_mul_ps(r_tmp, cos_tmp);
            v16sf y_tmp = _mm512_mul_ps(r_tmp, sin_tmp);
            _mm512_storeu_ps(x + i, x_tmp);
            _mm512_storeu_ps(y + i, y_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        float sin_tmp, cos_tmp;
        mysincosf(theta[i], &sin_tmp, &cos_tmp);
        x[i] = r[i] * cos_tmp;
        y[i] = r[i] * sin_tmp;
    }
}

static inline void cart2pol2D512f(float *x, float *y, float *r, float *theta, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), AVX512_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf x_tmp = _mm512_load_ps(x + i);
            v16sf y_tmp = _mm512_load_ps(y + i);
            v16sf y_square = _mm512_mul_ps(y_tmp, y_tmp);
            v16sf r_tmp = _mm512_fmadd_ps_custom(x_tmp, x_tmp, y_square);
            r_tmp = _mm512_sqrt_ps(r_tmp);
            v16sf theta_tmp = atan2512f_ps(y_tmp, x_tmp);
            _mm512_store_ps(r + i, r_tmp);
            _mm512_store_ps(theta + i, theta_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf x_tmp = _mm512_loadu_ps(x + i);
            v16sf y_tmp = _mm512_loadu_ps(y + i);
            v16sf y_square = _mm512_mul_ps(y_tmp, y_tmp);
            v16sf r_tmp = _mm512_fmadd_ps_custom(x_tmp, x_tmp, y_square);
            r_tmp = _mm512_sqrt_ps(r_tmp);
            v16sf theta_tmp = atan2512f_ps(y_tmp, x_tmp);
            _mm512_storeu_ps(r + i, r_tmp);
            _mm512_storeu_ps(theta + i, theta_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        r[i] = sqrtf(x[i] * x[i] + (y[i] * y[i]));
        theta[i] = atan2f(y[i], x[i]);
    }
}

static inline void modf512f(float *src, float *integer, float *remainder, int len)
{
    int stop_len = len / (2 * AVX512_LEN_FLOAT);
    stop_len *= (2 * AVX512_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src), (uintptr_t) (integer), (uintptr_t) (remainder), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf src_tmp2 = _mm512_load_ps(src + i + AVX512_LEN_FLOAT);
            v16sf integer_tmp = _mm512_roundscale_ps(src_tmp, ROUNDTOZERO);
            v16sf integer_tmp2 = _mm512_roundscale_ps(src_tmp2, ROUNDTOZERO);
            v16sf remainder_tmp = _mm512_sub_ps(src_tmp, integer_tmp);
            v16sf remainder_tmp2 = _mm512_sub_ps(src_tmp2, integer_tmp2);
            _mm512_store_ps(integer + i, integer_tmp);
            _mm512_store_ps(integer + i + AVX512_LEN_FLOAT, integer_tmp2);
            _mm512_store_ps(remainder + i, remainder_tmp);
            _mm512_store_ps(remainder + i + AVX512_LEN_FLOAT, remainder_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf src_tmp2 = _mm512_loadu_ps(src + i + AVX512_LEN_FLOAT);
            v16sf integer_tmp = _mm512_roundscale_ps(src_tmp, ROUNDTOZERO);
            v16sf integer_tmp2 = _mm512_roundscale_ps(src_tmp2, ROUNDTOZERO);
            v16sf remainder_tmp = _mm512_sub_ps(src_tmp, integer_tmp);
            v16sf remainder_tmp2 = _mm512_sub_ps(src_tmp2, integer_tmp2);
            _mm512_storeu_ps(integer + i, integer_tmp);
            _mm512_storeu_ps(integer + i + AVX512_LEN_FLOAT, integer_tmp2);
            _mm512_storeu_ps(remainder + i, remainder_tmp);
            _mm512_storeu_ps(remainder + i + AVX512_LEN_FLOAT, remainder_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        remainder[i] = modff(src[i], &(integer[i]));
    }
}
