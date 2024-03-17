/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#define AMD_LIBM_VEC_EXPERIMENTAL
#include "amdlibm_vec.h"

#ifdef SSE
static inline void sin128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, amd_vrs4_sinf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, amd_vrs4_sinf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, amd_vrs4_cosf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, amd_vrs4_cosf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos128f_amdlibm(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf dst_sin_tmp = amd_vrs4_sinf(src_tmp);
            v4sf dst_cos_tmp = amd_vrs4_cosf(src_tmp);
            _mm_store_ps(dst_sin + i, dst_sin_tmp);
            _mm_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf dst_sin_tmp = amd_vrs4_sinf(src_tmp);
            v4sf dst_cos_tmp = amd_vrs4_cosf(src_tmp);
            _mm_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void sincos128d_amdlibm(double *src, double *dst_sin, double *dst_cos, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            v2sd dst_sin_tmp = amd_vrd2_sin(src_tmp);
            v2sd dst_cos_tmp = amd_vrd2_cos(src_tmp);
            _mm_store_pd(dst_sin + i, dst_sin_tmp);
            _mm_store_pd(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            v2sd dst_sin_tmp = amd_vrd2_sin(src_tmp);
            v2sd dst_cos_tmp = amd_vrd2_cos(src_tmp);
            _mm_storeu_pd(dst_sin + i, dst_sin_tmp);
            _mm_storeu_pd(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst_sin[i] = sin(src[i]);
        dst_cos[i] = cos(src[i]);
    }
}

static inline void exp128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, amd_vrs4_expf(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, amd_vrs4_expf(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

static inline void exp128d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, amd_vrd2_exp(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, amd_vrd2_exp(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = exp(src[i]);
    }
}

static inline void ln128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, amd_vrs4_logf(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, amd_vrs4_logf(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void ln128d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, amd_vrd2_log(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, amd_vrd2_log(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = exp(src[i]);
    }
}

static inline void log2128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, amd_vrs4_log2f(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, amd_vrs4_log2f(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline void log10128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, amd_vrs4_log10f(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, amd_vrs4_log10f(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void atan128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, amd_vrs4_atanf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, amd_vrs4_atanf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline void atan128d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, amd_vrd2_atan(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, amd_vrd2_atan(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan(src[i]);
    }
}

/*
static inline void atan2128f_amdlibm(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_atan2_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_atan2_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}
*/

static inline void asin128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, amd_vrs4_asinf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, amd_vrs4_asinf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}

/*
static inline void asin512d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, _mm_asin_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, _mm_asin_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}
*/

static inline void tan128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, amd_vrs4_tanf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, amd_vrs4_tanf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void tan128d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / SSE_LEN_DOUBLE;
    stop_len *= SSE_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_load_pd(src + i);
            _mm_store_pd(dst + i, amd_vrd2_tan(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_DOUBLE) {
            v2sd src_tmp = _mm_loadu_pd(src + i);
            _mm_storeu_pd(dst + i, amd_vrd2_tan(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tan(src[i]);
    }
}

/*
static inline void atanh128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_atanh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_atanh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline void acosh128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_acosh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_acosh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline void asinh128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_asinh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_asinh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline void sinh128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_sinh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_sinh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}
*/

static inline void cosh128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, amd_vrs4_coshf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, amd_vrs4_coshf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline void tanh128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, amd_vrs4_tanhf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, amd_vrs4_tanhf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}

static inline void cbrt128f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= (SSE_LEN_FLOAT);
    
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x = _mm_load_ps(src + i);
            v4sf dst_tmp = amd_vrs4_cbrtf(x);
            _mm_store_ps(dst + i, dst_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf x = _mm_loadu_ps(src + i);
            v4sf dst_tmp = amd_vrs4_cbrtf(x);
            _mm_storeu_ps(dst + i, dst_tmp);
        }
    }
    for (int i = stop_len; i < len; i++) {
        dst[i] = cbrtf(src[i]);
    }
}

#endif

#ifdef AVX

static inline void sin256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, amd_vrs8_sinf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, amd_vrs8_sinf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, amd_vrs8_cosf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, amd_vrs8_cosf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos256f_amdlibm(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf dst_sin_tmp = amd_vrs8_sinf(src_tmp);
            v8sf dst_cos_tmp = amd_vrs8_cosf(src_tmp);
            _mm256_store_ps(dst_sin + i, dst_sin_tmp);
            _mm256_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf dst_sin_tmp = amd_vrs8_sinf(src_tmp);
            v8sf dst_cos_tmp = amd_vrs8_cosf(src_tmp);
            _mm256_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm256_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void sincos256d_amdlibm(double *src, double *dst_sin, double *dst_cos, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            v4sd dst_sin_tmp = amd_vrd4_sin(src_tmp);
            v4sd dst_cos_tmp = amd_vrd4_cos(src_tmp);
            _mm256_store_pd(dst_sin + i, dst_sin_tmp);
            _mm256_store_pd(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            v4sd dst_sin_tmp = amd_vrd4_sin(src_tmp);
            v4sd dst_cos_tmp = amd_vrd4_cos(src_tmp);
            _mm256_storeu_pd(dst_sin + i, dst_sin_tmp);
            _mm256_storeu_pd(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst_sin[i] = sin(src[i]);
        dst_cos[i] = cos(src[i]);
    }
}

static inline void exp256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, amd_vrs8_expf(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, amd_vrs8_expf(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

static inline void exp256d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, amd_vrd4_exp(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, amd_vrd4_exp(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = exp(src[i]);
    }
}

static inline void ln256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, amd_vrs8_logf(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, amd_vrs8_logf(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void ln256d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, amd_vrd4_log(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, amd_vrd4_log(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = exp(src[i]);
    }
}

static inline void log2256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, amd_vrs8_log2f(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, amd_vrs8_log2f(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline void log10256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, amd_vrs8_log10f(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, amd_vrs8_log10f(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void atan256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, amd_vrs8_atanf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, amd_vrs8_atanf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline void atan256d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, amd_vrd4_atan(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, amd_vrd4_atan(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan(src[i]);
    }
}

/*
static inline void atan2256f_amdlibm(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_atan2_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_atan2_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}
*/

static inline void asin256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, amd_vrs8_asinf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, amd_vrs8_asinf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}

/*
static inline void asin256d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_asin_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, _mm256_asin_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}
*/

static inline void tan256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, amd_vrs8_tanf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, amd_vrs8_tanf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void tan256d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX_LEN_DOUBLE;
    stop_len *= AVX_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, amd_vrd4_tan(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_DOUBLE) {
            v4sd src_tmp = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, amd_vrd4_tan(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tan(src[i]);
    }
}

/*
static inline void atanh256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_atanh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_atanh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline void acosh256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_acosh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_acosh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline void asinh256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_asinh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_asinh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline void sinh256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_sinh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_sinh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}
*/

static inline void cosh256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, amd_vrs8_coshf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, amd_vrs8_coshf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline void tanh256f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, amd_vrs8_tanhf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, amd_vrs8_tanhf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}

#endif

#ifdef AVX512

static inline void sin512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, amd_vrs16_sinf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, amd_vrs16_sinf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, amd_vrs16_cosf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, amd_vrs16_cosf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos512f_amdlibm(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            v16sf dst_sin_tmp = amd_vrs16_sinf(src_tmp);
            v16sf dst_cos_tmp = amd_vrs16_cosf(src_tmp);
            _mm512_store_ps(dst_sin + i, dst_sin_tmp);
            _mm512_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            v16sf dst_sin_tmp = amd_vrs16_sinf(src_tmp);
            v16sf dst_cos_tmp = amd_vrs16_cosf(src_tmp);
            _mm512_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm512_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void sincos512d_amdlibm(double *src, double *dst_sin, double *dst_cos, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            v8sd dst_sin_tmp = amd_vrd8_sin(src_tmp);
            v8sd dst_cos_tmp = amd_vrd8_cos(src_tmp);
            _mm512_store_pd(dst_sin + i, dst_sin_tmp);
            _mm512_store_pd(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            v8sd dst_sin_tmp = amd_vrd8_sin(src_tmp);
            v8sd dst_cos_tmp = amd_vrd8_cos(src_tmp);
            _mm512_storeu_pd(dst_sin + i, dst_sin_tmp);
            _mm512_storeu_pd(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst_sin[i] = sin(src[i]);
        dst_cos[i] = cos(src[i]);
    }
}

static inline void exp512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, amd_vrs16_expf(_mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, amd_vrs16_expf(_mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

static inline void exp512d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, amd_vrd8_exp(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, amd_vrd8_exp(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = exp(src[i]);
    }
}

static inline void ln512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, amd_vrs16_logf(_mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, amd_vrs16_logf(_mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void ln512d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, amd_vrd8_log(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, amd_vrd8_log(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = exp(src[i]);
    }
}

static inline void log2512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, amd_vrs16_log2f(_mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, amd_vrs16_log2f(_mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline void log10512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, amd_vrs16_log10f(_mm512_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, amd_vrs16_log10f(_mm512_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void atan512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, amd_vrs16_atanf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, amd_vrs16_atanf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline void atan512d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, amd_vrd8_atan(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, amd_vrd8_atan(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan(src[i]);
    }
}

/*
static inline void atan2512f_amdlibm(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_store_ps(dst + i, _mm512_atan2_ps(_mm512_load_ps(src1 + i), _mm512_load_ps(src2 + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            _mm512_storeu_ps(dst + i, _mm512_atan2_ps(_mm512_loadu_ps(src1 + i), _mm512_loadu_ps(src2 + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}*/

static inline void asin512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, amd_vrs16_asinf(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, amd_vrs16_asinf(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}

static inline void asin512d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, amd_vrd8_asin(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, amd_vrd8_asin(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}

/*
static inline void tan512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_tan_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_tan_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void tan512d_amdlibm(double *src, double *dst, int len)
{
    int stop_len = len / AVX512_LEN_DOUBLE;
    stop_len *= AVX512_LEN_DOUBLE;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_load_pd(src + i);
            _mm512_store_pd(dst + i, _mm512_tan_pd(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_DOUBLE) {
            v8sd src_tmp = _mm512_loadu_pd(src + i);
            _mm512_storeu_pd(dst + i, _mm512_tan_pd(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tan(src[i]);
    }
}

static inline void atanh512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_atanh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_atanh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline void acosh512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_acosh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_acosh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline void asinh512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_asinh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_asinh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline void sinh512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_sinh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_sinh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}

static inline void cosh512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_cosh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_cosh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline void tanh512f_amdlibm(float *src, float *dst, int len)
{
    int stop_len = len / AVX512_LEN_FLOAT;
    stop_len *= AVX512_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX512_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_load_ps(src + i);
            _mm512_store_ps(dst + i, _mm512_tanh_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX512_LEN_FLOAT) {
            v16sf src_tmp = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_tanh_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}
*/
#endif
