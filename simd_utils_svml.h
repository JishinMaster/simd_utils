/*
 * Project : SIMD_Utils
 * Version : 0.1.11
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#ifdef SSE
static inline void sin128f_svml(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_sin_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_sin_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos128f_svml(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_cos_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_cos_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos128f_svml(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src), (uintptr_t)(dst_sin), (uintptr_t)(dst_cos), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            dst_sin_tmp = _mm_sincos_ps(&dst_cos_tmp, src_tmp);
            _mm_store_ps(dst_sin + i, dst_sin_tmp);
            _mm_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            dst_sin_tmp = _mm_sincos_ps(&dst_cos_tmp, src_tmp);
            _mm_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void ln_128f_svml(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_store_ps(dst + i, _mm_log_ps(_mm_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            _mm_storeu_ps(dst + i, _mm_log_ps(_mm_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void atan128f_svml(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_atan_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_atan_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline void atan2128f_svml(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), SSE_LEN_BYTES)) {
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

static inline void asin128f_svml(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_asin_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_asin_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}

static inline void tan128f_svml(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), SSE_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_load_ps(src + i);
            _mm_store_ps(dst + i, _mm_tan_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            _mm_storeu_ps(dst + i, _mm_tan_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}
#endif

#ifdef AVX

static inline void sin256f_svml(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_sin_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_sin_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cos256f_svml(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_cos_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_cos_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincos256f_svml(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (((uintptr_t)(const void *) (src) % AVX_LEN_BYTES) == 0) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            v8sf dst_sin_tmp;
            v8sf dst_cos_tmp;
            dst_sin_tmp = _mm256_sincos_ps(&dst_cos_tmp, src_tmp);
            _mm256_store_ps(dst_sin + i, dst_sin_tmp);
            _mm256_store_ps(dst_cos + i, dst_cos_tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            v8sf dst_sin_tmp;
            v8sf dst_cos_tmp;
            dst_sin_tmp = _mm256_sincos_ps(&dst_cos_tmp, src_tmp);
            _mm256_storeu_ps(dst_sin + i, dst_sin_tmp);
            _mm256_storeu_ps(dst_cos + i, dst_cos_tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void ln_256f_svml(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_store_ps(dst + i, _mm256_log_ps(_mm256_load_ps(src + i)));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            _mm256_storeu_ps(dst + i, _mm256_log_ps(_mm256_loadu_ps(src + i)));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void atan256f_svml(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_atan_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_atan_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline void atan2256f_svml(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
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

static inline void asin256f_svml(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_asin_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_asin_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}

static inline void tan256f_svml(float *src, float *dst, int len)
{
    int stop_len = len / AVX_LEN_FLOAT;
    stop_len *= AVX_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_tan_ps(src_tmp));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_FLOAT) {
            v8sf src_tmp = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_tan_ps(src_tmp));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

#endif
