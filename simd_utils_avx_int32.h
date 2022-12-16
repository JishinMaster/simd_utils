/*
 * Project : SIMD_Utils
 * Version : 0.2.4
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <stdint.h>
#include "immintrin.h"

#ifdef __AVX2__
static inline void add256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_add_epi32(_mm256_load_si256((__m256i *) (src1 + i)),
                                                                       _mm256_load_si256((__m256i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_add_epi32(_mm256_loadu_si256((__m256i *) (src1 + i)),
                                                                        _mm256_loadu_si256((__m256i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

#if 0
//Work in progress
static inline void mul256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_mul_epi32(_mm256_load_si256((__m256i *) (src1 + i)), _mm256_load_si256((__m256i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_mul_epi32(_mm256_loadu_si256((__m256i *) (src1 + i)), _mm256_loadu_si256((__m256i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}
#endif

static inline void sub256s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_sub_epi32(_mm256_load_si256((__m256i *) (src1 + i)),
                                                                       _mm256_load_si256((__m256i *) (src2 + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_sub_epi32(_mm256_loadu_si256((__m256i *) (src1 + i)),
                                                                        _mm256_loadu_si256((__m256i *) (src2 + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void addc256s(int32_t *src, int32_t value, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    const v8si tmp = _mm256_set1_epi32(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_add_epi32(tmp, _mm256_load_si256((__m256i *) (src + i))));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_add_epi32(tmp, _mm256_loadu_si256((__m256i *) (src + i))));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void vectorSlope256s(int *dst, int len, int offset, int slope)
{
    v8si coef = _mm256_set_epi32(7 * slope, 6 * slope, 5 * slope, 4 * slope, 3 * slope, 2 * slope, slope, 0);
    v8si slope16_vec = _mm256_set1_epi32(16 * slope);
    v8si curVal = _mm256_add_epi32(_mm256_set1_epi32(offset), coef);
    v8si curVal2 = _mm256_add_epi32(_mm256_set1_epi32(offset), coef);
    curVal2 = _mm256_add_epi32(curVal2, _mm256_set1_epi32(8 * slope));
    int stop_len = len / (2 * AVX_LEN_INT32);
    stop_len *= (2 * AVX_LEN_INT32);

    if (((uintptr_t) (const void *) (dst) % AVX_LEN_BYTES) == 0) {
        _mm256_storeu_si256((__m256i *) (dst + 0), curVal);
        _mm256_storeu_si256((__m256i *) (dst + AVX_LEN_INT32), curVal2);
    } else {
        _mm256_storeu_si256((__m256i *) (dst + 0), curVal);
        _mm256_storeu_si256((__m256i *) (dst + AVX_LEN_INT32), curVal2);
    }

    if (((uintptr_t) (const void *) (dst) % AVX_LEN_BYTES) == 0) {
        for (int i = 2 * AVX_LEN_INT32; i < stop_len; i += 2 * AVX_LEN_INT32) {
            curVal = _mm256_add_epi32(curVal, slope16_vec);
            _mm256_store_si256((__m256i *) (dst + i), curVal);
            curVal2 = _mm256_add_epi32(curVal2, slope16_vec);
            _mm256_store_si256((__m256i *) (dst + i + AVX_LEN_INT32), curVal2);
        }
    } else {
        for (int i = 2 * AVX_LEN_INT32; i < stop_len; i += 2 * AVX_LEN_INT32) {
            curVal = _mm256_add_epi32(curVal, slope16_vec);
            _mm256_storeu_si256((__m256i *) (dst + i), curVal);
            curVal2 = _mm256_add_epi32(curVal2, slope16_vec);
            _mm256_storeu_si256((__m256i *) (dst + i + AVX_LEN_INT32), curVal2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * i;
    }
}

// Experimental

static inline void copy256s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_store_si256((__m256i *) (dst + i), _mm256_load_si256((__m256i *) (src + i)));
        }
    } else {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_loadu_si256((__m256i *) (src + i)));
        }
    }
    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void copy256s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_INT32);
    stop_len *= (2 * AVX_LEN_INT32);


    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_INT32) {
            __m256i tmp1 = _mm256_load_si256((__m256i *) (src + i));
            __m256i tmp2 = _mm256_load_si256((__m256i *) (src + i + AVX_LEN_INT32));
            _mm256_store_si256((__m256i *) (dst + i), tmp1);
            _mm256_store_si256((__m256i *) (dst + i + AVX_LEN_INT32), tmp2);
        }
    } else {
#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_INT32) {
            __m256i tmp1 = _mm256_loadu_si256((__m256i *) (src + i));
            __m256i tmp2 = _mm256_loadu_si256((__m256i *) (src + i + AVX_LEN_INT32));
            _mm256_storeu_si256((__m256i *) (dst + i), tmp1);
            _mm256_storeu_si256((__m256i *) (dst + i + AVX_LEN_INT32), tmp2);
        }
    }


    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy256s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
        _mm256_stream_si256((__m256i *) (dst + i), _mm256_stream_load_si256((__m256i *) (src + i)));
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}


static inline void fast_copy256s_2(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * AVX_LEN_INT32);
    stop_len *= (2 * AVX_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 2 * AVX_LEN_INT32) {
        __m256i tmp1 = _mm256_stream_load_si256((__m256i *) (src + i));
        __m256i tmp2 = _mm256_stream_load_si256((__m256i *) (src + i + AVX_LEN_INT32));
        _mm256_stream_si256((__m256i *) (dst + i), tmp1);
        _mm256_stream_si256((__m256i *) (dst + i + AVX_LEN_INT32), tmp2);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void fast_copy256s_4(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (4 * AVX_LEN_INT32);
    stop_len *= (4 * AVX_LEN_INT32);

#ifdef OMP
#pragma omp parallel for schedule(auto)
#endif
    for (int i = 0; i < stop_len; i += 4 * AVX_LEN_INT32) {
        __m256i tmp1 = _mm256_stream_load_si256((__m256i *) (src + i));
        __m256i tmp2 = _mm256_stream_load_si256((__m256i *) (src + i + AVX_LEN_INT32));
        __m256i tmp3 = _mm256_stream_load_si256((__m256i *) (src + i + 2 * AVX_LEN_INT32));
        __m256i tmp4 = _mm256_stream_load_si256((__m256i *) (src + i + 3 * AVX_LEN_INT32));
        _mm256_stream_si256((__m256i *) (dst + i), tmp1);
        _mm256_stream_si256((__m256i *) (dst + i + AVX_LEN_INT32), tmp2);
        _mm256_stream_si256((__m256i *) (dst + i + 2 * AVX_LEN_INT32), tmp3);
        _mm256_stream_si256((__m256i *) (dst + i + 3 * AVX_LEN_INT32), tmp4);
    }
    _mm_mfence();

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline __m256i _mm256_absdiff_epi16(__m256i a, __m256i b)
{
    __m256i cmp, difab, difba;
    cmp = _mm256_cmpgt_epi16(a, b);
    difab = _mm256_sub_epi16(a, b);
    difba = _mm256_sub_epi16(b, a);
    difab = _mm256_and_si256(cmp, difab);
    difba = _mm256_andnot_si256(cmp, difba);
    return _mm256_or_si256(difab, difba);
}

static inline __m256i _mm256_absdiff_epi32(__m256i a, __m256i b)
{
    __m256i cmp, difab, difba;
    cmp = _mm256_cmpgt_epi32(a, b);
    difab = _mm256_sub_epi32(a, b);
    difba = _mm256_sub_epi32(b, a);
    difab = _mm256_and_si256(cmp, difab);
    difba = _mm256_andnot_si256(cmp, difba);
    return _mm256_or_si256(difab, difba);
}

static inline __m256i _mm256_absdiff_epi8(__m256i a, __m256i b)
{
    __m256i cmp, difab, difba;
    cmp = _mm256_cmpgt_epi8(a, b);
    difab = _mm256_sub_epi8(a, b);
    difba = _mm256_sub_epi8(b, a);
    difab = _mm256_and_si256(cmp, difab);
    difba = _mm256_andnot_si256(cmp, difba);
    return _mm256_or_si256(difab, difba);
}

static inline void absdiff16s_256s(int16_t *src1, int16_t *src2, int16_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT16;
    stop_len *= AVX_LEN_INT16;


    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT16) {
            __m256i a = _mm256_load_si256((__m256i *) (src1 + i));
            __m256i b = _mm256_load_si256((__m256i *) (src2 + i));
            _mm256_store_si256((__m256i *) (dst + i), _mm256_absdiff_epi16(a, b));
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT16) {
            __m256i a = _mm256_loadu_si256((__m256i *) (src1 + i));
            __m256i b = _mm256_loadu_si256((__m256i *) (src2 + i));
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_absdiff_epi16(a, b));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = abs(src1[i] - src2[i]);
    }
}

static inline void powerspect16s_256s_interleaved(complex16s_t *src, int32_t *dst, int len)
{
    int stop_len = len / AVX_LEN_INT32;
    stop_len *= AVX_LEN_INT32;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            __m256i reim = _mm256_load_si256((__m256i *) ((const int16_t *) src + j));
            // print8i(reim); printf("\n");
            _mm256_store_si256((__m256i *) (dst + i), _mm256_madd_epi16(reim, reim));
            j += AVX_LEN_INT16;
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            __m256i reim = _mm256_loadu_si256((__m256i *) ((const int16_t *) src + j));
            _mm256_storeu_si256((__m256i *) (dst + i), _mm256_madd_epi16(reim, reim));
            j += AVX_LEN_INT16;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (int32_t) src[i].re * (int32_t) src[i].re + (int32_t) src[i].im * (int32_t) src[i].im;
    }
}

static inline void sum16s32s256(int16_t *src, int len, int32_t *dst, int scale_factor)
{
    int stop_len = len / (4 * AVX_LEN_INT16);
    stop_len *= (4 * AVX_LEN_INT16);

    __attribute__((aligned(AVX_LEN_BYTES))) int32_t accumulate[AVX_LEN_INT32] = {0, 0, 0, 0,\
                                                                                 0, 0, 0, 0};
    int32_t tmp_acc = 0;
    int16_t scale = 1 << scale_factor;
    v8si one = _mm256_set1_epi16(1);
    v8si vec_acc1 = _mm256_setzero_si256();  // initialize the vector accumulator
    v8si vec_acc2 = _mm256_setzero_si256();  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_INT16) {
            v8si vec_src_tmp = _mm256_load_si256((__m256i *) ((const int16_t *)src + i));
            v8si vec_src_tmp2 = _mm256_load_si256((__m256i *) ((const int16_t *)src + i + AVX_LEN_INT16));
            v8si vec_src_tmp3 = _mm256_load_si256((__m256i *) ((const int16_t *)src + i + 2*AVX_LEN_INT16));
            v8si vec_src_tmp4 = _mm256_load_si256((__m256i *) ((const int16_t *)src + i + 3*AVX_LEN_INT16));
            vec_src_tmp = _mm256_madd_epi16(vec_src_tmp, one);
            vec_src_tmp2 = _mm256_madd_epi16(vec_src_tmp2, one);
            vec_src_tmp3 = _mm256_madd_epi16(vec_src_tmp3, one);
            vec_src_tmp4 = _mm256_madd_epi16(vec_src_tmp4, one);
            vec_src_tmp = _mm256_add_epi32(vec_src_tmp, vec_src_tmp2);
            vec_src_tmp3 = _mm256_add_epi32(vec_src_tmp3, vec_src_tmp4);
            vec_acc1 = _mm256_add_epi32(vec_src_tmp, vec_acc1);
            vec_acc2 = _mm256_add_epi32(vec_src_tmp3, vec_acc2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * AVX_LEN_INT16) {
            v8si vec_src_tmp = _mm256_loadu_si256((__m256i *) ((const int16_t *)src + i));
            v8si vec_src_tmp2 = _mm256_loadu_si256((__m256i *) ((const int16_t *)src + i + AVX_LEN_INT16));
            v8si vec_src_tmp3 = _mm256_loadu_si256((__m256i *) ((const int16_t *)src + i + 2*AVX_LEN_INT16));
            v8si vec_src_tmp4 = _mm256_loadu_si256((__m256i *) ((const int16_t *)src + i + 3*AVX_LEN_INT16));
            vec_src_tmp = _mm256_madd_epi16(vec_src_tmp, one);
            vec_src_tmp2 = _mm256_madd_epi16(vec_src_tmp2, one);
            vec_src_tmp3 = _mm256_madd_epi16(vec_src_tmp3, one);
            vec_src_tmp4 = _mm256_madd_epi16(vec_src_tmp4, one);
            vec_src_tmp = _mm256_add_epi32(vec_src_tmp, vec_src_tmp2);
            vec_src_tmp3 = _mm256_add_epi32(vec_src_tmp3, vec_src_tmp4);
            vec_acc1 = _mm256_add_epi32(vec_src_tmp, vec_acc1);
            vec_acc2 = _mm256_add_epi32(vec_src_tmp3, vec_acc2);
        }
    }
    
    vec_acc1 = _mm256_add_epi32(vec_acc1, vec_acc2);
    _mm256_store_si256((v8si*)accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += (int32_t)src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] +\
                        accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];
    tmp_acc /= scale;
    *dst = tmp_acc;
}


// no cmplt_epi32 with AVX2. gt(b,a) = le(a,b), not lt. To be improved?
static inline void threshold256_ltval_gtval_s(int32_t *src, int32_t *dst, int len, int32_t ltlevel, int32_t ltvalue, int32_t gtlevel, int32_t gtvalue)
{
    const v8si ltlevel_v = _mm256_set1_epi32(ltlevel);
    const v8si ltvalue_v = _mm256_set1_epi32(ltvalue);
    const v8si gtlevel_v = _mm256_set1_epi32(gtlevel);
    const v8si gtvalue_v = _mm256_set1_epi32(gtvalue);

    int stop_len = len / (2 * AVX_LEN_FLOAT);
    stop_len *= (2 * AVX_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_INT32) {
            v8si src_tmp = _mm256_load_si256((v8si *) (src + i));
            v8si src_tmp2 = _mm256_load_si256((v8si *) (src + i + AVX_LEN_INT32));
            v8si lt_mask = _mm256_cmpgt_epi32(ltlevel_v, src_tmp);
            v8si gt_mask = _mm256_cmpgt_epi32(src_tmp, gtlevel_v);
            v8si dst_tmp = _mm256_blendv_epi8(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm256_blendv_epi8(dst_tmp, gtvalue_v, gt_mask);
            _mm256_store_si256((v8si*)(dst + i), dst_tmp);
            v8si lt_mask2 = _mm256_cmpgt_epi32(ltlevel_v, src_tmp2);
            v8si gt_mask2 = _mm256_cmpgt_epi32(src_tmp2, gtlevel_v);
            v8si dst_tmp2 = _mm256_blendv_epi8(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm256_blendv_epi8(dst_tmp2, gtvalue_v, gt_mask2);
            _mm256_store_si256((v8si*)(dst + i + AVX_LEN_INT32), dst_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 2 * AVX_LEN_INT32) {
            v8si src_tmp = _mm256_loadu_si256((v8si *) (src + i));
            v8si src_tmp2 = _mm256_loadu_si256((v8si *) (src + i + AVX_LEN_INT32));
            v8si lt_mask = _mm256_cmpgt_epi32(ltlevel_v, src_tmp);
            v8si gt_mask = _mm256_cmpgt_epi32(src_tmp, gtlevel_v);
            v8si dst_tmp = _mm256_blendv_epi8(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = _mm256_blendv_epi8(dst_tmp, gtvalue_v, gt_mask);
            _mm256_storeu_si256((v8si*)(dst + i), dst_tmp);
            v8si lt_mask2 = _mm256_cmpgt_epi32(ltlevel_v, src_tmp2);
            v8si gt_mask2 = _mm256_cmpgt_epi32(src_tmp2, gtlevel_v);
            v8si dst_tmp2 = _mm256_blendv_epi8(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = _mm256_blendv_epi8(dst_tmp2, gtvalue_v, gt_mask2);
            _mm256_storeu_si256((v8si*)(dst + i + AVX_LEN_INT32), dst_tmp2);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}

// is it useful to unroll?
static inline void gatheri_256s(int32_t *src, int32_t *dst, int stride, int offset, int len)
{
    int stop_len = len / (AVX_LEN_INT32);
    stop_len *= (AVX_LEN_INT32);

    v8si vindex = _mm256_setr_epi32(offset, stride + offset, 2 * stride + offset, 3 * stride + offset,
                                    4 * stride + offset, 5 * stride + offset, 6 * stride + offset, 7 * stride + offset);

    if (isAligned((uintptr_t) (dst), AVX_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            v8si tmp = _mm256_i32gather_epi32((const int *) (src + i * AVX_LEN_INT32), vindex, 1);
            _mm256_store_si256((v8si *) (dst + i * AVX_LEN_INT32), tmp);
        }
    } else {
        for (int i = 0; i < stop_len; i += AVX_LEN_INT32) {
            v8si tmp = _mm256_i32gather_epi32((const int *) (src + i * AVX_LEN_INT32), vindex, 1);
            _mm256_storeu_si256((v8si *) (dst + i * AVX_LEN_INT32), tmp);
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i * stride + offset];
    }
}

#endif
