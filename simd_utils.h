/*
 * Project : SIMD_Utils
 * Version : 0.2.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define MAJOR_VERSION 0
#define MINOR_VERSION 2
#define SUB_VERSION 1

#ifdef OMP
#include <omp.h>
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "simd_utils_constants.h"

#include "mysincosf.h"

/* if the user insures that all of their pointers are aligned,
 * they can use ALWAYS_ALIGNED to hope for some minor speedup on small vectors
 */
static inline int isAligned(uintptr_t ptr, size_t alignment)
{
#ifndef ALWAYS_ALIGNED

#ifndef ARM  // ARM manages disalignment in hardware
    if (((uintptr_t) (ptr) % alignment) == 0)
        return 1;
    return 0;
#else
    return 1;
#endif

#else
    return 1;
#endif
}

static inline int areAligned2(uintptr_t ptr1, uintptr_t ptr2, size_t alignment)
{
#ifndef ALWAYS_ALIGNED

#ifndef ARM  // ARM manages disalignment in hardware
    if (((uintptr_t) (ptr1) % alignment) == 0)
        if (((uintptr_t) (ptr2) % alignment) == 0)
            return 1;
    return 0;
#else
    return 1;
#endif

#else
    return 1;
#endif
}

static inline int areAligned3(uintptr_t ptr1, uintptr_t ptr2, uintptr_t ptr3, size_t alignment)
{
#ifndef ALWAYS_ALIGNED

#ifndef ARM  // ARM manages disalignment in hardware
    if (((uintptr_t) (ptr1) % alignment) == 0)
        if (((uintptr_t) (ptr2) % alignment) == 0)
            if (((uintptr_t) (ptr3) % alignment) == 0)
                return 1;
    return 0;
#else
    return 1;
#endif

#else
    return 1;
#endif
}


static inline void simd_utils_get_version(void)
{
    printf("Simd Utils Version : %d.%d.%d\n", MAJOR_VERSION, MINOR_VERSION, SUB_VERSION);
}

#ifdef SSE

// For X86 devices with only SSE2
#ifdef NO_SSE3
static inline __m128 _mm_movehdup_ps (__m128 __X)
{
    return _mm_shuffle_ps (__X, __X, 0xF5);
}

static inline __m128 _mm_moveldup_ps (__m128 __X)
{
    return _mm_shuffle_ps (__X, __X, 0xA0);
}
#endif

// For X86 devices with only SSE2 and SSE3 (before CoreI7)
#ifdef NO_SSE4
static inline __m128i _mm_cmpeq_epi64(__m128i __X, __m128i __Y)
{
    int64_t* ptr_x = (int64_t*)&__X;
    int64_t* ptr_y = (int64_t*)&__Y;
    __m128i ret;
    int64_t* ptr_ret = (int64_t*)&ret;
    
    ptr_ret[0] = (ptr_x[0] == ptr_y[0])? 0xFFFFFFFFFFFFFFFF : 0;
    ptr_ret[1] = (ptr_x[1] == ptr_y[1])? 0xFFFFFFFFFFFFFFFF : 0;
    return ret;
}

static inline __m128d _mm_blendv_pd(__m128d __X, __m128d __Y, __m128d __M)
{
    __m128d b_tmp = _mm_and_pd(__Y, __M);
    __m128d a_tmp = _mm_and_pd(__X, _mm_cmpeq_pd(__M,*(__m128d *)_pd_zero));  
    return _mm_or_pd(a_tmp, b_tmp);
}

static inline __m128 _mm_blendv_ps(__m128 __X, __m128 __Y, __m128 __M)
{
    __m128 b_tmp = _mm_and_ps(__Y, __M);
    __m128 a_tmp = _mm_and_ps(__X, _mm_cmpeq_ps(__M,*(__m128 *)_ps_zero));  
    return _mm_or_ps(a_tmp, b_tmp);
}

static inline __m128i _mm_stream_load_si128(__m128i *__X)
{
    return _mm_load_si128(__X);
}

static inline __m128 _mm_round_ps(__m128 X, int mode)
{
__m128 ret;
__m128i reti;
unsigned int old_mode = _MM_GET_ROUNDING_MODE();
switch(mode){
    case _MM_FROUND_TRUNC:
    case _MM_ROUND_TOWARD_ZERO:
    case ROUNDTOZERO:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        break;    
    case ROUNDTOCEIL:
    case _MM_ROUND_UP:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        break;    
    case ROUNDTOFLOOR:
    case _MM_ROUND_DOWN:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        break;
    default:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        break;
}
reti = _mm_cvtps_epi32(X);
ret = _mm_cvtepi32_ps(reti);
_MM_SET_ROUNDING_MODE(old_mode);
return ret;
}
#endif
  
#ifndef ARM
#include "sse_mathfun.h"
#else /* ARM */
#include "neon_mathfun.h"
#endif /* ARM */

static inline v4sfx2 _mm_load2_ps(float const *mem_addr)
{
#ifdef ARM
    return vld2q_f32(mem_addr);
#else
    v4sf tmp1 = _mm_load_ps(mem_addr);
    v4sf tmp2 = _mm_load_ps(mem_addr + SSE_LEN_FLOAT);
    v4sfx2 ret;
    ret.val[0] = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(2, 0, 2, 0));
    ret.val[1] = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(3, 1, 3, 1));
    return ret;
#endif
}

static inline v4sfx2 _mm_load2u_ps(float const *mem_addr)
{
#ifdef ARM
    return vld2q_f32(mem_addr);
#else
    v4sf tmp1 = _mm_loadu_ps(mem_addr);
    v4sf tmp2 = _mm_loadu_ps(mem_addr + SSE_LEN_FLOAT);
    v4sfx2 ret;
    ret.val[0] = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(2, 0, 2, 0));
    ret.val[1] = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(3, 1, 3, 1));
    return ret;
#endif
}

static inline void _mm_store2_ps(float *mem_addr, v4sfx2 a)
{
#ifdef ARM
    vst2q_f32(mem_addr, a);
#else
    v4sf tmp1 = _mm_unpacklo_ps(a.val[0], a.val[1]);
    v4sf tmp2 = _mm_unpackhi_ps(a.val[0], a.val[1]);
    _mm_store_ps(mem_addr, tmp1);
    _mm_store_ps(mem_addr + SSE_LEN_FLOAT, tmp2);
#endif
}

static inline void _mm_store2u_ps(float *mem_addr, v4sfx2 a)
{
#ifdef ARM
    vst2q_f32(mem_addr, a);
#else
    v4sf tmp1 = _mm_unpacklo_ps(a.val[0], a.val[1]);
    v4sf tmp2 = _mm_unpackhi_ps(a.val[0], a.val[1]);
    _mm_storeu_ps(mem_addr, tmp1);
    _mm_storeu_ps(mem_addr + SSE_LEN_FLOAT, tmp2);
#endif
}

static inline __m128 _mm_fmadd_ps_custom(__m128 a, __m128 b, __m128 c)
{
// Haswell comes with avx2 and fma
//  ARM has vmla instead of fma in 32bits
#if defined(ARM) || defined(FMA)
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

static inline __m128 _mm_fmaddsub_ps_custom(__m128 a, __m128 b, __m128 c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm_addsub_ps(_mm_mul_ps(a, b), c);
#else  /* FMA */
    return _mm_fmaddsub_ps(a, b, c);
#endif /* FMA */
}

static inline __m128 _mm_fmsubadd_ps_custom(__m128 a, __m128 b, __m128 c)
{
#if !defined(FMA) || defined(ARM)
    v4sf d = _mm_mul_ps(*(v4sf *) _ps_conj_mask, c);
    return _mm_addsub_ps(_mm_mul_ps(a, b), d);
#else  /* FMA */
    return _mm_fmsubadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m128 _mm_fnmadd_ps_custom(__m128 a, __m128 b, __m128 c)
{
// Haswell comes with avx2 and fma
//  ARM has vmla instead of fma in 32bits
#if defined(ARM) || defined(FMA)
    return _mm_fnmadd_ps(a, b, c);
#else
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
#endif
}

static inline __m128d _mm_fmadd_pd_custom(__m128d a, __m128d b, __m128d c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm_add_pd(_mm_mul_pd(a, b), c);
#else  /* FMA */
    return _mm_fmadd_pd(a, b, c);
#endif /* FMA */
}

static inline __m128d _mm_fnmadd_pd_custom(__m128d a, __m128d b, __m128d c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm_sub_pd(c, _mm_mul_pd(a, b));
#else  /* FMA */
    return _mm_fnmadd_pd(a, b, c);
#endif /* FMA */
}

#include "simd_utils_sse_double.h"

#include "simd_utils_sse_float.h"
#include "simd_utils_sse_int32.h"

#endif /* SSE */

#ifdef AVX

#ifndef __clang__
#ifndef __INTEL_COMPILER
#ifndef __cplusplus                                       // TODO : it seems to be defined with G++ 9.2 and not GCC 9.2
static inline __m256 _mm256_set_m128(__m128 H, __m128 L)  // not present on every GCC version
{
    return _mm256_insertf128_ps(_mm256_castps128_ps256(L), H, 1);
}
#endif
#endif
#endif /* __clang__ */

static inline __m256 _mm256_fmadd_ps_custom(__m256 a, __m256 b, __m256 c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#else  /* FMA */
    return _mm256_fmadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m256 _mm256_fmaddsub_ps_custom(__m256 a, __m256 b, __m256 c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm256_addsub_ps(_mm256_mul_ps(a, b), c);
#else  /* FMA */
    return _mm256_fmaddsub_ps(a, b, c);
#endif /* FMA */
}

static inline __m256 _mm256_fnmadd_ps_custom(__m256 a, __m256 b, __m256 c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#else  /* FMA */
    return _mm256_fnmadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m256d _mm256_fmadd_pd_custom(__m256d a, __m256d b, __m256d c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#else  /* FMA */
    return _mm256_fmadd_pd(a, b, c);
#endif /* FMA */
}

static inline __m256d _mm256_fnmadd_pd_custom(__m256d a, __m256d b, __m256d c)
{
#ifndef FMA  // Haswell comes with avx2 and fma
    return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
#else  /* FMA */
    return _mm256_fnmadd_pd(a, b, c);
#endif /* FMA */
}

#include "avx_mathfun.h"

static inline v8sfx2 _mm256_load2_ps(float const *mem_addr)
{
    v4sfx2 src_1 = _mm_load2_ps(mem_addr);
    v4sfx2 src_2 = _mm_load2_ps(mem_addr + 2 * SSE_LEN_FLOAT);
    v8sfx2 ret;
    ret.val[0] = _mm256_set_m128(src_2.val[0], src_1.val[0]);
    ret.val[1] = _mm256_set_m128(src_2.val[1], src_1.val[1]);
    return ret;
}

static inline v8sfx2 _mm256_load2u_ps(float const *mem_addr)
{
    v4sfx2 src_1 = _mm_load2u_ps(mem_addr);
    v4sfx2 src_2 = _mm_load2u_ps(mem_addr + 2 * SSE_LEN_FLOAT);
    v8sfx2 ret;
    ret.val[0] = _mm256_set_m128(src_2.val[0], src_1.val[0]);
    ret.val[1] = _mm256_set_m128(src_2.val[1], src_1.val[1]);
    return ret;
}

#include "simd_utils_avx_double.h"
#include "simd_utils_avx_float.h"
#include "simd_utils_avx_int32.h"

#endif /* AVX */

#ifdef AVX512

static const float _ps512_conj_mask[16] __attribute__((aligned(64))) = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
                                                                        1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};

static inline __m512 _mm512_fmadd_ps_custom(__m512 a, __m512 b, __m512 c)
{
    return _mm512_fmadd_ps(a, b, c);
}

static inline __m512 _mm512_fmaddsub_ps_custom(__m512 a, __m512 b, __m512 c)
{
    return _mm512_fmaddsub_ps(a, b, c);
}

static inline __m512 _mm512_fnmadd_ps_custom(__m512 a, __m512 b, __m512 c)
{
    return _mm512_fnmadd_ps(a, b, c);
}

static inline __m512d _mm512_fmadd_pd_custom(__m512d a, __m512d b, __m512d c)
{
    return _mm512_fmadd_pd(a, b, c);
}

static inline __m512d _mm512_fnmadd_pd_custom(__m512d a, __m512d b, __m512d c)
{
    return _mm512_fnmadd_pd(a, b, c);
}

#include "avx512_mathfun.h"

#include "simd_utils_avx512_double.h"
#include "simd_utils_avx512_float.h"
#include "simd_utils_avx512_int32.h"

#endif /* AVX512 */

#ifdef ICC
#include "simd_utils_svml.h"
#endif

#ifdef RISCV /* RISCV */
#include "simd_utils_riscv.h"
#endif /* RISCV */

#ifdef ALTIVEC
#include "simd_utils_altivec_float.h"
#endif /* ALTIVEC */

#ifdef CUSTOM_MALLOC
// Thanks to Jpommier pfft https://bitbucket.org/jpommier/pffft/src/default/pffft.c
static inline int posix_memalign(void **pointer, size_t len, int alignement)
{
    void *p, *p0 = malloc(len + alignement);
    if (!p0)
        return (void *) NULL;
    p = (void *) (((size_t) p0 + alignement) & (~((size_t) (alignement - 1))));
    *((void **) p - 1) = p0;

    *pointer = p;
    return 0;
}

static inline void *aligned_malloc(size_t len, int alignement)
{
    void *p, *p0 = malloc(len + alignement);
    if (!p0)
        return (void *) NULL;
    p = (void *) (((size_t) p0 + alignement) & (~((size_t) (alignement - 1))));
    *((void **) p - 1) = p0;
    return p;
}

// Work in progress
static inline void aligned_free(void *p)
{
    if (p)
        free(*((void **) p - 1));
}

#endif /* CUSTOM_MALLOC */


//////////  C Test functions ////////////////
static inline void log10f_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++)
        dst[i] = log10f(src[i]);
}

static inline void log2f_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++)
        dst[i] = log2f(src[i]);
}

static inline void lnf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++)
        dst[i] = logf(src[i]);
}

static inline void expf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

static inline void fabsf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void setf_C(float *dst, float value, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zerof_C(float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = 0.0f;
    }
}


static inline void copyf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}


static inline void addcf_C(float *src, float value, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void addcs_C(int32_t *src, int32_t value, int32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void mulf_C(float *src1, float *src2, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void mulcf_C(float *src, float value, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladdf_C(float *_a, float *_b, float *_c, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (_a[i] * _b[i]) + _c[i];
    }
}

static inline void mulcaddf_C(float *_a, float _b, float *_c, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (_a[i] * _b) + _c[i];
    }
}

static inline void mulcaddcf_C(float *_a, float _b, float _c, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (_a[i] * _b) + _c;
    }
}

static inline void muladdcf_C(float *_a, float *_b, float _c, float *dst, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

static inline void muls_c(int32_t *a, int32_t *b, int32_t *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] * b[i];
    }
}

static inline void divf_C(float *src1, float *src2, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}

static inline void cplxtorealf_C(complex32_t *src, float *dstRe, float *dstIm, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
}


static inline void realtocplx_C(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
}

static inline void convert_64f32f_C(double *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (float) src[i];
    }
}

static inline void convert_32f64f_C(float *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (double) src[i];
    }
}

static inline void convertFloat32ToU8_C(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);

    if (rounding_mode == RndZero) {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = floorf(src[i] * scale_fact_mult);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);
        }
    } else if (rounding_mode == RndNear) {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = roundf(src[i] * scale_fact_mult);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);
        }
    } else if (rounding_mode == RndFinancial) {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);
        }
    } else {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = src[i] * scale_fact_mult;
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);
        }
    }
}


static inline void convertFloat32ToI16_C(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);

    // Default bankers rounding => round to nearest even
    if (rounding_mode == RndFinancial) {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (int16_t) (tmp > 32767.0f ? 32767.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        if (rounding_mode == RndZero) {
            fesetround(FE_TOWARDZERO);
        } else {
            fesetround(FE_TONEAREST);
        }

        // Default round toward zero
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (uint16_t) (tmp > 32767.0f ? 32767.0f : tmp);
        }
    }
}

static inline void convertInt16ToFloat32_C(int16_t *src, float *dst, int len, int scale_factor)
{
    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);

#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}

static inline void threshold_gt_f_C(float *src, float *dst, int len, float value)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold_gtabs_f_C(float *src, float *dst, int len, float value)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        if (src[i] >= 0.0f) {
            dst[i] = src[i] > value ? value : src[i];
        } else {
            dst[i] = src[i] < (-value) ? (-value) : src[i];
        }
    }
}

static inline void threshold_lt_f_C(float *src, float *dst, int len, float value)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

static inline void threshold_ltabs_f_C(float *src, float *dst, int len, float value)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        if (src[i] >= 0.0f) {
            dst[i] = src[i] < value ? value : src[i];
        } else {
            dst[i] = src[i] > (-value) ? (-value) : src[i];
        }
    }
}

static inline void threshold_ltval_gtval_f_C(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = dst[i] > gtlevel ? gtvalue : dst[i];
    }
}

static inline void magnitudef_C_interleaved(complex32_t *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = sqrtf(src[i].re * src[i].re + (src[i].im * src[i].im));
    }
}

static inline void magnitudef_C_split(float *srcRe, float *srcIm, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]));
    }
}


static inline void powerspectf_C_split(float *srcRe, float *srcIm, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]);
    }
}

static inline void powerspectf_C_interleaved(complex32_t *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i].re * src[i].re + (src[i].im * src[i].im);
    }
}

static inline void meanf_C(float *src, float *dst, int len)
{
    float acc = 0.0f;
    int i;

#ifdef OMP
#pragma omp simd reduction(+ \
                           : acc)
#endif
    for (i = 0; i < len; i++) {
        acc += src[i];
    }

    acc = acc / (float) len;
    *dst = acc;
}

static inline void sumf_C(float *src, float *dst, int len)
{
    float tmp_acc = 0.0f;

    for (int i = 0; i < len; i++) {
        tmp_acc += src[i];
    }
    *dst = tmp_acc;
}

static inline void maxlocf_C(float *src, float *max, int *idx, int len)
{
    float max_val = src[0];
    int i;
    int max_idx;
    for (i = 1; i < len; i++) {
        if (src[i] > max_val) {
            max_val = src[i];
            max_idx = i;
        }
    }
    *idx = max_idx;
    *max = max_val;
}

static inline void flipf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void asinf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}

static inline void asin_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = asin(src[i]);
    }
}

static inline void tanf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void tanhf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}

static inline void sinhf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}

static inline void coshf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline void atanhf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline void asinhf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline void acoshf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline void atan_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = atan(src[i]);
    }
}

static inline void atanf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

static inline void atan2f_C(float *src1, float *src2, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}

static inline void atan2f_interleaved_C(complex32_t *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = atan2f(src[i].im, src[i].re);
    }
}



static inline void sinf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = sinf(src[i]);
    }
}

static inline void cosf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = cosf(src[i]);
    }
}

static inline void sincosf_C(float *src, float *dst_sin, float *dst_cos, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void sincosd_C(double *src, double *dst_sin, double *dst_cos, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst_sin[i] = sin(src[i]);
        dst_cos[i] = cos(src[i]);
    }
}

static inline void sqrtf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static inline void floorf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}

static inline void ceilf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void roundf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void truncf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

static inline void floord_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = floor(src[i]);
    }
}

static inline void ceild_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = ceil(src[i]);
    }
}

static inline void roundd_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = round(src[i]);
    }
}

static inline void truncd_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = trunc(src[i]);
    }
}

static inline void cplxvecdiv_C(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        float c2d2 = src2[i].re * src2[i].re + src2[i].im * src2[i].im;
        dst[i].re = (src1[i].re * src2[i].re + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-src1[i].re * src2[i].im + (src2[i].re * src1[i].im)) / c2d2;
    }
}


static inline void cplxvecmul_C(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxvecmul_C_unrolled8(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / 8;
    stop_len *= 8;
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < stop_len; i += 8) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
        dst[i + 1].re = (src1[i + 1].re * src2[i + 1].re) - src1[i + 1].im * src2[i + 1].im;
        dst[i + 1].im = src1[i + 1].re * src2[i + 1].im + (src2[i + 1].re * src1[i + 1].im);
        dst[i + 2].re = (src1[i + 2].re * src2[i + 2].re) - src1[i + 2].im * src2[i + 2].im;
        dst[i + 2].im = src1[i + 2].re * src2[i + 2].im + (src2[i + 2].re * src1[i + 2].im);
        dst[i + 3].re = (src1[i + 3].re * src2[i + 3].re) - src1[i + 3].im * src2[i + 3].im;
        dst[i + 3].im = src1[i + 3].re * src2[i + 3].im + (src2[i + 3].re * src1[i + 3].im);
        dst[i + 4].re = (src1[i + 4].re * src2[i + 4].re) - src1[i + 4].im * src2[i + 4].im;
        dst[i + 4].im = src1[i + 4].re * src2[i + 4].im + (src2[i + 4].re * src1[i + 4].im);
        dst[i + 5].re = (src1[i + 5].re * src2[i + 5].re) - src1[i + 5].im * src2[i + 5].im;
        dst[i + 5].im = src1[i + 5].re * src2[i + 5].im + (src2[i + 5].re * src1[i + 5].im);
        dst[i + 6].re = (src1[i + 6].re * src2[i + 6].re) - src1[i + 6].im * src2[i + 6].im;
        dst[i + 6].im = src1[i + 6].re * src2[i + 6].im + (src2[i + 6].re * src1[i + 6].im);
        dst[i + 7].re = (src1[i + 7].re * src2[i + 7].re) - src1[i + 7].im * src2[i + 7].im;
        dst[i + 7].im = src1[i + 7].re * src2[i + 7].im + (src2[i + 7].re * src1[i + 7].im);
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxvecmul_C2(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = (float) ((double) src1[i].re * (double) src2[i].re - (double) src1[i].im * (double) src2[i].im);
        dst[i].im = (float) ((double) src1[i].re * (double) src2[i].im + (double) src2[i].re * (double) src1[i].im);
    }
}

static inline void cplxvecmul_C_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dstRe[i] = (src1Re[i] * src2Re[i]) - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i]);
    }
}


static inline void cplxconjvecmul_C(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + (src1[i].im * src2[i].im);
        dst[i].im = (src2[i].re * src1[i].im) - src1[i].re * src2[i].im;
    }
}

static inline void cplxconjvecmul_C2(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = (float) ((double) src1[i].re * (double) src2[i].re + (double) src1[i].im * (double) src2[i].im);
        dst[i].im = (float) ((double) src2[i].re * (double) src1[i].im - (double) src1[i].re * (double) src2[i].im);
    }
}

static inline void cplxconjvecmul_C_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i]);
        dstIm[i] = (src2Re[i] * src1Im[i]) - src1Re[i] * src2Im[i];
    }
}

static inline void cplxconjvecmul_C_split2(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dstRe[i] = (float) ((double) src1Re[i] * (double) src2Re[i] + (double) src1Im[i] * (double) src2Im[i]);
        dstIm[i] = (float) ((double) src2Re[i] * (double) src1Im[i] - (double) src1Re[i] * (double) src2Im[i]);
    }
}

static inline void cplxconj_C(complex32_t *src, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = src[i].re;
        dst[i].im = -src[i].im;
    }
}

static inline void vectorSlopef_C(float *dst, int len, float offset, float slope)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (float) i * slope + offset;
    }
}

static inline void vectorSloped_C(double *dst, int len, double offset, double slope)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (double) i * slope + offset;
    }
}

static inline void vectorSlopes_C(int32_t *dst, int len, int32_t offset, int32_t slope)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (int32_t) i * slope + offset;
    }
}

static inline void maxeveryf_c(float *src1, float *src2, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

static inline void mineveryf_c(float *src1, float *src2, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void minmaxf_c(float *src, int len, float *min_value, float *max_value)
{
    float min_tmp = src[0];
    float max_tmp = src[0];

#ifdef OMP
#pragma omp simd
#endif
    for (int i = 1; i < len; i++) {
        max_tmp = max_tmp > src[i] ? max_tmp : src[i];
        min_tmp = min_tmp < src[i] ? min_tmp : src[i];
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

static inline void addf_c(float *a, float *b, float *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}

static inline void adds_c(int32_t *a, int32_t *b, int32_t *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}


static inline void subf_c(float *a, float *b, float *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] - b[i];
    }
}

static inline void subcrevf_C(float *src, float value, float *dst, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = value - src[i];
    }
}

static inline void subs_c(int32_t *a, int32_t *b, int32_t *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] - b[i];
    }
}

/*static inline void orf_c(float *a, float *b, float *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] | b[i];
    }
}*/


static inline void setd_C(double *dst, double value, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zerod_C(double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = 0.0;
    }
}


static inline void copyd_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void copys_C(int32_t *src, int32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void sqrtd_C(double *src, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = sqrt(src[i]);
    }
}

static inline void addd_c(double *a, double *b, double *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}

static inline void muld_c(double *a, double *b, double *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] * b[i];
    }
}

static inline void subd_c(double *a, double *b, double *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] - b[i];
    }
}

static inline void divd_c(double *a, double *b, double *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] / b[i];
    }
}

static inline void mulcd_C(double *src, double value, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void muladdd_C(double *_a, double *_b, double *_c, double *dst, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcaddd_C(double *_a, double _b, double *_c, double *dst, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddcd_C(double *_a, double _b, double _c, double *dst, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
    }
}

static inline void muladdcd_C(double *_a, double *_b, double _c, double *dst, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = _a[i] * _b[i] + _c;
    }
}

static inline void addcd_C(double *src, double value, double *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void ors_C(int32_t *a, int32_t *b, int32_t *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] | b[i];
    }
}

/*static inline void andf_C(float *a, float *b, float *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] & b[i];
    }
}*/

static inline void ands_C(int32_t *a, int32_t *b, int32_t *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = a[i] & b[i];
    }
}

static inline void sigmoidf_C(float *src, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}

// parametric ReLU
// simple ReLU can be expressed as threshold_lt with value = 0
static inline void PReluf_C(float *src, float *dst, float alpha, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        if (src[i] > 0.0f)
            dst[i] = src[i];
        else
            dst[i] = alpha * src[i];
    }
}


static inline void softmaxf_C(float *src, float *dst, int len)
{
    float acc = 0.0f;

#ifdef OMP
#pragma omp simd reduction(+ \
                           : acc)
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = expf(src[i]);
        acc += dst[i];
    }

#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] /= acc;
    }
}


static inline void absdiff16s_c(int16_t *a, int16_t *b, int16_t *c, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        c[i] = abs(a[i] - b[i]);
    }
}

static inline void powerspect16s_c_interleaved(complex16s_t *src, int32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = (int32_t) src[i].re * (int32_t) src[i].re + (int32_t) src[i].im * (int32_t) src[i].im;
    }
}

#ifdef __cplusplus
}
#endif
