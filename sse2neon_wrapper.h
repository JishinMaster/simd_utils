/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

// JishinMaster : DTCollab sse2neon.h commit : 0d28a356e2f06b50372a7c4d60a4649e225a0acd
#include "sse2neon.h"


#if defined(__GNUC__) || defined(__clang__)

#pragma push_macro("FORCE_INLINE")
#pragma push_macro("ALIGN_STRUCT")
#define FORCE_INLINE static inline __attribute__((always_inline))
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))

#else

#error "Macro name collisions may happens with unknown compiler"
#ifdef FORCE_INLINE
#undef FORCE_INLINE
#endif
#define FORCE_INLINE static inline
#ifndef ALIGN_STRUCT
#define ALIGN_STRUCT(x) __declspec(align(x))
#endif

#endif

#include <stdint.h>
#include <stdlib.h>

#include <arm_neon.h>

// Round types
/*#define _MM_ROUND_MASK 0x6000
#define _MM_ROUND_NEAREST 0x0000
#define _MM_ROUND_DOWN 0x2000
#define _MM_ROUND_UP 0x4000
#define _MM_ROUND_TOWARD_ZERO 0x6000*/


/*#define _MM_FROUND_NINT \
    (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_FLOOR \
    (_MM_FROUND_TO_NEG_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_CEIL \
    (_MM_FROUND_TO_POS_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_TRUNC \
    (_MM_FROUND_TO_ZERO | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_RINT \
    (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_NEARBYINT \
    (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC)*/

/*
From https://developer.arm.com/docs/ddi0595/e/aarch64-system-registers/fpcr
RMode, bits [23:22]
Rounding Mode control field. The encoding of this field is:
RMode	Meaning
0b00	Round to Nearest (RN) mode.
0b01	Round towards Plus Infinity (RP) mode.
0b10	Round towards Minus Infinity (RM) mode.
0b11	Round towards Zero (RZ) mode.
*/


#if defined(__ARM_FEATURE_CRYPTO)

FORCE_INLINE __m128i _mm_aesdec_si128(__m128i a, __m128i RoundKey)
{
    return vreinterpretq_m128i_u8(vaesimcq_u8(vaesdq_u8(a, (__m128i){}))) ^ RoundKey;
}

FORCE_INLINE __m128i _mm_aesdeclast_si128(__m128i a, __m128i RoundKey)
{
    return vreinterpretq_m128i_u8(vaesdq_u8(a, (__m128i){})) ^ RoundKey;
}

inline __m128i _mm_aesimc_si128(__m128i a)
{
    return vreinterpretq_m128i_u8(vaesimcq_u8(vreinterpretq_u8_m128i(a)));
}

#endif


#if 0
//Armv8.3+
FORCE_INLINE __m128 _mm_cplx_mul_ps(__m128 r, __m128 a, __m128 b)
{
    return vreinterpretq_m128_f32(vcmlaq_f32(vreinterpretq_f32_m128(r), vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
}
#endif

#if 0
inline __m128 _mm_blend_ps(__m128 a, __m128 b, const int i32)
{
	//uint32x4_t mask = (uint32x4_t) {i32 & 0x000000FF, i32 & 0x0000FF00, i32 & 0x00FF0000, i32 & 0xFF000000};   
	uint32x4_t mask = (uint32x4_t) {i32 & 0xFF000000, i32 & 0x00FF0000, i32 & 0x0000FF00, i32 & 0x000000FF};   
	return vreinterpretq_m128_f32(vbslq_f32(mask, a, b));
}
#endif

/*FORCE_INLINE void _mm_lfence(void)
{
    __sync_synchronize();
}

FORCE_INLINE void _mm_mfence(void)
{
    __sync_synchronize();
}*/


#ifndef __aarch64__
#define _MM_SHUFFLE2(fp1, fp0) \
    (((fp1) << 1) | (fp0))
#endif

// Computes the fused multiple add product of 32-bit floating point numbers.
//
// Return Value
// Multiplies A and B, and adds C to the temporary result before returning it.
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmadd
FORCE_INLINE __m128 _mm_fmadd_ps(__m128 a, __m128 b, __m128 c)
{
#if defined(__aarch64__)
    return vreinterpretq_m128_f32(vfmaq_f32(vreinterpretq_f32_m128(c),
                                            vreinterpretq_f32_m128(b),
                                            vreinterpretq_f32_m128(a)));
#else
    return vreinterpretq_m128_f32(vmlaq_f32(vreinterpretq_f32_m128(c),
                                            vreinterpretq_f32_m128(b),
                                            vreinterpretq_f32_m128(a)));  //_mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

// Multiply packed single-precision (32-bit) floating-point elements in a and b,
// substract packed elements in c from the intermediate result,
// and store the results in dst.
// dst = a * b - c
// AARCH64 NEON has vfmsq_f32 but it computes A- B*C instead of A*B - C,
// an extra multiplication by -1.0f might not be worth it.
FORCE_INLINE __m128 _mm_fmsub_ps(__m128 a, __m128 b, __m128 c)
{
    return _mm_sub_ps(_mm_mul_ps(a, b), c);
}

// Negate Multiply packed single-precision (32-bit) floating-point elements in a
// and b, add packed elements in c from the intermediate result,
// and store the results in dst.
// dst =  - ( a * b ) + c
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmadd_ps
FORCE_INLINE __m128 _mm_fnmadd_ps(__m128 a, __m128 b, __m128 c)
{
#if defined(__aarch64__)
    return vreinterpretq_m128_f32(vfmsq_f32(vreinterpretq_f32_m128(c),
                                            vreinterpretq_f32_m128(b),
                                            vreinterpretq_f32_m128(a)));
#else
    return vreinterpretq_m128_f32(vmlsq_f32(vreinterpretq_f32_m128(c),
                                            vreinterpretq_f32_m128(b),
                                            vreinterpretq_f32_m128(a)));  //_mm_sub_ps(c, _mm_mul_ps(a, b));
#endif
}

FORCE_INLINE __m128 _mm_fmaddsub_ps(__m128 a, __m128 b, __m128 c)
{
    return _mm_addsub_ps(_mm_mul_ps(a, b), c);
}

// Negate Multiply packed single-precision (32-bit) floating-point elements in a
// and b, substract packed elements in c from the intermediate result,
// and store the results in dst.
// dst =  - ( a * b ) - c
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmsub_ps
FORCE_INLINE __m128 _mm_fnmsub_ps(__m128 a, __m128 b, __m128 c)
{
    __m128 negate_mask = {-1.0f, -1.0f, -1.0f, -1.0f};
    return _mm_mul_ps(negate_mask, _mm_fmadd_ps(a, b, c));
}

FORCE_INLINE __m128i _mm_cmple_epi32(__m128i a, __m128i b)
{
    return vreinterpretq_m128i_u32(
        vcleq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)));
}

FORCE_INLINE __m128i _mm_cmpge_epi32(__m128i a, __m128i b)
{
    return vreinterpretq_m128i_u32(
        vcgeq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)));
}

FORCE_INLINE __m128d _mm_cvtepi64_pd(__m128i a)
{
#if defined(__aarch64__)
    return vreinterpretq_m128d_f64(vcvtq_f64_s64(vreinterpretq_s64_m128i(a)));
#else
    double a0 = (double) ((int64_t *) &a)[0];
    double a1 = (double) ((int64_t *) &a)[1];
    return _mm_set_pd(a1, a0);
#endif
}


FORCE_INLINE __m128i _mm_cvtpd_epi64(__m128d a)
{
#if defined(__aarch64__)
    return vreinterpretq_m128i_s64(vcvtnq_s64_f64(a));
#else
    int64_t a0 = (int64_t) ((double *) &a)[0];
    int64_t a1 = (int64_t) ((double *) &a)[1];
    return _mm_set_epi64(a1, a0);
#endif
}

FORCE_INLINE __m128d _mm_fmadd_pd(__m128d a, __m128d b, __m128d c)
{
#if defined(__aarch64__)
    return vreinterpretq_m128d_f64(vfmaq_f64(vreinterpretq_f64_m128d(c),
                                             vreinterpretq_f64_m128d(b),
                                             vreinterpretq_f64_m128d(a)));
#else
    return _mm_add_pd(_mm_mul_pd(a, b), c);
#endif
}

FORCE_INLINE __m128d _mm_fnmadd_pd(__m128d a, __m128d b, __m128d c)
{
#if defined(__aarch64__)
    return vreinterpretq_m128d_f64(vfmsq_f64(vreinterpretq_f64_m128d(c),
                                             vreinterpretq_f64_m128d(b),
                                             vreinterpretq_f64_m128d(a)));
#else
    return _mm_add_pd(c, _mm_mul_pd(a, b));
#endif
}

#if defined(__GNUC__) || defined(__clang__)
#pragma pop_macro("ALIGN_STRUCT")
#pragma pop_macro("FORCE_INLINE")
#endif
