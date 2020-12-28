#pragma once

//JishinMaster : DTCollab sse2neon.h commit : 8508f728918ee20a56570f8b824ba77ac308a7c1
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

//Round types
#define _MM_ROUND_MASK        0x6000
#define _MM_ROUND_NEAREST     0x0000
#define _MM_ROUND_DOWN        0x2000
#define _MM_ROUND_UP          0x4000
#define _MM_ROUND_TOWARD_ZERO 0x6000
/* Rounding mode macros. */
#define _MM_FROUND_TO_NEAREST_INT	0x00
#define _MM_FROUND_TO_NEG_INF		0x01
#define _MM_FROUND_TO_POS_INF		0x02
#define _MM_FROUND_TO_ZERO		0x03
#define _MM_FROUND_CUR_DIRECTION	0x04

#define _MM_FROUND_RAISE_EXC		0x00
#define _MM_FROUND_NO_EXC		0x08

#define _MM_FROUND_NINT		\
  (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_FLOOR	\
  (_MM_FROUND_TO_NEG_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_CEIL		\
  (_MM_FROUND_TO_POS_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_TRUNC	\
  (_MM_FROUND_TO_ZERO | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_RINT		\
  (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_NEARBYINT	\
  (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC)


//To be checked
FORCE_INLINE __m128d _mm_shuffle_pd(__m128d a,
                                           __m128d b,
                                           __constrange(0, 255) int imm)
{
    __m128d ret;
    ret[0] = a[imm & 0x3];
    ret[1] = a[(imm >> 2) & 0x3];
    
    
    //ret[2] = b[(imm >> 4) & 0x03];
    //ret[3] = b[(imm >> 6) & 0x03];
    return ret;
}


FORCE_INLINE __m128d _mm_castsi128_pd(__m128i a)
{
    return vreinterpretq_m128d_s64(vreinterpretq_s64_m128i(a));
}


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

typedef struct{
    uint16_t res0;
    uint16_t res1 : 6;
    uint8_t  bit21 : 1;
    uint8_t  bit22 : 1;
    uint32_t res3;

} fpcr_bitfield;

typedef union{
    fpcr_bitfield field;
    uint64_t value;
} reg;

//TO BE TESTED
FORCE_INLINE void _MM_SET_ROUNDING_MODE (int rounding){

    reg r;
    asm volatile("mrs %0, FPCR" : "=r"(r.value)); /* read */
    
    if( rounding ==  _MM_ROUND_TOWARD_ZERO){
        r.field.bit21 = 1;
        r.field.bit22 = 1;
    }
    else if(rounding ==  _MM_FROUND_TO_NEG_INF ){
        r.field.bit21 = 0;
        r.field.bit22 = 1;
    }
    else if(rounding ==  _MM_FROUND_TO_POS_INF ){
        r.field.bit21 = 1;
        r.field.bit22 = 0;
    }
    else{ //_MM_ROUND_NEAREST
        r.field.bit21 = 0;
        r.field.bit22 = 0;
    }
    asm volatile("msr FPCR, %0" :: "r"(r)); /* write */
}


//TO BE TESTED
FORCE_INLINE __m64 _mm_cvtps_pi16( __m128 a ){
    return (__m64)vmovn_s32(vcvtnq_s32_f32(a));
}


#if defined(__ARM_FEATURE_CRYPTO)

FORCE_INLINE __m128i _mm_aesdec_si128 (__m128i a, __m128i RoundKey)
{
    return vreinterpretq_m128i_u8(vaesimcq_u8(vaesdq_u8(a, (__m128i){}))) ^ RoundKey;
}

FORCE_INLINE __m128i _mm_aesdeclast_si128 (__m128i a, __m128i RoundKey)
{
    return vreinterpretq_m128i_u8(vaesdq_u8(a, (__m128i){})) ^ RoundKey;
}

inline __m128i _mm_aesimc_si128 (__m128i a)
{
	return vreinterpretq_m128i_u8(vaesimcq_u8 (vreinterpretq_u8_m128i(a)));
}

#endif


//Armv8.3+
FORCE_INLINE __m128 _mm_cplx_mul_ps(__m128 r, __m128 a, __m128 b)
{
	return vreinterpretq_m128_f32(vcmlaq_f32(vreinterpretq_f32_m128(r), vreinterpretq_f32_m128(a),vreinterpretq_f32_m128(b)));
}

// Store the lower single-precision (32-bit) floating-point element from a into 4 contiguous elements in memory. mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_store1_ps&expand=5217,3606,3720,5595
FORCE_INLINE void _mm_store1_ps(float* mem_addr, __m128 a) {
	vst1q_lane_f32(mem_addr, a, 0);
}

#if 0
inline __m128 _mm_blend_ps(__m128 a, __m128 b, const int i32)
{
	//uint32x4_t mask = (uint32x4_t) {i32 & 0x000000FF, i32 & 0x0000FF00, i32 & 0x00FF0000, i32 & 0xFF000000};   
	uint32x4_t mask = (uint32x4_t) {i32 & 0xFF000000, i32 & 0x00FF0000, i32 & 0x0000FF00, i32 & 0x000000FF};   
	return vreinterpretq_m128_f32(vbslq_f32(mask, a, b));
}
#endif

FORCE_INLINE void _mm_empty(){
	return;
}

FORCE_INLINE void _mm_lfence(void)
{
    __sync_synchronize();
}

FORCE_INLINE void _mm_mfence(void)
{
    __sync_synchronize();
}


#if defined(__GNUC__) || defined(__clang__)
#pragma pop_macro("ALIGN_STRUCT")
#pragma pop_macro("FORCE_INLINE")
#endif



