/*
 * Project : SIMD_Utils
 * Version : 0.1.12
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define MAJOR_VERSION 0
#define MINOR_VERSION 1
#define SUB_VERSION 11

#ifdef OMP
#include <omp.h>
#endif

#include <math.h>
#include <stdint.h>

#include "mysincosf.h"

#define INVLN10 0.4342944819
#define IMM8_FLIP_VEC 0x1B              // change m128 from abcd to dcba
#define IMM8_LO_HI_VEC 0x1E             // change m128 from abcd to cdab
#define IMM8_PERMUTE_128BITS_LANES 0x1  // reverse abcd efgh to efgh abcd
#define M_PI 3.14159265358979323846

/* LATENCIES
SSE
_mm_store_ps     lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm_storeu_ps    lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm_load_ps      lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm_loadu_ps     lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm_min_ps	 lat 3, cpi 1 (ivy ) 1   (broadwell)
_mm_max_ps       lat 3, cpi 1 (ivy ) 1   (broadwell)
_mm_cvtpd_ps     lat 4, cpi 1 (ivy ) 1   (broadwell)
_mm_mul_ps	 lat 5 (ivy) 3 (broadwell), cpi 1 (ivy) 0.5 (broadwell)
_mm_div_ps	 lat 11-14 (ivy) <11 (broadwell), cpi 6 (ivy) 4 (broadwell)
_mm_movelh_ps    lat 1, cpi 1
_mm_hadd_ps		 lat 5, cpi 2 => useful for reduction!
_mm_shuffle_ps lat 1, cpi 1
_mm_cvtps_epi32 lat 3, cpi 1
_mm_round_ps
_mm_castsi128_ps


AVX/AVX2
_mm256_store_ps  lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm256_storeu_ps lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm256_load_ps   lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm256_loadu_ps  lat 1, cpi 1 (ivy ) 0.5 (broadwell)
_mm256_min_ps	 lat 3, cpi 1 (ivy ) 1   (broadwell)
_mm256_max_ps	 lat 3, cpi 1 (ivy ) 1   (broadwell)
_mm256_cvtpd_ps  lat 4 (ivy) 6 (broadwell), cpi 1 (ivy ) 1  (broadwell)
_mm256_mul_ps	 lat 5 (ivy) 3 (broadwell), cpi 1 (ivy) 0.5 (broadwell)
_mm256_div_ps	 lat 18-21 (ivy) 13-17 (broadwell), cpi 14 (ivy) 10 (broadwell)
_mm256_set_m128  lat 3, cpi 1
_mm256_hadd_ps
_mm256_permute_ps lat 1, cpi 1
_mm256_permute2f128_ps lat 2(ivy) 3 (broadwell) , cpi 1	
 */

typedef struct {
    float re;
    float im;
} complex32_t;


typedef struct {
    double re;
    double im;
} complex64_t;

typedef enum {
    RndZero,
    RndNear,
    RndFinancial,
} FloatRoundingMode;

/* if the user insures that all of their pointers are aligned, 
 * they can use ALWAYS_ALIGNED to hope for some minor speedup on small vectors 
 */
static inline int isAligned(uintptr_t ptr, size_t alignment)
{
#ifndef ALWAYS_ALIGNED
    if (((uintptr_t)(ptr) % alignment) == 0)
        return 1;
    return 0;
#else
    return 1;
#endif
}

static inline int areAligned2(uintptr_t ptr1, uintptr_t ptr2, size_t alignment)
{
#ifndef ALWAYS_ALIGNED
    if (((uintptr_t)(ptr1) % alignment) == 0)
        if (((uintptr_t)(ptr2) % alignment) == 0)
            return 1;
    return 0;
#else
    return 1;
#endif
}

static inline int areAligned3(uintptr_t ptr1, uintptr_t ptr2, uintptr_t ptr3, size_t alignment)
{
#ifndef ALWAYS_ALIGNED
    if (((uintptr_t)(ptr1) % alignment) == 0)
        if (((uintptr_t)(ptr2) % alignment) == 0)
            if (((uintptr_t)(ptr3) % alignment) == 0)
                return 1;
    return 0;
#else
    return 1;
#endif
}


static inline void simd_utils_get_version(void)
{
    printf("Simd Utils Version : %d.%d.%d\n", MAJOR_VERSION, MINOR_VERSION, SUB_VERSION);
}

#ifndef RISCV

#ifdef SSE
#define SSE_LEN_BYTES 16  // Size of SSE lane
#define SSE_LEN_INT32 4   // number of int32 with an SSE lane
#define SSE_LEN_FLOAT 4   // number of float with an SSE lane
#define SSE_LEN_DOUBLE 2  // number of double with an SSE lane

#ifndef ARM
#include "sse_mathfun.h"
#else /* ARM */
#include "neon_mathfun.h"

#define _PS_CONST(Name, Val) \
    static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PI32_CONST(Name, Val) \
    static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PS_CONST_TYPE(Name, Type, Val) \
    static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}

#endif /* ARM */


static inline __m128 _mm_fmadd_ps_custom(__m128 a, __m128 b, __m128 c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#else  /* FMA */
    return _mm_fmadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m128 _mm_fnmadd_ps_custom(__m128 a, __m128 b, __m128 c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
#else  /* FMA */
    return _mm_fnmadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m128d _mm_fmadd_pd_custom(__m128d a, __m128d b, __m128d c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm_add_pd(_mm_mul_pd(a, b), c);
#else  /* FMA */
    return _mm_fmadd_pd(a, b, c);
#endif /* FMA */
}

static inline __m128d _mm_fnmadd_pd_custom(__m128d a, __m128d b, __m128d c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm_sub_pd(c, _mm_mul_pd(a, b));
#else  /* FMA */
    return _mm_fnmadd_pd(a, b, c);
#endif /* FMA */
}

#define _PD_CONST(Name, Val) \
    static const ALIGN16_BEG double _pd_##Name[2] ALIGN16_END = {Val, Val}
#define _PI64_CONST(Name, Val) \
    static const ALIGN16_BEG int64_t _pi64_##Name[2] ALIGN16_END = {Val, Val}
#define _PD_CONST_TYPE(Name, Type, Val) \
    static const ALIGN16_BEG Type _pd_##Name[2] ALIGN16_END = {Val, Val}

/*
_PD_CONST_TYPE(min_norm_pos, int64_t, 0x380ffff83ce549caL);
_PD_CONST_TYPE(mant_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(inv_mant_mask, int64_t, ~0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x8000000000000000L);
_PD_CONST_TYPE(inv_sign_mask, int64_t, ~0x8000000000000000L);
*/

#ifdef ARM

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int) 0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, -1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, -1.2420140846E-1);
_PS_CONST(cephes_log_p4, +1.4249322787E-1);
_PS_CONST(cephes_log_p5, -1.6668057665E-1);
_PS_CONST(cephes_log_p6, +2.0000714765E-1);
_PS_CONST(cephes_log_p7, -2.4999993993E-1);
_PS_CONST(cephes_log_p8, +3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);
_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);

_PS_CONST(minus_cephes_DP1, -0.78515625);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS_CONST(sincof_p0, -1.9515295891E-4);
_PS_CONST(sincof_p1, 8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0, 2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.388731625493765E-003);
_PS_CONST(coscof_p2, 4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516);  // 4 / M_PI

#endif /* ARM */


/* the smallest non denormalized double number */
/*_PD_CONST_TYPE(min_norm_pos, int64_t, 0x00800000);
_PD_CONST_TYPE(mant_mask, int64_t, 0x7f800000);
_PD_CONST_TYPE(inv_mant_mask, int64_t, ~0x7f800000);

_PD_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x80000000);
_PD_CONST_TYPE(inv_sign_mask, int64_t, ~0x80000000);*/

_PD_CONST_TYPE(min_norm_pos, int64_t, 0x380ffff83ce549caL);
_PD_CONST_TYPE(mant_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(inv_mant_mask, int64_t, ~0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x8000000000000000L);
_PD_CONST_TYPE(inv_sign_mask, int64_t, ~0x8000000000000000L);


_PD_CONST(minus_cephes_DP1, -7.85398125648498535156E-1);
_PD_CONST(minus_cephes_DP2, -3.77489470793079817668E-8);
_PD_CONST(minus_cephes_DP3, -2.69515142907905952645E-15);
_PD_CONST(sincof_p0, 1.58962301576546568060E-10);
_PD_CONST(sincof_p1, -2.50507477628578072866E-8);
_PD_CONST(sincof_p2, 2.75573136213857245213E-6);
_PD_CONST(sincof_p3, -1.98412698295895385996E-4);
_PD_CONST(sincof_p4, 8.33333333332211858878E-3);
_PD_CONST(sincof_p5, -1.66666666666666307295E-1);
_PD_CONST(coscof_p0, -1.13585365213876817300E-11);
_PD_CONST(coscof_p1, 2.08757008419747316778E-9);
_PD_CONST(coscof_p2, -2.75573141792967388112E-7);
_PD_CONST(coscof_p3, 2.48015872888517045348E-5);
_PD_CONST(coscof_p4, -1.38888888888730564116E-3);
_PD_CONST(coscof_p5, 4.16666666666665929218E-2);
_PD_CONST(cephes_FOPI, 1.2732395447351626861510701069801148);  // 4 / M_PI

_PD_CONST(1, 1.0);
_PD_CONST(2, 2.0);
_PD_CONST(0p5, 0.5);
_PI64_CONST(1, 1);
_PI64_CONST(inv1, ~1);
_PI64_CONST(2, 2);
_PI64_CONST(4, 4);
_PI64_CONST(0x7f, 0x7f);

_PD_CONST(cephes_SQRTHF, 0.70710678118654752440);
_PD_CONST(cephes_log_p0, 1.01875663804580931796E-4);
_PD_CONST(cephes_log_p1, -4.97494994976747001425E-1);
_PD_CONST(cephes_log_p2, 4.70579119878881725854E0);
_PD_CONST(cephes_log_p3, -1.44989225341610930846E1);
_PD_CONST(cephes_log_p4, +1.79368678507819816313E1);
_PD_CONST(cephes_log_p5, -7.70838733755885391666E0);

_PD_CONST(cephes_log_q1, -1.12873587189167450590E1);
_PD_CONST(cephes_log_q2, 4.52279145837532221105E1);
_PD_CONST(cephes_log_q3, -8.29875266912776603211E1);
_PD_CONST(cephes_log_q4, 7.11544750618563894466E1);
_PD_CONST(cephes_log_q5, 4.52279145837532221105E1);
_PD_CONST(cephes_log_q6, -2.31251620126765340583E1);

_PD_CONST(exp_hi, 709.437);
_PD_CONST(exp_lo, -709.436139303);

_PD_CONST(cephes_LOG2EF, 1.4426950408889634073599);

_PD_CONST(cephes_exp_p0, 1.26177193074810590878e-4);
_PD_CONST(cephes_exp_p1, 3.02994407707441961300e-2);
_PD_CONST(cephes_exp_p2, 9.99999999999999999910e-1);

_PD_CONST(cephes_exp_q0, 3.00198505138664455042e-6);
_PD_CONST(cephes_exp_q1, 2.52448340349684104192e-3);
_PD_CONST(cephes_exp_q2, 2.27265548208155028766e-1);
_PD_CONST(cephes_exp_q3, 2.00000000000000000009e0);

_PD_CONST(cephes_exp_C1, 0.693145751953125);
_PD_CONST(cephes_exp_C2, 1.42860682030941723212e-6);

_PD_CONST_TYPE(positive_mask, int64_t, (int64_t) 0x7FFFFFFFFFFFFFFFL);
_PD_CONST_TYPE(negative_mask, int64_t, (int64_t) ~0x7FFFFFFFFFFFFFFFL);
_PD_CONST(ASIN_P0, 4.253011369004428248960E-3);
_PD_CONST(ASIN_P1, -6.019598008014123785661E-1);
_PD_CONST(ASIN_P2, 5.444622390564711410273E0);
_PD_CONST(ASIN_P3, -1.626247967210700244449E1);
_PD_CONST(ASIN_P4, 1.956261983317594739197E1);
_PD_CONST(ASIN_P5, -8.198089802484824371615E0);

_PD_CONST(ASIN_Q0, -1.474091372988853791896E1);
_PD_CONST(ASIN_Q1, 7.049610280856842141659E1);
_PD_CONST(ASIN_Q2, -1.471791292232726029859E2);
_PD_CONST(ASIN_Q3, 1.395105614657485689735E2);
_PD_CONST(ASIN_Q4, -4.918853881490881290097E1);

_PD_CONST(ASIN_R0, 2.967721961301243206100E-3);
_PD_CONST(ASIN_R1, -5.634242780008963776856E-1);
_PD_CONST(ASIN_R2, 6.968710824104713396794E0);
_PD_CONST(ASIN_R3, -2.556901049652824852289E1);
_PD_CONST(ASIN_R4, 2.853665548261061424989E1);

_PD_CONST(ASIN_S0, -2.194779531642920639778E1);
_PD_CONST(ASIN_S1, 1.470656354026814941758E2);
_PD_CONST(ASIN_S2, -3.838770957603691357202E2);
_PD_CONST(ASIN_S3, 3.424398657913078477438E2);

_PD_CONST(PIO2, 1.57079632679489661923);    /* pi/2 */
_PD_CONST(PIO4, 7.85398163397448309616E-1); /* pi/4 */

_PD_CONST(minMOREBITS, -6.123233995736765886130E-17);
_PD_CONST(MOREBITS, 6.123233995736765886130E-17);

_PD_CONST(ATAN_P0, -8.750608600031904122785E-1);
_PD_CONST(ATAN_P1, -1.615753718733365076637E1);
_PD_CONST(ATAN_P2, -7.500855792314704667340E1);
_PD_CONST(ATAN_P3, -1.228866684490136173410E2);
_PD_CONST(ATAN_P4, -6.485021904942025371773E1);

_PD_CONST(ATAN_Q0, 2.485846490142306297962E1);
_PD_CONST(ATAN_Q1, 1.650270098316988542046E2);
_PD_CONST(ATAN_Q2, 4.328810604912902668951E2);
_PD_CONST(ATAN_Q3, 4.853903996359136964868E2);
_PD_CONST(ATAN_Q4, 1.945506571482613964425E2);

_PD_CONST(TAN3PI8, 2.41421356237309504880); /* 3*pi/8 */

_PD_CONST(min1, -1.0);

#include "simd_utils_sse_double.h"

#include "simd_utils_sse_float.h"
#include "simd_utils_sse_int32.h"

#endif /* SSE */

#ifdef AVX

#ifndef __clang__
#ifndef __INTEL_COMPILER
#ifndef __cplusplus                                       // TODO : it seems to be defined with G++ 9.2 and not GCC 9.2
static inline __m256 _mm256_set_m128(__m128 H, __m128 L)  //not present on every GCC version
{
    return _mm256_insertf128_ps(_mm256_castps128_ps256(L), H, 1);
}
#endif
#endif
#endif /* __clang__ */

#define AVX_LEN_BYTES 32  // Size of AVX lane
#define AVX_LEN_INT32 8   // number of int32 with an AVX lane
#define AVX_LEN_FLOAT 8   // number of float with an AVX lane
#define AVX_LEN_DOUBLE 4  // number of double with an AVX lane

static inline __m256 _mm256_fmadd_ps_custom(__m256 a, __m256 b, __m256 c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#else  /* FMA */
    return _mm256_fmadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m256 _mm256_fnmadd_ps_custom(__m256 a, __m256 b, __m256 c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#else  /* FMA */
    return _mm256_fnmadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m256d _mm256_fmadd_pd_custom(__m256d a, __m256d b, __m256d c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#else  /* FMA */
    return _mm256_fmadd_pd(a, b, c);
#endif /* FMA */
}

static inline __m256d _mm256_fnmadd_pd_custom(__m256d a, __m256d b, __m256d c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
#else  /* FMA */
    return _mm256_fnmadd_pd(a, b, c);
#endif /* FMA */
}

#include "avx_mathfun.h"

#define _PD256_CONST(Name, Val) \
    static const ALIGN32_BEG double _pd256_##Name[4] ALIGN32_END = {Val, Val, Val, Val}
#define _PI256_64_CONST(Name, Val) \
    static const ALIGN32_BEG int64_t _pi256_64_##Name[4] ALIGN32_END = {Val, Val, Val, Val}
#define _PD256_CONST_TYPE(Name, Type, Val) \
    static const ALIGN32_BEG Type _pd256_##Name[4] ALIGN32_END = {Val, Val, Val, Val}

_PD256_CONST_TYPE(min_norm_pos, int64_t, 0x380ffff83ce549caL);
_PD256_CONST_TYPE(mant_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD256_CONST_TYPE(inv_mant_mask, int64_t, ~0xFFFFFFFFFFFFFL);
_PD256_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x8000000000000000L);
_PD256_CONST_TYPE(inv_sign_mask, int64_t, ~0x8000000000000000L);


_PD256_CONST(minus_cephes_DP1, -7.85398125648498535156E-1);
_PD256_CONST(minus_cephes_DP2, -3.77489470793079817668E-8);
_PD256_CONST(minus_cephes_DP3, -2.69515142907905952645E-15);
_PD256_CONST(sincof_p0, 1.58962301576546568060E-10);
_PD256_CONST(sincof_p1, -2.50507477628578072866E-8);
_PD256_CONST(sincof_p2, 2.75573136213857245213E-6);
_PD256_CONST(sincof_p3, -1.98412698295895385996E-4);
_PD256_CONST(sincof_p4, 8.33333333332211858878E-3);
_PD256_CONST(sincof_p5, -1.66666666666666307295E-1);
_PD256_CONST(coscof_p0, -1.13585365213876817300E-11);
_PD256_CONST(coscof_p1, 2.08757008419747316778E-9);
_PD256_CONST(coscof_p2, -2.75573141792967388112E-7);
_PD256_CONST(coscof_p3, 2.48015872888517045348E-5);
_PD256_CONST(coscof_p4, -1.38888888888730564116E-3);
_PD256_CONST(coscof_p5, 4.16666666666665929218E-2);
_PD256_CONST(cephes_FOPI, 1.2732395447351626861510701069801148);  // 4 / M_PI

_PD256_CONST_TYPE(positive_mask, int64_t, (int64_t) 0x7FFFFFFFFFFFFFFFL);
_PD256_CONST_TYPE(negative_mask, int64_t, (int64_t) ~0x7FFFFFFFFFFFFFFFL);
_PD256_CONST(ASIN_P0, 4.253011369004428248960E-3);
_PD256_CONST(ASIN_P1, -6.019598008014123785661E-1);
_PD256_CONST(ASIN_P2, 5.444622390564711410273E0);
_PD256_CONST(ASIN_P3, -1.626247967210700244449E1);
_PD256_CONST(ASIN_P4, 1.956261983317594739197E1);
_PD256_CONST(ASIN_P5, -8.198089802484824371615E0);

_PD256_CONST(ASIN_Q0, -1.474091372988853791896E1);
_PD256_CONST(ASIN_Q1, 7.049610280856842141659E1);
_PD256_CONST(ASIN_Q2, -1.471791292232726029859E2);
_PD256_CONST(ASIN_Q3, 1.395105614657485689735E2);
_PD256_CONST(ASIN_Q4, -4.918853881490881290097E1);

_PD256_CONST(ASIN_R0, 2.967721961301243206100E-3);
_PD256_CONST(ASIN_R1, -5.634242780008963776856E-1);
_PD256_CONST(ASIN_R2, 6.968710824104713396794E0);
_PD256_CONST(ASIN_R3, -2.556901049652824852289E1);
_PD256_CONST(ASIN_R4, 2.853665548261061424989E1);

_PD256_CONST(ASIN_S0, -2.194779531642920639778E1);
_PD256_CONST(ASIN_S1, 1.470656354026814941758E2);
_PD256_CONST(ASIN_S2, -3.838770957603691357202E2);
_PD256_CONST(ASIN_S3, 3.424398657913078477438E2);

_PD256_CONST(PIO2, 1.57079632679489661923);    /* pi/2 */
_PD256_CONST(PIO4, 7.85398163397448309616E-1); /* pi/4 */

_PD256_CONST(minMOREBITS, -6.123233995736765886130E-17);
_PD256_CONST(MOREBITS, 6.123233995736765886130E-17);

_PD256_CONST(ATAN_P0, -8.750608600031904122785E-1);
_PD256_CONST(ATAN_P1, -1.615753718733365076637E1);
_PD256_CONST(ATAN_P2, -7.500855792314704667340E1);
_PD256_CONST(ATAN_P3, -1.228866684490136173410E2);
_PD256_CONST(ATAN_P4, -6.485021904942025371773E1);

_PD256_CONST(ATAN_Q0, 2.485846490142306297962E1);
_PD256_CONST(ATAN_Q1, 1.650270098316988542046E2);
_PD256_CONST(ATAN_Q2, 4.328810604912902668951E2);
_PD256_CONST(ATAN_Q3, 4.853903996359136964868E2);
_PD256_CONST(ATAN_Q4, 1.945506571482613964425E2);

_PD256_CONST(TAN3PI8, 2.41421356237309504880); /* 3*pi/8 */

_PD256_CONST(min1, -1.0);
_PD256_CONST(1, 1.0);
_PD256_CONST(2, 2.0);
_PD256_CONST(0p5, 0.5);
_PI256_64_CONST(1, 1);
_PI256_64_CONST(inv1, ~1);
_PI256_64_CONST(2, 2);
_PI256_64_CONST(4, 4);
_PI256_64_CONST(0x7f, 0x7f);

#include "simd_utils_avx_double.h"
#include "simd_utils_avx_float.h"
#include "simd_utils_avx_int32.h"

#endif /* AVX */

#ifdef AVX512

#define AVX512_LEN_BYTES 64  // Size of AVX512 lane
#define AVX512_LEN_INT32 16  // number of int32 with an AVX512 lane
#define AVX512_LEN_FLOAT 16  // number of float with an AVX512 lane
#define AVX512_LEN_DOUBLE 8  // number of double with an AVX512 lane

static inline __m512 _mm512_fmadd_ps_custom(__m512 a, __m512 b, __m512 c)
{
#ifndef FMA
    return _mm512_add_ps(_mm512_mul_ps(a, b), c);
#else  /* FMA */
    return _mm512_fmadd_ps(a, b, c);
#endif /* FMA */
}

static inline __m512d _mm512_fmadd_pd_custom(__m512d a, __m512d b, __m512d c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm512_add_pd(_mm512_mul_pd(a, b), c);
#else  /* FMA */
    return _mm512_fmadd_pd(a, b, c);
#endif /* FMA */
}

static inline __m512d _mm512_fnmadd_pd_custom(__m512d a, __m512d b, __m512d c)
{
#ifndef FMA  //Haswell comes with avx2 and fma
    return _mm512_sub_pd(c, _mm512_mul_pd(a, b));
#else  /* FMA */
    return _mm512_fnmadd_pd(a, b, c);
#endif /* FMA */
}

#include "avx512_mathfun.h"

#define _PD512_CONST(Name, Val) \
    static const ALIGN64_BEG double _pd512_##Name[8] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI512_64_CONST(Name, Val) \
    static const ALIGN64_BEG int64_t _pi512_64_##Name[8] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PD512_CONST_TYPE(Name, Type, Val) \
    static const ALIGN64_BEG Type _pd512_##Name[8] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val}


_PD512_CONST_TYPE(min_norm_pos, int64_t, 0x380ffff83ce549caL);
_PD512_CONST_TYPE(mant_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD512_CONST_TYPE(inv_mant_mask, int64_t, ~0xFFFFFFFFFFFFFL);
_PD512_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x8000000000000000L);
_PD512_CONST_TYPE(inv_sign_mask, int64_t, ~0x8000000000000000L);

_PD512_CONST(minus_cephes_DP1, -7.85398125648498535156E-1);
_PD512_CONST(minus_cephes_DP2, -3.77489470793079817668E-8);
_PD512_CONST(minus_cephes_DP3, -2.69515142907905952645E-15);
_PD512_CONST(sincof_p0, 1.58962301576546568060E-10);
_PD512_CONST(sincof_p1, -2.50507477628578072866E-8);
_PD512_CONST(sincof_p2, 2.75573136213857245213E-6);
_PD512_CONST(sincof_p3, -1.98412698295895385996E-4);
_PD512_CONST(sincof_p4, 8.33333333332211858878E-3);
_PD512_CONST(sincof_p5, -1.66666666666666307295E-1);
_PD512_CONST(coscof_p0, -1.13585365213876817300E-11);
_PD512_CONST(coscof_p1, 2.08757008419747316778E-9);
_PD512_CONST(coscof_p2, -2.75573141792967388112E-7);
_PD512_CONST(coscof_p3, 2.48015872888517045348E-5);
_PD512_CONST(coscof_p4, -1.38888888888730564116E-3);
_PD512_CONST(coscof_p5, 4.16666666666665929218E-2);
_PD512_CONST(cephes_FOPI, 1.2732395447351626861510701069801148);  // 4 / M_PI

_PD512_CONST_TYPE(positive_mask, int64_t, (int64_t) 0x7FFFFFFFFFFFFFFFL);
_PD512_CONST_TYPE(negative_mask, int64_t, (int64_t) ~0x7FFFFFFFFFFFFFFFL);
_PD512_CONST(ASIN_P0, 4.253011369004428248960E-3);
_PD512_CONST(ASIN_P1, -6.019598008014123785661E-1);
_PD512_CONST(ASIN_P2, 5.444622390564711410273E0);
_PD512_CONST(ASIN_P3, -1.626247967210700244449E1);
_PD512_CONST(ASIN_P4, 1.956261983317594739197E1);
_PD512_CONST(ASIN_P5, -8.198089802484824371615E0);

_PD512_CONST(ASIN_Q0, -1.474091372988853791896E1);
_PD512_CONST(ASIN_Q1, 7.049610280856842141659E1);
_PD512_CONST(ASIN_Q2, -1.471791292232726029859E2);
_PD512_CONST(ASIN_Q3, 1.395105614657485689735E2);
_PD512_CONST(ASIN_Q4, -4.918853881490881290097E1);

_PD512_CONST(ASIN_R0, 2.967721961301243206100E-3);
_PD512_CONST(ASIN_R1, -5.634242780008963776856E-1);
_PD512_CONST(ASIN_R2, 6.968710824104713396794E0);
_PD512_CONST(ASIN_R3, -2.556901049652824852289E1);
_PD512_CONST(ASIN_R4, 2.853665548261061424989E1);

_PD512_CONST(ASIN_S0, -2.194779531642920639778E1);
_PD512_CONST(ASIN_S1, 1.470656354026814941758E2);
_PD512_CONST(ASIN_S2, -3.838770957603691357202E2);
_PD512_CONST(ASIN_S3, 3.424398657913078477438E2);

_PD512_CONST(PIO2, 1.57079632679489661923);    /* pi/2 */
_PD512_CONST(PIO4, 7.85398163397448309616E-1); /* pi/4 */

_PD512_CONST(minMOREBITS, -6.123233995736765886130E-17);
_PD512_CONST(MOREBITS, 6.123233995736765886130E-17);

_PD512_CONST(ATAN_P0, -8.750608600031904122785E-1);
_PD512_CONST(ATAN_P1, -1.615753718733365076637E1);
_PD512_CONST(ATAN_P2, -7.500855792314704667340E1);
_PD512_CONST(ATAN_P3, -1.228866684490136173410E2);
_PD512_CONST(ATAN_P4, -6.485021904942025371773E1);

_PD512_CONST(ATAN_Q0, 2.485846490142306297962E1);
_PD512_CONST(ATAN_Q1, 1.650270098316988542046E2);
_PD512_CONST(ATAN_Q2, 4.328810604912902668951E2);
_PD512_CONST(ATAN_Q3, 4.853903996359136964868E2);
_PD512_CONST(ATAN_Q4, 1.945506571482613964425E2);

_PD512_CONST(TAN3PI8, 2.41421356237309504880); /* 3*pi/8 */

_PD512_CONST(min1, -1.0);
_PD512_CONST(1, 1.0);
_PD512_CONST(2, 2.0);
_PD512_CONST(0p5, 0.5);
_PI512_64_CONST(1, 1);
_PI512_64_CONST(inv1, ~1);
_PI512_64_CONST(2, 2);
_PI512_64_CONST(4, 4);
_PI512_64_CONST(0x7f, 0x7f);

#include "simd_utils_avx512_double.h"
#include "simd_utils_avx512_float.h"
#include "simd_utils_avx512_int32.h"

#endif

#ifdef ICC
#include "simd_utils_svml.h"
#endif

#else /* RISCV */
#include "simd_utils_riscv.h"
#endif /* RISCV */

#ifdef CUSTOM_MALLOC
//Thanks to Jpommier pfft https://bitbucket.org/jpommier/pffft/src/default/pffft.c
static inline int posix_memalign(void **pointer, size_t len, int alignement)
{
    void *p, *p0 = malloc(len + alignement);
    if (!p0)
        return (void *) NULL;
    p = (void *) (((size_t) p0 + alignement) & (~((size_t)(alignement - 1))));
    *((void **) p - 1) = p0;

    *pointer = p;
    return 0;
}


static inline void *aligned_malloc(size_t len, int alignement)
{
    void *p, *p0 = malloc(len + alignement);
    if (!p0)
        return (void *) NULL;
    p = (void *) (((size_t) p0 + alignement) & (~((size_t)(alignement - 1))));
    *((void **) p - 1) = p0;
    return p;
}

//Work in progress
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

static inline void setf_C(float *src, float value, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        src[i] = value;
    }
}

static inline void zerof_C(float *src, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        src[i] = 0.0f;
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
        dst[i] = _a[i] * _b[i] + _c[i];
    }
}

static inline void mulcaddf_C(float *_a, float _b, float *_c, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = _a[i] * _b + _c[i];
    }
}

static inline void mulcaddcf_C(float *_a, float _b, float _c, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = _a[i] * _b + _c;
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

static inline void cplxtorealf_C(float *src, float *dstRe, float *dstIm, int len)
{
    int j = 0;
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < 2 * len; i += 2) {
        dstRe[j] = src[i];
        dstIm[j] = src[i + 1];
        j++;
    }
}


static inline void realtocplx_C(float *srcRe, float *srcIm, float *dst, int len)
{
    int j = 0;
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[j] = srcRe[i];
        dst[j + 1] = srcIm[i];
        j += 2;
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
            dst[i] = (uint8_t)(tmp > 255.0f ? 255.0f : tmp);
        }
    } else if (rounding_mode == RndNear) {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = roundf(src[i] * scale_fact_mult);
            dst[i] = (uint8_t)(tmp > 255.0f ? 255.0f : tmp);
        }
    } else if (rounding_mode == RndFinancial) {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint8_t)(tmp > 255.0f ? 255.0f : tmp);
        }
    } else {
#ifdef OMP
#pragma omp simd
#endif
        for (int i = 0; i < len; i++) {
            float tmp = src[i] * scale_fact_mult;
            dst[i] = (uint8_t)(tmp > 255.0f ? 255.0f : tmp);
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
        dst[i] = sqrtf(src[i].re * src[i].re + src[i].im * src[i].im);
    }
}

static inline void magnitudef_C_split(float *srcRe, float *srcIm, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i]);
    }
}


static inline void powerspectf_C_split(float *srcRe, float *srcIm, float *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i];
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

static inline void cplxvecmul_C(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + src2[i].re * src1[i].im;
    }
}

static inline void cplxvecmul_C_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + src2Re[i] * src1Im[i];
    }
}


static inline void cplxconjvecmul_C(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + src1[i].im * src2[i].im;
        dst[i].im = src2[i].re * src1[i].im - src1[i].re * src2[i].im;
    }
}

static inline void cplxconjvecmul_C_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        dstRe[i] = src1Re[i] * src2Re[i] + src1Im[i] * src2Im[i];
        dstIm[i] = src2Re[i] * src1Im[i] - src1Re[i] * src2Im[i];
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


static inline void setd_C(double *src, double value, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        src[i] = value;
    }
}

static inline void zerod_C(double *src, int len)
{
#ifdef OMP
#pragma omp simd
#endif
    for (int i = 0; i < len; i++) {
        src[i] = 0.0;
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

//parametric ReLU
//simple ReLU can be expressed as threshold_lt with value = 0
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

#ifdef __cplusplus
}
#endif
