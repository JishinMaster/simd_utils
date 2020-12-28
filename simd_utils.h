/*
 * Project : SIMD_Utils
 * Version : 0.1.3
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OMP
#include <omp.h>
#endif

#include <math.h>

#define INVLN10 0.4342944819
#define IMM8_FLIP_VEC  0x1B // change m128 from abcd to dcba
#define IMM8_LO_HI_VEC 0x1E // change m128 from abcd to cdab
#define IMM8_PERMUTE_128BITS_LANES 0x1 // reverse abcd efgh to efgh abcd
#define M_PI 3.14159265358979323846

#warning "TODO : add better alignment checks"

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

#ifdef SSE
#define SSE_LEN_BYTES 16 // Size of SSE lane
#define SSE_LEN_INT32  4 // number of int32 with an SSE lane
#define SSE_LEN_FLOAT  4 // number of float with an SSE lane
#define SSE_LEN_DOUBLE 2 // number of double with an SSE lane

#ifndef ARM
#include "sse_mathfun.h"
#else
#include "neon_mathfun.h"

#define _PS_CONST(Name, Val)                                            \
		static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
		static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
		static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

#endif


static inline __m128 _mm_fmadd_ps_custom (__m128 a, __m128 b, __m128 c) 
{
	#ifndef FMA //Haswell comes with avx2 and fma
   	return _mm_add_ps( _mm_mul_ps(a,b), c);
	#else
		return _mm_fmadd_ps( a, b, c);
	#endif
}

#ifndef ARM // no support for 64bit float with ARM yet
static inline __m128d _mm_fmadd_pd_custom (__m128d a, __m128d b, __m128d c) 
{
	#ifndef FMA //Haswell comes with avx2 and fma
   	return _mm_add_pd( _mm_mul_pd(a,b), c);
	#else
		return _mm_fmadd_pd( a, b, c);
	#endif
}
#endif

#define _PD_CONST(Name, Val)                                            \
		static const ALIGN16_BEG double _pd_##Name[2] ALIGN16_END = { Val, Val}
#define _PI64_CONST(Name, Val)                                            \
		static const ALIGN16_BEG int64_t _pi64_##Name[2] ALIGN16_END = { Val, Val}
#define _PD_CONST_TYPE(Name, Type, Val)                                 \
		static const ALIGN16_BEG Type _pd_##Name[2] ALIGN16_END = { Val, Val}

_PD_CONST_TYPE(min_norm_pos, long int, 0x380ffff83ce549caL);
_PD_CONST_TYPE(mant_mask, long int, 0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(inv_mant_mask, long int, ~0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(sign_mask, long int, (long int)0x8000000000000000L);
_PD_CONST_TYPE(inv_sign_mask, long int, ~0x8000000000000000L);

#ifdef ARM

_PS_CONST(1  , 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);
_PS_CONST(exp_hi,	88.3762626647949f);
_PS_CONST(exp_lo,	-88.3762626647949f);

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
_PS_CONST(sincof_p1,  8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0,  2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.388731625493765E-003);
_PS_CONST(coscof_p2,  4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI
#endif

_PD_CONST(1 , 1.0);
_PD_CONST(0p5, 0.5);

_PI64_CONST(1, 1);
_PI64_CONST(inv1, ~1);
_PI64_CONST(2, 2);
_PI64_CONST(4, 4);
_PI64_CONST(0x7f, 0x7f);

_PD_CONST(cephes_SQRTHF, 0.70710678118654752440);
_PD_CONST(cephes_log_p0, 1.01875663804580931796E-4);
_PD_CONST(cephes_log_p1, - 4.97494994976747001425E-1);
_PD_CONST(cephes_log_p2, 4.70579119878881725854E0);
_PD_CONST(cephes_log_p3, - 1.44989225341610930846E1);
_PD_CONST(cephes_log_p4, + 1.79368678507819816313E1);
_PD_CONST(cephes_log_p5, - 7.70838733755885391666E0);

_PD_CONST(cephes_log_q1, - 1.12873587189167450590E1);
_PD_CONST(cephes_log_q2, 4.52279145837532221105E1);
_PD_CONST(cephes_log_q3, - 8.29875266912776603211E1);
_PD_CONST(cephes_log_q4, 7.11544750618563894466E1);
_PD_CONST(cephes_log_q5, 4.52279145837532221105E1);
_PD_CONST(cephes_log_q6, - 2.31251620126765340583E1);

_PD_CONST(exp_hi,  709.437);
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


#include "simd_utils_sse_float.h"
#include "simd_utils_sse_int32.h"

#ifndef ARM
#include "simd_utils_sse_double.h"

#endif

#endif

#ifdef AVX

static inline __m256 _mm256_set_m128 ( __m128 H, __m128 L) //not present on every GCC version
{
	return _mm256_insertf128_ps(_mm256_castps128_ps256(L), H, 1);
}
#define AVX_LEN_BYTES 32 // Size of AVX lane
#define AVX_LEN_INT32  8 // number of int32 with an AVX lane
#define AVX_LEN_FLOAT  8 // number of float with an AVX lane
#define AVX_LEN_DOUBLE 4 // number of double with an AVX lane

static inline __m256 _mm256_fmadd_ps_custom (__m256 a, __m256 b, __m256 c) 
{
	#ifndef FMA //Haswell comes with avx2 and fma
   		return _mm256_add_ps( _mm256_mul_ps(a,b), c);
	#else
		return _mm256_fmadd_ps( a, b, c);
	#endif
}

static inline __m256d _mm256_fmadd_pd_custom (__m256d a, __m256d b, __m256d c) 
{
	#ifndef FMA //Haswell comes with avx2 and fma
   		return _mm256_add_pd( _mm256_mul_pd(a,b), c);
	#else
		return _mm256_fmadd_pd( a, b, c);
	#endif
}

#include "avx_mathfun.h"
#include "simd_utils_avx_float.h"
#include "simd_utils_avx_double.h"
#include "simd_utils_avx_int32.h"

#endif


#ifdef CUSTOM_MALLOC
//Thanks to Jpommier pfft https://bitbucket.org/jpommier/pffft/src/default/pffft.c
static inline int posix_memalign(void **pointer, size_t len, int alignement) {
  void *p, *p0 = malloc(len + alignement);
  if (!p0) return (void *) NULL;
  p = (void *) (((size_t) p0 + alignement) & (~((size_t) (alignement-1))));
  *((void **) p - 1) = p0;
  
  *pointer = p;
  return 0;
}


static inline void *aligned_malloc(size_t len, int alignement) {
  void *p, *p0 = malloc(len + alignement);
  if (!p0) return (void *) NULL;
  p = (void *) (((size_t) p0 + alignement) & (~((size_t) (alignement-1))));
  *((void **) p - 1) = p0;
  return p;
}

//Work in progress
static inline void aligned_free(void *p) {
  if (p) free(*((void **) p - 1));
}
	
#endif 



//////////  C Test functions ////////////////
static inline void log10f_C(float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++) dst[i] = log10f(src[i]);
}

static inline void lnf_C(float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++) dst[i] = logf(src[i]);
}

static inline void fabsf_C(float* src, float* dst, int len)
{

	for(int i = 0; i < len; i++){
		dst[i] = fabsf(src[i]);
	}		
}

static inline void setf_C( float* src, float value, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		src[i] = value;
	}		
}

static inline void copyf_C( float* src, float* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = src[i];
	}		
}


static inline void addcf_C( float* src, float value, float* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = src[i] + value;
	}		
}

static inline void mulf_C( float* src1, float* src2, float* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = src1[i] * src2[i];
	}		
}

static inline void mulcf_C( float* src, float value, float* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = src[i] * value;
	}		
}

static inline void divf_C( float* src1, float* src2, float* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = src1[i] / src2[i];
	}		
}

static inline void cplxtorealf_C( float* src, float* dstRe, float* dstIm, int len)
{
	int j = 0;
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < 2*len; i+=2){
		dstRe[j]   = src[i];
		dstIm[j]   = src[i+1];
		j++;
	}		
}

static inline void convert_64f32f_C(double* src, float* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = (float)src[i];
	}
}

static inline void convert_32f64f_C(float* src, double* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = (double)src[i];
	}
}

static inline void convertFloat32ToU8_C(float* src, uint8_t* dst, int len, int rounding_mode, int scale_factor)
{
    float scale_fact_mult = 1.0f/(float)(1 << scale_factor);
    
    if(rounding_mode == RndZero){
	    for(int i = 0; i < len; i++){
		    dst[i] = (uint8_t)floorf(src[i] * scale_fact_mult);
	    }
	}
	else if (rounding_mode == RndNear){
		for(int i = 0; i < len; i++){
		    dst[i] = (uint8_t)roundf(src[i] * scale_fact_mult);
	    }
	}
	else{
	#ifdef OMP
	#pragma omp simd
	#endif
		for(int i = 0; i < len; i++){
		    dst[i] = (uint8_t)(src[i] * scale_fact_mult);
	    }
	}
}

static inline void convertInt16ToFloat32_C(int16_t* src, float* dst, int len, int scale_factor)
{
	float scale_fact_mult = 1.0f/(float)(1 << scale_factor);

#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = (float)src[i] * scale_fact_mult;
	}

}

static inline void threshold_gt_f_C( float* src, float* dst, float value, int len)
{

#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = src[i]<value?src[i]:value;
	}	
}


static inline void threshold_lt_f_C( float* src, float* dst, float value, int len)
{

#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = src[i]>value?src[i]:value;
	}
}

static inline void magnitudef_C_interleaved( complex32_t* src, float* dst, int len)
{

#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = sqrtf(src[i].re*src[i].re + src[i].im*src[i].im);
	}
}

static inline void magnitudef_C_split( float* srcRe, float* srcIm, float* dst, int len)
{

#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = sqrtf(srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i]);
	}
}

static inline void meanf_C(float* src, float* dst, int len)
{
	float acc = 0.0f;
	int i;

#ifdef OMP
#pragma omp simd reduction(+:acc)
#endif
	for(i = 0; i < len; i++){
		acc += src[i];
	}

	acc  = acc/(float)len;
	*dst = acc;
}

static inline void flipf_C(float* src, float* dst, int len)
{

#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[len  -i - 1] = src[i];
	}
}

static inline void asinf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = asinf(src[i]);
	}
}

static inline void tanf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = tanf(src[i]);
	}
}

static inline void atanf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = atanf(src[i]);
	}
}

static inline void atan2f_C( float* src1, float* src2, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = atan2f(src1[i],src2[i]);
	}
}


static inline void sinf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = sinf(src[i]);
	}
}

static inline void cosf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = cosf(src[i]);
	}
}

static inline void floorf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = floorf(src[i]);
	}
}

static inline void ceilf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = ceilf(src[i]);
	}
}

static inline void roundf_C( float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = roundf(src[i]);
	}
}


static inline void cplxvecmul_C(complex32_t* src1, complex32_t* src2, complex32_t* dst, int len)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i].re   = src1[i].re*src2[i].re - src1[i].im*src2[i].im;
		dst[i].im   = src1[i].re*src2[i].im + src2[i].re*src1[i].im;
	}	
}


static inline void vectorSlopef_C(float* dst, int len, float offset, float slope)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = (float)i * slope + offset;
	}
}

static inline void vectorSloped_C(double* dst, int len, double offset, double slope)
{
#ifdef OMP
#pragma omp simd
#endif
	for(int i = 0; i < len; i++){
		dst[i] = (double)i * slope + offset;
	}
}


static inline void maxeveryf_c( float* src1, float* src2, float* dst,  int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = src1[i]>src2[i]?src1[i]:src2[i];
	}
}

static inline void mineveryf_c( float* src1, float* src2, float* dst,  int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = src1[i]<src2[i]?src1[i]:src2[i];
	}
}

void addf_c(float* a, float* b, float* c,int n){
	for(int i = 0; i < n; i++){
		c[i] = a[i] + b[i];
	}
}

void adds_c(int32_t* a, int32_t* b, int32_t* c,int n){
	for(int i = 0; i < n; i++){
		c[i] = a[i] + b[i];
	}
}

void subf_c(float* a, float* b, float* c,int n){
	for(int i = 0; i < n; i++){
		c[i] = a[i] - b[i];
	}
}
#ifdef __cplusplus
}
#endif
