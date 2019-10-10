#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef AVX
#include "avx_mathfun.h"
#endif

#include "sse_mathfun.h"

#ifdef OMP
#include <omp.h>
#endif

#include <time.h>
#include <sys/time.h>
#include <stdint.h>

#define INVLN10 0.4342944819

#define SSE_LEN_BYTES 16
#define AVX_LEN_BYTES 32
#define SSE_LEN_FLOAT 4 //single float
#define AVX_LEN_FLOAT 8 //single float

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
*/

typedef struct {
	float re;
	float im;
} complex;

#ifdef WINDOWS // TODO : find a way to align on Windows
void posix_memalign( void** inout , int alignement, size_t len){
	*inout = malloc(len);
}	
#endif 


void log10_128f(float* src, float* dst, int len)
{
	const __m128 invln10f = _mm_set1_ps((float)INVLN10);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = log_ps(_mm_load_ps(src + i));
			_mm_store_ps(dst + i, _mm_mul_ps(src_tmp, invln10f));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = log_ps(_mm_loadu_ps(src + i));
			_mm_storeu_ps(dst + i, _mm_mul_ps(src_tmp, invln10f));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = log10f(src[i]);
	}
}

void ln_128f(float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, log_ps(_mm_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, log_ps(_mm_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = logf(src[i]);
	}
}

void fabs128f(float* src, float* dst, int len)
{
	const __m128 mask = _mm_castsi128_ps (_mm_set1_epi32 (0x7FFFFFFF));

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_and_ps (mask, src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, _mm_and_ps (mask, src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = fabsf(src[i]);
	}		
}

void set128f( float* src, float value, int len)
{
	const __m128 tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(src + i, tmp);
		}
	}
	
	for(int i = stop_len; i < len; i++){
		src[i] = value;
	}		
}

void zero128f( float* src, int len)
{
	const __m128 tmp = _mm_setzero_ps();

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(src + i, tmp);
		}
	}
	
	for(int i = stop_len; i < len; i++){
		src[i] = 0.0f;
	}		
}

#pragma warning "Immediate add/mul?"
void addc128f( float* src, float value, float* dst, int len)
{
	const __m128 tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_add_ps(tmp, _mm_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_add_ps(tmp, _mm_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] + value;
	}		
}

void mulc128f( float* src, float value, float* dst, int len)
{
	const __m128 tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_mul_ps(tmp, _mm_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_add_ps(tmp, _mm_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] * value;
	}		
}

#pragma warning "src2 should have no 0.0f values!"
void div128f( float* src1, float* src2, float* dst, int len)
{

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_div_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_div_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] / src2[i];
	}		
}

// converts 32bits complex float to two arrays real and im
//TODO : find efficient intrinsics
void cplxtoreal128f( float* src, float* dstRe, float* dstIm, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;
	
	int j = 0;
	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			dstRe[j]   = src[i];
			dstIm[j]   = src[i+1];
			dstRe[j+1] = src[i+2];
			dstIm[j+1] = src[i+3];
			j+=2;
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			dstRe[j]   = src[i];
			dstIm[j]   = src[i+1];
			dstRe[j+1] = src[i+2];
			dstIm[j+1] = src[i+3];
			j+=2;
		}
	}

	for(int i = stop_len; i < len; i++){
		dstRe[j]   = src[i];
		dstIm[j]   = src[i+1];
		dstRe[j+1] = src[i+2];
		dstIm[j+1] = src[i+3];
		j+=2;
	}		
}

void convert128_64f32f(double* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128d src_lo = _mm_load_pd(src + i);
			__m128d src_hi = _mm_load_pd(src + i + 2);
			__m128	tmp    = _mm_movelh_ps(_mm_cvtpd_ps(src_lo), _mm_cvtpd_ps(src_hi));
			_mm_store_ps(dst + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128d src_lo = _mm_loadu_pd(src + i);
			__m128d src_hi = _mm_loadu_pd(src + i + 2);
			__m128	tmp    = _mm_movelh_ps(_mm_cvtpd_ps(src_lo), _mm_cvtpd_ps(src_hi));
			_mm_storeu_ps(dst + i, tmp);
		}
	}
	
	for(int i = stop_len; i < len; i++){
		dst[i] = (float)src[i];
	}
}

void threshold128_lt_f( float* src, float* dst, float value, int len)
{
	const __m128 tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_min_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, _mm_min_ps(src_tmp,tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i]<value?src[i]:value;
	}	
}


void threshold128_gt_f( float* src, float* dst, float value, int len)
{
	const __m128 tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(src + i, _mm_max_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, _mm_max_ps(src_tmp,tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i]>value?src[i]:value;
	}
}


void sin128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, sin_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, sin_ps(src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sinf(src[i]);
	}
}

void cos128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, cos_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, cos_ps(src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = cosf(src[i]);
	}
}

void sincos128f( float* src, float* dst_sin, float* dst_cos, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);
			__m128 dst_sin_tmp;
			__m128 dst_cos_tmp;
			sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
			_mm_store_ps(dst_sin + i, dst_sin_tmp);
			_mm_store_ps(dst_cos + i, dst_cos_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i);
			__m128 dst_sin_tmp;
			__m128 dst_cos_tmp;
			sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
			_mm_storeu_ps(dst_sin + i, dst_sin_tmp);
			_mm_storeu_ps(dst_cos + i, dst_cos_tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		dst_sin[i] = sinf(src[i]);
		dst_cos[i] = cosf(src[i]);
	}
}

#warning "TODO : create a single function, avoid div"
void tan128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_div_ps(sin_ps(src_tmp),cos_ps(src_tmp)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, _mm_div_ps(sin_ps(src_tmp),cos_ps(src_tmp)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = tanf(src[i]);
	}
}

void magnitude128f_split( float* srcRe, float* srcIm, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(srcRe) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 re_tmp = _mm_load_ps(srcRe + i);
			__m128 re2    = _mm_mul_ps(re_tmp,re_tmp);
			__m128 im_tmp = _mm_load_ps(srcIm + i);
			__m128 im2    = _mm_mul_ps(im_tmp,im_tmp);
			_mm_store_ps(dst + i, _mm_sqrt_ps(_mm_add_ps(re2,im2)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 re_tmp = _mm_loadu_ps(srcRe + i);
			__m128 re2    = _mm_mul_ps(re_tmp,re_tmp);
			__m128 im_tmp = _mm_loadu_ps(srcIm + i);
			__m128 im2    = _mm_mul_ps(im_tmp,im_tmp);
			_mm_storeu_ps(dst + i, _mm_sqrt_ps(_mm_add_ps(re2,im2)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i]);
	}
}

#warning "TODO : needs to be tested!"
void magnitude128f_interleaved( complex* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	int j = 0;
	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){ 
			__m128 cplx01          = _mm_load_ps(src + j);
			__m128 cplx23          = _mm_load_ps(src + j + 2 ); // complex is 2 floats
			__m128 cplx01_square   = _mm_mul_ps(cplx01,cplx01);
			__m128 cplx23_square   = _mm_mul_ps(cplx23,cplx23);
			__m128 square_sum_0123 = _mm_hadd_ps(cplx01_square, cplx23_square);
			_mm_store_ps(dst + i, _mm_sqrt_ps(square_sum_0123));
			j+= SSE_LEN_BYTES;
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= 2*SSE_LEN_FLOAT){
			__m128 cplx01          = _mm_loadu_ps(src + j);
			__m128 cplx23          = _mm_loadu_ps(src + j + 2 );
			__m128 cplx01_square   = _mm_mul_ps(cplx01,cplx01);
			__m128 cplx23_square   = _mm_mul_ps(cplx23,cplx23);
			__m128 square_sum_0123 = _mm_hadd_ps(cplx01_square, cplx23_square);
			_mm_storeu_ps(dst + i, _mm_sqrt_ps(square_sum_0123));
			j+= SSE_LEN_BYTES;
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(src[i].re*src[i].re + src[i].im*src[i].im);
	}
}

void mean128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;
	
	 __attribute__ ((aligned (SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f,0.0f,0.0f,0.0f};
	float tmp_acc = 0.0f;
	__m128 vec_acc = _mm_setzero_ps(); //initialize the vector accumulator
	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 vec_tmp = _mm_load_ps(src + i);
			vec_acc        = _mm_add_ps(vec_acc, vec_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 vec_tmp = _mm_loadu_ps(src + i);
			vec_acc        = _mm_add_ps(vec_acc, vec_tmp);
		}
	}

	_mm_store_ps(accumulate , vec_acc);
	
	for(int i = stop_len; i < len; i++){
		tmp_acc = src[i];
	}
	
	tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];
	tmp_acc /= (float)len;
	
	*dst = tmp_acc;
}

#ifdef AVX 

#ifdef AVX //not present on every GCC version
__m256 _mm256_set_m128 ( __m128 H, __m128 L)
{
      return _mm256_insertf128_ps(_mm256_castps128_ps256(L), H, 1);
}
#endif

void log10_256f(float* src, float* dst, int len)
{
	float invln10f_mask = (float)INVLN10;
	const __m256 invln10f = _mm256_set1_ps((float)INVLN10); //_mm256_broadcast_ss(&invln10f_mask);

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = log256_ps(_mm256_load_ps(src + i));
			_mm256_store_ps(dst + i, _mm256_mul_ps(src_tmp, invln10f));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = log256_ps(_mm256_loadu_ps(src + i));
			_mm256_storeu_ps(dst + i, _mm256_mul_ps(src_tmp, invln10f));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = log10f(src[i]);
	}
}

void ln_256f(float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, log256_ps(_mm256_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, log256_ps(_mm256_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = logf(src[i]);
	}
}

void fabs256f(float* src, float* dst, int len)
{
	const __m256 mask = _mm256_castsi256_ps (_mm256_set1_epi32 (0x7FFFFFFF));

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_and_ps (mask, src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_and_ps (mask, src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = fabsf(src[i]);
	}		
}


void set256f( float* src, float value, int len)
{
	const __m256 tmp = _mm256_set1_ps(value); //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(src + i, tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		src[i] = value;
	}		
}

void zero256f( float* src, int len)
{
	const __m256 tmp = _mm256_setzero_ps();

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(src + i, tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		src[i] = 0.0f;
	}		
}

void addc256f( float* src, float value, float* dst, int len)
{
	const __m256 tmp = _mm256_set1_ps(value); //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_add_ps(tmp, _mm256_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_add_ps(tmp, _mm256_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] + value;
	}		
}


void mulc256f( float* src, float value, float* dst, int len)
{
	const __m256 tmp = _mm256_set1_ps(value); //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_mul_ps(tmp, _mm256_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_mul_ps(tmp, _mm256_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] * value;
	}		
}

void div256f( float* src1, float* src2, float* dst, int len)
{

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_div_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_div_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] / src2[i];
	}		
}

void convert256_64f32f(double* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m128 src_lo = _mm256_cvtpd_ps(_mm256_load_pd(src + i));
			__m128 src_hi = _mm256_cvtpd_ps(_mm256_load_pd(src + i + 4));
			_mm256_store_ps(dst + i, _mm256_set_m128(src_hi, src_lo));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m128 src_lo = _mm256_cvtpd_ps(_mm256_loadu_pd(src + i));
			__m128 src_hi = _mm256_cvtpd_ps(_mm256_loadu_pd(src + i + 4));
			_mm256_storeu_ps(dst + i, _mm256_set_m128(src_hi, src_lo));
		}
	}
	
	for(int i = stop_len; i < len; i++){
		dst[i] = (float)src[i];
	}
}

void threshold256_lt_f( float* src, float* dst, float value, int len)
{
	__m256 tmp = _mm256_set1_ps(value);//_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_min_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_min_ps(src_tmp,tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i]<value?src[i]:value;
	}	
}

void threshold256_gt_f( float* src, float* dst, float value, int len)
{
	__m256 tmp = _mm256_set1_ps(value);//_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_max_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_max_ps(src_tmp,tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i]<value?src[i]:value;
	}	
}

void sin256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, sin256_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, sin256_ps(src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sinf(src[i]);
	}
}

void cos256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, cos256_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, cos256_ps(src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = cosf(src[i]);
	}
}

void sincos256f( float* src, float* dst_sin, float* dst_cos, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);
			__m256 dst_sin_tmp;
			__m256 dst_cos_tmp;
			sincos256_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
			_mm256_store_ps(dst_sin + i, dst_sin_tmp);
			_mm256_store_ps(dst_cos + i, dst_cos_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_loadu_ps(src + i);
			__m256 dst_sin_tmp;
			__m256 dst_cos_tmp;
			sincos256_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
			_mm256_storeu_ps(dst_sin + i, dst_sin_tmp);
			_mm256_storeu_ps(dst_cos + i, dst_cos_tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		dst_sin[i] = sinf(src[i]);
		dst_cos[i] = cosf(src[i]);
	}
}

#warning "TODO : create a single function, avoid div"
void tan256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_div_ps(sin256_ps(src_tmp),cos256_ps(src_tmp)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_div_ps(sin256_ps(src_tmp),cos256_ps(src_tmp)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = tanf(src[i]);
	}
}


void magnitude256f_split( float* srcRe, float* srcIm, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(srcRe) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 re_tmp = _mm256_load_ps(srcRe + i);
			__m256 re2    = _mm256_mul_ps(re_tmp,re_tmp);
			__m256 im_tmp = _mm256_load_ps(srcIm + i);
			__m256 im2    = _mm256_mul_ps(im_tmp,im_tmp);
			_mm256_store_ps(dst + i, _mm256_sqrt_ps(_mm256_add_ps(re2,im2)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 re_tmp = _mm256_loadu_ps(srcRe + i);
			__m256 re2    = _mm256_mul_ps(re_tmp,re_tmp);
			__m256 im_tmp = _mm256_loadu_ps(srcIm + i);
			__m256 im2    = _mm256_mul_ps(im_tmp,im_tmp);
			_mm256_store_ps(dst + i, _mm256_sqrt_ps(_mm256_add_ps(re2,im2)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i]);
	}
}

void mean256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;
	
	 __attribute__ ((aligned (AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
	float tmp_acc = 0.0f;
	__m256 vec_acc = _mm256_setzero_ps(); //initialize the vector accumulator
	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_FLOAT) == 0){
		//printf("Is aligned!\n");
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 vec_tmp = _mm256_load_ps(src + i);
			vec_acc        = _mm256_add_ps(vec_acc, vec_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 vec_tmp = _mm256_loadu_ps(src + i);
			vec_acc        = _mm256_add_ps(vec_acc, vec_tmp);
		}
	}

	_mm256_store_ps(accumulate , vec_acc);
	
	for(int i = stop_len; i < len; i++){
		tmp_acc += src[i];
	}
	
	tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];
	tmp_acc /= (float)len;
	
	*dst = tmp_acc;
}

#endif


//////////  C Test functions ////////////////
void log10f_C(float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++) dst[i] = log10f(src[i]);
}

void ln_C(float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++) dst[i] = logf(src[i]);
}

void fabsf_C(float* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = fabsf(src[i]);
	}		
}

void setf_C( float* src, float value, int len)
{
	for(int i = 0; i < len; i++){
		src[i] = value;
	}		
}

void addcf_C( float* src, float value, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = src[i] + value;
	}		
}

void mulcf_C( float* src, float value, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = src[i] * value;
	}		
}

void divf_C( float* src1, float* src2, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = src1[i] / src2[i];
	}		
}

void cplxtorealf_C( float* src, float* dstRe, float* dstIm, int len)
{
	int j = 0;
	for(int i = 0; i < len; i++){
		dstRe[j]   = src[i];
		dstIm[j]   = src[i+1];
		dstRe[j+1] = src[i+2];
		dstIm[j+1] = src[i+3];
		j+=2;
	}		
}

void convert_64f32f_C(double* src, float* dst, int len)
{
	#ifdef OMP
	#pragma omp simd
	#endif	
	for(int i = 0; i < len; i++){
		dst[i] = (float)src[i];
	}
}

void threshold_lt_f_C( float* src, float* dst, float value, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = src[i]<value?src[i]:value;
	}	
}


void threshold_gt_f_C( float* src, float* dst, float value, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = src[i]>value?src[i]:value;
	}
}

void magnitude_C_interleaved( complex* src, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = sqrtf(src[i].re*src[i].re + src[i].im*src[i].im);
	}
}

void magnitude_C_split( float* srcRe, float* srcIm, float* dst, int len)
{
	for(int i = 0; i < len; i++){
		dst[i] = sqrtf(srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i]);
	}
}

void mean_C(float* src, float* dst, int len)
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
////////////////////////
