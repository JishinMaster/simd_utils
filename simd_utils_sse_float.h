/*
 * Project : SIMD_Utils
 * Version : 0.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <stdint.h>
#include "immintrin.h"

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

//TODO : "Immediate add/mul?"
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

void convert128_32f64f(float* src, double* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i); //load a,b,c,d			
			__m128 src_tmp_hi = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_LO_HI_VEC);// rotate vec from abcd to cdab
			_mm_store_pd(dst + i, _mm_cvtps_pd(src_tmp)); //store the c and d converted in 64bits 
			_mm_store_pd(dst + i + 2, _mm_cvtps_pd(src_tmp_hi)); //store the a and b converted in 64bits 
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i); //load a,b,c,d			
			__m128 src_tmp_hi = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_LO_HI_VEC);// rotate vec from abcd to cdab
			_mm_storeu_pd(dst + i, _mm_cvtps_pd(src_tmp)); //store the c and d converted in 64bits 
			_mm_storeu_pd(dst + i + 2, _mm_cvtps_pd(src_tmp_hi)); //store the a and b converted in 64bits 
		}
	}
	
	for(int i = stop_len; i < len; i++){
		dst[i] = (double)src[i];
	}
}

//TODO : find a better way to work on aligned data
void flip128f(float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	for(int i = 0; i < SSE_LEN_FLOAT; i++){
		dst[len  -i - 1] = src[i];
	}

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = SSE_LEN_FLOAT; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_load_ps(src + i);	//load a,b,c,d
			__m128 src_tmp_flip = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_FLIP_VEC);// rotate vec from abcd to bcba
			_mm_storeu_ps(dst + len -i - SSE_LEN_FLOAT, src_tmp_flip); //store the flipped vector
		}
	}
	else{
		for(int i = SSE_LEN_FLOAT; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i); //load a,b,c,d			
			__m128 src_tmp_flip = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_FLIP_VEC);// rotate vec from abcd to bcba
			_mm_storeu_ps(dst + len -i - SSE_LEN_FLOAT, src_tmp_flip); //store the flipped vector
		}
	}

	for(int i = stop_len; i <  len; i++){
		dst[len  -i - 1] = src[i];
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
void magnitude128f_interleaved( complex32_t* src, float* dst, int len)
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

void sqrt128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_sqrt_ps( _mm_load_ps(src + i) ) );
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_sqrt_ps( _mm_loadu_ps(src + i) ) );
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(src[i]);
	}
}

void round128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
		  __m128 src_tmp   = _mm_load_ps(src + i);
		  _mm_store_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
		  __m128 src_tmp   = _mm_loadu_ps(src + i);
		  _mm_storeu_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = roundf(src[i]);
	}
}

void ceil128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
		  __m128 src_tmp   = _mm_load_ps(src + i);
		  _mm_store_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
		  __m128 src_tmp   = _mm_loadu_ps(src + i);
		  _mm_storeu_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = ceilf(src[i]);
	}
}

void floor128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
		  __m128 src_tmp   = _mm_load_ps(src + i);
		  _mm_store_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
		  __m128 src_tmp   = _mm_loadu_ps(src + i);
		  _mm_storeu_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = floorf(src[i]);
	}
}
