/*
 * Project : SIMD_Utils
 * Version : 0.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <stdint.h>
#include "immintrin.h"

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

void convert256_32f64f(float* src, double* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){ 
			__m128 src_tmp = _mm_load_ps(src + i); //load a,b,c,d
			_mm256_store_pd(dst + i, _mm256_cvtps_pd(src_tmp)); //store the abcd converted in 64bits 
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i); //load a,b,c,d
			_mm256_storeu_pd(dst + i, _mm256_cvtps_pd(src_tmp)); //store the c and d converted in 64bits 
		}
	}
	
	for(int i = stop_len; i < len; i++){
		dst[i] = (double)src[i];
	}
}

void flip256f(float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	for(int i = 0; i < AVX_LEN_FLOAT; i++){
		dst[len  -i - 1] = src[i];
	}

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = AVX_LEN_FLOAT; i < stop_len; i+= AVX_LEN_FLOAT){
			__m256 src_tmp = _mm256_load_ps(src + i);	//load a,b,c,d,e,f,g,h
 			__m256 src_tmp_flip = _mm256_permute2f128_ps (src_tmp, src_tmp, IMM8_PERMUTE_128BITS_LANES); // reverse lanes abcdefgh to efghabcd
			_mm256_storeu_ps(dst + len -i - AVX_LEN_FLOAT, _mm256_permute_ps (src_tmp_flip, IMM8_FLIP_VEC)); //store the flipped vector
		}
	}
	else{
		/*for(int i = AVX_LEN_FLOAT; i < stop_len; i+= AVX_LEN_FLOAT){
			__m128 src_tmp = _mm_loadu_ps(src + i); //load a,b,c,d,e,f,g,h		
			__m128 src_tmp_flip = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_FLIP_VEC);// rotate vec from abcd to bcba
			_mm_storeu_ps(dst + len -i - AVX_LEN_FLOAT, src_tmp_flip); //store the flipped vector
		}*/
	}

	for(int i = stop_len; i <  len; i++){
		dst[len  -i - 1] = src[i];
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

void sqrt256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_sqrt_ps( _mm256_load_ps(src + i) ) );
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_sqrt_ps( _mm256_loadu_ps(src + i) ) );
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(src[i]);
	}
}


void round256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
		  __m256 src_tmp   = _mm256_load_ps(src + i);
		  _mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
		  __m256 src_tmp   = _mm256_loadu_ps(src + i);
		  _mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = roundf(src[i]);
	}
}

void floor256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
		  __m256 src_tmp   = _mm256_load_ps(src + i);
		  _mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
		  __m256 src_tmp   = _mm256_loadu_ps(src + i);
		  _mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = floorf(src[i]);
	}
}

void ceil256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
		  __m256 src_tmp   = _mm256_load_ps(src + i);
		  _mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
		  __m256 src_tmp   = _mm256_loadu_ps(src + i);
		  _mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = ceilf(src[i]);
	}
}

