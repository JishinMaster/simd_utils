/*
 * Project : SIMD_Utils
 * Version : 0.1.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <stdint.h>
#include "immintrin.h"

// For tanf
_PS_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS_CONST(DP1,   -0.78515625);
_PS_CONST(DP2,   -2.4187564849853515625e-4);
_PS_CONST(DP3,   -3.77489497744594108e-8);

_PS_CONST(FOPI, 1.27323954473516); /* 4/pi */
_PS_CONST(TAN_P0, 9.38540185543E-3);
_PS_CONST(TAN_P1, 3.11992232697E-3);
_PS_CONST(TAN_P2, 2.44301354525E-2);
_PS_CONST(TAN_P3, 5.34112807005E-2);
_PS_CONST(TAN_P4, 1.33387994085E-1);
_PS_CONST(TAN_P5, 3.33331568548E-1);

_PS_CONST(ASIN_P0, 4.2163199048E-2);
_PS_CONST(ASIN_P1, 2.4181311049E-2);
_PS_CONST(ASIN_P2, 4.5470025998E-2);
_PS_CONST(ASIN_P3, 7.4953002686E-2);
_PS_CONST(ASIN_P4, 1.6666752422E-1);

_PS_CONST(PIF, 3.14159265358979323846); // PI
_PS_CONST(PIO2F, 1.570796326794896619); // PI/2

void log10_128f(float* src, float* dst, int len)
{
	const v4sf invln10f = _mm_set1_ps((float)INVLN10);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = log_ps(_mm_load_ps(src + i));
			_mm_store_ps(dst + i, _mm_mul_ps(src_tmp, invln10f));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = log_ps(_mm_loadu_ps(src + i));
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
	const v4sf mask = _mm_castsi128_ps (_mm_set1_epi32 (0x7FFFFFFF));

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_and_ps (mask, src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, _mm_and_ps (mask, src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = fabsf(src[i]);
	}		
}

void set128f( float* src, float value, int len)
{
	const v4sf tmp = _mm_set1_ps(value);

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

// Could be better to just use set(0)
void zero128f( float* src, int len)
{
	const v4sf tmp = _mm_setzero_ps();

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

void copy128f( float* src,  float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_load_ps(src + i));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_loadu_ps(src + i));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i];
	}
}

void add128f( float* src1, float* src2, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_add_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_add_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] + src2[i];
	}
}


void mul128f( float* src1, float* src2, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_mul_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_mul_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] * src2[i];
	}
}

void sub128f( float* src1, float* src2, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_sub_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_sub_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] - src2[i];
	}
}

//TODO : "Immediate add/mul?"
// No need for subc, just use addc(-value)
void addc128f( float* src, float value, float* dst, int len)
{
	const v4sf tmp = _mm_set1_ps(value);

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
	const v4sf tmp = _mm_set1_ps(value);

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
			v4sf	tmp    = _mm_movelh_ps(_mm_cvtpd_ps(src_lo), _mm_cvtpd_ps(src_hi));
			_mm_store_ps(dst + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			__m128d src_lo = _mm_loadu_pd(src + i);
			__m128d src_hi = _mm_loadu_pd(src + i + 2);
			v4sf	tmp    = _mm_movelh_ps(_mm_cvtpd_ps(src_lo), _mm_cvtpd_ps(src_hi));
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
			v4sf src_tmp = _mm_load_ps(src + i); //load a,b,c,d
			v4sf src_tmp_hi = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_LO_HI_VEC);// rotate vec from abcd to cdab
			_mm_store_pd(dst + i, _mm_cvtps_pd(src_tmp)); //store the c and d converted in 64bits 
			_mm_store_pd(dst + i + 2, _mm_cvtps_pd(src_tmp_hi)); //store the a and b converted in 64bits 
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i); //load a,b,c,d
			v4sf src_tmp_hi = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_LO_HI_VEC);// rotate vec from abcd to cdab
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
			v4sf src_tmp = _mm_load_ps(src + i);	//load a,b,c,d
			v4sf src_tmp_flip = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_FLIP_VEC);// rotate vec from abcd to bcba
			_mm_storeu_ps(dst + len -i - SSE_LEN_FLOAT, src_tmp_flip); //store the flipped vector
		}
	}
	else{
		for(int i = SSE_LEN_FLOAT; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i); //load a,b,c,d
			v4sf src_tmp_flip = _mm_shuffle_ps (src_tmp, src_tmp, IMM8_FLIP_VEC);// rotate vec from abcd to bcba
			_mm_storeu_ps(dst + len -i - SSE_LEN_FLOAT, src_tmp_flip); //store the flipped vector
		}
	}

	for(int i = stop_len; i <  len; i++){
		dst[len  -i - 1] = src[i];
	}
}

void maxevery128f( float* src1, float* src2, float* dst,  int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_max_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_max_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i]>src2[i]?src1[i]:src2[i];
	}
}

void minevery128f( float* src1, float* src2, float* dst,  int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_min_ps(_mm_load_ps(src1 + i), _mm_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_min_ps(_mm_loadu_ps(src1 + i), _mm_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i]<src2[i]?src1[i]:src2[i];
	}
}

void threshold128_lt_f( float* src, float* dst, float value, int len)
{
	const v4sf tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_min_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, _mm_min_ps(src_tmp,tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i]<value?src[i]:value;
	}	
}

void threshold128_gt_f( float* src, float* dst, float value, int len)
{
	const v4sf tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_max_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
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
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, sin_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
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
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, cos_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
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
			v4sf src_tmp = _mm_load_ps(src + i);
			v4sf dst_sin_tmp;
			v4sf dst_cos_tmp;
			sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
			_mm_store_ps(dst_sin + i, dst_sin_tmp);
			_mm_store_ps(dst_cos + i, dst_cos_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
			v4sf dst_sin_tmp;
			v4sf dst_cos_tmp;
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


v4sf asinf_ps(v4sf xx, const v4sf positive_mask, const v4sf negative_mask)
{
	v4sf a, x, z, x_tmp, z_tmp;
	v4sf sign, flag;
	v4sf ainfem4, asup0p5;
	v4sf tmp;
	x    = xx;
	a    = _mm_and_ps (positive_mask, x); //fabs(x)
	sign = _mm_cmplt_ps(x,_mm_setzero_ps()); //0xFFFFFFFF if x < 0.0

	//TODO : vectorize this
	/*if( a > 1.0f )
	{
		return( 0.0f );
	}*/


	ainfem4 = _mm_cmplt_ps(a,_mm_set1_ps(1.0e-4)); 	//if( a < 1.0e-4f )

	asup0p5 = _mm_cmpgt_ps(a,*(v4sf*)_ps_0p5); //if( a > 0.5f ) flag = 1 else 0
	z_tmp   = _mm_sub_ps(*(v4sf*)_ps_1,a);
	z_tmp   = _mm_mul_ps(*(v4sf*)_ps_0p5,z_tmp);
	z       = _mm_blendv_ps (_mm_mul_ps(a,a), z_tmp, asup0p5);
	x       = _mm_blendv_ps ( a, _mm_sqrt_ps(z), asup0p5);

	tmp     =  _mm_mul_ps(*(v4sf*)_ps_ASIN_P0, z);
	tmp     =  _mm_add_ps(*(v4sf*)_ps_ASIN_P1, tmp);
	tmp     =  _mm_mul_ps(z, tmp);
	tmp     =  _mm_add_ps(*(v4sf*)_ps_ASIN_P2, tmp);
	tmp     =  _mm_mul_ps(z, tmp);
	tmp     =  _mm_add_ps(*(v4sf*)_ps_ASIN_P3, tmp);
	tmp     =  _mm_mul_ps(z, tmp);
	tmp     =  _mm_add_ps(*(v4sf*)_ps_ASIN_P4, tmp);
	tmp     =  _mm_mul_ps(z, tmp);
	tmp     =  _mm_mul_ps(x, tmp);
	tmp     =  _mm_add_ps(x, tmp);

	z       = tmp;

	z_tmp   = _mm_add_ps(z, z);
	z_tmp   = _mm_sub_ps(*(v4sf*)_ps_PIO2F, z_tmp);
	z       = _mm_blendv_ps (z, z_tmp, asup0p5);

	//done:
	z       = _mm_blendv_ps (z, a, ainfem4);
	z       = _mm_blendv_ps (z, _mm_xor_ps(negative_mask,z), sign);

	return( z );
}

void asin128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	const v4sf positive_mask = _mm_castsi128_ps (_mm_set1_epi32 (0x7FFFFFFF));
	const v4sf negative_mask = _mm_castsi128_ps (_mm_set1_epi32 (~0x7FFFFFFF));

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, asinf_ps(src_tmp, positive_mask, negative_mask));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, asinf_ps(src_tmp, positive_mask, negative_mask));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = asinf(src[i]);
	}
}

v4sf tanf_ps(v4sf xx, const v4sf positive_mask, const v4sf negative_mask)
{
	v4sf x, y, z, zz;
	v4si j; //long?
	v4sf sign, xsupem4;
	v4sf tmp;
	v4si jandone, jandtwo;

	x = _mm_and_ps (positive_mask, xx); //fabs(xx)

	/* compute x mod PIO4 */

	//TODO : on neg values should be ceil and not floor
	j = _mm_cvtps_epi32( _mm_round_ps(_mm_mul_ps(*(v4sf*)_ps_FOPI,x), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC )); /* integer part of x/(PI/4), using floor */
	y = _mm_cvtepi32_ps(j);

	jandone = _mm_cmpgt_epi32(_mm_and_si128(j,*(v4si*)_pi32_1),_mm_setzero_si128 ());
	y = _mm_blendv_ps ( y, _mm_add_ps(y,*(v4sf*)_ps_1),_mm_cvtepi32_ps(jandone));
	j = _mm_cvtps_epi32( y); // no need to round again

	//z = ((x - y * DP1) - y * DP2) - y * DP3;

#if 1
	tmp = _mm_mul_ps(y, *(v4sf*)_ps_DP1);
	z   = _mm_add_ps(x, tmp);
	tmp = _mm_mul_ps(y, *(v4sf*)_ps_DP2);
	z   = _mm_add_ps(z, tmp);
	tmp = _mm_mul_ps(y, *(v4sf*)_ps_DP3);
	z   = _mm_add_ps(z, tmp);
#else // faster but less precision
	tmp = _mm_mul_ps(y,*(v4sf*)_ps_DP123);
	z   = _mm_sub_ps(x, tmp);
#endif
	zz = _mm_mul_ps(z,z); //z*z

	//TODO : should not be computed if X < 10e-4
	/* 1.7e-8 relative error in [-pi/4, +pi/4] */
	tmp =  _mm_mul_ps(*(v4sf*)_ps_TAN_P0, zz);
	tmp =  _mm_add_ps(*(v4sf*)_ps_TAN_P1, tmp);
	tmp =  _mm_mul_ps(zz, tmp);
	tmp =  _mm_add_ps(*(v4sf*)_ps_TAN_P2, tmp);
	tmp =  _mm_mul_ps(zz, tmp);
	tmp =  _mm_add_ps(*(v4sf*)_ps_TAN_P3, tmp);
	tmp =  _mm_mul_ps(zz, tmp);
	tmp =  _mm_add_ps(*(v4sf*)_ps_TAN_P4, tmp);
	tmp =  _mm_mul_ps(zz, tmp);
	tmp =  _mm_add_ps(*(v4sf*)_ps_TAN_P5, tmp);
	tmp =  _mm_mul_ps(zz, tmp);
	tmp =  _mm_mul_ps(z, tmp);
	tmp =  _mm_add_ps(z, tmp);

	xsupem4 = _mm_cmpgt_ps(x,_mm_set1_ps(1.0e-4)); 	//if( x > 1.0e-4 )
	y      = _mm_blendv_ps ( z, tmp, xsupem4);

	jandtwo = _mm_cmpgt_epi32(_mm_and_si128(j,*(v4si*)_pi32_2),_mm_setzero_si128 ());
	y = _mm_blendv_ps ( y, _mm_div_ps(_mm_set1_ps(-1.0f),y),_mm_cvtepi32_ps(jandtwo));

	sign   = _mm_cmplt_ps(xx,_mm_setzero_ps()); //0xFFFFFFFF if xx < 0.0
	y = _mm_blendv_ps (y, _mm_xor_ps(negative_mask,y), sign);

	return( y );
}

void tan128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	const v4sf positive_mask = _mm_castsi128_ps (_mm_set1_epi32 (0x7FFFFFFF));
	const v4sf negative_mask = _mm_castsi128_ps (_mm_set1_epi32 (~0x7FFFFFFF));

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, tanf_ps(src_tmp, positive_mask, negative_mask));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, tanf_ps(src_tmp, positive_mask, negative_mask));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = tanf(src[i]);
	}
}

void tan128f_naive( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_div_ps(sin_ps(src_tmp),cos_ps(src_tmp)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp = _mm_loadu_ps(src + i);
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
			v4sf re_tmp = _mm_load_ps(srcRe + i);
			v4sf re2    = _mm_mul_ps(re_tmp,re_tmp);
			v4sf im_tmp = _mm_load_ps(srcIm + i);
			v4sf im2    = _mm_mul_ps(im_tmp,im_tmp);
			_mm_store_ps(dst + i, _mm_sqrt_ps(_mm_add_ps(re2,im2)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf re_tmp = _mm_loadu_ps(srcRe + i);
			v4sf re2    = _mm_mul_ps(re_tmp,re_tmp);
			v4sf im_tmp = _mm_loadu_ps(srcIm + i);
			v4sf im2    = _mm_mul_ps(im_tmp,im_tmp);
			_mm_storeu_ps(dst + i, _mm_sqrt_ps(_mm_add_ps(re2,im2)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i]);
	}
}

void powerspect128f_split( float* srcRe, float* srcIm, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(srcRe) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf re_tmp = _mm_load_ps(srcRe + i);
			v4sf re2    = _mm_mul_ps(re_tmp,re_tmp);
			v4sf im_tmp = _mm_load_ps(srcIm + i);
			v4sf im2    = _mm_mul_ps(im_tmp,im_tmp);
			_mm_store_ps(dst + i, _mm_add_ps(re2,im2));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf re_tmp = _mm_loadu_ps(srcRe + i);
			v4sf re2    = _mm_mul_ps(re_tmp,re_tmp);
			v4sf im_tmp = _mm_loadu_ps(srcIm + i);
			v4sf im2    = _mm_mul_ps(im_tmp,im_tmp);
			_mm_storeu_ps(dst + i, _mm_add_ps(re2,im2));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i];
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
			v4sf cplx01          = _mm_load_ps(src + j);
			v4sf cplx23          = _mm_load_ps(src + j + 2 ); // complex is 2 floats
			v4sf cplx01_square   = _mm_mul_ps(cplx01,cplx01);
			v4sf cplx23_square   = _mm_mul_ps(cplx23,cplx23);
			v4sf square_sum_0123 = _mm_hadd_ps(cplx01_square, cplx23_square);
			_mm_store_ps(dst + i, _mm_sqrt_ps(square_sum_0123));
			j+= SSE_LEN_BYTES;
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= 2*SSE_LEN_FLOAT){
			v4sf cplx01          = _mm_loadu_ps(src + j);
			v4sf cplx23          = _mm_loadu_ps(src + j + 2 );
			v4sf cplx01_square   = _mm_mul_ps(cplx01,cplx01);
			v4sf cplx23_square   = _mm_mul_ps(cplx23,cplx23);
			v4sf square_sum_0123 = _mm_hadd_ps(cplx01_square, cplx23_square);
			_mm_storeu_ps(dst + i, _mm_sqrt_ps(square_sum_0123));
			j+= SSE_LEN_BYTES;
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(src[i].re*src[i].re + src[i].im*src[i].im);
	}
}

void subcrev128f( float* src, float value, float* dst, int len)
{
	const v4sf tmp = _mm_set1_ps(value);

	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_store_ps(dst + i, _mm_sub_ps(tmp, _mm_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			_mm_storeu_ps(dst + i, _mm_sub_ps(tmp, _mm_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] + value;
	}
}

void sum128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	__attribute__ ((aligned (SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f,0.0f,0.0f,0.0f};
	float tmp_acc = 0.0f;
	v4sf vec_acc = _mm_setzero_ps(); //initialize the vector accumulator
	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf vec_tmp = _mm_load_ps(src + i);
			vec_acc        = _mm_add_ps(vec_acc, vec_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf vec_tmp = _mm_loadu_ps(src + i);
			vec_acc        = _mm_add_ps(vec_acc, vec_tmp);
		}
	}

	_mm_store_ps(accumulate , vec_acc);

	for(int i = stop_len; i < len; i++){
		tmp_acc = src[i];
	}

	tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

	*dst = tmp_acc;
}

void mean128f( float* src, float* dst, int len)
{
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

	__attribute__ ((aligned (SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f,0.0f,0.0f,0.0f};
	float tmp_acc = 0.0f;
	v4sf vec_acc = _mm_setzero_ps(); //initialize the vector accumulator
	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf vec_tmp = _mm_load_ps(src + i);
			vec_acc        = _mm_add_ps(vec_acc, vec_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf vec_tmp = _mm_loadu_ps(src + i);
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
			v4sf src_tmp   = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp   = _mm_loadu_ps(src + i);
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
			v4sf src_tmp   = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp   = _mm_loadu_ps(src + i);
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
			v4sf src_tmp   = _mm_load_ps(src + i);
			_mm_store_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_FLOAT){
			v4sf src_tmp   = _mm_loadu_ps(src + i);
			_mm_storeu_ps(dst + i, _mm_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = floorf(src[i]);
	}
}
