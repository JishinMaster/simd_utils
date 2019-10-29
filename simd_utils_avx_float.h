/*
 * Project : SIMD_Utils
 * Version : 0.1.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <stdint.h>
#include "immintrin.h"


#if 0
/* Compare */
_CMP_EQ_OQ    0x00 /* Equal (ordered, non-signaling)  */
_CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
_CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
_CMP_UNORD_Q  0x03 /* Unordered (non-signaling)  */
_CMP_NEQ_UQ   0x04 /* Not-equal (unordered, non-signaling)  */
_CMP_NLT_US   0x05 /* Not-less-than (unordered, signaling)  */
_CMP_NLE_US   0x06 /* Not-less-than-or-equal (unordered, signaling)  */
_CMP_ORD_Q    0x07 /* Ordered (nonsignaling)   */
_CMP_EQ_UQ    0x08 /* Equal (unordered, non-signaling)  */
_CMP_NGE_US   0x09 /* Not-greater-than-or-equal (unord, signaling)  */
_CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */
_CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
_CMP_NEQ_OQ   0x0c /* Not-equal (ordered, non-signaling)  */
_CMP_GE_OS    0x0d /* Greater-than-or-equal (ordered, signaling)  */
_CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
_CMP_TRUE_UQ  0x0f /* True (unordered, non-signaling)  */
_CMP_EQ_OS    0x10 /* Equal (ordered, signaling)  */
_CMP_LT_OQ    0x11 /* Less-than (ordered, non-signaling)  */
_CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
_CMP_UNORD_S  0x13 /* Unordered (signaling)  */
_CMP_NEQ_US   0x14 /* Not-equal (unordered, signaling)  */
_CMP_NLT_UQ   0x15 /* Not-less-than (unordered, non-signaling)  */
_CMP_NLE_UQ   0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
_CMP_ORD_S    0x17 /* Ordered (signaling)  */
_CMP_EQ_US    0x18 /* Equal (unordered, signaling)  */
_CMP_NGE_UQ   0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
_CMP_NGT_UQ   0x1a /* Not-greater-than (unordered, non-signaling)  */
_CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
_CMP_NEQ_OS   0x1c /* Not-equal (ordered, signaling)  */
_CMP_GE_OQ    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
_CMP_GT_OQ    0x1e /* Greater-than (ordered, non-signaling)  */
_CMP_TRUE_US  0x1f /* True (unordered, signaling)  */
#endif

// For tanf
_PS256_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS256_CONST(DP1,   -0.78515625);
_PS256_CONST(DP2,   -2.4187564849853515625e-4);
_PS256_CONST(DP3,   -3.77489497744594108e-8);

_PS256_CONST(FOPI, 1.27323954473516); /* 4/pi */
_PS256_CONST(TAN_P0, 9.38540185543E-3);
_PS256_CONST(TAN_P1, 3.11992232697E-3);
_PS256_CONST(TAN_P2, 2.44301354525E-2);
_PS256_CONST(TAN_P3, 5.34112807005E-2);
_PS256_CONST(TAN_P4, 1.33387994085E-1);
_PS256_CONST(TAN_P5, 3.33331568548E-1);

void log10_256f(float* src, float* dst, int len)
{
	float invln10f_mask = (float)INVLN10;
	const v8sf invln10f = _mm256_set1_ps((float)INVLN10); //_mm256_broadcast_ss(&invln10f_mask);

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = log256_ps(_mm256_load_ps(src + i));
			_mm256_store_ps(dst + i, _mm256_mul_ps(src_tmp, invln10f));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = log256_ps(_mm256_loadu_ps(src + i));
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
	const v8sf mask = _mm256_castsi256_ps (_mm256_set1_epi32 (0x7FFFFFFF));

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_and_ps (mask, src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_and_ps (mask, src_tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = fabsf(src[i]);
	}		
}

void set256f( float* src, float value, int len)
{
	const v8sf tmp = _mm256_set1_ps(value); //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

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
	const v8sf tmp = _mm256_setzero_ps();

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


void copy256f( float* src,  float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_load_ps(src + i));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i];
	}
}

void add256f( float* src1, float* src2, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_add_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] + src2[i];
	}
}


void mul256f( float* src1, float* src2, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_mul_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] * src2[i];
	}
}

void sub256f( float* src1, float* src2, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_sub_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] - src2[i];
	}
}


void addc256f( float* src, float value, float* dst, int len)
{
	const v8sf tmp = _mm256_set1_ps(value); //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

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
	const v8sf tmp = _mm256_set1_ps(value); //_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

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
	int stop_len = len/SSE_LEN_FLOAT;
	stop_len    *= SSE_LEN_FLOAT;

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
			v8sf src_tmp = _mm256_load_ps(src + i);	//load a,b,c,d,e,f,g,h
			v8sf src_tmp_flip = _mm256_permute2f128_ps (src_tmp, src_tmp, IMM8_PERMUTE_128BITS_LANES); // reverse lanes abcdefgh to efghabcd
			_mm256_storeu_ps(dst + len -i - AVX_LEN_FLOAT, _mm256_permute_ps (src_tmp_flip, IMM8_FLIP_VEC)); //store the flipped vector
		}
	}
	else{
		/*for(int i = AVX_LEN_FLOAT; i < stop_len; i+= AVX_LEN_FLOAT){
			__m128 src_tmp = _mm256_loadu_ps(src + i); //load a,b,c,d,e,f,g,h
			__m128 src_tmp_flip = _mm256_shuffle_ps (src_tmp, src_tmp, IMM8_FLIP_VEC);// rotate vec from abcd to bcba
			_mm256_storeu_ps(dst + len -i - AVX_LEN_FLOAT, src_tmp_flip); //store the flipped vector
		}*/
	}

	for(int i = stop_len; i <  len; i++){
		dst[len  -i - 1] = src[i];
	}
}

void maxevery256f( float* src1, float* src2, float* dst,  int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_max_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_max_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i]>src2[i]?src1[i]:src2[i];
	}
}

void minevery256f( float* src1, float* src2, float* dst,  int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_min_ps(_mm256_load_ps(src1 + i), _mm256_load_ps(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_min_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i]<src2[i]?src1[i]:src2[i];
	}
}

void threshold256_lt_f( float* src, float* dst, float value, int len)
{
	v8sf tmp = _mm256_set1_ps(value);//_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_min_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_min_ps(src_tmp,tmp));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i]<value?src[i]:value;
	}	
}

void threshold256_gt_f( float* src, float* dst, float value, int len)
{
	v8sf tmp = _mm256_set1_ps(value);//_mm256_broadcast_ss(&value); //avx broadcast vs mm_set_ps?

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_max_ps(src_tmp,tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
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
			v8sf src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, sin256_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
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
			v8sf src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, cos256_ps(src_tmp));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
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
			v8sf src_tmp = _mm256_load_ps(src + i);
			v8sf dst_sin_tmp;
			v8sf dst_cos_tmp;
			sincos256_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
			_mm256_store_ps(dst_sin + i, dst_sin_tmp);
			_mm256_store_ps(dst_cos + i, dst_cos_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
			v8sf dst_sin_tmp;
			v8sf dst_cos_tmp;
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


#ifndef __AVX2__ // Needs AVX2 to  get _mm256_cmpgt_epi32
#warning "Using SSE2 to perform AVX2 integer ops"
AVX2_INTOP_USING_SSE2(cmpgt_epi32)
#endif

#if 1

v8sf tan256f_ps(v8sf xx, const v8sf positive_mask, const v8sf negative_mask)
{
	v8sf x, y, z, zz;
	v8si j; //long?
	v8sf sign, xsupem4;
	v8sf tmp;
	v8si jandone, jandtwo;

	x = _mm256_and_ps (positive_mask, xx); //fabs(xx)

	/* compute x mod PIO4 */

	//TODO : on neg values should be ceil and not floor
	j = _mm256_cvtps_epi32( _mm256_round_ps(_mm256_mul_ps(*(v8sf*)_ps256_FOPI,x), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC )); /* integer part of x/(PI/4), using floor */
	y = _mm256_cvtepi32_ps(j);


#ifndef __AVX2__
  v4si andone_gt_0, andone_gt_1;
  v8si andone_gt;
  v4si j_0, j_1;
  COPY_IMM_TO_XMM(j,j_0,j_1);

  //FT: 0 1 and not 1 0?
  andone_gt_0 = _mm_and_si128(j_0,*(v4si*)_pi32avx_1);
  andone_gt_1 = _mm_and_si128(j_1,*(v4si*)_pi32avx_1);
  COPY_XMM_TO_IMM(andone_gt_0,andone_gt_1,andone_gt);
  jandone = _mm256_cmpgt_epi32(andone_gt,_mm256_setzero_si256 ());
#else
  jandone = _mm256_cmpgt_epi32(_mm256_and_si256(j,*(v8si*)_pi32_256_1),_mm256_setzero_si256 ());
#endif

	y = _mm256_blendv_ps ( y, _mm256_add_ps(y,*(v8sf*)_ps256_1),_mm256_cvtepi32_ps(jandone));
	j = _mm256_cvtps_epi32( y); // no need to round again

	//z = ((x - y * DP1) - y * DP2) - y * DP3;

#if 1
	tmp = _mm256_mul_ps(y, *(v8sf*)_ps256_DP1);
	z   = _mm256_add_ps(x, tmp);
	tmp = _mm256_mul_ps(y, *(v8sf*)_ps256_DP2);
	z   = _mm256_add_ps(z, tmp);
	tmp = _mm256_mul_ps(y, *(v8sf*)_ps256_DP3);
	z   = _mm256_add_ps(z, tmp);
#else // faster but less precision
	tmp = _mm256_mul_ps(y,*(v8sf*)_ps256_DP123);
	z   = _mm256_sub_ps(x, tmp);
#endif
	zz = _mm256_mul_ps(z,z); //z*z

	//TODO : should not be computed if X < 10e-4
	/* 1.7e-8 relative error in [-pi/4, +pi/4] */
	tmp =  _mm256_mul_ps(*(v8sf*)_ps256_TAN_P0, zz);
	tmp =  _mm256_add_ps(*(v8sf*)_ps256_TAN_P1, tmp);
	tmp =  _mm256_mul_ps(zz, tmp);
	tmp =  _mm256_add_ps(*(v8sf*)_ps256_TAN_P2, tmp);
	tmp =  _mm256_mul_ps(zz, tmp);
	tmp =  _mm256_add_ps(*(v8sf*)_ps256_TAN_P3, tmp);
	tmp =  _mm256_mul_ps(zz, tmp);
	tmp =  _mm256_add_ps(*(v8sf*)_ps256_TAN_P4, tmp);
	tmp =  _mm256_mul_ps(zz, tmp);
	tmp =  _mm256_add_ps(*(v8sf*)_ps256_TAN_P5, tmp);
	tmp =  _mm256_mul_ps(zz, tmp);
	tmp =  _mm256_mul_ps(z, tmp);
	tmp =  _mm256_add_ps(z, tmp);

	xsupem4 = _mm256_cmp_ps(x,_mm256_set1_ps(1.0e-4), _CMP_GT_OS); 	//if( x > 1.0e-4 )
	y       = _mm256_blendv_ps ( z, tmp, xsupem4);

#ifndef __AVX2__
  v4si andtwo_gt_0, andtwo_gt_1;
  v8si andtwo_gt;
  COPY_IMM_TO_XMM(j,j_0,j_1);
  andtwo_gt_0 = _mm_and_si128(j_0,*(v4si*)_pi32avx_2);
  andtwo_gt_1 = _mm_and_si128(j_1,*(v4si*)_pi32avx_2);
  COPY_XMM_TO_IMM(andtwo_gt_0,andtwo_gt_1,andtwo_gt);
  jandtwo = _mm256_cmpgt_epi32(andtwo_gt,_mm256_setzero_si256 ());
#else
  jandtwo = _mm256_cmpgt_epi32(_mm256_and_si256(j,*(v8si*)_pi32_256_2),_mm256_setzero_si256 ());
#endif

	y       = _mm256_blendv_ps ( y, _mm256_div_ps(_mm256_set1_ps(-1.0f),y),_mm256_cvtepi32_ps(jandtwo));

	sign   = _mm256_cmp_ps(xx,_mm256_setzero_ps(), _CMP_LT_OS); //0xFFFFFFFF if xx < 0.0
	y      = _mm256_blendv_ps (y, _mm256_xor_ps(negative_mask,y), sign);

	return( y );
}

void tan256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	const v8sf positive_mask = _mm256_castsi256_ps (_mm256_set1_epi32 (0x7FFFFFFF));
	const v8sf negative_mask = _mm256_castsi256_ps (_mm256_set1_epi32 (~0x7FFFFFFF));

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, tan256f_ps(src_tmp, positive_mask, negative_mask));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, tan256f_ps(src_tmp, positive_mask, negative_mask));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = tanf(src[i]);
	}
}

#else

void tan256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_div_ps(sin256_ps(src_tmp),cos256_ps(src_tmp)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_div_ps(sin256_ps(src_tmp),cos256_ps(src_tmp)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = tanf(src[i]);
	}
}
#endif

void magnitude256f_split( float* srcRe, float* srcIm, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(srcRe) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf re_tmp = _mm256_load_ps(srcRe + i);
			v8sf re2    = _mm256_mul_ps(re_tmp,re_tmp);
			v8sf im_tmp = _mm256_load_ps(srcIm + i);
			v8sf im2    = _mm256_mul_ps(im_tmp,im_tmp);
			_mm256_store_ps(dst + i, _mm256_sqrt_ps(_mm256_add_ps(re2,im2)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf re_tmp = _mm256_loadu_ps(srcRe + i);
			v8sf re2    = _mm256_mul_ps(re_tmp,re_tmp);
			v8sf im_tmp = _mm256_loadu_ps(srcIm + i);
			v8sf im2    = _mm256_mul_ps(im_tmp,im_tmp);
			_mm256_store_ps(dst + i, _mm256_sqrt_ps(_mm256_add_ps(re2,im2)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrtf(srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i]);
	}
}

void powerspect256f_split( float* srcRe, float* srcIm, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(srcRe) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf re_tmp = _mm256_load_ps(srcRe + i);
			v8sf re2    = _mm256_mul_ps(re_tmp,re_tmp);
			v8sf im_tmp = _mm256_load_ps(srcIm + i);
			v8sf im2    = _mm256_mul_ps(im_tmp,im_tmp);
			_mm256_store_ps(dst + i, _mm256_add_ps(re2,im2));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf re_tmp = _mm256_loadu_ps(srcRe + i);
			v8sf re2    = _mm256_mul_ps(re_tmp,re_tmp);
			v8sf im_tmp = _mm256_loadu_ps(srcIm + i);
			v8sf im2    = _mm256_mul_ps(im_tmp,im_tmp);
			_mm256_store_ps(dst + i, _mm256_add_ps(re2,im2));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = srcRe[i]*srcRe[i] + srcIm[i]*srcIm[i];
	}
}

void subcrev256f( float* src, float value, float* dst, int len)
{
	const v8sf tmp = _mm256_set1_ps(value);

	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_store_ps(dst + i, _mm256_sub_ps(tmp, _mm256_load_ps(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			_mm256_storeu_ps(dst + i, _mm256_sub_ps(tmp, _mm256_loadu_ps(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] + value;
	}
}

void sum256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	__attribute__ ((aligned (AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
	float tmp_acc = 0.0f;
	v8sf vec_acc = _mm256_setzero_ps(); //initialize the vector accumulator
	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf vec_tmp   = _mm256_load_ps(src + i);
			vec_acc        = _mm256_add_ps(vec_acc, vec_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf vec_tmp   = _mm256_loadu_ps(src + i);
			vec_acc        = _mm256_add_ps(vec_acc, vec_tmp);
		}
	}

	_mm256_store_ps(accumulate , vec_acc);

	for(int i = stop_len; i < len; i++){
		tmp_acc += src[i];
	}

	tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3] + accumulate[4] + accumulate[5] + accumulate[6] + accumulate[7];

	*dst = tmp_acc;
}


void mean256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	__attribute__ ((aligned (AVX_LEN_BYTES))) float accumulate[AVX_LEN_FLOAT] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
	float tmp_acc = 0.0f;
	v8sf vec_acc = _mm256_setzero_ps(); //initialize the vector accumulator
	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf vec_tmp = _mm256_load_ps(src + i);
			vec_acc        = _mm256_add_ps(vec_acc, vec_tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf vec_tmp = _mm256_loadu_ps(src + i);
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
			v8sf src_tmp   = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp   = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = roundf(src[i]);
	}
}

void ceil256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp   = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp   = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = ceilf(src[i]);
	}
}

void floor256f( float* src, float* dst, int len)
{
	int stop_len = len/AVX_LEN_FLOAT;
	stop_len    *= AVX_LEN_FLOAT;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp   = _mm256_load_ps(src + i);
			_mm256_store_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_FLOAT){
			v8sf src_tmp   = _mm256_loadu_ps(src + i);
			_mm256_storeu_ps(dst + i, _mm256_round_ps(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = floorf(src[i]);
	}
}


