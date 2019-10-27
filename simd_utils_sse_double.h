/*
 * Project : SIMD_Utils
 * Version : 0.1.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <stdint.h>
#include "immintrin.h"


typedef __m128d  v2sd; // vector of 2 double (sse)
typedef __m128i  v2si; // vector of 2 int 64 (sse)

void set128d( double* src, double value, int len)
{
	const v2sd tmp = _mm_set1_pd(value);

	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(src + i, tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		src[i] = value;
	}
}

void zero128d( double* src, int len)
{
	const v2sd tmp = _mm_setzero_pd();

	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(src + i, tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		src[i] = 0.0;
	}
}

void copy128d( double* src,  double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_load_pd(src + i));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_loadu_pd(src + i));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i];
	}
}

void sqrt128d( double* src, double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_sqrt_pd( _mm_load_pd(src + i) ) );
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_sqrt_pd( _mm_loadu_pd(src + i) ) );
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrt(src[i]);
	}
}

void add128d( double* src1, double* src2, double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_add_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_add_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] + src2[i];
	}
}

void mul128d( double* src1, double* src2, double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_mul_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_mul_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] * src2[i];
	}
}

void sub128d( double* src1, double* src2, double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_sub_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_sub_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] - src2[i];
	}
}

void div128d( double* src1, double* src2, double* dst, int len)
{

	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_div_pd(_mm_load_pd(src1 + i), _mm_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_div_pd(_mm_loadu_pd(src1 + i), _mm_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] / src2[i];
	}
}

//TODO : "Immediate add/mul?"
void addc128d( double* src, double value, double* dst, int len)
{
	const v2sd tmp = _mm_set1_pd(value);

	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_add_pd(tmp, _mm_load_pd(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_add_pd(tmp, _mm_loadu_pd(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] + value;
	}
}

void mulc128d( double* src, double value, double* dst, int len)
{
	const v2sd tmp = _mm_set1_pd(value);

	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_store_pd(dst + i, _mm_mul_pd(tmp, _mm_load_pd(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			_mm_storeu_pd(dst + i, _mm_add_pd(tmp, _mm_loadu_pd(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] * value;
	}
}

void round128d( double* src, double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			v2sd src_tmp   = _mm_load_pd(src + i);
			_mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			v2sd src_tmp   = _mm_loadu_pd(src + i);
			_mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = round(src[i]);
	}
}

void ceil128d( double* src, double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			v2sd src_tmp   = _mm_load_pd(src + i);
			_mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			v2sd src_tmp   = _mm_loadu_pd(src + i);
			_mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = ceil(src[i]);
	}
}

void floor128d( double* src, double* dst, int len)
{
	int stop_len = len/SSE_LEN_DOUBLE;
	stop_len    *= SSE_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			v2sd src_tmp   = _mm_load_pd(src + i);
			_mm_store_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_DOUBLE){
			v2sd src_tmp   = _mm_loadu_pd(src + i);
			_mm_storeu_pd(dst + i, _mm_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = floor(src[i]);
	}
}
