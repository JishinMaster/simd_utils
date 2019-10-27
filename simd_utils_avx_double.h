/*
 * Project : SIMD_Utils
 * Version : 0.1.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <stdint.h>
#include "immintrin.h"

typedef __m256d  v4sd; // vector of 4 double (avx)

void set256d( double* src, double value, int len)
{
	const v4sd tmp = _mm256_set1_pd(value);

	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(src + i, tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		src[i] = value;
	}
}

void zero256d( double* src, int len)
{
	const v4sd tmp = _mm256_setzero_pd();

	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(src + i, tmp);
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(src + i, tmp);
		}
	}

	for(int i = stop_len; i < len; i++){
		src[i] = 0.0;
	}
}

void copy256d( double* src,  double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_load_pd(src + i));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i];
	}
}

void sqrt256d( double* src, double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_sqrt_pd( _mm256_load_pd(src + i) ) );
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_sqrt_pd( _mm256_loadu_pd(src + i) ) );
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = sqrt(src[i]);
	}
}

void add256d( double* src1, double* src2, double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_add_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_add_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] + src2[i];
	}
}

void mul256d( double* src1, double* src2, double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_mul_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] * src2[i];
	}
}

void sub256d( double* src1, double* src2, double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_sub_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_sub_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] - src2[i];
	}
}

void div256d( double* src1, double* src2, double* dst, int len)
{

	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src1) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_div_pd(_mm256_load_pd(src1 + i), _mm256_load_pd(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_div_pd(_mm256_loadu_pd(src1 + i), _mm256_loadu_pd(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] / src2[i];
	}
}

//TODO : "Immediate add/mul?"
void addc256d( double* src, double value, double* dst, int len)
{
	const v4sd tmp = _mm256_set1_pd(value);

	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_add_pd(tmp, _mm256_load_pd(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_add_pd(tmp, _mm256_loadu_pd(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] + value;
	}
}

void mulc256d( double* src, double value, double* dst, int len)
{
	const v4sd tmp = _mm256_set1_pd(value);

	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_store_pd(dst + i, _mm256_mul_pd(tmp, _mm256_load_pd(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			_mm256_storeu_pd(dst + i, _mm256_add_pd(tmp, _mm256_loadu_pd(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] * value;
	}
}

void round256d( double* src, double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			v4sd src_tmp   = _mm256_load_pd(src + i);
			_mm256_store_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			v4sd src_tmp   = _mm256_loadu_pd(src + i);
			_mm256_storeu_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = round(src[i]);
	}
}

void ceil256d( double* src, double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			v4sd src_tmp   = _mm256_load_pd(src + i);
			_mm256_store_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			v4sd src_tmp   = _mm256_loadu_pd(src + i);
			_mm256_storeu_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = ceil(src[i]);
	}
}

void floor256d( double* src, double* dst, int len)
{
	int stop_len = len/AVX_LEN_DOUBLE;
	stop_len    *= AVX_LEN_DOUBLE;

	if( ( (uintptr_t)(const void*)(src) % AVX_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			v4sd src_tmp   = _mm256_load_pd(src + i);
			_mm256_store_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= AVX_LEN_DOUBLE){
			v4sd src_tmp   = _mm256_loadu_pd(src + i);
			_mm256_storeu_pd(dst + i, _mm256_round_pd(src_tmp, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC ));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = floor(src[i]);
	}
}
