/*
 * Project : SIMD_Utils
 * Version : 0.1.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <stdint.h>
#ifndef ARM
#include <immintrin.h>
#else
#include "sse2neon.h"
#endif


void add128s( int32_t* src1, int32_t* src2, int32_t* dst, int len)
{
	int stop_len = len/SSE_LEN_INT32;
	stop_len    *= SSE_LEN_INT32;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_store_si128(dst + i, _mm_add_epi32(_mm_load_si128(src1 + i), _mm_load_si128(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_storeu_si128(dst + i, _mm_add_epi32(_mm_loadu_si128(src1 + i), _mm_loadu_si128(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] + src2[i];
	}
}

void mul128s( int32_t* src1, int32_t* src2, int32_t* dst, int len)
{
	int stop_len = len/SSE_LEN_INT32;
	stop_len    *= SSE_LEN_INT32;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_store_si128(dst + i, _mm_mul_epi32(_mm_load_si128(src1 + i), _mm_load_si128(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_storeu_si128(dst + i, _mm_mul_epi32(_mm_loadu_si128(src1 + i), _mm_loadu_si128(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] * src2[i];
	}
}

void sub128s( int32_t* src1, int32_t* src2, int32_t* dst, int len)
{
	int stop_len = len/SSE_LEN_INT32;
	stop_len    *= SSE_LEN_INT32;

	if( ( (uintptr_t)(const void*)(src1) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_store_si128(dst + i, _mm_sub_epi32(_mm_load_si128(src1 + i), _mm_load_si128(src2 + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_storeu_si128(dst + i, _mm_sub_epi32(_mm_loadu_si128(src1 + i), _mm_loadu_si128(src2 + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src1[i] - src2[i];
	}
}

void addc128s( int32_t* src, int32_t value, int32_t* dst, int len)
{
	int stop_len = len/SSE_LEN_INT32;
	stop_len    *= SSE_LEN_INT32;

	const v4si tmp = _mm_set1_epi32(value);

	if( ( (uintptr_t)(const void*)(src) % SSE_LEN_BYTES) == 0){
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_store_si128(dst + i, _mm_add_epi32(tmp, _mm_load_si128(src + i)));
		}
	}
	else{
		for(int i = 0; i < stop_len; i+= SSE_LEN_INT32){
			_mm_storeu_si128(dst + i, _mm_add_epi32(tmp, _mm_loadu_si128(src + i)));
		}
	}

	for(int i = stop_len; i < len; i++){
		dst[i] = src[i] + value;
	}
}
