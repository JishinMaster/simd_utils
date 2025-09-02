/*
 * Project : SIMD_Utils
 * Version : 0.2.6
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once
#include <fenv.h>
#include <stdint.h>
#ifndef ARM
#include <immintrin.h>
#else
#include "AVX2neon_wrapper.h"
#endif

#include <math.h>
#include <string.h>

#ifdef __AVX2__
static inline size_t strnlen_s_512(const char *s, size_t maxlen) {
    if (s == NULL)
        return 0;

    size_t stop_len = maxlen / (AVX512_LEN_BYTES);
    stop_len *= (AVX512_LEN_BYTES);

    if (isAligned((uintptr_t) (s), AVX512_LEN_BYTES)) {
        for (size_t i = 0; i < stop_len; i += AVX512_LEN_BYTES) {
            __m512i chunk = _mm512_load_si512((const __m512i *)(s + i));
            __mmask64 mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_setzero_si512());
            if (mask) {
                int offset = __builtin_ctzll(mask); // __builtin_ctzll works on 64-bit masks.
				return i + offset;
			}
        }
    } else {
		 for (size_t i = 0; i < stop_len; i += AVX512_LEN_BYTES) {
			__m512i chunk = _mm512_loadu_si512((const __m512i *)(s + i));
            __mmask64 mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_setzero_si512());
            if (mask) {
                int offset = __builtin_ctzll(mask); // __builtin_ctzll works on 64-bit masks.
				return i + offset;
			}
        }
    }

	if((maxlen - stop_len) > 32){
		__m256i chunk = _mm256_loadu_si256((const __m256i *)(s + stop_len));
		__m256i cmp = _mm256_cmpeq_epi8(chunk, _mm256_setzero_si256());
		size_t mask = _mm256_movemask_epi8(cmp);
		if (mask != 0) {
			size_t offset = __builtin_ctz(mask);
			return stop_len + offset;
		}
		stop_len+=32;
	}

	if((maxlen - stop_len) > 16){
		__m128i chunk = _mm_loadu_si128((const __m128i *)(s + stop_len));
		__m128i cmp = _mm_cmpeq_epi8(chunk, _mm_setzero_si128());
		size_t mask = _mm_movemask_epi8(cmp);
		if (mask != 0) {
			size_t offset = __builtin_ctz(mask);
			return stop_len + offset;
		}
		stop_len+=16;
	}

    for (int i = stop_len; i < maxlen; i++) {
		if (s[i] == '\0') {
            return i;
        }
    }

	return maxlen;
}

#endif