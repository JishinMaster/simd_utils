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
#include "sse2neon_wrapper.h"
#endif

#include <math.h>
#include <string.h>

static inline size_t strnlen_s_128(const char *s, size_t maxlen) {
    if (s == NULL)
        return 0;

    size_t stop_len = maxlen / (SSE_LEN_BYTES);
    stop_len *= (SSE_LEN_BYTES);

    if (isAligned((uintptr_t) (s), SSE_LEN_BYTES)) {
        for (size_t i = 0; i < stop_len; i += SSE_LEN_BYTES) {
			// Load 16 bytes from the string 
			__m128i chunk = _mm_load_si128((const __m128i *)(s + i));
			// Compare each byte in the chunk with zero
			__m128i cmp = _mm_cmpeq_epi8(chunk, _mm_setzero_si128());
			// Create a 16-bit mask where each bit corresponds to a byte comparison.
			size_t mask = _mm_movemask_epi8(cmp);
			if (mask != 0) {
				// A zero byte was found. Use count trailing zeros (ctz)
				// to locate the index of the first zero.
				size_t offset = __builtin_ctz(mask);  // For GCC/Clang. On MSVC, use _BitScanForward.
				return i + offset;
			}
        }
    } else {
		 for (size_t i = 0; i < stop_len; i += SSE_LEN_BYTES) {
			__m128i chunk = _mm_loadu_si128((const __m128i *)(s + i));
			__m128i cmp = _mm_cmpeq_epi8(chunk, _mm_setzero_si128());
			size_t mask = _mm_movemask_epi8(cmp);
			if (mask != 0) {
				size_t offset = __builtin_ctz(mask);
				return i + offset;
			}
        }
    }

    for (int i = stop_len; i < maxlen; i++) {
		if (s[i] == '\0') {
            return i;
        }
    }

	return maxlen;
}