/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/
#include <immintrin.h>

#ifndef FMA

/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
static inline v8sf log256_ps(v8sf x)
{
    v8si imm0;
    v8sf one = *(v8sf *) _ps256_1;

    // v8sf invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
    v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, *(v8sf *) _ps256_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(v8sf *) _ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, *(v8si *) _pi32_256_0x7f);
    v8sf e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
    // v8sf mask = _mm256_cmplt_ps(x, *(v8sf*)_ps256_cephes_SQRTHF);
    v8sf mask = _mm256_cmp_ps(x, *(v8sf *) _ps256_cephes_SQRTHF, _CMP_LT_OS);
    v8sf tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    v8sf z = _mm256_mul_ps(x, x);

    v8sf y = *(v8sf *) _ps256_cephes_log_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    tmp = _mm256_mul_ps(e, *(v8sf *) _ps256_cephes_log_q1);
    y = _mm256_add_ps(y, tmp);


    tmp = _mm256_mul_ps(z, *(v8sf *) _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, *(v8sf *) _ps256_cephes_log_q2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline v8sf exp256_ps(v8sf x)
{
    v8sf tmp = _mm256_setzero_ps(), fx;
    v8si imm0;
    v8sf one = *(v8sf *) _ps256_1;

    x = _mm256_min_ps(x, *(v8sf *) _ps256_exp_hi);
    x = _mm256_max_ps(x, *(v8sf *) _ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_mul_ps(x, *(v8sf *) _ps256_cephes_LOG2EF);
    fx = _mm256_add_ps(fx, *(v8sf *) _ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    // imm0 = _mm256_cvttps_epi32(fx);
    // tmp  = _mm256_cvtepi32_ps(imm0);

    tmp = _mm256_floor_ps(fx);

    /* if greater, substract 1 */
    // v8sf mask = _mm256_cmpgt_ps(tmp, fx);
    v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, *(v8sf *) _ps256_cephes_exp_C1);
    v8sf z = _mm256_mul_ps(fx, *(v8sf *) _ps256_cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x, x);

    v8sf y = *(v8sf *) _ps256_cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, *(v8si *) _pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    v8sf pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
static inline v8sf sin256_ps(v8sf x)
{  // any x
    v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
    v8si imm0, imm2;

#ifndef __AVX2__
    v4si imm0_1, imm0_2;
    v4si imm2_1, imm2_2;
#endif

    sign_bit = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_ps(sign_bit, *(v8sf *) _ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf *) _ps256_cephes_FOPI);

    /*
    Here we start a series of integer operations, which are in the
    realm of AVX2.
    If we don't have AVX, let's perform them using SSE2 directives
  */

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm256_add_epi32(imm2, *(v8si *) _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_inv1);  // FT
    y = _mm256_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    imm0 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_4);  // FT
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_2);  // FT
    imm2 = _mm256_cmpeq_epi32(imm2, *(v8si *) _pi32_256_0);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(v4si *) _pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(v4si *) _pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm0_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    v8sf swap_sign_bit = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);
    sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(v8sf *) _ps256_minus_cephes_DP1;
    xmm2 = *(v8sf *) _ps256_minus_cephes_DP2;
    xmm3 = *(v8sf *) _ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(v8sf *) _ps256_coscof_p0;
    v8sf z = _mm256_mul_ps(x, x);

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, *(v8sf *) _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = *(v8sf *) _ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf *) _ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf *) _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm256_blendv_ps(y, y2, poly_mask);
#else
    y2 = _mm256_and_ps(poly_mask, y2);  //, xmm3);
    y = _mm256_andnot_ps(poly_mask, y);
    y = _mm256_add_ps(y, y2);
#endif

    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
static inline v8sf cos256_ps(v8sf x)
{  // any x
    v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
    v8si imm0, imm2;

#ifndef __AVX2__
    v4si imm0_1, imm0_2;
    v4si imm2_1, imm2_2;
#endif

    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf *) _ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, *(v8si *) _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm2 = _mm256_sub_epi32(imm2, *(v8si *) _pi32_256_2);

    /* get the swap sign flag */
    imm0 = _mm256_andnot_si256(imm2, *(v8si *) _pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(v8si *) _pi32_256_0);
#else

    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(v4si *) _pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(v4si *) _pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm2_1 = _mm_sub_epi32(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_sub_epi32(imm2_2, *(v4si *) _pi32avx_2);

    imm0_1 = _mm_andnot_si128(imm2_1, *(v4si *) _pi32avx_4);
    imm0_2 = _mm_andnot_si128(imm2_2, *(v4si *) _pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    v8sf sign_bit = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(v8sf *) _ps256_minus_cephes_DP1;
    xmm2 = *(v8sf *) _ps256_minus_cephes_DP2;
    xmm3 = *(v8sf *) _ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(v8sf *) _ps256_coscof_p0;
    v8sf z = _mm256_mul_ps(x, x);

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, *(v8sf *) _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = *(v8sf *) _ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf *) _ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf *) _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm256_blendv_ps(y, y2, poly_mask);
#else
    y2 = _mm256_and_ps(poly_mask, y2);  //, xmm3);
    y = _mm256_andnot_ps(poly_mask, y);
    y = _mm256_add_ps(y, y2);
#endif
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
static inline void sincos256_ps(v8sf x, v8sf *s, v8sf *c)
{
    v8sf xmm1, xmm2, xmm3, sign_bit_sin, y;
    v8si imm0, imm2, imm4;

#ifndef __AVX2__
    v4si imm0_1, imm0_2;
    v4si imm2_1, imm2_2;
    v4si imm4_1, imm4_2;
#endif

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(v8sf *) _ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf *) _ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in imm2 */
    imm2 = _mm256_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, *(v8si *) _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_inv1);

    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    // v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(v8si *) _pi32_256_0);
    // v8sf poly_mask = _mm256_castsi256_ps(imm2);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(v4si *) _pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(v4si *) _pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm4_1 = imm2_1;
    imm4_2 = imm2_2;

    imm0_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif
    v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(v8sf *) _ps256_minus_cephes_DP1;
    xmm2 = *(v8sf *) _ps256_minus_cephes_DP2;
    xmm3 = *(v8sf *) _ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

#ifdef __AVX2__
    imm4 = _mm256_sub_epi32(imm4, *(v8si *) _pi32_256_2);
    imm4 = _mm256_andnot_si256(imm4, *(v8si *) _pi32_256_4);
    imm4 = _mm256_slli_epi32(imm4, 29);
#else
    imm4_1 = _mm_sub_epi32(imm4_1, *(v4si *) _pi32avx_2);
    imm4_2 = _mm_sub_epi32(imm4_2, *(v4si *) _pi32avx_2);

    imm4_1 = _mm_andnot_si128(imm4_1, *(v4si *) _pi32avx_4);
    imm4_2 = _mm_andnot_si128(imm4_2, *(v4si *) _pi32avx_4);

    imm4_1 = _mm_slli_epi32(imm4_1, 29);
    imm4_2 = _mm_slli_epi32(imm4_2, 29);

    COPY_XMM_TO_IMM(imm4_1, imm4_2, imm4);
#endif

    v8sf sign_bit_cos = _mm256_castsi256_ps(imm4);

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v8sf z = _mm256_mul_ps(x, x);
    y = *(v8sf *) _ps256_coscof_p0;

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, *(v8sf *) _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = *(v8sf *) _ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf *) _ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf *) _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
#if 1
    xmm1 = _mm256_blendv_ps(y, y2, poly_mask);
    xmm2 = _mm256_blendv_ps(y2, y, poly_mask);
#else
    v8sf ysin2 = _mm256_and_ps(poly_mask, y2);
    v8sf ysin1 = _mm256_andnot_ps(poly_mask, y);
    y2 = _mm256_sub_ps(y2, ysin2);
    y = _mm256_sub_ps(y, ysin1);
    xmm1 = _mm256_add_ps(ysin1, ysin2);
    xmm2 = _mm256_add_ps(y, y2);
#endif

    /* update the sign */
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

#else /* FMA */


/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
static inline v8sf log256_ps(v8sf x)
{
    v8si imm0;
    v8sf one = *(v8sf *) _ps256_1;

    v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, *(v8sf *) _ps256_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(v8sf *) _ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, *(v8si *) _pi32_256_0x7f);
    v8sf e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    v8sf mask = _mm256_cmp_ps(x, *(v8sf *) _ps256_cephes_SQRTHF, _CMP_LT_OS);
    v8sf tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    v8sf z = _mm256_mul_ps(x, x);

    v8sf y = _mm256_fmadd_ps(*(v8sf *) _ps256_cephes_log_p0, x, *(v8sf *) _ps256_cephes_log_p1);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_log_p2);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_log_p3);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_log_p4);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_log_p5);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_log_p6);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_log_p7);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    y = _mm256_fmadd_ps(e, *(v8sf *) _ps256_cephes_log_q1, y);
    y = _mm256_fnmadd_ps(z, *(v8sf *) _ps256_0p5, y);

    tmp = _mm256_fmadd_ps(e, *(v8sf *) _ps256_cephes_log_q2, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline v8sf exp256_ps(v8sf x)
{
    v8sf tmp = _mm256_setzero_ps(), fx;
    v8si imm0;
    v8sf one = *(v8sf *) _ps256_1;

    x = _mm256_min_ps(x, *(v8sf *) _ps256_exp_hi);
    x = _mm256_max_ps(x, *(v8sf *) _ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_fmadd_ps(x, *(v8sf *) _ps256_cephes_LOG2EF, *(v8sf *) _ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    tmp = _mm256_floor_ps(fx);

    /* if greater, substract 1 */
    v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    x = _mm256_fnmadd_ps(fx, *(v8sf *) _ps256_cephes_exp_C1, x);
    x = _mm256_fnmadd_ps(fx, *(v8sf *) _ps256_cephes_exp_C2, x);

    v8sf z = _mm256_mul_ps(x, x);

    v8sf y = _mm256_fmadd_ps(*(v8sf *) _ps256_cephes_exp_p0, x, *(v8sf *) _ps256_cephes_exp_p1);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_exp_p2);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_exp_p3);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_exp_p4);
    y = _mm256_fmadd_ps(y, x, *(v8sf *) _ps256_cephes_exp_p5);
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, *(v8si *) _pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    v8sf pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
static inline v8sf sin256_ps(v8sf x)
{  // any x
    v8sf xmm3, sign_bit, y;
    v8si imm0, imm2;

#ifndef __AVX2__
    v4si imm0_1, imm0_2;
    v4si imm2_1, imm2_2;
#endif

    sign_bit = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_ps(sign_bit, *(v8sf *) _ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf *) _ps256_cephes_FOPI);

    /*
    Here we start a series of integer operations, which are in the
    realm of AVX2.
    If we don't have AVX, let's perform them using SSE2 directives
  */

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm256_add_epi32(imm2, *(v8si *) _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_inv1);  // FT
    y = _mm256_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    imm0 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_4);  // FT
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_2);  // FT
    imm2 = _mm256_cmpeq_epi32(imm2, *(v8si *) _pi32_256_0);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(v4si *) _pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(v4si *) _pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm0_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    v8sf swap_sign_bit = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);
    sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP1, x);
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP2, x);
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v8sf z = _mm256_mul_ps(x, x);

    y = _mm256_fmadd_ps(*(v8sf *) _ps256_coscof_p0, z, *(v8sf *) _ps256_coscof_p1);
    y = _mm256_fmadd_ps(y, z, *(v8sf *) _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fnmadd_ps(z, *(v8sf *) _ps256_0p5, y);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = _mm256_fmadd_ps(*(v8sf *) _ps256_sincof_p0, z, *(v8sf *) _ps256_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, *(v8sf *) _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm256_blendv_ps(y, y2, poly_mask);
#else
    y2 = _mm256_and_ps(poly_mask, y2);  //, xmm3);
    y = _mm256_andnot_ps(poly_mask, y);
    y = _mm256_add_ps(y, y2);
#endif
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
static inline v8sf cos256_ps(v8sf x)
{  // any x
    v8sf xmm3, y;
    v8si imm0, imm2;

#ifndef __AVX2__
    v4si imm0_1, imm0_2;
    v4si imm2_1, imm2_2;
#endif

    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf *) _ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, *(v8si *) _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm2 = _mm256_sub_epi32(imm2, *(v8si *) _pi32_256_2);

    /* get the swap sign flag */
    imm0 = _mm256_andnot_si256(imm2, *(v8si *) _pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(v8si *) _pi32_256_0);
#else

    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(v4si *) _pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(v4si *) _pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm2_1 = _mm_sub_epi32(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_sub_epi32(imm2_2, *(v4si *) _pi32avx_2);

    imm0_1 = _mm_andnot_si128(imm2_1, *(v4si *) _pi32avx_4);
    imm0_2 = _mm_andnot_si128(imm2_2, *(v4si *) _pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    v8sf sign_bit = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP1, x);
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP2, x);
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v8sf z = _mm256_mul_ps(x, x);

    y = _mm256_fmadd_ps(*(v8sf *) _ps256_coscof_p0, z, *(v8sf *) _ps256_coscof_p1);
    y = _mm256_fmadd_ps(y, z, *(v8sf *) _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fnmadd_ps(z, *(v8sf *) _ps256_0p5, y);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = _mm256_fmadd_ps(*(v8sf *) _ps256_sincof_p0, z, *(v8sf *) _ps256_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, *(v8sf *) _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm256_blendv_ps(y, y2, poly_mask);
#else
    y2 = _mm256_and_ps(poly_mask, y2);  //, xmm3);
    y = _mm256_andnot_ps(poly_mask, y);
    y = _mm256_add_ps(y, y2);
#endif
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
static inline void sincos256_ps(v8sf x, v8sf *s, v8sf *c)
{
    v8sf xmm1, xmm2, sign_bit_sin, y;
    v8si imm0, imm2, imm4;

#ifndef __AVX2__
    v4si imm0_1, imm0_2;
    v4si imm2_1, imm2_2;
    v4si imm4_1, imm4_2;
#endif

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf *) _ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(v8sf *) _ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf *) _ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in imm2 */
    imm2 = _mm256_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, *(v8si *) _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_inv1);

    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    // v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm256_and_si256(imm2, *(v8si *) _pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(v8si *) _pi32_256_0);
    // v8sf poly_mask = _mm256_castsi256_ps(imm2);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(v4si *) _pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(v4si *) _pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm4_1 = imm2_1;
    imm4_2 = imm2_2;

    imm0_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(v4si *) _pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(v4si *) _pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif
    v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP1, x);
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP2, x);
    x = _mm256_fmadd_ps(y, *(v8sf *) _ps256_minus_cephes_DP3, x);

#ifdef __AVX2__
    imm4 = _mm256_sub_epi32(imm4, *(v8si *) _pi32_256_2);
    imm4 = _mm256_andnot_si256(imm4, *(v8si *) _pi32_256_4);
    imm4 = _mm256_slli_epi32(imm4, 29);
#else
    imm4_1 = _mm_sub_epi32(imm4_1, *(v4si *) _pi32avx_2);
    imm4_2 = _mm_sub_epi32(imm4_2, *(v4si *) _pi32avx_2);

    imm4_1 = _mm_andnot_si128(imm4_1, *(v4si *) _pi32avx_4);
    imm4_2 = _mm_andnot_si128(imm4_2, *(v4si *) _pi32avx_4);

    imm4_1 = _mm_slli_epi32(imm4_1, 29);
    imm4_2 = _mm_slli_epi32(imm4_2, 29);

    COPY_XMM_TO_IMM(imm4_1, imm4_2, imm4);
#endif

    v8sf sign_bit_cos = _mm256_castsi256_ps(imm4);

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v8sf z = _mm256_mul_ps(x, x);

    y = _mm256_fmadd_ps(*(v8sf *) _ps256_coscof_p0, z, *(v8sf *) _ps256_coscof_p1);
    y = _mm256_fmadd_ps(y, z, *(v8sf *) _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fnmadd_ps(z, *(v8sf *) _ps256_0p5, y);
    y = _mm256_add_ps(y, *(v8sf *) _ps256_1);


    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    v8sf y2 = _mm256_fmadd_ps(*(v8sf *) _ps256_sincof_p0, z, *(v8sf *) _ps256_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, *(v8sf *) _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    xmm1 = _mm256_blendv_ps(y, y2, poly_mask);
    xmm2 = _mm256_blendv_ps(y2, y, poly_mask);
#else
    v8sf ysin2 = _mm256_and_ps(poly_mask, y2);
    v8sf ysin1 = _mm256_andnot_ps(poly_mask, y);
    y2 = _mm256_sub_ps(y2, ysin2);
    y = _mm256_sub_ps(y, ysin1);
    xmm1 = _mm256_add_ps(ysin1, ysin2);
    xmm2 = _mm256_add_ps(y, y2);
#endif

    /* update the sign */
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

#endif /* FMA */
