/* NEON implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2011  Julien Pommier

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

// Update : use of FMA

#ifndef INC_SIMD_NEON_MATHFUN_H_
#define INC_SIMD_NEON_MATHFUN_H_

#include <arm_neon.h>

#include "sse2neon_wrapper.h"

#ifndef FMA

/* natural logarithm computed for 4 simultaneous float
   return NaN for x <= 0
*/
static inline v4sf log_ps(v4sf x)
{
    v4sf one = vdupq_n_f32(1);

    x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
    v4su invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

    v4si ux = vreinterpretq_s32_f32(x);

    v4si emm0 = vshrq_n_s32(ux, 23);

    /* keep only the fractional part */
    ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ux);

    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    v4sf e = vcvtq_f32_s32(emm0);

    e = vaddq_f32(e, one);

    /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
    v4su mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
    v4sf tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);

    v4sf z = vmulq_f32(x, x);

    v4sf y = vdupq_n_f32(c_cephes_log_p0);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
    y = vmulq_f32(y, x);

    y = vmulq_f32(y, z);


    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
    y = vaddq_f32(y, tmp);


    tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
    y = vsubq_f32(y, tmp);

    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
    x = vaddq_f32(x, y);
    x = vaddq_f32(x, tmp);
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask));  // negative arg will be NAN
    return x;
}

/* exp() computed for 4 float at once */
static inline v4sf exp_ps(v4sf x)
{
    v4sf tmp, fx;

    v4sf one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    v4su mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    v4sf z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    static const float cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    v4sf y = vld1q_dup_f32(cephes_exp_p + 0);
    v4sf c1 = vld1q_dup_f32(cephes_exp_p + 1);
    v4sf c2 = vld1q_dup_f32(cephes_exp_p + 2);
    v4sf c3 = vld1q_dup_f32(cephes_exp_p + 3);
    v4sf c4 = vld1q_dup_f32(cephes_exp_p + 4);
    v4sf c5 = vld1q_dup_f32(cephes_exp_p + 5);

    y = vmulq_f32(y, x);
    z = vmulq_f32(x, x);
    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c4);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c5);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    v4sf pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

/* evaluation of 4 sines & cosines at once.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

   Note also that when you compute sin(x), cos(x) is available at
   almost no extra price so both sin_ps and cos_ps make use of
   sincos_ps..
  */
static inline void sincos_ps(v4sf x, v4sf *ysin, v4sf *ycos)
{  // any x
    v4sf xmm1, xmm2, xmm3, y;

    v4su emm2;

    v4su sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f32(x, vdupq_n_f32(0));
    x = vabsq_f32(x);

    /* scale by 4/Pi */
    y = vmulq_f32(x, vdupq_n_f32(c_cephes_FOPI));

    /* store the integer part of y in mm0 */
    emm2 = vcvtq_u32_f32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);

    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
    v4su poly_mask = vtstq_u32(emm2, vdupq_n_u32(2));

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = vmulq_n_f32(y, c_minus_cephes_DP1);
    xmm2 = vmulq_n_f32(y, c_minus_cephes_DP2);
    xmm3 = vmulq_n_f32(y, c_minus_cephes_DP3);
    x = vaddq_f32(x, xmm1);
    x = vaddq_f32(x, xmm2);
    x = vaddq_f32(x, xmm3);

    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, vdupq_n_u32(4)));
    sign_mask_cos = vtstq_u32(vsubq_u32(emm2, vdupq_n_u32(2)), vdupq_n_u32(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    v4sf z = vmulq_f32(x, x);
    v4sf y1, y2;

    y1 = vmulq_n_f32(z, c_coscof_p0);
    y2 = vmulq_n_f32(z, c_sincof_p0);
    y1 = vaddq_f32(y1, vdupq_n_f32(c_coscof_p1));
    y2 = vaddq_f32(y2, vdupq_n_f32(c_sincof_p1));
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vaddq_f32(y1, vdupq_n_f32(c_coscof_p2));
    y2 = vaddq_f32(y2, vdupq_n_f32(c_sincof_p2));
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, x);
    y1 = vsubq_f32(y1, vmulq_f32(z, vdupq_n_f32(0.5f)));
    y2 = vaddq_f32(y2, x);
    y1 = vaddq_f32(y1, vdupq_n_f32(1));

    /* select the correct result from the two polynoms */
    v4sf ys = vbslq_f32(poly_mask, y1, y2);
    v4sf yc = vbslq_f32(poly_mask, y2, y1);
    *ysin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ycos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}

#else /* FMA */

// FMA version
static inline v4sf log_ps(v4sf x)
{
    v4sf one = vdupq_n_f32(1);

    x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
    v4su invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

    v4si ux = vreinterpretq_s32_f32(x);

    v4si emm0 = vshrq_n_s32(ux, 23);

    /* keep only the fractional part */
    ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ux);

    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    v4sf e = vcvtq_f32_s32(emm0);

    e = vaddq_f32(e, one);

    /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
    v4su mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
    v4sf tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);

    v4sf z = vmulq_f32(x, x);


    v4sf y;
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p1), vdupq_n_f32(c_cephes_log_p0), x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p2), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p3), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p4), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p5), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p6), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p7), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_log_p8), y, x);
    y = vmulq_f32(y, x);
    y = vmulq_f32(y, z);
    y = vfmaq_f32(y, e, vdupq_n_f32(c_cephes_log_q1));
    y = vfmaq_f32(y, z, vdupq_n_f32(-0.5f));


    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
    x = vfmaq_f32(x, e, vdupq_n_f32(c_cephes_log_q2));
    x = vaddq_f32(x, y);

    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask));  // negative arg will be NAN
    return x;
}

// FMA version
static inline v4sf exp_ps(v4sf x)
{
    v4sf tmp, fx;

    v4sf one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    v4su mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    // tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    // x = vsubq_f32(x, tmp);
    x = vfmaq_f32(x, fx, vdupq_n_f32(-c_cephes_exp_C1));
    // v4sf z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    // x = vsubq_f32(x, z);
    x = vfmaq_f32(x, fx, vdupq_n_f32(-c_cephes_exp_C2));

    static const float cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    v4sf y = vld1q_dup_f32(cephes_exp_p + 0);
    v4sf c1 = vld1q_dup_f32(cephes_exp_p + 1);
    v4sf c2 = vld1q_dup_f32(cephes_exp_p + 2);
    v4sf c3 = vld1q_dup_f32(cephes_exp_p + 3);
    v4sf c4 = vld1q_dup_f32(cephes_exp_p + 4);
    v4sf c5 = vld1q_dup_f32(cephes_exp_p + 5);

    y = vfmaq_f32(c1, y, x);
    y = vfmaq_f32(c2, y, x);
    y = vfmaq_f32(c3, y, x);
    y = vfmaq_f32(c4, y, x);
    y = vfmaq_f32(c5, y, x);

    v4sf z = vmulq_f32(x, x);
    y = vfmaq_f32(x, y, z);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    v4sf pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

// FMA version
static inline void sincos_ps(v4sf x, v4sf *ysin, v4sf *ycos)
{  // any x
    v4sf y;

    v4su emm2;

    v4su sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f32(x, vdupq_n_f32(0));
    x = vabsq_f32(x);

    /* scale by 4/Pi */
    y = vmulq_f32(x, vdupq_n_f32(c_cephes_FOPI));

    /* store the integer part of y in mm0 */
    emm2 = vcvtq_u32_f32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);

    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
    v4su poly_mask = vtstq_u32(emm2, vdupq_n_u32(2));

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */

    x = vfmaq_n_f32(x, y, c_minus_cephes_DP1);
    x = vfmaq_n_f32(x, y, c_minus_cephes_DP2);
    x = vfmaq_n_f32(x, y, c_minus_cephes_DP3);

    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, vdupq_n_u32(4)));
    sign_mask_cos = vtstq_u32(vsubq_u32(emm2, vdupq_n_u32(2)), vdupq_n_u32(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    v4sf z = vmulq_f32(x, x);
    v4sf y1, y2;

    y1 = vfmaq_n_f32(vdupq_n_f32(c_coscof_p1), z, c_coscof_p0);
    y2 = vfmaq_n_f32(vdupq_n_f32(c_sincof_p1), z, c_sincof_p0);
    y1 = vfmaq_f32(vdupq_n_f32(c_coscof_p2), y1, z);
    y2 = vfmaq_f32(vdupq_n_f32(c_sincof_p2), y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vfmaq_f32(x, y2, x);
    y1 = vfmaq_n_f32(y1, z, -0.5f);
    y1 = vaddq_f32(y1, vdupq_n_f32(1));

    /* select the correct result from the two polynoms */
    v4sf ys = vbslq_f32(poly_mask, y1, y2);
    v4sf yc = vbslq_f32(poly_mask, y2, y1);
    *ysin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ycos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}

#endif /* FMA */


static inline v4sf sin_ps(v4sf x)
{
    v4sf ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ysin;
}

static inline v4sf cos_ps(v4sf x)
{
    v4sf ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ycos;
}


static inline v4sf pow_ps(v4sf y, v4sf x)
{
    v4sf logvec = log_ps(y);
    v4sf expvec = vmulq_f32(logvec, x);
    v4sf ret = exp_ps(expvec);
    return ret;
}

static inline v4sf sqrt_ps(v4sf val)
{
#if defined(__aarch64__)
    return vsqrtq_f32(val);
#else
    v4sf est = vrsqrteq_f32(val);
    // Perform 4 iterations
    v4sf vec = vmulq_f32(est, est);
    vec = vrsqrtsq_f32(val, vec);
    est = vmulq_f32(vec, est);
    vec = vmulq_f32(est, est);
    vec = vrsqrtsq_f32(val, vec);
    est = vmulq_f32(vec, est);
    vec = vmulq_f32(est, est);
    vec = vrsqrtsq_f32(val, vec);
    est = vmulq_f32(vec, est);
    vec = vmulq_f32(est, est);
    vec = vrsqrtsq_f32(val, vec);
    est = vmulq_f32(vec, est);
    // Multiply by val
    est = vmulq_f32(est, val);
    return est;
#endif
}

#pragma GCC diagnostic pop

#endif  // INC_SIMD_NEON_MATHFUN_H_
