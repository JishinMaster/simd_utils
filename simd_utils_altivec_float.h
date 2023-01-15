/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <altivec.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef __MACH__
#include "fpu_control.h"
#endif

// In Altivec there is only mad and not mul
static inline v4sf vec_mul(v4sf a, v4sf b)
{
    return vec_madd(a, b, *(v4sf *) _ps_0);
}

// Useful link : http://mirror.informatimago.com/next/developer.apple.com/hardware/ve/algorithms.html

// In Altivec there is no div, hence a/b = a*(1/b)
static inline v4sf vec_div(v4sf a, v4sf b)
{
    return vec_mul(a, vec_re(b));
}

// In Altivec there is no nand
static inline v4sf vec_nand(v4sf a, v4sf b)
{
    return vec_andc(b, a);
}

static inline v4si vec_nandi(v4si a, v4si b)
{
    return vec_andc(b, a);
}


#if 0  // vec_div does not exist on older GCC?
static inline v4sf vec_div_less_precise(v4sf a, v4sf b)
{
    // Get the reciprocal estimate
    v4sf estimate = vec_re(b);
    // One round of Newton-Raphson refinement
    v4sf re = vec_madd(vec_nmsub(estimate, b, *(v4sf *) _ps_1), estimate, estimate);
    return vec_mul(a, re);
}

// http://preserve.mactech.com/articles/mactech/Vol.15/15.07/AltiVecRevealed/index.html
// precise to full IEEE 24 bits
static inline vector float vec_div_precise(vector float A, vector float B)
{
    vector float y0;
    vector float y1;
    vector float y2;
    vector float Q;
    vector float R;

    y0 = vec_re(B);  // approximate 1/B

    // y1 = y0*(-(y0*B - 1.0))+y0  i.e. y0+y0*(1.0 - y0*B)
    y1 = vec_madd(y0, vec_nmsub(y0, B, *(v4sf *) _ps_1), y0);

    // REPEAT the Newton-Raphson to get the required 24 bits
    y2 = vec_madd(y1, vec_nmsub(y1, B, *(v4sf *) _ps_1), y1);

    // y2 = y1*(-(y1*B - 1.0))+y1  i.e. y1+y1*(1.0 - y1*B)
    // y2 is now the correctly rounded reciprocal, and the manual considers this
    // OK for use in computing the remainder: Q = A*y2, R = A - B*Q

    Q = vec_mul(A, y2);
    R = vec_nmsub(B, Q, A);  // -(B*Q-A) == (A-B*Q)

    // final rouding adjustment
    return (vec_madd(R, y2, Q));
}
#else

static inline vector float vec_div_precise(vector float A, vector float B)
{
    return vec_div(A, B);
}
#endif

#if 1  // vec_sqrt does not exist on older GCC?
// In Altivec there is no sqrt, hence sqrt(a)= a*rsqrt(a)
static inline v4sf vec_sqrt(v4sf a)
{
#if 1  // Add a quantum so that sqrt(0) = 0 and not NaN
    const v4sf quantum = {1.180E-38, 1.180E-38, 1.180E-38, 1.180E-38};
    a = vec_add(a, quantum);
#endif
    return vec_mul(a, vec_rsqrte(a));
}

static inline v4sf vec_sqrt_precise(v4sf a)
{
#if 1  // Add a quantum so that sqrt(0) = 0 and not NaN
    const v4sf quantum = {1.180E-38, 1.180E-38, 1.180E-38, 1.180E-38};
    a = vec_add(a, quantum);
#endif
    // Get the square root reciprocal estimate
    v4sf estimate = vec_rsqrte(a);

    // One round of Newton-Raphson refinement
    v4sf estimateSquared = vec_mul(estimate, estimate);
    v4sf halfEstimate = vec_mul(estimate, *(v4sf *) _ps_0p5);
    v4sf re = vec_madd(vec_nmsub(a, estimateSquared, *(v4sf *) _ps_1), halfEstimate, estimate);

    return vec_mul(a, re);
}

#else
static inline v4sf vec_sqrt(v4sf a)
{
    return __builtin_vec_sqrt(a);
}

static inline v4sf vec_sqrt_precise(v4sf a)
{
    return __builtin_vec_sqrt(a);
}
#endif


static inline void set128f(float *dst, float value, int len)
{
    v4sf tmp = vec_splats(value);

    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (isAligned((uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            vec_st(tmp, 0, dst + i);
        }
    } else {
        int unaligned_float = (uintptr_t) (dst) % ALTIVEC_LEN_FLOAT;  // could this happen though?
        if (unaligned_float == 0) {                                   // dst is not aligned on 16bytes boundary but is at least aligned on float
            int unaligned_elts = ((uintptr_t) (dst) % ALTIVEC_LEN_BYTES) / sizeof(float);
            for (int i = 0; i < unaligned_elts; i++) {
                dst[i] = value;
            }
            for (int i = unaligned_elts + 1; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
                vec_st(tmp, 0, dst + i);
            }
        } else {  // do not use SIMD in this case, skip to scalar part
            stop_len = 0;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value;
    }
}

static inline void zero128f(float *dst, int len)
{
    set128f(dst, 0.0f, len);
}

static inline void copy128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            vec_st(src_tmp, 0, dst + i);
            vec_st(src_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            if (unalign_dst) {
                vec_stu(*(v16u8 *) &src_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &src_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(src_tmp, 0, dst + i);
                vec_st(src_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_add(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src1) {
                a = (v4sf) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4sf) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4sf c = vec_add(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

static inline void addc128f(float *src, float value, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    v4sf b = vec_splats(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_add(a, b), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }

            v4sf c = vec_add(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] + value;
    }
}

static inline void sub128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_sub(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src1) {
                a = (v4sf) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4sf) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4sf c = vec_sub(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] - src2[i];
    }
}

static inline void subc128f(float *src, float value, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    v4sf b = vec_splats(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_sub(a, b), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }

            v4sf c = vec_sub(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] - value;
    }
}

static inline void subcrev128f(float *src, float value, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    v4sf b = vec_splats(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_sub(b, a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }

            v4sf c = vec_sub(b, a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = value - src[i];
    }
}

static inline void mul128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_mul(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src1) {
                a = (v4sf) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4sf) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4sf c = vec_mul(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

static inline void mulc128f(float *src, float value, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    v4sf b = vec_splats(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_mul(a, b), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }

            v4sf c = vec_mul(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] * value;
    }
}

static inline void div128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_div_precise(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src1) {
                a = (v4sf) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4sf) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4sf c = vec_div_precise(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] / src2[i];
    }
}


static inline void fabs128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_and(*(v4sf *) _ps_pos_sign_mask, src_tmp);
            v4sf dst_tmp2 = vec_and(*(v4sf *) _ps_pos_sign_mask, src_tmp2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_and(*(v4sf *) _ps_pos_sign_mask, src_tmp);
            v4sf dst_tmp2 = vec_and(*(v4sf *) _ps_pos_sign_mask, src_tmp2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void minevery128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_min(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src1) {
                a = (v4sf) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4sf) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4sf c = vec_min(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] < src2[i] ? src1[i] : src2[i];
    }
}

static inline void maxevery128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_max(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src1) {
                a = (v4sf) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4sf) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4sf c = vec_max(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] > src2[i] ? src1[i] : src2[i];
    }
}

// converts 32bits complex float to two arrays real and im
static inline void cplxtoreal128f(complex32_t *src, float *dstRe, float *dstIm, int len)
{
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN cplxtoreal128f" ::
                       : "memory");
#endif
    int stop_len = 2 * len / (4 * ALTIVEC_LEN_FLOAT);
    stop_len *= 4 * ALTIVEC_LEN_FLOAT;
    int j = 0;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dstRe), (uintptr_t) (dstIm), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 vec1 = vec_ld2((float *) (src) + i);
            v4sfx2 vec2 = vec_ld2((float *) (src) + i + 2 * ALTIVEC_LEN_FLOAT);
            vec_st(vec1.val[0], 0, dstRe + j);
            vec_st(vec1.val[1], 0, dstIm + j);
            vec_st(vec2.val[0], 0, dstRe + j + ALTIVEC_LEN_FLOAT);
            vec_st(vec2.val[1], 0, dstIm + j + ALTIVEC_LEN_FLOAT);
            j += 2 * ALTIVEC_LEN_FLOAT;
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dstRe = (uintptr_t) (dstRe) % ALTIVEC_LEN_BYTES;
        int unalign_dstIm = (uintptr_t) (dstIm) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 vec1, vec2;

            if (unalign_src) {
                vec1 = vec_ld2u((float *) (src) + i);
                vec2 = vec_ld2u((float *) (src) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                vec1 = vec_ld2((float *) (src) + i);
                vec2 = vec_ld2((float *) (src) + i + 2 * ALTIVEC_LEN_FLOAT);
            }

            if (unalign_dstRe) {
                vec_stu(*(v16u8 *) &vec1.val[0], (unsigned char *) (dstRe + j));
                vec_stu(*(v16u8 *) &vec2.val[0], (unsigned char *) (dstRe + j + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(vec1.val[0], 0, dstRe + j);
                vec_st(vec2.val[0], 0, dstRe + j + ALTIVEC_LEN_FLOAT);
            }

            if (unalign_dstIm) {
                vec_stu(*(v16u8 *) &vec1.val[1], (unsigned char *) (dstIm + j));
                vec_stu(*(v16u8 *) &vec2.val[1], (unsigned char *) (dstIm + j + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(vec1.val[1], 0, dstIm + j);
                vec_st(vec2.val[1], 0, dstIm + j + ALTIVEC_LEN_FLOAT);
            }
            j += 2 * ALTIVEC_LEN_FLOAT;
        }
    }

    for (int i = j; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END cplxtoreal128f" ::
                       : "memory");
#endif
}

static inline void realtocplx128f(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN realtocplx128f" ::
                       : "memory");
#endif
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);
    int j = 0;

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 cplx1, cplx2;
            cplx1.val[0] = vec_ld(0, (float const *) (srcRe) + i);
            cplx1.val[1] = vec_ld(0, (float const *) (srcIm) + i);
            cplx2.val[0] = vec_ld(0, (float const *) (srcRe) + i + ALTIVEC_LEN_FLOAT);
            cplx2.val[1] = vec_ld(0, (float const *) (srcIm) + i + ALTIVEC_LEN_FLOAT);

            vec_st2(cplx1, (float *) (dst) + j);
            vec_st2(cplx2, (float *) (dst) + j + 2 * ALTIVEC_LEN_FLOAT);
            j += 4 * ALTIVEC_LEN_FLOAT;
        }
    } else {
        int unalign_srcRe = (uintptr_t) (srcRe) % ALTIVEC_LEN_BYTES;
        int unalign_srcIm = (uintptr_t) (srcIm) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 cplx1, cplx2;

            if (unalign_srcRe) {
                cplx1.val[0] = (v4sf) vec_ldu((unsigned char *) ((float const *) (srcRe) + i));
                cplx2.val[0] = (v4sf) vec_ldu((unsigned char *) ((float const *) (srcRe) + i + ALTIVEC_LEN_FLOAT));
            } else {
                cplx1.val[0] = vec_ld(0, (float const *) (srcRe) + i);
                cplx2.val[0] = vec_ld(0, (float const *) (srcRe) + i + ALTIVEC_LEN_FLOAT);
            }

            if (unalign_srcIm) {
                cplx1.val[1] = (v4sf) vec_ldu((unsigned char *) ((float const *) (srcIm) + i));
                cplx2.val[1] = (v4sf) vec_ldu((unsigned char *) ((float const *) (srcIm) + i + ALTIVEC_LEN_FLOAT));
            } else {
                cplx1.val[1] = vec_ld(0, (float const *) (srcIm) + i);
                cplx2.val[1] = vec_ld(0, (float const *) (srcIm) + i + ALTIVEC_LEN_FLOAT);
            }

            if (unalign_dst) {
                vec_st2u(cplx1, (float *) (dst) + j);
                vec_st2u(cplx2, (float *) (dst) + j + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                vec_st2(cplx1, (float *) (dst) + j);
                vec_st2(cplx2, (float *) (dst) + j + 2 * ALTIVEC_LEN_FLOAT);
            }
            j += 4 * ALTIVEC_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END realtocplx128f" ::
                       : "memory");
#endif
}

static inline v4sf exp_ps_alternate(v4sf x)
{
    v4sf z_tmp, z, fx;
    v4si n;
    v4bi xsupmaxlogf, xinfminglogf;

    xsupmaxlogf = vec_cmpgt(x, *(v4sf *) _ps_MAXLOGF);
    xinfminglogf = vec_cmplt(x, *(v4sf *) _ps_MINLOGF);

    /* Express e**x = e**g 2**n
     *   = e**g e**( n loge(2) )
     *   = e**( g + n loge(2) )
     */
    fx = vec_madd(*(v4sf *) _ps_cephes_LOG2EF, x, *(v4sf *) _ps_0p5);
    z = vec_floor(fx);  // round to floor

    x = vec_madd(z, *(v4sf *) _ps_cephes_exp_minC1, x);
    x = vec_madd(z, *(v4sf *) _ps_cephes_exp_minC2, x);

    n = vec_cts(z, 0);
    n = vec_add(n, *(v4si *) _pi32_0x7f);
    v4ui shift_bits = vec_splats((unsigned int) 23);
    n = vec_sl(n, shift_bits);

    z = vec_mul(x, x);

    z_tmp = vec_madd(*(v4sf *) _ps_cephes_exp_p0, x, *(v4sf *) _ps_cephes_exp_p1);
    z_tmp = vec_madd(z_tmp, x, *(v4sf *) _ps_cephes_exp_p2);
    z_tmp = vec_madd(z_tmp, x, *(v4sf *) _ps_cephes_exp_p3);
    z_tmp = vec_madd(z_tmp, x, *(v4sf *) _ps_cephes_exp_p4);
    z_tmp = vec_madd(z_tmp, x, *(v4sf *) _ps_cephes_exp_p5);
    z_tmp = vec_madd(z_tmp, z, x);

    /* build 2^n */
    z_tmp = vec_madd(z_tmp, *(v4sf *) &n, *(v4sf *) &n);

    z = vec_sel(z_tmp, *(v4sf *) _ps_MAXNUMF, xsupmaxlogf);
    z = vec_sel(z, *(v4sf *) _ps_0, xinfminglogf);
    return z;
}

static inline void exp_128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(exp_ps_alternate(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = exp_ps_alternate(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

static inline void log2_128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_loge(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = vec_loge(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

static inline v4sf log2_ps(v4sf x)
{
    v4si emm0;
    v4sf one = *(v4sf *) _ps_1;
    v4sf invalid_mask = (v4sf) vec_cmple(x, *(v4sf *) _ps_0);
    x = vec_max(x, *(v4sf *) _ps_min_norm_pos); /* cut off denormalized stuff */
    v4ui shift_bits = vec_splats((unsigned int) 23);
    emm0 = vec_sr(*(v4si *) &x, shift_bits);

    /* keep only the fractional part */
    x = vec_and(x, *(v4sf *) _ps_inv_mant_mask);
    x = vec_or(x, *(v4sf *) _ps_0p5);
    emm0 = vec_sub(emm0, *(v4si *) _pi32_0x7f);
    v4sf e = vec_ctf(emm0, 0);
    e = vec_add(e, one);

    v4sf mask = (v4sf) vec_cmplt(x, *(v4sf *) _ps_cephes_SQRTHF);
    v4sf tmp = vec_and(x, mask);
    x = vec_sub(x, one);
    e = vec_sub(e, vec_and(one, mask));
    x = vec_add(x, tmp);

    v4sf z = vec_mul(x, x);
    v4sf y = vec_madd(*(v4sf *) _ps_cephes_log_p0, x, *(v4sf *) _ps_cephes_log_p1);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p2);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p3);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p4);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p5);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p6);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p7);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p8);
    y = vec_mul(y, x);
    y = vec_mul(y, z);

    y = vec_nmsub(z, *(v4sf *) _ps_0p5, y);

    tmp = vec_add(y, x);
    z = vec_mul(y, *(v4sf *) _ps_cephes_LOG2EA);
    z = vec_madd(x, *(v4sf *) _ps_cephes_LOG2EA, z);
    z = vec_add(z, tmp);
    x = vec_add(z, e);
    x = vec_or(x, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline void log2_128f_precise(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(log2_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = log2_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log2f(src[i]);
    }
}

// less precise
static inline void ln_128f_less_precise(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    const v4sf ln2_vec = {LN2, LN2, LN2, LN2};

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_mul(vec_loge(a), ln2_vec), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = vec_mul(vec_loge(a), ln2_vec);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline void log10_128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    const v4sf ln2_ln10_vec = {LN2_DIV_LN10, LN2_DIV_LN10, LN2_DIV_LN10, LN2_DIV_LN10};

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_mul(vec_loge(a), ln2_ln10_vec), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = vec_mul(vec_loge(a), ln2_ln10_vec);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}


static inline v4sf log10_ps(v4sf x)
{
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-BEGIN log10_ps" ::
                       : "memory");
#endif

    v4si emm0;
    v4sf one = *(v4sf *) _ps_1;
    v4sf invalid_mask = (v4sf) vec_cmple(x, *(v4sf *) _ps_0);
    x = vec_max(x, *(v4sf *) _ps_min_norm_pos); /* cut off denormalized stuff */
    v4ui shift_bits = vec_splats((unsigned int) 23);
    emm0 = vec_sr(*(v4si *) &x, shift_bits);

    /* keep only the fractional part */
    x = vec_and(x, *(v4sf *) _ps_inv_mant_mask);
    x = vec_or(x, *(v4sf *) _ps_0p5);
    emm0 = vec_sub(emm0, *(v4si *) _pi32_0x7f);
    v4sf e = vec_ctf(emm0, 0);
    e = vec_add(e, one);

    v4sf mask = (v4sf) vec_cmplt(x, *(v4sf *) _ps_cephes_SQRTHF);
    v4sf tmp = vec_and(x, mask);
    x = vec_sub(x, one);
    e = vec_sub(e, vec_and(one, mask));
    x = vec_add(x, tmp);

    v4sf z = vec_mul(x, x);
    v4sf y = vec_madd(*(v4sf *) _ps_cephes_log_p0, x, *(v4sf *) _ps_cephes_log_p1);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p2);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p3);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p4);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p5);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p6);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p7);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p8);
    y = vec_mul(y, x);
    y = vec_mul(y, z);

    y = vec_nmsub(z, *(v4sf *) _ps_0p5, y);

    // Could it be improved with more parallelism or would it worsen precision?
    tmp = vec_add(x, y);
    z = vec_mul(tmp, *(v4sf *) _ps_cephes_L10EB);
    z = vec_madd(y, *(v4sf *) _ps_cephes_L10EA, z);
    z = vec_madd(x, *(v4sf *) _ps_cephes_L10EA, z);
    z = vec_madd(e, *(v4sf *) _ps_cephes_L102B, z);
    x = vec_madd(e, *(v4sf *) _ps_cephes_L102A, z);

    x = vec_or(x, invalid_mask);  // negative arg will be NAN
#ifdef LLVMMCA
    __asm volatile("# LLVM-MCA-END log10_ps" ::
                       : "memory");
#endif
    return x;
}

static inline void log10_128f_precise(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(log10_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = log10_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline v4sf log_ps(v4sf x)
{
    v4si emm0;
    v4sf one = *(v4sf *) _ps_1;

    v4bi invalid_mask = vec_cmple(x, *(v4sf *) _ps_0);

    x = vec_max(x, *(v4sf *) _ps_min_norm_pos); /* cut off denormalized stuff */

    v4ui shift_bits = vec_splats((unsigned int) 23);
    emm0 = vec_sr(*(v4si *) &x, shift_bits);

    /* keep only the fractional part */
    x = vec_and(x, *(v4sf *) _ps_inv_mant_mask);
    x = vec_or(x, *(v4sf *) _ps_0p5);

    emm0 = vec_sub(emm0, *(v4si *) _pi32_0x7f);
    v4sf e = vec_ctf(emm0, 0);

    e = vec_add(e, one);

    v4bi mask = vec_cmplt(x, *(v4sf *) _ps_cephes_SQRTHF);
    v4sf tmp = vec_and(x, *(v4sf *) &mask);
    x = vec_sub(x, one);
    e = vec_sub(e, vec_and(one, mask));
    x = vec_add(x, tmp);

    v4sf z = vec_mul(x, x);

    v4sf y = vec_madd(*(v4sf *) _ps_cephes_log_p0, x, *(v4sf *) _ps_cephes_log_p1);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p2);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p3);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p4);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p5);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p6);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p7);
    y = vec_madd(y, x, *(v4sf *) _ps_cephes_log_p8);
    y = vec_mul(y, x);

    y = vec_mul(y, z);

    y = vec_madd(e, *(v4sf *) _ps_cephes_log_q1, y);
    y = vec_nmsub(z, *(v4sf *) _ps_0p5, y);

    tmp = vec_madd(e, *(v4sf *) _ps_cephes_log_q2, y);
    x = vec_add(x, tmp);
    x = vec_or(x, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline void ln_128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(log_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = log_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline void magnitude128f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf re_tmp = vec_ld(0, srcRe + i);
            v4sf re2 = vec_mul(re_tmp, re_tmp);
            v4sf im_tmp = vec_ld(0, srcIm + i);
            v4sf im2 = vec_mul(im_tmp, im_tmp);
            vec_st(vec_sqrt(vec_add(re2, im2)), 0, dst + i);
        }
    } else {
        int unalign_srcRe = (uintptr_t) (srcRe) % ALTIVEC_LEN_BYTES;
        int unalign_srcIm = (uintptr_t) (srcRe) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf re_tmp, re2, im_tmp, im2, res;

            if (unalign_srcRe) {
                re_tmp = (v4sf) vec_ldu((unsigned char *) (srcRe + i));
            } else {
                re_tmp = vec_ld(0, srcRe + i);
            }

            if (unalign_srcIm) {
                im_tmp = (v4sf) vec_ldu((unsigned char *) (srcIm + i));
            } else {
                im_tmp = vec_ld(0, srcIm + i);
            }

            re2 = vec_mul(re_tmp, re_tmp);
            im_tmp = vec_ld(0, srcIm + i);
            im2 = vec_mul(im_tmp, im_tmp);
            res = vec_sqrt(vec_add(re2, im2));

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &res, (unsigned char *) (dst + i));
            } else {
                vec_st(res, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]));
    }
}

static inline void powerspect128f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (srcRe), (uintptr_t) (srcIm), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf re_tmp = vec_ld(0, srcRe + i);
            v4sf re2 = vec_mul(re_tmp, re_tmp);
            v4sf im_tmp = vec_ld(0, srcIm + i);
            v4sf im2 = vec_mul(im_tmp, im_tmp);
            vec_st(vec_add(re2, im2), 0, dst + i);
        }
    } else {
        int unalign_srcRe = (uintptr_t) (srcRe) % ALTIVEC_LEN_BYTES;
        int unalign_srcIm = (uintptr_t) (srcRe) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf re_tmp, re2, im_tmp, im2, res;

            if (unalign_srcRe) {
                re_tmp = (v4sf) vec_ldu((unsigned char *) (srcRe + i));
            } else {
                re_tmp = vec_ld(0, srcRe + i);
            }

            if (unalign_srcIm) {
                im_tmp = (v4sf) vec_ldu((unsigned char *) (srcIm + i));
            } else {
                im_tmp = vec_ld(0, srcIm + i);
            }

            re2 = vec_mul(re_tmp, re_tmp);
            im_tmp = vec_ld(0, srcIm + i);
            im2 = vec_mul(im_tmp, im_tmp);
            res = vec_add(re2, im2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &res, (unsigned char *) (dst + i));
            } else {
                vec_st(res, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = srcRe[i] * srcRe[i] + (srcIm[i] * srcIm[i]);
    }
}

// vec_nmsub(a,b,c) => -(a*b -c)
// (ac -bd) + i(ad + bc)
static inline void cplxvecmul128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (4 * ALTIVEC_LEN_FLOAT);
    stop_len *= 4 * ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 src1_split = vec_ld2((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = vec_ld2((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = vec_ld2((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sfx2 src2_split2 = vec_ld2((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sf ac = vec_mul(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = vec_mul(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = vec_mul(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = vec_mul(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = vec_nmsub(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = vec_madd(src1_split.val[1], src2_split.val[0], ad);
            dst_split2.val[0] = vec_nmsub(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = vec_madd(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_st2(dst_split, (float *) (dst) + i);
            vec_st2(dst_split2, (float *) (dst) + i + 2 * ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < 2 * stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 src1_split, src1_split2, src2_split, src2_split2;

            if (unalign_src1) {
                src1_split = vec_ld2u((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
                src1_split2 = vec_ld2u((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                src1_split = vec_ld2((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
                src1_split2 = vec_ld2((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            }

            if (unalign_src2) {
                src2_split = vec_ld2u((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
                src2_split2 = vec_ld2u((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                src2_split = vec_ld2((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
                src2_split2 = vec_ld2((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            }

            v4sf ac = vec_mul(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = vec_mul(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = vec_mul(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = vec_mul(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = vec_nmsub(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = vec_madd(src1_split.val[1], src2_split.val[0], ad);
            dst_split2.val[0] = vec_nmsub(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = vec_madd(src1_split2.val[1], src2_split2.val[0], ad2);

            if (unalign_dst) {
                vec_st2u(dst_split, (float *) (dst) + i);
                vec_st2u(dst_split2, (float *) (dst) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                vec_st2(dst_split, (float *) (dst) + i);
                vec_st2(dst_split2, (float *) (dst) + i + 2 * ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxvecmul128f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (ALTIVEC_LEN_FLOAT);
    stop_len = stop_len * ALTIVEC_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), ALTIVEC_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), ALTIVEC_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), ALTIVEC_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src1Re_tmp = vec_ld(0, (float *) (src1Re) + i);
            v4sf src1Im_tmp = vec_ld(0, (float *) (src1Im) + i);
            v4sf src2Re_tmp = vec_ld(0, (float *) (src2Re) + i);
            v4sf src2Im_tmp = vec_ld(0, (float *) (src2Im) + i);
            v4sf ac = vec_mul(src1Re_tmp, src2Re_tmp);
            v4sf bc = vec_mul(src1Im_tmp, src2Re_tmp);
            v4sf tmp = vec_nmsub(src1Im_tmp, src2Im_tmp, ac);
            v4sf tmp2 = vec_madd(src1Re_tmp, src2Im_tmp, bc);
            vec_st(tmp, 0, dstRe + i);   // ac - bd
            vec_st(tmp2, 0, dstIm + i);  // ad + bc
        }
    } else {
        int unalign_src1Re = (uintptr_t) (src1Re) % ALTIVEC_LEN_BYTES;
        int unalign_src1Im = (uintptr_t) (src1Im) % ALTIVEC_LEN_BYTES;
        int unalign_src2Re = (uintptr_t) (src2Re) % ALTIVEC_LEN_BYTES;
        int unalign_src2Im = (uintptr_t) (src2Im) % ALTIVEC_LEN_BYTES;
        int unalign_dstRe = (uintptr_t) (dstRe) % ALTIVEC_LEN_BYTES;
        int unalign_dstIm = (uintptr_t) (dstIm) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src1Re_tmp, src1Im_tmp, src2Re_tmp, src2Im_tmp;

            if (unalign_src1Re) {
                src1Re_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src1Re) + i));
            } else {
                src1Re_tmp = vec_ld(0, (float *) (src1Re) + i);
            }

            if (unalign_src2Re) {
                src2Re_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src2Re) + i));
            } else {
                src2Re_tmp = vec_ld(0, (float *) (src2Re) + i);
            }

            if (unalign_src1Im) {
                src1Im_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src1Im) + i));
            } else {
                src1Im_tmp = vec_ld(0, (float *) (src1Im) + i);
            }

            if (unalign_src2Im) {
                src2Im_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src2Im) + i));
            } else {
                src2Im_tmp = vec_ld(0, (float *) (src2Im) + i);
            }

            v4sf ac = vec_mul(src1Re_tmp, src2Re_tmp);
            v4sf bc = vec_mul(src1Im_tmp, src2Re_tmp);
            v4sf tmp = vec_nmsub(src1Im_tmp, src2Im_tmp, ac);
            v4sf tmp2 = vec_madd(src1Re_tmp, src2Im_tmp, bc);

            if (unalign_dstRe) {
                vec_stu(*(v16u8 *) &tmp, (unsigned char *) (dstRe + i));
            } else {
                vec_st(tmp, 0, dstRe + i);
            }

            if (unalign_dstIm) {
                vec_stu(*(v16u8 *) &tmp2, (unsigned char *) (dstIm + i));
            } else {
                vec_st(tmp2, 0, dstIm + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dstRe[i] = (src1Re[i] * src2Re[i]) - src1Im[i] * src2Im[i];
        dstIm[i] = src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i]);
    }
}

static inline void cplxvecdiv128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (4 * ALTIVEC_LEN_FLOAT);
    stop_len *= 4 * ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 src1_split = vec_ld2((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = vec_ld2((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = vec_ld2((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sfx2 src2_split2 = vec_ld2((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sf c2 = vec_mul(src2_split.val[0], src2_split.val[0]);
            v4sf c2d2 = vec_madd(src2_split.val[1], src2_split.val[1], c2);
            v4sf c2_ = vec_mul(src2_split2.val[0], src2_split2.val[0]);
            v4sf c2d2_ = vec_madd(src2_split2.val[1], src2_split2.val[1], c2_);
            v4sf ac = vec_mul(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf bc = vec_mul(src1_split.val[1], src2_split.val[0]);     // bc
            v4sf ac2 = vec_mul(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf bc2 = vec_mul(src1_split2.val[1], src2_split2.val[0]);  // bc
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = vec_madd(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = vec_nmsub(src1_split.val[0], src2_split.val[1], bc);
            dst_split2.val[0] = vec_madd(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = vec_nmsub(src1_split2.val[0], src2_split2.val[1], bc2);
            dst_split.val[0] = vec_div_precise(dst_split.val[0], c2d2);
            dst_split.val[1] = vec_div_precise(dst_split.val[1], c2d2);
            dst_split2.val[0] = vec_div_precise(dst_split2.val[0], c2d2_);
            dst_split2.val[1] = vec_div_precise(dst_split2.val[1], c2d2_);
            vec_st2(dst_split, (float *) (dst) + i);
            vec_st2(dst_split2, (float *) (dst) + i + 2 * ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < 2 * stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 src1_split, src1_split2, src2_split, src2_split2;

            if (unalign_src1) {
                src1_split = vec_ld2u((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
                src1_split2 = vec_ld2u((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                src1_split = vec_ld2((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
                src1_split2 = vec_ld2((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            }

            if (unalign_src2) {
                src2_split = vec_ld2u((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
                src2_split2 = vec_ld2u((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                src2_split = vec_ld2((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
                src2_split2 = vec_ld2((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            }

            v4sf c2 = vec_mul(src2_split.val[0], src2_split.val[0]);
            v4sf c2d2 = vec_madd(src2_split.val[1], src2_split.val[1], c2);
            v4sf c2_ = vec_mul(src2_split2.val[0], src2_split2.val[0]);
            v4sf c2d2_ = vec_madd(src2_split2.val[1], src2_split2.val[1], c2_);
            v4sf ac = vec_mul(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf bc = vec_mul(src1_split.val[1], src2_split.val[0]);     // bc
            v4sf ac2 = vec_mul(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf bc2 = vec_mul(src1_split2.val[1], src2_split2.val[0]);  // bc
            v4sfx2 dst_split;
            v4sfx2 dst_split2;
            dst_split.val[0] = vec_madd(src1_split.val[1], src2_split.val[1], ac);
            dst_split.val[1] = vec_nmsub(src1_split.val[0], src2_split.val[1], bc);
            dst_split2.val[0] = vec_madd(src1_split2.val[1], src2_split2.val[1], ac2);
            dst_split2.val[1] = vec_nmsub(src1_split2.val[0], src2_split2.val[1], bc2);
            dst_split.val[0] = vec_div_precise(dst_split.val[0], c2d2);
            dst_split.val[1] = vec_div_precise(dst_split.val[1], c2d2);
            dst_split2.val[0] = vec_div_precise(dst_split2.val[0], c2d2_);
            dst_split2.val[1] = vec_div_precise(dst_split2.val[1], c2d2_);

            if (unalign_dst) {
                vec_st2u(dst_split, (float *) (dst) + i);
                vec_st2u(dst_split2, (float *) (dst) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                vec_st2(dst_split, (float *) (dst) + i);
                vec_st2(dst_split2, (float *) (dst) + i + 2 * ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2[i].re * src2[i].re + src2[i].im * src2[i].im;
        dst[i].re = ((src1[i].re * src2[i].re) + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-(src1[i].re * src2[i].im) + (src2[i].re * src1[i].im)) / c2d2;
    }
}

static inline void cplxvecdiv128f_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    int stop_len = len / (ALTIVEC_LEN_FLOAT);
    stop_len = stop_len * ALTIVEC_LEN_FLOAT;

    int i;
    if (areAligned2((uintptr_t) (src1Re), (uintptr_t) (src1Im), ALTIVEC_LEN_BYTES) &&
        areAligned2((uintptr_t) (src2Re), (uintptr_t) (src2Im), ALTIVEC_LEN_BYTES) &&
        areAligned2((uintptr_t) (dstRe), (uintptr_t) (dstIm), ALTIVEC_LEN_BYTES)) {
        for (i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src1Re_tmp = vec_ld(0, (float *) (src1Re) + i);
            v4sf src1Im_tmp = vec_ld(0, (float *) (src1Im) + i);
            v4sf src2Re_tmp = vec_ld(0, (float *) (src2Re) + i);
            v4sf src2Im_tmp = vec_ld(0, (float *) (src2Im) + i);
            v4sf c2 = vec_mul(src2Re_tmp, src2Re_tmp);
            v4sf c2d2 = vec_madd(src2Im_tmp, src2Im_tmp, c2);
            v4sf ac = vec_mul(src1Re_tmp, src2Re_tmp);
            v4sf bc = vec_mul(src1Im_tmp, src2Re_tmp);
            v4sf dstRe_tmp = vec_madd(src1Im_tmp, src2Im_tmp, ac);
            v4sf dstIm_tmp = vec_nmsub(src1Re_tmp, src2Im_tmp, bc);
            dstRe_tmp = vec_div_precise(dstRe_tmp, c2d2);
            dstIm_tmp = vec_div_precise(dstIm_tmp, c2d2);
            vec_st(dstRe_tmp, 0, dstRe + i);
            vec_st(dstIm_tmp, 0, dstIm + i);
        }
    } else {
        int unalign_src1Re = (uintptr_t) (src1Re) % ALTIVEC_LEN_BYTES;
        int unalign_src1Im = (uintptr_t) (src1Im) % ALTIVEC_LEN_BYTES;
        int unalign_src2Re = (uintptr_t) (src2Re) % ALTIVEC_LEN_BYTES;
        int unalign_src2Im = (uintptr_t) (src2Im) % ALTIVEC_LEN_BYTES;
        int unalign_dstRe = (uintptr_t) (dstRe) % ALTIVEC_LEN_BYTES;
        int unalign_dstIm = (uintptr_t) (dstIm) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src1Re_tmp, src1Im_tmp, src2Re_tmp, src2Im_tmp;

            if (unalign_src1Re) {
                src1Re_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src1Re) + i));
            } else {
                src1Re_tmp = vec_ld(0, (float *) (src1Re) + i);
            }

            if (unalign_src2Re) {
                src2Re_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src2Re) + i));
            } else {
                src2Re_tmp = vec_ld(0, (float *) (src2Re) + i);
            }

            if (unalign_src1Im) {
                src1Im_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src1Im) + i));
            } else {
                src1Im_tmp = vec_ld(0, (float *) (src1Im) + i);
            }

            if (unalign_src2Im) {
                src2Im_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src2Im) + i));
            } else {
                src2Im_tmp = vec_ld(0, (float *) (src2Im) + i);
            }

            v4sf c2 = vec_mul(src2Re_tmp, src2Re_tmp);
            v4sf c2d2 = vec_madd(src2Im_tmp, src2Im_tmp, c2);
            v4sf ac = vec_mul(src1Re_tmp, src2Re_tmp);
            v4sf bc = vec_mul(src1Im_tmp, src2Re_tmp);
            v4sf dstRe_tmp = vec_madd(src1Im_tmp, src2Im_tmp, ac);
            v4sf dstIm_tmp = vec_nmsub(src1Re_tmp, src2Im_tmp, bc);
            dstRe_tmp = vec_div_precise(dstRe_tmp, c2d2);
            dstIm_tmp = vec_div_precise(dstIm_tmp, c2d2);

            if (unalign_dstRe) {
                vec_stu(*(v16u8 *) &dstRe_tmp, (unsigned char *) (dstRe + i));
            } else {
                vec_st(dstRe_tmp, 0, dstRe + i);
            }

            if (unalign_dstIm) {
                vec_stu(*(v16u8 *) &dstIm_tmp, (unsigned char *) (dstIm + i));
            } else {
                vec_st(dstIm_tmp, 0, dstIm + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        float c2d2 = src2Re[i] * src2Re[i] + src2Im[i] * src2Im[i];
        dstRe[i] = (src1Re[i] * src2Re[i] + (src1Im[i] * src2Im[i])) / c2d2;
        dstIm[i] = (-src1Re[i] * src2Im[i] + (src2Re[i] * src1Im[i])) / c2d2;
    }
}

static inline void minmax128f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = (len - ALTIVEC_LEN_FLOAT) / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    float min_f[ALTIVEC_LEN_FLOAT] __attribute__((aligned(ALTIVEC_LEN_BYTES)));
    float max_f[ALTIVEC_LEN_FLOAT] __attribute__((aligned(ALTIVEC_LEN_BYTES)));
    v4sf max_v, min_v, max_v2, min_v2;
    v4sf src_tmp, src_tmp2;

    float min_tmp = src[0];
    float max_tmp = src[0];

    if (len >= ALTIVEC_LEN_FLOAT) {
        if (isAligned((uintptr_t) (src), ALTIVEC_LEN_BYTES)) {
            src_tmp = vec_ld(0, src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = ALTIVEC_LEN_FLOAT; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
                max_v = vec_max(max_v, src_tmp);
                min_v = vec_min(min_v, src_tmp);
                max_v2 = vec_max(max_v2, src_tmp2);
                min_v2 = vec_min(min_v2, src_tmp2);
            }
        } else {
            src_tmp = (v4sf) vec_ldu((unsigned char *) (src + 0));
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = ALTIVEC_LEN_FLOAT; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
                max_v = vec_max(max_v, src_tmp);
                min_v = vec_min(min_v, src_tmp);
                max_v2 = vec_max(max_v2, src_tmp2);
                min_v2 = vec_min(min_v2, src_tmp2);
            }
        }

        max_v = vec_max(max_v, max_v2);
        min_v = vec_min(min_v, min_v2);

        vec_st(max_v, 0, max_f);
        vec_st(min_v, 0, min_f);

        max_tmp = max_f[0];
        max_tmp = max_tmp > max_f[1] ? max_tmp : max_f[1];
        max_tmp = max_tmp > max_f[2] ? max_tmp : max_f[2];
        max_tmp = max_tmp > max_f[3] ? max_tmp : max_f[3];

        min_tmp = min_f[0];
        min_tmp = min_tmp < min_f[1] ? min_tmp : min_f[1];
        min_tmp = min_tmp < min_f[2] ? min_tmp : min_f[2];
        min_tmp = min_tmp < min_f[3] ? min_tmp : min_f[3];
    }

    for (int i = stop_len; i < len; i++) {
        max_tmp = max_tmp > src[i] ? max_tmp : src[i];
        min_tmp = min_tmp < src[i] ? min_tmp : src[i];
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

static inline void sum128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    __attribute__((aligned(ALTIVEC_LEN_BYTES))) float accumulate[ALTIVEC_LEN_FLOAT];
    float tmp_acc = 0.0f;

    v4sf vec_acc1 = vec_splats(0.0f);  // initialize the vector accumulator
    v4sf vec_acc2 = vec_splats(0.0f);  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf vec_tmp1 = vec_ld(0, src + i);
            vec_acc1 = vec_add(vec_acc1, vec_tmp1);
            v4sf vec_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            vec_acc2 = vec_add(vec_acc2, vec_tmp2);
        }
    } else {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf vec_tmp1, vec_tmp2;
            vec_tmp1 = (v4sf) vec_ldu((unsigned char *) (src + i));
            vec_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            vec_acc1 = vec_add(vec_acc1, vec_tmp1);
            vec_acc2 = vec_add(vec_acc2, vec_tmp2);
        }
    }
    vec_acc1 = vec_add(vec_acc1, vec_acc2);
    vec_st(vec_acc1, 0, accumulate);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

    *dst = tmp_acc;
}

static inline void mean128f(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sum128f(src, dst, len);
    *dst *= coeff;
}


static inline void threshold128_gt_f(float *src, float *dst, int len, float value)
{
    const v4sf tmp = {value, value, value, value};

    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_min(src_tmp, tmp);
            v4sf dst_tmp2 = vec_min(src_tmp2, tmp);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_min(src_tmp, tmp);
            v4sf dst_tmp2 = vec_min(src_tmp2, tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold128_gtabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = vec_splats(value);

    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf src_sign = vec_and(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = vec_and(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = vec_and(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = vec_and(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = vec_min(src_abs, pval);
            v4sf dst_tmp2 = vec_min(src_abs2, pval);
            dst_tmp = vec_xor(dst_tmp, src_sign);
            dst_tmp2 = vec_xor(dst_tmp2, src_sign2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf src_sign = vec_and(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = vec_and(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = vec_and(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = vec_and(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = vec_min(src_abs, pval);
            v4sf dst_tmp2 = vec_min(src_abs2, pval);
            dst_tmp = vec_xor(dst_tmp, src_sign);
            dst_tmp2 = vec_xor(dst_tmp2, src_sign2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] >= 0.0f) {
            dst[i] = src[i] > value ? value : src[i];
        } else {
            dst[i] = src[i] < (-value) ? (-value) : src[i];
        }
    }
}


static inline void threshold128_lt_f(float *src, float *dst, int len, float value)
{
    const v4sf tmp = {value, value, value, value};

    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_max(src_tmp, tmp);
            v4sf dst_tmp2 = vec_max(src_tmp2, tmp);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_max(src_tmp, tmp);
            v4sf dst_tmp2 = vec_max(src_tmp2, tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

static inline void threshold128_ltabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = vec_splats(value);

    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf src_sign = vec_and(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = vec_and(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = vec_and(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = vec_and(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = vec_max(src_abs, pval);
            v4sf dst_tmp2 = vec_max(src_abs2, pval);
            dst_tmp = vec_xor(dst_tmp, src_sign);
            dst_tmp2 = vec_xor(dst_tmp2, src_sign2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf src_sign = vec_and(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_sign2 = vec_and(src_tmp2, *(v4sf *) _ps_sign_mask);
            v4sf src_abs = vec_and(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf src_abs2 = vec_and(src_tmp2, *(v4sf *) _ps_pos_sign_mask);
            v4sf dst_tmp = vec_max(src_abs, pval);
            v4sf dst_tmp2 = vec_max(src_abs2, pval);
            dst_tmp = vec_xor(dst_tmp, src_sign);
            dst_tmp2 = vec_xor(dst_tmp2, src_sign2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] >= 0.0f) {
            dst[i] = src[i] < value ? value : src[i];
        } else {
            dst[i] = src[i] > (-value) ? (-value) : src[i];
        }
    }
}


static inline void threshold128_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v4sf ltlevel_v = {ltlevel, ltlevel, ltlevel, ltlevel};
    const v4sf ltvalue_v = {ltvalue, ltvalue, ltvalue, ltvalue};
    const v4sf gtlevel_v = {gtlevel, gtlevel, gtlevel, gtlevel};
    const v4sf gtvalue_v = {gtvalue, gtvalue, gtvalue, gtvalue};

    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4bi lt_mask = vec_cmplt(src_tmp, ltlevel_v);
            v4bi gt_mask = vec_cmpgt(src_tmp, gtlevel_v);
            v4sf dst_tmp = vec_sel(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = vec_sel(dst_tmp, gtvalue_v, gt_mask);
            vec_st(dst_tmp, 0, dst + i);
            v4bi lt_mask2 = vec_cmplt(src_tmp2, ltlevel_v);
            v4bi gt_mask2 = vec_cmpgt(src_tmp2, gtlevel_v);
            v4sf dst_tmp2 = vec_sel(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = vec_sel(dst_tmp2, gtvalue_v, gt_mask2);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4bi lt_mask = vec_cmplt(src_tmp, ltlevel_v);
            v4bi gt_mask = vec_cmpgt(src_tmp, gtlevel_v);
            v4sf dst_tmp = vec_sel(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = vec_sel(dst_tmp, gtvalue_v, gt_mask);
            vec_st(dst_tmp, 0, dst + i);
            v4bi lt_mask2 = vec_cmplt(src_tmp2, ltlevel_v);
            v4bi gt_mask2 = vec_cmpgt(src_tmp2, gtlevel_v);
            v4sf dst_tmp2 = vec_sel(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = vec_sel(dst_tmp2, gtvalue_v, gt_mask2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}

static inline void round128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_round(src_tmp);
            v4sf dst_tmp2 = vec_round(src_tmp2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_round(src_tmp);
            v4sf dst_tmp2 = vec_round(src_tmp2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_ceil(src_tmp);
            v4sf dst_tmp2 = vec_ceil(src_tmp2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_ceil(src_tmp);
            v4sf dst_tmp2 = vec_ceil(src_tmp2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_floor(src_tmp);
            v4sf dst_tmp2 = vec_floor(src_tmp2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_floor(src_tmp);
            v4sf dst_tmp2 = vec_floor(src_tmp2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void trunc128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_trunc(src_tmp);
            v4sf dst_tmp2 = vec_trunc(src_tmp2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_trunc(src_tmp);
            v4sf dst_tmp2 = vec_trunc(src_tmp2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

static inline void sqrt128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_sqrt_precise(src_tmp);
            v4sf dst_tmp2 = vec_sqrt_precise(src_tmp2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_sqrt_precise(src_tmp);
            v4sf dst_tmp2 = vec_sqrt_precise(src_tmp2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static inline void flip128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    int mini = ((len - 1) < (2 * ALTIVEC_LEN_FLOAT)) ? (len - 1) : (2 * ALTIVEC_LEN_FLOAT);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst + len - ALTIVEC_LEN_FLOAT), ALTIVEC_LEN_BYTES)) {
        for (int i = 2 * ALTIVEC_LEN_FLOAT; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);  // load a,b,c,d
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf src_tmp_flip = vec_perm(src_tmp, src_tmp, flip_vector);  // rotate vec from abcd to bcba
            v4sf src_tmp_flip2 = vec_perm(src_tmp2, src_tmp2, flip_vector);
            vec_st(src_tmp_flip, 0, dst + len - i - ALTIVEC_LEN_FLOAT);  // store the flipped vector
            vec_st(src_tmp_flip2, 0, dst + len - i - 2 * ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 2 * ALTIVEC_LEN_FLOAT; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf src_tmp_flip = vec_perm(src_tmp, src_tmp, flip_vector);  // rotate vec from abcd to bcba
            v4sf src_tmp_flip2 = vec_perm(src_tmp2, src_tmp2, flip_vector);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &src_tmp_flip, (unsigned char *) (dst + len - i - ALTIVEC_LEN_FLOAT));
                vec_stu(*(v16u8 *) &src_tmp_flip2, (unsigned char *) (dst + len - i - 2 * ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(src_tmp_flip, 0, dst + len - i - ALTIVEC_LEN_FLOAT);
                vec_st(src_tmp_flip2, 0, dst + len - i - 2 * ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void sincos_ps(v4sf x, v4sf *s, v4sf *c)
{
    v4sf xmm1, xmm2, sign_bit_sin, y;
    v4si emm0, emm2, emm4;

#if 1
    v4bi xmm3;
#else
    v4sf xmm3 = *(v4sf *) _ps_0;
#endif

    sign_bit_sin = x;
    /* take the absolute value */
    x = vec_and(x, *(v4sf *) _ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = vec_and(sign_bit_sin, *(v4sf *) _ps_sign_mask);

    /* scale by 4/Pi */
    y = vec_mul(x, *(v4sf *) _ps_cephes_FOPI);

    /* store the integer part of y in emm2 */
    emm2 = vec_cts(y, 0);

    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vec_add(emm2, *(v4si *) _pi32_1);
    emm2 = vec_and(emm2, *(v4si *) _pi32_inv1);
    y = vec_ctf(emm2, 0);
    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0 = vec_and(emm2, *(v4si *) _pi32_4);
    // emm0 = vec_slli(emm0, 29);
    // v4ui shift_val = vec_splat_u32(13); //between -16 and +15. asm tests gives 13 for 29 bits: 16+13?
    v4ui shift_val = vec_splats((unsigned int) 29);
    emm0 = vec_sl(emm0, shift_val);
    v4si *swap_sign_bit_sin = &emm0;

    /* get the polynom selection mask for the sine*/
    emm2 = vec_and(emm2, *(v4si *) _pi32_2);
    emm2 = vec_cmpeq(emm2, *(v4si *) _pi32_0);
    v4si *poly_mask = &emm2;

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = vec_madd(y, *(v4sf *) _ps_minus_cephes_DP1, x);
    x = vec_madd(y, *(v4sf *) _ps_minus_cephes_DP2, x);
    x = vec_madd(y, *(v4sf *) _ps_minus_cephes_DP3, x);

    emm4 = vec_sub(emm4, *(v4si *) _pi32_2);
    emm4 = vec_andc(*(v4si *) _pi32_4, emm4);
    emm4 = vec_sl(emm4, shift_val);
    v4si *sign_bit_cos = &emm4;

    sign_bit_sin = vec_xor(sign_bit_sin, *(v4sf *) swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v4sf z = vec_mul(x, x);

    y = vec_madd(*(v4sf *) _ps_coscof_p0, z, *(v4sf *) _ps_coscof_p1);
    y = vec_madd(y, z, *(v4sf *) _ps_coscof_p2);
    y = vec_mul(y, z);
    y = vec_mul(y, z);
    y = vec_nmsub(z, *(v4sf *) _ps_0p5, y);
    y = vec_add(y, *(v4sf *) _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    v4sf y2 = vec_madd(*(v4sf *) _ps_sincof_p0, z, *(v4sf *) _ps_sincof_p1);
    y2 = vec_madd(y2, z, *(v4sf *) _ps_sincof_p2);
    y2 = vec_mul(y2, z);
    y2 = vec_madd(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    xmm3 = *(v4bi *) poly_mask;
    xmm1 = vec_sel(y, y2, xmm3);
    xmm2 = vec_sel(y2, y, xmm3);
#else
    xmm3 = *(v4sf *) poly_mask;
    v4sf ysin2 = vec_and(xmm3, y2);
    v4sf ysin1 = vec_andc(y, xmm3);
    y2 = vec_sub(y2, ysin2);
    y = vec_sub(y, ysin1);
    xmm1 = vec_add(ysin1, ysin2);
    xmm2 = vec_add(y, y2);
#endif

    /* update the sign */
    *s = vec_xor(xmm1, sign_bit_sin);
    *c = vec_xor(xmm2, *(v4sf *) sign_bit_cos);
}

static inline void sincos128f(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src), (uintptr_t) (dst_sin), (uintptr_t) (dst_cos), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            vec_st(dst_sin_tmp, 0, dst_sin + i);
            vec_st(dst_cos_tmp, 0, dst_cos + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst_sin = (uintptr_t) (dst_sin) % ALTIVEC_LEN_BYTES;
        int unalign_dst_cos = (uintptr_t) (dst_cos) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                src_tmp = vec_ld(0, src + i);
            }
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);

            if (unalign_dst_sin) {
                vec_stu(*(v16u8 *) &dst_sin_tmp, (unsigned char *) (dst_sin + i));
            } else {
                vec_st(dst_sin_tmp, 0, dst_sin + i);
            }

            if (unalign_dst_cos) {
                vec_stu(*(v16u8 *) &dst_cos_tmp, (unsigned char *) (dst_cos + i));
            } else {
                vec_st(dst_cos_tmp, 0, dst_cos + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void sincos128f_interleaved(float *src, complex32_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            v4sf sin_cos_h = vec_mergeh(dst_cos_tmp, dst_sin_tmp);
            v4sf sin_cos_l = vec_mergel(dst_cos_tmp, dst_sin_tmp);
            vec_st(sin_cos_h, 0, (float *) (dst) + j);
            vec_st(sin_cos_l, 0, (float *) (dst) + j + ALTIVEC_LEN_FLOAT);
            j += 2 * ALTIVEC_LEN_FLOAT;
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                src_tmp = vec_ld(0, src + i);
            }
            v4sf dst_sin_tmp;
            v4sf dst_cos_tmp;
            sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
            v4sf sin_cos_h = vec_mergeh(dst_cos_tmp, dst_sin_tmp);
            v4sf sin_cos_l = vec_mergel(dst_cos_tmp, dst_sin_tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &sin_cos_h, (unsigned char *) ((float *) (dst) + j));
                vec_stu(*(v16u8 *) &sin_cos_l, (unsigned char *) ((float *) (dst) + j + ALTIVEC_LEN_FLOAT));

            } else {
                vec_st(sin_cos_h, 0, (float *) (dst) + j);
                vec_st(sin_cos_l, 0, (float *) (dst) + j + ALTIVEC_LEN_FLOAT);
            }

            j += 2 * ALTIVEC_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], &(dst[i].im), &(dst[i].re));
    }
}

static inline v4sf tanf_ps(v4sf xx)
{
    v4sf x, y, z, zz;
    v4si j;  // long?
    v4sf sign;
    v4sf tmp;
    v4si tmpi;
    __vector bool jandone, jandtwo, xsupem4;

    x = vec_and(*(v4sf *) _ps_pos_sign_mask, xx);  // fabs(xx) //OK
    sign = vec_and(xx, *(v4sf *) _ps_sign_mask);

    /* compute x mod PIO4 */
    tmp = vec_mul(*(v4sf *) _ps_FOPI, x);
    j = vec_cts(tmp, 0);
    y = vec_ctf(j, 0);

    jandone = vec_cmpgt(vec_and(j, *(v4si *) _pi32_1), *(v4si *) _pi32_0);  // Ok?
    tmp = vec_and(*(v4sf *) _ps_1, jandone);
    y = vec_add(y, tmp);
    tmpi = vec_and(*(v4si *) _pi32_1, jandone);
    j = vec_add(j, tmpi);

    z = vec_madd(y, *(v4sf *) _ps_DP1, x);
    z = vec_madd(y, *(v4sf *) _ps_DP2, z);
    z = vec_madd(y, *(v4sf *) _ps_DP3, z);

    zz = vec_mul(z, z);  // z*z

    // TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = vec_madd(*(v4sf *) _ps_TAN_P0, zz, *(v4sf *) _ps_TAN_P1);
    tmp = vec_madd(tmp, zz, *(v4sf *) _ps_TAN_P2);
    tmp = vec_madd(tmp, zz, *(v4sf *) _ps_TAN_P3);
    tmp = vec_madd(tmp, zz, *(v4sf *) _ps_TAN_P4);
    tmp = vec_madd(tmp, zz, *(v4sf *) _ps_TAN_P5);
    tmp = vec_mul(zz, tmp);

    tmp = vec_madd(tmp, z, z);
    xsupem4 = vec_cmpgt(x, *(v4sf *) _ps_1em4);  // if( x > 1.0e-4 )
    y = vec_sel(z, tmp, xsupem4);

    jandtwo = vec_cmpgt(vec_and(j, *(v4si *) _pi32_2), *(v4si *) _pi32_0);

    tmp = vec_div_precise(*(v4sf *) _ps_min1, y);
    y = vec_sel(y, tmp, jandtwo);
    y = vec_xor(y, sign);

    return (y);
}

static inline void tan128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(tanf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = tanf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void tan128f_naive(float *restrict src, float *restrict dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            v4sf sin_tmp, cos_tmp;
            sincos_ps(a, &sin_tmp, &cos_tmp);
            vec_st(vec_div(sin_tmp, cos_tmp), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf sin_tmp, cos_tmp;
            sincos_ps(a, &sin_tmp, &cos_tmp);
            v4sf c = vec_div(sin_tmp, cos_tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline v4sf atanf_ps(v4sf xx)
{
    v4sf x, y, z;
    v4sf sign;
    v4bi suptan3pi8, inftan3pi8suppi8;
    v4sf tmp;

    x = vec_and(*(v4sf *) _ps_pos_sign_mask, xx);
    sign = vec_and(xx, *(v4sf *) _ps_sign_mask);

    /* range reduction */

    y = *(v4sf *) _ps_0;
    suptan3pi8 = vec_cmpgt(x, *(v4sf *) _ps_TAN3PI8F);  // if( x > tan 3pi/8 )
    x = vec_sel(x, vec_div_precise(*(v4sf *) _ps_min1, x), suptan3pi8);
    y = vec_sel(y, *(v4sf *) _ps_PIO2F, suptan3pi8);

    inftan3pi8suppi8 = vec_and(vec_cmple(x, *(v4sf *) _ps_TAN3PI8F), vec_cmpgt(x, *(v4sf *) _ps_TANPI8F));  // if( x > tan 3pi/8 )

    // To be optimised with RCP?
    x = vec_sel(x, vec_div_precise(vec_sub(x, *(v4sf *) _ps_1), vec_add(x, *(v4sf *) _ps_1)), inftan3pi8suppi8);
    y = vec_sel(y, *(v4sf *) _ps_PIO4F, inftan3pi8suppi8);

    z = vec_mul(x, x);

    tmp = vec_madd(*(v4sf *) _ps_ATAN_P0, z, *(v4sf *) _ps_ATAN_P1);
    tmp = vec_madd(tmp, z, *(v4sf *) _ps_ATAN_P2);
    tmp = vec_madd(tmp, z, *(v4sf *) _ps_ATAN_P3);
    tmp = vec_mul(z, tmp);
    tmp = vec_madd(tmp, x, x);

    y = vec_add(y, tmp);

    y = vec_xor(y, sign);
    return (y);
}

static inline void atan128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(atanf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = atanf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}


static inline v4sf atan2f_ps(v4sf y, v4sf x)
{
    v4sf z, w;
    v4bi xinfzero, yinfzero, xeqzero, yeqzero;
    v4bi xeqzeroandyinfzero, yeqzeroandxinfzero;
    v4bi specialcase;
    v4sf tmp;

    xinfzero = vec_cmplt(x, *(v4sf *) _ps_0);  // code =2
    yinfzero = vec_cmplt(y, *(v4sf *) _ps_0);  // code = code |1;

    xeqzero = vec_cmpeq(x, *(v4sf *) _ps_0);
    yeqzero = vec_cmpeq(y, *(v4sf *) _ps_0);

    xeqzeroandyinfzero = vec_and(xeqzero, yinfzero);
    yeqzeroandxinfzero = vec_and(yeqzero, xinfzero);

#if 1
    xeqzeroandyinfzero = vec_and(xeqzeroandyinfzero, *(v4bi *) _ps_sign_mask);
    tmp = vec_xor(*(v4sf *) _ps_PIO2F, xeqzeroandyinfzero);  // either PI or -PI
    z = vec_andc(tmp, yeqzero);                              // not(yeqzero) and tmp => 0, PI/2, -PI/2
#else
    z = *(v4sf *) _ps_PIO2F;
    z = vec_sel(z, *(v4sf *) _ps_mPIO2F, xeqzeroandyinfzero);
    z = vec_sel(z, *(v4sf *) _ps_0, yeqzero);
#endif
    z = vec_sel(z, *(v4sf *) _ps_PIF, yeqzeroandxinfzero);
    specialcase = vec_xor(xeqzero, yeqzero);

#if 0
    tmp = vec_and(*(v4sf *) _ps_PIF, vec_andc(xinfzero, yinfzero));
    v4sf tmp2 = vec_and(*(v4sf *) _ps_mPIF, vec_and(yinfzero, xinfzero));
    w = vec_add(tmp, tmp2);
#else
    w = *(v4sf *) _ps_0;
    w = vec_sel(w, *(v4sf *) _ps_PIF, vec_andc(xinfzero, yinfzero));  // y >= 0 && x<0
    w = vec_sel(w, *(v4sf *) _ps_mPIF, vec_and(yinfzero, xinfzero));  // y < 0 && x<0
#endif

    tmp = vec_div_precise(y, x);
    tmp = atanf_ps(tmp);
    tmp = vec_add(w, tmp);
    z = vec_sel(tmp, z, specialcase);  // atanf(y/x) if not in special case
    return (z);
}

static inline void atan2128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(atan2f_ps(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src1) {
                a = (v4sf) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4sf) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4sf c = atan2f_ps(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src1[i], src2[i]);
    }
}

static inline void atan2128f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    int j = 0;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, (float *) src + j);
            v4sf b = vec_ld(0, (float *) src + j + ALTIVEC_LEN_FLOAT);
            v4sf re = vec_perm(a, b, re_mask);
            v4sf im = vec_perm(a, b, im_mask);
            vec_st(atan2f_ps(im, re), 0, dst + i);
            j += 2 * ALTIVEC_LEN_FLOAT;
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a, b;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) ((float *) src + j));
                b = (v4sf) vec_ldu((unsigned char *) ((float *) src + j + ALTIVEC_LEN_FLOAT));
            } else {
                a = vec_ld(0, (float *) src + j);
                b = vec_ld(0, (float *) src + j + ALTIVEC_LEN_FLOAT);
            }
            v4sf re = vec_perm(a, b, re_mask);
            v4sf im = vec_perm(a, b, im_mask);
            v4sf c = atan2f_ps(im, re);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
            j += 2 * ALTIVEC_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atan2f(src[i].im, src[i].re);
    }
}

static inline v4sf tanhf_ps(v4sf xx)
{
    v4sf x, z, z_first_branch, z_second_branch;
    v4bi xxsup0, xsupmaxlogfdiv2, xsup0p625;

    xxsup0 = vec_cmpgt(xx, *(v4sf *) _ps_0);
    xsupmaxlogfdiv2 = vec_cmpgt(xx, *(v4sf *) _ps_MAXLOGFDIV2);

    x = vec_and(*(v4sf *) _ps_pos_sign_mask, xx);

    xsup0p625 = vec_cmpge(x, *(v4sf *) _ps_0p625);
    x = vec_sel(x, exp_ps_alternate(vec_add(x, x)), xsup0p625);

    // z = 1.0 - 2.0 / (x + 1.0);
    z_first_branch = vec_add(x, *(v4sf *) _ps_1);
    z_first_branch = vec_div_precise(*(v4sf *) _ps_min2, z_first_branch);
    z_first_branch = vec_add(*(v4sf *) _ps_1, z_first_branch);
    z_first_branch = vec_sel(vec_xor(*(v4sf *) _ps_neg_sign_mask, z_first_branch), z_first_branch, xxsup0);

    // z = x * x;
    z = vec_mul(x, x);

    z_second_branch = vec_madd(z, *(v4sf *) _ps_TANH_P0, *(v4sf *) _ps_TANH_P1);
    z_second_branch = vec_madd(z_second_branch, z, *(v4sf *) _ps_TANH_P2);
    z_second_branch = vec_madd(z_second_branch, z, *(v4sf *) _ps_TANH_P3);
    z_second_branch = vec_madd(z_second_branch, z, *(v4sf *) _ps_TANH_P4);
    z_second_branch = vec_mul(z_second_branch, z);
    z_second_branch = vec_madd(z_second_branch, xx, xx);

    z = vec_sel(z_second_branch, z_first_branch, xsup0p625);
    // if (x > 0.5 * MAXLOGF), return (xx > 0)? 1.0f: -1.0f
    z = vec_sel(z, *(v4sf *) _ps_min1, xsupmaxlogfdiv2);
    z = vec_sel(z, *(v4sf *) _ps_1, vec_and(xxsup0, xsupmaxlogfdiv2));

    return (z);
}

static inline void tanh128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(tanhf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = tanhf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanhf(src[i]);
    }
}

static inline v4sf atanhf_ps(v4sf x)
{
    v4sf z, tmp, tmp2, z_first_branch, z_second_branch;
    v4bi xsup1, xinfmin1, zinf1emin4, zinf0p5;

    z = vec_and(*(v4sf *) _ps_pos_sign_mask, x);

    xsup1 = vec_cmpge(x, *(v4sf *) _ps_1);
    xinfmin1 = vec_cmple(x, *(v4sf *) _ps_min1);
    zinf1emin4 = vec_cmplt(z, *(v4sf *) _ps_1emin4);
    zinf0p5 = vec_cmplt(z, *(v4sf *) _ps_0p5);

    // First branch
    tmp = vec_mul(x, x);
    z_first_branch = vec_madd(*(v4sf *) _ps_ATANH_P0, tmp, *(v4sf *) _ps_ATANH_P1);
    z_first_branch = vec_madd(z_first_branch, tmp, *(v4sf *) _ps_ATANH_P2);
    z_first_branch = vec_madd(z_first_branch, tmp, *(v4sf *) _ps_ATANH_P3);
    z_first_branch = vec_madd(z_first_branch, tmp, *(v4sf *) _ps_ATANH_P4);
    z_first_branch = vec_mul(z_first_branch, tmp);
    z_first_branch = vec_madd(z_first_branch, x, x);

    // Second branch
    // precision of rcp vs div?
    tmp = vec_sub(*(v4sf *) _ps_1, x);
    tmp2 = vec_re(tmp);
    tmp = vec_madd(tmp2, x, tmp2);
    z_second_branch = log_ps(tmp);
    z_second_branch = vec_mul(*(v4sf *) _ps_0p5, z_second_branch);

    z = vec_sel(z_second_branch, z_first_branch, zinf0p5);
    z = vec_sel(z, x, zinf1emin4);

    z = vec_sel(z, *(v4sf *) _ps_MAXNUMF, xsup1);
    z = vec_sel(z, *(v4sf *) _ps_minMAXNUMF, xinfmin1);

    return (z);
}

static inline void atanh128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(atanhf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = atanhf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline v4sf sinhf_ps(v4sf x)
{
    v4sf z, z_first_branch, z_second_branch, tmp;
    v4bi xsupmaxlogf, zsup1;
    v4sf sign;

    // x = xx; if x < 0, z = -x, else x
    z = vec_and(*(v4sf *) _ps_pos_sign_mask, x);
    sign = vec_and(x, *(v4sf *) _ps_sign_mask);

    xsupmaxlogf = vec_cmpgt(z, *(v4sf *) _ps_MAXLOGF);

    // First branch
    zsup1 = vec_cmpgt(z, *(v4sf *) _ps_1);
    z_first_branch = exp_ps_alternate(z);
    tmp = vec_div_precise(*(v4sf *) _ps_min0p5, z_first_branch);
    z_first_branch = vec_madd(*(v4sf *) _ps_0p5, z_first_branch, tmp);

#if 1
    z_first_branch = vec_xor(z_first_branch, sign);
#else
    v4sf xinf0 = vec_cmplt(x, _mm_setzero_ps());
    z_first_branch = vec_sel(z_first_branch, vec_xor(*(v4sf *) _ps_neg_sign_mask, z_first_branch), xinf0);
#endif

    // Second branch
    tmp = vec_mul(x, x);
    z_second_branch = vec_madd(*(v4sf *) _ps_SINH_P0, tmp, *(v4sf *) _ps_SINH_P1);
    z_second_branch = vec_madd(z_second_branch, tmp, *(v4sf *) _ps_SINH_P2);
    z_second_branch = vec_mul(z_second_branch, tmp);
    z_second_branch = vec_madd(z_second_branch, x, x);

    // Choose between first and second branch
    z = vec_sel(z_second_branch, z_first_branch, zsup1);

    // Set value to MAXNUMF if abs(x) > MAGLOGF
    // Set value to -MAXNUMF if abs(x) > MAGLOGF and x < 0
    tmp = vec_xor(*(v4sf *) _ps_MAXNUMF, sign);
    z = vec_sel(z, tmp, xsupmaxlogf);

    return (z);
}

static inline void sinh128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(sinhf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = sinhf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = sinhf(src[i]);
    }
}

static inline v4sf coshf_ps(v4sf xx)
{
    v4sf x, y, tmp;
    v4bi xsupmaxlogf;

    x = vec_and(*(v4sf *) _ps_pos_sign_mask, xx);
    xsupmaxlogf = vec_cmpgt(x, *(v4sf *) _ps_MAXLOGF);

    y = exp_ps_alternate(x);
    tmp = vec_div_precise(*(v4sf *) _ps_0p5, y);  // or 1/(2*y)
    y = vec_madd(*(v4sf *) _ps_0p5, y, tmp);
    y = vec_sel(y, *(v4sf *) _ps_MAXNUMF, xsupmaxlogf);

    return y;
}

static inline void cosh128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(coshf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = coshf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = coshf(src[i]);
    }
}

static inline v4sf asinhf_ps(v4sf xx)
{
    v4sf x, tmp, z, z_first_branch, z_second_branch;
    v4bi xsup1500, xinf0p5;
    v4sf xxinf0;

    x = vec_and(*(v4sf *) _ps_pos_sign_mask, xx);
    xsup1500 = vec_cmpgt(x, *(v4sf *) _ps_1500);
    xinf0p5 = vec_cmplt(x, *(v4sf *) _ps_0p5);

    xxinf0 = vec_and(xx, *(v4sf *) _ps_sign_mask);

    tmp = vec_mul(x, x);
    // First Branch (x < 0.5)
    z_first_branch = vec_madd(*(v4sf *) _ps_ASINH_P0, tmp, *(v4sf *) _ps_ASINH_P1);
    z_first_branch = vec_madd(z_first_branch, tmp, *(v4sf *) _ps_ASINH_P2);
    z_first_branch = vec_madd(z_first_branch, tmp, *(v4sf *) _ps_ASINH_P3);
    z_first_branch = vec_mul(z_first_branch, tmp);
    z_first_branch = vec_madd(z_first_branch, x, x);

    // Second Branch
    z_second_branch = vec_sqrt_precise(vec_add(tmp, *(v4sf *) _ps_1));
    z_second_branch = log_ps(vec_add(z_second_branch, x));

    z = vec_sel(z_second_branch, z_first_branch, xinf0p5);
    tmp = log_ps(x);
    tmp = vec_add(tmp, *(v4sf *) _ps_LOGE2F);
    z = vec_sel(z, tmp, xsup1500);
    z = vec_xor(z, xxinf0);
    return z;
}

static inline void asinh128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(asinhf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = asinhf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinhf(src[i]);
    }
}

static inline v4sf acoshf_ps(v4sf x)
{
    v4sf z, z_first_branch, z_second_branch;
    v4bi xsup1500, zinf0p5, xinf1;
    v4sf tmp;
    xsup1500 = vec_cmpgt(x, *(v4sf *) _ps_1500);  // return  (logf(x) + LOGE2F)
    xinf1 = vec_cmplt(x, *(v4sf *) _ps_1);        // return 0

    z = vec_sub(x, *(v4sf *) _ps_1);

    zinf0p5 = vec_cmplt(z, *(v4sf *) _ps_0p5);  // first and second branch

    // First Branch (z < 0.5)
    z_first_branch = vec_madd(*(v4sf *) _ps_ACOSH_P0, z, *(v4sf *) _ps_ACOSH_P1);
    z_first_branch = vec_madd(z_first_branch, z, *(v4sf *) _ps_ACOSH_P2);
    z_first_branch = vec_madd(z_first_branch, z, *(v4sf *) _ps_ACOSH_P3);
    z_first_branch = vec_madd(z_first_branch, z, *(v4sf *) _ps_ACOSH_P4);
    z_first_branch = vec_mul(z_first_branch, vec_sqrt_precise(z));

    // Second Branch
    z_second_branch = vec_madd(z, x, z);
    z_second_branch = vec_sqrt_precise(z_second_branch);
    z_second_branch = vec_add(x, z_second_branch);
    z_second_branch = log_ps(z_second_branch);

    z = vec_sel(z_second_branch, z_first_branch, zinf0p5);
    tmp = log_ps(x);
    tmp = vec_add(tmp, *(v4sf *) _ps_LOGE2F);
    z = vec_sel(z, tmp, xsup1500);

    v4bi *xinf1_ptr = &xinf1;
    z = vec_andc(z, *(v4sf *) xinf1_ptr);

    return z;
}

static inline void acosh128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(acoshf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = acoshf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = acoshf(src[i]);
    }
}

static inline v4sf asinf_ps(v4sf xx)
{
    v4sf a, x, z, z_tmp;
    v4sf sign;
    v4bi ainfem4, asup0p5;
    v4sf tmp;
    x = xx;
    a = vec_and(*(v4sf *) _ps_pos_sign_mask, x);  // fabs(x)
    // sign = vec_cmplt(x, *(v4sf *) _ps_0);        // 0xFFFFFFFF if x < 0.0
    sign = vec_and(xx, *(v4sf *) _ps_sign_mask);

    const v4sf ps1em4 = {1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4};
    ainfem4 = vec_cmplt(a, ps1em4);  // if( a < 1.0e-4f )

    asup0p5 = vec_cmpgt(a, *(v4sf *) _ps_0p5);  // if( a > 0.5f ) flag = 1 else 0
    z_tmp = vec_sub(*(v4sf *) _ps_1, a);
    z_tmp = vec_mul(*(v4sf *) _ps_0p5, z_tmp);
    z = vec_sel(vec_mul(a, a), z_tmp, asup0p5);
    x = vec_sel(a, vec_sqrt_precise(z), asup0p5);

    tmp = vec_madd(*(v4sf *) _ps_ASIN_P0, z, *(v4sf *) _ps_ASIN_P1);
    tmp = vec_madd(z, tmp, *(v4sf *) _ps_ASIN_P2);
    tmp = vec_madd(z, tmp, *(v4sf *) _ps_ASIN_P3);
    tmp = vec_madd(z, tmp, *(v4sf *) _ps_ASIN_P4);
    tmp = vec_mul(z, tmp);
    tmp = vec_madd(x, tmp, x);

    z = tmp;

    z_tmp = vec_add(z, z);
    z_tmp = vec_sub(*(v4sf *) _ps_PIO2F, z_tmp);
    z = vec_sel(z, z_tmp, asup0p5);

    // done:
    z = vec_sel(z, a, ainfem4);
    // z = vec_sel(z, vec_xor(*(v4sf *) _ps_neg_sign_mask, z), sign);
    z = vec_xor(z, sign);

    // if (x > 1.0) then return 0.0
    z = vec_sel(z, *(v4sf *) _ps_0, vec_cmpgt(x, *(v4sf *) _ps_1));
    return (z);
}

static inline void asin128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(asinf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = asinf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = asinf(src[i]);
    }
}


static inline void cplxconj128f(complex32_t *src, complex32_t *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);  //(len << 1) >> 2;
    stop_len *= 2 * ALTIVEC_LEN_FLOAT;             // stop_len << 2;

    __attribute__((aligned(ALTIVEC_LEN_BYTES))) int32_t conj_mask[ALTIVEC_LEN_FLOAT] = {(int) 0x00000000, (int) 0x80000000, (int) 0x00000000, (int) 0x80000000};
    int i;
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (i = 0; i < 2 * stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, (float *) (src) + i);
            v4sf src_tmp2 = vec_ld(0, (float *) (src) + i + ALTIVEC_LEN_FLOAT);
            v4sf dst_tmp = vec_xor(src_tmp, *(v4sf *) &conj_mask);
            v4sf dst_tmp2 = vec_xor(src_tmp2, *(v4sf *) &conj_mask);
            vec_st(dst_tmp, 0, (float *) (dst) + i);
            vec_st(dst_tmp2, 0, (float *) (dst) + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (i = 0; i < 2 * stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src) + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) ((float *) (src) + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, (float *) (src) + i);
                src_tmp2 = vec_ld(0, (float *) (src) + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf dst_tmp = vec_xor(src_tmp, *(v4sf *) &conj_mask);
            v4sf dst_tmp2 = vec_xor(src_tmp2, *(v4sf *) &conj_mask);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) ((float *) (dst) + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) ((float *) (dst) + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(dst_tmp, 0, (float *) (dst) + i);
                vec_st(dst_tmp2, 0, (float *) (dst) + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i].re = src[i].re;
        dst[i].im = -src[i].im;
    }
}

static inline void PRelu128f(float *src, float *dst, float alpha, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    v4sf alpha_vec = {alpha, alpha, alpha, alpha};
    const v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf tmp = vec_mul(alpha_vec, src_tmp);
            v4bi cmp = vec_cmpgt(src_tmp, zero);
            v4sf blend_res = vec_sel(tmp, src_tmp, cmp);
            vec_st(blend_res, 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) ((float *) (src) + i));
            } else {
                src_tmp = vec_ld(0, (float *) (src) + i);
            }
            v4sf tmp = vec_mul(alpha_vec, src_tmp);
            v4bi cmp = vec_cmpgt(src_tmp, zero);
            v4sf blend_res = vec_sel(tmp, src_tmp, cmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &blend_res, (unsigned char *) ((float *) (dst) + i));
            } else {
                vec_st(blend_res, 0, (float *) (dst) + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] > 0.0f)
            dst[i] = src[i];
        else
            dst[i] = alpha * src[i];
    }
}

static inline void vectorSlope128f(float *dst, int len, float offset, float slope)
{
    v4sf coef = {0.0f, slope, 2.0f * slope, 3.0f * slope};
    v4sf slope8_vec = vec_splats(8.0f * slope);
    v4sf curVal = vec_add(vec_splats(offset), coef);
    v4sf curVal2 = vec_add(vec_splats(offset), coef);
    curVal2 = vec_add(curVal2, vec_splats(4.0f * slope));

    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (len >= ALTIVEC_LEN_BYTES) {
        if (isAligned((uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
            vec_st(curVal, 0, dst + 0);
            vec_st(curVal2, 0, dst + ALTIVEC_LEN_FLOAT);
        } else {
            vec_stu(*(v16u8 *) &curVal, (unsigned char *) ((float *) (dst) + 0));
            vec_stu(*(v16u8 *) &curVal2, (unsigned char *) ((float *) (dst) + ALTIVEC_LEN_FLOAT));
        }

        if (isAligned((uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
            for (int i = 2 * ALTIVEC_LEN_FLOAT; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
                curVal = vec_add(curVal, slope8_vec);
                vec_st(curVal, 0, dst + i);
                curVal2 = vec_add(curVal2, slope8_vec);
                vec_st(curVal2, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        } else {
            for (int i = 2 * ALTIVEC_LEN_FLOAT; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
                curVal = vec_add(curVal, slope8_vec);
                vec_stu(*(v16u8 *) &curVal, (unsigned char *) ((float *) (dst) + i));
                curVal2 = vec_add(curVal2, slope8_vec);
                vec_stu(*(v16u8 *) &curVal2, (unsigned char *) ((float *) (dst) + i + ALTIVEC_LEN_FLOAT));
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

// from https://stackoverflow.com/questions/57454416/sse-integer-2n-powers-of-2-for-32-bit-integers-without-avx2
static inline v4sf power_of_twof(v4si b)
{
    v4si exp = vec_add(b, vec_splats((int) 127));
    v4si s;
    v4ui shift_bits = vec_splats((unsigned int) 23);
    s = vec_sl(exp, shift_bits);
    return *(v4sf *) &s;
}

static inline v4sf cbrtf_ps(v4sf xx)
{
    v4sf e, rem, sign;
    v4sf x, z;
    v4sf tmp, tmp2;

    x = xx;
    sign = vec_and(xx, *(v4sf *) _ps_sign_mask);
    x = vec_and(x, *(v4sf *) _ps_pos_sign_mask);

    z = x;
    /* extract power of 2, leaving
     * mantissa between 0.5 and 1
     */
    // x = frexpf(x, &e);
    // solve problem for zero
    v4ui shift_bits = vec_splats((unsigned int) 23);
    v4si emm0 = vec_sr(*(v4si *) &x, shift_bits);
    x = vec_and(x, *(v4sf *) _ps_inv_mant_mask);
    x = vec_or(x, *(v4sf *) _ps_0p5);
    emm0 = vec_sub(emm0, *(v4si *) _pi32_0x7e);  // -7f + 1
    e = vec_ctf(emm0, 0);
    tmp = vec_madd(*(v4sf *) _ps_CBRTF_P0, x, *(v4sf *) _ps_CBRTF_P1);
    tmp = vec_madd(x, tmp, *(v4sf *) _ps_CBRTF_P2);
    tmp = vec_madd(x, tmp, *(v4sf *) _ps_CBRTF_P3);
    x = vec_madd(x, tmp, *(v4sf *) _ps_CBRTF_P4);

    /* exponent divided by 3 */
    v4bi e_sign = vec_cmpge(e, *(v4sf *) _ps_0);
    e = vec_and(e, *(v4sf *) _ps_pos_sign_mask);

    rem = e;
    e = vec_mul(e, *(v4sf *) _ps_0p3);
    v4sf e_tmp = vec_mul(*(v4sf *) _ps_3, vec_floor(e));
    rem = vec_sub(rem, e_tmp);

    v4sf mul1, mul2;
    v4sf mul_cst1 = vec_sel(*(v4sf *) _ps_cephes_invCBRT2, *(v4sf *) _ps_cephes_CBRT2, e_sign);
    v4sf mul_cst2 = vec_sel(*(v4sf *) _ps_cephes_invCBRT4, *(v4sf *) _ps_cephes_CBRT4, e_sign);
    mul1 = vec_mul(x, mul_cst1);
    mul2 = vec_mul(x, mul_cst2);

    v4si remi = vec_cts(rem, 0);  // rem integer
    v4bi rem1 = vec_cmpeq(remi, *(v4si *) _pi32_1);
    v4bi rem2 = vec_cmpeq(remi, *(v4si *) _pi32_2);

    x = vec_sel(x, mul1, rem1);  // rem==1
    x = vec_sel(x, mul2, rem2);  // rem==2

    /* multiply by power of 2 */
    //  x = ldexpf(x, e);
    // x= x* (1 >> e)
    v4sf cst = power_of_twof(vec_cts(e, 0));
    // blend sign of e
    tmp = vec_mul(x, cst);
    tmp2 = vec_div_precise(x, cst);
    x = vec_sel(tmp2, tmp, e_sign);

    /* Newton iteration */
    // x -= (x - (z / (x * x))) * 0.333333333333;
    tmp2 = vec_mul(x, x);
    tmp2 = vec_div_precise(z, tmp2);
    tmp2 = vec_sub(x, tmp2);
    tmp2 = vec_mul(tmp2, *(v4sf *) _ps_0p3);
    x = vec_sub(x, tmp2);

    x = vec_xor(x, sign);
    return x;
}

static inline void cbrt128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(cbrtf_ps(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a;
            if (unalign_src) {
                a = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }
            v4sf c = cbrtf_ps(a);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = cbrtf(src[i]);
    }
}

static inline void modf128f(float *src, float *integer, float *remainder, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    if (areAligned3((uintptr_t) (src), (uintptr_t) (integer), (uintptr_t) (remainder), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf integer_tmp = vec_trunc(src_tmp);
            v4sf integer_tmp2 = vec_trunc(src_tmp2);
            v4sf remainder_tmp = vec_sub(src_tmp, integer_tmp);
            v4sf remainder_tmp2 = vec_sub(src_tmp2, integer_tmp2);
            vec_st(integer_tmp, 0, integer + i);
            vec_st(integer_tmp2, 0, integer + i + ALTIVEC_LEN_FLOAT);
            vec_st(remainder_tmp, 0, remainder + i);
            vec_st(remainder_tmp2, 0, remainder + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_integer = (uintptr_t) (integer) % ALTIVEC_LEN_BYTES;
        int unalign_remainder = (uintptr_t) (remainder) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }
            v4sf integer_tmp = vec_trunc(src_tmp);
            v4sf integer_tmp2 = vec_trunc(src_tmp2);
            v4sf remainder_tmp = vec_sub(src_tmp, integer_tmp);
            v4sf remainder_tmp2 = vec_sub(src_tmp2, integer_tmp2);

            if (unalign_integer) {
                vec_stu(*(v16u8 *) &integer_tmp, (unsigned char *) (integer + i));
                vec_stu(*(v16u8 *) &integer_tmp2, (unsigned char *) (integer + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(integer_tmp, 0, integer + i);
                vec_st(integer_tmp2, 0, integer + i + ALTIVEC_LEN_FLOAT);
            }

            if (unalign_remainder) {
                vec_stu(*(v16u8 *) &remainder_tmp, (unsigned char *) (remainder + i));
                vec_stu(*(v16u8 *) &remainder_tmp2, (unsigned char *) (remainder + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(remainder_tmp, 0, remainder + i);
                vec_st(remainder_tmp2, 0, remainder + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        remainder[i] = modff(src[i], &(integer[i]));
    }
}

static inline void dot128f(float *src1, float *src2, int len, float *dst)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    __attribute__((aligned(ALTIVEC_LEN_BYTES))) float accumulate[ALTIVEC_LEN_FLOAT];
    float tmp_acc = 0.0f;

    v4sf vec_acc1 = vec_splats(0.0f);  // initialize the vector accumulator
    v4sf vec_acc2 = vec_splats(0.0f);  // initialize the vector accumulator

    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf src1_tmp = vec_ld(0, src1 + i);
            v4sf src1_tmp2 = vec_ld(0, src1 + i + ALTIVEC_LEN_FLOAT);
            v4sf src2_tmp = vec_ld(0, src2 + i);
            v4sf src2_tmp2 = vec_ld(0, src2 + i + ALTIVEC_LEN_FLOAT);
            vec_acc1 = vec_madd(src1_tmp, src2_tmp, vec_acc1);
            vec_acc2 = vec_madd(src1_tmp2, src2_tmp2, vec_acc2);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src1_tmp, src1_tmp2, src2_tmp, src2_tmp2;
            if (unalign_src1) {
                src1_tmp = (v4sf) vec_ldu((unsigned char *) (src1 + i));
                src1_tmp2 = (v4sf) vec_ldu((unsigned char *) (src1 + i + ALTIVEC_LEN_FLOAT));
            } else {
                src1_tmp = vec_ld(0, src1 + i);
                src1_tmp2 = vec_ld(0, src1 + i + ALTIVEC_LEN_FLOAT);
            }

            if (unalign_src2) {
                src2_tmp = (v4sf) vec_ldu((unsigned char *) (src2 + i));
                src2_tmp2 = (v4sf) vec_ldu((unsigned char *) (src2 + i + ALTIVEC_LEN_FLOAT));
            } else {
                src2_tmp = vec_ld(0, src2 + i);
                src2_tmp2 = vec_ld(0, src2 + i + ALTIVEC_LEN_FLOAT);
            }
            vec_acc1 = vec_madd(src1_tmp, src2_tmp, vec_acc1);
            vec_acc2 = vec_madd(src1_tmp2, src2_tmp2, vec_acc2);
        }
    }
    vec_acc1 = vec_add(vec_acc1, vec_acc2);
    vec_st(vec_acc1, 0, accumulate);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src1[i] * src2[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

    *dst = tmp_acc;
}

static inline void dotc128f(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    int stop_len = len / (4 * ALTIVEC_LEN_FLOAT);
    stop_len *= 4 * ALTIVEC_LEN_FLOAT;

    v4sfx2 vec_acc1 = {{*(v4sf *) _ps_0, *(v4sf *) _ps_0}};  // initialize the vector accumulator
    v4sfx2 vec_acc2 = {{*(v4sf *) _ps_0, *(v4sf *) _ps_0}};  // initialize the vector accumulator

    complex32_t dst_tmp = {{0.0f, 0.0f}};

    __attribute__((aligned(ALTIVEC_LEN_BYTES))) float accumulateRe[ALTIVEC_LEN_FLOAT];
    __attribute__((aligned(ALTIVEC_LEN_BYTES))) float accumulateIm[ALTIVEC_LEN_FLOAT];

    if (areAligned2((uintptr_t) (src1), (uintptr_t) (src2), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < 2 * stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 src1_split = vec_ld2((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
            v4sfx2 src2_split = vec_ld2((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
            v4sfx2 src1_split2 = vec_ld2((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sfx2 src2_split2 = vec_ld2((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sf ac = vec_mul(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = vec_mul(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = vec_mul(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = vec_mul(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 tmp_split;
            v4sfx2 tmp_split2;
            tmp_split.val[0] = vec_nmsub(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = vec_madd(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = vec_nmsub(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = vec_madd(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = vec_add(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = vec_add(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = vec_add(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = vec_add(vec_acc2.val[1], tmp_split2.val[1]);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < 2 * stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sfx2 src1_split, src1_split2, src2_split, src2_split2;

            if (unalign_src1) {
                src1_split = vec_ld2u((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
                src1_split2 = vec_ld2u((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                src1_split = vec_ld2((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
                src1_split2 = vec_ld2((float *) (src1) + i + 2 * ALTIVEC_LEN_FLOAT);
            }

            if (unalign_src2) {
                src2_split = vec_ld2u((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
                src2_split2 = vec_ld2u((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            } else {
                src2_split = vec_ld2((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
                src2_split2 = vec_ld2((float *) (src2) + i + 2 * ALTIVEC_LEN_FLOAT);
            }

            v4sf ac = vec_mul(src1_split.val[0], src2_split.val[0]);     // ac
            v4sf ad = vec_mul(src1_split.val[0], src2_split.val[1]);     // ad
            v4sf ac2 = vec_mul(src1_split2.val[0], src2_split2.val[0]);  // ac
            v4sf ad2 = vec_mul(src1_split2.val[0], src2_split2.val[1]);  // ad
            v4sfx2 tmp_split;
            v4sfx2 tmp_split2;
            tmp_split.val[0] = vec_nmsub(src1_split.val[1], src2_split.val[1], ac);
            tmp_split.val[1] = vec_madd(src1_split.val[1], src2_split.val[0], ad);
            tmp_split2.val[0] = vec_nmsub(src1_split2.val[1], src2_split2.val[1], ac2);
            tmp_split2.val[1] = vec_madd(src1_split2.val[1], src2_split2.val[0], ad2);
            vec_acc1.val[0] = vec_add(vec_acc1.val[0], tmp_split.val[0]);
            vec_acc1.val[1] = vec_add(vec_acc1.val[1], tmp_split.val[1]);
            vec_acc2.val[0] = vec_add(vec_acc2.val[0], tmp_split2.val[0]);
            vec_acc2.val[1] = vec_add(vec_acc2.val[1], tmp_split2.val[1]);
        }
    }

    vec_acc1.val[0] = vec_add(vec_acc1.val[0], vec_acc2.val[0]);
    vec_acc1.val[1] = vec_add(vec_acc1.val[1], vec_acc2.val[1]);
    vec_st(vec_acc1.val[0], 0, accumulateRe);
    vec_st(vec_acc1.val[1], 0, accumulateIm);

    for (int i = stop_len; i < len; i++) {
        dst_tmp.re += src1[i].re * src2[i].re - (src1[i].im * src2[i].im);
        dst_tmp.im += src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }

    dst_tmp.re = dst_tmp.re + accumulateRe[0] + accumulateRe[1] + accumulateRe[2] + accumulateRe[3];
    dst_tmp.im = dst_tmp.im + accumulateIm[0] + accumulateIm[1] + accumulateIm[2] + accumulateIm[3];

    dst->re = dst_tmp.re;
    dst->im = dst_tmp.im;
}

#ifndef __MACH__
static inline void convertFloat32ToU8_128(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * ALTIVEC_LEN_FLOAT);
    stop_len *= (4 * ALTIVEC_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = vec_splats(scale_fact_mult);

    fpu_control_t _mm_rounding_ori;  // save rounding mode
    _FPU_GETCW(_mm_rounding_ori);

    int rounding_ori = fegetround();

    if (rounding_mode == RndZero) {
        _FPU_SETCW(_FPU_RC_ZERO | _FPU_DEFAULT);  // rounding_vec = ROUNDTOZERO;
        fesetround(FE_TOWARDZERO);
    } else if (rounding_mode == RndFinancial) {  // nothing to do, Default bankers rounding => round to nearest even
    } else {
        _FPU_SETCW(_FPU_RC_NEAREST | _FPU_DEFAULT);  // rounding_vec = ROUNDTONEAREST;
        fesetround(FE_TONEAREST);
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp1 = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf src_tmp3 = vec_ld(0, src + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sf src_tmp4 = vec_ld(0, src + i + 3 * ALTIVEC_LEN_FLOAT);
            v4sf tmp1 = vec_mul(src_tmp1, scale_fact_vec);
            v4sf tmp2 = vec_mul(src_tmp2, scale_fact_vec);
            v4sf tmp3 = vec_mul(src_tmp3, scale_fact_vec);
            v4sf tmp4 = vec_mul(src_tmp4, scale_fact_vec);
            v4si tmp1_int = vec_cts(tmp1, 0);
            v4si tmp2_int = vec_cts(tmp2, 0);
            v4si tmp3_int = vec_cts(tmp3, 0);
            v4si tmp4_int = vec_cts(tmp4, 0);
            v8ss tmp5 = vec_packs(tmp1_int, tmp2_int);
            v8ss tmp6 = vec_packs(tmp3_int, tmp4_int);
            v16u8 tmp7 = vec_packsu(tmp5, tmp6);
            vec_st(tmp7, 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp1, src_tmp2, src_tmp3, src_tmp4;
            if (unalign_src) {
                src_tmp1 = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
                src_tmp3 = (v4sf) vec_ldu((unsigned char *) (src + i + 2 * ALTIVEC_LEN_FLOAT));
                src_tmp4 = (v4sf) vec_ldu((unsigned char *) (src + i + 3 * ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp1 = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
                src_tmp3 = vec_ld(0, src + i + 2 * ALTIVEC_LEN_FLOAT);
                src_tmp4 = vec_ld(0, src + i + 3 * ALTIVEC_LEN_FLOAT);
            }

            v4sf tmp1 = vec_mul(src_tmp1, scale_fact_vec);
            v4sf tmp2 = vec_mul(src_tmp2, scale_fact_vec);
            v4sf tmp3 = vec_mul(src_tmp3, scale_fact_vec);
            v4sf tmp4 = vec_mul(src_tmp4, scale_fact_vec);
            v4si tmp1_int = vec_cts(tmp1, 0);
            v4si tmp2_int = vec_cts(tmp2, 0);
            v4si tmp3_int = vec_cts(tmp3, 0);
            v4si tmp4_int = vec_cts(tmp4, 0);
            v8ss tmp5 = vec_packs(tmp1_int, tmp2_int);
            v8ss tmp6 = vec_packs(tmp3_int, tmp4_int);
            v16u8 tmp7 = vec_packsu(tmp5, tmp6);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &tmp7, (unsigned char *) (dst + i));
            } else {
                vec_st(tmp7, 0, dst + i);
            }
        }
    }

    if (rounding_mode == RndFinancial) {
        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (uint8_t) (tmp > 255.0f ? 255.0f : tmp);
        }
        _FPU_SETCW(_mm_rounding_ori);  // restore previous rounding mode
        fesetround(rounding_ori);
    }
}

static inline void convertFloat32ToI16_128(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * ALTIVEC_LEN_FLOAT);
    stop_len *= (4 * ALTIVEC_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = vec_splats(scale_fact_mult);

    fpu_control_t _mm_rounding_ori;  // save rounding mode
    _FPU_GETCW(_mm_rounding_ori);

    int rounding_ori = fegetround();

    if (rounding_mode == RndZero) {
        _FPU_SETCW(_FPU_RC_ZERO | _FPU_DEFAULT);  // rounding_vec = ROUNDTOZERO;
        fesetround(FE_TOWARDZERO);
    } else if (rounding_mode == RndFinancial) {  // nothing to do, Default bankers rounding => round to nearest even
    } else {
        _FPU_SETCW(_FPU_RC_NEAREST | _FPU_DEFAULT);  // rounding_vec = ROUNDTONEAREST;
        fesetround(FE_TONEAREST);
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp1 = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf src_tmp3 = vec_ld(0, src + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sf src_tmp4 = vec_ld(0, src + i + 3 * ALTIVEC_LEN_FLOAT);
            v4sf tmp1 = vec_mul(src_tmp1, scale_fact_vec);
            v4sf tmp2 = vec_mul(src_tmp2, scale_fact_vec);
            v4sf tmp3 = vec_mul(src_tmp3, scale_fact_vec);
            v4sf tmp4 = vec_mul(src_tmp4, scale_fact_vec);
            v4si tmp1_int = vec_cts(tmp1, 0);
            v4si tmp2_int = vec_cts(tmp2, 0);
            v4si tmp3_int = vec_cts(tmp3, 0);
            v4si tmp4_int = vec_cts(tmp4, 0);
            v8ss tmp5 = vec_packs(tmp1_int, tmp2_int);
            v8ss tmp6 = vec_packs(tmp3_int, tmp4_int);
            vec_st(tmp5, 0, dst + i);
            vec_st(tmp6, 0, dst + i + ALTIVEC_LEN_INT16);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp1, src_tmp2, src_tmp3, src_tmp4;
            if (unalign_src) {
                src_tmp1 = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
                src_tmp3 = (v4sf) vec_ldu((unsigned char *) (src + i + 2 * ALTIVEC_LEN_FLOAT));
                src_tmp4 = (v4sf) vec_ldu((unsigned char *) (src + i + 3 * ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp1 = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
                src_tmp3 = vec_ld(0, src + i + 2 * ALTIVEC_LEN_FLOAT);
                src_tmp4 = vec_ld(0, src + i + 3 * ALTIVEC_LEN_FLOAT);
            }

            v4sf tmp1 = vec_mul(src_tmp1, scale_fact_vec);
            v4sf tmp2 = vec_mul(src_tmp2, scale_fact_vec);
            v4sf tmp3 = vec_mul(src_tmp3, scale_fact_vec);
            v4sf tmp4 = vec_mul(src_tmp4, scale_fact_vec);
            v4si tmp1_int = vec_cts(tmp1, 0);
            v4si tmp2_int = vec_cts(tmp2, 0);
            v4si tmp3_int = vec_cts(tmp3, 0);
            v4si tmp4_int = vec_cts(tmp4, 0);
            v8ss tmp5 = vec_packs(tmp1_int, tmp2_int);
            v8ss tmp6 = vec_packs(tmp3_int, tmp4_int);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &tmp5, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &tmp6, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(tmp5, 0, dst + i);
                vec_st(tmp6, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    if (rounding_mode == RndFinancial) {
        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (int16_t) (tmp > 32767.0f ? 32767.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (int16_t) (tmp > 32767.0f ? 32767.0f : tmp);
        }
        _FPU_SETCW(_mm_rounding_ori);  // restore previous rounding mode
        fesetround(rounding_ori);
    }
}

static inline void convertFloat32ToU16_128(float *src, uint16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (4 * ALTIVEC_LEN_FLOAT);
    stop_len *= (4 * ALTIVEC_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = vec_splats(scale_fact_mult);

    fpu_control_t _mm_rounding_ori;  // save rounding mode
    _FPU_GETCW(_mm_rounding_ori);

    int rounding_ori = fegetround();

    if (rounding_mode == RndZero) {
        _FPU_SETCW(_FPU_RC_ZERO | _FPU_DEFAULT);  // rounding_vec = ROUNDTOZERO;
        fesetround(FE_TOWARDZERO);
    } else if (rounding_mode == RndFinancial) {  // nothing to do, Default bankers rounding => round to nearest even
    } else {
        _FPU_SETCW(_FPU_RC_NEAREST | _FPU_DEFAULT);  // rounding_vec = ROUNDTONEAREST;
        fesetround(FE_TONEAREST);
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp1 = vec_ld(0, src + i);
            v4sf src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf src_tmp3 = vec_ld(0, src + i + 2 * ALTIVEC_LEN_FLOAT);
            v4sf src_tmp4 = vec_ld(0, src + i + 3 * ALTIVEC_LEN_FLOAT);
            v4sf tmp1 = vec_mul(src_tmp1, scale_fact_vec);
            v4sf tmp2 = vec_mul(src_tmp2, scale_fact_vec);
            v4sf tmp3 = vec_mul(src_tmp3, scale_fact_vec);
            v4sf tmp4 = vec_mul(src_tmp4, scale_fact_vec);
            v4si tmp1_int = vec_cts(tmp1, 0);
            v4si tmp2_int = vec_cts(tmp2, 0);
            v4si tmp3_int = vec_cts(tmp3, 0);
            v4si tmp4_int = vec_cts(tmp4, 0);
            v8us tmp5 = vec_packsu(tmp1_int, tmp2_int);
            v8us tmp6 = vec_packsu(tmp3_int, tmp4_int);
            vec_st(tmp5, 0, dst + i);
            vec_st(tmp6, 0, dst + i + ALTIVEC_LEN_INT16);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp1, src_tmp2, src_tmp3, src_tmp4;
            if (unalign_src) {
                src_tmp1 = (v4sf) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
                src_tmp3 = (v4sf) vec_ldu((unsigned char *) (src + i + 2 * ALTIVEC_LEN_FLOAT));
                src_tmp4 = (v4sf) vec_ldu((unsigned char *) (src + i + 3 * ALTIVEC_LEN_FLOAT));
            } else {
                src_tmp1 = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
                src_tmp3 = vec_ld(0, src + i + 2 * ALTIVEC_LEN_FLOAT);
                src_tmp4 = vec_ld(0, src + i + 3 * ALTIVEC_LEN_FLOAT);
            }

            v4sf tmp1 = vec_mul(src_tmp1, scale_fact_vec);
            v4sf tmp2 = vec_mul(src_tmp2, scale_fact_vec);
            v4sf tmp3 = vec_mul(src_tmp3, scale_fact_vec);
            v4sf tmp4 = vec_mul(src_tmp4, scale_fact_vec);
            v4si tmp1_int = vec_cts(tmp1, 0);
            v4si tmp2_int = vec_cts(tmp2, 0);
            v4si tmp3_int = vec_cts(tmp3, 0);
            v4si tmp4_int = vec_cts(tmp4, 0);
            v8us tmp5 = vec_packsu(tmp1_int, tmp2_int);
            v8us tmp6 = vec_packsu(tmp3_int, tmp4_int);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &tmp5, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &tmp6, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(tmp5, 0, dst + i);
                vec_st(tmp6, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    if (rounding_mode == RndFinancial) {
        for (int i = stop_len; i < len; i++) {
            float tmp = (roundf(src[i] * scale_fact_mult * 0.5f) / 2.0f);
            dst[i] = (uint16_t) (tmp > 65535.0f ? 65535.0f : tmp);  // round to nearest even with round(x/2)*2
        }
    } else {
        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = nearbyintf(src[i] * scale_fact_mult);
            dst[i] = (uint16_t) (tmp > 65535.0f ? 65535.0f : tmp);  // round to nearest even with round(x/2)*2
        }
        _FPU_SETCW(_mm_rounding_ori);  // restore previous rounding mode
        fesetround(rounding_ori);
    }
}

static inline void convertInt16ToFloat32_128(int16_t *src, float *dst, int len, int scale_factor)
{
    int stop_len = len / (2 * ALTIVEC_LEN_FLOAT);
    stop_len *= (2 * ALTIVEC_LEN_FLOAT);

    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    v4sf scale_fact_vec = vec_splats(scale_fact_mult);
    v4ui shift = vec_splats((unsigned int) 16);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v8ss vec = vec_ld(0, src + i);                 // loads 1 2 3 4 5 6 7 8 8
            v8ss low = vec_mergeh(vec, vec);               // low 1 1 2 2 3 3 4 4
            v8ss high = vec_mergel(vec, vec);              // high 5 5 6 6 7 7 8 8
            v4si lows = vec_sra(*(v4si *) &low, shift);    // make low 1 -1 2 -1 3 -1 4 -4
            v4si highs = vec_sra(*(v4si *) &high, shift);  // make high 5 -1 6 -1 7 -1 8 -1
            v4sf lowf = vec_ctf(*(v4si *) &lows, 0);
            v4sf highf = vec_ctf(*(v4si *) &highs, 0);

            // convert the vector to float and scale it
            v4sf floatlo = vec_mul(lowf, scale_fact_vec);
            v4sf floathi = vec_mul(highf, scale_fact_vec);

            vec_st(floatlo, 0, dst + i);
            vec_st(floathi, 0, dst + i + ALTIVEC_LEN_FLOAT);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v8ss vec;
            if (unalign_src) {
                vec = (v8ss) vec_ldu((unsigned char *) (src + i));
            } else {
                vec = vec_ld(0, src + i);
            }

            v8ss low = vec_mergeh(vec, vec);               // low 1 1 2 2 3 3 4 4
            v8ss high = vec_mergel(vec, vec);              // high 5 5 6 6 7 7 8 8
            v4si lows = vec_sra(*(v4si *) &low, shift);    // make low 1 -1 2 -1 3 -1 4 -4
            v4si highs = vec_sra(*(v4si *) &high, shift);  // make high 5 -1 6 -1 7 -1 8 -1
            v4sf lowf = vec_ctf(*(v4si *) &lows, 0);
            v4sf highf = vec_ctf(*(v4si *) &highs, 0);

            // convert the vector to float and scale it
            v4sf floatlo = vec_mul(lowf, scale_fact_vec);
            v4sf floathi = vec_mul(highf, scale_fact_vec);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &floatlo, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &floathi, (unsigned char *) (dst + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec_st(floatlo, 0, dst + i);
                vec_st(floathi, 0, dst + i + ALTIVEC_LEN_FLOAT);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}
#endif

static inline void pol2cart2D128f(float *r, float *theta, float *x, float *y, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), ALTIVEC_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf r_tmp = vec_ld(0, r + i);
            v4sf theta_tmp = vec_ld(0, theta + i);
            v4sf sin_tmp;
            v4sf cos_tmp;
            sincos_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v4sf x_tmp = vec_mul(r_tmp, cos_tmp);
            v4sf y_tmp = vec_mul(r_tmp, sin_tmp);
            vec_st(x_tmp, 0, x + i);
            vec_st(y_tmp, 0, y + i);
        }
    } else {
        int unalign_r = (uintptr_t) (r) % ALTIVEC_LEN_BYTES;
        int unalign_theta = (uintptr_t) (theta) % ALTIVEC_LEN_BYTES;
        int unalign_x = (uintptr_t) (x) % ALTIVEC_LEN_BYTES;
        int unalign_y = (uintptr_t) (y) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf r_tmp, theta_tmp;
            if (unalign_r) {
                r_tmp = (v4sf) vec_ldu((unsigned char *) (r + i));
            } else {
                r_tmp = vec_ld(0, r + i);
            }

            if (unalign_theta) {
                theta_tmp = (v4sf) vec_ldu((unsigned char *) (theta + i));
            } else {
                theta_tmp = vec_ld(0, theta + i);
            }

            v4sf sin_tmp;
            v4sf cos_tmp;
            sincos_ps(theta_tmp, &sin_tmp, &cos_tmp);
            v4sf x_tmp = vec_mul(r_tmp, cos_tmp);
            v4sf y_tmp = vec_mul(r_tmp, sin_tmp);

            if (unalign_x) {
                vec_stu(*(v16u8 *) &x_tmp, (unsigned char *) (x + i));
            } else {
                vec_st(x_tmp, 0, x + i);
            }

            if (unalign_y) {
                vec_stu(*(v16u8 *) &y_tmp, (unsigned char *) (y + i));
            } else {
                vec_st(y_tmp, 0, y + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        float sin_tmp, cos_tmp;
        mysincosf(theta[i], &sin_tmp, &cos_tmp);
        x[i] = r[i] * cos_tmp;
        y[i] = r[i] * sin_tmp;
    }
}

static inline void cart2pol2D128f(float *x, float *y, float *r, float *theta, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (r), (uintptr_t) (theta), ALTIVEC_LEN_BYTES) &&
        areAligned2((uintptr_t) (x), (uintptr_t) (y), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf x_tmp = vec_ld(0, x + i);
            v4sf y_tmp = vec_ld(0, y + i);
            v4sf y_square = vec_mul(y_tmp, y_tmp);
            v4sf r_tmp = vec_madd(x_tmp, x_tmp, y_square);
            r_tmp = vec_sqrt_precise(r_tmp);
            v4sf theta_tmp = atan2f_ps(y_tmp, x_tmp);
            vec_st(r_tmp, 0, r + i);
            vec_st(theta_tmp, 0, theta + i);
        }
    } else {
        int unalign_r = (uintptr_t) (r) % ALTIVEC_LEN_BYTES;
        int unalign_theta = (uintptr_t) (theta) % ALTIVEC_LEN_BYTES;
        int unalign_x = (uintptr_t) (x) % ALTIVEC_LEN_BYTES;
        int unalign_y = (uintptr_t) (y) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf x_tmp, y_tmp;
            if (unalign_x) {
                x_tmp = (v4sf) vec_ldu((unsigned char *) (x + i));
            } else {
                x_tmp = vec_ld(0, x + i);
            }

            if (unalign_y) {
                y_tmp = (v4sf) vec_ldu((unsigned char *) (y + i));
            } else {
                y_tmp = vec_ld(0, y + i);
            }

            v4sf y_square = vec_mul(y_tmp, y_tmp);
            v4sf r_tmp = vec_madd(x_tmp, x_tmp, y_square);
            r_tmp = vec_sqrt_precise(r_tmp);
            v4sf theta_tmp = atan2f_ps(y_tmp, x_tmp);

            if (unalign_r) {
                vec_stu(*(v16u8 *) &r_tmp, (unsigned char *) (r + i));
            } else {
                vec_st(r_tmp, 0, r + i);
            }

            if (unalign_theta) {
                vec_stu(*(v16u8 *) &theta_tmp, (unsigned char *) (theta + i));
            } else {
                vec_st(theta_tmp, 0, theta + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        r[i] = sqrtf(x[i] * x[i] + (y[i] * y[i]));
        theta[i] = atan2f(y[i], x[i]);
    }
}

static inline void sigmoid128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf tmp = vec_add(*(v4sf *) _ps_1, exp_ps_alternate(vec_xor(*(v4sf *) _ps_neg_sign_mask, src_tmp)));
            v4sf tmp2 = vec_div_precise(*(v4sf *) _ps_1, tmp);
            vec_st(tmp2, 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                src_tmp = vec_ld(0, src + i);
            }
            v4sf tmp = vec_add(*(v4sf *) _ps_1, exp_ps_alternate(vec_xor(*(v4sf *) _ps_neg_sign_mask, src_tmp)));
            v4sf tmp2 = vec_div_precise(*(v4sf *) _ps_1, tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &tmp2, (unsigned char *) (dst + i));
            } else {
                vec_st(tmp2, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}

static inline void softmax128f(float *src, float *dst, int len)
{
    int stop_len = len / (ALTIVEC_LEN_FLOAT);
    stop_len *= (ALTIVEC_LEN_FLOAT);

    __attribute__((aligned(ALTIVEC_LEN_BYTES))) float accumulate[ALTIVEC_LEN_FLOAT];
    float acc = 0.0f;

    v4sf vec_acc1 = *(v4sf *) _ps_0;  // initialize the vector accumulator

    int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp = vec_ld(0, src + i);
            v4sf dst_tmp = exp_ps_alternate(src_tmp);
            vec_acc1 = vec_add(vec_acc1, dst_tmp);
            vec_st(dst_tmp, 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf src_tmp;
            if (unalign_src) {
                src_tmp = (v4sf) vec_ldu((unsigned char *) (src + i));
            } else {
                src_tmp = vec_ld(0, src + i);
            }

            v4sf dst_tmp = exp_ps_alternate(src_tmp);
            vec_acc1 = vec_add(vec_acc1, dst_tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
            } else {
                vec_st(dst_tmp, 0, dst + i);
            }
        }
    }

    vec_st(vec_acc1, 0, accumulate);

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
        acc += dst[i];
    }

    acc = acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];
    vec_acc1 = vec_splats(acc);

    if (unalign_dst == 0) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf dst_tmp = vec_ld(0, dst + i);
            v4sf tmp = vec_div_precise(dst_tmp, vec_acc1);
            vec_st(tmp, 0, dst + i);
        }
    } else {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf dst_tmp = (v4sf) vec_ldu((unsigned char *) (dst + i));
            v4sf tmp = vec_div_precise(dst_tmp, vec_acc1);
            vec_stu(*(v16u8 *) &tmp, (unsigned char *) (dst + i));
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] /= acc;
    }
}
