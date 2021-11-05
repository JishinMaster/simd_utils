/*
 * Project : SIMD_Utils
 * Version : 0.2.0
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <altivec.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

//Compare and perm operations => perm unit
// On e6500, VPERM operations take 2 cycles. VFPU operations take 6 cycles.
// Complex FPU operations take 7 cycles (and block the unit for 2 cycles)

//use pointer dereferencing to make it generic?
static inline v16u8 vec_ldu(unsigned char *v)
{
    v16u8 permute = vec_lvsl(0, v);
    v16u8 MSQ = vec_ld(0, v);
    v16u8 LSQ = vec_ld(16, v);
    return vec_perm(MSQ, LSQ, permute);
}

/// From http://mirror.informatimago.com/next/developer.apple.com/hardware/ve/alignment.html
static inline void vec_stu(v16u8 src, unsigned char *target)
{
    v16u8 MSQ, LSQ, result;
    v16u8 mask, align;

    MSQ = vec_ld(0, target);                                        // most significant quadword
    LSQ = vec_ld(16, target);                                       // least significant quadword
    align = vec_lvsr(0, target);                                    // create alignment vector
    mask = vec_perm(*(v16u8 *) _pi8_0, *(v16s8 *) _pi8_ff, align);  // Create select mask
    src = vec_perm(src, src, align);                                // Right rotate stored data
    MSQ = vec_sel(MSQ, src, mask);                                  // Insert data into MSQ part
    LSQ = vec_sel(src, LSQ, mask);                                  // Insert data into LSQ part
    vec_st(MSQ, 0, target);                                         // Store the MSQ part
    vec_st(LSQ, 16, target);                                        // Store the LSQ part
}

//In Altivec there is only mad and not mul
static inline v4sf vec_mul(v4sf a, v4sf b)
{
    return vec_madd(a, b, *(v4sf *) _ps_0);
}

//Useful link : http://mirror.informatimago.com/next/developer.apple.com/hardware/ve/algorithms.html

//In Altivec there is no div, hence a/b = a*(1/b)
static inline v4sf vec_div(v4sf a, v4sf b)
{
    return vec_mul(a, vec_re(b));
}

static inline v4sf vec_div_precise(v4sf a, v4sf b)
{
    //Get the reciprocal estimate
    v4sf estimate = vec_re(b);

    //One round of Newton-Raphson refinement
    v4sf re = vec_madd(vec_nmsub(estimate, b, *(v4sf *) _ps_1), estimate, estimate);
    return vec_mul(a, re);
}


/*static inline void print4(v4sf v)
{
    float *p = (float *) &v;
    printf("[%3.24g, %3.24g, %3.24g, %3.24g]", p[0], p[1], p[2], p[3]);
}*/

//In Altivec there is no sqrt, hence sqrt(a)= a*rsqrt(a)
static inline v4sf vec_sqrt(v4sf a)
{
#if 1  //Add a quantum so that sqrt(0) = 0 and not NaN
    const v4sf quantum = {1.180E-38, 1.180E-38, 1.180E-38, 1.180E-38};
    a = vec_add(a, quantum);
#endif
    return vec_mul(a, vec_rsqrte(a));
}

static inline v4sf vec_sqrt_precise(v4sf a)
{
#if 1  //Add a quantum so that sqrt(0) = 0 and not NaN
    const v4sf quantum = {1.180E-38, 1.180E-38, 1.180E-38, 1.180E-38};
    a = vec_add(a, quantum);
#endif
    //Get the square root reciprocal estimate
    v4sf estimate = vec_rsqrte(a);

    //One round of Newton-Raphson refinement
    v4sf estimateSquared = vec_madd(estimate, estimate, *(v4sf *) _ps_0);
    v4sf halfEstimate = vec_madd(estimate, *(v4sf *) _ps_0p5, *(v4sf *) _ps_0);
    v4sf re = vec_madd(vec_nmsub(a, estimateSquared, *(v4sf *) _ps_1), halfEstimate, estimate);

    return vec_mul(a, re);
}


static inline v4sf vec_set1_ps(float value)
{
    v4sf a = {value, value, value, value};
    return a;
}

static inline void set128f(float *dst, float value, int len)
{
    v4sf tmp = vec_set1_ps(value);

    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (isAligned((uintptr_t)(dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            vec_st(tmp, 0, dst + i);
        }
    } else {
        int unaligned_float = (uintptr_t)(dst) % ALTIVEC_LEN_FLOAT;  //could this happen though?
        if (unaligned_float == 0) {                                  //dst is not aligned on 16bytes boundary but is at least aligned on float
            int unaligned_elts = ((uintptr_t)(dst) % ALTIVEC_LEN_BYTES) / sizeof(float);
            for (int i = 0; i < unaligned_elts; i++) {
                dst[i] = value;
            }
            for (int i = unaligned_elts + 1; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
                vec_st(tmp, 0, dst + i);
            }
        } else {  //do not use SIMD in this case, skip to scalar part
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

static inline void mul128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_mul(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t)(src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t)(src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t)(dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        //The following loop relies on good branch prediction architecture
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

static inline void minevery128f(float *src1, float *src2, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t)(src1), (uintptr_t)(src2), (uintptr_t)(dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src1 + i);
            v4sf b = vec_ld(0, src2 + i);
            vec_st(vec_min(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t)(src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t)(src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t)(dst) % ALTIVEC_LEN_BYTES;

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

// converts 32bits complex float to two arrays real and im
static inline void cplxtoreal128f(float *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (ALTIVEC_LEN_FLOAT);
    stop_len *= ALTIVEC_LEN_FLOAT;
    int j = 0;

    const v16u8 re_mask = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
    const v16u8 im_mask = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

    if (areAligned3((uintptr_t)(src), (uintptr_t)(dstRe), (uintptr_t)(dstIm), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf vec1 = vec_ld(0, src + i);
            v4sf vec2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            v4sf re = vec_perm(vec1, vec2, re_mask);
            v4sf im = vec_perm(vec1, vec2, im_mask);
            vec_st(re, 0, dstRe + j);
            vec_st(im, 0, dstIm + j);
            j += ALTIVEC_LEN_FLOAT;
        }
    } else {
        int unalign_src = (uintptr_t)(src) % ALTIVEC_LEN_BYTES;
        int unalign_dstRe = (uintptr_t)(dstRe) % ALTIVEC_LEN_BYTES;
        int unalign_dstIm = (uintptr_t)(dstIm) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_FLOAT) {
            v4sf vec1, vec2;

            if (unalign_src) {
                vec1 = (v4sf) vec_ldu((unsigned char *) (src + i));
                vec2 = (v4sf) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_FLOAT));
            } else {
                vec1 = vec_ld(0, src + i);
                vec2 = vec_ld(0, src + i + ALTIVEC_LEN_FLOAT);
            }

            v4sf re = vec_perm(vec1, vec2, re_mask);
            v4sf im = vec_perm(vec1, vec2, im_mask);

            if (unalign_dstRe) {
                vec_stu(*(v16u8 *) &re, (unsigned char *) (dstRe + i));
            } else {
                vec_st(re, 0, dstRe + j);
            }

            if (unalign_dstIm) {
                vec_stu(*(v16u8 *) &im, (unsigned char *) (dstIm + i));
            } else {
                vec_st(im, 0, dstIm + j);
            }
            j += ALTIVEC_LEN_FLOAT;
        }
    }

    for (int i = stop_len; i < 2 * len; i += 2) {
        dstRe[j] = src[i];
        dstIm[j] = src[i + 1];
        j++;
    }
}

static inline void log2_128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_loge(a), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t)(src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t)(dst) % ALTIVEC_LEN_BYTES;

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

static inline void ln_128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    const v4sf ln2_vec = {LN2, LN2, LN2, LN2};

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_mul(vec_loge(a), ln2_vec), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t)(src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t)(dst) % ALTIVEC_LEN_BYTES;

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
        dst[i] = log2f(src[i]);
    }
}

static inline void log10_128f(float *src, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    const v4sf ln2_ln10_vec = {LN2_DIV_LN10, LN2_DIV_LN10, LN2_DIV_LN10, LN2_DIV_LN10};

    if (areAligned2((uintptr_t)(src), (uintptr_t)(dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf a = vec_ld(0, src + i);
            vec_st(vec_mul(vec_loge(a), ln2_ln10_vec), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t)(src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t)(dst) % ALTIVEC_LEN_BYTES;

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
        dst[i] = log2f(src[i]);
    }
}

static inline void magnitude128f_split(float *srcRe, float *srcIm, float *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_FLOAT;
    stop_len *= ALTIVEC_LEN_FLOAT;

    if (areAligned3((uintptr_t)(srcRe), (uintptr_t)(srcIm), (uintptr_t)(dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_FLOAT) {
            v4sf re_tmp = vec_ld(0, srcRe + i);
            v4sf re2 = vec_mul(re_tmp, re_tmp);
            v4sf im_tmp = vec_ld(0, srcIm + i);
            v4sf im2 = vec_mul(im_tmp, im_tmp);
            vec_st(vec_sqrt(vec_add(re2, im2)), 0, dst + i);
        }
    } else {
        int unalign_srcRe = (uintptr_t)(srcRe) % ALTIVEC_LEN_BYTES;
        int unalign_srcIm = (uintptr_t)(srcRe) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t)(dst) % ALTIVEC_LEN_BYTES;

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
        dst[i] = sqrtf(srcRe[i] * srcRe[i] + srcIm[i] * srcIm[i]);
    }
}



//Work in progress
#if 0
static inline void print16(v16u8 v)
{
    unsigned char *p = (unsigned char *) &v;
    printf("[%x, %x, %x, %x, %x, %x, %x, %x,%x, %x, %x, %x, %x, %x, %x, %x]\n",\
     p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],\
     p[12], p[13], p[14], p[15]);
    //printf("[%3.24g, %3.24g, %3.24g, %3.24g]", p[0], p[1], p[2], p[3]);
}

//Best would be vec_sel(a,b,mask)?
static inline v16u8 vec_blend(v16u8 a, v16u8 b, v16u8 mask)
{
    v16u8 b_tmp = vec_and(b, mask);
    v16u8 a_tmp = vec_and(a, vec_cmpeq(mask,*(v16u8 *) _pi8_0));  
    return vec_or(a_tmp, b_tmp);
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
            v4sf tmp = vec_mul(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)

            // if x > 0
            //vec_cmpgt(src_tmp, zero)
            __vector bool cmp = vec_cmpgt(src_tmp, zero);
            v16u8 blend_res = vec_blend(*(v16u8*)&tmp, *(v16u8*)&src_tmp, *(v16u8*)&cmp);
            vec_st(*(v4sf*)&blend_res, 0, dst + i);
        }
    } else {
        /*for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
            v4sf src_tmp = _mm_loadu_ps(src + i);
            v4sf tmp = _mm_mul_ps(alpha_vec, src_tmp);  // tmp = a*x (used when x < 0)

            // if x > 0
            _mm_storeu_ps(dst + i, _mm_blendv_ps(tmp, src_tmp, _mm_cmpgt_ps(src_tmp, zero)));
        }*/
    }

    for (int i = stop_len; i < len; i++) {
        if (src[i] > 0.0f)
            dst[i] = src[i];
        else
            dst[i] = alpha * src[i];
    }
}
#endif
