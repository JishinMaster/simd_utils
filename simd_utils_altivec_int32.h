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


#if 0  // work in progress

// https://gist.github.com/cxd4/8137986
#define SWAP(d3, d2, d1, d0) ((d3 << 6) | (d2 << 4) | (d1 << 2) | (d0 << 0))

static v4si vec_mullo(v4si a, v4si b)
{
#if 0
    v8ss prod_m; /* alternating FFFFFFFF00000000FFFFFFFF00000000 */
    v8ss prod_n; /* alternating 00000000FFFFFFFF00000000FFFFFFFF */

    //static const v16u8 swap_mask = {0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13};
    //static const v16u8 swap_mask = {0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13};
    
    prod_n = (v8ss)vec_mulo(*(v8us*)&a, *(v8us*)&b);
    //a = vec_perm(a, SWAP(2,3,0,1)); /* old SWAP(3,2,1,0) */
    //b = vec_perm(b, SWAP(2,3,0,1)); /* old SWAP(3,2,1,0) */
    prod_m = (v8ss)vec_mule(*(v8ss*)&a, *(v8ss*)&b);
/*
 * prod_m = { a[0] * b[0], a[2] * b[2] }
 * prod_n = { a[1] * b[1], a[3] * b[3] }
 */

    a = vec_unpackh(prod_n, prod_m);
    a = vec_sll(a, 64/8);
    a = vec_srl(a, 64/8);
    b = vec_unpackl(prod_n, prod_m);
    b = vec_sll(b, 64/8);
    b = vec_or(b, a); /* Ans = (hi << 64) | (lo & 0x00000000FFFFFFFF) */
    return (b);
#else
    v4si a13 = vec_perm(a, a, 0xF5);                        // (-,a3,-,a1)
    v4si b13 = vec_perm(b, b, 0xF5);                        // (-,b3,-,b1)
    v4si prod02 = (v4si) vec_mule((v4ui) a, (v4ui) b);      // (-,a2*b2,-,a0*b0)
    v4si prod13 = (v4si) vec_mule((v4ui) a13, (v4ui) b13);  // (-,a3*b3,-,a1*b1)
    v4si prod01 = vec_mergeh(prod02, prod13);               // (-,-,a1*b1,a0*b0)
    v4si prod23 = vec_mergel(prod02, prod13);               // (-,-,a3*b3,a2*b2)
    v4si prod = vec_mergeh((__vector long long) prod01, (__vector long long) prod23);
    (, );  // (ab3,ab2,ab1,ab0)
#endif
}
#endif

static inline void mul128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src1_tmp = vec_ld(0, src1 + i);
            v4si src2_tmp = vec_ld(0, src2 + i);
            v4si src1_tmp2 = vec_ld(0, src1 + i + ALTIVEC_LEN_INT32);
            v4si src2_tmp2 = vec_ld(0, src2 + i + ALTIVEC_LEN_INT32);
            v4si tmp = vec_mullo(src1_tmp, src2_tmp);
            v4si tmp2 = vec_mullo(src1_tmp2, src2_tmp2);
            vec_st(tmp, 0, dst + i);
            vec_st(tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
        }
    } else {
        // TODO
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

#endif

static inline void copy128s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp = vec_ld(0, src + i);
            v4si src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            vec_st(src_tmp, 0, dst + i);
            vec_st(src_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            }
            if (unalign_dst) {
                vec_stu(*(v16u8 *) &src_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &src_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_INT32));
            } else {
                vec_st(src_tmp, 0, dst + i);
                vec_st(src_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i];
    }
}

static inline void add128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_INT32;
    stop_len *= ALTIVEC_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a = vec_ld(0, src1 + i);
            v4si b = vec_ld(0, src2 + i);
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
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a, b;
            if (unalign_src1) {
                a = (v4si) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4si) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4si c = vec_add(a, b);

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

static inline void addc128s(int32_t *src, int32_t value, int32_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_INT32;
    stop_len *= ALTIVEC_LEN_INT32;

    v4si b = vec_splats(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a = vec_ld(0, src + i);
            vec_st(vec_add(a, b), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a;
            if (unalign_src) {
                a = (v4si) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }

            v4si c = vec_add(a, b);

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

static inline void sub128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_INT32;
    stop_len *= ALTIVEC_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a = vec_ld(0, src1 + i);
            v4si b = vec_ld(0, src2 + i);
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
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a, b;
            if (unalign_src1) {
                a = (v4si) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4si) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4si c = vec_sub(a, b);

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

static inline void subc128s(int32_t *src, int32_t value, int32_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_INT32;
    stop_len *= ALTIVEC_LEN_INT32;

    v4si b = vec_splats(value);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a = vec_ld(0, src + i);
            vec_st(vec_sub(a, b), 0, dst + i);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a;
            if (unalign_src) {
                a = (v4si) vec_ldu((unsigned char *) (src + i));
            } else {
                a = vec_ld(0, src + i);
            }

            v4si c = vec_sub(a, b);

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

static inline void flip128s(int32_t *src, int32_t *dst, int len)
{
    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    int mini = ((len - 1) < (2 * ALTIVEC_LEN_INT32)) ? (len - 1) : (2 * ALTIVEC_LEN_INT32);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst + len - ALTIVEC_LEN_INT32), ALTIVEC_LEN_BYTES)) {
        for (int i = 2 * ALTIVEC_LEN_INT32; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp = vec_ld(0, src + i);  // load a,b,c,d
            v4si src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            v4si src_tmp_flip = vec_perm(src_tmp, src_tmp, flip_vector);  // rotate vec from abcd to bcba
            v4si src_tmp_flip2 = vec_perm(src_tmp2, src_tmp2, flip_vector);
            vec_st(src_tmp_flip, 0, dst + len - i - ALTIVEC_LEN_INT32);  // store the flipped vector
            vec_st(src_tmp_flip2, 0, dst + len - i - 2 * ALTIVEC_LEN_INT32);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 2 * ALTIVEC_LEN_INT32; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            }
            v4si src_tmp_flip = vec_perm(src_tmp, src_tmp, flip_vector);  // rotate vec from abcd to bcba
            v4si src_tmp_flip2 = vec_perm(src_tmp2, src_tmp2, flip_vector);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &src_tmp_flip, (unsigned char *) (dst + len - i - ALTIVEC_LEN_INT32));
                vec_stu(*(v16u8 *) &src_tmp_flip2, (unsigned char *) (dst + len - i - 2 * ALTIVEC_LEN_INT32));
            } else {
                vec_st(src_tmp_flip, 0, dst + len - i - ALTIVEC_LEN_INT32);
                vec_st(src_tmp_flip2, 0, dst + len - i - 2 * ALTIVEC_LEN_INT32);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void minevery128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_INT32;
    stop_len *= ALTIVEC_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a = vec_ld(0, src1 + i);
            v4si b = vec_ld(0, src2 + i);
            vec_st(vec_min(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a, b;
            if (unalign_src1) {
                a = (v4si) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4si) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4si c = vec_min(a, b);

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

static inline void maxevery128s(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_INT32;
    stop_len *= ALTIVEC_LEN_INT32;

    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a = vec_ld(0, src1 + i);
            v4si b = vec_ld(0, src2 + i);
            vec_st(vec_max(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT32) {
            v4si a, b;
            if (unalign_src1) {
                a = (v4si) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v4si) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v4si c = vec_max(a, b);

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

static inline void minmax128s(int32_t *src, int len, int32_t *min_value, int32_t *max_value)
{
    int stop_len = (len - ALTIVEC_LEN_INT32) / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    int32_t min_f[ALTIVEC_LEN_INT32] __attribute__((aligned(ALTIVEC_LEN_BYTES)));
    int32_t max_f[ALTIVEC_LEN_INT32] __attribute__((aligned(ALTIVEC_LEN_BYTES)));
    v4si max_v, min_v, max_v2, min_v2;
    v4si src_tmp, src_tmp2;

    int32_t min_tmp = src[0];
    int32_t max_tmp = src[0];

    if (len >= ALTIVEC_LEN_INT32) {
        if (isAligned((uintptr_t) (src), ALTIVEC_LEN_BYTES)) {
            src_tmp = vec_ld(0, src + 0);
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = ALTIVEC_LEN_INT32; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
                max_v = vec_max(max_v, src_tmp);
                min_v = vec_min(min_v, src_tmp);
                max_v2 = vec_max(max_v2, src_tmp2);
                min_v2 = vec_min(min_v2, src_tmp2);
            }
        } else {
            src_tmp = (v4si) vec_ldu((unsigned char *) (src + 0));
            max_v = src_tmp;
            min_v = src_tmp;
            max_v2 = src_tmp;
            min_v2 = src_tmp;

            for (int i = ALTIVEC_LEN_INT32; i < stop_len; i += ALTIVEC_LEN_INT32) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
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

static inline void threshold128_gt_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si tmp = {value, value, value, value};

    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp = vec_ld(0, src + i);
            v4si src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            v4si dst_tmp = vec_min(src_tmp, tmp);
            v4si dst_tmp2 = vec_min(src_tmp2, tmp);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            }
            v4si dst_tmp = vec_min(src_tmp, tmp);
            v4si dst_tmp2 = vec_min(src_tmp2, tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_INT32));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < value ? src[i] : value;
    }
}

static inline void threshold128_gtabs_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si pval = vec_splats(value);
    const v4si mval = vec_splats(-value);

    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp = vec_ld(0, src + i);
            v4si src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            v4si src_abs = vec_abs(src_tmp);  // take absolute value
            v4si src_abs2 = vec_abs(src_tmp2);
            v4ui eqmask = vec_cmpeq(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4ui eqmask2 = vec_cmpeq(src_abs2, src_tmp2);
            v4si max = vec_min(src_tmp, pval);
            v4si max2 = vec_min(src_tmp2, pval);
            v4si min = vec_max(src_tmp, mval);
            v4si min2 = vec_max(src_tmp2, mval);
            v4si dst_tmp = vec_sel(min, max, eqmask);
            v4si dst_tmp2 = vec_sel(min2, max2, eqmask2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            }

            v4si src_abs = vec_abs(src_tmp);  // take absolute value
            v4si src_abs2 = vec_abs(src_tmp2);
            v4ui eqmask = vec_cmpeq(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4ui eqmask2 = vec_cmpeq(src_abs2, src_tmp2);
            v4si max = vec_min(src_tmp, pval);
            v4si max2 = vec_min(src_tmp2, pval);
            v4si min = vec_max(src_tmp, mval);
            v4si min2 = vec_max(src_tmp2, mval);
            v4si dst_tmp = vec_sel(min, max, eqmask);
            v4si dst_tmp2 = vec_sel(min2, max2, eqmask2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_INT32));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
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

static inline void threshold128_lt_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si tmp = {value, value, value, value};

    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp = vec_ld(0, src + i);
            v4si src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            v4si dst_tmp = vec_max(src_tmp, tmp);
            v4si dst_tmp2 = vec_max(src_tmp2, tmp);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            }
            v4si dst_tmp = vec_max(src_tmp, tmp);
            v4si dst_tmp2 = vec_max(src_tmp2, tmp);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_INT32));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

static inline void threshold128_ltabs_s(int32_t *src, int32_t *dst, int len, int32_t value)
{
    const v4si pval = vec_splats(value);
    const v4si mval = vec_splats(-value);

    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp = vec_ld(0, src + i);
            v4si src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            v4si src_abs = vec_abs(src_tmp);  // take absolute value
            v4si src_abs2 = vec_abs(src_tmp2);
            v4ui eqmask = vec_cmpeq(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4ui eqmask2 = vec_cmpeq(src_abs2, src_tmp2);
            v4si max = vec_max(src_tmp, pval);
            v4si max2 = vec_max(src_tmp2, pval);
            v4si min = vec_min(src_tmp, mval);
            v4si min2 = vec_min(src_tmp2, mval);
            v4si dst_tmp = vec_sel(min, max, eqmask);
            v4si dst_tmp2 = vec_sel(min2, max2, eqmask2);
            vec_st(dst_tmp, 0, dst + i);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            }
            v4si src_abs = vec_abs(src_tmp);  // take absolute value
            v4si src_abs2 = vec_abs(src_tmp2);
            v4ui eqmask = vec_cmpeq(src_abs, src_tmp);  // if A = abs(A), then A is >= 0 (mask 0xFFFFFFFF)
            v4ui eqmask2 = vec_cmpeq(src_abs2, src_tmp2);
            v4si max = vec_max(src_tmp, pval);
            v4si max2 = vec_max(src_tmp2, pval);
            v4si min = vec_min(src_tmp, mval);
            v4si min2 = vec_min(src_tmp2, mval);
            v4si dst_tmp = vec_sel(min, max, eqmask);
            v4si dst_tmp2 = vec_sel(min2, max2, eqmask2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_INT32));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
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


static inline void threshold128_ltval_gtval_s(int32_t *src, int32_t *dst, int len, int32_t ltlevel, int32_t ltvalue, int32_t gtlevel, int32_t gtvalue)
{
    const v4si ltlevel_v = {ltlevel, ltlevel, ltlevel, ltlevel};
    const v4si ltvalue_v = {ltvalue, ltvalue, ltvalue, ltvalue};
    const v4si gtlevel_v = {gtlevel, gtlevel, gtlevel, gtlevel};
    const v4si gtvalue_v = {gtvalue, gtvalue, gtvalue, gtvalue};

    int stop_len = len / (2 * ALTIVEC_LEN_INT32);
    stop_len *= (2 * ALTIVEC_LEN_INT32);

    if (areAligned2((uintptr_t) (src), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp = vec_ld(0, src + i);
            v4si src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            __vector bool int lt_mask = vec_cmplt(src_tmp, ltlevel_v);
            __vector bool int gt_mask = vec_cmpgt(src_tmp, gtlevel_v);
            v4si dst_tmp = vec_sel(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = vec_sel(dst_tmp, gtvalue_v, gt_mask);
            vec_st(dst_tmp, 0, dst + i);
            __vector bool int lt_mask2 = vec_cmplt(src_tmp2, ltlevel_v);
            __vector bool int gt_mask2 = vec_cmpgt(src_tmp2, gtlevel_v);
            v4si dst_tmp2 = vec_sel(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = vec_sel(dst_tmp2, gtvalue_v, gt_mask2);
            vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
        }
    } else {
        int unalign_src = (uintptr_t) (src) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        for (int i = 0; i < stop_len; i += 2 * ALTIVEC_LEN_INT32) {
            v4si src_tmp, src_tmp2;
            if (unalign_src) {
                src_tmp = (v4si) vec_ldu((unsigned char *) (src + i));
                src_tmp2 = (v4si) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT32));
            } else {
                src_tmp = vec_ld(0, src + i);
                src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT32);
            }
            __vector bool int lt_mask = vec_cmplt(src_tmp, ltlevel_v);
            __vector bool int gt_mask = vec_cmpgt(src_tmp, gtlevel_v);
            v4si dst_tmp = vec_sel(src_tmp, ltvalue_v, lt_mask);
            dst_tmp = vec_sel(dst_tmp, gtvalue_v, gt_mask);
            vec_st(dst_tmp, 0, dst + i);
            __vector bool int lt_mask2 = vec_cmplt(src_tmp2, ltlevel_v);
            __vector bool int gt_mask2 = vec_cmpgt(src_tmp2, gtlevel_v);
            v4si dst_tmp2 = vec_sel(src_tmp2, ltvalue_v, lt_mask2);
            dst_tmp2 = vec_sel(dst_tmp2, gtvalue_v, gt_mask2);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &dst_tmp, (unsigned char *) (dst + i));
                vec_stu(*(v16u8 *) &dst_tmp2, (unsigned char *) (dst + i + ALTIVEC_LEN_INT32));
            } else {
                vec_st(dst_tmp, 0, dst + i);
                vec_st(dst_tmp2, 0, dst + i + ALTIVEC_LEN_INT32);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] < ltlevel ? ltvalue : src[i];
        dst[i] = src[i] > gtlevel ? gtvalue : dst[i];
    }
}

static inline v8ss vec_absdiff(v8ss a, v8ss b)
{
    v16u8 cmp;
    v8ss difab, difba;
    cmp = vec_cmpgt(a, b);
    difab = vec_sub(a, b);
    difba = vec_sub(b, a);
#if 1  // should be faster
    return vec_sel(difba, difab, cmp);
#else
    difab = vec_and(*(v8ss *) &cmp, difab);
    difba = vec_andc(difba, *(v8ss *) &cmp);
    return vec_or(difab, difba);
#endif
}

static inline void absdiff16s_128s(int16_t *src1, int16_t *src2, int16_t *dst, int len)
{
    int stop_len = len / ALTIVEC_LEN_INT16;
    stop_len *= ALTIVEC_LEN_INT16;


    if (areAligned3((uintptr_t) (src1), (uintptr_t) (src2), (uintptr_t) (dst), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT16) {
            v8ss a = vec_ld(0, src1 + i);
            v8ss b = vec_ld(0, src2 + i);
            vec_st(vec_absdiff(a, b), 0, dst + i);
        }
    } else {
        int unalign_src1 = (uintptr_t) (src1) % ALTIVEC_LEN_BYTES;
        int unalign_src2 = (uintptr_t) (src2) % ALTIVEC_LEN_BYTES;
        int unalign_dst = (uintptr_t) (dst) % ALTIVEC_LEN_BYTES;

        /*To be improved : we constantly use unaligned load or store of those data
        There exist better unaligned stream load or store which could improve performance
        */
        // The following loop relies on good branch prediction architecture
        for (int i = 0; i < stop_len; i += ALTIVEC_LEN_INT16) {
            v8ss a, b;
            if (unalign_src1) {
                a = (v8ss) vec_ldu((unsigned char *) (src1 + i));
            } else {
                a = vec_ld(0, src1 + i);
            }
            if (unalign_src2) {
                b = (v8ss) vec_ldu((unsigned char *) (src2 + i));
            } else {
                b = vec_ld(0, src2 + i);
            }
            v8ss c = vec_absdiff(a, b);

            if (unalign_dst) {
                vec_stu(*(v16u8 *) &c, (unsigned char *) (dst + i));
            } else {
                vec_st(c, 0, dst + i);
            }
        }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = abs(src1[i] - src2[i]);
    }
}

// Works with positive scale_factor (divides final value)
static inline void sum16s32s128(int16_t *src, int len, int32_t *dst, int scale_factor)
{
    int stop_len = len / (4 * ALTIVEC_LEN_INT16);
    stop_len *= (4 * ALTIVEC_LEN_INT16);

    __attribute__((aligned(ALTIVEC_LEN_BYTES))) int32_t accumulate[ALTIVEC_LEN_INT32];
    int32_t tmp_acc = 0;
    int16_t scale = 1 << scale_factor;
    v8ss one = vec_splats(1);
    v4si vec_acc1 = *(v4si *) _ps_0;  // initialize the vector accumulator
    v4si vec_acc2 = *(v4si *) _ps_0;  // initialize the vector accumulator

    if (isAligned((uintptr_t) (src), ALTIVEC_LEN_BYTES)) {
        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_INT16) {
            v8ss vec_src_tmp = vec_ld(0, src + i);
            v8ss vec_src_tmp2 = vec_ld(0, src + i + ALTIVEC_LEN_INT16);
            v8ss vec_src_tmp3 = vec_ld(0, src + i + 2 * ALTIVEC_LEN_INT16);
            v8ss vec_src_tmp4 = vec_ld(0, src + i + 3 * ALTIVEC_LEN_INT16);
            v4si vec_src_tmpi = vec_msum(vec_src_tmp, one, *(v4si *) _ps_0);
            v4si vec_src_tmp2i = vec_msum(vec_src_tmp2, one, *(v4si *) _ps_0);
            v4si vec_src_tmp3i = vec_msum(vec_src_tmp3, one, *(v4si *) _ps_0);
            v4si vec_src_tmp4i = vec_msum(vec_src_tmp4, one, *(v4si *) _ps_0);
            vec_src_tmpi = vec_add(vec_src_tmpi, vec_src_tmp2i);
            vec_src_tmp3i = vec_add(vec_src_tmp3i, vec_src_tmp4i);
            vec_acc1 = vec_add(vec_src_tmpi, vec_acc1);
            vec_acc2 = vec_add(vec_src_tmp3i, vec_acc2);
        }
    } else {
        for (int i = 0; i < stop_len; i += 4 * ALTIVEC_LEN_INT16) {
            v8ss vec_src_tmp = (v8ss) vec_ldu((unsigned char *) (src + i));
            v8ss vec_src_tmp2 = (v8ss) vec_ldu((unsigned char *) (src + i + ALTIVEC_LEN_INT16));
            v8ss vec_src_tmp3 = (v8ss) vec_ldu((unsigned char *) (src + i + 2 * ALTIVEC_LEN_INT16));
            v8ss vec_src_tmp4 = (v8ss) vec_ldu((unsigned char *) (src + i + 3 * ALTIVEC_LEN_INT16));
            v4si vec_src_tmpi = vec_msum(vec_src_tmp, one, *(v4si *) _ps_0);
            v4si vec_src_tmp2i = vec_msum(vec_src_tmp2, one, *(v4si *) _ps_0);
            v4si vec_src_tmp3i = vec_msum(vec_src_tmp3, one, *(v4si *) _ps_0);
            v4si vec_src_tmp4i = vec_msum(vec_src_tmp4, one, *(v4si *) _ps_0);
            vec_src_tmpi = vec_add(vec_src_tmpi, vec_src_tmp2i);
            vec_src_tmp3i = vec_add(vec_src_tmp3i, vec_src_tmp4i);
            vec_acc1 = vec_add(vec_src_tmpi, vec_acc1);
            vec_acc2 = vec_add(vec_src_tmp3i, vec_acc2);
        }
    }

    vec_acc1 = vec_add(vec_acc1, vec_acc2);
    vec_st(vec_acc1, 0, accumulate);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += (int32_t) src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];

    tmp_acc /= scale;
    *dst = tmp_acc;
}
