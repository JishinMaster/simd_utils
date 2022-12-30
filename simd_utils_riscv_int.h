/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <fenv.h>
#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline void copys_vec(int32_t *src, int32_t *dst, int len)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        VSTORE_INT(dst_tmp, VLOAD_INT(src_tmp, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void subs_vec(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    size_t i;
    int32_t *src1_tmp = src1;
    int32_t *src2_tmp = src2;
    int32_t *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLOAD_INT(src1_tmp, i);
        vb = VLOAD_INT(src2_tmp, i);

        VSTORE_INT(dst_tmp, VSUB_INT(va, vb, i), i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void adds_vec(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    size_t i;
    int32_t *src1_tmp = src1;
    int32_t *src2_tmp = src2;
    int32_t *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLOAD_INT(src1_tmp, i);
        vb = VLOAD_INT(src2_tmp, i);

        VSTORE_INT(dst_tmp, VADD_INT(va, vb, i), i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void addcs_vec(int32_t *src, int32_t value, int32_t *dst, int len)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va;
        va = VLOAD_INT(src_tmp, i);

        VSTORE_INT(dst_tmp, VADD1_INT(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void muls_vec(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    size_t i;
    int32_t *src1_tmp = src1;
    int32_t *src2_tmp = src2;
    int32_t *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLOAD_INT(src1_tmp, i);
        vb = VLOAD_INT(src2_tmp, i);

        VSTORE_INT(dst_tmp, VMUL_INT(va, vb, i), i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulcs_vec(int32_t *src, int32_t value, int32_t *dst, int len)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va;
        va = VLOAD_INT(src_tmp, i);

        VSTORE_INT(dst_tmp, VMUL1_INT(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void vectorSlopes_vec(int32_t *dst, int len, int32_t offset, int32_t slope)
{
    size_t i;
    int32_t *dst_tmp = dst;

    int32_t coef_max[MAX_ELTS32];

    // to be improved!
    for (int s = 0; s < MAX_ELTS32; s++) {
        coef_max[s] = (int32_t) (s) *slope;
    }

    i = VSETVL32(len);

    V_ELT_INT coef = VLOAD_INT(coef_max, i);
    V_ELT_INT slope_vec = VLOAD1_INT((int32_t) (i) *slope, i);
    V_ELT_INT curVal = VADD1_INT(coef, offset, i);

    VSTORE_INT(dst_tmp, curVal, i);
    dst_tmp += i;
    len -= i;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        curVal = VADD_INT(curVal, slope_vec, i);
        VSTORE_INT(dst_tmp, curVal, i);
        dst_tmp += i;
    }
}

static inline void threshold_gt_s_vec(int32_t *src, int32_t *dst, int len, float value)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va;
        va = VLOAD_INT(src_tmp, i);
        VSTORE_INT(dst_tmp, VMIN1_INT(va, value, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_lt_s_vec(int32_t *src, int32_t *dst, int len, float value)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va;
        va = VLOAD_INT(src_tmp, i);
        VSTORE_INT(dst_tmp, VMAX1_INT(va, value, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_gtabs_s_vec(int32_t *src, int32_t *dst, int len, int32_t value)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT src_tmp_vec = VLOAD_INT(src_tmp, i);
        V_ELT_BOOL srcsup0 = VGT1_INT_BOOL(src_tmp_vec, 0, i);
        V_ELT_INT src_tmp_neg_vec = VNEG_INT(src_tmp_vec, i);
        V_ELT_INT src_abs = VMERGE_INT(srcsup0, src_tmp_vec, src_tmp_neg_vec, i);
        V_ELT_BOOL eqmask = VEQ_INT_BOOL(src_abs, src_tmp_vec, i);
        V_ELT_INT min = VMIN1_INT(src_tmp_vec, value, i);
        V_ELT_INT max = VMAX1_INT(src_tmp_vec, -value, i);
        V_ELT_INT dst_tmp_vec = VMERGE_INT(eqmask, min, max, i);
        VSTORE_INT(dst_tmp, dst_tmp_vec, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_ltabs_s_vec(int32_t *src, int32_t *dst, int len, int32_t value)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT src_tmp_vec = VLOAD_INT(src_tmp, i);
        V_ELT_BOOL srcsup0 = VGT1_INT_BOOL(src_tmp_vec, 0, i);
        V_ELT_INT src_tmp_neg_vec = VNEG_INT(src_tmp_vec, i);
        V_ELT_INT src_abs = VMERGE_INT(srcsup0, src_tmp_vec, src_tmp_neg_vec, i);
        V_ELT_BOOL eqmask = VEQ_INT_BOOL(src_abs, src_tmp_vec, i);
        V_ELT_INT max = VMAX1_INT(src_tmp_vec, value, i);
        V_ELT_INT min = VMIN1_INT(src_tmp_vec, -value, i);
        V_ELT_INT dst_tmp_vec = VMERGE_INT(eqmask, max, min, i);
        VSTORE_INT(dst_tmp, dst_tmp_vec, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void minmaxs_vec(int32_t *src, int len, int32_t *min_value, int32_t *max_value)
{
    size_t i, i_last;

    int32_t *src_tmp = src;

    i = VSETVL32(len);

    vint32m1_t min0 = vle32_v_i32m1(src_tmp, 1);  // or vfmv_v_f_f32m1
    vint32m1_t max0 = min0;
    V_ELT_INT minv, maxv, v1;
    minv = VLOAD_INT(src_tmp, i);
    maxv = minv;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        v1 = VLOAD_INT(src_tmp, i);
        minv = VMIN_INT(v1, minv, i);
        maxv = VMAX_INT(v1, maxv, i);
        src_tmp += i;
        i_last = i;
    }
    min0 = VREDMIN_INT(min0, minv, min0, i_last);
    max0 = VREDMAX_INT(max0, maxv, max0, i_last);
    vse32_v_i32m1(min_value, min0, 1);
    vse32_v_i32m1(max_value, max0, 1);
}

static inline void maxeverys_vec(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    size_t i;
    int32_t *src1_tmp = src1;
    int32_t *src2_tmp = src2;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLOAD_INT(src1_tmp, i);
        vb = VLOAD_INT(src2_tmp, i);
        VSTORE_INT(dst_tmp, VMAX_INT(va, vb, i), i);

        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void mineverys_vec(int32_t *src1, int32_t *src2, int32_t *dst, int len)
{
    size_t i;
    int32_t *src1_tmp = src1;
    int32_t *src2_tmp = src2;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLOAD_INT(src1_tmp, i);
        vb = VLOAD_INT(src2_tmp, i);
        VSTORE_INT(dst_tmp, VMIN_INT(va, vb, i), i);

        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_ltval_gtval_s_vec(int32_t *src, int32_t *dst, int len, int32_t ltlevel, int32_t ltvalue, int32_t gtlevel, int32_t gtvalue)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_INT va = VLOAD_INT(src_tmp, i);
        V_ELT_BOOL lt_mask = VLT1_INT_BOOL(va, ltlevel, i);
        V_ELT_BOOL gt_mask = VGT1_INT_BOOL(va, gtlevel, i);
        V_ELT_INT tmp = VMERGE1_INT(lt_mask, va, ltvalue, i);
        tmp = VMERGE1_INT(gt_mask, tmp, gtvalue, i);
        VSTORE_INT(dst_tmp, tmp, i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void flips_vec(int32_t *src, int32_t *dst, int len)
{
    size_t i;
    i = VSETVL32(len);
    int vec_size = VSETVL32(4096);
    int32_t *src_tmp = src + len - i;
    int32_t *dst_tmp = dst;

#if MAX_ELTS32 == 32
    // max vector size is 1024bits, but could be less (128bits on C906 core)
    uint32_t index[32] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
                          19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                          6, 5, 4, 3, 2, 1, 0};
#else
    uint32_t index[MAX_ELTS32];
    for (int l = 0; l < MAX_ELTS32; l++)
        index[l] = MAX_ELTS32 - l - 1;
#endif

    V_ELT_UINT index_vec = VLOAD_UINT(index + MAX_ELTS32 - vec_size, i);
    V_ELT_INT a, b;
    for (; (i = VSETVL32(len)) >= vec_size; len -= i) {
        a = VLOAD_INT(src_tmp, i);
        b = VGATHER_INT(a, index_vec, i);
        VSTORE_INT(dst_tmp, b, i);
        src_tmp -= i;
        dst_tmp += i;
    }

    if (i) {
        src_tmp = src;
        index_vec = VLOAD_UINT(index + MAX_ELTS32 - i, i);
        print_vec_uint(index_vec, i);
        a = VLOAD_INT(src_tmp, i);
        b = VGATHER_INT(a, index_vec, i);
        VSTORE_INT(dst_tmp, b, i);
    }
}
