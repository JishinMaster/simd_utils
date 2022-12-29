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

    int32_t coef_max[32];

    // to be improved!
    for (int s = 0; s < 32; s++) {
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
