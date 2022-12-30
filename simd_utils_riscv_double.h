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

#if ELEN >= 64

static inline void roundd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;
    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE a, b;
        a = VLOAD_DOUBLE(src_tmp, i);
        b = VCVT_INT_DOUBLE(VCVT_DOUBLE_INT(a, i), i);
        VSTORE_DOUBLE(dst_tmp, b, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void ceild_vec(double *src, double *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    roundd_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void floord_vec(double *src, double *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    roundd_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void truncd_vec(double *src, double *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    roundd_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

void addd_vec(double *a, double *b, double *c, int len)
{
    size_t i;
    double *a_tmp = a;
    double *b_tmp = b;
    double *c_tmp = c;
    V_ELT_DOUBLE va, vb, vc;
    for (; (i = VSETVL64(len)) > 0; len -= i) {
        va = VLOAD_DOUBLE(a_tmp, i);
        vb = VLOAD_DOUBLE(b_tmp, i);
        vc = VADD_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void addcd_vec(double *src, double value, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;
    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLOAD_DOUBLE(src_tmp, i);

        VSTORE_DOUBLE(dst_tmp, VADD1_DOUBLE(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void muld_vec(double *a, double *b, double *c, int len)
{
    size_t i;
    double *a_tmp = a;
    double *b_tmp = b;
    double *c_tmp = c;
    V_ELT_DOUBLE va, vb, vc;
    for (; (i = VSETVL64(len)) > 0; len -= i) {
        va = VLOAD_DOUBLE(a_tmp, i);
        vb = VLOAD_DOUBLE(b_tmp, i);
        vc = VMUL_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void divd_vec(double *a, double *b, double *c, int len)
{
    size_t i;
    double *a_tmp = a;
    double *b_tmp = b;
    double *c_tmp = c;
    V_ELT_DOUBLE va, vb, vc;
    for (; (i = VSETVL64(len)) > 0; len -= i) {
        va = VLOAD_DOUBLE(a_tmp, i);
        vb = VLOAD_DOUBLE(b_tmp, i);
        vc = VDIV_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void subd_vec(double *a, double *b, double *c, int len)
{
    size_t i;
    double *a_tmp = a;
    double *b_tmp = b;
    double *c_tmp = c;
    V_ELT_DOUBLE va, vb, vc;
    for (; (i = VSETVL64(len)) > 0; len -= i) {
        va = VLOAD_DOUBLE(a_tmp, i);
        vb = VLOAD_DOUBLE(b_tmp, i);
        vc = VSUB_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void muladdd_vec(double *a, double *b, double *c, double *dst, int len)
{
    size_t i;
    double *a_tmp = a;
    double *b_tmp = b;
    double *c_tmp = c;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vb, vc;
        va = VLOAD_DOUBLE(a_tmp, i);
        vb = VLOAD_DOUBLE(b_tmp, i);
        vc = VLOAD_DOUBLE(c_tmp, i);
        vc = VFMA_DOUBLE(vc, va, vb, i);
        VSTORE_DOUBLE(dst_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulcaddd_vec(double *a, double b, double *c, double *dst, int len)
{
    size_t i;
    double *a_tmp = a;
    double *c_tmp = c;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vc;
        va = VLOAD_DOUBLE(a_tmp, i);
        vc = VLOAD_DOUBLE(c_tmp, i);
        vc = VFMA1_DOUBLE(vc, b, va, i);
        VSTORE_DOUBLE(dst_tmp, vc, i);

        a_tmp += i;
        c_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulcaddcd_vec(double *a, double b, double c, double *dst, int len)
{
    size_t i;
    double *a_tmp = a;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vc;
        va = VLOAD_DOUBLE(a_tmp, i);
        vc = VLOAD1_DOUBLE(c, i);
        vc = VFMA1_DOUBLE(vc, b, va, i);
        VSTORE_DOUBLE(dst_tmp, vc, i);

        a_tmp += i;
        dst_tmp += i;
    }
}

static inline void muladdcd_vec(double *a, double *b, double c, double *dst, int len)
{
    size_t i;
    double *a_tmp = a;
    double *b_tmp = b;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vb, vc;
        va = VLOAD_DOUBLE(a_tmp, i);
        vb = VLOAD_DOUBLE(b_tmp, i);
        vc = VLOAD1_DOUBLE(c, i);
        vc = VFMA_DOUBLE(vc, va, vb, i);
        VSTORE_DOUBLE(dst_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulcd_vec(double *src, double value, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;
    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLOAD_DOUBLE(src_tmp, i);

        VSTORE_DOUBLE(dst_tmp, VMUL1_DOUBLE(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void setd_vec(double *dst, double value, int len)
{
    size_t i;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        VSTORE_DOUBLE(dst_tmp, VLOAD1_DOUBLE(value, i), i);
        dst_tmp += i;
    }
}

static inline void zerod_vec(double *dst, int len)
{
    setd_vec(dst, 0.0, len);
}

static inline void copyd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        VSTORE_DOUBLE(dst_tmp, VLOAD_DOUBLE(src_tmp, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void sqrtd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLOAD_DOUBLE(src_tmp, i);
        VSTORE_DOUBLE(dst_tmp, VSQRT_DOUBLE(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void fabsd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLOAD_DOUBLE(src_tmp, i);
        VSTORE_DOUBLE(dst_tmp, VABS_DOUBLE(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void sumd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double acc[32];

    i = VSETVL64(len);
    V_ELT_DOUBLE vacc = VLOAD1_DOUBLE(0.0, i);  // initialised at 0?

    int len_ori = len;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        V_ELT_DOUBLE va = VLOAD_DOUBLE(src_tmp, i);
        vacc = VADD_DOUBLE(vacc, va, i);
        src_tmp += i;
    }

    size_t vlen_ori = VSETVL64(len_ori);
    VSTORE_DOUBLE(acc, vacc, len_ori);
    for (int j = 1; j < vlen_ori; j++) {
        acc[0] += acc[j];
    }
    *dst = acc[0];
}

static inline void meand_vec(double *src, double *dst, int len)
{
    double coeff = 1.0 / ((double) len);
    sumd_vec(src, dst, len);
    *dst *= coeff;
}

static inline void vectorSloped_vec(double *dst, int len, double offset, double slope)
{
    size_t i;
    double *dst_tmp = dst;

    double coef_max[MAX_ELTS64];

    // to be improved!
    for (int s = 0; s < MAX_ELTS64; s++) {
        coef_max[s] = (double) (s) *slope;
    }

    i = VSETVL64(len);

    V_ELT_DOUBLE coef = VLOAD_DOUBLE(coef_max, i);
    V_ELT_DOUBLE slope_vec = VLOAD1_DOUBLE((double) (i) *slope, i);
    V_ELT_DOUBLE curVal = VADD1_DOUBLE(coef, offset, i);

    VSTORE_DOUBLE(dst_tmp, curVal, i);
    dst_tmp += i;
    len -= i;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        curVal = VADD_DOUBLE(curVal, slope_vec, i);
        VSTORE_DOUBLE(dst_tmp, curVal, i);
        dst_tmp += i;
    }
}

#else

#warning "No support for double precision functions"

static inline void roundd_vec(double *src, double *dst, int len) {}
static inline void ceild_vec(double *src, double *dst, int len) {}
static inline void floord_vec(double *src, double *dst, int len) {}
static inline void truncd_vec(double *src, double *dst, int len) {}
void addd_vec(double *a, double *b, double *c, int len) {}
static inline void addcd_vec(double *src, double value, double *dst, int len) {}
static inline void muld_vec(double *a, double *b, double *c, int len) {}
static inline void divd_vec(double *a, double *b, double *c, int len) {}
static inline void subd_vec(double *a, double *b, double *c, int len) {}
static inline void muladdd_vec(double *a, double *b, double *c, double *dst, int len) {}
static inline void mulcaddd_vec(double *a, double b, double *c, double *dst, int len) {}
static inline void mulcaddcd_vec(double *a, double b, double c, double *dst, int len) {}
static inline void muladdcd_vec(double *a, double *b, double c, double *dst, int len) {}
static inline void mulcd_vec(double *src, double value, double *dst, int len) {}
static inline void setd_vec(double *dst, double value, int len) {}
static inline void zerod_vec(double *dst, int len) {}
static inline void copyd_vec(double *src, double *dst, int len) {}
static inline void sqrtd_vec(double *src, double *dst, int len) {}
static inline void fabsd_vec(double *src, double *dst, int len) {}
static inline void sumd_vec(double *src, double *dst, int len) {}
static inline void meand_vec(double *src, double *dst, int len) {}
static inline void vectorSloped_vec(double *dst, int len, double offset, double slope) {}

#endif
