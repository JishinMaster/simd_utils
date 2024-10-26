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

static inline void rintd_vec(double *src, double *dst, int len)
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

static inline void roundd_vec(double *src, double *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_AWAY);
    rintd_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void ceild_vec(double *src, double *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    rintd_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void floord_vec(double *src, double *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    rintd_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void truncd_vec(double *src, double *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    rintd_vec(src, dst, len);
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

static inline void cplxtoreald_vec(complex64_t *src, double *dstRe, double *dstIm, int len)
{
    size_t i;
    double *src_tmp = (double *) src;
    double *dstRe_tmp = dstRe;
    double *dstIm_tmp = dstIm;
    int cplx_len = len;

    for (; (i = VSETVL64H(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_DOUBLEH dstRe_vec;
        V_ELT_DOUBLEH dstIm_vec;
        VLOAD_DOUBLEH2(&dstRe_vec, &dstIm_vec, src_tmp, i);
        VSTORE_DOUBLEH(dstRe_tmp, dstRe_vec, i);
        VSTORE_DOUBLEH(dstIm_tmp, dstIm_vec, i);
        src_tmp += 2 * i;
        dstRe_tmp += i;
        dstIm_tmp += i;
    }
}

static inline void realtocplxd_vec(double *srcRe, double *srcIm, complex64_t *dst, int len)
{
    size_t i;

    double *dst_tmp = (double *) dst;
    double *srcRe_tmp = srcRe;
    double *srcIm_tmp = srcIm;
    int cplx_len = len;

    for (; (i = VSETVL64H(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_DOUBLEH srcRe_vec = VLOAD_DOUBLEH(srcRe_tmp, i);
        V_ELT_DOUBLEH srcIm_vec = VLOAD_DOUBLEH(srcIm_tmp, i);
        VSTORE_DOUBLEH2(dst_tmp, srcRe_vec, srcIm_vec, i);
        dst_tmp += 2 * i;
        srcRe_tmp += i;
        srcIm_tmp += i;
    }
}



static inline void sincos_pd(V_ELT_DOUBLEH x,
                              V_ELT_DOUBLEH *sin_tmp,
                              V_ELT_DOUBLEH *cos_tmp,
                              V_ELT_DOUBLEH coscof_1_vec,
                              V_ELT_DOUBLEH coscof_2_vec,
							  V_ELT_DOUBLEH coscof_3_vec,
							  V_ELT_DOUBLEH coscof_4_vec,
							  V_ELT_DOUBLEH coscof_5_vec,
                              V_ELT_DOUBLEH sincof_1_vec,
                              V_ELT_DOUBLEH sincof_2_vec,
                              V_ELT_DOUBLEH sincof_3_vec,
							  V_ELT_DOUBLEH sincof_4_vec,
							  V_ELT_DOUBLEH sincof_5_vec,							  
                              size_t i)
{
    V_ELT_DOUBLEH y;
    V_ELT_INT64H  j;
    V_ELT_BOOL64H jandone, jsup3, jsup1, j1or2, xinf0;
    V_ELT_BOOL64H sign_sin, sign_cos, poly_mask;

    sign_sin = VCLEAR_BOOL64H(i);
    sign_cos = VCLEAR_BOOL64H(i);
    // if (x < 0)
    xinf0 = VLT1_DOUBLEH_BOOLH(x, 0.0, i);
    sign_sin = VXOR_BOOL64H(sign_sin, xinf0, i);

    /* take the absolute value */
    x = VINTERP_INTH_DOUBLEH(VAND1_INT64H(VINTERP_DOUBLEH_INTH(x), inv_sign_maskd, i));
    /* scale by 4/Pi */
    y = VMUL1_DOUBLEH(x, FOPId, i);

    /* store the integer part of y in mm2 */
#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif	
    j = VCVT_RTZ_DOUBLEH_INTH(y, i);
#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
	
    // if (j&1))
    jandone = VNE1_INTH_BOOL64H(VAND1_INT64H(j, 1, i), 0, i);
    j = VADD1_INT64H_MASK(jandone, j, 1, i);
    y = VCVT_INTH_DOUBLEH(j, i);
	
    // j&=7
    j = VAND1_INT64H(j, 7, i);

    // if (j > 3)
    jsup3 = VGT1_INTH_BOOL64H(j, 3, i);
    sign_sin = VXOR_BOOL64H(sign_sin, jsup3, i);
    sign_cos = VXOR_BOOL64H(sign_cos, jsup3, i);
    j = VSUB1_INT64H_MASK(jsup3, j, 4, i);

    // if (j > 1)
    jsup1 = VGT1_INTH_BOOL64H(j, 1, i);
    sign_cos = VXOR_BOOL64H(sign_cos, jsup1, i);

    j1or2 = VOR_BOOL64H(VEQ1_INTH_BOOL64H(j, 1, i),
                      VEQ1_INTH_BOOL64H(j, 2, i), i);


    /* The magic pass: "Extended precision modular arithmetic"
    x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = VFMACC1_DOUBLEH(x, minus_cephes_DP1, y, i);
    x = VFMACC1_DOUBLEH(x, minus_cephes_DP2, y, i);
    x = VFMACC1_DOUBLEH(x, minus_cephes_DP3, y, i);
	//printf("x ");print_vec64h(x,i);
    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    V_ELT_DOUBLEH z = VMUL_DOUBLEH(x, x, i);
    y = z;
    y = VFMADD1_DOUBLEH(y, coscod[0], coscof_1_vec, i);
    y = VFMADD_DOUBLEH(y, z, coscof_2_vec, i);
    y = VFMADD_DOUBLEH(y, z, coscof_3_vec, i);
    y = VFMADD_DOUBLEH(y, z, coscof_4_vec, i);
    y = VFMADD_DOUBLEH(y, z, coscof_5_vec, i);	
    y = VMUL_DOUBLEH(y, z, i);
    y = VMUL_DOUBLEH(y, z, i);
    y = VFMACC1_DOUBLEH(y, -0.5, z, i);  // y = y -0.5*z
    y = VADD1_DOUBLEH(y, 1.0, i);
	//printf("y ");print_vec64h(y,i);
    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    V_ELT_DOUBLEH y2;
    y2 = z;
    y2 = VFMADD1_DOUBLEH(y2, sincod[0], sincof_1_vec, i);
    y2 = VFMADD_DOUBLEH(y2, z, sincof_2_vec, i);
    y2 = VFMADD_DOUBLEH(y2, z, sincof_3_vec, i);
    y2 = VFMADD_DOUBLEH(y2, z, sincof_4_vec, i);
    y2 = VFMADD_DOUBLEH(y2, z, sincof_5_vec, i);	
    y2 = VMUL_DOUBLEH(y2, z, i);
    y2 = VFMADD_DOUBLEH(y2, x, x, i);
	//printf("y2 ");print_vec64h(y2,i);
    /* select the correct result from the two polynoms */
    //V_ELT_DOUBLEH y_sin = VMERGE_DOUBLEH(poly_mask, y, y2, i);
    //V_ELT_DOUBLEH y_cos = VMERGE_DOUBLEH(poly_mask, y2, y, i);
    V_ELT_DOUBLEH y_sin = VMERGE_DOUBLEH(j1or2, y2, y, i);
    V_ELT_DOUBLEH y_cos = VMERGE_DOUBLEH(j1or2, y, y2, i);
	
    y_sin = VMUL1_DOUBLEH_MASK(sign_sin, y_sin, -1.0, i);
    y_cos = VMUL1_DOUBLEH_MASK(sign_cos, y_cos, -1.0, i);
    *sin_tmp = y_sin;
    *cos_tmp = y_cos;
}

static inline void sincosd_vec(double *src, double *s, double *c, int len)
{
    size_t i;
    double *src_tmp = src;
    double *s_tmp = s;
    double *c_tmp = c;

    i = VSETVL64H(len);
    V_ELT_DOUBLEH coscof_1_vec = VLOAD1_DOUBLEH(coscod[1], i);
    V_ELT_DOUBLEH coscof_2_vec = VLOAD1_DOUBLEH(coscod[2], i);
	V_ELT_DOUBLEH coscof_3_vec = VLOAD1_DOUBLEH(coscod[3], i);
	V_ELT_DOUBLEH coscof_4_vec = VLOAD1_DOUBLEH(coscod[4], i);
	V_ELT_DOUBLEH coscof_5_vec = VLOAD1_DOUBLEH(coscod[5], i);	
    V_ELT_DOUBLEH sincof_1_vec = VLOAD1_DOUBLEH(sincod[1], i);
    V_ELT_DOUBLEH sincof_2_vec = VLOAD1_DOUBLEH(sincod[2], i);
    V_ELT_DOUBLEH sincof_3_vec = VLOAD1_DOUBLEH(sincod[3], i);
    V_ELT_DOUBLEH sincof_4_vec = VLOAD1_DOUBLEH(sincod[4], i);
    V_ELT_DOUBLEH sincof_5_vec = VLOAD1_DOUBLEH(sincod[5], i);
	
    for (; (i = VSETVL64H(len)) > 0; len -= i) {
        V_ELT_DOUBLEH x = VLOAD_DOUBLEH(src_tmp, i);
        V_ELT_DOUBLEH y_sin, y_cos;
        sincos_pd(x, &y_sin, &y_cos,
                   coscof_1_vec, coscof_2_vec, coscof_3_vec,
				   coscof_4_vec, coscof_5_vec,
                   sincof_1_vec, sincof_2_vec,  sincof_3_vec,
				   sincof_4_vec, sincof_5_vec, i);
        VSTORE_DOUBLEH(s_tmp, y_sin, i);
        VSTORE_DOUBLEH(c_tmp, y_cos, i);

        src_tmp += i;
        s_tmp += i;
        c_tmp += i;
    }
}

//TODO : check with a real target, QEMU shows low precision
static inline void sincosd_interleaved_vec(double *src, complex64_t *dst, int len)
{
    size_t i;
    double *src_tmp = src;
	double *dst_tmp = (double *) dst;

    i = VSETVL64H(len);
    V_ELT_DOUBLEH coscof_1_vec = VLOAD1_DOUBLEH(coscod[1], i);
    V_ELT_DOUBLEH coscof_2_vec = VLOAD1_DOUBLEH(coscod[2], i);
	V_ELT_DOUBLEH coscof_3_vec = VLOAD1_DOUBLEH(coscod[3], i);
	V_ELT_DOUBLEH coscof_4_vec = VLOAD1_DOUBLEH(coscod[4], i);
	V_ELT_DOUBLEH coscof_5_vec = VLOAD1_DOUBLEH(coscod[5], i);	
    V_ELT_DOUBLEH sincof_1_vec = VLOAD1_DOUBLEH(sincod[1], i);
    V_ELT_DOUBLEH sincof_2_vec = VLOAD1_DOUBLEH(sincod[2], i);
    V_ELT_DOUBLEH sincof_3_vec = VLOAD1_DOUBLEH(sincod[3], i);
    V_ELT_DOUBLEH sincof_4_vec = VLOAD1_DOUBLEH(sincod[4], i);
    V_ELT_DOUBLEH sincof_5_vec = VLOAD1_DOUBLEH(sincod[5], i);
	
    for (; (i = VSETVL64H(len)) > 0; len -= i) {
        V_ELT_DOUBLEH x = VLOAD_DOUBLEH(src_tmp, i);
        V_ELT_DOUBLEH y_sin, y_cos;
        sincos_pd(x, &y_sin, &y_cos,
                   coscof_1_vec, coscof_2_vec, coscof_3_vec,
				   coscof_4_vec, coscof_5_vec,
                   sincof_1_vec, sincof_2_vec,  sincof_3_vec,
				   sincof_4_vec, sincof_5_vec, i);
        VSTORE_DOUBLEH2(dst_tmp, y_cos, y_sin, i);
        dst_tmp += 2 * i;
        src_tmp += i;
    }
}


static inline V_ELT_DOUBLEH atand_pd(V_ELT_DOUBLEH xx,
                                    V_ELT_DOUBLEH ATAN_P1_vec,
                                    V_ELT_DOUBLEH ATAN_P2_vec,
                                    V_ELT_DOUBLEH ATAN_P3_vec,
                                    V_ELT_DOUBLEH ATAN_P4_vec,
									V_ELT_DOUBLEH ATAN_Q1_vec,
                                    V_ELT_DOUBLEH ATAN_Q2_vec,
                                    V_ELT_DOUBLEH ATAN_Q3_vec,
                                    V_ELT_DOUBLEH ATAN_Q4_vec,
									V_ELT_DOUBLEH MOREBITS_vec,	
									V_ELT_DOUBLEH PIO4_vec,	
                                    V_ELT_DOUBLEH min1_vec,
                                    size_t i)
{
    V_ELT_DOUBLEH x, y, z;
    V_ELT_INT64H sign;
    V_ELT_BOOL64H suptan3pi8, inftan3pi8inf0p66;
    V_ELT_DOUBLEH tmp, tmp2;
    V_ELT_BOOL64H tmpb1, tmpb2;
	V_ELT_INT64H flag;
	
    x = VINTERP_INTH_DOUBLEH(VAND1_INT64H(VINTERP_DOUBLEH_INTH(xx), inv_sign_maskd, i));
    sign = VAND1_INT64H(VINTERP_DOUBLEH_INTH(xx), sign_maskd, i);
	//printf("x : ");print_vec64h(x,i);
    /* range reduction */
    y = VLOAD1_DOUBLEH(0.0, i);
	flag = VLOAD1_INT64H(0, i);
    suptan3pi8 = VGT1_DOUBLEH_BOOLH(x, TAN3PI8d, i);
    tmp = VDIV_DOUBLEH(min1_vec, x, i);
    x = VMERGE_DOUBLEH(suptan3pi8, x, tmp, i);
    y = VMERGE1_DOUBLEH(suptan3pi8, y, PIO2d, i);
    flag = VMERGE1_INT64H(suptan3pi8, flag, 1, i);	


    tmpb1 = VLE1_DOUBLEH_BOOLH(x, TAN3PI8d, i);
    tmpb2 = VLE1_DOUBLEH_BOOLH(x, 0.66, i);
    inftan3pi8inf0p66 = VAND_BOOL64H(tmpb1, tmpb2, i);
    y = VMERGE_DOUBLEH(inftan3pi8inf0p66, PIO4_vec, y, i);

    // To be optimised with RCP?
	tmp2 = VADD1_DOUBLEH(x, 1.0, i);
    tmp = VSUB1_DOUBLEH(x, 1.0, i);
    tmp = VDIV_DOUBLEH(tmp, tmp2, i);
    x = VMERGE_DOUBLEH(inftan3pi8inf0p66, tmp, x, i);
	V_ELT_BOOL64H yeqpio4 = VEQ1_DOUBLEH_BOOLH(y, PIO4d, i);
    flag = VMERGE1_INT64H(yeqpio4, flag, 2, i);	
	
    z = VMUL_DOUBLEH(x, x, i);

    tmp = z;
    tmp = VFMADD1_DOUBLEH(tmp, ATAN_P0d, ATAN_P1_vec, i);
    tmp = VFMADD_DOUBLEH(tmp, z, ATAN_P2_vec, i);
    tmp = VFMADD_DOUBLEH(tmp, z, ATAN_P3_vec, i);
    tmp = VFMADD_DOUBLEH(tmp, z, ATAN_P4_vec, i);	
    tmp = VMUL_DOUBLEH(z, tmp, i);
	
	tmp2 = z;
	tmp2 = VADD1_DOUBLEH(tmp2, ATAN_Q0d, i);
	tmp2 = VFMADD_DOUBLEH(tmp2, z, ATAN_Q1_vec, i);
	tmp2 = VFMADD_DOUBLEH(tmp2, z, ATAN_Q2_vec, i);
	tmp2 = VFMADD_DOUBLEH(tmp2, z, ATAN_Q3_vec, i);
	tmp2 = VFMADD_DOUBLEH(tmp2, z, ATAN_Q4_vec, i);		
	z = VDIV_DOUBLEH(tmp, tmp2, i);
	
	z = VFMADD_DOUBLEH(z, x, x, i);
	V_ELT_BOOL64H flageq1 = VEQ1_INTH_BOOL64H(flag, 1, i);
	V_ELT_BOOL64H flageq2 = VEQ1_INTH_BOOL64H(flag, 2, i);
	tmp = MOREBITS_vec;
	tmp = VFMADD1_DOUBLEH(MOREBITS_vec, 0.5, z, i);
	z = VMERGE_DOUBLEH(flageq2, z, tmp, i);
    z = VADD1_DOUBLEH_MASK(flageq1, z, MOREBITSd, i);
	y = VADD_DOUBLEH(y, z, i);
	V_ELT_BOOL64H xeq0 = VEQ1_DOUBLEH_BOOLH(x, 0.0, i);
    y = VINTERP_INTH_DOUBLEH(VXOR_INT64H(VINTERP_DOUBLEH_INTH(y), sign, i));
	y = VMERGE_DOUBLEH(xeq0, y, xx, i);
    return y;
}

static inline void atand_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;

    i = VSETVL64H(len);
    V_ELT_DOUBLEH ATAN_P1_vec = VLOAD1_DOUBLEH(ATAN_P1d, i);
    V_ELT_DOUBLEH ATAN_P2_vec = VLOAD1_DOUBLEH(ATAN_P2d, i);
    V_ELT_DOUBLEH ATAN_P3_vec = VLOAD1_DOUBLEH(ATAN_P3d, i);
    V_ELT_DOUBLEH ATAN_P4_vec = VLOAD1_DOUBLEH(ATAN_P4d, i);
    V_ELT_DOUBLEH ATAN_Q1_vec = VLOAD1_DOUBLEH(ATAN_Q1d, i);
    V_ELT_DOUBLEH ATAN_Q2_vec = VLOAD1_DOUBLEH(ATAN_Q2d, i);
    V_ELT_DOUBLEH ATAN_Q3_vec = VLOAD1_DOUBLEH(ATAN_Q3d, i);
    V_ELT_DOUBLEH ATAN_Q4_vec = VLOAD1_DOUBLEH(ATAN_Q4d, i);
    V_ELT_DOUBLEH MOREBITS_vec = VLOAD1_DOUBLEH(MOREBITSd, i);
	V_ELT_DOUBLEH PIO4_vec = VLOAD1_DOUBLEH(PIO4d, i);
	
    V_ELT_DOUBLEH min1_vec = VLOAD1_DOUBLEH(-1.0, i);
	
    for (; (i = VSETVL64H(len)) > 0; len -= i) {
        V_ELT_DOUBLEH xx = VLOAD_DOUBLEH(src_tmp, i);
        V_ELT_DOUBLEH y;
        y = atand_pd(xx, ATAN_P1_vec, ATAN_P2_vec, ATAN_P3_vec, ATAN_P4_vec,
					ATAN_Q1_vec, ATAN_Q2_vec, ATAN_Q3_vec, ATAN_Q4_vec,
					MOREBITS_vec, PIO4_vec, min1_vec, i);
        VSTORE_DOUBLEH(dst_tmp, y, i);
        src_tmp += i;
        dst_tmp += i;
    }
}


static inline V_ELT_DOUBLEH atan2_pd(V_ELT_DOUBLEH y, V_ELT_DOUBLEH x, V_ELT_DOUBLEH ATAN_P1_vec,
                                    V_ELT_DOUBLEH ATAN_P2_vec,
                                    V_ELT_DOUBLEH ATAN_P3_vec,
                                    V_ELT_DOUBLEH ATAN_P4_vec,
									V_ELT_DOUBLEH ATAN_Q1_vec,
                                    V_ELT_DOUBLEH ATAN_Q2_vec,
                                    V_ELT_DOUBLEH ATAN_Q3_vec,
                                    V_ELT_DOUBLEH ATAN_Q4_vec,
									V_ELT_DOUBLEH MOREBITS_vec,	
									V_ELT_DOUBLEH PIO4_vec,	
                                    V_ELT_DOUBLEH min1_vec,
                                    size_t i)
{
    V_ELT_DOUBLEH z, w;
    V_ELT_BOOL64H xinfzero, yinfzero, xeqzero, yeqzero;
    V_ELT_BOOL64H xeqzeroandyinfzero, yeqzeroandxinfzero;
    V_ELT_BOOL64H specialcase;
    V_ELT_DOUBLEH tmp, tmp2;

    xinfzero = VLT1_DOUBLEH_BOOLH(x, 0.0, i);  // code =2
    yinfzero = VLT1_DOUBLEH_BOOLH(y, 0.0, i);  // code = code |1;

    xeqzero = VEQ1_DOUBLEH_BOOLH(x, 0.0, i);
    yeqzero = VEQ1_DOUBLEH_BOOLH(y, 0.0, i);

    xeqzeroandyinfzero = VAND_BOOL64H(xeqzero, yinfzero, i);
    yeqzeroandxinfzero = VAND_BOOL64H(yeqzero, xinfzero, i);

    z = VLOAD1_DOUBLEH(PIO2d, i);
    z = VMERGE1_DOUBLEH(xeqzeroandyinfzero, z, -PIO2d, i);
    z = VMERGE1_DOUBLEH(yeqzero, z, 0.0, i);
    z = VMERGE1_DOUBLEH(yeqzeroandxinfzero, z, PId, i);
    specialcase = VOR_BOOL64H(xeqzero, yeqzero, i);

    w = VLOAD1_DOUBLEH(0.0, i);
    w = VMERGE1_DOUBLEH(VAND_BOOL64H(VNOT_BOOL64H(yinfzero, i), xinfzero, i), w, PId, i);  // y >= 0 && x<0
    w = VMERGE1_DOUBLEH(VAND_BOOL64H(yinfzero, xinfzero, i), w, -PId, i);                // y < 0 && x<0

    tmp = VDIV_DOUBLEH(y, x, i);
    tmp = atand_pd(tmp, ATAN_P1_vec, ATAN_P2_vec, ATAN_P3_vec, ATAN_P4_vec,
					ATAN_Q1_vec, ATAN_Q2_vec, ATAN_Q3_vec, ATAN_Q4_vec,
					MOREBITS_vec, PIO4_vec, min1_vec, i);
    tmp = VADD_DOUBLEH(w, tmp, i);
    z = VMERGE_DOUBLEH(specialcase, tmp, z, i);  // atan(y/x) if not in special case
    return z;
}

static inline void atan2d_vec(double *src1, double *src2, double *dst, int len)
{
    size_t i;
    double *src1_tmp = src1;
    double *src2_tmp = src2;
    double *dst_tmp = dst;

    i = VSETVL64H(len);
    V_ELT_DOUBLEH ATAN_P1_vec = VLOAD1_DOUBLEH(ATAN_P1d, i);
    V_ELT_DOUBLEH ATAN_P2_vec = VLOAD1_DOUBLEH(ATAN_P2d, i);
    V_ELT_DOUBLEH ATAN_P3_vec = VLOAD1_DOUBLEH(ATAN_P3d, i);
    V_ELT_DOUBLEH ATAN_P4_vec = VLOAD1_DOUBLEH(ATAN_P4d, i);
    V_ELT_DOUBLEH ATAN_Q1_vec = VLOAD1_DOUBLEH(ATAN_Q1d, i);
    V_ELT_DOUBLEH ATAN_Q2_vec = VLOAD1_DOUBLEH(ATAN_Q2d, i);
    V_ELT_DOUBLEH ATAN_Q3_vec = VLOAD1_DOUBLEH(ATAN_Q3d, i);
    V_ELT_DOUBLEH ATAN_Q4_vec = VLOAD1_DOUBLEH(ATAN_Q4d, i);
    V_ELT_DOUBLEH MOREBITS_vec = VLOAD1_DOUBLEH(MOREBITSd, i);
	V_ELT_DOUBLEH PIO4_vec = VLOAD1_DOUBLEH(PIO4d, i);
    V_ELT_DOUBLEH min1_vec = VLOAD1_DOUBLEH(-1.0, i);

    for (; (i = VSETVL64H(len)) > 0; len -= i) {
        V_ELT_DOUBLEH y = VLOAD_DOUBLEH(src1_tmp, i);
        V_ELT_DOUBLEH x = VLOAD_DOUBLEH(src2_tmp, i);

        V_ELT_DOUBLEH z = atan2_pd(y, x,
                                   ATAN_P1_vec, ATAN_P2_vec, ATAN_P3_vec, ATAN_P4_vec,
					ATAN_Q1_vec, ATAN_Q2_vec, ATAN_Q3_vec, ATAN_Q4_vec,
					MOREBITS_vec, PIO4_vec, min1_vec, i);
        VSTORE_DOUBLEH(dst_tmp, z, i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void atan2d_interleaved_vec(complex64_t *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = (double*)src;
    double *dst_tmp = dst;

    i = VSETVL64H(len);
    V_ELT_DOUBLEH ATAN_P1_vec = VLOAD1_DOUBLEH(ATAN_P1d, i);
    V_ELT_DOUBLEH ATAN_P2_vec = VLOAD1_DOUBLEH(ATAN_P2d, i);
    V_ELT_DOUBLEH ATAN_P3_vec = VLOAD1_DOUBLEH(ATAN_P3d, i);
    V_ELT_DOUBLEH ATAN_P4_vec = VLOAD1_DOUBLEH(ATAN_P4d, i);
    V_ELT_DOUBLEH ATAN_Q1_vec = VLOAD1_DOUBLEH(ATAN_Q1d, i);
    V_ELT_DOUBLEH ATAN_Q2_vec = VLOAD1_DOUBLEH(ATAN_Q2d, i);
    V_ELT_DOUBLEH ATAN_Q3_vec = VLOAD1_DOUBLEH(ATAN_Q3d, i);
    V_ELT_DOUBLEH ATAN_Q4_vec = VLOAD1_DOUBLEH(ATAN_Q4d, i);
    V_ELT_DOUBLEH MOREBITS_vec = VLOAD1_DOUBLEH(MOREBITSd, i);
	V_ELT_DOUBLEH PIO4_vec = VLOAD1_DOUBLEH(PIO4d, i);
    V_ELT_DOUBLEH min1_vec = VLOAD1_DOUBLEH(-1.0, i);

    for (; (i = VSETVL64H(len)) > 0; len -= i) {
        V_ELT_DOUBLEH y, x;
        VLOAD_DOUBLEH2(&x, &y, src_tmp, i);
        V_ELT_DOUBLEH z = atan2_pd(y, x,
                                   ATAN_P1_vec, ATAN_P2_vec, ATAN_P3_vec, ATAN_P4_vec,
					ATAN_Q1_vec, ATAN_Q2_vec, ATAN_Q3_vec, ATAN_Q4_vec,
					MOREBITS_vec, PIO4_vec, min1_vec, i);
        VSTORE_DOUBLEH(dst_tmp, z, i);
        src_tmp += 2*i;
        dst_tmp += i;
    }
}


static inline V_ELT_DOUBLEH exp_pd(V_ELT_DOUBLEH x,
                                  V_ELT_DOUBLEH Op5_vec,
                                  V_ELT_DOUBLEH cephes_exp_p1_vec,
                                  V_ELT_DOUBLEH cephes_exp_p2_vec,
                                  V_ELT_DOUBLEH cephes_exp_q1_vec,
                                  V_ELT_DOUBLEH cephes_exp_q2_vec,
                                  V_ELT_DOUBLEH cephes_exp_q3_vec,
								  V_ELT_DOUBLEH one_vec,
                                  size_t i)
{
    V_ELT_DOUBLEH px, xx, tmp, tmp2;
    V_ELT_INT64H n;

    /* Express e**x = e**g 2**n
     *   = e**g e**( n loge(2) )
     *   = e**( g + n loge(2) )
     */
    px = x;
    px = VFMADD1_DOUBLEH(px, cephes_LOG2Ed, Op5_vec, i);
    px = VCVT_INTH_DOUBLEH(VCVT_DOUBLEH_INTH(px, i), i);
    n = VCVT_DOUBLEH_INTH(px, i);
    x = VFMACC1_DOUBLEH(x, cephes_exp_minC1d, px, i);
    x = VFMACC1_DOUBLEH(x, cephes_exp_minC2d, px, i);

    xx = VMUL_DOUBLEH(x, x, i);
    tmp = xx;
    tmp = VFMADD1_DOUBLEH(tmp, cephes_exp_p0d, cephes_exp_p1_vec, i);
    tmp = VFMADD_DOUBLEH(tmp, xx, cephes_exp_p2_vec, i);
	px = VMUL_DOUBLEH(tmp, x, i);
	
	tmp2 = xx;
    tmp2 = VFMADD1_DOUBLEH(tmp2, cephes_exp_q0d, cephes_exp_q1_vec, i);
    tmp2 = VFMADD_DOUBLEH(tmp2, xx, cephes_exp_q2_vec, i);
    tmp2 = VFMADD_DOUBLEH(tmp2, xx, cephes_exp_q3_vec, i);
	tmp2 = VSUB_DOUBLEH(tmp2, px, i);
	x = VDIV_DOUBLEH(px, tmp2, i);
    x = VFMADD1_DOUBLEH(x, 2.0, one_vec, i);

	/* build 2^n */
    n = VADD1_INT64H(n, (unsigned int)1023, i);
    n = VSLL1_INT64H(n, 52, i);
    V_ELT_DOUBLEH pow2n = VINTERP_INTH_DOUBLEH(n);

    /* multiply by power of 2 */
    x = VMUL_DOUBLEH(x, pow2n, i);
    return x;
}

static inline void expd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;

    i = VSETVL64H(len);

    uint64_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

    V_ELT_DOUBLEH cephes_exp_p1_vec = VLOAD1_DOUBLEH(cephes_exp_p1d, i);
    V_ELT_DOUBLEH cephes_exp_p2_vec = VLOAD1_DOUBLEH(cephes_exp_p2d, i);
    V_ELT_DOUBLEH cephes_exp_q1_vec = VLOAD1_DOUBLEH(cephes_exp_q1d, i);
    V_ELT_DOUBLEH cephes_exp_q2_vec = VLOAD1_DOUBLEH(cephes_exp_q2d, i);
    V_ELT_DOUBLEH cephes_exp_q3_vec = VLOAD1_DOUBLEH(cephes_exp_q3d, i);
    V_ELT_DOUBLEH Op5_vec = VLOAD1_DOUBLEH(0.5, i);
    V_ELT_DOUBLEH one_vec = VLOAD1_DOUBLEH(1.0, i);	

    for (; (i = VSETVL64H(len)) > 0; len -= i) {
        V_ELT_DOUBLEH x = VLOAD_DOUBLEH(src_tmp, i);
        x = exp_pd(x, Op5_vec, cephes_exp_p1_vec,
                   cephes_exp_p2_vec, cephes_exp_q1_vec,
                   cephes_exp_q2_vec, cephes_exp_q3_vec, one_vec, i);
        VSTORE_DOUBLEH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

//Work in progress
//Should improve precision or check on real hardware
static inline void asind_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;
	
    i = VSETVL64H(len);
    V_ELT_DOUBLEH ASIN_S1_vec = VLOAD1_DOUBLEH(ASIN_S1d, i);
    V_ELT_DOUBLEH ASIN_S2_vec = VLOAD1_DOUBLEH(ASIN_S2d, i);
    V_ELT_DOUBLEH ASIN_S3_vec = VLOAD1_DOUBLEH(ASIN_S3d, i);
    V_ELT_DOUBLEH ASIN_P1_vec = VLOAD1_DOUBLEH(ASIN_P1d, i);
    V_ELT_DOUBLEH ASIN_P2_vec = VLOAD1_DOUBLEH(ASIN_P2d, i);
    V_ELT_DOUBLEH ASIN_P3_vec = VLOAD1_DOUBLEH(ASIN_P3d, i);
    V_ELT_DOUBLEH ASIN_P4_vec = VLOAD1_DOUBLEH(ASIN_P4d, i);
    V_ELT_DOUBLEH ASIN_P5_vec = VLOAD1_DOUBLEH(ASIN_P5d, i);
    V_ELT_DOUBLEH ASIN_Q1_vec = VLOAD1_DOUBLEH(ASIN_Q1d, i);
    V_ELT_DOUBLEH ASIN_Q2_vec = VLOAD1_DOUBLEH(ASIN_Q2d, i);
    V_ELT_DOUBLEH ASIN_Q3_vec = VLOAD1_DOUBLEH(ASIN_Q3d, i);
	V_ELT_DOUBLEH ASIN_Q4_vec = VLOAD1_DOUBLEH(ASIN_Q4d, i);
	V_ELT_DOUBLEH ASIN_R1_vec = VLOAD1_DOUBLEH(ASIN_R1d, i);
	V_ELT_DOUBLEH ASIN_R2_vec = VLOAD1_DOUBLEH(ASIN_R2d, i);
	V_ELT_DOUBLEH ASIN_R3_vec = VLOAD1_DOUBLEH(ASIN_R3d, i);
	V_ELT_DOUBLEH ASIN_R4_vec = VLOAD1_DOUBLEH(ASIN_R4d, i);
    V_ELT_DOUBLEH minMOREBITS_vec = VLOAD1_DOUBLEH(minMOREBITSd, i);
	
    for (; (i = VSETVL64H(len)) > 0; len -= i) {
        V_ELT_DOUBLEH x = VLOAD_DOUBLEH(src_tmp, i);
		
        V_ELT_DOUBLEH a, z, z_first_branch, zz_first_branch, p, zz_second_branch; 
		V_ELT_DOUBLEH tmp_first_branch, tmp_second_branch, z_second_branch;
        V_ELT_INT64H sign;
        V_ELT_BOOL64H ainfem8, asup0p625, xsup1;

        a = VINTERP_INTH_DOUBLEH(VAND1_INT64H(VINTERP_DOUBLEH_INTH(x), inv_sign_mask, i));
        sign = VAND1_INT64H(VINTERP_DOUBLEH_INTH(x), sign_mask, i);

        ainfem8 = VLT1_DOUBLEH_BOOLH(a, 1.0e-8, i);
        asup0p625 = VGT1_DOUBLEH_BOOLH(a, 0.625, i); 
		
		// fist branch		
        zz_first_branch = VRSUB1_DOUBLEH(a, 1.0, i);
		p = zz_first_branch;
		p = VFMADD1_DOUBLEH(p, ASIN_R0d, ASIN_R1_vec, i);
		p = VFMADD_DOUBLEH(p, zz_first_branch, ASIN_R2_vec, i);
		p = VFMADD_DOUBLEH(p, zz_first_branch, ASIN_R3_vec, i);
		p = VFMADD_DOUBLEH(p, zz_first_branch, ASIN_R4_vec, i);
		p = VMUL_DOUBLEH(p, zz_first_branch, i);		

		tmp_first_branch = VADD1_DOUBLEH(zz_first_branch, ASIN_S0d, i);
		tmp_first_branch = VFMADD_DOUBLEH(tmp_first_branch, zz_first_branch, ASIN_S1_vec, i);
		tmp_first_branch = VFMADD_DOUBLEH(tmp_first_branch, zz_first_branch, ASIN_S2_vec, i);
		tmp_first_branch = VFMADD_DOUBLEH(tmp_first_branch, zz_first_branch, ASIN_S3_vec, i);
		p = VDIV_DOUBLEH(p, tmp_first_branch, i);

		zz_first_branch = VSQRT_DOUBLEH(VADD_DOUBLEH(zz_first_branch, zz_first_branch, i), i);
		z_first_branch = VRSUB1_DOUBLEH(zz_first_branch, PIO4d, i);
		zz_first_branch = VFMADD_DOUBLEH(zz_first_branch, p, minMOREBITS_vec, i);
		z_first_branch = VSUB_DOUBLEH(z_first_branch, zz_first_branch, i);
	
	    // second branch
		zz_second_branch = VMUL_DOUBLEH(a, a, i);
		z_second_branch = zz_second_branch;
		z_second_branch = VFMADD1_DOUBLEH(z_second_branch, ASIN_P0d, ASIN_P1_vec, i);
		z_second_branch = VFMADD_DOUBLEH(z_second_branch, zz_second_branch, ASIN_P2_vec, i);
		z_second_branch = VFMADD_DOUBLEH(z_second_branch, zz_second_branch, ASIN_P3_vec, i);
		z_second_branch = VFMADD_DOUBLEH(z_second_branch, zz_second_branch, ASIN_P4_vec, i);
		z_second_branch = VFMADD_DOUBLEH(z_second_branch, zz_second_branch, ASIN_P5_vec, i);
		z_second_branch = VMUL_DOUBLEH(z_second_branch, zz_second_branch, i);

		tmp_second_branch = VADD1_DOUBLEH(zz_second_branch, ASIN_Q0d, i);
		tmp_second_branch = VFMADD_DOUBLEH(tmp_second_branch, zz_second_branch, ASIN_Q1_vec, i);
		tmp_second_branch = VFMADD_DOUBLEH(tmp_second_branch, zz_second_branch, ASIN_Q2_vec, i);
		tmp_second_branch = VFMADD_DOUBLEH(tmp_second_branch, zz_second_branch, ASIN_Q3_vec, i);
		tmp_second_branch = VFMADD_DOUBLEH(tmp_second_branch, zz_second_branch, ASIN_Q4_vec, i);

		z_second_branch = VDIV_DOUBLEH(z_second_branch, tmp_second_branch, i);
		z_second_branch = VFMADD_DOUBLEH(z_second_branch, a, a, i);
		/*printf("x : ");print_vec64h(x, i);
		printf("z_first_branch : ");print_vec64h(z_first_branch, i);		
		printf("z_second_branch : ");print_vec64h(z_second_branch, i);*/
		z = VADD1_DOUBLEH_MASKEDOFF(asup0p625, z_second_branch, z_first_branch, PIO4d, i);
		//printf("z : ");print_vec64h(z, i);
        z = VINTERP_INTH_DOUBLEH(VXOR_INT64H(VINTERP_DOUBLEH_INTH(z), sign, i));
		z = VMERGE_DOUBLEH(ainfem8, z, x, i);
		//printf("z : ");print_vec64h(z, i);
        // if (x > 1.0) then return 0.0
		xsup1 = VGT1_DOUBLEH_BOOLH(x, 1.0, i);		
        z = VMERGE1_DOUBLEH(xsup1, z, 0.0, i);
		//printf("z : ");print_vec64h(z, i);

        VSTORE_DOUBLEH(dst_tmp, z, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

#else

#warning "No support for double precision functions"

static inline void rintd_vec(double *src, double *dst, int len) {}
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
static inline void cplxtoreald_vec(complex64_t *src, double *dstRe, double *dstIm, int len) {}
static inline void realtocplxd_vec(double *srcRe, double *srcIm, complex64_t *dst, int len) {}
static inline void sincosd_vec(double *src, double *s, double *c, int len) {}
static inline void sincosd_interleaved_vec(double *src, complex64_t *dst, int len) {}
static inline void atand_vec(double *src, double *dst, int len) {}
static inline void atan2d_vec(double *src1, double *src2, double *dst, int len) {}
static inline void atan2d_interleaved_vec(complex64_t *src, double *dst, int len) {}
static inline void asind_vec(double *src, double *dst, int len) {}
static inline void expd_vec(double *src, double *dst, int len) {}
#endif
