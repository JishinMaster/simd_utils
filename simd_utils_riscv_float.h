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


// e32 => float32 (e64 float 64)
// m8 8 elements (m4 4 elements)
/* i = vsetvl_e32m8(len) asks for
    n float32 elements grouped by 8. l returns the total number of elements achievable
    with this configuration
*/
void addf_vec(float *a, float *b, float *c, int len)
{
    size_t i;
    float *a_tmp = a;
    float *b_tmp = b;
    float *c_tmp = c;
    V_ELT_FLOAT va, vb, vc;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        va = VLOAD_FLOAT(a_tmp, i);
        vb = VLOAD_FLOAT(b_tmp, i);
        vc = VADD_FLOAT(va, vb, i);
        VSTORE_FLOAT(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void addcf_vec(float *src, float value, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src_tmp, i);

        VSTORE_FLOAT(dst_tmp, VADD1_FLOAT(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulf_vec(float *a, float *b, float *c, int len)
{
    size_t i;
    float *a_tmp = a;
    float *b_tmp = b;
    float *c_tmp = c;
    V_ELT_FLOAT va, vb, vc;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        va = VLOAD_FLOAT(a_tmp, i);
        vb = VLOAD_FLOAT(b_tmp, i);
        vc = VMUL_FLOAT(va, vb, i);
        VSTORE_FLOAT(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void divf_vec(float *a, float *b, float *c, int len)
{
    size_t i;
    float *a_tmp = a;
    float *b_tmp = b;
    float *c_tmp = c;
    V_ELT_FLOAT va, vb, vc;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        va = VLOAD_FLOAT(a_tmp, i);
        vb = VLOAD_FLOAT(b_tmp, i);
        vc = VDIV_FLOAT(va, vb, i);
        VSTORE_FLOAT(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void subf_vec(float *a, float *b, float *c, int len)
{
    size_t i;
    float *a_tmp = a;
    float *b_tmp = b;
    float *c_tmp = c;
    V_ELT_FLOAT va, vb, vc;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        va = VLOAD_FLOAT(a_tmp, i);
        vb = VLOAD_FLOAT(b_tmp, i);
        vc = VSUB_FLOAT(va, vb, i);
        VSTORE_FLOAT(c_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
    }
}

static inline void muladdf_vec(float *a, float *b, float *c, float *dst, int len)
{
    size_t i;
    float *a_tmp = a;
    float *b_tmp = b;
    float *c_tmp = c;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb, vc;
        va = VLOAD_FLOAT(a_tmp, i);
        vb = VLOAD_FLOAT(b_tmp, i);
        vc = VLOAD_FLOAT(c_tmp, i);
        vc = VFMACC_FLOAT(vc, va, vb, i);
        VSTORE_FLOAT(dst_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        c_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulcaddf_vec(float *a, float b, float *c, float *dst, int len)
{
    size_t i;
    float *a_tmp = a;
    float *c_tmp = c;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vc;
        va = VLOAD_FLOAT(a_tmp, i);
        vc = VLOAD_FLOAT(c_tmp, i);
        vc = VFMACC1_FLOAT(vc, b, va, i);
        VSTORE_FLOAT(dst_tmp, vc, i);

        a_tmp += i;
        c_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulcaddcf_vec(float *a, float b, float c, float *dst, int len)
{
    size_t i;
    float *a_tmp = a;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vc;
        va = VLOAD_FLOAT(a_tmp, i);
        vc = VLOAD1_FLOAT(c, i);
        vc = VFMACC1_FLOAT(vc, b, va, i);
        VSTORE_FLOAT(dst_tmp, vc, i);

        a_tmp += i;
        dst_tmp += i;
    }
}

static inline void muladdcf_vec(float *a, float *b, float c, float *dst, int len)
{
    size_t i;
    float *a_tmp = a;
    float *b_tmp = b;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb, vc;
        va = VLOAD_FLOAT(a_tmp, i);
        vb = VLOAD_FLOAT(b_tmp, i);
        vc = VLOAD1_FLOAT(c, i);
        vc = VFMACC_FLOAT(vc, va, vb, i);
        VSTORE_FLOAT(dst_tmp, vc, i);

        a_tmp += i;
        b_tmp += i;
        dst_tmp += i;
    }
}

static inline void mulcf_vec(float *src, float value, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src_tmp, i);

        VSTORE_FLOAT(dst_tmp, VMUL1_FLOAT(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

//TODO : could be improved with FMA
static inline void sinf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLOAD_FLOAT(src_tmp, i);

        V_ELT_FLOAT sign_bit, y;
        V_ELT_INT emm0, emm2;
        sign_bit = x;

        /* take the absolute value */
        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

        /* extract the sign bit (upper one) */
        // not 0 if input < 0
        V_ELT_INT sign_bit_int = VAND1_INT(VINTERP_FLOAT_INT(sign_bit), sign_mask, i);

        /* scale by 4/Pi */
        y = VMUL1_FLOAT(x, FOPI, i);

        /* store the integer part of y in mm0 */
        emm2 = VCVT_RTZ_FLOAT_INT(y, i);

        /* j=(j+1) & (~1) (see the cephes sources) */
        emm2 = VADD1_INT(emm2, 1, i);
        emm2 = VAND1_INT(emm2, ~1, i);
        y = VCVT_INT_FLOAT(emm2, i);

        /* get the swap sign flag */
        emm0 = VAND1_INT(emm2, 4, i);
        emm0 = VSLL1_INT(emm0, 29, i);

        /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
         */
        emm2 = VAND1_INT(emm2, 2, i);

        /// emm2 == 0 ? 0xFFFFFFFF : 0x00000000
        V_ELT_BOOL poly_mask = VEQ1_INT_BOOL(emm2, 0, i);

        sign_bit_int = VXOR_INT(sign_bit_int, emm0, i);  // emm0 is swap_sign_bit

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMACC1_FLOAT(x, minus_cephes_DP1, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP2, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT_FLOAT z = VMUL_FLOAT(x, x, i);
        y = VMUL1_FLOAT(z, coscof[0], i);
        y = VADD1_FLOAT(y, coscof[1], i);
        y = VMUL_FLOAT(y, z, i);
        y = VADD1_FLOAT(y, coscof[2], i);
        y = VMUL_FLOAT(y, z, i);
        y = VMUL_FLOAT(y, z, i);
        V_ELT_FLOAT tmp = VMUL1_FLOAT(z, 0.5f, i);
        y = VSUB_FLOAT(y, tmp, i);
        y = VADD1_FLOAT(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT_FLOAT y2;
        y2 = VMUL1_FLOAT(z, sincof[0], i);
        y2 = VADD1_FLOAT(y2, sincof[1], i);
        y2 = VMUL_FLOAT(y2, z, i);
        y2 = VADD1_FLOAT(y2, sincof[2], i);
        y2 = VMUL_FLOAT(y2, z, i);
        y2 = VMUL_FLOAT(y2, x, i);
        y2 = VADD_FLOAT(y2, x, i);

        /* select the correct result from the two polynoms */
        y = VMERGE_FLOAT(poly_mask, y, y2, i);

        /* update the sign */
        y = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(y), sign_bit_int, i));

        VSTORE_FLOAT(dst_tmp, y, i);

        src_tmp += i;
        dst_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

// TODO : could be improved
static inline void cosf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLOAD_FLOAT(src_tmp, i);

        V_ELT_FLOAT y;
        V_ELT_INT j;
        V_ELT_BOOL jandone, jsup3, jsup1, j1or2;
        V_ELT_BOOL sign_cos;

        sign_cos = VCLEAR_BOOL(i);

        /* take the absolute value */
        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

        /* scale by 4/Pi */
        y = VMUL1_FLOAT(x, FOPI, i);

        /* store the integer part of y in mm2 */
        j = VCVT_RTZ_FLOAT_INT(y, i);

        // if (j&1))
        jandone = VNE1_INT_BOOL(VAND1_INT(j, 1, i), 0, i);
        j = VADD1_INT_MASK(jandone, j, j, 1, i);
        y = VCVT_INT_FLOAT(j, i);

        // j&=7
        j = VAND1_INT(j, 7, i);

        // if (j > 3)
        jsup3 = VGT1_INT_BOOL(j, 3, i);
        sign_cos = VXOR_BOOL(sign_cos, jsup3, i);
        j = VSUB1_INT_MASK(jsup3, j, j, 4, i);

        // if (j > 1)
        jsup1 = VGT1_INT_BOOL(j, 1, i);
        sign_cos = VXOR_BOOL(sign_cos, jsup1, i);

        j1or2 = VOR_BOOL(VEQ1_INT_BOOL(j, 1, i),
                           VEQ1_INT_BOOL(j, 2, i), i);

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMACC1_FLOAT(x, minus_cephes_DP1, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP2, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT_FLOAT z = VMUL_FLOAT(x, x, i);
        y = VMUL1_FLOAT(z, coscof[0], i);
        y = VADD1_FLOAT(y, coscof[1], i);
        y = VMUL_FLOAT(y, z, i);
        y = VADD1_FLOAT(y, coscof[2], i);
        y = VMUL_FLOAT(y, z, i);
        y = VMUL_FLOAT(y, z, i);
        V_ELT_FLOAT tmp = VMUL1_FLOAT(z, 0.5f, i);
        y = VSUB_FLOAT(y, tmp, i);
        y = VADD1_FLOAT(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT_FLOAT y2;
        y2 = VMUL1_FLOAT(z, sincof[0], i);
        y2 = VADD1_FLOAT(y2, sincof[1], i);
        y2 = VMUL_FLOAT(y2, z, i);
        y2 = VADD1_FLOAT(y2, sincof[2], i);
        y2 = VMUL_FLOAT(y2, z, i);
        y2 = VMUL_FLOAT(y2, x, i);
        y2 = VADD_FLOAT(y2, x, i);

        /* select the correct result from the two polynoms */
        V_ELT_FLOAT y_cos = VMERGE_FLOAT(j1or2, y, y2, i);
        y_cos = VMUL1_FLOAT_MASK(sign_cos, y_cos, y_cos, -1.0f, i);
        VSTORE_FLOAT(dst_tmp, y_cos, i);

        src_tmp += i;
        dst_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}


#if 1  // should be faster

static inline void sincosf_vec(float *src, float *s, float *c, int len)
{
    size_t i;
    float *src_tmp = src;
    float *s_tmp = s;
    float *c_tmp = c;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    i = VSETVL32H(len);
    V_ELT_FLOATH coscof_1_vec = VLOAD1_FLOATH(coscof[1], i);
    V_ELT_FLOATH coscof_2_vec = VLOAD1_FLOATH(coscof[2], i);
    V_ELT_FLOATH sincof_1_vec = VLOAD1_FLOATH(sincof[1], i);
    V_ELT_FLOATH sincof_2_vec = VLOAD1_FLOATH(sincof[2], i);
    
    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);

        V_ELT_FLOATH y;
        V_ELT_INTH j;
        V_ELT_BOOLH jandone, jsup3, jsup1, j1or2, xinf0;
        V_ELT_BOOLH sign_sin, sign_cos;

        sign_sin = VCLEAR_BOOLH(i);
        sign_cos = VCLEAR_BOOLH(i);

        // if (x < 0)
        xinf0 = VLT1_FLOATH_BOOLH(x, 0.0f, i);
        sign_sin = VXOR_BOOLH(sign_sin, xinf0, i);

        /* take the absolute value */
        x = VINTHERP_INTH_FLOATH(VAND1_INTH(VINTHERP_FLOATH_INTH(x), inv_sign_mask, i));

        /* scale by 4/Pi */
        y = VMUL1_FLOATH(x, FOPI, i);

        /* store the integer part of y in mm2 */
        j = VCVT_RTZ_FLOATH_INTH(y, i);

        // if (j&1))
        jandone = VNE1_INTH_BOOLH(VAND1_INTH(j, 1, i), 0, i);
        j = VADD1_INTH_MASK(jandone, j, j, 1, i);
        y = VCVT_INTH_FLOATH(j, i);

        // j&=7
        j = VAND1_INTH(j, 7, i);

        // if (j > 3)
        jsup3 = VGT1_INTH_BOOLH(j, 3, i);
        sign_sin = VXOR_BOOLH(sign_sin, jsup3, i);
        sign_cos = VXOR_BOOLH(sign_cos, jsup3, i);
        j = VSUB1_INTH_MASK(jsup3, j, j, 4, i);

        // if (j > 1)
        jsup1 = VGT1_INTH_BOOLH(j, 1, i);
        sign_cos = VXOR_BOOLH(sign_cos, jsup1, i);

        j1or2 = VOR_BOOLH(VEQ1_INTH_BOOLH(j, 1, i),
                           VEQ1_INTH_BOOLH(j, 2, i), i);

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMACC1_FLOATH(x, minus_cephes_DP1, y, i);
        x = VFMACC1_FLOATH(x, minus_cephes_DP2, y, i);
        x = VFMACC1_FLOATH(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT_FLOATH z = VMUL_FLOATH(x, x, i);
        y = z;
        y = VFMADD1_FLOATH(y, coscof[0], coscof_1_vec, i);
        y = VFMADD_FLOATH(y, z, coscof_2_vec, i);
        y = VMUL_FLOATH(y, z, i);
        y = VMUL_FLOATH(y, z, i);
        y = VFMACC1_FLOATH(y, -0.5f, z, i);  // y = y -0.5*z
        y = VADD1_FLOATH(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT_FLOATH y2;
        y2 = z;
        y2 = VFMADD1_FLOATH(y2, sincof[0], sincof_1_vec, i);
        y2 = VFMADD_FLOATH(y2, z, sincof_2_vec, i);
        y2 = VMUL_FLOATH(y2, z, i);
        y2 = VFMADD_FLOATH(y2, x, x, i);

        /* select the correct result from the two polynoms */
        V_ELT_FLOATH y_sin = VMERGE_FLOATH(j1or2, y2, y, i);
        V_ELT_FLOATH y_cos = VMERGE_FLOATH(j1or2, y, y2, i);

        y_sin = VMUL1_FLOATH_MASK(sign_sin, y_sin, y_sin, -1.0f, i);
        y_cos = VMUL1_FLOATH_MASK(sign_cos, y_cos, y_cos, -1.0f, i);

        VSTORE_FLOATH(s_tmp, y_sin, i);
        VSTORE_FLOATH(c_tmp, y_cos, i);

        src_tmp += i;
        s_tmp += i;
        c_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

#else
static inline void sincosf_vec(float *src, float *s, float *c, int len)
{
    size_t i;
    float *src_tmp = src;
    float *s_tmp = s;
    float *c_tmp = c;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLOAD_FLOAT(src_tmp, i);

        V_ELT_FLOAT y;
        V_ELT_INT j;
        V_ELT_BOOL jandone, jsup3, jsup1, j1or2, xinf0;
        V_ELT_BOOL sign_sin, sign_cos;

        sign_sin = VCLEAR_BOOL(i);
        sign_cos = VCLEAR_BOOL(i);

        // if (x < 0)
        xinf0 = VLT1_FLOAT_BOOL(x, 0.0f, i);
        sign_sin = VXOR_BOOL(sign_sin, xinf0, i);

        /* take the absolute value */
        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

        /* scale by 4/Pi */
        y = VMUL1_FLOAT(x, FOPI, i);

        /* store the integer part of y in mm2 */
        j = VCVT_RTZ_FLOAT_INT(y, i);

        // if (j&1))
        jandone = VNE1_INT_BOOL(VAND1_INT(j, 1, i), 0, i);
        j = VADD1_INT_MASK(jandone, j, j, 1, i);
        y = VCVT_INT_FLOAT(j, i);

        // j&=7
        j = VAND1_INT(j, 7, i);

        // if (j > 3)
        jsup3 = VGT1_INT_BOOL(j, 3, i);
        sign_sin = VXOR_BOOL(sign_sin, jsup3, i);
        sign_cos = VXOR_BOOL(sign_cos, jsup3, i);
        j = VSUB1_INT_MASK(jsup3, j, j, 4, i);

        // if (j > 1)
        jsup1 = VGT1_INT_BOOL(j, 1, i);
        sign_cos = VXOR_BOOL(sign_cos, jsup1, i);

        j1or2 = VOR_BOOL(VEQ1_INT_BOOL(j, 1, i),
                           VEQ1_INT_BOOL(j, 2, i), i);

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMACC1_FLOAT(x, minus_cephes_DP1, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP2, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT_FLOAT z = VMUL_FLOAT(x, x, i);
        y = VMUL1_FLOAT(z, coscof[0], i);
        y = VADD1_FLOAT(y, coscof[1], i);
        y = VMUL_FLOAT(y, z, i);
        y = VADD1_FLOAT(y, coscof[2], i);
        y = VMUL_FLOAT(y, z, i);
        y = VMUL_FLOAT(y, z, i);
        V_ELT_FLOAT tmp = VMUL1_FLOAT(z, 0.5f, i);
        y = VSUB_FLOAT(y, tmp, i);
        y = VADD1_FLOAT(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT_FLOAT y2;
        y2 = VMUL1_FLOAT(z, sincof[0], i);
        y2 = VADD1_FLOAT(y2, sincof[1], i);
        y2 = VMUL_FLOAT(y2, z, i);
        y2 = VADD1_FLOAT(y2, sincof[2], i);
        y2 = VMUL_FLOAT(y2, z, i);
        y2 = VMUL_FLOAT(y2, x, i);
        y2 = VADD_FLOAT(y2, x, i);

        /* select the correct result from the two polynoms */
        V_ELT_FLOAT y_sin = VMERGE_FLOAT(j1or2, y2, y, i);
        V_ELT_FLOAT y_cos = VMERGE_FLOAT(j1or2, y, y2, i);

        y_sin = VMUL1_FLOAT_MASK(sign_sin, y_sin, y_sin, -1.0f, i);
        y_cos = VMUL1_FLOAT_MASK(sign_cos, y_cos, y_cos, -1.0f, i);

        VSTORE_FLOAT(s_tmp, y_sin, i);
        VSTORE_FLOAT(c_tmp, y_cos, i);

        src_tmp += i;
        s_tmp += i;
        c_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}
#endif


static inline void sumf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    i = VSETVL32(len);
    V_ELT_FLOAT vacc = VLOAD1_FLOAT(0.0f, i);

#if 1
    vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, i);
    size_t i_last;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLOAD_FLOAT(src_tmp, i);
        vacc = VADD_FLOAT(vacc, va, i);
        src_tmp += i;
        i_last = i;
    }

    acc = VREDSUM_FLOAT(acc, vacc, acc, i_last);
    vse32_v_f32m1(dst, acc, 1);
#else
    float acc[32];
    int len_ori = len;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLOAD_FLOAT(src_tmp, i);
        vacc = VADD_FLOAT(vacc, va, i);
        src_tmp += i;
    }

    size_t vlen_ori = VSETVL32(len_ori);
    VSTORE_FLOAT(acc, vacc, len_ori);
    for (int j = 1; j < vlen_ori; j++) {
        acc[0] += acc[j];
    }
    *dst = acc[0];
#endif
}

static inline void meanf_vec(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sumf_vec(src, dst, len);
    *dst *= coeff;
}

static inline void dotf_vec(float *src1, float *src2, int len, float *dst)
{
    size_t i;
    float *src_tmp1 = src1;
    float *src_tmp2 = src2;
    i = VSETVL32(len);
    V_ELT_FLOAT vacc = VLOAD1_FLOAT(0.0f, i);

    vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, i);
    size_t i_last;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLOAD_FLOAT(src_tmp1, i);
        V_ELT_FLOAT vb = VLOAD_FLOAT(src_tmp2, i);
        vacc = VFMACC_FLOAT(vacc, va, vb, i);
        src_tmp1 += i;
        src_tmp2 += i;
        i_last = i;
    }

    acc = VREDSUM_FLOAT(acc, vacc, acc, i_last);
    vse32_v_f32m1(dst, acc, 1);
}

static inline void dotcf_vec(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    size_t i;
    float *src1_tmp = (float *) src1;
    float *src2_tmp = (float *) src2;
    int cplx_len = 2 * len;

    i = VSETVL32(cplx_len);
    V_ELT_FLOATH vacc_Re = VLOAD1_FLOATH(0.0f, i);
    V_ELT_FLOATH vacc_Im = VLOAD1_FLOATH(0.0f, i);

    vfloat32m1_t acc_Re = vfmv_v_f_f32m1(0.0f, i);
    vfloat32m1_t acc_Im = vfmv_v_f_f32m1(0.0f, i);
    size_t i_last;

    int vec_size = VSETVL32(4096);
    int nb_elts = 0;

    for (; (i = VSETVL32(cplx_len)) >= vec_size; cplx_len -= i) {
        V_ELT_FLOATH src1Re_vec;
        V_ELT_FLOATH src1Im_vec;
        V_ELT_FLOATH src2Re_vec;
        V_ELT_FLOATH src2Im_vec;
        VLOAD_FLOATH2(&src1Re_vec, &src1Im_vec, src1_tmp, i);
        VLOAD_FLOATH2(&src2Re_vec, &src2Im_vec, src2_tmp, i);
        V_ELT_FLOATH tmp1 = VMUL_FLOATH(src1Im_vec, src2Im_vec, i);
        V_ELT_FLOATH dstRe_vec = VFMSUB_FLOATH(src1Re_vec, src2Re_vec, tmp1, i);
        V_ELT_FLOATH tmp2 = VMUL_FLOATH(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOATH dstIm_vec = VFMACC_FLOATH(tmp2, src2Re_vec, src1Im_vec, i);
        vacc_Re = VADD_FLOATH(vacc_Re, dstRe_vec, i);
        vacc_Im = VADD_FLOATH(vacc_Im, dstIm_vec, i);
        src1_tmp += i;
        src2_tmp += i;
        i_last = i;
        nb_elts += vec_size;
    }

    acc_Re = VREDSUM_FLOATH(acc_Re, vacc_Re, acc_Re, i_last);
    acc_Im = VREDSUM_FLOATH(acc_Im, vacc_Im, acc_Im, i_last);
    vse32_v_f32m1(&(dst->re), acc_Re, 1);
    vse32_v_f32m1(&(dst->im), acc_Im, 1);

    i = nb_elts / 2;
    for (; i < len; i++) {
        dst->re += src1[i].re * src2[i].re - (src1[i].im * src2[i].im);
        dst->im += src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxtorealf_vec(complex32_t *src, float *dstRe, float *dstIm, int len)
{
    size_t i;
    float *src_tmp = (float *) src;
    float *dstRe_tmp = dstRe;
    float *dstIm_tmp = dstIm;
    int cplx_len = 2 * len;

    for (; (i = VSETVL32(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_FLOATH dstRe_vec;
        V_ELT_FLOATH dstIm_vec;
        VLOAD_FLOATH2(&dstRe_vec, &dstIm_vec, src_tmp, i);
        VSTORE_FLOATH(dstRe_tmp, dstRe_vec, i);
        VSTORE_FLOATH(dstIm_tmp, dstIm_vec, i);
        src_tmp += i;
        dstRe_tmp += i / 2;
        dstIm_tmp += i / 2;
    }
}

static inline void realtocplxf_vec(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
    size_t i;

    float *dst_tmp = (float *) dst;
    float *srcRe_tmp = srcRe;
    float *srcIm_tmp = srcIm;
    int cplx_len = len;

    for (; (i = VSETVL32(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_FLOATH srcRe_vec = VLOAD_FLOATH(srcRe_tmp, i);
        V_ELT_FLOATH srcIm_vec = VLOAD_FLOATH(srcIm_tmp, i);
        VSTORE_FLOATH2(dst_tmp, srcRe_vec, srcIm_vec, i);
        dst_tmp += 2 * i;
        srcRe_tmp += i;
        srcIm_tmp += i;
    }
}

// Work in progress
// We work on m4 instead of m8 in order to use load/store interleaved
static inline void cplxvecmul_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    size_t i;
    float *src1_tmp = (float *) src1;
    float *src2_tmp = (float *) src2;
    float *dst_tmp = (float *) dst;
    int cplx_len = 2 * len;

    int vec_size = VSETVL32(4096);
    int nb_elts = 0;

    for (; (i = VSETVL32(cplx_len)) >= vec_size; cplx_len -= i) {
        V_ELT_FLOATH src1Re_vec;
        V_ELT_FLOATH src1Im_vec;
        V_ELT_FLOATH src2Re_vec;
        V_ELT_FLOATH src2Im_vec;
        VLOAD_FLOATH2(&src1Re_vec, &src1Im_vec, src1_tmp, i);
        VLOAD_FLOATH2(&src2Re_vec, &src2Im_vec, src2_tmp, i);
        V_ELT_FLOATH tmp1 = VMUL_FLOATH(src1Im_vec, src2Im_vec, i);
        V_ELT_FLOATH dstRe_vec = VFMSUB_FLOATH(src1Re_vec, src2Re_vec, tmp1, i);
        V_ELT_FLOATH tmp2 = VMUL_FLOATH(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOATH dstIm_vec = VFMACC_FLOATH(tmp2, src2Re_vec, src1Im_vec, i);

        VSTORE_FLOATH2(dst_tmp, dstRe_vec, dstIm_vec, i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
        nb_elts += vec_size;
    }

    i = nb_elts / 2;
    for (; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxvecmul_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    size_t i;
    float *src1Re_tmp = src1Re;
    float *src1Im_tmp = src1Im;
    float *src2Re_tmp = src2Re;
    float *src2Im_tmp = src2Im;
    float *dstRe_tmp = dstRe;
    float *dstIm_tmp = dstIm;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT src1Re_vec = VLOAD_FLOAT(src1Re_tmp, i);
        V_ELT_FLOAT src1Im_vec = VLOAD_FLOAT(src1Im_tmp, i);
        V_ELT_FLOAT src2Re_vec = VLOAD_FLOAT(src2Re_tmp, i);
        V_ELT_FLOAT src2Im_vec = VLOAD_FLOAT(src2Im_tmp, i);

        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src1Im_vec, src2Im_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMSUB_FLOAT(src1Re_vec, src2Re_vec, tmp1, i);
        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMACC_FLOAT(tmp2, src2Re_vec, src1Im_vec, i);
        VSTORE_FLOAT(dstRe_tmp, dstRe_vec, i);
        VSTORE_FLOAT(dstIm_tmp, dstIm_vec, i);

        src1Re_tmp += i;
        src1Im_tmp += i;
        src2Re_tmp += i;
        src2Im_tmp += i;
        dstRe_tmp += i;
        dstIm_tmp += i;
    }
}

static inline void cplxvecdiv_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    size_t i;
    float *src1_tmp = (float *) src1;
    float *src2_tmp = (float *) src2;
    float *dst_tmp = (float *) dst;
    int cplx_len = 2 * len;

    int vec_size = VSETVL32(4096);
    int nb_elts = 0;

    for (; (i = VSETVL32(cplx_len)) >= vec_size; cplx_len -= i) {
        V_ELT_FLOATH src1Re_vec;
        V_ELT_FLOATH src1Im_vec;
        V_ELT_FLOATH src2Re_vec;
        V_ELT_FLOATH src2Im_vec;
        VLOAD_FLOATH2(&src1Re_vec, &src1Im_vec, src1_tmp, i);
        VLOAD_FLOATH2(&src2Re_vec, &src2Im_vec, src2_tmp, i);

        V_ELT_FLOATH tmp1 = VMUL_FLOATH(src2Re_vec, src2Re_vec, i);
        V_ELT_FLOATH c2d2 = VFMACC_FLOATH(tmp1, src2Im_vec, src2Im_vec, i);

        V_ELT_FLOATH tmp2 = VMUL_FLOATH(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOATH dstRe_vec = VFMACC_FLOATH(tmp2, src1Im_vec, src2Im_vec, i);
        dstRe_vec = VDIV_FLOATH(dstRe_vec, c2d2, i);

        V_ELT_FLOATH tmp3 = VMUL_FLOATH(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOATH dstIm_vec = VFMSUB_FLOATH(src2Re_vec, src1Im_vec, tmp3, i);
        dstIm_vec = VDIV_FLOATH(dstIm_vec, c2d2, i);

        VSTORE_FLOATH2(dst_tmp, dstRe_vec, dstIm_vec, i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
        nb_elts += vec_size;
    }

    i = nb_elts / 2;
    for (; i < len; i++) {
        float c2d2 = src2[i].re * src2[i].re + src2[i].im * src2[i].im;
        dst[i].re = (src1[i].re * src2[i].re + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-src1[i].re * src2[i].im + (src2[i].re * src1[i].im)) / c2d2;
    }
}

static inline void cplxvecdiv_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    size_t i;
    float *src1Re_tmp = src1Re;
    float *src1Im_tmp = src1Im;
    float *src2Re_tmp = src2Re;
    float *src2Im_tmp = src2Im;
    float *dstRe_tmp = dstRe;
    float *dstIm_tmp = dstIm;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT src1Re_vec = VLOAD_FLOAT(src1Re_tmp, i);
        V_ELT_FLOAT src1Im_vec = VLOAD_FLOAT(src1Im_tmp, i);
        V_ELT_FLOAT src2Re_vec = VLOAD_FLOAT(src2Re_tmp, i);
        V_ELT_FLOAT src2Im_vec = VLOAD_FLOAT(src2Im_tmp, i);

        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src2Re_vec, src2Re_vec, i);
        V_ELT_FLOAT c2d2 = VFMACC_FLOAT(tmp1, src2Im_vec, src2Im_vec, i);

        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMACC_FLOAT(tmp2, src1Im_vec, src2Im_vec, i);
        dstRe_vec = VDIV_FLOAT(dstRe_vec, c2d2, i);

        V_ELT_FLOAT tmp3 = VMUL_FLOAT(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMSUB_FLOAT(src2Re_vec, src1Im_vec, tmp3, i);
        dstIm_vec = VDIV_FLOAT(dstIm_vec, c2d2, i);
        VSTORE_FLOAT(dstRe_tmp, dstRe_vec, i);
        VSTORE_FLOAT(dstIm_tmp, dstIm_vec, i);

        src1Re_tmp += i;
        src1Im_tmp += i;
        src2Re_tmp += i;
        src2Im_tmp += i;
        dstRe_tmp += i;
        dstIm_tmp += i;
    }
}

static inline void cplxconjf_vec(complex32_t *src, complex32_t *dst, int len)
{
    size_t i;
    float *src_tmp = (float *) src;
    float *dst_tmp = (float *) dst;
    int cplx_len = 2 * len;

    int vec_size = VSETVL32(4096);
    int nb_elts = 0;

    for (; (i = VSETVL32(cplx_len)) >= vec_size; cplx_len -= i) {
        V_ELT_FLOATH srcRe_vec;
        V_ELT_FLOATH srcIm_vec;
        VLOAD_FLOATH2(&srcRe_vec, &srcIm_vec, src_tmp, i);
        srcIm_vec = VINTERP_INTH_FLOATH(VXOR1_INTH(VINTERP_FLOATH_INTH(srcIm_vec), (int32_t) 0x80000000, i));
        VSTORE_FLOATH2(dst_tmp, srcRe_vec, srcIm_vec, i);
        src_tmp += i;
        dst_tmp += i;
        nb_elts += vec_size;
    }

    i = nb_elts / 2;
    for (; i < len; i++) {
        dst[i].re = src[i].re;
        dst[i].im = -src[i].im;
    }
}

static inline void magnitudef_split_vec(float *srcRe, float *srcIm, float *dst, int len)
{
    size_t i;
    float *srcRe_tmp = srcRe;
    float *srcIm_tmp = srcIm;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT re_tmp = VLOAD_FLOAT(srcRe_tmp, i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT im_tmp = VLOAD_FLOAT(srcIm_tmp, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, im_tmp, im_tmp, i);

        VSTORE_FLOAT(dst_tmp, VSQRT_FLOAT(tmp, i), i);

        srcRe_tmp += i;
        srcIm_tmp += i;
        dst_tmp += i;
    }
}

static inline void powerspectf_split_vec(float *srcRe, float *srcIm, float *dst, int len)
{
    size_t i;
    float *srcRe_tmp = srcRe;
    float *srcIm_tmp = srcIm;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT re_tmp = VLOAD_FLOAT(srcRe_tmp, i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT im_tmp = VLOAD_FLOAT(srcIm_tmp, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, im_tmp, im_tmp, i);

        VSTORE_FLOAT(dst_tmp, tmp, i);

        srcRe_tmp += i;
        srcIm_tmp += i;
        dst_tmp += i;
    }
}

static inline void powerspectf_interleaved_vec(complex32_t *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = (float *) src;
    float *dst_tmp = dst;

#if 1
    int cplx_len = 2 * len;

    for (; (i = VSETVL32(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_FLOATH dstRe_vec;
        V_ELT_FLOATH dstIm_vec;
        VLOAD_FLOATH2(&dstRe_vec, &dstIm_vec, src_tmp, i);
        V_ELT_FLOATH re2 = VMUL_FLOATH(dstRe_vec, dstRe_vec, i);
        V_ELT_FLOATH tmp = VFMACC_FLOATH(re2, dstIm_vec, dstIm_vec, i);
        VSTORE_FLOATH(dst_tmp, tmp, i);
        src_tmp += i;
        dst_tmp += i / 2;
    }
#else

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        // complex are a + ib, c + id, e + if, etc
        // load in re_tmp a,c,e, etc => i elements in range 0..2*i with a stride of 2
        // load in im_tmp b,d,f, etc => i elements in range 0..2*i with a stride of 2
        V_ELT_FLOAT re_tmp = VLE_FLOAT_STRIDE(src_tmp, 2 * sizeof(float), i);
        V_ELT_FLOAT im_tmp = VLE_FLOAT_STRIDE(src_tmp + 1, 2 * sizeof(float), i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, im_tmp, im_tmp, i);
        VSTORE_FLOAT(dst_tmp, tmp, i);

        // src_tmp increases twice as fast since it's complex and not float
        src_tmp += 2 * i;
        dst_tmp += i;
    }
#endif
}

static inline void magnitudef_interleaved_vec(complex32_t *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = (float *) src;
    float *dst_tmp = dst;

#if 1
    int cplx_len = 2 * len;

    for (; (i = VSETVL32(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_FLOATH dstRe_vec;
        V_ELT_FLOATH dstIm_vec;
        VLOAD_FLOATH2(&dstRe_vec, &dstIm_vec, src_tmp, i);
        V_ELT_FLOATH re2 = VMUL_FLOATH(dstRe_vec, dstRe_vec, i);
        V_ELT_FLOATH tmp = VFMACC_FLOATH(re2, dstIm_vec, dstIm_vec, i);
        tmp = VSQRT_FLOATH(tmp, i);
        VSTORE_FLOATH(dst_tmp, tmp, i);
        src_tmp += i;
        dst_tmp += i / 2;
    }
#else
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        // complex are a + ib, c + id, e + if, etc
        // load in re_tmp a,c,e, etc => i elements in range 0..2*i with a stride of 2
        // load in im_tmp b,d,f, etc => i elements in range 0..2*i with a stride of 2
        V_ELT_FLOAT re_tmp = VLE_FLOAT_STRIDE(src_tmp, 2 * sizeof(float), i);
        V_ELT_FLOAT im_tmp = VLE_FLOAT_STRIDE(src_tmp + 1, 2 * sizeof(float), i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, im_tmp, im_tmp, i);
        VSTORE_FLOAT(dst_tmp, VSQRT_FLOAT(tmp, i), i);

        // src_tmp increases twice as fast since it's complex and not float
        src_tmp += 2 * i;
        dst_tmp += i;
    }
#endif
}

static inline void maxeveryf_vec(float *src1, float *src2, float *dst, int len)
{
    size_t i;
    float *src1_tmp = src1;
    float *src2_tmp = src2;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb;
        va = VLOAD_FLOAT(src1_tmp, i);
        vb = VLOAD_FLOAT(src2_tmp, i);
        VSTORE_FLOAT(dst_tmp, VMAX_FLOAT(va, vb, i), i);

        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void mineveryf_vec(float *src1, float *src2, float *dst, int len)
{
    size_t i;
    float *src1_tmp = src1;
    float *src2_tmp = src2;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb;
        va = VLOAD_FLOAT(src1_tmp, i);
        vb = VLOAD_FLOAT(src2_tmp, i);
        VSTORE_FLOAT(dst_tmp, VMIN_FLOAT(va, vb, i), i);

        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void minmaxf_vec(float *src, int len, float *min_value, float *max_value)
{
    size_t i, i_last;

    float *src_tmp = src;

    i = VSETVL32(len);

    vfloat32m1_t min0 = vle32_v_f32m1(src_tmp, 1);  // or vfmv_v_f_f32m1
    vfloat32m1_t max0 = min0;
    V_ELT_FLOAT minv, maxv, v1;
    minv = VLOAD_FLOAT(src_tmp, i);
    maxv = minv;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        v1 = VLOAD_FLOAT(src_tmp, i);
        minv = VMIN_FLOAT(v1, minv, i);
        maxv = VMAX_FLOAT(v1, maxv, i);
        src_tmp += i;
        i_last = i;
    }
    min0 = VREDMIN_FLOAT(min0, minv, min0, i_last);
    max0 = VREDMAX_FLOAT(max0, maxv, max0, i_last);
    vse32_v_f32m1(min_value, min0, 1);
    vse32_v_f32m1(max_value, max0, 1);
}

static inline void threshold_gt_f_vec(float *src, float *dst, int len, float value)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src_tmp, i);
        VSTORE_FLOAT(dst_tmp, VMIN1_FLOAT(va, value, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_lt_f_vec(float *src, float *dst, int len, float value)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src_tmp, i);
        VSTORE_FLOAT(dst_tmp, VMAX1_FLOAT(va, value, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_gtabs_f_vec(float *src, float *dst, int len, float value)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLOAD_FLOAT(src_tmp, i);

#if 1
        V_ELT_INT va_sign = VAND1_INT(VINTERP_FLOAT_INT(va), sign_mask, i);
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        V_ELT_FLOAT sval = VMIN1_FLOAT(va_abs, value, i);
        sval = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(sval), va_sign, i));
        VSTORE_FLOAT(dst_tmp, sval, i);
#else  // should be removed?
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        V_ELT_BOOL eqmask = VEQ_FLOAT_BOOL(va, va_abs, i);
        V_ELT_BOOL gtmask = VGT1_FLOAT_BOOL(va_abs, value, i);

        V_ELT_FLOAT sval;
        sval = VMERGE1_FLOAT(VNOT_BOOL(eqmask, i), sval, -value, i);
        sval = VMERGE1_FLOAT(eqmask, sval, value, i);
        VSTORE_FLOAT(dst_tmp, VMERGE_FLOAT(gtmask, va, sval, i), i);
#endif
        src_tmp += i;
        dst_tmp += i;
    }
}

/*
            v4sf src_tmp = _mm_load_ps(src + i);
            v4sf src_sign = _mm_and_ps(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
            v4sf src_abs = _mm_and_ps(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
            v4sf dst_tmp = _mm_max_ps(src_abs, pval);
            dst_tmp = _mm_xor_ps(dst_tmp, src_sign);
            _mm_store_ps(dst + i, dst_tmp);
*/
static inline void threshold_ltabs_f_vec(float *src, float *dst, int len, float value)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLOAD_FLOAT(src_tmp, i);
        V_ELT_INT va_sign = VAND1_INT(VINTERP_FLOAT_INT(va), sign_mask, i);
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        V_ELT_FLOAT sval = VMAX1_FLOAT(va_abs, value, i);
        sval = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(sval), va_sign, i));
        VSTORE_FLOAT(dst_tmp, sval, i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_ltval_gtval_f_vec(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLOAD_FLOAT(src_tmp, i);
        V_ELT_BOOL lt_mask = VLT1_FLOAT_BOOL(va, ltlevel, i);
        V_ELT_BOOL gt_mask = VGT1_FLOAT_BOOL(va, gtlevel, i);
        V_ELT_FLOAT tmp = VMERGE1_FLOAT(lt_mask, va, ltvalue, i);
        tmp = VMERGE1_FLOAT(gt_mask, tmp, gtvalue, i);
        VSTORE_FLOAT(dst_tmp, tmp, i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void sqrtf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src_tmp, i);
        VSTORE_FLOAT(dst_tmp, VSQRT_FLOAT(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void fabsf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src_tmp, i);
        VSTORE_FLOAT(dst_tmp, VABS_FLOAT(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

#if 1 // should be faster

static inline void log10_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);
    V_ELT_FLOATH zero_vec = VLOAD1_FLOATH(0.0f, i);
    V_ELT_FLOATH c_cephes_log_p1_vec = VLOAD1_FLOATH(c_cephes_log_p1, i);
    V_ELT_FLOATH c_cephes_log_p2_vec = VLOAD1_FLOATH(c_cephes_log_p2, i);
    V_ELT_FLOATH c_cephes_log_p3_vec = VLOAD1_FLOATH(c_cephes_log_p3, i);
    V_ELT_FLOATH c_cephes_log_p4_vec = VLOAD1_FLOATH(c_cephes_log_p4, i);
    V_ELT_FLOATH c_cephes_log_p5_vec = VLOAD1_FLOATH(c_cephes_log_p5, i);
    V_ELT_FLOATH c_cephes_log_p6_vec = VLOAD1_FLOATH(c_cephes_log_p6, i);
    V_ELT_FLOATH c_cephes_log_p7_vec = VLOAD1_FLOATH(c_cephes_log_p7, i);
    V_ELT_FLOATH c_cephes_log_p8_vec = VLOAD1_FLOATH(c_cephes_log_p8, i);
    
    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        V_ELT_INTH imm0;

        V_ELT_BOOLH invalid_mask = VLE1_FLOATH_BOOLH(x, 0.0f, i);
        x = VMAX1_FLOATH(x, 1.17549e-38f, i); /* cut off denormalized stuff */
        imm0 = VSRA1_INTH(VINTHERP_FLOATH_INTH(x), 23, i);

        /* keep only the fractional part */
        x = VINTHERP_INTH_FLOATH(VAND1_INTH(VINTHERP_FLOATH_INTH(x), c_inv_mant_mask, i));
        // 0x3f000000 is the hex representation of 0.5f
        x = VINTHERP_INTH_FLOATH(VOR1_INTH(VINTHERP_FLOATH_INTH(x), 0x3f000000, i));
        imm0 = VSUB1_INTH(imm0, 0x7f, i);
        V_ELT_FLOATH e = VCVT_INTH_FLOATH(imm0, i);
        e = VADD1_FLOATH(e, 1.0f, i);

        // could lead to errors since we take the inverted mask after?
        V_ELT_BOOLH mask = VLT1_FLOATH_BOOLH(x, c_cephes_SQRTHF, i);

        V_ELT_FLOATH tmp = VMERGE1_FLOATH(VNOT_BOOLH(mask, i), x, 0.0f, i);
        x = VSUB1_FLOATH(x, 1.0f, i);  // x ok

        // substract 1.0f if mask is true (x < SQRTHF). To be optimised
        e = VSUB_FLOATH(e, VMERGE1_FLOATH(mask, zero_vec, 1.0f, i), i);
        x = VADD_FLOATH(x, tmp, i);

        V_ELT_FLOATH z = VMUL_FLOATH(x, x, i);
        V_ELT_FLOATH y = x;
        y = VFMADD1_FLOATH(y, c_cephes_log_p0, c_cephes_log_p1_vec, i);
        y = VFMADD_FLOATH(y, x, c_cephes_log_p2_vec, i);
        y = VFMADD_FLOATH(y, x, c_cephes_log_p3_vec, i);
        y = VFMADD_FLOATH(y, x, c_cephes_log_p4_vec, i);
        y = VFMADD_FLOATH(y, x, c_cephes_log_p5_vec, i);
        y = VFMADD_FLOATH(y, x, c_cephes_log_p6_vec, i);
        y = VFMADD_FLOATH(y, x, c_cephes_log_p7_vec, i);
        y = VFMADD_FLOATH(y, x, c_cephes_log_p8_vec, i);
        y = VMUL_FLOATH(y, x, i);
        y = VMUL_FLOATH(y, z, i);
        y = VFMACC1_FLOATH(y, -0.5f, z, i);  // y = y -0.5*z

        tmp = VADD_FLOATH(x, y, i);
        z = VMUL1_FLOATH(tmp, c_cephes_L10EB, i);
        V_ELT_FLOATH tmp2 = VMUL1_FLOATH(y, c_cephes_L10EA, i);
        z = VADD_FLOATH(z, tmp2, i);
        tmp2 = VMUL1_FLOATH(x, c_cephes_L10EA, i);
        z = VADD_FLOATH(z, tmp2, i);
        tmp2 = VMUL1_FLOATH(e, c_cephes_L102B, i);
        z = VADD_FLOATH(z, tmp2, i);
        tmp2 = VMUL1_FLOATH(e, c_cephes_L102A, i);
        x = VADD_FLOATH(z, tmp2, i);
        
        // print_vec(x);printf("\n");
        // could we use merge function? VMERGE_FLOATH? create a nan vec?
        x = VMERGE1_FLOATH(invalid_mask, x, 0xFFFFFFFF, i);

        VSTORE_FLOATH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }
}
#else

static inline void log10_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32(len);
    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLOAD_FLOAT(src_tmp, i);
        V_ELT_INT imm0;

        V_ELT_BOOL invalid_mask = VLE1_FLOAT_BOOL(x, 0.0f, i);
        x = VMAX1_FLOAT(x, 1.17549e-38f, i); /* cut off denormalized stuff */
        imm0 = VSRA1_INT(VINTERP_FLOAT_INT(x), 23, i);

        /* keep only the fractional part */
        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), c_inv_mant_mask, i));
        // 0x3f000000 is the hex representation of 0.5f
        x = VINTERP_INT_FLOAT(VOR1_INT(VINTERP_FLOAT_INT(x), 0x3f000000, i));
        imm0 = VSUB1_INT(imm0, 0x7f, i);
        V_ELT_FLOAT e = VCVT_INT_FLOAT(imm0, i);
        e = VADD1_FLOAT(e, 1.0f, i);

        // could lead to errors since we take the inverted mask after?
        V_ELT_BOOL mask = VLT1_FLOAT_BOOL(x, c_cephes_SQRTHF, i);

        V_ELT_FLOAT tmp = VMERGE1_FLOAT(VNOT_BOOL(mask, i), x, 0.0f, i);
        x = VSUB1_FLOAT(x, 1.0f, i);  // x ok

        // substract 1.0f if mask is true (x < SQRTHF). To be optimised
        e = VSUB_FLOAT(e, VMERGE1_FLOAT(mask, zero_vec, 1.0f, i), i);
        x = VADD_FLOAT(x, tmp, i);

        V_ELT_FLOAT z = VMUL_FLOAT(x, x, i);
        V_ELT_FLOAT y = VMUL1_FLOAT(x, c_cephes_log_p0, i);
        y = VADD1_FLOAT(y, c_cephes_log_p1, i);
        y = VMUL_FLOAT(y, x, i);
        y = VADD1_FLOAT(y, c_cephes_log_p2, i);
        y = VMUL_FLOAT(y, x, i);
        y = VADD1_FLOAT(y, c_cephes_log_p3, i);
        y = VMUL_FLOAT(y, x, i);
        y = VADD1_FLOAT(y, c_cephes_log_p4, i);
        y = VMUL_FLOAT(y, x, i);
        y = VADD1_FLOAT(y, c_cephes_log_p5, i);
        y = VMUL_FLOAT(y, x, i);
        y = VADD1_FLOAT(y, c_cephes_log_p6, i);
        y = VMUL_FLOAT(y, x, i);
        y = VADD1_FLOAT(y, c_cephes_log_p7, i);
        y = VMUL_FLOAT(y, x, i);
        y = VADD1_FLOAT(y, c_cephes_log_p8, i);
        y = VMUL_FLOAT(y, x, i);
        y = VMUL_FLOAT(y, z, i);
        y = VFMADD1_FLOAT(z, -0.5f, y, i);  // y = y -0.5*z

        tmp = VADD_FLOAT(x, y, i);
        z = VMUL1_FLOAT(tmp, c_cephes_L10EB, i);
        V_ELT_FLOAT tmp2 = VMUL1_FLOAT(y, c_cephes_L10EA, i);
        z = VADD_FLOAT(z, tmp2, i);
        tmp2 = VMUL1_FLOAT(x, c_cephes_L10EA, i);
        z = VADD_FLOAT(z, tmp2, i);
        tmp2 = VMUL1_FLOAT(e, c_cephes_L102B, i);
        z = VADD_FLOAT(z, tmp2, i);
        tmp2 = VMUL1_FLOAT(e, c_cephes_L102A, i);
        x = VADD_FLOAT(z, tmp2, i);
        // print_vec(x);printf("\n");
        // could we use merge function? VMERGE_FLOAT? create a nan vec?
        x = VMERGE1_FLOAT(invalid_mask, x, 0xFFFFFFFF, i);

        VSTORE_FLOAT(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

#endif

static inline void vectorSlopef_vec(float *dst, int len, float offset, float slope)
{
    size_t i;
    float *dst_tmp = dst;

    float coef_max[32];

    // to be improved!
    for (int s = 0; s < 32; s++) {
        coef_max[s] = (float) (s) *slope;
    }

    i = VSETVL32(len);

    V_ELT_FLOAT coef = VLOAD_FLOAT(coef_max, i);
    V_ELT_FLOAT slope_vec = VLOAD1_FLOAT((float) (i) *slope, i);
    V_ELT_FLOAT curVal = VADD1_FLOAT(coef, offset, i);

    VSTORE_FLOAT(dst_tmp, curVal, i);
    dst_tmp += i;
    len -= i;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        curVal = VADD_FLOAT(curVal, slope_vec, i);
        VSTORE_FLOAT(dst_tmp, curVal, i);
        dst_tmp += i;
    }
}

static inline void setf_vec(float *dst, float value, int len)
{
    size_t i;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        VSTORE_FLOAT(dst_tmp, VLOAD1_FLOAT(value, i), i);
        dst_tmp += i;
    }
}

static inline void zerof_vec(float *dst, int len)
{
    setf_vec(dst, 0.0f, len);
}

static inline void copyf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        VSTORE_FLOAT(dst_tmp, VLOAD_FLOAT(src_tmp, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void modf_vec(float *src, float *integer, float *remainder, int len)
{
    size_t i;
    float *src_tmp = src;
    float *integer_tmp = integer;
    float *remainder_tmp = remainder;

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT src_vec, integer_vec, remainer_vec;
        src_vec = VLOAD_FLOAT(src_tmp, i);
        integer_vec = VCVT_INT_FLOAT(VCVT_RTZ_FLOAT_INT(src_vec, i), i);
        VSTORE_FLOAT(integer_tmp, integer_vec, i);
        remainer_vec = VSUB_FLOAT(src_vec, integer_vec, i);
        VSTORE_FLOAT(remainder_tmp, remainer_vec, i);
        src_tmp += i;
        integer_tmp += i;
        remainder_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void roundf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT a, b;
        a = VLOAD_FLOAT(src_tmp, i);
        b = VCVT_INT_FLOAT(VCVT_FLOAT_INT(a, i), i);
        VSTORE_FLOAT(dst_tmp, b, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void ceilf_vec(float *src, float *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    roundf_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void floorf_vec(float *src, float *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    roundf_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void truncf_vec(float *src, float *dst, int len)
{
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    roundf_vec(src, dst, len);
    _MM_SET_ROUNDING_MODE(reg_ori);
}

#if 1  // should be a better version
static inline void flipf_vec(float *src, float *dst, int len)
{
    size_t i, i_last;
    i = VSETVL32(len);
    int vec_size = VSETVL32(4096);
    float *src_tmp = src + len - i;
    float *dst_tmp = dst;

    // max vector size is 1024bits, but could be less (128bits on C906 core)
    uint32_t index[32] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
                          19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                          6, 5, 4, 3, 2, 1, 0};

    V_ELT_UINT index_vec = VLOAD_UINT(index + 32 - vec_size, i);
    V_ELT_FLOAT a, b;
    for (; (i = VSETVL32(len)) >= vec_size; len -= i) {
        a = VLOAD_FLOAT(src_tmp, i);
        b = VGATHER_FLOAT(a, index_vec, i);
        VSTORE_FLOAT(dst_tmp, b, i);
        src_tmp -= i;
        dst_tmp += i;
        i_last = i;
    }

    if (i_last) {
        index_vec = VLOAD_UINT(index + 32 - i_last, i_last);
        a = VLOAD_FLOAT(src_tmp, i_last);
        b = VGATHER_FLOAT(a, index_vec, i_last);
        VSTORE_FLOAT(dst_tmp, b, i_last);
    }
}
#else
static inline void flipf_vec(float *src, float *dst, int len)
{
    size_t i;
    int j = len;
    int len_ori = len;
    i = vsetvl_e32m2(len);
    int vec_size = i;
    float *src_tmp = src + len - i;
    float *dst_tmp = dst;

    // max vector size is 1024bits, but could be less (128bits on C906 core)
    uint32_t index[32] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
                          19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                          6, 5, 4, 3, 2, 1, 0};

    vuint32m2_t index_vec = vle32_v_u32m2(index + 32 - vec_size, i);
    for (; (i = vsetvl_e32m2(len)) >= vec_size; len -= i) {
        vfloat32m2_t a = vle32_v_f32m2(src_tmp, i);
        vfloat32m2_t b = vrgather_vv_f32m2(a, index_vec, i);
        vse32_v_f32m2(dst_tmp, b, i);
        j = (int) len;
        src_tmp -= i;
        dst_tmp += i;
    }
    j -= vec_size;
    for (; j >= 0; j--) {
        dst[len_ori - j - 1] = src[j];
    }
}
#endif

static inline void convert_32f64f_vec(float *src, double *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    double *dst_tmp = dst;

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        VSTORE_DOUBLE(dst_tmp, VCVT_FLOAT_DOUBLE(VLOAD_FLOATH(src_tmp, i), i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void convert_64f32f_vec(double *src, float *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL64(len)) > 0; len -= i) {
        VSTORE_FLOATH(dst_tmp, VCVT_DOUBLE_FLOAT(VLOAD_DOUBLE(src_tmp, i), i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}
