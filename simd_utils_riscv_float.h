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

static inline void subcrevf_vec(float *src, float value, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src_tmp, i);
        VSTORE_FLOAT(dst_tmp, VRSUB1_FLOAT(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
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

// TODO : could be improved with FMA
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
        V_ELT_BOOL32 poly_mask = VEQ1_INT_BOOL(emm2, 0, i);

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

    i = VSETVL32H(len);
    V_ELT_FLOATH c_coscof_p1_vec = VLOAD1_FLOATH(c_coscof_p1, i);
    V_ELT_FLOATH c_coscof_p2_vec = VLOAD1_FLOATH(c_coscof_p2, i);
    V_ELT_FLOATH c_sincof_p1_vec = VLOAD1_FLOATH(c_sincof_p1, i);
    V_ELT_FLOATH c_sincof_p2_vec = VLOAD1_FLOATH(c_sincof_p2, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH xmm3, y;

        V_ELT_INTH emm0, emm2;

        /* take the absolute value */
        x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(x), inv_sign_mask, i));

        /* scale by 4/Pi */
        y = VMUL1_FLOATH(x, FOPI, i);

        /* store the integer part of y in mm0 */
        emm2 = VCVT_FLOATH_INTH(y, i);
        /* j=(j+1) & (~1) (see the cephes sources) */
        emm2 = VADD1_INTH(emm2, 1, i);
        emm2 = VAND1_INTH(emm2, ~1, i);
        y = VCVT_INTH_FLOATH(emm2, i);

        emm2 = VSUB1_INTH(emm2, 2, i);

        /* get the swap sign flag */
        emm0 = VAND1_INTH(VNOT_INTH(emm2, i), 4, i);
        emm0 = VSLL1_INTH(emm0, 29, i);
        /* get the polynom selection mask */
        emm2 = VAND1_INTH(emm2, 2, i);
        V_ELT_BOOL32H poly_mask = VEQ1_INTH_BOOLH(emm2, 0, i);

        /* The magic pass: "Extended precision modular arithmetic"
         x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMACC1_FLOATH(x, minus_cephes_DP1, y, i);
        x = VFMACC1_FLOATH(x, minus_cephes_DP2, y, i);
        x = VFMACC1_FLOATH(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT_FLOATH z = VMUL_FLOATH(x, x, i);

        y = z;
        y = VFMADD1_FLOATH(y, c_coscof_p0, c_coscof_p1_vec, i);
        y = VFMADD_FLOATH(y, z, c_coscof_p2_vec, i);
        y = VMUL_FLOATH(y, z, i);
        y = VMUL_FLOATH(y, z, i);
        y = VFMACC1_FLOATH(y, -0.5f, z, i);  // y = y -0.5*z
        y = VADD1_FLOATH(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT_FLOATH y2 = z;
        y2 = VFMADD1_FLOATH(y2, c_sincof_p0, c_sincof_p1_vec, i);
        y2 = VFMADD_FLOATH(y2, z, c_sincof_p2_vec, i);
        y2 = VMUL_FLOATH(y2, z, i);
        y2 = VFMADD_FLOATH(y2, x, x, i);

        /* select the correct result from the two polynoms */
        y = VMERGE_FLOATH(poly_mask, y, y2, i);

        /* update the sign */
        y = VINTERP_INTH_FLOATH(VXOR_INTH(VINTERP_FLOATH_INTH(y), emm0, i));
        VSTORE_FLOATH(dst_tmp, y, i);
        src_tmp += i;
        dst_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

static inline void sincosf_ps(V_ELT_FLOATH x,
                              V_ELT_FLOATH *sin_tmp,
                              V_ELT_FLOATH *cos_tmp,
                              V_ELT_FLOATH coscof_1_vec,
                              V_ELT_FLOATH coscof_2_vec,
                              V_ELT_FLOATH sincof_1_vec,
                              V_ELT_FLOATH sincof_2_vec,
                              size_t i)
{
    V_ELT_FLOATH y;
    V_ELT_INTH j;
    V_ELT_BOOL32H jandone, jsup3, jsup1, j1or2, xinf0;
    V_ELT_BOOL32H sign_sin, sign_cos;

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
    *sin_tmp = y_sin;
    *cos_tmp = y_cos;
}

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
        V_ELT_FLOATH y_sin, y_cos;
        sincosf_ps(x, &y_sin, &y_cos,
                   coscof_1_vec, coscof_2_vec,
                   sincof_1_vec, sincof_2_vec, i);
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

static inline void sincosf_interleaved_vec(float *src, complex32_t *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = (float *) dst;
    int cplx_len = len;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    i = VSETVL32H(cplx_len);
    V_ELT_FLOATH coscof_1_vec = VLOAD1_FLOATH(coscof[1], i);
    V_ELT_FLOATH coscof_2_vec = VLOAD1_FLOATH(coscof[2], i);
    V_ELT_FLOATH sincof_1_vec = VLOAD1_FLOATH(sincof[1], i);
    V_ELT_FLOATH sincof_2_vec = VLOAD1_FLOATH(sincof[2], i);

    for (; (i = VSETVL32H(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH y_sin, y_cos;
        sincosf_ps(x, &y_sin, &y_cos,
                   coscof_1_vec, coscof_2_vec,
                   sincof_1_vec, sincof_2_vec, i);
        VSTORE_FLOATH2(dst_tmp, y_cos, y_sin, i);
        dst_tmp += 2 * i;
        src_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

#if 0  // old version with no FMA. Waiting for real HW benchmark to remove
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
        V_ELT_BOOL32 jandone, jsup3, jsup1, j1or2, xinf0;
        V_ELT_BOOL32 sign_sin, sign_cos;

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

static inline void tanf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    i = VSETVL32H(len);
    V_ELT_FLOATH TAN_P1_vec = VLOAD1_FLOATH(TAN_P1, i);
    V_ELT_FLOATH TAN_P2_vec = VLOAD1_FLOATH(TAN_P2, i);
    V_ELT_FLOATH TAN_P3_vec = VLOAD1_FLOATH(TAN_P3, i);
    V_ELT_FLOATH TAN_P4_vec = VLOAD1_FLOATH(TAN_P4, i);
    V_ELT_FLOATH TAN_P5_vec = VLOAD1_FLOATH(TAN_P5, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH xx = VLOAD_FLOATH(src_tmp, i);

        V_ELT_FLOATH x, y, z, zz;
        V_ELT_INTH j;
        V_ELT_INTH sign;
        V_ELT_FLOATH tmp;
        V_ELT_INTH tmpi;
        V_ELT_BOOL32H jandone, jandtwo, xsupem4;

        x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(xx), inv_sign_mask, i));
        sign = VAND1_INTH(VINTERP_FLOATH_INTH(xx), sign_mask, i);

        // compute x mod PIO4
        tmp = VMUL1_FLOATH(x, FOPI, i);
        j = VCVT_FLOATH_INTH(tmp, i);
        y = VCVT_INTH_FLOATH(j, i);

        jandone = VGT1_INTH_BOOLH(VAND1_INTH(j, 1, i), 0, i);

        // TODO : could it be improved like X86 (replace add and merge with and and and add)?
        tmp = VADD1_FLOATH(y, 1.0f, i);
        y = VMERGE_FLOATH(jandone, y, tmp, i);
        tmpi = VADD1_INTH(j, 1, i);
        j = VMERGE_INTH(jandone, j, tmpi, i);
        z = x;
        z = VFMACC1_FLOATH(z, minus_cephes_DP1, y, i);
        z = VFMACC1_FLOATH(z, minus_cephes_DP2, y, i);
        z = VFMACC1_FLOATH(z, minus_cephes_DP3, y, i);
        zz = VMUL_FLOATH(z, z, i);  // z*z

        // TODO : should not be computed if X < 10e-4
        // 1.7e-8 relative error in [-pi/4, +pi/4]
        tmp = zz;
        tmp = VFMADD1_FLOATH(tmp, TAN_P0, TAN_P1_vec, i);
        tmp = VFMADD_FLOATH(tmp, zz, TAN_P2_vec, i);
        tmp = VFMADD_FLOATH(tmp, zz, TAN_P3_vec, i);
        tmp = VFMADD_FLOATH(tmp, zz, TAN_P4_vec, i);
        tmp = VFMADD_FLOATH(tmp, zz, TAN_P5_vec, i);
        tmp = VMUL_FLOATH(zz, tmp, i);

        tmp = VFMADD_FLOATH(tmp, z, z, i);
        xsupem4 = VGT1_FLOATH_BOOLH(x, 1e-4f, i);
        y = VMERGE_FLOATH(xsupem4, z, tmp, i);

        jandtwo = VGT1_INTH_BOOLH(VAND1_INTH(j, 2, i), 0, i);

        // xor(rcp(y)) gives not good enough result
        tmp = VRDIV1_FLOATH(y, -1.0f, i);
        y = VMERGE_FLOATH(jandtwo, y, tmp, i);
        y = VINTERP_INTH_FLOATH(VXOR_INTH(VINTERP_FLOATH_INTH(y), sign, i));
        VSTORE_FLOATH(dst_tmp, y, i);
        src_tmp += i;
        dst_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

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
    float acc[MAX_ELTS32];
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
    int cplx_len = len;

    for (; (i = VSETVL32H(cplx_len)) > 0; cplx_len -= i) {
        V_ELT_FLOATH dstRe_vec;
        V_ELT_FLOATH dstIm_vec;
        VLOAD_FLOATH2(&dstRe_vec, &dstIm_vec, src_tmp, i);
        VSTORE_FLOATH(dstRe_tmp, dstRe_vec, i);
        VSTORE_FLOATH(dstIm_tmp, dstIm_vec, i);
        src_tmp += 2 * i;
        dstRe_tmp += i;
        dstIm_tmp += i;
    }
}

static inline void realtocplxf_vec(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
    size_t i;

    float *dst_tmp = (float *) dst;
    float *srcRe_tmp = srcRe;
    float *srcIm_tmp = srcIm;
    int cplx_len = len;

    for (; (i = VSETVL32H(cplx_len)) > 0; cplx_len -= i) {
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
static inline void cplxvecmulf_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
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

static inline void cplxvecmulf_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
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

static inline void cplxvecdivf_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
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

static inline void cplxvecdivf_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
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

static inline void cplxconjvecmulf_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
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
        V_ELT_FLOATH dstRe_vec = VFMACC_FLOATH(tmp1, src1Re_vec, src2Re_vec, i);
        V_ELT_FLOATH tmp2 = VMUL_FLOATH(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOATH dstIm_vec = VFMSUB_FLOATH(src2Re_vec, src1Im_vec, tmp2, i);  // vs1*vd - vs2
        VSTORE_FLOATH2(dst_tmp, dstRe_vec, dstIm_vec, i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
        nb_elts += vec_size;
    }

    i = nb_elts / 2;
    for (; i < len; i++) {
        dst[i].re = src1[i].re * src2[i].re + (src1[i].im * src2[i].im);
        dst[i].im = -src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxconjvecmulf_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
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
        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMACC_FLOAT(tmp1, src1Im_vec, src2Im_vec, i);
        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMSUB_FLOAT(src2Re_vec, src1Im_vec, tmp2, i);  // vs1*vd - vs2
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
        V_ELT_BOOL32 eqmask = VEQ_FLOAT_BOOL(va, va_abs, i);
        V_ELT_BOOL32 gtmask = VGT1_FLOAT_BOOL(va_abs, value, i);

        V_ELT_FLOAT sval;
        sval = VMERGE1_FLOAT(VNOT_BOOL(eqmask, i), sval, -value, i);
        sval = VMERGE1_FLOAT(eqmask, sval, value, i);
        VSTORE_FLOAT(dst_tmp, VMERGE_FLOAT(gtmask, va, sval, i), i);
#endif
        src_tmp += i;
        dst_tmp += i;
    }
}

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
        V_ELT_BOOL32 lt_mask = VLT1_FLOAT_BOOL(va, ltlevel, i);
        V_ELT_BOOL32 gt_mask = VGT1_FLOAT_BOOL(va, gtlevel, i);
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

#if 1  // should be faster

static inline void log10f_vec(float *src, float *dst, int len)
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

        V_ELT_BOOL32H invalid_mask = VLE1_FLOATH_BOOLH(x, 0.0f, i);
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
        V_ELT_BOOL32H mask = VLT1_FLOATH_BOOLH(x, c_cephes_SQRTHF, i);

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

static inline void log10f_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32(len);
    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLOAD_FLOAT(src_tmp, i);
        V_ELT_INT imm0;

        V_ELT_BOOL32 invalid_mask = VLE1_FLOAT_BOOL(x, 0.0f, i);
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
        V_ELT_BOOL32 mask = VLT1_FLOAT_BOOL(x, c_cephes_SQRTHF, i);

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

static inline V_ELT_FLOATH log_ps(V_ELT_FLOATH x,
                                  V_ELT_FLOATH zero_vec,
                                  V_ELT_FLOATH c_cephes_log_p1_vec,
                                  V_ELT_FLOATH c_cephes_log_p2_vec,
                                  V_ELT_FLOATH c_cephes_log_p3_vec,
                                  V_ELT_FLOATH c_cephes_log_p4_vec,
                                  V_ELT_FLOATH c_cephes_log_p5_vec,
                                  V_ELT_FLOATH c_cephes_log_p6_vec,
                                  V_ELT_FLOATH c_cephes_log_p7_vec,
                                  V_ELT_FLOATH c_cephes_log_p8_vec,
                                  size_t i)
{
    V_ELT_INTH imm0;
    V_ELT_BOOL32H invalid_mask = VLE1_FLOATH_BOOLH(x, 0.0f, i);
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
    V_ELT_BOOL32H mask = VLT1_FLOATH_BOOLH(x, c_cephes_SQRTHF, i);

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

    y = VFMACC1_FLOATH(y, c_cephes_log_q1, e, i);
    y = VFMACC1_FLOATH(y, -0.5f, z, i);  // y = y -0.5*z
    tmp = y;
    tmp = VFMACC1_FLOATH(tmp, c_cephes_log_q2, e, i);
    x = VADD_FLOATH(x, tmp, i);

    x = VMERGE1_FLOATH(invalid_mask, x, 0xFFFFFFFF, i);
    return x;
}

static inline void lnf_vec(float *src, float *dst, int len)
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
        x = log_ps(x, zero_vec, c_cephes_log_p1_vec,
                   c_cephes_log_p2_vec, c_cephes_log_p3_vec,
                   c_cephes_log_p4_vec, c_cephes_log_p5_vec,
                   c_cephes_log_p6_vec, c_cephes_log_p7_vec,
                   c_cephes_log_p8_vec, i);
        VSTORE_FLOATH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void log2f_vec(float *src, float *dst, int len)
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

        V_ELT_BOOL32H invalid_mask = VLE1_FLOATH_BOOLH(x, 0.0f, i);
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
        V_ELT_BOOL32H mask = VLT1_FLOATH_BOOLH(x, c_cephes_SQRTHF, i);

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

        z = VMUL1_FLOATH(y, c_cephes_LOG2EA, i);
        z = VFMACC1_FLOATH(z, c_cephes_LOG2EA, x, i);
        z = VADD_FLOATH(z, tmp, i);
        x = VADD_FLOATH(z, e, i);

        // print_vec(x);printf("\n");
        // could we use merge function? VMERGE_FLOATH? create a nan vec?
        x = VMERGE1_FLOATH(invalid_mask, x, 0xFFFFFFFF, i);

        VSTORE_FLOATH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline V_ELT_FLOATH atanf_ps(V_ELT_FLOATH xx,
                                    V_ELT_FLOATH ATAN_P1_vec,
                                    V_ELT_FLOATH ATAN_P2_vec,
                                    V_ELT_FLOATH ATAN_P3_vec,
                                    V_ELT_FLOATH min1_vec,
                                    size_t i)
{
    V_ELT_FLOATH x, y, z;
    V_ELT_INTH sign;
    V_ELT_BOOL32H suptan3pi8, inftan3pi8suppi8;
    V_ELT_FLOATH tmp, tmp2;
    V_ELT_BOOL32H tmpb1, tmpb2;

    x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(xx), inv_sign_mask, i));
    sign = VAND1_INTH(VINTERP_FLOATH_INTH(xx), sign_mask, i);

    /* range reduction */
    y = VLOAD1_FLOATH(0.0f, i);
    suptan3pi8 = VGT1_FLOATH_BOOLH(x, TAN3PI8F, i);
    tmp = VDIV_FLOATH(min1_vec, x, i);
    x = VMERGE_FLOATH(suptan3pi8, x, tmp, i);
    y = VMERGE1_FLOATH(suptan3pi8, y, PIO2F, i);

    tmpb1 = VLE1_FLOATH_BOOLH(x, TAN3PI8F, i);
    tmpb2 = VGT1_FLOATH_BOOLH(x, TANPI8F, i);
    inftan3pi8suppi8 = VAND_BOOLH(tmpb1, tmpb2, i);

    // To be optimised with RCP?
    tmp = VSUB1_FLOATH(x, 1.0f, i);
    tmp2 = VADD1_FLOATH(x, 1.0f, i);
    tmp = VDIV_FLOATH(tmp, tmp2, i);
    x = VMERGE_FLOATH(inftan3pi8suppi8, x, tmp, i);
    y = VMERGE1_FLOATH(inftan3pi8suppi8, y, PIO4F, i);
    z = VMUL_FLOATH(x, x, i);

    tmp = z;
    tmp = VFMADD1_FLOATH(tmp, ATAN_P0, ATAN_P1_vec, i);
    tmp = VFMADD_FLOATH(tmp, z, ATAN_P2_vec, i);
    tmp = VFMADD_FLOATH(tmp, z, ATAN_P3_vec, i);
    tmp = VMUL_FLOATH(z, tmp, i);
    tmp = VFMADD_FLOATH(tmp, x, x, i);
    y = VADD_FLOATH(y, tmp, i);
    y = VINTERP_INTH_FLOATH(VXOR_INTH(VINTERP_FLOATH_INTH(y), sign, i));
    return y;
}

static inline void atanf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);
    V_ELT_FLOATH ATAN_P1_vec = VLOAD1_FLOATH(ATAN_P1, i);
    V_ELT_FLOATH ATAN_P2_vec = VLOAD1_FLOATH(ATAN_P2, i);
    V_ELT_FLOATH ATAN_P3_vec = VLOAD1_FLOATH(ATAN_P3, i);
    V_ELT_FLOATH min1_vec = VLOAD1_FLOATH(-1.0f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH xx = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH y;
        y = atanf_ps(xx, ATAN_P1_vec, ATAN_P2_vec, ATAN_P3_vec, min1_vec, i);
        VSTORE_FLOATH(dst_tmp, y, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline V_ELT_FLOATH atan2f_ps(V_ELT_FLOATH y, V_ELT_FLOATH x, V_ELT_FLOATH ATAN_P1_vec, V_ELT_FLOATH ATAN_P2_vec, V_ELT_FLOATH ATAN_P3_vec, V_ELT_FLOATH min1_vec, size_t i)
{
    V_ELT_FLOATH z, w;
    V_ELT_BOOL32H xinfzero, yinfzero, xeqzero, yeqzero;
    V_ELT_BOOL32H xeqzeroandyinfzero, yeqzeroandxinfzero;
    V_ELT_BOOL32H specialcase;
    V_ELT_FLOATH tmp, tmp2;

    xinfzero = VLT1_FLOATH_BOOLH(x, 0.0f, i);  // code =2
    yinfzero = VLT1_FLOATH_BOOLH(y, 0.0f, i);  // code = code |1;

    xeqzero = VEQ1_FLOATH_BOOLH(x, 0.0f, i);
    yeqzero = VEQ1_FLOATH_BOOLH(y, 0.0f, i);

    xeqzeroandyinfzero = VAND_BOOLH(xeqzero, yinfzero, i);
    yeqzeroandxinfzero = VAND_BOOLH(yeqzero, xinfzero, i);

#if 0  // not ported on RISCV version
        xeqzeroandyinfzero = _mm_and_ps(xeqzeroandyinfzero, *(V_ELT_FLOATH *) _ps_sign_mask);
        tmp = _mm_xor_ps(*(V_ELT_FLOATH *) _ps_PIO2F, xeqzeroandyinfzero);  // either PI or -PI
        z = _mm_andnot_ps(yeqzero, tmp);                            // not(yeqzero) and tmp => 0, PI/2, -PI/2
#else
    z = VLOAD1_FLOATH(PIO2F, i);
    z = VMERGE1_FLOATH(xeqzeroandyinfzero, z, mPIO2F, i);
    z = VMERGE1_FLOATH(yeqzero, z, 0.0f, i);
#endif
    z = VMERGE1_FLOATH(yeqzeroandxinfzero, z, PIF, i);
    specialcase = VOR_BOOLH(xeqzero, yeqzero, i);

#if 0  // not ported on RISCV version
        tmp = _mm_and_ps(*(V_ELT_FLOATH *) _ps_PIF, _mm_andnot_ps(yinfzero, xinfzero));
        tmp2 = _mm_and_ps(*(V_ELT_FLOATH *) _ps_mPIF, _mm_and_ps(yinfzero, xinfzero));
        w = _mm_add_ps(tmp, tmp2);
#else
    w = VLOAD1_FLOATH(0.0f, i);
    w = VMERGE1_FLOATH(VAND_BOOLH(VNOT_BOOLH(yinfzero, i), xinfzero, i), w, PIF, i);  // y >= 0 && x<0
    w = VMERGE1_FLOATH(VAND_BOOLH(yinfzero, xinfzero, i), w, mPIF, i);                // y < 0 && x<0
#endif

    tmp = VDIV_FLOATH(y, x, i);
    tmp = atanf_ps(tmp, ATAN_P1_vec, ATAN_P2_vec, ATAN_P3_vec, min1_vec, i);
    tmp = VADD_FLOATH(w, tmp, i);
    z = VMERGE_FLOATH(specialcase, tmp, z, i);  // atanf(y/x) if not in special case
    return z;
}

static inline V_ELT_FLOATH atan2f_vec(float *src1, float *src2, float *dst, int len)
{
    size_t i;
    float *src1_tmp = src1;
    float *src2_tmp = src2;
    float *dst_tmp = dst;

    i = VSETVL32H(len);
    V_ELT_FLOATH ATAN_P1_vec = VLOAD1_FLOATH(ATAN_P1, i);
    V_ELT_FLOATH ATAN_P2_vec = VLOAD1_FLOATH(ATAN_P2, i);
    V_ELT_FLOATH ATAN_P3_vec = VLOAD1_FLOATH(ATAN_P3, i);
    V_ELT_FLOATH min1_vec = VLOAD1_FLOATH(-1.0f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH y = VLOAD_FLOATH(src1_tmp, i);
        V_ELT_FLOATH x = VLOAD_FLOATH(src2_tmp, i);

        V_ELT_FLOATH z = atan2f_ps(y, x,
                                   ATAN_P1_vec, ATAN_P2_vec,
                                   ATAN_P3_vec, min1_vec, i);
        VSTORE_FLOATH(dst_tmp, z, i);
        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void atan2f_interleaved_vec(complex32_t *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = (float *) src;
    float *dst_tmp = dst;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    i = VSETVL32H(len);
    V_ELT_FLOATH ATAN_P1_vec = VLOAD1_FLOATH(ATAN_P1, i);
    V_ELT_FLOATH ATAN_P2_vec = VLOAD1_FLOATH(ATAN_P2, i);
    V_ELT_FLOATH ATAN_P3_vec = VLOAD1_FLOATH(ATAN_P3, i);
    V_ELT_FLOATH min1_vec = VLOAD1_FLOATH(-1.0f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x;
        V_ELT_FLOATH y;
        VLOAD_FLOATH2(&x, &y, src_tmp, i);
        V_ELT_FLOATH z = atan2f_ps(y, x,
                                   ATAN_P1_vec, ATAN_P2_vec,
                                   ATAN_P3_vec, min1_vec, i);
        VSTORE_FLOATH(dst_tmp, z, i);
        src_tmp += 2 * i;
        dst_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

static inline void asinf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);
    V_ELT_FLOATH ASIN_P1_vec = VLOAD1_FLOATH(ASIN_P1, i);
    V_ELT_FLOATH ASIN_P2_vec = VLOAD1_FLOATH(ASIN_P2, i);
    V_ELT_FLOATH ASIN_P3_vec = VLOAD1_FLOATH(ASIN_P3, i);
    V_ELT_FLOATH ASIN_P4_vec = VLOAD1_FLOATH(ASIN_P4, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH a, z, z_tmp;
        V_ELT_INTH sign;
        V_ELT_BOOL32H ainfem4, asup0p5, xsup1;
        V_ELT_FLOATH tmp;

        a = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(x), inv_sign_mask, i));
        sign = VAND1_INTH(VINTERP_FLOATH_INTH(x), sign_mask, i);

        ainfem4 = VLT1_FLOATH_BOOLH(a, 1.0e-4f, i);  // if( a < 1.0e-4f )
        asup0p5 = VGT1_FLOATH_BOOLH(a, 0.5f, i);     // if( a > 0.5f ) flag = 1 else 0
        z_tmp = VRSUB1_FLOATH(a, 1.0f, i);
        z_tmp = VMUL1_FLOATH(z_tmp, 0.5f, i);
        tmp = VMUL_FLOATH(a, a, i);
        z = VMERGE_FLOATH(asup0p5, tmp, z_tmp, i);
        x = VMERGE_FLOATH(asup0p5, a, VSQRT_FLOATH(z, i), i);
        xsup1 = VGT1_FLOATH_BOOLH(x, 1.0f, i);

        tmp = z;
        tmp = VFMADD1_FLOATH(tmp, ASIN_P0, ASIN_P1_vec, i);
        tmp = VFMADD_FLOATH(tmp, z, ASIN_P2_vec, i);
        tmp = VFMADD_FLOATH(tmp, z, ASIN_P3_vec, i);
        tmp = VFMADD_FLOATH(tmp, z, ASIN_P4_vec, i);
        tmp = VMUL_FLOATH(z, tmp, i);
        tmp = VFMADD_FLOATH(tmp, x, x, i);

        z = tmp;
        z_tmp = VADD_FLOATH(z, z, i);
        z_tmp = VRSUB1_FLOATH(z_tmp, PIO2F, i);
        z = VMERGE_FLOATH(asup0p5, z, z_tmp, i);

        // done:
        z = VMERGE_FLOATH(ainfem4, z, a, i);
        z = VINTERP_INTH_FLOATH(VXOR_INTH(VINTERP_FLOATH_INTH(z), sign, i));

        // if (x > 1.0) then return 0.0
        z = VMERGE1_FLOATH(xsup1, z, 0.0f, i);

        VSTORE_FLOATH(dst_tmp, z, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void acoshf_vec(float *src, float *dst, int len)
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

    V_ELT_FLOATH ACOSH_P1_vec = VLOAD1_FLOATH(ACOSH_P1, i);
    V_ELT_FLOATH ACOSH_P2_vec = VLOAD1_FLOATH(ACOSH_P2, i);
    V_ELT_FLOATH ACOSH_P3_vec = VLOAD1_FLOATH(ACOSH_P3, i);
    V_ELT_FLOATH ACOSH_P4_vec = VLOAD1_FLOATH(ACOSH_P4, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH z, z_first_branch, z_second_branch;
        V_ELT_BOOL32H xsup1500, zinf0p5, xinf1;
        V_ELT_FLOATH tmp;

        xsup1500 = VGT1_FLOATH_BOOLH(x, 1500.0f, i);  // return  (logf(x) + LOGE2F)
        xinf1 = VLT1_FLOATH_BOOLH(x, 1.0f, i);        // return 0

        z = VSUB1_FLOATH(x, 1.0f, i);
        zinf0p5 = VLT1_FLOATH_BOOLH(z, 0.5f, i);  // first and second branch

        // First Branch (z < 0.5)
        z_first_branch = VFMADD1_FLOATH(z, ACOSH_P0, ACOSH_P1_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, z, ACOSH_P2_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, z, ACOSH_P3_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, z, ACOSH_P4_vec, i);
        z_first_branch = VMUL_FLOATH(z_first_branch, VSQRT_FLOATH(z, i), i);

        // Second Branch
        z_second_branch = VFMADD_FLOATH(z, x, z, i);
        z_second_branch = VSQRT_FLOATH(z_second_branch, i);
        z_second_branch = VADD_FLOATH(x, z_second_branch, i);
        z_second_branch = log_ps(z_second_branch, zero_vec, c_cephes_log_p1_vec,
                                 c_cephes_log_p2_vec, c_cephes_log_p3_vec,
                                 c_cephes_log_p4_vec, c_cephes_log_p5_vec,
                                 c_cephes_log_p6_vec, c_cephes_log_p7_vec,
                                 c_cephes_log_p8_vec, i);

        z = VMERGE_FLOATH(zinf0p5, z_second_branch, z_first_branch, i);
        tmp = log_ps(x, zero_vec, c_cephes_log_p1_vec,
                     c_cephes_log_p2_vec, c_cephes_log_p3_vec,
                     c_cephes_log_p4_vec, c_cephes_log_p5_vec,
                     c_cephes_log_p6_vec, c_cephes_log_p7_vec,
                     c_cephes_log_p8_vec, i);
        tmp = VADD1_FLOATH(tmp, LOGE2F, i);
        z = VMERGE_FLOATH(xsup1500, z, tmp, i);

#if 0  // not ported on RISCV yet
        z = _mm_andnot_ps(xinf1, z);
#else
        z = VMERGE1_FLOATH(xinf1, z, 0.0f, i);
#endif

        VSTORE_FLOATH(dst_tmp, z, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void asinhf_vec(float *src, float *dst, int len)
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

    V_ELT_FLOATH ASINH_P1_vec = VLOAD1_FLOATH(ASINH_P1, i);
    V_ELT_FLOATH ASINH_P2_vec = VLOAD1_FLOATH(ASINH_P2, i);
    V_ELT_FLOATH ASINH_P3_vec = VLOAD1_FLOATH(ASINH_P3, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH xx = VLOAD_FLOATH(src_tmp, i);

        V_ELT_FLOATH x, tmp, z, z_first_branch, z_second_branch;
        V_ELT_BOOL32H xsup1500, xinf0p5;
        V_ELT_INTH xxinf0;

        x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(xx), inv_sign_mask, i));
        xsup1500 = VGT1_FLOATH_BOOLH(x, 1500.0f, i);
        xinf0p5 = VLT1_FLOATH_BOOLH(x, 0.5f, i);

        xxinf0 = VAND1_INTH(VINTERP_FLOATH_INTH(xx), sign_mask, i);

        tmp = VMUL_FLOATH(x, x, i);
        // First Branch (x < 0.5)
        z_first_branch = tmp;
        z_first_branch = VFMADD1_FLOATH(z_first_branch, ASINH_P0, ASINH_P1_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, tmp, ASINH_P2_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, tmp, ASINH_P3_vec, i);
        z_first_branch = VMUL_FLOATH(z_first_branch, tmp, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, x, x, i);

        // Second Branch
        z_second_branch = VSQRT_FLOATH(VADD1_FLOATH(tmp, 1.0f, i), i);
        z_second_branch = log_ps(VADD_FLOATH(z_second_branch, x, i), zero_vec, c_cephes_log_p1_vec,
                                 c_cephes_log_p2_vec, c_cephes_log_p3_vec,
                                 c_cephes_log_p4_vec, c_cephes_log_p5_vec,
                                 c_cephes_log_p6_vec, c_cephes_log_p7_vec,
                                 c_cephes_log_p8_vec, i);

        z = VMERGE_FLOATH(xinf0p5, z_second_branch, z_first_branch, i);
        tmp = log_ps(x, zero_vec, c_cephes_log_p1_vec,
                     c_cephes_log_p2_vec, c_cephes_log_p3_vec,
                     c_cephes_log_p4_vec, c_cephes_log_p5_vec,
                     c_cephes_log_p6_vec, c_cephes_log_p7_vec,
                     c_cephes_log_p8_vec, i);
        tmp = VADD1_FLOATH(tmp, LOGE2F, i);
        z = VMERGE_FLOATH(xsup1500, z, tmp, i);
        z = VINTERP_INTH_FLOATH(VXOR_INTH(VINTERP_FLOATH_INTH(z), xxinf0, i));

        VSTORE_FLOATH(dst_tmp, z, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void atanhf_vec(float *src, float *dst, int len)
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

    V_ELT_FLOATH ATANH_P1_vec = VLOAD1_FLOATH(ATANH_P1, i);
    V_ELT_FLOATH ATANH_P2_vec = VLOAD1_FLOATH(ATANH_P2, i);
    V_ELT_FLOATH ATANH_P3_vec = VLOAD1_FLOATH(ATANH_P3, i);
    V_ELT_FLOATH ATANH_P4_vec = VLOAD1_FLOATH(ATANH_P4, i);
    V_ELT_FLOATH one_vec = VLOAD1_FLOATH(1.0f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH z, tmp, tmp2, z_first_branch, z_second_branch;
        V_ELT_BOOL32H xsup1, xinfmin1, zinf1emin4, zinf0p5;

        z = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(x), inv_sign_mask, i));

        xsup1 = VGE1_FLOATH_BOOLH(x, 1.0f, i);
        xinfmin1 = VLE1_FLOATH_BOOLH(x, -1.0f, i);
        zinf1emin4 = VLT1_FLOATH_BOOLH(z, -1e-4f, i);
        zinf0p5 = VLT1_FLOATH_BOOLH(z, 0.5f, i);

        // First branch
        tmp = VMUL_FLOATH(x, x, i);
        z_first_branch = tmp;
        z_first_branch = VFMADD1_FLOATH(z_first_branch, ATANH_P0, ATANH_P1_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, tmp, ATANH_P2_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, tmp, ATANH_P3_vec, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, tmp, ATANH_P4_vec, i);
        z_first_branch = VMUL_FLOATH(z_first_branch, tmp, i);
        z_first_branch = VFMADD_FLOATH(z_first_branch, x, x, i);

        // Second branch
        // RISCV, could be replace with rcp equivalent vfrec?
        // only 7 bits precision vs rcp 12bits (out of 24)
        tmp = VRSUB1_FLOATH(x, 1.0f, i);  // 1 -x
        tmp2 = VDIV_FLOATH(one_vec, tmp, i);
        tmp = VFMADD_FLOATH(tmp2, x, tmp2, i);
        z_second_branch = log_ps(tmp, zero_vec, c_cephes_log_p1_vec,
                                 c_cephes_log_p2_vec, c_cephes_log_p3_vec,
                                 c_cephes_log_p4_vec, c_cephes_log_p5_vec,
                                 c_cephes_log_p6_vec, c_cephes_log_p7_vec,
                                 c_cephes_log_p8_vec, i);
        z_second_branch = VMUL1_FLOATH(z_second_branch, 0.5f, i);

        z = VMERGE_FLOATH(zinf0p5, z_second_branch, z_first_branch, i);
        z = VMERGE_FLOATH(zinf1emin4, z, x, i);

        z = VMERGE1_FLOATH(xsup1, z, MAXNUMF, i);
        z = VMERGE1_FLOATH(xinfmin1, z, -MAXNUMF, i);

        VSTORE_FLOATH(dst_tmp, z, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline V_ELT_FLOATH exp_ps(V_ELT_FLOATH x,
                                  V_ELT_FLOATH Op5_vec,
                                  V_ELT_FLOATH cephes_exp_p1_vec,
                                  V_ELT_FLOATH cephes_exp_p2_vec,
                                  V_ELT_FLOATH cephes_exp_p3_vec,
                                  V_ELT_FLOATH cephes_exp_p4_vec,
                                  V_ELT_FLOATH cephes_exp_p5_vec,
                                  size_t i)
{
    V_ELT_FLOATH z_tmp, z, fx;
    V_ELT_INTH n;
    V_ELT_BOOL32H xsupmaxlogf, xinfminglogf;

    xsupmaxlogf = VGT1_FLOATH_BOOLH(x, MAXLOGF, i);
    xinfminglogf = VLT1_FLOATH_BOOLH(x, MINLOGF, i);

    /* Express e**x = e**g 2**n
     *   = e**g e**( n loge(2) )
     *   = e**( g + n loge(2) )
     */
    fx = x;
    fx = VFMADD1_FLOATH(x, c_cephes_LOG2EF, Op5_vec, i);
    z = VCVT_INTH_FLOATH(VCVT_FLOATH_INTH(fx, i), i);

    x = VFMACC1_FLOATH(x, -c_cephes_exp_C1, z, i);
    x = VFMACC1_FLOATH(x, -c_cephes_exp_C2, z, i);

    n = VCVT_FLOATH_INTH(z, i);
    n = VADD1_INTH(n, 0x7f, i);
    n = VSLL1_INTH(n, 23, i);
    V_ELT_FLOATH pow2n = VINTERP_INTH_FLOATH(n);

    z = VMUL_FLOATH(x, x, i);

    z_tmp = x;
    z_tmp = VFMADD1_FLOATH(z_tmp, c_cephes_exp_p0, cephes_exp_p1_vec, i);
    z_tmp = VFMADD_FLOATH(z_tmp, x, cephes_exp_p2_vec, i);
    z_tmp = VFMADD_FLOATH(z_tmp, x, cephes_exp_p3_vec, i);
    z_tmp = VFMADD_FLOATH(z_tmp, x, cephes_exp_p4_vec, i);
    z_tmp = VFMADD_FLOATH(z_tmp, x, cephes_exp_p5_vec, i);
    z_tmp = VFMADD_FLOATH(z_tmp, z, x, i);

    /* build 2^n */
    z_tmp = VFMADD_FLOATH(z_tmp, pow2n, pow2n, i);

    z = VMERGE1_FLOATH(xsupmaxlogf, z_tmp, MAXNUMF, i);

#if 0  // not ported on RISCV yey
  z = _mm_andnot_ps(xinfminglogf, z);
#else
    z = VMERGE1_FLOATH(xinfminglogf, z, 0.0f, i);
#endif

    return z;
}

static inline void expf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

    V_ELT_FLOATH cephes_exp_p1_vec = VLOAD1_FLOATH(c_cephes_exp_p1, i);
    V_ELT_FLOATH cephes_exp_p2_vec = VLOAD1_FLOATH(c_cephes_exp_p2, i);
    V_ELT_FLOATH cephes_exp_p3_vec = VLOAD1_FLOATH(c_cephes_exp_p3, i);
    V_ELT_FLOATH cephes_exp_p4_vec = VLOAD1_FLOATH(c_cephes_exp_p4, i);
    V_ELT_FLOATH cephes_exp_p5_vec = VLOAD1_FLOATH(c_cephes_exp_p5, i);
    V_ELT_FLOATH Op5_vec = VLOAD1_FLOATH(0.5f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        x = exp_ps(x, Op5_vec, cephes_exp_p1_vec,
                   cephes_exp_p2_vec, cephes_exp_p3_vec,
                   cephes_exp_p4_vec, cephes_exp_p5_vec, i);
        VSTORE_FLOATH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void coshf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

    V_ELT_FLOATH cephes_exp_p1_vec = VLOAD1_FLOATH(c_cephes_exp_p1, i);
    V_ELT_FLOATH cephes_exp_p2_vec = VLOAD1_FLOATH(c_cephes_exp_p2, i);
    V_ELT_FLOATH cephes_exp_p3_vec = VLOAD1_FLOATH(c_cephes_exp_p3, i);
    V_ELT_FLOATH cephes_exp_p4_vec = VLOAD1_FLOATH(c_cephes_exp_p4, i);
    V_ELT_FLOATH cephes_exp_p5_vec = VLOAD1_FLOATH(c_cephes_exp_p5, i);
    V_ELT_FLOATH Op5_vec = VLOAD1_FLOATH(0.5f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH xx = VLOAD_FLOATH(src_tmp, i);

        V_ELT_FLOATH x, tmp;
        V_ELT_BOOL32H xsupmaxlogf;

        x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(xx), inv_sign_mask, i));
        xsupmaxlogf = VGT1_FLOATH_BOOLH(x, MAXLOGF, i);

        tmp = exp_ps(x, Op5_vec, cephes_exp_p1_vec,
                     cephes_exp_p2_vec, cephes_exp_p3_vec,
                     cephes_exp_p4_vec, cephes_exp_p5_vec, i);
        x = VRDIV1_FLOATH(tmp, 0.5f, i);  // or 1/(2*y)
        x = VFMACC1_FLOATH(x, 0.5f, tmp, i);
        x = VMERGE1_FLOATH(xsupmaxlogf, x, MAXNUMF, i);
        VSTORE_FLOATH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

// TODO : ULP bigger than X86?
static inline void sinhf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);

    V_ELT_FLOATH SINH_P1_vec = VLOAD1_FLOATH(SINH_P1, i);
    V_ELT_FLOATH SINH_P2_vec = VLOAD1_FLOATH(SINH_P2, i);
    V_ELT_FLOATH cephes_exp_p1_vec = VLOAD1_FLOATH(c_cephes_exp_p1, i);
    V_ELT_FLOATH cephes_exp_p2_vec = VLOAD1_FLOATH(c_cephes_exp_p2, i);
    V_ELT_FLOATH cephes_exp_p3_vec = VLOAD1_FLOATH(c_cephes_exp_p3, i);
    V_ELT_FLOATH cephes_exp_p4_vec = VLOAD1_FLOATH(c_cephes_exp_p4, i);
    V_ELT_FLOATH cephes_exp_p5_vec = VLOAD1_FLOATH(c_cephes_exp_p5, i);
    V_ELT_FLOATH Op5_vec = VLOAD1_FLOATH(0.5f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH z, z_first_branch, z_second_branch, tmp;
        V_ELT_BOOL32H xsupmaxlogf, zsup1;
        V_ELT_INTH sign;

        // x = xx; if x < 0, z = -x, else x
        z = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(x), inv_sign_mask, i));
        sign = VAND1_INTH(VINTERP_FLOATH_INTH(x), sign_mask, i);

        xsupmaxlogf = VGT1_FLOATH_BOOLH(z, MAXLOGF, i);

        // First branch
        zsup1 = VGT1_FLOATH_BOOLH(z, 1.0f, i);
        tmp = exp_ps(z, Op5_vec, cephes_exp_p1_vec,
                     cephes_exp_p2_vec, cephes_exp_p3_vec,
                     cephes_exp_p4_vec, cephes_exp_p5_vec, i);
        z_first_branch = VRDIV1_FLOATH(tmp, -0.5f, i);
        z_first_branch = VFMACC1_FLOATH(z_first_branch, 0.5f, tmp, i);

#if 0  // not ported on RISCV yet
      z_first_branch = _mm_xor_ps(z_first_branch, sign);
#else
        V_ELT_BOOL32H xinf0 = VLT1_FLOATH_BOOLH(x, 0.0f, i);
        V_ELT_FLOATH tmp2 = VINTERP_INTH_FLOATH(VXOR1_INTH(VINTERP_FLOATH_INTH(z_first_branch), neg_sign_mask, i));
        z_first_branch = VMERGE_FLOATH(xinf0, z_first_branch, tmp2, i);
#endif

        // Second branch
        tmp = VMUL_FLOATH(x, x, i);
        z_second_branch = tmp;
        z_second_branch = VFMADD1_FLOATH(z_second_branch, SINH_P0, SINH_P1_vec, i);
        z_second_branch = VFMADD_FLOATH(z_second_branch, tmp, SINH_P2_vec, i);
        z_second_branch = VMUL_FLOATH(z_second_branch, tmp, i);
        z_second_branch = VFMADD_FLOATH(z_second_branch, x, x, i);

        // Choose between first and second branch
        z = VMERGE_FLOATH(zsup1, z_second_branch, z_first_branch, i);

        // Set value to MAXNUMF if abs(x) > MAGLOGF
        // Set value to -MAXNUMF if abs(x) > MAGLOGF and x < 0
        tmp = VINTERP_INTH_FLOATH(VXOR1_INTH(sign, *(int32_t *) &MAXNUMF, i));
        z = VMERGE_FLOATH(xsupmaxlogf, z, tmp, i);

        VSTORE_FLOATH(dst_tmp, z, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void tanhf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

    V_ELT_FLOATH cephes_exp_p1_vec = VLOAD1_FLOATH(c_cephes_exp_p1, i);
    V_ELT_FLOATH cephes_exp_p2_vec = VLOAD1_FLOATH(c_cephes_exp_p2, i);
    V_ELT_FLOATH cephes_exp_p3_vec = VLOAD1_FLOATH(c_cephes_exp_p3, i);
    V_ELT_FLOATH cephes_exp_p4_vec = VLOAD1_FLOATH(c_cephes_exp_p4, i);
    V_ELT_FLOATH cephes_exp_p5_vec = VLOAD1_FLOATH(c_cephes_exp_p5, i);
    V_ELT_FLOATH Op5_vec = VLOAD1_FLOATH(0.5f, i);
    V_ELT_FLOATH TANH_P1_vec = VLOAD1_FLOATH(TANH_P1, i);
    V_ELT_FLOATH TANH_P2_vec = VLOAD1_FLOATH(TANH_P2, i);
    V_ELT_FLOATH TANH_P3_vec = VLOAD1_FLOATH(TANH_P3, i);
    V_ELT_FLOATH TANH_P4_vec = VLOAD1_FLOATH(TANH_P4, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH xx = VLOAD_FLOATH(src_tmp, i);
        V_ELT_FLOATH x, z, z_first_branch, z_second_branch, tmp;
        V_ELT_BOOL32H xxsup0, xsupmaxlogfdiv2, xsup0p625;
        xxsup0 = VGT1_FLOATH_BOOLH(xx, 0.0f, i);
        xsupmaxlogfdiv2 = VGT1_FLOATH_BOOLH(xx, MAXLOGFDIV2, i);

        x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(xx), inv_sign_mask, i));

        xsup0p625 = VGE1_FLOATH_BOOLH(x, 0.625f, i);
        tmp = VADD_FLOATH(x, x, i);
        tmp = exp_ps(tmp, Op5_vec, cephes_exp_p1_vec,
                     cephes_exp_p2_vec, cephes_exp_p3_vec,
                     cephes_exp_p4_vec, cephes_exp_p5_vec, i);
        x = VMERGE_FLOATH(xsup0p625, x, tmp, i);

        // z = 1.0 - 2.0 / (x + 1.0);
        z_first_branch = VADD1_FLOATH(x, 1.0f, i);
        z_first_branch = VRDIV1_FLOATH(z_first_branch, -2.0f, i);
        z_first_branch = VADD1_FLOATH(z_first_branch, 1.0f, i);
        tmp = VINTERP_INTH_FLOATH(VXOR1_INTH(VINTERP_FLOATH_INTH(z_first_branch), neg_sign_mask, i));
        z_first_branch = VMERGE_FLOATH(xxsup0, tmp, z_first_branch, i);

        // to speed up the last merge
        xxsup0 = VAND_BOOLH(xxsup0, xsupmaxlogfdiv2, i);

        // z = x * x;
        z = VMUL_FLOATH(x, x, i);
        z_second_branch = z;
        z_second_branch = VFMADD1_FLOATH(z_second_branch, TANH_P0, TANH_P1_vec, i);
        z_second_branch = VFMADD_FLOATH(z_second_branch, z, TANH_P2_vec, i);
        z_second_branch = VFMADD_FLOATH(z_second_branch, z, TANH_P3_vec, i);
        z_second_branch = VFMADD_FLOATH(z_second_branch, z, TANH_P4_vec, i);
        z_second_branch = VMUL_FLOATH(z_second_branch, z, i);
        z_second_branch = VFMADD_FLOATH(z_second_branch, xx, xx, i);

        z = VMERGE_FLOATH(xsup0p625, z_second_branch, z_first_branch, i);
        // if (x > 0.5 * MAXLOGF), return (xx > 0)? 1.0f: -1.0f
        z = VMERGE1_FLOATH(xsupmaxlogfdiv2, z, -1.0f, i);
        z = VMERGE1_FLOATH(xxsup0, z, 1.0f, i);  // xxsup0.xsupmaxlogfdiv2 has already been done
        VSTORE_FLOATH(dst_tmp, z, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void vectorSlopef_vec(float *dst, int len, float offset, float slope)
{
    size_t i;
    float *dst_tmp = dst;

    float coef_max[MAX_ELTS32];

    // to be improved!
    for (int s = 0; s < MAX_ELTS32; s++) {
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
    size_t i;
    i = VSETVL32(len);
    int vec_size = VSETVL32(4096);
    float *src_tmp = src + len - i;
    float *dst_tmp = dst;

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
    V_ELT_FLOAT a, b;
    for (; (i = VSETVL32(len)) >= vec_size; len -= i) {
        a = VLOAD_FLOAT(src_tmp, i);
        b = VGATHER_FLOAT(a, index_vec, i);
        VSTORE_FLOAT(dst_tmp, b, i);
        src_tmp -= i;
        dst_tmp += i;
    }

    if (i) {
        src_tmp = src;
        index_vec = VLOAD_UINT(index + MAX_ELTS32 - i, i);
        a = VLOAD_FLOAT(src_tmp, i);
        b = VGATHER_FLOAT(a, index_vec, i);
        VSTORE_FLOAT(dst_tmp, b, i);
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


    vuint32m2_t index_vec = vle32_v_u32m2(index + MAX_ELTS32 - vec_size, i);
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

// Work in progress
#if 1
static inline V_ELT_FLOATH power_of_twof(V_ELT_INTH b, size_t i)
{
    V_ELT_INTH exp = VADD1_INTH(b, 127, i);
    V_ELT_FLOATH f = VINTERP_INTH_FLOATH(VSLL1_INTH(exp, 23, i));
    return f;
}

static inline void cbrtf_vec(float *src, float *dst, int len)
{
#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);
    V_ELT_FLOATH invCBRT2_vec = VLOAD1_FLOATH(cephes_invCBRT2, i);
    V_ELT_FLOATH invCBRT4_vec = VLOAD1_FLOATH(cephes_invCBRT4, i);
    V_ELT_FLOATH CBRTF_P1_vec = VLOAD1_FLOATH(CBRTF_P1, i);
    V_ELT_FLOATH CBRTF_P2_vec = VLOAD1_FLOATH(CBRTF_P2, i);
    V_ELT_FLOATH CBRTF_P3_vec = VLOAD1_FLOATH(CBRTF_P3, i);
    V_ELT_FLOATH CBRTF_P4_vec = VLOAD1_FLOATH(CBRTF_P4, i);

    const float Op5 = 0.5f;

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH xx = VLOAD_FLOATH(src_tmp, i);

        V_ELT_INTH sign;
        V_ELT_FLOATH x, z, e, rem;
        V_ELT_FLOATH tmp, tmp2;

        x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(xx), inv_sign_mask, i));
        sign = VAND1_INTH(VINTERP_FLOATH_INTH(xx), sign_mask, i);

        z = x;
        /* extract power of 2, leaving
         * mantissa between 0.5 and 1
         */
        // x = frexpf(x, &e);
        // solve problem for zero
        V_ELT_INTH emm0 = VSRA1_INTH(VINTHERP_FLOATH_INTH(x), 23, i);
        x = VINTERP_INTH_FLOATH(VAND1_INTH(VINTHERP_FLOATH_INTH(x), c_inv_mant_mask, i));
        x = VINTERP_INTH_FLOATH(VOR1_INTH(VINTHERP_FLOATH_INTH(x), *(int32_t *) &Op5, i));
        emm0 = VSUB1_INTH(emm0, 0x7e, i);
        e = VCVT_INTH_FLOATH(emm0, i);

        /* Approximate cube root of number between .5 and 1,
         * peak relative error = 9.2e-6
         */
        tmp = x;
        tmp = VFMADD1_FLOATH(tmp, CBRTF_P0, CBRTF_P1_vec, i);
        tmp = VFMADD_FLOATH(tmp, x, CBRTF_P2_vec, i);
        tmp = VFMADD_FLOATH(tmp, x, CBRTF_P3_vec, i);
        x = VFMADD_FLOATH(x, tmp, CBRTF_P4_vec, i);

        /* exponent divided by 3 */
        V_ELT_BOOL32H e_sign = VGE1_FLOATH_BOOLH(e, 0.0f, i);
        e = VINTERP_INTH_FLOATH(VAND1_INTH(VINTERP_FLOATH_INTH(e), inv_sign_mask, i));
        rem = e;
        e = VMUL1_FLOATH(e, 0.333333333333f, i);
        V_ELT_INTH e_int = VCVT_RTZ_FLOATH_INTH(e, i);
        e = VCVT_INTH_FLOATH(e_int, i);
        V_ELT_FLOATH e_tmp = VMUL1_FLOATH(e, 3.0f, i);
        rem = VSUB_FLOATH(rem, e_tmp, i);

        V_ELT_FLOATH mul1, mul2;
        V_ELT_FLOATH mul_cst1 = VMERGE1_FLOATH(e_sign, invCBRT2_vec, cephes_CBRT2, i);
        V_ELT_FLOATH mul_cst2 = VMERGE1_FLOATH(e_sign, invCBRT4_vec, cephes_CBRT4, i);
        mul1 = VMUL_FLOATH(x, mul_cst1, i);
        mul2 = VMUL_FLOATH(x, mul_cst2, i);

        V_ELT_INTH remi = VCVT_FLOATH_INTH(rem, i);  // rem integer
        V_ELT_BOOL32H rem1 = VEQ1_INTH_BOOLH(remi, 1, i);
        V_ELT_BOOL32H rem2 = VEQ1_INTH_BOOLH(remi, 2, i);

        x = VMERGE_FLOATH(rem1, x, mul1, i);  // rem==1
        x = VMERGE_FLOATH(rem2, x, mul2, i);  // rem==2

        /* multiply by power of 2 */
        //  x = ldexpf(x, e);
        // x= x* (1 >> e)
        V_ELT_FLOATH cst = power_of_twof(e_int, i);

        // blend sign of e
        tmp = VMUL_FLOATH(x, cst, i);
        tmp2 = VDIV_FLOATH(x, cst, i);
        x = VMERGE_FLOATH(e_sign, tmp2, tmp, i);

        /* Newton iteration */
        // x -= (x - (z / (x * x))) * 0.333333333333;
        tmp2 = VMUL_FLOATH(x, x, i);
        tmp2 = VDIV_FLOATH(z, tmp2, i);
        tmp2 = VSUB_FLOATH(x, tmp2, i);
        tmp2 = VMUL1_FLOATH(tmp2, 0.333333333333f, i);
        x = VSUB_FLOATH(x, tmp2, i);
        x = VINTERP_INTH_FLOATH(VXOR_INTH(VINTERP_FLOATH_INTH(x), sign, i));

        VSTORE_FLOATH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}
#endif

static inline void convertInt16ToFloat32_vec(int16_t *src, float *dst, int len, int scale_factor)
{
    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    size_t i;
    int16_t *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL16H(len)) > 0; len -= i) {
        V_ELT_SHORTH src_short;
        V_ELT_INT src_int;
        V_ELT_FLOAT dst_float;
        src_short = VLOAD_SHORTH(src_tmp, i);
        src_int = VCVT_SHORTH_INT(src_short, i);
        dst_float = VMUL1_FLOAT(VCVT_INT_FLOAT(src_int, i), scale_fact_mult, i);
        VSTORE_FLOAT(dst_tmp, dst_float, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

// TODO: specific case for RISCV RndFinancial?
static inline void convertFloat32ToI16_vec(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();

    if (rounding_mode == RndZero) {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
    } else if (rounding_mode == RndFinancial) {
    } else {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
    }

    size_t i;
    float *src_tmp = src;
    int16_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_SHORTH dst_short;
        V_ELT_INT dst_int;
        V_ELT_FLOAT src_float;
        V_ELT_FLOAT tmp;

        src_float = VLOAD_FLOAT(src_tmp, i);
        tmp = VMUL1_FLOAT(src_float, scale_fact_mult, i);
        dst_int = VCVT_FLOAT_INT(tmp, i);
        dst_short = VCVT_INT_SHORTH(dst_int, 0, i);
        VSTORE_SHORTH(dst_tmp, dst_short, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

// TODO: specific case for RISCV RndFinancial?
//  could scale factor come directly from VCVT_UINT_USHORTH? (shift parameter)
static inline void convertFloat32ToU16_vec(float *src, uint16_t *dst, int len, int rounding_mode, int scale_factor)
{
    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();

    if (rounding_mode == RndZero) {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
    } else if (rounding_mode == RndFinancial) {
    } else {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
    }

    size_t i;
    float *src_tmp = src;
    uint16_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_USHORTH dst_short;
        V_ELT_UINT dst_int;
        V_ELT_FLOAT src_float;
        V_ELT_FLOAT tmp;

        src_float = VLOAD_FLOAT(src_tmp, i);
        tmp = VMUL1_FLOAT(src_float, scale_fact_mult, i);
        dst_int = VCVT_FLOAT_UINT(tmp, i);
        dst_short = VCVT_UINT_USHORTH(dst_int, 0, i);
        VSTORE_USHORTH(dst_tmp, dst_short, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void convertFloat32ToU8_vec(float *src, uint8_t *dst, int len, int rounding_mode, int scale_factor)
{
    float scale_fact_mult = 1.0f / (float) (1 << scale_factor);

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();

    if (rounding_mode == RndZero) {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);  // rounding_vec = ROUNDTOZERO;
    } else if (rounding_mode == RndFinancial) {
    } else {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);  // rounding_vec = ROUNDTONEAREST;
    }

    size_t i;
    float *src_tmp = src;
    uint8_t *dst_tmp = dst;

    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_UBYTEHH dst_ubyte;
        V_ELT_USHORTH dst_short;
        V_ELT_UINT dst_int;
        V_ELT_FLOAT src_float;
        V_ELT_FLOAT tmp;

        src_float = VLOAD_FLOAT(src_tmp, i);
        tmp = VMUL1_FLOAT(src_float, scale_fact_mult, i);
        dst_int = VCVT_FLOAT_UINT(tmp, i);
        dst_short = VCVT_UINT_USHORTH(dst_int, 0, i);
        dst_ubyte = VCVT_USHORTH_UBYTEHH(dst_short, 0, i);
        VSTORE_UBYTEHH(dst_tmp, dst_ubyte, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void pol2cart2Df_vec(float *r, float *theta, float *x, float *y, int len)
{
    float *r_tmp = r;
    float *theta_tmp = theta;
    float *x_tmp = x;
    float *y_tmp = y;
    size_t i;

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
        V_ELT_FLOATH r_vec = VLOAD_FLOATH(r_tmp, i);
        V_ELT_FLOATH theta_vec = VLOAD_FLOATH(theta_tmp, i);
        V_ELT_FLOATH sin_vec, cos_vec;
        sincosf_ps(theta_vec, &sin_vec, &cos_vec,
                   coscof_1_vec, coscof_2_vec,
                   sincof_1_vec, sincof_2_vec, i);
        V_ELT_FLOATH x_vec = VMUL_FLOATH(r_vec, cos_vec, i);
        V_ELT_FLOATH y_vec = VMUL_FLOATH(r_vec, sin_vec, i);
        VSTORE_FLOATH(x_tmp, x_vec, i);
        VSTORE_FLOATH(y_tmp, y_vec, i);

        r_tmp += i;
        theta_tmp += i;
        x_tmp += i;
        y_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

static inline void cart2pol2Df_vec(float *x, float *y, float *r, float *theta, int len)
{
    float *r_tmp = r;
    float *theta_tmp = theta;
    float *x_tmp = x;
    float *y_tmp = y;
    size_t i;

#ifdef NO_RTZ
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
#endif

    i = VSETVL32H(len);
    V_ELT_FLOATH ATAN_P1_vec = VLOAD1_FLOATH(ATAN_P1, i);
    V_ELT_FLOATH ATAN_P2_vec = VLOAD1_FLOATH(ATAN_P2, i);
    V_ELT_FLOATH ATAN_P3_vec = VLOAD1_FLOATH(ATAN_P3, i);
    V_ELT_FLOATH min1_vec = VLOAD1_FLOATH(-1.0f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x_vec = VLOAD_FLOATH(x_tmp, i);
        V_ELT_FLOATH y_vec = VLOAD_FLOATH(y_tmp, i);
        V_ELT_FLOATH y_square = VMUL_FLOATH(y_vec, y_vec, i);
        V_ELT_FLOATH r_vec = x_vec;
        r_vec = VFMADD_FLOATH(r_vec, x_vec, y_square, i);
        r_vec = VSQRT_FLOATH(r_vec, i);
        V_ELT_FLOATH theta_vec = atan2f_ps(y_vec, x_vec,
                                           ATAN_P1_vec, ATAN_P2_vec,
                                           ATAN_P3_vec, min1_vec, i);
        VSTORE_FLOATH(r_tmp, r_vec, i);
        VSTORE_FLOATH(theta_tmp, theta_vec, i);

        r_tmp += i;
        theta_tmp += i;
        x_tmp += i;
        y_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

static inline void PReluf_vec(float *src, float *dst, float alpha, int len)
{
    float *src_tmp = src;
    float *dst_tmp = dst;
    size_t i;

    i = VSETVL32(len);
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT src_vec = VLOAD_FLOAT(src_tmp, i);
        V_ELT_FLOAT tmp = VMUL1_FLOAT(src_vec, alpha, i);
        V_ELT_BOOL32 mask = VGT1_FLOAT_BOOL(src_vec, 0.0f, i);
        V_ELT_FLOAT dst_vec = VMERGE_FLOAT(mask, tmp, src_vec, i);
        VSTORE_FLOAT(dst_tmp, dst_vec, i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void sigmoidf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL32H(len);

    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

    V_ELT_FLOATH cephes_exp_p1_vec = VLOAD1_FLOATH(c_cephes_exp_p1, i);
    V_ELT_FLOATH cephes_exp_p2_vec = VLOAD1_FLOATH(c_cephes_exp_p2, i);
    V_ELT_FLOATH cephes_exp_p3_vec = VLOAD1_FLOATH(c_cephes_exp_p3, i);
    V_ELT_FLOATH cephes_exp_p4_vec = VLOAD1_FLOATH(c_cephes_exp_p4, i);
    V_ELT_FLOATH cephes_exp_p5_vec = VLOAD1_FLOATH(c_cephes_exp_p5, i);
    V_ELT_FLOATH Op5_vec = VLOAD1_FLOATH(0.5f, i);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH x = VLOAD_FLOATH(src_tmp, i);
        x = VINTERP_INTH_FLOATH(VXOR1_INTH(VINTERP_FLOATH_INTH(x), neg_sign_mask, i));
        x = exp_ps(x, Op5_vec, cephes_exp_p1_vec,
                   cephes_exp_p2_vec, cephes_exp_p3_vec,
                   cephes_exp_p4_vec, cephes_exp_p5_vec, i);
        x = VADD1_FLOATH(x, 1.0f, i);
        x = VRDIV1_FLOATH(x, 1.0f, i); // 1/x
        VSTORE_FLOATH(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

static inline void softmaxf_vec(float *src, float *dst, int len)
{
    size_t i;
    size_t i_last;
    float *src_tmp = src;
    float *dst_tmp = dst;
    size_t len_ori = len;
    
    uint32_t reg_ori;
    reg_ori = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

    i = VSETVL32H(len);
    V_ELT_FLOATH cephes_exp_p1_vec = VLOAD1_FLOATH(c_cephes_exp_p1, i);
    V_ELT_FLOATH cephes_exp_p2_vec = VLOAD1_FLOATH(c_cephes_exp_p2, i);
    V_ELT_FLOATH cephes_exp_p3_vec = VLOAD1_FLOATH(c_cephes_exp_p3, i);
    V_ELT_FLOATH cephes_exp_p4_vec = VLOAD1_FLOATH(c_cephes_exp_p4, i);
    V_ELT_FLOATH cephes_exp_p5_vec = VLOAD1_FLOATH(c_cephes_exp_p5, i);
    V_ELT_FLOATH Op5_vec = VLOAD1_FLOATH(0.5f, i);
    
    V_ELT_FLOATH vacc = VLOAD1_FLOATH(0.0f, i);
    vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, i);
    float acc_scalar = 0.0f;
    
    vse32_v_f32m1(&acc_scalar, acc, 1);

    for (; (i = VSETVL32H(len)) > 0; len -= i) {
        V_ELT_FLOATH va = VLOAD_FLOATH(src_tmp, i);
        va = exp_ps(va, Op5_vec, cephes_exp_p1_vec,
                   cephes_exp_p2_vec, cephes_exp_p3_vec,
                   cephes_exp_p4_vec, cephes_exp_p5_vec, i);
        vacc = VADD_FLOATH(vacc, va, i);
        VSTORE_FLOATH(dst_tmp, va, i);
        src_tmp += i;
        dst_tmp += i;
        i_last = i;
    }
    acc = VREDSUM_FLOATH(acc, vacc, acc, i_last);
    vse32_v_f32m1(&acc_scalar, acc, 1);
    
    len = len_ori;
    dst_tmp = dst;
    for (; (i = VSETVL32(len)) > 0; len -= i) {
        V_ELT_FLOAT dst_vec = VLOAD_FLOAT(dst_tmp, i);
        dst_vec = VDIV1_FLOAT(dst_vec, acc_scalar, i);
        VSTORE_FLOAT(dst_tmp, dst_vec, i);
        dst_tmp += i;
    }

    _MM_SET_ROUNDING_MODE(reg_ori);
}

#if ELEN >= 64
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
#else

#warning "No support for double precision functions"
static inline void convert_32f64f_vec(float *src, double *dst, int len)
{
}
static inline void convert_64f32f_vec(double *src, float *dst, int len) {}

#endif  // ELEN >= 64
