/*
 * Project : SIMD_Utils
 * Version : 0.2.6
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

#include <fenv.h>
#include <math.h>
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


void addf_vec(float *a, float *b, float *c, int len)
{
    V_ELT_FLOAT va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        va = VLOAD_FLOAT(a+l, i);
        vb = VLOAD_FLOAT(b+l, i);
        vc = VADD_FLOAT(va, vb, i);
        VSTORE_FLOAT(c+l, vc, i);
    }
}

static inline void addcf_vec(float *src, float value, float *dst, int len)
{
    V_ELT_FLOAT va, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        va = VLOAD_FLOAT(src+l, i);
        vc = VADD1_FLOAT(va, value, i);
        VSTORE_FLOAT(dst+l, vc, i);
    }
}

static inline void mulf_vec(float *a, float *b, float *c, int len)
{
    V_ELT_FLOAT va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        va = VLOAD_FLOAT(a+l, i);
        vb = VLOAD_FLOAT(b+l, i);
        vc = VMUL_FLOAT(va, vb, i);
        VSTORE_FLOAT(c+l, vc, i);
    }
}

static inline void divf_vec(float *a, float *b, float *c, int len)
{
    V_ELT_FLOAT va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        va = VLOAD_FLOAT(a+l, i);
        vb = VLOAD_FLOAT(b+l, i);
        vc = VDIV_FLOAT(va, vb, i);
        VSTORE_FLOAT(c+l, vc, i);
    }
}

static inline void subf_vec(float *a, float *b, float *c, int len)
{
    V_ELT_FLOAT va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        va = VLOAD_FLOAT(a+l, i);
        vb = VLOAD_FLOAT(b+l, i);
        vc = VSUB_FLOAT(va, vb, i);
        VSTORE_FLOAT(c+l, vc, i);
    }
}

static inline void subcrevf_vec(float *src, float value, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src + l, i);
        VSTORE_FLOAT(dst + l, VRSUB1_FLOAT(va, value, i), i);
    }
}

static inline void muladdf_vec(float *a, float *b, float *c, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va, vb, vc;
        va = VLOAD_FLOAT(a+l, i);
        vb = VLOAD_FLOAT(b+l, i);
        vc = VLOAD_FLOAT(c+l, i);
        vc = VFMACC_FLOAT(vc, va, vb, i);
        VSTORE_FLOAT(dst+l, vc, i);
    }
}

static inline void mulcaddf_vec(float *a, float b, float *c, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va, vc;
        va = VLOAD_FLOAT(a+l, i);
        vc = VLOAD_FLOAT(c+l, i);
        vc = VFMACC1_FLOAT(vc, b, va, i);
        VSTORE_FLOAT(dst+l, vc, i);
    }
}

static inline void mulcaddcf_vec(float *a, float b, float c, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va, vc;
        va = VLOAD_FLOAT(a+l, i);
        vc = VLOAD1_FLOAT(c, i);
        vc = VFMACC1_FLOAT(vc, b, va, i);
        VSTORE_FLOAT(dst+l, vc, i);
    }
}

static inline void muladdcf_vec(float *a, float *b, float c, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va, vb, vc;
        va = VLOAD_FLOAT(a+l, i);
        vb = VLOAD_FLOAT(b+l, i);
        vc = VLOAD1_FLOAT(c, i);
        vc = VFMACC_FLOAT(vc, va, vb, i);
        VSTORE_FLOAT(dst+l, vc, i);
    }
}

static inline void mulcf_vec(float *src, float value, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src+l, i);
        VSTORE_FLOAT(dst+l, VMUL1_FLOAT(va, value, i), i);
    }
}


static inline void sinf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);

        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);

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
        emm2 = VCVT_FLOAT_INT(y, i);

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

        VSTORE_FLOAT(dst+l, y, i);
    }
}


static inline void cosf_vec(float *src, float *dst, int len)
{

    V_ELT_FLOAT c_coscof_p1_vec = VLOAD1_FLOAT(c_coscof_p1, i);
    V_ELT_FLOAT c_coscof_p2_vec = VLOAD1_FLOAT(c_coscof_p2, i);
    V_ELT_FLOAT c_sincof_p1_vec = VLOAD1_FLOAT(c_sincof_p1, i);
    V_ELT_FLOAT c_sincof_p2_vec = VLOAD1_FLOAT(c_sincof_p2, i);

	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);

        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT xmm3, y;

        V_ELT_INT emm0, emm2;

        /* take the absolute value */
        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

        /* scale by 4/Pi */
        y = VMUL1_FLOAT(x, FOPI, i);

        /* store the integer part of y in mm0 */
	
        emm2 = VCVT_RTZ_FLOAT_INT(y, i);
	
        /* j=(j+1) & (~1) (see the cephes sources) */
        emm2 = VADD1_INT(emm2, 1, i);
        emm2 = VAND1_INT(emm2, ~1, i);
        y = VCVT_INT_FLOAT(emm2, i);

        emm2 = VSUB1_INT(emm2, 2, i);

        /* get the swap sign flag */
        emm0 = VAND1_INT(VNOT_INT(emm2, i), 4, i);
        emm0 = VSLL1_INT(emm0, 29, i);
        /* get the polynom selection mask */
        emm2 = VAND1_INT(emm2, 2, i);
        V_ELT_BOOL32 poly_mask = VEQ1_INT_BOOL(emm2, 0, i);

        /* The magic pass: "Extended precision modular arithmetic"
         x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMACC1_FLOAT(x, minus_cephes_DP1, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP2, y, i);
        x = VFMACC1_FLOAT(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT_FLOAT z = VMUL_FLOAT(x, x, i);

        y = z;
        y = VFMADD1_FLOAT(y, c_coscof_p0, c_coscof_p1_vec, i);
        y = VFMADD_FLOAT(y, z, c_coscof_p2_vec, i);
        y = VMUL_FLOAT(y, z, i);
        y = VMUL_FLOAT(y, z, i);
        y = VFMACC1_FLOAT(y, -0.5f, z, i);  // y = y -0.5*z
        y = VADD1_FLOAT(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT_FLOAT y2 = z;
        y2 = VFMADD1_FLOAT(y2, c_sincof_p0, c_sincof_p1_vec, i);
        y2 = VFMADD_FLOAT(y2, z, c_sincof_p2_vec, i);
        y2 = VMUL_FLOAT(y2, z, i);
        y2 = VFMADD_FLOAT(y2, x, x, i);

        /* select the correct result from the two polynoms */
        y = VMERGE_FLOAT(poly_mask, y, y2, i);

        /* update the sign */
        y = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(y), emm0, i));
        VSTORE_FLOAT(dst+l, y, i);
    }
}



void sincosf_ps(V_ELT_FLOAT x,
                              V_ELT_FLOAT *sin_tmp,
                              V_ELT_FLOAT *cos_tmp,
                              V_ELT_FLOAT coscof_1_vec,
                              V_ELT_FLOAT coscof_2_vec,
                              V_ELT_FLOAT sincof_1_vec,
                              V_ELT_FLOAT sincof_2_vec,
                              V_ELT_BOOL32 i)
{
    V_ELT_FLOAT y, sign_bit_sin;
    V_ELT_INT imm0, imm2, imm4;

	sign_bit_sin = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), sign_mask, i));
	
    /* take te absolute value */
    x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

    /* scale by 4/Pi */
    y = VMUL1_FLOAT(x, FOPI, i);

    /* store te integer part of y in mm2 */
    imm2 = VCVT_RTZ_FLOAT_INT(y, i);
    /* j=(j+1) & (~1) (see te cepes sources) */
    imm2 = VADD1_INT(imm2, 1, i);
    imm2 = VAND1_INT(imm2,  ~1, i);
	
    y = VCVT_INT_FLOAT(imm2, i);
    imm4 = imm2;

    imm0 = VAND1_INT(imm2, 4, i);
    imm0 = VSLL1_INT(imm0, 29, i);

	/* get te polynom selection mask for te sine*/
	imm2 =  VAND1_INT(imm2, 2, i);
    V_ELT_FLOAT swap_sign_bit_sin = VINTERP_INT_FLOAT(imm0);
    V_ELT_BOOL32 poly_mask = VEQ1_INT_BOOL(imm2, 0, i);

    /* Te magic pass: "Extended precision modular aritmetic"
    x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = VFMACC1_FLOAT(x, minus_cephes_DP1, y, i);
    x = VFMACC1_FLOAT(x, minus_cephes_DP2, y, i);
    x = VFMACC1_FLOAT(x, minus_cephes_DP3, y, i);

    imm4 = VSUB1_INT(imm4, 2, i);
    imm4 = VAND1_INT(VNOT_INT(imm4, i), 4, i);
    imm4 = VSLL1_INT(imm4, 29, i);

	V_ELT_FLOAT sign_bit_cos = VINTERP_INT_FLOAT(imm4);
    sign_bit_sin = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(sign_bit_sin),imm0, i));

    /* Evaluate te first polynom  (0 <= x <= Pi/4) */
    V_ELT_FLOAT z = VMUL_FLOAT(x, x, i);
    y = z;
    y = VFMADD1_FLOAT(y, coscof[0], coscof_1_vec, i);
    y = VFMADD_FLOAT(y, z, coscof_2_vec, i);
    y = VMUL_FLOAT(y, z, i);
    y = VMUL_FLOAT(y, z, i);
    y = VFMACC1_FLOAT(y, -0.5f, z, i);  // y = y -0.5*z
    y = VADD1_FLOAT(y, 1.0f, i);
	
    /* Evaluate te second polynom  (Pi/4 <= x <= 0) */
    V_ELT_FLOAT y2;
    y2 = z;
    y2 = VFMADD1_FLOAT(y2, sincof[0], sincof_1_vec, i);
    y2 = VFMADD_FLOAT(y2, z, sincof_2_vec, i);
    y2 = VMUL_FLOAT(y2, z, i);
    y2 = VFMADD_FLOAT(y2, x, x, i);
	
    /* select te correct result from te two polynoms */
    V_ELT_FLOAT y_sin = VMERGE_FLOAT(poly_mask, y, y2, i);
    V_ELT_FLOAT y_cos = VMERGE_FLOAT(poly_mask, y2, y, i);
	
	*sin_tmp = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(y_sin), VINTERP_FLOAT_INT(sign_bit_sin), i));
	*cos_tmp = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(y_cos), VINTERP_FLOAT_INT(sign_bit_cos), i));
}

static inline void sincosf_vec(float *src, float *s, float *c, int len)
{
    size_t i;
    V_ELT_FLOAT coscof_1_vec = VLOAD1_FLOAT(coscof[1], i);
    V_ELT_FLOAT coscof_2_vec = VLOAD1_FLOAT(coscof[2], i);
    V_ELT_FLOAT sincof_1_vec = VLOAD1_FLOAT(sincof[1], i);
    V_ELT_FLOAT sincof_2_vec = VLOAD1_FLOAT(sincof[2], i);

	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT y_sin, y_cos;
        sincosf_ps(x, &y_sin, &y_cos,
                   coscof_1_vec, coscof_2_vec,
                   sincof_1_vec, sincof_2_vec, i);
        VSTORE_FLOAT(s+l, y_sin, i);
        VSTORE_FLOAT(c+l, y_cos, i);
    }
}

static inline void sincosf_interleaved_vec(float *src, complex32_t *dst, int len)
{
    size_t i;
    V_ELT_FLOAT coscof_1_vec = VLOAD1_FLOAT(coscof[1], i);
    V_ELT_FLOAT coscof_2_vec = VLOAD1_FLOAT(coscof[2], i);
    V_ELT_FLOAT sincof_1_vec = VLOAD1_FLOAT(sincof[1], i);
    V_ELT_FLOAT sincof_2_vec = VLOAD1_FLOAT(sincof[2], i);

	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT y_sin, y_cos;
        sincosf_ps(x, &y_sin, &y_cos,
                   coscof_1_vec, coscof_2_vec,
                   sincof_1_vec, sincof_2_vec, i);
		VSTORE_FLOAT2SPLIT(dst+l,y_cos,y_sin,i);
    }
}

static inline void tanf_vec(float *src, float *dst, int len)
{
    size_t i;
    V_ELT_FLOAT TAN_P1_vec = VLOAD1_FLOAT(TAN_P1, i);
    V_ELT_FLOAT TAN_P2_vec = VLOAD1_FLOAT(TAN_P2, i);
    V_ELT_FLOAT TAN_P3_vec = VLOAD1_FLOAT(TAN_P3, i);
    V_ELT_FLOAT TAN_P4_vec = VLOAD1_FLOAT(TAN_P4, i);
    V_ELT_FLOAT TAN_P5_vec = VLOAD1_FLOAT(TAN_P5, i);

	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT xx = VLOAD_FLOAT(src+l, i);

        V_ELT_FLOAT x, y, z, zz;
        V_ELT_INT j;
        V_ELT_INT sign;
        V_ELT_FLOAT tmp;
        V_ELT_INT tmpi;
        V_ELT_BOOL32 jandone, jandtwo, xsupem4;

        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(xx), inv_sign_mask, i));
        sign = VAND1_INT(VINTERP_FLOAT_INT(xx), sign_mask, i);

        // compute x mod PIO4
        tmp = VMUL1_FLOAT(x, FOPI, i);
        j = VCVT_RTZ_FLOAT_INT(tmp, i);	
        y = VRTZ_FLOAT(tmp, i);

        jandone = VGT1_INT_BOOL(VAND1_INT(j, 1, i), 0, i);
        y = VADD1_FLOAT_MASK(jandone, y, 1.0f, i);
        j = VADD1_INT_MASK(jandone, j, 1, i);
        z = x;
        z = VFMACC1_FLOAT(z, minus_cephes_DP1, y, i);
        z = VFMACC1_FLOAT(z, minus_cephes_DP2, y, i);
        z = VFMACC1_FLOAT(z, minus_cephes_DP3, y, i);
        zz = VMUL_FLOAT(z, z, i);  // z*z

        // TODO : sould not be computed if X < 10e-4
        // 1.7e-8 relative error in [-pi/4, +pi/4]
        tmp = zz;
        tmp = VFMADD1_FLOAT(tmp, TAN_P0, TAN_P1_vec, i);
        tmp = VFMADD_FLOAT(tmp, zz, TAN_P2_vec, i);
        tmp = VFMADD_FLOAT(tmp, zz, TAN_P3_vec, i);
        tmp = VFMADD_FLOAT(tmp, zz, TAN_P4_vec, i);
        tmp = VFMADD_FLOAT(tmp, zz, TAN_P5_vec, i);
        tmp = VMUL_FLOAT(zz, tmp, i);

        tmp = VFMADD_FLOAT(tmp, z, z, i);
        xsupem4 = VGT1_FLOAT_BOOL(x, 1e-4f, i);
        y = VMERGE_FLOAT(xsupem4, z, tmp, i);

        jandtwo = VGT1_INT_BOOL(VAND1_INT(j, 2, i), 0, i);

        // xor(rcp(y)) gives not good enoug result
        tmp = VRDIV1_FLOAT(y, -1.0f, i);
        y = VMERGE_FLOAT(jandtwo, y, tmp, i);
        y = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(y), sign, i));
        VSTORE_FLOAT(dst+l, y, i);
    }
}

static inline void sumf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
    V_ELT_FLOAT vacc = VLOAD1_FLOAT(0.0f, n);
	svbool_t i;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va = VLOAD_FLOAT(src+l, i);
        vacc = VADD_FLOAT(vacc, va, i);
    }
#if 0 //ordered sum
	float tmp = 0.0f;
    *dst = VREDSUMORD_FLOAT(tmp, vacc, i);
#else	
    *dst = VREDSUM_FLOAT(vacc, i);
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
	size_t n = (size_t)len;
	V_ELT_FLOAT dummy;
    V_ELT_FLOAT vacc = VLOAD1_FLOAT(0.0f, n);
	svbool_t i;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va = VLOAD_FLOAT(src1+l, i);
        V_ELT_FLOAT vb = VLOAD_FLOAT(src2+l, i);
        vacc = VFMACC_FLOAT(vacc, va, vb, i);
    }

#if 0 //ordered sum
	float tmp = 0.0f;
    *dst = VREDSUMORD_FLOAT(tmp, vacc, i);
#else	
    *dst = VREDSUM_FLOAT(vacc, i);
#endif
}

static inline void dotcf_vec(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    size_t n = (size_t)(len);

    V_ELT_FLOAT vacc_Re = VLOAD1_FLOAT(0.0f, n);
    V_ELT_FLOAT vacc_Im = VLOAD1_FLOAT(0.0f, n);

	V_ELT_FLOAT dummy;
	svbool_t i;	
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT2 src1_vec = VLOAD_FLOAT2(src1+l, i);
        V_ELT_FLOAT2 src2_vec = VLOAD_FLOAT2(src2+l, i);
#if 1
        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src1_vec.__val[0], src2_vec.__val[0], i);
        V_ELT_FLOAT dstRe_vec = VFMSUB_FLOAT(src1_vec.__val[1], src2_vec.__val[1], tmp1, i);
        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1_vec.__val[0], src2_vec.__val[1], i);
        V_ELT_FLOAT dstIm_vec = VFMACC_FLOAT(tmp2, src2_vec.__val[0], src1_vec.__val[1], i);
#else
	    V_ELT_FLOAT src1Re_vec = svget2_f32(src1_vec,0);
        V_ELT_FLOAT src1Im_vec = svget2_f32(src1_vec,1);
        V_ELT_FLOAT src2Re_vec = svget2_f32(src2_vec,0);
        V_ELT_FLOAT src2Im_vec =  svget2_f32(src2_vec,1);
        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMSUB_FLOAT(src1Im_vec, src2Im_vec, tmp1, i);
        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMACC_FLOAT(tmp2, src2Re_vec, src1Im_vec, i);
			
#endif
        vacc_Re = VADD_FLOAT(vacc_Re, dstRe_vec, i);
        vacc_Im = VADD_FLOAT(vacc_Im, dstIm_vec, i);
    }
    dst->re = VREDSUM_FLOAT(vacc_Re, i);
    dst->im = VREDSUM_FLOAT(vacc_Im, i);
}

static inline void cplxtorealf_vec(complex32_t *src, float *dstRe, float *dstIm, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	svbool_t i;	
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT2 dst_vec = VLOAD_FLOAT2(src+l, i);
        V_ELT_FLOAT dstRe_vec = svget2_f32(dst_vec,0); 
        V_ELT_FLOAT dstIm_vec = svget2_f32(dst_vec,1);		
        VSTORE_FLOAT(dstRe+l, dstRe_vec, i);
        VSTORE_FLOAT(dstIm+l, dstIm_vec, i);
    }
}

static inline void realtocplxf_vec(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	svbool_t i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT srcRe_vec = VLOAD_FLOAT(srcRe+l, i);
        V_ELT_FLOAT srcIm_vec = VLOAD_FLOAT(srcIm+l, i);
        VSTORE_FLOAT2SPLIT(dst+l, srcRe_vec, srcIm_vec, i);
    }
}


static inline void cplxvecmulf_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    size_t n = (size_t)(len*2);
	V_ELT_FLOAT dummy;
	svbool_t i;		
	uint64_t numVals = svlen_f32(dummy);
	V_ELT_FLOAT zero = VLOAD1_FLOAT(0.0f,dummy);
	
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT src1_vec = VLOAD_FLOAT((float*)src1+l, i);
        V_ELT_FLOAT src2_vec = VLOAD_FLOAT((float*)src2+l, i);
		V_ELT_FLOAT dst_vec = VLOAD1_FLOAT(0.0f,i);
		dst_vec = VMUL_CFLOAT(dst_vec,src1_vec,src2_vec,i);
        VSTORE_FLOAT((float*)dst+l, dst_vec, i);

    }
}

static inline void cplxvecmulf_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    float *src1Re_tmp = src1Re;
    float *src1Im_tmp = src1Im;
    float *src2Re_tmp = src2Re;
    float *src2Im_tmp = src2Im;
    float *dstRe_tmp = dstRe;
    float *dstIm_tmp = dstIm;

    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	svbool_t i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT src1Re_vec = VLOAD_FLOAT(src1Re_tmp, i);
        V_ELT_FLOAT src1Im_vec = VLOAD_FLOAT(src1Im_tmp, i);
        V_ELT_FLOAT src2Re_vec = VLOAD_FLOAT(src2Re_tmp, i);
        V_ELT_FLOAT src2Im_vec = VLOAD_FLOAT(src2Im_tmp, i);
        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMSUB_FLOAT(src1Im_vec, src2Im_vec, tmp1, i);
        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMACC_FLOAT(tmp2, src2Re_vec, src1Im_vec, i);
        VSTORE_FLOAT(dstRe_tmp, dstRe_vec, i);
        VSTORE_FLOAT(dstIm_tmp, dstIm_vec, i);

        src1Re_tmp += numVals;
        src1Im_tmp += numVals;
        src2Re_tmp += numVals;
        src2Im_tmp += numVals;
        dstRe_tmp += numVals;
        dstIm_tmp += numVals;
    }
}

static inline void cplxvecdivf_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	svbool_t i;		
	uint64_t numVals = svlen_f32(dummy);
	
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT2 src1_vec = VLOAD_FLOAT2(src1+l, i);
        V_ELT_FLOAT2 src2_vec = VLOAD_FLOAT2(src2+l, i);
	    V_ELT_FLOAT src1Re_vec = svget2_f32(src1_vec,0);
        V_ELT_FLOAT src1Im_vec = svget2_f32(src1_vec,1);
        V_ELT_FLOAT src2Re_vec = svget2_f32(src2_vec,0);
        V_ELT_FLOAT src2Im_vec =  svget2_f32(src2_vec,1);

        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src2Re_vec, src2Re_vec, i);
        V_ELT_FLOAT c2d2 = VFMACC_FLOAT(tmp1, src2Im_vec, src2Im_vec, i);

        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMACC_FLOAT(tmp2, src1Im_vec, src2Im_vec, i);
        dstRe_vec = VDIV_FLOAT(dstRe_vec, c2d2, i);

        V_ELT_FLOAT tmp3 = VMUL_FLOAT(src1Im_vec, src2Re_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMSUB_FLOAT(src2Im_vec, src1Re_vec, tmp3, i);
        dstIm_vec = VDIV_FLOAT(dstIm_vec, c2d2, i);

        VSTORE_FLOAT2SPLIT(dst+l, dstRe_vec, dstIm_vec, i);
    }
}

static inline void cplxvecdivf_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    float *src1Re_tmp = src1Re;
    float *src1Im_tmp = src1Im;
    float *src2Re_tmp = src2Re;
    float *src2Im_tmp = src2Im;
    float *dstRe_tmp = dstRe;
    float *dstIm_tmp = dstIm;

    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	svbool_t i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT src1Re_vec = VLOAD_FLOAT(src1Re_tmp, i);
        V_ELT_FLOAT src1Im_vec = VLOAD_FLOAT(src1Im_tmp, i);
        V_ELT_FLOAT src2Re_vec = VLOAD_FLOAT(src2Re_tmp, i);
        V_ELT_FLOAT src2Im_vec = VLOAD_FLOAT(src2Im_tmp, i);

        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src2Re_vec, src2Re_vec, i);
        V_ELT_FLOAT c2d2 = VFMACC_FLOAT(tmp1, src2Im_vec, src2Im_vec, i);

        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMACC_FLOAT(tmp2, src1Im_vec, src2Im_vec, i);
        dstRe_vec = VDIV_FLOAT(dstRe_vec, c2d2, i);

        V_ELT_FLOAT tmp3 = VMUL_FLOAT(src1Im_vec, src2Re_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMSUB_FLOAT(src2Im_vec, src1Re_vec, tmp3, i);
        dstIm_vec = VDIV_FLOAT(dstIm_vec, c2d2, i);
        VSTORE_FLOAT(dstRe_tmp, dstRe_vec, i);
        VSTORE_FLOAT(dstIm_tmp, dstIm_vec, i);

        src1Re_tmp += numVals;
        src1Im_tmp += numVals;
        src2Re_tmp += numVals;
        src2Im_tmp += numVals;
        dstRe_tmp += numVals;
        dstIm_tmp += numVals;
    }
}

static inline void cplxconjvecmulf_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    size_t n = (size_t)(len*2);
	V_ELT_FLOAT dummy;
	svbool_t i;		
	uint64_t numVals = svlen_f32(dummy);
	V_ELT_FLOAT zero = VLOAD1_FLOAT(0.0f,dummy);
	
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT src1_vec = VLOAD_FLOAT((float*)src1+l, i);
        V_ELT_FLOAT src2_vec = VLOAD_FLOAT((float*)src2+l, i);
		V_ELT_FLOAT dst_vec = VLOAD1_FLOAT(0.0f,i);
		dst_vec = VMULCONJA_CFLOAT(dst_vec,src2_vec,src1_vec,i);
        VSTORE_FLOAT((float*)dst+l, dst_vec, i);

    }
}


static inline void cplxconjvecmulf_vec_split(float *src1Re, float *src1Im, float *src2Re, float *src2Im, float *dstRe, float *dstIm, int len)
{
    float *src1Re_tmp = src1Re;
    float *src1Im_tmp = src1Im;
    float *src2Re_tmp = src2Re;
    float *src2Im_tmp = src2Im;
    float *dstRe_tmp = dstRe;
    float *dstIm_tmp = dstIm;

    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	svbool_t i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT src1Re_vec = VLOAD_FLOAT(src1Re_tmp, i);
        V_ELT_FLOAT src1Im_vec = VLOAD_FLOAT(src1Im_tmp, i);
        V_ELT_FLOAT src2Re_vec = VLOAD_FLOAT(src2Re_tmp, i);
        V_ELT_FLOAT src2Im_vec = VLOAD_FLOAT(src2Im_tmp, i);
        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMACC_FLOAT(tmp1, src1Im_vec, src2Im_vec, i);
        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Im_vec, src2Re_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMSUB_FLOAT(src2Im_vec, src1Re_vec, tmp2, i);  // vs1*vd - vs2
        VSTORE_FLOAT(dstRe_tmp, dstRe_vec, i);
        VSTORE_FLOAT(dstIm_tmp, dstIm_vec, i);

        src1Re_tmp += numVals;
        src1Im_tmp += numVals;
        src2Re_tmp += numVals;
        src2Im_tmp += numVals;
        dstRe_tmp += numVals;
        dstIm_tmp += numVals;
    }
}

// IEEE 754 round to nearest even
static inline void rintf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT a, b;
        a = VLOAD_FLOAT(src+l, i);
        b = VRNE_FLOAT(a, i);
        VSTORE_FLOAT(dst+l, b, i);
    }
}

//C Roundf, round away from zero
static inline void roundf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT a, b;
        a = VLOAD_FLOAT(src+l, i);
        b = VRNA_FLOAT(a, i);
        VSTORE_FLOAT(dst+l, b, i);
    }
}

static inline void ceilf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT a, b;
        a = VLOAD_FLOAT(src+l, i);
        b = VRINF_FLOAT(a, i);
        VSTORE_FLOAT(dst+l, b, i);
    }
}

static inline void floorf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT a, b;
        a = VLOAD_FLOAT(src+l, i);
        b = VRMINF_FLOAT(a, i);
        VSTORE_FLOAT(dst+l, b, i);
    }
}

static inline void truncf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		svbool_t i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT a, b;
        a = VLOAD_FLOAT(src+l, i);
        b = VRTZ_FLOAT(a, i);
        VSTORE_FLOAT(dst+l, b, i);
    }
}