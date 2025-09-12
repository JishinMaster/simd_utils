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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src+l, i);
        VSTORE_FLOAT(dst+l, VMUL1_FLOAT(va, value, i), i);
    }
}

#warning "to be improved with fma"
static inline void sinf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);

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
    V_ELT_FLOAT c_sincof_p1_vec = VLOAD1_FLOAT(c_sincof_p1, i);

	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);

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
        y = VFMASQ1_FLOAT(y, z, c_coscof_p2, i);
        y = VMUL_FLOAT(y, z, i);
        y = VMUL_FLOAT(y, z, i);
        y = VFMACC1_FLOAT(y, -0.5f, z, i);  // y = y -0.5*z
        y = VADD1_FLOAT(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT_FLOAT y2 = z;
        y2 = VFMADD1_FLOAT(y2, c_sincof_p0, c_sincof_p1_vec, i);
        y2 = VFMASQ1_FLOAT(y2, z, c_sincof_p2, i);
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
                              V_ELT_FLOAT sincof_1_vec,
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
    y = VFMASQ1_FLOAT(y, z, coscof[2], i);
    y = VMUL_FLOAT(y, z, i);
    y = VMUL_FLOAT(y, z, i);
    y = VFMACC1_FLOAT(y, -0.5f, z, i);  // y = y -0.5*z
    y = VADD1_FLOAT(y, 1.0f, i);
	
    /* Evaluate te second polynom  (Pi/4 <= x <= 0) */
    V_ELT_FLOAT y2;
    y2 = z;
    y2 = VFMADD1_FLOAT(y2, sincof[0], sincof_1_vec, i);
    y2 = VFMASQ1_FLOAT(y2, z, sincof[2], i);
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
    V_ELT_FLOAT sincof_1_vec = VLOAD1_FLOAT(sincof[1], i);

	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT y_sin, y_cos;
        sincosf_ps(x, &y_sin, &y_cos,
                   coscof_1_vec, sincof_1_vec, i);
        VSTORE_FLOAT(s+l, y_sin, i);
        VSTORE_FLOAT(c+l, y_cos, i);
    }
}

static inline void sincosf_interleaved_vec(float *src, complex32_t *dst, int len)
{
    size_t i;
    V_ELT_FLOAT coscof_1_vec = VLOAD1_FLOAT(coscof[1], i);
    V_ELT_FLOAT sincof_1_vec = VLOAD1_FLOAT(sincof[1], i);

	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT y_sin, y_cos;
        sincosf_ps(x, &y_sin, &y_cos,
                   coscof_1_vec, sincof_1_vec, i);
		VSTORE_FLOAT2SPLIT(dst+l,y_cos,y_sin,i);
    }
}

static inline void tanf_vec(float *src, float *dst, int len)
{
    size_t i;
    V_ELT_FLOAT TAN_P1_vec = VLOAD1_FLOAT(TAN_P1, i);
	
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
        tmp = VFMASQ1_FLOAT(tmp, zz, TAN_P2, i);
        tmp = VFMASQ1_FLOAT(tmp, zz, TAN_P3, i);
        tmp = VFMASQ1_FLOAT(tmp, zz, TAN_P4, i);
        tmp = VFMASQ1_FLOAT(tmp, zz, TAN_P5, i);
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
	V_ELT_BOOL32 i;
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

#warning "TODO : result not good enough"
static inline void dotf_vec(float *src1, float *src2, int len, float *dst)
{
	size_t n = (size_t)len;
	V_ELT_FLOAT dummy;
    V_ELT_FLOAT vacc = VLOAD1_FLOAT(0.0f, n);
	V_ELT_BOOL32 i;
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

#warning "TODO : result not good enough"
static inline void dotcf_vec(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    size_t n = (size_t)(len);

    V_ELT_FLOAT vacc_Re = VLOAD1_FLOAT(0.0f, n);
    V_ELT_FLOAT vacc_Im = VLOAD1_FLOAT(0.0f, n);

	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;	
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
	V_ELT_BOOL32 i;	
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
	V_ELT_BOOL32 i;		
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
	V_ELT_BOOL32 i;		
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
	V_ELT_BOOL32 i;		
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
	V_ELT_BOOL32 i;		
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
	V_ELT_BOOL32 i;		
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

static inline void cplxconjf_vec(complex32_t *src, complex32_t *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT2 src_vec = VLOAD_FLOAT2(src+l, i);
	    V_ELT_FLOAT srcRe_vec = svget2_f32(src_vec,0);
        V_ELT_FLOAT srcIm_vec = svget2_f32(src_vec,1);
        srcIm_vec = VINTERP_INT_FLOAT(VXOR1_INT(VINTERP_FLOAT_INT(srcIm_vec), (int32_t) 0x80000000, i));
        VSTORE_FLOAT2SPLIT(dst+l, srcRe_vec, srcIm_vec, i);
    }
}

static inline void cplxconjvecmulf_vec(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    size_t n = (size_t)(len*2);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
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
	V_ELT_BOOL32 i;		
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

static inline void magnitudef_split_vec(float *srcRe, float *srcIm, float *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT re_tmp = VLOAD_FLOAT(srcRe+l, i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT im_tmp = VLOAD_FLOAT(srcIm+l, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, im_tmp, im_tmp, i);
        VSTORE_FLOAT(dst+l, VSQRT_FLOAT(tmp, i), i);
    }
}

static inline void powerspectf_split_vec(float *srcRe, float *srcIm, float *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT re_tmp = VLOAD_FLOAT(srcRe+l, i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT im_tmp = VLOAD_FLOAT(srcIm+l, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, im_tmp, im_tmp, i);
        VSTORE_FLOAT(dst+l, tmp, i);
    }
}

static inline void powerspectf_interleaved_vec(complex32_t *src, float *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT2 src_vec = VLOAD_FLOAT2(src+l, i);
	    V_ELT_FLOAT dstRe_vec = svget2_f32(src_vec,0);
        V_ELT_FLOAT dstIm_vec = svget2_f32(src_vec,1);		
        V_ELT_FLOAT re2 = VMUL_FLOAT(dstRe_vec, dstRe_vec, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, dstIm_vec, dstIm_vec, i);
        VSTORE_FLOAT(dst+l, tmp, i);
    }
}

static inline void magnitudef_interleaved_vec(complex32_t *src, float *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT2 src_vec = VLOAD_FLOAT2(src+l, i);
	    V_ELT_FLOAT dstRe_vec = svget2_f32(src_vec,0);
        V_ELT_FLOAT dstIm_vec = svget2_f32(src_vec,1);		
        V_ELT_FLOAT re2 = VMUL_FLOAT(dstRe_vec, dstRe_vec, i);
        V_ELT_FLOAT tmp = VFMACC_FLOAT(re2, dstIm_vec, dstIm_vec, i);
        tmp = VSQRT_FLOAT(tmp, i);
        VSTORE_FLOAT(dst+l, tmp, i);
    }
}

static inline void maxeveryf_vec(float *src1, float *src2, float *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va, vb;
        va = VLOAD_FLOAT(src1+l, i);
        vb = VLOAD_FLOAT(src2+l, i);
        VSTORE_FLOAT(dst+l, VMAX_FLOAT(va, vb, i), i);
    }
}

static inline void mineveryf_vec(float *src1, float *src2, float *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;		
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va, vb;
        va = VLOAD_FLOAT(src1+l, i);
        vb = VLOAD_FLOAT(src2+l, i);
        VSTORE_FLOAT(dst+l, VMIN_FLOAT(va, vb, i), i);
    }
}

static inline void minmaxf_vec(float *src, int len, float *min_value, float *max_value)
{
    size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
    V_ELT_FLOAT minv = VLOAD_FLOAT(src, i);  // or vfmv_v_f_f32m1
    V_ELT_FLOAT maxv = minv;
	for (size_t l=numVals; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT v1 = VLOAD_FLOAT(src+l, i);
        minv = VMIN_FLOAT(v1, minv, i);
        maxv = VMAX_FLOAT(v1, maxv, i);
    }
    *min_value = VREDMIN_FLOAT(minv, i);
    *max_value = VREDMAX_FLOAT(maxv, i); 
}

static inline void threshold_gt_f_vec(float *src, float *dst, int len, float value)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src+l, i);
        VSTORE_FLOAT(dst+l, VMIN1_FLOAT(va, value, i), i);
    }
}

static inline void threshold_lt_f_vec(float *src, float *dst, int len, float value)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT va;
        va = VLOAD_FLOAT(src+l, i);
        VSTORE_FLOAT(dst+l, VMAX1_FLOAT(va, value, i), i);
    }
}

static inline void threshold_gtabs_f_vec(float *src, float *dst, int len, float value)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;	
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);		
        V_ELT_FLOAT va = VLOAD_FLOAT(src+l, i);
        V_ELT_INT va_sign = VAND1_INT(VINTERP_FLOAT_INT(va), sign_mask, i);
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        V_ELT_FLOAT sval = VMIN1_FLOAT(va_abs, value, i);
        sval = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(sval), va_sign, i));
        VSTORE_FLOAT(dst+l, sval, i);
    }
}

static inline void threshold_ltabs_f_vec(float *src, float *dst, int len, float value)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;	
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT va = VLOAD_FLOAT(src+l, i);
        V_ELT_INT va_sign = VAND1_INT(VINTERP_FLOAT_INT(va), sign_mask, i);
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        V_ELT_FLOAT sval = VMAX1_FLOAT(va_abs, value, i);
        sval = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(sval), va_sign, i));
        VSTORE_FLOAT(dst+l, sval, i);
    }
}

static inline void threshold_ltval_gtval_f_vec(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;	
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT va = VLOAD_FLOAT(src+l, i);
        V_ELT_BOOL32 lt_mask = VLT1_FLOAT_BOOL(va, ltlevel, i);
        V_ELT_BOOL32 gt_mask = VGT1_FLOAT_BOOL(va, gtlevel, i);
        V_ELT_FLOAT tmp = VMERGE1_FLOAT(lt_mask, va, ltvalue, i);
        tmp = VMERGE1_FLOAT(gt_mask, tmp, gtvalue, i);
        VSTORE_FLOAT(dst+l, tmp, i);
    }
}

static inline void sqrtf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;	
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT va = VLOAD_FLOAT(src+l, i);
        VSTORE_FLOAT(dst+l, VSQRT_FLOAT(va, i), i);
    }
}

static inline void fabsf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i;	
	uint64_t numVals = svlen_f32(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT va = VLOAD_FLOAT(src+l, i);
        VSTORE_FLOAT(dst+l, VABS_FLOAT(va, i), i);
    }
}


static inline void log10f_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);
    V_ELT_FLOAT c_cephes_log_p1_vec = VLOAD1_FLOAT(c_cephes_log_p1, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
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

        // substract 1.0f if mask is true (x < SQRTF). To be optimised
        e = VSUB_FLOAT(e, VMERGE1_FLOAT(mask, zero_vec, 1.0f, i), i);
        x = VADD_FLOAT(x, tmp, i);

        V_ELT_FLOAT z = VMUL_FLOAT(x, x, i);
        V_ELT_FLOAT y = x;
        y = VFMADD1_FLOAT(y, c_cephes_log_p0, c_cephes_log_p1_vec, i);
        y = VFMASQ1_FLOAT(y, x, c_cephes_log_p2, i);
        y = VFMASQ1_FLOAT(y, x, c_cephes_log_p3, i);
        y = VFMASQ1_FLOAT(y, x, c_cephes_log_p4, i);
        y = VFMASQ1_FLOAT(y, x, c_cephes_log_p5, i);
        y = VFMASQ1_FLOAT(y, x, c_cephes_log_p6, i);
        y = VFMASQ1_FLOAT(y, x, c_cephes_log_p7, i);
        y = VFMASQ1_FLOAT(y, x, c_cephes_log_p8, i);
        y = VMUL_FLOAT(y, x, i);
        y = VMUL_FLOAT(y, z, i);
        y = VFMACC1_FLOAT(y, -0.5f, z, i);  // y = y -0.5*z

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

        x = VMERGE1_FLOAT(invalid_mask, x, 0xFFFFFFFF, i);

        VSTORE_FLOAT(dst+l, x, i);
    }
}

static inline V_ELT_FLOAT log_ps(V_ELT_FLOAT x,
                                  V_ELT_FLOAT zero_vec,
                                  V_ELT_FLOAT c_cephes_log_p1_vec,
                                  V_ELT_BOOL32 i)
{
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
    V_ELT_FLOAT y = x;
    y = VFMADD1_FLOAT(y, c_cephes_log_p0, c_cephes_log_p1_vec, i);
    y = VFMASQ1_FLOAT(y, x, c_cephes_log_p2, i);
    y = VFMASQ1_FLOAT(y, x, c_cephes_log_p3, i);
    y = VFMASQ1_FLOAT(y, x, c_cephes_log_p4, i);
    y = VFMASQ1_FLOAT(y, x, c_cephes_log_p5, i);
    y = VFMASQ1_FLOAT(y, x, c_cephes_log_p6, i);
    y = VFMASQ1_FLOAT(y, x, c_cephes_log_p7, i);
    y = VFMASQ1_FLOAT(y, x, c_cephes_log_p8, i);
    y = VMUL_FLOAT(y, x, i);
    y = VMUL_FLOAT(y, z, i);

    y = VFMACC1_FLOAT(y, c_cephes_log_q1, e, i);
    y = VFMACC1_FLOAT(y, -0.5f, z, i);  // y = y -0.5*z
    tmp = y;
    tmp = VFMACC1_FLOAT(tmp, c_cephes_log_q2, e, i);
    x = VADD_FLOAT(x, tmp, i);

    x = VMERGE1_FLOAT(invalid_mask, x, 0xFFFFFFFF, i);
    return x;
}

static inline void lnf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);
    V_ELT_FLOAT c_cephes_log_p1_vec = VLOAD1_FLOAT(c_cephes_log_p1, i);


	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        x = log_ps(x, zero_vec, c_cephes_log_p1_vec, i);
        VSTORE_FLOAT(dst+l, x, i);
    }
}

static inline void log2f_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);
    V_ELT_FLOAT c_cephes_log_p1_vec = VLOAD1_FLOAT(c_cephes_log_p1, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
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
        V_ELT_FLOAT y = x;
		y = VFMADD1_FLOAT(y, c_cephes_log_p0, c_cephes_log_p1_vec, i);
		y = VFMASQ1_FLOAT(y, x, c_cephes_log_p2, i);
		y = VFMASQ1_FLOAT(y, x, c_cephes_log_p3, i);
		y = VFMASQ1_FLOAT(y, x, c_cephes_log_p4, i);
		y = VFMASQ1_FLOAT(y, x, c_cephes_log_p5, i);
		y = VFMASQ1_FLOAT(y, x, c_cephes_log_p6, i);
		y = VFMASQ1_FLOAT(y, x, c_cephes_log_p7, i);
		y = VFMASQ1_FLOAT(y, x, c_cephes_log_p8, i);
        y = VMUL_FLOAT(y, x, i);
        y = VMUL_FLOAT(y, z, i);
        y = VFMACC1_FLOAT(y, -0.5f, z, i);  // y = y -0.5*z

        tmp = VADD_FLOAT(x, y, i);

        z = VMUL1_FLOAT(y, c_cephes_LOG2EA, i);
        z = VFMACC1_FLOAT(z, c_cephes_LOG2EA, x, i);
        z = VADD_FLOAT(z, tmp, i);
        x = VADD_FLOAT(z, e, i);

        // print_vec(x);printf("\n");
        // could we use merge function? VMERGE_FLOAT? create a nan vec?
        x = VMERGE1_FLOAT(invalid_mask, x, 0xFFFFFFFF, i);
        VSTORE_FLOAT(dst+l, x, i);
    }
}

static inline V_ELT_FLOAT atanf_ps(V_ELT_FLOAT xx,
                                    V_ELT_FLOAT ATAN_P1_vec,
                                    V_ELT_FLOAT min1_vec,
                                    V_ELT_BOOL32 i)
{
    V_ELT_FLOAT x, y, z;
    V_ELT_INT sign;
    V_ELT_BOOL32 suptan3pi8, inftan3pi8suppi8;
    V_ELT_FLOAT tmp, tmp2;
    V_ELT_BOOL32 tmpb1, tmpb2;

    x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(xx), inv_sign_mask, i));
    sign = VAND1_INT(VINTERP_FLOAT_INT(xx), sign_mask, i);

    /* range reduction */
    y = VLOAD1_FLOAT(0.0f, i);
    suptan3pi8 = VGT1_FLOAT_BOOL(x, TAN3PI8F, i);
    tmp = VDIV_FLOAT(min1_vec, x, i);
    x = VMERGE_FLOAT(suptan3pi8, x, tmp, i);
    y = VMERGE1_FLOAT(suptan3pi8, y, PIO2F, i);

    tmpb1 = VLE1_FLOAT_BOOL(x, TAN3PI8F, i);
    tmpb2 = VGT1_FLOAT_BOOL(x, TANPI8F, i);
    inftan3pi8suppi8 = VAND_BOOL(tmpb1, tmpb2, i);

    // To be optimised with RCP?
    tmp = VSUB1_FLOAT(x, 1.0f, i);
    tmp2 = VADD1_FLOAT(x, 1.0f, i);
    tmp = VDIV_FLOAT(tmp, tmp2, i);
    x = VMERGE_FLOAT(inftan3pi8suppi8, x, tmp, i);
    y = VMERGE1_FLOAT(inftan3pi8suppi8, y, PIO4F, i);
    z = VMUL_FLOAT(x, x, i);

    tmp = z;
    tmp = VFMADD1_FLOAT(tmp, ATAN_P0, ATAN_P1_vec, i);
    tmp = VFMASQ1_FLOAT(tmp, z, ATAN_P2, i);
    tmp = VFMASQ1_FLOAT(tmp, z, ATAN_P3, i);
    tmp = VMUL_FLOAT(z, tmp, i);
    tmp = VFMADD_FLOAT(tmp, x, x, i);
    y = VADD_FLOAT(y, tmp, i);
    y = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(y), sign, i));
    return y;
}

static inline void atanf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
    V_ELT_FLOAT ATAN_P1_vec = VLOAD1_FLOAT(ATAN_P1, i);
    V_ELT_FLOAT min1_vec = VLOAD1_FLOAT(-1.0f, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT xx = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT y;
        y = atanf_ps(xx, ATAN_P1_vec, min1_vec, i);
        VSTORE_FLOAT(dst+l, y, i);
    }
}

static inline V_ELT_FLOAT atan2f_ps(V_ELT_FLOAT y, V_ELT_FLOAT x, V_ELT_FLOAT ATAN_P1_vec, V_ELT_FLOAT min1_vec, V_ELT_BOOL32 i)
{
    V_ELT_FLOAT z, w;
    V_ELT_BOOL32 xinfzero, yinfzero, xeqzero, yeqzero;
    V_ELT_BOOL32 xeqzeroandyinfzero, yeqzeroandxinfzero;
    V_ELT_BOOL32 specialcase;
    V_ELT_FLOAT tmp, tmp2;

    xinfzero = VLT1_FLOAT_BOOL(x, 0.0f, i);  // code =2
    yinfzero = VLT1_FLOAT_BOOL(y, 0.0f, i);  // code = code |1;

    xeqzero = VEQ1_FLOAT_BOOL(x, 0.0f, i);
    yeqzero = VEQ1_FLOAT_BOOL(y, 0.0f, i);

    xeqzeroandyinfzero = VAND_BOOL(xeqzero, yinfzero, i);
    yeqzeroandxinfzero = VAND_BOOL(yeqzero, xinfzero, i);

    z = VLOAD1_FLOAT(PIO2F, i);
    z = VMERGE1_FLOAT(xeqzeroandyinfzero, z, mPIO2F, i);
    z = VMERGE1_FLOAT(yeqzero, z, 0.0f, i);
    z = VMERGE1_FLOAT(yeqzeroandxinfzero, z, PIF, i);
    specialcase = VOR_BOOL(xeqzero, yeqzero, i);

    w = VLOAD1_FLOAT(0.0f, i);
    w = VMERGE1_FLOAT(VAND_BOOL(VNOT_BOOL(yinfzero, i), xinfzero, i), w, PIF, i);  // y >= 0 && x<0
    w = VMERGE1_FLOAT(VAND_BOOL(yinfzero, xinfzero, i), w, mPIF, i);                // y < 0 && x<0

    tmp = VDIV_FLOAT(y, x, i);
    tmp = atanf_ps(tmp, ATAN_P1_vec, min1_vec, i);
    tmp = VADD_FLOAT(w, tmp, i);
    z = VMERGE_FLOAT(specialcase, tmp, z, i);  // atanf(y/x) if not in special case
    return z;
}

static inline void atan2f_vec(float *src1, float *src2, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
    V_ELT_FLOAT ATAN_P1_vec = VLOAD1_FLOAT(ATAN_P1, i);
    V_ELT_FLOAT min1_vec = VLOAD1_FLOAT(-1.0f, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT y = VLOAD_FLOAT(src1+l, i);
        V_ELT_FLOAT x = VLOAD_FLOAT(src2+l, i);

        V_ELT_FLOAT z = atan2f_ps(y, x,
                                   ATAN_P1_vec, min1_vec, i);
        VSTORE_FLOAT(dst+l, z, i);
    }
}

static inline void atan2f_interleaved_vec(complex32_t *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
    V_ELT_FLOAT ATAN_P1_vec = VLOAD1_FLOAT(ATAN_P1, i);
    V_ELT_FLOAT min1_vec = VLOAD1_FLOAT(-1.0f, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT2 src_vec = VLOAD_FLOAT2(src+l, i);
	    V_ELT_FLOAT x = svget2_f32(src_vec,0);
        V_ELT_FLOAT y = svget2_f32(src_vec,1);		
        V_ELT_FLOAT z = atan2f_ps(y, x,
                                   ATAN_P1_vec, min1_vec, i);
        VSTORE_FLOAT(dst+l, z, i);
    }
}

static inline void asinf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
    V_ELT_FLOAT ASIN_P1_vec = VLOAD1_FLOAT(ASIN_P1, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT a, z, z_tmp;
        V_ELT_INT sign;
        V_ELT_BOOL32 ainfem4, asup0p5, xsup1;
        V_ELT_FLOAT tmp;

        a = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));
        sign = VAND1_INT(VINTERP_FLOAT_INT(x), sign_mask, i);

        ainfem4 = VLT1_FLOAT_BOOL(a, 1.0e-4f, i);  // if( a < 1.0e-4f )
        asup0p5 = VGT1_FLOAT_BOOL(a, 0.5f, i);     // if( a > 0.5f ) flag = 1 else 0
        z_tmp = VRSUB1_FLOAT(a, 1.0f, i);
        z_tmp = VMUL1_FLOAT(z_tmp, 0.5f, i);
        tmp = VMUL_FLOAT(a, a, i);
        z = VMERGE_FLOAT(asup0p5, tmp, z_tmp, i);
        x = VMERGE_FLOAT(asup0p5, a, VSQRT_FLOAT(z, i), i);
        xsup1 = VGT1_FLOAT_BOOL(x, 1.0f, i);

        tmp = z;
        tmp = VFMADD1_FLOAT(tmp, ASIN_P0, ASIN_P1_vec, i);
        tmp = VFMASQ1_FLOAT(tmp, z, ASIN_P2, i);
        tmp = VFMASQ1_FLOAT(tmp, z, ASIN_P3, i);
        tmp = VFMASQ1_FLOAT(tmp, z, ASIN_P4, i);
        tmp = VMUL_FLOAT(z, tmp, i);
        tmp = VFMADD_FLOAT(tmp, x, x, i);

        z = tmp;
        z_tmp = VADD_FLOAT(z, z, i);
        z_tmp = VRSUB1_FLOAT(z_tmp, PIO2F, i);
        z = VMERGE_FLOAT(asup0p5, z, z_tmp, i);

        // done:
        z = VMERGE_FLOAT(ainfem4, z, a, i);
        z = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(z), sign, i));

        // if (x > 1.0) then return 0.0
        z = VMERGE1_FLOAT(xsup1, z, 0.0f, i);
        VSTORE_FLOAT(dst+l, z, i);
    }
}

static inline void acoshf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);
    V_ELT_FLOAT c_cephes_log_p1_vec = VLOAD1_FLOAT(c_cephes_log_p1, i);

    V_ELT_FLOAT ACOSH_P1_vec = VLOAD1_FLOAT(ACOSH_P1, i);


	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT z, z_first_branch, z_second_branch;
        V_ELT_BOOL32 xsup1500, zinf0p5, xinf1;
        V_ELT_FLOAT tmp;

        xsup1500 = VGT1_FLOAT_BOOL(x, 1500.0f, i);  // return  (logf(x) + LOGE2F)
        xinf1 = VLT1_FLOAT_BOOL(x, 1.0f, i);        // return 0

        z = VSUB1_FLOAT(x, 1.0f, i);
        zinf0p5 = VLT1_FLOAT_BOOL(z, 0.5f, i);  // first and second branch

        // First Branch (z < 0.5)
        z_first_branch = VFMADD1_FLOAT(z, ACOSH_P0, ACOSH_P1_vec, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, z, ACOSH_P2, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, z, ACOSH_P3, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, z, ACOSH_P4, i);
        z_first_branch = VMUL_FLOAT(z_first_branch, VSQRT_FLOAT(z, i), i);

        // Second Branch
        z_second_branch = VFMADD_FLOAT(z, x, z, i);
        z_second_branch = VSQRT_FLOAT(z_second_branch, i);
        z_second_branch = VADD_FLOAT(x, z_second_branch, i);
        z_second_branch = log_ps(z_second_branch, zero_vec, c_cephes_log_p1_vec, i);

        z = VMERGE_FLOAT(zinf0p5, z_second_branch, z_first_branch, i);
        tmp = log_ps(x, zero_vec, c_cephes_log_p1_vec, i);
        tmp = VADD1_FLOAT(tmp, LOGE2F, i);
        z = VMERGE_FLOAT(xsup1500, z, tmp, i);
        z = VMERGE1_FLOAT(xinf1, z, 0.0f, i);
        VSTORE_FLOAT(dst+l, z, i);
    }
}

static inline void asinhf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);
    V_ELT_FLOAT c_cephes_log_p1_vec = VLOAD1_FLOAT(c_cephes_log_p1, i);

    V_ELT_FLOAT ASINH_P1_vec = VLOAD1_FLOAT(ASINH_P1, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT xx = VLOAD_FLOAT(src+l, i);

        V_ELT_FLOAT x, tmp, z, z_first_branch, z_second_branch;
        V_ELT_BOOL32 xsup1500, xinf0p5;
        V_ELT_INT xxinf0;

        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(xx), inv_sign_mask, i));
        xsup1500 = VGT1_FLOAT_BOOL(x, 1500.0f, i);
        xinf0p5 = VLT1_FLOAT_BOOL(x, 0.5f, i);

        xxinf0 = VAND1_INT(VINTERP_FLOAT_INT(xx), sign_mask, i);

        tmp = VMUL_FLOAT(x, x, i);
        // First Branch (x < 0.5)
        z_first_branch = tmp;
        z_first_branch = VFMADD1_FLOAT(z_first_branch, ASINH_P0, ASINH_P1_vec, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, tmp, ASINH_P2, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, tmp, ASINH_P3, i);
        z_first_branch = VMUL_FLOAT(z_first_branch, tmp, i);
        z_first_branch = VFMADD_FLOAT(z_first_branch, x, x, i);

        // Second Branch
        z_second_branch = VSQRT_FLOAT(VADD1_FLOAT(tmp, 1.0f, i), i);
        z_second_branch = log_ps(VADD_FLOAT(z_second_branch, x, i), zero_vec, c_cephes_log_p1_vec, i);

        z = VMERGE_FLOAT(xinf0p5, z_second_branch, z_first_branch, i);
        tmp = log_ps(x, zero_vec, c_cephes_log_p1_vec, i);
        tmp = VADD1_FLOAT(tmp, LOGE2F, i);
        z = VMERGE_FLOAT(xsup1500, z, tmp, i);
        z = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(z), xxinf0, i));
        VSTORE_FLOAT(dst+l, z, i);
    }
}

static inline void atanhf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
    V_ELT_FLOAT zero_vec = VLOAD1_FLOAT(0.0f, i);
    V_ELT_FLOAT c_cephes_log_p1_vec = VLOAD1_FLOAT(c_cephes_log_p1, i);

    V_ELT_FLOAT ATANH_P1_vec = VLOAD1_FLOAT(ATANH_P1, i);
    V_ELT_FLOAT one_vec = VLOAD1_FLOAT(1.0f, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT z, tmp, tmp2, z_first_branch, z_second_branch;
        V_ELT_BOOL32 xsup1, xinfmin1, zinf1emin4, zinf0p5;

        z = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

        xsup1 = VGE1_FLOAT_BOOL(x, 1.0f, i);
        xinfmin1 = VLE1_FLOAT_BOOL(x, -1.0f, i);
        zinf1emin4 = VLT1_FLOAT_BOOL(z, -1e-4f, i);
        zinf0p5 = VLT1_FLOAT_BOOL(z, 0.5f, i);

        // First branch
        tmp = VMUL_FLOAT(x, x, i);
        z_first_branch = tmp;
        z_first_branch = VFMADD1_FLOAT(z_first_branch, ATANH_P0, ATANH_P1_vec, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, tmp, ATANH_P2, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, tmp, ATANH_P3, i);
        z_first_branch = VFMASQ1_FLOAT(z_first_branch, tmp, ATANH_P4, i);
        z_first_branch = VMUL_FLOAT(z_first_branch, tmp, i);
        z_first_branch = VFMADD_FLOAT(z_first_branch, x, x, i);

        // Second branch
        // RISCV, could be replace with rcp equivalent vfrec?
        // only 7 bits precision vs rcp 12bits (out of 24)
        tmp = VRSUB1_FLOAT(x, 1.0f, i);  // 1 -x
        tmp2 = VDIV_FLOAT(one_vec, tmp, i);
        tmp = VFMADD_FLOAT(tmp2, x, tmp2, i);
        z_second_branch = log_ps(tmp, zero_vec, c_cephes_log_p1_vec, i);
        z_second_branch = VMUL1_FLOAT(z_second_branch, 0.5f, i);

        z = VMERGE_FLOAT(zinf0p5, z_second_branch, z_first_branch, i);
        z = VMERGE_FLOAT(zinf1emin4, z, x, i);

        z = VMERGE1_FLOAT(xsup1, z, MAXNUMF, i);
        z = VMERGE1_FLOAT(xinfmin1, z, -MAXNUMF, i);
        VSTORE_FLOAT(dst+l, z, i);
    }
}

static inline V_ELT_FLOAT exp_ps(V_ELT_FLOAT x,
                                  V_ELT_FLOAT Op5_vec,
                                  V_ELT_FLOAT cephes_exp_p1_vec,
                                  V_ELT_BOOL32 i)
{
    V_ELT_FLOAT z_tmp, z, fx;
    V_ELT_INT n;
    V_ELT_BOOL32 xsupmaxlogf, xinfminglogf;

    xsupmaxlogf = VGT1_FLOAT_BOOL(x, MAXLOGF, i);
    xinfminglogf = VLT1_FLOAT_BOOL(x, MINLOGF, i);

    /* Express e**x = e**g 2**n
     *   = e**g e**( n loge(2) )
     *   = e**( g + n loge(2) )
     */
    fx = VFMADD1_FLOAT(x, c_cephes_LOG2EF, Op5_vec, i);
    z = VRMINF_FLOAT(fx, i);	
    n = VCVT_FLOAT_INT(z, i);

    x = VFMACC1_FLOAT(x, -c_cephes_exp_C1, z, i);
    x = VFMACC1_FLOAT(x, -c_cephes_exp_C2, z, i);	

    n = VADD1_INT(n, 0x7f, i);
    n = VSLL1_INT(n, 23, i);
    V_ELT_FLOAT pow2n = VINTERP_INT_FLOAT(n);

    z = VMUL_FLOAT(x, x, i);

    z_tmp = x;
    z_tmp = VFMADD1_FLOAT(z_tmp, c_cephes_exp_p0, cephes_exp_p1_vec, i);
    z_tmp = VFMASQ1_FLOAT(z_tmp, x, c_cephes_exp_p2, i);
    z_tmp = VFMASQ1_FLOAT(z_tmp, x, c_cephes_exp_p3, i);
    z_tmp = VFMASQ1_FLOAT(z_tmp, x, c_cephes_exp_p4, i);
    z_tmp = VFMASQ1_FLOAT(z_tmp, x, c_cephes_exp_p5, i);
    z_tmp = VFMADD_FLOAT(z_tmp, z, x, i);

    /* build 2^n */
    z_tmp = VFMADD_FLOAT(z_tmp, pow2n, pow2n, i);

    z = VMERGE1_FLOAT(xsupmaxlogf, z_tmp, MAXNUMF, i);
    z = VMERGE1_FLOAT(xinfminglogf, z, 0.0f, i);

    return z;
}

static inline void expf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

    V_ELT_FLOAT cephes_exp_p1_vec = VLOAD1_FLOAT(c_cephes_exp_p1, i);
    V_ELT_FLOAT Op5_vec = VLOAD1_FLOAT(0.5f, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        x = exp_ps(x, Op5_vec, cephes_exp_p1_vec, i);
        VSTORE_FLOAT(dst+l, x, i);
    }
}

static inline void coshf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

    V_ELT_FLOAT cephes_exp_p1_vec = VLOAD1_FLOAT(c_cephes_exp_p1, i);
    V_ELT_FLOAT Op5_vec = VLOAD1_FLOAT(0.5f, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT xx = VLOAD_FLOAT(src+l, i);

        V_ELT_FLOAT x, tmp;
        V_ELT_BOOL32 xsupmaxlogf;

        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(xx), inv_sign_mask, i));
        xsupmaxlogf = VGT1_FLOAT_BOOL(x, MAXLOGF, i);

        tmp = exp_ps(x, Op5_vec, cephes_exp_p1_vec, i);
        x = VRDIV1_FLOAT(tmp, 0.5f, i);  // or 1/(2*y)
        x = VFMACC1_FLOAT(x, 0.5f, tmp, i);
        x = VMERGE1_FLOAT(xsupmaxlogf, x, MAXNUMF, i);
        VSTORE_FLOAT(dst+l, x, i);
    }
}

static inline void sinhf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

    V_ELT_FLOAT SINH_P1_vec = VLOAD1_FLOAT(SINH_P1, i);
    V_ELT_FLOAT cephes_exp_p1_vec = VLOAD1_FLOAT(c_cephes_exp_p1, i);
    V_ELT_FLOAT Op5_vec = VLOAD1_FLOAT(0.5f, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT z, z_first_branch, z_second_branch, tmp;
        V_ELT_BOOL32 xsupmaxlogf, zsup1;
        V_ELT_INT sign;

        // x = xx; if x < 0, z = -x, else x
        z = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));
        sign = VAND1_INT(VINTERP_FLOAT_INT(x), sign_mask, i);

        xsupmaxlogf = VGT1_FLOAT_BOOL(z, MAXLOGF, i);

        // First branch
        zsup1 = VGT1_FLOAT_BOOL(z, 1.0f, i);
        tmp = exp_ps(z, Op5_vec, cephes_exp_p1_vec, i);
        z_first_branch = VRDIV1_FLOAT(tmp, -0.5f, i);
        z_first_branch = VFMACC1_FLOAT(z_first_branch, 0.5f, tmp, i);

        V_ELT_BOOL32 xinf0 = VLT1_FLOAT_BOOL(x, 0.0f, i);
        V_ELT_FLOAT tmp2 = VINTERP_INT_FLOAT(VXOR1_INT(VINTERP_FLOAT_INT(z_first_branch), neg_sign_mask, i));
        z_first_branch = VMERGE_FLOAT(xinf0, z_first_branch, tmp2, i);

        // Second branch
        tmp = VMUL_FLOAT(x, x, i);
        z_second_branch = tmp;
        z_second_branch = VFMADD1_FLOAT(z_second_branch, SINH_P0, SINH_P1_vec, i);
        z_second_branch = VFMASQ1_FLOAT(z_second_branch, tmp, SINH_P2, i);
        z_second_branch = VMUL_FLOAT(z_second_branch, tmp, i);
        z_second_branch = VFMADD_FLOAT(z_second_branch, x, x, i);

        // Choose between first and second branch
        z = VMERGE_FLOAT(zsup1, z_second_branch, z_first_branch, i);

        // Set value to MAXNUMF if abs(x) > MAGLOGF
        // Set value to -MAXNUMF if abs(x) > MAGLOGF and x < 0
        tmp = VINTERP_INT_FLOAT(VXOR1_INT(sign, *(int32_t *) &MAXNUMF, i));
        z = VMERGE_FLOAT(xsupmaxlogf, z, tmp, i);

        VSTORE_FLOAT(dst+l, z, i);
    }
}

static inline void tanhf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

    V_ELT_FLOAT cephes_exp_p1_vec = VLOAD1_FLOAT(c_cephes_exp_p1, i);
    V_ELT_FLOAT Op5_vec = VLOAD1_FLOAT(0.5f, i);
    V_ELT_FLOAT TANH_P1_vec = VLOAD1_FLOAT(TANH_P1, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT xx = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT x, z, z_first_branch, z_second_branch, tmp;
        V_ELT_BOOL32 xxsup0, xsupmaxlogfdiv2, xsup0p625;
        xxsup0 = VGT1_FLOAT_BOOL(xx, 0.0f, i);
        xsupmaxlogfdiv2 = VGT1_FLOAT_BOOL(xx, MAXLOGFDIV2, i);

        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(xx), inv_sign_mask, i));

        xsup0p625 = VGE1_FLOAT_BOOL(x, 0.625f, i);
        tmp = VADD_FLOAT(x, x, i);
        tmp = exp_ps(tmp, Op5_vec, cephes_exp_p1_vec, i);
        x = VMERGE_FLOAT(xsup0p625, x, tmp, i);

        // z = 1.0 - 2.0 / (x + 1.0);
        z_first_branch = VADD1_FLOAT(x, 1.0f, i);
        z_first_branch = VRDIV1_FLOAT(z_first_branch, -2.0f, i);
        z_first_branch = VADD1_FLOAT(z_first_branch, 1.0f, i);
        tmp = VINTERP_INT_FLOAT(VXOR1_INT(VINTERP_FLOAT_INT(z_first_branch), neg_sign_mask, i));
        z_first_branch = VMERGE_FLOAT(xxsup0, tmp, z_first_branch, i);

        // to speed up the last merge
        xxsup0 = VAND_BOOL(xxsup0, xsupmaxlogfdiv2, i);

        // z = x * x;
        z = VMUL_FLOAT(x, x, i);
        z_second_branch = z;
        z_second_branch = VFMADD1_FLOAT(z_second_branch, TANH_P0, TANH_P1_vec, i);
        z_second_branch = VFMASQ1_FLOAT(z_second_branch, z, TANH_P2, i);
        z_second_branch = VFMASQ1_FLOAT(z_second_branch, z, TANH_P3, i);
        z_second_branch = VFMASQ1_FLOAT(z_second_branch, z, TANH_P4, i);
        z_second_branch = VMUL_FLOAT(z_second_branch, z, i);
        z_second_branch = VFMADD_FLOAT(z_second_branch, xx, xx, i);

        z = VMERGE_FLOAT(xsup0p625, z_second_branch, z_first_branch, i);
        // if (x > 0.5 * MAXLOGF), return (xx > 0)? 1.0f: -1.0f
        z = VMERGE1_FLOAT(xsupmaxlogfdiv2, z, -1.0f, i);
        z = VMERGE1_FLOAT(xxsup0, z, 1.0f, i);  // xxsup0.xsupmaxlogfdiv2 has already been done
        VSTORE_FLOAT(dst+l, z, i);
    }
}

#if 1 //generated using chat gpt, should be tested on real HW for performance review
static inline void vectorSlopef_vec(float *dst, int len, float offset, float slope)
{
    int index_base = 0; // current index offset
    while (index_base < len) {
        // Predicate for active lanes
        svbool_t pg = svwhilelt_b32(index_base, len);
        // lane indices: [0,1,2,...]
        svuint32_t idx = svindex_u32(0, 1);
        // absolute indices = index_base + idx
        svuint32_t abs_idx = svadd_u32_z(pg, svdup_u32(index_base), idx);
        // convert to float
        svfloat32_t fidx = svcvt_f32_u32_z(pg, abs_idx);
        // compute: offset + slope * index
        svfloat32_t slope_vec = svdup_f32(slope);
        svfloat32_t offs_vec  = svdup_f32(offset);
        svfloat32_t val = svmad_f32_x(pg, fidx, slope_vec, offs_vec);
        // store
        svst1(pg, dst + index_base, val);
        // advance by VL
        index_base += svcntw();
    }
}

#else
static inline void vectorSlopef_vec(float *dst, int len, float offset, float slope)
{
    float __attribute__((aligned(128))) coef_max[MAX_ELTS32];

	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);
	
    for (int s = 0; s < numVals; s++) {
        coef_max[s] = (float) (s) *slope;
    }

    V_ELT_FLOAT coef = VLOAD_FLOAT(coef_max, i);
    V_ELT_FLOAT slope_vec = VLOAD1_FLOAT((float) (numVals) *slope, i);
    V_ELT_FLOAT curVal = VADD1_FLOAT(coef, offset, i);
    VSTORE_FLOAT(dst, curVal, i);
	for (size_t l=numVals; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        curVal = VADD_FLOAT(curVal, slope_vec, i);
        VSTORE_FLOAT(dst+l, curVal, i);
    }
}
#endif

static inline void setf_vec(float *dst, float value, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        VSTORE_FLOAT(dst+l, VLOAD1_FLOAT(value, i), i);
    }
}

static inline void zerof_vec(float *dst, int len)
{
    setf_vec(dst, 0.0f, len);
}

static inline void copyf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        VSTORE_FLOAT(dst+l, VLOAD_FLOAT(src+l, i), i);
    }
}

static inline void modf_vec(float *src, float *integer, float *remainder, int len)
{
	size_t n = (size_t)(len);
	V_ELT_FLOAT dummy;
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	uint64_t numVals = svlen_f32(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);	
        V_ELT_FLOAT src_vec, integer_vec, remainer_vec;
        src_vec = VLOAD_FLOAT(src+l, i);
        integer_vec = VRTZ_FLOAT(src_vec, i);	
        VSTORE_FLOAT(integer+l, integer_vec, i);
        remainer_vec = VSUB_FLOAT(src_vec, integer_vec, i);
        VSTORE_FLOAT(remainder+l, remainer_vec, i);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
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
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT a, b;
        a = VLOAD_FLOAT(src+l, i);
        b = VRTZ_FLOAT(a, i);
        VSTORE_FLOAT(dst+l, b, i);
    }
}

//generated using chat gpt, should be tested on real HW for performance review
static inline void flipf_vec(const float *src, float *dst, int len)
{
    int remaining = len;
    int dst_off   = 0;

    while (remaining > 0) {
        // Predicate for active lanes
        svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)remaining);

        // Number of active lanes this iteration
        uint32_t n = svcntp_b32(svptrue_b32(), pg);

        // Load 'n' floats from the tail of src
        const float *src_chunk = src + (remaining - n);
        svfloat32_t a = svld1(pg, src_chunk);

        // Build reversed *byte* offsets: (n-1-i) * sizeof(float)
        svuint32_t idx   = svindex_u32(0, 1);
        svuint32_t n_1   = svdup_u32(n - 1);
        svuint32_t ridx  = svsub_u32_z(pg, n_1, idx);      // (n-1-i)
        svuint32_t offs  = svlsl_n_u32_z(pg, ridx, 2);     // *4 bytes

        // Scatter-store reversed into dst
        svst1_scatter_offset(pg, dst + dst_off, offs, a);

        dst_off   += n;
        remaining -= n;
    }
}

// Work in progress
static inline V_ELT_FLOAT power_of_twof(V_ELT_INT b, V_ELT_BOOL32 i)
{
    V_ELT_INT exp = VADD1_INT(b, 127, i);
    V_ELT_FLOAT f = VINTERP_INT_FLOAT(VSLL1_INT(exp, 23, i));
    return f;
}

static inline void cbrtf_vec(float *src, float *dst, int len)
{

	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
    V_ELT_FLOAT invCBRT2_vec = VLOAD1_FLOAT(cephes_invCBRT2, i);
    V_ELT_FLOAT invCBRT4_vec = VLOAD1_FLOAT(cephes_invCBRT4, i);
    V_ELT_FLOAT CBRTF_P1_vec = VLOAD1_FLOAT(CBRTF_P1, i);

    const float Op5 = 0.5f;

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT xx = VLOAD_FLOAT(src+l, i);

        V_ELT_INT sign;
        V_ELT_FLOAT x, z, e, rem;
        V_ELT_FLOAT tmp, tmp2;

        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(xx), inv_sign_mask, i));
        sign = VAND1_INT(VINTERP_FLOAT_INT(xx), sign_mask, i);

        z = x;
        /* extract power of 2, leaving
         * mantissa between 0.5 and 1
         */
        // x = frexpf(x, &e);
        // solve problem for zero
        V_ELT_INT emm0 = VSRA1_INT(VINTERP_FLOAT_INT(x), 23, i);
        x = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(x), c_inv_mant_mask, i));
        x = VINTERP_INT_FLOAT(VOR1_INT(VINTERP_FLOAT_INT(x), *(int32_t *) &Op5, i));
        emm0 = VSUB1_INT(emm0, 0x7e, i);
        e = VCVT_INT_FLOAT(emm0, i);

        /* Approximate cube root of number between .5 and 1,
         * peak relative error = 9.2e-6
         */
        tmp = x;
        tmp = VFMADD1_FLOAT(tmp, CBRTF_P0, CBRTF_P1_vec, i);
        tmp = VFMASQ1_FLOAT(tmp, x, CBRTF_P2, i);
        tmp = VFMASQ1_FLOAT(tmp, x, CBRTF_P3, i);
        x = VFMASQ1_FLOAT(x, tmp, CBRTF_P4, i);

        /* exponent divided by 3 */
        V_ELT_BOOL32 e_sign = VGE1_FLOAT_BOOL(e, 0.0f, i);
        e = VINTERP_INT_FLOAT(VAND1_INT(VINTERP_FLOAT_INT(e), inv_sign_mask, i));
        rem = e;
        e = VMUL1_FLOAT(e, 0.333333333333f, i);	
        V_ELT_INT e_int = VCVT_FLOAT_INT(e, i);	
        e = VCVT_INT_FLOAT(e_int, i);
        V_ELT_FLOAT e_tmp = VMUL1_FLOAT(e, 3.0f, i);
        rem = VSUB_FLOAT(rem, e_tmp, i);

        V_ELT_FLOAT mul1, mul2;
        V_ELT_FLOAT mul_cst1 = VMERGE1_FLOAT(e_sign, invCBRT2_vec, cephes_CBRT2, i);
        V_ELT_FLOAT mul_cst2 = VMERGE1_FLOAT(e_sign, invCBRT4_vec, cephes_CBRT4, i);
        mul1 = VMUL_FLOAT(x, mul_cst1, i);
        mul2 = VMUL_FLOAT(x, mul_cst2, i);

        V_ELT_INT remi = VCVT_FLOAT_INT(rem, i);  // rem integer
        V_ELT_BOOL32 rem1 = VEQ1_INT_BOOL(remi, 1, i);
        V_ELT_BOOL32 rem2 = VEQ1_INT_BOOL(remi, 2, i);

        x = VMERGE_FLOAT(rem1, x, mul1, i);  // rem==1
        x = VMERGE_FLOAT(rem2, x, mul2, i);  // rem==2

        /* multiply by power of 2 */
        //  x = ldexpf(x, e);
        // x= x* (1 >> e)
        V_ELT_FLOAT cst = power_of_twof(e_int, i);

        // blend sign of e
        tmp = VMUL_FLOAT(x, cst, i);
        tmp2 = VDIV_FLOAT(x, cst, i);
        x = VMERGE_FLOAT(e_sign, tmp2, tmp, i);

        /* Newton iteration */
        // x -= (x - (z / (x * x))) * 0.333333333333;
        tmp2 = VMUL_FLOAT(x, x, i);
        tmp2 = VDIV_FLOAT(z, tmp2, i);
        tmp2 = VSUB_FLOAT(x, tmp2, i);
        tmp2 = VMUL1_FLOAT(tmp2, 0.333333333333f, i);
        x = VSUB_FLOAT(x, tmp2, i);
        x = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(x), sign, i));

        VSTORE_FLOAT(dst+l, x, i);
    }
}

static inline void convertInt16ToFloat32_vec(int16_t *src, float *dst, int len, int scale_factor)
{
    float scale_fact_mult;
    if(scale_factor >= 0)
    	scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    else
    	scale_fact_mult = (float) (1 << -scale_factor);

    int i = 0;
    while (i < len) {
        V_ELT_BOOL16 pg = svwhilelt_b16_s32(i, len);
        V_ELT_SHORT v_i16 = VLOAD_SHORT(src + i,pg);
        V_ELT_INT v_i32_lo = VCVT_SHORT_INT_LOW(v_i16);
        V_ELT_INT v_i32_hi = VCVT_SHORT_INT_HIGH(v_i16);
        V_ELT_BOOL32 pg_lo = svwhilelt_b32_s32(i, len);
        V_ELT_BOOL32 pg_hi = svwhilelt_b32_s32(i + svcntw(), len);
        V_ELT_FLOAT v_f32_lo = VCVT_INT_FLOAT(v_i32_lo, pg_lo);
        V_ELT_FLOAT v_f32_hi = VCVT_INT_FLOAT(v_i32_hi, pg_hi);
		v_f32_lo = VMUL1_FLOAT(v_f32_lo, scale_fact_mult, pg_lo);
		v_f32_hi = VMUL1_FLOAT(v_f32_hi, scale_fact_mult, pg_lo);		
        VSTORE_FLOAT(dst + i, v_f32_lo,pg_lo);
        VSTORE_FLOAT(dst + i + svcntw(), v_f32_hi,pg_hi);
        i += svcntw() * 2; // we processed 2×VL16 elements
    }
}

//TODO : check whether float16 offers good enough precision
static inline void convertFloat32ToI16_vec(float *src, int16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int i = 0;
	
    float scale_fact_mult;
    if(scale_factor >= 0)
    	scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    else
    	scale_fact_mult = (float) (1 << -scale_factor);
	float16_t scale_fact_mult_16 = (float16_t)scale_fact_mult;
	uint32_t reg_ori = fegetround();
    if (rounding_mode == RndZero) {
		fesetround(FE_TOWARDZERO);
		while (i < len) {
			V_ELT_BOOL32 pg0 = svwhilelt_b32_s32(i, len);
			V_ELT_BOOL32 pg1 = svwhilelt_b32_s32(i+svcntw(), len);		
			V_ELT_FLOAT f0 = VLOAD_FLOAT(src + i, pg0);
			V_ELT_FLOAT f1 = VLOAD_FLOAT(src + i + svcntw(), pg1);
			svfloat16_t lo = svcvt_f16_f32_z(pg0, f0);
			svfloat16_t hi = svcvt_f16_f32_z(pg1, f1);
			svfloat16_t h = svuzp1_f16(lo, hi);
			V_ELT_BOOL16 pg16 = svwhilelt_b16(i, len);
			h = VMUL1_FLOAT16(h, scale_fact_mult_16, pg16);			
			V_ELT_SHORT s16 = VCVT_FLOAT16_SHORT(h, pg16);
			VSTORE_SHORT(dst + i, s16, pg16);
			i += svcnth(); // VL 16-bit elements processed
		}
    } else if (rounding_mode == RndFinancial) {
		while (i < len) {
			V_ELT_BOOL32 pg0 = svwhilelt_b32_s32(i, len);
			V_ELT_BOOL32 pg1 = svwhilelt_b32_s32(i+svcntw(), len);		
			V_ELT_FLOAT f0 = VLOAD_FLOAT(src + i, pg0);
			V_ELT_FLOAT f1 = VLOAD_FLOAT(src + i + svcntw(), pg1);
			svfloat16_t lo = svcvt_f16_f32_z(pg0, f0);
			svfloat16_t hi = svcvt_f16_f32_z(pg1, f1);
			svfloat16_t h = svuzp1_f16(lo, hi);
			V_ELT_BOOL16 pg16 = svwhilelt_b16(i, len);
			h = VMUL1_FLOAT16(h, scale_fact_mult_16, pg16);
			h  = svrinta_f16_z(pg16, h);
			V_ELT_SHORT s16 = VCVT_FLOAT16_SHORT(h, pg16);
			VSTORE_SHORT(dst + i, s16, pg16);
			i += svcnth(); // VL 16-bit elements processed
		}		
    } else {
		fesetround(FE_TONEAREST);
		while (i < len) {
			V_ELT_BOOL32 pg0 = svwhilelt_b32_s32(i, len);
			V_ELT_BOOL32 pg1 = svwhilelt_b32_s32(i+svcntw(), len);		
			V_ELT_FLOAT f0 = VLOAD_FLOAT(src + i, pg0);
			V_ELT_FLOAT f1 = VLOAD_FLOAT(src + i + svcntw(), pg1);
			svfloat16_t lo = svcvt_f16_f32_z(pg0, f0);
			svfloat16_t hi = svcvt_f16_f32_z(pg1, f1);
			svfloat16_t h = svuzp1_f16(lo, hi);
			V_ELT_BOOL16 pg16 = svwhilelt_b16(i, len);
			h = VMUL1_FLOAT16(h, scale_fact_mult_16, pg16);
			h  = svrintn_f16_z(pg16, h);
			V_ELT_SHORT s16 = VCVT_FLOAT16_SHORT(h, pg16);
			VSTORE_SHORT(dst + i, s16, pg16);
			i += svcnth(); // VL 16-bit elements processed
		}		
    }	
	fesetround(reg_ori);	
}

static inline void convertFloat32ToU16_vec(float *src, uint16_t *dst, int len, int rounding_mode, int scale_factor)
{
    int i = 0;
	
    float scale_fact_mult;
    if(scale_factor >= 0)
    	scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    else
    	scale_fact_mult = (float) (1 << -scale_factor);
	float16_t scale_fact_mult_16 = (float16_t)scale_fact_mult;
	uint32_t reg_ori = fegetround();
    if (rounding_mode == RndZero) {
		fesetround(FE_TOWARDZERO);
		while (i < len) {
			V_ELT_BOOL32 pg0 = svwhilelt_b32_s32(i, len);
			V_ELT_BOOL32 pg1 = svwhilelt_b32_s32(i+svcntw(), len);		
			V_ELT_FLOAT f0 = VLOAD_FLOAT(src + i, pg0);
			V_ELT_FLOAT f1 = VLOAD_FLOAT(src + i + svcntw(), pg1);
			svfloat16_t lo = svcvt_f16_f32_z(pg0, f0);
			svfloat16_t hi = svcvt_f16_f32_z(pg1, f1);
			svfloat16_t h = svuzp1_f16(lo, hi);
			V_ELT_BOOL16 pg16 = svwhilelt_b16(i, len);
			h = VMUL1_FLOAT16(h, scale_fact_mult_16, pg16);			
			V_ELT_USHORT u16 = VCVT_FLOAT16_USHORT(h, pg16);
			VSTORE_USHORT(dst + i, u16, pg16);
			i += svcnth(); // VL 16-bit elements processed
		}
    } else if (rounding_mode == RndFinancial) {
		while (i < len) {
			V_ELT_BOOL32 pg0 = svwhilelt_b32_s32(i, len);
			V_ELT_BOOL32 pg1 = svwhilelt_b32_s32(i+svcntw(), len);		
			V_ELT_FLOAT f0 = VLOAD_FLOAT(src + i, pg0);
			V_ELT_FLOAT f1 = VLOAD_FLOAT(src + i + svcntw(), pg1);
			svfloat16_t lo = svcvt_f16_f32_z(pg0, f0);
			svfloat16_t hi = svcvt_f16_f32_z(pg1, f1);
			svfloat16_t h = svuzp1_f16(lo, hi);
			V_ELT_BOOL16 pg16 = svwhilelt_b16(i, len);
			h = VMUL1_FLOAT16(h, scale_fact_mult_16, pg16);
			h  = svrinta_f16_z(pg16, h);
			V_ELT_USHORT u16 = VCVT_FLOAT16_USHORT(h, pg16);
			VSTORE_USHORT(dst + i, u16, pg16);
			i += svcnth(); // VL 16-bit elements processed
		}		
    } else {
		fesetround(FE_TONEAREST);
		while (i < len) {
			V_ELT_BOOL32 pg0 = svwhilelt_b32_s32(i, len);
			V_ELT_BOOL32 pg1 = svwhilelt_b32_s32(i+svcntw(), len);		
			V_ELT_FLOAT f0 = VLOAD_FLOAT(src + i, pg0);
			V_ELT_FLOAT f1 = VLOAD_FLOAT(src + i + svcntw(), pg1);
			svfloat16_t lo = svcvt_f16_f32_z(pg0, f0);
			svfloat16_t hi = svcvt_f16_f32_z(pg1, f1);
			svfloat16_t h = svuzp1_f16(lo, hi);
			V_ELT_BOOL16 pg16 = svwhilelt_b16(i, len);
			h = VMUL1_FLOAT16(h, scale_fact_mult_16, pg16);
			h  = svrintn_f16_z(pg16, h);
			V_ELT_USHORT u16 = VCVT_FLOAT16_USHORT(h, pg16);
			VSTORE_USHORT(dst + i, u16, pg16);
			i += svcnth(); // VL 16-bit elements processed
		}		
    }	
	fesetround(reg_ori);	
}

static inline void pol2cart2Df_vec(float *r, float *theta, float *x, float *y, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
	
    V_ELT_FLOAT coscof_1_vec = VLOAD1_FLOAT(coscof[1], i);
    V_ELT_FLOAT sincof_1_vec = VLOAD1_FLOAT(sincof[1], i);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT r_vec = VLOAD_FLOAT(r+l, i);
        V_ELT_FLOAT theta_vec = VLOAD_FLOAT(theta+l, i);
        V_ELT_FLOAT sin_vec, cos_vec;
        sincosf_ps(theta_vec, &sin_vec, &cos_vec,
                   coscof_1_vec, sincof_1_vec, i);
        V_ELT_FLOAT x_vec = VMUL_FLOAT(r_vec, cos_vec, i);
        V_ELT_FLOAT y_vec = VMUL_FLOAT(r_vec, sin_vec, i);
        VSTORE_FLOAT(x+l, x_vec, i);
        VSTORE_FLOAT(y+l, y_vec, i);
    }
}

static inline void cart2pol2Df_vec(float *x, float *y, float *r, float *theta, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
    V_ELT_FLOAT ATAN_P1_vec = VLOAD1_FLOAT(ATAN_P1, i);
    V_ELT_FLOAT min1_vec = VLOAD1_FLOAT(-1.0f, i);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT x_vec = VLOAD_FLOAT(x+l, i);
        V_ELT_FLOAT y_vec = VLOAD_FLOAT(y+l, i);
        V_ELT_FLOAT y_square = VMUL_FLOAT(y_vec, y_vec, i);
        V_ELT_FLOAT r_vec = x_vec;
        r_vec = VFMADD_FLOAT(r_vec, x_vec, y_square, i);
        r_vec = VSQRT_FLOAT(r_vec, i);
        V_ELT_FLOAT theta_vec = atan2f_ps(y_vec, x_vec,
                                           ATAN_P1_vec, min1_vec, i);
        VSTORE_FLOAT(r+l, r_vec, i);
        VSTORE_FLOAT(theta+l, theta_vec, i);
    }
}

static inline void PReluf_vec(float *src, float *dst, float alpha, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL32 i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT src_vec = VLOAD_FLOAT(src+l, i);
        V_ELT_FLOAT tmp = VMUL1_FLOAT(src_vec, alpha, i);
        V_ELT_BOOL32 mask = VGT1_FLOAT_BOOL(src_vec, 0.0f, i);
        V_ELT_FLOAT dst_vec = VMERGE_FLOAT(mask, tmp, src_vec, i);
        VSTORE_FLOAT(dst+l, dst_vec, i);
    }
}

static inline void sigmoidf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
    V_ELT_FLOAT cephes_exp_p1_vec = VLOAD1_FLOAT(c_cephes_exp_p1, i);
    V_ELT_FLOAT Op5_vec = VLOAD1_FLOAT(0.5f, i);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b32_s32(l, n);
        V_ELT_FLOAT x = VLOAD_FLOAT(src+l, i);
        x = VINTERP_INT_FLOAT(VXOR1_INT(VINTERP_FLOAT_INT(x), neg_sign_mask, i));
        x = exp_ps(x, Op5_vec, cephes_exp_p1_vec, i);
        x = VADD1_FLOAT(x, 1.0f, i);
        x = VRDIV1_FLOAT(x, 1.0f, i);  // 1/x
        VSTORE_FLOAT(dst+l, x, i);
    }
}

static inline void softmaxf_vec(float *src, float *dst, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	V_ELT_BOOL32 i = svwhilelt_b32_s32(0, n);
    V_ELT_FLOAT cephes_exp_p1_vec = VLOAD1_FLOAT(c_cephes_exp_p1, i);
    V_ELT_FLOAT Op5_vec = VLOAD1_FLOAT(0.5f, i);

    V_ELT_FLOAT vacc = VLOAD1_FLOAT(0.0f, i);
    float acc = 0.0f;

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);		
        V_ELT_FLOAT va = VLOAD_FLOAT(src+l, i);
        va = exp_ps(va, Op5_vec, cephes_exp_p1_vec, i);
        vacc = VADD_FLOAT(vacc, va, i);
        VSTORE_FLOAT(dst+l, va, i);
    }
    acc = VREDSUM_FLOAT(vacc, i);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);			
        V_ELT_FLOAT dst_vec = VLOAD_FLOAT(dst+l, i);
        dst_vec = VDIV1_FLOAT(dst_vec, acc, i);
        VSTORE_FLOAT(dst+l, dst_vec, i);
    }
}

static inline void convert_32f64f_vec(float *src, double *dst, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
	V_ELT_BOOL32 i;
	
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b32_s32(l, n);
		unsigned long active_f = svcntp_b32(svptrue_b32(), i);
        unsigned long even_cnt = (active_f + 1u) / 2u;
        unsigned long odd_cnt  = active_f / 2u;
        svbool_t pg64_even = svwhilelt_b64((size_t)0, even_cnt);
        svbool_t pg64_odd  = svwhilelt_b64((size_t)0, odd_cnt);
		V_ELT_FLOAT src_vec = VLOAD_FLOAT(src+l, i);
        svfloat64_t even = svcvt_f64_f32_x(pg64_even, src_vec);
        svfloat64_t odd = svcvtlt_f64_f32_x(pg64_odd, src_vec);
        svfloat64_t lo = svzip1_f64(even, odd);
        svfloat64_t hi = svzip2_f64(even, odd);		
        svst1(pg64_even, &dst[l], lo);
		svst1(pg64_odd, &dst[l + svcntd()], hi);
    }
}

static inline void convert_64f32f_vec(double *src, float *dst, int len)
{
	size_t n = (size_t)len;	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	V_ELT_FLOAT dummyf;
	uint64_t numVals = svlen_f64(dummy);
	V_ELT_BOOL64 ilo,ihi;
	V_ELT_BOOL32 i32;
	svuint32_t idx = svindex_u32(0,1); // indices 0,1,2,3 for contiguous packing
	
	for (size_t l=0; l<n; l+=(2*numVals)) {
		ilo = svwhilelt_b64_s64(l, n);	
		ihi = svwhilelt_b64_s64(l+numVals, n);			
		i32 = svwhilelt_b32_s32(l, n);
		V_ELT_DOUBLE src_tmp_lo = VLOAD_DOUBLE(src+l, ilo);
		V_ELT_DOUBLE src_tmp_hi = VLOAD_DOUBLE(src+l+numVals, ihi);			
		V_ELT_FLOAT dstlo = svcvt_f32_f64_x(ilo,src_tmp_lo);		
		V_ELT_FLOAT dsthi = svcvt_f32_f64_x(ihi,src_tmp_hi);
		V_ELT_FLOAT dst_tmp = svuzp1_f32(dstlo,dsthi);
        VSTORE_FLOAT(dst+l, dst_tmp, i32);
    }
}