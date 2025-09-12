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


static inline void rintd_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE a, b;
        a = VLOAD_DOUBLE(src+l, i);
        b = VRNE_DOUBLE(a, i);
        VSTORE_DOUBLE(dst+l, b, i);
    }
}

static inline void roundd_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE a, b;
        a = VLOAD_DOUBLE(src+l, i);
        b = VRNA_DOUBLE(a, i);
        VSTORE_DOUBLE(dst+l, b, i);
    }
}

static inline void ceild_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE a, b;
        a = VLOAD_DOUBLE(src+l, i);
        b = VRINF_DOUBLE(a, i);
        VSTORE_DOUBLE(dst+l, b, i);
    }
}

static inline void floord_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE a, b;
        a = VLOAD_DOUBLE(src+l, i);
        b = VRMINF_DOUBLE(a, i);
        VSTORE_DOUBLE(dst+l, b, i);
    }
}

static inline void truncd_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE a, b;
        a = VLOAD_DOUBLE(src+l, i);
        b = VRTZ_DOUBLE(a, i);
        VSTORE_DOUBLE(dst+l, b, i);
    }
}

void addd_vec(double *a, double *b, double *c, int len)
{
    V_ELT_DOUBLE va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        va = VLOAD_DOUBLE(a+l, i);
        vb = VLOAD_DOUBLE(b+l, i);
        vc = VADD_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c+l, vc, i);
    }
}

static inline void addcd_vec(double *src, double value, double *dst, int len)
{
    V_ELT_DOUBLE va, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        va = VLOAD_DOUBLE(src+l, i);
        vc = VADD1_DOUBLE(va, value, i);
        VSTORE_DOUBLE(dst+l, vc, i);
    }
}

static inline void muld_vec(double *a, double *b, double *c, int len)
{
    V_ELT_DOUBLE va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        va = VLOAD_DOUBLE(a+l, i);
        vb = VLOAD_DOUBLE(b+l, i);
        vc = VMUL_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c+l, vc, i);
    }
}

static inline void divd_vec(double *a, double *b, double *c, int len)
{
    V_ELT_DOUBLE va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        va = VLOAD_DOUBLE(a+l, i);
        vb = VLOAD_DOUBLE(b+l, i);
        vc = VDIV_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c+l, vc, i);
    }
}

static inline void subd_vec(double *a, double *b, double *c, int len)
{
    V_ELT_DOUBLE va, vb, vc;
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        va = VLOAD_DOUBLE(a+l, i);
        vb = VLOAD_DOUBLE(b+l, i);
        vc = VSUB_DOUBLE(va, vb, i);
        VSTORE_DOUBLE(c+l, vc, i);
    }
}

static inline void subcrevd_vec(double *src, double value, double *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE va;
        va = VLOAD_DOUBLE(src + l, i);
        VSTORE_DOUBLE(dst + l, VRSUB1_DOUBLE(va, value, i), i);
    }
}

static inline void muladdd_vec(double *a, double *b, double *c, double *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE va, vb, vc;
        va = VLOAD_DOUBLE(a+l, i);
        vb = VLOAD_DOUBLE(b+l, i);
        vc = VLOAD_DOUBLE(c+l, i);
        vc = VFMACC_DOUBLE(vc, va, vb, i);
        VSTORE_DOUBLE(dst+l, vc, i);
    }
}

static inline void mulcaddd_vec(double *a, double b, double *c, double *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE va, vc;
        va = VLOAD_DOUBLE(a+l, i);
        vc = VLOAD_DOUBLE(c+l, i);
        vc = VFMACC1_DOUBLE(vc, b, va, i);
        VSTORE_DOUBLE(dst+l, vc, i);
    }
}

static inline void mulcaddcd_vec(double *a, double b, double c, double *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE va, vc;
        va = VLOAD_DOUBLE(a+l, i);
        vc = VLOAD1_DOUBLE(c, i);
        vc = VFMACC1_DOUBLE(vc, b, va, i);
        VSTORE_DOUBLE(dst+l, vc, i);
    }
}

static inline void muladdcd_vec(double *a, double *b, double c, double *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE va, vb, vc;
        va = VLOAD_DOUBLE(a+l, i);
        vb = VLOAD_DOUBLE(b+l, i);
        vc = VLOAD1_DOUBLE(c, i);
        vc = VFMACC_DOUBLE(vc, va, vb, i);
        VSTORE_DOUBLE(dst+l, vc, i);
    }
}

static inline void mulcd_vec(double *src, double value, double *dst, int len)
{
	size_t n = (size_t)len;
	
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=n; l<n; l+=numVals) {
		// set predicate 
		V_ELT_BOOL64 i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE va;
        va = VLOAD_DOUBLE(src+l, i);
        VSTORE_DOUBLE(dst+l, VMUL1_DOUBLE(va, value, i), i);
    }
}

static inline void setd_vec(double *dst, double value, int len)
{
	size_t n = (size_t)(len);
	V_ELT_DOUBLE dummy;
	V_ELT_BOOL64 i = svwhilelt_b64_s64(0, n);
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b64_s64(l, n);	
        VSTORE_DOUBLE(dst+l, VLOAD1_DOUBLE(value, i), i);
    }
}

static inline void zerod_vec(double *dst, int len)
{
    setd_vec(dst, 0.0, len);
}

static inline void copyd_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_DOUBLE dummy;
	V_ELT_BOOL64 i = svwhilelt_b64_s64(0, n);
	uint64_t numVals = svlen_f64(dummy);

	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b64_s64(l, n);	
        VSTORE_DOUBLE(dst+l, VLOAD_DOUBLE(src+l, i), i);
    }
}

static inline void sqrtd_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_DOUBLE dummy;
	V_ELT_BOOL64 i;	
	uint64_t numVals = svlen_f64(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b64_s64(l, n);	
        V_ELT_DOUBLE va = VLOAD_DOUBLE(src+l, i);
        VSTORE_DOUBLE(dst+l, VSQRT_DOUBLE(va, i), i);
    }
}

static inline void fabsd_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)(len);
	V_ELT_DOUBLE dummy;
	V_ELT_BOOL64 i;	
	uint64_t numVals = svlen_f64(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b64_s64(l, n);	
        V_ELT_DOUBLE va = VLOAD_DOUBLE(src+l, i);
        VSTORE_DOUBLE(dst+l, VABS_DOUBLE(va, i), i);
    }
}

static inline void sumd_vec(double *src, double *dst, int len)
{
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
    V_ELT_DOUBLE vacc = VLOAD1_DOUBLE(0.0, n);
	V_ELT_BOOL64 i;
	uint64_t numVals = svlen_f64(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE va = VLOAD_DOUBLE(src+l, i);
        vacc = VADD_DOUBLE(vacc, va, i);
    }
#if 0 //ordered sum
	double tmp = 0.0;
    *dst = VREDSUMORD_DOUBLE(tmp, vacc, i);
#else	
    *dst = VREDSUM_DOUBLE(vacc, i);
#endif
}

static inline void meand_vec(double *src, double *dst, int len)
{
    double coeff = 1.0 / ((double) len);
    sumd_vec(src, dst, len);
    *dst *= coeff;
}

static inline void vectorSloped_vec(double *dst, int len, double offset, double slope)
{
    int index_base = 0; // current index offset
    while (index_base < len) {
        // Predicate for active lanes
        svbool_t pg = svwhilelt_b64(index_base, len);
        // lane indices: [0,1,2,...]
        svuint64_t idx = svindex_u64(0, 1);
        // absolute indices = index_base + idx
        svuint64_t abs_idx = svadd_u64_z(pg, svdup_u64(index_base), idx);
        // convert to double
        V_ELT_DOUBLE fidx = svcvt_f64_u64_z(pg, abs_idx);
        // compute: offset + slope * index
        V_ELT_DOUBLE slope_vec = svdup_f64(slope);
        V_ELT_DOUBLE offs_vec  = svdup_f64(offset);
        V_ELT_DOUBLE val = svmad_f64_x(pg, fidx, slope_vec, offs_vec);
        // store
        svst1(pg, dst + index_base, val);
        // advance by VL
        index_base += svcntd();
    }
}

static inline void cplxtoreald_vec(complex64_t *src, double *dstRe, double *dstIm, int len)
{
    size_t n = (size_t)(len);
	V_ELT_DOUBLE dummy;
	V_ELT_BOOL64 i;	
	uint64_t numVals = svlen_f64(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE2 dst_vec = VLOAD_DOUBLE2(src+l, i);
        V_ELT_DOUBLE dstRe_vec = svget2_f64(dst_vec,0); 
        V_ELT_DOUBLE dstIm_vec = svget2_f64(dst_vec,1);		
        VSTORE_DOUBLE(dstRe+l, dstRe_vec, i);
        VSTORE_DOUBLE(dstIm+l, dstIm_vec, i);
    }
}

static inline void realtocplxd_vec(double *srcRe, double *srcIm, complex64_t *dst, int len)
{
    size_t n = (size_t)(len);
	V_ELT_DOUBLE dummy;
	V_ELT_BOOL64 i;		
	uint64_t numVals = svlen_f64(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		// set predicate 
		i = svwhilelt_b64_s64(l, n);
        V_ELT_DOUBLE srcRe_vec = VLOAD_DOUBLE(srcRe+l, i);
        V_ELT_DOUBLE srcIm_vec = VLOAD_DOUBLE(srcIm+l, i);
        VSTORE_DOUBLE2SPLIT(dst+l, srcRe_vec, srcIm_vec, i);
    }
}

static inline void sincos_pd(V_ELT_DOUBLE x,
                              V_ELT_DOUBLE *sin_tmp,
                              V_ELT_DOUBLE *cos_tmp,
                              V_ELT_DOUBLE coscof_1_vec,
                              V_ELT_DOUBLE sincof_1_vec,
							  V_ELT_INT64 four,
                              V_ELT_BOOL64 i)
{
    V_ELT_DOUBLE y;
    V_ELT_INT64  emm0,emm2,emm4;
	V_ELT_INT64 sign_bit_sin, swap_sign_bit_sin, sign_bit_cos;

	sign_bit_sin = VINTERP_DOUBLE_INT64(x);

    /* take the absolute value */
    x = VINTERP_INT64_DOUBLE(VAND1_INT64(VINTERP_DOUBLE_INT64(x), inv_sign_maskd, i));
	
    /* extract the sign bit (upper one) */
    sign_bit_sin = VAND1_INT64(sign_bit_sin, sign_maskd, i);

    /* scale by 4/Pi */
    y = VMUL1_DOUBLE(x, FOPId, i);
    y = VRMINF_DOUBLE(y, i);

    V_ELT_DOUBLE z;
	
	int64_t one = (int64_t)1;
	
    /* store the integer part of y in mm2 */
    emm2 = VCVT_DOUBLE_INT64(y, i);
    /* j=(j+1) & (~1) (see the cephes sources) */	
    emm2 = VADD1_INT64(emm2, one, i);	
    emm2 = VAND1_INT64(emm2, ~one, i);	
	y = VCVT_INT64_DOUBLE(emm2,i);
	emm4 = emm2;
	
    /* get the swap sign flag for the sine */
    emm0 = VAND1_INT64(emm2, (int64_t)4, i);	
    emm0 = VSLL1_INT64(emm0, (int64_t)61, i);
	
    swap_sign_bit_sin = emm0;
	
    /* get the polynom selection mask for the sine*/
    emm2 = VAND1_INT64(emm2, (int64_t)2, i);	
    V_ELT_BOOL64 poly_mask = VEQ1_INT64_BOOL(emm2, (int64_t)0, i);
	 
    /* The magic pass: "Extended precision modular arithmetic"
    x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = VFMACC1_DOUBLE(x, minus_cephes_DP1, y, i);
    x = VFMACC1_DOUBLE(x, minus_cephes_DP2, y, i);
    x = VFMACC1_DOUBLE(x, minus_cephes_DP3, y, i);
	
    emm4 = VSUB1_INT64(emm4, (int64_t)2, i);

	#warning "To be improved!"
    emm4 = VANDNOT_INT64(emm4, four, i);
    emm4 = VSLL1_INT64(emm4,(int64_t)61, i);
	
    sign_bit_cos = emm4;
    sign_bit_sin = VXOR_INT64(sign_bit_sin, swap_sign_bit_sin, i);
	
    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    z = VMUL_DOUBLE(x, x, i);
    y = z;
    y = VFMADD1_DOUBLE(y, coscod[0], coscof_1_vec, i);
    y = VFMASQ1_DOUBLE(y, z, coscod[2], i);
    y = VFMASQ1_DOUBLE(y, z, coscod[3], i);
    y = VFMASQ1_DOUBLE(y, z, coscod[4], i);
    y = VFMASQ1_DOUBLE(y, z, coscod[5], i);	
    y = VMUL_DOUBLE(y, z, i);
    y = VMUL_DOUBLE(y, z, i);
    y = VFMACC1_DOUBLE(y, -0.5, z, i);  // y = y -0.5*z
    y = VADD1_DOUBLE(y, 1.0, i);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    V_ELT_DOUBLE y2;
    y2 = z;
    y2 = VFMADD1_DOUBLE(y2, sincod[0], sincof_1_vec, i);
    y2 = VFMASQ1_DOUBLE(y2, z, sincod[2], i);
    y2 = VFMASQ1_DOUBLE(y2, z, sincod[3], i);
    y2 = VFMASQ1_DOUBLE(y2, z, sincod[4], i);
    y2 = VFMASQ1_DOUBLE(y2, z, sincod[5], i);	
    y2 = VMUL_DOUBLE(y2, z, i);
    y2 = VFMADD_DOUBLE(y2, x, x, i);

    /* select the correct result from the two polynoms */
    V_ELT_DOUBLE y_sin = VMERGE_DOUBLE(poly_mask, y, y2, i);
    V_ELT_DOUBLE y_cos = VMERGE_DOUBLE(poly_mask, y2, y, i);
    y_sin = VINTERP_INT64_DOUBLE(VXOR_INT64(VINTERP_DOUBLE_INT64(y_sin), sign_bit_sin, i));
    y_cos = VINTERP_INT64_DOUBLE(VXOR_INT64(VINTERP_DOUBLE_INT64(y_cos), sign_bit_cos, i));

    *sin_tmp = y_sin;
    *cos_tmp = y_cos;
}	

static inline void sincosd_vec(double *src, double *s, double *c, int len)
{
    V_ELT_BOOL64 i;
    V_ELT_DOUBLE coscof_1_vec = VLOAD1_DOUBLE(coscod[1], i);
    V_ELT_DOUBLE sincof_1_vec = VLOAD1_DOUBLE(sincod[1], i);
	V_ELT_INT64 four = VLOAD1_INT64(4,i);
	uint32_t reg_ori = fegetround();
	fesetround(FE_TONEAREST);
		
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b64_s64(l, n);		
        V_ELT_DOUBLE x = VLOAD_DOUBLE(src+l, i);
        V_ELT_DOUBLE y_sin, y_cos;
        sincos_pd(x, &y_sin, &y_cos,
                   coscof_1_vec, sincof_1_vec, four, i);
        VSTORE_DOUBLE(s+l, y_sin, i);
        VSTORE_DOUBLE(c+l, y_cos, i);
    }
	fesetround(reg_ori);	
}

static inline void sincosd_interleaved_vec(double *src, complex64_t *dst, int len)
{
    V_ELT_BOOL64 i;
    V_ELT_DOUBLE coscof_1_vec = VLOAD1_DOUBLE(coscod[1], i);
    V_ELT_DOUBLE sincof_1_vec = VLOAD1_DOUBLE(sincod[1], i);
	V_ELT_INT64 four = VLOAD1_INT64(4,i);
	
	size_t n = (size_t)len;
	// get the vector length being used, so we know how to increment the loop (1)
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);
	for (size_t l=0; l<n; l+=numVals) {
		i = svwhilelt_b64_s64(l, n);		
        V_ELT_DOUBLE x = VLOAD_DOUBLE(src+l, i);
        V_ELT_DOUBLE y_sin, y_cos;
        sincos_pd(x, &y_sin, &y_cos,
                   coscof_1_vec, sincof_1_vec, four, i);
        VSTORE_DOUBLE2SPLIT(dst+l, y_cos, y_sin, i);
    }
}