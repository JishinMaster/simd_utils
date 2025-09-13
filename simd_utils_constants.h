/*
 * Project : SIMD_Utils
 * Version : 0.2.6
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#if defined(SSE) || defined(AVX) || defined(AVX512)
#ifndef ARM
#include <immintrin.h>
#else

#if !defined(__aarch64__)
#define SSE2NEON_PRECISE_SQRT 1
#define SSE2NEON_PRECISE_DIV 1
#endif

// Also includes arm_neon.h
#include "sse2neon_wrapper.h"
#endif
#endif

#ifdef SVE2
#include <arm_sve.h>

#define MAX_ELTS8 1024
#define MAX_ELTS32 256
#define MAX_ELTS64 128

#define V_ELT_FLOAT svfloat32_t
#define VLOAD_FLOAT(a,b) svld1_f32((b),(a))
#define VLOAD_INT(a,b) svld1_s32((b),(a))
#define VLOAD_UINT(a,b) svld1_u32((b),(a))
#define VLOAD1_FLOAT(a,b) svdup_n_f32((a))
#define VSTORE_FLOAT(a,b,c) svst1_f32((c),(a),(b))
#define VSTORE_INT(a,b,c) svst1_s32((c),(a),(b))
#define V_ELT_INT svint32_t
#define V_ELT_UINT svuint32_t
#define VINTERP_INT_FLOAT svreinterpret_f32_s32
#define VINTERP_FLOAT_INT svreinterpret_s32_f32
#define VAND1_INT(a,b,c) svand_n_u32_z((c),(a),(b))
#define VMUL1_FLOAT(a,b,c) svmul_n_f32_z((c),(a),(b))
#define VDIV1_FLOAT(a,b,c) svdiv_n_f32_z((c),(a),(b))
#define VMUL1_FLOAT_MASK(a,b,c,d) svmul_n_f32_m((a),(b),(c))
#define VMUL_FLOAT(a,b,c) svmul_f32_z((c),(a),(b))
#define VDIV_FLOAT(a,b,c) svdiv_f32_z((c),(a),(b))
#define VADD1_FLOAT(a,b,c) svadd_n_f32_z((c),(a),(b))
#define VADD1_FLOAT_MASK(a,b,c,d) svadd_n_f32_m((a),(b),(c))
#define VADD1_INT(a,b,c) svadd_n_s32_z((c),(a),(b))
#define VADD1_INT_MASK(a,b,c,d) svadd_n_s32_m((a),(b),(c))
#define VSUB1_INT_MASK(a,b,c,d) svsub_n_s32_m((a),(b),(c))
#define VADD_FLOAT(a,b,c) svadd_f32_z((c),(a),(b))
#define VSUB_FLOAT(a,b,c) svsub_f32_z((c),(a),(b))
#define VSUB1_FLOAT(a,b,c) svsub_n_f32_z((c),(a),(b))
#define VSUB1_INT(a,b,c) svsub_n_s32_z((c),(a),(b))
#define VSLL1_INT(a,b,c) svlsl_n_s32_z((c),(a),(b))
#define VCVT_INT_FLOAT(a,b) svcvt_f32_s32_z((b),(a))
#define VCVT_FLOAT_INT(a,b) svcvt_s32_f32_z((b),(a))
#define VCVT_RTZ_FLOAT_INT VCVT_FLOAT_INT
// x current exact, a nearest away from zero, i current mode inexact, m mininf,
// n nearest even, p plusinf
#define VR_FLOAT(a,b) svrintx_f32_z((b),(a))
#define VRTZ_FLOAT(a,b) svrintz_f32_z((b),(a))
#define VRNE_FLOAT(a,b) svrintn_f32_z((b),(a))
#define VRMINF_FLOAT(a,b) svrintm_f32_z((b),(a))
#define VRINF_FLOAT(a,b) svrintp_f32_z((b),(a))
#define VRNA_FLOAT(a,b) svrinta_f32_z((b),(a))

#define V_ELT_BOOL32 svbool_t
#define VEQ1_INT_BOOL(a,b,c) svcmpeq_n_s32((c),(a),(b))
#define VGT1_FLOAT_BOOL(a,b,c) svcmpgt_n_f32((c),(a),(b))
#define VGE1_FLOAT_BOOL(a,b,c) svcmpge_n_f32((c),(a),(b))
#define VLT1_FLOAT_BOOL(a,b,c) svcmplt_n_f32((c),(a),(b))
#define VLE1_FLOAT_BOOL(a,b,c) svcmple_n_f32((c),(a),(b))
#define VEQ1_FLOAT_BOOL(a,b,c) svcmpeq_n_f32((c),(a),(b))
//#define VFMACC1_FLOAT(a,b,c,d) svmad_n_f32_m((d),(b),(c),(a)) // op3 + op1[i]*op2[i]
#define VFMASQ1_FLOAT(a,b,c,d) svmad_n_f32_x((d),(a),(b),(c)) 
#define VFMACC1_FLOAT(a,b,c,d) svmla_n_f32_z((d),(a),(c),(b)) // op1[i] + op2[i] * op3
#define VFMACC_FLOAT(a,b,c,d) svmla_f32_z((d),(a),(c),(b)) // op1[i] + op2[i] * op3[i]
#define VFMADD1_FLOAT(a,b,c,d) svmla_n_f32_z((d),(c),(a),(b)) //  op1[i] + op2[i] * op3

//SVE2 computes a - b*c where riscv does b*c - a! 
#define VFMSUB_FLOAT(a,b,c,d) svmls_f32_z((d),(c),(a),(b)) //  op1[i] - op2[i] * op3[i]

#define VFMADD_FLOAT(a,b,c,d) svmad_f32_z((d),(a),(b),(c)) // op1[i] * op2[i] + op3[i]
#define VMERGE_FLOAT(mask, t, f, i) svsel_f32((mask), (f), (t))     /* select */
#define VMERGE1_FLOAT(mask, v, f, i) svsel_f32((mask), svdup_n_f32((float)(f)),(v))
#define VXOR_INT(a,b,c)  sveor_s32_m((c),(a),(b))
#define VOR_INT(a,b,c)  svorr_s32_m((c),(a),(b))
#define VOR1_INT(a,b,c)  svorr_n_s32_z((c),(a),(b))
#define VNOT_INT(a,b)  svnot_s32_z((b),(a))
#define VXOR1_INT(a,b,c)  sveor_n_s32_z((c),(a),(b))
#define VCLEAR_BOOL(a) svpfalse_b()

#define VXOR_BOOL(a,b,c)  sveor_b_z((c),(a),(b))
#define VOR_BOOL(a,b,c)  svorr_b_z((c),(a),(b))
#define VAND_BOOL(a,b,c)  svand_b_z((c),(a),(b))
#define VOR_BOOL(a,b,c)  svorr_b_z((c),(a),(b))
#define VNOT_BOOL(a,b)  svnot_b_z((b),(a))
#define VGT1_INT_BOOL(a,b,c) svcmpgt_n_s32((c),(a),(b))
#define VNE1_INT_BOOL(a,b,c) svcmpne_n_s32((c),(a),(b))

#define VRDIV1_FLOAT(a,b,c) svdivr_n_f32_z((c),(a),(b))
#define VRSUB1_FLOAT(a,b,c) svsubr_n_f32_z((c),(a),(b))

#define V_ELT_FLOAT2 svfloat32x2_t
#define VSTORE_FLOAT2SPLIT(a,b,c,d) svst2_f32((d),(float*)(a),(V_ELT_FLOAT2){b,c})
#define VSTORE_FLOAT2(a,b,c) svst2_f32((c),(float*)(a),(b))
#define VLOAD_FLOAT2(a,b) svld2_f32((b),(float*)(a))

#define VREDSUM_FLOAT(a,b)  svaddv_f32((b),(a))
#define VREDSUMORD_FLOAT(a,b,c)  svadda_f32((c),(a),(b))
#define VSQRT_FLOAT(a,b) svsqrt_f32_z((b),(a))
#define VMAX_FLOAT(a,b,c) svmax_f32_z((c),(a),(b))
#define VMIN_FLOAT(a,b,c) svmin_f32_z((c),(a),(b))
#define VMAX1_FLOAT(a,b,c) svmax_n_f32_z((c),(a),(b))
#define VMIN1_FLOAT(a,b,c) svmin_n_f32_z((c),(a),(b))
#define VREDMAX_FLOAT(a,b) svmaxv_f32((b),(a))
#define VREDMIN_FLOAT(a,b) svminv_f32((b),(a))
#define VABS_FLOAT(a,b) svabs_f32_z((b),(a))
#define VSRA1_INT(a,b,c) svasr_n_s32_z((c),(a),(b))

#define VGATHER_FLOAT(a,b,c) svld1_gather_s32offset_f32((c),(a),(b))

//converts only to even index! svcvtlt for odd
// needs zip1 zip2 to interleave properly
#define VCVT_DOUBLE_FLOAT(a,b) svcvt_f32_f64_z((b),(a))
#define VCVT_FLOAT_DOUBLE(a,b) svcvt_f64_f32_z((b),(a))
// returns dst = dst  + a*b
static inline V_ELT_FLOAT VMUL_CFLOAT(V_ELT_FLOAT dst, V_ELT_FLOAT a, V_ELT_FLOAT b, V_ELT_BOOL32 i){
	dst = svcmla_f32_z(i,dst,a,b,0);
	dst = svcmla_f32_z(i,dst,a,b,90);
	return dst;
}

//returns dst = dst + conj(a)*b
static inline V_ELT_FLOAT VMULCONJA_CFLOAT(V_ELT_FLOAT dst, V_ELT_FLOAT a, V_ELT_FLOAT b, V_ELT_BOOL32 i){
	dst = svcmla_f32_z(i,dst,a,b,0);
	dst = svcmla_f32_z(i,dst,a,b,270);
	return dst;
}

/// 16 bits
#define V_ELT_FLOAT16 svfloat16_t
#define V_ELT_BOOL16 svbool_t
#define V_ELT_SHORT svint16_t
#define V_ELT_USHORT svuint16_t
#define VCVT_SHORT_FLOAT16(a,b) svcvt_f16_s16_z((b),(a))
#define VCVT_USHORT_FLOAT16(a,b) svcvt_f16_u16_z((b),(a))
#define VCVT_SHORT_INT_LOW(a) svunpklo_s32((a))
#define VCVT_SHORT_INT_HIGH(a) svunpkhi_s32((a))
#define VLOAD_SHORT(a,b) svld1_s16((b),(a))
#define VCVT_FLOAT16_SHORT(a,b) svcvt_s16_f16_z((b),(a))
#define VCVT_FLOAT16_USHORT(a,b) svcvt_u16_f16_z((b),(a))
#define VCVT_FLOAT16_UBYTE(a,b) svcvt_u8_f16_z((b),(a))
#define VCVT_FLOAT16_FLOAT(a,b) svcvt_f32_f16_z((b),(a))
#define VMUL1_FLOAT16(a,b,c) svmul_n_f16_z((c),(a),(b))
#define VSTORE_SHORT(a,b,c) svst1_s16((c),(a),(b))
#define VSTORE_USHORT(a,b,c) svst1_u16((c),(a),(b))
#define VSTORE_UBYTE(a,b,c) svst1_u8((c),(a),(b))

///////////64bits
#define V_ELT_DOUBLE svfloat64_t
#define VLOAD_DOUBLE(a,b) svld1_f64((b),(a))
#define VLOAD_INT64(a,b) svld1_s64((b),(a))
#define VLOAD_UINT64(a,b) svld1_u64((b),(a))
#define VLOAD1_DOUBLE(a,b) svdup_n_f64((a))
#define VLOAD1_INT64(a,b) svdup_n_s64((a))
#define VSTORE_DOUBLE(a,b,c) svst1_f64((c),(a),(b))
#define VSTORE_INT64(a,b,c) svst1_s64((c),(a),(b))
#define V_ELT_INT64 svint64_t
#define V_ELT_UINT64 svuint64_t
#define VINTERP_INT64_DOUBLE svreinterpret_f64_s64
#define VINTERP_DOUBLE_INT64 svreinterpret_s64_f64
#define VAND1_INT64(a,b,c) svand_n_u64_z((c),(a),(b))
#define VMUL1_DOUBLE(a,b,c) svmul_n_f64_z((c),(a),(b))
#define VMUL1_DOUBLE_MASK(a,b,c,d) svmul_n_f64_m((a),(b),(c))
#define VMUL_DOUBLE(a,b,c) svmul_f64_z((c),(a),(b))
#define VDIV_DOUBLE(a,b,c) svdiv_f64_z((c),(a),(b))
#define VADD1_DOUBLE(a,b,c) svadd_n_f64_z((c),(a),(b))
#define VADD1_DOUBLE_MASK(a,b,c,d) svadd_n_f64_m((a),(b),(c))
#define VADD1_INT64(a,b,c) svadd_n_s64_z((c),(a),(b))
#define VADD1_INT64_MASK(a,b,c,d) svadd_n_s64_m((a),(b),(c))
#define VSUB1_INT64_MASK(a,b,c,d) svsub_n_s64_m((a),(b),(c))
#define VADD_DOUBLE(a,b,c) svadd_f64_z((c),(a),(b))
#define VSUB_DOUBLE(a,b,c) svsub_f64_z((c),(a),(b))
#define VSUB1_DOUBLE(a,b,c) svsub_n_f64_z((c),(a),(b))
#define VSUB1_INT64(a,b,c) svsub_n_s64_z((c),(a),(b))
#define VSLL1_INT64(a,b,c) svlsl_n_s64_z((c),(a),(b))
#define VCVT_INT64_DOUBLE(a,b) svcvt_f64_s64_z((b),(a))
#define VCVT_DOUBLE_INT64(a,b) svcvt_s64_f64_z((b),(a))
#define VCVT_RTZ_DOUBLE_INT64 VCVT_DOUBLE_INT64
#define VGT1_INT64_BOOL(a,b,c) svcmpgt_n_s64((c),(a),(b))
#define VNE1_INT64_BOOL(a,b,c) svcmpne_n_s64((c),(a),(b))
#define VSLL1_INT64(a,b,c) svlsl_n_s64_z((c),(a),(b))

#warning "SSE andnot => ( not a) and b, vbic => a & (not b)
#define VANDNOT1_INT64(a,b,c) svbic_n_s64_z((c),(a),(b))
#define VANDNOT_INT64(a,b,c) svbic_s64_z((c),(b),(a)) //equivalent to andnot

// x current exact, a nearest away from zero, i current mode inexact, m mininf,
// n nearest even, p plusinf
#define VR_DOUBLE(a,b) svrintx_f64_z((b),(a))
#define VRTZ_DOUBLE(a,b) svrintz_f64_z((b),(a))
#define VRNE_DOUBLE(a,b) svrintn_f64_z((b),(a))
#define VRMINF_DOUBLE(a,b) svrintm_f64_z((b),(a))
#define VRINF_DOUBLE(a,b) svrintp_f64_z((b),(a))
#define VRNA_DOUBLE(a,b) svrinta_f64_z((b),(a))

#define V_ELT_BOOL64 svbool_t
#define VEQ1_INT64_BOOL(a,b,c) svcmpeq_n_s64((c),(a),(b))
#define VGT1_DOUBLE_BOOL(a,b,c) svcmpgt_n_f64((c),(a),(b))
#define VGE1_DOUBLE_BOOL(a,b,c) svcmpge_n_f64((c),(a),(b))
#define VLT1_DOUBLE_BOOL(a,b,c) svcmplt_n_f64((c),(a),(b))
#define VLE1_DOUBLE_BOOL(a,b,c) svcmple_n_f64((c),(a),(b))
#define VEQ1_DOUBLE_BOOL(a,b,c) svcmpeq_n_f64((c),(a),(b))

//(svbool_t pg, svfloat64_t op1, svfloat64_t op2, float64_t op3) //op3 + op1[i]*op2[i]
#define VFMASQ1_DOUBLE(a,b,c,d) svmad_n_f64_x((d),(a),(b),(c)) 
#define VFMACC1_DOUBLE(a,b,c,d) svmla_n_f64_z((d),(a),(c),(b)) // op1[i] + op2[i] * op3
#define VFMACC_DOUBLE(a,b,c,d) svmla_f64_z((d),(a),(c),(b)) // op1[i] + op2[i] * op3[i]
#define VFMADD1_DOUBLE(a,b,c,d) svmla_n_f64_z((d),(c),(a),(b)) //  op1[i] + op2[i] * op3

//SVE2 computes a - b*c where riscv does b*c - a! 
#define VFMSUB_DOUBLE(a,b,c,d) svmls_f64_z((d),(c),(a),(b)) //  op1[i] - op2[i] * op3[i]

#define VFMADD_DOUBLE(a,b,c,d) svmad_f64_z((d),(a),(b),(c)) // op1[i] * op2[i] + op3[i]
#define VMERGE_DOUBLE(mask, t, f, i) svsel_f64((mask), (f), (t))     /* select */
#define VMERGE1_DOUBLE(mask, v, f, i) svsel_f64((mask), svdup_n_f64((float)(f)),(v))
#define VOR_INT64(a,b,c)  svorr_s64_m((c),(a),(b))
#define VOR1_INT64(a,b,c)  svorr_n_s64_z((c),(a),(b))
#define VNOT_INT64(a,b)  svnot_s64_z((b),(a))
#define VXOR_INT64(a,b,c)  sveor_s64_z((c),(a),(b))
//#define VXOR_DOUBLE(a,b,c)  sveor_f64((c),(a),(b))
#define VXOR1_INT64(a,b,c)  sveor_n_s64_z((c),(a),(b))
#define VCLEAR_BOOL(a) svpfalse_b()

#define VXOR_BOOL(a,b,c)  sveor_b_z((c),(a),(b))
#define VOR_BOOL(a,b,c)  svorr_b_z((c),(a),(b))
#define VAND_BOOL(a,b,c)  svand_b_z((c),(a),(b))
#define VOR_BOOL(a,b,c)  svorr_b_z((c),(a),(b))
#define VNOT_BOOL(a,b)  svnot_b_z((b),(a))
#define VGT1_INT64_BOOL(a,b,c) svcmpgt_n_s64((c),(a),(b))
#define VNE1_INT64_BOOL(a,b,c) svcmpne_n_s64((c),(a),(b))

#define VRDIV1_DOUBLE(a,b,c) svdivr_n_f64_z((c),(a),(b))
#define VRSUB1_DOUBLE(a,b,c) svsubr_n_f64_z((c),(a),(b))

#define V_ELT_DOUBLE2 svfloat64x2_t
#define VSTORE_DOUBLE2SPLIT(a,b,c,d) svst2_f64((d),(double*)(a),(V_ELT_DOUBLE2){b,c})
#define VSTORE_DOUBLE2(a,b,c) svst2_f64((c),(double*)(a),(b))
#define VLOAD_DOUBLE2(a,b) svld2_f64((b),(double*)(a))

#define VREDSUM_DOUBLE(a,b)  svaddv_f64((b),(a))
#define VREDSUMORD_DOUBLE(a,b,c)  svadda_f64((c),(a),(b))
#define VSQRT_DOUBLE(a,b) svsqrt_f64_z((b),(a))
#define VMAX_DOUBLE(a,b,c) svmax_f64_z((c),(a),(b))
#define VMIN_DOUBLE(a,b,c) svmin_f64_z((c),(a),(b))
#define VMAX1_DOUBLE(a,b,c) svmax_n_f64_z((c),(a),(b))
#define VMIN1_DOUBLE(a,b,c) svmin_n_f64_z((c),(a),(b))
#define VREDMAX_DOUBLE(a,b) svmaxv_f64((b),(a))
#define VREDMIN_DOUBLE(a,b) svminv_f64((b),(a))
#define VABS_DOUBLE(a,b) svabs_f64_z((b),(a))
#define VSRA1_INT64(a,b,c) svasr_n_s64_z((c),(a),(b))

#define VGATHER_DOUBLE(a,b,c) svld1_gather_s64offset_f64((c),(a),(b))

#define VCVT_DOUBLE_DOUBLE(a,b) svcvt_f64_f64_z((b),(a))
#define VCVT_DOUBLE_DOUBLE(a,b) svcvt_f64_f64_z((b),(a))
// returns dst = dst  + a*b
static inline V_ELT_DOUBLE VMUL_CDOUBLE(V_ELT_DOUBLE dst, V_ELT_DOUBLE a, V_ELT_DOUBLE b, V_ELT_BOOL64 i){
	dst = svcmla_f64_z(i,dst,a,b,0);
	dst = svcmla_f64_z(i,dst,a,b,90);
	return dst;
}

//returns dst = dst + conj(a)*b
static inline V_ELT_DOUBLE VMULCONJA_CDOUBLE(V_ELT_DOUBLE dst, V_ELT_DOUBLE a, V_ELT_DOUBLE b, V_ELT_BOOL64 i){
	dst = svcmla_f64_z(i,dst,a,b,0);
	dst = svcmla_f64_z(i,dst,a,b,270);
	return dst;
}

#endif // SVE2

#ifdef RISCV
#include <riscv_vector.h>

/* ELEN : element length, 8,16,32,64bits
    VLEN : Vector Length, at least 128bits
    32 registers in the 0.10 standard, plus vstart, vxsat, vxrm, vcsr, vtype, vl, vlenb
    VSEW : Vector Standard Element Width (dynamic), with of the base element : 8,16,32,64,...,1024bits
    (up to 64bit in the current intrinsics
    LMUL : Vector register grouping => may group multiple VLEN registers, so that 1 instruction can be applied to multiple registers. If LMUL is < 1, the operation applies only to a part of the register
    LMUL = 1,2,4,8, 1, 1/2, 1/4, 1/8
    VLMAX = LMUL*VLEN/SEW
    Vector Tail Agnostic and Vector Mask Agnostic vta and vma allow to mask operations on vector such as only part of a vector is modified
    Vector Fixed-Point Rounding Mode Register vxrm for rounding mode : round-to-nearest-up rnu, round-to-nearest-even rne, round-down rdn, round-to-odd rod

Need a real CPU with CPI/latency to have better choice of instructions..
fmadd vs fmacc, load stride vs segment, etc

*/

// 0 to nearest, 1 to zero (trunc), 2 round down, 3 round up, 4 round to nearest
#define _MM_ROUND_NEAREST 0
#define _MM_ROUND_TOWARD_ZERO 1
#define _MM_ROUND_DOWN 2
#define _MM_ROUND_UP 3
#define _MM_ROUND_AWAY 4

// load vector float32, 8
// "1" in name means either vector scalar instructions, or load/store scalar to vector

/*
# FP multiply-accumulate, overwrites addend
vfmacc.vv vd, vs1, vs2, vm
# vd[i] = +(vs1[i] * vs2[i]) + vd[i]
vfmacc.vf vd, rs1, vs2, vm
# vd[i] = +(f[rs1] * vs2[i]) + vd[i]
# FP negate-(multiply-accumulate), overwrites subtrahend
vfnmacc.vv vd, vs1, vs2, vm
# vd[i] = -(vs1[i] * vs2[i]) - vd[i]
vfnmacc.vf vd, rs1, vs2, vm
# vd[i] = -(f[rs1] * vs2[i]) - vd[i]
# FP multiply-subtract-accumulator, overwrites subtrahend
vfmsac.vv vd, vs1, vs2, vm
# vd[i] = +(vs1[i] * vs2[i]) - vd[i]
vfmsac.vf vd, rs1, vs2, vm
# vd[i] = +(f[rs1] * vs2[i]) - vd[i]
# FP negate-(multiply-subtract-accumulator), overwrites minuend
vfnmsac.vv vd, vs1, vs2, vm
# vd[i] = -(vs1[i] * vs2[i]) + vd[i]
vfnmsac.vf vd, rs1, vs2, vm
# vd[i] = -(f[rs1] * vs2[i]) + vd[i]
# FP multiply-add, overwrites multiplicand
vfmadd.vv vd, vs1, vs2, vm
# vd[i] = +(vs1[i] * vd[i]) + vs2[i]
vfmadd.vf vd, rs1, vs2, vm
# vd[i] = +(f[rs1] * vd[i]) + vs2[i]
# FP negate-(multiply-add), overwrites multiplicand
vfnmadd.vv vd, vs1, vs2, vm
# vd[i] = -(vs1[i] * vd[i]) - vs2[i]
vfnmadd.vf vd, rs1, vs2, vm
# vd[i] = -(f[rs1] * vd[i]) - vs2[i]
# FP multiply-sub, overwrites multiplicand
vfmsub.vv vd, vs1, vs2, vm
# vd[i] = +(vs1[i] * vd[i]) - vs2[i]
vfmsub.vf vd, rs1, vs2, vm
# vd[i] = +(f[rs1] * vd[i]) - vs2[i]
# FP negate-(multiply-sub), overwrites multiplicand
vfnmsub.vv vd, vs1, vs2, vm
# vd[i] = -(vs1[i] * vd[i]) + vs2[i]
vfnmsub.vf vd, rs1, vs2, vm
# vd[i] = -(f[rs1] * vd[i]) + vs2[i]
*/

#ifndef ELEN
#define ELEN 64  // vector support elements up to 64 bits
#endif

#ifndef VECTOR_LENGTH
#define MAX_ELTS8 128  // 1024bits*4 registers(m4) => 512 int8
#define MAX_ELTS32 32  // 1024bits*4 registers(m4) => 128 float/int32
#define MAX_ELTS64 16  // 1024bits*4 registers(m4) => 64 double/int64
#define VECTOR_LENGTH 1024
#else
#define MAX_ELTS8 VECTOR_LENGTH
#define MAX_ELTS32 VECTOR_LENGTH / 4
#define MAX_ELTS64 VECTOR_LENGTH / 8
#endif


#if __riscv_v == 7000
#define NO_RTZ
#warning "RVV v0.7, NO_RTZ"
#endif

#if defined(__clang__) ||  (defined(__GNUC__) && __GNUC__ < 11)
#warning "RVV intrinsics v0.10"

#define vfcvt_rtz_x_f_v_i32m4 vfcvt_x_f_v_i32m4
#define vfcvt_rtz_x_f_v_i32m2 vfcvt_x_f_v_i32m2

///////////////////// FULL VECTOR  m4 //////////////
#define VSETVL32 vsetvl_e32m4
#define VSETVL16 vsetvl_e16m4

//// FLOAT
#define V_ELT_FLOAT vfloat32m4_t
#define VLOAD_FLOAT vle32_v_f32m4
#define VLOAD1_FLOAT vfmv_v_f_f32m4
#define VSTORE_FLOAT vse32_v_f32m4
#define VADD_FLOAT vfadd_vv_f32m4
#define VADD1_FLOAT vfadd_vf_f32m4
#define VSUB_FLOAT vfsub_vv_f32m4
#define VSUB1_FLOAT vfsub_vf_f32m4
#define VRSUB1_FLOAT vfrsub_vf_f32m4  // v2 = f - v1
#define VMUL_FLOAT vfmul_vv_f32m4
#define VMUL1_FLOAT vfmul_vf_f32m4
#define VDIV_FLOAT vfdiv_vv_f32m4
#define VDIV1_FLOAT vfdiv_vf_f32m4
#define VRDIV1_FLOAT vfrdiv_vf_f32m4
#define VFMACC_FLOAT vfmacc_vv_f32m4  // vd[i] = +(vs1[i] * vs2[i]) + vd[i]
#define VFMACC1_FLOAT vfmacc_vf_f32m4
#define VFMADD_FLOAT vfmadd_vv_f32m4  // vd[i] = +(vs1[i] * vd[i]) + vs2[i]
#define VFMADD1_FLOAT vfmadd_vf_f32m4
#define VFMSUB_FLOAT vfmsub_vv_f32m4  // d = a*b - c
#define VREDSUM_FLOAT vfredosum_vs_f32m4_f32m1
#define VREDMAX_FLOAT vfredmax_vs_f32m4_f32m1
#define VREDMIN_FLOAT vfredmin_vs_f32m4_f32m1
#define VMIN_FLOAT vfmin_vv_f32m4
#define VMAX_FLOAT vfmax_vv_f32m4
#define VMIN1_FLOAT vfmin_vf_f32m4
#define VMAX1_FLOAT vfmax_vf_f32m4
#define VINTERP_FLOAT_INT vreinterpret_v_f32m4_i32m4
#define VINTERP_INT_FLOAT vreinterpret_v_i32m4_f32m4
#define VCVT_RTZ_FLOAT_INT vfcvt_rtz_x_f_v_i32m4
#define VCVT_FLOAT_INT vfcvt_x_f_v_i32m4
#define VCVT_INT_FLOAT vfcvt_f_x_v_f32m4
#define VMERGE_FLOAT vmerge_vvm_f32m4
static inline vfloat32m4_t VMUL1_FLOAT_MASK(vbool8_t mask, vfloat32m4_t op1, float op2, size_t vl){
	return vfmul_vf_f32m4_m(mask, op1, op1, op2, vl);
}
#define VSQRT_FLOAT vfsqrt_v_f32m4
#define VLE_FLOAT_STRIDE vlse32_v_f32m4
#define VEQ1_FLOAT_BOOL vmfeq_vf_f32m4_b8
#define VEQ_FLOAT_BOOL vmfeq_vv_f32m4_b8
#define VGT1_FLOAT_BOOL vmfgt_vf_f32m4_b8
#define VNE1_FLOAT_BOOL vmfne_vf_f32m4_b8
#define VLT1_FLOAT_BOOL vmflt_vf_f32m4_b8
#define VLE1_FLOAT_BOOL vmfle_vf_f32m4_b8
#define VABS_FLOAT vfabs_v_f32m4
#define VMERGE1_FLOAT vfmerge_vfm_f32m4
#define VGATHER_FLOAT vrgather_vv_f32m4

//// INT
#define V_ELT_INT vint32m4_t
#define VLOAD_INT vle32_v_i32m4
#define VLOAD1_INT vmv_v_x_i32m4
#define VSTORE_INT vse32_v_i32m4
#define VADD_INT vadd_vv_i32m4
#define VADD1_INT vadd_vx_i32m4
#define VMUL_INT vmul_vv_i32m4
#define VMUL1_INT vmul_vx_i32m4
#define VSUB_INT vsub_vv_i32m4
#define VSUB1_INT vsub_vx_i32m4
#define VAND1_INT vand_vx_i32m4
#define VAND_INT vand_vv_i32m4
#define VXOR_INT vxor_vv_i32m4
#define VSLL1_INT vsll_vx_i32m4
#define VEQ1_INT_BOOL vmseq_vx_i32m4_b8
#define VEQ_INT_BOOL vmseq_vv_i32m4_b8
#define VGT1_INT_BOOL vmsgt_vx_i32m4_b8
#define VNE1_INT_BOOL vmsne_vx_i32m4_b8
#define VLT1_INT_BOOL vmslt_vx_i32m4_b8
#define VLE1_INT_BOOL vmsle_vx_i32m4_b8
static inline vint32m4_t VADD1_INT_MASK(vbool8_t mask, vint32m4_t op1, int32_t op2, size_t vl){
	return vadd_vx_i32m4_m(mask, op1, op1, op2, vl);
}
static inline vint32m4_t VSUB1_INT_MASK(vbool8_t mask, vint32m4_t op1, int32_t op2, size_t vl){
	return vsub_vx_i32m4_m(mask, op1, op1, op2, vl);
}
#define VSUB1_INT vsub_vx_i32m4
#define VOR1_INT vor_vx_i32m4
#define VSRA1_INT vsra_vx_i32m4
#define VMIN_INT vmin_vv_i32m4
#define VMIN1_INT vmin_vx_i32m4
#define VMAX_INT vmax_vv_i32m4
#define VMAX1_INT vmax_vx_i32m4
#define VMERGE1_INT vmerge_vxm_i32m4
#define VMERGE_INT vmerge_vvm_i32m4
#define VNEG_INT vneg_v_i32m4
#define VREDSUM_INT vredosum_vs_i32m4_i32m1
#define VREDMAX_INT vredmax_vs_i32m4_i32m1
#define VREDMIN_INT vredmin_vs_i32m4_i32m1
#define VGATHER_INT vrgather_vv_i32m4
#define VNOT_INT vnot_v_i32m4

//// UINT
#define VLOAD_UINT vle32_v_u32m4
#define VSTORE_UINT vse32_v_u32m4
#define V_ELT_UINT vuint32m4_t
#define VCVT_FLOAT_UINT vfcvt_xu_f_v_u32m4

//// SHORT
#define V_ELT_SHORT vint16m4_t
#define VLOAD_SHORT vle16_v_i16m4
#define VLOAD1_SHORT vmv_v_x_i16m4
#define VSTORE_SHORT vse16_v_i16m4
#define VADD_SHORT vadd_vv_i16m4
#define VSUB_SHORT vsub_vv_i16m4
#define VREDSUMW_SHORT vwredsum_vs_i16m4_i32m1
#define VGT_SHORT_BOOL vmsgt_vv_i16m4_b4
#define VMERGE_SHORT vmerge_vvm_i16m4

//// BOOL for 16 bits elements
#define V_ELT_BOOL16 vbool4_t

//// BOOL for 32 bits elements
#define V_ELT_BOOL32 vbool8_t
#define VNOT_BOOL vmnot_m_b8
#define VCLEAR_BOOL vmclr_m_b8
#define VXOR_BOOL vmxor_mm_b8
#define VOR_BOOL vmor_mm_b8
#define VAND_BOOL vmand_mm_b8
#define VANDNOT_BOOL vmandn_mm_b8

/////////////////////////// HALF VECTOR, m2 ///////////////
#define VSETVL32H vsetvl_e32m2
#define VSETVL16H vsetvl_e16m2

//// FLOATH
#define V_ELT_FLOATH vfloat32m2_t
 
#define VLOAD_FLOATH vle32_v_f32m2
#define VLOAD1_FLOATH vfmv_v_f_f32m2
#define VLOAD_FLOATH2 vlseg2e32_v_f32m2
#define VLOAD_FLOATH_STRIDE vlse32_v_f32m2
#define VSTORE_FLOATH vse32_v_f32m2
#define VSTORE_FLOATHH vse32_v_f32m1
#define VLOAD_FLOATHH vle32_v_f32m1
#define VSTORE_FLOATH2 vsseg2e32_v_f32m2
#define VINTERP_FLOATH_INTH vreinterpret_v_f32m2_i32m2
#define VINTERP_INTH_FLOATH vreinterpret_v_i32m2_f32m2
#define VXOR1_INTH vxor_vx_i32m2
#define VADD_FLOATH vfadd_vv_f32m2
#define VADD1_FLOATH vfadd_vf_f32m2
static inline vfloat32m2_t VADD1_FLOATH_MASK(vbool16_t mask, vfloat32m2_t op1, float op2, size_t vl){
	return vfadd_vf_f32m2_m(mask, op1, op1, op2, vl);
}
#define VSUB_FLOATH vfsub_vv_f32m2
#define VSUB1_FLOATH vfsub_vf_f32m2    // v2 = v1 - f
#define VRSUB1_FLOATH vfrsub_vf_f32m2  // v2 = f - v1
#define VMUL_FLOATH vfmul_vv_f32m2
#define VMUL1_FLOATH vfmul_vf_f32m2
static inline vfloat32m2_t VMUL1_FLOATH_MASK(vbool16_t mask, vfloat32m2_t op1, float op2, size_t vl){
	return vfmul_vf_f32m2_m(mask, op1, op1, op2, vl);
}
#define VDIV_FLOATH vfdiv_vv_f32m2
#define VDIV1_FLOATH vfdiv_vf_f32m2
#define VRDIV1_FLOATH vfrdiv_vf_f32m2
#define VFMACC_FLOATH vfmacc_vv_f32m2  // d = a + b*c
#define VFMACC1_FLOATH vfmacc_vf_f32m2
#define VFMADD_FLOATH vfmadd_vv_f32m2  // vd[i] = +(vs1[i] * vd[i]) + vs2[i]
#define VFMADD1_FLOATH vfmadd_vf_f32m2
#define VFMSUB_FLOATH vfmsub_vv_f32m2  // d = a*b - c
#define VREDSUM_FLOATH vfredosum_vs_f32m2_f32m1
#define VREDMAX_FLOATH vfredmax_vs_f32m2_f32m1
#define VREDMIN_FLOATH vfredmin_vs_f32m2_f32m1
#define VMIN_FLOATH vfmin_vv_f32m2
#define VMIN1_FLOATH vfmin_vf_f32m2
#define VMAX_FLOATH vfmax_vv_f32m2
#define VMAX1_FLOATH vfmax_vf_f32m2
#define VINTERP_FLOATH_INTH vreinterpret_v_f32m2_i32m2
#define VINTERP_INTH_FLOATH vreinterpret_v_i32m2_f32m2
#define VCVT_RTZ_FLOATH_INTH vfcvt_rtz_x_f_v_i32m2
#define VCVT_FLOATH_INTH vfcvt_x_f_v_i32m2
#define VCVT_INTH_FLOATH vfcvt_f_x_v_f32m2
#define VMERGE_FLOATH vmerge_vvm_f32m2
#define VSQRT_FLOATH vfsqrt_v_f32m2
#define VEQ1_FLOATH_BOOLH vmfeq_vf_f32m2_b16
#define VEQ_FLOATH_BOOLH vmfeq_vv_f32m2_b16
#define VGE1_FLOATH_BOOLH vmfge_vf_f32m2_b16
#define VGT1_FLOATH_BOOLH vmfgt_vf_f32m2_b16
#define VNE1_FLOATH_BOOLH vmfne_vf_f32m2_b16
#define VLT1_FLOATH_BOOLH vmflt_vf_f32m2_b16
#define VLE1_FLOATH_BOOLH vmfle_vf_f32m2_b16
#define VABS_FLOATH vfabs_v_f32m2
#define VMERGE1_FLOATH vfmerge_vfm_f32m2
#define VGATHER_FLOATH vrgather_vv_f32m2

//// INTH
#define V_ELT_INTH vint32m2_t
#define VSTORE_INTHH vse32_v_i32m1
#define VLOAD_INTHH  vle32_v_i32m1
#define VLOAD_INTH vle32_v_i32m2
#define VLOAD1_INTH vmv_v_x_i32m2
#define VLOAD1_INTHH vmv_v_x_i32m1
#define VSTORE_INTH vse32_v_i32m2
#define VADD_INTH vadd_vv_i32m2
#define VADD1_INTH vadd_vx_i32m2
static inline vint32m2_t VADD1_INTH_MASK(vbool16_t mask, vint32m2_t op1, int32_t op2, size_t vl){
	return vadd_vx_i32m2_m(mask, op1, op1, op2, vl);
}

#define VADD1_INTH_MASKEDOFF vadd_vx_i32m2_m
#define VADD1_INT64H_MASKEDOFF vadd_vx_i64m2_m
#define VMUL_INTH vmul_vv_i32m2
#define VMUL1_INTH vmul_vx_i32m2
#define VSUB_INTH vsub_vv_i32m2
#define VSUB1_INTH vsub_vx_i32m2
static inline vint32m2_t VSUB1_INTH_MASK(vbool16_t mask, vint32m2_t op1, int32_t op2, size_t vl){
	return vsub_vx_i32m2_m(mask, op1, op1, op2, vl);
}
#define VAND1_INTH vand_vx_i32m2
#define VAND_INTH vand_vv_i32m2
#define VXOR_INTH vxor_vv_i32m2
#define VSLL1_INTH vsll_vx_i32m2
#define VEQ1_INTH_BOOLH vmseq_vx_i32m2_b16
#define VGT1_INTH_BOOLH vmsgt_vx_i32m2_b16
#define VNE1_INTH_BOOLH vmsne_vx_i32m2_b16
#define VLT1_INTH_BOOLH vmflt_vf_f32m2_b16
#define VLE1_INTH_BOOLH vmsle_vx_i32m2_b16
#define VEQ_INTH_BOOLH vmseq_vv_i32m2_b16
#define VOR1_INTH vor_vx_i32m2
#define VSRA1_INTH vsra_vx_i32m2
#define VMIN_INTH vmin_vv_i32m2
#define VMIN1_INTH vmin_vx_i32m2
#define VMAX_INTH vmax_vv_i32m2
#define VMAX1_INTH vmax_vx_i32m2
#define VNOT_INTH vnot_v_i32m2
#define VMERGE_INTH vmerge_vvm_i32m2

//// UINTH
#define VLOAD_UINTH vle32_v_u32m2
#define V_ELT_UINTH vuint32m2_t
#define VCVT_FLOATH_UINTH vfcvt_xu_f_v_u32m2

//// SHORTH
#define V_ELT_SHORTH vint16m2_t
#define VLOAD_SHORTH vle16_v_i16m2
#define VLOAD1_SHORTH vmv_v_x_i16m2
#define VSTORE_SHORTH vse16_v_i16m2
#define VADD_SHORTH vadd_vv_i16m2
#define VREDSUMW_SHORTH vwredsum_vs_i16m4_i32m1
#define VCVT_INT_SHORTH vnclip_wx_i16m2

#if __riscv_v != 7000
#define VCVT_SHORTH_INT vsext_vf2_i32m4
#else
#define VCVT_SHORTH_INT(a, b) vwmul_vx_i32m4((a), 1, (b))
#endif

//// USHORTH
#define V_ELT_USHORTH vuint16m2_t
#define VLOAD_USHORTH vle16_v_u16m2
#define VSTORE_USHORTH vse16_v_u16m2
#define VCVT_UINT_USHORTH vnclipu_wx_u16m2

//// UBYTEHH
#define V_ELT_UBYTEHH vuint8m1_t
#define VLOAD_UBYTEHH vle8_v_u8m1
#define VSTORE_UBYTEHH vse8_v_u8m1
#define VCVT_USHORTH_UBYTEHH vnclipu_wx_u8m1

//// BOOL for Half length vector 32 bits elements
#define V_ELT_BOOL32H vbool16_t
#define VNOT_BOOLH vmnot_m_b16
#define VCLEAR_BOOLH vmclr_m_b16
#define VXOR_BOOLH vmxor_mm_b16
#define VOR_BOOLH vmor_mm_b16
#define VAND_BOOLH vmand_mm_b16
#define VANDNOT_BOOLH vmandn_mm_b16
//#define VANDNOT_BOOLH vmnand_mm_b16

#if ELEN >= 64
#define VSETVL64 vsetvl_e64m4

//// DOUBLE
#define V_ELT_DOUBLE vfloat64m4_t
#define VLOAD_DOUBLE vle64_v_f64m4
#define VLOAD1_DOUBLE vfmv_v_f_f64m4
#define VSTORE_DOUBLE vse64_v_f64m4
#define VADD_DOUBLE vfadd_vv_f64m4
#define VADD1_DOUBLE vfadd_vf_f64m4
#define VSUB_DOUBLE vfsub_vv_f64m4
#define VSUB1_DOUBLE vfsub_vf_f64m4
#define VMUL_DOUBLE vfmul_vv_f64m4
#define VMUL1_DOUBLE vfmul_vf_f64m4
#define VDIV_DOUBLE vfdiv_vv_f64m4
#define VFMA_DOUBLE vfmacc_vv_f64m4  // d = a + b*c
#define VFMA1_DOUBLE vfmacc_vf_f64m4
#define VFMSUB_DOUBLE vfmsub_vv_f64m4  // d = a*b - c
#define VREDSUM_DOUBLE vfredosum_vs_f64m4_f64m1
#define VREDMAX_DOUBLE vfredmax_vs_f64m4_f64m1
#define VREDMIN_DOUBLE vfredmin_vs_f64m4_f64m1
#define VMIN_DOUBLE vfmin_vv_f64m4
#define VMAX_DOUBLE vfmax_vv_f64m4
#define VMIN1_DOUBLE vfmin_vf_f64m4
#define VMAX1_DOUBLE vfmax_vf_f64m4
#define VINTERP_DOUBLE_INT vreinterpret_v_f64m4_i64m4
#define VINTERP_INT_DOUBLE vreinterpret_v_i64m4_f64m4
#define VCVT_RTZ_DOUBLE_INT vfcvt_rtz_x_f_v_i64m4
#define VCVT_DOUBLE_INT vfcvt_x_f_v_i64m4
#define VCVT_INT_DOUBLE vfcvt_f_x_v_f64m4
#define VABS_DOUBLE vfabs_v_f64m4
#define VSQRT_DOUBLE vfsqrt_v_f64m4
#define VCVT_DOUBLE_FLOAT vfncvt_f_f_w_f32m2
#define VCVT_FLOAT_DOUBLE vfwcvt_f_f_v_f64m4

//// DOUBLEH
#define VSETVL64H vsetvl_e64m2
#define V_ELT_DOUBLEH vfloat64m2_t
#define VLOAD_DOUBLEH2 vlseg2e64_v_f64m2
#define VLOAD_DOUBLEH_STRIDE vlse64_v_f64m2
#define VSTORE_DOUBLEH2 vsseg2e64_v_f64m2
#define VLOAD_DOUBLEH vle64_v_f64m2
#define VLOAD1_DOUBLEH vfmv_v_f_f64m2
#define VSTORE_DOUBLEH vse64_v_f64m2
#define VADD_DOUBLEH vfadd_vv_f64m2
#define VADD1_DOUBLEH vfadd_vf_f64m2
#define VSUB_DOUBLEH vfsub_vv_f64m2
#define VSUB1_DOUBLEH vfsub_vf_f64m2
#define VRSUB1_DOUBLEH vfrsub_vf_f64m2  // v2 = f - v1
#define VMUL_DOUBLEH vfmul_vv_f64m2
#define VMUL1_DOUBLEH vfmul_vf_f64m2
#define VDIV_DOUBLEH vfdiv_vv_f64m2
#define VFMACC_DOUBLEH vfmacc_vv_f64m2  // d = a + b*c
#define VFMACC1_DOUBLEH vfmacc_vf_f64m2  
#define VFMADD_DOUBLEH vfmadd_vv_f64m2  
#define VFMADD1_DOUBLEH vfmadd_vf_f64m2 
#define VFMA1_DOUBLEH vfmacc_vf_f64m2
#define VFMSUB_DOUBLEH vfmsub_vv_f64m2  // d = a*b - c
#define VREDSUM_DOUBLEH vfredosum_vs_f64m2_f64m1
#define VREDMAX_DOUBLEH vfredmax_vs_f64m2_f64m1
#define VREDMIN_DOUBLEH vfredmin_vs_f64m2_f64m1
#define VMIN_DOUBLEH vfmin_vv_f64m2
#define VMAX_DOUBLEH vfmax_vv_f64m2
#define VMIN1_DOUBLEH vfmin_vf_f64m2
#define VMAX1_DOUBLEH vfmax_vf_f64m2
#define VINTERP_DOUBLEH_INTH vreinterpret_v_f64m2_i64m2
#define VINTERP_INTH_DOUBLEH vreinterpret_v_i64m2_f64m2
#define VCVT_RTZ_DOUBLEH_INTH vfcvt_rtz_x_f_v_i64m2
#define VCVT_DOUBLEH_INTH vfcvt_x_f_v_i64m2
#define VCVT_INTH_DOUBLEH vfcvt_f_x_v_f64m2
#define VABS_DOUBLEH vfabs_v_f64m2
#define VSQRT_DOUBLEH vfsqrt_v_f64m2
#define VCVT_DOUBLEH_FLOATH vfncvt_f_f_w_f32m2
#define VCVT_FLOATH_DOUBLEH vfwcvt_f_f_v_f64m2
#define VLT1_DOUBLEH_BOOLH vmflt_vf_f64m2_b32
#define VMERGE_DOUBLEH vmerge_vvm_f64m2
#define VMERGE1_DOUBLEH vfmerge_vfm_f64m2
static inline vfloat64m2_t VMUL1_DOUBLEH_MASK(vbool32_t mask, vfloat64m2_t op1, double op2, size_t vl){
	return vfmul_vf_f64m2_m(mask, op1, op1, op2, vl);
}
static inline vfloat64m2_t VADD1_DOUBLEH_MASK(vbool32_t mask, vfloat64m2_t op1, double op2, size_t vl){
	return vfadd_vf_f64m2_m(mask, op1, op1, op2, vl);
}
#define VADD1_DOUBLEH_MASKEDOFF vfadd_vf_f64m2_m
#define VEQ1_DOUBLEH_BOOLH vmfeq_vf_f64m2_b32
#define VEQ_DOUBLEH_BOOLH vmfeq_vv_f64m2_b32
#define VGE1_DOUBLEH_BOOLH vmfge_vf_f64m2_b32
#define VGT1_DOUBLEH_BOOLH vmfgt_vf_f64m2_b32
#define VNE1_DOUBLEH_BOOLH vmfne_vf_f64m2_b32
#define VLT1_DOUBLEH_BOOLH vmflt_vf_f64m2_b32
#define VLE1_DOUBLEH_BOOLH vmfle_vf_f64m2_b32


// INT64H
#define V_ELT_INT64H vint64m2_t
#define VLOAD1_INT64H vmv_v_x_i64m2
#define VSLL1_INT64H vsll_vx_i64m2
#define VADD1_INT64H vadd_vx_i64m2
static inline vint64m2_t VADD1_INT64H_MASK(vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl){
	return vadd_vx_i64m2_m(mask, op1, op1, op2, vl);
}
static inline vint64m2_t VSUB1_INT64H_MASK(vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl){
	return vsub_vx_i64m2_m(mask, op1, op1, op2, vl);
}
#define VAND1_INT64H vand_vx_i64m2
#define VXOR_INT64H vxor_vv_i64m2
#define VNE1_INTH_BOOL64H vmsne_vx_i64m2_b32
#define VEQ1_INTH_BOOL64H vmseq_vx_i64m2_b32
#define VEQ1_INTH_BOOL64H vmseq_vx_i64m2_b32
#define VGT1_INTH_BOOL64H vmsgt_vx_i64m2_b32
#define VMERGE_INT64H vmerge_vvm_i64m2
#define VMERGE1_INT64H vmerge_vxm_i64m2
#define VSTORE_INT64H vse64_v_i64m2

#define vfcvt_rtz_x_f_v_i64m4 vfcvt_x_f_v_i64m4
#define vfcvt_rtz_x_f_v_i64m2 vfcvt_x_f_v_i64m2

//BOOL64H
#define V_ELT_BOOL64H vbool32_t
#define VNOT_BOOL64H vmnot_m_b32
#define VCLEAR_BOOL64H vmclr_m_b32
#define VXOR_BOOL64H vmxor_mm_b32
#define VOR_BOOL64H vmor_mm_b32
#define VAND_BOOL64H vmand_mm_b32
#define VANDNOT_BOOL64H vmandn_mm_b32
#endif  // ELEN >= 64


#else
#warning "RVV intrinsics v1.0"

///////////////////// FULL VECTOR  m4 //////////////
#define VSETVL32 __riscv_vsetvl_e32m4
#define VSETVL16 __riscv_vsetvl_e16m4

//// FLOAT
#define V_ELT_FLOAT vfloat32m4_t
#define VLOAD_FLOAT __riscv_vle32_v_f32m4
#define VLOAD1_FLOAT __riscv_vfmv_v_f_f32m4
#define VSTORE_FLOAT __riscv_vse32_v_f32m4
#define VADD_FLOAT __riscv_vfadd_vv_f32m4
#define VADD1_FLOAT __riscv_vfadd_vf_f32m4
#define VSUB_FLOAT __riscv_vfsub_vv_f32m4
#define VSUB1_FLOAT __riscv_vfsub_vf_f32m4
#define VRSUB1_FLOAT __riscv_vfrsub_vf_f32m4  // v2 = f - v1
#define VMUL_FLOAT __riscv_vfmul_vv_f32m4
#define VMUL1_FLOAT __riscv_vfmul_vf_f32m4
#define VDIV_FLOAT __riscv_vfdiv_vv_f32m4
#define VDIV1_FLOAT __riscv_vfdiv_vf_f32m4
#define VRDIV1_FLOAT __riscv_vfrdiv_vf_f32m4
#define VFMACC_FLOAT __riscv_vfmacc_vv_f32m4  // vd[i] = +(vs1[i] * vs2[i]) + vd[i]
#define VFMACC1_FLOAT __riscv_vfmacc_vf_f32m4
#define VFMADD_FLOAT __riscv_vfmadd_vv_f32m4  // vd[i] = +(vs1[i] * vd[i]) + vs2[i]
#define VFMADD1_FLOAT __riscv_vfmadd_vf_f32m4
#define VFMSUB_FLOAT __riscv_vfmsub_vv_f32m4  // d = a*b - c
#define VREDSUM_FLOAT __riscv_vfredosum_vs_f32m4_f32m1_tu
/*static inline vfloat32m1_t VREDSUM_FLOAT(vfloat32m1_t dest,
vfloat32m4_t vector, vfloat32m1_t scalar, size_t vl){
	return  __riscv_vfredosum_vs_f32m4_f32m1(vector, scalar, vl);
}*/
#define VREDMAX_FLOAT __riscv_vfredmax_vs_f32m4_f32m1_tu
#define VREDMIN_FLOAT __riscv_vfredmin_vs_f32m4_f32m1_tu
#define VMIN_FLOAT __riscv_vfmin_vv_f32m4
#define VMAX_FLOAT __riscv_vfmax_vv_f32m4
#define VMIN1_FLOAT __riscv_vfmin_vf_f32m4
#define VMAX1_FLOAT __riscv_vfmax_vf_f32m4
#define VINTERP_FLOAT_INT __riscv_vreinterpret_v_f32m4_i32m4
#define VINTERP_INT_FLOAT __riscv_vreinterpret_v_i32m4_f32m4

#define VCVT_FLOAT_INT __riscv_vfcvt_x_f_v_i32m4
#ifdef NO_RTZ
#define VCVT_RTZ_FLOAT_INT VCVT_FLOAT_INT
#else
#define VCVT_RTZ_FLOAT_INT __riscv_vfcvt_rtz_x_f_v_i32m4
#endif

#define VCVT_INT_FLOAT __riscv_vfcvt_f_x_v_f32m4
static inline vfloat32m4_t VMERGE_FLOAT(vbool8_t mask, vfloat32m4_t op1,
vfloat32m4_t op2, size_t vl){
	return __riscv_vmerge_vvm_f32m4(op1, op2, mask, vl);
}
#define VMUL1_FLOAT_MASK __riscv_vfmul_vf_f32m4_m

#define VSQRT_FLOAT __riscv_vfsqrt_v_f32m4
#define VLE_FLOAT_STRIDE __riscv_vlse32_v_f32m4
#define VEQ1_FLOAT_BOOL __riscv_vmfeq_vf_f32m4_b8
#define VEQ_FLOAT_BOOL __riscv_vmfeq_vv_f32m4_b8
#define VGT1_FLOAT_BOOL __riscv_vmfgt_vf_f32m4_b8
#define VNE1_FLOAT_BOOL __riscv_vmfne_vf_f32m4_b8
#define VLT1_FLOAT_BOOL __riscv_vmflt_vf_f32m4_b8
#define VLE1_FLOAT_BOOL __riscv_vmfle_vf_f32m4_b8
#define VABS_FLOAT __riscv_vfabs_v_f32m4
static inline vfloat32m4_t VMERGE1_FLOAT(vbool8_t mask, vfloat32m4_t op1,
float op2, size_t vl){
	return __riscv_vfmerge_vfm_f32m4(op1, op2, mask, vl);
}
#define VGATHER_FLOAT __riscv_vrgather_vv_f32m4
#define vfmv_v_f_f32m1 __riscv_vfmv_v_f_f32m1

//// INT
#define V_ELT_INT vint32m4_t
#define VLOAD_INT __riscv_vle32_v_i32m4
#define VLOAD1_INT __riscv_vmv_v_x_i32m4
#define VSTORE_INT __riscv_vse32_v_i32m4
#define VADD_INT __riscv_vadd_vv_i32m4
#define VADD1_INT __riscv_vadd_vx_i32m4
#define VMUL_INT __riscv_vmul_vv_i32m4
#define VMUL1_INT __riscv_vmul_vx_i32m4
#define VSUB_INT __riscv_vsub_vv_i32m4
#define VSUB1_INT __riscv_vsub_vx_i32m4
#define VAND1_INT __riscv_vand_vx_i32m4
#define VAND_INT __riscv_vand_vv_i32m4
#define VXOR_INT __riscv_vxor_vv_i32m4
#define VSLL1_INT __riscv_vsll_vx_i32m4
#define VEQ1_INT_BOOL __riscv_vmseq_vx_i32m4_b8
#define VEQ_INT_BOOL __riscv_vmseq_vv_i32m4_b8
#define VGT1_INT_BOOL __riscv_vmsgt_vx_i32m4_b8
#define VNE1_INT_BOOL __riscv_vmsne_vx_i32m4_b8
#define VLT1_INT_BOOL __riscv_vmslt_vx_i32m4_b8
#define VLE1_INT_BOOL __riscv_vmsle_vx_i32m4_b8
#define VADD1_INT_MASK __riscv_vadd_vx_i32m4_m
#define VSUB1_INT_MASK __riscv_vsub_vx_i32m4_m
#define VSUB1_INT __riscv_vsub_vx_i32m4
#define VOR1_INT __riscv_vor_vx_i32m4
#define VSRA1_INT __riscv_vsra_vx_i32m4
#define VMIN_INT __riscv_vmin_vv_i32m4
#define VMIN1_INT __riscv_vmin_vx_i32m4
#define VMAX_INT __riscv_vmax_vv_i32m4
#define VMAX1_INT __riscv_vmax_vx_i32m4
static inline vint32m4_t VMERGE_INT(vbool8_t mask, vint32m4_t op1,
vint32m4_t op2, size_t vl){
	return __riscv_vmerge_vvm_i32m4(op1, op2, mask, vl);
}

static inline vint32m4_t VMERGE1_INT(vbool8_t mask, vint32m4_t op1,
int32_t op2, size_t vl){
	return __riscv_vmerge_vxm_i32m4(op1, op2, mask, vl);
}
#define VNEG_INT __riscv_vneg_v_i32m4
#define VREDSUM_INT vredsum_vs_i32m4_i32m1_tu
/*static inline vint32m1_t VREDSUM_INT(vint32m1_t dest,
vint32m4_t vector, vint32m1_t scalar, size_t vl){
	return  __riscv_vredsum_vs_i32m4_i32m1(vector, scalar, vl);
}*/
#define VREDMAX_INT __riscv_vredmax_vs_i32m4_i32m1_tu
#define VREDMIN_INT __riscv_vredmin_vs_i32m4_i32m1_tu
#define VGATHER_INT __riscv_vrgather_vv_i32m4
#define VNOT_INT __riscv_vnot_v_i32m4

//// UINT
#define VLOAD_UINT __riscv_vle32_v_u32m4
#define VSTORE_UINT __riscv_vse32_v_u32m4
#define V_ELT_UINT vuint32m4_t
#define VCVT_FLOAT_UINT __riscv_vfcvt_xu_f_v_u32m4

//// SHORT
#define V_ELT_SHORT vint16m4_t
#define VLOAD_SHORT __riscv_vle16_v_i16m4
#define VLOAD1_SHORT __riscv_vmv_v_x_i16m4
#define VSTORE_SHORT __riscv_vse16_v_i16m4
#define VADD_SHORT __riscv_vadd_vv_i16m4
#define VSUB_SHORT __riscv_vsub_vv_i16m4
#define VREDSUMW_SHORT __riscv_vwredsum_vs_i16m4_i32m1_tu
#define VGT_SHORT_BOOL __riscv_vmsgt_vv_i16m4_b4
static inline vint16m4_t VMERGE_SHORT(vbool4_t mask, vint16m4_t op1,
vint16m4_t op2, size_t vl){
	return __riscv_vmerge_vvm_i16m4(op1, op2, mask, vl);
}

//// BOOL for 16 bits elements
#define V_ELT_BOOL16 vbool4_t

//// BOOL for 32 bits elements
#define V_ELT_BOOL32 vbool8_t
#define VNOT_BOOL __riscv_vmnot_m_b8
#define VCLEAR_BOOL __riscv_vmclr_m_b8
#define VXOR_BOOL __riscv_vmxor_mm_b8
#define VOR_BOOL __riscv_vmor_mm_b8
#define VAND_BOOL __riscv_vmand_mm_b8
#define VANDNOT_BOOL __riscv_vmandn_mm_b8

/////////////////////////// HALF VECTOR, m2 ///////////////
#define VSETVL32H __riscv_vsetvl_e32m2
#define VSETVL16H __riscv_vsetvl_e16m2

//// FLOATH
#define V_ELT_FLOATH vfloat32m2_t
#define VLOAD_FLOATH __riscv_vle32_v_f32m2
#define VLOAD1_FLOATH __riscv_vfmv_v_f_f32m2
static inline void VLOAD_FLOATH2(vfloat32m2_t *v0, vfloat32m2_t *v1, float* base, size_t vl){
	vfloat32m2x2_t v = __riscv_vlseg2e32_v_f32m2x2(base, vl);
	*v0 = __riscv_vget_v_f32m2x2_f32m2(v, 0);
	*v1 = __riscv_vget_v_f32m2x2_f32m2(v, 1);
}
static inline void VSTORE_FLOATH2(float* base, vfloat32m2_t v0, vfloat32m2_t v1, size_t vl){
	vfloat32m2x2_t v = __riscv_vcreate_v_f32m2x2(v0, v1);
	__riscv_vsseg2e32_v_f32m2x2(base, v, vl);
}
#define VSTORE_FLOATHH __riscv_vse32_v_f32m1
#define VLOAD_FLOATHH __riscv_vle32_v_f32m1
#define VLOAD_FLOATH_STRIDE __riscv_vlse32_v_f32m2
#define VSTORE_FLOATH __riscv_vse32_v_f32m2
#define VINTERP_FLOATH_INTH __riscv_vreinterpret_v_f32m2_i32m2
#define VINTERP_INTH_FLOATH __riscv_vreinterpret_v_i32m2_f32m2
#define VXOR1_INTH __riscv_vxor_vx_i32m2
#define VADD_FLOATH __riscv_vfadd_vv_f32m2
#define VADD1_FLOATH __riscv_vfadd_vf_f32m2
#define VADD1_FLOATH_MASK __riscv_vfadd_vf_f32m2_m
#define VSUB_FLOATH __riscv_vfsub_vv_f32m2
#define VSUB1_FLOATH __riscv_vfsub_vf_f32m2    // v2 = v1 - f
#define VRSUB1_FLOATH __riscv_vfrsub_vf_f32m2  // v2 = f - v1
#define VMUL_FLOATH __riscv_vfmul_vv_f32m2
#define VMUL1_FLOATH __riscv_vfmul_vf_f32m2
#define VMUL1_FLOATH_MASK __riscv_vfmul_vf_f32m2_m
#define VDIV_FLOATH __riscv_vfdiv_vv_f32m2
#define VDIV1_FLOATH __riscv_vfdiv_vf_f32m2
#define VRDIV1_FLOATH __riscv_vfrdiv_vf_f32m2
#define VFMACC_FLOATH __riscv_vfmacc_vv_f32m2  // d = a + b*c
#define VFMACC1_FLOATH __riscv_vfmacc_vf_f32m2
#define VFMADD_FLOATH __riscv_vfmadd_vv_f32m2  // vd[i] = +(vs1[i] * vd[i]) + vs2[i]
#define VFMADD1_FLOATH __riscv_vfmadd_vf_f32m2
#define VFMSUB_FLOATH __riscv_vfmsub_vv_f32m2  // d = a*b - c
#define VREDSUM_FLOATH __riscv_vfredosum_vs_f32m2_f32m1_tu
/*static inline vfloat32m1_t VREDSUM_FLOATH(vfloat32m1_t dest,
vfloat32m2_t vector, vfloat32m1_t scalar, size_t vl){
	return  __riscv_vfredosum_vs_f32m2_f32m1(vector, scalar, vl);
}*/
#define VREDMAX_FLOATH __riscv_vfredmax_vs_f32m2_f32m1_tu
#define VREDMIN_FLOATH __riscv_vfredmin_vs_f32m2_f32m1_tu
#define VMIN_FLOATH __riscv_vfmin_vv_f32m2
#define VMIN1_FLOATH __riscv_vfmin_vf_f32m2
#define VMAX_FLOATH __riscv_vfmax_vv_f32m2
#define VMAX1_FLOATH __riscv_vfmax_vf_f32m2
#define VINTERP_FLOATH_INTH __riscv_vreinterpret_v_f32m2_i32m2
#define VINTERP_INTH_FLOATH __riscv_vreinterpret_v_i32m2_f32m2
#define VCVT_FLOATH_INTH __riscv_vfcvt_x_f_v_i32m2
#ifdef NO_RTZ
#define VCVT_RTZ_FLOATH_INTH VCVT_FLOATH_INTH
#else
#define VCVT_RTZ_FLOATH_INTH __riscv_vfcvt_rtz_x_f_v_i32m2	
#endif

#define VCVT_INTH_FLOATH __riscv_vfcvt_f_x_v_f32m2
static inline vfloat32m2_t VMERGE_FLOATH(vbool16_t mask, vfloat32m2_t op1,
vfloat32m2_t op2, size_t vl){
 return __riscv_vmerge_vvm_f32m2(op1, op2, mask, vl);
}
#define VSQRT_FLOATH __riscv_vfsqrt_v_f32m2
#define VEQ1_FLOATH_BOOLH __riscv_vmfeq_vf_f32m2_b16
#define VEQ_FLOATH_BOOLH __riscv_vmfeq_vv_f32m2_b16
#define VGE1_FLOATH_BOOLH __riscv_vmfge_vf_f32m2_b16
#define VGT1_FLOATH_BOOLH __riscv_vmfgt_vf_f32m2_b16
#define VNE1_FLOATH_BOOLH __riscv_vmfne_vf_f32m2_b16
#define VLT1_FLOATH_BOOLH __riscv_vmflt_vf_f32m2_b16
#define VLE1_FLOATH_BOOLH __riscv_vmfle_vf_f32m2_b16
#define VABS_FLOATH __riscv_vfabs_v_f32m2
static inline vfloat32m2_t VMERGE1_FLOATH(vbool16_t mask, vfloat32m2_t op1,
float op2, size_t vl){
	return __riscv_vfmerge_vfm_f32m2(op1, op2, mask, vl);
}
#define VGATHER_FLOATH __riscv_vrgather_vv_f32m2

//// INTH
#define V_ELT_INTH vint32m2_t
#define VSTORE_INTHH __riscv_vse32_v_i32m1
#define VLOAD_INTHH __riscv_vle32_v_i32m1
#define VLOAD_INTH __riscv_vle32_v_i32m2
#define VLOAD1_INTH __riscv_vmv_v_x_i32m2
#define VLOAD1_INTHH __riscv_vmv_v_x_i32m1
#define VSTORE_INTH __riscv_vse32_v_i32m2
#define VADD_INTH __riscv_vadd_vv_i32m2
#define VADD1_INTH __riscv_vadd_vx_i32m2
#define VADD1_INTH_MASK __riscv_vadd_vx_i32m2_m
static inline vint32m2_t VADD1_INTH_MASKEDOFF(vbool16_t mask, vint32m2_t op1, vint32m2_t maskedoff,
int32_t op2, size_t vl){
	op1 = __riscv_vmerge_vvm_i32m2(maskedoff, op1, mask, vl);
	return __riscv_vadd_vx_i32m2_m(mask, op1, op2, vl);
}	
static inline vint64m2_t VADD1_INT64H_MASKEDOFF(vbool32_t mask, vint64m2_t op1, vint64m2_t maskedoff,
int64_t op2, size_t vl){
	op1 = __riscv_vmerge_vvm_i64m2(maskedoff, op1, mask, vl);
	return __riscv_vadd_vx_i64m2_m(mask, op1, op2, vl);
}	
#define VMUL_INTH __riscv_vmul_vv_i32m2
#define VMUL1_INTH __riscv_vmul_vx_i32m2
#define VSUB_INTH __riscv_vsub_vv_i32m2
#define VSUB1_INTH __riscv_vsub_vx_i32m2
#define VSUB1_INTH_MASK __riscv_vsub_vx_i32m2_m
#define VAND1_INTH __riscv_vand_vx_i32m2
#define VAND_INTH __riscv_vand_vv_i32m2
#define VXOR_INTH __riscv_vxor_vv_i32m2
#define VSLL1_INTH __riscv_vsll_vx_i32m2
#define VEQ1_INTH_BOOLH __riscv_vmseq_vx_i32m2_b16
#define VGT1_INTH_BOOLH __riscv_vmsgt_vx_i32m2_b16
#define VNE1_INTH_BOOLH __riscv_vmsne_vx_i32m2_b16
#define VLT1_INTH_BOOLH __riscv_vmflt_vf_f32m2_b16
#define VLE1_INTH_BOOLH __riscv_vmsle_vx_i32m2_b16
#define VEQ_INTH_BOOLH __riscv_vmseq_vv_i32m2_b16
#define VOR1_INTH __riscv_vor_vx_i32m2
#define VSRA1_INTH __riscv_vsra_vx_i32m2
#define VMIN_INTH __riscv_vmin_vv_i32m2
#define VMIN1_INTH __riscv_vmin_vx_i32m2
#define VMAX_INTH __riscv_vmax_vv_i32m2
#define VMAX1_INTH __riscv_vmax_vx_i32m2
#define VNOT_INTH __riscv_vnot_v_i32m2

static inline vint32m2_t VMERGE_INTH(vbool16_t mask, vint32m2_t op1,
vint32m2_t op2, size_t vl){
	return __riscv_vmerge_vvm_i32m2(op1, op2, mask, vl);
}

//// UINTH
#define VLOAD_UINTH __riscv_vle32_v_u32m2
#define V_ELT_UINTH __riscv_vuint32m2_t
#define VCVT_FLOATH_UINTH __riscv_vfcvt_xu_f_v_u32m2

//// SHORTH
#define V_ELT_SHORTH vint16m2_t
#define VLOAD_SHORTH __riscv_vle16_v_i16m2
#define VLOAD1_SHORTH __riscv_vmv_v_x_i16m2
#define VSTORE_SHORTH __riscv_vse16_v_i16m2
#define VADD_SHORTH __riscv_vadd_vv_i16m2
#define VREDSUMW_SHORTH __riscv_vwredsum_vs_i16m4_i32m1_tu
static inline vint16m2_t VCVT_INT_SHORTH (vint32m4_t op1, size_t shift, size_t vl){
	return __riscv_vnclip_wx_i16m2(op1, shift, __RISCV_VXRM_RNU, vl);
}

#if __riscv_v != 7000
#define VCVT_SHORTH_INT __riscv_vsext_vf2_i32m4
#else
#define VCVT_SHORTH_INT(a, b)  __riscv_vwmul_vx_i32m4((a), 1, (b))
#endif


//// USHORTH
#define V_ELT_USHORTH vuint16m2_t
#define VLOAD_USHORTH __riscv_vle16_v_u16m2
#define VSTORE_USHORTH __riscv_vse16_v_u16m2
static inline vuint16m2_t VCVT_UINT_USHORTH (vuint32m4_t op1, size_t shift, size_t vl){
	return __riscv_vnclipu_wx_u16m2(op1, shift, __RISCV_VXRM_RNU, vl);
}
//// UBYTEHH
#define V_ELT_UBYTEHH vuint8m1_t
#define VLOAD_UBYTEHH __riscv_vle8_v_u8m1
#define VSTORE_UBYTEHH __riscv_vse8_v_u8m1
static inline vuint8m1_t VCVT_USHORTH_UBYTEHH (vuint16m2_t op1, size_t shift, size_t vl){
	return __riscv_vnclipu_wx_u8m1(op1, shift, __RISCV_VXRM_RNU, vl);
}
//// BOOL for Half length __riscv_vector 32 bits elements
#define V_ELT_BOOL32H vbool16_t
#define VNOT_BOOLH __riscv_vmnot_m_b16
#define VCLEAR_BOOLH __riscv_vmclr_m_b16
#define VXOR_BOOLH __riscv_vmxor_mm_b16
#define VOR_BOOLH __riscv_vmor_mm_b16
#define VAND_BOOLH __riscv_vmand_mm_b16
#define VANDNOT_BOOLH __riscv_vmandn_mm_b16
//#define VANDNOT_BOOLH __riscv_vmnand_mm_b16

#if ELEN >= 64
#define VSETVL64 __riscv_vsetvl_e64m4
#define VSETVL64H __riscv_vsetvl_e64m2

//// DOUBLE
#define V_ELT_DOUBLE vfloat64m4_t
#define VLOAD_DOUBLE __riscv_vle64_v_f64m4
#define VLOAD1_DOUBLE __riscv_vfmv_v_f_f64m4
#define VSTORE_DOUBLE __riscv_vse64_v_f64m4
#define VADD_DOUBLE __riscv_vfadd_vv_f64m4
#define VADD1_DOUBLE __riscv_vfadd_vf_f64m4
#define VSUB_DOUBLE __riscv_vfsub_vv_f64m4
#define VSUB1_DOUBLE __riscv_vfsub_vf_f64m4
#define VMUL_DOUBLE __riscv_vfmul_vv_f64m4
#define VMUL1_DOUBLE __riscv_vfmul_vf_f64m4
#define VDIV_DOUBLE __riscv_vfdiv_vv_f64m4
#define VFMA_DOUBLE __riscv_vfmacc_vv_f64m4  // d = a + b*c
#define VFMA1_DOUBLE __riscv_vfmacc_vf_f64m4
#define VFMSUB_DOUBLE __riscv_vfmsub_vv_f64m4  // d = a*b - c
#define VREDSUM_DOUBLE __riscv_vfredosum_vs_f64m4_f64m1_tu
#define VREDMAX_DOUBLE __riscv_vfredmax_vs_f64m4_f64m1_tu
#define VREDMIN_DOUBLE __riscv_vfredmin_vs_f64m4_f64m1_tu
#define VMIN_DOUBLE __riscv_vfmin_vv_f64m4
#define VMAX_DOUBLE __riscv_vfmax_vv_f64m4
#define VMIN1_DOUBLE __riscv_vfmin_vf_f64m4
#define VMAX1_DOUBLE __riscv_vfmax_vf_f64m4
#define VINTERP_DOUBLE_INT __riscv_vreinterpret_v_f64m4_i64m4
#define VINTERP_INT_DOUBLE __riscv_vreinterpret_v_i64m4_f64m4
#define VCVT_DOUBLE_INT __riscv_vfcvt_x_f_v_i64m4

#ifdef NO_RTZ
#define VCVT_RTZ_DOUBLE_INT VCVT_DOUBLE_INT
#else
#define VCVT_RTZ_DOUBLE_INT __riscv_vfcvt_rtz_x_f_v_i64m4
#endif

#define VCVT_INT_DOUBLE __riscv_vfcvt_f_x_v_f64m4
#define VABS_DOUBLE __riscv_vfabs_v_f64m4
#define VSQRT_DOUBLE __riscv_vfsqrt_v_f64m4
#define VCVT_DOUBLE_FLOAT __riscv_vfncvt_f_f_w_f32m2
#define VCVT_FLOAT_DOUBLE __riscv_vfwcvt_f_f_v_f64m4

//// DOUBLEH
#define V_ELT_DOUBLEH vfloat64m2_t
static inline void VLOAD_DOUBLEH2(vfloat64m2_t *v0, vfloat64m2_t *v1, double* base, size_t vl){
	vfloat64m2x2_t v = __riscv_vlseg2e64_v_f64m2x2(base, vl);
	*v0 = __riscv_vget_v_f64m2x2_f64m2(v, 0);
	*v1 = __riscv_vget_v_f64m2x2_f64m2(v, 1);
}
static inline void VSTORE_DOUBLEH2(double* base, vfloat64m2_t v0, vfloat64m2_t v1, size_t vl){
	vfloat64m2x2_t v = __riscv_vcreate_v_f64m2x2(v0, v1);
	__riscv_vsseg2e64_v_f64m2x2(base, v, vl);
}
#define VLOAD_DOUBLEH_STRIDE __riscv_vlse64_v_f64m2x2
#define VLOAD_DOUBLEH __riscv_vle64_v_f64m2
#define VLOAD1_DOUBLEH __riscv_vfmv_v_f_f64m2
#define VSTORE_DOUBLEH __riscv_vse64_v_f64m2
#define VADD_DOUBLEH __riscv_vfadd_vv_f64m2
#define VADD1_DOUBLEH __riscv_vfadd_vf_f64m2
#define VSUB_DOUBLEH __riscv_vfsub_vv_f64m2
#define VSUB1_DOUBLEH __riscv_vfsub_vf_f64m2
#define VRSUB1_DOUBLEH __riscv_vfrsub_vf_f64m2  // v2 = f - v1
#define VMUL_DOUBLEH __riscv_vfmul_vv_f64m2
#define VMUL1_DOUBLEH __riscv_vfmul_vf_f64m2
#define VDIV_DOUBLEH __riscv_vfdiv_vv_f64m2
#define VFMACC_DOUBLEH __riscv_vfmacc_vv_f64m2  // d = a + b*c
#define VFMACC1_DOUBLEH __riscv_vfmacc_vf_f64m2  
#define VFMADD_DOUBLEH __riscv_vfmadd_vv_f64m2  
#define VFMADD1_DOUBLEH __riscv_vfmadd_vf_f64m2 
#define VFMA1_DOUBLEH __riscv_vfmacc_vf_f64m2
#define VFMSUB_DOUBLEH __riscv_vfmsub_vv_f64m2  // d = a*b - c
#define VREDSUM_DOUBLEH __riscv_vfredosum_vs_f64m2_f64m1_tu
#define VREDMAX_DOUBLEH __riscv_vfredmax_vs_f64m2_f64m1_tu
#define VREDMIN_DOUBLEH __riscv_vfredmin_vs_f64m2_f64m1_tu
#define VMIN_DOUBLEH __riscv_vfmin_vv_f64m2
#define VMAX_DOUBLEH __riscv_vfmax_vv_f64m2
#define VMIN1_DOUBLEH __riscv_vfmin_vf_f64m2
#define VMAX1_DOUBLEH __riscv_vfmax_vf_f64m2
#define VINTERP_DOUBLEH_INTH __riscv_vreinterpret_v_f64m2_i64m2
#define VINTERP_INTH_DOUBLEH __riscv_vreinterpret_v_i64m2_f64m2

#define VCVT_DOUBLEH_INTH __riscv_vfcvt_x_f_v_i64m2
#ifdef NO_RTZ
#define VCVT_RTZ_DOUBLEH_INTH VCVT_DOUBLEH_INTH
#else
#define VCVT_RTZ_DOUBLEH_INTH __riscv_vfcvt_rtz_x_f_v_i64m2
#endif

#define VCVT_INTH_DOUBLEH __riscv_vfcvt_f_x_v_f64m2
#define VABS_DOUBLEH __riscv_vfabs_v_f64m2
#define VSQRT_DOUBLEH __riscv_vfsqrt_v_f64m2
#define VCVT_DOUBLEH_FLOATH __riscv_vfncvt_f_f_w_f32m2
#define VCVT_FLOATH_DOUBLEH __riscv_vfwcvt_f_f_v_f64m2
#define VLT1_DOUBLEH_BOOLH __riscv_vmflt_vf_f64m2_b32
static inline vfloat64m2_t VMERGE_DOUBLEH(vbool32_t mask, vfloat64m2_t op1,
vfloat64m2_t op2, size_t vl){
 return __riscv_vmerge_vvm_f64m2(op1, op2, mask, vl);
}

static inline vfloat64m2_t VMERGE1_DOUBLEH(vbool32_t mask, vfloat64m2_t op1,
double op2, size_t vl){
	return __riscv_vfmerge_vfm_f64m2(op1, op2, mask, vl);
}
#define VMUL1_DOUBLEH_MASK __riscv_vfmul_vf_f64m2_m
#define VEQ1_DOUBLEH_BOOLH __riscv_vmfeq_vf_f64m2_b32
#define VEQ_DOUBLEH_BOOLH __riscv_vmfeq_vv_f64m2_b32
#define VGE1_DOUBLEH_BOOLH __riscv_vmfge_vf_f64m2_b32
#define VGT1_DOUBLEH_BOOLH __riscv_vmfgt_vf_f64m2_b32
#define VNE1_DOUBLEH_BOOLH __riscv_vmfne_vf_f64m2_b32
#define VLT1_DOUBLEH_BOOLH __riscv_vmflt_vf_f64m2_b32
#define VLE1_DOUBLEH_BOOLH __riscv_vmfle_vf_f64m2_b32
#define VADD1_DOUBLEH_MASK __riscv_vfadd_vf_f64m2_m
#define VMUL1_DOUBLEH_MASK __riscv_vfmul_vf_f64m2_m
static inline vfloat64m2_t VADD1_DOUBLEH_MASKEDOFF(vbool32_t mask, vfloat64m2_t op1, vfloat64m2_t maskedoff,
double op2, size_t vl){
	op1 = __riscv_vmerge_vvm_f64m2(maskedoff, op1, mask, vl);
	return __riscv_vfadd_vf_f64m2_m(mask, op1, op2, vl);
}	

// INT64H
#define V_ELT_INT64H vint64m2_t
#define VLOAD1_INT64H __riscv_vmv_v_x_i64m2
#define VSLL1_INT64H __riscv_vsll_vx_i64m2
#define VADD1_INT64H __riscv_vadd_vx_i64m2
#define VAND1_INT64H __riscv_vand_vx_i64m2
#define VXOR_INT64H __riscv_vxor_vv_i64m2
#define VNE1_INTH_BOOL64H __riscv_vmsne_vx_i64m2_b32
#define VEQ1_INTH_BOOL64H __riscv_vmseq_vx_i64m2_b32
#define VEQ1_INTH_BOOL64H __riscv_vmseq_vx_i64m2_b32
#define VGT1_INTH_BOOL64H __riscv_vmsgt_vx_i64m2_b32
#define VSUB1_INT64H_MASK __riscv_vsub_vx_i64m2_m
static inline vint64m2_t VMERGE_INT64H(vbool32_t mask, vint64m2_t op1,
vint64m2_t op2, size_t vl){
	return __riscv_vmerge_vvm_i64m2(op1, op2, mask, vl);
}

static inline vint64m2_t VMERGE1_INT64H(vbool32_t mask, vint64m2_t op1,
int64_t op2, size_t vl){
	return __riscv_vmerge_vxm_i64m2(op1, op2, mask, vl);
}
#define VADD1_INT64H_MASK __riscv_vadd_vx_i64m2_m

#define VSTORE_INT64H __riscv_vse64_v_i64m2

//BOOL64H
#define V_ELT_BOOL64H vbool32_t
#define VNOT_BOOL64H __riscv_vmnot_m_b32
#define VCLEAR_BOOL64H __riscv_vmclr_m_b32
#define VXOR_BOOL64H __riscv_vmxor_mm_b32
#define VOR_BOOL64H __riscv_vmor_mm_b32
#define VAND_BOOL64H __riscv_vmand_mm_b32
#define VANDNOT_BOOL64H __riscv_vmandn_mm_b32
#endif  // ELEN >= 64

#endif

#endif  // RISCV

#ifdef ALTIVEC
#include <altivec.h>
#endif

#ifdef _MSC_VER /* visual c++ */
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#define ALIGN32_BEG
#define ALIGN32_END __declspec(align(32))
#define ALIGN64_BEG
#define ALIGN64_END __declspec(align(64))
#else /* gcc,icc, clang */
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))
#define ALIGN64_BEG
#define ALIGN64_END __attribute__((aligned(64)))
#endif


static const float FOPI = 1.27323954473516f;
static const float PIO4F = 0.7853981633974483096f;
static const double FOPId = 1.2732395447351626861510701069801148;

#define PIF 3.14159265358979323846f      // PI
#define mPIF -3.14159265358979323846f    // -PI
#define PIO2F 1.57079632679489661923f    // PI/2 1.570796326794896619
#define mPIO2F -1.57079632679489661923f  // -PI/2 1.570796326794896619

/* Note, these constants are for a 32-bit significand: */
/*
static const float DP1 = 0.7853851318359375f;
static const float DP2 = 1.30315311253070831298828125e-5f;
static const float DP3 = 3.03855025325309630e-11f;
static const float lossth = 65536.f;
*/

/* These are for a 24-bit significand: */
static const float minus_cephes_DP1 = -0.78515625f;
static const float minus_cephes_DP2 = -2.4187564849853515625e-4f;
static const float minus_cephes_DP3 = -3.77489497744594108e-8f;
static const float lossth = 8192.;

static const float T24M1 = 16777215.f;

static const float sincof[] = {-1.9515295891E-4f, 8.3321608736E-3f, -1.6666654611E-1f};
static const float coscof[] = {2.443315711809948E-5f, -1.388731625493765E-3f,
                               4.166664568298827E-2f};

static const double minus_cephes_DP1d = -7.85398125648498535156E-1;
static const double minus_cephes_DP2d = -3.77489470793079817668E-8;
static const double minus_cephes_DP3d = -2.69515142907905952645E-15;
static const double sincod[] = {1.58962301576546568060E-10, -2.50507477628578072866E-8,
2.75573136213857245213E-6, -1.98412698295895385996E-4, 8.33333333332211858878E-3, -1.66666666666666307295E-1};
static const double coscod[] = {-1.13585365213876817300E-11, 2.08757008419747316778E-9, -2.75573141792967388112E-7, 2.48015872888517045348E-5, -1.38888888888730564116E-3, 4.16666666666665929218E-2};

static const double ASIN_P0d =  4.253011369004428248960E-3;
static const double ASIN_P1d =  -6.019598008014123785661E-1;
static const double ASIN_P2d =  5.444622390564711410273E0;
static const double ASIN_P3d =  -1.626247967210700244449E1;
static const double ASIN_P4d =  1.956261983317594739197E1;
static const double ASIN_P5d =  -8.198089802484824371615E0;

static const double PId =  3.1415926535897932384626433832795028841971693993751058209749445923;      // PI
static const double mPId =  -3.1415926535897932384626433832795028841971693993751058209749445923;    // -PI
static const double ASIN_Q0d =  -1.474091372988853791896E1;
static const double ASIN_Q1d =  7.049610280856842141659E1;
static const double ASIN_Q2d =  -1.471791292232726029859E2;
static const double ASIN_Q3d =  1.395105614657485689735E2;
static const double ASIN_Q4d =  -4.918853881490881290097E1;

static const double ASIN_R0d =  2.967721961301243206100E-3;
static const double ASIN_R1d =  -5.634242780008963776856E-1;
static const double ASIN_R2d =  6.968710824104713396794E0;
static const double ASIN_R3d =  -2.556901049652824852289E1;
static const double ASIN_R4d =  2.853665548261061424989E1;

static const double ASIN_S0d =  -2.194779531642920639778E1;
static const double ASIN_S1d =  1.470656354026814941758E2;
static const double ASIN_S2d =  -3.838770957603691357202E2;
static const double ASIN_S3d =  3.424398657913078477438E2;

static const double PIO2d =  1.5707963267948966192313216916397514420985846996875529104874722961;    /* pi/2 */
static const double PIO4d =  0.7853981633974483096156608458198757210492923498437764552437361480; /* pi/4 */

static const double minMOREBITSd =  -6.123233995736765886130E-17;
static const double MOREBITSd =  6.123233995736765886130E-17;
//static const double 0p5xMOREBITSd =  3.061616997868382943065e-17;

static const double ATAN_P0d =  -8.750608600031904122785E-1;
static const double ATAN_P1d =  -1.615753718733365076637E1;
static const double ATAN_P2d =  -7.500855792314704667340E1;
static const double ATAN_P3d =  -1.228866684490136173410E2;
static const double ATAN_P4d =  -6.485021904942025371773E1;

static const double ATAN_Q0d =  2.485846490142306297962E1;
static const double ATAN_Q1d =  1.650270098316988542046E2;
static const double ATAN_Q2d =  4.328810604912902668951E2;
static const double ATAN_Q3d =  4.853903996359136964868E2;
static const double ATAN_Q4d =  1.945506571482613964425E2;

static const double TAN3PI8d =  2.41421356237309504880; /* 3*pi/8 */
static const double min1d =  -1.0;
//static const double 1m14d =  1.0e-14;
static const double TAN_P0d =  -1.30936939181383777646E4;
static const double TAN_P1d =  1.15351664838587416140E6;
static const double TAN_P2d =  -1.79565251976484877988E7;
static const double TAN_Q0d =  1.36812963470692954678E4;
static const double TAN_Q1d =  -1.32089234440210967447E6;
static const double TAN_Q2d =  2.50083801823357915839E7;
static const double TAN_Q3d =  -5.38695755929454629881E7;
static const double TAN_mDP1d =  -7.853981554508209228515625E-1;
static const double TAN_mDP2d =  -7.94662735614792836714E-9;
static const double TAN_mDP3d =  -3.06161699786838294307E-17;
static const double tanlossthd =  1.073741824e9;

static const double cephes_LOG2Ed =  1.4426950408889634073599;  /* 1/log(2) */
static const double cephes_LOGE2d =  6.93147180559945309417E-1; /* log(2) */

static const double cephes_exp_p0d =  1.26177193074810590878e-4;
static const double cephes_exp_p1d =  3.02994407707441961300e-2;
static const double cephes_exp_p2d =  9.99999999999999999910e-1;

static const double cephes_exp_q0d =  3.00198505138664455042e-6;
static const double cephes_exp_q1d =  2.52448340349684104192e-3;
static const double cephes_exp_q2d =  2.27265548208155028766e-1;
static const double cephes_exp_q3d =  2.00000000000000000009e0;

static const double cephes_exp_minC1d =  -0.693145751953125;
static const double cephes_exp_minC2d =  -1.42860682030941723212e-6;


#define SIGN_MASK 0x80000000
#define SIGN_MASKD 0x8000000000000000L
static const int32_t sign_mask = SIGN_MASK;
static const int32_t inv_sign_mask = ~SIGN_MASK;
static const int64_t sign_maskd = SIGN_MASKD;
static const int64_t inv_sign_maskd = ~SIGN_MASKD;

#define neg_sign_mask ~0x7FFFFFFF
#define min_norm_pos 0x00800000
#define INVLN10 0.4342944819032518f  // 0.4342944819f
#define INVLN2 1.4426950408889634f   // 1.44269504089f
#define LN2 0.6931471805599453094172321214581765680755001343602552541206800094f
#define LN2_DIV_LN10 0.3010299956639811952137388947244930267681898814621085413104274611f
#define IMM8_FLIP_VEC 0x1B              // change m128 from abcd to dcba
#define IMM8_LO_HI_VEC 0x1E             // change m128 from abcd to cdab
#define IMM8_PERMUTE_128BITS_LANES 0x1  // reverse abcd efgh to efgh abcd

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

typedef struct {
	int16_t re;
	int16_t im;
} complex16s_t;

typedef struct {
	int32_t re;
	int32_t im;
} complex32s_t;

typedef struct {
	float re;
	float im;
} complex32_t;

typedef struct {
	double re;
	double im;
} complex64_t;

typedef enum {
    RndZero,
    RndNear,
    RndFinancial,
} FloatRoundingMode;

typedef  struct {
	int16_t x;
	int16_t y;
} point_t;

typedef struct {
	point_t* modified_points;
	int counter;
} modified_t;

#ifndef max
#define max(a,b) ((a) > (b))? (a):(b)
#endif

#ifndef min
#define min(a,b) ((a) < (b))? (a):(b)
#endif


#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524f
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 -1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 -1.2420140846E-1
#define c_cephes_log_p4 +1.4249322787E-1
#define c_cephes_log_p5 -1.6668057665E-1
#define c_cephes_log_p6 +2.0000714765E-1
#define c_cephes_log_p7 -2.4999993993E-1
#define c_cephes_log_p8 +3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

#define c_cephes_L102A 3.0078125E-1f
#define c_cephes_L102B 2.48745663981195213739E-4f
#define c_cephes_L10EA 4.3359375E-1f
#define c_cephes_L10EB 7.00731903251827651129E-4f

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341f
#define c_cephes_LOG2EA 0.44269504088896340735992f
#define c_cephes_exp_C1 0.693359375f
#define c_cephes_exp_C2 -2.12194440e-4f

#define c_cephes_exp_p0 1.9875691500E-4f
#define c_cephes_exp_p1 1.3981999507E-3f
#define c_cephes_exp_p2 8.3334519073E-3f
#define c_cephes_exp_p3 4.1665795894E-2f
#define c_cephes_exp_p4 1.6666665459E-1f
#define c_cephes_exp_p5 5.0000001201E-1f

// TODO redundant with previous static float definitions, one should be removed
#define c_minus_cephes_DP1 -0.78515625f
#define c_minus_cephes_DP2 -2.4187564849853515625e-4f
#define c_minus_cephes_DP3 -3.77489497744594108e-8f

#define c_sincof_p0 -1.9515295891E-4f
#define c_sincof_p1 8.3321608736E-3f
#define c_sincof_p2 -1.6666654611E-1f
#define c_coscof_p0 2.443315711809948E-005f
#define c_coscof_p1 -1.388731625493765E-003f
#define c_coscof_p2 4.166664568298827E-002f
#define c_cephes_FOPI 1.27323954473516f  // 4 / M_PI

#define ATAN_P0 8.05374449538e-2f
#define ATAN_P1 -1.38776856032E-1f
#define ATAN_P2 1.99777106478E-1f
#define ATAN_P3 -3.33329491539E-1f

#define TAN_P0 9.38540185543E-3f
#define TAN_P1 3.11992232697E-3f
#define TAN_P2 2.44301354525E-2f
#define TAN_P3 5.34112807005E-2f
#define TAN_P4 1.33387994085E-1f
#define TAN_P5 3.33331568548E-1f

#define ASIN_P0 4.2163199048E-2f
#define ASIN_P1 2.4181311049E-2f
#define ASIN_P2 4.5470025998E-2f
#define ASIN_P3 7.4953002686E-2f
#define ASIN_P4 1.6666752422E-1f

#define TANH_P0 -5.70498872745E-3f
#define TANH_P1 2.06390887954E-2f
#define TANH_P2 -5.37397155531E-2f
#define TANH_P3 1.33314422036E-1f
#define TANH_P4 -3.33332819422E-1f

#define SINH_P0 2.03721912945E-4f
#define SINH_P1 8.33028376239E-3f
#define SINH_P2 1.66667160211E-1f

#define ATANH_P0 1.81740078349E-1f
#define ATANH_P1 8.24370301058E-2f
#define ATANH_P2 1.46691431730E-1f
#define ATANH_P3 1.99782164500E-1f
#define ATANH_P4 3.33337300303E-1f

#define LOGE2F 0.693147180559945309f
#define ASINH_P0 2.0122003309E-2f
#define ASINH_P1 -4.2699340972E-2f
#define ASINH_P2 7.4847586088E-2f
#define ASINH_P3 -1.6666288134E-1f

#define ACOSH_P0 1.7596881071E-3f
#define ACOSH_P1 -7.5272886713E-3f
#define ACOSH_P2 2.6454905019E-2f
#define ACOSH_P3 -1.1784741703E-1f
#define ACOSH_P4 1.4142135263E0f

#define cephes_CBRT2 1.25992104989487316477f
#define cephes_CBRT4 1.58740105196819947475f
#define cephes_invCBRT2 0.7937005259840997373740956123328f
#define cephes_invCBRT4 0.6299605249474365823842821870329f
#define CBRTF_P0 -0.13466110473359520655053f
#define CBRTF_P1 0.54664601366395524503440f
#define CBRTF_P2 -0.95438224771509446525043f
#define CBRTF_P3 1.1399983354717293273738f
#define CBRTF_P4 0.40238979564544752126924f

#define TANPI8F 0.414213562373095048802f   // tan(pi/8) => 0.4142135623730950
#define TAN3PI8F 2.414213562373095048802f  // tan(3*pi/8) => 2.414213562373095

static const float MAXNUMF = 3.4028234663852885981170418348451692544e38f;

#define MAXLOGF 88.72283905206835f
#define MINLOGF -103.278929903431851103f
#define MAXLOGFDIV2 44.361419526034176f

#ifdef ALTIVEC

#define ALTIVEC_LEN_FLOAT 4
#define ALTIVEC_LEN_INT32 4
#define ALTIVEC_LEN_INT16 8
#define ALTIVEC_LEN_BYTES 16

#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))

#define _PI8_CONST(Name, Val)                                                                           \
    static const ALIGN16_BEG unsigned char _pi8_##Name[16] ALIGN16_END = {Val, Val, Val, Val, Val, Val, \
                                                                          Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}

_PI8_CONST(0, 0x00);
_PI8_CONST(ff, 0xFF);

typedef __vector float v4sf;
typedef __vector int v4si;
typedef __vector unsigned int v4ui;
typedef __vector short v8ss;
typedef __vector unsigned short v8us;
typedef __vector unsigned char v16u8;
typedef __vector char v16s8;
typedef __vector bool int v4bi;

typedef struct {
    v4sf val[2];
} v4sfx2;

// extract real and imaginary part with
//   v4sf re = vec_perm(vec1, vec2, re_mask);
//   v4sf im = vec_perm(vec1, vec2, im_mask);
static const v16u8 re_mask = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
static const v16u8 im_mask = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
static const v16u8 reim_mask_hi = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
static const v16u8 reim_mask_lo = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
static const v16u8 flip_vector = {12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3};

#endif

#ifdef SSE

#define SSE_LEN_BYTES 16  // Size of SSE lane
#define SSE_LEN_INT16 8   // number of int16 with an SSE lane
#define SSE_LEN_INT32 4   // number of int32 with an SSE lane
#define SSE_LEN_FLOAT 4   // number of float with an SSE lane
#define SSE_LEN_DOUBLE 2  // number of double with an SSE lane

typedef __m128d v2sd;   // vector of 2 double (sse)
typedef __m128i v2sid;  // vector of 2 int64 (sse2)

#ifdef ARM

typedef float32x4_t v4sf;      // vector of 4 float
typedef float32x4x2_t v4sfx2;  // vector of 4 float
typedef uint32x4_t v4su;       // vector of 4 uint32
typedef int32x4_t v4si;        // vector of 4 uint32
typedef float32x4x2_t v4sfx2;

#if defined(__aarch64__)
typedef float64x2x2_t v2sdx2;
#else
typedef float32x4x2_t v2sdx2;
#endif


typedef int8x16_t v8ss;
typedef uint8x16_t v8us;
typedef uint16x8_t v16u8;
typedef uint16x8_t v16s8;

#else

typedef __m128 v4sf;   // vector of 4 float (sse1)
typedef __m128i v4si;  // vector of 4 int (sse2)
typedef __m128i v8ss;
typedef __m128i v8us;
typedef __m128i v16u8;
typedef __m128i v16s8;

typedef struct {
    v4sf val[2];
} v4sfx2;

typedef struct {
    v2sd val[2];
} v2sdx2;

#endif  // ARM

#define ROUNDTONEAREST (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define ROUNDTOFLOOR (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
#define ROUNDTOCEIL (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
#define ROUNDTOZERO (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)


#define _PD_CONST(Name, Val) \
    static const ALIGN16_BEG double _pd_##Name[2] ALIGN16_END = {Val, Val}
#define _PI64_CONST(Name, Val) \
    static const ALIGN16_BEG int64_t _pi64_##Name[2] ALIGN16_END = {Val, Val}
#define _PD_CONST_TYPE(Name, Type, Val) \
    static const ALIGN16_BEG Type _pd_##Name[2] ALIGN16_END = {Val, Val}

#endif  // SSE


#if defined(SSE) || defined(ALTIVEC)

#define _PS_CONST(Name, Val) \
    static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PI32_CONST(Name, Val) \
    static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PS_CONST_TYPE(Name, Type, Val) \
    static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}

// Warning, declared in reverse order since it's little endian :
//  const v4sf conj_mask = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
#if defined(SSE)
static const float _ps_conj_mask[4] __attribute__((aligned(16))) = {1.0f, -1.0f, 1.0f, -1.0f};
#else  // ALTIVEC, big endian
static const float _ps_conj_mask[4] __attribute__((aligned(16))) = {-1.0f, 1.0f, -1.0f, 1.0f};
#endif

/////////////// INT //////////////////
_PI32_CONST(0, 0);
_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7e, 0x7e);
_PI32_CONST(0x7f, 0x7f);

/////////////// SINGLE //////////////////
_PS_CONST(0, 0.0f);
_PS_CONST(min0, -0.0f);
_PS_CONST(1, 1.0f);
_PS_CONST(0p3, 0.333333333333f);
_PS_CONST(min0p3, -0.333333333333f);
_PS_CONST(0p5, 0.5f);
_PS_CONST(min1, -1.0f);
_PS_CONST(min2, -2.0f);
_PS_CONST(2, 2.0f);
_PS_CONST(min0p5, -0.5f);
_PS_CONST(3, 3.0f);

/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);  // 1000 0000 0111 1111 1111 1111 1111 1111

_PS_CONST_TYPE(sign_mask, int, (int) 0x80000000);
_PS_CONST_TYPE(mid_mask, int, (int)  0x3effffff);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS_CONST(cephes_log_q1, -2.12194440e-4f);
_PS_CONST(cephes_log_q2, 0.693359375f);

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

_PS_CONST(minus_cephes_DP1, -0.78515625f);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS_CONST(sincof_p0, -1.9515295891E-4f);
_PS_CONST(sincof_p1, 8.3321608736E-3f);
_PS_CONST(sincof_p2, -1.6666654611E-1f);
_PS_CONST(coscof_p0, 2.443315711809948E-005f);
_PS_CONST(coscof_p1, -1.388731625493765E-003f);
_PS_CONST(coscof_p2, 4.166664568298827E-002f);
_PS_CONST(cephes_FOPI, 1.27323954473516f);  // 4 / M_PI

// For tanf
_PS_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS_CONST(DP1, -0.78515625f);
_PS_CONST(DP2, -2.4187564849853515625E-4f);
_PS_CONST(DP3, -3.77489497744594108E-8f);

_PS_CONST(FOPI, 1.27323954473516f); /* 4/pi */
_PS_CONST(TAN_P0, 9.38540185543E-3f);
_PS_CONST(TAN_P1, 3.11992232697E-3f);
_PS_CONST(TAN_P2, 2.44301354525E-2f);
_PS_CONST(TAN_P3, 5.34112807005E-2f);
_PS_CONST(TAN_P4, 1.33387994085E-1f);
_PS_CONST(TAN_P5, 3.33331568548E-1f);

_PS_CONST(ASIN_P0, 4.2163199048E-2f);
_PS_CONST(ASIN_P1, 2.4181311049E-2f);
_PS_CONST(ASIN_P2, 4.5470025998E-2f);
_PS_CONST(ASIN_P3, 7.4953002686E-2f);
_PS_CONST(ASIN_P4, 1.6666752422E-1f);

_PS_CONST(PIF, 3.14159265358979323846f);      // PI
_PS_CONST(mPIF, -3.14159265358979323846f);    // -PI
_PS_CONST(PIO2F, 1.57079632679489661923f);    // PI/2 1.570796326794896619
_PS_CONST(mPIO2F, -1.57079632679489661923f);  // -PI/2 1.570796326794896619
_PS_CONST(PIO4F, 0.785398163397448309615f);   // PI/4 0.7853981633974483096

_PS_CONST(TANPI8F, 0.414213562373095048802f);   // tan(pi/8) => 0.4142135623730950
_PS_CONST(TAN3PI8F, 2.414213562373095048802f);  // tan(3*pi/8) => 2.414213562373095

_PS_CONST(ATAN_P0, 8.05374449538e-2f);
_PS_CONST(ATAN_P1, -1.38776856032E-1f);
_PS_CONST(ATAN_P2, 1.99777106478E-1f);
_PS_CONST(ATAN_P3, -3.33329491539E-1f);

_PS_CONST_TYPE(pos_sign_mask, int, (int) 0x7FFFFFFF);
_PS_CONST_TYPE(neg_sign_mask, int, (int) ~0x7FFFFFFF);

_PS_CONST(MAXLOGF, 88.72283905206835f);
_PS_CONST(MAXLOGFDIV2, 44.361419526034176f);
_PS_CONST(MINLOGF, -103.278929903431851103f);
_PS_CONST(cephes_exp_minC1, -0.693359375f);
_PS_CONST(cephes_exp_minC2, 2.12194440e-4f);

_PS_CONST(0p625, 0.625f);
_PS_CONST(TANH_P0, -5.70498872745E-3f);
_PS_CONST(TANH_P1, 2.06390887954E-2f);
_PS_CONST(TANH_P2, -5.37397155531E-2f);
_PS_CONST(TANH_P3, 1.33314422036E-1f);
_PS_CONST(TANH_P4, -3.33332819422E-1f);

_PS_CONST(MAXNUMF, 3.4028234663852885981170418348451692544e38f);
_PS_CONST(minMAXNUMF, -3.4028234663852885981170418348451692544e38f);
_PS_CONST(SINH_P0, 2.03721912945E-4f);
_PS_CONST(SINH_P1, 8.33028376239E-3f);
_PS_CONST(SINH_P2, 1.66667160211E-1f);

_PS_CONST(1emin4, 1e-4f);
_PS_CONST(ATANH_P0, 1.81740078349E-1f);
_PS_CONST(ATANH_P1, 8.24370301058E-2f);
_PS_CONST(ATANH_P2, 1.46691431730E-1f);
_PS_CONST(ATANH_P3, 1.99782164500E-1f);
_PS_CONST(ATANH_P4, 3.33337300303E-1f);

_PS_CONST_TYPE(zero, int, (int) 0x00000000);
_PS_CONST(1500, 1500.0f);
_PS_CONST(LOGE2F, 0.693147180559945309f);
_PS_CONST(ASINH_P0, 2.0122003309E-2f);
_PS_CONST(ASINH_P1, -4.2699340972E-2f);
_PS_CONST(ASINH_P2, 7.4847586088E-2f);
_PS_CONST(ASINH_P3, -1.6666288134E-1f);

_PS_CONST(ACOSH_P0, 1.7596881071E-3f);
_PS_CONST(ACOSH_P1, -7.5272886713E-3f);
_PS_CONST(ACOSH_P2, 2.6454905019E-2f);
_PS_CONST(ACOSH_P3, -1.1784741703E-1f);
_PS_CONST(ACOSH_P4, 1.4142135263E0f);

/* For log10f */
_PS_CONST(cephes_L102A, 3.0078125E-1f);
_PS_CONST(cephes_L102B, 2.48745663981195213739E-4f);
_PS_CONST(cephes_L10EA, 4.3359375E-1f);
_PS_CONST(cephes_L10EB, 7.00731903251827651129E-4f);

/* For log2f */
_PS_CONST(cephes_LOG2EA, 0.44269504088896340735992f);

/* For cbrtf */
_PS_CONST(cephes_CBRT2, 1.25992104989487316477f);
_PS_CONST(cephes_CBRT4, 1.58740105196819947475f);
_PS_CONST(cephes_invCBRT2, 0.7937005259840997373740956123328f);
_PS_CONST(cephes_invCBRT4, 0.6299605249474365823842821870329f);
_PS_CONST(CBRTF_P0, -0.13466110473359520655053f);
_PS_CONST(CBRTF_P1, 0.54664601366395524503440f);
_PS_CONST(CBRTF_P2, -0.95438224771509446525043f);
_PS_CONST(CBRTF_P3, 1.1399983354717293273738f);
_PS_CONST(CBRTF_P4, 0.40238979564544752126924f);

#endif  // SSE/ARM || ALTIVEC

#ifdef SSE  // or ARM. no double precision for ALTIVEC
/////////////// INT64 //////////////
_PI64_CONST(1, 1);
_PI64_CONST(inv1, ~1);
_PI64_CONST(2, 2);
_PI64_CONST(4, 4);
_PI64_CONST(0x7f, 0x7f);
_PI64_CONST(0x3ff, 0x3ff);
_PI64_CONST(1023, (unsigned int) 1023);

/////////////// DOUBLE //////////////////
_PD_CONST_TYPE(zero, int, (int) 0x00000000);
_PD_CONST_TYPE(min_norm_pos, int64_t, 0x380ffff83ce549caL);
_PD_CONST_TYPE(mant_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(inv_mant_mask, int64_t, (int64_t) 0x800FFFFFFFFFFFFFL);
_PD_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x8000000000000000L);
_PD_CONST_TYPE(mid_mask, int64_t, (int64_t)  0x3fdfffffffffffffL);
_PD_CONST_TYPE(inv_sign_mask, int64_t, ~0x8000000000000000L);
_PD_CONST_TYPE(pos_sign_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD_CONST_TYPE(epi64_mask, double, 0x0018000000000000L);

_PD_CONST(minus_cephes_DP1, -7.85398125648498535156E-1);
_PD_CONST(minus_cephes_DP2, -3.77489470793079817668E-8);
_PD_CONST(minus_cephes_DP3, -2.69515142907905952645E-15);
_PD_CONST(sincof_p0, 1.58962301576546568060E-10);
_PD_CONST(sincof_p1, -2.50507477628578072866E-8);
_PD_CONST(sincof_p2, 2.75573136213857245213E-6);
_PD_CONST(sincof_p3, -1.98412698295895385996E-4);
_PD_CONST(sincof_p4, 8.33333333332211858878E-3);
_PD_CONST(sincof_p5, -1.66666666666666307295E-1);
_PD_CONST(coscof_p0, -1.13585365213876817300E-11);
_PD_CONST(coscof_p1, 2.08757008419747316778E-9);
_PD_CONST(coscof_p2, -2.75573141792967388112E-7);
_PD_CONST(coscof_p3, 2.48015872888517045348E-5);
_PD_CONST(coscof_p4, -1.38888888888730564116E-3);
_PD_CONST(coscof_p5, 4.16666666666665929218E-2);
_PD_CONST(cephes_FOPI, 1.2732395447351626861510701069801148);  // 4 / M_PI

_PD_CONST(1, 1.0);
_PD_CONST(2, 2.0);
_PD_CONST(min8, -8.0);
_PD_CONST(0p5, 0.5);
_PD_CONST(min0p5, -0.5);
_PD_CONST(0p125, 0.125);
_PD_CONST(0p625, 0.625);
_PD_CONST(0p66, 0.66);
_PD_CONST(1em8, 1E-8);
_PD_CONST(min_212emin4, -2.121944400546905827679e-4);
_PD_CONST(0p69, 0.693359375);

_PD_CONST(cephes_SQRTHF, 0.70710678118654752440);

_PD_CONST(cephes_log_p0, 1.01875663804580931796E-4);
_PD_CONST(cephes_log_p1, 4.97494994976747001425E-1);
_PD_CONST(cephes_log_p2, 4.70579119878881725854E0);
_PD_CONST(cephes_log_p3, 1.44989225341610930846E1);
_PD_CONST(cephes_log_p4, 1.79368678507819816313E1);
_PD_CONST(cephes_log_p5, 7.70838733755885391666E0);

_PD_CONST(cephes_log_q0, 1.12873587189167450590E1);
_PD_CONST(cephes_log_q1, 4.52279145837532221105E1);
_PD_CONST(cephes_log_q2, 8.29875266912776603211E1);
_PD_CONST(cephes_log_q3, 7.11544750618563894466E1);
_PD_CONST(cephes_log_q4, 4.52279145837532221105E1);
_PD_CONST(cephes_log_q5, 2.31251620126765340583E1);

_PD_CONST(cephes_log_r0, -7.89580278884799154124E-1);
_PD_CONST(cephes_log_r1, 1.63866645699558079767E1);
_PD_CONST(cephes_log_r2, -6.41409952958715622951E1);

_PD_CONST(cephes_log_s0, -3.56722798256324312549E1);
_PD_CONST(cephes_log_s1, 3.12093766372244180303E2);
_PD_CONST(cephes_log_s2, -7.69691943550460008604E2);

_PD_CONST(exp_hi, 709.437);
_PD_CONST(exp_lo, -709.436139303);

_PD_CONST(cephes_LOG2E, 1.4426950408889634073599);  /* 1/log(2) */
_PD_CONST(cephes_LOGE2, 6.93147180559945309417E-1); /* log(2) */

_PD_CONST(cephes_exp_p0, 1.26177193074810590878e-4);
_PD_CONST(cephes_exp_p1, 3.02994407707441961300e-2);
_PD_CONST(cephes_exp_p2, 9.99999999999999999910e-1);

_PD_CONST(cephes_exp_q0, 3.00198505138664455042e-6);
_PD_CONST(cephes_exp_q1, 2.52448340349684104192e-3);
_PD_CONST(cephes_exp_q2, 2.27265548208155028766e-1);
_PD_CONST(cephes_exp_q3, 2.00000000000000000009e0);

_PD_CONST(cephes_exp_minC1, -0.693145751953125);
_PD_CONST(cephes_exp_minC2, -1.42860682030941723212e-6);

_PD_CONST_TYPE(positive_mask, int64_t, (int64_t) 0x7FFFFFFFFFFFFFFFL);
_PD_CONST_TYPE(negative_mask, int64_t, (int64_t) ~0x7FFFFFFFFFFFFFFFL);
_PD_CONST(ASIN_P0, 4.253011369004428248960E-3);
_PD_CONST(ASIN_P1, -6.019598008014123785661E-1);
_PD_CONST(ASIN_P2, 5.444622390564711410273E0);
_PD_CONST(ASIN_P3, -1.626247967210700244449E1);
_PD_CONST(ASIN_P4, 1.956261983317594739197E1);
_PD_CONST(ASIN_P5, -8.198089802484824371615E0);

_PD_CONST(PIF, 3.1415926535897932384626433832795028841971693993751058209749445923);      // PI
_PD_CONST(mPIF, -3.1415926535897932384626433832795028841971693993751058209749445923);    // -PI
_PD_CONST(PIO2F, 1.5707963267948966192313216916397514420985846996875529104874722961);    // PI/2 1.570796326794896619
_PD_CONST(mPIO2F, -1.5707963267948966192313216916397514420985846996875529104874722961);  // -PI/2 1.570796326794896619
_PD_CONST(PIO4F, 0.7853981633974483096156608458198757210492923498437764552437361480);    // PI/4 0.7853981633974483096

_PD_CONST(ASIN_Q0, -1.474091372988853791896E1);
_PD_CONST(ASIN_Q1, 7.049610280856842141659E1);
_PD_CONST(ASIN_Q2, -1.471791292232726029859E2);
_PD_CONST(ASIN_Q3, 1.395105614657485689735E2);
_PD_CONST(ASIN_Q4, -4.918853881490881290097E1);

_PD_CONST(ASIN_R0, 2.967721961301243206100E-3);
_PD_CONST(ASIN_R1, -5.634242780008963776856E-1);
_PD_CONST(ASIN_R2, 6.968710824104713396794E0);
_PD_CONST(ASIN_R3, -2.556901049652824852289E1);
_PD_CONST(ASIN_R4, 2.853665548261061424989E1);

_PD_CONST(ASIN_S0, -2.194779531642920639778E1);
_PD_CONST(ASIN_S1, 1.470656354026814941758E2);
_PD_CONST(ASIN_S2, -3.838770957603691357202E2);
_PD_CONST(ASIN_S3, 3.424398657913078477438E2);

_PD_CONST(PIO2, 1.57079632679489661923);    /* pi/2 */
_PD_CONST(PIO4, 7.85398163397448309616E-1); /* pi/4 */

_PD_CONST(minMOREBITS, -6.123233995736765886130E-17);
_PD_CONST(MOREBITS, 6.123233995736765886130E-17);
_PD_CONST(0p5xMOREBITS, 3.061616997868382943065e-17);

_PD_CONST(ATAN_P0, -8.750608600031904122785E-1);
_PD_CONST(ATAN_P1, -1.615753718733365076637E1);
_PD_CONST(ATAN_P2, -7.500855792314704667340E1);
_PD_CONST(ATAN_P3, -1.228866684490136173410E2);
_PD_CONST(ATAN_P4, -6.485021904942025371773E1);

_PD_CONST(ATAN_Q0, 2.485846490142306297962E1);
_PD_CONST(ATAN_Q1, 1.650270098316988542046E2);
_PD_CONST(ATAN_Q2, 4.328810604912902668951E2);
_PD_CONST(ATAN_Q3, 4.853903996359136964868E2);
_PD_CONST(ATAN_Q4, 1.945506571482613964425E2);

_PD_CONST(TAN3PI8, 2.41421356237309504880); /* 3*pi/8 */

_PD_CONST(min1, -1.0);

_PD_CONST(1m14, 1.0e-14);
_PD_CONST(TAN_P0, -1.30936939181383777646E4);
_PD_CONST(TAN_P1, 1.15351664838587416140E6);
_PD_CONST(TAN_P2, -1.79565251976484877988E7);
_PD_CONST(TAN_Q0, 1.36812963470692954678E4);
_PD_CONST(TAN_Q1, -1.32089234440210967447E6);
_PD_CONST(TAN_Q2, 2.50083801823357915839E7);
_PD_CONST(TAN_Q3, -5.38695755929454629881E7);

_PD_CONST(TAN_mDP1, -7.853981554508209228515625E-1);
_PD_CONST(TAN_mDP2, -7.94662735614792836714E-9);
_PD_CONST(TAN_mDP3, -3.06161699786838294307E-17);
_PD_CONST(tanlossth, 1.073741824e9);
_PD_CONST(PDEPI64U, 0x0010000000000000);


#endif  // SSE/ARM


#ifdef AVX

#define AVX_LEN_BYTES 32  // Size of AVX lane
#define AVX_LEN_INT16 16  // number of int16 with an AVX lane
#define AVX_LEN_INT32 8   // number of int32 with an AVX lane
#define AVX_LEN_FLOAT 8   // number of float with an AVX lane
#define AVX_LEN_DOUBLE 4  // number of double with an AVX lane

#define _PI32AVX_CONST(Name, Val) \
    static const ALIGN32_BEG int _pi32avx_##Name[4] ALIGN32_END = {Val, Val, Val, Val}
#define _PS256_CONST(Name, Val) \
    static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST256(Name, Val) \
    static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS256_CONST_TYPE(Name, Type, Val) \
    static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PD256_CONST(Name, Val) \
    static const ALIGN32_BEG double _pd256_##Name[4] ALIGN32_END = {Val, Val, Val, Val}
#define _PI256_64_CONST(Name, Val) \
    static const ALIGN32_BEG int64_t _pi256_64_##Name[4] ALIGN32_END = {Val, Val, Val, Val}
#define _PD256_CONST_TYPE(Name, Type, Val) \
    static const ALIGN32_BEG Type _pd256_##Name[4] ALIGN32_END = {Val, Val, Val, Val}

static const float _ps256_conj_mask[8] __attribute__((aligned(32))) = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
// static int32_t vindex256_arr[8] __attribute__((aligned[32])) = {0,1,2,3,4,5,6,7};

typedef __m256 v8sf;    // vector of 8 float (avx)
typedef __m256i v8si;   // vector of 8 int   (avx)
typedef __m256i v4sid;  // vector of 4 64 bits int   (avx)
typedef __m128i v4si;   // vector of 4 int   (avx)
typedef __m256d v4sd;   // vector of 4 double (avx)
typedef struct {
    v8sf val[2];
} v8sfx2;

typedef struct {
    v4sd val[2];
} v4sdx2;

#ifndef __AVX2__

typedef union imm_xmm_union {
    v8si imm;
    v4si xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_)           \
    {                                                 \
        imm_xmm_union u __attribute__((aligned(32))); \
        u.imm = imm_;                                 \
        xmm0_ = u.xmm[0];                             \
        xmm1_ = u.xmm[1];                             \
    }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_)           \
    {                                                 \
        imm_xmm_union u __attribute__((aligned(32))); \
        u.xmm[0] = xmm0_;                             \
        u.xmm[1] = xmm1_;                             \
        imm_ = u.imm;                                 \
    }

#define AVX2_BITOP_USING_SSE2(fn)                            \
    static inline v8si _mm256_##fn(v8si x, int a)            \
    {                                                        \
        /* use SSE2 instruction to perform the bitop AVX2 */ \
        v4si x1, x2;                                         \
        v8si ret;                                            \
        COPY_IMM_TO_XMM(x, x1, x2);                          \
        x1 = _mm_##fn(x1, a);                                \
        x2 = _mm_##fn(x2, a);                                \
        COPY_XMM_TO_IMM(x1, x2, ret);                        \
        return (ret);                                        \
    }

#warning "Using SSE2 to perform AVX2 bitshift ops"
AVX2_BITOP_USING_SSE2(slli_epi32)
AVX2_BITOP_USING_SSE2(srli_epi32)

#define AVX2_INTOP_USING_SSE2(fn)                                         \
    static inline v8si _mm256_##fn(v8si x, v8si y)                        \
    {                                                                     \
        /* use SSE2 instructions to perform the AVX2 integer operation */ \
        v4si x1, x2;                                                      \
        v4si y1, y2;                                                      \
        v8si ret;                                                         \
        COPY_IMM_TO_XMM(x, x1, x2);                                       \
        COPY_IMM_TO_XMM(y, y1, y2);                                       \
        x1 = _mm_##fn(x1, y1);                                            \
        x2 = _mm_##fn(x2, y2);                                            \
        COPY_XMM_TO_IMM(x1, x2, ret);                                     \
        return (ret);                                                     \
    }

#warning "Using SSE2 to perform AVX2 integer ops"
AVX2_INTOP_USING_SSE2(and_si128)
AVX2_INTOP_USING_SSE2(andnot_si128)
AVX2_INTOP_USING_SSE2(cmpeq_epi32)
AVX2_INTOP_USING_SSE2(sub_epi32)
AVX2_INTOP_USING_SSE2(add_epi32)

#endif /* __AVX2__ */

/////////////// INT //////////////////
_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7e, 0x7e);
_PI32_CONST256(0x7f, 0x7f);

_PI256_64_CONST(1, 1);
_PI256_64_CONST(inv1, ~1);
_PI256_64_CONST(2, 2);
_PI256_64_CONST(4, 4);
_PI256_64_CONST(0x7f, 0x7f);
_PI256_64_CONST(0x3ff, 0x3ff);
_PI256_64_CONST(1023, (unsigned int) 1023);

/////////////// SINGLE //////////////////
_PS256_CONST(1, 1.0f);
_PS256_CONST(3, 3.0f);
_PS256_CONST(0p3, 0.333333333333f);
_PS256_CONST(min0p3, -0.333333333333f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int) 0x80000000);
_PS256_CONST_TYPE(mid_mask, int, (int)  0x3effffff);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS256_CONST(cephes_log_q1, -2.12194440e-4f);
_PS256_CONST(cephes_log_q2, 0.693359375f);

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

_PS256_CONST(minus_cephes_DP1, -0.78515625f);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS256_CONST(sincof_p0, -1.9515295891E-4f);
_PS256_CONST(sincof_p1, 8.3321608736E-3f);
_PS256_CONST(sincof_p2, -1.6666654611E-1f);
_PS256_CONST(coscof_p0, 2.443315711809948E-005f);
_PS256_CONST(coscof_p1, -1.388731625493765E-003f);
_PS256_CONST(coscof_p2, 4.166664568298827E-002f);
_PS256_CONST(cephes_FOPI, 1.27323954473516f);  // 4 / M_PI

_PS256_CONST(min1, -1.0f);
_PS256_CONST(min2, -2.0f);
_PS256_CONST(2, 2.0f);
_PS256_CONST(min0p5, -0.5f);

// For tanf
_PS256_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS256_CONST(DP1, -0.78515625);
_PS256_CONST(DP2, -2.4187564849853515625e-4);
_PS256_CONST(DP3, -3.77489497744594108e-8);

_PS256_CONST(FOPI, 1.27323954473516); /* 4/pi */
_PS256_CONST(TAN_P0, 9.38540185543E-3);
_PS256_CONST(TAN_P1, 3.11992232697E-3);
_PS256_CONST(TAN_P2, 2.44301354525E-2);
_PS256_CONST(TAN_P3, 5.34112807005E-2);
_PS256_CONST(TAN_P4, 1.33387994085E-1);
_PS256_CONST(TAN_P5, 3.33331568548E-1);

_PS256_CONST(ASIN_P0, 4.2163199048E-2);
_PS256_CONST(ASIN_P1, 2.4181311049E-2);
_PS256_CONST(ASIN_P2, 4.5470025998E-2);
_PS256_CONST(ASIN_P3, 7.4953002686E-2);
_PS256_CONST(ASIN_P4, 1.6666752422E-1);

_PS256_CONST(PIF, 3.14159265358979323846);      // PI
_PS256_CONST(mPIF, -3.14159265358979323846);    // -PI
_PS256_CONST(PIO2F, 1.57079632679489661923);    // PI/2 1.570796326794896619
_PS256_CONST(mPIO2F, -1.57079632679489661923);  // -PI/2 1.570796326794896619
_PS256_CONST(PIO4F, 0.785398163397448309615);   // PI/4 0.7853981633974483096

_PS256_CONST(TANPI8F, 0.414213562373095048802);   // tan(pi/8) => 0.4142135623730950
_PS256_CONST(TAN3PI8F, 2.414213562373095048802);  // tan(3*pi/8) => 2.414213562373095

_PS256_CONST(ATAN_P0, 8.05374449538e-2);
_PS256_CONST(ATAN_P1, -1.38776856032E-1);
_PS256_CONST(ATAN_P2, 1.99777106478E-1);
_PS256_CONST(ATAN_P3, -3.33329491539E-1);

_PS256_CONST_TYPE(pos_sign_mask, int, (int) 0x7FFFFFFF);
_PS256_CONST_TYPE(neg_sign_mask, int, (int) ~0x7FFFFFFF);

_PS256_CONST(MAXLOGF, 88.72283905206835f);
_PS256_CONST(MAXLOGFDIV2, 44.361419526034176f);
_PS256_CONST(MINLOGF, -103.278929903431851103f);
_PS256_CONST(cephes_exp_minC1, -0.693359375f);
_PS256_CONST(cephes_exp_minC2, 2.12194440e-4f);
_PS256_CONST(0p625, 0.625f);

_PS256_CONST(TANH_P0, -5.70498872745E-3f);
_PS256_CONST(TANH_P1, 2.06390887954E-2f);
_PS256_CONST(TANH_P2, -5.37397155531E-2f);
_PS256_CONST(TANH_P3, 1.33314422036E-1f);
_PS256_CONST(TANH_P4, -3.33332819422E-1f);

_PS256_CONST(MAXNUMF, 3.4028234663852885981170418348451692544e38f);
_PS256_CONST(minMAXNUMF, -3.4028234663852885981170418348451692544e38f);
_PS256_CONST(SINH_P0, 2.03721912945E-4f);
_PS256_CONST(SINH_P1, 8.33028376239E-3f);
_PS256_CONST(SINH_P2, 1.66667160211E-1f);

_PS256_CONST(1emin4, 1e-4f);
_PS256_CONST(ATANH_P0, 1.81740078349E-1f);
_PS256_CONST(ATANH_P1, 8.24370301058E-2f);
_PS256_CONST(ATANH_P2, 1.46691431730E-1f);
_PS256_CONST(ATANH_P3, 1.99782164500E-1f);
_PS256_CONST(ATANH_P4, 3.33337300303E-1f);

_PS256_CONST(1500, 1500.0f);
_PS256_CONST(LOGE2F, 0.693147180559945309f);
_PS256_CONST(ASINH_P0, 2.0122003309E-2f);
_PS256_CONST(ASINH_P1, -4.2699340972E-2f);
_PS256_CONST(ASINH_P2, 7.4847586088E-2f);
_PS256_CONST(ASINH_P3, -1.6666288134E-1f);

_PS256_CONST(ACOSH_P0, 1.7596881071E-3f);
_PS256_CONST(ACOSH_P1, -7.5272886713E-3f);
_PS256_CONST(ACOSH_P2, 2.6454905019E-2f);
_PS256_CONST(ACOSH_P3, -1.1784741703E-1f);
_PS256_CONST(ACOSH_P4, 1.4142135263E0f);

/* For log10f */
_PS256_CONST(cephes_L102A, 3.0078125E-1f);
_PS256_CONST(cephes_L102B, 2.48745663981195213739E-4f);
_PS256_CONST(cephes_L10EA, 4.3359375E-1f);
_PS256_CONST(cephes_L10EB, 7.00731903251827651129E-4f);

/* For log2f */
_PS256_CONST(cephes_LOG2EA, 0.44269504088896340735992f);

/* For cbrtf */
_PS256_CONST(cephes_CBRT2, 1.25992104989487316477f);
_PS256_CONST(cephes_CBRT4, 1.58740105196819947475f);
_PS256_CONST(cephes_invCBRT2, 0.7937005259840997373740956123328f);
_PS256_CONST(cephes_invCBRT4, 0.6299605249474365823842821870329f);
_PS256_CONST(CBRTF_P0, -0.13466110473359520655053f);
_PS256_CONST(CBRTF_P1, 0.54664601366395524503440f);
_PS256_CONST(CBRTF_P2, -0.95438224771509446525043f);
_PS256_CONST(CBRTF_P3, 1.1399983354717293273738f);
_PS256_CONST(CBRTF_P4, 0.40238979564544752126924f);

/////////////// DOUBLE //////////////////
_PD256_CONST_TYPE(min_norm_pos, int64_t, 0x380ffff83ce549caL);
_PD256_CONST_TYPE(mant_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD256_CONST_TYPE(inv_mant_mask, int64_t, (int64_t)0x800FFFFFFFFFFFFFL);
_PD256_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x8000000000000000L);
_PD256_CONST_TYPE(mid_mask, int64_t, (int64_t)  0x3fdfffffffffffffL);
_PD256_CONST_TYPE(inv_sign_mask, int64_t, ~0x8000000000000000L);
_PD256_CONST_TYPE(pos_sign_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD256_CONST_TYPE(epi64_mask, double, 0x0018000000000000L);

_PD256_CONST(minus_cephes_DP1, -7.85398125648498535156E-1);
_PD256_CONST(minus_cephes_DP2, -3.77489470793079817668E-8);
_PD256_CONST(minus_cephes_DP3, -2.69515142907905952645E-15);
_PD256_CONST(sincof_p0, 1.58962301576546568060E-10);
_PD256_CONST(sincof_p1, -2.50507477628578072866E-8);
_PD256_CONST(sincof_p2, 2.75573136213857245213E-6);
_PD256_CONST(sincof_p3, -1.98412698295895385996E-4);
_PD256_CONST(sincof_p4, 8.33333333332211858878E-3);
_PD256_CONST(sincof_p5, -1.66666666666666307295E-1);
_PD256_CONST(coscof_p0, -1.13585365213876817300E-11);
_PD256_CONST(coscof_p1, 2.08757008419747316778E-9);
_PD256_CONST(coscof_p2, -2.75573141792967388112E-7);
_PD256_CONST(coscof_p3, 2.48015872888517045348E-5);
_PD256_CONST(coscof_p4, -1.38888888888730564116E-3);
_PD256_CONST(coscof_p5, 4.16666666666665929218E-2);
_PD256_CONST(cephes_FOPI, 1.2732395447351626861510701069801148);  // 4 / M_PI

_PD256_CONST_TYPE(positive_mask, int64_t, (int64_t) 0x7FFFFFFFFFFFFFFFL);
_PD256_CONST_TYPE(negative_mask, int64_t, (int64_t) ~0x7FFFFFFFFFFFFFFFL);
_PD256_CONST(ASIN_P0, 4.253011369004428248960E-3);
_PD256_CONST(ASIN_P1, -6.019598008014123785661E-1);
_PD256_CONST(ASIN_P2, 5.444622390564711410273E0);
_PD256_CONST(ASIN_P3, -1.626247967210700244449E1);
_PD256_CONST(ASIN_P4, 1.956261983317594739197E1);
_PD256_CONST(ASIN_P5, -8.198089802484824371615E0);

_PD256_CONST(PIF, 3.1415926535897932384626433832795028841971693993751058209749445923);      // PI
_PD256_CONST(mPIF, -3.1415926535897932384626433832795028841971693993751058209749445923);    // -PI
_PD256_CONST(PIO2F, 1.5707963267948966192313216916397514420985846996875529104874722961);    // PI/2 1.570796326794896619
_PD256_CONST(mPIO2F, -1.5707963267948966192313216916397514420985846996875529104874722961);  // -PI/2 1.570796326794896619
_PD256_CONST(PIO4F, 0.7853981633974483096156608458198757210492923498437764552437361480);    // PI/4 0.7853981633974483096

_PD256_CONST(ASIN_Q0, -1.474091372988853791896E1);
_PD256_CONST(ASIN_Q1, 7.049610280856842141659E1);
_PD256_CONST(ASIN_Q2, -1.471791292232726029859E2);
_PD256_CONST(ASIN_Q3, 1.395105614657485689735E2);
_PD256_CONST(ASIN_Q4, -4.918853881490881290097E1);

_PD256_CONST(ASIN_R0, 2.967721961301243206100E-3);
_PD256_CONST(ASIN_R1, -5.634242780008963776856E-1);
_PD256_CONST(ASIN_R2, 6.968710824104713396794E0);
_PD256_CONST(ASIN_R3, -2.556901049652824852289E1);
_PD256_CONST(ASIN_R4, 2.853665548261061424989E1);

_PD256_CONST(ASIN_S0, -2.194779531642920639778E1);
_PD256_CONST(ASIN_S1, 1.470656354026814941758E2);
_PD256_CONST(ASIN_S2, -3.838770957603691357202E2);
_PD256_CONST(ASIN_S3, 3.424398657913078477438E2);

_PD256_CONST(PIO2, 1.57079632679489661923);    /* pi/2 */
_PD256_CONST(PIO4, 7.85398163397448309616E-1); /* pi/4 */

_PD256_CONST(minMOREBITS, -6.123233995736765886130E-17);
_PD256_CONST(MOREBITS, 6.123233995736765886130E-17);

_PD256_CONST(ATAN_P0, -8.750608600031904122785E-1);
_PD256_CONST(ATAN_P1, -1.615753718733365076637E1);
_PD256_CONST(ATAN_P2, -7.500855792314704667340E1);
_PD256_CONST(ATAN_P3, -1.228866684490136173410E2);
_PD256_CONST(ATAN_P4, -6.485021904942025371773E1);

_PD256_CONST(ATAN_Q0, 2.485846490142306297962E1);
_PD256_CONST(ATAN_Q1, 1.650270098316988542046E2);
_PD256_CONST(ATAN_Q2, 4.328810604912902668951E2);
_PD256_CONST(ATAN_Q3, 4.853903996359136964868E2);
_PD256_CONST(ATAN_Q4, 1.945506571482613964425E2);

_PD256_CONST(TAN3PI8, 2.41421356237309504880); /* 3*pi/8 */

_PD256_CONST(min1, -1.0);
_PD256_CONST(1, 1.0);
_PD256_CONST(2, 2.0);
_PD256_CONST(min8, -8.0);
_PD256_CONST(0p5, 0.5);
_PD256_CONST(min0p5, -0.5);
_PD256_CONST(0p125, 0.125);
_PD256_CONST(0p625, 0.625);
_PD256_CONST(0p66, 0.66);
_PD256_CONST(1em8, 1E-8);
_PD256_CONST(min_212emin4, -2.121944400546905827679e-4);
_PD256_CONST(0p69, 0.693359375);

_PD256_CONST(cephes_SQRTHF, 0.70710678118654752440);

_PD256_CONST(cephes_log_p0, 1.01875663804580931796E-4);
_PD256_CONST(cephes_log_p1, 4.97494994976747001425E-1);
_PD256_CONST(cephes_log_p2, 4.70579119878881725854E0);
_PD256_CONST(cephes_log_p3, 1.44989225341610930846E1);
_PD256_CONST(cephes_log_p4, 1.79368678507819816313E1);
_PD256_CONST(cephes_log_p5, 7.70838733755885391666E0);

_PD256_CONST(cephes_log_q0, 1.12873587189167450590E1);
_PD256_CONST(cephes_log_q1, 4.52279145837532221105E1);
_PD256_CONST(cephes_log_q2, 8.29875266912776603211E1);
_PD256_CONST(cephes_log_q3, 7.11544750618563894466E1);
_PD256_CONST(cephes_log_q4, 4.52279145837532221105E1);
_PD256_CONST(cephes_log_q5, 2.31251620126765340583E1);

_PD256_CONST(cephes_log_r0, -7.89580278884799154124E-1);
_PD256_CONST(cephes_log_r1, 1.63866645699558079767E1);
_PD256_CONST(cephes_log_r2, -6.41409952958715622951E1);

_PD256_CONST(cephes_log_s0, -3.56722798256324312549E1);
_PD256_CONST(cephes_log_s1, 3.12093766372244180303E2);
_PD256_CONST(cephes_log_s2, -7.69691943550460008604E2);

_PD256_CONST(1m14, 1.0e-14);
_PD256_CONST(TAN_P0, -1.30936939181383777646E4);
_PD256_CONST(TAN_P1, 1.15351664838587416140E6);
_PD256_CONST(TAN_P2, -1.79565251976484877988E7);
_PD256_CONST(TAN_Q0, 1.36812963470692954678E4);
_PD256_CONST(TAN_Q1, -1.32089234440210967447E6);
_PD256_CONST(TAN_Q2, 2.50083801823357915839E7);
_PD256_CONST(TAN_Q3, -5.38695755929454629881E7);

_PD256_CONST(TAN_mDP1, -7.853981554508209228515625E-1);
_PD256_CONST(TAN_mDP2, -7.94662735614792836714E-9);
_PD256_CONST(TAN_mDP3, -3.06161699786838294307E-17);
_PD256_CONST(tanlossth, 1.073741824e9);
_PD256_CONST(PDEPI64U, 0x0010000000000000);

_PD256_CONST(cephes_LOG2E, 1.4426950408889634073599);  /* 1/log(2) */
_PD256_CONST(cephes_LOGE2, 6.93147180559945309417E-1); /* log(2) */

_PD256_CONST(cephes_exp_p0, 1.26177193074810590878e-4);
_PD256_CONST(cephes_exp_p1, 3.02994407707441961300e-2);
_PD256_CONST(cephes_exp_p2, 9.99999999999999999910e-1);

_PD256_CONST(cephes_exp_q0, 3.00198505138664455042e-6);
_PD256_CONST(cephes_exp_q1, 2.52448340349684104192e-3);
_PD256_CONST(cephes_exp_q2, 2.27265548208155028766e-1);
_PD256_CONST(cephes_exp_q3, 2.00000000000000000009e0);

_PD256_CONST(cephes_exp_minC1, -0.693145751953125);
_PD256_CONST(cephes_exp_minC2, -1.42860682030941723212e-6);

#endif


#ifdef AVX512

#define AVX512_LEN_BYTES 64  // Size of AVX512 lane
#define AVX512_LEN_INT16 32  // number of int16 with an AVX512 lane
#define AVX512_LEN_INT32 16  // number of int32 with an AVX512 lane
#define AVX512_LEN_FLOAT 16  // number of float with an AVX512 lane
#define AVX512_LEN_DOUBLE 8  // number of double with an AVX512 lane

typedef __m512 v16sf;   // vector of 16 float (avx512)
typedef __m512i v16si;  // vector of 16 int   (avx512)
typedef __m512i v8sid;  // vector of 8 64bits int   (avx512)
typedef __m256i v8si;   // vector of 8 int   (avx)
typedef __m512d v8sd;   // vector of 8 double (avx512)
typedef struct {
    v16sf val[2];
} v16sfx2;
typedef struct {
    v8sd val[2];
} v8sdx2;

#define _PI64AVX512_CONST(Name, Val) \
    static const ALIGN64_BEG int _pi64avx_##Name[8] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val}

/* declare some AVX512 constants */
#define _PS512_CONST(Name, Val) \
    static const ALIGN64_BEG float _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST512(Name, Val) \
    static const ALIGN64_BEG int _pi32_512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS512_CONST_TYPE(Name, Type, Val) \
    static const ALIGN64_BEG Type _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PD512_CONST(Name, Val) \
    static const ALIGN64_BEG double _pd512_##Name[8] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI512_64_CONST(Name, Val) \
    static const ALIGN64_BEG int64_t _pi512_64_##Name[8] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PD512_CONST_TYPE(Name, Type, Val) \
    static const ALIGN64_BEG Type _pd512_##Name[8] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val}

////////// INT /////////////
// Named to be unified ..
_PI32_CONST512(0, 0);
_PI32_CONST512(1, 1);
_PI32_CONST512(inv1, ~1);
_PI32_CONST512(2, 2);
_PI32_CONST512(4, 4);
_PI32_CONST512(0x7e, 0x7e);
_PI32_CONST512(0x7f, 0x7f);
_PI512_64_CONST(1, 1);
_PI512_64_CONST(inv1, ~1);
_PI512_64_CONST(2, 2);
_PI512_64_CONST(4, 4);
_PI512_64_CONST(0x7f, 0x7f);
_PI512_64_CONST(0x3ff, 0x3ff);
_PI512_64_CONST(1023, (unsigned int) 1023);
_PI64AVX512_CONST(1, 1);
_PI64AVX512_CONST(inv1, ~1);
_PI64AVX512_CONST(2, 2);
_PI64AVX512_CONST(4, 4);

// used for cplxtoreal transforms

// Select alternatively indexes between Real and Complex Elements of the two 512bit vectors
//  indexes with 0x1X means second vector argument and X position (hexa)
static const int _pi32_512_idx_re[16] __attribute__((aligned(64))) = {0x10, 0x12, 0x14, 0x16,
                                                                      0x18, 0x1A, 0x1C, 0x1E, 0, 2, 4, 6, 8, 10, 12, 14};
static const int _pi32_512_idx_im[16] __attribute__((aligned(64))) = {0x11, 0x13, 0x15, 0x17,
                                                                      0x19, 0x1B, 0x1D, 0x1F, 1, 3, 5, 7, 9, 11, 13, 15};
static const int64_t _pi64_512_idx_re[8] __attribute__((aligned(64))) = {8, 10, 12, 14, 0, 2, 4, 6};
static const int64_t _pi64_512_idx_im[8] __attribute__((aligned(64))) = {9, 8 + 3, 8 + 5, 8 + 7, 1, 3, 5, 7};


// used for realtocplx transforms
static const int _pi32_512_idx_cplx_lo[16] __attribute__((aligned(64))) = {0x10, 0, 0x11, 1,
                                                                           0x12, 2, 0x13, 3, 0x14, 4, 0x15, 5, 0x16, 6, 0x17, 7};
static const int _pi32_512_idx_cplx_hi[16] __attribute__((aligned(64))) = {0x18, 8, 0x19, 9,
                                                                           0x1A, 10, 0x1B, 11, 0x1C, 12, 0x1D, 13, 0x1E, 14, 0x1F, 15};
static const int64_t _pi64_512_idx_cplx_lo[8] __attribute__((aligned(64))) = {0x8, 0x0, 0x9, 0x1, 0xA, 0x2, 0xB, 0x3};
static const int64_t _pi64_512_idx_cplx_hi[8] __attribute__((aligned(64))) = {0xC, 0x4, 0xD, 0x5, 0xE, 6, 0xF, 0x7};

////////// SINGLE /////////////
_PS512_CONST(1, 1.0f);
_PS512_CONST(3, 3.0f);
_PS512_CONST(0p3, 0.333333333333f);
_PS512_CONST(min0p3, -0.333333333333f);
_PS512_CONST(0p5, 0.5f);

/* the smallest non denormalized float number */
_PS512_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS512_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS512_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS512_CONST_TYPE(sign_mask, int, (int) 0x80000000);
_PS512_CONST_TYPE(mid_mask, int, (int)  0x3effffff);
_PS512_CONST_TYPE(inv_sign_mask, int, ~0x80000000);
_PS512_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS512_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS512_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS512_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS512_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS512_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS512_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS512_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS512_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS512_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS512_CONST(cephes_log_q1, -2.12194440e-4f);
_PS512_CONST(cephes_log_q2, 0.693359375f);

_PS512_CONST(exp_hi, 88.3762626647949f);
_PS512_CONST(exp_lo, -88.3762626647949f);

_PS512_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS512_CONST(cephes_exp_C1, 0.693359375f);
_PS512_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS512_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS512_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS512_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS512_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS512_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS512_CONST(cephes_exp_p5, 5.0000001201E-1f);

_PS512_CONST(minus_cephes_DP1, -0.78515625f);
_PS512_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS512_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS512_CONST(sincof_p0, -1.9515295891E-4f);
_PS512_CONST(sincof_p1, 8.3321608736E-3f);
_PS512_CONST(sincof_p2, -1.6666654611E-1f);
_PS512_CONST(coscof_p0, 2.443315711809948E-005f);
_PS512_CONST(coscof_p1, -1.388731625493765E-003f);
_PS512_CONST(coscof_p2, 4.166664568298827E-002f);
_PS512_CONST(cephes_FOPI, 1.27323954473516f);  // 4 / M_PI

_PS512_CONST(min1, -1.0f);
_PS512_CONST(plus1, 1.0f);
_PS512_CONST(min2, -2.0f);
_PS512_CONST(2, 2.0f);
_PS512_CONST(min0p5, -0.5f);


// For tanf
_PS512_CONST(DP123, 0.78515625 + 2.4187564849853515625e-4 + 3.77489497744594108e-8);

// Neg values to better migrate to FMA
_PS512_CONST(DP1, -0.78515625f);
_PS512_CONST(DP2, -2.4187564849853515625e-4f);
_PS512_CONST(DP3, -3.77489497744594108e-8f);

_PS512_CONST(FOPI, 1.27323954473516f); /* 4/pi */
_PS512_CONST(TAN_P0, 9.38540185543E-3f);
_PS512_CONST(TAN_P1, 3.11992232697E-3f);
_PS512_CONST(TAN_P2, 2.44301354525E-2f);
_PS512_CONST(TAN_P3, 5.34112807005E-2f);
_PS512_CONST(TAN_P4, 1.33387994085E-1f);
_PS512_CONST(TAN_P5, 3.33331568548E-1f);

_PS512_CONST(ASIN_P0, 4.2163199048E-2f);
_PS512_CONST(ASIN_P1, 2.4181311049E-2f);
_PS512_CONST(ASIN_P2, 4.5470025998E-2f);
_PS512_CONST(ASIN_P3, 7.4953002686E-2f);
_PS512_CONST(ASIN_P4, 1.6666752422E-1f);

_PS512_CONST(PIF, 3.14159265358979323846f);      // PI
_PS512_CONST(mPIF, -3.14159265358979323846f);    // -PI
_PS512_CONST(PIO2F, 1.57079632679489661923f);    // PI/2 1.570796326794896619
_PS512_CONST(mPIO2F, -1.57079632679489661923f);  // -PI/2 1.570796326794896619
_PS512_CONST(PIO4F, 0.785398163397448309615f);   // PI/4 0.7853981633974483096

_PS512_CONST(TANPI8F, 0.414213562373095048802f);   // tan(pi/8) => 0.4142135623730950
_PS512_CONST(TAN3PI8F, 2.414213562373095048802f);  // tan(3*pi/8) => 2.414213562373095

_PS512_CONST(ATAN_P0, 8.05374449538e-2f);
_PS512_CONST(ATAN_P1, -1.38776856032E-1f);
_PS512_CONST(ATAN_P2, 1.99777106478E-1f);
_PS512_CONST(ATAN_P3, -3.33329491539E-1f);

_PS512_CONST_TYPE(pos_sign_mask, int, (int) 0x7FFFFFFF);
_PS512_CONST_TYPE(neg_sign_mask, int, (int) ~0x7FFFFFFF);

_PS512_CONST(MAXLOGF, 88.72283905206835f);
_PS512_CONST(MAXLOGFDIV2, 44.361419526034176f);
_PS512_CONST(0p625, 0.625f);
_PS512_CONST(TANH_P0, -5.70498872745E-3f);
_PS512_CONST(TANH_P1, 2.06390887954E-2f);
_PS512_CONST(TANH_P2, -5.37397155531E-2f);
_PS512_CONST(TANH_P3, 1.33314422036E-1f);
_PS512_CONST(TANH_P4, -3.33332819422E-1f);

_PS512_CONST(MAXNUMF, 3.4028234663852885981170418348451692544e38f);
_PS512_CONST(minMAXNUMF, -3.4028234663852885981170418348451692544e38f);
_PS512_CONST(SINH_P0, 2.03721912945E-4f);
_PS512_CONST(SINH_P1, 8.33028376239E-3f);
_PS512_CONST(SINH_P2, 1.66667160211E-1f);

_PS512_CONST(1emin4, 1e-4f);
_PS512_CONST(ATANH_P0, 1.81740078349E-1f);
_PS512_CONST(ATANH_P1, 8.24370301058E-2f);
_PS512_CONST(ATANH_P2, 1.46691431730E-1f);
_PS512_CONST(ATANH_P3, 1.99782164500E-1f);
_PS512_CONST(ATANH_P4, 3.33337300303E-1f);

_PS512_CONST(1500, 1500.0f);
_PS512_CONST(LOGE2F, 0.693147180559945309f);
_PS512_CONST(ASINH_P0, 2.0122003309E-2f);
_PS512_CONST(ASINH_P1, -4.2699340972E-2f);
_PS512_CONST(ASINH_P2, 7.4847586088E-2f);
_PS512_CONST(ASINH_P3, -1.6666288134E-1f);

_PS512_CONST(ACOSH_P0, 1.7596881071E-3f);
_PS512_CONST(ACOSH_P1, -7.5272886713E-3f);
_PS512_CONST(ACOSH_P2, 2.6454905019E-2f);
_PS512_CONST(ACOSH_P3, -1.1784741703E-1f);
_PS512_CONST(ACOSH_P4, 1.4142135263E0f);

/* For log10f */
_PS512_CONST(cephes_L102A, 3.0078125E-1f);
_PS512_CONST(cephes_L102B, 2.48745663981195213739E-4f);
_PS512_CONST(cephes_L10EA, 4.3359375E-1f);
_PS512_CONST(cephes_L10EB, 7.00731903251827651129E-4f);

/* For log2f */
_PS512_CONST(cephes_LOG2EA, 0.44269504088896340735992f);

/* For cbrtf */
_PS512_CONST(cephes_CBRT2, 1.25992104989487316477f);
_PS512_CONST(cephes_CBRT4, 1.58740105196819947475f);
_PS512_CONST(cephes_invCBRT2, 0.7937005259840997373740956123328f);
_PS512_CONST(cephes_invCBRT4, 0.6299605249474365823842821870329f);
_PS512_CONST(CBRTF_P0, -0.13466110473359520655053f);
_PS512_CONST(CBRTF_P1, 0.54664601366395524503440f);
_PS512_CONST(CBRTF_P2, -0.95438224771509446525043f);
_PS512_CONST(CBRTF_P3, 1.1399983354717293273738f);
_PS512_CONST(CBRTF_P4, 0.40238979564544752126924f);


////////// DOUBLE /////////////
_PD512_CONST_TYPE(min_norm_pos, int64_t, 0x380ffff83ce549caL);
_PD512_CONST_TYPE(mant_mask, int64_t, 0xFFFFFFFFFFFFFL);
_PD512_CONST_TYPE(inv_mant_mask, int64_t, (int64_t)0x800FFFFFFFFFFFFFL);
_PD512_CONST_TYPE(sign_mask, int64_t, (int64_t) 0x8000000000000000L);
_PD512_CONST_TYPE(mid_mask, int64_t, (int64_t)  0x3fdfffffffffffffL);
_PD512_CONST_TYPE(inv_sign_mask, int64_t, ~0x8000000000000000L);

_PD512_CONST(minus_cephes_DP1, -7.85398125648498535156E-1);
_PD512_CONST(minus_cephes_DP2, -3.77489470793079817668E-8);
_PD512_CONST(minus_cephes_DP3, -2.69515142907905952645E-15);
_PD512_CONST(sincof_p0, 1.58962301576546568060E-10);
_PD512_CONST(sincof_p1, -2.50507477628578072866E-8);
_PD512_CONST(sincof_p2, 2.75573136213857245213E-6);
_PD512_CONST(sincof_p3, -1.98412698295895385996E-4);
_PD512_CONST(sincof_p4, 8.33333333332211858878E-3);
_PD512_CONST(sincof_p5, -1.66666666666666307295E-1);
_PD512_CONST(coscof_p0, -1.13585365213876817300E-11);
_PD512_CONST(coscof_p1, 2.08757008419747316778E-9);
_PD512_CONST(coscof_p2, -2.75573141792967388112E-7);
_PD512_CONST(coscof_p3, 2.48015872888517045348E-5);
_PD512_CONST(coscof_p4, -1.38888888888730564116E-3);
_PD512_CONST(coscof_p5, 4.16666666666665929218E-2);
_PD512_CONST(cephes_FOPI, 1.2732395447351626861510701069801148);  // 4 / M_PI

_PD512_CONST_TYPE(positive_mask, int64_t, (int64_t) 0x7FFFFFFFFFFFFFFFL);
_PD512_CONST_TYPE(negative_mask, int64_t, (int64_t) ~0x7FFFFFFFFFFFFFFFL);
_PD512_CONST_TYPE(pos_sign_mask, int64_t, 0xFFFFFFFFFFFFFL);

_PD512_CONST(ASIN_P0, 4.253011369004428248960E-3);
_PD512_CONST(ASIN_P1, -6.019598008014123785661E-1);
_PD512_CONST(ASIN_P2, 5.444622390564711410273E0);
_PD512_CONST(ASIN_P3, -1.626247967210700244449E1);
_PD512_CONST(ASIN_P4, 1.956261983317594739197E1);
_PD512_CONST(ASIN_P5, -8.198089802484824371615E0);

_PD512_CONST(PIF, 3.1415926535897932384626433832795028841971693993751058209749445923);      // PI
_PD512_CONST(mPIF, -3.1415926535897932384626433832795028841971693993751058209749445923);    // -PI
_PD512_CONST(PIO2F, 1.5707963267948966192313216916397514420985846996875529104874722961);    // PI/2 1.570796326794896619
_PD512_CONST(mPIO2F, -1.5707963267948966192313216916397514420985846996875529104874722961);  // -PI/2 1.570796326794896619
_PD512_CONST(PIO4F, 0.7853981633974483096156608458198757210492923498437764552437361480);    // PI/4 0.7853981633974483096

_PD512_CONST(ASIN_Q0, -1.474091372988853791896E1);
_PD512_CONST(ASIN_Q1, 7.049610280856842141659E1);
_PD512_CONST(ASIN_Q2, -1.471791292232726029859E2);
_PD512_CONST(ASIN_Q3, 1.395105614657485689735E2);
_PD512_CONST(ASIN_Q4, -4.918853881490881290097E1);

_PD512_CONST(ASIN_R0, 2.967721961301243206100E-3);
_PD512_CONST(ASIN_R1, -5.634242780008963776856E-1);
_PD512_CONST(ASIN_R2, 6.968710824104713396794E0);
_PD512_CONST(ASIN_R3, -2.556901049652824852289E1);
_PD512_CONST(ASIN_R4, 2.853665548261061424989E1);

_PD512_CONST(ASIN_S0, -2.194779531642920639778E1);
_PD512_CONST(ASIN_S1, 1.470656354026814941758E2);
_PD512_CONST(ASIN_S2, -3.838770957603691357202E2);
_PD512_CONST(ASIN_S3, 3.424398657913078477438E2);

_PD512_CONST(PIO2, 1.57079632679489661923);    /* pi/2 */
_PD512_CONST(PIO4, 7.85398163397448309616E-1); /* pi/4 */

_PD512_CONST(minMOREBITS, -6.123233995736765886130E-17);
_PD512_CONST(MOREBITS, 6.123233995736765886130E-17);

_PD512_CONST(ATAN_P0, -8.750608600031904122785E-1);
_PD512_CONST(ATAN_P1, -1.615753718733365076637E1);
_PD512_CONST(ATAN_P2, -7.500855792314704667340E1);
_PD512_CONST(ATAN_P3, -1.228866684490136173410E2);
_PD512_CONST(ATAN_P4, -6.485021904942025371773E1);

_PD512_CONST(ATAN_Q0, 2.485846490142306297962E1);
_PD512_CONST(ATAN_Q1, 1.650270098316988542046E2);
_PD512_CONST(ATAN_Q2, 4.328810604912902668951E2);
_PD512_CONST(ATAN_Q3, 4.853903996359136964868E2);
_PD512_CONST(ATAN_Q4, 1.945506571482613964425E2);

_PD512_CONST(TAN3PI8, 2.41421356237309504880); /* 3*pi/8 */

_PD512_CONST(min1, -1.0);
_PD512_CONST(min8, -8.0);
_PD512_CONST(min0p5, -0.5);
_PD512_CONST(1, 1.0);
_PD512_CONST(2, 2.0);
_PD512_CONST(0p5, 0.5);
_PD512_CONST(0p125, 0.125);
_PD512_CONST(PDEPI64U, 0x0010000000000000);
_PD512_CONST(min_212emin4, -2.121944400546905827679e-4);
_PD512_CONST(0p69, 0.693359375);

_PD512_CONST(1m14, 1.0e-14);
_PD512_CONST(TAN_P0, -1.30936939181383777646E4);
_PD512_CONST(TAN_P1, 1.15351664838587416140E6);
_PD512_CONST(TAN_P2, -1.79565251976484877988E7);
_PD512_CONST(TAN_Q0, 1.36812963470692954678E4);
_PD512_CONST(TAN_Q1, -1.32089234440210967447E6);
_PD512_CONST(TAN_Q2, 2.50083801823357915839E7);
_PD512_CONST(TAN_Q3, -5.38695755929454629881E7);
_PD512_CONST(TAN_mDP1, -7.853981554508209228515625E-1);
_PD512_CONST(TAN_mDP2, -7.94662735614792836714E-9);
_PD512_CONST(TAN_mDP3, -3.06161699786838294307E-17);
_PD512_CONST(tanlossth, 1.073741824e9);

_PD512_CONST(cephes_LOG2E, 1.4426950408889634073599);  /* 1/log(2) */
_PD512_CONST(cephes_LOGE2, 6.93147180559945309417E-1); /* log(2) */

_PD512_CONST(cephes_exp_p0, 1.26177193074810590878e-4);
_PD512_CONST(cephes_exp_p1, 3.02994407707441961300e-2);
_PD512_CONST(cephes_exp_p2, 9.99999999999999999910e-1);

_PD512_CONST(cephes_exp_q0, 3.00198505138664455042e-6);
_PD512_CONST(cephes_exp_q1, 2.52448340349684104192e-3);
_PD512_CONST(cephes_exp_q2, 2.27265548208155028766e-1);
_PD512_CONST(cephes_exp_q3, 2.00000000000000000009e0);

_PD512_CONST(cephes_exp_minC1, -0.693145751953125);
_PD512_CONST(cephes_exp_minC2, -1.42860682030941723212e-6);

_PD512_CONST(cephes_SQRTHF, 0.70710678118654752440);

_PD512_CONST(cephes_log_p0, 1.01875663804580931796E-4);
_PD512_CONST(cephes_log_p1, 4.97494994976747001425E-1);
_PD512_CONST(cephes_log_p2, 4.70579119878881725854E0);
_PD512_CONST(cephes_log_p3, 1.44989225341610930846E1);
_PD512_CONST(cephes_log_p4, 1.79368678507819816313E1);
_PD512_CONST(cephes_log_p5, 7.70838733755885391666E0);

_PD512_CONST(cephes_log_q0, 1.12873587189167450590E1);
_PD512_CONST(cephes_log_q1, 4.52279145837532221105E1);
_PD512_CONST(cephes_log_q2, 8.29875266912776603211E1);
_PD512_CONST(cephes_log_q3, 7.11544750618563894466E1);
_PD512_CONST(cephes_log_q4, 4.52279145837532221105E1);
_PD512_CONST(cephes_log_q5, 2.31251620126765340583E1);

_PD512_CONST(cephes_log_r0, -7.89580278884799154124E-1);
_PD512_CONST(cephes_log_r1, 1.63866645699558079767E1);
_PD512_CONST(cephes_log_r2, -6.41409952958715622951E1);

_PD512_CONST(cephes_log_s0, -3.56722798256324312549E1);
_PD512_CONST(cephes_log_s1, 3.12093766372244180303E2);
_PD512_CONST(cephes_log_s2, -7.69691943550460008604E2);

_PD512_CONST(exp_hi, 709.437);
_PD512_CONST(exp_lo, -709.436139303);

_PD512_CONST(0p625, 0.625);
_PD512_CONST(0p66, 0.66);
_PD512_CONST(1em8, 1E-8);
#endif


/// PRINT FUNCTIONS */
#if 0

#if defined(RISCV)

static inline void print_vec(V_ELT_FLOAT vec, int l)
{
    float observ[32];
    VSTORE_FLOAT(observ, vec, l);
    for (int i = 0; i < l; i++)
        printf("%3.4g ", observ[i]);
    printf("\n");
}

static inline void print_vech(V_ELT_FLOATH vec, int l)
{
    float observ[32];
    VSTORE_FLOATH(observ, vec, l);
    for (int i = 0; i < l; i++)
        printf("%g ", observ[i]);
    printf("\n");
}

static inline void print_vec64h(V_ELT_DOUBLEH vec, int l)
{
    double observ[16];
    VSTORE_DOUBLEH(observ, vec, l);
    for (int i = 0; i < l; i++)
        printf("%13.8g ", observ[i]);
    printf("\n");
}

static inline void print_vec_int(V_ELT_INT vec, int l)
{
    int observ[32];
    VSTORE_INT(observ, vec, l);
    for (int i = 0; i < l; i++)
        printf("%x ", observ[i]);
    printf("\n");
}

static inline void print_vec_inth(V_ELT_INTH vec, int l)
{
    int observ[32];
    VSTORE_INTH(observ, vec, l);
    for (int i = 0; i < l; i++)
        printf("%x ", observ[i]);
    printf("\n");
}

static inline void print_vec_int64h(V_ELT_INT64H vec, int l)
{
    long int observ[16];
    VSTORE_INT64H(observ, vec, l);
    for (int i = 0; i < l; i++)
        printf("%x ", observ[i]);
    printf("\n");
}

/*
static inline void print_bool4(vbool4_t vec)
{
    char observ[32];
    VSTORE_INT(observ, vec, 32);
    for (int i = 0; i < 32; i++)
        printf("%x ", observ[i]);
    printf("\n");
}
*/

static inline void print_vec_uint(V_ELT_UINT vec, int l)
{
    unsigned int observ[32];
    VSTORE_UINT(observ, vec, l);
    for (int i = 0; i < l; i++)
        printf("%x ", observ[i]);
    printf("\n");
}

#endif

#if defined(SVE2)

static inline void print_vec(V_ELT_FLOAT vec, V_ELT_BOOL64 l)
{
    float observ[32];
    VSTORE_FLOAT(observ, vec, l);
	V_ELT_FLOAT dummy;
	uint64_t numVals = svlen_f32(dummy);
    for (int i = 0; i < numVals; i++)
        printf("%3.4g ", observ[i]);
    printf("\n");
}

static inline void print_vec64(V_ELT_DOUBLE vec, V_ELT_BOOL64 l)
{
    double observ[16];
    VSTORE_DOUBLE(observ, vec, l);
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);	
    for (int i = 0; i < numVals; i++)
        printf("%13.8g ", observ[i]);
    printf("\n");
}

static inline void print_vec64x(V_ELT_DOUBLE vec, V_ELT_BOOL64 l)
{
    double observ[16];
    VSTORE_DOUBLE(observ, vec, l);
	V_ELT_DOUBLE dummy;
	uint64_t numVals = svlen_f64(dummy);	
    for (int i = 0; i < numVals; i++)
        printf("%16lx ", observ[i]);
    printf("\n");
}

static inline void print_vec_int(V_ELT_INT vec, V_ELT_BOOL64 l)
{
    int observ[32];
    VSTORE_INT(observ, vec, l);
	V_ELT_INT dummy;
	uint64_t numVals = svlen_s32(dummy);		
    for (int i = 0; i < numVals; i++)
        printf("%x ", observ[i]);
    printf("\n");
}

static inline void print_vec_int64(V_ELT_INT64 vec, V_ELT_BOOL64 l)
{
    long int observ[16];
    VSTORE_INT64(observ, vec, l);
	V_ELT_INT64 dummy;
	uint64_t numVals = svlen_s64(dummy);		
    for (int i = 0; i < numVals; i++)
        printf("%ld ", observ[i]);
    printf("\n");
}

static inline void print_vec_int64x(V_ELT_INT64 vec, V_ELT_BOOL64 l)
{
    long int observ[16];
    VSTORE_INT64(observ, vec, l);
	V_ELT_INT64 dummy;
	uint64_t numVals = svlen_s64(dummy);		
    for (int i = 0; i < numVals; i++)
        printf("%16lx ", observ[i]);
    printf("\n");
}

#endif

#if defined(SSE) || defined(ALTIVEC)

static inline void print4_4digits(v4sf v)
{
    float *p = (float *) &v;
#ifndef __SSE2__
#ifndef ALTIVEC
    _mm_empty();
#endif
#endif
    printf("[%3.4g, %3.4g, %3.4g, %3.4g]", p[0], p[1], p[2], p[3]);
    // printf("[%0.3f, %0.3f, %0.3f, %0.3f]", p[0], p[1], p[2], p[3]);
}

static inline void print4(v4sf v)
{
    float *p = (float *) &v;
#ifndef __SSE2__
#ifndef ALTIVEC
    _mm_empty();
#endif
#endif
    printf("[%3.24g, %3.24g, %3.24g, %3.24g]", p[0], p[1], p[2], p[3]);
    // printf("[%0.3f, %0.3f, %0.3f, %0.3f]", p[0], p[1], p[2], p[3]);
}

static inline void print4x(v4sf v)
{
    float *p = (float *) &v;
#ifndef __SSE2__
#ifndef ALTIVEC
    _mm_empty();
#endif
#endif
    printf("[%08x, %08x, %08x, %08x]", p[0], p[1], p[2], p[3]);
    // printf("[%0.3f, %0.3f, %0.3f, %0.3f]", p[0], p[1], p[2], p[3]);
}

static inline void print8xs(v8ss v)
{
    short *p = (short *) &v;
#ifndef __SSE2__
#ifndef ALTIVEC
    _mm_empty();
#endif
#endif
    printf("[%04x, %04x, %04x, %04x, %04x, %04x, %04x, %04x]",
           p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}

static inline void print16xu(v16u8 v)
{
    uint8_t *p = (uint8_t *) &v;
#ifndef __SSE2__
#ifndef ALTIVEC
    _mm_empty();
#endif
#endif
    printf("[%02x, %02x, %02x, %02x, %02x, %02x, %02x, %02x,%02x, %02x, %02x, %02x, %02x, %02x, %02x, %02x]",
           p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
           p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
}

static inline void print4i(v4si v)
{
    int *p = (int *) &v;
#ifndef __SSE2__
#ifndef ALTIVEC
    _mm_empty();
#endif
#endif
    printf("[%d, %d, %d, %d]", p[0], p[1], p[2], p[3]);
}

static inline void print4xi(v4si v)
{
    int *p = (int *) &v;
#ifndef __SSE2__
#ifndef ALTIVEC
    _mm_empty();
#endif
#endif
    printf("[%08x, %08x, %08x, %08x]", p[0], p[1], p[2], p[3]);
    // printf("[%0.3f, %0.3f, %0.3f, %0.3f]", p[0], p[1], p[2], p[3]);
}
#endif

#ifdef SSE
static inline void print4short(__m64 v)
{
    uint16_t *p = (uint16_t *) &v;
#ifndef __SSE2__
    _mm_empty();
#endif
    printf("[%u, %u, %u, %u]", p[0], p[1], p[2], p[3]);
}

static inline void print2(__m128d v)
{
    double *p = (double *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%13.8g, %13.8g]\n", p[0], p[1]);
}

static inline void print2x(__m128d v)
{
    double *p = (double *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%16lx, %16lx]\n", p[0], p[1]);
}

static inline void print2i(__m128i v)
{
    int64_t *p = (int64_t *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%ld, %ld]\n", p[0], p[1]);
}

static inline void print2xi(__m128i v)
{
    int64_t *p = (int64_t *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%16lx, %16lx]\n", p[0], p[1]);
}

#endif

#ifdef AVX
static inline void print8(__m256 v)
{
    float *p = (float *) &v;
#ifndef __SSE2__
    _mm_empty();
#endif
    printf("[%3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}

static inline void print4d(__m256d v)
{
    double *p = (double *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%13.8g, %13.8g, %13.8g, %13.8g]", p[0], p[1], p[2], p[3]);
}

static inline void print4id(__m256i v)
{
    int64_t *p = (int64_t *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%ld, %ld, %ld, %ld]", p[0], p[1], p[2], p[3]);
}

#endif

#ifdef AVX512
static inline void print16(__m512 v)
{
    float *p = (float *) &v;
    printf("[%3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g, %3.5g]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
}

static inline void print8d(__m512d v)
{
    double *p = (double *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%13.8g, %13.8g, %13.8g, %13.8g, %13.8g, %13.8g, %13.8g, %13.8g ]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}

static inline void print8id(__m512i v)
{
    int64_t *p = (int64_t *) &v;
#ifndef USE_SSE2
    _mm_empty();
#endif
    printf("[%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld ]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}

#endif

#endif
