/*
 * Project : SIMD_Utils
 * Version : 0.2.4
 * Author  : JishinMaster
 * Licence : BSD-2
 */

// TODO : look at scatter/gather/compress/decompress opcodes
#include <fenv.h>
#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef __linux__
/* Get current value of CLOCK and store it in TP.  */
int clock_gettime(clockid_t clock_id, struct timespec *tp)
{
    struct timeval tv;
    int retval = gettimeofday(&tv, NULL);
    if (retval == 0)
        /* Convert into `timespec'.  */
        TIMEVAL_TO_TIMESPEC(&tv, tp);
    return retval;
}
#endif

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

static inline uint32_t _MM_GET_ROUNDING_MODE()
{
    uint32_t reg;
    asm volatile("frrm %0"
                 : "=r"(reg));
    return reg;
}

static inline void _MM_SET_ROUNDING_MODE(uint32_t mode)
{
    uint32_t reg;

    switch (mode) {
    case _MM_ROUND_NEAREST:
        asm volatile("fsrmi %0,0"
                     : "=r"(reg));
        break;
    case _MM_ROUND_TOWARD_ZERO:  // trunc
        asm volatile("fsrmi %0,1"
                     : "=r"(reg));
        break;
    case _MM_ROUND_DOWN:
        asm volatile("fsrmi %0,2"
                     : "=r"(reg));
        break;
    case _MM_ROUND_UP:
        asm volatile("fsrmi %0,3"
                     : "=r"(reg));
        break;
    default:
        printf("_MM_SET_ROUNDING_MODE wrong mod requested %d\n", mode);
    }
}

#ifndef vfcvt_rtz_x_f_v_i64m8
#define NO_RTZ
#define vfcvt_rtz_x_f_v_i64m8 vfcvt_x_f_v_i64m8
#endif

#ifndef vfcvt_rtz_x_f_v_i32m8
#define NO_RTZ
#define vfcvt_rtz_x_f_v_i32m8 vfcvt_x_f_v_i32m8
#endif

// load vector float32, 8
// "1" in name means either vector scalar instructions, or load/store scalar to vector
#define VSETVL vsetvl_e32m8
#define V_ELT_FLOAT vfloat32m8_t
#define VLEV_FLOAT vle32_v_f32m8
#define VSEV_FLOAT vse32_v_f32m8
#define VLE1_FLOAT vfmv_v_f_f32m8
#define VADD_FLOAT vfadd_vv_f32m8
#define VADD1_FLOAT vfadd_vf_f32m8
#define VSUB_FLOAT vfsub_vv_f32m8
#define VSUB1_FLOAT vfsub_vf_f32m8
#define VMUL_FLOAT vfmul_vv_f32m8
#define VMUL1_FLOAT vfmul_vf_f32m8
#define VDIV_FLOAT vfdiv_vv_f32m8
#define VFMA_FLOAT vfmacc_vv_f32m8  // d = a + b*c
#define VFMA1_FLOAT vfmacc_vf_f32m8
#define VFMSUB_FLOAT vfmsub_vv_f32m8  // d = a*b - c
#define VREDSUM_FLOAT vfredosum_vs_f32m8_f32m1
#define VREDMAX_FLOAT vfredmax_vs_f32m8_f32m1
#define VREDMIN_FLOAT vfredmin_vs_f32m8_f32m1
#define VMIN_FLOAT vfmin_vv_f32m8
#define VMAX_FLOAT vfmax_vv_f32m8
#define VMIN1_FLOAT vfmin_vf_f32m8
#define VMAX1_FLOAT vfmax_vf_f32m8
#define VINTERP_FLOAT_INT vreinterpret_v_f32m8_i32m8
#define VINTERP_INT_FLOAT vreinterpret_v_i32m8_f32m8

#define VCVT_FLOAT_INT vfcvt_rtz_x_f_v_i32m8
#define VCVT_INT_FLOAT vfcvt_f_x_v_f32m8

#define VSETVL_DOUBLE vsetvl_e64m8
#define V_ELT_DOUBLE vfloat64m8_t
#define VLEV_DOUBLE vle64_v_f64m8
#define VLE1_DOUBLE vfmv_v_f_f64m8
#define VSEV_DOUBLE vse64_v_f64m8
#define VADD_DOUBLE vfadd_vv_f64m8
#define VADD1_DOUBLE vfadd_vf_f64m8
#define VSUB_DOUBLE vfsub_vv_f64m8
#define VSUB1_DOUBLE vfsub_vf_f64m8
#define VMUL_DOUBLE vfmul_vv_f64m8
#define VMUL1_DOUBLE vfmul_vf_f64m8
#define VDIV_DOUBLE vfdiv_vv_f64m8
#define VFMA_DOUBLE vfmacc_vv_f64m8  // d = a + b*c
#define VFMA1_DOUBLE vfmacc_vf_f64m8
#define VFMSUB_DOUBLE vfmsub_vv_f64m8  // d = a*b - c
#define VREDSUM_DOUBLE vfredosum_vs_f64m8_f64m1
#define VREDMAX_DOUBLE vfredmax_vs_f64m8_f64m1
#define VREDMIN_DOUBLE vfredmin_vs_f64m8_f64m1
#define VMIN_DOUBLE vfmin_vv_f64m8
#define VMAX_DOUBLE vfmax_vv_f64m8
#define VMIN1_DOUBLE vfmin_vf_f64m8
#define VMAX1_DOUBLE vfmax_vf_f64m8
#define VINTERP_DOUBLE_INT vreinterpret_v_f64m8_i64m8
#define VINTERP_INT_DOUBLE vreinterpret_v_i64m8_f64m8
#define VCVT_DOUBLE_INT vfcvt_rtz_x_f_v_i64m8
#define VCVT_INT_DOUBLE vfcvt_f_x_v_f64m8

#define V_ELT_INT vint32m8_t
#define VLEV_INT vle32_v_i32m8
#define VLE1_INT vmv_v_x_i32m8
#define VSEV_INT vse32_v_i32m8
#define VADD_INT vadd_vv_i32m8
#define VADD1_INT vadd_vx_i32m8
#define VMUL_INT vmul_vv_i32m8
#define VMUL1_INT vmul_vx_i32m8
#define VSUB_INT vsub_vv_i32m8
#define VSUB1_INT vsub_vx_i32m8
#define VAND_INT vand_vx_i32m8
#define VXOR_INT vxor_vv_i32m8

static inline void print_vec(V_ELT_FLOAT vec)
{
    float observ[32];
    VSEV_FLOAT(observ, vec, 32);
    for (int i = 0; i < 32; i++)
        printf("%0.3f ", observ[i]);
    printf("\n");
}

static inline void print_vec_int(V_ELT_INT vec)
{
    int observ[32];
    VSEV_INT(observ, vec, 32);
    for (int i = 0; i < 32; i++)
        printf("%x ", observ[i]);
    printf("\n");
}

/*
static inline void print_bool4(vbool4_t vec)
{
    char observ[32];
    VSEV_INT(observ, vec, 32);
    for (int i = 0; i < 32; i++)
        printf("%x ", observ[i]);
    printf("\n");
}
*/

static inline void print_vec_uint(vuint32m8_t vec)
{
    unsigned int observ[32];
    vse32_v_u32m8(observ, vec, 32);
    for (int i = 0; i < 32; i++)
        printf("%x ", observ[i]);
    printf("\n");
}
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        va = VLEV_FLOAT(a_tmp, i);
        vb = VLEV_FLOAT(b_tmp, i);
        vc = VADD_FLOAT(va, vb, i);
        VSEV_FLOAT(c_tmp, vc, i);

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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLEV_FLOAT(src_tmp, i);

        VSEV_FLOAT(dst_tmp, VADD1_FLOAT(va, value, i), i);
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        va = VLEV_FLOAT(a_tmp, i);
        vb = VLEV_FLOAT(b_tmp, i);
        vc = VMUL_FLOAT(va, vb, i);
        VSEV_FLOAT(c_tmp, vc, i);

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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        va = VLEV_FLOAT(a_tmp, i);
        vb = VLEV_FLOAT(b_tmp, i);
        vc = VDIV_FLOAT(va, vb, i);
        VSEV_FLOAT(c_tmp, vc, i);

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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        va = VLEV_FLOAT(a_tmp, i);
        vb = VLEV_FLOAT(b_tmp, i);
        vc = VSUB_FLOAT(va, vb, i);
        VSEV_FLOAT(c_tmp, vc, i);

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb, vc;
        va = VLEV_FLOAT(a_tmp, i);
        vb = VLEV_FLOAT(b_tmp, i);
        vc = VLEV_FLOAT(c_tmp, i);
        vc = VFMA_FLOAT(vc, va, vb, i);
        VSEV_FLOAT(dst_tmp, vc, i);

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vc;
        va = VLEV_FLOAT(a_tmp, i);
        vc = VLEV_FLOAT(c_tmp, i);
        vc = VFMA1_FLOAT(vc, b, va, i);
        VSEV_FLOAT(dst_tmp, vc, i);

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vc;
        va = VLEV_FLOAT(a_tmp, i);
        vc = VLE1_FLOAT(c, i);
        vc = VFMA1_FLOAT(vc, b, va, i);
        VSEV_FLOAT(dst_tmp, vc, i);

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb, vc;
        va = VLEV_FLOAT(a_tmp, i);
        vb = VLEV_FLOAT(b_tmp, i);
        vc = VLE1_FLOAT(c, i);
        vc = VFMA_FLOAT(vc, va, vb, i);
        VSEV_FLOAT(dst_tmp, vc, i);

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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLEV_FLOAT(src_tmp, i);

        VSEV_FLOAT(dst_tmp, VMUL1_FLOAT(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLEV_FLOAT(src_tmp, i);

        V_ELT_FLOAT xmm3, sign_bit, y;
        V_ELT_INT emm0, emm2;
        sign_bit = x;

        /* take the absolute value */
        x = VINTERP_INT_FLOAT(VAND_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

        /* extract the sign bit (upper one) */
        // not 0 if input < 0
        V_ELT_INT sign_bit_int = VAND_INT(VINTERP_FLOAT_INT(sign_bit), sign_mask, i);

        /* scale by 4/Pi */
        y = VMUL1_FLOAT(x, FOPI, i);

        /* store the integer part of y in mm0 */
        emm2 = VCVT_FLOAT_INT(y, i);

        /* j=(j+1) & (~1) (see the cephes sources) */
        emm2 = VADD1_INT(emm2, 1, i);
        emm2 = VAND_INT(emm2, ~1, i);
        y = VCVT_INT_FLOAT(emm2, i);

        /* get the swap sign flag */
        emm0 = VAND_INT(emm2, 4, i);
        emm0 = vsll_vx_i32m8(emm0, 29, i);

        /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
         */
        emm2 = VAND_INT(emm2, 2, i);

        /// emm2 == 0 ? 0xFFFFFFFF : 0x00000000
        vbool4_t poly_mask = vmseq_vx_i32m8_b4(emm2, 0, i);
        // vbool4_t not_poly_mask=vmnot_m_b4(poly_mask, i);

        sign_bit_int = VXOR_INT(sign_bit_int, emm0, i);  // emm0 is swap_sign_bit

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMA1_FLOAT(x, minus_cephes_DP1, y, i);
        x = VFMA1_FLOAT(x, minus_cephes_DP2, y, i);
        x = VFMA1_FLOAT(x, minus_cephes_DP3, y, i);

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
        y = vmerge_vvm_f32m8(poly_mask, y, y2, i);

        /* update the sign */
        y = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(y), sign_bit_int, i));

        VSEV_FLOAT(dst_tmp, y, i);

        src_tmp += i;
        dst_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

// Work in progress; could use fma?
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLEV_FLOAT(src_tmp, i);

        V_ELT_FLOAT y;
        V_ELT_INT j;
        vbool4_t jandone, jsup3, jsup1, j1or2, xinf0;
        vbool4_t sign_sin, sign_cos;

        sign_sin = vmclr_m_b4(i);
        sign_cos = vmclr_m_b4(i);

        // if (x < 0)
        xinf0 = vmflt_vf_f32m8_b4(x, 0.0f, i);
        sign_sin = vmxor_mm_b4(sign_sin, xinf0, i);

        /* take the absolute value */
        x = VINTERP_INT_FLOAT(VAND_INT(VINTERP_FLOAT_INT(x), inv_sign_mask, i));

        /* scale by 4/Pi */
        y = VMUL1_FLOAT(x, FOPI, i);

        /* store the integer part of y in mm2 */
        j = VCVT_FLOAT_INT(y, i);

        // if (j&1))
        jandone = vmsne_vx_i32m8_b4(VAND_INT(j, 1, i), 0, i);
        j = vadd_vx_i32m8_m(jandone, j, j, 1, i);
        y = VCVT_INT_FLOAT(j, i);

        // j&=7
        j = VAND_INT(j, 7, i);

        // if (j > 3)
        jsup3 = vmsgt_vx_i32m8_b4(j, 3, i);
        sign_sin = vmxor_mm_b4(sign_sin, jsup3, i);
        sign_cos = vmxor_mm_b4(sign_cos, jsup3, i);
        j = vsub_vx_i32m8_m(jsup3, j, j, 4, i);

        // if (j > 1)
        jsup1 = vmsgt_vx_i32m8_b4(j, 1, i);
        sign_cos = vmxor_mm_b4(sign_cos, jsup1, i);

        j1or2 = vmor_mm_b4(vmseq_vx_i32m8_b4(j, 1, i),
                           vmseq_vx_i32m8_b4(j, 2, i), i);

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = VFMA1_FLOAT(x, minus_cephes_DP1, y, i);
        x = VFMA1_FLOAT(x, minus_cephes_DP2, y, i);
        x = VFMA1_FLOAT(x, minus_cephes_DP3, y, i);

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
        V_ELT_FLOAT y_sin = vmerge_vvm_f32m8(j1or2, y2, y, i);
        V_ELT_FLOAT y_cos = vmerge_vvm_f32m8(j1or2, y, y2, i);

        y_sin = vfmul_vf_f32m8_m(sign_sin, y_sin, y_sin, -1.0f, i);
        y_cos = vfmul_vf_f32m8_m(sign_cos, y_cos, y_cos, -1.0f, i);

        VSEV_FLOAT(s_tmp, y_sin, i);
        VSEV_FLOAT(c_tmp, y_cos, i);

        src_tmp += i;
        s_tmp += i;
        c_tmp += i;
    }

#ifdef NO_RTZ
    _MM_SET_ROUNDING_MODE(reg_ori);
#endif
}

static inline void sumf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    i = VSETVL(len);
    V_ELT_FLOAT vacc = VLE1_FLOAT(0.0f, i);

#if 1
    vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, i);
    size_t i_last;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLEV_FLOAT(src_tmp, i);
        vacc = VADD_FLOAT(vacc, va, i);
        src_tmp += i;
        i_last = i;
    }

    acc = vfredosum_vs_f32m8_f32m1(acc, vacc, acc, i_last);
    vse32_v_f32m1(dst, acc, 1);
#else
    float acc[32];
    int len_ori = len;
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLEV_FLOAT(src_tmp, i);
        vacc = VADD_FLOAT(vacc, va, i);
        src_tmp += i;
    }

    size_t vlen_ori = VSETVL(len_ori);
    VSEV_FLOAT(acc, vacc, len_ori);
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
    i = VSETVL(len);
    V_ELT_FLOAT vacc = VLE1_FLOAT(0.0f, i);

    vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, i);
    size_t i_last;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLEV_FLOAT(src_tmp1, i);
        V_ELT_FLOAT vb = VLEV_FLOAT(src_tmp2, i);
        vacc = VFMA_FLOAT(vacc, va, vb, i);
        src_tmp1 += i;
        src_tmp2 += i;
        i_last = i;
    }

    acc = vfredosum_vs_f32m8_f32m1(acc, vacc, acc, i_last);
    vse32_v_f32m1(dst, acc, 1);
}

static inline void dotcf_vec(complex32_t *src1, complex32_t *src2, int len, complex32_t *dst)
{
    size_t i;
    float *src1_tmp = (float *) src1;
    float *src2_tmp = (float *) src2;
    int cplx_len = 2 * len;

    i = VSETVL(cplx_len);
    vfloat32m4_t vacc_Re = vfmv_v_f_f32m4(0.0f, i);
    vfloat32m4_t vacc_Im = vfmv_v_f_f32m4(0.0f, i);

    vfloat32m1_t acc_Re = vfmv_v_f_f32m1(0.0f, i);
    vfloat32m1_t acc_Im = vfmv_v_f_f32m1(0.0f, i);
    size_t i_last;

    int vec_size = VSETVL(4096);
    int nb_elts = 0;

    for (; (i = VSETVL(cplx_len)) >= vec_size; cplx_len -= i) {
        vfloat32m4_t src1Re_vec;
        vfloat32m4_t src1Im_vec;
        vfloat32m4_t src2Re_vec;
        vfloat32m4_t src2Im_vec;
        vlseg2e32_v_f32m4(&src1Re_vec, &src1Im_vec, src1_tmp, i);
        vlseg2e32_v_f32m4(&src2Re_vec, &src2Im_vec, src2_tmp, i);
        vfloat32m4_t tmp1 = vfmul_vv_f32m4(src1Im_vec, src2Im_vec, i);
        vfloat32m4_t dstRe_vec = vfmsub_vv_f32m4(src1Re_vec, src2Re_vec, tmp1, i);
        vfloat32m4_t tmp2 = vfmul_vv_f32m4(src1Re_vec, src2Im_vec, i);
        vfloat32m4_t dstIm_vec = vfmacc_vv_f32m4(tmp2, src2Re_vec, src1Im_vec, i);
        vacc_Re = vfadd_vv_f32m4(vacc_Re, dstRe_vec, i);
        vacc_Im = vfadd_vv_f32m4(vacc_Im, dstIm_vec, i);
        src1_tmp += i;
        src2_tmp += i;
        i_last = i;
        nb_elts += vec_size;
    }

    acc_Re = vfredosum_vs_f32m4_f32m1(acc_Re, vacc_Re, acc_Re, i_last);
    acc_Im = vfredosum_vs_f32m4_f32m1(acc_Im, vacc_Im, acc_Im, i_last);
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

    for (; (i = VSETVL(cplx_len)) > 0; cplx_len -= i) {
        vfloat32m4_t dstRe_vec;
        vfloat32m4_t dstIm_vec;
        vlseg2e32_v_f32m4(&dstRe_vec, &dstIm_vec, src_tmp, i);
        vse32_v_f32m4(dstRe_tmp, dstRe_vec, i);
        vse32_v_f32m4(dstIm_tmp, dstIm_vec, i);
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

    for (; (i = VSETVL(cplx_len)) > 0; cplx_len -= i) {
        vfloat32m4_t srcRe_vec = vle32_v_f32m4(srcRe_tmp, i);
        vfloat32m4_t srcIm_vec = vle32_v_f32m4(srcIm_tmp, i);
        vsseg2e32_v_f32m4(dst_tmp, srcRe_vec, srcIm_vec, i);
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

    int vec_size = VSETVL(4096);
    int nb_elts = 0;

    for (; (i = VSETVL(cplx_len)) >= vec_size; cplx_len -= i) {
        vfloat32m4_t src1Re_vec;
        vfloat32m4_t src1Im_vec;
        vfloat32m4_t src2Re_vec;
        vfloat32m4_t src2Im_vec;
        vlseg2e32_v_f32m4(&src1Re_vec, &src1Im_vec, src1_tmp, i);
        vlseg2e32_v_f32m4(&src2Re_vec, &src2Im_vec, src2_tmp, i);
        vfloat32m4_t tmp1 = vfmul_vv_f32m4(src1Im_vec, src2Im_vec, i);
        vfloat32m4_t dstRe_vec = vfmsub_vv_f32m4(src1Re_vec, src2Re_vec, tmp1, i);
        vfloat32m4_t tmp2 = vfmul_vv_f32m4(src1Re_vec, src2Im_vec, i);
        vfloat32m4_t dstIm_vec = vfmacc_vv_f32m4(tmp2, src2Re_vec, src1Im_vec, i);

        vsseg2e32_v_f32m4(dst_tmp, dstRe_vec, dstIm_vec, i);
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT src1Re_vec = VLEV_FLOAT(src1Re_tmp, i);
        V_ELT_FLOAT src1Im_vec = VLEV_FLOAT(src1Im_tmp, i);
        V_ELT_FLOAT src2Re_vec = VLEV_FLOAT(src2Re_tmp, i);
        V_ELT_FLOAT src2Im_vec = VLEV_FLOAT(src2Im_tmp, i);

        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src1Im_vec, src2Im_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMSUB_FLOAT(src1Re_vec, src2Re_vec, tmp1, i);
        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMA_FLOAT(tmp2, src2Re_vec, src1Im_vec, i);
        VSEV_FLOAT(dstRe_tmp, dstRe_vec, i);
        VSEV_FLOAT(dstIm_tmp, dstIm_vec, i);

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

    int vec_size = VSETVL(4096);
    int nb_elts = 0;

    for (; (i = VSETVL(cplx_len)) >= vec_size; cplx_len -= i) {
        vfloat32m4_t src1Re_vec;
        vfloat32m4_t src1Im_vec;
        vfloat32m4_t src2Re_vec;
        vfloat32m4_t src2Im_vec;
        vlseg2e32_v_f32m4(&src1Re_vec, &src1Im_vec, src1_tmp, i);
        vlseg2e32_v_f32m4(&src2Re_vec, &src2Im_vec, src2_tmp, i);

        vfloat32m4_t tmp1 = vfmul_vv_f32m4(src2Re_vec, src2Re_vec, i);
        vfloat32m4_t c2d2 = vfmacc_vv_f32m4(tmp1, src2Im_vec, src2Im_vec, i);

        vfloat32m4_t tmp2 = vfmul_vv_f32m4(src1Re_vec, src2Re_vec, i);
        vfloat32m4_t dstRe_vec = vfmacc_vv_f32m4(tmp2, src1Im_vec, src2Im_vec, i);
        dstRe_vec = vfdiv_vv_f32m4(dstRe_vec, c2d2, i);

        vfloat32m4_t tmp3 = vfmul_vv_f32m4(src1Re_vec, src2Im_vec, i);
        vfloat32m4_t dstIm_vec = vfmsub_vv_f32m4(src2Re_vec, src1Im_vec, tmp3, i);
        dstIm_vec = vfdiv_vv_f32m4(dstIm_vec, c2d2, i);

        vsseg2e32_v_f32m4(dst_tmp, dstRe_vec, dstIm_vec, i);
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT src1Re_vec = VLEV_FLOAT(src1Re_tmp, i);
        V_ELT_FLOAT src1Im_vec = VLEV_FLOAT(src1Im_tmp, i);
        V_ELT_FLOAT src2Re_vec = VLEV_FLOAT(src2Re_tmp, i);
        V_ELT_FLOAT src2Im_vec = VLEV_FLOAT(src2Im_tmp, i);

        V_ELT_FLOAT tmp1 = VMUL_FLOAT(src2Re_vec, src2Re_vec, i);
        V_ELT_FLOAT c2d2 = VFMA_FLOAT(tmp1, src2Im_vec, src2Im_vec, i);

        V_ELT_FLOAT tmp2 = VMUL_FLOAT(src1Re_vec, src2Re_vec, i);
        V_ELT_FLOAT dstRe_vec = VFMA_FLOAT(tmp2, src1Im_vec, src2Im_vec, i);
        dstRe_vec = VDIV_FLOAT(dstRe_vec, c2d2, i);

        V_ELT_FLOAT tmp3 = VMUL_FLOAT(src1Re_vec, src2Im_vec, i);
        V_ELT_FLOAT dstIm_vec = VFMSUB_FLOAT(src2Re_vec, src1Im_vec, tmp3, i);
        dstIm_vec = VDIV_FLOAT(dstIm_vec, c2d2, i);
        VSEV_FLOAT(dstRe_tmp, dstRe_vec, i);
        VSEV_FLOAT(dstIm_tmp, dstIm_vec, i);

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

    int vec_size = VSETVL(4096);
    int nb_elts = 0;

    for (; (i = VSETVL(cplx_len)) >= vec_size; cplx_len -= i) {
        vfloat32m4_t srcRe_vec;
        vfloat32m4_t srcIm_vec;
        vlseg2e32_v_f32m4(&srcRe_vec, &srcIm_vec, src_tmp, i);
        srcIm_vec = vreinterpret_v_i32m4_f32m4(vxor_vx_i32m4(vreinterpret_v_f32m4_i32m4(srcIm_vec), (int32_t) 0x80000000, i));
        vsseg2e32_v_f32m4(dst_tmp, srcRe_vec, srcIm_vec, i);
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT re_tmp = VLEV_FLOAT(srcRe_tmp, i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT im_tmp = VLEV_FLOAT(srcIm_tmp, i);
        V_ELT_FLOAT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);

        VSEV_FLOAT(dst_tmp, vfsqrt_v_f32m8(tmp, i), i);

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT re_tmp = VLEV_FLOAT(srcRe_tmp, i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT im_tmp = VLEV_FLOAT(srcIm_tmp, i);
        V_ELT_FLOAT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);

        VSEV_FLOAT(dst_tmp, tmp, i);

        srcRe_tmp += i;
        srcIm_tmp += i;
        dst_tmp += i;
    }
}

// should we use vlseg2e32_v_f32m4?
static inline void powerspectf_interleaved_vec(complex32_t *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = (float *) src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        // complex are a + ib, c + id, e + if, etc
        // load in re_tmp a,c,e, etc => i elements in range 0..2*i with a stride of 2
        // load in im_tmp b,d,f, etc => i elements in range 0..2*i with a stride of 2
        V_ELT_FLOAT re_tmp = vlse32_v_f32m8(src_tmp, 2 * sizeof(float), i);
        V_ELT_FLOAT im_tmp = vlse32_v_f32m8(src_tmp + 1, 2 * sizeof(float), i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);
        VSEV_FLOAT(dst_tmp, tmp, i);

        // src_tmp increases twice as fast since it's complex and not float
        src_tmp += 2 * i;
        dst_tmp += i;
    }
}

static inline void magnitudef_interleaved_vec(complex32_t *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = (float *) src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        // complex are a + ib, c + id, e + if, etc
        // load in re_tmp a,c,e, etc => i elements in range 0..2*i with a stride of 2
        // load in im_tmp b,d,f, etc => i elements in range 0..2*i with a stride of 2
        V_ELT_FLOAT re_tmp = vlse32_v_f32m8(src_tmp, 2 * sizeof(float), i);
        V_ELT_FLOAT im_tmp = vlse32_v_f32m8(src_tmp + 1, 2 * sizeof(float), i);
        V_ELT_FLOAT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT_FLOAT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);
        VSEV_FLOAT(dst_tmp, vfsqrt_v_f32m8(tmp, i), i);

        // src_tmp increases twice as fast since it's complex and not float
        src_tmp += 2 * i;
        dst_tmp += i;
    }
}

static inline void maxeveryf_vec(float *src1, float *src2, float *dst, int len)
{
    size_t i;
    float *src1_tmp = src1;
    float *src2_tmp = src2;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb;
        va = VLEV_FLOAT(src1_tmp, i);
        vb = VLEV_FLOAT(src2_tmp, i);
        VSEV_FLOAT(dst_tmp, VMAX_FLOAT(va, vb, i), i);

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va, vb;
        va = VLEV_FLOAT(src1_tmp, i);
        vb = VLEV_FLOAT(src2_tmp, i);
        VSEV_FLOAT(dst_tmp, VMIN_FLOAT(va, vb, i), i);

        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void minmaxf_vec(float *src, int len, float *min_value, float *max_value)
{
    size_t i, i_last;

    float *src_tmp = src;

    i = VSETVL(len);

    vfloat32m1_t min0 = vle32_v_f32m1(src_tmp, 1);  // or vfmv_v_f_f32m1
    vfloat32m1_t max0 = min0;
    V_ELT_FLOAT minv, maxv, v1;
    minv = VLEV_FLOAT(src_tmp, i);
    maxv = minv;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        v1 = VLEV_FLOAT(src_tmp, i);
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLEV_FLOAT(src_tmp, i);
        VSEV_FLOAT(dst_tmp, VMIN1_FLOAT(va, value, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_lt_f_vec(float *src, float *dst, int len, float value)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLEV_FLOAT(src_tmp, i);
        VSEV_FLOAT(dst_tmp, VMAX1_FLOAT(va, value, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_gtabs_f_vec(float *src, float *dst, int len, float value)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLEV_FLOAT(src_tmp, i);

#if 1
        V_ELT_INT va_sign = VAND_INT(VINTERP_FLOAT_INT(va), sign_mask, i);
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        V_ELT_FLOAT sval = VMIN1_FLOAT(va_abs, value, i);
        sval = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(sval), va_sign, i));
        VSEV_FLOAT(dst_tmp, sval, i);
#else  // should be removed?
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        vbool4_t eqmask = vmfeq_vv_f32m8_b4(va, va_abs, i);
        vbool4_t gtmask = vmfgt_vf_f32m8_b4(va_abs, value, i);

        V_ELT_FLOAT sval;
        sval = vfmerge_vfm_f32m8(vmnot_m_b4(eqmask, i), sval, -value, i);
        sval = vfmerge_vfm_f32m8(eqmask, sval, value, i);
        VSEV_FLOAT(dst_tmp, vmerge_vvm_f32m8(gtmask, va, sval, i), i);
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLEV_FLOAT(src_tmp, i);
        V_ELT_INT va_sign = VAND_INT(VINTERP_FLOAT_INT(va), sign_mask, i);
        V_ELT_FLOAT va_abs = VINTERP_INT_FLOAT(VAND_INT(VINTERP_FLOAT_INT(va), inv_sign_mask, i));
        V_ELT_FLOAT sval = VMAX1_FLOAT(va_abs, value, i);
        sval = VINTERP_INT_FLOAT(VXOR_INT(VINTERP_FLOAT_INT(sval), va_sign, i));
        VSEV_FLOAT(dst_tmp, sval, i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_ltval_gtval_f_vec(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va = VLEV_FLOAT(src_tmp, i);
        vbool4_t lt_mask = vmflt_vf_f32m8_b4(va, ltlevel, i);
        vbool4_t gt_mask = vmfgt_vf_f32m8_b4(va, gtlevel, i);
        V_ELT_FLOAT tmp = vfmerge_vfm_f32m8(lt_mask, va, ltvalue, i);
        tmp = vfmerge_vfm_f32m8(gt_mask, tmp, gtvalue, i);
        VSEV_FLOAT(dst_tmp, tmp, i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void sqrtf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLEV_FLOAT(src_tmp, i);
        VSEV_FLOAT(dst_tmp, vfsqrt_v_f32m8(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void fabsf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT va;
        va = VLEV_FLOAT(src_tmp, i);
        VSEV_FLOAT(dst_tmp, vfabs_v_f32m8(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void log10_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    i = VSETVL(len);
    V_ELT_FLOAT zero_vec = VLE1_FLOAT(0.0f, i);
    V_ELT_FLOAT one_vec = VLE1_FLOAT(1.0f, i);

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT x = VLEV_FLOAT(src_tmp, i);
        V_ELT_INT imm0;

        vbool4_t invalid_mask = vmfle_vf_f32m8_b4(x, 0.0f, i);
        x = VMAX1_FLOAT(x, 1.17549e-38f, i); /* cut off denormalized stuff */
        imm0 = vsra_vx_i32m8(VINTERP_FLOAT_INT(x), 23, i);

        /* keep only the fractional part */
        x = VINTERP_INT_FLOAT(VAND_INT(VINTERP_FLOAT_INT(x), c_inv_mant_mask, i));
        // 0x3f000000 is the hex representation of 0.5f
        x = VINTERP_INT_FLOAT(vor_vx_i32m8(VINTERP_FLOAT_INT(x), 0x3f000000, i));
        imm0 = vsub_vx_i32m8(imm0, 0x7f, i);
        V_ELT_FLOAT e = VCVT_INT_FLOAT(imm0, i);
        e = VADD1_FLOAT(e, 1.0f, i);

        // could lead to errors since we take the inverted mask after?
        vbool4_t mask = vmflt_vf_f32m8_b4(x, c_cephes_SQRTHF, i);

        V_ELT_FLOAT tmp = vfmerge_vfm_f32m8(vmnot_m_b4(mask, i), x, 0.0f, i);
        x = VSUB1_FLOAT(x, 1.0f, i);  // x ok

        // substract 1.0f if mask is true (x < SQRTHF). To be optimised
        e = VSUB_FLOAT(e, vfmerge_vfm_f32m8(mask, zero_vec, 1.0f, i), i);
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
        y = vfmadd_vf_f32m8(z, -0.5f, y, i);  // y = y -0.5*z

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
        // could we use merge function? vmerge_vvm_f32m8? create a nan vec?
        x = vfmerge_vfm_f32m8(invalid_mask, x, 0xFFFFFFFF, i);

        VSEV_FLOAT(dst_tmp, x, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void vectorSlopef_vec(float *dst, int len, float offset, float slope)
{
    size_t i;
    float *dst_tmp = dst;

    float coef_max[32];

    // to be improved!
    for (int s = 0; s < 32; s++) {
        coef_max[s] = (float) (s) *slope;
    }

    i = VSETVL(len);

    V_ELT_FLOAT coef = VLEV_FLOAT(coef_max, i);
    V_ELT_FLOAT slope_vec = VLE1_FLOAT((float) (i) *slope, i);
    V_ELT_FLOAT curVal = VADD1_FLOAT(coef, offset, i);

    VSEV_FLOAT(dst_tmp, curVal, i);
    dst_tmp += i;
    len -= i;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        curVal = VADD_FLOAT(curVal, slope_vec, i);
        VSEV_FLOAT(dst_tmp, curVal, i);
        dst_tmp += i;
    }
}

static inline void setf_vec(float *dst, float value, int len)
{
    size_t i;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        VSEV_FLOAT(dst_tmp, VLE1_FLOAT(value, i), i);
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        VSEV_FLOAT(dst_tmp, VLEV_FLOAT(src_tmp, i), i);
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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT src_vec, integer_vec, remainer_vec;
        src_vec = VLEV_FLOAT(src_tmp, i);
        integer_vec = vfcvt_f_x_v_f32m8(vfcvt_rtz_x_f_v_i32m8(src_vec, i), i);
        VSEV_FLOAT(integer_tmp, integer_vec, i);
        remainer_vec = VSUB_FLOAT(src_vec, integer_vec, i);
        VSEV_FLOAT(remainder_tmp, remainer_vec, i);
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_FLOAT a, b;
        a = VLEV_FLOAT(src_tmp, i);
        b = vfcvt_f_x_v_f32m8(vfcvt_x_f_v_i32m8(a, i), i);
        VSEV_FLOAT(dst_tmp, b, i);
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

#if 1
static inline void flipf_vec(float *src, float *dst, int len)
{
    size_t i, i_last;
    i = VSETVL(len);
    int vec_size = VSETVL(4096);
    float *src_tmp = src + len - i;
    float *dst_tmp = dst;

    // max vector size is 1024bits, but could be less (128bits on C906 core)
    uint32_t index[32] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
                          19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                          6, 5, 4, 3, 2, 1, 0};

    vuint32m8_t index_vec = vle32_v_u32m8(index + 32 - vec_size, i);
    V_ELT_FLOAT a, b;
    for (; (i = VSETVL(len)) >= vec_size; len -= i) {
        a = VLEV_FLOAT(src_tmp, i);
        b = vrgather_vv_f32m8(a, index_vec, i);
        VSEV_FLOAT(dst_tmp, b, i);
        src_tmp -= i;
        dst_tmp += i;
        i_last = i;
    }

    if (i_last) {
        index_vec = vle32_v_u32m8(index + 32 - i_last, i_last);
        a = VLEV_FLOAT(src_tmp, i_last);
        b = vrgather_vv_f32m8(a, index_vec, i_last);
        VSEV_FLOAT(dst_tmp, b, i_last);
    }
}
#else
// could be improved
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

static inline void copys_vec(int32_t *src, int32_t *dst, int len)
{
    size_t i;
    int32_t *src_tmp = src;
    int32_t *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        VSEV_INT(dst_tmp, VLEV_INT(src_tmp, i), i);
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLEV_INT(src1_tmp, i);
        vb = VLEV_INT(src2_tmp, i);

        VSEV_INT(dst_tmp, VSUB_INT(va, vb, i), i);
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLEV_INT(src1_tmp, i);
        vb = VLEV_INT(src2_tmp, i);

        VSEV_INT(dst_tmp, VADD_INT(va, vb, i), i);
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_INT va;
        va = VLEV_INT(src_tmp, i);

        VSEV_INT(dst_tmp, VADD1_INT(va, value, i), i);
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_INT va, vb;
        va = VLEV_INT(src1_tmp, i);
        vb = VLEV_INT(src2_tmp, i);

        VSEV_INT(dst_tmp, VMUL_INT(va, vb, i), i);
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
    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT_INT va;
        va = VLEV_INT(src_tmp, i);

        VSEV_INT(dst_tmp, VMUL1_INT(va, value, i), i);
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

    i = VSETVL(len);

    V_ELT_INT coef = VLEV_INT(coef_max, i);
    V_ELT_INT slope_vec = VLE1_INT((int32_t) (i) *slope, i);
    V_ELT_INT curVal = VADD1_INT(coef, offset, i);

    VSEV_INT(dst_tmp, curVal, i);
    dst_tmp += i;
    len -= i;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        curVal = VADD_INT(curVal, slope_vec, i);
        VSEV_INT(dst_tmp, curVal, i);
        dst_tmp += i;
    }
}

static inline void roundd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;
    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE a, b;
        a = VLEV_DOUBLE(src_tmp, i);
        b = vfcvt_f_x_v_f64m8(vfcvt_x_f_v_i64m8(a, i), i);
        VSEV_DOUBLE(dst_tmp, b, i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void ceild_vec(double *src, double *dst, int len)
{
    // 0 to nearest, 1 to zero (trunc), 2 round down, 3 round up, 4 round to nearest
    uint32_t reg = 0, reg_ori;
    asm volatile("frrm %0"
                 : "=r"(reg_ori));
    asm volatile("fsrmi %0,3"
                 : "=r"(reg));
    roundd_vec(src, dst, len);
    asm volatile("fsrm %0,%1"
                 : "=r"(reg), "=r"(reg_ori));
    asm volatile("frrm %0"
                 : "=r"(reg_ori));
}

static inline void floord_vec(double *src, double *dst, int len)
{
    // 0 to nearest, 1 to zero (trunc), 2 round down, 3 round up, 4 round to nearest
    uint32_t reg = 0, reg_ori;
    asm volatile("frrm %0"
                 : "=r"(reg_ori));
    asm volatile("fsrmi %0,2"
                 : "=r"(reg));
    roundd_vec(src, dst, len);
    asm volatile("fsrm %0,%1"
                 : "=r"(reg), "=r"(reg_ori));
    asm volatile("frrm %0"
                 : "=r"(reg_ori));
}

static inline void truncd_vec(double *src, double *dst, int len)
{
    // 0 to nearest, 1 to zero (trunc), 2 round down, 3 round up, 4 round to nearest
    uint32_t reg = 0, reg_ori;
    asm volatile("frrm %0"
                 : "=r"(reg_ori));
    asm volatile("fsrmi %0,1"
                 : "=r"(reg));
    roundd_vec(src, dst, len);
    asm volatile("fsrm %0,%1"
                 : "=r"(reg), "=r"(reg_ori));
    asm volatile("frrm %0"
                 : "=r"(reg_ori));
}

void addd_vec(double *a, double *b, double *c, int len)
{
    size_t i;
    double *a_tmp = a;
    double *b_tmp = b;
    double *c_tmp = c;
    V_ELT_DOUBLE va, vb, vc;
    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        va = VLEV_DOUBLE(a_tmp, i);
        vb = VLEV_DOUBLE(b_tmp, i);
        vc = VADD_DOUBLE(va, vb, i);
        VSEV_DOUBLE(c_tmp, vc, i);

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
    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLEV_DOUBLE(src_tmp, i);

        VSEV_DOUBLE(dst_tmp, VADD1_DOUBLE(va, value, i), i);
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
    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        va = VLEV_DOUBLE(a_tmp, i);
        vb = VLEV_DOUBLE(b_tmp, i);
        vc = VMUL_DOUBLE(va, vb, i);
        VSEV_DOUBLE(c_tmp, vc, i);

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
    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        va = VLEV_DOUBLE(a_tmp, i);
        vb = VLEV_DOUBLE(b_tmp, i);
        vc = VDIV_DOUBLE(va, vb, i);
        VSEV_DOUBLE(c_tmp, vc, i);

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
    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        va = VLEV_DOUBLE(a_tmp, i);
        vb = VLEV_DOUBLE(b_tmp, i);
        vc = VSUB_DOUBLE(va, vb, i);
        VSEV_DOUBLE(c_tmp, vc, i);

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

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vb, vc;
        va = VLEV_DOUBLE(a_tmp, i);
        vb = VLEV_DOUBLE(b_tmp, i);
        vc = VLEV_DOUBLE(c_tmp, i);
        vc = VFMA_DOUBLE(vc, va, vb, i);
        VSEV_DOUBLE(dst_tmp, vc, i);

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

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vc;
        va = VLEV_DOUBLE(a_tmp, i);
        vc = VLEV_DOUBLE(c_tmp, i);
        vc = VFMA1_DOUBLE(vc, b, va, i);
        VSEV_DOUBLE(dst_tmp, vc, i);

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

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vc;
        va = VLEV_DOUBLE(a_tmp, i);
        vc = VLE1_DOUBLE(c, i);
        vc = VFMA1_DOUBLE(vc, b, va, i);
        VSEV_DOUBLE(dst_tmp, vc, i);

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

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va, vb, vc;
        va = VLEV_DOUBLE(a_tmp, i);
        vb = VLEV_DOUBLE(b_tmp, i);
        vc = VLE1_DOUBLE(c, i);
        vc = VFMA_DOUBLE(vc, va, vb, i);
        VSEV_DOUBLE(dst_tmp, vc, i);

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
    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLEV_DOUBLE(src_tmp, i);

        VSEV_DOUBLE(dst_tmp, VMUL1_DOUBLE(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void setd_vec(double *dst, double value, int len)
{
    size_t i;
    double *dst_tmp = dst;

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        VSEV_DOUBLE(dst_tmp, VLE1_DOUBLE(value, i), i);
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

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        VSEV_DOUBLE(dst_tmp, VLEV_DOUBLE(src_tmp, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void sqrtd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLEV_DOUBLE(src_tmp, i);
        VSEV_DOUBLE(dst_tmp, vfsqrt_v_f64m8(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void fabsd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double *dst_tmp = dst;

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va;
        va = VLEV_DOUBLE(src_tmp, i);
        VSEV_DOUBLE(dst_tmp, vfabs_v_f64m8(va, i), i);

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void sumd_vec(double *src, double *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    double acc[32];

    i = VSETVL_DOUBLE(len);
    V_ELT_DOUBLE vacc = VLE1_DOUBLE(0.0, i);  // initialised at 0?

    int len_ori = len;

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        V_ELT_DOUBLE va = VLEV_DOUBLE(src_tmp, i);
        vacc = VADD_DOUBLE(vacc, va, i);
        src_tmp += i;
    }

    size_t vlen_ori = VSETVL_DOUBLE(len_ori);
    VSEV_DOUBLE(acc, vacc, len_ori);
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

    double coef_max[32];

    // to be improved!
    for (int s = 0; s < 32; s++) {
        coef_max[s] = (double) (s) *slope;
    }

    i = VSETVL_DOUBLE(len);

    V_ELT_DOUBLE coef = VLEV_DOUBLE(coef_max, i);
    V_ELT_DOUBLE slope_vec = VLE1_DOUBLE((double) (i) *slope, i);
    V_ELT_DOUBLE curVal = VADD1_DOUBLE(coef, offset, i);

    VSEV_DOUBLE(dst_tmp, curVal, i);
    dst_tmp += i;
    len -= i;

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        curVal = VADD_DOUBLE(curVal, slope_vec, i);
        VSEV_DOUBLE(dst_tmp, curVal, i);
        dst_tmp += i;
    }
}

static inline void convert_32f64f_vec(float *src, double *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    double *dst_tmp = dst;

    for (; (i = vsetvl_e32m4(len)) > 0; len -= i) {
        VSEV_DOUBLE(dst_tmp, vfwcvt_f_f_v_f64m8(vle32_v_f32m4(src_tmp, i), i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void convert_64f32f_vec(double *src, float *dst, int len)
{
    size_t i;
    double *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL_DOUBLE(len)) > 0; len -= i) {
        vse32_v_f32m4(dst_tmp, vfncvt_f_f_w_f32m4(VLEV_DOUBLE(src_tmp, i), i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}
