/*
 * Project : SIMD_Utils
 * Version : 0.2.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


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
*/
// load vector float32, 8
#define VSETVL vsetvl_e32m8

#define VLEV_FLOAT vle32_v_f32m8
#define VSEV_FLOAT vse32_v_f32m8
#define VADD_FLOAT vfadd_vv_f32m8
#define VMUL_FLOAT vfmul_vv_f32m8
#define VFMA_FLOAT vfmacc_vv_f32m8
#define V_ELT vfloat32m8_t

#define VLEV_INT vle32_v_i32m8
#define VSEV_INT vse32_v_i32m8
#define VADD_INT vadd_vv_i32m8
#define VSUB_INT vsub_vv_i32m8
#define V_ELT_INT vint32m8_t



static inline void print_vec(V_ELT vec)
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
    V_ELT va, vb, vc;
    for (; (i = VSETVL(len)) > 0; len -= i) {
        printf("Loaded %lu elts\n", i);
        va = VLEV_FLOAT(a, i);
        vb = VLEV_FLOAT(b, i);
        vc = VADD_FLOAT(va, vb, i);
        VSEV_FLOAT(c, vc, i);

        a += i;
        b += i;
        c += i;
    }
}


static inline void sinf_vec(float *src, float *dst, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;



    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT x = VLEV_FLOAT(src_tmp, i);

        V_ELT xmm3, sign_bit, y;
        V_ELT_INT emm0, emm2;
        sign_bit = x;

        /* take the absolute value */
        x = vreinterpret_v_i32m8_f32m8(vand_vx_i32m8(vreinterpret_v_f32m8_i32m8(x), inv_sign_mask, i));

        /* extract the sign bit (upper one) */
        // not 0 if input < 0
        V_ELT_INT sign_bit_int = vand_vx_i32m8(vreinterpret_v_f32m8_i32m8(sign_bit), sign_mask, i);

        /* scale by 4/Pi */
        y = vfmul_vf_f32m8(x, FOPI, i);

        /* store the integer part of y in mm0 */
        emm2 = vfcvt_rtz_x_f_v_i32m8(y, i);

        /* j=(j+1) & (~1) (see the cephes sources) */
        emm2 = vadd_vx_i32m8(emm2, 1, i);
        emm2 = vand_vx_i32m8(emm2, ~1, i);
        y = vfcvt_f_x_v_f32m8(emm2, i);

        /* get the swap sign flag */
        emm0 = vand_vx_i32m8(emm2, 4, i);
        emm0 = vsll_vx_i32m8(emm0, 29, i);

        /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
         */
        emm2 = vand_vx_i32m8(emm2, 2, i);

        /// emm2 == 0 ? 0xFFFFFFFF : 0x00000000
        vbool4_t poly_mask = vmseq_vx_i32m8_b4(emm2, 0, i);
        // vbool4_t not_poly_mask=vmnot_m_b4(poly_mask, i);

        sign_bit_int = vxor_vv_i32m8(sign_bit_int, emm0, i);  // emm0 is swap_sign_bit

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        x = vfmacc_vf_f32m8(x, minus_cephes_DP1, y, i);
        x = vfmacc_vf_f32m8(x, minus_cephes_DP2, y, i);
        x = vfmacc_vf_f32m8(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT z = vfmul_vv_f32m8(x, x, i);
        y = vfmul_vf_f32m8(z, coscof[0], i);
        y = vfadd_vf_f32m8(y, coscof[1], i);
        y = vfmul_vv_f32m8(y, z, i);
        y = vfadd_vf_f32m8(y, coscof[2], i);
        y = vfmul_vv_f32m8(y, z, i);
        y = vfmul_vv_f32m8(y, z, i);
        V_ELT tmp = vfmul_vf_f32m8(z, 0.5f, i);
        y = vfsub_vv_f32m8(y, tmp, i);
        y = vfadd_vf_f32m8(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT y2;
        y2 = vfmul_vf_f32m8(z, sincof[0], i);
        y2 = vfadd_vf_f32m8(y2, sincof[1], i);
        y2 = vfmul_vv_f32m8(y2, z, i);
        y2 = vfadd_vf_f32m8(y2, sincof[2], i);
        y2 = vfmul_vv_f32m8(y2, z, i);
        y2 = vfmul_vv_f32m8(y2, x, i);
        y2 = vfadd_vv_f32m8(y2, x, i);

        /* select the correct result from the two polynoms */
        y = vmerge_vvm_f32m8(poly_mask, y, y2, i);

        /* update the sign */
        y = vreinterpret_v_i32m8_f32m8(vxor_vv_i32m8(vreinterpret_v_f32m8_i32m8(y), sign_bit_int, i));

        VSEV_FLOAT(dst_tmp, y, i);

        src_tmp += i;
        dst_tmp += i;
    }
}

// Work in progress
#if 1
static inline void sincosf_vec(float *src, float *s, float *c, int len)
{
    size_t i;
    float *src_tmp = src;
    float *s_tmp = s;
    float *c_tmp = c;


    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT x = VLEV_FLOAT(src_tmp, i);

        V_ELT y;
        V_ELT_INT j;
        vbool4_t jandone, jsup3, jsup1, j1or2, xinf0;
        vbool4_t sign_sin, sign_cos;

        sign_sin = vmclr_m_b4(i);
        sign_cos = vmclr_m_b4(i);

        // if (x < 0)
        xinf0 = vmflt_vf_f32m8_b4(x, 0.0f, i);
        sign_sin = vmxor_mm_b4(sign_sin, xinf0, i);

        /* take the absolute value */
        x = vreinterpret_v_i32m8_f32m8(vand_vx_i32m8(vreinterpret_v_f32m8_i32m8(x), inv_sign_mask, i));

        /* scale by 4/Pi */
        y = vfmul_vf_f32m8(x, FOPI, i);

        /* store the integer part of y in mm2 */
        j = vfcvt_rtz_x_f_v_i32m8(y, i);

        // if (j&1))
        jandone = vmsne_vx_i32m8_b4(vand_vx_i32m8(j, 1, i), 0, i);
        j = vadd_vx_i32m8_m(jandone, j, j, 1, i);
        y = vfcvt_f_x_v_f32m8(j, i);

        // j&=7
        j = vand_vx_i32m8(j, 7, i);

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
        x = vfmacc_vf_f32m8(x, minus_cephes_DP1, y, i);
        x = vfmacc_vf_f32m8(x, minus_cephes_DP2, y, i);
        x = vfmacc_vf_f32m8(x, minus_cephes_DP3, y, i);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        V_ELT z = vfmul_vv_f32m8(x, x, i);
        y = vfmul_vf_f32m8(z, coscof[0], i);
        y = vfadd_vf_f32m8(y, coscof[1], i);
        y = vfmul_vv_f32m8(y, z, i);
        y = vfadd_vf_f32m8(y, coscof[2], i);
        y = vfmul_vv_f32m8(y, z, i);
        y = vfmul_vv_f32m8(y, z, i);
        V_ELT tmp = vfmul_vf_f32m8(z, 0.5f, i);
        y = vfsub_vv_f32m8(y, tmp, i);
        y = vfadd_vf_f32m8(y, 1.0f, i);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
        V_ELT y2;
        y2 = vfmul_vf_f32m8(z, sincof[0], i);
        y2 = vfadd_vf_f32m8(y2, sincof[1], i);
        y2 = vfmul_vv_f32m8(y2, z, i);
        y2 = vfadd_vf_f32m8(y2, sincof[2], i);
        y2 = vfmul_vv_f32m8(y2, z, i);
        y2 = vfmul_vv_f32m8(y2, x, i);
        y2 = vfadd_vv_f32m8(y2, x, i);

        /* select the correct result from the two polynoms */
        V_ELT y_sin = vmerge_vvm_f32m8(j1or2, y2, y, i);
        V_ELT y_cos = vmerge_vvm_f32m8(j1or2, y, y2, i);

        y_sin = vfmul_vf_f32m8_m(sign_sin, y_sin, y_sin, -1.0f, i);
        y_cos = vfmul_vf_f32m8_m(sign_cos, y_cos, y_cos, -1.0f, i);

        VSEV_FLOAT(s_tmp, y_sin, i);
        VSEV_FLOAT(c_tmp, y_cos, i);

        src_tmp += i;
        s_tmp += i;
        c_tmp += i;
    }
}
#endif

static inline void sumf_vec(float *src, float *dst, int len)
{
    float acc[32] = {0};  // max size?

    size_t i;
    float *src_tmp = src;

    i = VSETVL(len);
    V_ELT vacc = VLEV_FLOAT(acc, i);  // initialised at 0?

    int len_ori = len;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT va = VLEV_FLOAT(src_tmp, i);
        vacc = VADD_FLOAT(vacc, va, i);
        src_tmp += i;
    }

    size_t vlen_ori = VSETVL(len_ori);
    VSEV_FLOAT(acc, vacc, len_ori);
    for (int j = 1; j < vlen_ori; j++) {
        acc[0] += acc[j];
    }
    *dst = acc[0];
}

static inline void meanf_vec(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sumf_vec(src, dst, len);
    *dst *= coeff;
}

static inline void magnitudef_split_vec(float *srcRe, float *srcIm, float *dst, int len)
{
    size_t i;
    float *srcRe_tmp = srcRe;
    float *srcIm_tmp = srcIm;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT re_tmp = VLEV_FLOAT(srcRe_tmp, i);
        V_ELT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT im_tmp = VLEV_FLOAT(srcIm_tmp, i);
        V_ELT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);

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
        V_ELT re_tmp = VLEV_FLOAT(srcRe_tmp, i);
        V_ELT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT im_tmp = VLEV_FLOAT(srcIm_tmp, i);
        V_ELT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);

        VSEV_FLOAT(dst_tmp, tmp, i);

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

    for (; (i = VSETVL(len)) > 0; len -= i) {
        // complex are a + ib, c + id, e + if, etc
        // load in re_tmp a,c,e, etc => i elements in range 0..2*i with a stride of 2
        // load in im_tmp b,d,f, etc => i elements in range 0..2*i with a stride of 2
        V_ELT re_tmp = vlse32_v_f32m8(src_tmp, 2 * sizeof(float), i);
        V_ELT im_tmp = vlse32_v_f32m8(src_tmp + 1, 2 * sizeof(float), i);
        V_ELT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);
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
        V_ELT re_tmp = vlse32_v_f32m8(src_tmp, 2 * sizeof(float), i);
        V_ELT im_tmp = vlse32_v_f32m8(src_tmp + 1, 2 * sizeof(float), i);
        V_ELT re2 = VMUL_FLOAT(re_tmp, re_tmp, i);
        V_ELT tmp = VFMA_FLOAT(re2, im_tmp, im_tmp, i);
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
        V_ELT va, vb;
        va = VLEV_FLOAT(src1_tmp, i);
        vb = VLEV_FLOAT(src2_tmp, i);
        VSEV_FLOAT(dst_tmp, vfmax_vv_f32m8(va, vb, i), i);

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
        V_ELT va, vb;
        va = VLEV_FLOAT(src1_tmp, i);
        vb = VLEV_FLOAT(src2_tmp, i);
        VSEV_FLOAT(dst_tmp, vfmin_vv_f32m8(va, vb, i), i);

        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_gt_f_vec(float *src, float *dst, int len, float value)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT va;
        va = VLEV_FLOAT(src_tmp, i);
        VSEV_FLOAT(dst_tmp, vfmin_vf_f32m8(va, value, i), i);

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
        V_ELT va;
        va = VLEV_FLOAT(src_tmp, i);
        VSEV_FLOAT(dst_tmp, vfmax_vf_f32m8(va, value, i), i);

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
        V_ELT va = VLEV_FLOAT(src_tmp, i);
        V_ELT va_abs = (V_ELT) vand_vx_i32m8((vint32m8_t) va, inv_sign_mask, i);
        vbool4_t eqmask = vmfeq_vv_f32m8_b4(va, va_abs, i);
        vbool4_t gtmask = vmfgt_vf_f32m8_b4(va_abs, value, i);

        /*weird way to achieve the equivalent of _mm_blendv_ps(mval, pval, eqmask);
         no way to broadcast a float to a vector yet, so we have to use an undefined vector
         merged with the negative value (not(mask))and then merge with the value */
        V_ELT sval;
        sval = vfmerge_vfm_f32m8(vmnot_m_b4(eqmask, i), sval, -value, i);
        sval = vfmerge_vfm_f32m8(eqmask, sval, value, i);
        VSEV_FLOAT(dst_tmp, vmerge_vvm_f32m8(gtmask, va, sval, i), i);

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
        V_ELT va = VLEV_FLOAT(src_tmp, i);
        vbool4_t lt_mask = vmflt_vf_f32m8_b4(va, ltlevel, i);
        vbool4_t gt_mask = vmfgt_vf_f32m8_b4(va, gtlevel, i);
        V_ELT tmp = vfmerge_vfm_f32m8(lt_mask, va, ltvalue, i);
        tmp = vfmerge_vfm_f32m8(gt_mask, tmp, gtvalue, i);
        VSEV_FLOAT(dst_tmp, tmp, i);

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

        VSEV_INT(dst_tmp, vsub_vv_i32m8(va, vb, i), i);
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

        VSEV_INT(dst_tmp, vadd_vv_i32m8(va, vb, i), i);
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

        VSEV_INT(dst_tmp, vadd_vx_i32m8(va, value, i), i);
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

        VSEV_INT(dst_tmp, vmul_vv_i32m8(va, vb, i), i);
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

        VSEV_INT(dst_tmp, vmul_vx_i32m8(va, value, i), i);
        src_tmp += i;
        dst_tmp += i;
    }
}
