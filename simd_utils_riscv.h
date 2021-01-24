/*
 * Project : SIMD_Utils
 * Version : 0.1.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdlib.h>

/* ELEN : element length, 8,16,32,64bits
	VLEN : Vector Length
	VSEW : Vector Standard Element Width (dynamic), with of the base element : 8,16,32,64,...,1024bits
	(up to 64bit in the current intrinsics
	LMUL : Vector register grouping => may group multiple VLEN registers, so that 1 instruction can be applied to multiple registers. If LMUL is <, the operation applies only to a part of the register
	LMUL = 1,2,4,8, 1, 1/2, 1/4, 1/8
	VLMAX = LMUL*VLEN/SEW
*/
//load vector float32, 8
#define VLEV_FLOAT vle32_v_f32m8
#define VSEV_FLOAT vse32_v_f32m8
#define VADD_FLOAT vfadd_vv_f32m8
#define V_ELT vfloat32m8_t
#define VSETVL vsetvl_e32m8


static inline void maxeveryf_vec(float *src1, float *src2, float *dst, int len)
{
    size_t i;
    float *src1_tmp = src1;
    float *src2_tmp = src2;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT va, vb;
        va = VLEV_FLOAT(src1_tmp);
        vb = VLEV_FLOAT(src2_tmp);
        VSEV_FLOAT(dst_tmp, vfmax_vv_f32m8(va, vb));

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
        va = VLEV_FLOAT(src1_tmp);
        vb = VLEV_FLOAT(src2_tmp);
        VSEV_FLOAT(dst_tmp, vfmin_vv_f32m8(va, vb));

        src1_tmp += i;
        src2_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_gt_f_vec(float *src, float *dst, float value, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT va;
        va = VLEV_FLOAT(src_tmp);
        VSEV_FLOAT(dst_tmp, vfmin_vf_f32m8(va, value));

        src_tmp += i;
        dst_tmp += i;
    }
}

static inline void threshold_lt_f_vec(float *src, float *dst, float value, int len)
{
    size_t i;
    float *src_tmp = src;
    float *dst_tmp = dst;

    for (; (i = VSETVL(len)) > 0; len -= i) {
        V_ELT va;
        va = VLEV_FLOAT(src_tmp);
        VSEV_FLOAT(dst_tmp, vfmax_vf_f32m8(va, value));

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
        vint32m8_t va, vb;
        va = vle32_v_i32m8(src1_tmp);
        vb = vle32_v_i32m8(src2_tmp);

        vse32_v_i32m8(dst_tmp, vsub_vv_i32m8(va, vb));
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
        vint32m8_t va, vb;
        va = vle32_v_i32m8(src1_tmp);
        vb = vle32_v_i32m8(src2_tmp);

        vse32_v_i32m8(dst_tmp, vadd_vv_i32m8(va, vb));
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
        vint32m8_t va;
        va = vle32_v_i32m8(src_tmp);

        vse32_v_i32m8(dst_tmp, vadd_vx_i32m8(va, value));
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
        vint32m8_t va, vb;
        va = vle32_v_i32m8(src1_tmp);
        vb = vle32_v_i32m8(src2_tmp);

        vse32_v_i32m8(dst_tmp, vmul_vv_i32m8(va, vb));
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
        vint32m8_t va;
        va = vle32_v_i32m8(src_tmp);

        vse32_v_i32m8(dst_tmp, vmul_vx_i32m8(va, value));
        src_tmp += i;
        dst_tmp += i;
    }
}

// e32 => float32 (e64 float 64)
// m8 8 elements (m4 4 elements)
/* l = vsetvl_e32m8(n) asks for 
	n float32 elements grouped by 8. l returns the total number of elements achievable
	with this configuration
*/
void addf_vec(float *a, float *b, float *c, int n)
{
    size_t l;
    V_ELT va, vb, vc;
    for (; (l = VSETVL(n)) > 0; n -= l) {
        printf("Loaded %lu elts\n", l);
        va = VLEV_FLOAT(a);
        a += l;
        vb = VLEV_FLOAT(b);
        b += l;
        vc = VADD_FLOAT(va, vb);
        VSEV_FLOAT(c, vc);
        c += l;
    }
}
