#define SSE_LEN_FLOAT 4
#define SSE_LEN_BYTES 16
typedef float32x4_t v4sf;
typedef float32x4x2_t v4sfx2;
typedef int32x4_t v4si;

static inline void add128f(float *a, float* b, float* c, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf a_vec = vld1q_f32(a + i);
		v4sf b_vec = vld1q_f32(b + i);
		vst1q_f32(c + i, vaddq_f32(a_vec, b_vec));
	}
    for (int i = stop_len; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}

static inline void cplxtoreal128f(complex32_t *src, float *dstRe, float *dstIm, int len)
{
    int stop_len = 2 * len / (4 * SSE_LEN_FLOAT);
    stop_len *= 4 * SSE_LEN_FLOAT;

    int j = 0;
	for (int i = 0; i < stop_len; i += 4 * SSE_LEN_FLOAT) {
		/*__builtin_prefetch(&src[i+SSE_LEN_FLOAT], 0, 3);
		__builtin_prefetch(&dstRe[i+SSE_LEN_FLOAT], 1, 3);
		__builtin_prefetch(&dstIm[i+SSE_LEN_FLOAT], 1, 3);*/
		v4sfx2 vec1 = vld2q_f32((float const *) (src) + i);
		v4sfx2 vec2 = vld2q_f32((float const *) (src) + i + 2 * SSE_LEN_FLOAT);
		vst1q_f32(dstRe + j, vec1.val[0]);
		vst1q_f32(dstIm + j, vec1.val[1]);
		vst1q_f32(dstRe + j + SSE_LEN_FLOAT, vec2.val[0]);
		vst1q_f32(dstIm + j + SSE_LEN_FLOAT, vec2.val[1]);
		j += 2 * SSE_LEN_FLOAT;
	}

    for (int i = j; i < len; i++) {
        dstRe[i] = src[i].re;
        dstIm[i] = src[i].im;
    }
}

static inline void realtocplx128f(float *srcRe, float *srcIm, complex32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

    int j = 0;
	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		/*__builtin_prefetch(&srcRe[i+SSE_LEN_FLOAT], 0, 3);
		__builtin_prefetch(&srcIm[i+SSE_LEN_FLOAT], 0, 3);
		__builtin_prefetch(&dst[i+SSE_LEN_FLOAT], 1, 3);*/
		v4sf re = vld1q_f32(srcRe + i);
		v4sf im = vld1q_f32(srcIm + i);
		v4sf re2 = vld1q_f32(srcRe + i + SSE_LEN_FLOAT);
		v4sf im2 = vld1q_f32(srcIm + i + SSE_LEN_FLOAT);
		v4sfx2 reim = {{re, im}};
		v4sfx2 reim2 = {{re2, im2}};
		vst2q_f32((float *) (dst) + j, reim);
		vst2q_f32((float *) (dst) + j + 2 * SSE_LEN_FLOAT, reim2);
		j += 4 * SSE_LEN_FLOAT;
	}

    for (int i = stop_len; i < len; i++) {
        dst[i].re = srcRe[i];
        dst[i].im = srcIm[i];
    }
}

void powerspect128f_interleaved(complex32_t *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

    int j = 0;
	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sfx2 src_split = vld2q_f32((float *) (src) + j);  // a0a1a2a3, b0b1b2b3
		v4sfx2 src_split2 = vld2q_f32((float *) (src) + j + 2 * SSE_LEN_FLOAT);
		v4sf split_square0 = vmulq_f32(src_split.val[0], src_split.val[0]);
		v4sf split2_square0 = vmulq_f32(src_split2.val[0], src_split2.val[0]);
		v4sfx2 dst_split;
		dst_split.val[0] = vfmaq_f32(split_square0, src_split.val[1], src_split.val[1]);
		dst_split.val[1] = vfmaq_f32(split2_square0, src_split2.val[1], src_split2.val[1]);
		vst1q_f32((dst + i), dst_split.val[0]);
		vst1q_f32((dst + i + SSE_LEN_FLOAT), dst_split.val[1]);
		j += 4 * SSE_LEN_FLOAT;
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i].re * src[i].re + (src[i].im * src[i].im);
    }
}

static inline float32x4_t vrsqrtq_f32(float32x4_t a)
{
    // Quake initial approximation via bit trick
    uint32x4_t ai = vreinterpretq_u32_f32(a);
    ai = vsubq_u32(vdupq_n_u32(0x5f3759dfu), vshrq_n_u32(ai, 1));
    float32x4_t y = vreinterpretq_f32_u32(ai);

    // Newton-Raphson refinements: y = y*(1.5 - 0.5*a*y*y)
    float32x4_t x2 = vmulq_n_f32(a, 0.5f);
    // 1st iteration
    float32x4_t y2 = vmulq_f32(y, y);
    y = vmulq_f32(y, vsubq_f32(*(v4sf *) _ps_1p5, vmulq_f32(x2, y2)));
    // 2nd iteration -> almost full float precision
    y2 = vmulq_f32(y, y);
    y = vmulq_f32(y, vsubq_f32(*(v4sf *) _ps_1p5, vmulq_f32(x2, y2)));
    // 3nd iteration -> ~full float precision
    y2 = vmulq_f32(y, y);
    y = vmulq_f32(y, vsubq_f32(*(v4sf *) _ps_1p5, vmulq_f32(x2, y2)));
    return y; // ~1/sqrt(a)
}

// Vector sqrt using the inverse-sqrt result (no div; only +/*)
static inline float32x4_t vsqrtq_f32(float32x4_t a)
{
    float32x4_t rinv = vrsqrtq_f32(a);
    return vmulq_f32(a, rinv); // sqrt(a) = a * (1/sqrt(a))
}

// 1/y
static inline float32x4_t vrecipq_f32(float32x4_t a)
{
    // --- Bit-hack initial approximation ---
    uint32x4_t ai = vreinterpretq_u32_f32(a);
    // magic constant chosen for reciprocal
    ai = vsubq_u32(vdupq_n_u32(0x7EEEEEEE), ai);
    float32x4_t r = vreinterpretq_f32_u32(ai);

    // --- Newton–Raphson refinement: r = r * (2 - a*r) ---
    // 1st iteration
    float32x4_t ar = vmulq_f32(a, r);
    r = vmulq_f32(r, vsubq_f32(*(v4sf *) _ps_2, ar));
    // 2nd iteration -> almost full float32 precision (~23 bits)
    ar = vmulq_f32(a, r);
    r = vmulq_f32(r, vsubq_f32(*(v4sf *) _ps_2, ar));
    // 3nd iteration -> full float32 precision
    ar = vmulq_f32(a, r);
    r = vmulq_f32(r, vsubq_f32(*(v4sf *) _ps_2, ar));
    return r;
}

// -1/y
static inline float32x4_t vminrecipq_f32(float32x4_t a)
{
    // --- Bit-hack initial approximation ---
    uint32x4_t ai = vreinterpretq_u32_f32(a);
    // magic constant chosen for reciprocal
    ai = vsubq_u32(vdupq_n_u32(0x7EEEEEEE), ai);
    float32x4_t r = vreinterpretq_f32_u32(ai);

    // --- Newton–Raphson refinement: r = r * (2 - a*r) ---
    // 1st iteration
    float32x4_t ar = vmulq_f32(a, r);
    r = vmulq_f32(r, vsubq_f32(*(v4sf *) _ps_2, ar));
    // 2nd iteration -> almost full float32 precision (~23 bits)
    ar = vmulq_f32(a, r);
    r = vmulq_f32(r, vsubq_f32(*(v4sf *) _ps_2, ar));
    // 3nd iteration -> full float32 precision
    ar = vmulq_f32(a, r);
    r = vmulq_f32(r, vsubq_f32(*(v4sf *) _ps_2, ar));
    return vmulq_n_f32(r,-1.0f);
}

static inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b)
{
    float32x4_t oneDivb = vrecipq_f32(b);
    return vmulq_f32(a,oneDivb);
}

static inline void print4_4digits(v4sf v)
{
    float *p = (float *) &v;
    printf("[%3.4g, %3.4g, %3.4g, %3.4g]", p[0], p[1], p[2], p[3]);
}

static inline void print4i(v4si v)
{
    int32_t *p = (int32_t *) &v;
    printf("[%ld %ld, %ld, %ld]", p[0], p[1], p[2], p[3]);
}

static inline void print4u(v4su v)
{
    uint32_t *p = (int32_t *) &v;
    printf("[%lu %lu, %lu, %lu]", p[0], p[1], p[2], p[3]);
}

static inline v4sf log10_ps(v4sf x)
{
    //printf("x ");print4_4digits(x);printf("\r\n");
    v4si emm0;
    mve_pred16_t invalid_mask = vcmpleq_f32(x, *(v4sf *) _ps_0);
    x = vmaxnmq_f32(x, *(v4sf *) _ps_min_norm_pos); /* cut off denormalized stuff */
    emm0 = vshrq_n_s32(vreinterpretq_s32_f32(x), 23);
   // printf("emm0 ");print4i_4digits(emm0);printf("\r\n");
    /* keep only the fractional part */
    x = vandq_f32(x, *(v4sf *) _ps_inv_mant_mask);
    x = vorrq_f32(x, *(v4sf *) _ps_0p5);
    //printf("x2 ");print4_4digits(x);printf("\r\n");
    emm0 = vsubq_n_s32(emm0, 0x7F);
    v4sf e = vcvtq_f32_s32(emm0);
    e = vaddq_n_f32(e, 1.0f);
    //printf("e ");print4_4digits(e);printf("\r\n");
    mve_pred16_t mask = vcmpltq_f32(x, *(v4sf *) _ps_cephes_SQRTHF);
    x = vsubq_n_f32(x, 1.0f);
    e = vsubq_m_n_f32(e, e, 1.0f, mask);
    //printf("e2 ");print4_4digits(e);printf("\r\n");
    x = vaddq_m_f32(x, x, x, mask);
    //printf("x3 ");print4_4digits(x);printf("\r\n");
    v4sf z = vmulq_f32(x, x);
    // vfmasq_n_f32 : a*b + c (scalar)
    v4sf y = vfmasq_n_f32(x, *(v4sf *) _ps_cephes_log_p0, c_cephes_log_p1);
    //printf("y ");print4_4digits(y);printf("\r\n");
    y = vfmasq_n_f32(y, x, c_cephes_log_p2);
    //printf("y2 ");print4_4digits(y);printf("\r\n");
    y = vfmasq_n_f32(y, x, c_cephes_log_p3);
    y = vfmasq_n_f32(y, x, c_cephes_log_p4);
    y = vfmasq_n_f32(y, x, c_cephes_log_p5);
    y = vfmasq_n_f32(y, x, c_cephes_log_p6);
    y = vfmasq_n_f32(y, x, c_cephes_log_p7);
    y = vfmasq_n_f32(y, x, c_cephes_log_p8);
    y = vmulq_f32(y, x);
    y = vmulq_f32(y, z);
    //printf("y3 ");print4_4digits(y);printf("\r\n");
    // vfmaq_n_f32 a + b*c (scalar)
    y = vfmaq_n_f32(y, z, -0.5f);
    //printf("y4 ");print4_4digits(y);printf("\r\n");
    // Could it be improved with more parallelism or would it worsen precision?
    float32x4_t tmp = vaddq_f32(x, y);

    z = vmulq_n_f32(tmp, c_cephes_L10EB);
    //printf("z ");print4_4digits(z);printf("\r\n");
    z = vfmaq_n_f32(z, y, c_cephes_L10EA);
    //printf("z2 ");print4_4digits(z);printf("\r\n");
    z = vfmaq_n_f32(z, x, c_cephes_L10EA);
    //printf("z3 ");print4_4digits(z);printf("\r\n");
    z = vfmaq_n_f32(z, e, c_cephes_L102B);
    //printf("z4 ");print4_4digits(z);printf("\r\n");
    x = vfmaq_n_f32(z, e, c_cephes_L102A);
    //printf("x5 ");print4_4digits(x);printf("\r\n");
    x = vorrq_m_f32(x, x, *(v4sf *) _pi32_0xFFFFFFFF, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline void log10128f_precise(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf vec = vld1q_f32(src + i);
		vec = log10_ps(vec);
		vst1q_f32(dst + i, vec);
	}
    for (int i = stop_len; i < len; i++) {
        dst[i] = log10f(src[i]);
    }
}

static inline mve_pred16_t vtstq_u32_m(uint32x4_t a, uint32x4_t b) {
	uint32x4_t p = vandq_u32(a, b);
	//printf("a");print4u(a);printf("\n\r");
	//printf("b");print4u(b);printf("\n\r");
	//printf("p");print4u(p);printf("\n\r");
    return vcmpneq_u32(p, *(v4su *) _pu32_0);
}

//useful to better match y = a*b + c instead of y = a + b*c
#define vfmasq_f32(b,c,add) vfmaq_f32(add,b,c)

static inline mve_pred16_t xor_pred(mve_pred16_t a, mve_pred16_t b) {
    // Materialize predicate bits into uint16_t scalars
    uint16_t pa = (uint16_t)a;
    uint16_t pb = (uint16_t)b;
    uint16_t pc = pa ^ pb;
    //printf("%x %x %x\r\n",pa,pb,pc);
    return (mve_pred16_t)(pc);
}

static inline mve_pred16_t and_pred(mve_pred16_t a, mve_pred16_t b) {
    // Materialize predicate bits into uint16_t scalars
    uint16_t pa = (uint16_t)a;
    uint16_t pb = (uint16_t)b;
    uint16_t pc = pa & pb;
    //printf("%x %x %x\r\n",pa,pb,pc);
    return (mve_pred16_t)(pc);
}

// FMA version
static inline void sincos_ps(v4sf x, v4sf *ysin, v4sf *ycos)
{  // any x
    v4sf y;

    v4su emm2;

    mve_pred16_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcmpltq_f32(x, *(v4sf *) _ps_0);
    x = vabsq_f32(x);
    //printf("x ");print4_4digits(x);printf("\r\n");
    /* scale by 4/Pi */
    y = vmulq_n_f32(x, c_cephes_FOPI);

    /* store the integer part of y in mm0 */
    emm2 = vcvtq_u32_f32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_n_u32(emm2, 1);
    emm2 = vandq_u32(emm2, *(v4su *) _pu32_inv1);
    y = vcvtq_f32_u32(emm2);
    //printf("y ");print4_4digits(y);printf("\r\n");
    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
    mve_pred16_t poly_mask = vtstq_u32_m(emm2,  *(v4su *) _pu32_2);
    //printf("poly_mask %x\r\n",(uint16_t)poly_mask);
    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */

    x = vfmaq_n_f32(x, y, c_minus_cephes_DP1);
    x = vfmaq_n_f32(x, y, c_minus_cephes_DP2);
    x = vfmaq_n_f32(x, y, c_minus_cephes_DP3);

    mve_pred16_t sign_mask_sin2 = vcmpeqq_u32(emm2, *(v4su *) _pu32_4);
    //printf("sign_mask_sin2 %x\r\n",(uint16_t)sign_mask_sin2);
    //printf("sign_mask_sin %x\r\n",(uint16_t)sign_mask_sin);
    sign_mask_sin = xor_pred(sign_mask_sin, sign_mask_sin2);
    //printf("sign_mask_sin %x\r\n",(uint16_t)sign_mask_sin);
    v4su emm2min2 = vsubq_n_u32(emm2, 2);
    //printf("emm2 ");print4u(emm2);printf("\r\n");
    //printf("emm2min2 ");print4u(emm2min2);printf("\r\n");
    sign_mask_cos = vtstq_u32_m(emm2min2, *(v4su *) _pu32_4);
    //printf("sign_mask_cos %x\r\n",(uint16_t)sign_mask_sin);
    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    v4sf z = vmulq_f32(x, x);
    v4sf y1, y2;

    y1 = vfmaq_n_f32(vdupq_n_f32(c_coscof_p1), z, c_coscof_p0);
    y2 = vfmaq_n_f32(vdupq_n_f32(c_sincof_p1), z, c_sincof_p0);
    y1 = vfmasq_n_f32(y1, z, c_coscof_p2);
    y2 = vfmasq_n_f32(y2, z, c_sincof_p2);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vfmaq_f32(x, y2, x);
    y1 = vfmaq_n_f32(y1, z, -0.5f);
    y1 = vaddq_n_f32(y1, 1);

    /* select the correct result from the two polynoms */
    //printf("y1 ");print4_4digits(y1);printf("\r\n");
    //printf("y2 ");print4_4digits(y2);printf("\r\n");
    v4sf ys = vpselq_f32(y1, y2,poly_mask);
    v4sf yc = vpselq_f32(y2, y1, poly_mask);
    //printf("ys ");print4_4digits(ys);printf("\r\n");
    //printf("yc ");print4_4digits(yc);printf("\r\n");
    *ysin = vpselq_f32(vnegq_f32(ys), ys, sign_mask_sin);
    *ycos = vpselq_f32(yc, vnegq_f32(yc), sign_mask_cos);
    //printf("sin ");print4_4digits(*ysin);printf("\r\n");
    //printf("cos ");print4_4digits(*ycos);printf("\r\n");
}

static inline void sincos128f(float *src, float *dst_sin, float *dst_cos, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;
	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf dst_sin_tmp;
		v4sf dst_cos_tmp;
		sincos_ps(src_tmp, &dst_sin_tmp, &dst_cos_tmp);
		vst1q_f32(dst_sin + i, dst_sin_tmp);
		vst1q_f32(dst_cos + i, dst_cos_tmp);
	}

    for (int i = stop_len; i < len; i++) {
        mysincosf(src[i], dst_sin + i, dst_cos + i);
    }
}

static inline void sqrt128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf vec = vld1q_f32(src + i);
		vec = vsqrtq_f32(vec);
		vst1q_f32(dst + i, vec);
	}
    for (int i = stop_len; i < len; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static inline v4sf exp_ps(v4sf x)
{
    v4sf fx;

    x = vminnmq_f32(x, *(v4sf *) _ps_exp_hi);
    x = vmaxnmq_f32(x, *(v4sf *) _ps_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vfmaq_n_f32(*(v4sf *) _ps_0p5, x, c_cephes_LOG2EF);
    fx = vrndmq_f32(fx); /* perform a floorf */

    x = vfmaq_n_f32(x, fx, c_cephes_exp_minC1);
    x = vfmaq_n_f32(x, fx, c_cephes_exp_minC2);

    v4sf z = vmulq_f32(x, x);

    v4sf y = vfmaq_n_f32(*(v4sf *) _ps_cephes_exp_p1, x, c_cephes_exp_p0);
    y = vfmasq_n_f32(y, x, c_cephes_exp_p2);
    y = vfmasq_n_f32(y, x, c_cephes_exp_p3);
    y = vfmasq_n_f32(y, x, c_cephes_exp_p4);
    y = vfmasq_n_f32(y, x, c_cephes_exp_p5);
    y = vfmaq_f32(x, y, z);
    y = vaddq_n_f32(y, 1.0f);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_n_s32(mm, 0x7F);
    mm = vshlq_n_s32(mm, 23);
    v4sf pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

static inline void exp128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		vst1q_f32(dst + i, exp_ps(vld1q_f32(src + i)));
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = expf(src[i]);
    }
}

static inline v4sf tanf_ps(v4sf xx)
{
    v4sf x, y, z, zz;
    v4si j;  // long?
    v4sf sign;
    mve_pred16_t xsupem4;
    v4sf tmp;
    mve_pred16_t jandone, jandtwo;

    x = vandq_f32(*(v4sf *) _ps_pos_sign_mask, xx);  // fabs(xx) //OK
    sign = vandq_f32(xx, *(v4sf *) _ps_sign_mask);
  //  printf("x");print4_4digits(x);printf("\r\n");
   // printf("xx");print4_4digits(xx);printf("\r\n");
   // printf("sign");print4_4digits(sign);printf("\r\n");

    /* compute x mod PIO4 */
    tmp = vmulq_n_f32(x, c_cephes_FOPI);
   // printf("tmp2");print4_4digits(tmp);printf("\r\n");
    j = vcvtq_s32_f32(tmp);
    //printf("j");print4i(j);printf("\r\n");
    y = vcvtq_f32_s32(j);
   // printf("y0");print4_4digits(y);printf("\r\n");
    jandone = vcmpgtq_s32(vandq_s32(j, *(v4si *) _pi32_1),  *(v4si *) _pi32_0);
    y = vaddq_m_n_f32(y, y, 1.0f, jandone);
   // printf("y1");print4_4digits(y);printf("\r\n");
    j = vaddq_m_n_s32(j, j, 1, jandone);
   // printf("j1");print4i(j);printf("\r\n");
    z = vfmaq_n_f32(x, y, c_minus_cephes_DP1);
    z = vfmaq_n_f32(z, y, c_minus_cephes_DP2);
    z = vfmaq_n_f32(z, y, c_minus_cephes_DP3);
    zz = vmulq_f32(z, z);  // z*z
   // printf("zz");print4_4digits(z);printf("\r\n");
    // TODO : should not be computed if X < 10e-4
    /* 1.7e-8 relative error in [-pi/4, +pi/4] */
    tmp = vfmasq_n_f32(zz, *(v4sf *) _ps_TAN_P0, TAN_P1);
    tmp = vfmasq_n_f32(tmp, zz, TAN_P2);
    tmp = vfmasq_n_f32(tmp, zz, TAN_P3);
    tmp = vfmasq_n_f32(tmp, zz, TAN_P4);
    tmp = vfmasq_n_f32(tmp, zz, TAN_P5);
    tmp = vmulq_f32(zz, tmp);
    tmp = vmulq_f32(tmp, z);
    xsupem4 = vcmpgtq_f32(x, *(v4sf *) _ps_1emin4);  // if( x > 1.0e-4 )
    y = vaddq_m_f32(z, z, tmp, xsupem4);

    jandtwo = vcmpgtq_s32(vandq_s32(j, *(v4si *) _pi32_2), *(v4si *) _pi32_0);
    //printf("y");print4_4digits(y);printf("\r\n");
    // xor(rcp(y)) gives not good enough result
    tmp = vminrecipq_f32(y);
    y = vpselq_f32(tmp, y, jandtwo);
    //printf("tmp2");print4_4digits(tmp);printf("\r\n");
   // printf("y2");print4_4digits(y);printf("\r\n");
    y = veorq_f32(y, sign);
   // printf("yxor");print4_4digits(y);printf("\r\n"); printf("\r\n");
    return (y);
}

static inline void tan128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;
	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		vst1q_f32(dst + i, tanf_ps(vld1q_f32(src + i)));
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = tanf(src[i]);
    }
}

static inline void threshold128_ltval_gtval_f(float *src, float *dst, int len, float ltlevel, float ltvalue, float gtlevel, float gtvalue)
{
    const v4sf ltlevel_v = vdupq_n_f32(ltlevel);
    const v4sf ltvalue_v = vdupq_n_f32(ltvalue);
    const v4sf gtlevel_v = vdupq_n_f32(gtlevel);
    const v4sf gtvalue_v = vdupq_n_f32(gtvalue);

    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= ( SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i); //vldrwq_f32 ?
		mve_pred16_t lt_mask = vcmpltq_f32(src_tmp, ltlevel_v);
		mve_pred16_t gt_mask = vcmpgtq_f32(src_tmp, gtlevel_v);
		v4sf dst_tmp = vpselq_f32(ltvalue_v, src_tmp, lt_mask);
		dst_tmp = vpselq_f32(gtvalue_v, dst_tmp, gt_mask);
		vst1q_f32(dst + i, dst_tmp);
	}

    for (int i = stop_len; i < len; i++) {
		float tmp = src[i];
        float tmp2 = tmp < ltlevel ? ltvalue : tmp;
        dst[i] = tmp > gtlevel ? gtvalue : tmp2;
    }
}

static inline void threshold128_ltabs_f(float *src, float *dst, int len, float value)
{
    const v4sf pval = vdupq_n_f32(value);

    int stop_len = len / (SSE_LEN_FLOAT);
    stop_len *= (SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_sign = vandq_f32(src_tmp, *(v4sf *) _ps_sign_mask);  // extract sign
		v4sf src_abs = vandq_f32(src_tmp, *(v4sf *) _ps_pos_sign_mask);  // take absolute value
		v4sf dst_tmp = vmaxnmq_f32(src_abs, pval);
		dst_tmp = veorq_f32(dst_tmp, src_sign);
		vst1q_f32(dst + i, dst_tmp);
	}

    for (int i = stop_len; i < len; i++) {
        if (src[i] >= 0.0f) {
            dst[i] = src[i] < value ? value : src[i];
        } else {
            dst[i] = src[i] > (-value) ? (-value) : src[i];
        }
    }
}


static inline void threshold128_lt_f(float *src, float *dst, int len, float value)
{
    const v4sf tmp = vdupq_n_f32(value);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		v4sf dst_tmp = vmaxnmq_f32(src_tmp, tmp);
		v4sf dst_tmp2 = vmaxnmq_f32(src_tmp2, tmp);
		vst1q_f32(dst + i, dst_tmp);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = src[i] > value ? src[i] : value;
    }
}

static inline void minmax128f(float *src, int len, float *min_value, float *max_value)
{
    int stop_len = (len - SSE_LEN_FLOAT) / (2*SSE_LEN_FLOAT);
    stop_len *= ( 2*SSE_LEN_FLOAT);
    stop_len = (stop_len < 0) ? 0 : stop_len;

    v4sf max_v, min_v, max_v2, min_v2;
    v4sf src_tmp, src_tmp2;

    float min_tmp = src[0];
    float max_tmp = src[0];
    float min_f[SSE_LEN_FLOAT] __attribute__((aligned(SSE_LEN_BYTES)));
    float max_f[SSE_LEN_FLOAT] __attribute__((aligned(SSE_LEN_BYTES)));

    if (len >= SSE_LEN_FLOAT) {
		src_tmp = vld1q_f32(src + 0);
		max_v = src_tmp;
		min_v = src_tmp;
		max_v2 = src_tmp;
		min_v2 = src_tmp;

		for (int i = SSE_LEN_FLOAT; i < stop_len; i += 2*SSE_LEN_FLOAT) {
			src_tmp = vld1q_f32(src + i);
			src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
			max_v = vmaxnmq_f32(max_v, src_tmp);
			min_v = vminnmq_f32(min_v, src_tmp);
			max_v2 = vmaxnmq_f32(max_v2, src_tmp2);
			min_v2 = vminnmq_f32(min_v2, src_tmp2);
		}

        max_v = vmaxnmq_f32(max_v, max_v2);
        min_v = vminnmq_f32(min_v, min_v2);
        vst1q_f32(max_f, max_v);
        vst1q_f32(min_f, min_v);

        max_f[0] = fmaxf(max_f[0],max_f[1]);
        max_f[3] = fmaxf(max_f[2],max_f[3]);
        max_f[0] = fmaxf(max_f[0],max_f[3]);
        max_tmp = max_f[0];

        min_f[0] = fminf(min_f[0],min_f[1]);
        min_f[3] = fminf(min_f[2],min_f[3]);
        min_f[0] = fminf(min_f[0],min_f[3]);
        min_tmp = min_f[0];
    }

    for (int i = stop_len; i < len; i++) {
        max_tmp =  fmaxf(max_tmp,src[i]);
        min_tmp = fminf(min_tmp,src[i]);
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

static inline void flip128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    int mini = ((len - 1) < (2 * SSE_LEN_FLOAT)) ? (len - 1) : (2 * SSE_LEN_FLOAT);
    for (int i = 0; i < mini; i++) {
        dst[len - i - 1] = src[i];
    }

    uint32x4_t idx = {3,2,1,0};
	for (int i = 2 * SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src_tmp = vldrwq_gather_shifted_offset_f32(src + i,idx);  // load a,b,c,d
		v4sf src_tmp2 = vldrwq_gather_shifted_offset_f32(src + i + SSE_LEN_FLOAT,idx);
		vst1q_f32(dst + len - i - SSE_LEN_FLOAT, src_tmp);  // store the flipped vector
		vst1q_f32(dst + len - i - 2 * SSE_LEN_FLOAT, src_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[len - i - 1] = src[i];
    }
}

static inline void fabs128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);
	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		v4sf dst_tmp = vandq_f32(*(v4sf *) _ps_pos_sign_mask, src_tmp);
		v4sf dst_tmp2 = vandq_f32(*(v4sf *) _ps_pos_sign_mask, src_tmp2);
		vst1q_f32(dst + i, dst_tmp);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static inline void subcrev128f(float *src, float value, float *dst, int len)
{
    const v4sf tmp = vdupq_n_f32(value);

    int stop_len = len / (2*SSE_LEN_FLOAT);
    stop_len *= 2*SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += 2*SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		v4sf dst_tmp = vsubq_f32(tmp, src_tmp);
		v4sf dst_tmp2 = vsubq_f32(tmp, src_tmp2);
		vst1q_f32(dst + i, dst_tmp);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = value - src[i];
    }
}

static inline void sum128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    __attribute__((aligned(SSE_LEN_BYTES))) float accumulate[SSE_LEN_FLOAT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tmp_acc = 0.0f;
    v4sf vec_acc1 = vdupq_n_f32(0.0f);  // initialize the vector accumulator
    v4sf vec_acc2 = vdupq_n_f32(0.0f);  // initialize the vector accumulator

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf vec_tmp1 = vld1q_f32(src + i);
		vec_acc1 = vaddq_f32(vec_acc1, vec_tmp1);
		v4sf vec_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		vec_acc2 = vaddq_f32(vec_acc2, vec_tmp2);
	}

    vec_acc1 = vaddq_f32(vec_acc1, vec_acc2);
    vst1q_f32(accumulate, vec_acc1);

    for (int i = stop_len; i < len; i++) {
        tmp_acc += src[i];
    }

    tmp_acc = tmp_acc + accumulate[0] + accumulate[1] + accumulate[2] + accumulate[3];
    *dst = tmp_acc;
}

static inline void mean128f(float *src, float *dst, int len)
{
    float coeff = 1.0f / ((float) len);
    sum128f(src, dst, len);
    *dst *= coeff;
}

static inline v4sf log_ps(v4sf x)
{
    x = vmaxnmq_f32(x, *(v4sf *) _ps_0); /* force flush to zero on denormal values */
    mve_pred16_t invalid_mask = vcmpleq_f32(x, *(v4sf *) _ps_0);
    v4si ux = vreinterpretq_s32_f32(x);
    v4si emm0 = vshrq_n_s32(ux, 23);

    /* keep only the fractional part */
    ux = vandq_s32(ux, *(v4si *) _ps_inv_mant_mask);
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(*(v4sf *) _ps_0p5));
    x = vreinterpretq_f32_s32(ux);

    emm0 = vsubq_n_s32(emm0, 0x7F);
    v4sf e = vcvtq_f32_s32(emm0);

    e = vaddq_n_f32(e, 1.0f);

    /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
    mve_pred16_t mask = vcmpltq_f32(x, *(v4sf *) _ps_cephes_SQRTHF);
    x = vsubq_n_f32(x, 1.0f);
    e = vsubq_m_n_f32(e, e, 1.0f, mask);
    x = vaddq_m_f32(x, x, x, mask);

    v4sf z = vmulq_f32(x, x);
    v4sf y = vfmasq_n_f32(x, *(v4sf *) _ps_cephes_log_p0, c_cephes_log_p1);
    //printf("y ");print4_4digits(y);printf("\r\n");
    y = vfmasq_n_f32(y, x, c_cephes_log_p2);
    //printf("y2 ");print4_4digits(y);printf("\r\n");
    y = vfmasq_n_f32(y, x, c_cephes_log_p3);
    y = vfmasq_n_f32(y, x, c_cephes_log_p4);
    y = vfmasq_n_f32(y, x, c_cephes_log_p5);
    y = vfmasq_n_f32(y, x, c_cephes_log_p6);
    y = vfmasq_n_f32(y, x, c_cephes_log_p7);
    y = vfmasq_n_f32(y, x, c_cephes_log_p8);
    y = vmulq_f32(y, x);
    y = vmulq_f32(y, z);
    y = vfmaq_n_f32(y, e, c_cephes_log_q1);
    y = vfmaq_n_f32(y, z, -0.5f);

    y = vfmaq_n_f32(y, e, c_cephes_log_q2);
    x = vaddq_f32(x, y);

    x = vorrq_m_f32(x, x, *(v4sf *) _pi32_0xFFFFFFFF, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline void ln128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		vst1q_f32(dst + i, log_ps(vld1q_f32(src + i)));
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = logf(src[i]);
    }
}

static inline v4sf atanhf_ps(v4sf x)
{
    v4sf z, tmp, tmp2, z_first_branch, z_second_branch;
    mve_pred16_t xsup1, xinfmin1, zinf1emin4, zinf0p5;

    z = vandq_f32(*(v4sf *) _ps_pos_sign_mask, x);
    xsup1 = vcmpgeq_f32(x, *(v4sf *) _ps_1);
    xinfmin1 = vcmpleq_f32(x, *(v4sf *) _ps_min1);
    zinf1emin4 = vcmpltq_f32(z, *(v4sf *) _ps_1emin4);
    zinf0p5 = vcmpltq_f32(z, *(v4sf *) _ps_0p5);

    // First branch
    tmp = vmulq_f32(x, x);
    z_first_branch = vfmasq_n_f32(tmp, *(v4sf *) _ps_ATANH_P0, ATANH_P1);
    z_first_branch = vfmasq_n_f32(z_first_branch, tmp, ATANH_P2);
    z_first_branch = vfmasq_n_f32(z_first_branch, tmp, ATANH_P3);
    z_first_branch = vfmasq_n_f32(z_first_branch, tmp, ATANH_P4);
    z_first_branch = vmulq_f32(z_first_branch, tmp);
    z_first_branch = vfmaq_f32(x, z_first_branch, x);

    // Second branch
    tmp = vsubq_f32(*(v4sf *) _ps_1, x);
    tmp2 = vrecipq_f32(tmp);
    tmp = vfmaq_f32(tmp2, x, tmp2);
    z_second_branch = log_ps(tmp);
    z_second_branch = vmulq_n_f32(z_second_branch, 0.5f);

    z = vpselq_f32(z_first_branch, z_second_branch, zinf0p5);
    z = vpselq_f32(x, z, zinf1emin4);

    z = vpselq_f32(*(v4sf *) _ps_MAXNUMF, z, xsup1);
    z = vpselq_f32(*(v4sf *) _ps_minMAXNUMF, z, xinfmin1);

    return (z);
}

static inline void atanh128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		vst1q_f32(dst + i, atanhf_ps(src_tmp));
	}
    for (int i = stop_len; i < len; i++) {
        dst[i] = atanhf(src[i]);
    }
}

static inline v4sf atanf_ps(v4sf xx)
{
    v4sf x, y, z;
    v4sf sign;
    mve_pred16_t suptan3pi8, inftan3pi8suppi8;
    v4sf tmp, tmp2;

    x = vandq_f32(*(v4sf *) _ps_pos_sign_mask, xx);
    sign = vandq_f32(xx, *(v4sf *) _ps_sign_mask);

    /* range reduction */
    suptan3pi8 = vcmpgtq_f32(x, *(v4sf *) _ps_TAN3PI8F);  // if( x > tan 3pi/8 )
    tmp = vminrecipq_f32(x);
    x = vpselq_f32(tmp, x, suptan3pi8);
    y = vpselq_f32(*(v4sf *) _ps_PIO2F, *(v4sf *) _ps_0, suptan3pi8);

    mve_pred16_t letan3pi8 = vcmpleq_f32(x, *(v4sf *) _ps_TAN3PI8F);
    mve_pred16_t gttanpi8 = vcmpgtq_f32(x, *(v4sf *) _ps_TANPI8F);
    inftan3pi8suppi8 = and_pred(letan3pi8, gttanpi8);  // if( x > tan 3pi/8 )

    tmp = vsubq_n_f32(x, 1.0f);
    tmp2 = vaddq_n_f32(x, 1.0f);
    tmp = vdivq_f32(tmp, tmp2);

    x = vpselq_f32(tmp, x, inftan3pi8suppi8);
    y = vpselq_f32(*(v4sf *) _ps_PIO4F, y, inftan3pi8suppi8);

    z = vmulq_f32(x, x);

    tmp = vfmasq_n_f32(z, *(v4sf *) _ps_ATAN_P0, ATAN_P1);
    tmp = vfmasq_n_f32(tmp, z, ATAN_P2);
    tmp = vfmasq_n_f32(tmp, z, ATAN_P3);
    tmp = vmulq_f32(z, tmp);
    tmp = vfmasq_f32(tmp, x, x);

    y = vaddq_f32(y, tmp);
    y = veorq_f32(y, sign);
    return (y);
}

static inline void atan128f(float *src, float *dst, int len)
{
    int stop_len = len / SSE_LEN_FLOAT;
    stop_len *= SSE_LEN_FLOAT;

	for (int i = 0; i < stop_len; i += SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		vst1q_f32(dst + i, atanf_ps(src_tmp));
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = atanf(src[i]);
    }
}

//rounds away from zero like IPP, whereas SSE/AVx/AVX512 rounds to nearest even (IEEE-754)
static inline void rint128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		v4sf dst_tmp = vrndnq_f32(src_tmp);
		v4sf dst_tmp2 = vrndnq_f32(src_tmp2);
		vst1q_f32(dst + i, dst_tmp);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = rintf(src[i]);
    }
}


//rounds away from zero like IPP, whereas SSE/AVx/AVX512 rounds to nearest even (IEEE-754)
static inline void round128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
        v4sf src_tmp = vld1q_f32(src + i);
        v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
#if 1
        v4sf dst_tmp = vrndaq(src_tmp);
        v4sf dst_tmp2 = vrndaq(src_tmp2);
#else
        v4sf spe1 = vandq_f32(src_tmp, *(v4sf*)_ps_sign_mask);
        spe1 = vorrq_f32(spe1,*(v4sf*)_ps_mid_mask);
        spe1 = vaddq_f32(src_tmp, spe1);
        v4sf spe2 = vandq_f32(src_tmp2, *(v4sf*)_ps_sign_mask);
        spe2 = vorrq_f32(spe2,*(v4sf*)_ps_mid_mask);
        spe2 = vaddq_f32(src_tmp2, spe2);
        v4sf dst_tmp = vrndq_f32(spe1);
        v4sf dst_tmp2 = vrndq_f32(spe2);
#endif
        vst1q_f32(dst + i, dst_tmp);
        vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = roundf(src[i]);
    }
}

static inline void ceil128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		v4sf dst_tmp = vrndpq_f32(src_tmp);
		v4sf dst_tmp2 = vrndpq_f32(src_tmp2);
		vst1q_f32(dst + i, dst_tmp);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = ceilf(src[i]);
    }
}

static inline void floor128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		v4sf dst_tmp = vrndmq_f32(src_tmp);
		v4sf dst_tmp2 = vrndmq_f32(src_tmp2);
		vst1q_f32(dst + i, dst_tmp);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = floorf(src[i]);
    }
}

static inline void trunc128f(float *src, float *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src_tmp = vld1q_f32(src + i);
		v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		v4sf dst_tmp = vrndq_f32(src_tmp);
		v4sf dst_tmp2 = vrndq_f32(src_tmp2);
		vst1q_f32(dst + i, dst_tmp);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, dst_tmp2);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = truncf(src[i]);
    }
}

static inline void cplxvecmul128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

#if 1
	for (int i = 0; i < 2 * stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sf src1_vec = vld1q_f32((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
		v4sf src2_vec = vld1q_f32((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
		v4sf src1_vec2 = vld1q_f32((float *) (src1) + i + SSE_LEN_FLOAT);  // a0a1a2a3, b0b1b2b3
		v4sf src2_vec2 = vld1q_f32((float *) (src2) + i + SSE_LEN_FLOAT);  // c0c1c2c3 d0d1d2d3
		v4sf dst_vec = vcmulq(src1_vec,src2_vec);
		dst_vec = vcmlaq_rot90(dst_vec, src1_vec,src2_vec);
		v4sf dst_vec2 = vcmulq(src1_vec2,src2_vec2);
		dst_vec2 = vcmlaq_rot90(dst_vec2, src1_vec2,src2_vec2);
		vst1q_f32((float *) (dst) + i, dst_vec);
		vst1q_f32((float *) (dst) + i + SSE_LEN_FLOAT, dst_vec2);
	}
#else
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

	for (int i = 0; i < 2 * stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sfx2 src1_split = vld2q_f32((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
		v4sfx2 src2_split = vld2q_f32((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
		v4sf ac = vmulq_f32(src1_split.val[0], src2_split.val[0]);     // ac
		v4sf ad = vmulq_f32(src1_split.val[0], src2_split.val[1]);     // ad
		v4sfx2 dst_split;
		dst_split.val[0] = vfmsq_f32(ac, src1_split.val[1], src2_split.val[1]); //ac - bd
		dst_split.val[1] = vfmasq_f32(src1_split.val[1], src2_split.val[0], ad);
		vst2q_f32((float *) (dst) + i, dst_split);
	}
#endif
    for (int i = stop_len; i < len; i++) {
        dst[i].re = (src1[i].re * src2[i].re) - src1[i].im * src2[i].im;
        dst[i].im = src1[i].re * src2[i].im + (src2[i].re * src1[i].im);
    }
}

static inline void cplxvecdiv128f(complex32_t *src1, complex32_t *src2, complex32_t *dst, int len)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= 2 * SSE_LEN_FLOAT;

	for (int i = 0; i < 2 * stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4sfx2 src1_split = vld2q_f32((float *) (src1) + i);  // a0a1a2a3, b0b1b2b3
		v4sfx2 src2_split = vld2q_f32((float *) (src2) + i);  // c0c1c2c3 d0d1d2d3
		v4sf c2 = vmulq_f32(src2_split.val[0], src2_split.val[0]);
		v4sf c2d2 = vfmaq_f32(c2, src2_split.val[1], src2_split.val[1]);
		v4sf ac = vmulq_f32(src1_split.val[0], src2_split.val[0]);     // ac
		v4sf bc = vmulq_f32(src1_split.val[1], src2_split.val[0]);     // bc
		v4sfx2 dst_split;
		dst_split.val[0] = vfmaq_f32(ac, src1_split.val[1], src2_split.val[1]);
		dst_split.val[1] = vfmsq_f32(bc, src1_split.val[0], src2_split.val[1]);
		c2d2 = vrecipq_f32(c2d2);
		dst_split.val[0] = vmulq_f32(dst_split.val[0], c2d2);
		dst_split.val[1] = vmulq_f32(dst_split.val[1], c2d2);
		vst2q_f32((float *) (dst) + i, dst_split);
	}

    for (int i = stop_len; i < len; i++) {
        float c2d2 = (src2[i].re * src2[i].re) + src2[i].im * src2[i].im;
        dst[i].re = ((src1[i].re * src2[i].re) + (src1[i].im * src2[i].im)) / c2d2;
        dst[i].im = (-(src1[i].re * src2[i].im) + (src2[i].re * src1[i].im)) / c2d2;
    }
}

static inline void vectorSlope128f(float *dst, int len, float offset, float slope)
{
    float coeff[SSE_LEN_FLOAT] __attribute__((aligned(SSE_LEN_BYTES)));
    coeff[0] = 0.0f;
    coeff[1] = slope;
    coeff[2] = 2.0f*slope;
    coeff[3] = 3.0f*slope;

    v4sf coef = vld1q_f32(coeff);
    float slope8 = 8.0f * slope;
    v4sf curVal = vaddq_n_f32(coef, offset);
    v4sf curVal2 = vaddq_n_f32(curVal, 4.0f * slope);

    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    if (len >= 2*SSE_LEN_FLOAT) {
    		vst1q_f32(dst + 0, curVal);
    		vst1q_f32(dst + SSE_LEN_FLOAT, curVal2);
            for (int i = 2 * SSE_LEN_FLOAT; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
                curVal = vaddq_n_f32(curVal, slope8);
                vst1q_f32(dst + i, curVal);
                curVal2 = vaddq_n_f32(curVal2, slope8);
                vst1q_f32(dst + i + SSE_LEN_FLOAT, curVal2);
            }
    }

    for (int i = stop_len; i < len; i++) {
        dst[i] = offset + slope * (float) i;
    }
}

static inline void convertInt16ToFloat32_128(const int16_t *__restrict__ src, float  *__restrict__ dst, int len, int scale_factor)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    float scale_fact_mult;
    if(scale_factor >= 0)
    	scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    else
    	scale_fact_mult = (float) (1 << -scale_factor);

	for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		v4si low = vldrhq_s32(src + i);  // loads widening 1 2 3 4
		v4si high = vldrhq_s32(src + i + SSE_LEN_FLOAT);  // loads widening 5 6 7 8
		// convert the vector to float and scale it
		v4sf floatlo = vcvtq_f32_s32(low);
		floatlo = vmulq_n_f32(floatlo, scale_fact_mult);
		v4sf floathi = vcvtq_f32_s32(high);
		floathi = vmulq_n_f32(floathi, scale_fact_mult);
		vst1q_f32(dst + i, floatlo);
		vst1q_f32(dst + i + SSE_LEN_FLOAT, floathi);
	}

    for (int i = stop_len; i < len; i++) {
        dst[i] = (float) src[i] * scale_fact_mult;
    }
}

static inline void convertFloat32ToI16_128(const float  *__restrict__ src, int16_t *__restrict__ dst, int len, int rounding_mode, int scale_factor)
{
    int stop_len = len / (2 * SSE_LEN_FLOAT);
    stop_len *= (2 * SSE_LEN_FLOAT);

    float scale_fact_mult;
    if(scale_factor >= 0)
    	scale_fact_mult = 1.0f / (float) (1 << scale_factor);
    else
    	scale_fact_mult = (float) (1 << -scale_factor);

    int rounding_ori = fegetround();

    if (rounding_mode == RndZero) {
        fesetround(FE_TOWARDZERO);
    } else if (rounding_mode == RndNear){
        fesetround(FE_TONEAREST);
    }

    if(rounding_mode != RndFinancial){
	  for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		  v4sf src_tmp1 = vld1q_f32(src + i);
		  v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		  v4sf tmp1 = vmulq_n_f32(src_tmp1, scale_fact_mult);
		  v4sf tmp2 = vmulq_n_f32(src_tmp2, scale_fact_mult);
		  v4si tmp1_int = vcvtq_s32_f32(tmp1);
		  v4si tmp2_int = vcvtq_s32_f32(tmp2);
		  vstrhq_s32(dst + i, tmp1_int);
		  vstrhq_s32(dst + i + SSE_LEN_FLOAT, tmp2_int);
	  }
    } else {
	  for (int i = 0; i < stop_len; i += 2 * SSE_LEN_FLOAT) {
		  v4sf src_tmp1 = vld1q_f32(src + i);
		  v4sf src_tmp2 = vld1q_f32(src + i + SSE_LEN_FLOAT);
		  v4sf tmp1 = vmulq_n_f32(src_tmp1, scale_fact_mult);
		  v4sf tmp2 = vmulq_n_f32(src_tmp2, scale_fact_mult);
		  tmp1 = vrndaq(tmp1);
		  tmp2 = vrndaq(tmp2);
		  v4si tmp1_int = vcvtq_s32_f32(tmp1);
		  v4si tmp2_int = vcvtq_s32_f32(tmp2);
		  vstrhq_s32(dst + i, tmp1_int);
		  vstrhq_s32(dst + i + SSE_LEN_FLOAT, tmp2_int);
	  }
    }

    if (rounding_mode == RndFinancial) {
        for (int i = stop_len; i < len; i++) {
            float tmp = roundf(src[i] * scale_fact_mult);
            tmp = (int16_t)fminf(tmp,32767.0f);
			dst[i] = (int16_t)fmaxf(-32768.0f, tmp);
        }
    } else {
        // Default round toward zero
        for (int i = stop_len; i < len; i++) {
            float tmp = rintf(src[i] * scale_fact_mult);
            tmp = (int16_t)fminf(tmp,32767.0f);
			dst[i] = (int16_t)fmaxf(-32768.0f, tmp);
        }
        fesetround(rounding_ori);
    }
}
