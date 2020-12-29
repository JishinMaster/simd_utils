#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "simd_utils.h"

#ifdef IPP
#include <ipp.h>
#endif

#ifdef MKL
#include <mkl.h>
#endif


static float DP1 = 0.78515625;
static float DP2 = 2.4187564849853515625e-4;
static float DP3 = 3.77489497744594108e-8;
float FOPI = 1.27323954473516; /* 4/pi */
static float lossth = 8192.;
#define TLOSS 5 /* total loss of precision */

static float tanf_cephes(float xx)
{
    float x, y, z, zz;
    long j;
    int sign;


    /* make argument positive but save the sign */
    if (xx < 0.0) {
        x = -xx;
        sign = -1;
    } else {
        x = xx;
        sign = 1;
    }

    if (x > lossth) {
        printf("Error > TLOSS %f %d\n", x, TLOSS);
        return (0.0);
    }

    /* compute x mod PIO4 */
    j = FOPI * x; /* integer part of x/(PI/4) */
    y = j;

    /* map zeros and singularities to origin */
    if (j & 1) {
        j += 1;
        y += 1.0;
    }

    z = ((x - y * DP1) - y * DP2) - y * DP3;
    //z = (x - y * DP1);
    //z = z -y*DP2;
    //z = z -y*DP3;
    //printf("%13.8g\n",z);
    //printf("%13.8g\n",y);
    //printf("%13.8g\n",z);

    zz = z * z;

    if (x > 1.0e-4) {
        /* 1.7e-8 relative error in [-pi/4, +pi/4] */
        y =
            (((((9.38540185543E-3 * zz + 3.11992232697E-3) * zz + 2.44301354525E-2) * zz + 5.34112807005E-2) * zz + 1.33387994085E-1) * zz + 3.33331568548E-1) * zz * z + z;
    } else {
        y = z;
    }

    if (j & 2) {
        y = -1.0 / y;
    }

    if (sign < 0)
        y = -y;

    return (y);
}

#if 0
/* natural logarithm computed for 4 simultaneous float 
   return NaN for x <= 0
 */
/* __m128 is ugly to write */
typedef __m128d v2df;  // vector of 4 float (sse1)
typedef __m128i v2di;  // vector of 4 float (sse1)
v2df log_pd(v2df x) {
	v2di emm0;

	v2df one = *(v2df*)_pd_1;

	v2df invalid_mask = _mm_cmple_pd(x, _mm_setzero_pd());

	x = _mm_max_pd(x, *(v2df*)_pd_min_norm_pos);  /* cut off denormalized stuff */

	emm0 = _mm_srli_epi64(_mm_castpd_si128(x), 23+32); // ????

	/* keep only the fractional part */
	x = _mm_and_pd(x, *(v2df*)_pd_inv_mant_mask);
	x = _mm_or_pd(x, *(v2df*)_pd_0p5);

	emm0 = _mm_sub_epi64(emm0, *(v2di*)_pi32_0x7f);
	v2df e = _mm_cvtepi32_pd(emm0);

	e = _mm_add_pd(e, one);

	/* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
	 */
	v2df mask = _mm_cmplt_pd(x, *(v2df*)_pd_cephes_SQRTHF);
	v2df tmp = _mm_and_pd(x, mask);
	x = _mm_sub_pd(x, one);
	e = _mm_sub_pd(e, _mm_and_pd(one, mask));
	x = _mm_add_pd(x, tmp);


	v2df z = _mm_mul_pd(x,x);

	v2df y = *(v2df*)_pd_cephes_log_p0;
	y = _mm_mul_pd(y, x);
	y = _mm_add_pd(y, *(v2df*)_pd_cephes_log_p1);
	y = _mm_mul_pd(y, x);
	y = _mm_add_pd(y, *(v2df*)_pd_cephes_log_p2);
	y = _mm_mul_pd(y, x);
	y = _mm_add_pd(y, *(v2df*)_pd_cephes_log_p3);
	y = _mm_mul_pd(y, x);
	y = _mm_add_pd(y, *(v2df*)_pd_cephes_log_p4);
	y = _mm_mul_pd(y, x);
	y = _mm_add_pd(y, *(v2df*)_pd_cephes_log_p5);
	y = _mm_mul_pd(y, x);

	y = _mm_mul_pd(y, z);


	tmp = _mm_mul_pd(e, *(v2df*)_pd_cephes_log_q1);
	y = _mm_add_pd(y, tmp);
	tmp = _mm_mul_pd(z, *(v2df*)_pd_0p5);
	y = _mm_sub_pd(y, tmp);

	tmp = _mm_mul_pd(e, *(v2df*)_pd_cephes_log_q2);
	x = _mm_add_pd(x, y);
	x = _mm_add_pd(x, tmp);
	x = _mm_or_pd(x, invalid_mask); // negative arg will be NAN
	return x;
}
#endif


typedef ALIGN16_BEG union {
    float f[4];
    int i[4];
    v4sf v;
} ALIGN16_END V4SF;

/*void print4(__m128 v) {
	float *p = (float*)&v;
#ifndef USE_SSE2
	_mm_empty();
#endif
	printf("[%13.8g, %13.8g, %13.8g, %13.8g]", p[0], p[1], p[2], p[3]);
}*/


#ifdef AVX
typedef ALIGN32_BEG union {
    float f[8];
    int i[8];
    v8sf v;
} ALIGN32_END V8SF;

/*void print8(__m256 v) {
	float *p = (float*)&v;
#ifndef USE_SSE2
	_mm_empty();
#endif
	printf("[%13.8g, %13.8g, %13.8g, %13.8g %13.8g, %13.8g, %13.8g, %13.8g]", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}*/
#endif

float l2_err(float *test, float *ref, int len)
{
    float l2_err = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (ref[i] - test[i]) * (ref[i] - test[i]);
    }

#ifdef RELEASE
    if (l2_err > 0.00001f)
        printf("L2 ERR %0.7f\n", l2_err);
#else
    printf("L2 ERR %0.7f\n", l2_err);
#endif
    return l2_err;
}

float l2_err_u8(uint8_t *test, uint8_t *ref, int len)
{
    float l2_err = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
    }

#ifdef RELEASE
    if (l2_err > 0.00001f)
        printf("L2 ERR %0.7f\n", l2_err);
#else
    printf("L2 ERR %0.7f\n", l2_err);
#endif
    return l2_err;
}

float l2_err_i32(int32_t *test, int32_t *ref, int len)
{
    float l2_err = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
    }

#ifdef RELEASE
    if (l2_err > 0.00001f)
        printf("L2 ERR %0.7f\n", l2_err);
#else
    printf("L2 ERR %0.7f\n", l2_err);
#endif

    return l2_err;
}

float l2_errd(double *test, double *ref, int len)
{
    double l2_err = 0.0;

    for (int i = 0; i < len; i++) {
        l2_err += (ref[i] - test[i]) * (ref[i] - test[i]);
    }

#ifdef RELEASE
    if (l2_err > 0.00001)
        printf("L2 ERR %0.7f\n", l2_err);
#else
    printf("L2 ERR %0.7f\n", l2_err);
#endif

    return l2_err;
}

int main(int argc, char **argv)
{
#if 0
	V4SF sseVecIn = {M_PI,-M_PI/2.0f,-M_PI/4.0f,0.0f};
	V4SF sseVecOut;

	sseVecOut.v = cos_ps(sseVecIn.v);
	print4(sseVecOut.v);
	printf("\n");
	sseVecOut.v = sin_ps(sseVecIn.v);
	print4(sseVecOut.v);
	printf("\n");

	set128f(sseVecOut.f, 14.5f, 4);
	print4(sseVecOut.v);
	printf("\n");

	sseVecOut.v = vector_abs(sseVecIn.v);
	print4(sseVecIn.v);
	printf("\n");
	print4(sseVecOut.v);
	printf("\n");


	V8SF avxVecIn = {{M_PI,-M_PI/2.0f,-M_PI/4.0f,0.0f, M_PI,-M_PI/2.0f,-M_PI/4.0f,0.0f}};
	V8SF avxVecOut;

	avxVecOut.v = vector256_abs(avxVecIn.v);
	print8(avxVecIn.v);
	printf("\n");
	print8(avxVecOut.v);
	printf("\n");


#endif

#ifdef IPP
    ippInit();
    const IppLibraryVersion *lib;
    lib = ippGetLibVersion();
    IppStatus status;
    Ipp64u mask, emask;
    printf("%s %s\n", lib->Name, lib->Version);

    /* Get CPU features and features enabled with selected library level */
    status = ippGetCpuFeatures(&mask, 0);
    if (ippStsNoErr == status) {
        emask = ippGetEnabledCpuFeatures();
        printf("Features supported by CPU\tby Intel IPP\n");
        printf("-----------------------------------------\n");
        printf("  ippCPUID_MMX        = ");
        printf("%c\t%c\t", (mask & ippCPUID_MMX) ? 'Y' : 'N', (emask & ippCPUID_MMX) ? 'Y' : 'N');
        printf("Intel(R) architecture MMX(TM) technology supported\n");
        printf("  ippCPUID_SSE        = ");
        printf("%c\t%c\t", (mask & ippCPUID_SSE) ? 'Y' : 'N', (emask & ippCPUID_SSE) ? 'Y' : 'N');
        printf("Intel(R) Streaming SIMD Extensions\n");
        printf("  ippCPUID_SSE2       = ");
        printf("%c\t%c\t", (mask & ippCPUID_SSE2) ? 'Y' : 'N', (emask & ippCPUID_SSE2) ? 'Y' : 'N');
        printf("Intel(R) Streaming SIMD Extensions 2\n");
        printf("  ippCPUID_SSE3       = ");
        printf("%c\t%c\t", (mask & ippCPUID_SSE3) ? 'Y' : 'N', (emask & ippCPUID_SSE3) ? 'Y' : 'N');
        printf("Intel(R) Streaming SIMD Extensions 3\n");
        printf("  ippCPUID_SSSE3      = ");
        printf("%c\t%c\t", (mask & ippCPUID_SSSE3) ? 'Y' : 'N', (emask & ippCPUID_SSSE3) ? 'Y' : 'N');
        printf("Supplemental Streaming SIMD Extensions 3\n");
        printf("  ippCPUID_MOVBE      = ");
        printf("%c\t%c\t", (mask & ippCPUID_MOVBE) ? 'Y' : 'N', (emask & ippCPUID_MOVBE) ? 'Y' : 'N');
        printf("The processor supports MOVBE instruction\n");
        printf("  ippCPUID_SSE41      = ");
        printf("%c\t%c\t", (mask & ippCPUID_SSE41) ? 'Y' : 'N', (emask & ippCPUID_SSE41) ? 'Y' : 'N');
        printf("Intel(R) Streaming SIMD Extensions 4.1\n");
        printf("  ippCPUID_SSE42      = ");
        printf("%c\t%c\t", (mask & ippCPUID_SSE42) ? 'Y' : 'N', (emask & ippCPUID_SSE42) ? 'Y' : 'N');
        printf("Intel(R) Streaming SIMD Extensions 4.2\n");
        printf("  ippCPUID_AVX        = ");
        printf("%c\t%c\t", (mask & ippCPUID_AVX) ? 'Y' : 'N', (emask & ippCPUID_AVX) ? 'Y' : 'N');
        printf("Intel(R) Advanced Vector Extensions instruction set\n");
        printf("  ippAVX_ENABLEDBYOS  = ");
        printf("%c\t%c\t", (mask & ippAVX_ENABLEDBYOS) ? 'Y' : 'N', (emask & ippAVX_ENABLEDBYOS) ? 'Y' : 'N');
        printf("The operating system supports Intel(R) AVX\n");
        printf("  ippCPUID_AES        = ");
        printf("%c\t%c\t", (mask & ippCPUID_AES) ? 'Y' : 'N', (emask & ippCPUID_AES) ? 'Y' : 'N');
        printf("AES instruction\n");
        printf("  ippCPUID_SHA        = ");
        printf("%c\t%c\t", (mask & ippCPUID_SHA) ? 'Y' : 'N', (emask & ippCPUID_SHA) ? 'Y' : 'N');
        printf("Intel(R) SHA new instructions\n");
        printf("  ippCPUID_CLMUL      = ");
        printf("%c\t%c\t", (mask & ippCPUID_CLMUL) ? 'Y' : 'N', (emask & ippCPUID_CLMUL) ? 'Y' : 'N');
        printf("PCLMULQDQ instruction\n");
        printf("  ippCPUID_RDRAND     = ");
        printf("%c\t%c\t", (mask & ippCPUID_RDRAND) ? 'Y' : 'N', (emask & ippCPUID_RDRAND) ? 'Y' : 'N');
        printf("Read Random Number instructions\n");
        printf("  ippCPUID_F16C       = ");
        printf("%c\t%c\t", (mask & ippCPUID_F16C) ? 'Y' : 'N', (emask & ippCPUID_F16C) ? 'Y' : 'N');
        printf("Float16 instructions\n");
        printf("  ippCPUID_AVX2       = ");
        printf("%c\t%c\t", (mask & ippCPUID_AVX2) ? 'Y' : 'N', (emask & ippCPUID_AVX2) ? 'Y' : 'N');
        printf("Intel(R) Advanced Vector Extensions 2 instruction set\n");
        printf("  ippCPUID_AVX512F    = ");
        printf("%c\t%c\t", (mask & ippCPUID_AVX512F) ? 'Y' : 'N', (emask & ippCPUID_AVX512F) ? 'Y' : 'N');
        printf("Intel(R) Advanced Vector Extensions 3.1 instruction set\n");
        printf("  ippCPUID_AVX512CD   = ");
        printf("%c\t%c\t", (mask & ippCPUID_AVX512CD) ? 'Y' : 'N', (emask & ippCPUID_AVX512CD) ? 'Y' : 'N');
        printf("Intel(R) Advanced Vector Extensions CD (Conflict Detection) instruction set\n");
        printf("  ippCPUID_AVX512ER   = ");
        printf("%c\t%c\t", (mask & ippCPUID_AVX512ER) ? 'Y' : 'N', (emask & ippCPUID_AVX512ER) ? 'Y' : 'N');
        printf("Intel(R) Advanced Vector Extensions ER instruction set\n");
        printf("  ippCPUID_ADCOX      = ");
        printf("%c\t%c\t", (mask & ippCPUID_ADCOX) ? 'Y' : 'N', (emask & ippCPUID_ADCOX) ? 'Y' : 'N');
        printf("ADCX and ADOX instructions\n");
        printf("  ippCPUID_RDSEED     = ");
        printf("%c\t%c\t", (mask & ippCPUID_RDSEED) ? 'Y' : 'N', (emask & ippCPUID_RDSEED) ? 'Y' : 'N');
        printf("The RDSEED instruction\n");
        printf("  ippCPUID_PREFETCHW  = ");
        printf("%c\t%c\t", (mask & ippCPUID_PREFETCHW) ? 'Y' : 'N', (emask & ippCPUID_PREFETCHW) ? 'Y' : 'N');
        printf("The PREFETCHW instruction\n");
        printf("  ippCPUID_KNC        = ");
        printf("%c\t%c\t", (mask & ippCPUID_KNC) ? 'Y' : 'N', (emask & ippCPUID_KNC) ? 'Y' : 'N');
        printf("Intel(R) Xeon Phi(TM) Coprocessor instruction set\n");
    }

#endif

    float *inout = NULL, *inout2 = NULL, *inout3 = NULL, *inout4 = NULL, *inout_ref = NULL, *inout2_ref = NULL;
    double *inoutd = NULL, *inoutd_ref = NULL;
    uint8_t *inout_u1 = NULL, *inout_u2 = NULL;
    int16_t *inout_s1 = NULL, *inout_s2 = NULL;
    int32_t *inout_i1 = NULL, *inout_i2 = NULL, *inout_iref = NULL;
    int len = atoi(argv[1]);

#ifndef USE_MALLOC
    posix_memalign((void **) &inout, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout == NULL) {
        printf("posix_memalign inout failed\n");
        return -1;
    }
    posix_memalign((void **) &inout2, atoi(argv[2]), len * sizeof(float));
    if (inout2 == NULL) {
        printf("posix_memalign inout2 failed\n");
        return -1;
    }
    posix_memalign((void **) &inout3, atoi(argv[2]), len * sizeof(float));
    if (inout3 == NULL) {
        printf("posix_memalign inout3 failed\n");
        return -1;
    }
    posix_memalign((void **) &inout4, atoi(argv[2]), len * sizeof(float));
    if (inout4 == NULL) {
        printf("posix_memalign inout4 failed\n");
        return -1;
    }
    posix_memalign((void **) &inout_ref, atoi(argv[2]), len * sizeof(float));
    if (inout_ref == NULL) {
        printf("posix_memalign inout_ref failed\n");
        return -1;
    }
    posix_memalign((void **) &inout2_ref, atoi(argv[2]), len * sizeof(float));
    if (inout2_ref == NULL) {
        printf("posix_memalign inout2_ref failed\n");
        return -1;
    }
    posix_memalign((void **) &inoutd, atoi(argv[2]), len * sizeof(double));
    if (inoutd == NULL) {
        printf("posix_memalign inoutd failed\n");
        return -1;
    }
    posix_memalign((void **) &inoutd_ref, atoi(argv[2]), len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("posix_memalign inoutd_ref failed\n");
        return -1;
    }

    posix_memalign((void **) &inout_u1, atoi(argv[2]), len * sizeof(uint8_t));
    if (inout_u1 == NULL) {
        printf("posix_memalign inout_u1 failed\n");
        return -1;
    }
    posix_memalign((void **) &inout_u2, atoi(argv[2]), len * sizeof(uint8_t));
    if (inout_u2 == NULL) {
        printf("posix_memalign inout_u2 failed\n");
        return -1;
    }

    posix_memalign((void **) &inout_s1, atoi(argv[2]), len * sizeof(int16_t));
    if (inout_s1 == NULL) {
        printf("posix_memalign inout_s1 failed\n");
        return -1;
    }
    posix_memalign((void **) &inout_s2, atoi(argv[2]), len * sizeof(int16_t));
    if (inout_s2 == NULL) {
        printf("posix_memalign inout_s2 failed\n");
        return -1;
    }

    posix_memalign((void **) &inout_i1, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_i1 == NULL) {
        printf("posix_memalign inout_i1 failed\n");
        return -1;
    }
    posix_memalign((void **) &inout_i2, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_i2 == NULL) {
        printf("posix_memalign inout_i2 failed\n");
        return -1;
    }
    posix_memalign((void **) &inout_iref, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_iref == NULL) {
        printf("posix_memalign inout_iref failed\n");
        return -1;
    }


#else
    inout = (float *) malloc(2 * len * sizeof(float));
    if (inout == NULL) {
        printf("malloc inout failed\n");
        return -1;
    }
    inout2 = (float *) malloc(len * sizeof(float));
    if (inout2 == NULL) {
        printf("malloc inout2 failed\n");
        return -1;
    }
    inout3 = (float *) malloc(len * sizeof(float));
    if (inout3 == NULL) {
        printf("malloc inout3 failed\n");
        return -1;
    }
    inout4 = (float *) malloc(len * sizeof(float));
    if (inout4 == NULL) {
        printf("malloc inout4 failed\n");
        return -1;
    }
    inout_ref = (float *) malloc(len * sizeof(float));
    if (inout_ref == NULL) {
        printf("malloc inout_ref failed\n");
        return -1;
    }
    inout2_ref = (float *) malloc(len * sizeof(float));
    if (inout2_ref == NULL) {
        printf("malloc inout2_ref failed\n");
        return -1;
    }
    inoutd = (double *) malloc(len * sizeof(double));
    if (inoutd == NULL) {
        printf("malloc inoutd failed\n");
        return -1;
    }
    inoutd_ref = (double *) malloc(len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("malloc inoutd_ref failed\n");
        return -1;
    }

    inout_u1 = (uint8_t *) malloc(len * sizeof(uint8_t));
    if (inout_u1 == NULL) {
        printf("malloc inout_u1 failed\n");
        return -1;
    }
    inout_u2 = (uint8_t *) malloc(len * sizeof(uint8_t));
    if (inout_u2 == NULL) {
        printf("malloc inout_u2 failed\n");
        return -1;
    }

    inout_s1 = (int16_t *) malloc(len * sizeof(int16_t));
    if (inout_s1 == NULL) {
        printf("malloc inout_s1 failed\n");
        return -1;
    }
    inout_s2 = (int16_t *) malloc(len * sizeof(int16_t));
    if (inout_s2 == NULL) {
        printf("malloc inout_s2 failed\n");
        return -1;
    }


    inout_i1 = (int32_t *) malloc(len * sizeof(int32_t));
    if (inout_i1 == NULL) {
        printf("posix_memalign inout_i1 failed\n");
        return -1;
    }
    inout_i2 = (int32_t *) malloc(len * sizeof(int32_t));
    if (inout_i2 == NULL) {
        printf("posix_memalign inout_i2 failed\n");
        return -1;
    }
    inout_iref = (int32_t *) malloc(len * sizeof(int32_t));
    if (inout_iref == NULL) {
        printf("posix_memalign inout_iref failed\n");
        return -1;
    }

#endif

    struct timespec start, stop;
    double elapsed = 0.0;


    clock_gettime(CLOCK_REALTIME, &start);
    memset(inout_ref, 0.0f, len * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("memset %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    memset(inout_ref, 0.0f, len * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("memset %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    setf_C(inout_ref, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("setf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsZero_32f(inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsZero_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsZero_32f(inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsZero_32f %d %lf\n", len, elapsed);


    clock_gettime(CLOCK_REALTIME, &start);
    ippsSet_32f(0.001f, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSet_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsSet_32f(0.08f, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSet_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    zero128f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    zero128f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero128f %d %lf\n", len, elapsed);


    clock_gettime(CLOCK_REALTIME, &start);
    set128f(inout, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    set128f(inout, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set128f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif



#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    zero256f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    zero256f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    set256f(inout, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    set256f(inout, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set256f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif


    clock_gettime(CLOCK_REALTIME, &start);
    memcpy(inout2_ref, inout, len * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("memcpy %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    memcpy(inout2_ref, inout, len * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("memcpy %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    copyf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copyf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCopy_32f(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCopy_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCopy_32f(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCopy_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    copy128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    copy128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy128f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    copy256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    copy256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy256f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

    /*for(int i = 0; i < len; i++){
		printf("%f ",inout[i]);
	}
	printf("\n");*/

    inout[3] = 0.04f;

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 20.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
    }


    clock_gettime(CLOCK_REALTIME, &start);
    threshold_lt_f_C(inout, inout2_ref, 0.02f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_lt_f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LT_32f(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreashold_LT_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LT_32f(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreashold_LT_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_lt_f(inout, inout2, 0.07f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_lt_f(inout, inout2, 0.02f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_lt_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_lt_f(inout, inout2, 0.07f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_lt_f(inout, inout2, 0.02f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_lt_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif


    /*for (int i = 0; i < len; i++)
{
	printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
}*/


    clock_gettime(CLOCK_REALTIME, &start);
    fabsf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabsf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsZero_32f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsZero_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsZero_32f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsZero_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAbs_32f(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAbs_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAbs_32f(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAbs_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    fabs128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    fabs128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs128f %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    fabs256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    fabs256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif



    for (int i = 0; i < len; i++)
        inoutd[i] = (double) i;



    clock_gettime(CLOCK_REALTIME, &start);
    convert_64f32f_C(inoutd, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert_64f32f_C %d %lf\n", len, elapsed);


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_64f32f(inoutd, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_64f32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_64f32f(inoutd, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_64f32f %d %lf\n", len, elapsed);
#endif



#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    convert128_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert128_64f32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    convert128_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert128_64f32f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convert256_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert256_64f32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    convert256_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert256_64f32f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif
    /*for(int i = 0; i < len; i++){
		printf("%f ",inout[i]);
	}
printf("\n");*/
    /*
	threshold128_gt_f(inout, 0.07f, len);
for(int i = 0; i < len; i++){
		printf("%f ",inout[i]);
	}

printf("\n");
	 */


    for (int i = 0; i < len; i++) {
        inout[i] = (float) i;
        inout_ref[i] = inout[i];
    }
    float mean = 0.0f, mean_ref = 0.0f;


    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / (float) 2.7f;
        inout_ref[i] = inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    meanf_C(inout_ref, &mean_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("meanf_C %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    mean = 0.0f;
    ippsMean_32f(inout_ref, len, &mean_ref, (IppHintAlgorithm) 0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMean_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMean_32f(inout_ref, len, &mean_ref, (IppHintAlgorithm) 0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMean_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    mean128f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mean128f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean128f %d %lf\n", len, elapsed);

    printf("mean %f ref %f\n", mean, mean_ref);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    mean256f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mean256f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean256f %d %lf\n", len, elapsed);

    printf("mean %f ref %f\n", mean, mean_ref);
#endif


    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 10.5f;
        inout2[i] = (float) i / 35.77f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    magnitudef_C_split(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitudef_C_split %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude128f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitude128f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude128f_split %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);

    /*for(int i = 0; i < len; i++){
		printf("%f %f %f %f\n",inout[i],inout2[i],inout2_ref[i],inout_ref[i]);
	}*/
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude256f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude256f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitude256f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude256f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif


    int j = 0;
    for (int i = 0; i < 2 * len; i+=2) {
        inout[i]   = (float) j / 10.5f;
        inout[i+1] = (float) j / 35.77f;
        j+= 1;
    }
    for (int i = 0; i < len; i++) {
        inout_ref[i] = 0.0f;
        inout2_ref[i] = 0.0f;
    }

    ////////////////////////:

    clock_gettime(CLOCK_REALTIME, &start);
    magnitudef_C_interleaved((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitudef_C_interleaved %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32fc((const Ipp32fc *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32fc((const Ipp32fc *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32fc %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude128f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitude128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude128f_interleaved %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif


    /*int k = 0;
	for(int i = 0; i < len; i++){
		printf("Int : %f %f || %f %f\n", inout[k],inout[k+1], inout_ref[i],inout2_ref[i]);
		k+=2;
	}*/



    memset(inout_ref, 0.5555f, len * sizeof(float));
    memset(inout2_ref, 0.5555f, len * sizeof(float));


    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32fc_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32fc_A24 %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len / 2);
#endif

    /*for(int i = 0; i < len; i+=2){
		printf("%f %f // %f %f // %f %f || %f %f\n",inout[i],inout[i+1],inout2[i],inout2[i+1],inout_ref[i],inout_ref[i+1], inout2_ref[i], inout2_ref[i+1]);
	}*/

    /*for(int i = 0; i < len/2; i+=2){
		printf("%f %f\n",inout_ref[i],inout_ref[i+1]);
	}*/

    ///////////////////////

    clock_gettime(CLOCK_REALTIME, &start);
    addcf_C(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addcf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAddC_32f(inout, 5.7f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAddC_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAddC_32f(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAddC_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    addc128f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    addc128f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    addc256f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    addc256f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc256f %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
#endif


    clock_gettime(CLOCK_REALTIME, &start);
    mulcf_C(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulcf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMulC_32f(inout, 5.7f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulC_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMulC_32f(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulC_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    mulc128f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mulc128f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    mulc256f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mulc256f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc256f %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
#endif

    /////////////
    clock_gettime(CLOCK_REALTIME, &start);
    mulf_C(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulcf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulC_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulC_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    mul128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mul128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    mul256f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mul256f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif


    /////////:
    clock_gettime(CLOCK_REALTIME, &start);
    sinf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinf_C %d %lf\n", len, elapsed);


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSin_32f_A24(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSin_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsSin_32f_A24(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSin_32f_A24 %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    sin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sin256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sin256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif



    clock_gettime(CLOCK_REALTIME, &start);
    cosf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCos_32f_A24(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCos_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCos_32f_A24(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCos_32f_A24 %d %lf\n", len, elapsed);
#endif


#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cos128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cos128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif



#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cos256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cos256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif


    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 1.82f;
        inout_ref[i] = inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    lnf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("lnf_C %d %lf\n", len, elapsed);


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsLn_32f(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsLn_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsLn_32f(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsLn_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    ln_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ln_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    ln_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ln_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 1.82f;
        inoutd_ref[i] = 0.0;
        inoutd[i] = 0.0;
    }



    clock_gettime(CLOCK_REALTIME, &start);
    convert_32f64f_C(inout, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert_32f64f_C %d %lf\n", len, elapsed);

#ifndef ARM

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f64f(inout, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f64f(inout, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f64f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    convert128_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert128_32f64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    convert128_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert128_32f64f %d %lf\n", len, elapsed);

    l2_errd(inoutd, inoutd_ref, len);
    /*for(int i =0; i < 16; i++){
		printf("%lf %lf %f\n",inoutd_ref[i],inoutd[i],inout[i]);
	}*/
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convert256_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert256_32f64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    convert256_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert256_32f64f %d %lf\n", len, elapsed);

    l2_errd(inoutd, inoutd_ref, len);
    /*for(int i =0; i < 16; i++){
		printf("%lf %lf %f\n",inoutd_ref[i],inoutd[i],inout[i]);
	}*/
#endif

#endif

    clock_gettime(CLOCK_REALTIME, &start);
    flipf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip256f %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsFlip_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFlip_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsFlip_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFlip_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    flip128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    flip128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    flip256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    flip256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif


    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 1.82f;
        inout_ref[i] = inout[i];
    }



    clock_gettime(CLOCK_REALTIME, &start);
    floorf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor256f %d %lf\n", len, elapsed);

    //#ifndef ARM

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsFloor_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFloor_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsFloor_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFloor_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    floor128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    floor128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    floor256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    floor256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif


    clock_gettime(CLOCK_REALTIME, &start);
    ceilf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceilf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCeil_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCeil_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCeil_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCeil_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    ceil128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ceil128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    ceil256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ceil256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif


    clock_gettime(CLOCK_REALTIME, &start);
    roundf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("roundf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsRound_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRound_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsRound_32f(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRound_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    round128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("round128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    round128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("round128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    round256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("round256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    round256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("round256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    //#endif //ARM

    clock_gettime(CLOCK_REALTIME, &start);
    tanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsTan_32f_A24(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTan_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsTan_32f_A24(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTan_32f_A24 %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    tan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    tan128f_naive(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f_old %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tan128f_naive(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f_old %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    tan256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tan256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#endif


    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 1.82f / (float) (5 * len);
    }



    clock_gettime(CLOCK_REALTIME, &start);
    asinf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsin_32f_A24(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsin_32f_A24(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_32f_A24 %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    asin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    asin256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif



    clock_gettime(CLOCK_REALTIME, &start);
    atanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_32f_A24(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_32f_A24(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_32f_A24 %d %lf\n", len, elapsed);
#endif


#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    atan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    atan256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256f %d %lf\n", len, elapsed);
#endif

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (-1.0f * i + 0.15f) / 2.5f / (float) (5 * len);
        inout_ref[i] = 50.0f;
        inout2_ref[i] = 50.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    atan2f_C(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan2_32f_A24(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan2_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan2_32f_A24(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan2_32f_A24 %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    atan2128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan2128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    atan2256f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan2256f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif


    for (int i = 0; i < len; i++) {
        inout_ref[i] = 0.0f;
        inout2_ref[i] = 0.0f;
        inout3[i] = 0.0f;
        inout4[i] = 0.0f;
    }

    //#ifndef ARM


    clock_gettime(CLOCK_REALTIME, &start);
    cplxtorealf_C(inout, inout_ref, inout2_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtorealf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCplxToReal_32fc((const Ipp32fc *) inout, inout_ref, inout2_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCplxToReal_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCplxToReal_32fc((const Ipp32fc *) inout, inout_ref, inout2_ref, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCplxToReal_32fc %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal128f(inout, inout3, inout4, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal128f(inout, inout3, inout4, len / 2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal128f %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len / 2);
    l2_err(inout4, inout2_ref, len / 2);
#endif

#ifdef AVX
    /*clock_gettime(CLOCK_REALTIME,&start);
	cplxtoreal256f(inout, inout3, inout4, len/2);
	clock_gettime(CLOCK_REALTIME,&stop);
	elapsed = (stop.tv_sec - start.tv_sec)*1e6 + (stop.tv_nsec - start.tv_nsec)*1e-3;
	printf("cplxtoreal256f %d %lf\n",len, elapsed);

	clock_gettime(CLOCK_REALTIME,&start);
	cplxtoreal256f(inout, inout3, inout4, len/2);
	clock_gettime(CLOCK_REALTIME,&stop);
	elapsed = (stop.tv_sec - start.tv_sec)*1e6 + (stop.tv_nsec - start.tv_nsec)*1e-3;
	printf("cplxtoreal256f %d %lf\n",len, elapsed);
	l2_err(inout3,inout_ref,len/2);
	l2_err(inout4,inout2_ref,len/2);*/
#endif

    /*for(int i = 0; i < len; i++)
		printf("%f %f || %f %f \n",inout3[i],inout_ref[i], inout4[i],inout2_ref[i]);*/

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (i) *10;
        inout_u1[i] = 0x0;
        inout_u2[i] = 0x0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_C(inout, inout_u2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_C %d %lf\n", len, elapsed);

#ifndef ARM

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f8u_Sfs(inout, inout_u2, len, ippRndZero, 4);  //ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f8u_Sfs(inout, inout_u2, len, ippRndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_128(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_128(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_128 %d %lf\n", len, elapsed);

    l2_err_u8(inout_u1, inout_u2, len);
#endif

#endif
    /*for(int i = 0; i < len; i++)
		printf("%x %x\n" ,inout_u1[i],inout_u2[i]);
	*/

    for (int i = 0; i < len; i++)
        inout_s1[i] = -len + i;

    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_C(inout_s1, inout2_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_16s32f_Sfs(inout_s1, inout2_ref, len, 4);  //ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_16s32f_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_16s32f_Sfs(inout_s1, inout2_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_16s32f_Sfs %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_128(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_128(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_128 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif


    //for(int i = 0; i < len; i++)
    //	printf("%d %f %f %f\n" ,inout_s1[i], (float)inout_s1[i]/(1 << 4),inout_ref[i],inout2_ref[i]);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlopef_C(inout2_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_32f(inout2_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_32f(inout2_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_32f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope128f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope128f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope256f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope256f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif



    clock_gettime(CLOCK_REALTIME, &start);
    vectorSloped_C(inoutd_ref, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSloped_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_64f(inoutd_ref, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_64f(inoutd_ref, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_64f %d %lf\n", len, elapsed);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope128d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope128d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope256d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope256d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd, len);
#endif

    double GB = (double) (len * sizeof(int32_t)) / 1e9;

    clock_gettime(CLOCK_REALTIME, &start);
    memcpy(inout_iref, inout_i1, len * sizeof(int32_t));
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("memcpy %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    memcpy(inout_iref, inout_i1, len * sizeof(int32_t));
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("memcpy %d BW %lf GB/s\n", len, (GB / elapsed));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCopy_32s((const Ipp32s *) inout_i1, (Ipp32s *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("ippsCopy_32s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCopy_32s((const Ipp32s *) inout_i1, (Ipp32s *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("ippsCopy_32s %d BW %lf GB/s\n", len, (GB / elapsed));
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);


    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

#endif

#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);


    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);
#endif


    free(inout);
    free(inout2);
    free(inout3);
    free(inout4);
    free(inout_u1);
    free(inout_u2);
    free(inout_s1);
    free(inout_s2);
    free(inout_ref);
    free(inout2_ref);
    free(inoutd);
    free(inoutd_ref);

    free(inout_i1);
    free(inout_i2);
    free(inout_iref);

    return 0;
}
