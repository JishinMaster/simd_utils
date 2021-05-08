/*
 * Project : SIMD_Utils
 * Version : 0.1.11
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "simd_utils.h"

#ifdef IPP
#include <ipp.h>
#include <ippdefs.h>
#include <ipptypes.h>

#define PX_FM (ippCPUID_MMX | ippCPUID_SSE | ippCPUID_SSE2)
#define M7_FM (PX_FM | ippCPUID_SSE3)
#define U8_FM (M7_FM | ippCPUID_SSSE3)
#define N8_FM (U8_FM)  //not supported on corei7? | ippCPUID_MOVBE )
#define Y8_FM (U8_FM | ippCPUID_SSE41 | ippCPUID_SSE42 | ippCPUID_AES | ippCPUID_CLMUL | ippCPUID_SHA)
#define E9_FM (Y8_FM | ippCPUID_AVX | ippAVX_ENABLEDBYOS | ippCPUID_RDRAND | ippCPUID_F16C)
#define L9_FM (E9_FM | ippCPUID_MOVBE | ippCPUID_AVX2 | ippCPUID_ADCOX | ippCPUID_RDSEED | ippCPUID_PREFETCHW)
#define K0_FM (L9_FM | ippCPUID_AVX512F)

#define SSE4_MASK (ippCPUID_MMX | ippCPUID_SSE | ippCPUID_SSE2 | ippCPUID_SSE3 | ippCPUID_SSE41 | ippCPUID_SSE42 | ippCPUID_CLMUL | ippCPUID_AES)
#define AVX_MASK (SSE4_MASK | ippCPUID_AVX | ippAVX_ENABLEDBYOS)
#define AVX2_MASK (AVX_MASK | ippCPUID_AVX2)
#define AVX512_MASK (AVX2_MASK | ippCPUID_AVX512F)

#endif

#ifdef MKL
#include <mkl.h>
#include <mkl_vml.h>
#endif

typedef ALIGN16_BEG union {
    float f[4];
    int i[4];
    v4sf v;
} ALIGN16_END V4SF;

#ifdef AVX
typedef ALIGN32_BEG union {
    float f[8];
    int i[8];
    v8sf v;
} ALIGN32_END V8SF;

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
#ifdef IPP

    IppStatus status;
    Ipp64u mask, emask;

#if defined(AVX512)
    mask = AVX512_MASK;
#elif defined(__AVX2__)
    mask = AVX2_MASK;
#elif defined(AVX)
    mask = AVX_MASK;
#else
    mask = SSE4_MASK;
#endif

    status = ippSetCpuFeatures(mask);
    if (status != ippStsNoErr) {
        printf("error ippSetCpuFeatures() %llx %s\n", mask, ippGetStatusString(status));
        //return -1;
    }

    //ippInit();
    const IppLibraryVersion *lib;
    lib = ippGetLibVersion();
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

#endif /* IPP */

#ifdef MKL

    int mkl_mask;

#if defined(AVX512)
    mkl_mask = MKL_ENABLE_AVX512;
#elif defined(__AVX2__)
    mkl_mask = MKL_ENABLE_AVX2;
#elif defined(AVX)
    mkl_mask = MKL_ENABLE_AVX;
#else
    mkl_mask = MKL_ENABLE_SSE4_2;
#endif
    if (mkl_enable_instructions(mkl_mask) != 1)
        printf("Could not enable required MKL instructions\n");

#endif /* MKL */

    float *inout = NULL, *inout2 = NULL, *inout3 = NULL, *inout4 = NULL, *inout5 = NULL;
    float *inout6 = NULL, *inout_ref = NULL, *inout2_ref = NULL;
    double *inoutd = NULL, *inoutd2 = NULL, *inoutd3 = NULL, *inoutd_ref = NULL, *inoutd2_ref = NULL;
    uint8_t *inout_u1 = NULL, *inout_u2 = NULL;
    int16_t *inout_s1 = NULL, *inout_s2 = NULL;
    int32_t *inout_i1 = NULL, *inout_i2 = NULL, *inout_iref = NULL;
    int len = atoi(argv[1]);
    int ret;
#ifndef USE_MALLOC
    ret = posix_memalign((void **) &inout, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout == NULL) {
        printf("ret = posix_memalign inout failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout2, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout2 == NULL) {
        printf("ret = posix_memalign inout2 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout3, atoi(argv[2]), len * sizeof(float));
    if (inout3 == NULL) {
        printf("ret = posix_memalign inout3 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout4, atoi(argv[2]), len * sizeof(float));
    if (inout4 == NULL) {
        printf("ret = posix_memalign inout4 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout5, atoi(argv[2]), len * sizeof(float));
    if (inout3 == NULL) {
        printf("ret = posix_memalign inout5 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout6, atoi(argv[2]), len * sizeof(float));
    if (inout4 == NULL) {
        printf("ret = posix_memalign inout6 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout_ref, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout_ref == NULL) {
        printf("ret = posix_memalign inout_ref failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout2_ref, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout2_ref == NULL) {
        printf("ret = posix_memalign inout2_ref failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inoutd, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("ret = posix_memalign inoutd failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inoutd2, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("ret = posix_memalign inoutd2 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inoutd3, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("ret = posix_memalign inoutd3 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inoutd_ref, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("ret = posix_memalign inoutd_ref failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inoutd2_ref, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("ret = posix_memalign inoutd2_ref failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout_u1, atoi(argv[2]), len * sizeof(uint8_t));
    if (inout_u1 == NULL) {
        printf("ret = posix_memalign inout_u1 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout_u2, atoi(argv[2]), len * sizeof(uint8_t));
    if (inout_u2 == NULL) {
        printf("ret = posix_memalign inout_u2 failed\n");
        return -1;
    }

    ret = posix_memalign((void **) &inout_s1, atoi(argv[2]), len * sizeof(int16_t));
    if (inout_s1 == NULL) {
        printf("ret = posix_memalign inout_s1 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout_s2, atoi(argv[2]), len * sizeof(int16_t));
    if (inout_s2 == NULL) {
        printf("ret = posix_memalign inout_s2 failed\n");
        return -1;
    }

    ret = posix_memalign((void **) &inout_i1, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_i1 == NULL) {
        printf("ret = posix_memalign inout_i1 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout_i2, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_i2 == NULL) {
        printf("ret = posix_memalign inout_i2 failed\n");
        return -1;
    }
    ret = posix_memalign((void **) &inout_iref, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_iref == NULL) {
        printf("ret = posix_memalign inout_iref failed\n");
        return -1;
    }

#else /* USE_MALLOC */
    //TODO : add missing new arrays
    inout = (float *) malloc(2 * len * sizeof(float));
    if (inout == NULL) {
        printf("malloc inout failed\n");
        return -1;
    }
    inout2 = (float *) malloc(2 * len * sizeof(float));
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
    inout5 = (float *) malloc(len * sizeof(float));
    if (inout5 == NULL) {
        printf("malloc inout5 failed\n");
        return -1;
    }
    inout6 = (float *) malloc(len * sizeof(float));
    if (inout6 == NULL) {
        printf("malloc inout6 failed\n");
        return -1;
    }
    inout_ref = (float *) malloc(2 * len * sizeof(float));
    if (inout_ref == NULL) {
        printf("malloc inout_ref failed\n");
        return -1;
    }
    inout2_ref = (float *) malloc(2 * len * sizeof(float));
    if (inout2_ref == NULL) {
        printf("malloc inout2_ref failed\n");
        return -1;
    }
    inoutd = (double *) malloc(2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("malloc inoutd failed\n");
        return -1;
    }
    inoutd2 = (double *) malloc(2 * len * sizeof(double));
    if (inoutd2 == NULL) {
        printf("malloc inoutd2 failed\n");
        return -1;
    }
    inoutd3 = (double *) malloc(2 * len * sizeof(double));
    if (inoutd3 == NULL) {
        printf("malloc inoutd3 failed\n");
        return -1;
    }
    inoutd_ref = (double *) malloc(2 * len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("malloc inoutd_ref failed\n");
        return -1;
    }
    inoutd2_ref = (double *) malloc(2 * len * sizeof(double));
    if (inoutd2_ref == NULL) {
        printf("malloc inoutd2_ref failed\n");
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
        printf("ret = posix_memalign inout_i1 failed\n");
        return -1;
    }
    inout_i2 = (int32_t *) malloc(len * sizeof(int32_t));
    if (inout_i2 == NULL) {
        printf("ret = posix_memalign inout_i2 failed\n");
        return -1;
    }
    inout_iref = (int32_t *) malloc(len * sizeof(int32_t));
    if (inout_iref == NULL) {
        printf("ret = posix_memalign inout_iref failed\n");
        return -1;
    }

#endif /* USE_MALLOC */


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////// BEGIN //////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct timespec start, stop;
    double elapsed = 0.0;

    printf("\n");
    /////////////////////////////////////////////////////////// MEMSET //////////////////////////////////////////////////////////////////////////////
    printf("MEMSET\n");

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
    ippsSet_32f(0.001f, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSet_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsSet_32f(0.08f, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSet_32f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    zero512f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    zero512f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    set512f(inout, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    set512f(inout, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set512f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MEMCPY //////////////////////////////////////////////////////////////////////////////
    printf("MEMCPY\n");

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
    ippsCopy_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCopy_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCopy_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCopy_32f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    copy512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    copy512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy512f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

    /*for(int i = 0; i < len; i++){
		printf("%f ",inout[i]);
	}
	printf("\n");*/


    printf("\n");
    /////////////////////////////////////////////////////////// threshold_lt_f //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_LT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 20.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = 0.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_lt_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_lt_f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LT_32f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreashold_LT_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LT_32f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreashold_LT_32f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_lt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_lt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_lt_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_lt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_lt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_lt_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_lt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_lt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_lt_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif



    printf("\n");
    /////////////////////////////////////////////////////////// threshold_gt_f //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_GTABS\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (-i) / 20.0f + 30.0f - 0.05f * (float) i;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = 0.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gtabs_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gtabs_f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_GTAbs_32f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_GTAbs_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_GTAbs_32f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_GTAbs_32f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_gtabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_gtabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_gtabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_gtabs_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_gtabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_gtabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_gtabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_gtabs_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_gtabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_gtabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_gtabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_gtabs_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

    /*for(int i = 0; i < len; i++){
      printf("%f %f %f\n", inout[i],inout2_ref[i],inout2[i]);
    }*/

    printf("\n");
    /////////////////////////////////////////////////////////// THRESHOLD_LTValGTVal //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_LTValGTVal\n");


    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltval_gtval_f_C(inout, inout2_ref, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltval_gtval_f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LTValGTVal_32f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_LTValGTVal_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LTValGTVal_32f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_LTValGTVal_32f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_ltval_gtval_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_ltval_gtval_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_ltval_gtval_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_ltval_gtval_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_ltval_gtval_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_ltval_gtval_f %d %lf\n", len, elapsed);

    l2_err(inout2, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MAXEVERY //////////////////////////////////////////////////////////////////////////////
    printf("MAXEVERY\n");

    clock_gettime(CLOCK_REALTIME, &start);
    maxeveryf_c(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxeveryf_c %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMaxEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMaxEvery_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMaxEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMaxEvery_32f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    maxevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery128f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    maxevery256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery256f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    maxevery512f(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery512f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MINEVERY //////////////////////////////////////////////////////////////////////////////
    printf("MINEVERY\n");

    clock_gettime(CLOCK_REALTIME, &start);
    mineveryf_c(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mineveryf_c %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMinEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMinEvery_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMinEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMinEvery_32f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    minevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    minevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery128f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    minevery256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    minevery256f(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery256f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    minevery512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    minevery512f(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery512f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
#endif

    /*for (int i = 0; i < len; i++)
{
	printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
}*/

    printf("\n");
    /////////////////////////////////////////////////////////// MINMAX //////////////////////////////////////////////////////////////////////////////
    printf("MINMAX\n");

    float min, max, min_ref, max_ref;
    clock_gettime(CLOCK_REALTIME, &start);
    minmaxf_c(inout, len, &min_ref, &max_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmaxf_c %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMinMax_32f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMinMax_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMinMax_32f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMinMax_32f %d %lf\n", len, elapsed);

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif


#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    minmax128f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    minmax128f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax128f %d %lf\n", len, elapsed);

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    minmax256f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    minmax256f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax256f %d %lf\n", len, elapsed);

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    minmax512f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    minmax512f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax512f %d %lf\n", len, elapsed);

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// FABSF //////////////////////////////////////////////////////////////////////////////
    printf("FABSF\n");

    clock_gettime(CLOCK_REALTIME, &start);
    fabsf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabsf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAbs_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAbs_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAbs_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAbs_32f %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAbsI(len, inout, 1, inout2, 1);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAbsI %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsAbsI(len, inout, 1, inout2, 1);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAbsI %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    fabs512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    fabs512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CONVERT_64_32 //////////////////////////////////////////////////////////////////////////////
    printf("CONVERT_64_32\n");

    for (int i = 0; i < len; i++)
        inoutd[i] = (double) i;

    clock_gettime(CLOCK_REALTIME, &start);
    convert_64f32f_C(inoutd, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert_64f32f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_64f32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_64f32f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
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



    printf("\n");
    /////////////////////////////////////////////////////////// MEAN //////////////////////////////////////////////////////////////////////////////
    printf("MEAN\n");

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
    ippsMean_32f(inout_ref, len, &mean, (IppHintAlgorithm) 0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMean_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMean_32f(inout_ref, len, &mean, (IppHintAlgorithm) 0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMean_32f %d %lf\n", len, elapsed);

    printf("mean %f ref %f\n", mean, mean_ref);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    mean512f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mean512f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean512f %d %lf\n", len, elapsed);

    printf("mean %f ref %f\n", mean, mean_ref);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MAGNITUDE_SPLIT //////////////////////////////////////////////////////////////////////////////
    printf("MAGNITUDE_SPLIT\n");

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
    ippsMagnitude_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude512f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude512f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitude512f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude512f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MAGNITUDE_INTERLEAVE //////////////////////////////////////////////////////////////////////////////
    printf("MAGNITUDE_INTERLEAVE\n");

    int j = 0;
    for (int i = 0; i < 2 * len; i += 2) {
        inout[i] = (float) j / 10.5f;
        inout[i + 1] = (float) j / 35.77f;
        j += 1;
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
    ippsMagnitude_32fc((const Ipp32fc *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32fc((const Ipp32fc *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32fc %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
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

    /*#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude256f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude256f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitude256f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude256f_interleaved %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif


    int k = 0;
	for(int i = 0; i < len; i++){
		printf("Int : %f %f || %f %f\n", inout[k],inout[k+1], inout_ref[i],inout2_ref[i]);
		k+=2;
	}*/


    printf("\n");
    /////////////////////////////////////////////////////////// CPLXVECMUL //////////////////////////////////////////////////////////////////////////////
    printf("CPLXVECMUL\n");

    for (int i = 0; i < 2 * len; i++) {
        inout[i] = (float) i / 300.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32fc_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32fc_A24 %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);

    /*for(int i = 0; i < len; i+=2){
		printf("%f %f // %f %f // %f %f || %f %f\n",inout[i],inout[i+1],inout2[i],inout2[i+1],inout_ref[i],inout_ref[i+1], inout2_ref[i], inout2_ref[i+1]);
	 }*/
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CPLXVECMUL_SPLIT //////////////////////////////////////////////////////////////////////////////
    printf("CPLXVECMUL_SPLIT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 300.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
        inout3[i] = (float) i / 1024.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout4[i] = (float) i / (-11112.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C_split %d %lf\n", len, elapsed);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul256f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul256f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul512f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul512f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

    /*for(int i = 0; i < len; i+=2){
		printf("%f %f // %f %f // %f %f || %f %f\n",inout[i],inout[i+1],inout2[i],inout2[i+1],inout_ref[i],inout_ref[i+1], inout2_ref[i], inout2_ref[i+1]);
	}*/

    /*for(int i = 0; i < len/2; i+=2){
		printf("%f %f\n",inout_ref[i],inout_ref[i+1]);
	}*/

    printf("\n");

    /////////////////////////////////////////////////////////// CPLXCONJVECMUL //////////////////////////////////////////////////////////////////////////////
    printf("CPLXCONJVECMUL\n");

    for (int i = 0; i < 2 * len; i++) {
        inout[i] = (float) i / 300.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
    }


    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul_C %d %lf\n", len, elapsed);

#ifdef IPP
// TODO
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif
    printf("\n");

    /////////////////////////////////////////////////////////// CPLXCONJVECMUL_SPLIT //////////////////////////////////////////////////////////////////////////////
    printf("CPLXCONJVECMUL_SPLIT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 300.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
        inout3[i] = (float) i / 1024.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout4[i] = (float) i / (-11112.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C_split %d %lf\n", len, elapsed);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul256f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul256f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul512f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul512f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

    /*for(int i = 0; i < len; i+=2){
		printf("%f %f // %f %f // %f %f || %f %f\n",inout[i],inout[i+1],inout2[i],inout2[i+1],inout_ref[i],inout_ref[i+1], inout2_ref[i], inout2_ref[i+1]);
	}*/

    /*for(int i = 0; i < len/2; i+=2){
		printf("%f %f\n",inout_ref[i],inout_ref[i+1]);
	}*/

    printf("\n");


    /////////////////////////////////////////////////////////// CPLXCONJ //////////////////////////////////////////////////////////////////////////////
    printf("CPLXCONJ\n");

    for (int i = 0; i < 2 * len; i++) {
        inout[i] = (float) i / 300.0f;  //printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
    }


    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj_C((complex32_t *) inout, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConj_32fc((const Ipp32fc *) inout, (const Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConj_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConj_32fc((const Ipp32fc *) inout, (const Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj128f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj128f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj256f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj256f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj512f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj512f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");


    /////////////////////////////////////////////////////////// ADDCST //////////////////////////////////////////////////////////////////////////////
    printf("ADDCST\n");

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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    addc512f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    addc512f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc512f %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MULCST //////////////////////////////////////////////////////////////////////////////
    printf("MULCST\n");

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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    mulc512f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mulc512f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc512f %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MUL //////////////////////////////////////////////////////////////////////////////
    printf("MUL\n");

    clock_gettime(CLOCK_REALTIME, &start);
    mulf_C(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulcf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsMul(len, inout, inout2, inout_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsMul %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsMul(len, inout, inout2, inout_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsMul %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    mul512f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    mul512f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SIN //////////////////////////////////////////////////////////////////////////////
    printf("SIN\n");

    clock_gettime(CLOCK_REALTIME, &start);
    sinf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSin_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsSin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSin_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsSin(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsSin %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsSin(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsSin %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sin512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sin512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif


    printf("\n");
    /////////////////////////////////////////////////////////// COS //////////////////////////////////////////////////////////////////////////////
    printf("COS\n");

    clock_gettime(CLOCK_REALTIME, &start);
    cosf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCos_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCos_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCos_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCos_32f_A24 %d %lf\n", len, elapsed);

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsCos(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsCos %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsCos(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsCos %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cos128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cos128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cos256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cos256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cos512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cos512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SINCOS //////////////////////////////////////////////////////////////////////////////
    printf("SINCOS\n");

    clock_gettime(CLOCK_REALTIME, &start);
    sincosf_C(inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSinCos_32f_A24(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSinCos_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsSinCos_32f_A24(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSinCos_32f_A24 %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsSinCos(len, inout, inout2, inout3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsSinCos %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsSinCos(len, inout, inout2, inout3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsSinCos %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sincos128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sincos128f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#endif

    /*for (int i = 0; i < len; i++) {
        printf("%f %f %f %f %f\n",inout[i], inout_ref[i],inout2[i],inout2_ref[i],inout3[i]);
    }*/

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sincos256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sincos256f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sincos512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SINCOSD //////////////////////////////////////////////////////////////////////////////
    printf("SINCOSD\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = -1.0 + (double) i / 10.0;
        inout[i] = -1.0f + (float) i / 10.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    sincosd_C(inoutd, inoutd_ref, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosd_C %d %lf\n", len, elapsed);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif
#endif

// work in progress
#if 0
#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif
#endif

    /*for(int i = 0; i < len; i++){
        printf("%lf || %lf %lf || %lf %lf\n",inoutd[i], inoutd_ref[i],inoutd2[i], inoutd2_ref[i], inoutd3[i]);
    }*/

    printf("\n");
    /////////////////////////////////////////////////////////// LOGN //////////////////////////////////////////////////////////////////////////////
    printf("LOGN\n");

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
    ippsLn_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsLn_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsLn_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsLn_32f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsLn(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsLn %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsLn(len, inout, inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsLn %d %lf\n", len, elapsed);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ln_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ln_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    ln_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ln_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif


    printf("\n");
    /////////////////////////////////////////////////////////// LOGN //////////////////////////////////////////////////////////////////////////////
    printf("EXP\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 10000.0f;
        inout_ref[i] = inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    expf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("expf_C %d %lf\n", len, elapsed);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    exp_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    exp_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    exp_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    exp_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    exp_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    exp_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CONVERT_32_64 //////////////////////////////////////////////////////////////////////////////
    printf("CONVERT_32_64\n");


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

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f64f %d %lf\n", len, elapsed);
    l2_errd(inoutd, inoutd_ref, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convert512_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert512_32f64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    convert512_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert512_32f64f %d %lf\n", len, elapsed);

    l2_errd(inoutd, inoutd_ref, len);
    /*for(int i =0; i < 16; i++){
		printf("%lf %lf %f\n",inoutd_ref[i],inoutd[i],inout[i]);
	}*/
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// FLIP //////////////////////////////////////////////////////////////////////////////
    printf("FLIP\n");

    clock_gettime(CLOCK_REALTIME, &start);
    flipf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flipf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsFlip_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFlip_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsFlip_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFlip_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
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


    printf("\n");
    /////////////////////////////////////////////////////////// FLOOR //////////////////////////////////////////////////////////////////////////////
    printf("FLOOR\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 1.82f;
        inout_ref[i] = inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    floorf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floorf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsFloor_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFloor_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsFloor_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFloor_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsFloor(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsFloor %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsFloor(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsFloor %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    floor512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    floor512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CEIL //////////////////////////////////////////////////////////////////////////////
    printf("CEIL\n");

    clock_gettime(CLOCK_REALTIME, &start);
    ceilf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceilf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCeil_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCeil_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCeil_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCeil_32f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsCeil(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsCeil %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsCeil(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsCeil %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    ceil512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ceil512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ROUND //////////////////////////////////////////////////////////////////////////////
    printf("ROUND\n");

    clock_gettime(CLOCK_REALTIME, &start);
    roundf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("roundf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsRound_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRound_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsRound_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRound_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsRound(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsRound %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsRound(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsRound %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    round512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("round512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    round512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("round512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif



    printf("\n");
    /////////////////////////////////////////////////////////// TRUNC //////////////////////////////////////////////////////////////////////////////
    printf("TRUNC\n");

    clock_gettime(CLOCK_REALTIME, &start);
    truncf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("truncf_C %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsTrunc_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTrunc_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsTrunc_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTrunc_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
//TODO
#endif

#ifdef MKL
//TODO
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    trunc128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("trunc128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    trunc128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("trunc128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    trunc256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("trunc256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    trunc256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("trunc256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    trunc512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("trunc512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    trunc512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("trunc512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// TAN //////////////////////////////////////////////////////////////////////////////
    printf("TAN\n");

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
    ippsTan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTan_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsTan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTan_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsTan(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsTan %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsTan(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsTan %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    tan512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tan512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#endif


    printf("\n");
    /////////////////////////////////////////////////////////// TANH //////////////////////////////////////////////////////////////////////////////
    printf("TANH\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 2000) / 1000.0f;
        if (i % 2 == 0)
            inout[i] = -inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    tanhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tanhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanhf_C %d %lf\n", len, elapsed);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    tanh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tanh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    tanh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tanh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    tanh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tanh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SINH //////////////////////////////////////////////////////////////////////////////
    printf("SINH\n");

    clock_gettime(CLOCK_REALTIME, &start);
    sinhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sinhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinhf_C %d %lf\n", len, elapsed);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    sinh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sinh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sinh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sinh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sinh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sinh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    /*for(int i = 0; i < len; i++)
		  printf("%f %f %f \n",inout[i],inout_ref[i], inout2[i]);
		printf("\n\n");*/

    printf("\n");
    /////////////////////////////////////////////////////////// ASINF //////////////////////////////////////////////////////////////////////////////
    printf("ASINF\n");

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
    ippsAsin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAsin(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAsin %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsAsin(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAsin %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    asin512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif


    printf("\n");
    /////////////////////////////////////////////////////////// ASIN //////////////////////////////////////////////////////////////////////////////
    printf("ASIN\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = (float) (1.0 * i + 1.0) / 1.82 / (float) (5 * len);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    asin_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsin_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsin_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

    /*
for (int i = 0; i < len; i++){
    printf("%lf , %lf, %lf %lf\n",inoutd[i],inoutd_ref[i],inoutd2[i],asin(inoutd[i]));
}*/

#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    asin128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);


    clock_gettime(CLOCK_REALTIME, &start);
    asin128d_(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d_ %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin128d_(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d_ %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

    /*
for (int i = 0; i < len; i++){
    printf("%lf , %lf, %lf %lf\n",inoutd[i],inoutd_ref[i],inoutd2[i],asin(inoutd[i]));
}*/

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    asin256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    asin512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asin512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ATANF //////////////////////////////////////////////////////////////////////////////
    printf("ATANF\n");

    clock_gettime(CLOCK_REALTIME, &start);
    atanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAtan(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAtan %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsAtan(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAtan %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

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
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    atan512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ATANF2 //////////////////////////////////////////////////////////////////////////////
    printf("ATANF2\n");

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
    ippsAtan2_32f_A24(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan2_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan2_32f_A24(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan2_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAtan2(len, inout, inout2, inout_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAtan2 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vsAtan2(len, inout, inout2, inout_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAtan2 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan2128f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan2128f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan2256f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan2256f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    atan2512f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan2512f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif


    printf("\n");
    /////////////////////////////////////////////////////////// ATAN //////////////////////////////////////////////////////////////////////////////
    printf("ATAN\n");

    for (int i = 0; i < len / 2; i++) {
        inoutd[i] = (float) (-1.0 * i + 1.0) / 1.82 / (float) (5 * len);
    }
    for (int i = len / 2; i < len; i++) {
        inoutd[i] = (float) (1.0 * i + 1.0) / 1.82 / (float) (5 * len);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    atan_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan_C %d %lf\n", len, elapsed);


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);


    /*for (int i = 0; i < len; i++){
        printf("%lf , %lf, %lf %lf\n",inoutd[i],inoutd_ref[i],inoutd2[i],atan(inoutd[i]));
    }*/

#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    atan128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    atan256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    atan512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CPLX2REAL //////////////////////////////////////////////////////////////////////////////
    printf("CPLX2REAL\n");

    for (int i = 0; i < len; i++) {
        inout_ref[i] = 0.0f;
        inout2_ref[i] = 0.0f;
        inout3[i] = 0.0f;
        inout4[i] = 0.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxtorealf_C(inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtorealf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCplxToReal_32fc((const Ipp32fc *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCplxToReal_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsCplxToReal_32fc((const Ipp32fc *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCplxToReal_32fc %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal128f(inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal128f(inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal128f %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal256f(inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal256f(inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal256f %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

    /*for(int i = 0; i < len; i++)
		  printf("%f %f %f %f \n",inout3[i],inout_ref[i], inout4[i], inout2_ref[i]);
		printf("\n\n");*/
#endif


    printf("\n");
    /////////////////////////////////////////////////////////// REAL2CPLX //////////////////////////////////////////////////////////////////////////////
    printf("REAL2CPLX\n");

    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx_C(inout3, inout4, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsRealToCplx_32f(inout3, inout4, (Ipp32fc *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRealToCplx_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsRealToCplx_32f(inout3, inout4, (Ipp32fc *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRealToCplx_32f %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, 2 * len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx128f(inout3, inout4, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx128f(inout3, inout4, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx128f %d %lf\n", len, elapsed);
    l2_err(inout, inout_ref, 2 * len);

    /*for(int i = 0; i < 2*len; i++)
		  printf("%f %f\n",inout[i],inout_ref[i]);
		printf("\n\n");*/
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx256f(inout3, inout4, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx256f(inout3, inout4, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx256f %d %lf\n", len, elapsed);
    l2_err(inout, inout_ref, 2 * len);

    /*for(int i = 0; i < 2*len; i++)
		  printf("%f %f\n",inout[i],inout_ref[i]);*/
#endif

    /*for(int i = 0; i < len; i++)
		printf("%f %f || %f %f \n",inout3[i],inout_ref[i], inout4[i],inout2_ref[i]);*/


    printf("\n");
    /////////////////////////////////////////////////////////// CONVERTFLOAT32_U8 //////////////////////////////////////////////////////////////////////////////
    printf("CONVERTFLOAT32_U8\n");

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


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f8u_Sfs(inout, inout_u1, len, ippRndZero, 4);  //ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f8u_Sfs(inout, inout_u1, len, ippRndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);
    l2_err_u8(inout_u1, inout_u2, len);

    /*for(int i = 0; i < len; i++)
		  printf("%x %x\n" ,inout_u1[i],inout_u2[i]);*/


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

    /*for(int i = 0; i < len; i++)
		printf("%x %x\n" ,inout_u1[i],inout_u2[i]);
	*/

    for (int i = 0; i < len; i++)
        inout_s1[i] = -len + i;

    printf("\n");
    /////////////////////////////////////////////////////////// CONVERTINT16_FLOAT32 //////////////////////////////////////////////////////////////////////////////
    printf("CONVERTINT16_FLOAT32\n");

    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_C(inout_s1, inout2_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_16s32f_Sfs(inout_s1, inout_ref, len, 4);  //ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_16s32f_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_16s32f_Sfs(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_16s32f_Sfs %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
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


    printf("\n");
    /////////////////////////////////////////////////////////// VECTOR_SLOPE //////////////////////////////////////////////////////////////////////////////
    printf("VECTOR_SLOPE\n");

    //for(int i = 0; i < len; i++)
    //	printf("%d %f %f %f\n" ,inout_s1[i], (float)inout_s1[i]/(1 << 4),inout_ref[i],inout2_ref[i]);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlopef_C(inout2_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_32f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_32f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope512f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope512f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// VECTOR_SLOPE_D //////////////////////////////////////////////////////////////////////////////
    printf("VECTOR_SLOPE_D\n");

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSloped_C(inoutd_ref, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSloped_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_64f(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_64f(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_64f %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd, len);
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


    printf("\n");
    /////////////////////////////////////////////////////////// EXPERIMENTAL //////////////////////////////////////////////////////////////////////////////
    printf("EXPERIMENTAL\n");

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

#ifdef AVX
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
#endif

    free(inout);
    free(inout2);
    free(inout3);
    free(inout4);
    free(inout5);
    free(inout6);
    free(inout_u1);
    free(inout_u2);
    free(inout_s1);
    free(inout_s2);
    free(inout_ref);
    free(inout2_ref);
    free(inoutd);
    free(inoutd2);
    free(inoutd3);
    free(inoutd_ref);
    free(inoutd2_ref);

    free(inout_i1);
    free(inout_i2);
    free(inout_iref);

    return 0;
}
