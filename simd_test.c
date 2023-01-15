/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "simd_utils.h"

#ifdef __MACH__
#include "macosx_wrapper.h"
#endif

#ifdef VDSP
#include "vDSP.h"
#include "vForce.h"
#endif

#ifdef IPP
#include <ipp.h>
#include <ippdefs.h>
#include <ipptypes.h>

#define PX_FM (ippCPUID_MMX | ippCPUID_SSE | ippCPUID_SSE2)
#define M7_FM (PX_FM | ippCPUID_SSE3)
#define U8_FM (M7_FM | ippCPUID_SSSE3)
#define N8_FM (U8_FM)  // not supported on corei7? | ippCPUID_MOVBE )
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

// Does not take care of NAN and INF
int32_t ulpsDistance32(const float a, const float b)
{
    if (a == b)
        return 0;
    int32_t ia, ib;
    memcpy(&ia, &a, sizeof(float));
    memcpy(&ib, &b, sizeof(float));

    if ((ia < 0) != (ib < 0))
        return -1;

    int32_t dist = abs(ia - ib);
    return dist;
}

// Does not take care of NAN and INF
int64_t ulpsDistance64(const double a, const double b)
{
    if (a == b)
        return 0;
    int64_t ia, ib;
    memcpy(&ia, &a, sizeof(double));
    memcpy(&ib, &b, sizeof(double));

    if ((ia < 0) != (ib < 0))
        return -1;

    int64_t dist = abs(ia - ib);
    return dist;
}


float l2_err(float *test, float *ref, int len)
{
    float l2_err = 0.0;
    int sup3ulps = 0;
    float sup3ulps_percent = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (ref[i] - test[i]) * (ref[i] - test[i]);
        int32_t dist = ulpsDistance32(ref[i], test[i]);
        if (dist > 3)
            sup3ulps++;
    }

    sup3ulps_percent = (float) sup3ulps / (float) len * 100.0f;
    printf("L2 ERR %0.9g SUP_3ULPS %2.4g %% \n", l2_err, sup3ulps_percent);
    return l2_err;
}

double l2_errd(double *test, double *ref, int len)
{
    double l2_err = 0.0;
    int sup3ulps = 0;
    float sup3ulps_percent = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (ref[i] - test[i]) * (ref[i] - test[i]);
        int64_t dist = ulpsDistance64(ref[i], test[i]);
        if (dist > 3)
            sup3ulps++;
    }

    sup3ulps_percent = (float) sup3ulps / (float) len * 100.0f;
    printf("L2 ERR %0.13g SUP_3ULPS %2.4g %% \n", l2_err, sup3ulps_percent);
    return l2_err;
}

float l2_err_u8(uint8_t *test, uint8_t *ref, int len)
{
    float l2_err = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
    }

    printf("L2 ERR %0.9g\n", l2_err);
    return l2_err;
}

float l2_err_i16(int16_t *test, int16_t *ref, int len)
{
    float l2_err = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
    }

    printf("L2 ERR %0.9g\n", l2_err);
    return l2_err;
}

float l2_err_i32(int32_t *test, int32_t *ref, int len)
{
    float l2_err = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
    }

    printf("L2 ERR %0.9g\n", l2_err);
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
        // return -1;
    }

    // ippInit();
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
    int16_t *inout_s1 = NULL, *inout_s2 = NULL, *inout_s3 = NULL, *inout_sref = NULL;
    int32_t *inout_i1 = NULL, *inout_i2 = NULL, *inout_i3 = NULL, *inout_iref = NULL;
    int len = atoi(argv[1]);

#ifndef USE_MALLOC
    int ret = 0;
    ret |= posix_memalign((void **) &inout, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout == NULL) {
        printf("posix_memalign inout failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout2, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout2 == NULL) {
        printf("posix_memalign inout2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout3, atoi(argv[2]), len * sizeof(float));
    if (inout3 == NULL) {
        printf("posix_memalign inout3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout4, atoi(argv[2]), len * sizeof(float));
    if (inout4 == NULL) {
        printf("posix_memalign inout4 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout5, atoi(argv[2]), len * sizeof(float));
    if (inout3 == NULL) {
        printf("posix_memalign inout5 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout6, atoi(argv[2]), len * sizeof(float));
    if (inout4 == NULL) {
        printf("posix_memalign inout6 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_ref, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout_ref == NULL) {
        printf("posix_memalign inout_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout2_ref, atoi(argv[2]), 2 * len * sizeof(float));
    if (inout2_ref == NULL) {
        printf("posix_memalign inout2_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("posix_memalign inoutd failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd2, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("posix_memalign inoutd2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd3, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("posix_memalign inoutd3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd_ref, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("posix_memalign inoutd_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd2_ref, atoi(argv[2]), 2 * len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("posix_memalign inoutd2_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_u1, atoi(argv[2]), len * sizeof(uint8_t));
    if (inout_u1 == NULL) {
        printf("posix_memalign inout_u1 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_u2, atoi(argv[2]), len * sizeof(uint8_t));
    if (inout_u2 == NULL) {
        printf("posix_memalign inout_u2 failed\n");
        return -1;
    }

    ret |= posix_memalign((void **) &inout_s1, atoi(argv[2]), 2 * len * sizeof(int16_t));
    if (inout_s1 == NULL) {
        printf("posix_memalign inout_s1 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_s2, atoi(argv[2]), 2 * len * sizeof(int16_t));
    if (inout_s2 == NULL) {
        printf("posix_memalign inout_s2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_s3, atoi(argv[2]), 2 * len * sizeof(int16_t));
    if (inout_s3 == NULL) {
        printf("posix_memalign inout_s3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_sref, atoi(argv[2]), 2 * len * sizeof(int16_t));
    if (inout_sref == NULL) {
        printf("posix_memalign inout_sref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_i1, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_i1 == NULL) {
        printf("posix_memalign inout_i1 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_i2, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_i2 == NULL) {
        printf("posix_memalign inout_i2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_i3, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_i3 == NULL) {
        printf("posix_memalign inout_i3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_iref, atoi(argv[2]), len * sizeof(int32_t));
    if (inout_iref == NULL) {
        printf("posix_memalign inout_iref failed\n");
        return -1;
    }

    if (ret) {
        printf("Error in posix_memalign calls");
        return -1;
    }
#else /* USE_MALLOC */
    // TODO : add missing new arrays
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

    inout_s1 = (int16_t *) malloc(2 * len * sizeof(int16_t));
    if (inout_s1 == NULL) {
        printf("malloc inout_s1 failed\n");
        return -1;
    }
    inout_s2 = (int16_t *) malloc(2 * len * sizeof(int16_t));
    if (inout_s2 == NULL) {
        printf("malloc inout_s2 failed\n");
        return -1;
    }
    inout_s3 = (int16_t *) malloc(2 * len * sizeof(int16_t));
    if (inout_s3 == NULL) {
        printf("malloc inout_s3 failed\n");
        return -1;
    }
    inout_sref = (int16_t *) malloc(2 * len * sizeof(int16_t));
    if (inout_sref == NULL) {
        printf("malloc inout_sref failed\n");
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
    inout_i3 = (int32_t *) malloc(len * sizeof(int32_t));
    if (inout_i3 == NULL) {
        printf("posix_memalign inout_i3 failed\n");
        return -1;
    }
    inout_iref = (int32_t *) malloc(len * sizeof(int32_t));
    if (inout_iref == NULL) {
        printf("posix_memalign inout_iref failed\n");
        return -1;
    }

#endif /* USE_MALLOC */


    simd_utils_get_version();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////// BEGIN //////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct timespec start, stop;
    double elapsed = 0.0;
    volatile int loop = 10;
    volatile int l = 0;
    double flops;

    printf("\n");
    /////////////////////////////////////////////////////////// MEMSET //////////////////////////////////////////////////////////////////////////////
    printf("MEMSET\n");

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    memset(inout_ref, 0.0f, len * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("memset %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        memset(inout_ref, 0.0f, len * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("memset %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    setf_C(inout_ref, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("setf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        setf_C(inout_ref, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("setf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsZero_32f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsZero_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsZero_32f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsZero_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


    clock_gettime(CLOCK_REALTIME, &start);
    ippsSet_32f(0.001f, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSet_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSet_32f(0.08f, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsSet_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout, inout_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    zero128f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        zero128f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("zero128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


    clock_gettime(CLOCK_REALTIME, &start);
    set128f(inout, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        set128f(inout, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("set128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    zero256f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        zero256f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("zero256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    set256f(inout, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        set256f(inout, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("set256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    zero512f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zero512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        zero512f(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("zero512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    set512f(inout, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("set512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        set512f(inout, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("set512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout, inout_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    zerof_vec(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("zerof_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        zerof_vec(inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("zerof_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    setf_vec(inout, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("setf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        setf_vec(inout, 0.08f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("setf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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
    for (l = 0; l < loop; l++)
        memcpy(inout2_ref, inout, len * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("memcpy %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    copyf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copyf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copyf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("copyf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCopy_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCopy_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCopy_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsCopy_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    copy128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copy128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("copy128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    copy256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copy256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("copy256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    copy512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copy512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copy512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("copy512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    copyf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("copyf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copyf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("copyf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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
        inout[i] = (float) i / 20.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = 0.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_lt_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_lt_f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_lt_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_lt_f_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LT_32f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreashold_LT_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_LT_32f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreashold_LT_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_lt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_lt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_lt_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_lt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_lt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_lt_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_lt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_lt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_lt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_lt_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_lt_f_vec(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_lt_f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_lt_f_vec(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_lt_f_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif


    printf("\n");
    /////////////////////////////////////////////////////////// threshold_lt_f //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_GT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 20.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = 0.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gt_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gt_f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gt_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gt_f_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_GT_32f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_GT_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_GT_32f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_GT_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_gt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_gt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_gt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_gt_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_gt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_gt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_gt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_gt_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_gt_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_gt_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_gt_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_gt_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gt_f_vec(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gt_f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gt_f_vec(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gt_f_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// threshold_gt_f //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_GTABS\n");

    flops = 2 * len;

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (-i) / 20.0f + 30.0f - 0.05f * (float) i;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = 0.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gtabs_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gtabs_f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gtabs_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gtabs_f_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_GTAbs_32f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_GTAbs_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_GTAbs_32f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_GTAbs_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_gtabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_gtabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_gtabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_gtabs_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_gtabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_gtabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_gtabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_gtabs_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_gtabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_gtabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_gtabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_gtabs_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gtabs_f_vec(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gtabs_f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gtabs_f_vec(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gtabs_f_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

    /*for(int i = 0; i < len; i++){
      printf("%f %f %f\n", inout[i],inout2_ref[i],inout2[i]);
    }*/

    printf("\n");
    /////////////////////////////////////////////////////////// threshold_gt_f //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_LTABS\n");

    flops = 2 * len;

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (-i) / 20.0f + 30.0f - 0.05f * (float) i;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = 0.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltabs_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltabs_f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltabs_f_C(inout, inout2_ref, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltabs_f_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LTAbs_32f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_LTAbs_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_LTAbs_32f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_LTAbs_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_ltabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_ltabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_ltabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_ltabs_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_ltabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_ltabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_ltabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_ltabs_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_ltabs_f(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_ltabs_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_ltabs_f(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_ltabs_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltabs_f_vec(inout, inout2, len, 0.07f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltabs_f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltabs_f_vec(inout, inout2, len, 0.02f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltabs_f_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// THRESHOLD_LTValGTVal //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_LTValGTVal\n");

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltval_gtval_f_C(inout, inout2_ref, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltval_gtval_f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltval_gtval_f_C(inout, inout2_ref, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltval_gtval_f_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LTValGTVal_32f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_LTValGTVal_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_LTValGTVal_32f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_LTValGTVal_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_ltval_gtval_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_ltval_gtval_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_ltval_gtval_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_ltval_gtval_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_ltval_gtval_f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_ltval_gtval_f(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_ltval_gtval_f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltval_gtval_f_vec(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltval_gtval_f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltval_gtval_f_vec(inout, inout2, len, 0.5f, 0.35f, 0.7f, 0.77f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltval_gtval_f_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout2_ref, len);
#endif

    /*for(int i = 0; i < len; i++){
      printf("%f %f %f\n", inout[i],inout2_ref[i],inout2[i]);
    }*/

    printf("\n");
    /////////////////////////////////////////////////////////// THRESHOLD_LTValGTVals //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_LTValGTVals\n");

    for (int i = 0; i < len; i++) {
        inout_i1[i] = (rand() % 1234567);
        if (i % 4 == 0)
            inout_i1[i] = -inout_i1[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltval_gtval_s_C(inout_i1, inout_iref, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltval_gtval_s_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltval_gtval_s_C(inout_i1, inout_iref, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltval_gtval_s_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LTValGTVal_32s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_LTValGTVal_32s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_LTValGTVal_32s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_LTValGTVal_32s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_ltval_gtval_s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_ltval_gtval_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_ltval_gtval_s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_ltval_gtval_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_ltval_gtval_s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_ltval_gtval_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_ltval_gtval_s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_ltval_gtval_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_ltval_gtval_s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_ltval_gtval_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_ltval_gtval_s(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_ltval_gtval_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltval_gtval_s_vec(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltval_gtval_s_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltval_gtval_s_vec(inout_i1, inout_i2, len, 80, 10, 100, 90);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltval_gtval_s_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MAXEVERY //////////////////////////////////////////////////////////////////////////////
    printf("MAXEVERY\n");

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    maxeveryf_c(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxeveryf_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxeveryf_c(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxeveryf_c %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMaxEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMaxEvery_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMaxEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMaxEvery_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxevery128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxevery256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxevery256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxevery512f(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxevery512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    maxeveryf_vec(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxeveryf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxeveryf_vec(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxeveryf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mineveryf_c(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mineveryf_c %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMinEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMinEvery_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMinEvery_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMinEvery_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    minevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minevery128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minevery128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    minevery256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minevery256f(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minevery256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    minevery512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minevery512f(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minevery512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    mineveryf_vec(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mineveryf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mineveryf_vec(inout, inout2, inout3, len);

    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mineveryf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
#endif

    /*for (int i = 0; i < len; i++)
{
    printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
}*/

    printf("\n");
    /////////////////////////////////////////////////////////// MINMAX //////////////////////////////////////////////////////////////////////////////
    printf("MINMAX\n");

    flops = 2 * len;
    float min, max, min_ref, max_ref;

    clock_gettime(CLOCK_REALTIME, &start);
    minmaxf_c(inout, len, &min_ref, &max_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmaxf_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmaxf_c(inout, len, &min_ref, &max_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmaxf_c %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMinMax_32f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMinMax_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMinMax_32f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMinMax_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif


#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    minmax128f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmax128f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmax128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    minmax256f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmax256f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmax256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    minmax512f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmax512f(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmax512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    minmaxf_vec(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmaxf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmaxf_vec(inout, len, &min, &max);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmaxf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f || %f %f\n", min_ref, min, max_ref, max);
#endif
    printf("\n");
    /////////////////////////////////////////////////////////// FABSF //////////////////////////////////////////////////////////////////////////////
    printf("FABSF\n");

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    fabsf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabsf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fabsf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fabsf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAbs_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAbs_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAbs_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAbs_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAbsI(len, inout, 1, inout2, 1);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAbsI %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsAbsI(len, inout, 1, inout2, 1);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsAbsI %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    fabs128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fabs128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fabs128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);

    /*
    #ifdef VDSP // OS X 10.5
        clock_gettime(CLOCK_REALTIME, &start);
        vvfabf(inout2, inout, &len);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("vvfabf %d %lf\n", len, elapsed);

        clock_gettime(CLOCK_REALTIME, &start);
        for (l = 0; l < loop; l++)
          vvfabf(inout2, inout, &len);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
        printf("vvfabf %d %lf\n", len, elapsed);
        l2_err(inout2_ref, inout2, len);
    #endif
    */

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    fabs256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fabs256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fabs256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    fabs512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabs512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fabs512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fabs512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    fabsf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("fabsf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fabsf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fabsf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convert_64f32f_C(inoutd, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convert_64f32f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_64f32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        convert128_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        convert256_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convert256_64f32f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convert512_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert512_64f32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convert512_64f32f(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convert512_64f32f %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif

#if defined(RISCV) && (ELEN >= 64)
    clock_gettime(CLOCK_REALTIME, &start);
    convert_64f32f_vec(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert_64f32f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convert_64f32f_vec(inoutd, inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convert_64f32f_vec %d %lf\n", len, elapsed);

    l2_err(inout, inout_ref, len);
#endif

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        meanf_C(inout_ref, &mean_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("meanf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    meanf_C_precise(inout_ref, &mean_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("meanf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    mean = 0.0f;
    ippsMean_32f(inout_ref, len, &mean, (IppHintAlgorithm) 0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMean_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMean_32f(inout_ref, len, &mean, (IppHintAlgorithm) 0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMean_32f %d %lf\n", len, elapsed);

    printf("mean %f ref %f ULPS %d\n", mean, mean_ref, ulpsDistance32(mean, mean_ref));
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    mean128f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mean128f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mean128f %d %lf\n", len, elapsed);

    printf("mean %f ref %f ULPS %d\n", mean, mean_ref, ulpsDistance32(mean, mean_ref));

#ifndef ALTIVEC
    clock_gettime(CLOCK_REALTIME, &start);
    meankahan128f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("meankahan128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        meankahan128f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("meankahan128f %d %lf\n", len, elapsed);

    printf("mean %f ref %f ULPS %d\n", mean, mean_ref, ulpsDistance32(mean, mean_ref));
#endif
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    mean256f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mean256f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mean256f %d %lf\n", len, elapsed);

    printf("mean %f ref %f ULPS %d\n", mean, mean_ref, ulpsDistance32(mean, mean_ref));
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    mean512f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mean512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mean512f(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mean512f %d %lf\n", len, elapsed);

    printf("mean %f ref %f ULPS %d\n", mean, mean_ref, ulpsDistance32(mean, mean_ref));
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    meanf_vec(inout, &mean, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("meanf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++) {
        meanf_vec(inout, &mean, len);
    }
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("meanf_vec %d %lf\n", len, elapsed);

    printf("mean %f ref %f ULPS %d\n", mean, mean_ref, ulpsDistance32(mean, mean_ref));
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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitudef_C_split(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitudef_C_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitudef_C_split_precise(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitudef_C_split_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitudef_C_split_precise(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitudef_C_split_precise %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMagnitude_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMagnitude_32f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude128f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitude128f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitude128f_split %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);

    /* for(int i = 0; i < len; i++){
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
    for (l = 0; l < loop; l++)
        magnitude256f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        magnitude512f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitude512f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    magnitudef_split_vec(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitudef_split_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitudef_split_vec(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitudef_split_vec %d %lf\n", len, elapsed);

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitudef_C_interleaved((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitudef_C_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitudef_C_interleaved_precise((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitudef_C_interleaved_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitudef_C_interleaved_precise((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitudef_C_interleaved_precise %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMagnitude_32fc((const Ipp32fc *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMagnitude_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMagnitude_32fc((const Ipp32fc *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMagnitude_32fc %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vcAbs(len, (MKL_Complex8 *) inout, inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvcAbs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vcAbs(len, (MKL_Complex8 *) inout, inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvcAbs %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif


#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude128f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitude128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitude128f_interleaved %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX
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

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    magnitude512f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude512f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    magnitude512f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitude512f_interleaved %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    magnitudef_interleaved_vec((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("magnitudef_interleaved_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        magnitudef_interleaved_vec((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("magnitudef_interleaved_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif
    /*int k = 0;
    for (int i = 0; i < len; i++) {
        printf("Int : %f %f || %f %f\n", inout[k], inout[k + 1], inout_ref[i], inout2_ref[i]);
        k += 2;
    }*/

    printf("\n");
    /////////////////////////////////////////////////////////// POWERSPECT_SPLIT //////////////////////////////////////////////////////////////////////////////
    printf("POWERSPECT_SPLIT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 10.5f;
        inout2[i] = (float) i / 35.77f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    powerspectf_C_split(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspectf_C_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspectf_C_split(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspectf_C_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    powerspectf_C_split_precise(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspectf_C_split_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspectf_C_split_precise(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspectf_C_split_precise %d %lf\n", len, elapsed);


#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsPowerSpectr_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsPowerSpectr_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsPowerSpectr_32f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsPowerSpectr_32f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);

    /*for(int i = 0; i < len; i++){
    if(fabsf(inout_ref[i]-inout2_ref[i]) > 0.000001f)
        printf("%d %.12f %.12f %.12f %.12f\n",i, inout[i],inout2[i], inout_ref[i],inout2_ref[i]);
   } */
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect128f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect128f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect128f_split %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);

    /* for(int i = 0; i < len; i++){
        printf("%f %f %f %f\n",inout[i],inout2[i],inout2_ref[i],inout_ref[i]);
    }*/
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect256f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect256f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect256f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect256f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect512f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect512f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect512f_split(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect512f_split %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    powerspectf_split_vec(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspectf_split_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspectf_split_vec(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspectf_split_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// POWERSPECT_INTERLEAVE //////////////////////////////////////////////////////////////////////////////
    printf("POWERSPECT_INTERLEAVE\n");

    j = 0;
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
    powerspectf_C_interleaved((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspectf_C_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspectf_C_interleaved((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspectf_C_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    powerspectf_C_interleaved_precise((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspectf_C_interleaved_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspectf_C_interleaved_precise((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspectf_C_interleaved_precise %d %lf\n", len, elapsed);

#ifdef IPP

    clock_gettime(CLOCK_REALTIME, &start);
    ippsPowerSpectr_32fc((const Ipp32fc *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsPowerSpectr_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsPowerSpectr_32fc((const Ipp32fc *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsPowerSpectr_32fc %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect128f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect128f_interleaved %d %lf\n", len, elapsed);

    /*for(int i = 0; i < 2*len; i++){
      printf("%f %f\n",inout_ref[i], inout2_ref[i]);
    }*/

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect256f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect256f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect256f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect256f_interleaved %d %lf\n", len, elapsed);

    /*for(int i = 0; i < 2*len; i++){
      printf("%f %f\n",inout_ref[i], inout2_ref[i]);
    }*/

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect512f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect512f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect512f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect512f_interleaved %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    powerspectf_interleaved_vec((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspectf_interleaved_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspectf_interleaved_vec((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspectf_interleaved_vec %d %lf\n", len, elapsed);

    /*for(int i = 0; i < 2*len; i++){
      printf("%f %f\n",inout_ref[i], inout2_ref[i]);
    }*/

    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CPLXVECDIV //////////////////////////////////////////////////////////////////////////////
    printf("CPLXVECDIVF\n");

    flops = 6 * len;
    for (int i = 0; i < 2 * len; i++) {
        inout[i] = (float) i / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv_C_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv_C_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv_C_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv_C_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsDiv_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsDiv_32fc_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsDiv_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsDiv_32fc_A24 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#if defined(MKL)
    clock_gettime(CLOCK_REALTIME, &start);
    vcDiv(len, (const MKL_Complex8 *) inout, (const MKL_Complex8 *) inout2, (MKL_Complex8 *) inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvcDiv %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vcDiv(len, (const MKL_Complex8 *) inout, (const MKL_Complex8 *) inout2, (MKL_Complex8 *) inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvcDiv %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);

#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdivf_vec((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdivf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdivf_vec((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdivf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif
    printf("\n");

    printf("\n");
    /////////////////////////////////////////////////////////// CPLXVECDIV_SPLIT //////////////////////////////////////////////////////////////////////////////
    printf("CPLXVECDIV_SPLIT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
        inout3[i] = (float) (i + 1) / 1024.0f + 1.575494 * 1e-22f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout4[i] = (float) (i + 1) / (-11112.577f) + 1.575494 * 1e-22f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv_C_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv_C_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv_C_split_precise(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv_C_split_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv_C_split_precise(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv_C_split_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv128f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);

    /*for(int i = 0; i < len; i++){
      printf("%f %f %f %f || %f %f %f %f\n",inout[i],inout2[i], inout3[i], inout4[i], inout_ref[i], inout2_ref[i], inout5[i], inout6[i]);
    }*/
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv256f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv256f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdiv512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdiv512f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdiv512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdiv512f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecdivf_vec_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecdivf_vec_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecdivf_vec_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecdivf_vec_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif
    /*for(int i = 0; i < len; i++){
        printf("%f %f // %f %f // %f %f || %f %f\n",inout[i],inout2[i],inout3[i],inout4[i],inout5[i],inout_ref[i], inout6[i], inout2_ref[i]);
     }*/

    printf("\n");

    /////////////////////////////////////////////////////////// CPLXVECMUL //////////////////////////////////////////////////////////////////////////////
    printf("CPLXVECMUL\n");

    flops = 6 * len;
    for (int i = 0; i < 2 * len; i++) {
        inout[i] = (float) i / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul_C_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul_C_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul_C_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32fc_A11((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32fc_A11 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMul_32fc_A11((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMul_32fc_A11 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32fc_A21((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32fc_A21 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMul_32fc_A21((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMul_32fc_A21 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32fc_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMul_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMul_32fc_A24 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);

    /*for(int i = 0; i < len; i+=2){
        printf("%f %f // %f %f // %f %f || %f %f\n",inout[i],inout[i+1],inout2[i],inout2[i+1],inout_ref[i],inout_ref[i+1], inout2_ref[i], inout2_ref[i+1]);
     }*/
#endif

#if defined(MKL)
    clock_gettime(CLOCK_REALTIME, &start);
    vcMul(len, (const MKL_Complex8 *) inout, (const MKL_Complex8 *) inout2, (MKL_Complex8 *) inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvcMul %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vcMul(len, (const MKL_Complex8 *) inout, (const MKL_Complex8 *) inout2, (MKL_Complex8 *) inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvcMul %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);

    /*for(int i = 0; i < 2*len; i+=2){
      printf("%f %f %f %f ||| %f %f || %f %f\n", inout[i], inout[i+1],inout2[i], inout2[i+1],\
              inout_ref[i], inout2_ref[i], inout_ref[i+1], inout2_ref[i+1]);
    }*/
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmulf_vec((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmulf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmulf_vec((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmulf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
    
    /*for(int i = 0; i < len; i+=2){
        printf("%0.6g %0.6g  // %0.6g  %0.6g  // %0.6g  %0.6g  || %0.6g  %0.6g \n",inout[i],inout[i+1],\
                inout2[i],inout2[i+1],inout_ref[i],inout_ref[i+1], inout2_ref[i], inout2_ref[i+1]);
     }*/
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CPLXVECMUL_SPLIT //////////////////////////////////////////////////////////////////////////////
    printf("CPLXVECMUL_SPLIT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
        inout3[i] = (float) i / 1024.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout4[i] = (float) i / (-11112.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul_C_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul_C_split_precise(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C_split_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul_C_split_precise(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul_C_split_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul128f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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
    for (l = 0; l < loop; l++)
        cplxvecmul256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul256f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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
    for (l = 0; l < loop; l++)
        cplxvecmul512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul512f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxvecmulf_vec_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmulf_vec_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxvecmulf_vec_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmulf_vec_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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

    for (int i = 0; i < 2 * len; i += 2) {
        inout[i] = (float) i / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
        inout[i + 1] = (float) i / -500.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i + 1] = (float) i / (199.577f);
    }

    flops = 7 * len;

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul_C((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul_C_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul_C_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul_C_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul_C_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMulByConj_32fc_A11((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulByConj_32fc_A11 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMulByConj_32fc_A11((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMulByConj_32fc_A11 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMulByConj_32fc_A21((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulByConj_32fc_A21 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMulByConj_32fc_A21((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMulByConj_32fc_A21 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);

    clock_gettime(CLOCK_REALTIME, &start);
    ippsMulByConj_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulByConj_32fc_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMulByConj_32fc_A24((const Ipp32fc *) inout, (const Ipp32fc *) inout2, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMulByConj_32fc_A24 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul128f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);


    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f_kahan((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f_kahan %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul128f_kahan((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul128f_kahan %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul128f_precise((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul128f_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul256f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul512f((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmulf_vec((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmulf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmulf_vec((complex32_t *) inout, (complex32_t *) inout2, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmulf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, 2 * len);
#endif

    printf("\n");

    /////////////////////////////////////////////////////////// CPLXCONJVECMUL_SPLIT //////////////////////////////////////////////////////////////////////////////
    printf("CPLXCONJVECMUL_SPLIT\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
        inout3[i] = (float) i / 1024.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout4[i] = (float) i / (-11112.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxvecmul_C_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul_C_split(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxvecmul_C_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul_C_split_precise(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul_C_split_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul_C_split_precise(inout, inout2, inout3, inout4, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul_C_split_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul128f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul128f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f_split_kahan(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f_split_kahan %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul128f_split_kahan(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul128f_split_kahan %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmul128f_split_precise(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmul128f_split_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmul128f_split_precise(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul128f_split_precise %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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
    for (l = 0; l < loop; l++)
        cplxconjvecmul256f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul256f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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
    for (l = 0; l < loop; l++)
        cplxconjvecmul512f_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmul512f_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout5, len);
    l2_err(inout2_ref, inout6, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjvecmulf_vec_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjvecmulf_vec_split %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjvecmulf_vec_split(inout, inout2, inout3, inout4, inout5, inout6, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjvecmulf_vec_split %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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
        inout[i] = (float) i / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-127.577f);
    }

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj_C((complex32_t *) inout, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconj_C((complex32_t *) inout, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconj_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConj_32fc((const Ipp32fc *) inout, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConj_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConj_32fc((const Ipp32fc *) inout, (Ipp32fc *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConj_32fc %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconj128f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconj128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconj128f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        cplxconj256f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        cplxconj512f((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconj512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxconjf_vec((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxconjf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxconjf_vec((complex32_t *) inout, (complex32_t *) inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxconjf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// DOTF //////////////////////////////////////////////////////////////////////////////
    printf("DOTF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 3000.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-1270.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    dotf_C(inout, inout2, len, &inout_ref[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dotf_C(inout, inout2, len, &inout_ref[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dotf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    dotf_C_precise(inout, inout2, len, &inout_ref[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsDotProd_32f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsDotProd_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsDotProd_32f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsDotProd_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    dot128f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dot128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dot128f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dot128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    dot256f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dot256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dot256f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dot256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    dot512f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dot512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dot512f(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dot512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    dotf_vec(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dotf_vec(inout, inout2, len, &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dotf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// DOTF //////////////////////////////////////////////////////////////////////////////
    printf("DOTCF\n");

    for (int i = 0; i < 2 * len; i++) {
        inout[i] = (float) i / 3000.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-1270.577f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    dotcf_C((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout_ref[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotcf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dotcf_C((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout_ref[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dotcf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    dotcf_C_precise((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout_ref[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotcf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsDotProd_32fc((const Ipp32fc *) inout, (const Ipp32fc *) inout2, len, (Ipp32fc *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsDotProd_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsDotProd_32fc((const Ipp32fc *) inout, (const Ipp32fc *) inout2, len, (Ipp32fc *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsDotProd_32fc %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
    printf("%f %f ULPS %d\n", inout_ref[1], inout3[1], ulpsDistance32(inout_ref[1], inout3[1]));
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    dotc128f((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotc128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dotc128f((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dotc128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
    printf("%f %f ULPS %d\n", inout_ref[1], inout3[1], ulpsDistance32(inout_ref[1], inout3[1]));
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    dotc256f((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotc256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dotc256f((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dotc256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
    printf("%f %f ULPS %d\n", inout_ref[1], inout3[1], ulpsDistance32(inout_ref[1], inout3[1]));
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    dotc512f((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotc512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dotc512f((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dotc512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
    printf("%f %f ULPS %d\n", inout_ref[1], inout3[1], ulpsDistance32(inout_ref[1], inout3[1]));
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    dotcf_vec((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("dotcf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        dotcf_vec((complex32_t *) inout, (complex32_t *) inout2, len, (complex32_t *) &inout3[0]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("dotcf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%f %f ULPS %d\n", inout_ref[0], inout3[0], ulpsDistance32(inout_ref[0], inout3[0]));
    printf("%f %f ULPS %d\n", inout_ref[1], inout3[1], ulpsDistance32(inout_ref[1], inout3[1]));
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MULADD //////////////////////////////////////////////////////////////////////////////
    printf("MULADD\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i / 3000.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout2[i] = (float) i / (-1270.577f);
        inout3[i] = (float) i / (12070.12345f);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    muladdf_C(inout, inout2, inout3, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("muladdf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        muladdf_C(inout, inout2, inout3, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("muladdf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    muladd128f(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("muladd128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        muladd128f(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("muladd128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, len);

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    muladd256f(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("muladd256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        muladd256f(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("muladd256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    muladd512f(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("muladd512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        muladd512f(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("muladd512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    muladdf_vec(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("muladdf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        muladdf_vec(inout, inout2, inout3, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("muladdf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ADDCST //////////////////////////////////////////////////////////////////////////////
    printf("ADDCST\n");

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    addcf_C(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addcf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        addcf_C(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("addcf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAddC_32f(inout, 5.7f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAddC_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAddC_32f(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAddC_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    addc128f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        addc128f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("addc128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    addc256f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        addc256f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("addc256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    addc512f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addc512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        addc512f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("addc512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    addcf_vec(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("addcf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        addcf_vec(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("addcf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SUBCREVF //////////////////////////////////////////////////////////////////////////////
    printf("SUBCREVF\n");

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    subcrevf_C(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("subcrevf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        subcrevf_C(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("subcrevf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSubCRev_32f(inout, 5.7f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSubCRev_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSubCRev_32f(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsSubCRev_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    subcrev128f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("subcrev128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        subcrev128f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("subcrev128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    subcrev256f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("subcrev256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        subcrev256f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("subcrev256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    subcrev512f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("subcrev512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        subcrev512f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("subcrev512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    subcrevf_vec(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("subcrevf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        subcrevf_vec(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("subcrevf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mulcf_C(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mulcf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMulC_32f(inout, 5.7f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMulC_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMulC_32f(inout, 6.3f, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMulC_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    mulc128f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mulc128f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mulc128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef VDSP
    float val = 5.7f;
    clock_gettime(CLOCK_REALTIME, &start);
    vDSP_vsmul(inout, 1, &val, inout2, 1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vDSP_vsmul %d %lf\n", len, elapsed);

    val = 6.3f;
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vDSP_vsmul(inout, 1, &val, inout2, 1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vDSP_vsmul %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    mulc256f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mulc256f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mulc256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    mulc512f(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulc512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mulc512f(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mulc512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    mulcf_vec(inout, 5.7f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulcf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mulcf_vec(inout, 6.3f, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mulcf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MUL //////////////////////////////////////////////////////////////////////////////
    printf("MUL\n");

    clock_gettime(CLOCK_REALTIME, &start);
    mulf_C(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mulf_C(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mulf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMul_32f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMul_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMul_32f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMul_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsMul(len, inout, inout2, inout_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsMul %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsMul(len, inout, inout2, inout_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsMul %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    mul128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mul128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mul128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout_ref, len);
#endif


#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    mul256f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mul256f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mul256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    mul512f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mul512f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mul512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    mulf_vec(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mulf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mulf_vec(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mulf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SIN //////////////////////////////////////////////////////////////////////////////
    printf("SIN\n");

    flops = 34 * len;  // TODO : check the right theoretical value

    for (int i = 0; i < len; i++) {
        inout[i] = -(float) len / 16.0f + 0.1f * (float) i;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    sinf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinf_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    clock_gettime(CLOCK_REALTIME, &start);
    sinf_C_precise(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSin_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsSin_32f_A24 %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsSin(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsSin %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsSin(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsSin %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    sin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin128f_svml %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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
    for (l = 0; l < loop; l++)
        sin256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin256f_svml %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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
    for (l = 0; l < loop; l++)
        sin512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sin512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sin512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin512f_svml %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    sinf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);

    // for(int i = 0; i < len; i++) printf("%f %f %f\n",inout[i], inout2[i], inout2_ref[i]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// COS //////////////////////////////////////////////////////////////////////////////
    printf("COSF\n");

    clock_gettime(CLOCK_REALTIME, &start);
    cosf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cosf_C_precise(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCos_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCos_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCos_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsCos(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        cos128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cos128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cos128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cos128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        cos256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cos256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cos256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cos256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        cos512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cos512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cos512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cos512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cos512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif


#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cosf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
    l2_err(inout2_ref, inout2, len);

    // for(int i = 0; i < len; i++) printf("%f %f %f\n",inout[i], inout2[i], inout2_ref[i]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SINCOS //////////////////////////////////////////////////////////////////////////////
    printf("SINCOS\n");

    clock_gettime(CLOCK_REALTIME, &start);
    sincosf_C(inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincosf_C(inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincosf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sincosf_C_precise(inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSinCos_32f_A24(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSinCos_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSinCos_32f_A24(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsSinCos(len, inout, inout2, inout3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsSinCos %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        sincos128f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvsincosf(inout2, inout3, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvsincosf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvsincosf(inout2, inout3, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvsincosf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        sincos256f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        sincos512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos512f_svml(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    sincosf_vec(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    // for (l = 0; l < loop; l++)
    //     sincosf_vec(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincosf_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
    l2_err(inout2_ref, inout3, len);
#endif

    /*for (int i = 0; i < len; i++) {
        printf("%f %f %f %f %f\n",inout[i], inout_ref[i],inout2[i],inout2_ref[i],inout3[i]);
    }*/

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincosd_C(inoutd, inoutd_ref, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincosd_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSinCos_64f_A53(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSinCos_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSinCos_64f_A53(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsSinCos_64f_A53 %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif


#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vdSinCos(len, inoutd, inoutd2, inoutd3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvdSinCos %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vdSinCos(len, inoutd, inoutd2, inoutd3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvdSinCos %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos128d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128d_svml(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos128d_svml(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos128d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos256d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256d_svml(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos256d_svml(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos256d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos512d(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

    /*for(int i = 0; i < len; i++){
        printf("%lf || %lf %lf || %lf %lf\n",inoutd[i], inoutd_ref[i],inoutd2[i], inoutd2_ref[i], inoutd3[i]);
    }*/

    printf("\n");
    ///////////////////////////////////////////////// SINCOSF_INTERLEAVED //////////////////////////////////////////////////////////////////////
    printf("SINCOSF_INTERLEAVED\n");

    clock_gettime(CLOCK_REALTIME, &start);
    sincosf_C_interleaved(inout, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosf_C_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincosf_C_interleaved(inout, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincosf_C_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sincosf_C_interleaved_precise(inout, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosf_C_interleaved_precise %d %lf\n", len, elapsed);

#if defined(IPP)
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCIS_32fc_A24(inout, (Ipp32fc *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCIS_32fc_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCIS_32fc_A24(inout, (Ipp32fc *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsCIS_32fc_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, 2 * len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128f_interleaved(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos128f_interleaved(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos128f_interleaved %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, 2 * len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256f_interleaved(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos256f_interleaved(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos256f_interleaved %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, 2 * len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512f_interleaved(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos512f_interleaved(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512f_interleaved %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, 2 * len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    sincosf_interleaved_vec(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosf_interleaved_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincosf_interleaved_vec(inout, (complex32_t *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincosf_interleaved_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, 2 * len);
#endif
    /*for (int i = 0; i < 2*len; i+=2) {
        printf("%f %f %f %f\n",inout_ref[i],inout2[i],inout_ref[i+1],inout2[i+1]);
    }*/

    printf("\n");
    ///////////////////////////////////////////////// SINCOSF_INTERLEAVED //////////////////////////////////////////////////////////////////////
    printf("SINCOS_INTERLEAVED\n");

    clock_gettime(CLOCK_REALTIME, &start);
    sincosd_C_interleaved(inoutd, (complex64_t *) inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincosd_C_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincosd_C_interleaved(inoutd, (complex64_t *) inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincosd_C_interleaved %d %lf\n", len, elapsed);

#if defined(IPP)
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCIS_64fc_A53(inoutd, (Ipp64fc *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCIS_64fc_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCIS_64fc_A53(inoutd, (Ipp64fc *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsCIS_64fc_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, 2 * len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128d_interleaved(inoutd, (complex64_t *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128d_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos128d_interleaved(inoutd, (complex64_t *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos128d_interleaved %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, 2 * len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256d_interleaved(inoutd, (complex64_t *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256d_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos256d_interleaved(inoutd, (complex64_t *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos256d_interleaved %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, 2 * len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512d_interleaved(inoutd, (complex64_t *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512d_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos512d_interleaved(inoutd, (complex64_t *) inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512d_interleaved %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, 2 * len);
#endif

    // for(int i = 0; i < 2*len; i++) printf("%0.2lf %0.2lf\n",inoutd_ref[i],inoutd2[i]);

    printf("\n");
    /////////////////////////////////////////////////////////// LOGN //////////////////////////////////////////////////////////////////////////////
    printf("LOGNF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 1.82f;
        inout_ref[i] = inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    lnf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("lnf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        lnf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("lnf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    lnf_C_precise(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("lnf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsLn_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsLn_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsLn_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsLn_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsLn(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsLn %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsLn(len, inout, inout2_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsLn %d %lf\n", len, elapsed);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    ln_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln_128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln_128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvlogf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvlogf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvlogf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvlogf %d %lf\n", len, elapsed);
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
    for (l = 0; l < loop; l++)
        ln_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln_256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        ln_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln_512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln_512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    lnf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("lnf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        lnf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("lnf_vec %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// LOG2 //////////////////////////////////////////////////////////////////////////////
    printf("LOG2F\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 0.000001f) / 1.82f;
        inout_ref[i] = inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    log2f_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2f_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    log2f_C_precise(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2f_C_precise %d %lf\n", len, elapsed);

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsLog2(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsLog2 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsLog2(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsLog2 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    log2_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log2_128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_128f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log2_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    log2_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log2_256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_256f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_256f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log2_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    log2_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log2_512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_512f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_512f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log2_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2_512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2_512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    log2f_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2f_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2f_vec %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// LOG10 //////////////////////////////////////////////////////////////////////////////
    printf("LOG10F\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 0.000001f) / 1.82f;
        inout_ref[i] = inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    log10f_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10f_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    log10f_C_precise(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10f_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsLog10_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsLog10_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsLog10_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsLog10_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsLog10(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsLog10 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsLog10(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsLog10 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    log10_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log10_128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_128f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log10_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvlog10f(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvlog10f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvlog10f(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvlog10f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    log10_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log10_256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_256f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_256f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log10_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    log10_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log10_512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_512f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_512f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log10_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10_512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10_512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    log10f_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10f_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10f_vec %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    /*for(int i = 0; i < len; i++)
          printf("%f %f %f\n",inout[i],inout2_ref[i], inout2[i]);
        printf("\n\n");*/
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// EXP //////////////////////////////////////////////////////////////////////////////
    printf("EXPF\n");

    /*for (int i = 0; i < len; i++) {
        inout[i] = (float) (1.0f * i + 1.0f) / 10000.0f;
        inout_ref[i] = inout[i];
    }*/

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 50000) / 10000.0f;
        if (i % 2 == 0)
            inout[i] = -inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    expf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("expf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        expf_C(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("expf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    expf_C_precise(inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("expf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsExp_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsExp_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsExp_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsExp_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsExp(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsExp %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsExp(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsExp %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    exp_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifndef ALTIVEC
    clock_gettime(CLOCK_REALTIME, &start);
    exp_128f_(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_128f_ %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_128f_(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_128f_ %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    exp_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    exp_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    expf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("expf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        expf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("expf_vec %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

    /*for(int i = 0; i < len; i++)
          printf("%f %f %f\n",inout[i],inout2_ref[i], inout2[i]);
        printf("\n\n");*/

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convert_32f64f_C(inout, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convert_32f64f_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        convert128_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        convert256_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        convert512_32f64f(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convert512_32f64f %d %lf\n", len, elapsed);

    l2_errd(inoutd, inoutd_ref, len);
    /*for(int i =0; i < 16; i++){
        printf("%lf %lf %f\n",inoutd_ref[i],inoutd[i],inout[i]);
    }*/
#endif

#if defined(RISCV) && (ELEN >= 64)
    clock_gettime(CLOCK_REALTIME, &start);
    convert_32f64f_vec(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convert_32f64f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convert_32f64f_vec(inout, inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convert_32f64f_vec %d %lf\n", len, elapsed);

    l2_errd(inoutd, inoutd_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// FLIP //////////////////////////////////////////////////////////////////////////////
    printf("FLIP\n");

    clock_gettime(CLOCK_REALTIME, &start);
    flipf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flipf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flipf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flipf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsFlip_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFlip_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsFlip_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsFlip_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    flip128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flip128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        flip256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flip256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    flip512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flip512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flip512f %d %lf\n", len, elapsed);

    // for(int i = 0; i < len; i++)
    //   printf("%f %f\n",inout[i], inout2[i]);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    flipf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flipf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flipf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flipf_vec %d %lf\n", len, elapsed);

    // for(int i = 0; i < len; i++)
    //   printf("%f %f\n",inout[i], inout2[i]);
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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        floorf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("floorf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsFloor_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsFloor_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsFloor_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsFloor(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsFloor %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    floor128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        floor128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("floor128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvfloorf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvfloorf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvfloorf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvfloorf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    floor256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floor256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        floor256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        floor512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("floor512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    floorf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("floorf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        floorf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("floorf_vec %d %lf\n", len, elapsed);

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ceilf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ceilf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCeil_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCeil_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCeil_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsCeil(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsCeil %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    ceil128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ceil128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ceil128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvceilf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvceilf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvceilf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvceilf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    ceil256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceil256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ceil256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        ceil512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ceil512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    ceilf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ceilf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ceilf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ceilf_vec %d %lf\n", len, elapsed);

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        roundf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("roundf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsRound_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRound_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsRound_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsRound(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsRound %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    round128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("round128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        round128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        round256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        round512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("round512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    roundf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("roundf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        roundf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("roundf_vec %d %lf\n", len, elapsed);

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        truncf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("truncf_C %d %lf\n", len, elapsed);


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsTrunc_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTrunc_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsTrunc_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsTrunc_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsTrunc(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsTrunc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsTrunc(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsTrunc %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    trunc128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("trunc128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        trunc128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        trunc256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        trunc512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("trunc512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    truncf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("truncf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        truncf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("truncf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SQRTF //////////////////////////////////////////////////////////////////////////////
    printf("SQRTF\n");

    for (int i = 0; i < len; i++)
        inout[i] = (float) (rand() / 123456) / 3.55555f;

    clock_gettime(CLOCK_REALTIME, &start);
    sqrtf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sqrtf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sqrtf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sqrtf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sqrtf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sqrtf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSqrt_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippssqrt_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSqrt_32f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippssqrt_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    sqrt128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sqrt128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sqrt128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sqrt128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sqrt256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sqrt256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sqrt256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sqrt256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sqrt512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sqrt512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sqrt512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sqrt512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    sqrtf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sqrtf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sqrtf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sqrtf_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// TAN //////////////////////////////////////////////////////////////////////////////
    printf("TANF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 8000) / 1000.0f;
        if (i % 2 == 0)
            inout[i] = -inout[i];
    }


    clock_gettime(CLOCK_REALTIME, &start);
    tanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tanf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsTan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTan_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsTan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsTan(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsTan %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    tan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    tan128f_naive(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f_old %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan128f_naive(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan128f_old %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvtanf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvtanf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvtanf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvtanf %d %lf\n", len, elapsed);
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
    for (l = 0; l < loop; l++)
        tan256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        tan512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan512f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif


#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    tanf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanf_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// TAN //////////////////////////////////////////////////////////////////////////////
    printf("TAN\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = (double) (rand() % 8000) / 1000.0;
        if (i % 2 == 0)
            inoutd[i] = -inoutd[i];
    }


    clock_gettime(CLOCK_REALTIME, &start);
    tan_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsTan_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTan_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsTan_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsTan_64f_A53 %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    tan128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan128d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan128d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    tan256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan256d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan256d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    tan512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan512d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
    // for(int i = 0;  i < 512len; i++) printf("%lf %lf %lf \n",inoutd[i],inoutd_ref[i],inoutd2[i]);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tan512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan512d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan512d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

    printf("\n");
    /////////////////////////////////////////////////////////// TANH //////////////////////////////////////////////////////////////////////////////
    printf("TANHF\n");

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
    for (l = 0; l < loop; l++)
        tanhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    tanhf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanhf_C_precise %d %lf\n", len, elapsed);


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsTanh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsTanh_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsTanh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsTanh_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsTanh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsTanh %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsTanh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsTanh %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    tanh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tanh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh128f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvtanhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvtanhf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvtanhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvtanhf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    tanh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tanh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh256f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    tanh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    tanh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh512f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    tanhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanhf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanhf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// SINH //////////////////////////////////////////////////////////////////////////////
    printf("SINHF\n");

    clock_gettime(CLOCK_REALTIME, &start);
    sinhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sinhf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinhf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSinh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSinh_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSinh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsSinh_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsSinh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsSinh %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsSinh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsSinh %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    sinh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sinh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinh128f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvsinhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvsinhf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvsinhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvsinhf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sinh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sinh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinh256f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sinh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sinh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinh512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinh512f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    sinhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sinhf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sinhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sinhf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// COSH //////////////////////////////////////////////////////////////////////////////
    printf("COSHF\n");

    clock_gettime(CLOCK_REALTIME, &start);
    coshf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("coshf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        coshf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("coshf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    coshf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("coshf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCosh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCosh_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCosh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsCosh_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsCosh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsCosh %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsCosh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsCosh %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cosh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cosh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh128f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvcoshf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvcoshf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvcoshf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvcoshf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cosh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cosh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh256f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cosh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cosh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh512f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    coshf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("coshf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        coshf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("coshf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ATANH //////////////////////////////////////////////////////////////////////////////
    printf("ATANH\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 2000) / 10000.0f;
        if (i % 2 == 0)
            inout[i] = -inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    atanhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atanhf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanhf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtanh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtanh_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAtanh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAtanh_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAtanh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAtanh %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsAtanh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsAtanh %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    atanh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atanh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanh128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanh128f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvatanhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvatanhf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvatanhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvatanhf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    atanh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atanh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanh256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanh256f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    atanh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atanh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanh512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanh512f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    atanhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanhf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanhf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ASINH //////////////////////////////////////////////////////////////////////////////
    printf("ASINHF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 2000) / 10000.0f;
        if (i % 2 == 0)
            inout[i] = -inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    asinhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinhf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinhf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asinhf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinhf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsinh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsinh_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAsinh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAsinh_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAsinh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAsinh %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsAsinh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsAsinh %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    asinh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asinh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinh128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinh128f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvasinhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvasinhf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvasinhf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvasinhf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    asinh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asinh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinh256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinh256f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    asinh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asinh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinh512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinh512f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    asinhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinhf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinhf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinhf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ACOSH //////////////////////////////////////////////////////////////////////////////
    printf("ACOSHF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 2000) / 1000.0f + 1.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    acoshf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acoshf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acoshf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acoshf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    acoshf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acoshf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAcosh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAcosh_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAcosh_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAcosh_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsAcosh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvsAcosh %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsAcosh(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsAcosh %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    acosh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acosh128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acosh128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acosh128f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    acosh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acosh128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acosh128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acosh128f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvacoshf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvacoshf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvacoshf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvacoshf %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

    /* for(int i = 0; i < len; i++)
          printf("%f %f %f \n",inout[i],inout_ref[i], inout2[i]);
        printf("\n\n");*/

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    acosh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acosh256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acosh256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acosh256f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    acosh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acosh256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acosh256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acosh256f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    acosh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acosh512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acosh512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acosh512f %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    acosh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acosh512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acosh512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acosh512f_svml %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    acoshf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("acoshf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        acoshf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("acoshf_vec %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    asinf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAsin_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsAsin(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsAsin %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    asin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvasinf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvasinf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvasinf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvasinf %d %lf\n", len, elapsed);
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
    for (l = 0; l < loop; l++)
        asin256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        asin512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin512f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    asinf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asinf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asinf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asinf_vec %d %lf\n", len, elapsed);
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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAsin_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAsin_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAsin_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAsin_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

    /*
for (int i = 0; i < len; i++){
    printf("%lf , %lf, %lf %lf\n",inoutd[i],inoutd_ref[i],inoutd2[i],asin(inoutd[i]));
}*/

#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vdAsin(len, inoutd, inoutd2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvdAsin %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vdAsin(len, inoutd, inoutd2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvdAsin %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    asin128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);


    clock_gettime(CLOCK_REALTIME, &start);
    asin128d_(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d_ %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin128d_(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin128d_ %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

    /*
for (int i = 0; i < len; i++){
    printf("%lf , %lf, %lf %lf\n",inoutd[i],inoutd_ref[i],inoutd2[i],asin(inoutd[i]));
}*/

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin128d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    asin256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin256d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    asin512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    asin512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin512d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ATANF //////////////////////////////////////////////////////////////////////////////
    printf("ATANF\n");

    clock_gettime(CLOCK_REALTIME, &start);
    atanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atanf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanf_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAtan_32f_A24(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsAtan(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsAtan %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    atan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("atan128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan128f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvatanf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvatanf %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvatanf(inout2, inout, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvatanf %d %lf\n", len, elapsed);
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
    for (l = 0; l < loop; l++)
        atan256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        atan512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    atanf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atanf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atanf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atanf_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

    // for(int i = 0; i < len; i++)
    //   printf("%f %f %f\n", inout[i], inout2[i], inout_ref[i]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ATAN2F //////////////////////////////////////////////////////////////////////////////
    printf("ATAN2F\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (-1.0f * i + 0.15f) / 2.5f / (float) (5 * len);
        inout2[i] = (float) (rand() % 12345) / 123456.0f + 0.123f;
        if (i % 4 == 0)
            inout2[i] = -inout2[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    atan2f_C(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2f_C(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2f_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan2f_C_precise(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2f_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan2_32f_A24(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan2_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAtan2_32f_A24(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vsAtan2(len, inout, inout2, inout_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsAtan2 %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    atan2128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2128f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan2128f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2128f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef VDSP
    clock_gettime(CLOCK_REALTIME, &start);
    vvatan2f(inout_ref, inout, inout2, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vvatan2f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vvatan2f(inout_ref, inout, inout2, &len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vvatan2f %d %lf\n", len, elapsed);
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
    for (l = 0; l < loop; l++)
        atan2256f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan2256f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2256f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        atan2512f(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan2512f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2512f_svml(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    atan2f_vec(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2f_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2f_vec(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2f_vec %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ATANF2_INTERLEAVED /////////////////////////////////////////////////////
    printf("ATAN2F_INTERLEAVED\n");

    for (int i = 0; i < 2 * len; i++) {
        inout[i] = (float) (-1.0f * i + 0.15f) / 2.5f / (float) (5 * len);
        inout_ref[i] = 50.0f;
        inout2_ref[i] = 50.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    atan2f_interleaved_C((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2f_interleaved_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2f_interleaved_C((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2f_interleaved_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    atan2f_interleaved_C_precise((complex32_t *) inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2f_interleaved_C_precise %d %lf\n", len, elapsed);

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    atan2128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2128f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2128f_interleaved %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    atan2256f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2256f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2256f_interleaved %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    atan2512f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2512f_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2512f_interleaved((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2512f_interleaved %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    atan2f_interleaved_vec((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2f_interleaved_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2f_interleaved_vec((complex32_t *) inout, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2f_interleaved_vec %d %lf\n", len, elapsed);
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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan_C %d %lf\n", len, elapsed);


#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAtan_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAtan_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);


    /*for (int i = 0; i < len; i++){
        printf("%lf , %lf, %lf %lf\n",inoutd[i],inoutd_ref[i],inoutd2[i],atan(inoutd[i]));
    }*/

#endif

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vdAtan(len, inoutd, inoutd2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("MKLvdAtan %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vdAtan(len, inoutd, inoutd2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvdAtan %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    atan128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("atan128d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan128d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    atan256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("atan256d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan256d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    atan512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("atan512d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan512d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

    printf("\n");
    /////////////////////////////////////////////////////////// ATANF2 //////////////////////////////////////////////////////////////////////////////
    printf("ATAN2\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = (double) (-1.0 * i + 0.15) / 2.5 / (double) (5 * len);
        inoutd2[i] = (double) (rand() % 12345) / 123456.0 + 0.123;
        if (i % 4 == 0)
            inout2[i] = -inout2[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    atan2_C(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2_C(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsAtan2_64f_A53(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsAtan2_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsAtan2_64f_A53(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsAtan2_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd2_ref, inoutd_ref, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    atan2128d(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2128d(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2128d %d %lf\n", len, elapsed);
    l2_errd(inoutd2_ref, inoutd_ref, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    atan2256d(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2256d(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2256d %d %lf\n", len, elapsed);
    l2_errd(inoutd2_ref, inoutd_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    atan2512d(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2512d(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2512d %d %lf\n", len, elapsed);
    l2_errd(inoutd2_ref, inoutd_ref, len);
#endif


    printf("\n");
    /////////////////////////////////////////////////////////// ATANF2_INTERLEAVED /////////////////////////////////////////////////////
    printf("ATAN2_INTERLEAVED\n");

    for (int i = 0; i < 2 * len; i++) {
        inoutd[i] = (double) (-1.0 * i + 0.15) / 2.5 / (double) (5 * len);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    atan2_interleaved_C((complex64_t *) inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2d_interleaved_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2_interleaved_C((complex64_t *) inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2d_interleaved_C %d %lf\n", len, elapsed);

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    atan2128d_interleaved((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2128d_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2128d_interleaved((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2128d_interleaved %d %lf\n", len, elapsed);
    l2_errd(inoutd2, inoutd_ref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    atan2256d_interleaved((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2256d_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2256d_interleaved((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2256d_interleaved %d %lf\n", len, elapsed);
    l2_errd(inoutd2, inoutd_ref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    atan2512d_interleaved((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2512d_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2512d_interleaved((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2512d_interleaved %d %lf\n", len, elapsed);
    l2_errd(inoutd2, inoutd_ref, len);
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
    cplxtorealf_C((complex32_t *) inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtorealf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtorealf_C((complex32_t *) inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtorealf_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCplxToReal_32fc((const Ipp32fc *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCplxToReal_32fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCplxToReal_32fc((const Ipp32fc *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsCplxToReal_32fc %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal128f((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreal128f((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreal128f %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

#ifdef VDSP
    DSPSplitComplex dsp_split;
    dsp_split.realp = inout3;
    dsp_split.imagp = inout4;

    clock_gettime(CLOCK_REALTIME, &start);
    vDSP_ctoz((DSPComplex *) inout, 2, &dsp_split, 1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vDSP_ctoz %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vDSP_ctoz((DSPComplex *) inout, 2, &dsp_split, 1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("vDSP_ctoz %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif


#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal256f((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreal256f((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreal256f %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal512f((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreal512f((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreal512f %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

    /*for(int i = 0; i < len; i++)
          printf("%f %f | %f %f %f %f \n",inout[2*i], inout[2*i + 1], inout3[i],inout_ref[i], inout4[i], inout2_ref[i]);
        printf("\n\n");*/
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtorealf_vec((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtorealf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtorealf_vec((complex32_t *) inout, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtorealf_vec %d %lf\n", len, elapsed);
    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// REAL2CPLX //////////////////////////////////////////////////////////////////////////////
    printf("REAL2CPLX\n");

    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx_C(inout3, inout4, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplx_C(inout3, inout4, (complex32_t *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("realtocplx_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsRealToCplx_32f(inout3, inout4, (Ipp32fc *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRealToCplx_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsRealToCplx_32f(inout3, inout4, (Ipp32fc *) inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsRealToCplx_32f %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, 2 * len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx128f(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx128f %d %lf\n", len, elapsed);

    /*for(int i = 0; i < 2*len; i++)
          printf("%f %f\n",inout[i],inout_ref[i]);
        printf("\n\n");*/

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplx128f(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("realtocplx128f %d %lf\n", len, elapsed);
    l2_err(inout, inout_ref, 2 * len);


#endif

#ifdef VDSP
    //	DSPSplitComplex dsp_split;
    //	dsp_split.realp=inout3;
    //	dsp_split.imagp=inout4;

    clock_gettime(CLOCK_REALTIME, &start);
    vDSP_ztoc(&dsp_split, 1, (DSPComplex *) inout, 2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vDSP_ztoc %d %lf\n", len, elapsed);

    /*for(int i = 0; i < 2*len; i++)
          printf("%f %f\n",inout[i],inout_ref[i]);
        printf("\n\n");*/

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vDSP_ztoc(&dsp_split, 1, (DSPComplex *) inout, 2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("vDSP_ztoc %d %lf\n", len, elapsed);
    l2_err(inout, inout_ref, 2 * len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx256f(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplx256f(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("realtocplx256f %d %lf\n", len, elapsed);
    l2_err(inout, inout_ref, 2 * len);

    /*for(int i = 0; i < 2*len; i++)
          printf("%f %f\n",inout[i],inout_ref[i]);*/
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx512f(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplx512f(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("realtocplx512f %d %lf\n", len, elapsed);
    l2_err(inout, inout_ref, 2 * len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplxf_vec(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplxf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplxf_vec(inout3, inout4, (complex32_t *) inout, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("realtocplxf_vec %d %lf\n", len, elapsed);
    l2_err(inout, inout_ref, 2 * len);
#endif

    /*for(int i = 0; i < len; i++)
        printf("%f %f || %f %f \n",inout3[i],inout_ref[i], inout4[i],inout2_ref[i]);*/



    printf("\n");
    /////////////////////////////////////////////////////////// CPLX2REALD //////////////////////////////////////////////////////////////////////////////
    printf("CPLX2REALD\n");
    for (int i = 0; i < 2 * len; i++) {
        inoutd[i] = (double) (rand() % 12345);
    }

    for (int i = 0; i < len; i++) {
        inoutd2[i] = 0.0;
        inoutd3[i] = 0.0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreald_C((complex64_t *) inoutd, inoutd_ref, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreald_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreald_C((complex64_t *) inoutd, inoutd_ref, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreald_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCplxToReal_64fc((const Ipp64fc *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCplxToReal_64fc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCplxToReal_64fc((const Ipp64fc *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsCplxToReal_64fc %d %lf\n", len, elapsed);
    l2_errd(inoutd2, inoutd_ref, len);
    l2_errd(inoutd3, inoutd2_ref, len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal128d((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreal128d((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreal128d %d %lf\n", len, elapsed);
    l2_errd(inoutd2, inoutd_ref, len);
    l2_errd(inoutd3, inoutd2_ref, len);

#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal256d((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreal256d((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreal256d %d %lf\n", len, elapsed);
    l2_errd(inoutd2, inoutd_ref, len);
    l2_errd(inoutd3, inoutd2_ref, len);

#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreal512d((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreal512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreal512d((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreal512d %d %lf\n", len, elapsed);
    l2_errd(inoutd2, inoutd_ref, len);
    l2_errd(inoutd3, inoutd2_ref, len);

#endif
    printf("\n");
    /////////////////////////////////////////////////////////// REAL2CPLXD //////////////////////////////////////////////////////////////////////////////
    printf("REAL2CPLXD\n");

    clock_gettime(CLOCK_REALTIME, &start);
    realtocplxd_C(inoutd2, inoutd3, (complex64_t *) inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplxd_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplxd_C(inoutd2, inoutd3, (complex64_t *) inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("realtocplxd_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsRealToCplx_64f(inoutd2, inoutd3, (Ipp64fc *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsRealToCplx_64f %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsRealToCplx_64f(inoutd2, inoutd3, (Ipp64fc *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsRealToCplx_64f %d %lf\n", len, elapsed);
    l2_errd(inoutd, inoutd_ref, 2 * len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx128d(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx128d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplx128d(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("realtocplx128d %d %lf\n", len, elapsed);
    l2_errd(inoutd, inoutd_ref, 2 * len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx256d(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx256d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplx256d(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("realtocplx256d %d %lf\n", len, elapsed);
    l2_errd(inoutd, inoutd_ref, 2 * len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplx512d(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplx512d %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplx512d(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("realtocplx512d %d %lf\n", len, elapsed);
    l2_errd(inoutd, inoutd_ref, 2 * len);
#endif

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_C(inout, inout_u2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f8u_Sfs(inout, inout_u1, len, ippRndZero, 4);  // ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32f8u_Sfs(inout, inout_u1, len, ippRndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);
    l2_err_u8(inout_u1, inout_u2, len);

    /*for(int i = 0; i < len; i++)
          printf("%x %x\n" ,inout_u1[i],inout_u2[i]);*/


#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_128(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_128(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_128 %d %lf\n", len, elapsed);
    l2_err_u8(inout_u1, inout_u2, len);
#endif
#endif

    /*for(int i = 0; i < len; i++)
        printf("%x %x\n" ,inout_u1[i],inout_u2[i]);
    */
#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_256(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_256(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_256 %d %lf\n", len, elapsed);

    l2_err_u8(inout_u1, inout_u2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_512(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_512(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_512 %d %lf\n", len, elapsed);

    l2_err_u8(inout_u1, inout_u2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_vec(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_vec(inout, inout_u1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_vec %d %lf\n", len, elapsed);

    l2_err_u8(inout_u1, inout_u2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CONVERTFLOAT32_I16 //////////////////////////////////////////////////////////////////////////////
    printf("CONVERTFLOAT32_I16\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (i) *10.0f;
        inout_s1[i] = 0;
        inout_s2[i] = 0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_C(inout, inout_s1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_C(inout, inout_s1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f16s_Sfs(inout, inout_s2, len, ippRndZero, 4);  // ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f16s_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32f16s_Sfs(inout, inout_s2, len, ippRndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_32f16s_Sfs %d %lf\n", len, elapsed);
    l2_err_i16(inout_s1, inout_s2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_128(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_128(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_128 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_256(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_256(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_256 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_512(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_512(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_512 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_vec(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_vec(inout, inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_vec %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CONVERTFLOAT32_U16 //////////////////////////////////////////////////////////////////////////////
    printf("CONVERTFLOAT32_U16\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (i) *10.0f;
        inout_s1[i] = 0;
        inout_s2[i] = 0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_C(inout, (uint16_t *) inout_s1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_C(inout, (uint16_t *) inout_s1, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f16u_Sfs(inout, (uint16_t *) inout_s2, len, ippRndZero, 4);  // ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f16u_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32f16u_Sfs(inout, (uint16_t *) inout_s2, len, ippRndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_32f16u_Sfs %d %lf\n", len, elapsed);
    l2_err_i16(inout_s1, inout_s2, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_128(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_128(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_128 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif
    /*for(int i=0; i < len; i++)
      printf("%f %u %u\n",inout[i], (uint16_t)inout_s1[i], (uint16_t)inout_s2[i]);*/
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_256(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_256(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_256 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_512(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_512(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_512 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_vec(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_vec(inout, (uint16_t *) inout_s2, len, RndZero, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_vec %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt16ToFloat32_C(inout_s1, inout2_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt16ToFloat32_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_16s32f_Sfs(inout_s1, inout_ref, len, 4);  // ippRndNear ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_16s32f_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_16s32f_Sfs(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_16s32f_Sfs %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_128(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt16ToFloat32_128(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt16ToFloat32_128 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif
#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_256(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt16ToFloat32_256(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt16ToFloat32_256 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_512(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt16ToFloat32_512(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt16ToFloat32_512 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt16ToFloat32_vec(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt16ToFloat32_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt16ToFloat32_vec(inout_s1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt16ToFloat32_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif


    /*for(int i = 0; i < len; i++)
        printf("%d %f %f %f\n" ,inout_s1[i], (float)inout_s1[i]/(1 << 4),inout_ref[i],inout2_ref[i]);*/

    printf("\n");
    /////////////////////////////////////////////////////////// VECTOR_SLOPE //////////////////////////////////////////////////////////////////////////////
    printf("VECTOR_SLOPE\n");

    // for(int i = 0; i < len; i++)
    //	printf("%d %f %f %f\n" ,inout_s1[i], (float)inout_s1[i]/(1 << 4),inout_ref[i],inout2_ref[i]);

    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlopef_C(inout2_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vectorSlopef_C(inout2_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vectorSlope_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_32f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsVectorSlope_32f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsVectorSlope_32f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope128f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vectorSlope128f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vectorSlope256f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vectorSlope512f(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vectorSlope512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlopef_vec(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlopef_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vectorSlopef_vec(inout_ref, len, 2.5f, 3.0f);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vectorSlopef_vec %d %lf\n", len, elapsed);
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

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vectorSloped_C(inoutd_ref, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vectorSloped_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsVectorSlope_64f(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsVectorSlope_64f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsVectorSlope_64f(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vectorSlope128d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
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
    for (l = 0; l < loop; l++)
        vectorSlope256d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vectorSlope256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSlope512d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSlope512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vectorSlope512d(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vectorSlope512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd, len);
#endif

#if defined(RISCV) && (ELEN >= 64)
    clock_gettime(CLOCK_REALTIME, &start);
    vectorSloped_vec(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("vectorSloped_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vectorSloped_vec(inoutd, len, 2.5, 3.0);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("vectorSloped_vec %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd, len);
#endif

    // for(int i = 0; i < len; i++)printf("%lf %lf\n",inoutd_ref[i], inoutd[i]);

    printf("\n");
    /////////////////////////////////////////////////////////// SIGMOID //////////////////////////////////////////////////////////////////////////////
    printf("SIGMOID\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 80000) / 1000.0f;
        if (i % 2 == 0)
            inout[i] = -inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    sigmoidf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sigmoidf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sigmoidf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sigmoidf_C %d %lf\n", len, elapsed);

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    sigmoid128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sigmoid128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sigmoid128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sigmoid128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    sigmoid128f_(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sigmoid128f_ %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sigmoid128f_(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sigmoid128f_ %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sigmoid256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sigmoid256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sigmoid256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sigmoid256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sigmoid512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sigmoid512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sigmoid512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sigmoid512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    sigmoidf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sigmoidf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sigmoidf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sigmoidf_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    /*for(int i = 0; i < len; i++)
      printf("%f %f %f\n",inout[i], inout_ref[i],inout2[i]);*/

    printf("\n");
    /////////////////////////////////////////////////////////// PReluf //////////////////////////////////////////////////////////////////////////////
    printf("PReluf\n");

    clock_gettime(CLOCK_REALTIME, &start);
    PReluf_C(inout, inout_ref, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("PReluf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        PReluf_C(inout, inout_ref, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("PReluf_C %d %lf\n", len, elapsed);

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    PRelu128f(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("PRelu128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        PRelu128f(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("PRelu128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    PRelu256f(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("PRelu256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        PRelu256f(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("PRelu256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    PRelu512f(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("PRelu512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        PRelu512f(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("PRelu512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    PReluf_vec(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("PReluf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        PReluf_vec(inout, inout2, 0.05f, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("PReluf_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// Softmax //////////////////////////////////////////////////////////////////////////////
    printf("Softmax\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 50000) / 1000.0f;
        if (i % 2 == 0)
            inout[i] = -inout[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    softmaxf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("softmaxf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        softmaxf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("softmaxf_C %d %lf\n", len, elapsed);

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    softmax128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("softmax128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        softmax128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("softmax128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    softmax128f_dualacc(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("softmax128f_dualacc %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        softmax128f_dualacc(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("softmax128f_dualacc %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    softmax256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("softmax256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        softmax256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("softmax256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    softmax512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("softmax512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        softmax512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("softmax512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    softmaxf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("softmaxf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        softmaxf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("softmaxf_vec %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

    /*for(int i = 0; i < len; i++)
          printf("%f %f %f\n",inout[i],inout_ref[i], inout2[i]);
        printf("\n\n");*/

    printf("\n");
    /////////////////////////////////////////////////////////// ABSDIFF_S16 //////////////////////////////////////////////////////////////////////////////
    printf("ABSDIFF_S16\n");


    for (int i = 0; i < len; i++) {
        inout_s1[i] = (rand() % 32767);
        if (i % 4 == 0)
            inout_s1[i] = -inout_s1[i];
        inout_s2[i] = (rand() % 10500);
        if (i % 7 == 0)
            inout_s2[i] = -inout_s2[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    absdiff16s_c(inout_s1, inout_s2, inout_sref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("absdiff16s_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        absdiff16s_c(inout_s1, inout_s2, inout_sref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("absdiff16s_c %d %lf\n", len, elapsed);

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    absdiff16s_128s(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("absdiff16s_128s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        absdiff16s_128s(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("absdiff16s_128s %d %lf\n", len, elapsed);
    l2_err_i16(inout_sref, inout_s3, len);
#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    absdiff16s_256s(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("absdiff16s_256s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        absdiff16s_256s(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("absdiff16s_256s %d %lf\n", len, elapsed);
    l2_err_i16(inout_sref, inout_s3, len);
#endif
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    absdiff16s_512s(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("absdiff16s_512s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        absdiff16s_512s(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("absdiff16s_512s %d %lf\n", len, elapsed);
    l2_err_i16(inout_sref, inout_s3, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    absdiff16s_vec(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("absdiff16s_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        absdiff16s_vec(inout_s1, inout_s2, inout_s3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("absdiff16s_vec %d %lf\n", len, elapsed);
    l2_err_i16(inout_sref, inout_s3, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// POWERSPECT_S16_INTERLEAVED //////////////////////////////////////////////////////////////////////////////
    printf("POWERSPECT_S16_INTERLEAVED\n");


    for (int i = 0; i < 2 * len; i++) {
        inout_s1[i] = (rand() % 32767);
        if (i % 4 == 0)
            inout_s1[i] = -inout_s1[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    powerspect16s_c_interleaved((complex16s_t *) inout_s1, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect16s_c_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect16s_c_interleaved((complex16s_t *) inout_s1, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect16s_c_interleaved %d %lf\n", len, elapsed);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect16s_128s_interleaved((complex16s_t *) inout_s1, inout_i1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect16s_128s_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect16s_128s_interleaved((complex16s_t *) inout_s1, inout_i1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect16s_128s_interleaved %d %lf\n", len, elapsed);
    l2_err_i32(inout_i1, inout_iref, len);
#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect16s_256s_interleaved((complex16s_t *) inout_s1, inout_i1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect16s_256s_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect16s_256s_interleaved((complex16s_t *) inout_s1, inout_i1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect16s_256s_interleaved %d %lf\n", len, elapsed);
    l2_err_i32(inout_i1, inout_iref, len);
#endif
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    powerspect16s_512s_interleaved((complex16s_t *) inout_s1, inout_i1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("powerspect16s_512s_interleaved %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powerspect16s_512s_interleaved((complex16s_t *) inout_s1, inout_i1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powerspect16s_512s_interleaved %d %lf\n", len, elapsed);
    l2_err_i32(inout_i1, inout_iref, len);
#endif
    /*for(int i = 0; i < len; i++)
          printf("%d %d %d\n",inout_s1[i], inout_i1[i], inout_iref[i]);
        printf("\n\n");*/

    printf("\n");
    /////////////////////////////////////////////////////////// SUM16S32S //////////////////////////////////////////////////////////////////////////////
    printf("SUM16S32S\n");


    for (int i = 0; i < 2 * len; i++) {
        inout_s1[i] = (rand() % 32767);
        if (i % 4 == 0)
            inout_s1[i] = -inout_s1[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    sum16s32s_C(inout_s1, len, &inout_iref[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sum16s32s_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sum16s32s_C(inout_s1, len, &inout_iref[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sum16s32s_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsSum_16s32s_Sfs(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsSum_16s32s_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsSum_16s32s_Sfs(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsSum_16s32s_Sfs %d %lf\n", len, elapsed);
    printf("%d %d\n", inout_iref[0], inout_i1[0]);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    sum16s32s128(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sum16s32s128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sum16s32s128(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sum16s32s128 %d %lf\n", len, elapsed);
    printf("%d %d\n", inout_iref[0], inout_i1[0]);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    sum16s32s256(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sum16s32s256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sum16s32s256(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sum16s32s256 %d %lf\n", len, elapsed);
    printf("%d %d\n", inout_iref[0], inout_i1[0]);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    sum16s32s512(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sum16s32s512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sum16s32s512(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sum16s32s512 %d %lf\n", len, elapsed);
    printf("%d %d\n", inout_iref[0], inout_i1[0]);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    sum16s32s_vec(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sum16s32s_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sum16s32s_vec(inout_s1, len, &inout_i1[0], 3);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sum16s32s_vec %d %lf\n", len, elapsed);
    printf("%d %d\n", inout_iref[0], inout_i1[0]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MINMAX //////////////////////////////////////////////////////////////////////////////
    printf("MINMAXS\n");

    flops = 2 * len;
    int32_t mins, maxs, mins_ref, maxs_ref;

    for (int i = 0; i < len; i++) {
        inout_i1[i] = rand();
        if (i % 4 == 0)
            inout_i1[i] = -inout_i1[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    minmaxs_c(inout_i1, len, &mins_ref, &maxs_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmaxs_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmaxs_c(inout_i1, len, &mins_ref, &maxs_ref);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmaxs_c %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsMinMax_32s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsMinMax_32s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsMinMax_32s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsMinMax_32s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%d %d || %d %d\n", mins_ref, mins, maxs_ref, maxs);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    minmax128s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax128s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmax128s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmax128s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%d %d || %d %d\n", mins_ref, mins, maxs_ref, maxs);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    minmax256s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax256s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmax256s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmax256s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%d %d || %d %d\n", mins_ref, mins, maxs_ref, maxs);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    minmax512s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmax512s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmax512s(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmax512s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%d %d || %d %d\n", mins_ref, mins, maxs_ref, maxs);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    minmaxs_vec(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minmaxs_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minmaxs_vec(inout_i1, len, &mins, &maxs);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minmaxs_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    printf("%d %d || %d %d\n", mins_ref, mins, maxs_ref, maxs);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MAXEVERYS //////////////////////////////////////////////////////////////////////////////
    printf("MAXEVERYS\n");

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    maxeverys_c(inout_i1, inout_i2, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxeverys_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxeverys_c(inout_i1, inout_i2, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxeverys_c %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery128s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxevery128s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxevery128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery256s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery256s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxevery256s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxevery256s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    maxevery512s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxevery512s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxevery512s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxevery512s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    maxeverys_vec(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("maxeverys_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        maxeverys_vec(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("maxeverys_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MINEVERYS //////////////////////////////////////////////////////////////////////////////
    printf("MINEVERYS\n");

    flops = len;

    clock_gettime(CLOCK_REALTIME, &start);
    mineverys_c(inout_i1, inout_i2, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mineverys_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mineverys_c(inout_i1, inout_i2, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mineverys_c %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    minevery128s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery128s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minevery128s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minevery128s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    minevery256s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery256s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minevery256s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minevery256s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    minevery512s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("minevery512s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        minevery512s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("minevery512s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    mineverys_vec(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mineverys_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mineverys_vec(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mineverys_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i3, inout_iref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// THRESHOLD_LTS //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_LTS\n");

    for (int i = 0; i < len; i++) {
        inout_i1[i] = i / 2 - 37;
        inout_i2[i] = 0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_lt_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_lt_s_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_lt_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_lt_s_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LT_32s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreashold_LT_32s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_LT_32s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreashold_LT_32s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_lt_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_lt_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_lt_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_lt_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_lt_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_lt_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_lt_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_lt_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_lt_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_lt_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_lt_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_lt_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_lt_s_vec(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_lt_s_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_lt_s_vec(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_lt_s_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// THRESHOLD_GTS //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_GTS\n");

    for (int i = 0; i < len; i++) {
        inout_i1[i] = i / 3 - 18;
        inout_i2[i] = 0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gt_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gt_s_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gt_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gt_s_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_GT_32s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_GT_32s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_GT_32s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_GT_32s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_gt_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_gt_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_gt_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_gt_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_gt_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_gt_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_gt_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_gt_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_gt_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_gt_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_gt_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_gt_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gt_s_vec(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gt_s_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gt_s_vec(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gt_s_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// THRESHOLD_GTABSS //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_GTABSS\n");

    flops = 2 * len;

    for (int i = 0; i < len; i++) {
        inout_i1[i] = (-i) / 4 + 17 - 3 * i;
        inout_i2[i] = 0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gtabs_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gtabs_s_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gtabs_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gtabs_s_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_GTAbs_32s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_GTAbs_32s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_GTAbs_32s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_GTAbs_32s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_gtabs_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_gtabs_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_gtabs_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_gtabs_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_gtabs_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_gtabs_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_gtabs_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_gtabs_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_gtabs_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_gtabs_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_gtabs_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_gtabs_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_gtabs_s_vec(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_gtabs_s_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_gtabs_s_vec(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_gtabs_s_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);

    // for(int i = 0; i < len; i++) printf("%d %d %d\n", inout_i1[i], inout_i2[i], inout_iref[i]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// THRESHOLD_LTABSS //////////////////////////////////////////////////////////////////////////////
    printf("THRESHOLD_LTABSS\n");

    flops = 2 * len;

    for (int i = 0; i < len; i++) {
        inout_i1[i] = (-i) / 4 + 17 - 3 * i;
        inout_i2[i] = 0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltabs_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltabs_s_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltabs_s_C(inout_i1, inout_iref, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltabs_s_C %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsThreshold_LTAbs_32s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsThreshold_LTAbs_32s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsThreshold_LTAbs_32s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsThreshold_LTAbs_32s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold128_ltabs_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold128_ltabs_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold128_ltabs_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold128_ltabs_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold256_ltabs_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold256_ltabs_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold256_ltabs_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold256_ltabs_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold512_ltabs_s(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold512_ltabs_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold512_ltabs_s(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold512_ltabs_s %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);
#endif
    /*for(int i = 0; i < len; i++){
        printf("%d %d %d\n",inout_i1[i],inout_i2[i],inout_iref[i]);
    }*/

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    threshold_ltabs_s_vec(inout_i1, inout_i2, len, 700);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("threshold_ltabs_s_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        threshold_ltabs_s_vec(inout_i1, inout_i2, len, 500);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("threshold_ltabs_s_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err_i32(inout_i2, inout_iref, len);

    // for(int i = 0; i < len; i++) printf("%d %d %d\n", inout_i1[i], inout_i2[i], inout_iref[i]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// FLIPS //////////////////////////////////////////////////////////////////////////////
    printf("FLIPS\n");

    clock_gettime(CLOCK_REALTIME, &start);
    flips_C(inout_i1, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flips_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flips_C(inout_i1, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flips_C %d %lf\n", len, elapsed);

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    flip128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip128s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flip128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flip128s %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i2, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    flip256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip256s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flip256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flip256s %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i2, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    flip512s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flip512s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flip512s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flip512s %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i2, len);
#endif

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    flips_vec(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("flips_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        flips_vec(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("flips_vec %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i2, len);

    // for(int i = 0; i < len; i++) printf("%d %d %d\n",inout_i1[i], inout_i2[i], inout_iref[i]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// MULS //////////////////////////////////////////////////////////////////////////////
    printf("MULS\n");

    for (int i = 0; i < len; i++) {
        inout_i1[i] = (-32000) + rand() % 65000;
        inout_i2[i] = (65000) + -3 * (rand() % 22000);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    muls_c(inout_i1, inout_i2, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("muls_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        muls_c(inout_i1, inout_i2, inout_iref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("muls_c %d %lf\n", len, elapsed);

#if defined(SSE)  // || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    mul128s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul128s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mul128s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mul128s %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i3, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    mul256s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul256s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mul256s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mul256s %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i3, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    mul512s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("mul512s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        mul512s(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("mul512s %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i3, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    muls_vec(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("muls_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        muls_vec(inout_i1, inout_i2, inout_i3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("muls_vec %d %lf\n", len, elapsed);
    l2_err_i32(inout_iref, inout_i3, len);
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
    for (l = 0; l < loop; l++)
        memcpy(inout_iref, inout_i1, len * sizeof(int32_t));
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("memcpy %d BW %lf GB/s\n", len, (GB / elapsed));

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCopy_32s((const Ipp32s *) inout_i1, (Ipp32s *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("ippsCopy_32s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCopy_32s((const Ipp32s *) inout_i1, (Ipp32s *) inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("ippsCopy_32s %d BW %lf GB/s\n", len, (GB / elapsed));
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);


    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fast_copy128s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("fast_copy128s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fast_copy128s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("fast_copy128s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy128s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy128s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fast_copy128s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("fast_copy128s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);
#endif

#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);


    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fast_copy256s(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("fast_copy256s %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fast_copy256s_2(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("fast_copy256s_2 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    fast_copy256s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("fast_copy256s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fast_copy256s_4(inout_i1, inout_i2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) * 1e-9) / (double) loop;
    printf("fast_copy256s_4 %d BW %lf GB/s\n", len, (GB / elapsed));

    l2_err_i32(inout_i2, inout_iref, len);
#endif
#endif

    printf("\n");
    ////////////////////////////////////////////////// MODFF ////////////////////////////////////////////////////////////////////
    printf("MODFF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 10000) / 1235.6f;
        if (i % 4 == 0)
            inout[i] *= -1.0f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    modff_C(inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("modff_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        modff_C(inout, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("modff_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsModf_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsModf_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsModf_32f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsModf_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


    l2_err(inout2, inout_ref, len);
    l2_err(inout3, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    modf128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("modf128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        modf128f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("modf128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


    l2_err(inout2, inout_ref, len);
    l2_err(inout3, inout2_ref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    modf256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("modf256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        modf256f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("modf256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


    l2_err(inout2, inout_ref, len);
    l2_err(inout3, inout2_ref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    modf512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("modf512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        modf512f(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("modf512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout_ref, len);
    l2_err(inout3, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    modf_vec(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("modf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        modf_vec(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("modf_vec %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout2, inout_ref, len);
    l2_err(inout3, inout2_ref, len);

    // for(int i = 0; i < len; i++) printf("%f %f %f %f %f\n",inout[i],inout2[i], inout_ref[i], inout3[i],inout2_ref[i]);
#endif

    printf("\n");
    /////////////////////////////////////////////////////////// CBTRF //////////////////////////////////////////////////////////////////////////////
    printf("CBTRF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = 1.0f / ((float) i + 0.000000001f);  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cbrtf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrtf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrtf_C(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrtf_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cbrtf_C_precise(inout, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrtf_C_precise %d %lf\n", len, elapsed);

#ifdef MKL
    clock_gettime(CLOCK_REALTIME, &start);
    vsCbrt(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("MKLvsCbrt %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        vsCbrt(len, inout, inout2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("MKLvsCbrt %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cbrt128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrt128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrt128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrt128f %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);

    /*for (int i = 0; i < len; i++) {
        printf("%f %f %f\n", inout[i],inout_ref[i],inout2[i]);
    }*/
#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    cbrt256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrt256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrt256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrt256f %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);
#endif
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cbrt512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrt512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrt512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrt512f %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cbrtf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrtf_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrtf_vec(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrtf_vec %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);
#endif


    printf("\n");
    ////////////////////////////////////////////////// POL2CART2DF ////////////////////////////////////////////////////////////////////
    printf("POL2CART2DF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i;
        inout2[i] = 1.0f / ((float) i + 0.000000001f) + sqrtf((float) i);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2Df_C(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2Df_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2Df_C(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2Df_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2Df_C_precise(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2Df_C_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2Df_C_precise(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2Df_C_precise %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsPolarToCart_32f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsPolarToCart_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsPolarToCart_32f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsPolarToCart_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));


    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2D128f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2D128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2D128f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2D128f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2D128f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2D128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2D128f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2D128f_precise %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2D256f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2D256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2D256f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2D256f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2D256f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2D256f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2D256f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2D256f_precise %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2D512f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2D512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2D512f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2D512f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2D512f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2D512f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2D512f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2D512f_precise %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    pol2cart2Df_vec(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pol2cart2Df_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pol2cart2Df_vec(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pol2cart2Df_vec %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

    printf("\n");
    ////////////////////////////////////////////////// CART2POL2DF ////////////////////////////////////////////////////////////////////
    printf("CART2POL2DF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i + 1E-38f;
        inout2[i] = 1.0f / ((float) i + 0.1f) + sqrtf((float) i);
    }

    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2Df_C(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2Df_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2Df_C(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2Df_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2Df_C_precise(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2Df_C_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2Df_C_precise(inout, inout2, inout_ref, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2Df_C_precise %d %lf\n", len, elapsed);
#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsCartToPolar_32f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsCartToPolar_32f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsCartToPolar_32f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsCartToPolar_32f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

#endif

#if defined(SSE) || defined(ALTIVEC)
    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2D128f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2D128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2D128f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2D128f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2D128f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2D128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2D128f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2D128f_precise %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif
    /*for(int i = 0; i < len; i++)
         printf("%0.7f %0.7f %0.7f %0.7f %0.7f %0.7f\n",inout[i], inout2[i],inout3[i],inout_ref[i], inout4[i],inout2_ref[i]);*/
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2D256f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2D256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2D256f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2D256f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2D256f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2D256f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2D256f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2D256f_precise %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2D512f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2D512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2D512f(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2D512f %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2D512f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2D512f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2D512f_precise(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2D512f_precise %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    cart2pol2Df_vec(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cart2pol2Df_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cart2pol2Df_vec(inout, inout2, inout3, inout4, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cart2pol2Df_vec %d %lf\n", len, elapsed);

    l2_err(inout3, inout_ref, len);
    l2_err(inout4, inout2_ref, len);
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
    free(inout_s3);
    free(inout_sref);
    free(inout_ref);
    free(inout2_ref);
    free(inoutd);
    free(inoutd2);
    free(inoutd3);
    free(inoutd_ref);
    free(inoutd2_ref);

    free(inout_i1);
    free(inout_i2);
    free(inout_i3);
    free(inout_iref);

    return 0;
}
