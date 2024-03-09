
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

//Relative Error
// For complex arrays this is not the good way to compute the error
// but it should give a good enough idea of the precision
float l2_err(float *test, float *ref, int len)
{
    float l2_rel_err = 0.0f;
    float sum = 0.0f;
    int sup1ulps = 0;
    float sup1ulps_percent = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_rel_err += (ref[i] - test[i]) * (ref[i] - test[i]);
        sum += ref[i]*ref[i];
        int32_t dist = ulpsDistance32(ref[i], test[i]);
        if (dist > 1)
            sup1ulps++;
    }

    sup1ulps_percent = (float) sup1ulps / (float) len * 100.0f;
    l2_rel_err =  sqrtf(l2_rel_err)/sqrtf(sum);
    printf("L2 REL ERR %0.9g SUP_1ULPS %2.4g %% \n", l2_rel_err, sup1ulps_percent);
    return l2_rel_err;
}

double l2_errd(double *test, double *ref, int len)
{
    double l2_rel_err = 0.0;
    double sum = 0.0;
    int sup1ulps = 0;
    float sup1ulps_percent = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_rel_err += (ref[i] - test[i]) * (ref[i] - test[i]);
        sum += ref[i]*ref[i];
        int64_t dist = ulpsDistance64(ref[i], test[i]);
        if (dist > 1)
            sup1ulps++;
    }

    sup1ulps_percent = (float) sup1ulps / (float) len * 100.0f;
    l2_rel_err =  sqrt(l2_rel_err)/sqrt(sum);
    printf("L2 REL ERR %0.18g SUP_1ULPS %2.4g %% \n", l2_rel_err, sup1ulps_percent);
    return l2_rel_err;
}

float l2_err_u8(uint8_t *test, uint8_t *ref, int len)
{
    float l2_rel_err = 0.0f;
    float sum = 0.0;
    
    for (int i = 0; i < len; i++) {
        l2_rel_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
        sum +=  (float) (ref[i]*ref[i]);
    }

    l2_rel_err =  sqrtf(l2_rel_err)/sqrtf(sum);
    printf("L2 REL ERR %0.9g\n", l2_rel_err);
    return l2_rel_err;
}

float l2_err_i16(int16_t *test, int16_t *ref, int len)
{
    float l2_rel_err = 0.0f;
    float sum = 0.0;

    for (int i = 0; i < len; i++) {
        l2_rel_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
        sum +=  (float) (ref[i]*ref[i]);
    }

    l2_rel_err =  sqrtf(l2_rel_err)/sqrtf(sum);
    printf("L2 REL ERR %0.9g\n", l2_rel_err);
    return l2_rel_err;
}

#warning "TODO : check integer overflow"
float l2_err_i32(int32_t *test, int32_t *ref, int len)
{
    float l2_rel_err = 0.0f;
    //float sum = 0.0;

    for (int i = 0; i < len; i++) {
        l2_rel_err += (float) (ref[i] - test[i]) * (ref[i] - test[i]);
        //sum +=  (float) (ref[i]*ref[i]);
    }

    l2_rel_err =  sqrtf(l2_rel_err);///sqrtf(sum);
    printf("L2 REL ERR %0.9g\n", l2_rel_err);
    return l2_rel_err;
}

#ifdef IPP
void init_ipp(void){
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
}
#endif

#ifdef MKL
    void init_mkl(void){
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
    }
#endif /* MKL */
