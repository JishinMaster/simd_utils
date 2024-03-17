/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include "common_test.h"

int main(int argc, char **argv)
{
#ifdef IPP
    init_ipp();
#endif /* IPP */

#ifdef MKL
    init_mkl();
#endif /* MKL */

    if(argc < 4){
      printf("Usage simd_test : len alignment offset\n");
      return -1;
    }

    int align = atoi(argv[2]);
    int offset = atoi(argv[3]);// offset to test unaligned cases
    int len = atoi(argv[1]) + offset;
        
    float *inout = NULL, *inout2 = NULL, *inout3 = NULL, *inout4 = NULL, *inout5 = NULL;
    float *inout6 = NULL, *inout_ref = NULL, *inout2_ref = NULL;
    double *inoutd = NULL, *inoutd2 = NULL, *inoutd3 = NULL, *inoutd_ref = NULL, *inoutd2_ref = NULL;
    uint8_t *inout_u1 = NULL, *inout_u2 = NULL;
    int16_t *inout_s1 = NULL, *inout_s2 = NULL, *inout_s3 = NULL, *inout_sref = NULL;
    int32_t *inout_i1 = NULL, *inout_i2 = NULL, *inout_i3 = NULL, *inout_iref = NULL;

#ifndef USE_MALLOC
    int ret = 0;
    ret |= posix_memalign((void **) &inout, align, 2 * len * sizeof(float));
    if (inout == NULL) {
        printf("posix_memalign inout failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout2, align, 2 * len * sizeof(float));
    if (inout2 == NULL) {
        printf("posix_memalign inout2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout3, align, len * sizeof(float));
    if (inout3 == NULL) {
        printf("posix_memalign inout3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout4, align, len * sizeof(float));
    if (inout4 == NULL) {
        printf("posix_memalign inout4 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout5, align, len * sizeof(float));
    if (inout3 == NULL) {
        printf("posix_memalign inout5 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout6, align, len * sizeof(float));
    if (inout4 == NULL) {
        printf("posix_memalign inout6 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_ref, align, 2 * len * sizeof(float));
    if (inout_ref == NULL) {
        printf("posix_memalign inout_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout2_ref, align, 2 * len * sizeof(float));
    if (inout2_ref == NULL) {
        printf("posix_memalign inout2_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd, align, 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("posix_memalign inoutd failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd2, align, 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("posix_memalign inoutd2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd3, align, 2 * len * sizeof(double));
    if (inoutd == NULL) {
        printf("posix_memalign inoutd3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd_ref, align, 2 * len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("posix_memalign inoutd_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inoutd2_ref, align, 2 * len * sizeof(double));
    if (inoutd_ref == NULL) {
        printf("posix_memalign inoutd2_ref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_u1, align, len * sizeof(uint8_t));
    if (inout_u1 == NULL) {
        printf("posix_memalign inout_u1 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_u2, align, len * sizeof(uint8_t));
    if (inout_u2 == NULL) {
        printf("posix_memalign inout_u2 failed\n");
        return -1;
    }

    ret |= posix_memalign((void **) &inout_s1, align, 2 * len * sizeof(int16_t));
    if (inout_s1 == NULL) {
        printf("posix_memalign inout_s1 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_s2, align, 2 * len * sizeof(int16_t));
    if (inout_s2 == NULL) {
        printf("posix_memalign inout_s2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_s3, align, 2 * len * sizeof(int16_t));
    if (inout_s3 == NULL) {
        printf("posix_memalign inout_s3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_sref, align, 2 * len * sizeof(int16_t));
    if (inout_sref == NULL) {
        printf("posix_memalign inout_sref failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_i1, align, len * sizeof(int32_t));
    if (inout_i1 == NULL) {
        printf("posix_memalign inout_i1 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_i2, align, len * sizeof(int32_t));
    if (inout_i2 == NULL) {
        printf("posix_memalign inout_i2 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_i3, align, len * sizeof(int32_t));
    if (inout_i3 == NULL) {
        printf("posix_memalign inout_i3 failed\n");
        return -1;
    }
    ret |= posix_memalign((void **) &inout_iref, align, len * sizeof(int32_t));
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
    len = atoi(argv[1]);
    inout += offset;
    inout2 += offset;
    inout3 += offset;
    inout4 += offset;
    inout5 += offset;
    inout6 += offset;
    inout_ref += offset;
    inout2_ref += offset;
    inoutd += offset;
    inoutd2 += offset;
    inoutd3 += offset;
    inoutd_ref += offset;
    inoutd2_ref += offset;
    inout_u1 += offset;
    inout_u2 += offset;
    inout_s1 += offset;
    inout_s2 += offset;
    inout_s3 += offset;
    inout_sref += offset;
    inout_i1 += offset;
    inout_i2 += offset;
    inout_i3 += offset;
    inout_iref += offset;

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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sin128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sin128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin128f_amdlibm %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sin256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sin256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin256f_amdlibm %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sin512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sin512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sin512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sin512f_amdlibm %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    cos128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cos128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cos128f_amdlibm %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    cos256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cos256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cos256f_amdlibm %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    cos512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cos512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cos512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cos512f_amdlibm %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128f_amdlibm(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos128f_amdlibm(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos128f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256f_amdlibm(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos256f_amdlibm(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos256f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512f_amdlibm(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos512f_amdlibm(inout, inout2, inout3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sincos128d_amdlibm(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos128d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos128d_amdlibm(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos128d_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sincos256d_amdlibm(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos256d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos256d_amdlibm(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos256d_amdlibm %d %lf\n", len, elapsed);
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
    
#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512d_svml(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos512d_svml(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512d_svml %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    sincos512d_amdlibm(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("sincos512d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        sincos512d_amdlibm(inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("sincos512d_amdlibm %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
    l2_errd(inoutd2_ref, inoutd3, len);
#endif

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
    ln128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    ln128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln128f_amdlibm %d %lf\n", len, elapsed);
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
    ln256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    ln256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln256f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    ln512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    ln512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln512f_amdlibm %d %lf\n", len, elapsed);
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
    log2128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log2128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2128f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log2128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    log2128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2128f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    log2256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log2256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2256f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2256f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log2256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    log2256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2256f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    log2512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log2512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2512f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2512f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log2512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    log2512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log2512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log2512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log2512f_amdlibm %d %lf\n", len, elapsed);
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
    log10128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log10128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10128f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10128f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log10128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    log10128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10128f_amdlibm %d %lf\n", len, elapsed);
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
    log10256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log10256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10256f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10256f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10256f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log10256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    log10256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10256f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    log10512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

    clock_gettime(CLOCK_REALTIME, &start);
    log10512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10512f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10512f_precise(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10512f_precise %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    log10512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    log10512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("log10512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        log10512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("log10512f_amdlibm %d %lf\n", len, elapsed);
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
    exp128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp128f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp128f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifndef ALTIVEC
    clock_gettime(CLOCK_REALTIME, &start);
    exp128f_(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp128f_ %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp128f_(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp128f_ %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp128f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    exp128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp128f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    exp256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp256f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp256f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp256f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    exp256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp256f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    exp512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp512f(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp512f %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp512f_svml %d %lf\n", len, elapsed);
    l2_err(inout2_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    exp512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp512f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tan128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan128f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tan256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan256f_amdlibm %d %lf\n", len, elapsed);
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
    /////////////////////////////////////////////////////////// EXP //////////////////////////////////////////////////////////////////////////////
    printf("EXP\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = (double) (rand() % 8000) / 100.0;
        if (i % 2 == 0)
            inoutd[i] = -inoutd[i];
    }

    clock_gettime(CLOCK_REALTIME, &start);
    exp_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsExp_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsExp_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsExp_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsExp_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    exp128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp128d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp128d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp128d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    exp128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp128d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp128d_amdlibm %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    exp256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp256d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
    
#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp256d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp256d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    exp256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp256d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp256d_amdlibm %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    exp512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp512d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
    
#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    exp512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp512d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp512d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    exp512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("exp512d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        exp512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("exp512d_amdlibm %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

    printf("\n");
    /////////////////////////////////////////////////////////// LOG //////////////////////////////////////////////////////////////////////////////
    printf("LOG\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = (double) (rand() % 8000) / 100.0 + 0.0000001;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    ln_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln_C(inoutd, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsLn_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsLn_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsLn_64f_A53(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsLn_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef SSE
    clock_gettime(CLOCK_REALTIME, &start);
    ln128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln128d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln128d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
    
#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln128d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln128d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln128d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    ln128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln128d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln128d_amdlibm %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    ln256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln256d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln256d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
    
#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln256d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln256d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln256d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    ln256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln256d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln256d_amdlibm %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    ln512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln512d(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln512d %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
    
#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    ln512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln512d_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln512d_svml(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln512d_svml %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    ln512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ln512d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ln512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ln512d_amdlibm %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif

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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tan128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan128d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan128d_amdlibm %d %lf\n", len, elapsed);

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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tan256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan256d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan256d_amdlibm %d %lf\n", len, elapsed);

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

/*#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tan512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tan512d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tan512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tan512d_amdlibm %d %lf\n", len, elapsed);

    l2_errd(inoutd_ref, inoutd2, len);
#endif*/

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

/*#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tanh128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh128f_amdlibm %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif*/

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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tanh256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh256f_amdlibm %d %lf\n", len, elapsed);

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

/*#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    tanh512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("tanh512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        tanh512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("tanh512f_amdlibm %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif*/

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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    cosh128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh128f_amdlibm %d %lf\n", len, elapsed);

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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    cosh256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh256f_amdlibm %d %lf\n", len, elapsed);

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

/*#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    cosh512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cosh512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cosh512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cosh512f_amdlibm %d %lf\n", len, elapsed);

    l2_err(inout_ref, inout2, len);
#endif*/

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


    inout -= offset;
    inout2 -= offset;
    inout3 -= offset;
    inout4 -= offset;
    inout5 -= offset;
    inout6 -= offset;
    inout_ref -= offset;
    inout2_ref -= offset;
    inoutd -= offset;
    inoutd2 -= offset;
    inoutd3 -= offset;
    inoutd_ref -= offset;
    inoutd2_ref -= offset;
    inout_u1 -= offset;
    inout_u2 -= offset;
    inout_s1 -= offset;
    inout_s2 -= offset;
    inout_s3 -= offset;
    inout_sref -= offset;
    inout_i1 -= offset;
    inout_i2 -= offset;
    inout_i3 -= offset;
    inout_iref -= offset;

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
