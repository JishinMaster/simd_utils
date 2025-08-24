/*
 * Project : SIMD_Utils
 * Version : 0.2.6
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    asin128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin128f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    asin256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin256f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    asin512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin512f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#if defined(RISCV) || defined(SVE2)
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

/*#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    asin128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin128d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin128d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin128d_amdlibm %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif*/

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

/*#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    asin256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin256d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin256d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin256d_amdlibm %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif*/

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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    asin512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asin512d_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asin512d_amdlibm(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asin512d_amdlibm %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
#endif

#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    asind_vec(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("asind_vec %d %lf\n", len, elapsed);

    /*clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        asind_vec(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("asind_vec %d %lf\n", len, elapsed);*/
    l2_errd(inoutd_ref, inoutd2, len);
    
	/*for(int i = 0; i < len; i++){
        printf("%lf || %g %g || %g\n",inoutd[i], inoutd_ref[i],inoutd2[i],
		fabs(inoutd_ref[i]-inoutd2[i]));
    }*/	
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    atan128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("atan128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan128f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    atan256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("atan256f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan256f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan256f_amdlibm %d %lf\n", len, elapsed);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    atan512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan512f_svml %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    atan512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("atan512f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan512f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("atan512f_amdlibm %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2, len);
#endif

#endif

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    atand_vec(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atand_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atand_vec(inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atand_vec %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2, len);
	
    /*for(int i = 0; i < len; i++){
        printf("%lf || %lf %lf || %g\n",inoutd[i], inoutd_ref[i],inoutd2[i],
		fabs(inoutd_ref[i]-inoutd2[i]));
    }*/
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


#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    atan2d_vec(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2d_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2d_vec(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2d_vec %d %lf\n", len, elapsed);
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

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    atan2d_interleaved_vec((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("atan2d_interleaved_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        atan2d_interleaved_vec((complex64_t *) inoutd, inoutd2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("atan2d_interleaved_vec %d %lf\n", len, elapsed);
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    cplxtoreald_vec((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("cplxtoreald_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cplxtoreald_vec((complex64_t *) inoutd, inoutd2, inoutd3, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cplxtoreald_vec %d %lf\n", len, elapsed);
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

#if defined(RISCV)
    clock_gettime(CLOCK_REALTIME, &start);
    realtocplxd_vec(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("realtocplxd_vec %d %lf\n", len, elapsed);
    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        realtocplxd_vec(inoutd2, inoutd3, (complex64_t *) inoutd, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3 / (double) loop;
    printf("realtocplxd_vec %d %lf\n", len, elapsed);
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
    convertFloat32ToU8_C(inout, inout_u2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_C(inout, inout_u2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f8u_Sfs(inout, inout_u1, len, ippRndFinancial, 4);  // ippRndFinancial ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32f8u_Sfs(inout, inout_u1, len, ippRndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_32f8u_Sfs %d %lf\n", len, elapsed);
    l2_err_u8(inout_u1, inout_u2, len);

    /*printf("Scale : %g\n", 1.0f / (float) (1 << 4));
    for(int i = 0; i < len; i++)
        printf("IPP : %g %x %x\n" ,inout[i], inout_u1[i],inout_u2[i]);*/


#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_128(inout, inout_u1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_128(inout, inout_u1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_128 %d %lf\n", len, elapsed);
    l2_err_u8(inout_u1, inout_u2, len);
#endif
    for(int i = 0; i < len; i++)
        if(inout_u1[i] != inout_u2[i])
          printf("SSE : %d %g %g %x %x\n" ,i, inout[i], inout[i]/(1<<4),inout_u1[i],inout_u2[i]);
#endif
    
#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_256(inout, inout_u1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_256(inout, inout_u1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_256 %d %lf\n", len, elapsed);

    l2_err_u8(inout_u1, inout_u2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_512(inout, inout_u1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_512(inout, inout_u1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU8_512 %d %lf\n", len, elapsed);

    l2_err_u8(inout_u1, inout_u2, len);
#endif

#ifdef RISCV
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU8_vec(inout, inout_u1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU8_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU8_vec(inout, inout_u1, len, RndFinancial, 4);
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
        if(i%2==0) inout[i] = - inout[i];
        inout_s1[i] = 0;
        inout_s2[i] = 0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_C(inout, inout_s1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_C(inout, inout_s1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f16s_Sfs(inout, inout_s2, len, ippRndFinancial, 4);  // ippRndFinancial ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f16s_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32f16s_Sfs(inout, inout_s2, len, ippRndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_32f16s_Sfs %d %lf\n", len, elapsed);
    l2_err_i16(inout_s1, inout_s2, len);

    for(int i = 0; i < len; i++)
        if(inout_s1[i] != inout_s2[i])
          printf("IPP : %d %g %g %d %d\n" ,i, inout[i], inout[i]/(1<<4),inout_s1[i],inout_s2[i]);
#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_128(inout, inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_128(inout, inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_128 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);

    for(int i = 0; i < len; i++)
        if(inout_s1[i] != inout_s2[i])
          printf("SSE : %d %g %g %d %d\n" ,i, inout[i], inout[i]/(1<<4),inout_s1[i],inout_s2[i]);
#endif
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_256(inout, inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_256(inout, inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_256 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_512(inout, inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_512(inout, inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToI16_512 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#if defined(RISCV) || defined(SVE2)
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToI16_vec(inout, inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToI16_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToI16_vec(inout, inout_s2, len, RndFinancial, 4);
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
    convertFloat32ToU16_C(inout, (uint16_t *) inout_s1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_C(inout, (uint16_t *) inout_s1, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32f16u_Sfs(inout, (uint16_t *) inout_s2, len, ippRndFinancial, 4);  // ippRndFinancial ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32f16u_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32f16u_Sfs(inout, (uint16_t *) inout_s2, len, ippRndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_32f16u_Sfs %d %lf\n", len, elapsed);
    l2_err_i16(inout_s1, inout_s2, len);

    /*printf("Scale : %g\n", 1.0f / (float) (1 << 4));
    for(int i = 0; i < len; i++)
        printf("IPP : %g %d %d\n" ,inout[i], inout_s1[i],inout_s2[i]);*/
#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_128(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_128(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_128 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif
    for(int i = 0; i < len; i++)
        if(inout_s1[i] != inout_s2[i])
          printf("SSE : %d %g %g %d %d\n" ,i, inout[i], inout[i]/(1<<4),inout_s1[i],inout_s2[i]);
#endif

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_256(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_256(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_256 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_512(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_512(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertFloat32ToU16_512 %d %lf\n", len, elapsed);

    l2_err_i16(inout_s1, inout_s2, len);
#endif

#if defined(RISCV) || defined(SVE2)
    clock_gettime(CLOCK_REALTIME, &start);
    convertFloat32ToU16_vec(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertFloat32ToU16_vec %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertFloat32ToU16_vec(inout, (uint16_t *) inout_s2, len, RndFinancial, 4);
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
    ippsConvert_16s32f_Sfs(inout_s1, inout_ref, len, 4);  // ippRndFinancial ippRndZero ippRndFinancial
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

#if defined(RISCV) || defined(SVE2)
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


    printf("\n");
    /////////////////////////////////////////////////////////// CONVERTINT32_FLOAT32 //////////////////////////////////////////////////////////////////////////////
    printf("CONVERTINT32_FLOAT32\n");

    for (int i = 0; i < len; i++)
        inout_i1[i] = -len/2 + i;

    clock_gettime(CLOCK_REALTIME, &start);
    convertInt32ToFloat32_C(inout_i1, inout2_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt32ToFloat32_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt32ToFloat32_C(inout_i1, inout2_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt32ToFloat32_C %d %lf\n", len, elapsed);

#ifdef IPP
    clock_gettime(CLOCK_REALTIME, &start);
    ippsConvert_32s32f_Sfs(inout_i1, inout_ref, len, 4);  // ippRndFinancial ippRndZero ippRndFinancial
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("ippsConvert_32s32f_Sfs %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsConvert_32s32f_Sfs(inout_i1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsConvert_32s32f_Sfs %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(SSE) || defined(ALTIVEC)
#ifndef __MACH__
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt32ToFloat32_128(inout_i1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt32ToFloat32_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt32ToFloat32_128(inout_i1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt32ToFloat32_128 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif
#endif

#ifdef AVX
#ifdef __AVX2__
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt32ToFloat32_256(inout_i1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt32ToFloat32_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt32ToFloat32_256(inout_i1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt32ToFloat32_256 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif
#endif

#ifdef AVX512
    clock_gettime(CLOCK_REALTIME, &start);
    convertInt32ToFloat32_512(inout_i1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("convertInt32ToFloat32_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        convertInt32ToFloat32_512(inout_i1, inout_ref, len, 4);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("convertInt32ToFloat32_512 %d %lf\n", len, elapsed);
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

#if defined(RISCV) || defined(SVE2)
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
