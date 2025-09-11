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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if (defined(RISCV) && (ELEN >= 64)) || defined(SVE2)
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

#if defined (RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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
      printf("%g %g %g %g ||| %g %g || %g %g\n", inout[i], inout[i+1],inout2[i], inout2[i+1],\
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

#if defined (RISCV) || defined(SVE2)
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
        inout[i] = (float) (2*i) / 300.0f;  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
		inout2[i] = (float) (2*i+1) /300.0f;
        inout3[i] = (float) (2*i) /  (-127.577f);  // printf("%f %f %f\n",inout[i],inout2[i],inout2_ref[i]);
        inout4[i] = (float) (2*i+1) /  (-127.577f);
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
			
    /*for(int i = 0; i < len; i++){
      printf("%g %g %g %g ||| %g %g || %g %g\n", inout[i], inout2[i],inout3[i], inout4[i],\
              inout_ref[i], inout5[i], inout2_ref[i], inout6[i]);
    }*/	
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

#if defined (RISCV) || defined(SVE2)
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

#if defined (RISCV) || defined(SVE2)
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

#if defined (RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined (RISCV) || defined(SVE2)
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

#if defined (RISCV) || defined(SVE2)
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
