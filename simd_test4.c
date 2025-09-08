/*
 * Project : SIMD_Utils
 * Version : 0.2.6
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include "common_test.h"

#include <strings.h>

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
    double flops =  2 * len;

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

#if defined(RISCV) || defined(SVE2)
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cbrt128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrt128f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrt128f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrt128f_svml %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);
#endif

#ifdef AMDLIBM
    clock_gettime(CLOCK_REALTIME, &start);
    cbrt128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrt128f_amdlibm %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrt128f_amdlibm(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrt128f_amdlibm %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);
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

#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cbrt256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrt256f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrt256f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrt256f_svml %d %lf\n", len, elapsed);

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
    
#ifdef ICC
    clock_gettime(CLOCK_REALTIME, &start);
    cbrt512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("cbrt512f_svml %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        cbrt512f_svml(inout, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("cbrt512f_svml %d %lf\n", len, elapsed);

    l2_err(inout2, inout_ref, len);
#endif

#endif

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

#if defined(RISCV) || defined(SVE2)
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

    printf("\n");
    ////////////////////////////////////////////////// FP32TOFP16 ////////////////////////////////////////////////////////////////////
    printf("FP32TOFP16\n");

	for(int i = 0; i < len; i++)
		inout[i]= (float)(-len)*0.0073456f + 0.0123456789f*(float)i;	

    clock_gettime(CLOCK_REALTIME, &start);
    fp32tofp16_C(inout,  (uint16_t *)inout_sref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("fp32tofp16_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fp32tofp16_C(inout,  (uint16_t *)inout_sref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fp32tofp16_C %d %lf\n", len, elapsed);

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    fp32tofp16128(inout,  (uint16_t *)inout_s1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("fp32tofp16128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fp32tofp16128(inout,  (uint16_t *)inout_s1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fp32tofp16128 %d %lf\n", len, elapsed);

    for(int i = 0; i < len; i++){
		if(inout_s1[i] != inout_sref[i])
			printf("error at %d : %08x != %08x\n",i,inout_s1[i],inout_sref[i]);
	}

    clock_gettime(CLOCK_REALTIME, &start);
    fp32tofp16256(inout,  (uint16_t *)inout_s1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("fp32tofp16256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fp32tofp16256(inout,  (uint16_t *)inout_s1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fp32tofp16256 %d %lf\n", len, elapsed);

    for(int i = 0; i < len; i++){
		if(inout_s1[i] != inout_sref[i])
			printf("error at %d : %08x != %08x\n",i,inout_s1[i],inout_sref[i]);
	}
#endif


    printf("\n");
    ////////////////////////////////////////////////// FP16TOFP32 ////////////////////////////////////////////////////////////////////
    printf("FP16TOFP32\n");

    clock_gettime(CLOCK_REALTIME, &start);
    fp16tofp32_C( (uint16_t *)inout_sref, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("fp16tofp32_C %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fp16tofp32_C( (uint16_t *)inout_sref, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fp16tofp32_C %d %lf\n", len, elapsed);

#ifdef AVX
    clock_gettime(CLOCK_REALTIME, &start);
    fp16tofp32128( (uint16_t *)inout_sref, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("fp16tofp32128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fp16tofp32128( (uint16_t *)inout_sref, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fp16tofp32128 %d %lf\n", len, elapsed);

	//we convert format so it makes sense to directly compare float values
    for(int i = 0; i < len; i++){
		if(inout2[i] != inout_ref[i])
			printf("error at %d : %g != %g\n",i,inout2[i],inout_ref[i]);
	}

    clock_gettime(CLOCK_REALTIME, &start);
    fp16tofp32256( (uint16_t *)inout_sref, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("fp16tofp32256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        fp16tofp32256( (uint16_t *)inout_sref, inout2, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("fp16tofp32256 %d %lf\n", len, elapsed);

	//we convert format so it makes sense to directly compare float values
    for(int i = 0; i < len; i++){
		if(inout2[i] != inout_ref[i])
			printf("error at %d : %g != %g\n",i,inout2[i],inout_ref[i]);
	}
#endif

    printf("\n");
    ////////////////////////////////////////////////// POWF ////////////////////////////////////////////////////////////////////
    printf("POWF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i + 1E-38f;
        inout2[i] = (float)i * 0.00007f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    powf_c(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powf_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powf_c(inout, inout2, inout_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powf_c %d %lf\n", len, elapsed);

#if defined(IPP)
    clock_gettime(CLOCK_REALTIME, &start);
    ippsPow_32f_A24(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("ippsPow_32f_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsPow_32f_A24(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsPow_32f_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    pow128f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pow128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pow128f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pow128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);

    clock_gettime(CLOCK_REALTIME, &start);
    pow128f_precise(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pow128f_precise %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pow128f_precise(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pow128f_precise %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    pow256f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pow256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pow256f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pow256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    pow512f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pow512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pow512f(inout, inout2, inout2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pow512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

    printf("\n");
    ////////////////////////////////////////////////// POW ////////////////////////////////////////////////////////////////////
    printf("POW\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = (double) i + 1E-38f;
        inoutd2[i] = (double)i * 0.00007;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    powd_c(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powd_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powd_c(inoutd, inoutd2, inoutd_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powd_c %d %lf\n", len, elapsed);

#if defined(IPP)
    clock_gettime(CLOCK_REALTIME, &start);
    ippsPow_64f_A53(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("ippsPow_64f_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsPow_64f_A53(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsPow_64f_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    pow128d(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pow128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pow128d(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pow128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    pow256d(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pow256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pow256d(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pow256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    pow512d(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("pow512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        pow512d(inoutd, inoutd2, inoutd2_ref, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("pow512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif


    printf("\n");
    ////////////////////////////////////////////////// POWCPLXF ////////////////////////////////////////////////////////////////////
    printf("POWCPLXF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) i + 1E-38f;
        inout2[i] = (float)i * 0.00007f;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    powcplxf_c((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplxf_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplxf_c((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplxf_c %d %lf\n", len, elapsed);

#if defined(IPP)
    clock_gettime(CLOCK_REALTIME, &start);
    ippsPow_32fc_A24((Ipp32fc*)inout, (Ipp32fc*)inout2, (Ipp32fc*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("ippsPow_32fc_A24 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsPow_32fc_A24((Ipp32fc*)inout, (Ipp32fc*)inout2, (Ipp32fc*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsPow_32fc_A24 %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    powcplx128f((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplx128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplx128f((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplx128f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    powcplx256f((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplx256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplx256f((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplx256f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    powcplx512f((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplx512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplx512f((complex32_t*)inout, (complex32_t*)inout2, (complex32_t*)inout2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplx512f %d %lf\n", len, elapsed);
    l2_err(inout_ref, inout2_ref, len);
#endif


    printf("\n");
    ////////////////////////////////////////////////// POW ////////////////////////////////////////////////////////////////////
    printf("POW\n");

    for (int i = 0; i < len; i++) {
        inoutd[i] = (double) i + 1E-38f;
        inoutd2[i] = (double)i * 0.00007;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    powcplxd_c((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplxd_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplxd_c((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplxd_c %d %lf\n", len, elapsed);

#if defined(IPP)
    clock_gettime(CLOCK_REALTIME, &start);
    ippsPow_64fc_A53((const Ipp64fc*)inoutd, (const Ipp64fc*)inoutd2,(Ipp64fc*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("ippsPow_64fc_A53 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        ippsPow_64fc_A53((const Ipp64fc*)inoutd, (const Ipp64fc*)inoutd2,(Ipp64fc*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("ippsPow_64fc_A53 %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    powcplx128d((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplx128d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplx128d((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplx128d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    powcplx256d((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplx256d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplx256d((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplx256d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    powcplx512d((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("powcplx512d %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        powcplx512d((complex64_t*)inoutd, (complex64_t*)inoutd2, (complex64_t*)inoutd2_ref, len/2);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("powcplx512d %d %lf\n", len, elapsed);
    l2_errd(inoutd_ref, inoutd2_ref, len);
#endif

    printf("\n");
    ////////////////////////////////////////////////// STRNLEN_S ////////////////////////////////////////////////////////////////////
    printf("STRNLEN_S\n");

	for(int i = 0; i < len; i++)
		inout_u1[i] =  'a';
	inout_u1[len-3] =  '\0';

	size_t lengthstr_ref;
    clock_gettime(CLOCK_REALTIME, &start);
    lengthstr_ref = strnlen(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("strnlen_s %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        lengthstr_ref = strnlen(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("strnlen_s %d %lf\n", len, elapsed);

#if defined(SSE)
	size_t lengthstr;
    clock_gettime(CLOCK_REALTIME, &start);
    lengthstr = strnlen_s_128(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("strnlen_s_128 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        lengthstr = strnlen_s_128(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("strnlen_s_128 %d %lf\n", len, elapsed);
	printf("%lu %lu\n",lengthstr_ref, lengthstr_ref);
#endif

#if defined(AVX)
#ifdef __AVX2__
	lengthstr;
    clock_gettime(CLOCK_REALTIME, &start);
    lengthstr = strnlen_s_256(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("strnlen_s_256 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        lengthstr = strnlen_s_256(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("strnlen_s_256 %d %lf\n", len, elapsed);
	printf("%lu %lu\n",lengthstr_ref, lengthstr_ref);
#endif
#endif

#if defined(AVX512)
	lengthstr;
    clock_gettime(CLOCK_REALTIME, &start);
    lengthstr = strnlen_s_512(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("strnlen_s_512 %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
        lengthstr = strnlen_s_512(inout_u1, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("strnlen_s_512 %d %lf\n", len, elapsed);
	printf("%lu %lu\n",lengthstr_ref, lengthstr_ref);
#endif

    printf("\n");
    ////////////////////////////////////////////////// MODFF ////////////////////////////////////////////////////////////////////
    printf("CHECK NANINFF\n");

    for (int i = 0; i < len; i++) {
        inout[i] = (float) (rand() % 10000) / 1235.6f;
        if (i % 4 == 0)
            inout[i] *= -1.0f;
    }

	inout[len/2] /= 0.0f;
	inout[len/4] = NAN;
    clock_gettime(CLOCK_REALTIME, &start);
	int32_t nan, inf;
    checkNanInff_c(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3);
    printf("checkNanInff_c %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
		checkNanInff_c(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("checkNanInff_c %d %lf\n", len, elapsed);
	printf(" nan %08x inf %08x\n",nan,inf);

#if defined(SSE)
    clock_gettime(CLOCK_REALTIME, &start);
    checkNanInf128f(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("checkNanInf128f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
		checkNanInf128f(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("checkNanInf128f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

	printf(" nan %08x inf %08x\n",nan,inf);
#endif

#if defined(AVX)
    clock_gettime(CLOCK_REALTIME, &start);
    checkNanInf256f(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("checkNanInf256f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
		checkNanInf256f(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("checkNanInf256f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

	printf(" nan %08x inf %08x\n",nan,inf);
#endif

#if defined(AVX512)
    clock_gettime(CLOCK_REALTIME, &start);
    checkNanInf512f(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);

    elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
    printf("checkNanInf512f %d %lf\n", len, elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    for (l = 0; l < loop; l++)
		checkNanInf512f(inout,&nan, &inf, len);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed = ((stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3) / (double) loop;
    printf("checkNanInf512f %d %lf %0.3lf GFlops/s\n", len, elapsed, flops / (elapsed * 1e3));

	printf(" nan %08x inf %08x\n",nan,inf);
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
