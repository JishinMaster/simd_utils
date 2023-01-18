# simd_utils

A header only library implementing common mathematical functions using SIMD intrinsics.
This library is C/C++ compatible (tested with GCC7.5/9.3/10.2/11.3/12.0, clang 9 and icc 2021).

Thanks to Julien Pommier and Giovanni Garberoglio for their work on sin,cos,log, and exp functions in SSE, AVX, and NEON intrinsics.
Thanks to the DLTcollab team for their work on sse2neon.

## What is SIMD Utils?

The purpose of this library is to give an open-source implementation of SIMD optimized commonly used algorithms, such as type conversion (float32, float64, uint16, ...), trigonometry (sin, cos, atan, ...), log/exp, min/max, and other functions.
Its API was thought as a simple replacement for Intel IPP/MKL libraries.
Some of the functions are vectorised version of the cephes maths library (https://www.netlib.org/cephes/)

## Why use SIMD Utils?

- It's free
- It's open source
- It works on a wide range of machines, including Arm 32bits (with NEON) and 64bits

## Targets

Supported targets are : 
- SSE (SSE4.X mostly)
- AVX (AVX and AVX2)
- AVX512
- ARM Neon (through sse2neon plus some optimized functions).
- RISC-V Vector extension 1.0
- PowerPC Alitivec (no double precision suppport)

128 bit functions (SSE, NEON, ALTIVEC) are name function128type, such as asin128f, which computes the arcsinus function on an float32 array. Float64 functions have the "d" suffix.
256 bit functions (AVX/AVX2) have 256 instead of 128 in their name, such as asin256f.
256 bit functions (AVX512) have 512 instead of 128 in their name, such as cos512f.
Vector functions (RISCV) for which the SIMD length makes less sense, are name functionType_vec, such as subs_vec, which substract an int32 array from and other one.

The project has been tested on :
- Intel Atom
- Intel Ivy Bridge Core-i7
- Intel Skylake Core-i7
- Intel Cannonlake Core-i7
- Intel SDE (emulator) for AVX-512
- Qemu 5.X (emulator) )for arm/aarch64, ppc and riscv
- Cortex-a53 (Raspberry Pi 3B)
- Cortex-a9 (ZYBO)
- PowerPC G5 (iMac G5)
- RISCV Ox64 (C906 core)

## Building

To build the project you will need the sse_mathfun.h, avx_mathfun.h and neon_mathun.h headers available here http://gruntthepeon.free.fr/ssemath/, and there http://software-lisc.fbk.eu/avx_mathfun/
This project also uses a forked version of sse2neon (https://github.com/DLTcollab/sse2neon) adding functions such as double precision and Fused Multiple Add.

Simply include simd_utils.h in your C/C++ file, and compile with : 
- SSE support : gcc -DSSE -msse4.2 -c file.c -I .
- AVX support : gcc -DSSE -DAVX -mavx2  -c file.c -I .
- AVX512 support : gcc -DSSE -DAVX -DAVX512 -march=skylake-avx512 -mprefer-vector-width=512 -c file.c -I .
- ARM V7 NEON support : arm-none-linux-gnueabihf-gcc -march=armv7-a -mfpu=neon -DARM -DSSE -flax-vector-conversions -c file.c -I .
- ARM V8 NEON support : aarch64-linux-gnu-gcc -DARM -DFMA -DSSE -flax-vector-conversions -c file.c -I .
- RISCV support : riscv64-unknown-linux-gnu-gcc -DRISCV -march=rv64gcv -c file.c-I .
- ALTIVEC support : powerpc64-linux-gnu-gcc -DALTIVEC -DFMA -maltivec -flax-vector-conversions -c file.c -I .

For FMA support you need to add -DFMA and -mfma to x86 targets, and -DFMA to Armv8 targets.
For ARMV7 targets, you could also add -DSSE2NEON_PRECISE_SQRT for improved accuracy with sqrt and rsqrt
For X86 targets with ICC compiler, simply add -DICC to activate Intel SVML intrinsics.
Altivec support is intended mostly for older Big Endian PowerPC. Newer Little Endian might benefit from a direct conversion from SSE similar to sse2neon.

## OpenCL (experimental)

The same approach is applied to OpenCL kernels as an experiment, focused on GPUs, but other OpenCL devices may work.
At the moment only some functions are supported (log, exp, sincos, tan, atan, atan2, asin, sqrt), based on the cephes library, which seems to be faster that the OpenCL native functions (tested on Intel GPU with beignet 1.3) 
To try it out, simply use : 
- gcc -DSSE -msse4.2 -march=native simd_test_opencl.c -lOpenCL -lrt -lm (add -DSIMPLE_BUFFERS for CPU devices)


## Supported Functions 

SSE/NEON are 128bits wide. SSE functions use up to SSE4.2 features.
Some functions are directly coded using NEON intrinsics (for performance reasons), but most functions translate SSE code to NEON using sse2neon header.
Some AVX functions, such as integer ones, require AVX2. The 256 bit integer functions are emulated using SSE for some floating point functions if AVX2 is unavailable.
Altivec implemented functions are indicated with "(a)".

The following table is a work in progress, "?" means there is not yet an implemented function (or a directly equivalent Intel IPP function) :

| SSE/NEON/ALTIVEC (X=128), AVX (X=256), AVX512 (X=512) |           C_REF             |              IPP_REF           |            RISCV              |
|-------------------------------------------------------|-----------------------------|--------------------------------|-------------------------------|
|                                                       |                             |                                |                               |
| log10_Xf/precise (a)                                  | log10f_C                    | ippsLog10_32f_A24              | log10f_vec                    |
| log2_Xf/precise  (a)                                  | log2f_C                     |                                | log2f_vec                     |
| ln_Xf  (a)                                            | lnf_C                       | ippsLn_32f_A24                 | lnf_vec                       |
| exp_Xf (a)                                            | expf_C                      | ippsExp_32f_A24                | expf_vec                      |
| cbrtXf  (a)                                           | cbrtf_C                     | ?                              | cbrtf_vec                     |
| fabsXf (a)                                            | fabsf_C                     | ippsAbs_32f                    | fabsf_vec                     |
| setXf  (a)                                            | setf_C                      | ippsSet_32f                    | setf_vec                      |
| zeroXf (a)                                            | zerof_C                     | ippsZero_32f                   | zerof_vec                     |
| copyXf (a)                                            | copyf_C                     | ippsCopy_32f                   | copyf_vec                     |
| addXf  (a)                                            | addf_c                      | ippsAdd_32f                    | addf_vec                      |
| mulXf  (a)                                            | mulf_C                      | ippsMul_32f                    | mulf_vec                      |
| subXf  (a)                                            | subf_c                      | ippsSub_32f                    | subf_vec                      |
| addcXf (a)                                            | addcf_C                     | ippsAddC_32f                   | addcf_vec                     |
| mulcXf (a)                                            | mulcf_C                     | ippsMulC_32f                   | mulcf_vec                     |
| muladdXf                                              | muladdf_C                   | ?                              | muladdf_vec                   |
| mulcaddXf                                             | mulcaddf_C                  | ?                              | mulcaddf_vec                  |
| mulcaddcXf                                            | mulcaddcf_C                 | ?                              | mulcaddcf_vec                 |
| muladdcXf                                             | muladdcf_C                  | ?                              | muladdcf_vec                  |
| divXf  (a)                                            | divf_C                      | ippsDiv_32f_A24                | divf_vec                      |
| dotXf  (a)                                            | dotf_C                      | ippsDotProd_32f                | dotf_vec                      |
| dotcXf (a)                                            | dotcf_C                     | ippsDotProd_32fc               | dotcf_vec                     |
| vectorSlopeXf    (a)                                  | vectorSlopef_C              | ippsVectorSlope_32f            | vectorSlopef_vec              |
| convertFloat32ToU8_X  (a)                             | convertFloat32ToU8_C        | ippsConvert_32f8u_Sfs          | convertFloat32ToU8_vec        |
| convertFloat32ToU16_X (a)                             | convertFloat32ToI16_C       | ippsConvert_32f16u_Sfs         | convertFloat32ToU16_vec       |
| convertFloat32ToI16_X  (a)                            | convertFloat32ToI16_C       | ippsConvert_32f16s_Sfs         | convertFloat32ToI16_vec       |
| convertInt16ToFloat32_X  (a)                          | convertInt16ToFloat32_C     | ippsConvert_16s32f_Sfs         | convertInt16ToFloat32_vec     |
| cplxtorealXf   (a)                                    | cplxtorealf_C               | ippsCplxToReal_32fc            | cplxtorealf_vec               |
| realtocplxXf   (a)                                    | realtocplx_C                | ippsRealToCplx_32f             | realtocplxf_vec               |
| convertX_64f32f                                       | convert_64f32f_C            | ippsConvert_64f32f             | convert_64f32f_vec            |
| convertX_32f64f                                       | convert_32f64f_C            | ippsConvert_32f64f             | convert_32f64f_vec            |
| flipXf   (a)                                          | flipf_C                     | ippsFlip_32f                   | flipf_vec                     |
| maxeveryXf  (a)                                       | maxeveryf_c                 | ippsMaxEvery_32f               | maxeveryf_vec                 |
| mineveryXf  (a)                                       | mineveryf_c                 | ippsMinEvery_32f               | mineveryf_vec                 |
| minmaxXf    (a)                                       | minmaxf_c                   | ippsMinMax_32f                 | minmaxf_vec                   |
| thresholdX_gt_f       (a)                             | threshold_gt_f_C            | ippsThreshold_GT_32f           | threshold_gt_f_vec            |
| thresholdX_gtabs_f    (a)                             | threshold_gtabs_f_C         | ippsThreshold_GTAbs_32f        | threshold_gtabs_f_vec         |
| thresholdX_lt_f       (a)                             | threshold_lt_f_C            | ippsThreshold_LT_32f           | threshold_lt_f_vec            |
| thresholdX_ltabs_f    (a)                             | threshold_ltabs_f_C         | ippsThreshold_LTAbs_32f        | threshold_ltabs_f_vec         |
| thresholdX_ltval_gtval_f (a)                          | threshold_ltval_gtval_f_C   | ippsThreshold_LTValGTVal_32f   | threshold_ltval_gtval_f_vec   |
| sinXf                                                 | sinf_C                      | ippsSin_32f_A24                | sinf_vec                      |
| cosXf                                                 | cosf_C                      | ippsCos_32f_A24                | cosf_vec                      |
| sincosXf (a)                                          | sincosf_C                   | ippsSinCos_32f_A24             | sincosf_vec                   |
| sincosXf_interleaved (a)                              | sincosf_C_interleaved       | ippsCIS_32fc_A24               | sincosf_interleaved_vec       |
| coshXf  (a)                                           | coshf_C                     | ippsCosh_32f_A24               | coshf_vec                     |
| sinhXf  (a)                                           | sinhf_C                     | ippsSinh_32f_A24               | sinhf_vec                     |
| acoshXf (a)                                           | acoshf_C                    | ippsAcosh_32f_A24              | acoshf_vec                    |
| asinhXf (a)                                           | asinhf_C                    | ippsAsinh_32f_A24              | asinhf_vec                    |
| atanhXf (a)                                           | atanhf_C                    | ippsAtanh_32f_A24              | atanh_vec                     |
| atanXf  (a)                                           | atanf_C                     | ippsAtan_32f_A24               | atanf_vec                     |
| atan2Xf (a)                                           | atan2f_C                    | ippsAtan2_32f_A24              | atan2f_vec                    |
| atan2Xf_interleaved (a)                               | atan2f_interleaved_C        | ?                              | atan2f_interleaved_vec        |
| asinXf (a)                                            | asinf_C                     | ippsAsin_32f_A24               | asinf_vec                     |
| tanhXf (a)                                            | tanhf_C                     | ippsTanh_32f_A24               | tanhf_vec                     |
| tanXf  (a)                                            | tanf_C                      | ippsTan_32f_A24                | tanf_vec                      |
| magnitudeXf_split  (a)                                | magnitudef_C_split          | ippsMagnitude_32f              | magnitudef_split_vec          |
| powerspectXf_split (a)                                | powerspectf_C_split         | ippsPowerSpectr_32f            | powerspectf_split_vec         |
| magnitudeXf_interleaved                               | magnitudef_C_interleaved    | ippsMagnitude_32fc             | magnitudef_interleaved_vec    |
| powerspectXf_interleaved                              | powerspectf_C_interleaved   | ippsPowerSpectr_32fc           | powerspectf_interleaved_vec   |
| subcrevXf (a)                                         | subcrevf_C                  | ippsSubCRev_32f                | subcrevf_vec                  |
| sumXf    (a)                                          | sumf_C                      | ippsSum_32f                    | sumf_vec                      |
| meanXf   (a)                                          | meanf_C                     | ippsMean_32f                   | meanf_vec                     |
| sqrtXf   (a)                                          | sqrtf_C                     | ippsSqrt_32f                   | sqrtf_vec                     |
| roundXf  (a)                                          | roundf_C                    | ippsRound_32f                  | roundf_vec                    |
| ceilXf   (a)                                          | ceilf_C                     | ippsCeil_32f                   | ceilf_vec                     |
| floorXf  (a)                                          | floorf_C                    | ippsFloor_32f                  | floorf_vec                    |
| truncXf  (a)                                          | truncf_C                    | ippsTrunc_32f                  | truncf_vec                    |
| modfXf  (a)                                           | modff_C                     | ippsModf_32f                   | modf_vec                      |
| cplxvecmulXf  (a)                                     | cplxvecmul_C/precise        | ippsMul_32fc_A11/24            | cplxvecmulf_vec               |
| cplxvecmulXf_split  (a)                               | cplxvecmul_C_split/precise  | ?                              | cplxvecmulf_vec_split         |
| cplxconjvecmulXf   (a)                                | cplxconjvecmul_C            | ippsMulByConj_32fc_A24         | cplxconjvecmulf_vec           |
| cplxconjvecmulXf_split                                | cplxconjvecmul_C_split      | ?                              | cplxconjvecmulf_vec_split     |
| cplxconjXf          (a)                               | cplxconj_C                  | ippsConj_32fc_A24              | cplxconjf_vec                 |
| cplxvecdivXf        (a)                               | cplxvecdiv_C                | ?                              | cplxvecdivf_vec               |
| cplxvecdivXf_split  (a)                               | cplxvecdiv_C_split          | ?                              | cplxvecdivf_vec_split         |
| setXd                                                 | setd_C                      | ippsSet_64f                    | setd_vec                      |
| zeroXd                                                | zerod_C                     | ippsZero_64f                   | zerod_vec                     |
| copyXd                                                | copyd_C                     | ippsCopy_64f                   | copyd_vec                     |
| sqrtXd                                                | sqrtd_C                     | ippsSqrt_64f                   | sqrtd_vec                     |
| addXd                                                 | addd_c                      | ippsAdd_64f                    | addd_vec                      |
| mulXd                                                 | muld_c                      | ippsMul_64f                    | muld_vec                      |
| subXd                                                 | subd_c                      | ippsSub_64f                    | subd_vec                      |
| divXd                                                 | divd_c                      | ippsDiv_64f                    | divd_vec                      |
| addcXd                                                | addcd_C                     | ippsAddC_64f                   | addcd_vec                     |
| mulcXd                                                | mulcd_C                     | ippsMulC_64f                   | mulcd_vec                     |
| muladdXd                                              | muladdd_C                   | ?                              | muladdd_vec                   |
| mulcaddXd                                             | mulcaddd_C                  | ?                              | muladdcd_vec                  |
| mulcaddcXd                                            | mulcaddcd_C                 | ?                              | mulcaddcd_vec                 |
| muladdcXd                                             | muladdcd_C                  | ?                              | muladdcd_vec                  |
| roundXd                                               | roundd_C                    | ippsRound_64f                  | roundd_vec                    |
| ceilXd                                                | ceild_C                     | ippsCeil_64f                   | ceild_vec                     |
| floorXd                                               | floord_C                    | ippsFloor_64f                  | floord_vec                    |
| truncXd                                               | truncd_C                    | ippsTrunc_64f                  | truncd_vec                    |
| vectorSlopeXd                                         | vectorSloped_C              | ippsVectorSlope_64f            | vectorSloped_vec              |
| sincosXd                                              | sincosd_C                   | ippsSinCos_64f_A53             | ?                             |
| sincosXd_interleaved                                  | sincosd_C_interleaved       | ippsCIS_64fc_A53               | ?                             |
| atanXd                                                | atan_C                      | ippsAtan_64f_A53               | ?                             |
| atan2Xd                                               | atan2d_C                    | ippsAtan2_64f_A53              | ?                             |
| atan2Xd_interleaved                                   | atan2_interleaved_C         | ?                              | ?                             |
| asinXd                                                | asin_C                      | ippsAsin_64f_A53               | ?                             |
| cplxtorealXd                                          | cplxtoreald_C               | ippsCplxToReal_64fc            | ?                             |
| realtocplxXd                                          | realtocplxd_C               | ippsRealToCplx_64f             | ?                             |
| addXs   (a)                                           | adds_c                      | ?                              | adds_vec                      |
| mulXs                                                 | muls_c                      | ?                              | muls_vec                      |
| subXs   (a)                                           | subs_c                      | ?                              | subs_vec                      |
| addcXs  (a)                                           | addcs_C                     | ?                              | addcs_vec                     |
| vectorSlopeXs (a)                                     | vectorSlopes_C              | ippsVectorSlope_32s            | vectorSlopes_vec              |
| flipXs  (a)                                           | flips_C                     | ?                              | flips_vec                     |
| maxeveryXs (a)                                        | maxeverys_c                 | ?                              | maxeverys_vec                 |
| mineveryXs (a)                                        | mineverys_c                 | ?                              | mineverys_vec                 |
| minmaxXs   (a)                                        | minmaxs_c                   | ippsMinMax_32s                 | minmaxs_vec                   |
| thresholdX_gt_s  (a)                                  | threshold_gt_s_C            | ippsThreshold_GT_32s           | thresholdX_gt_s_vec           |
| thresholdX_gtabs_s (a)                                | threshold_gtabs_s_C         | ippsThreshold_GTAbs_32s        | thresholdX_gtabs_s_vec        |
| thresholdX_lt_s     (a)                               | threshold_lt_s_C            | ippsThreshold_LT_32s           | thresholdX_lt_s_vec           |
| thresholdX_ltabs_s  (a)                               | threshold_ltabs_s_C         | ippsThreshold_LTAbs_32s        | thresholdX_ltabs_s_vec        |
| thresholdX_ltval_gtval_s (a)                          | threshold_ltval_gtval_s_C   | ippsThreshold_LTValGTVal_32s   | threshold_ltval_gtval_s_vec   |
| copyXs  (a)                                           | copys_C                     | ippsCopy_32s                   | copys_vec                     |
| ?                                                     | ?                           | ?                              | mulcs_vec                     |
| absdiff16s_Xs (a)                                     | absdiff16s_c                | ?                              | absdiff16s_vec                |
| sum16s32sX (a)                                        | sum16s32s_C                 | ippsSum_16s32s_Sfs             | sum16s32s_vec                 |
| ?                                                     | ors_c                       | ippsOr_32u                     | ?                             |
| ?                                                     | ands_c                      | ippsAnd_32u                    | ?                             |
| sigmoidXf  (a)                                        | sigmoidf_C                  | ?                              | sigmoidf_vec                  |
| PReluXf    (a)                                        | PReluf_C                    | ?                              | PReluf_vec                    |
| softmaxXf  (a)                                        | softmaxf_C                  | ?                              | softmaxf_vec                  |
| pol2cart2DXf (a)                                      | pol2cart2Df_C               | ?                              | pol2cart2Df_vec               |
| cart2pol2DXf (a)                                      | cart2pol2Df_C               | ?                              | cart2pol2Df_vec               |
| gatheri_256/512s                                      | gatheri_C                   | ?                              | ?                             |


## Licence

This library is released under BSD licence so that everyone can freely use it in their project, find bugs, propose new functions or enhance existing ones.
