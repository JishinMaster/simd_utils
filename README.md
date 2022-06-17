# simd_utils

A header only library implementing common mathematical functions using SIMD intrinsics.
This library is C/C++ compatible (tested with GCC7.5/9.3 and clang 9).

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
- AVX512 (experimental, most of float32 functions)
- ARM Neon (through sse2neon plus some optimized functions).
- RISC-V Vector extension (experimental)
- PowerPC Alitivec (experimental)

128 bit functions (SSE and NEON) are name function128type, such as asin128f, which computes the arcsinus function on an float32 array. Float64 functions have the "d" suffix.
256 bit functions (AVX/AVX2) have 256 instead of 128 in their name, such as asin256f.
256 bit functions (AVX512) have 512 instead of 128 in their name, such as cos512f.
Vector functions (RISCV) for which the SIMD length makes less sense, are name functionType_vec, such as subs_vec, which substract an int32 array from and other one.

The project has been tested on :
- Intel Atom
- Intel Ivy Bridge Core-i7
- Intel Skylake Core-i7
- Intel Cannonlake Core-i7
- Intel SDE (emulator) for AVX-512
- Spike (emulator) for RISCV Vector
- Qemu 5.X (emulator) )for arm/aarch64, and ppc
- Cortex-a53 (Raspberry Pi 3B)
- Cortex-a9 (ZYBO)

## Building

To build the project you will need the sse_mathfun.h, avx_mathfun.h and neon_mathun.h headers available here http://gruntthepeon.free.fr/ssemath/, and there http://software-lisc.fbk.eu/avx_mathfun/
This project also uses a forked version of sse2neon (https://github.com/DLTcollab/sse2neon) adding functions such as double precision and Fused Multiple Add.

Simply include simd_utils.h in your C/C++ file, and compile with : 
- SSE support : gcc -DSSE -msse4.2 -c file.c -I .
- AVX support : gcc -DSSE -DAVX -mavx2  -c file.c -I .
- AVX512 support : gcc -DSSE -DAVX -DAVX512 -march=skylake-avx512 -mprefer-vector-width=512 -c file.c -I .
- ARM V7 NEON support : arm-none-linux-gnueabihf-gcc -march=armv7-a -mfpu=neon -DARM -DSSE -flax-vector-conversions -c file.c -I .
- ARM V8 NEON support : aarch64-linux-gnu-gcc -DARM -DFMA -DSSE -flax-vector-conversions -c file.c -I .
- ALTIVEC support : powerpc64-linux-gnu-gcc -DALTIVEC -DFMA -maltivec -flax-vector-conversions -c file.c -I .

For FMA support you need to add -DFMA and -mfma to x86 targets, and -DFMA to Armv8 targets.
For ARMV7 targets, you could also add -DSSE2NEON_PRECISE_SQRT for improved accuracy with sqrt and rsqrt
For X86 targets with ICC compiler, simply add -DICC to activate Intel SVML intrinsics.

## OpenCL (experimental)

The same approach is applied to OpenCL kernels as an experiment, focused on GPUs, but other OpenCL devices may work.
At the moment only some functions are supported (log, exp, sincos, tan, atan, atan2, asin, sqrt), based on the cephes library, which seems to be faster that the OpenCL native functions (tested on Intel GPU with beignet 1.3) 
To try it out, simply use : 
- gcc -DSSE -msse4.2 -march=native simd_test_opencl.c -lOpenCL -lrt -lm (add -DSIMPLE_BUFFERS for CPU devices)


## Supported Functions 

SSE/NEON are 128bits wide. SSE functions use up to SSE4.2 features.
Some functions are directly coded using NEON intrinsics (for performance reasons), but most functions translate SSE code to NEON using sse2neon header.
Some AVX functions, such as integer ones, require AVX2. The 256 bit integer functions are emulated using SSE for some floating point functions if AVX2 is unavailable.

The following table is a work in progress, "X" means there is not yet an implemented function (or a directly equivalent Intel IPP function) :
 
| SSE/NEON                   | AVX/AVX2                   | AVX512                     | C_REF                     | IPP_REF                      | RISCV                       | ALTIVEC              |
|----------------------------|----------------------------|----------------------------|---------------------------|------------------------------|-----------------------------|----------------------|
|                            |                            |                            |                           |                              |                             |                      |
| log10_128f/precise         | log10_256f/precise         | log10_512f/precise         | log10f_C                  | ippsLog10_32f_A24            | X                           | log10_128f           |
| log2_128f/precise          | log2_256f/precise          | log2_512f/precise          | log2f_C                   |                              | X                           | log2_128f            |
| ln_128f                    | ln_256f                    | ln_512f                    | lnf_C                     | ippsLn_32f_A24               | X                           | ln_128f              |
| exp_128f                   | exp_256f                   | exp_512f                   | expf_C                    | ippsExp_32f_A24              | X                           | X                    |
| cbrt128f                   | cbrt256f                   | cbrt512f                   | cbrtf_C                   | X                            | X                           | X                    |
| fabs128f                   | fabs256f                   | fabs512f                   | fabsf_C                   | ippsAbs_32f                  | X                           | X                    |
| set128f                    | set256f                    | set512f                    | setf_C                    | ippsSet_32f                  | X                           | set128f              |
| zero128f                   | zero256f                   | zero512f                   | zerof_C                   | ippsZero_32f                 | X                           | zero128f             |
| copy128f                   | copy256f                   | copy512f                   | copyf_C                   | ippsCopy_32f                 | X                           | X                    |
| add128f                    | add256f                    | add512f                    | addf_c                    | ippsAdd_32f                  | addf_vec                    | X                    |
| mul128f                    | mul256f                    | mul512f                    | mulf_C                    | ippsMul_32f                  | X                           | mul128f              |
| sub128f                    | sub256f                    | sub512f                    | subf_c                    | ippsSub_32f                  | X                           | X                    |
| addc128f                   | addc256f                   | addc512f                   | addcf_C                   | ippsAddC_32f                 | X                           | X                    |
| mulc128f                   | mulc256f                   | mulc512f                   | mulcf_C                   | ippsMulC_32f                 | X                           | X                    |
| muladd128f                 | muladd256f                 | muladd512f                 | muladdf_C                 | X                            | X                           | X                    |
| mulcadd128f                | mulcadd256f                | mulcadd512f                | mulcaddf_C                | X                            | X                           | X                    |
| mulcaddc128f               | mulcaddc256f               | mulcaddc512f               | mulcaddcf_C               | X                            | X                           | X                    |
| muladdc128f                | muladdc256f                | muladdc512f                | muladdcf_C                | X                            | X                           | X                    |
| div128f                    | div256f                    | div512f                    | divf_C                    | ippsDiv_32f_A24              | X                           | X                    |
| vectorSlope128f            | vectorSlope256f            | vectorSlope512f            | vectorSlopef_C            | ippsVectorSlope_32f          | X                           | X                    |
| convertFloat32ToU8_128     | convertFloat32ToU8_256     | convertFloat32ToU8_512     | convertFloat32ToU8_C      | ippsConvert_32f8u_Sfs        | X                           | X                    |
| convertFloat32ToU16_128    | convertFloat32ToU16_256    | convertFloat32ToU16_256    | convertFloat32ToI16_C     | ippsConvert_32f16u_Sfs       | X                           | X                    |
| convertFloat32ToI16_128    | convertFloat32ToI16_256    | convertFloat32ToI16_256    | convertFloat32ToI16_C     | ippsConvert_32f16s_Sfs       | X                           | X                    |
| convertInt16ToFloat32_128  | convertInt16ToFloat32_256  | convertInt16ToFloat32_512  | convertInt16ToFloat32_C   | ippsConvert_16s32f_Sfs       | X                           | X                    |
| cplxtoreal128f             | cplxtoreal256f             | cplxtoreal512f             | cplxtorealf_C             | ippsCplxToReal_32fc          | X                           | cplxtoreal128f       |
| realtocplx128f             | realtocplx256f             | realtocplx512f             | realtocplx_C              | ippsRealToCplx_32f           | X                           | X                    |
| convert128_64f32f          | convert256_64f32f          | convert512_64f32f          | convert_64f32f_C          | ippsConvert_64f32f           | X                           | X                    |
| convert128_32f64f          | convert256_32f64f          | convert512_32f64f          | convert_32f64f_C          | ippsConvert_32f64f           | X                           | X                    |
| flip128f                   | flip256f                   | flip512f                   | flipf_C                   | ippsFlip_32f                 | X                           | X                    |
| maxevery128f               | maxevery256f               | maxevery512f               | maxeveryf_c               | ippsMaxEvery_32f             | maxeveryf_vec               | X                    |
| minevery128f               | minevery256f               | minevery512f               | mineveryf_c               | ippsMinEvery_32f             | mineveryf_vec               | minevery128f         |
| minmax128f                 | minmax256f                 | minmax512f                 | minmaxf_c                 | ippsMinMax_32f               | X                           | X                    |
| threshold128_gt_f          | threshold256_gt_f          | threshold512_gt_f          | threshold_gt_f_C          | ippsThreshold_GT_32f         | threshold_gt_f_vec          | X                    |
| threshold128_gtabs_f       | threshold256_gtabs_f       | threshold512_gtabs_f       | threshold_gtabs_f_C       | ippsThreshold_GTAbs_32f      | X                           | X                    |
| threshold128_lt_f          | threshold256_lt_f          | threshold512_lt_f          | threshold_lt_f_C          | ippsThreshold_LT_32f         | threshold_lt_f_vec          | X                    |
| threshold128_ltabs_f       | threshold256_ltabs_f       | threshold512_ltabs_f       | threshold_ltabs_f_C       | ippsThreshold_LTAbs_32f      | X                           | X                    |
| threshold128_ltval_gtval_f | threshold256_ltval_gtval_f | threshold512_ltval_gtval_f | threshold_ltval_gtval_f_C | ippsThreshold_LTValGTVal_32f | threshold_ltval_gtval_f_vec | X                    |
| sin128f                    | sin256f                    | sin512f                    | sinf_C                    | ippsSin_32f_A24              | sinf_vec                    | X                    |
| cos128f                    | cos256f                    | cos512f                    | cosf_C                    | ippsCos_32f_A24              | X                           | X                    |
| sincos128f                 | sincos256f                 | sincos512f                 | sincosf_C                 | ippsSinCos_32f_A24           | sincosf_vec                 | X                    |
| sincos128f_interleaved     | sincos256f_interleaved     | sincos512f_interleaved     | sincosf_C_interleaved     | X                            | X                           | X                    |
| cosh128f                   | cosh256f                   | cosh512f                   | coshf_C                   | ippsCosh_32f_A24             | X                           | X                    |
| sinh128f                   | sinh256f                   | sinh512f                   | sinhf_C                   | ippsSinh_32f_A24             | X                           | X                    |
| acosh128f                  | acosh256f                  | acosh512f                  | acoshf_C                  | ippsAcosh_32f_A24            | X                           | X                    |
| asinh128f                  | asinh256f                  | asinh512f                  | asinhf_C                  | ippsAsinh_32f_A24            | X                           | X                    |
| atanh128f                  | atanh256f                  | atanh512f                  | atanhf_C                  | ippsAtanh_32f_A24            | X                           | X                    |
| atan128f                   | atan256f                   | atan512f                   | atanf_C                   | ippsAtan_32f_A24             | X                           | X                    |
| atan2128f                  | atan2256f                  | atan2512f                  | atan2f_C                  | ippsAtan2_32f_A24            | X                           | X                    |
| atan2128f_interleaved      | atan2256f_interleaved      | atan2512f_interleaved      | atan2f_interleaved_C      | X                            | X                           | X                    |
| asin128f                   | asin256f                   | asin512f                   | asinf_C                   | ippsAsin_32f_A24             | X                           | X                    |
| tanh128f                   | tanh256f                   | tanh512f                   | tanhf_C                   | ippsTanh_32f_A24             | X                           | X                    |
| tan128f                    | tan256f                    | tan512f                    | tanf_C                    | ippsTan_32f_A24              | X                           | X                    |
| magnitude128f_split        | magnitude256f_split        | magnitude512f_split        | magnitudef_C_split        | ippsMagnitude_32f            | magnitudef_split_vec        | magnitude128f_split  |
| powerspect128f_split       | powerspect256f_split       | powerspect512f_split       | powerspectf_C_split       | ippsPowerSpectr_32f          | powerspectf_split_vec       | powerspect128f_split |
| magnitude128f_interleaved  | magnitude256f_interleaved  | magnitude512f_interleaved  | magnitudef_C_interleaved  | ippsMagnitude_32fc           | X                           | X                    |
| powerspect128f_interleaved | powerspect256f_interleaved | powerspect512f_interleaved | powerspectf_C_interleaved | ippsPowerSpectr_32fc         | X                           | X                    |
| subcrev128f                | subcrev256f                | subcrev512f                | subcrevf_C                | ippsSubCRev_32f              | X                           | X                    |
| sum128f                    | sum256f                    | sum512f                    | sumf_C                    | ippsSum_32f                  | sumf_vec                    | X                    |
| mean128f                   | mean256f                   | mean512f                   | meanf_C                   | ippsMean_32f                 | meanf_vec                   | X                    |
| sqrt128f                   | sqrt256f                   | sqrt512f                   | sqrtf_C                   | ippsSqrt_32f                 | X                           | X                    |
| round128f                  | round256f                  | round512f                  | roundf_C                  | ippsRound_32f                | X                           | X                    |
| ceil128f                   | ceil256f                   | ceil512f                   | ceilf_C                   | ippsCeil_32f                 | X                           | X                    |
| floor128f                  | floor256f                  | floor512f                  | floorf_C                  | ippsFloor_32f                | X                           | X                    |
| trunc128f                  | trunc256f                  | trunc512f                  | truncf_C                  | ippsTrunc_32f                | X                           | X                    |
| cplxvecmul128f             | cplxvecmul256f             | cplxvecmul512f             | cplxvecmul_C              | ippsMul_32fc_A24             | X                           | X                    |
| cplxvecmul128f_split       | cplxvecmul256f_split       | cplxvecmul512f_split       | cplxvecmul_C_split        | X                            | X                           | X                    |
| cplxconjvecmul128f         | cplxconjvecmul256f         | cplxconjvecmul512f         | cplxconjvecmul_C          | ippsMulByConj_32fc_A24       | X                           | X                    |
| cplxconjvecmul128f_split   | cplxconjvecmul256f_split   | cplxconjvecmul512f_split   | cplxconjvecmul_C_split    | X                            | X                           | X                    |
| cplxconj128f               | cplxconj256f               | cplxconj512f               | cplxconj_C                | ippsConj_32fc_A24            | X                           | X                    |
| cplxvecdiv128f             | cplxvecdiv256f             | cplxvecdiv512f             | cplxvecdiv_C              | X                            | X                           | X                    |
| cplxvecdiv128f_split       | cplxvecdiv256f_split       | cplxvecdiv512f_split       | cplxvecdiv_C_split        | X                            | X                           | X                    |
| set128d                    | set256d                    | set512d                    | setd_C                    | ippsSet_64f                  | X                           | X                    |
| zero128d                   | zero256d                   | zero512d                   | zerod_C                   | ippsZero_64f                 | X                           | X                    |
| copy128d                   | copy256d                   | copy512d                   | copyd_C                   | ippsCopy_64f                 | X                           | X                    |
| sqrt128d                   | sqrt256d                   | sqrt512d                   | sqrtd_C                   | ippsSqrt_64f                 | X                           | X                    |
| add128d                    | add256d                    | add512d                    | addd_c                    | ippsAdd_64f                  | X                           | X                    |
| mul128d                    | mul256d                    | mul512d                    | muld_c                    | ippsMul_64f                  | X                           | X                    |
| sub128d                    | sub256d                    | sub512d                    | subd_c                    | ippsSub_64f                  | X                           | X                    |
| div128d                    | div256d                    | div512d                    | divd_c                    | ippsDiv_64f                  | X                           | X                    |
| addc128d                   | addc256d                   | addc512d                   | addcd_C                   | ippsAddC_64f                 | X                           | X                    |
| mulc128d                   | mulc256d                   | mulc512d                   | mulcd_C                   | ippsMulC_64f                 | X                           | X                    |
| muladd128d                 | muladd256d                 | muladd512d                 | muladdd_C                 | X                            | X                           | X                    |
| mulcadd128d                | mulcadd256d                | mulcadd512d                | mulcaddd_C                | X                            | X                           | X                    |
| mulcaddc128d               | mulcaddc256d               | mulcaddc512d               | mulcaddcd_C               | X                            | X                           | X                    |
| muladdc128d                | muladdc256d                | muladdc512d                | muladdcd_C                | X                            | X                           | X                    |
| round128d                  | round256d                  | round512d                  | roundd_C                  | ippsRound_64f                | X                           | X                    |
| ceil128d                   | ceil256d                   | ceil512d                   | ceild_C                   | ippsCeil_64f                 | X                           | X                    |
| floor128d                  | floor256d                  | floor512d                  | floord_C                  | ippsFloor_64f                | X                           | X                    |
| trunc128d                  | trunc256d                  | trunc512d                  | truncd_C                  | ippsTrunc_64f                | X                           | X                    |
| vectorSlope128d            | vectorSlope256d            | vectorSlope512d            | vectorSloped_C            | ippsVectorSlope_64f          | X                           | X                    |
| sincos128d                 | sincos256d                 | sincos512d                 | sincosd_C                 | ippsSinCos_64f_A53           | X                           | X                    |
| atan128d                   | atan256d                   | atan512d                   | atan_C                    | ippsAtan_64f_A53             | X                           | X                    |
| asin128d                   | asin256d                   | asin512d                   | asin_C                    | ippsAsin_64f_A53             | X                           | X                    |
| add128s                    | add256s                    | add512s                    | adds_c                    | X                            | adds_vec                    | X                    |
| mul128s                    | mul256s                    | mul512s                    | muls_c                    | X                            | muls_vec                    | X                    |
| sub128s                    | sub256s                    | sub512s                    | subs_c                    | X                            | subs_vec                    | X                    |
| addc128s                   | addc256s                   | addc512s                   | addcs_C                   | X                            | addcs_vec                   | X                    |
| vectorSlope128s            | vectorSlope256s            | vectorSlope512s            | vectorSlopes_C            | ippsVectorSlope_32s          | X                           | X                    |
| copy128s                   | copy256s                   | copy512s                   | copys_C                   | ippsCopy_32s                 | X                           | X                    |
| X                          | X                          | X                          | X                         | X                            | mulcs_vec                   | X                    |
| absdiff16s_128s            | absdiff16s_256s            | absdiff16s_512s            | absdiff16s_c              | X                            | X                           | X                    |
| X                          | X                          | X                          | ors_c                     | ippsOr_32u                   | X                           | X                    |
| X                          | X                          | X                          | ands_c                    | ippsAnd_32u                  | X                           | X                    |
| sigmoid128f                | sigmoid256f                | sigmoid512f                | sigmoidf_C                | X                            | X                           | X                    |
| PRelu128f                  | PRelu256f                  | PRelu512f                  | PReluf_C                  | X                            | X                           | PRelu128f            |
| softmax128f                | softmax256f                | softmax512f                | softmaxf_C                | X                            | X                           | X                    |
| pol2cart2D128f             | pol2cart2D256f             | pol2cart2D512f             | pol2cart2Df_C             | X                            | X                           | X                    |
| cart2pol2D128f             | cart2pol2D256f             | cart2pol2D512f             | cart2pol2Df_C             | X                            | X                           | X                    |
| X                          | gatheri_256s               | gatheri_512s               | gatheri_C                 | X                            | X                           | X                    |

## Licence

This library is released under BSD licence so that everyone can freely use it in their project, find bugs, propose new functions or enhance existing ones.
