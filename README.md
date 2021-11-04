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
- AVX512 support : gcc -DSSE -DAVX -DAVX512 -march=skylake-avx512 -c file.c -I .
- NEON support : aarch64-linux-gnu-gcc -DARM -DFMA -DSSE -flax-vector-conversions -c file.c -I .
- ALTIVEC support : powerpc64-linux-gnu-gcc -DALTIVEC -DFMA -maltivec -flax-vector-conversions -c file.c -I .

For FMA support you need to add -DFMA and -mfma to x86 targets, and -DFMA to Armv8 targets.

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
 

| SSE/NEON                   | AVX/AVX2                   | AVX512                     | C_REF                     | IPP_REF                      | RISCV              |
|:----------------------------:|:----------------------------:|:----------------------------:|:---------------------------:|:------------------------------:|:--------------------:|
|                            |                            |                            |                           |                              |                    |
| log10_128f                 | log10_256f                 | log10_512f                 | log10f_C                  | ippsLog10_32f_A24            | X                  |
| ln_128f                    | ln_256f                    | ln_512f                    | lnf_C                     | ippsLn_32f                   | X                  |
| exp_128f                   | exp_256f                   | exp_512f                   | expf_C                    | X                            | X                  |
| fabs128f                   | fabs256f                   | fabs512f                   | fabsf_C                   | ippsAbs_32f                  | X                  |
| set128f                    | set256f                    | set512f                    | setf_C                    | ippsSet_32f                  | X                  |
| zero128f                   | zero256f                   | zero512f                   | zerof_C                   | ippsZero_32f                 | X                  |
| copy128f                   | copy256f                   | copy512f                   | copyf_C                   | ippsCopy_32f                 | X                  |
| add128f                    | add256f                    | add512f                    | addf_c                    | ippsAdd_32f                  | addf_vec           |
| mul128f                    | mul256f                    | mul512f                    | mulf_C                    | ippsMul_32f                  | X                  |
| sub128f                    | sub256f                    | sub512f                    | subf_c                    | ippsSub_32f                  | X                  |
| addc128f                   | addc256f                   | addc512f                   | addcf_C                   | ippsAddC_32f                 | X                  |
| mulc128f                   | mulc256f                   | mulc512f                   | mulcf_C                   | ippsMulC_32f                 | X                  |
| muladd128f                 | muladd256f                 | muladd512f                 | muladdf_C                 | X                            | X                  |
| mulcadd128f                | mulcadd256f                | mulcadd512f                | mulcaddf_C                | X                            | X                  |
| mulcaddc128f               | mulcaddc256f               | mulcaddc512f               | mulcaddcf_C               | X                            | X                  |
| muladdc128f                | muladdc256f                | muladdc512f                | muladdcf_C                | X                            | X                  |
| div128f                    | div256f                    | div512f                    | divf_C                    | ippsDiv_32f                  | X                  |
| vectorSlope128f            | vectorSlope256f            | vectorSlope512f            | vectorSlopef_C            | ippsVectorSlope_32f          | X                  |
| convertFloat32ToU8_128     | X                          | X                          | convertFloat32ToU8_C      | ippsConvert_32f8u_Sfs        | X                  |
| convertInt16ToFloat32_128  | X                          | X                          | convertInt16ToFloat32_C   | ippsConvert_16s32f_Sfs       | X                  |
| cplxtoreal128f             | cplxtoreal256f             | X                          | cplxtorealf_C             | ippsCplxToReal_32fc          | X                  |
| realtocplx128f             | realtocplx256f             | X                          | realtocplx_C              | ippsRealToCplx_32f           | X                  |
| convert128_64f32f          | convert256_64f32f          | X                          | convert_64f32f_C          | ippsConvert_64f32f           | X                  |
| convert128_32f64f          | convert256_32f64f          | convert512_32f64f          | convert_32f64f_C          | ippsConvert_32f64f           | X                  |
| flip128f                   | flip256f                   | X                          | flipf_C                   | ippsFlip_32f                 | X                  |
| maxevery128f               | maxevery256f               | maxevery512f               | maxeveryf_c               | ippsMaxEvery_32f             | maxeveryf_vec      |
| minevery128f               | minevery256f               | minevery512f               | mineveryf_c               | ippsMinEvery_32f             | mineveryf_vec      |
| minmax128f                 | minmax256f                 | minmax512f                 | minmaxf_c                 | ippsMinMax_32f               | X                  |
| threshold128_gt_f          | threshold256_gt_f          | threshold512_gt_f          | threshold_gt_f_C          | ippsThreshold_GT_32f         | threshold_gt_f_vec |
| threshold128_gtabs_f       | threshold256_gtabs_f       | threshold512_gtabs_f       | threshold_gtabs_f_C       | ippsThreshold_GTAbs_32f      | X                  |
| threshold128_lt_f          | threshold256_lt_f          | threshold512_lt_f          | threshold_lt_f_C          | ippsThreshold_LT_32f         | threshold_lt_f_vec |
| threshold128_ltabs_f       | threshold256_ltabs_f       | threshold512_ltabs_f       | threshold_ltabs_f_C       | ippsThreshold_LTAbs_32f      | X                  |
| threshold128_ltval_gtval_f | threshold256_ltval_gtval_f | threshold512_ltval_gtval_f | threshold_ltval_gtval_f_C | ippsThreshold_LTValGTVal_32f | X                  |
| sin128f                    | sin256f                    | sin512f                    | sinf_C                    | ippsSin_32f_A24              | X                  |
| cos128f                    | cos256f                    | cos512f                    | cosf_C                    | ippsCos_32f_A24              | X                  |
| sincos128f                 | sincos256f                 | sincos512f                 | sincosf_C                 | ippsSinCos_32f_A24           | X                  |
| atan128f                   | atan256f                   | atan512f                   | atanf_C                   | ippsAtan_32f_A24             | X                  |
| atan2128f                  | atan2256f                  | atan2512f                  | atan2f_C                  | ippsAtan2_32f_A24            | X                  |
| asin128f                   | asin256f                   | asin512f                   | asinf_C                   | ippsAsin_32f_A24             | X                  |
| tan128f                    | tan256f                    | tan512f                    | tanf_C                    | ippsTan_32f_A24              | X                  |
| magnitude128f_split        | magnitude256f_split        | magnitude512f_split        | magnitudef_C_split        | ippsMagnitude_32f            | X                  |
| powerspect128f_split       | powerspect256f_split       | powerspect512f_split       | powerspectf_C_split       | ippsPowerSpectr_32f          | X                  |
| magnitude128f_interleaved  | X                          | X                          | magnitudef_C_interleaved  | ippsMagnitude_32fc           | X                  |
| subcrev128f                | subcrev256f                | subcrev512f                | subcrevf_C                | ippsSubCRev_32f              | X                  |
| sum128f                    | sum256f                    | sum512f                    | sumf_C                    | ippsSum_32f                  | X                  |
| mean128f                   | mean256f                   | mean512f                   | meanf_C                   | ippsMean_32f                 | X                  |
| sqrt128f                   | sqrt256f                   | sqrt512f                   | sqrtf_C                   | ippsSqrt_32f                 | X                  |
| round128f                  | round256f                  | round512f                  | roundf_C                  | ippsRound_32f                | X                  |
| ceil128f                   | ceil256f                   | ceil512f                   | ceilf_C                   | ippsCeil_32f                 | X                  |
| floor128f                  | floor256f                  | floor512f                  | floorf_C                  | ippsFloor_32f                | X                  |
| trunc128f                  | trunc256f                  | trunc512f                  | truncf_C                  | ippsTrunc_32f                | X                  |
| cplxvecmul128f             | cplxvecmul256f             | cplxvecmul512f             | cplxvecmul_C              | ippsMul_32fc_A24             | X                  |
| cplxvecmul128f_split       | cplxvecmul256f_split       | cplxvecmul512f_split       | cplxvecmul_C_split        | X                            | X                  |
| cplxconjvecmul128f         | cplxconjvecmul256f         | cplxconjvecmul512f         | cplxconjvecmul_C          | ippsMulByConj_32fc_A24       | X                  |
| cplxconjvecmul128f_split   | cplxconjvecmul256f_split   | cplxconjvecmul512f_split   | cplxconjvecmul_C_split    | X                            | X                  |
| cplxconj128f               | cplxconj256f               | cplxconj512f               | cplxconj_C                | ippsConj_32fc_A24            | X                  |
| set128d                    | set256d                    | set512d                    | setd_C                    | ippsSet_64f                  | X                  |
| zero128d                   | zero256d                   | zero512d                   | zerod_C                   | ippsZero_64f                 | X                  |
| copy128d                   | copy256d                   | copy512d                   | copyd_C                   | ippsCopy_64f                 | X                  |
| sqrt128d                   | sqrt256d                   | sqrt512d                   | sqrtd_C                   | ippsSqrt_64f                 | X                  |
| add128d                    | add256d                    | add512d                    | addd_c                    | ippsAdd_64f                  | X                  |
| mul128d                    | mul256d                    | mul512d                    | muld_c                    | ippsMul_64f                  | X                  |
| sub128d                    | sub256d                    | sub512d                    | subd_c                    | ippsSub_64f                  | X                  |
| div128d                    | div256d                    | div512d                    | divd_c                    | ippsDiv_64f                  | X                  |
| addc128d                   | addc256d                   | addc512d                   | addcd_C                   | ippsAddC_64f                 | X                  |
| mulc128d                   | mulc256d                   | mulc512d                   | mulcd_C                   | ippsMulC_64f                 | X                  |
| muladd128d                 | muladd256d                 | muladd512d                 | muladdd_C                 | X                            | X                  |
| mulcadd128d                | mulcadd256d                | mulcadd512d                | mulcaddd_C                | X                            | X                  |
| mulcaddc128d               | mulcaddc256d               | mulcaddc512d               | mulcaddcd_C               | X                            | X                  |
| muladdc128d                | muladdc256d                | muladdc512d                | muladdcd_C                | X                            | X                  |
| round128d                  | round256d                  | round512d                  | roundd_C                  | ippsRound_64f                | X                  |
| ceil128d                   | ceil256d                   | ceil512d                   | ceild_C                   | ippsCeil_64f                 | X                  |
| floor128d                  | floor256d                  | floor512d                  | floord_C                  | ippsFloor_64f                | X                  |
| trunc128d                  | trunc256d                  | trunc512d                  | truncd_C                  | ippsTrunc_64f                | X                  |
| vectorSlope128d            | vectorSlope256d            | vectorSlope512d            | vectorSloped_C            | ippsVectorSlope_64f          | X                  |
| sincos128d                 | sincos256d                 | X                          | sincosd_C                 | ippsSinCos_64f_A53           | X                  |
| atan128d                   | atan256d                   | atan512d                   | atan_C                    | ippsAtan_64f_A53             | X                  |
| asin128d                   | asin256d                   | asin512d                   | asin_C                    | ippsAsin_64f_A53             | X                  |
| add128s                    | add256s                    | add512s                    | adds_c                    | X                            | adds_vec           |
| mul128s                    | mul256s                    | mul512s                    | muls_c                    | X                            | muls_vec           |
| sub128s                    | sub256s                    | sub512s                    | subs_c                    | X                            | subs_vec           |
| addc128s                   | addc256s                   | addc512s                   | addcs_C                   | X                            | addcs_vec          |
| vectorSlope128s            | X                          | X                          | vectorSlopes_C            | ippsVectorSlope_32s          | X                  |
| copy128s                   | copy256s                   | copy512s                   | copys_C                   | ippsCopy_32s                 | X                  |
| X                          | X                          | X                          | X                         | X                            | mulcs_vec          |
| X                          | X                          | X                          | ors_c                     | ippsOr_32u                   | X                  |
| X                          | X                          | X                          | ands_c                    | ippsAnd_32u                  | X                  |
| cosh128f                   | cosh256f                   | cosh512f                   | coshf_C                   | X                            | X                  |
| acosh128f                  | acosh256f                  | acosh512f                  | acoshf_C                  | X                            | X                  |
| sinh128f                   | sinh256f                   | sinh512f                   | sinhf_C                   | X                            | X                  |
| asinh128f                  | asinh256f                  | asinh512f                  | asinhf_C                  | X                            | X                  |
| tanh128f                   | tanh256f                   | tanh512f                   | tanhf_C                   | X                            | X                  |
| atanh128f                  | atanh256f                  | atanh512f                  | atanhf_C                  | X                            | X                  |
| sigmoid128f                | sigmoid256f                | X                          | sigmoidf_C                | X                            | X                  |
| PRelu128f                  | PRelu256f                  | X                          | PReluf_C                  | X                            | X                  |
| softmax128f                | softmax256f                | X                          | softmaxf_C                | X                            | X                  |

## Licence

This library is released under BSD licence so that everyone can freely use it in their project, find bugs, propose new functions or enhance existing ones.
