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
- Qemu 5.X (emulator) )for arm/aarch64
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

For FMA support you need to add -DFMA and -mfma to x86 targets, and -DFMA to Armv8 targets.

## Supported Functions 

The following table is a work in progress : 

| SSE/NEON                   | AVX/AVX2                   | AVX512                     | RISCV              | C_REF                     | IPP_REF                      |
|:----------------------------:|:----------------------------:|:----------------------------:|:--------------------:|:---------------------------:|------------------------------|
|                            |                            |                            |                    |                           |                              |
| log10_128f                 | log10_256f                 | log10_512f                 | X                  | log10f_C                  |                              |
| ln_128f                    | ln_256f                    | ln_512f                    | X                  | lnf_C                     | ippsLn_32f                   |
| fabs128f                   | fabs256f                   | fabs512f                   | X                  | fabsf_C                   | ippsAbs_32f                  |
| set128f                    | set256f                    | set512f                    | X                  | setf_C                    |                              |
| zero128f                   | zero256f                   | zero512f                   | X                  | zerof_C                   | ippsZero_32f                 |
| copy128f                   | copy256f                   | copy512f                   | X                  | copyf_C                   | ippsCopy_32f                 |
| add128f                    | add256f                    | add512f                    | addf_vec           | addf_c                    |                              |
| mul128f                    | mul256f                    | mul512f                    | X                  | mulf_C                    | ippsMul_32f                  |
| sub128f                    | sub256f                    | sub512f                    | X                  | subf_c                    |                              |
| addc128f                   | addc256f                   | addc512f                   | X                  | addcf_C                   | ippsAddC_32f                 |
| mulc128f                   | mulc256f                   | mulc512f                   | X                  | mulcf_C                   | ippsMulC_32f                 |
| muladd128f                 | muladd256f                 | muladd512f                 | X                  | muladdf_C                 |                              |
| mulcadd128f                | mulcadd256f                | mulcadd512f                | X                  | mulcaddf_C                |                              |
| mulcaddc128f               | mulcaddc256f               | mulcaddc512f               | X                  | mulcaddcf_C               |                              |
| muladdc128f                | muladdc256f                | muladdc512f                | X                  | muladdcf_C                |                              |
| div128f                    | div256f                    | div512f                    | X                  | divf_C                    |                              |
| vectorSlope128f            | vectorSlope256f            | vectorSlope512f            | X                  | vectorSlopef_C            | ippsVectorSlope_32f          |
| convertFloat32ToU8_128     | X                          | X                          | X                  | convertFloat32ToU8_C      | ippsConvert_32f8u_Sfs        |
| convertInt16ToFloat32_128  | X                          | X                          | X                  | convertInt16ToFloat32_C   | ippsConvert_16s32f_Sfs       |
| cplxtoreal128f             | cplxtoreal256f             | X                          | X                  | cplxtorealf_C             | ippsCplxToReal_32fc          |
| realtocplx128f             | realtocplx256f             | X                          | X                  | realtocplx_C              | ippsRealToCplx_32f           |
| convert128_64f32f          | convert256_64f32f          | X                          | X                  | convert_64f32f_C          | ippsConvert_64f32f           |
| convert128_32f64f          | convert256_32f64f          | convert512_32f64f          | X                  | convert_32f64f_C          | ippsConvert_32f64f           |
| flip128f                   | flip256f                   | X                          | X                  | flipf_C                   | ippsFlip_32f                 |
| maxevery128f               | maxevery256f               | maxevery512f               | maxeveryf_vec      | maxeveryf_c               | ippsMaxEvery_32f             |
| minevery128f               | minevery256f               | minevery512f               | mineveryf_vec      | mineveryf_c               | ippsMinEvery_32f             |
| minmax128f                 | minmax256f                 | minmax512f                 | X                  | minmaxf_c                 | ippsMinMax_32f               |
| threshold128_gt_f          | threshold256_gt_f          | threshold512_gt_f          | threshold_gt_f_vec | threshold_gt_f_C          |                              |
| threshold128_gtabs_f       | threshold256_gtabs_f       | threshold512_gtabs_f       | X                  | threshold_gtabs_f_C       | ippsThreshold_GTAbs_32f      |
| threshold128_lt_f          | threshold256_lt_f          | threshold512_lt_f          | threshold_lt_f_vec | threshold_lt_f_C          | ippsThreshold_LT_32f         |
| threshold128_ltabs_f       | threshold256_ltabs_f       | threshold512_ltabs_f       | X                  | threshold_ltabs_f_C       |                              |
| threshold128_ltval_gtval_f | threshold256_ltval_gtval_f | threshold512_ltval_gtval_f | X                  | threshold_ltval_gtval_f_C | ippsThreshold_LTValGTVal_32f |
| sin128f                    | sin256f                    | sin512f                    | X                  | sinf_C                    | ippsSin_32f_A24              |
| cos128f                    | cos256f                    | cos512f                    | X                  | cosf_C                    | ippsCos_32f_A24              |
| sincos128f                 | sincos256f                 | sincos512f                 | X                  | sincosf_C                 | ippsSinCos_32f_A24           |
| atan128f                   | atan256f                   | atan512f                   | X                  | atanf_C                   | ippsAtan_32f_A24             |
| atan2128f                  | atan2256f                  | atan2512f                  | X                  | atan2f_C                  | ippsAtan2_32f_A24            |
| asin128f                   | asin256f                   | asin512f                   | X                  | asinf_C                   | ippsAsin_32f_A24             |
| tan128f                    | tan256f                    | tan512f                    | X                  | tanf_C                    | ippsTan_32f_A24              |
| magnitude128f_split        | magnitude256f_split        | magnitude512f_split        | X                  | magnitudef_C_split        | ippsMagnitude_32f            |
| powerspect128f_split       | powerspect256f_split       | powerspect512f_split       | X                  | powerspectf_C_split       |                              |
| magnitude128f_interleaved  | X                          | X                          | X                  | magnitudef_C_interleaved  | ippsMagnitude_32fc           |
| subcrev128f                | subcrev256f                | subcrev512f                | X                  | subcrevf_C                |                              |
| sum128f                    | sum256f                    | sum512f                    | X                  | sumf_C                    |                              |
| mean128f                   | mean256f                   | mean512f                   | X                  | meanf_C                   | ippsMean_32f                 |
| sqrt128f                   | sqrt256f                   | sqrt512f                   | X                  | sqrtf_C                   |                              |
| round128f                  | round256f                  | round512f                  | X                  | roundf_C                  | ippsRound_32f                |
| ceil128f                   | ceil256f                   | ceil512f                   | X                  | ceilf_C                   | ippsCeil_32f                 |
| floor128f                  | floor256f                  | floor512f                  | X                  | floorf_C                  | ippsFloor_32f                |
| trunc128f                  | trunc256f                  | trunc512f                  | X                  | truncf_C                  | ippsTrunc_32f                |
| cplxvecmul128f             | cplxvecmul256f             | cplxvecmul512f             | X                  | cplxvecmul_C              | ippsMul_32fc_A24             |
| cplxvecmul128f_split       | cplxvecmul256f_split       | cplxvecmul512f_split       | X                  | cplxvecmul_C_split        |                              |
| cplxconjvecmul128f         | cplxconjvecmul256f         | cplxconjvecmul512f         | X                  | cplxconjvecmul_C          |                              |
| cplxconjvecmul128f_split   | cplxconjvecmul256f_split   | cplxconjvecmul512f_split   | X                  | cplxconjvecmul_C_split    |                              |
| cplxconj128f               | cplxconj256f               | cplxconj512f               | X                  | cplxconj_C                | ippsConj_32fc                |
| set128d                    | set256d                    | set512d                    | X                  | setd_C                    |                              |
| zero128d                   | zero256d                   | zero512d                   | X                  | zerod_C                   |                              |
| copy128d                   | copy256d                   | copy512d                   | X                  | copyd_C                   |                              |
| sqrt128d                   | sqrt256d                   | sqrt512d                   | X                  | sqrtd_C                   |                              |
| add128d                    | add256d                    | add512d                    | X                  | addd_c                    |                              |
| mul128d                    | mul256d                    | mul512d                    | X                  | muld_c                    |                              |
| sub128d                    | sub256d                    | sub512d                    | X                  | subd_c                    |                              |
| div128d                    | div256d                    | div512d                    | X                  | divd_c                    |                              |
| addc128d                   | addc256d                   | addc512d                   | X                  | addcd_C                   |                              |
| mulc128d                   | mulc256d                   | mulc512d                   | X                  | mulcd_C                   |                              |
| muladd128d                 | muladd256d                 | muladd512d                 | X                  | muladdd_C                 |                              |
| mulcadd128d                | mulcadd256d                | mulcadd512d                | X                  | mulcaddd_C                |                              |
| mulcaddc128d               | mulcaddc256d               | mulcaddc512d               | X                  | mulcaddcd_C               |                              |
| muladdc128d                | muladdc256d                | muladdc512d                | X                  | muladdcd_C                |                              |
| round128d                  | round256d                  | round512d                  | X                  | roundd_C                  |                              |
| ceil128d                   | ceil256d                   | ceil512d                   | X                  | ceild_C                   |                              |
| floor128d                  | floor256d                  | floor512d                  | X                  | floord_C                  |                              |
| trunc128d                  | trunc256d                  | trunc512d                  | X                  | truncd_C                  |                              |
| vectorSlope128d            | vectorSlope256d            | vectorSlope512d            | X                  | vectorSloped_C            | ippsVectorSlope_64f          |
| sincos128d                 | sincos256d                 | X                          | X                  | sincosd_C                 |                              |
| atan128d                   | atan256d                   | atan512d                   | X                  | atan_C                    | ippsAtan_64f_A53             |
| asin128d                   | asin256d                   | asin512d                   | X                  | asin_C                    | ippsAsin_64f_A53             |
| add128s                    | add256s                    | add512s                    | adds_vec           | adds_c                    |                              |
| mul128s                    | mul256s                    | mul512s                    | muls_vec           | muls_c                    |                              |
| sub128s                    | sub256s                    | sub512s                    | subs_vec           | subs_c                    |                              |
| addc128s                   | addc256s                   | addc512s                   | addcs_vec          | addcs_C                   |                              |
| vectorSlope128s            | X                          | X                          | X                  | vectorSlopes_C            |                              |
| copy128s                   | copy256s                   | copy512s                   | X                  | copys_C                   | ippsCopy_32s                 |
| X                          | X                          | X                          | mulcs_vec          | X                         |                              |
| X                          | X                          | X                          | X                  | ors_c                     |                              |
| X                          | X                          | X                          | X                  | ands_c                    |                              |

## Licence

This library is released under BSD licence so that everyone can freely use it in their project, find bugs, propose new functions or enhance existing ones.
