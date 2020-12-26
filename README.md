# simd_utils

A header only library implementing common mathematical functions using SIMD intrinsics.
This library is C/C++ compatible (tested with GCC7.5/9.3 and clang 9).

Thanks to Julien Pommier and Giovanni Garberoglio for their work on sin,cos,log, and exp functions in SSE, AVX, and NEON intrinsics.
Thanks to the DLTcollab team for their work on sse2neon.

## What is SIMD Utils?

The purpose of this library is to give an open-source implementation of SIMD optimized commonly used algorithms, such as type conversion (float32, float64, uint16, ...), trigonometry (sin, cos, atan, ...), log/exp, min/max, and other functions.
Its API was thought as a simple replacement for Intel IPP/MKL libraries.


## Building

To build the project you will need the sse_mathfun.h, avx_mathfun.h and neon_mathun.h headers available here http://gruntthepeon.free.fr/ssemath/, and there http://software-lisc.fbk.eu/avx_mathfun/
This project also uses a forked version of sse2neon (https://github.com/DLTcollab/sse2neon) adding Fused Multiple Add functions

## Targets

Supported targets are : 
- SSE (SSE4.X mostly)
- AVX (AVX1.0 mostly, some AVX2 functions)
- ARM Neon (through sse2neon plus some optimized functions).
- RISC-V Vector extension (experimental)

## Licence

This library is released under BSD licence so that everyone can freely use it in their project, find bugs, propose new functions or enhance existing ones.
