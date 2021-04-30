/*
 * Project : SIMD_Utils
 * Version : 0.1.9
 * Author  : JishinMaster
 * Licence : BSD-2
 */


inline void sin_vec16(__global float *A, __global float *C, int i)
{
    float16 a_vec = vload16(0, &A[16 * i]);
    float16 c_vec = sin(a_vec);
    vstore16(c_vec, 0, &C[16 * i]);
}

inline void sin_vec8(__global float *A, __global float *C, int i)
{
    float8 a_vec = vload8(0, &A[8 * i]);
    float8 c_vec = sin(a_vec);
    vstore8(c_vec, 0, &C[8 * i]);
}

inline void sin_vec4(__global float *A, __global float *C, int i)
{
    float4 a_vec = vload4(0, &A[4 * i]);
    float4 c_vec = sin(a_vec);
    vstore4(c_vec, 0, &C[4 * i]);
}

inline void sin_vec2(__global float *A, __global float *C, int i)
{
    float2 a_vec = vload2(0, &A[2 * i]);
    float2 c_vec = sin(a_vec);
    vstore2(c_vec, 0, &C[2 * i]);
}

__constant float FOPI = 1.27323954473516f;
__constant float mPIO4F = 0.7853981633974483096f;

/* Note, these constants are for a 32-bit significand: */
__constant float DP1 = -0.7853851318359375f;
__constant float DP2 = -1.30315311253070831298828125e-5f;
__constant float DP3 = -3.03855025325309630e-11f;
__constant float lossth = 65536.f;

__constant float T24M1 = 16777215.f;

__constant float sincof[] = {-1.9515295891E-4f, 8.3321608736E-3f,
                             -1.6666654611E-1f};
__constant float coscof[] = {2.443315711809948E-005f, -1.388731625493765E-003f,
                             4.166664568298827E-002f};

__constant float MAXNUMF = 3.4028234663852885981170418348451692544e38f;
__constant float MAXLOGF = 88.72283905206835f;
__constant float MINLOGF = -103.278929903431851103f; /* log(2^-149) */
__constant float LOG2EF = 1.44269504088896341f;
__constant float LOGE2F = 0.693147180559945309f;
__constant float SQRTHF = 0.707106781186547524f;
__constant float PIF = 3.141592653589793238f;
__constant float PIO2F = 1.5707963267948966192f;
__constant float MACHEPF = 5.9604644775390625E-8f;
__constant float mC1 = -0.693359375f;
__constant float mC2 = 2.12194440e-4f;

int mysincosf(float xx, __global float *s, __global float *c)
{
    float *p;
    float x, y, y1, y2, z;
    int j, sign_sin, sign_cos;

    sign_sin = 1;
    sign_cos = 1;

    x = xx;
    if (xx < 0) {
        sign_sin = -1;
        x = -xx;
    }
    if (x > T24M1) {
        return -1;
    }
    j = FOPI * x; /* integer part of x/(PI/4) */
    y = j;
    /* map zeros to origin */
    if (j & 1) {
        j += 1;
        y += 1.0f;
    }
    j &= 7; /* octant modulo 360 degrees */
    /* reflect in x axis */
    if (j > 3) {
        sign_sin = -sign_sin;
        sign_cos = -sign_cos;
        j -= 4;
    }

    if (j > 1)
        sign_cos = -sign_cos;

    if (x > lossth) {
        x = fma(y, mPIO4F, x);
    } else {
        /* Extended precision modular arithmetic */
        x = fma(y, DP1, x);
        x = fma(y, DP2, x);
        x = fma(y, DP3, x);
    }
    /*einits();*/
    z = x * x;

    /* measured relative error in +/- pi/4 is 7.8e-8 */
    y1 = fma(coscof[0], z, coscof[1]);
    y1 = fma(y1, z, coscof[2]);
    y1 = y1 * z * z;
    y1 = fma(-0.5f, z, y1);
    y1 += 1.0f;

    /* Theoretical relative error = 3.8e-9 in [-pi/4, +pi/4] */
    y2 = fma(sincof[0], z, sincof[1]);
    y2 = fma(y2, z, sincof[2]);
    y2 = y2 * z;
    y2 = fma(y2, x, x);

    if ((j == 1) || (j == 2)) {
        *s = y1;
        *c = y2;
    } else {
        *s = y2;
        *c = y1;
    }
    // COS

    /*einitd();*/
    if (sign_sin < 0) {
        *s = -(*s);
    }
    if (sign_cos < 0) {
        *c = -(*c);
    }

    return 0;
}

float myexpf(float xx)
{
    float x, z, z_back;
    int n;

    x = xx;

    if (x > MAXLOGF) {
        //mtherr("expf", OVERFLOW);
        return (MAXNUMF);
    }

    if (x < MINLOGF) {
        //mtherr("expf", UNDERFLOW);
        return (0.0f);
    }

    /* Express e**x = e**g 2**n
   *   = e**g e**( n loge(2) )
   *   = e**( g + n loge(2) )
   */
    z = floor(fma(LOG2EF, x, 0.5f)); /* floor() truncates toward -infinity. */
    x = fma(z, mC1, x);
    x = fma(z, mC2, x);
    n = z;

    z_back = x * x;
    /* Theoretical peak relative error in [-0.5, +0.5] is 4.2e-9. */
    z = fma(1.9875691500E-4f, x, 1.3981999507E-3f);
    z = fma(z, x, 8.3334519073E-3f);
    z = fma(z, x, 4.1665795894E-2f);
    z = fma(z, x, 1.6666665459E-1f);
    z = fma(z, x, 5.0000001201E-1f);
    z = fma(z, z_back, x);
    z = z + 1.0f;
    /* multiply by power of 2 */
    x = ldexp(z, n);

    return (x);
}

float mylogf(float xx)
{
    float y;
    float x, z, fe;
    int e;

    x = xx;
    fe = 0.0f;

    x = frexp(x, &e);
    if (x < SQRTHF) {
        e -= 1;
        x = fma(2.0f, x, -1.0f); /*  2x - 1  */
    } else {
        x = x - 1.0f;
    }
    z = x * x;

    y = fma(7.0376836292E-2, x, -1.1514610310E-1f);
    y = fma(y, x, 1.1676998740E-1f);
    y = fma(y, x, -1.2420140846E-1f);
    y = fma(y, x, 1.4249322787E-1f);
    y = fma(y, x, -1.6668057665E-1f);
    y = fma(y, x, 2.0000714765E-1f);
    y = fma(y, x, -2.4999993993E-1f);
    y = fma(y, x, 3.3333331174E-1f);
    y = y * x * z;

    if (e) {
        fe = e;
        y = fma(-2.12194440e-4f, fe, y);
    }

    y = fma(-0.5f, z, y); /* y - 0.5 x^2 */
    z = x + y;            /* ... + x  */

    if (e)
        z = fma(0.693359375f, fe, z);

    return (z);
}

__kernel void kernel_test(__global float *restrict A,
                          __global float *restrict B,
                          __global float *restrict C,
                          int nbElts,
                          int batch,
                          int type)
{
    int i = get_global_id(0);

    if (i < nbElts) {
        switch (type) {
        case 0:
            mysincosf(A[i], &B[i], &C[i]);
            break;
        case 1:
            C[i] = mylogf(A[i]);
            break;
        case 2:
            C[i] = myexpf(A[i]);
            break;
        default:
            break;
        }
    }
}
