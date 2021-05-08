/*
 * Project : SIMD_Utils
 * Version : 0.1.12
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

static float FOPI = 1.27323954473516;
static float PIO4F = 0.7853981633974483096;

/* Note, these constants are for a 32-bit significand: */
static float DP1 = 0.7853851318359375;
static float DP2 = 1.30315311253070831298828125e-5;
static float DP3 = 3.03855025325309630e-11;
static float lossth = 65536.;

/* These are for a 24-bit significand: */
/*static float DP1 = 0.78515625;
static float DP2 = 2.4187564849853515625e-4;
static float DP3 = 3.77489497744594108e-8;
static float lossth = 8192.;*/

static float T24M1 = 16777215.;

static float sincof[] = {-1.9515295891E-4, 8.3321608736E-3, -1.6666654611E-1};
static float coscof[] = {2.443315711809948E-005, -1.388731625493765E-003,
                         4.166664568298827E-002};

static inline int mysincosf(float xx, float *s, float *c)
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
        return (0.0f);
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
        x = x - y * PIO4F;
    } else {
        /* Extended precision modular arithmetic */
        x = ((x - y * DP1) - y * DP2) - y * DP3;
    }
    /*einits();*/
    z = x * x;

    /* measured relative error in +/- pi/4 is 7.8e-8 */
    y1 = ((coscof[0] * z + coscof[1]) * z + coscof[2]) * z * z;
    y1 -= 0.5f * z;
    y1 += 1.0f;

    /* Theoretical relative error = 3.8e-9 in [-pi/4, +pi/4] */
    y2 = ((sincof[0] * z + sincof[1]) * z + sincof[2]) * z * x;
    y2 += x;

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
