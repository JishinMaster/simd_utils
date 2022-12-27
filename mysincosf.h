/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#pragma once

static inline int mysincosf(float xx, float *s, float *c)
{
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
        x = ((x + y * minus_cephes_DP1) + y * minus_cephes_DP2) + y * minus_cephes_DP3;
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
