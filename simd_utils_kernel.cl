#define CL_PROGRAM_STRING_DEBUG_INFO

/*
 * Project : SIMD_Utils
 * Version : 0.2.1
 * Author  : JishinMaster
 * Licence : BSD-2
 */


#ifdef CLANG
typedef float float16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
#endif

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

__constant float tanmDP1 = -0.78515625f;
__constant float tanmDP2 = -2.4187564849853515625e-4f;
__constant float tanmDP3 = -3.77489497744594108e-8f;
__constant float tanlossth = 8192.0f;

__constant float PIO4F = 0.7853981633974483096f;


// Might be slower since directly available on most devices
float mysqrtf(float xx)
{
    float f, x, y, x_tmp;
    int e;

    f = xx;
    if (f <= 0.0f) {
        return (0.0f);
    }

    x = frexp(f, &e); /* f = x * 2**e,   0.5 <= x < 1.0 */
    /* If power of 2 is odd, double x and decrement the power of 2. */
    if (e & 1) {
        x = x + x;
        e -= 1;
    }

    e >>= 1; /* The power of 2 of the square root. */

    if (x > 1.41421356237f) {
        /* x is between sqrt(2) and 2. */
        x = x - 2.0f;
        y = fma(-9.8843065718E-4f, x, 7.9479950957E-4f);
        y = fma(y, x, -3.5890535377E-3f);
        y = fma(y, x, 1.1028809744E-2f);
        y = fma(y, x, -4.4195203560E-2f);
        y = fma(y, x, 3.5355338194E-1f);
        y = fma(y, x, 1.41421356237E0f);
        goto sqdon;
    }

    if (x > 0.707106781187f) {
        /* x is between sqrt(2)/2 and sqrt(2). */
        x = x - 1.0f;

        x_tmp = fma(x, 0.5f, 1.0f);
        y = fma(1.35199291026E-2f, x, -2.26657767832E-2f);
        y = fma(y, x, 2.78720776889E-2f);
        y = fma(y, x, -3.89582788321E-2f);
        y = fma(y, x, 6.24811144548E-2f);
        y = fma(y, x, -1.25001503933E-1f);
        y = y * x;
        y = fma(y, x, x_tmp);

        goto sqdon;
    }

    /* x is between 0.5 and sqrt(2)/2. */
    x = x - 0.5f;

    y = fma(-3.9495006054E-1f, x, 5.1743034569E-1f);
    y = fma(y, x, -4.3214437330E-1f);
    y = fma(y, x, 3.5310730460E-1f);
    y = fma(y, x, -3.5354581892E-1f);
    y = fma(y, x, 7.0710676017E-1f);
    y = fma(y, x, 7.07106781187E-1f);

sqdon:
    y = ldexp(y, e); /* y = y * 2**e */
    return (y);
}

float myasinf(float xx)
{
    float a, x, z, z_tmp;
    int sign, flag;

    x = xx;

    if (x > 0.0f) {
        sign = 1;
        a = x;
    } else {
        sign = -1;
        a = -x;
    }

    if (a > 1.0f) {
        return (0.0f);
    }

    if (a < 1.0e-4f) {
        z = a;
        goto done;
    }

    if (a > 0.5f) {
        z = fma(a, -0.5f, 0.5f);
        x = sqrt(z);
        flag = 1;
    } else {
        x = a;
        z = x * x;
        flag = 0;
    }

    z_tmp = fma(4.2163199048E-2f, z, 2.4181311049E-2f);
    z_tmp = fma(z_tmp, z, 4.5470025998E-2f);
    z_tmp = fma(z_tmp, z, 4.5470025998E-2f);
    z_tmp = fma(z_tmp, z, 7.4953002686E-2f);
    z_tmp = fma(z_tmp, z, 1.6666752422E-1f);
    z_tmp = z_tmp * z;
    z_tmp = fma(z_tmp, x, x);
    z = z_tmp;

    if (flag != 0) {
        z = z + z;
        z = PIO2F - z;
    }
done:
    if (sign < 0)
        z = -z;
    return (z);
}


float myatanf(float xx)
{
    float x, y, z, y_tmp;
    int sign;

    x = xx;

    /* make argument positive and save the sign */
    if (xx < 0.0f) {
        sign = -1;
        x = -xx;
    } else {
        sign = 1;
        x = xx;
    }
    /* range reduction */
    if (x > 2.414213562373095f) /* tan 3pi/8 */
    {
        y = PIO2F;
        x = -(1.0f / x);
    }

    else if (x > 0.4142135623730950f) /* tan pi/8 */
    {
        y = PIO4F;
        x = (x - 1.0f) / (x + 1.0f);
    } else
        y = 0.0f;

    z = x * x;

    y_tmp = fma(8.05374449538e-2f, z, -1.38776856032E-1f);
    y_tmp = fma(y_tmp, z, 1.99777106478E-1f);
    y_tmp = fma(y_tmp, z, -3.33329491539E-1f);
    y_tmp = y_tmp * z;
    y_tmp = fma(y_tmp, x, x);
    y = y + y_tmp;

    if (sign < 0)
        y = -y;

    return (y);
}


float myatan2f(float y, float x)
{
    float z, w;
    int code;

    code = 0;

    if (x < 0.0f)
        code = 2;
    if (y < 0.0f)
        code |= 1;

    if (x == 0.0f) {
        if (code & 1) {
            return (-PIO2F);
        }
        if (y == 0.0f)
            return (0.0f);
        return (PIO2F);
    }

    if (y == 0.0f) {
        if (code & 2)
            return (PIF);
        return (0.0f);
    }


    switch (code) {
    default:
    case 0:
    case 1:
        w = 0.0f;
        break;
    case 2:
        w = PIF;
        break;
    case 3:
        w = -PIF;
        break;
    }

    z = myatanf(y / x);

    return (w + z);
}

float mytanf(float xx)
{
    float x, y, z, zz;
    long j;
    int sign;


    /* make argument positive but save the sign */
    if (xx < 0.0f) {
        x = -xx;
        sign = -1;
    } else {
        x = xx;
        sign = 1;
    }

    if (x > tanlossth) {
        return (0.0f);
    }

    /* compute x mod PIO4 */
    j = FOPI * x; /* integer part of x/(PI/4) */
    y = j;

    /* map zeros and singularities to origin */
    if (j & 1) {
        j += 1;
        y += 1.0f;
    }

    z = fma(y, tanmDP1, x);
    z = fma(y, tanmDP2, z);
    z = fma(y, tanmDP3, z);

    zz = z * z;

    if (x > 1.0e-4f) {
        /* 1.7e-8 relative error in [-pi/4, +pi/4] */
        y = fma(9.38540185543E-3f, zz, 3.11992232697E-3f);
        y = fma(y, zz, 2.44301354525E-2f);
        y = fma(y, zz, 5.34112807005E-2f);
        y = fma(y, zz, 1.33387994085E-1f);
        y = fma(y, zz, 3.33331568548E-1f);
        y = y * zz;
        y = fma(y, z, z);
    } else {
        y = z;
    }

    if (j & 2) {
        y = -1.0f / y;
    }

    if (sign < 0)
        y = -y;

    return (y);
}



float2 mytanf_vec2(float2 xx)
{
    float2 x, y, y2,  z, zz;
    long2 j;
    int2 sign;


    /* make argument positive but save the sign */
    if (xx.x < 0.0f) {
        x.x = -xx.x;
        sign.x = -1;
    } else {
        x.x = xx.x;
        sign.x = 1;
    }

    if (xx.y < 0.0f) {
        x.y = -xx.y;
        sign.y = -1;
    } else {
        x.y = xx.y;
        sign.y = 1;
    }

    /* compute x mod PIO4 */
    j = convert_long2(FOPI * x); /* integer part of x/(PI/4) */
    y = convert_float2(j);

    /* map zeros and singularities to origin */
    if (j.x & 1) {
        j.x += 1;
        y.x += 1.0f;
    }

    if (j.y & 1) {
        j.y += 1;
        y.y += 1.0f;
    }

    z = fma(y, tanmDP1, x);
    z = fma(y, tanmDP2, z);
    z = fma(y, tanmDP3, z);

    zz = z * z;

    y = fma(9.38540185543E-3f, zz, 3.11992232697E-3f);
    y = fma(y, zz, 2.44301354525E-2f);
    y = fma(y, zz, 5.34112807005E-2f);
    y = fma(y, zz, 1.33387994085E-1f);
    y = fma(y, zz, 3.33331568548E-1f);
    y = y * zz;
    y = fma(y, z, z);

   if (x.x <= 1.0e-4f)
      y.x = z.x;
   if (x.y <= 1.0e-4f)
      y.y = z.y;
      
    if (j.x & 2) {
        y.x = -1.0f / y.x;
    }

    if (j.y & 2) {
        y.y = -1.0f / y.y;
    }

    if (sign.x < 0)
        y.x = -y.x;

    if (sign.y < 0)
        y.y = -y.y;

    if (x.x > tanlossth) {
        y.x = 0.0f;
    }
    if (x.y > tanlossth) {
        y.y = 0.0f;
    }

    return (y);
}


float4 mytanf_vec4(float4 xx)
{
    float4 x, y, y2,  z, zz;
    long4 j;
    int4 sign;


    /* make argument positive but save the sign */
    if (xx.x < 0.0f) {
        x.x = -xx.x;
        sign.x = -1;
    } else {
        x.x = xx.x;
        sign.x = 1;
    }

    if (xx.y < 0.0f) {
        x.y = -xx.y;
        sign.y = -1;
    } else {
        x.y = xx.y;
        sign.y = 1;
    }

    if (xx.z < 0.0f) {
        x.z = -xx.z;
        sign.z = -1;
    } else {
        x.z = xx.z;
        sign.z = 1;
    }
    
    if (xx.w < 0.0f) {
        x.w = -xx.w;
        sign.w = -1;
    } else {
        x.w = xx.w;
        sign.w = 1;
    }

    /* compute x mod PIO4 */
    j = convert_long4(FOPI * x); /* integer part of x/(PI/4) */
    y = convert_float4(j);

    /* map zeros and singularities to origin */
    if (j.x & 1) {
        j.x += 1;
        y.x += 1.0f;
    }

    if (j.y & 1) {
        j.y += 1;
        y.y += 1.0f;
    }

    if (j.z & 1) {
        j.z += 1;
        y.z += 1.0f;
    }
    
    if (j.w & 1) {
        j.w += 1;
        y.w += 1.0f;
    }

    z = fma(y, tanmDP1, x);
    z = fma(y, tanmDP2, z);
    z = fma(y, tanmDP3, z);

    zz = z * z;

    y = fma(9.38540185543E-3f, zz, 3.11992232697E-3f);
    y = fma(y, zz, 2.44301354525E-2f);
    y = fma(y, zz, 5.34112807005E-2f);
    y = fma(y, zz, 1.33387994085E-1f);
    y = fma(y, zz, 3.33331568548E-1f);
    y = y * zz;
    y = fma(y, z, z);

   if (x.x <= 1.0e-4f)
      y.x = z.x;
   if (x.y <= 1.0e-4f)
      y.y = z.y;
   if (x.z <= 1.0e-4f)
      y.z = z.z;
   if (x.w <= 1.0e-4f)
      y.w = z.w;

    if (j.x & 2) {
        y.x = -1.0f / y.x;
    }

    if (j.y & 2) {
        y.y = -1.0f / y.y;
    }

    if (j.z & 2) {
        y.z = -1.0f / y.z;
    }
    
    if (j.w & 2) {
        y.w = -1.0f / y.w;
    }

    if (sign.x < 0)
        y.x = -y.x;

    if (sign.y < 0)
        y.y = -y.y;

    if (sign.z < 0)
        y.z = -y.z;
        
    if (sign.w < 0)
        y.w = -y.w;

    if (x.x > tanlossth) {
        y.x = 0.0f;
    }
    if (x.y > tanlossth) {
        y.y = 0.0f;
    }
    if (x.z > tanlossth) {
        y.z = 0.0f;
    }
    if (x.w > tanlossth) {
        y.w = 0.0f;
    }
    
    return (y);
}


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

    y = fma(7.0376836292E-2f, x, -1.1514610310E-1f);
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
            B[i] = sincos(A[i],&C[i]);
            break;
        case 2:
            C[i] = mylogf(A[i]);
            break;
        case 3:
            C[i] = log(A[i]);
            break;
        case 4:
            C[i] = myexpf(A[i]);
            break;
        case 5:
            C[i] = exp(A[i]);
            break;
        case 6:
            C[i] = mytanf(A[i]);
            break;
        case 7:
            C[i] = tan(A[i]);
            break;
        case 8:
            C[i] = myatanf(A[i]);
            break;
        case 9:
            C[i] = atan(A[i]);
            break;
        case 10:
            C[i] = myatan2f(A[i], B[i]);
            break;
        case 11:
            C[i] = atan2(A[i],B[i]);
            break;
        case 12:
            C[i] = myasinf(A[i]);
            break;
        case 13:
            C[i] = asin(A[i]);
            break;
        case 14:
            if(i < nbElts/2){
              float2 a_vec = vload2(0, &A[2 * i]);
              float2 c_vec  = mytanf_vec2(a_vec);
              vstore2(c_vec, 0, &C[2 * i]);
            }
            break;
        case 15:
            if(i < nbElts/4){
              float4 a_vec = vload4(0, &A[4 * i]);
              float4 c_vec  = mytanf_vec4(a_vec);
              vstore4(c_vec, 0, &C[4 * i]);
            }
            break;
        case 16:
            if(i < nbElts/2){
              float2 a_vec = vload2(0, &A[2 * i]);
              float2 c_vec  = tan(a_vec);
              vstore2(c_vec, 0, &C[2 * i]);
            }
            break;
        case 17:
            if(i < nbElts/4){
              float4 a_vec = vload4(0, &A[4 * i]);
              float4 c_vec  = tan(a_vec);
              vstore4(c_vec, 0, &C[4 * i]);
            }
            break;
        default:
            break;
        }
    }
}
