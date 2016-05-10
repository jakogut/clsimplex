/* Adapted from Simplex noise algorithm written by Stefan Gustavson
                                       (stefan.gustavson@gmail.com)
*/

#define F3 0.333333333f
#define G3 0.166666667f

__constant unsigned char perm[512] = {151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
  129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
  151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
  129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

__constant float grad3lut[16][3] = {
  { 1.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f }, // 12 cube edges
  { -1.0f, 0.0f, 1.0f }, { 0.0f, -1.0f, 1.0f },
  { 1.0f, 0.0f, -1.0f }, { 0.0f, 1.0f, -1.0f },
  { -1.0f, 0.0f, -1.0f }, { 0.0f, -1.0f, -1.0f },
  { 1.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 0.0f },
  { -1.0f, 1.0f, 0.0f }, { -1.0f, -1.0f, 0.0f },
  { 1.0f, 0.0f, 1.0f }, { -1.0f, 0.0f, 1.0f }, // 4 repeats to make 16
  { 0.0f, 1.0f, -1.0f }, { 0.0f, -1.0f, -1.0f }
};

void grad3( int hash, float *gx, float *gy, float *gz ) {
    int h = hash & 15;
    *gx = grad3lut[h][0];
    *gy = grad3lut[h][1];
    *gz = grad3lut[h][2];
    return;
}

/* This kernel takes one three dimensional vector for the origin
 * of the noise block, as well as three integers to specify the
 * dimensions of the block to generate, and finally a pointer to
 * global memory to which to write the results. */
__kernel void
sdnoise3(const float x, const float y, const float z,
        const unsigned cs_x, const unsigned cs_y, const unsigned cs_z,
        __global float *chunk) {

    // The global id is used to write the results to a flat array
    unsigned idx = get_global_id(0);

    if (idx > 0 && idx < cs_x * cs_y * cs_z){
        // Find the x, y, and z coordinates from the index and block dims
        float zo = floor((float)idx / (cs_z*cs_z));
        float yo = floor((float)(idx - (zo*cs_z*cs_z)) / cs_y);
        float xo = floor((float)idx - (yo*cs_y + zo*cs_z*cs_z));

        float n0, n1, n2, n3;
        float gx0, gy0, gz0, gx1, gy1, gz1;
        float gx2, gy2, gz2, gx3, gy3, gz3;

        float s = (xo + yo + zo) * F3;
        float xs = xo + s;
        float ys = yo + s;
        float zs = zo + s;
        int i = floor((float)xs);
        int j = floor((float)ys);
        int k = floor((float)zs);

        float t = (float)(i+j+k)*G3;
        float X0 = i-t;
        float Y0 = j-t;
        float Z0 = k-t;
        float x0 = xo-X0;
        float y0 = yo-Y0;
        float z0 = zo-Z0;

        int i1, j1, k1;
        int i2, j2, k2;

        if (x0 >= y0) {
            if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
            else if (x0 >= z0) {i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
            else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
        } else {
            if ( y0<z0 ) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
            else if(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
            else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
        }

        float x1 = x0 - i1 + G3; /* Offsets for second corner in (x,y,z) coords */
        float y1 = y0 - j1 + G3;
        float z1 = z0 - k1 + G3;
        float x2 = x0 - i2 + 2.0f * G3; /* Offsets for third corner in (x,y,z) coords */
        float y2 = y0 - j2 + 2.0f * G3;
        float z2 = z0 - k2 + 2.0f * G3;
        float x3 = x0 - 1.0f + 3.0f * G3; /* Offsets for last corner in (x,y,z) coords */
        float y3 = y0 - 1.0f + 3.0f * G3;
        float z3 = z0 - 1.0f + 3.0f * G3;

        /* Wrap the integer indices at 256, to avoid indexing perm[] out of bounds */
        int ii = i & 0xff;
        int jj = j & 0xff;
        int kk = k & 0xff;

        /* Calculate the contribution from the four corners */
        float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
        float t20, t40;
        if(t0 < 0.0f) n0 = t0 = t20 = t40 = gx0 = gy0 = gz0 = 0.0f;
        else {
            grad3( perm[ii + perm[jj + perm[kk]]], &gx0, &gy0, &gz0 );
            t20 = t0 * t0;
            t40 = t20 * t20;
            n0 = t40 * ( gx0 * x0 + gy0 * y0 + gz0 * z0 );
        }

        float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
        float t21, t41;
        if(t1 < 0.0f) n1 = t1 = t21 = t41 = gx1 = gy1 = gz1 = 0.0f;
        else {
            grad3( perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]], &gx1, &gy1, &gz1 );
            t21 = t1 * t1;
            t41 = t21 * t21;
            n1 = t41 * ( gx1 * x1 + gy1 * y1 + gz1 * z1 );
        }

        float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
        float t22, t42;
        if(t2 < 0.0f) n2 = t2 = t22 = t42 = gx2 = gy2 = gz2 = 0.0f;
        else {
            grad3( perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]], &gx2, &gy2, &gz2 );
            t22 = t2 * t2;
            t42 = t22 * t22;
            n2 = t42 * ( gx2 * x2 + gy2 * y2 + gz2 * z2 );
        }

        float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
        float t23, t43;
        if(t3 < 0.0f) n3 = t3 = t23 = t43 = gx3 = gy3 = gz3 = 0.0f;
        else {
            grad3( perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]], &gx3, &gy3, &gz3 );
            t23 = t3 * t3;
            t43 = t23 * t23;
            n3 = t43 * ( gx3 * x3 + gy3 * y3 + gz3 * z3 );
        }

        /*  Add contributions from each corner to get the final noise value.
         * The result is scaled to return values in the range [-1,1] */
        chunk[idx] = 28.0f * (n0 + n1 + n2 + n3);
    } /*else {
        //printf("Index %i out of range.\n", idx);
    }*/

}
