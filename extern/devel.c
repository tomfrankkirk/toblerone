#include <stdio.h>
#include <math.h>

#include "ctoblerone.h"

void normalToVector(float vec[3], float out[3]) {

    if (abs(vec[2]) < abs(vec[0])) {
        out[0] = vec[1]; 
        out[1] = -vec[0]; 
        out[2] = 0; 
    } else {
        out[0] = 0; 
        out[1] = -vec[2]; 
        out[2] = vec[1]; 
    }

}

void intersections3D(float *pnt, float *ray, float *ps, float *ts, float*normals, int nTris, float *mus) {

    float d1[3], d2[3], vtx[3]; 
    float start = {0, 0, 0}; 
    float onPlane[3][3]; 
    float dp; 


     for (int v=0; v < 3; v++) {
        onPlane[v][0] = 0; 
    }

    for (int t = 0; t < nTris; t++) {

        normalToVector(ray, d1); 
        cross(ray, d1, d2); 


        for (int v=0; v < 3; v++) {
            dp = dot(ray)
        }


    }


}