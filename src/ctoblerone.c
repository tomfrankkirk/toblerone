#include <stdio.h>
#include <math.h>
#include "tribox.h"

static void cross(const float* v1, const float* v2, float* out)
{
    out[0] = (v1[1] * v2[2]) - (v1[2] * v2[1]);
    out[1] = (v1[2] * v2[0]) - (v1[0] * v2[2]);
    out[2] = (v1[0] * v2[1]) - (v1[1] * v2[0]);
}

static float dot(const float* v1, const float* v2)
{
    return (v1[0] * v2[0]) + (v1[1] * v2[1]) + (v1[2] * v2[2]);
}

char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2)
{
    char intersection = 0; 

    for (int j = 2, i = 0; i < 3; ++i) //start with the wraparound case
    {
        if ((tri[i][ax1] < start[ax1]) != (tri[j][ax1] < start[ax1]))
        {//if one vertex is on one side of the point in the x direction, and the other is on the other side (equal case is treated as greater)
            int ti, tj;
            if (tri[i][ax1] < tri[j][ax1])//reorient the segment consistently to get a consistent answer
            {
                ti = i; tj = j;
            } else {
                ti = j; tj = i;
            }
            if ((tri[ti][ax2] - tri[tj][ax2]) / (tri[ti][ax1] - tri[tj][ax1]) * (start[ax1] - tri[tj][ax1]) + tri[tj][ax2] > start[ax2])
            {//if the point on the line described by the two vertices with the same x coordinate is above (greater y) than the test point
                intersection = !intersection; //even/odd winding rule
            }
        }
        j = i;//consecutive vertices, does 2,0 then 0,1 then 1,2
    }
    return intersection;
}


char ray_wrapper(float s1, float s2, float s3, 
                   float tv11, float tv12, float tv13,
                   float tv21, float tv22, float tv23,
                   float tv31, float tv32, float tv33, 
                   int ax1, int ax2)
{
    float s[3], tv[3][3];
    s[0] = s1;
    s[1] = s2;
    s[2] = s3;
    tv[0][0] = tv11;
    tv[0][1] = tv12;
    tv[0][2] = tv13;
    tv[1][0] = tv21;
    tv[1][1] = tv22;
    tv[1][2] = tv23;
    tv[2][0] = tv31;
    tv[2][1] = tv32;
    tv[2][2] = tv33;
    return testRayTriangleIntersection(tv, s, ax1, ax2);
}   





void triPlaneIntersections(const float* points, const int* tris, int nTris, const float* testPnt, const float* ray, int normDF, float* output)
{
    float normal[3], v1[3], v2[3], toPoint[3]; 
    float dotRN, dotPN; 

    for (int t = 0; t < nTris; t++)
    {
        const float *p1 = &points[3* tris[0]];
        const float *p2 = &points[3* tris[1]];
        const float *p3 = &points[3* tris[2]];

        for (int j=0; j<3; j++) {
            v1[j] = p3[j] - p1[j];
            v2[j] = p2[j] - p1[j];
            toPoint[j] = p1[j] - testPnt[j]; 
        }

        cross(v1, v2, normal); 
        dotRN = dot(ray, normal); 
        dotPN = dot(toPoint, normal); 

        if (dotRN)
        {
            output[t] = dotPN / dotRN; 
        } else {
            output[t] = 0.0; 
        }
    }
}



