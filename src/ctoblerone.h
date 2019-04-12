#include <stdio.h>


char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2); 

static void cross(const float* v1, const float* v2, float* out); 

static float dot(const float* v1, const float* v2); 