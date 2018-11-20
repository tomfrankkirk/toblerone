#include <stdio.h>


char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2); 

void testTrianglesVoxelIntersection(const int* triangles, const float* points, int nTris, const float* voxCent, const float* voxHalfSize, char* results); 

void ctestManyRayTriangleIntersections(const int* triangles, const float* points, const float* start, const int nTris, const int ax1, const int ax2, char* results); 


char ray_wrapper(float s1, float s2, float s3, 
                   float tv11, float tv12, float tv13,
                   float tv21, float tv22, float tv23,
                   float tv31, float tv32, float tv33, 
                   int ax1, int ax2); 

void triPlaneIntersections(const float* points, const int* tris, int nTris, const float* testPnt, const float* ray, int normDF, float* output);

void testManyTriangleVoxelIntersections(int *tris, float *points, float *vC, float *hS, int nTris, char *results); 