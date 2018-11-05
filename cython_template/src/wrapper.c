#include "wrapper.h"

#include <math.h>

/**
 * Area of triangle from two side vectors
 */
static float area(const float *v1, const float *v2)
{
   float cp1 = (v1[1] * v2[2]) - (v1[2] * v2[1]);
   float cp2 = (v1[2] * v2[0]) - (v1[0] * v2[2]);
   float cp3 = (v1[0] * v2[1]) - (v1[1] * v2[0]);

   return sqrt(cp1*cp1 + cp2*cp2 + cp3*cp3) / 2;
}

/**
 * trig_vertices is an num_trigsx3 array of ints. Each set of 3 integers represents the vertices of a triangle
 * float_array is an num_verticesx3 array of floats. The'm'th entry represents the co-ordinates of vertex 'm'
 * output_buffer is an num_trigsx1 array of floats to be filled in with the areas of triangles defined by trig_vertices
 */
void get_areas(const int *trig_vertices, int num_trigs, const float *float_array, int num_vertices, float *output_buffer)
{
  for (int trig=0; trig<num_trigs; trig++) {
    /* Here should check that trig_vertices[0], [1] and [2] are less than num_vertices! */

    const float *p1 = &float_array[3* trig_vertices[0]];
    const float *p2 = &float_array[3* trig_vertices[1]];
    const float *p3 = &float_array[3* trig_vertices[2]];

    float v1[3], v2[3];
    for (int j=0; j<3; j++) {
      v1[j] = p3[j] - p1[j];
      v2[j] = p2[j] - p1[j];
    }

    *output_buffer = area(v1, v2);

    trig_vertices += 3;
    output_buffer++;
  }
}

