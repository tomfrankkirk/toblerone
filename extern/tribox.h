#include <stdio.h>
#include <math.h>

char triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3]); 

char tribox_wrapper(float bc1, float bc2, float bc3, 
                   float fs1, float fs2, float fs3,
                   float tv11, float tv12, float tv13,
                   float tv21, float tv22, float tv23,
                   float tv31, float tv32, float tv33);