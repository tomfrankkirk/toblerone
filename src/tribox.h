#include <stdio.h>
#include <math.h>

int planeBoxOverlap(const float normal[3], float d, const float maxbox[3]); 

char triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3]); 
