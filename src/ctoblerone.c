#include <stdio.h>
#include <math.h>
#include "tribox.h"

char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2)
{
    char intersection = 0; 

    int i, j;
    j=2;
    for (i=0; i < 3; ++i) //start with the wraparound case
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

