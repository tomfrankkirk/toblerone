import numpy as np
cimport numpy as np
import cython

# External function imports from ../src directory
cdef extern from "tribox.h":
    char triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3])

    #use separating axis theorem to test overlap between triangle and box
    #need to test for overlap in these directions:
    #1) the {x,y,z}-directions (actually, since we use the AABB of the triangle
    #   we do not even need to test these)
    #2) normal of the triangle
    #3) crossproduct(edge from tri, {x,y,z}-direction)
    #   this gives 3x3=9 more tests

    cdef float v0[3]
    cdef float v1[3]
    cdef float v2[3]
    cdef float normal[3]
    cdef float e0[3]
    cdef float e1[3]
    cdef float e2[3] 
    cdef float vmin[3]
    cdef float vmax[3]
    cdef Py_ssize_t idx 
    cdef float mini,maxi,d,p0,p1,p2,rad,fex,fey,fez

    for idx in range(3):            
        v0[idx] = verts[0,idx] - box_cent[idx]
        v1[idx] = verts[1,idx] - box_cent[idx]
        v2[idx] = verts[2,idx] - box_cent[idx]
        e0[idx] = v1[idx] - v0[idx]
        e1[idx] = v2[idx] - v1[idx] 
        e2[idx] = v0[idx] - v2[idx] 

    #Bullet 3: 
    # test the 9 tests first (this was faster)
    fex = fabs(e0[0])
    fey = fabs(e0[1])
    fez = fabs(e0[2])

    #AXISTEST_X01(e0[Z], e0[Y], fez, fey);
    p0 = e0[2] * v0[1] - e0[1] * v0[2];                    
    p2 = e0[2] * v2[1] - e0[1] * v2[2];                    
    if (p0 < p2): 
        mini = p0
        maxi = p2 
    else:
        mini = p2
        maxi = p0

    rad = fez * half_size[1] + fey * half_size[2];  
    if (mini > rad) or (maxi <- rad): 
        return 0 

    #AXISTEST_Y02(e0[Z], e0[X], fez, fex);
    p0 = -e0[2] * v0[0] + e0[0] * v0[2];                    
    p2 = -e0[2] * v2[0] + e0[0] * v2[2];   
    if (p0 < p2): 
        mini = p0
        maxi = p2 
    else:
        mini = p2
        maxi = p0

    rad = fez * half_size[0] + fex * half_size[2]
    if (mini > rad) or (maxi <- rad): 
        return 0 

    #AXISTEST_Z12(e0[Y], e0[X], fey, fex);
    p1 = e0[1] * v1[0] - e0[0] * v1[1];                   
    p2 = e0[1] * v2[0] - e0[0] * v2[1];  
    if (p2 < p1):
        mini=p2
        maxi=p1
    else:
        mini=p1 
        maxi=p2
        
    rad = fey * half_size[0] + fex * half_size[1];
    if (mini > rad) or (maxi <- rad): 
        return 0 


    fex = fabs(e1[0])
    fey = fabs(e1[1])
    fez = fabs(e1[2])

    #AXISTEST_X01(e1[Z], e1[Y], fez, fey);
    p0 = e1[2] * v0[1] - e1[1] * v0[2];                    
    p2 = e1[2] * v2[1] - e1[1] * v2[2];                    
    if (p0 < p2): 
        mini = p0
        maxi = p2 
    else:
        mini = p2
        maxi = p0

    rad = fez * half_size[1] + fey * half_size[2];  
    if (mini > rad) or (maxi <- rad): 
        return 0 

    #AXISTEST_Y02(e1[Z], e1[X], fez, fex);
    p0 = -e1[2] * v0[0] + e1[0] * v0[2];                    
    p2 = -e1[2] * v2[0] + e1[0] * v2[2];   
    if (p0 < p2): 
        mini = p0
        maxi = p2 
    else:
        mini = p2
        maxi = p0

    rad = fez * half_size[0] + fex * half_size[2]
    if (mini > rad) or (maxi <- rad): 
        return 0 

    #AXISTEST_Z0(e1[Y], e1[X], fey, fex);
    p0 = e1[1] * v0[0] - e1[0] * v0[1];  
    p1 = e1[1] * v1[0] - e1[0] * v1[1];  
    if (p0 < p1):
        mini=p0 
        maxi=p1
    else:
        mini=p1
        maxi=p0

    rad = fey * half_size[0] + fex * half_size[1];   
    if (mini > rad) or (maxi <- rad): 
        return 0 


    fex = fabs(e2[0])
    fey = fabs(e2[1])
    fez = fabs(e2[2])

    #AXISTEST_X2(e2[Z], e2[Y], fez, fey);
    p0 = e2[2] * v0[1] - e2[1] * v0[2]                  
    p1 = e2[2] * v1[1] - e2[1] * v1[2]                   
    if(p0<p1):
        mini = p0
        maxi = p1
    else:
        mini = p1
        maxi = p0
    
    rad = fez * half_size[1] + fey * half_size[2];  
    if (mini > rad) or (maxi <- rad): 
        return 0 

    #AXISTEST_Y1(e2[Z], e2[X], fez, fex);
    p0 = -e2[2] * v0[0] + e2[0] * v0[2]                  
    p1 = -e2[2] * v1[0] + e2[0] * v1[2]                     
    if(p0<p1):
        mini = p0
        maxi = p1
    else:
        mini = p1
        maxi = p0    

    rad = fez * half_size[0] + fex * half_size[2];  
    if (mini > rad) or (maxi <- rad): 
        return 0 

    #AXISTEST_Z12(e2[Y], e2[X], fey, fex);
    p1 = e2[1] * v1[0] - e2[0] * v1[1];                    
    p2 = e2[1] * v2[0] - e2[0] * v2[1];                    
    if (p2 < p1):
        mini=p2
        maxi=p1
    else:
        mini=p1 
        maxi=p2   

    rad = fey * half_size[0] + fex * half_size[1];   
    if (mini > rad) or (maxi <- rad): 
        return 0 

    #Bullet 1: 
    # first test overlap in the {x,y,z}-directions 
    # find min, max of the triangle each direction, and test for overlap in 
    # that direction -- this is equivalent to testing a minimal AABB around 
    # the triangle against the AABB 

    mini = v0[0]
    maxi = v0[0]
    if v1[0] < mini: mini = v1[0]
    if v1[0] > maxi: maxi = v1[0] 
    if v2[0] < mini: mini = v2[0] 
    if v2[0] > maxi: maxi = v2[0] 
    if (mini > half_size[0]) or (maxi<-half_size[0]): 
        return 0 

    mini = v0[1]
    maxi = v0[1]
    if v1[1] < mini: mini = v1[1]
    if v1[1] > maxi: maxi = v1[1] 
    if v2[1] < mini: mini = v2[1] 
    if v2[1] > maxi: maxi = v2[1] 
    if (mini > half_size[1]) or (maxi<-half_size[1]): 
        return 0 

    mini = v0[2]
    maxi = v0[2]
    if v1[2] < mini: mini = v1[2]
    if v1[2] > maxi: maxi = v1[2] 
    if v2[2] < mini: mini = v2[2] 
    if v2[2] > maxi: maxi = v2[2] 
    if (mini > half_size[2]) or (maxi<-half_size[2]): 
        return 0 

    normal[0] = (e0[1] * e1[2]) - (e0[2] * e1[1])
    normal[1] = (e0[2] * e1[0]) - (e0[0] * e1[2])
    normal[2] = (e0[0] * e1[1]) - (e0[1] * e1[0])
    d = -dot(normal, v0)

    #Bullet 2:
    # test if the box intersects the plane of the triangle
    # compute plane equation of triangle: normal*x+d=0
    for idx in range(3): 
        if (normal[idx] > 0.0):
            vmin[idx] = -half_size[idx]
            vmax[idx] = half_size[idx]

        else:
            vmin[idx] = half_size[idx]
            vmax[idx] = -half_size[idx]

    if (dot(&normal[0], &vmin[0]) + d) > 0.0:
        return 0
    if (dot(&normal[0], &vmax[0]) + d) >= 0.0:
        return 1
    return 0 

@cython.boundscheck(False) 
@cython.wraparound(False) 
def _ctestTriangleVoxelIntersection(voxCent, halfSize, tri):
    """
    Test if triangle intersects voxel. 
    WARNING: this function expects voxel half size, not full size, 
    as is the case for _cyfilterTriangles()
    """

    cdef float[:] vC = voxCent.flatten()
    cdef float[:] hS = halfSize.flatten()
    cdef float verts[3][3]
    for i in range(3):
        verts[0][i] = tri[0,i]
        verts[1][i] = tri[1,i]
        verts[2][i] = tri[2,i]

    return bool(triBoxOverlap(&vC[0], &hS[0], &verts[0]))


@cython.boundscheck(False) 
@cython.wraparound(False) 
def _cyfilterTriangles(tris, points, vC, vS):
    """
    Test if multiple triangles intersect voxel defined
    by centre vox_cent and half_size 
    """
    
    cdef Py_ssize_t t, a, b, c, i 
    cdef np.ndarray[char, ndim=1, cast=True] fltr = \
        np.zeros(tris.shape[0], dtype=bool)
    
    cdef np.ndarray[float, ndim=1] voxCent = vC.flatten() 
    cdef np.ndarray[float, ndim=1] halfSize = vS.flatten() 
    cdef float tri[3][3]
    for i in range(3):
        halfSize[i] = halfSize[i]/2 

    for t in range(tris.shape[0]):
        a = tris[t,0]
        b = tris[t,1]
        c = tris[t,2]

        for i in range(3):
            tri[0][i] = points[a,i]
            tri[1][i] = points[b,i]
            tri[2][i] = points[c,i]

        fltr[t] = triBoxOverlap(&voxCent[0], &halfSize[0], &tri[0])     

    return fltr 


@cython.boundscheck(False) 
@cython.wraparound(False) 
def _cytestManyRayTriangleIntersections(int[:,:] tris, float[:,:] points, start, int ax1, int ax2):
    """
    Test if a ray intersects triangles. The ray originates from the point 
    defined by start and travels along the dimension NOT specified by ax1 
    and ax2 (e.g 0 corresponds to X)
    """

    cdef np.ndarray[float, ndim=1] st = start.flatten()
    cdef np.ndarray[char, ndim=1, cast=True] fltr = np.zeros(tris.shape[0], dtype=np.bool)
    cdef Py_ssize_t t, a, b, c
    cdef float tri[3][3]

    for t in range(tris.shape[0]):
        a = tris[t,0]
        b = tris[t,1]
        c = tris[t,2]

        for i in range(3):
            tri[0][i] = points[a,i]
            tri[1][i] = points[b,i]
            tri[2][i] = points[c,i]

        fltr[t] = testRayTriangleIntersection(tri, &st[0], ax1, ax2)     

            fltr[t] = intersection 

    return fltr_array

@cython.boundscheck(False) 
@cython.wraparound(False) 
def _quick_cross(float[::] a, float[::] b):
    """
    Unsafe (no bounds check) cross product of a,b
    Args:
        a (np.array): 3 elements
        b (np.array): 3 elements
    Returns: 
        np.array, 3 elements
    """

    cdef float[::] out = np.empty(3, dtype=np.float32)
    with nogil: 
        out[0] = (a[1]*b[2]) - (a[2]*b[1])
        out[1] = (a[2]*b[0]) - (a[0]*b[2])
        out[2] = (a[0]*b[1]) - (a[1]*b[0])

    return out 