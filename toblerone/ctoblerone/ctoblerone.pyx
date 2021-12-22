import numpy as np
cimport numpy as np
import cython

# External function imports from ../src directory
cdef extern from "tribox.h":
    char triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3])


cdef extern from "ctoblerone.h":
    char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2)


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
    by centre vC and full size vS
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
    cdef np.ndarray[char, ndim=1, cast=True] fltr = np.zeros(tris.shape[0], dtype=bool)
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

    return fltr 


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