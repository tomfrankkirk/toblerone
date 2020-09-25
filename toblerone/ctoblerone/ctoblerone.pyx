import numpy as np
cimport numpy as np
import cython

from libc.math cimport fabs


# External function imports from ../src directory
cdef extern from "tribox.h":
    char triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3])


cdef extern from "ctoblerone.h":
    char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2)


# TODO: could re-optimise by passing vx, vy, vz instead of vox_cent
# - ensure memory integrity? Or re-implement those functions in cython. 

@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef _ctestTriangleVoxelIntersection(float[:] vox_cent, 
                                      float[:] halfSize, 
                                      float[:,:] tri):
    """
    Test if triangle intersects voxel. 

    WARNING: this function expects voxel half size, not full size, 
    as is the case for _cyfilterTriangles()
    """

    cdef float verts[3][3]
    cdef float vc[3]
    verts = np.ascontiguousarray(tri)
    vc = np.ascontiguousarray(vox_cent)

    return bool(triBoxOverlap(&vc[0], &halfSize[0], &verts[0]))


@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef _cyfilterTriangles(int[:,:] tris, 
                         float[:,:] points, 
                         float[:] vox_cent, 
                         float[:] half_size):
    """
    Test if multiple triangles intersect voxel defined
    by centre vC and full size vS
    """
    
    cdef Py_ssize_t t, i, a, b, c
    cdef np.ndarray[char, ndim=1, cast=True] fltr = np.zeros(tris.shape[0], dtype=np.bool)
    cdef float verts[3][3]
    cdef float vc[3]

    vc = np.ascontiguousarray(vox_cent)

    for t in range(tris.shape[0]):
        with nogil:
            a = tris[t,0]
            b = tris[t,1]
            c = tris[t,2]

            verts[0][0] = points[a,0]
            verts[0][1] = points[a,1]
            verts[0][2] = points[a,2]

            verts[1][0] = points[b,0]
            verts[1][1] = points[b,1]
            verts[1][2] = points[b,2]

            verts[2][0] = points[c,0]
            verts[2][1] = points[c,1]
            verts[2][2] = points[c,2]

        fltr[t] = triBoxOverlap(&vc[0], &half_size[0], &verts[0])     

    return fltr 


@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef _cytestManyRayTriangleIntersections(int[:,:] tris, 
                                          float[:,:] points, 
                                          float[:] start, 
                                          int ax1, 
                                          int ax2):
    """
    Test if a ray intersects triangles. The ray originates from the point 
    defined by start and travels along the dimension NOT specified by ax1 
    and ax2 (e.g 0 corresponds to X)
    """

    cdef np.ndarray[char, ndim=1, cast=True] fltr = np.zeros(tris.shape[0], dtype=np.bool)
    cdef Py_ssize_t t, a, b, c
    cdef float verts[3][3]

    for t in range(tris.shape[0]):
        with nogil: 
            a = tris[t,0]
            b = tris[t,1]
            c = tris[t,2]

            verts[0][0] = points[a,0]
            verts[0][1] = points[a,1]
            verts[0][2] = points[a,2]

            verts[1][0] = points[b,0]
            verts[1][1] = points[b,1]
            verts[1][2] = points[b,2]

            verts[2][0] = points[c,0]
            verts[2][2] = points[c,2]
            verts[2][1] = points[c,1]

        fltr[t] = testRayTriangleIntersection(verts, &start[0], ax1, ax2)     

    return fltr 


@cython.boundscheck(False) 
@cython.wraparound(False) 
def quick_cross(float[::] a, float[::] b):
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


@cython.boundscheck(False) 
@cython.wraparound(False) 
def normal_to_vector(float[::] a):
    """
    Return a new vector normal to input 

    Args: 
        a (np.array): 3 elements (no bounds check)

    Returns: 
        (np.array): 3 elements 
    """

    cdef float[::] out = np.zeros(3, dtype=np.float32) 
    with nogil:
        if fabs(a[2]) < fabs(a[0]):
            out[0] = a[1] 
            out[1] = -a[0]
        else:
            out[1] = -a[2]
            out[2] = a[1]

    return out 
    