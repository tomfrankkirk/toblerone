import numpy as np
cimport numpy as np
import cython


cdef extern from "tribox.h":
    char triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3]); 

cdef extern from "ctoblerone.h":
    void ctestManyRayTriangleIntersections(const int* triangles, const float* points, const float* start, const int nTris, const int ax1, const int ax2, char* results)

    int ray_wrapper(float s1, float s2, float s3, 
                   float tv11, float tv12, float tv13,
                   float tv21, float tv22, float tv23,
                   float tv31, float tv32, float tv33, 
                   int ax1, int ax2)

    void testTrianglesVoxelIntersection(int* triangles, float* points, int nTris, float* voxCent, float* voxHalfSize, char* results)
    
    char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2)

    void test(const float* points, const int* tris, int nTris, const float* testPnt, const float* ray, int normDF, float* output)


@cython.boundscheck(False) 
@cython.wraparound(False) 
def _ctestTriangleVoxelIntersection(voxCent, voxSize, tri):

    cdef float[:] vC = voxCent.flatten()
    cdef float[:] halfSize = voxSize
    for i in range(3):
        halfSize[i] = halfSize[i]/2 
    cdef float verts[3][3]
    for i in range(3):
        verts[0][i] = tri[0,i]
        verts[1][i] = tri[1,i]
        verts[2][i] = tri[2,i]

    return triBoxOverlap(&vC[0], &halfSize[0], verts)

@cython.boundscheck(False) 
@cython.wraparound(False) 
def _cfilterTriangles(tris, 
                      points, 
                      voxCent, 
                      voxSize):

    cdef np.ndarray[char, ndim=1, cast=True] fltr = \
        np.zeros(tris.shape[0], dtype=np.bool)
    cdef np.ndarray[int, ndim=1] ts = tris.flatten()
    cdef np.ndarray[np.float32_t, ndim=1] ps = points.flatten()
    cdef int nTris = tris.shape[0]
    cdef float[:] halfSize = voxSize
    for i in range(3):
        halfSize[i] = halfSize[i]/2 
    cdef float[:] cent = voxCent.flatten()

    testTrianglesVoxelIntersection(&ts[0], &ps[0], nTris, 
        &cent[0], &halfSize[0], &fltr[0]); 

    return fltr



@cython.boundscheck(False) 
@cython.wraparound(False) 
def _cytestManyRayTriangleIntersections(int[:,:] tris, float[:,:] points, float[:] start, int ax1, int ax2):

    cdef np.ndarray[char, ndim=1, cast=True] fltr = \
        np.zeros(tris.shape[0], dtype=np.bool)

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

        fltr[t] = testRayTriangleIntersection(tri, &start[0], ax1, ax2)     

    return fltr 


@cython.boundscheck(False) 
@cython.wraparound(False) 
def _ctestManyRayTriangleIntersections(tris, points, float[:] start,
                                   int ax1, int ax2):

    cdef int[:] ts = tris.flatten()
    cdef float[:] ps = points.flatten()
    cdef int nTris = tris.shape[0]
    cdef np.ndarray[char, ndim=1, cast=True] fltr = \
        np.zeros(nTris, dtype=np.bool)

    ctestManyRayTriangleIntersections(&ts[0], &ps[0], &start[0], 
        nTris, ax1, ax2, &fltr[0])
    return fltr


# Cython method: loop over triangles in cython and do the maths there. 
def _cyfindRayTriPlaneIntersections(float[:,:] points, int[:,:] tris, 
    float[:] testPnt, float[:] ray, int normDF):

    cdef Py_ssize_t t, a, b, c, d 
    cdef Py_ssize_t nTris = tris.shape[0]
    cdef float[:] e1 = np.empty(3, dtype=np.float32)
    cdef float[:] e2 = np.empty(3, dtype=np.float32)
    cdef float[:] normal = np.empty(3, dtype=np.float32)
    cdef float[:] toPoint = np.empty(3, dtype=np.float32)
    cdef float dotRN, mu
    cdef np.ndarray[float, ndim=1] mus = np.empty(nTris, dtype=np.float32) 

    for t in range(nTris):

        toPoint = points[tris[t,0]:] - testPnt
        normal = np.cross(np.subtract(points[tris[t,2],:], points[tris[t,0],:]), 
            np.subtract(points[tris[t,1],:], points[tris[t,0],:]))
        dotRN = np.dot(ray, normal)

        if dotRN:
            mus[t] = np.dot(toPoint, normal) / dotRN
        else:
            mus[t] = 0.0

    return mus


# Pure C method: loop and do vector maths in C. 
def _cfindRayTriPlaneIntersections(points, tris, float[:] testPnt, float[:] ray, normDF):

    cdef int nTris = tris.shape[0]
    cdef float[:] ps = points.flatten()
    cdef int[:] ts = tris.flatten()
    cdef np.ndarray[float, ndim=1] fltr = \
        np.zeros(nTris, dtype=np.float32)

    test(&ps[0], &ts[0], nTris, &testPnt[0], &ray[0], normDF, &fltr[0])
    return fltr 
    











