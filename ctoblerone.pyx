import numpy as np
cimport numpy as np
import cython


cdef extern from "tribox.h":
    char triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3])

cdef extern from "ctoblerone.h":
    char testRayTriangleIntersection(const float tri[3][3], const float start[3], int ax1, int ax2)


@cython.boundscheck(False) 
@cython.wraparound(False) 
def _ctestTriangleVoxelIntersection(voxCent, halfSize, tri):
    """WARNING: this function expects voxel half size, not full size, 
    as is the case for _cyfilterTriangles()"""

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

    # cdef np.ndarray[int, ndim=1] ts = tris.flatten()
    # cdef np.ndarray[float, ndim=1] ps = points.flatten()
    cdef np.ndarray[float, ndim=1] st = start.flatten()

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

        fltr[t] = testRayTriangleIntersection(tri, &st[0], ax1, ax2)     

    return fltr 
















# The below are in progress 



cdef cynormalToVector(float[:] vec):
    if np.abs(vec[2]) < np.abs(vec[0]):
        normal = np.array([vec[1], -vec[0], 0])
    else:
        normal = np.array([0, -vec[2], vec[1]])

    return normal.astype(np.float32)

cdef cyquickCross(float[:] a, float[:] b):
    return np.array([
        (a[1]*b[2]) - (a[2]*b[1]),
        (a[2]*b[0]) - (a[0]*b[2]), 
        (a[0]*b[1]) - (a[1]*b[0])], dtype = np.float32)    


cdef cydotVectorAndMatrix(float[:] vec, float[:,:] mat):
    return np.sum(np.multiply(vec, mat), axis=1)

cdef intersections3D(testPnt, ray, ps, ts, xps):

    cdef float[:] d2 = cynormalToVector(ray)
    cdef float[:] d1 = cyquickCross(d2, ray)

    cdef float[:] lmbda = cydotVectorAndMatrix(ray, ps)
    cdef np.ndarray[float, ndim=2] onPlane = \
        np.subtract(np.subtract(ps, np.outer(lmbda, ray)), testPnt)

    cdef np.ndarray[float, ndim=2] onPlane2D = np.array([
        cydotVectorAndMatrix(d1, onPlane), 
        cydotVectorAndMatrix(d2, onPlane),
        np.zeros(onPlane.shape[0])], dtype=np.float32)

    cdef np.ndarray[float, ndim=1] start = np.zeros(3, dtype=np.float32)
    cdef np.ndarray[char, ndim=1, cast=True] fltr = \
        _cytestManyRayTriangleIntersections(ts, onPlane2D.T, 
        start, 0, 1)

    cdef np.ndarray[int] intersects = np.flatnonzero(fltr).astype(np.int32)

    cdef Py_ssize_t t, c 
    cdef float[:] mus = np.zeros(intersects.size, dtype=np.float32)

    for t in intersects: 
        mus[c] = np.multiply(np.subtract(ps[ts[t,0],:], testPnt), xps[t,:]) / np.dot(ray, xps[t,:])
        c += 1   

    return mus 


def redTest(np.ndarray[float, ndim=2] testPnts, 
            np.ndarray[float, ndim=2] ps, 
            np.ndarray[int, ndim=2] ts, 
            np.ndarray[float, ndim=2] xps, 
            float[:] rootPoint, 
            char flip):

    cdef np.ndarray[char, ndim=1, cast=True] flags = \
        np.zeros(testPnts.shape[0], dtype=bool)

    cdef np.ndarray[float, ndim=2] rays = \
        np.subtract(rootPoint, testPnts)

    cdef np.ndarray[float] intMus 
        
    cdef np.ndarray[char, ndim=1, cast=True] inRange 

    cdef Py_ssize_t p 

    for p in range(testPnts.shape[0]):

        if not np.any(rays[p,:]):
            flags[p] = True 
            continue 
        
        else: 

            intMus = intersections3D(testPnts[p,:], rays[p,:], ps, 
                ts, xps)

            if intMus.shape[0]:
                inRange = np.logical_and(intMus < 1, intMus > 0)

                if not np.any(inRange):
                    flags[p] = True 
                
                else: 
                    flags[p] = not(inRange.size % 2)
                    
            else: 
                flags[p] = True 

    if flip:
        return ~flags
    return flags 