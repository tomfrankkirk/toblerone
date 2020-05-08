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

    return fltr 


@cython.boundscheck(False) 
@cython.wraparound(False) 
def quick_cross(np.ndarray[np.float32_t, ndim=1] a, np.ndarray[np.float32_t, ndim=1] b):

    cdef out = np.empty(3, dtype=np.float32)
    out[0] = (a[1]*b[2]) - (a[2]*b[1])
    out[1] = (a[2]*b[0]) - (a[0]*b[2])
    out[2] = (a[0]*b[1]) - (a[1]*b[0])

    return out 


@cython.boundscheck(False) 
@cython.wraparound(False) 
def normal_to_vector(np.ndarray[np.float32_t, ndim=1] a):

    cdef out = np.zeros(3, dtype=np.float32)    
    if np.abs(a[2]) < np.abs(a[0]):
        out[0] = a[1] 
        out[1] = -a[0]
    else:
        out[1] = -a[2]
        out[2] = a[1]

    return out 


def point_groups_intersect(list grps, np.ndarray[np.int32_t, ndim=2] tris):

    cdef Py_ssize_t g, h
    for g in range(len(grps)):
        for h in range(g + 1, len(grps)): 
            if np.intersect1d(tris[grps[g],:], tris[grps[h],:]).any(): 
                return True 

    return False 


def separate_point_clouds(np.ndarray[np.int32_t, ndim=2] tris):
    """Separate patches of a surface that intersect a voxel into disconnected
    groups, ie, point clouds. If the patch is cointguous within the voxel
    a single group will be returned.
    
    Args: 
        tris: n x 3 matrix of triangle indices into a points matrix

    Returns: 
        list of m arrays representing the point clouds, each of which is 
            list of row numbers into the given tris matrix 
    """

    if not tris.shape[0]:
        return [] 

    cdef Py_ssize_t t, g, h
    cdef list groups = [] 
    cdef bint newGroupNeeded, didMerge

    for t in range(tris.shape[0]):

        # If any node of the triangle is contained within the existing
        # groups, then append to that group. Assume new group needed
        # until proven otherwise
        newGroupNeeded = True 
        for g in range(len(groups)):
            if np.in1d(tris[t,:], tris[groups[g],:]).any():
                newGroupNeeded = False
                break 
        
        # Append triangle to existing group, using the break-value of g
        if not newGroupNeeded:
            groups[g].append(t)
        
        # New group needed
        else: 
            groups.append([t])

    # Merge groups that intersect 
    if len(groups) > 1: 
        while point_groups_intersect(groups, tris): 
            didMerge = False 

            for g in range(len(groups)):
                if didMerge: break 

                for h in range(g + 1, len(groups)):
                    if didMerge: break

                    if np.intersect1d(tris[groups[g],:], tris[groups[h],:]).any():
                        groups[g] = groups[g] + groups[h]
                        groups.pop(h)
                        didMerge = True  

    # Check for empty groups 
    for g in range(len(groups)):
        assert len(groups[g]), 'Empty group remains after merging'
    
    return groups 