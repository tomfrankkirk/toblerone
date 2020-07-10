import numpy as np
cimport numpy as np
import cython

from libc.math cimport fabs


@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef char tribox_overlap(float[:] box_cent, 
                    float[:] half_size, 
                    float[:,:] verts) nogil:

    """
    Cython implementation of Tomas Akenine-Moller's triangle-box overlap test. 
    Reproduced with original comments by Tom Kirk, 2020. 

    AABB-triangle overlap test code                     
    by Tomas Akenine-Möller                             
    Function: int triBoxOverlap(float boxcenter[3],     
             float boxhalfsize[3],float triverts[3][3]);
    History:                                            
      2001-03-05: released the code in its first version
      2001-06-18: changed the order of the tests, faster
                                                        
    Acknowledgement: Many thanks to Pierre Terdiman for 
    suggestions and discussions on how to optimize code.
    Thanks to David Hunt for finding a ">="-bug!        
    """

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
cdef float dot(float a[3], float b[3]) nogil:
    cdef float out = ((a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]))
    return out 


@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef filterTriangles(int[:,:] tris, 
                         float[:,:] points, 
                         float[:] vox_cent, 
                         float[:] half_size):
    """
    Test if multiple triangles intersect voxel defined
    by centre vox_cent and half_size 
    """
    
    cdef Py_ssize_t t
    cdef np.ndarray[char, ndim=1, cast=True] fltr = np.zeros(tris.shape[0], dtype=np.bool)
    verts_array = np.empty((3,3), dtype=np.float32)
    cdef float[:,:] verts = verts_array

    for t in range(tris.shape[0]):
        with nogil:
            verts[0,:] = points[tris[t,0],:]
            verts[1,:] = points[tris[t,1],:]
            verts[2,:] = points[tris[t,2],:]
        fltr[t] = tribox_overlap(vox_cent, half_size, verts_array)   

    return fltr 


@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef test_ray_tris_intersection(int[:,:] tris, float[:,:] points, 
                                float[::] start, int ax1, int ax2):
    """
    Test if a ray intersects a group of triangles. With thanks to 
    Tim Coalson, this is a direct port of his HCP wb_command code. 

    Args: 
        tris (np.array): 2D array of int32 triangle indices. 
        points (np.array): 2D array of float32 triangle points. 
        start (np.array): 3-vector of float32, ray origin 
        ax1 (int): one of the axes (X1, Y2, Z3) ray does not travel along
        ax2 (int): the other axis ray does not travel along 

    Returns: 
        (np.array), 1D of bool, length equal to triangles
    """
    

    cdef np.ndarray[char, ndim=1, cast=True] fltr_array = \
        np.zeros(tris.shape[0], dtype=np.bool)
    cdef char[::] fltr = fltr_array
    verts_array = np.empty((3,3), dtype=np.float32)
    cdef float[:,:] verts = verts_array
    cdef Py_ssize_t t,i,j,ti,tj 
    cdef char intersection 

    with nogil: 

        for t in range(tris.shape[0]):
            verts[0,:] = points[tris[t,0],:]
            verts[1,:] = points[tris[t,1],:]
            verts[2,:] = points[tris[t,2],:]
            intersection = 0 
            j = 2 
            for i in range(3):

                # if one vertex is on one side of the point in the x direction, 
                # and the other is on the other side (equal case is treated as greater)
                if ((verts[i,ax1] < start[ax1]) != (verts[j,ax1] < start[ax1])): 

                    #reorient the segment consistently to get a consistent answer
                    if (verts[i,ax1] < verts[j,ax1]):
                        ti = i; tj = j;
                    else:
                        ti = j; tj = i;

                    # if the point on the line described by the two vertices with 
                    # the same x coordinate is above (greater y) than the test point
                    if (((verts[ti,ax2] - verts[tj,ax2]) / (verts[ti,ax1] - verts[tj,ax1])) 
                        * (start[ax1] - verts[tj,ax1]) + verts[tj,ax2] > start[ax2]):
                        intersection = not intersection # even/odd winding rule
    
                # consecutive vertices, does 2,0 then 0,1 then 1,2
                j = i

            fltr[t] = intersection 

    return fltr_array

@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef quick_cross(float[::] a, float[::] b):
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


def point_groups_intersect(list grps, np.ndarray[np.int32_t, ndim=2] tris):
    """
    Check if a group 
    """

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