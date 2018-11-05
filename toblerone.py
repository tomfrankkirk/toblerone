# Toblerone: partial volume estimation on the cortical ribbon

import copy
import warnings
import functools
import itertools
import multiprocessing
import collections
import os 
import os.path as op
import pickle

import numpy as np 
import nibabel
import nibabel.freesurfer.io
import nibabel.nifti2
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError 
from ctoblerone import _ctestTriangleVoxelIntersection, _cfilterTriangles
from ctoblerone import _cfindRayTriPlaneIntersections, _cytestManyRayTriangleIntersections, _ctestManyRayTriangleIntersections

__NWORKERS__ = multiprocessing.cpu_count() 


def _affineTransformPoints(points, affine):
    """Apply affine transformation to set of points.

    Args: 
        points: n x 3 matrix of points to transform
        affine: 4 x 4 matrix for transformation

    Returns: 
        transformed copy of points 
    """

    # Add 1s on the 4th column, transpose and multiply, 
    # then re-transpose and drop 4th column  
    transfd = np.ones((points.shape[0], 4))
    transfd[:,0:3] = points
    transfd = np.matmul(affine, transfd.T).astype(np.float32)
    return (transfd[0:3,:]).T



def _triangleNormal(tri, normDirFlag):
    """Normal to the plane of a triangle. 

    Args: 
        tri: 3 x 3 matrix of vertices, XYZ along cols 
        normDirFlag: 1/-1 to denote ordering (1 default)

    Returns:
        normal: unit inwards normal to the surface
    """

    normal = normDirFlag * np.cross(tri[2,:] - tri[0,:], 
        tri[1,:] - tri[0,:])
    return normal / np.linalg.norm(normal)



def _generateVoxelCorners(cent, size):
    """Produce 8 x 3 matrix of voxel vertices. 

    Args: 
        cent: centre coordinate of voxel
        size: vector of dimensions in xyz 
    """

    bounds = _getVoxelBounds(cent, size)
    return np.array([
        [bounds[0,0], bounds[1,0], bounds[2,0]],
        [bounds[0,1], bounds[1,0], bounds[2,0]],
        [bounds[0,0], bounds[1,1], bounds[2,0]],
        [bounds[0,1], bounds[1,1], bounds[2,0]],
        [bounds[0,0], bounds[1,0], bounds[2,1]],
        [bounds[0,1], bounds[1,0], bounds[2,1]],
        [bounds[0,0], bounds[1,1], bounds[2,1]],
        [bounds[0,1], bounds[1,1], bounds[2,1]],
    ])



def _filterPoints(points, voxCent, voxSize):
    """Logical filter of points that are in the voxel specified.

    Args: 
        points: n x 3 matrix of points
        voxCent: vector for voxel centre coordinate 
        voxSize: vector of voxel dimensions in XYZ

    Returns: 
        n x 1 logical vector 
    """

    return np.all(np.less_equal(np.abs(points - voxCent), voxSize/2), \
        axis=1)



def _getVoxelBounds(voxCent, voxSize):
    """Return 3x2 vector of bounds for the voxel, 
    arranged [xMin xMax; yMin yMax; ...]
    """

    return np.array([
        [ voxCent[d] + (i * 0.5 * voxSize[d]) for i in [-1, 1] ]
        for d in range(3) ])



def _dotVectorAndMatrix(vec, mat):
    """Row-wise dot product of a vector and matrix. 
    Returns a vector with the same number of rows as
    the matrix with the dot prod of that row with the
    vector in each row"""

    return np.sum(mat * vec, axis=1)



def _rebaseTriangles(points, tris):
    """Re-express a patch of a larger surface as a new points and triangle matrix pair, indexed from 0. Useful for reducing computational complexity when working with a small
    patch of a surface where only a few nodes in the points 
    array are required by the triangles matrix. 

    Args: 
        points: n x 3 matrix of surface nodes
        tris: t x 3 matrix of triangle node indices. 
    
    Returns: 
        (localPoints, localTris) tuple of re-indexed points/tris. 
    """

    localPoints = np.empty((0, 3), dtype=np.float32)
    localTris = np.zeros(tris.shape, dtype=np.int32)
    pointsLUT = []

    for t in range(tris.shape[0]):
        for v in range(3):

            # For each vertex of each tri, check if we
            # have already processed it in the LUT
            vtx = tris[t,v]
            idx = np.argwhere(pointsLUT == vtx)
            if len(idx) > 1: 
                print("sdjk")

            # If not in the LUT, then add it and record that
            # as the new position. Write the missing vertex
            # into the local points array
            if not idx.size:
                pointsLUT.append(vtx)
                idx = len(pointsLUT) - 1
                localPoints = np.vstack([localPoints, points[vtx,:]])

            # Update the local triangle
            localTris[t,v] = idx

    return (localPoints, localTris)



def _sub2ind(dims, subs): 
    """Equivalent of MATLAB's sub2ind function
    
    Args:
        dims: tuple of ints for matrix dimensions
        subs: tuple of int arrays, one for each dim
    
    Returns: 
        list of flat indices 
    """

    return np.ravel_multi_index(subs, dims)



def _ind2sub(dims, inds):
    """Equivalent of MATLAB's ind2sub function
    
    Args:
        dims: tuple of ints for matrix dimensions
        inds: int array of flat indices
    
    Returns: 
        tuple of int subscript arrays, one for each dimension 
    """

    return np.unravel_index(inds, dims)



def _getVoxelSubscripts(voxList, imgSize):
    """Convert linear voxel indices into IJK subscripts
    within a matrix of imgSize. Returns an n x 3 matrix
    """
    
    return np.array(list(map(lambda v: _ind2sub(imgSize, v), voxList)))


def _separatePointClouds(tris):
    """Separate patches of a surface that intersect a voxel into disconnected
    groups, ie, point clouds. If the patch is cointguous within the voxel
    a single group will be returned.
    
    Args: 
        tris: n x 3 matrix of triangle indices into a points matrix

    Returns: 
        list of m arrays representing the point clouds, each of which is 
            list of row numbers into the given tris matrix 
    """

    # Local functions --------------------

    def __pointGroupsIntersect(grps): 
        """Break as soon as overlap is found. Brute force approach."""
        for g in range(len(grps)):
            for h in range(g + 1, len(grps)): 
                if np.any(np.intersect1d(grps[g], grps[h])):
                    return True 

        return False 

    # Main function ---------------------

    groups = [] 

    if not tris.shape[0]:
        return [] 

    for t in range(tris.shape[0]):

        # If any node of the triangle is contained within the existing
        # groups, then append to that group. Assume new group needed
        # until proven otherwise
        newGroupNeeded = True 
        for g in range(len(groups)):
            if np.any(np.in1d(tris[t,:], tris[groups[g],:])):
                newGroupNeeded = False
                break 
        
        # Append triangle to existing group, using the break-value of g
        if not newGroupNeeded:
            groups[g].append(t)
        
        # New group needed
        else: 
            groups.append([t])

    # Check for empty groups 
    assert all(map(lambda g: len(g) > 0, groups))

    # Merge groups that intersect 
    if len(groups) > 1: 
        assert not __pointGroupsIntersect(groups)
        while __pointGroupsIntersect(groups): 
            for g in range(len(groups)):
                for h in range(g + 1, len(groups)):
                    if np.any(np.intersect1d(groups[g], groups[h])):
                        groups[g] = groups[g] + groups[h]
                        groups.pop(h)

    return groups 



def _distributeObjects(objs, nWorkers):
    """Distribute a set of objects amongst n workers.
    Returns a list of length nWorkers with each workers objects"""

    chunkSize = np.floor(len(objs) / nWorkers).astype(np.int32)
    chunks = [] 

    for n in range(nWorkers):
        if n != nWorkers - 1: 
            chunks.append(objs[n * chunkSize : (n+1) * chunkSize])
        else:
            chunks.append(objs[n * chunkSize :])

    assert sum(map(len, chunks)) == len(objs), \
        "Distribute objects error: not all objects distributed"        

    return chunks 



def _formAssociations(points, tris, FoVsize):
    """Identify which triangles of a surface each voxel. This reduces the
    number of tests that must be performed in the main Toblerone algorithm.

    Args: 
        points: p x 3 matrix of surface nodes
        tris: t x 3 matrix of triangle node indices
        FoV: 1 x 3 vector of image dimensions (units of voxels) reqiured to
            fully enclose the surface

    Returns: 
        (associations, LUT) tuple of associations (a list of lists) and 
            a LUT used to index between associations table and vox index. 
    """

    global __NWORKERS__

    # Check for negative coordinates: these should have been sripped. 
    if np.round(np.min(points)) < 0: 
        raise RuntimeError("formAssociations: negative coordinate found")

    if np.any(np.round(np.max(points, axis=0)) >= FoVsize): 
        raise RuntimeError("formAssociations: coordinate outside FoV")

    chunks = _distributeObjects(np.arange(tris.shape[0]), __NWORKERS__)
    workerFunc = functools.partial(_formAssociationsWorker, tris, \
        points, FoVsize)

    with multiprocessing.Pool(__NWORKERS__) as p:
        allResults = p.map(workerFunc, chunks, chunksize=1)
    # allResults = list(map(workerFunc, chunks))

    # Flatten results down from each worker. Iterate only over the keys
    # present in each dict. Use a default dict of empty [] to hold results
    associations = collections.defaultdict(list)
    for res in allResults: 
        for k in res.keys():
            associations[k] = associations[k] + res[k]

    return dict(associations)



def _formAssociationsWorker(tris, points, FoVsize, triInds):
    workerResults = {}

    for t in triInds:

        # Get vertices of triangle in voxel space (to nearest vox)
        tri = points[tris[t,:]]
        minVs = np.floor(np.min(tri, axis=0)).astype(np.int16)
        maxVs = np.ceil(np.max(tri, axis=0)).astype(np.int16)

        # Loop over the neighbourhood voxels of this bounding box
        for I in range(minVs[0], maxVs[0] + 1):
            for J in range(minVs[1], maxVs[1] + 1):
                for K in range(minVs[2], maxVs[2] + 1):

                    ind = _sub2ind(FoVsize, (I,J,K))

                    if _ctestTriangleVoxelIntersection([I,J,K], \
                        np.ones(3, dtype=np.int8), tri):

                        if ind in workerResults:
                            workerResults[ind].append(t)
                        else: 
                            workerResults[ind] = [t]
    
    return workerResults



def _voxeliseSurfaces(FoVsize, inPs, inTris, outPs, outTris, inAssocs, \
    inLUT, outAssocs, outLUT, normDF):
    """Fill the volume contained within the given surfaces using orthographic
    projection along the X,Y or Z dimension. 
    See "Simplification and repair of polygonal models using volumetric techniques", Nooruddin & Turk, 2003 for an overview. 

    Args:
        inPs/outPs: p x 3 matrices of surface nodes
        inTris/outTris: t x 3 matrices of triangle node indices
        inAssocs/outAssocs: triangle associations, generated via formAssocs() 
        normDF: 1/-1 for triangle node ordering (1 default)

    Returns: 
        mask of dimensions FoV size, 1 = GM, 2 = WM, 3 = CSF, 4 = special
            error case of outside the pial but inside the white surface 
            (handled elsewhere)
    """


    dim = np.argmax(np.argsort(FoVsize))
    with multiprocessing.Pool(__NWORKERS__) as p: 
        def __createMask(ts, ps, assocs, LUT):
            return _voxeliseSurfaceAlongDimension(FoVsize, dim, \
                ts, ps, assocs, LUT, normDF)

        masks = p.starmap(__createMask, [[inTris, outTris], [inPs, outPs], \
            [inAssocs, outAssocs], [inLUT, outLUT]], chunksize=1)

    inMask = masks[0]; outMask = masks[1]
    mask = np.zeros(FoVsize, dtype=np.int8)
    mask[inMask & outMask] = 2
    mask[outMask & ~inMask] = 1
    mask[~outMask & ~inMask] = 3

    # Class 4 is used for anatomically impossible voxels, caused by 
    # imperfections in the FS surfaces (outside pial but inside white)
    # We will handle these cases elsewhere
    mask[~outMask & inMask] = 4
    return mask 



def _voxeliseSurfaceAlongDimension(FoVsize, dim, tris, points, assocs, LUT, normDF):
    """Fill the volume contained within the surface by projecting rays
    in a given direction. See "Simplification and repair of polygonal models 
    using volumetric techniques", Nooruddin & Turk, 2003 for an overview of 
    the conceptual approach taken.

    Args: 
        FoVsize: 1 x 3 vector of FoV dimensions in which surface is enclosed
        dim: int 0/1/2 dimension along which to project rays
        tris: t x 3 matrix of triangle node indices
        points: p x 3 matrix of surface nodes
        assocs: list of lists recording the triangle associations for voxels
        LUT: vector look up table between assocs and linear vox indices. 
        normDF: 1/-1 for triangle node ordering (1 default)

    Returns: 
        logical array of FoV size, true where voxel is contained in surface
    """

    # Initialsie empty mask, and loop over the OTHER dims to the one specified. 
    mask = np.zeros(FoVsize, dype=bool)
    otherDims = [ (dim+1)%3, (dim+2)%3 ]

    # Define a ray 
    ray = np.zeros((1,3))
    ray[dim] = 1

    for d1 in range(FoVsize[otherDims[0]]):
        for d2 in range(FoVsize[otherDims[1]]):

            # Defined the start/end of the ray and gather all 
            # linear indices of voxels along the ray
            IJK = np.zeros((FoVsize[dim], 3), dtype=np.int32)
            IJK[:,dim] = np.arange(0, FoVsize[dim])
            IJK[:,otherDims[0]] = d1
            IJK[:,otherDims[1]] = d2
            startPoint = np.ones((1,3))
            startPoint[otherDims] = [d1, d2]
            voxRange = _sub2ind(FoVsize, (IJK[:,1], IJK[:,2], IJK[:,3]))

            # Find all associated triangles lying along this ray
            # and test for intersection
            triNums = np.vstack(assocs[np.in1d(LUT, voxRange)])

            if len(triNums):
                inTs = list(set(triNums))
                intersectionMus = _findRayTriangleIntersections2D(startPoint, \
                    ray, tris[inTs,:], points, dim, normDF)

                if not intersectionMus.shape[0]:
                    continue
                
                # If intersections were found, perform a parity test. 
                # Any ray should make an even number of intersections
                # as it crosses from -ve to +ve infinity
                if (intersectionMus.shape[0] % 2):
                    raise RuntimeError("fillSurfaceAlongDimension: \
                        even number of intersections found. Does the FoV \
                        cover the full extents of the surface")

                # Calculate points of intersection along the ray
                sorted = np.argsort(intersectionMus)
                intDs = startPoint[dim] + (intersectionMus[sorted] * ray[dim])

                # Assignment. All voxels before the first point of intersection
                # are outside. The mask is already zeroed for these. All voxels
                # between point 1 and n could be in or out depending on parity
                for i in range(1, len(sorted)+1):

                    # Starting from infinity, all points between an odd numbered
                    # intersection and the next even one are inside the mask 
                    if (i % 2) & (i+1 <= len(sorted)):
                        indices = (IJK[:,dim] > intDs[i] & IJK[:,dim] < intDs(i+1))
                        mask[voxRange[indices]] = 1
                
                # All voxels beyond the last point of intersection are also outside. 

    return mask 



def _findRayTriangleIntersections2D(testPnt, ray, tris, points, axis, normDF):
    """Find intersections between a ray and a patch of surface, testing along
    one coordinate axis only (XYZ). As the triangle intersection test used within
    is 2D only, triangles are first projected down onto a 2D plane normal to the 
    test ray and then tested for intersection. This is intended to be used for
    voxeliseSurfaces(), for ray testing in the general 3D case use
    findRayTriangleIntersections3D() instead. 

    Args: 
        testPnt: 1 x 3 vector for ray origin
        ray: 1 x 3 vector of ray direction 
        triangles: m x 3 matrix of triangle node indices 
        points: n x 3 matrix of surface nodes for triangles
        axis: 0 for X, 1 for Y, 2 for Z, along which to test 
        normDF: 1/-1 for triangle node ordering (1 default)

    Returns: 
        1 x j vector of multipliers along the ray to the points of intersection
    """

    # Filter triangles that intersect with this ray 
    fltr = _pTestManyRayTriangleIntersections(tris, points, 
        testPnt, (axis+1)%3, (axis+2)%3)

    # And find the multipliers for those that do intersect 
    mus = _findRayTriPlaneIntersections(points, tris[fltr,:], \
        testPnt, ray, normDF)

    return mus




def _findRayTriPlaneIntersections(points, tris, testPnt, ray, normDF):
    """Find points of intersection between a ray and the planes defined by a
    set of triangles. As these points may not lie within their respective 
    triangles, these results must be further filtered using 
    vectorTestForRayTriangleIntersection() to identify genuine intersections.

    Args:
        points: p x 3 matrix of surface nodes
        tris: t x 3 matrix of triangle node indices 
        testPnt: 1 x 3 vector of ray origin
        ray: 1 x 3 direction vector of ray
        normDF: 1/-1 for triangle node ordering (1 default)

    Returns: 
        1 x j vector of multipliers along the ray to the points of intersection
    """

    # Compute edge vectors of all triangles and their planar normals
    e1 = points[tris[:,2],:] - points[tris[:,0],:]
    e2 = points[tris[:,1],:] - points[tris[:,0],:]
    normals = normDF * np.cross(e1, e2, axis=1) 

    # Compute dot of each normal with the ray
    dotRN = _dotVectorAndMatrix(ray, normals)

    # And calculate the multiplier.
    mu = np.sum((points[tris[:,0],:] - testPnt) * normals, axis=1) / dotRN 

    return mu 



def _normalToVector(vec):
    """Return a normal to the given vector"""
    vec = vec / np.linalg.norm(vec)

    if vec[2] < vec[0]:
        normal = np.array([vec[1], -vec[0], 0])
    else:
        normal = np.array([0, -vec[2], vec[1]])

    # The above can fail if unit cardinal vecs are passed in 
    if np.all(normal == 0):
        if np.array_equal(vec, [0, 0, -1]):
            normal = np.array([0, 1, 0])
        elif np.array_equal(vec, [-1, 0, 0]):
            normal = np.array([0, 1, 0])
        elif np.array_equal(vec, [0, -1, 0]):
            normal = np.array([1, 0, 0])
        else: 
            raise RuntimeError('Normal?')
        
    assert np.dot(normal, vec) == 0, 'Normals are not normal'
    return normal 



def _findRayTriangleIntersections3D(testPnt, ray, tris, points, normDF):
    """Find points of intersection between a ray and a surface. Triangles
    are projected down onto a 2D plane normal to the ray and then tested for
    intersection using findRayTriangleIntersections2D(). See: 
    https://stackoverflow.com/questions/2500499/howto-project-a-planar-polygon-on-a-plane-in-3d-space
    https://stackoverflow.com/questions/11132681/what-is-a-formula-to-get-a-vector-perpendicular-to-another-vector

    Args: 
        testPnt: 1 x 3 vector for origin of ray
        ray: 1 x 3 direction vector of ray
        tris: t x 3 matrix of triangle node indices
        points: p x 3 matrix of surface nodes
        normDF: 1/-1 for triangle node ordering (1 default)

    Returns: 
        1 x j vector of distance multipliers along the ray at each point
        of intersection 
    """
    
    # Main function 

    # Intersection is tested using Tim Coalson's adaptation of PNPOLY for careful
    # testing of intersections between infinite rays and points. As TC's adaptation
    # is a 2D test only (with the third dimension being the direction of ray
    # projection), triangles are flattened into 2D before testing. This is done by
    # projecting all triangles onto the plane defined by the ray (acting as planar
    # normal) and then testing for ray intersection (where the ray now represents
    # the Z direction in this new projected space) amongst all the triangles in
    # dimensions 1 and 2 (XY). Define a new coordinate system (d unit vectors) 
    # with d3 along the ray, d2 and d1 in plane.
    d3 = ray / np.linalg.norm(ray)
    d2 = _normalToVector(ray)
    d1 = np.cross(d2,d3)

    # Calculate the projection of each point onto the direction vector of the
    # surface normal. Then subtract this component off each to leave their position
    # on the plane and shift coordinates so the test point is the origin.
    lmbda = _dotVectorAndMatrix(d3, points)
    onPlane = points - np.outer(lmbda, d3)
    onPlane = onPlane - testPnt

    # Re-express the points in 2d planar coordiantes by evaluating dot products with
    # the d2 and d3 in-plane orthonormal unit vectors
    onPlane2d = np.zeros(onPlane.shape, dtype=np.float32)
    onPlane2d[:,0] = _dotVectorAndMatrix(d1, onPlane)
    onPlane2d[:,1] = _dotVectorAndMatrix(d2, onPlane)

    # Now perform the test 
    start = np.zeros(3, dtype=np.float32)
    fltr = _pTestManyRayTriangleIntersections(tris, onPlane2d, start, 0, 1)

    # For those trianglest that passed, calculate multiplier to point of 
    # intersection
    mus = _findRayTriPlaneIntersections(points, tris[fltr,:], testPnt, \
        ray, normDF)
    
    return mus



def _fullRayIntersectionTest(testPnt, points, tris, assocs, LUT, \
    voxIJK, imgSize, normDF):
    """To be used in conjunction with reducedRayIntersectionTest(). Determine if a 
    point lies within a surface by performing a ray intersection test against
    all the triangles of a surface (not the reduced form used elsewhere for 
    speed) This is used to define a root point for the reduced ray intersection
    test defined in reducedRayIntersectionTest(). 

    Inputs: 
        testPnt: 1 x 3 vector for point under test
        points: p x 3 matrix of surface nodes
        tris: t x 3 matrix of triangle node indices
        assocs: list of lists recordining triangle voxel associations
        LUT: a LUT used for indexing voxel indices into the assocs list
        voxIJK: the IJK subscripts of the voxel in which the testPnt lies
        imgSize: the dimensions of the image in voxels
        normDF: 1/-1 for triangle node ordering (1 default)

    Returns: 
        bool flag if the point lies within the surface
    """

    # Get the voxel indices that lie along the ray (inc the current vox)
    subs = np.zeros((imgSize[0], 3), dtype=np.int32)
    subs[:,0] = np.arange(0, imgSize[0])
    subs[:,1] = voxIJK[1]
    subs[:,2] = voxIJK[2]
    inds = _sub2ind(imgSize, (subs[:,0], subs[:,1], subs[:,2]))

    # Form the ray and fetch all appropriate triangles from assocs data
    ray = np.array([1, 0, 0], dtype=np.float32)
    axisTris = _flattenList(assocs[np.isin(LUT, inds)])
    axisTris = list(set(axisTris))
    axisTris = tris[axisTris,:]
    intXs = _findRayTriangleIntersections2D(testPnt, ray, axisTris, \
        points, 0, normDF)

    # Classify according to parity of intersections. If odd number of ints
    # found between -inf and the point under test, then it is inside
    assert (len(intXs) % 2) == 0, 'Odd number of intersections returned'
    return ((np.sum(intXs <= 0) % 2) == 1)



# @numba.jit(nopython=True)
def _reducedRayIntersectionTest(testPnts, tris, triPoints, rootPoint,\
    normDF, flip):
    """Shortened form of the full ray intersection test, working with only 
    a small patch of surface. Determine if test points are contained or 
    not within the patch. 

    Args: 
        testPnt: p x 3 matrix of points to test
        tris: t x 3 matrix of triangle node indices
        triPoints: n x 3 matrix of surface nodes
        rootPoint: 1 x 3 vector for a point to which all rays used for 
            testing will be drawn from point under test
        normDF: 1/-1 for triangle node ordering (1 default)
        flip: bool flag, false if the root point is OUTSIDE the surface
    
    Returns: 
        vector of bools, length p, denoting if the points are INSIDE.
    """

    # If no tris passed in throw error
    if not tris.shape[0]:
        raise RuntimeError("No triangles to test against")
    
    # Each point will be tested by drawing a ray to the root point and 
    # testing for intersection against all available triangles
    flags = np.zeros(testPnts.shape[0], dtype=bool)
    for p in range(testPnts.shape[0]):

        # Ray is defined from the test pnt to the root point
        pnt = testPnts[p,:]
        ray = rootPoint - pnt

        # Find intersections, if they exist classify according to parity. 
        # Note the following logic assumes the rootPoint is inisde, ie, 
        # flip is false, if this is not the case then we will simply invert
        # the results at the end. 
        intMus = _findRayTriangleIntersections3D(pnt, ray, tris, \
            triPoints, normDF)

        if intMus.shape[0]:

            # Filter down to intersections in the range (0,1)
            intMus = intMus[(intMus < 1) & (intMus > 0)]

            # if no ints in this reduced range then point is inside
            # because an intersection would otherwise be guaranteed
            if not intMus.shape[0]:
                flags[p] = True 

            # If even number, then point inside
            else: 
                flags[p] = not(intMus.shape[0] % 2)
            
        # Finally, if there were no intersections at all then the point is
        # also inside (given the root point is also inside, if the test pnt
        # was outside there would necessarily be an intersection)
        else: 
            flags[p] = True 
    
    # Flip results if required
    if flip:
        flags = ~flags 
    
    return flags 



def _findTrianglePlaneIntersections(tris, points, voxCent, voxSize):
    """Find the points of intersection of all triangle edges with the 
    planar faces of the voxel. 

    Args: 
        tris: t x 3 matrix of triangle node indices
        points: p x 3 matrix of surface nodes
        voxCent: 1 x 3 vector of voxel centre
        voxSize: 1 x 3 vector of voxel dimensions

    Returns: 
        n x 3 matrix of n intersection coordinates 
    """
    if not tris.shape[0]:
        return np.zeros((0,3))

    # Form all the edge vectors of the voxel cube, strip out repeats
    edges = np.hstack((np.vstack((tris[:,2], tris[:,1])), \
        np.vstack((tris[:,1], tris[:,0])), \
        np.vstack((tris[:,0], tris[:,2])))).T 

    nonrepeats = np.empty((0,2), dtype=np.int16)
    for k in range(edges.shape[0]):
        if not np.any(np.all(np.isin(edges[k+1:,:], edges[k,:]), axis=1)):
            nonrepeats = np.vstack((nonrepeats, edges[k,:]))
    
    edges = nonrepeats 
    intXs = np.empty((0,3))

    # Iterate over each dimension, moving +0.5 and -0.5 of the voxel size
    # from the vox centre to define a point on the planar face of the vox
    for dim in range(3):
        pNormal = np.zeros(3, dtype=np.int8)
        pNormal[dim] = 1

        for k in [-1, 1]: 
            pPlane = voxCent + (k/2 * pNormal * voxSize[dim])
            pStart = points[edges[:,0],:]
            edgeVecs = points[edges[:,1],:] - points[edges[:,0],:]

            # Sneaky trick here: because planar normals are aligned with the 
            # coord axes we don't need to do a full dot product, just extract
            # the appropriate component of the difference vectors
            mus = (pPlane - pStart)[:,dim] / edgeVecs[:,dim]
            pInts = pStart + (edgeVecs.T * mus).T
            pInts2D = pInts - pPlane 
            keep = np.all(np.abs(pInts2D) <= (voxSize/2), 1)    
            keep = np.logical_and(keep, np.logical_and(mus <= 1, mus >= 0))
            intXs = np.vstack((intXs, pInts[keep,:]))

    return intXs 



def _findVoxelSurfaceIntersections(tris, points, vertices, normDF):
    """Find points of intersection between edge and body vectors of a voxel
    with surface. Also detects folds along any of these vectors.

    Args: 
        tris: t x 3 matrix of triangle node indices
        points: p x 3 matrix of surface nodes
        vertices: vertices of the voxel
        normDF: 1/-1 for triangle node ordering (1 default)
    
    Returns: 
        (intersects, fold) tuple: intersects the points of intersection, 
            fold a bool if one has been detected. If a fold is detected, 
            function returns immediately (without complete set of intersections)
    """

    intersects = np.empty((0,3))
    fold = False 

    # If nothing to test, silently return 
    if not tris.shape[0]:
        return (intersects, fold)

    # 8 vertices correspond to 12 edge vectors along exterior edges and 
    # 4 body diagonals. 
    origins = np.array([1, 1, 1, 4, 4, 4, 5, 5, 8, 8, 6, 7, 1, 2, 3, 4, \
        1, 1, 1, 8, 8, 8, 2, 2, 3, 4, 4, 6], dtype=np.int8) - 1
    ends = np.array([2, 3, 5, 8, 2, 3, 6, 7, 6, 7, 2, 3, 8, 7, 6, 5, \
        6, 4, 7, 5, 3, 2, 3, 5, 5, 6, 7, 7], dtype=np.int8) - 1
    edges = vertices[ends,:] - vertices[origins,:]

    # Test each vector against the surface
    for e in range(16):
        edge = edges[e,:]
        pnt = vertices[origins[e],:]
        intMus = _findRayTriangleIntersections3D(pnt, edge, tris, points, normDF)

        if intMus.shape[0]:
            intPnts = pnt + np.outer(intMus, edge)
            accept = np.logical_and(intMus <= 1, intMus >= 0)
            
            if sum(accept) > 1:
                fold = True
                return (intersects, fold)

            intersects = np.vstack((intersects, intPnts[accept,:]))

    return (intersects, fold)


def _flattenList(lists):
    """Little utility to flatten a list of lists (depth of 2 max)"""
    out = []
    for sublist in lists:
         out = out + sublist 
    return out 


def _adjustFLIRT(source, reference, transform):
    """Adjust a FLIRT transformation matrix into a true world-world 
    transform. Required as FSL matrices are encoded in a specific form 
    such that they can only be applied alongside the requisite images (extra
    information is required from those images). With thanks to Martin Craig
    and Tim Coalson. See: https://github.com/Washington-University/workbench/blob/9c34187281066519e78841e29dc14bef504776df/src/Nifti/NiftiHeader.cxx#L168 
    https://github.com/Washington-University/workbench/blob/335ad0c910ca9812907ea92ba0e5563225eee1e6/src/Files/AffineFile.cxx#L144

    Args: 
        source: path to source image, the image to be deformed 
        reference: path to reference image, the target of the transform
        transform: affine matrix produced by FLIRT from src to ref 

    Returns: 
        complete transformation matrix between the two. 
    """

    # Local function to read out an FSL-specific affine matrix from an image
    def __getFSLspace(imgPth):
        obj = nibabel.load(imgPth)
        if obj.header['dim'][0] < 3:
            raise RuntimeError("Volume has less than 3 dimensions" + \
                 "cannot resolve space")

        sform = obj.affine
        det = np.linalg.det(sform[0:4, 0:4])
        ret = np.identity(4)
        pixdim = obj.header['pixdim'][1:4]
        for d in range(3):
            ret[d,d] = pixdim[d]

        # Check the xyzt field to find the spatial units. 
        xyzt =str(obj.header['xyzt_units'])
        if xyzt == '01': 
            multi = 1000
        elif xyzt == '10':
            multi = 1 
        elif xyzt =='11':
            multi = 1e-3
        else: 
            raise RuntimeError("Unknown units")

        if det > 0:
            ret[0,0] = -pixdim[0]
            ret[0,3] = (obj.header['dim'][1] - 1) * pixdim[0]

        ret = ret * multi
        ret[3,3] = 1
        return ret

    # Main function
    srcSpace = __getFSLspace(source)
    refSpace = __getFSLspace(reference)

    refObj = nibabel.load(reference)
    refAff = refObj.affine 
    srcObj = nibabel.load(source)
    srcAff = srcObj.affine 

    outAff = ((refAff @ np.linalg.inv(refSpace)) @ transform) @ srcSpace
    return (outAff @ np.linalg.inv(srcAff))


def _estimateVoxelFraction(allPoints, allTris, assocs, LUT, voxIJK, voxIdx,
    imgSize, supersampler, normDF):
    """The Big Daddy that does the Heavy Lifting. 
    Recursive estimation of PVs within a single voxel. Overview as follows: 
    - split voxel into subvoxels according to supersampler
    - check if they are intersected by any triangles
    - if not, perform whole-tissue type classification by ray testing 
    - if so, check the extent of intersection with the surface and use
        convex hulls to estimate volumes
    - if complex intersection (folds, multiple intersections etc) then 
        use recursion 
    """

    # Local functions to warm up ----------------------------------------------

    def __safeFormHull(points):
        """If three or less points are provided, or not enough distinct points 
        (eg coplanar points), then return 0 volume (recursion will be used 
        elsewhere). For everything else, let the exception continue up. 
        """

        if points.shape[0] > 3:
            try:
                hull = ConvexHull(points)
                return hull.volume
            except QhullError: 
                return 0
            except Exception: 
                raise 
        else: 
            return 0

    def __classifyVoxelViaRecursion(ps, ts, voxCent, voxSize, normDF, \
        containedFlag):
        """Classify a voxel entirely via recursion (not using any 
        convex hulls)
        """

        super2 = 5
        Nsubs2 = super2**3

        sX = np.arange(1 / (2 * super2), 1, 1 / super2) - 0.5
        sX, sY, sZ = np.meshgrid(sX, sX, sX)
        subVoxCents = np.vstack((sX.flatten(), sY.flatten(), sZ.flatten())).T
        flags = _reducedRayIntersectionTest(subVoxCents, ts, ps, voxCent, \
            normDF, ~containedFlag)

        return (np.sum(flags) / Nsubs2)

    def __fetchSubVoxCornerIndices(linIdx, supersampler):
        """Map between linear subvox index number and the indices of its
        vertices within the larger grid of subvoxel vertices
        """

        # Get the IJK coords within the subvoxel grid. 
        # Vertices are then +1/0 from these coords
        i, j, k = _ind2sub(supersampler, linIdx)
        subs = np.array([ [i, j, k], [i+1, j, k], [i, j+1, k], \
            [i+1, j+1, k], [i, j, k+1], [i+1, j, k+1], [i, j+1, k+1], \
            [i+1, j+1, k+1] ], dtype=np.int16) 

        # And map these vertix subscripts to linear indices within the 
        # grid of subvox vertices (which is always + 1 larger than supersamp)
        corners = _sub2ind(supersampler + 1, (subs[:,0], subs[:,1], subs[:,2]))

        return corners 

    def __getAllSubVoxCorners(supersampler, voxCent, voxSize):
        """Produce a grid of subvoxel vertices within a given voxel.

        Args: 
            supersampler: 1 x 3 vector of supersampling factor
            voxCent: 1 x 3 vector centre of voxel
            voxSize: 1 x 3 vector voxel dimensions
        
        Returns: 
            s x 3 matrix of subvoxel vertices, arranged by linear index
                of IJK along the column
        """

        # Get the origin for the grid of vertices (corner with smallest xyz)
        root = voxCent - voxSize/2

        # Grid will have s+1 points in each dimension 
        X, Y, Z = np.meshgrid(
            np.linspace(root[0], root[0] + voxSize[0], supersampler[0] + 1), \
            np.linspace(root[1], root[1] + voxSize[1], supersampler[1] + 1), \
            np.linspace(root[2], root[2] + voxSize[2], supersampler[2] + 1))

        return np.vstack((X.flatten(), Y.flatten(), Z.flatten())).astype(np.float32).T

    
    # The main function, here we go... ----------------------------------------

    verbose = False
    print(voxIdx)

    # Hardcode voxel size as we now work in voxel coords. Intialise results
    voxSize = np.array([1,1,1], dtype=np.int8)
    inFraction = 0

    # Set up the subvoxel sizes and vols. 
    samplr = supersampler 
    subVoxSize = (voxSize / supersampler).astype(np.float32)
    subVoxVol = np.prod(subVoxSize).astype(np.float32)
    subVox2World = np.array([
        [1/samplr[0], 0, 0, voxIJK[0] + (1-samplr[0])/(2*samplr[0])], \
        [0, 1/samplr[1], 0, voxIJK[1] + (1-samplr[1])/(2*samplr[1])], \
        [0, 0, 1/samplr[2], voxIJK[2] + (1-samplr[2])/(2*samplr[2])], \
        [0, 0, 0, 1]
    ])

    # Rebase triangles and points for this voxel
    ps, ts = _rebaseTriangles(allPoints, \
        allTris[_flattenList(assocs[LUT == voxIdx]), :])

    # Test all subvox corners now and store the results for later
    allCorners = __getAllSubVoxCorners(supersampler, voxIJK, voxSize)
    voxCentFlag = _fullRayIntersectionTest(voxIJK, allPoints, allTris, \
        assocs, LUT, voxIJK, imgSize, normDF)
    allCornerFlags = _reducedRayIntersectionTest(allCorners, ts, ps, \
            voxIJK, normDF, ~voxCentFlag)


    # Subvoxel loop starts here -----------------------------------------------

    for s in range(np.prod(supersampler)):

        # Get the centre and corners, prepare the sanity check
        subVoxClassified = False
        sijk = np.array([_ind2sub(supersampler, s)])
        subVoxCent = np.squeeze(_affineTransformPoints(sijk, subVox2World))
        cornerIndices = __fetchSubVoxCornerIndices(s, supersampler)
        corners = allCorners[cornerIndices,:]

        # Set the flag for the subVoxCent to serve as the root point
        subVoxFlag = _fullRayIntersectionTest(subVoxCent, allPoints, allTris, \
            assocs, LUT, voxIJK, imgSize, normDF)

        # Do any triangles intersect the subvox?
        triFltr = _cfilterTriangles(ts, ps, subVoxCent, subVoxSize)

        # CASE 1 --------------------------------------------------------------

        # If no triangles intersect the subvox then whole-tissue classification
        # using the flip flags that were set by fullRayIntersectTest()
        if not np.any(triFltr): 
            if verbose: print("Whole subvox assignment")
            
            inFraction += (subVoxFlag * subVoxVol)        
            subVoxClassified = True 

        # CASE 2: some triangles intersect the subvox -------------------------

        else: 
            cornerFlags = allCornerFlags[cornerIndices] 

            # Check for subvoxel edge intersections with the local patch of
            # triangles and for folds
            edgeIntXs, fold = _findVoxelSurfaceIntersections( \
                ts[triFltr,:], ps, corners, normDF)

            # Separate points within the voxel into distinct clouds, to check
            # for multiple surface intersection
            groups = _separatePointClouds(ts[triFltr,:])

            # If neither surface is folded within the subvox and there 
            # are no multiple intersections, we can form hulls. 
            if (not fold) & (len(groups) < 2):

                # Filter down surface nodes that are in the subvox
                localPs = ps[np.unique(ts[triFltr,:]),:]
                
                # Gather points together in preparation for forming 
                # hulls (the corners will be added just beforehand)
                hullPts = np.vstack( (localPs, edgeIntXs, \
                    _findTrianglePlaneIntersections(ts[triFltr,:], \
                    ps, subVoxCent, subVoxSize)) )

                if verbose: print('Hulls: mixed in/out')

                # If all corner flags homogenous but triangles intersect
                # then we have a situation where a small amount of surface
                # intersects a face in a highly convex manner
                if np.all(cornerFlags[0] == cornerFlags):
                    
                    # All corners are inside so small exterior hull
                    # Inverse otherwise
                    classes = [0, 1] if cornerFlags[0] else [1, 0]

                # Non homogenous corner flags: use convex hulls. 
                # Aim to form the smallest possible hull
                else:
                    
                    # Smaller interior hull 
                    if np.sum(cornerFlags) < 4:
                        hullPts = np.vstack((hullPts, corners[cornerFlags,:]))
                        classes = [1, 0]
                    
                    # Smaller exterior hull
                    else:
                        hullPts = np.vstack((hullPts, corners[~cornerFlags,:]))
                        classes = [0, 1]

                V = __safeFormHull(hullPts)
                if not V: 
                    L2fraction = __classifyVoxelViaRecursion(ps, ts, \
                        subVoxCent, subVoxSize, normDF, subVoxFlag)
                    inFraction += (subVoxVol * L2fraction) 
                else:
                    inFraction += np.matmul(classes, [V, (subVoxVol - V)])       
                
                subVoxClassified = True 

            # CASE 3: voxels that require recursion ---------------------------
            
            else: 

                # Folded surface or multiple intersection. 
                if fold | (len(groups) > 1):
                    if verbose & (len(groups) > 1): 
                        print("Multiple intersections detected, using recursion")
                    
                    if verbose & fold:
                        print("Fold detected, using recursion")

                    L2fraction = __classifyVoxelViaRecursion(ps, ts, subVoxCent, \
                        subVoxSize, normDF, subVoxFlag)
                    inFraction += (L2fraction * subVoxVol)
                    subVoxClassified = True 
            
        # Sanity check: we should have classified the voxel by now
        if not subVoxClassified: 
            raise RuntimeError("Subvoxel has not been classified")

        if verbose: print(inFraction)

    # END of subvoxel loop ----------------------------------------------------
      
    if inFraction > 1.00000001:
        raise RuntimeError('Fraction exceeds 1')
    return inFraction 



def _estimateFractions(points, tris, assocs, LUT, FoVsize, supersampler, \
    voxList, normDF):
    """Estimate fraction of voxels lying interior to surface. 

    Args: 
        points: surface nodes for either surface
        tris: triangles for either surface
        assocs: list of lists recording triangle/voxel associations
        LUT: mapping between linear voxel index and posn with assocs
        FoVsize: dimensions of voxel grid required to contain surfaces, 
            to which the voxels in voxList are indexed. 
        supersampler: 1 x 3 vector of supersampling factor
        voxList: list of linear voxel indices within the FoV to process
            (note any ROI is applied in the calling function)
        normDF: 1/-1 for triangle node ordering (1 default)

    Returns: 
        matrix of size prod(FoV) x 2, where the first column  
    """

    # Check inputs
    if points.shape[1] != 3: 
        raise RuntimeError("Points matrices should be p x 3")

    if tris.shape[1] != 3: 
        raise RuntimeError("Triangles matrices should be t x 3")

    if FoVsize.size != 3: 
        raise RuntimeError("FoV size should be a 1 x 3 vector")

    debug = False

    # Compute all voxel centres
    I, J, K = _ind2sub(FoVsize, np.arange(np.prod(FoVsize)))
    voxIJKs = np.vstack((I.flatten(), J.flatten(), K.flatten())).T
    voxIJKs = voxIJKs.astype(np.float32)

    # Shuffle voxel list in place, then prepare the worker function
    np.random.shuffle(voxList)
    workerChunks = _distributeObjects(voxList, __NWORKERS__)

    estimateFractionsPartial = functools.partial(__estimateFractionsWorker, \
        points, tris, assocs, LUT, voxIJKs, FoVsize, supersampler, normDF)

    # with multiprocessing.Pool(__NWORKERS__) as p: 
    #     workerResults = p.starmap(estimatePVsPartial, [workerChunks, \
    #     range(__NWORKERS__)])

    workerFractions = list(map(estimateFractionsPartial, workerChunks, \
        range(__NWORKERS__)))

    # Aggregate the results back together. 
    fractions = np.hstack(workerFractions)

    # Sanity check: did all voxels in toEstimate get processeed?
    if not all(np.any(fractions >= 0, axis=1)):
        raise RuntimeError("Not all voxels in voxList processed.")

    if np.any(fractions < 0): 
        raise RuntimeError("Negative fraction returned.")

    if np.any(fractions > 1):
        raise RuntimeError("Fraction greater than 1 returned.")

    return (fractions, voxList)



def __estimateFractionsWorker(points, tris, assocs, LUT, voxIJK, imgSize, \
        supersampler, normDF, workerVoxList, workerNumber):

    partialVolumes = np.zeros((len(workerVoxList), 3))
    counter = 0
    progFactor = (len(workerVoxList) // 10)

    for v in workerVoxList: 

        if (workerNumber == 0) & (counter % progFactor == 0): 
            print("{}..".format(round(10 * counter / progFactor)), \
            end='', flush=True)

        # Load voxel centre, estimate vols, and rescale to PVs
        voxijk = voxIJK[v,:]
        fraction = _estimateVoxelFraction(points, tris, assocs, LUT, \
            voxijk, v, imgSize, supersampler, normDF)
        
        partialVolumes[counter,:] = fraction
        counter += 1 
    
    return partialVolumes



def toblerone(**kwargs):
    """Estimate partial volumes on the cortical ribbon"""

    verbose = True 


    # If subdir given, then get all the surfaces out of the surf dir
    # If individual surface paths were given they will already be in scope
    if 'FSSubDir' in kwargs:
        FSSubDir = op.join(kwargs['FSSubDir'], 'surf')
        
        if not op.isdir(FSSubDir):
            raise RuntimeError("Subject's surf/ directory does not exist")

        kwargs['LWS'] = op.join(FSSubDir, 'lh.white')
        kwargs['LPS'] = op.join(FSSubDir, 'lh.pial')
        kwargs['RWS'] = op.join(FSSubDir, 'rh.white')
        kwargs['RPS'] = op.join(FSSubDir, 'rh.pial')

    # Define the hemispheres we will be working with and check surfaces exist. 
    # Check file formats
    hemispheres = []
    if (kwargs.get('LWS') is not None) & (kwargs.get('LPS') is not None): 
        hemispheres.append('L')
        assert all(map(op.isfile, (kwargs['LPS'], kwargs['LWS']))), \
            "LWS/LPS surface does not exist"
        surfExt = op.splitext(kwargs['LWS'])[-1]
    if (kwargs.get('RWS') is not None) & (kwargs.get('RPS') is not None): 
        hemispheres.append('R')
        assert all(map(op.isfile, (kwargs['RPS'], kwargs['RWS']))), \
            "RWS/RPS surface does not exist"
        surfExt = op.splitext(kwargs['RWS'])[-1]
    assert len(hemispheres), "Specify at least one hemisphere's surfaces (eg LWS/LPS)"

    # Manual surface offset, in which all checks disabled. 
    if 'cRAS' in kwargs: 
        assert kwargs['cRAS'].shape == (1,3), "cRAS must be a 1x3 vector"
    else: 
        cRAS = None 

    # Reference image path 
    assert ('reference' in kwargs), "Path to reference image must be given"
    assert op.isfile(kwargs['reference']), "Reference image does not exist"
    inExt = op.splitext(kwargs['reference'])[-1]
    assert inExt in [".nii", ".gz", ".mgh", ".mgz"], \
        "Reference must be in the following formats: nii, nii.gz, mgh, mgz"
    if '.nii.gz' in kwargs['reference']:
        inExt = '.nii.gz'

    # Structural to reference transformation 
    if 'struct2ref' in kwargs:
        if (type(kwargs['struct2ref']) is str):
            kwargs['struct2ref'] = np.fromfile(kwargs['struct2ref'], dtype=float)
        assert kwargs['struct2ref'].shape == (4,4), "struct2ref must be a 4x4 matrix"

    # Is this a FLIRT transform? If so we need to do some clever preprocessing
    if 'flirt' in kwargs:
        assert ('structural' in kwargs), "If using a FLIRT transform, the path to the \
            structural image must also be given"
        assert op.isfile(kwargs['structural']), "If using a FLIRT transform, the path to the \
            structural image must also be given"

    # Read in input image properties 
    imgObj = nibabel.load(kwargs['reference'])
    voxSize = (imgObj.header['pixdim'])[1:4]
    imgSize = (imgObj.header['dim'])[1:4]
    vox2world = imgObj.affine
    world2vox = np.linalg.inv(imgObj.affine)

    # Prepare output directory. If given then create it 
    # If not then use the input image dir
    if 'outDir' in kwargs: 
        if not op.isdir(kwargs['outDir']):
            os.mkdir(kwargs['outDir'])
    else: 
        kwargs['outDir'] = op.dirname(kwargs['reference'])

    # Create output dir for transformed surfaces
    transSurfDir = op.join(kwargs['outDir'], 'surf_transform')
    if not op.isdir(transSurfDir):
        os.mkdir(transSurfDir)
    
    # Check if the flip flag should be set (det of the 
    # 3 x 3 matrix in top left of vox2world)
    normDF = -1 if (np.linalg.det(vox2world[0:3,0:3]) < 0) else 1
    
    # Prepare the output filename. If not given then we pull it 
    # from the reference
    if 'outName' in kwargs:
        outName = kwargs['outName']
    else:  
        outName = op.split(kwargs['reference'])[-1]

    outExt = ''
    fname = outName
    while '.' in fname: 
        fname, e = op.splitext(fname)
        try:
            ext = float(e)
        except ValueError: 
            outExt = e
            ext = None
        if type(ext) is float: 
            fname + ".{:g}".format(ext)
            break 
    
    if outExt == '': 
        outExt = inExt 
    
    maskName = outName + "_surfmask"
    assocsName = fname + "_assocs"
    outPath = op.join(kwargs['outDir'], fname + outExt)
    maskPath = op.join(kwargs['outDir'], maskName + outExt)
    assocsPath = op.join(kwargs['outDir'], assocsName + '.pkl')

    # Debug tools 
    if 'solution' in kwargs: 
        sol = nibabel.load(kwargs['solution'])
        sol = sol.get_fdata()
        sol = np.reshape(sol, (-1, 3))
    else:
        sol = [] 

    # Load surface geometry information 
    # If FS binary format, we can read cRAS straight out of the meta dict.
    if cRAS is None: 
        if surfExt != '.gii':
            surfName = hemispheres[0] + 'PS'
            _, _, meta = tuple(nibabel.freesurfer.io.read_geometry(\
                        kwargs[surfName], read_metadata=True))
            cRAS = meta['cras']

        else: 
            cRAS = np.zeros(3)
    
    # Form the final transformation matrix to bring the surfaces to 
    # the same world (mm, not voxel) space as the reference image
    transform = np.identity(4)
    transform[0:3,3] = cRAS
    transform = np.matmul(kwargs['struct2ref'], transform)
    if verbose: 
        np.set_printoptions(precision=3, suppress=True)
        print("Transformation matrix will be applied:\n", transform)
    
    # Load all surfaces and transform into reference voxel space
    inPs = {}; outPs = {}; inTs = {}; outTs = {}
    for h in hemispheres: 
        for s in ['P', 'W']: 
            surfName = h + s + 'S'

            if surfExt != '.gii':
                ps, ts = tuple(nibabel.freesurfer.io.read_geometry(\
                    kwargs[surfName]))
            else: 
                ps, ts = tuple(map(lambda o: o.data, \
                    nibabel.load(kwargs[surfName]).darrays))

            # Transform the surfaces
            ps = _affineTransformPoints(ps, np.matmul(world2vox, transform))
            ts = ts.astype(np.int32)
            ps = ps.astype(np.float32)

            # Final indexing checks
            assert (np.min(ts) == 0) & (np.max(ts) == ps.shape[0] - 1), \
                "Vertex/triangle indexing incorrect"

            # Load the ps/ts into their respective dicts
            if s == 'P':
                outPs[h] = ps; outTs[h] = ts
            else: 
                inPs[h] = ps; inTs[h] = ts 


    # FoV and associations loop -----------------------------------------------

    # Check the FoV of the reference image against the extents of the surfaces
    # Warn if FoV does not contain whole surface
    s1 = outPs[hemispheres[0]]; s2 = outPs[hemispheres[-1]]
    if (np.any(s1 < -1) | np.any(s1 > imgSize-1) | \
        np.any(s2 < -1) |  np.any(s2 > imgSize-1)):
        print("Warning: the FoV of the reference image does not cover the", 
            "full extents of the surfaces provided. PVs will only be", 
            "estimated within the reference FoV")

    # Determine the full FoV needed to contain the surfaces
    minFoV = np.floor(np.minimum(np.min(s1, axis=0), \
        np.min(s2, axis=0)))
    maxFoV = np.ceil(np.maximum(np.max(s1, axis=0), \
        np.max(s2, axis=0)))

    # If the full FoV is larger than the reference FoV, work out the coord
    # shift between the spaces. [0 0 0] in reference space becomes [X Y Z] 
    # in full FoV space, where XYZ is the FoV offset. 
    # The max() function catches cases where no offset is necessary
    FoVoffset = np.maximum(-minFoV, np.zeros(3)).astype(np.int8)
    fullFoVsize = (np.maximum(maxFoV - minFoV + 1, imgSize)).astype(np.int16)
    assert np.all(fullFoVsize >= imgSize), \
        "Full FoV has not been expanded to at least reference FoV"

    # Shift surfaces to remove negative coordinates, check max limits
    # are within range of the full FoV
    if np.any(FoVoffset != 0):
        for h in hemispheres:
            for s in ["inPs", "outPs"]: 
                surf = s + '[\'' + h + '\']'
                exec(surf + '=' + surf + '+ FoVoffset')

                assert np.all(np.floor(np.min(eval(surf), axis=0)) >= 0), \
                    "FoV offset does not remove negative coordinates"
                assert np.all(np.ceil(np.max(eval(surf), axis=0)) < \
                    fullFoVsize), "Full FoV does not contain all surface coordinates"

    # Form (or read in) associations
    # We use a "stamp" matrix to check that saved/loaded assocs
    # are referenced to the right image space
    stamp = np.matmul(kwargs['struct2ref'], world2vox)
    inAssocs = {}; outAssocs = {} 
    recomputeAssocs = False 
    if op.isfile(assocsPath):
        print("Loading pre-computed associations found in the output", \
            "directory:", assocsPath)

        with open(assocsPath, 'rb') as f:

            oldStamp, inAssocs, outAssocs = pickle.load(f)
            if not np.array_equal(oldStamp, stamp): 
                raise RuntimeError("Pre-loaded associations stamp does not match", \
                    "surface/image geometry. Delete the associations.")
        
        # Check the loaded assocs are good to go (not nones)
        for h in hemispheres: 
            if not ((inAssocs[h] is not None) & (outAssocs[h] is not None)):
                print("Loaded associations are not complete, will recompute.")
                recomputeAssocs = True 

    elif (recomputeAssocs) or not (op.isfile(assocsPath)): 
        if 'saveAssocs' in kwargs: 
            print("Forming voxel associations, saving to:", assocsPath)
        else: 
            print("Forming voxel associations")

        # Initialise all these as none and overwrite them with true values if the 
        # hemispheres are available to do so 
        for h in hemispheres: 
            inAssocs[h] = _formAssociations(inPs[h], inTs[h], fullFoVsize)
            outAssocs[h] = _formAssociations(outPs[h], outTs[h], fullFoVsize)

        if 'saveAssocs' in kwargs: 
            with open(assocsPath, 'wb') as f: 
                pickle.dump((stamp, inAssocs, outAssocs), f)

    # Unpack the associations into their keys (LUT) and values. 
    inLUT = {}; outLUT = {} 
    for h in hemispheres: 
        inLUT[h] = list(inAssocs[h].keys())
        outLUT[h] = list(outAssocs[h].keys())
        inAssocs[h] = np.array(list(inAssocs[h].values()))
        outAssocs[h] = np.array(list(outAssocs[h].values()))

    # And now pass off to the actual toblerone estimation
    supersampler = (np.ceil(2 * voxSize) +1).astype(np.int8)
    print("Supersampling factor set at:", supersampler)

    # Generate the list of voxels to be processed. Start with the set of all
    # voxels within the reference image 
    I,J,K = np.meshgrid(np.arange(imgSize[0]), np.arange(imgSize[1]), \
        np.arange(imgSize[2]))
    voxSubs = np.vstack((I.flatten(), J.flatten(), K.flatten())).T

    # Shift these into full FOV space with the offset
    voxSubs = voxSubs + FoVoffset 

    # If ROI was given then apply it now
    if 'ROI' in kwargs:
        voxSubs = voxSubs[kwargs['ROI'].flatten(),:]
    
    # Convert to linear indices within full FOV space
    voxList = _sub2ind(fullFoVsize, (voxSubs[:,0], voxSubs[:,1], voxSubs[:,2]))

    # Process each hemisphere
    PVs = {}
    for h in hemispheres:

        if np.any(np.max(np.abs(inPs[h])) > np.max(np.abs(outPs[h]))):
            raise RuntimeWarning("Inner surface vertices appear to be further",\
                "from the origin than the outer vertices. Are the surfaces in",\
                "the correct order?")
        
        print("{} in: ".format(h), end='')
        inList = np.intersect1d(voxList, inLUT[h])
        inFracs, inList = _estimateFractions(inPs[h], inTs[h], inAssocs[h], \
            inLUT[h], fullFoVsize, supersampler, inList, normDF)

        print("{}out: ".format(h))
        outList = np.intersect1d(voxList, outLUT[h])
        outFracs, outList = _estimateFractions(outPs[h], outTs[h], \
            outAssocs[h], outLUT[h], fullFoVsize, supersampler, outList, normDF)
        
        voxelisePartial = functools.partial(_voxeliseSurfaceAlongDimension, \
            FoVsize=fullFoVsize, dim=0, normDF=normDF)

        # with multiprocessing.Pool(min(2, __NWORKERS__)) as p:
        #     fills = p.starmap(voxelisePartial, [inTs[h], outTs[h]], \
        #         [inPs[h], outPs[h]], [inAssocs[h], outAssocs[h]], \
        #         [inLUT[h], outLUT[h]])


        fills = list(map(voxelisePartial, [inTs[h], outTs[h]], \
            [inPs[h], outPs[h]], [inAssocs[h], outAssocs[h]], \
            [inLUT[h], outLUT[h]]))


        inFilled, outFilled = fills 

        inFractions = (inFilled.flatten()).astype(np.float32)
        inFractions[inList] = inFracs 

        outFractions = (outFilled.flatten()).astype(np.float32)
        outFractions[outList] = outFracs

        hemiPVs = np.zeros((np.prod(fullFoVsize), 3))
        hemiPVs[:,1] = inFractions 
        hemiPVs[:,0] = np.maximum(0, outFractions - inFractions)
        hemiPVs[:,2] = 1 - np.sum(hemiPVs[:,0:2], axis=1)

        PVs[h] = hemiPVs

    # Merge the fill masks by giving priority to GM, then WM, then CSF.
    if hemispheres == ['L']:
        outPVs = PVs['L']
    elif hemispheres == ['R']:
        outPVs = PVs['R']
    else:
        outPVs[:,0] = np.minimum(1, PVs['R'][:,0] + PVs['L'][:,0])
        outPVs[:,1] = np.minimum(1 - outPVs[:,0], \
            PVs['R'][:,1] + PVs['L'][:,1])
        outPVs[:,2] = 1 - np.sum(outPVs[:,0:2], axis=1)

    # Sanity checks
    assert (not np.any(outPVs > 1))
    assert (not np.any(outPVs < 0))
    assert (not np.any(np.sum(outPVs, axis=1) > 1.00001))
    assert all(np.any(outPVs[voxList,:] > 0, axis=1))

    # Form the surface mask (3D logical) as any voxel containing GM or 
    # intersecting the cortex (these definitions should always be equivalent)
    assocsMask = np.zeros(outPVs.shape, dtype=bool)
    assocsFltr = list(functools.reduce(lambda a, b: np.union1d(a,b), 
        (inLUT.values()).join(outLUT.values())))
    assocsMask[assocsFltr] = True 

    # Reshape images back into 4D or 3D images
    outPVs = np.reshape(outPVs, (fullFoVsize[0], fullFoVsize[1], \
        fullFoVsize[2], 3))
    assocsMask = np.reshape(assocsMask, tuple(fullFoVsize[0:3]))

    # Extract the output within the FoV of the reference image
    outPVs = outPVs[ FoVoffset[0] : FoVoffset[0] + imgSize[0], \
        FoVoffset[1] : FoVoffset[1] + imgSize[1], \
        FoVoffset[2] : FoVoffset[2] + imgSize[2] ]
    assocsMask = outPVs[ FoVoffset[0] : FoVoffset[0] + imgSize[0], \
        FoVoffset[1] : FoVoffset[1] + imgSize[1], \
        FoVoffset[2] : FoVoffset[2] + imgSize[2] ]

    # Finally, form the NIFTI objects and save the images. 
    print("Saving PVs to", outPath)
    PVhdr = copy.deepcopy(imgObj.header)
    PVhdr['dim'][0] = 4
    PVhdr['dim'][4] = 3
    PVobj = nibabel.nifti2.Nifti2Image(outPVs, imgObj.affine, header=PVhdr)
    nibabel.save(PVobj, outPath)

    print("Saving surface mask to", maskPath)
    maskHdr = copy.deepcopy(imgObj.header)
    maskObj = nibabel.nifti2.Nifti2Image(assocsMask, imgObj.affine, \
        header=maskHdr)
    nibabel.save(maskObj, maskPath)

    return # We are done. 





# Deprecated 
# def _testTriangleVoxelIntersection(voxCent, voxSize, tri): 
#     """Test if triangle intersects voxel, return bool flag

#     Args: 
#         voxCent: 1 x 3 vector voxel centre
#         voxSize: 1 x 3 vector vox dimensions. NB original implementation
#             expects *half* voxel sizes, this expects *full* sizes
#         tri: 3 x 3 matrix of voxel coordinates XYZ along cols 

#     This is a direct MATLAB port of the code available at:
#     http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
#     """

#     # Main function body ---------------

#     # Rescale voxel size and shift coordinates to cent
#     voxSize = voxSize / 2 
#     v0 = tri[0,:] - voxCent
#     v1 = tri[1,:] - voxCent
#     v2 = tri[2,:] - voxCent

#     # Edge vecs
#     e0 = v1 - v0
#     e1 = v2 - v1
#     e2 = v0 - v2

#     # Bullet 3 tests
#     flag = False
#     fex = np.abs(e0[0])
#     fey = np.abs(e0[1])
#     fez = np.abs(e0[2])
#     if not (__axistestX01(e0[2], e0[1], fez, fey, v0, v2, voxSize) & \
#         __axistestY02(e0[2], e0[0], fez, fex, v0, v2, voxSize) & \
#         __axistestZ12(e0[1], e0[0], fey, fex, v1, v2, voxSize)):
#         return flag 

#     fex = np.abs(e1[0])
#     fey = np.abs(e1[1])
#     fez = np.abs(e1[2])
#     if not (__axistestX01(e1[2], e1[1], fez, fey, v0, v2, voxSize) & \
#         __axistestY02(e1[2], e1[0], fez, fex, v0, v2, voxSize) & \
#         __axistestZ0(e1[1], e1[0], fey, fex, v0, v1, voxSize)):
#         return flag 

#     fex = np.abs(e2[0])
#     fey = np.abs(e2[1])
#     fez = np.abs(e2[2])
#     if not (__axistestX2(e2[2], e2[1], fez, fey, v0, v1, voxSize) & \
#         __axistestY1(e2[2], e2[0], fez, fex, v0, v1, voxSize) & \
#         __axistestZ12(e2[1], e2[0], fey, fex, v1, v2, voxSize)):
#         return flag 
        
#     # Bullet 1 tests
#     mi, ma = __findMinMax((v0[0], v1[0], v2[0]))
#     if ((mi > voxSize[0]) | (ma < -voxSize[0])): 
#         return flag 

#     mi, ma = __findMinMax((v0[1], v1[1], v2[1]))
#     if ((mi > voxSize[1]) | (ma < -voxSize[1])): 
#         return flag

#     mi, ma = __findMinMax((v0[2], v1[2], v2[2]))
#     if ((mi > voxSize[2]) | (ma < -voxSize[2])): 
#         return flag 

#     # Bullet 2 tests
#     normal = np.cross(e0, e1)
#     d = -np.dot(normal, v0)
#     if not __planeBoxOverlap(normal, d, voxSize):
#         return flag
    
#     return True 



# def __planeBoxOverlap(normal, d, box):
#     vmin = np.array([0,0,0], dtype=float)
#     vmax = np.array([0,0,0], dtype=float)

#     for q in range(3):
#         if normal[q] > 0.0:
#             vmin[q] = -box[q]
#             vmax[q] = box[q]
#         else:
#             vmin[q] = box[q]
#             vmax[q] = -box[q]

#     if (np.dot(normal, vmin) + d > 0.0):
#         return False
#     if (np.dot(normal, vmax) + d >= 0.0): 
#         return True 
    
#     return False 



# def __findMinMax(seta): 
#     return np.array([np.min(seta), np.max(seta)])



# def __axistestX01(a, b, fa, fb, v0, v2, voxSize):
#     p0 = a*v0[1] - b*v0[2]
#     p2 = a*v2[1] - b*v2[2]
#     mi, ma = __findMinMax([p0, p2])
#     rad = (fa * voxSize[1]) + (fb * voxSize[2])
#     return ~((mi > rad) | (ma < -rad))



# def __axistestX2(a, b, fa, fb, v0, v1, voxSize):
#     p0 = a*v0[1] - b*v0[2]
#     p1 = a*v1[1] - b*v1[2]
#     mi, ma = __findMinMax([p0, p1])
#     rad = (fa * voxSize[1]) + (fb * voxSize[2])
#     return ~((mi > rad) | (ma < -rad))



# def __axistestY02(a, b, fa, fb, v0, v2, voxSize):
#     p0 = -a*v0[0] + b*v0[2]
#     p2 = -a*v2[0] + b*v2[2]
#     mi, ma = __findMinMax([p0, p2])
#     rad = (fa * voxSize[0]) + (fb * voxSize[2])
#     return ~((mi > rad) | (ma < -rad))



# def __axistestY1(a, b, fa, fb, v0, v1, voxSize):
#     p0 = -a*v0[0] + b*v0[2]
#     p1 = -a*v1[0] + b*v1[2]
#     mi, ma = __findMinMax([p0, p1])
#     rad = (fa * voxSize[0]) + (fb * voxSize[2])
#     return ~((mi > rad) | (ma < -rad))



# def __axistestZ12(a, b, fa, fb, v1, v2, voxSize):
#     p1 = a*v1[0] - b*v1[1]
#     p2 = a*v2[0] - b*v2[1]
#     mi, ma = __findMinMax([p1, p2])
#     rad = (fa * voxSize[0]) + (fb * voxSize[1])
#     return ~((mi > rad) | (ma < -rad))



# def __axistestZ0(a, b, fa, fb, v0, v1, voxSize): 
#     p0 = a*v0[0] - b*v0[1]
#     p1 = a*v1[0] - b*v1[1]
#     mi, ma = __findMinMax([p0, p1])
#     rad = (fa * voxSize[0]) + (fb * voxSize[1])
#     return ~((mi > rad) | (ma < -rad))



# def _oldfilterTriangles(tris, points, voxCent, voxSize):
#     """Logical filter triangles inside specified voxel

#     Args: 
#         tris: t x 3 matrix of triangle node indices
#         points: p x 3 matrix of surface nodes
#         voxCent: 1 x 3 vector voxel centre
#         voxSize: 1 x 3 vector voxel size
    
#     Returns
#         t x 1 vector of bool values, true if tri intersects voxel
#     """

#     fltr = np.zeros(tris.shape[0], dtype=bool)
#     for t in range(tris.shape[0]):
#         fltr[t] = tribox_wrapper(voxCent, voxSize, \
#             points[tris[t,:],:])

#     return fltr 


def _pTestManyRayTriangleIntersections(tris, points, start, ax1, ax2):
    """Vectorised version of testForIntersectionByAxis(). 
    This is a 2D test only. Although the triangles can be defined in 3D, 
    the test is performed in one dimension only (the one not specified
    by ax1 and ax2), the dimension along which the ray lies (the ray must
    be zero in ax1 and ax2). See findIntersections3D() for a general 
    3D implementation of this function. 

    Args:
        triangles: m x 3 matrix of triangle nodes
        points: n x 3 matrix of surface nodes for triangles 
        start: 1 x 3 vector from which the ray used for testing originates (the
            ray by definition is defined in only the ax3 dimension so it need not 
            be explicitly specified here)
        ax1, ax2: the other axes along which the ray DOES NOT travel 

    Returns: 
        logical filter of triangles intersected by ray
           
    With thanks to Tim Coalson for his adaptation of PNPOLY, for which 
    this is a direct MATLAB port. See: 
    https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.cxx#L510
    https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
    """

    szTs = tris.shape 
    flags = np.zeros(szTs[0], dtype=bool)
    
    # Loop variables 
    TIs = np.zeros(szTs[0], dtype=np.int16)
    TJs = np.zeros(szTs[0], dtype=np.int16)
    j = 2

    for i in range(3):

        # This logical filter replaces the if statement at the top level. We 
        # maintain the behaviour by constraining the below operations to
        # points that pass this test
        disc = np.not_equal(points[tris[:,i], ax1] < start[ax1], \
            points[tris[:,j], ax1] < start[ax1])

        fltr = (points[tris[:,i], ax1] < points[tris[:,j], ax1])
        TIs[fltr] = i
        TJs[fltr] = j 
        TIs[~fltr] = j
        TJs[~fltr] = i 

        # TIs / TJs are column subscripts for each row of logical matrices
        # (one for each row). Form a tuple pair of arrays for indexing, where
        # the first element is 1:N of row numbers, second are col numbers
        TIidx = (np.arange(szTs[0]), TIs)
        TJidx = (np.arange(szTs[0]), TJs)
        divs = (points[tris[TIidx], ax1] - points[tris[TJidx], ax1])
        
        fltr = (divs == 0)
        assert np.all(~disc[fltr])
        divs[fltr] = 1

        flips = ((points[tris[TIidx], ax2] - points[tris[TJidx], ax2]) \
                / divs) * (start[ax1] - points[tris[TJidx], ax1]) \
                + points[tris[TJidx], ax2] > start[ax2]

        # Apply the flip for points that pass both tests, update loop var
        flags[flips & disc] = ~flags[flips & disc]
        j = i 

    return flags