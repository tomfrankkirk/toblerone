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
import argparse

import numpy as np 
import nibabel
import nibabel.freesurfer.io
import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError 

from . import pvcore
from .ctoblerone import _ctestTriangleVoxelIntersection, _cyfilterTriangles, \
    _cytestManyRayTriangleIntersections
from .classes import ImageSpace, Structure, Hemisphere, Surface, Patch



BAR_FORMAT = '{l_bar}{bar} {elapsed} | {remaining}'

def voxelise(imgSize, surface):
    """Voxelise (create binary in/out mask) a surface within a voxel grid
    of size imgSize. Surface coordinates must be in 0-indexed voxel units, 
    as will the voxel grid be interpreted (ie, 0 : imgSize - 1 in xyz). 
    Method is defined as static on the class for compataibility with 
    multiprocessing.Pool().

        Args: 
            imgSize: 3-vector of voxel grid dimensions
            surface: surface object. 

        Returns: 
            a flat boolean mask indicating voxels contained within the
                surface. Use reshape(imgSize) as required. 
    """

    # Test along the largest dim possible to minimise number of loops
    # We loop over the other two dims 
    dim = np.argmax(imgSize)

    try: 

        mask = np.zeros(np.prod(imgSize), dtype=bool)
        otherDims = [ (dim+1)%3, (dim+2)%3 ]
        startPoint = np.zeros(3, dtype=np.float32)

        for (d1,d2) in itertools.product(
            range(imgSize[otherDims[0]]), range(imgSize[otherDims[1]])):

            # Defined the start/end of the ray and gather all 
            # linear indices of voxels along the ray
            IJK = np.zeros((imgSize[dim], 3), dtype=np.int16)
            IJK[:,dim] = np.arange(0, imgSize[dim])
            IJK[:,otherDims[0]] = d1
            IJK[:,otherDims[1]] = d2
            startPoint[otherDims] = [d1, d2]
            voxRange = np.ravel_multi_index((IJK[:,0], IJK[:,1], IJK[:,2]), 
                imgSize)

            # Find all associated triangles lying along this ray
            # and test for intersection
            patches = surface.toPatchesForVoxels(voxRange)

            if patches is not None:
                intersectionMus = _findRayTriangleIntersections2D(startPoint, \
                    patches, dim)

                if not intersectionMus.shape[0]:
                    continue
                
                # If intersections were found, perform a parity test. 
                # Any ray should make an even number of intersections
                # as it crosses from -ve to +ve infinity
                if (intersectionMus.shape[0] % 2):
                    raise RuntimeError("fillSurfaceAlongDimension: \
                        odd number of intersections found. Does the FoV \
                        cover the full extents of the surface?")

                # Calculate points of intersection along the ray
                sorted = np.argsort(intersectionMus)
                intDs = startPoint[dim] + (intersectionMus[sorted])

                # Assignment. All voxels before the first point of intersection
                # are outside. The mask is already zeroed for these. All voxels
                # between point 1 and n could be in or out depending on parity
                for i in range(1, len(sorted)+1):

                    # Starting from infinity, all points between an odd numbered
                    # intersection and the next even one are inside the mask 
                    # Points beyond the last intersection are outside the mask
                    if ((i % 2) & ((i+1) <= len(sorted))):
                        indices = ((IJK[:,dim] > intDs[i-1]) 
                            & (IJK[:,dim] < intDs[i]))
                        mask[voxRange[indices]] = 1

        return mask

    except Exception as e:
        print("Error voxelising surface.")
        raise e 



def _quickCross(a, b):
    return np.array([
        (a[1]*b[2]) - (a[2]*b[1]),
        (a[2]*b[0]) - (a[0]*b[2]), 
        (a[0]*b[1]) - (a[1]*b[0])], dtype = np.float32)



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



def _dotVectorAndMatrix(vec, mat):
    """Row-wise dot product of a vector and matrix. 
    Returns a vector with the same number of rows as
    the matrix with the dot prod of that row with the
    vector in each row"""

    return np.sum(mat * vec, axis=1)



def _pointGroupsIntersect(grps, tris): 
    """Break as soon as overlap is found. Brute force approach."""
    for g in range(len(grps)):
        for h in range(g + 1, len(grps)): 
            if np.any(np.intersect1d(tris[grps[g],:], 
                tris[grps[h],:])):
                return True 

    return False 



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

    if not tris.shape[0]:
        return [] 

    groups = [] 

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

    # Merge groups that intersect 
    if len(groups) > 1: 
        while _pointGroupsIntersect(groups, tris): 
            didMerge = False 

            for g in range(len(groups)):
                if didMerge: break 

                for h in range(g + 1, len(groups)):
                    if didMerge: break

                    if np.any(np.intersect1d(tris[groups[g],:], 
                        tris[groups[h],:])):
                        groups[g] = groups[g] + groups[h]
                        groups.pop(h)
                        didMerge = True  

        # Check for empty groups 
    assert all(map(len, groups))
    
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



def _formAssociationsWorker(tris, points, FoVsize, triInds):
    workerResults = {}
    voxSize = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    for t in triInds:

        # Get vertices of triangle in voxel space (to nearest vox)
        tri = points[tris[t,:]]
        minVs = np.floor(np.min(tri, axis=0)).astype(np.int16)
        maxVs = np.ceil(np.max(tri, axis=0)).astype(np.int16)

        # Loop over the neighbourhood voxels of this bounding box
        neighbourhood = np.array(list(itertools.product(
            range(minVs[0], maxVs[0] + 1), 
            range(minVs[1], maxVs[1] + 1),
            range(minVs[2], maxVs[2] + 1))), dtype=np.float32)

        for voxel in neighbourhood: 

            if _ctestTriangleVoxelIntersection(voxel, voxSize, tri):

                ind = np.ravel_multi_index(voxel.astype(np.int16), 
                    FoVsize)
                if ind in workerResults:
                    workerResults[ind].append(t)
                else: 
                    workerResults[ind] = [t]
    
    return workerResults



def _findRayTriangleIntersections2D(testPnt, patch, axis):
    """Find intersections between a ray and a patch of surface, testing along
    one coordinate axis only (XYZ). As the triangle intersection test used within
    is 2D only, triangles are first projected down onto a 2D plane normal to the 
    test ray and then tested for intersection. This is intended to be used for
    voxeliseSurfaces(), for ray testing in the general 3D case use
    findRayTriangleIntersections3D() instead. 

    Args: 
        testPnt: 1 x 3 vector for ray origin
        patch: patch object for surface within the voxel
        axis: 0 for X, 1 for Y, 2 for Z, along which to test 

    Returns: 
        1 x j vector of multipliers along the ray to the points of intersection
    """

    ray = np.zeros(3, dtype=np.float32)
    ray[axis] = 1 

    # Filter triangles that intersect with this ray 
    fltr = _cytestManyRayTriangleIntersections(patch.tris, patch.points, 
        testPnt, (axis+1)%3, (axis+2)%3)

    # And find the multipliers for those that do intersect 
    if np.any(fltr):
        mus = _findRayTriPlaneIntersections(patch.points[patch.tris[fltr,0],:],
            patch.xProds[fltr,:], testPnt, ray)
    else:
        mus = np.array([])

    return mus



def _normalToVector(vec):
    """Return a normal to the given vector"""

    if np.abs(vec[2]) < np.abs(vec[0]):
        normal = np.array([vec[1], -vec[0], 0])
    else:
        normal = np.array([0, -vec[2], vec[1]])

    return normal 



def _findRayTriPlaneIntersections(planePoints, normals, testPnt, ray):
    """Find points of intersection between a ray and the planes defined by a
    set of triangles. As these points may not lie within their respective 
    triangles, these results must be further filtered using 
    vectorTestForRayTriangleIntersection() to identify genuine intersections.

    Args:
        planePoints: t x 3 matrix of points lying in each triangle's plane
        normals: t x 3 matrix of triangle plane normals
        testPnt: 1 x 3 vector of ray origin
        ray: 1 x 3 direction vector of ray

    Returns: 
        1 x j vector of multipliers along the ray to the points of intersection
    """

    # Compute dot of each normal with the ray
    dotRN = _dotVectorAndMatrix(ray, normals)

    # And calculate the multiplier.
    mu = np.sum((planePoints - testPnt) * normals, axis=1) / dotRN 

    return mu 



def _findRayTriangleIntersections3D(testPnt, ray, patch):
    """Find points of intersection between a ray and a surface. Triangles
    are projected down onto a 2D plane normal to the ray and then tested for
    intersection using findRayTriangleIntersections2D(). See: 
    https://stackoverflow.com/questions/2500499/howto-project-a-planar-polygon-on-a-plane-in-3d-space
    https://stackoverflow.com/questions/11132681/what-is-a-formula-to-get-a-vector-perpendicular-to-another-vector

    Args: 
        testPnt: 1 x 3 vector for origin of ray
        ray: 1 x 3 direction vector of ray
        patch: patch object for surface within the voxel

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
    d2 = _normalToVector(ray)
    d1 = _quickCross(d2, ray)

    # Calculate the projection of each point onto the direction vector of the
    # surface normal. Then subtract this component off each to leave their position
    # on the plane and shift coordinates so the test point is the origin.
    lmbda = _dotVectorAndMatrix(ray, patch.points)
    onPlane = (patch.points - np.outer(lmbda, ray)) - testPnt 

    # Re-express the points in 2d planar coordiantes by evaluating dot products with
    # the d2 and d3 in-plane orthonormal unit vectors
    onPlane2d = np.array(
        [_dotVectorAndMatrix(d1, onPlane), 
         _dotVectorAndMatrix(d2, onPlane),
         np.zeros(onPlane.shape[0])], dtype=np.float32)

    # Now perform the test 
    start = np.zeros(3, dtype=np.float32)
    fltr = _cytestManyRayTriangleIntersections(patch.tris, 
        onPlane2d.T, start, 0, 1)

    # For those trianglest that passed, calculate multiplier to point of 
    # intersection
    mus = _findRayTriPlaneIntersections(patch.points[patch.tris[fltr,0],:], 
        patch.xProds[fltr,:], testPnt, ray)
    
    return mus



def _fullRayIntersectionTest(testPnt, surf, voxIJK, imgSize):
    """To be used in conjunction with reducedRayIntersectionTest(). Determine if a 
    point lies within a surface by performing a ray intersection test against
    all the triangles of a surface (not the reduced form used elsewhere for 
    speed) This is used to define a root point for the reduced ray intersection
    test defined in reducedRayIntersectionTest(). 

    Inputs: 
        testPnt: 1 x 3 vector for point under test
        surf: complete surface object
        voxIJK: the IJK subscripts of the voxel in which the testPnt lies
        imgSize: the dimensions of the image in voxels

    Returns: 
        bool flag if the point lies within the surface
    """

    # Get the voxel indices that lie along the ray (inc the current vox)
    dim = np.argmin(imgSize)
    subs = np.tile(voxIJK, (imgSize[dim], 1)).astype(np.int32)
    subs[:,dim] = np.arange(0, imgSize[dim])
    inds = np.ravel_multi_index((subs[:,0], subs[:,1], subs[:,2]),
        imgSize)

    # Form the ray and fetch all appropriate triangles from assocs data
    patches = surf.toPatchesForVoxels(inds)
    intXs = _findRayTriangleIntersections2D(testPnt, patches, dim)

    # Classify according to parity of intersections. If odd number of ints
    # found between -inf and the point under test, then it is inside
    assert (len(intXs) % 2) == 0, 'Odd number of intersections returned'
    return ((np.sum(intXs <= 0) % 2) == 1)



def _reducedRayIntersectionTest(testPnts, patch, rootPoint, flip):
    """Shortened form of the full ray intersection test, working with only 
    a small patch of surface. Determine if test points are contained or 
    not within the patch. 

    Args: 
        testPnts: p x 3 matrix of points to test
        patch: patch object for surface within the voxel
        rootPoint: 1 x 3 vector for a point to which all rays used for 
            testing will be drawn from point under test
        flip: bool flag, false if the root point is OUTSIDE the surface
    
    Returns: 
        vector of bools, length p, denoting if the points are INSIDE.
    """

    # If no tris passed in throw error
    if not patch.tris.shape[0]:
        raise RuntimeError("No triangles to test against")
    
    # Each point will be tested by drawing a ray to the root point and 
    # testing for intersection against all available triangles
    flags = np.zeros(testPnts.shape[0], dtype=bool)
    rays = rootPoint - testPnts 

    for p in range(testPnts.shape[0]):

        # If the root point and test point are the same, then the ray
        # is zeros and shares the same classification as the root pt. 
        # Only proceed if the ray is non-zero
        if not np.any(rays[p,:]):
            shouldAppend = True 
        
        else: 

            # Find intersections, if they exist classify according to parity. 
            # Note the following logic assumes the rootPoint is inisde, ie, 
            # flip is false, if this is not the case then we will simply invert
            # the results at the end. 
            intMus = _findRayTriangleIntersections3D(testPnts[p,:], rays[p,:], patch)

            if intMus.shape[0]:

                # Filter down to intersections in the range (0,1)
                intMus = intMus[(intMus < 1) & (intMus > 0)]

                # if no ints in this reduced range then point is inside
                # because an intersection would otherwise be guaranteed
                if not intMus.shape[0]:
                    shouldAppend = True 

                # If even number, then point inside
                else: 
                    shouldAppend = not(intMus.shape[0] % 2)
                
            # Finally, if there were no intersections at all then the point is
            # also inside (given the root point is also inside, if the test pnt
            # was outside there would necessarily be an intersection)
            else: 
                shouldAppend = True 
        
        flags[p] = shouldAppend
    
    # Flip results if required
    if flip:
        flags = ~flags 
    
    return flags 



def _findTrianglePlaneIntersections(patch, voxCent, voxSize):
    """Find the points of intersection of all triangle edges with the 
    planar faces of the voxel. 

    Args: 
        patch: patch object for surface within the voxel
        voxCent: 1 x 3 vector of voxel centre
        voxSize: 1 x 3 vector of voxel dimensions

    Returns: 
        n x 3 matrix of n intersection coordinates 
    """
    if not patch.tris.shape[0]:
        return np.zeros((0,3))

    # Form all the edge vectors of the patch, strip out repeats
    edges = np.hstack((np.vstack((patch.tris[:,2], patch.tris[:,1])), \
        np.vstack((patch.tris[:,1], patch.tris[:,0])), \
        np.vstack((patch.tris[:,0], patch.tris[:,2])))).T 

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

            # Form the edge vectors and filter out those with zero component
            # in this dimension
            edgeVecs = patch.points[edges[:,1],:] - patch.points[edges[:,0],:]
            fltr = np.flatnonzero(edgeVecs[:,dim])
            edgeVecs = edgeVecs[fltr,:]
            pStart = patch.points[edges[fltr,0],:]

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



def _findVoxelSurfaceIntersections(patch, vertices):
    """Find points of intersection between edge and body vectors of a voxel
    with surface. Also detects folds along any of these vectors.

    Args: 
        patch: patch object for surface within the voxel
        vertices: vertices of the voxel
    
    Returns: 
        (intersects, fold) tuple: intersects the points of intersection, 
            fold a bool if one has been detected. If a fold is detected, 
            function returns immediately (without complete set of intersections)
    """

    intersects = np.empty((0,3))
    fold = False 

    # If nothing to test, silently return 
    if not patch.tris.shape[0]:
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
        intMus = _findRayTriangleIntersections3D(pnt, edge, patch)

        if intMus.shape[0]:
            intPnts = pnt + np.outer(intMus, edge)
            accept = np.logical_and(intMus <= 1, intMus >= 0)
            
            if sum(accept) > 1:
                fold = True
                return (intersects, fold)

            intersects = np.vstack((intersects, intPnts[accept,:]))

    return (intersects, fold)



def _safeFormHull(points):
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



def _classifyVoxelViaRecursion(patch, voxCent, voxSize, \
        containedFlag):
    """Classify a voxel entirely via recursion (not using any 
    convex hulls)
    """

    super2 = 5
    Nsubs2 = super2**3

    sX = np.arange(1 / (2 * super2), 1, 1 / super2, dtype=np.float32) - 0.5
    sX, sY, sZ = np.meshgrid(sX, sX, sX)
    subVoxCents = np.vstack((
        sX.flatten() * voxSize[0], 
        sY.flatten() * voxSize[1],  
        sZ.flatten() * voxSize[2])).T + voxCent
    flags = _reducedRayIntersectionTest(subVoxCents, patch, voxCent, \
        ~containedFlag)

    return (np.sum(flags) / Nsubs2)



def _fetchSubVoxCornerIndices(linIdx, supersampler):
    """Map between linear subvox index number and the indices of its
    vertices within the larger grid of subvoxel vertices
    """

    # Get the IJK coords within the subvoxel grid. 
    # Vertices are then +1/0 from these coords
    i, j, k = np.unravel_index(linIdx, supersampler)
    subs = np.array([ [i, j, k], [i+1, j, k], [i, j+1, k], \
        [i+1, j+1, k], [i, j, k+1], [i+1, j, k+1], [i, j+1, k+1], \
        [i+1, j+1, k+1] ], dtype=np.int16) 

    # And map these vertix subscripts to linear indices within the 
    # grid of subvox vertices (which is always + 1 larger than supersamp)
    corners = np.ravel_multi_index((subs[:,0], subs[:,1], subs[:,2]), 
        supersampler + 1)

    return corners 



def _getAllSubVoxCorners(supersampler, voxCent, voxSize):
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

    return np.vstack((X.flatten(), Y.flatten(), Z.flatten())
        ).astype(np.float32).T



def _estimateVoxelFraction(surf, voxIJK, voxIdx, imgSize, supersampler):
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
    
    # The main function, here we go... ----------------------------------------

    verbose = False
    # print(voxIdx)

    # Hardcode voxel size as we now work in voxel coords. Intialise results
    voxSize = np.array([1,1,1], dtype=np.int8)
    inFraction = 0.0

    # Set up the subvoxel sizes and vols. 
    subVoxSize = (1.0 / supersampler).astype(np.float32)
    subVoxVol = np.prod(subVoxSize).astype(np.float32)

    # Rebase triangles and points for this voxel
    patch = surf.toPatch(voxIdx)
    assert np.all(_cyfilterTriangles(patch.tris, patch.points,
        voxIJK, voxSize.astype(np.float32)))

    # Test all subvox corners now and store the results for later
    allCorners = _getAllSubVoxCorners(supersampler, voxIJK, voxSize)
    voxCentFlag = surf.voxelised[voxIdx]
    allCornerFlags = _reducedRayIntersectionTest(allCorners, patch, \
        voxIJK, ~voxCentFlag)

    # Test all subvox centres now and store the results for later
    si = np.linspace(0, 1, 2*supersampler[0] + 1, dtype=np.float32) - 0.5
    sj = np.linspace(0, 1, 2*supersampler[1] + 1, dtype=np.float32) - 0.5
    sk = np.linspace(0, 1, 2*supersampler[2] + 1, dtype=np.float32) - 0.5
    [si, sj, sk] = np.meshgrid(si[1:-1:2], sj[1:-1:2], sk[1:-1:2])
    allCents = np.vstack((si.flatten(), sj.flatten(), sk.flatten())).T \
        + voxIJK
    allCentFlags = _reducedRayIntersectionTest(allCents, patch, voxIJK, \
        ~voxCentFlag)


    # Subvoxel loop starts here -----------------------------------------------

    for s in range(np.prod(supersampler)):

        # Get the centre and corners, prepare the sanity check
        subVoxClassified = False
        subVoxCent = allCents[s,:]
        subVoxFlag = allCentFlags[s]

        # Do any triangles intersect the subvox?
        triFltr = _cyfilterTriangles(patch.tris, patch.points, 
            subVoxCent, subVoxSize)

        # CASE 1 --------------------------------------------------------------

        # If no triangles intersect the subvox then whole-tissue classification
        # using the flip flags that were set by fullRayIntersectTest()
        if not np.any(triFltr): 

            if verbose: print("Whole subvox assignment")
            inFraction += (int(subVoxFlag) * subVoxVol)        
            subVoxClassified = True 

        # CASE 2: some triangles intersect the subvox -------------------------

        else: 

            # Shrink the patch appropriately, load corner flags 
            smallPatch = patch.shrink(triFltr)
            cornerIndices = _fetchSubVoxCornerIndices(s, supersampler)
            corners = allCorners[cornerIndices,:]
            cornerFlags = allCornerFlags[cornerIndices] 

            # Check for subvoxel edge intersections with the local patch of
            # triangles and for folds
            edgeIntXs, fold = _findVoxelSurfaceIntersections( \
                smallPatch, corners)

            # Separate points within the voxel into distinct clouds, to check
            # for multiple surface intersection
            groups = _separatePointClouds(smallPatch.tris)

            # If neither surface is folded within the subvox and there 
            # are no multiple intersections, we can form hulls. 
            if (not fold) & (len(groups) < 2):

                # Filter down surface nodes that are in the subvox
                localPs = patch.points[np.unique(smallPatch.tris),:]
                localPs = localPs[_filterPoints(localPs, subVoxCent, 
                    subVoxSize)]
                
                # Gather points together in preparation for forming 
                # hulls (the corners will be added just beforehand)
                hullPts = np.vstack( (localPs, edgeIntXs, \
                    _findTrianglePlaneIntersections(smallPatch, \
                    subVoxCent, subVoxSize)) )

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

                V = _safeFormHull(hullPts)
                if not V: 
                    L2fraction = _classifyVoxelViaRecursion(smallPatch, \
                        subVoxCent, subVoxSize, subVoxFlag)
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

                    L2fraction = _classifyVoxelViaRecursion(smallPatch, 
                        subVoxCent, subVoxSize, subVoxFlag)
                    inFraction += (L2fraction * subVoxVol)
                    subVoxClassified = True 
            
        # Sanity check: we should have classified the voxel by now
        if not subVoxClassified: 
            raise RuntimeError("Subvoxel has not been classified")

        if verbose: print(inFraction)

    # END of subvoxel loop ----------------------------------------------------
      
    if inFraction > 1.000001:
        raise RuntimeError('Fraction exceeds 1 in', voxIdx)

    if inFraction < 0:
        raise RuntimeError('Negative fraction in', voxIdx)

    return inFraction 



def _estimateFractions(surf, FoVsize, supersampler, \
    voxList, descriptor, cores):
    """Estimate fraction of voxels lying interior to surface. 

    Args: 
        surf: complete surface object. 
        FoVsize: dimensions of voxel grid required to contain surfaces, 
            to which the voxels in voxList are indexed. 
        supersampler: 1 x 3 vector of supersampling factor
        voxList: list of linear voxel indices within the FoV to process

    Returns: 
        vector of size prod(FoV)
    """

    if len(FoVsize) != 3: 
        raise RuntimeError("FoV size should be a 1 x 3 vector or tuple")

    # Compute all voxel centres
    voxIJKs = pvcore._coordinatesForGrid(FoVsize).astype(np.float32)
    workerChunks = _distributeObjects(voxList, 40)
    workerFractions = []

    # Prepare a partial of the estimator for the map application on the pool. 
    estimatePartial = functools.partial(_estimateFractionsWorker, 
        surf, voxIJKs, FoVsize, supersampler)

    # Select the appropriate iterator function according to whether progress 
    # bar is requested. Tqdm provides progress bar.  
    if bool(descriptor):
        iterator = functools.partial(tqdm.tqdm,
            total=len(workerChunks), desc=descriptor, bar_format=BAR_FORMAT,
            ascii=True)
    else: 
        iterator = iter

    # And map across worker chunks either in parallel or serial. 
    if cores > 1:
        with multiprocessing.Pool(cores) as p: 
            for r in iterator(p.imap(estimatePartial, workerChunks)): 
                workerFractions.append(r)
    else: 
        for r in iterator(map(estimatePartial, workerChunks)):
            workerFractions.append(r)

    # Aggregate the results back together and check for exceptions
    if any([ isinstance(r, Exception) for r in workerFractions ]):
        print("Exception was raised during worker estimation:")
        raise workerFractions[0]

    fractions = np.concatenate(workerFractions)

    # Sanity check: did all voxels in toEstimate get processeed?
    if not np.all(fractions >= 0):
        raise RuntimeError("Not all voxels in voxList processed.")

    if np.any(fractions > 1):
        raise RuntimeError("Fraction greater than 1 returned.")

    return fractions



def _estimateFractionsWorker(surf, voxIJK, imgSize, \
        supersampler, workerVoxList):

    try:
        partialVolumes = np.zeros(len(workerVoxList), dtype=np.float32)

        for idx, v in enumerate(workerVoxList):

            # Load voxel coordinate and estimate faction
            voxijk = voxIJK[v,:]
            partialVolumes[idx] = _estimateVoxelFraction(surf,
                voxijk, v, imgSize, supersampler)  
        
        return partialVolumes

    except Exception as e:
        return e



def _determineFullFoV(surfs, refSpace):

    # if any(map(lambda s: np.any(s.points < -0.5) or 
    # np.any(s.points > refSpace.imgSize-0.5), surfs)):
    #     print("Warning: the FoV of the reference image does fully enclose", 
    #         "the surfaces.")
    #     print("PVs will only be estimated within the reference FoV.")

    # Find the min/max coordinates of the surfaces
    minFoV = np.floor(np.array(
        [np.min(s.points, axis=0) for s in surfs]).min(axis=0))
    maxFoV = np.ceil(np.array(
        [np.max(s.points, axis=0) for s in surfs]).max(axis=0))

    # If the min/max range is larger than the reference FoV, then shift and 
    # expand the coordinate system to the minimal size required for surfs
    FoVoffset = np.maximum(-minFoV, np.zeros(3)).astype(np.int16)
    FoVsize = np.maximum(refSpace.imgSize + FoVoffset, maxFoV+1).astype(np.int16)

    return (FoVoffset, FoVsize)





