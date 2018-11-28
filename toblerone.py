#!/usr/bin/env python3

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
import nibabel.nifti2
import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError 
from ctoblerone import _ctestTriangleVoxelIntersection, _cyfilterTriangles
from ctoblerone import _cytestManyRayTriangleIntersections

BAR_FORMAT = '{l_bar}{bar} {elapsed} | {remaining}'

# Class definitions -----------------------------------------------------------

class Hemisphere: 
    """Class to encapsulate the white and pial surfaces of a hemisphere.
    Initialise with a side identifier (L/R), the surfaces should be added
    directly via class to hemi.inSurf / hemi.outSurf
    """

    def __init__(self, side):
        self.side = side 
        self.inSurf = None 
        self.outSurf = None 
        return


    def surfs(self):
        """Iterator over the inner/outer surfaces"""
        return [self.inSurf, self.outSurf]


class Surface:
    """Class to contain a surface's points, triangles and normals matrix. 
    Normals are calculated upon initialisation, with the appropriate norm 
    direction flag. LUT and associations are intially set to None but will be 
    updated later on. Using associations data, a surface can be cast to patch
    for a particular voxel which reduces computational complexity by 
    discarding unncessary triangles.
    """

    def __init__(self, ps, ts):
        """ps: points, ts: triangles"""

        # Check inputs
        if ps.shape[1] != 3: 
            raise RuntimeError("Points matrices should be p x 3")

        if ts.shape[1] != 3: 
            raise RuntimeError("Triangles matrices should be t x 3")

        if (np.max(ts) != ps.shape[0]-1) | (np.min(ts) != 0):
            raise RuntimeError("Incorrect points/triangle indexing")
            
        self.points = ps
        self.tris = ts
        self.assocs = []
        self.LUT = []
        self.xProds = np.cross(
            ps[ts[:,2],:] - ps[ts[:,0],:], 
            ps[ts[:,1],:] - ps[ts[:,0],:], 
            axis=1)


    def rebaseTriangles(self, triNums):
        """Re-express a patch of a larger surface as a new points and triangle matrix pair, indexed from 0. Useful for reducing computational complexity when working with a small
        patch of a surface where only a few nodes in the points 
        array are required by the triangles matrix. 

        Args: 
            triNums: t x 1 list of triangle numbers to rebase. 
        
        Returns: 
            (localPoints, localTris) tuple of re-indexed points/tris. 
        """

        localPoints = np.empty((0, 3), dtype=np.float32)
        localTris = np.zeros((len(triNums), 3), dtype=np.int32)
        pointsLUT = []

        for t in range(len(triNums)):
            for v in range(3):

                # For each vertex of each tri, check if we
                # have already processed it in the LUT
                vtx = self.tris[triNums[t],v]
                idx = np.argwhere(pointsLUT == vtx)

                # If not in the LUT, then add it and record that
                # as the new position. Write the missing vertex
                # into the local points array
                if not idx.size:
                    pointsLUT.append(vtx)
                    idx = len(pointsLUT) - 1
                    localPoints = np.vstack([localPoints, 
                        self.points[vtx,:]])

                # Update the local triangle
                localTris[t,v] = idx

        return (localPoints, localTris)


    def toPatch(self, voxIdx):
        """Return a patch object specific to a voxel given by linear index.
        Look up the triangles intersecting the voxel, and then load and rebase
        the points / xprods as required. 
        """

        triNums = self.assocs[self.LUT == voxIdx][0]
        (ps, ts) = self.rebaseTriangles(triNums)

        return Patch(ps, ts, self.xProds[triNums,:])

    
    def toPatchesForVoxels(self, voxIndices):
        """Return the patches for the voxels in voxel indices, flattened into 
        a single set of ps, ts and xprods. 
        """

        triNums = _flattenList(self.assocs[np.isin(self.LUT, voxIndices)])
        triNums = list(set(triNums))

        return Patch(self.points, self.tris[triNums,:], 
            self.xProds[triNums,:])


class Patch(Surface):
    """Subclass of Surface that represents a small patch of surface. 
    Points, trianlges and xProds are all inherited from the parent surface. 
    This class should not be directly created but instead instantiated via
    the Surface.toPatch() / toPatchesForVoxels() methods. 
    """

    def __init__(self, points, tris, xProds):
        self.points = points 
        self.tris = tris
        self.xProds = xProds 

    def shrink(self, fltr):
        """Return a shrunk copy of the patch by applying the logical 
        filter fltr to the calling objects tris and xprods matrices
        """

        return Patch(self.points, self.tris[fltr,:], self.xProds[fltr,:])


# Function definitions --------------------------------------------------------

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


def _quickCross(a, b):
    return np.array([
        (a[1]*b[2]) - (a[2]*b[1]),
        (a[2]*b[0]) - (a[0]*b[2]), 
        (a[0]*b[1]) - (a[1]*b[0])], dtype = np.float32)


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

    def __pointGroupsIntersect(grps, tris): 
        """Break as soon as overlap is found. Brute force approach."""
        for g in range(len(grps)):
            for h in range(g + 1, len(grps)): 
                if np.any(np.intersect1d(tris[grps[g],:], 
                    tris[grps[h],:])):
                    return True 

        return False 

    # Main function ---------------------

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
        while __pointGroupsIntersect(groups, tris): 
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



def _formAssociations(surf, FoVsize, cores):
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

    # Check for negative coordinates: these should have been sripped. 
    if np.round(np.min(surf.points)) < 0: 
        raise RuntimeError("formAssociations: negative coordinate found")

    if np.any(np.round(np.max(surf.points, axis=0)) >= FoVsize): 
        raise RuntimeError("formAssociations: coordinate outside FoV")

    chunks = _distributeObjects(np.arange(surf.tris.shape[0]), cores)
    workerFunc = functools.partial(_formAssociationsWorker, surf.tris, \
        surf.points, FoVsize)

    with multiprocessing.Pool(cores) as p:
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

                ind = _sub2ind(FoVsize, voxel.astype(np.int16))
                if ind in workerResults:
                    workerResults[ind].append(t)
                else: 
                    workerResults[ind] = [t]
    
    return workerResults


def _voxeliseSurfaceAlongDimension(FoVsize, dim, surf):
    """Fill the volume contained within the surface by projecting rays
    in a given direction. See "Simplification and repair of polygonal models 
    using volumetric techniques", Nooruddin & Turk, 2003 for an overview of 
    the conceptual approach taken.

    Args: 
        FoVsize: 1 x 3 vector of FoV dimensions in which surface is enclosed
        dim: int 0/1/2 dimension along which to project rays
        surf: complete surface object. 

    Returns: 
        logical array of FoV size, true where voxel is contained in surface
    """

    try: 
        # Initialsie empty mask, and loop over the OTHER dims to the one specified. 
        mask = np.zeros(np.prod(FoVsize), dtype=bool)
        otherDims = [ (dim+1)%3, (dim+2)%3 ]
        startPoint = np.zeros(3, dtype=np.float32)

        for d1 in range(FoVsize[otherDims[0]]):
            for d2 in range(FoVsize[otherDims[1]]):

                # Defined the start/end of the ray and gather all 
                # linear indices of voxels along the ray
                IJK = np.zeros((FoVsize[dim], 3), dtype=np.int16)
                IJK[:,dim] = np.arange(0, FoVsize[dim])
                IJK[:,otherDims[0]] = d1
                IJK[:,otherDims[1]] = d2
                startPoint[otherDims] = [d1, d2]
                voxRange = _sub2ind(FoVsize, (IJK[:,0], IJK[:,1], IJK[:,2]))

                # Find all associated triangles lying along this ray
                # and test for intersection
                patches = surf.toPatchesForVoxels(voxRange)

                if patches.tris.shape[0]:
                    intersectionMus = _findRayTriangleIntersections2D(startPoint, \
                        patches, dim)

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
                    intDs = startPoint[dim] + (intersectionMus[sorted])

                    # Assignment. All voxels before the first point of intersection
                    # are outside. The mask is already zeroed for these. All voxels
                    # between point 1 and n could be in or out depending on parity
                    for i in range(1, len(sorted)+1):

                        # Starting from infinity, all points between an odd numbered
                        # intersection and the next even one are inside the mask 
                        if ((i % 2) & ((i+1) <= len(sorted))):
                            indices = ((IJK[:,dim] > intDs[i-1]) 
                                & (IJK[:,dim] < intDs[i]))
                            mask[voxRange[indices]] = 1
                    
                        # All voxels beyond the last point of intersection are also outside. 

        return np.reshape(mask, FoVsize) 

    except Exception as e:
        raise e 



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
    # fltr2 = _cytestManyRayTriangleIntersections(patch.tris, patch.points, 
    #     testPnt, (axis+1)%3, (axis+2)%3)
    # assert np.array_equal(fltr, fltr2)

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
    fltr = _cytestManyRayTriangleIntersections(patch.tris, onPlane2d.T, start, 0, 1)

    # For those trianglest that passed, calculate multiplier to point of 
    # intersection
    mus = _findRayTriPlaneIntersections(patch.points[patch.tris[fltr,0],:], 
        patch.xProds[fltr,:], testPnt, ray)
    
    return mus



def _fullRayIntersectionTest(testPnt, surf, \
        voxIJK, imgSize):
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
    inds = _sub2ind(imgSize, (subs[:,0], subs[:,1], subs[:,2]))

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

    outAff = np.matmul(np.matmul(
        np.matmul(refAff, np.linalg.inv(refSpace)),
        transform), srcSpace)
    return np.matmul(outAff, np.linalg.inv(srcAff))


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
    i, j, k = _ind2sub(supersampler, linIdx)
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


def _estimateVoxelFraction(surf, voxIJK, voxIdx,
    imgSize, supersampler):
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

    # Test all subvox corners now and store the results for later
    allCorners = _getAllSubVoxCorners(supersampler, voxIJK, voxSize)
    voxCentFlag = _fullRayIntersectionTest(voxIJK, surf, voxIJK, imgSize)
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
        triFltr = _cyfilterTriangles(patch.tris, patch.points, subVoxCent, 
            subVoxSize)

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
    I, J, K = _ind2sub(FoVsize, np.arange(np.prod(FoVsize)))
    voxIJKs = np.vstack((I.flatten(), J.flatten(), K.flatten())).T
    voxIJKs = voxIJKs.astype(np.float32)

    # Prepare partial function application for the estimation
    workerChunks = _distributeObjects(voxList, 50)
    workerFractions = []
    estimateFractionsPartial = functools.partial(_estimateFractionsWorker, \
        surf, voxIJKs, FoVsize, supersampler)

    # Parallel processing, tqdm provides the progress bar
    if True:
        with multiprocessing.Pool(cores) as p: 
            for _, r in enumerate(tqdm.tqdm(
                p.imap(estimateFractionsPartial, workerChunks), 
                total=len(workerChunks), desc=descriptor, 
                bar_format=BAR_FORMAT, ascii=True)):
                workerFractions.append(r)

    # Serial processing, again with progress bar.
    else: 
        for chunk in tqdm.trange(len(workerChunks), 
            desc=descriptor, bar_format=BAR_FORMAT, ascii=True):
            workerFractions.append(estimateFractionsPartial(workerChunks[chunk]))

    # Aggregate the results back together. 
    if any(map(lambda r: isinstance(r, Exception), workerFractions)):
        raise RuntimeError("Exception was raised during worker estimation")

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

        counter = 0 
        for v in workerVoxList:

            # Load voxel centre, estimate vols, and rescale to PVs
            voxijk = voxIJK[v,:]
            fraction = _estimateVoxelFraction(surf, \
                voxijk, v, imgSize, supersampler)
            
            partialVolumes[counter] = fraction
            counter += 1 
        
        return partialVolumes

    except Exception as e:
        return e



def estimatePVs(**kwargs):
    """Estimate partial volumes on the cortical ribbon"""

    verbose = True 
    if kwargs.get('cores') is not None:
        cores = kwargs['cores']
    else: 
        cores = multiprocessing.cpu_count()

    # If subdir given, then get all the surfaces out of the surf dir
    # If individual surface paths were given they will already be in scope
    if 'FSdir' in kwargs:
        FSdir = op.join(kwargs['FSdir'], 'surf')
        
        if not op.isdir(FSdir):
            raise RuntimeError("Subject's surf/ directory does not exist")

        kwargs['LWS'] = op.join(FSdir, 'lh.white')
        kwargs['LPS'] = op.join(FSdir, 'lh.pial')
        kwargs['RWS'] = op.join(FSdir, 'rh.white')
        kwargs['RPS'] = op.join(FSdir, 'rh.pial')

    # Define the hemispheres we will be working with and check surfaces exist. 
    # Check file formats
    hemispheres = []
    if (kwargs.get('LWS') is not None) & (kwargs.get('LPS') is not None): 
        hemispheres.append(Hemisphere('L'))
        if not all(map(op.isfile, (kwargs['LPS'], kwargs['LWS']))):
            raise RuntimeError("LWS/LPS surface does not exist")
        surfExt = op.splitext(kwargs['LWS'])[-1]

    if (kwargs.get('RWS') is not None) & (kwargs.get('RPS') is not None): 
        hemispheres.append(Hemisphere('R'))
        if not all(map(op.isfile, (kwargs['RPS'], kwargs['RWS']))):
            raise RuntimeError("RWS/RPS surface does not exist")
        surfExt = op.splitext(kwargs['RWS'])[-1]

    if not len(hemispheres):
        raise RuntimeError("Specify at least one hemisphere's surfaces (eg LWS/LPS)")


    # Reference image path 
    if not 'ref' in kwargs:
        raise RuntimeError("Path to reference image must be given")

    if not op.isfile(kwargs['ref']):
        raise RuntimeError("Reference image does not exist")

    inExt = op.splitext(kwargs['ref'])[-1]
    if not inExt in [".nii", ".gz", ".mgh", ".mgz"]:
        raise RuntimeError("Reference must be in the \
        following formats: nii, nii.gz, mgh, mgz")

    if '.nii.gz' in kwargs['ref']:
        inExt = '.nii.gz'


    # Structural to reference transformation. Either as array or path
    # to file containing matrix
    if kwargs.get('struct2ref') is not None:
        if (type(kwargs['struct2ref']) is str):
            _, matExt = op.splitext(kwargs['struct2ref'])

            try: 
                if matExt == '.txt':
                    matrix = np.loadtxt(kwargs['struct2ref'], 
                        dtype=np.float32)
                elif matExt in ['.npy', 'npz', '.pkl']:
                    matrix = np.load(kwargs['struct2ref'])
                else: 
                    matrix = np.fromfile(kwargs['struct2ref'], 
                        dtype=np.float32)
            except Exception as e:
                warnings.warn("""Could not load struct2ref matrix. 
                    File should be any type valid with numpy.load().""")
                raise e 
            kwargs['struct2ref'] = matrix

    else: 
        warnings.warn("No structural to reference transform given. Assuming identity.")
        kwargs['struct2ref'] = np.identity(4)

    if not kwargs['struct2ref'].shape == (4,4):
        raise RuntimeError("struct2ref must be a 4x4 matrix")

    # Is this a FLIRT transform? If so we need to do some clever preprocessing
    if kwargs.get('flirt'):
        if not 'struct' in kwargs:
            raise RuntimeError("If using a FLIRT transform, the path to the \
                structural image must also be given")
        if not op.isfile(kwargs['struct']):
            raise RuntimeError("Structural image does not exist")
        kwargs['struct2ref'] = _adjustFLIRT(kwargs['struct'], kwargs['ref'], 
            kwargs['struct2ref'])


    # Read in input image properties 
    imgObj = nibabel.load(kwargs['ref'])
    voxSize = (imgObj.header['pixdim'])[1:4]
    imgSize = (imgObj.header['dim'])[1:4]
    world2vox = np.linalg.inv(imgObj.affine)

    # Prepare output directory. If given then create it 
    # If not then use the input image dir
    if kwargs.get('outdir'):
        if not op.isdir(kwargs['outdir']):
            os.mkdir(kwargs['outdir'])
    else: 
        dname = op.dirname(kwargs['ref'])
        if dname == '':
            dname = os.getcwd()
        kwargs['outdir'] = dname

    # Create output dir for transformed surfaces
    if kwargs.get('savesurfs'):
        transSurfDir = op.join(kwargs['outdir'], 'surf_transform')
        if not op.isdir(transSurfDir):
            os.mkdir(transSurfDir)
    
    # Prepare the output filename. If not given then we pull it 
    # from the reference
    if  kwargs.get('name'):
        name = kwargs['name']
    else:  
        name = op.split(kwargs['ref'])[-1]
        name, inExt = op.splitext(name)
        name += '_tob'

    outExt = ''
    fname = name
    while ('.nii' in fname) | ('.nii.gz' in fname): 
        fname, e = op.splitext(fname)
        outExt = e + outExt
    
    if outExt == '': 
        outExt = inExt 
    
    name = fname
    maskName = name + "_surfmask"
    assocsName = name + "_assocs"
    assocsPath = op.join(kwargs['outdir'], assocsName + '.pkl')


    # Load surface geometry information 
    # If FS binary format, we can read cRAS straight out of the meta dict.
    if kwargs.get('cRAS') is None: 
        if surfExt != '.gii':
            surfName = hemispheres[0].side + 'PS'
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
    for h in hemispheres: 
        for s in ['P', 'W']: 
            surfName = h.side + s + 'S'
            surfName = kwargs[surfName]

            if surfExt != '.gii':
                ps, ts = tuple(nibabel.freesurfer.io.read_geometry(\
                    surfName))
            else: 
                gft = nibabel.load(surfName).darrays
                ps, ts = tuple(map(lambda o: o.data, gft))

            # Transform the surfaces to reference space
            ps = _affineTransformPoints(ps, transform)

            # # Save the transformed surfaces 
            # if kwargs.get('savesurfs'):
            #     _, fl = op.split(surfName) 
            #     transSurfName = op.join(transSurfDir, 'tob_' + fl)
            #     if surfExt != '.gii':
            #         nibabel.freesurfer.io.write_geometry(transSurfName, ps, ts)
            #     else: 
            #         gft2 = copy.deepcopy(gft)
            #         gft2.darrays[0:2] = (ps, ts)
            #         nibabel.save(gft2, transSurfName)

            # Transform the surfaces to voxel space
            ps = _affineTransformPoints(ps, world2vox)
            ps = ps.astype(np.float32)
            ts = ts.astype(np.int32)

            # Final indexing checks
            if not (np.min(ts) == 0) & (np.max(ts) == ps.shape[0] - 1):
                raise RuntimeError("Vertex/triangle indexing incorrect")

            # Write the surface into the hemisphere obj
            surf = Surface(ps, ts)
            if s == 'P': h.outSurf = surf 
            else: h.inSurf = surf 


    # FoV and associations loop -----------------------------------------------

    # Check the FoV of the reference image against the extents of the surfaces
    # Warn if FoV does not contain whole surface
    s1 = hemispheres[0].outSurf.points
    s2 = hemispheres[-1].outSurf.points
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
            for s in h.surfs(): 
                s.points = s.points + FoVoffset

                assert np.all(np.floor(np.min(s.points, axis=0)) >= 0), \
                    "FoV offset does not remove negative coordinates"
                assert np.all(np.ceil(np.max(s.points, axis=0)) < \
                    fullFoVsize), "Full FoV does not contain all surface coordinates"

    # Form (or read in) associations
    # We use a "stamp" matrix to check that saved/loaded assocs
    # are referenced to the right image space
    stamp = np.matmul(kwargs['struct2ref'], world2vox)
    inAssocs = {}; outAssocs = {} 
    recomputeAssocs = False 

    # Associations have been pre-computed 
    if op.isfile(assocsPath):
        print("Loading pre-computed associations found in the output", \
            "directory:", assocsPath)

        with open(assocsPath, 'rb') as f:

            oldStamp, inAssocsDict, outAssocsDict = pickle.load(f)
            if not np.array_equal(oldStamp, stamp): 
                raise RuntimeError("Pre-loaded associations stamp does not match", \
                    "surface/image geometry. Delete the associations.")
        
        # Check the loaded assocs are good to go (not nones)
        for h in hemispheres: 
            if not ((inAssocsDict[h.side] is not None) & (outAssocsDict[h.side] is not None)):
                print("Loaded associations are not complete, will recompute.")
                recomputeAssocs = True 
            
        # If so, unpack the associations into their keys (LUT) and values. 
            else:
                h.inSurf.LUT = np.array(list(inAssocsDict[h.side].keys()), dtype=np.int32)
                h.outSurf.LUT = np.array(list(outAssocsDict[h.side].keys()), dtype=np.int32)
                h.inSurf.assocs = np.array(list(inAssocsDict[h.side].values()))
                h.outSurf.assocs = np.array(list(outAssocsDict[h.side].values()))
    
    # Associations need computing
    elif (recomputeAssocs) or not (op.isfile(assocsPath)): 
        if 'saveassocs' in kwargs: 
            print("Forming voxel associations, saving to:", assocsPath)
        else: 
            print("Forming voxel associations")

        # Initialise all these as none and overwrite them with true values if the 
        # hemispheres are available to do so 
        if 'saveassocs' in kwargs: 
              inAssocsDict = {}; outAssocsDict = {}

        for h in hemispheres: 
            inAssocs, outAssocs = list(map(_formAssociations, h.surfs(), 
                [fullFoVsize, fullFoVsize], [cores, cores]))
            h.inSurf.LUT = np.array(list(inAssocs.keys()), dtype=np.int32)
            h.inSurf.assocs = np.array(list(inAssocs.values()))
            h.outSurf.LUT = np.array(list(outAssocs.keys()), dtype=np.int32)
            h.outSurf.assocs = np.array(list(outAssocs.values()))

            if 'saveassocs' in kwargs: 
                inAssocsDict[h.side] = inAssocs
                outAssocsDict[h.side] = outAssocs 

        if 'saveassocs' in kwargs: 
            with open(assocsPath, 'wb') as f: 
                pickle.dump((stamp, inAssocsDict, outAssocsDict), f)


    # And now pass off to the actual toblerone estimation
    supersampler = np.ceil(voxSize).astype(np.int8)
    print("Supersampling factor set at:", supersampler)

    # Generate the list of voxels to be processed. Start with the set of all
    # voxels within the reference image 
    I,J,K = np.meshgrid(np.arange(imgSize[0]), np.arange(imgSize[1]), \
        np.arange(imgSize[2]))
    voxSubs = np.vstack((I.flatten(), J.flatten(), K.flatten())).T

    # Shift these into full FOV space with the offset
    voxSubs = voxSubs + FoVoffset 
    
    # Convert to linear indices within full FOV space
    voxList = _sub2ind(fullFoVsize, (voxSubs[:,0], voxSubs[:,1], voxSubs[:,2]))

    # Process each hemisphere
    for h in hemispheres:

        if np.any(np.max(np.abs(h.inSurf.points)) > 
            np.max(np.abs(h.outSurf.points))):
            raise RuntimeWarning("Inner surface vertices appear to be further",\
                "from the origin than the outer vertices. Are the surfaces in",\
                "the correct order?")
        
        # Estimate fractions against the inner surface
        if not kwargs.get('hard'):

            fracs = []; vlists = []
            for s, d in zip(h.surfs(), ['in', 'out']):
                descriptor = " {} {}".format(h.side, d)
                slist = np.intersect1d(voxList, s.LUT).astype(np.int32)
                f = _estimateFractions(s, fullFoVsize, supersampler, 
                    slist, descriptor, cores)
                fracs.append(f)
                vlists.append(slist)

        
        # Estimate bool masks against both surfaces. 
        voxelisePartial = functools.partial(_voxeliseSurfaceAlongDimension, \
            fullFoVsize, 0)

        # Parallel mode
        with multiprocessing.Pool(min(2, cores)) as p:
            fills = p.map(voxelisePartial, h.surfs())

        # Single threaded mode
        # fills = list(map(voxelisePartial, h.surfs()))

        # Write the fractions on top of the binary masks. 
        inFilled, outFilled = fills 
        inFractions = (inFilled.flatten()).astype(np.float32)
        outFractions = (outFilled.flatten()).astype(np.float32)

        if not kwargs.get('hard'):
            inFractions[vlists[0]] = fracs[0] 
            outFractions[vlists[1]] = fracs[1]

        # Combine estimates from each surface into whole hemi PV estimates
        hemiPVs = np.zeros((np.prod(fullFoVsize), 3), dtype=np.float32)
        hemiPVs[:,1] = inFractions 
        hemiPVs[:,0] = np.maximum(0.0, outFractions - inFractions)
        hemiPVs[:,2] = 1.0 - np.sum(hemiPVs[:,0:2], axis=1)

        # And write into the hemisphere object. 
        h.PVs = hemiPVs

    # Merge the fill masks by giving priority to GM, then WM, then CSF.
    if len(hemispheres) == 1:
        outPVs = hemispheres[0].PVs
    else:
        h1, h2 = hemispheres
        outPVs = np.zeros((np.prod(fullFoVsize), 3), dtype=np.float32)
        outPVs[:,0] = np.minimum(1.0, h1.PVs[:,0] + h2.PVs[:,0])
        outPVs[:,1] = np.minimum(1.0 - outPVs[:,0], \
            h1.PVs[:,1] + h2.PVs[:,1])
        outPVs[:,2] = 1.0 - np.sum(outPVs[:,0:2], axis=1)

    # Sanity checks
    if np.any(outPVs > 1.0): 
        raise RuntimeError("PV exceeds 1")

    if np.any(outPVs < 0.0):
        raise RuntimeError("Negative PV returned")

    if not np.all(np.sum(outPVs, axis=1) == 1.0):
        raise RuntimeError("PVs do not sum to 1")

    # Form the surface mask (3D logical) as any voxel containing GM or 
    # intersecting the cortex (these definitions should always be equivalent)
    assocsMask = np.zeros((outPVs.shape[0], 1), dtype=bool)
    for h in hemispheres:
        for s in h.surfs(): 
            assocsMask[s.LUT] = True
    assocsMask[outPVs[:,0] > 0] = True 

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

    # Finally, form the NIFTI objects and save the PVs. 
    if kwargs.get('nosave'):
        print("Saving PVs to", kwargs['outdir'])
        PVhdr = copy.deepcopy(imgObj.header)
        if PVhdr.sizeof_hdr == 540:
            makeNifti = nibabel.nifti2.Nifti2Image
        else: 
            makeNifti = nibabel.nifti1.Nifti1Image

        tissues = ['GM', 'WM', 'NB']
        if kwargs.get('nostack'):
            for t in range(3):
                PVobj = makeNifti(outPVs[:,:,:,t], 
                    imgObj.affine, header=PVhdr)
                outPath = op.join(kwargs['outdir'], fname + tissues[t] + outExt)
                nibabel.save(PVobj, outPath)
        else:
            outPath = op.join(kwargs['outdir'], fname + outExt)
            PVhdr['dim'][0] = 4
            PVhdr['dim'][4] = 3
            PVobj = makeNifti(outPVs, imgObj.affine, header=PVhdr)
            nibabel.save(PVobj, outPath)

        # Save the mask
        maskPath = op.join(kwargs['outdir'], maskName + outExt)
        print("Saving surface mask to", maskPath)
        maskHdr = copy.deepcopy(imgObj.header)
        maskObj = makeNifti(assocsMask, imgObj.affine, \
            header=maskHdr)
        nibabel.save(maskObj, maskPath)

    return (outPVs, assocsMask)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Toblerone: partial volume estimation on the \
        cortical ribbon",

        usage=
"""

Toblerone: partial volume estimation on the cortical ribbon. 

By default output PV estimates will be saved as a single 4D image with GM, WM and non-brain in frames 0-2
This tool may be run either at the command line or as a module within a Python script. 

Required arguments: 
    --ref           path to reference image for which to estimate PVs
    --FSdir         path to a FreeSurfer subject directory; surfaces will be loaded from the /surf dir. 
                        Alternative to LWS/LPS/RWS/RPS, in .gii or .white/.pial format
    --LWS, --LPS    paths to left hemisphere white and pial surfaces respectively 
    --RWS, --RPS    as above for right hemisphere
    --struct2ref    path to structural (from which surfaces were produced) to reference transform.
                        Set '--struct2ref I' for identity transform. NB if this is a FSL FLIRT
                        transform then set the --flirt flag

Optional arguments:
    --name          output filename (ext optional). Defaults to reference filename with suffix _tob  
    --outdir        output directory. Defaults to directory containing the reference image   
    --flirt         flag, signifying that the --struct2ref transform was produced by FSL's FLIRT
                        If set, then a path to the structural image from which surfaces were 
                        produced must also be given     
    --struct        path to structural image (ie, what FreeSurfer was run on)
    --hard          don't estimate PVs, instead simply assign whole-voxel tissue volumes based on position
                        relative to surfaces
    --nostack       don't stack each tissue estimates into single image, save each separately 
    --saveassocs    save triangle/voxel associations data (debug tool)
    --cores         number of (logical) cores to use, default is maximum available
    

File formats:
    Surfaces are loaded either via the nibabel.freesurferio (.white/pial) or freesurfer.load (.gii)
    modules. 
    Images are loaded via the nibabel.load module (.nii/.nii.gz)
    Transformation matrices are loaded via np.fromtxt(), np.load() or np.fromfile() functions. 


Tom Kirk, November 2018
Institute of Biomedical Engineering / Welcome Centre for Integrative Neuroimaging
University of Oxford
thomas.kirk@eng.ox.ac.uk
"""
    )

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--FSdir', type=str, required=False)
    parser.add_argument('--LWS', type=str, required=False)
    parser.add_argument('--LPS', type=str, required=False)
    parser.add_argument('--RWS', type=str, required=False)        
    parser.add_argument('--RPS', type=str, required=False)
    parser.add_argument('--struct2ref', type=str, required=True) 
    parser.add_argument('--flirt', action='store_true')
    parser.add_argument('--struct', type=str, required=False)
    parser.add_argument('--outdir', type=str, required=False)
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--nostack', action='store_true', required=False)
    parser.add_argument('--saveassocs', action='store_true', required=False)
    parser.add_argument('--cores', type=int, required=False)

    args = vars(parser.parse_args())
    estimatePVs(**args)
    