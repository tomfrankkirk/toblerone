# Core numerical functions for Toblerone 

# This module contains the key functions that operate on patches of surface (see
# classes module) to estimate the fraction of a voxel enclosed within said patch.
# The most computationally intensive methods are handled by Cython in the module 
# ctoblerone (see extern/ctoblerone.pyx). Finally, certain functions that are re-
# used between modules are defined in pvcore. 

# This module should not be directly interacted with: the modules estimators and 
# pvtools provide outward-facing wrappers for actual PV estimation. 

import functools
import itertools
import multiprocessing as mp
from scipy import sparse
import copy 

import numpy as np 
import tqdm
import trimesh
from scipy.spatial import ConvexHull, QhullError, Delaunay

from toblerone.ctoblerone import (_ctestTriangleVoxelIntersection,  
                                  _cyfilterTriangles,
                                  _cytestManyRayTriangleIntersections,
                                  _quick_cross)
from toblerone import utils 
from toblerone.utils import NP_FLOAT


# Module level constants ------------------------------------------------------

ZERO_3 = np.zeros(3, dtype=NP_FLOAT)

SUPER2 = 6 * np.ones(3, dtype=NP_FLOAT)
SUPER2_DIV = SUPER2.prod()

# edge vectors for a single voxel 
ORIGINS = np.array([1, 1, 1, 4, 4, 4, 5, 5, 8, 8, 6, 7, 1, 2, 3, 4,
    1, 1, 1, 8, 8, 8, 2, 2, 3, 4, 4, 6], dtype=np.int32) - 1
ENDS = np.array([2, 3, 5, 8, 2, 3, 6, 7, 6, 7, 2, 3, 8, 7, 6, 5,
    6, 4, 7, 5, 3, 2, 3, 5, 5, 6, 7, 7], dtype=np.int32) - 1

# iterating over dimensions xyz, xyz 
DIMS = np.array([0,1,2,0,1,2])

# vectors to the 6 faces of a voxel from the centre point 
VOX_HALF_CYCLE = np.array(((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5))) 
VOX_HALF_VECS = np.array((VOX_HALF_CYCLE, -1 * VOX_HALF_CYCLE)).reshape(6,3)

# corners of a voxel centered on origin 
SUBVOXCORNERS = (np.array([ 
        [0, 0, 0], [1, 0, 0], 
        [0, 1, 0], [1, 1, 0], 
        [0, 0, 1], [1, 0, 1], 
        [0, 1, 1], [1, 1, 1]], 
        dtype=NP_FLOAT) - 0.5)

# See the _vox_tri_weights_worker() function for an explanation of the 
# naming convention here. 
TETRA1 = np.array([[0,3,4,5],   # aABC
                   [0,1,2,4],   # abcB
                   [0,2,4,5]],  # acBC
                   dtype=np.int32)  

TETRA2 = np.array([[0,3,4,5],   # aABC
                   [0,1,2,5],   # abcC
                   [0,1,4,5]],  # abBC
                   dtype=np.int32) 

# For defining the edges of triangle within a mesh
TRI_EDGE_INDEXING = [{1,0}, {2,0}, {2,1}]
TRI_FULL_SET = set(range(3))

# tdqm progress bar format
BAR_FORMAT = '{l_bar}{bar} {elapsed} | {remaining}'

# Functions -------------------------------------------------------------------


def _filterPoints(points, voxCent, vox_size):
    """Logical filter of points inside a voxel"""

    return ((np.abs(points - voxCent) - vox_size/2) < 1e-4).all(1)


def _pointGroupsIntersect(grps, tris): 
    """For _separatePointClouds. Break as soon as overlap is found"""
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
    assert all(map(len, groups)), 'Empty group remains after merging'
    
    return groups 


def form_associations(points_vox, tris, space, cores=mp.cpu_count()):
    """
    Identify which triangles of a surface intersect each voxel. This 
    reduces the number of operations that need be performed later. The 
    results will be stored on the surface object (ie, self)

    Returns: 
        None, but associations (sparse CSR matrix of size (voxs, tris)
        and assocs_keys (array of voxel indices containint the surface)
        will be set on the calling object. 
    """

    assert utils.space_encloses_surface(space, points_vox)
    workerFunc = functools.partial(_formAssociationsWorker, 
                                    tris, points_vox, space.size)

    if cores > 1:
        chunks = utils._distributeObjects(range(tris.shape[0]), cores)
        with mp.Pool(cores) as p:
            worker_assocs = p.map(workerFunc, chunks, chunksize=1)

        assocs = worker_assocs[0]
        for a in worker_assocs[1:]:
            assocs += a 

    else:
        assocs = workerFunc(range(tris.shape[0]))

    # Assocs keys is a list of all voxels touched by any triangle
    assocs_keys = np.flatnonzero(assocs.sum(1).A)
    return assocs, assocs_keys


def _formAssociationsWorker(tris, points, grid_size, triInds):
    """
    Worker function for use with multiprocessing. See formAssociations
    
    Returns: 
        sparse CSR matrix, shape (n_voxels, n_tris), boolean values.  
    """

    vox_size = np.array([0.5, 0.5, 0.5], dtype=NP_FLOAT)
    assocs = sparse.dok_matrix((grid_size.prod(), tris.shape[0]), dtype=bool)

    for t in triInds:

        # Get vertices of triangle in voxel space (to nearest vox)
        # Loop over neighbourhood of voxels in bounding box 
        tri = points[tris[t,:]]
        lims = np.vstack((tri.min(0), tri.max(0)+1)).round().astype(np.int16)
        nhood = np.array(list(itertools.product(
            range(*lims[:,0]), range(*lims[:,1]), range(*lims[:,2]))), 
            dtype=NP_FLOAT)

        for ijk in nhood: 
            if _ctestTriangleVoxelIntersection(ijk, vox_size, tri):
                vox = np.ravel_multi_index(ijk.astype(np.int16), grid_size)
                assocs[vox,t] = 1 
    
    return assocs.tocsr()


def _findRayTriangleIntersections2D(testPnt, patch, axis):
    """Find intersections between a ray and a patch of surface, testing along
    one coordinate axis only (XYZ). As the triangle intersection test used within
    is 2D only, triangles are first projected down onto a 2D plane normal to the 
    test ray and then tested for intersection. This is intended to be used for
    Surface.voxelise(), for ray testing in the general 3D case use
    findRayTriangleIntersections3D() instead. 

    Args: 
        testPnt: 1 x 3 vector for ray origin
        patch: patch object for surface within the voxel
        axis: 0 for X, 1 for Y, 2 for Z, along which to test 

    Returns: 
        1 x j vector of multipliers along the ray to the points of intersection
    """

    ray = np.zeros(3, dtype=NP_FLOAT)
    ray[axis] = 1 

    # Filter triangles that intersect with this ray 
    fltr = _cytestManyRayTriangleIntersections(patch.tris, patch.points, 
        testPnt, (axis+1)%3, (axis+2)%3)

    # And find the multipliers for those that do intersect 
    if np.any(fltr):
        mus = _findRayTriPlaneIntersections(patch.points[patch.tris[fltr,0],:],
            patch.xprods[fltr,:], testPnt, ray)
    else:
        mus = np.array([])

    return mus


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

    # mu is defined as dot((p_plane - p_test), normal_tri_plane) ...
    #   / dot(ray, normal_tri_plane)
    dotRN = np.einsum('ij,j->i', normals, ray, casting='no')
    mu = np.einsum('ij,ij->i', planePoints - testPnt, normals, casting='no')

    return mu / dotRN


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
    
    # Intersection is tested using Tim Coalson's adaptation of PNPOLY for careful
    # testing of intersections between infinite rays and points. As TC's adaptation
    # is a 2D test only (with the third dimension being the direction of ray
    # projection), triangles are flattened into 2D before testing. This is done by
    # projecting all triangles onto the plane defined by the ray (acting as planar
    # normal) and then testing for ray intersection (where the ray now represents
    # the Z direction in this new projected space) amongst all the triangles in
    # dimensions 1 and 2 (XY). Define a new coordinate system (d unit vectors) 
    # with d3 along the ray, d2 and d1 in plane.
    if np.abs(ray[2]) < np.abs(ray[0]):
        d2 = np.array([ray[1], -ray[0], 0], dtype=NP_FLOAT)
    else:
        d2 = np.array([0, -ray[2], ray[1]], dtype=NP_FLOAT)
    d1 = _quick_cross(d2, ray)

    # Calculate the projection of each point onto the direction vector of the
    # surface normal. Then subtract this component off each to leave their position
    # on the plane and shift coordinates so the test point is the origin.
    lmbda = np.einsum('ij,j->i', patch.points, ray, casting='no')
    onPlane = (patch.points 
                - np.einsum('i,j->ij', lmbda, ray, casting='no')
                - testPnt)

    # Re-express the points in 2d planar coordiantes by evaluating dot products with
    # the d2 and d3 in-plane orthonormal unit vectors
    onPlane2d = np.array([ np.einsum('ij,j->i', onPlane, d1, casting='no'),
                           np.einsum('ij,j->i', onPlane, d2, casting='no'),
                           np.zeros(lmbda.size, dtype=NP_FLOAT)])

    # Now perform the test 
    fltr = _cytestManyRayTriangleIntersections(patch.tris, onPlane2d.T, ZERO_3,
        0, 1)

    # For those trianglest that passed, calculate multiplier to point of 
    # intersection
    mus = _findRayTriPlaneIntersections(patch.points[patch.tris[fltr,0],:], 
        patch.xprods[fltr,:], testPnt, ray)
    
    return mus


def _fullRayIntersectionTest(testPnt, surf, voxIJK, size):
    """To be used in conjunction with reducedRayIntersectionTest(). Determine if a 
    point lies within a surface by performing a ray intersection test against
    all the triangles of a surface (not the reduced form used elsewhere for 
    speed). This is used to define a root point for the reduced test. 
. 

    Inputs: 
        testPnt: 1 x 3 vector for point under test
        surf: complete surface object
        voxIJK: the IJK subscripts of the voxel in which the testPnt lies
        size: the dimensions of the image in voxels

    Returns: 
        bool flag if the point lies within the surface
    """

    # Get the voxel indices that lie along the ray (inc the current vox)
    dim = np.argmin(size)
    subs = np.tile(voxIJK, (size[dim], 1)).astype(np.int32)
    subs[:,dim] = np.arange(0, size[dim])
    inds = np.ravel_multi_index((subs[:,0], subs[:,1], subs[:,2]),
        size)

    # Form the ray and fetch all appropriate triangles from assocs data
    patches = surf.to_patches(inds)

    if patches: 
        intXs = _findRayTriangleIntersections2D(testPnt, patches, dim)

        # Classify according to parity of intersections. If odd number of ints
        # found between -inf and the point under test, then it is inside
        assert ((intXs.size % 2) == 0), 'Odd number of intersections returned'
        return (((intXs <= 0).sum() % 2) == 1)
    
    else: 
        return False 


def _reducedRayIntersectionTest(testPnts, patch, rootPoint, flip):
    """Shortened form of the full ray intersection test, working with only 
    a small patch of surface. Determine if test points are contained or 
    not within the patch. 

    Args: 
        testPnts: p x 3 matrix of points to test
        patch: patch object for surface within the voxel
        rootPoint: 1 x 3 vector for a point to which all rays used for 
            testing will be drawn from point under test
        flip: bool flag, True if the root point is OUTSIDE the surface
    
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
            intMus = _findRayTriangleIntersections3D(testPnts[p,:], rays[p,:], 
                patch)

            if intMus.size:

                # Filter down to intersections in the range (0,1)
                intMus = intMus[(intMus < 1) & (intMus > 0)]

                # if no ints in this reduced range then point is inside
                # because an intersection would otherwise be guaranteed
                if not intMus.size:
                    shouldAppend = True 

                # If even number, then point inside
                else: 
                    shouldAppend = not(intMus.size % 2)
                
            # Finally, if there were no intersections at all then the point is
            # also inside (given the root point is also inside, if the test pnt
            # was outside there would necessarily be an intersection)
            else: 
                shouldAppend = True 
        
        flags[p] = shouldAppend
    
    if flip:
        flags = ~flags 
    
    return flags 


def _findTriangleVoxFaceIntersections(patch, voxCent, vox_size):
    """Find points of intersection of all triangle edges within a patch with 
    the faces of a voxel given by cent and size. 

    Args: 
        patch: patch object for region surface within the voxel
        voxCent: 1 x 3 vector of voxel centre
        vox_size: 1 x 3 vector of voxel dimensions

    Returns: 
        n x 3 matrix of n intersection coordinates 
    """

    if not patch.tris.shape[0]:
        return np.zeros((0,3), dtype=NP_FLOAT)

    # Form all the edge vectors of the patch, then strip out repeats
    edges = np.concatenate((patch.tris[:,0],patch.tris[:,0],patch.tris[:,2],
        patch.tris[:,1],patch.tris[:,2],patch.tris[:,1])).reshape(2,-1).T

    nonrpt = np.empty((0,2), dtype=np.int16)
    for k in range(edges.shape[0]):
        if not np.any(np.all(np.isin(edges[k+1:,:], edges[k,:]), axis=1)):
            nonrpt = np.vstack((nonrpt, edges[k,:]))
    
    intXs = np.empty((0,3), dtype=NP_FLOAT)
    edgeVecs = (patch.points[nonrpt[:,1],:] - patch.points[nonrpt[:,0],:])

    # Iterate over each dimension, moving +0.5 and -0.5 of the voxel size
    # from the vox centre to define a point on the planar face of the vox
    face_points = voxCent + (VOX_HALF_VECS * vox_size)

    for face_point,dim in zip(face_points, DIMS):

        # Filter to edge vectors that are non-zero in this dimension 
        fltr = np.flatnonzero(edgeVecs[:,dim])
        pStart = patch.points[nonrpt[fltr,0],:]

        # Sneaky trick here: because planar normals are aligned with the 
        # coord axes we don't need to do a full dot product, just extract
        # the appropriate component of the difference vectors
        mus = (face_point - pStart)[:,dim] / edgeVecs[fltr,dim]
        pInts = pStart + (edgeVecs[fltr,:].T * mus).T 
        pInts2D = pInts - face_point 
        keep = np.all(np.abs(pInts2D) <= (vox_size/2), 1)    
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

    intersects = np.empty((0,3), dtype=NP_FLOAT)
    fold = False 

    # If nothing to test, silently return empty results / false flag
    if not patch.tris.size:
        return (intersects, fold)

    # 8 vertices correspond to 12 edge vectors along exterior edges and 
    # 4 body diagonals. This function uses a particular numbering convention, 
    # encoded in the module level constants ENDS and ORIGINS 
    edges = vertices[ENDS,:] - vertices[ORIGINS,:]

    # Test each vector against the surface
    for e in range(16):
        edge = edges[e,:]
        pnt = vertices[ORIGINS[e],:]
        intMus = _findRayTriangleIntersections3D(pnt, edge, patch)

        if intMus.size:
            accept = np.logical_and(intMus <= 1, intMus >= 0)
            
            if accept.sum() > 1:
                fold = True
                return (intersects, fold)

            intPnts = pnt + np.einsum('i,j->ij', intMus, edge, casting='no')
            intersects = np.vstack((intersects, intPnts[accept,:]))

    return (intersects, fold)


def _safeFormHull(points):
    """If three or less points are provided, or not enough distinct points 
    (eg coplanar points), then return 0 volume (recursion will be used 
    elsewhere). For everything else, let the exception continue up. 
    """

    if points.size > 8:
        try:
            hull = ConvexHull(points)
            return hull.volume
        except QhullError: 
            return 0
        except Exception as e: 
            raise e
    else: 
        return 0
        

def _get_subvoxel_grid(supersampler):
    """Generate grid of subvoxel centers"""

    steps = 1.0 / supersampler
    subs = np.indices(supersampler.astype(int)).reshape(3,-1).T / supersampler
    subs += (steps / 2) - 0.5 

    # subs2 = np.meshgrid(*[np.arange(s/2, 1, s, dtype=NP_FLOAT) for s in steps ]
    # )
    # subs2 = np.stack(subs2, axis=-1).reshape(-1,3) - 0.5
    # np.array_equal(subs, subs2)
    return subs.astype(NP_FLOAT)


# Pre-compute the subvoxel grid used for recursive processing (it will
# be the same size for all such voxels)
SUPER2_GRID = _get_subvoxel_grid(SUPER2.astype(int))


def _classifyVoxelViaRecursion(patch, voxCent, vox_size, containedFlag):
    """Classify a voxel via recursion (not using convex hulls)"""

    # Create a grid of subvoxels, calculate fraction by simply testing
    # each subvoxel centre coordinate. 
    subVoxCents = SUPER2_GRID + voxCent
    flags = _reducedRayIntersectionTest(subVoxCents, patch, voxCent, \
        ~containedFlag)

    return flags.sum() / SUPER2_DIV


def _estimateVoxelFraction(surf, voxIJK, voxIdx, supersampler):
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
    inFraction = 0.0

    # Set up the subvoxel sizes and vols. 
    subvox_size = (1.0 / supersampler).astype(NP_FLOAT)
    subVoxVol = subvox_size.prod()

    # Rebase triangles and points for this voxel
    voxCentFlag = surf.indexed.voxelised[voxIdx]
    patch = surf.to_patch(voxIdx)

    # Test all subvox centres now and store the results for later
    allCents = _get_subvoxel_grid(supersampler) + voxIJK
    allCentFlags = _reducedRayIntersectionTest(allCents, patch, voxIJK,
        ~voxCentFlag)


    # Subvoxel loop starts here -----------------------------------------------

    for s in range(supersampler.prod()):

        # Get the centre and corners, prepare the sanity check
        subVoxClassified = False
        subVoxCent = allCents[s,:]
        subVoxFlag = allCentFlags[s]

        # Do any triangles intersect the subvox?
        triFltr = _cyfilterTriangles(patch.tris, patch.points, 
            subVoxCent, subvox_size)

        # CASE 1 --------------------------------------------------------------

        # If no triangles intersect the subvox then whole-tissue classification
        # using the flip flags that were set by fullRayIntersectTest()
        if not triFltr.any(): 

            if verbose: print("Whole subvox assignment")
            inFraction += (int(subVoxFlag) * subVoxVol)        
            subVoxClassified = True 

        # CASE 2: some triangles intersect the subvox -------------------------

        else: 

            # Shrink the patch appropriately, calculate subvox corners
            smallPatch = patch.shrink(triFltr)
            corners = subVoxCent + ((SUBVOXCORNERS) * (subvox_size[None,:]))
            cornerFlags = _reducedRayIntersectionTest(corners, patch,
                voxIJK, ~voxCentFlag)
            assert _filterPoints(corners, subVoxCent, subvox_size).all()

            # Check for subvoxel edge intersections with the local patch of
            # triangles and for folds
            edgeIntXs, fold = _findVoxelSurfaceIntersections(smallPatch, 
                corners)

            # Separate points within the voxel into distinct clouds, to check
            # for multiple surface intersection
            groups = _separatePointClouds(smallPatch.tris)

            # If neither surface is folded within the subvox and there 
            # are no multiple intersections, we can form hulls. 
            if (not fold) & (len(groups) < 2):

                # Filter down surface nodes that are in the subvox
                localPs = patch.points[np.unique(smallPatch.tris),:]
                localPs = localPs[_filterPoints(localPs, subVoxCent, 
                    subvox_size)]
                
                # Gather points together in preparation for forming 
                # hulls (the corners will be added just beforehand)
                hullPts = np.vstack( (localPs, edgeIntXs, \
                    _findTriangleVoxFaceIntersections(smallPatch, \
                    subVoxCent, subvox_size)) )

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

                    # Vertex defects is 2pi - (sum of triangle angles around each vertex)
                    # This will be positive in a convex region, negative for concave. 
                    # Concave: we assume the hull exterior to the surface is smaller so form that 
                    msh = trimesh.base.Trimesh(vertices=smallPatch.points, faces=smallPatch.tris)
                    if trimesh.curvature.vertex_defects(msh).sum() < 0: 
                        hullPts = np.vstack((hullPts, corners[~cornerFlags,:]))
                        classes = [0, 1]
                    
                    # Smaller interior hull
                    else:
                        hullPts = np.vstack((hullPts, corners[cornerFlags,:]))
                        classes = [1, 0]

                V = _safeFormHull(hullPts)
                if not V: 
                    L2fraction = _classifyVoxelViaRecursion(smallPatch, \
                        subVoxCent, subvox_size, subVoxFlag)
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
                        subVoxCent, subvox_size, subVoxFlag)
                    inFraction += (L2fraction * subVoxVol)
                    subVoxClassified = True 
            
        # Sanity check: we should have classified the voxel by now
        if not subVoxClassified: 
            raise RuntimeError("Subvoxel has not been classified")

        if verbose: print(inFraction)

    # END of subvoxel loop ----------------------------------------------------
      
    if inFraction > 1.0001:
        raise RuntimeError(f'Fraction is {inFraction} in vox {voxIdx} at {voxIJK}')

    return inFraction


def _estimateFractions(surf, supersampler, descriptor, cores):
    """Estimate fraction of voxels lying interior to surface. 
    Args: 
        surf: complete surface object. 
        size: dimensions of voxel grid required to contain surfaces, 
            to which the voxels in voxList are indexed. 
        supersampler: 1 x 3 vector of supersampling factor
    Returns: 
        vector of size prod(FoV)
    """

    supersampler = np.squeeze(np.array(supersampler, dtype=np.int16))

    # Compute all voxel centres, prepare a partial function application for 
    # use with the parallel pool map function 
    workerChunks = utils._distributeObjects(
                                range(surf.indexed.assocs_keys.size), 50)
    estimatePartial = functools.partial(_estimateFractionsWorker, 
        surf, supersampler)

    # Select the appropriate iterator function according to whether progress 
    # bar is requested. Tqdm provides progress bar.  
    if bool(descriptor):
        iterator = functools.partial(tqdm.tqdm,
            total=len(workerChunks), desc=descriptor, 
            bar_format=BAR_FORMAT, ascii=True)
    else: 
        iterator = iter

    # And map across worker chunks either in parallel or serial. 
    workerFractions = []
    if cores > 1:
        with mp.Pool(cores) as p: 
            for r in iterator(p.imap(estimatePartial, workerChunks)): 
                workerFractions.append(r)
    else: 
        for r in iterator(map(estimatePartial, workerChunks)):
            workerFractions.append(r)

    # Aggregate the results back together and check for exceptions
    # Then clip to range [0, 1] (reqd for geometric approximations)
    for r in workerFractions:
        if isinstance(r, Exception):
            print("Exception was raised during worker estimation:")
            raise r

    # Clip results to 1 (reqd due to geometric approximations)
    fractions = np.concatenate(workerFractions)
    return np.minimum(fractions, 1.0)


def _estimateFractionsWorker(surf, supersampler, chunk):
    """Wrapper for _estimateFractions() for use in multiprocessing pool"""

    # estimateVoxelFraction can throw, in which case we want to return the 
    # exception to the caller instead of raising it here (within a parallel
    # pool the exception will not be raised)
    try:
        pvs = np.zeros(len(chunk), dtype=NP_FLOAT)
        vox_inds = surf.indexed.assocs_keys[chunk]
        vox_ijks = np.array(np.unravel_index(vox_inds, surf.indexed.space.size),
            dtype=NP_FLOAT).T

        for idx in range(len(chunk)):
            pvs[idx] = _estimateVoxelFraction(surf, vox_ijks[idx,:], 
                vox_inds[idx], supersampler)  
        
        return pvs

    except Exception as e:
        return e


def _voxelise_worker(surf, dim_range, raysd1d2):
    """
    Worker function for Surface.voxelise()

    Args: 
        surf: the calling surface that is being voxelised  
        dim_range: the coordinates to operate over (the subset of the
            overall dimension that is being shared amongst workers)
        raysd1d2: Nx2 array, for the N rays to project through the 
            voxel grid, recording the D1 D2 coordinates of their origin
    """

    size = surf.indexed.space.size
    dim = np.argmax(size)
    other_dims = list({0,1,2} - {dim})
    mask_size = copy.deepcopy(size)
    mask_size[other_dims[0]] = len(dim_range)
    mask = np.zeros(mask_size.prod(), dtype=bool)

    if not raysd1d2.size: 
        return mask.reshape(mask_size) 

    else: 

        shift = np.zeros(3, np.int32)
        shift[other_dims[0]] = dim_range.start
        rayIJK = np.zeros((size[dim], 3), dtype=np.int32)
        rayIJK[:,dim] = np.arange(0, size[dim])
        start_point = np.zeros(3, dtype=NP_FLOAT)

        for d1d2 in raysd1d2: 

            start_point[other_dims] = d1d2    
            rayIJK[:,other_dims] = d1d2[None,:]
            ray_voxs_orig = np.ravel_multi_index(rayIJK.T, size)

            # Load patches along this ray, we can assert that at least 
            # one patch must be returned. Find intersections 
            patches = surf.to_patches(ray_voxs_orig)
            assert patches is not None, 'No patches returned for voxel in LUT'
            intersectionMus = _findRayTriangleIntersections2D(
                start_point, patches, dim)

            if not intersectionMus.size:
                continue
            
            # If intersections were found, perform a parity test. 
            # Any ray should make an even number of intersections
            # as it crosses from -ve to +ve infinity
            if (intersectionMus.shape[0] % 2):
                raise RuntimeError("voxelise: odd number of intersections" + 
                " found. Does the FoV cover the full extents of the surface?")

            # Calculate points of intersection along the ray. 
            sorted = np.argsort(intersectionMus)
            intDs = start_point[dim] + (intersectionMus[sorted])
            shiftIJK = rayIJK - shift[None,:]
            ray_voxs_sub = np.ravel_multi_index(shiftIJK.T, mask_size)

            # Assignment. All voxels before the first point of intersection
            # are outside. The mask is already zeroed for these. All voxels
            # between point 1 and n could be in or out depending on parity
            for i in range(1, len(sorted)+1):

                # Starting from infinity, all points between an odd numbered
                # intersection and the next even one are inside the mask 
                # Points beyond the last intersection are outside the mask
                if ((i % 2) & ((i+1) <= len(sorted))):
                    indices = ((rayIJK[:,dim] > intDs[i-1]) 
                        & (rayIJK[:,dim] < intDs[i]))
                    mask[ray_voxs_sub[indices]] = 1

        return mask.reshape(mask_size)


def vox_tri_weights(in_surf, out_surf, spc, factor, 
                    cores=mp.cpu_count(), descriptor='', ones=False):     
    """
    Form matrix of size (n_vox x n_tris), in which element (I,J) is the 
    fraction of samples from voxel I that are in triangle prism J. 

    Args: 
        in_surf: Surface object, inner surface of cortical ribbon
        out_surf: Surface object, outer surface of cortical ribbon
        spc: ImageSpace object within which to project 
        factor: voxel subdivision factor
        cores: number of cpu cores
        descriptor: string for tqdm progress bar 
        
    Returns: 
        vox_tri_weights: a scipy.sparse CSR matrix of shape
            (n_voxs, n_tris), in which each entry at index [I,J] gives the 
            number of samples from triangle prism J that are in voxel I. 
            NB this matrix is not normalised in any way!
    """

    points_vox = []
    for s in [in_surf, out_surf]:
        p_v = utils.affine_transform(s.points, spc.world2vox)   
        points_vox.append(p_v)

    n_tris = in_surf.tris.shape[0]
    t_ranges = utils._distributeObjects(range(n_tris), 50)
    worker = functools.partial(_vox_tri_weights_worker, 
        inps_vox=points_vox[0], outps_vox=points_vox[1], 
        tris=in_surf.tris, spc=spc, factor=factor, ones=ones)
    
    iterator = functools.partial(tqdm.tqdm,
        total=len(t_ranges), desc=descriptor, 
        bar_format=BAR_FORMAT, ascii=True)

    if cores > 1: 
        with mp.Pool(cores) as p: 
            vpmats = [ r for r in iterator(p.imap_unordered(worker, t_ranges)) ] 
            
    else: 
        vpmats = [ r for r in iterator(map(worker, t_ranges)) ]
         
    vpmat = vpmats[0]
    for vp in vpmats[1:]:
        vpmat += vp

    return vpmat / factor.prod()


def _vox_tri_weights_worker(t_range, inps_vox, outps_vox, tris, 
                            spc, factor, ones=False):
    """
    Helper method for vox_tri_weights(). 

    Args: 
        t_range: iterable of triangle numbers to process
        in_surf: inner surface of cortex, voxel coordinates
        out_surf: outer surface of cortex, voxel coordinates 
        spc: ImageSpace in which surfaces lie 
        factor: voxel subdivision factor

    Returns: 
        sparse CSR matrix of size (n_vox x n_tris)
    """

    # Initialise a grid of sample points, sized by (factor) in each dimension. 
    # We then shift the samples into each individual voxel. 
    vox_tri_samps = sparse.dok_matrix((spc.size.prod(), 
        tris.shape[0]), dtype=NP_FLOAT)
    samplers = [ np.linspace(0, 1, 2*f + 1, dtype=NP_FLOAT)[1:-1:2] 
                    for f in factor ]
    samples = (np.stack(np.meshgrid(*samplers), axis=-1)
               .reshape(-1,3) - 0.5)

    for t in t_range: 

        # Stack the vertices of the inner and outer triangles into a 6x3 array.
        # We will then refer to these points by the indices abc, ABC; lower 
        # case for the white surface, upper for the pial. We also cycle the 
        # vertices (note, NOT A SHUFFLE) such that the highest index is first 
        # (corresponding to A,a). The relative ordering of vertices remains the
        # same, so we use flagsum to check if B < C or C < B. 
        tri = tris[t,:]
        tri_max = np.argmax(tri)
        tri_sort = [ tri[(tri_max + i) % 3] for i in range(3) ]
        flagsum = sum([ int(tri_sort[v] < tri_sort[(v + 1) % 3]) 
                        for v in range(3) ])

        # Two positive divisions and one negative
        if flagsum == 2: 
            tets = TETRA1

        # This MUST be two negatives and one positive. 
        else:
            tets = TETRA2

        hull_ps = np.vstack((inps_vox[tri_sort,:], 
                             outps_vox[tri_sort,:]))

        # Get the neighbourhood of voxels through which this prism passes
        # in linear indices (note the +1 on the upper bound)
        bbox = (np.vstack((np.maximum(0, hull_ps.min(0)),
                           np.minimum(spc.size, hull_ps.max(0)+1)))
                           .round().astype(np.int32))
        hood = np.array(list(itertools.product(
                range(*bbox[:,0]), range(*bbox[:,1]), range(*bbox[:,2])
                )), dtype=np.int32)

        # The bbox may not intersect any voxels within the FoV at all, skip
        if not hood.size:
            continue 
        hood_vidx = np.ravel_multi_index(hood.T, spc.size)

        # Debug mode: just stick ones in all candidate voxels and continue 
        if ones: 
            vox_tri_samps[hood_vidx,t] = factor.prod()
            continue

        for vidx, ijk in zip(hood_vidx, hood.astype(NP_FLOAT)):
            v_samps = ijk + samples

            # The two triangles form an almost triangular prism in space (like a
            # toblerone bar...). It has 6 vertices and 8 triangular faces (2 end
            # caps, 3 almost rectangular side faces that are further split into 2
            # triangles each). Splitting the quadrilateral faces into triangles is 
            # the tricky bit as it can be done in two ways, as below. 
            # 
            #   pial 
            # N______N+1
            #  |\  /|
            #  | \/ |
            #  | /\ |
            # n|/__\|n+1
            #   white
            #   
            # It is important to ensure that neighbouring prisms share the same 
            # subdivision of their adjacent faces (ie, both of them agree to split
            # it in the \ or / direction) to avoid double counting regions of space.
            # This is achieved by enumerating the triangular faces of the prism in 
            # a specific order according to the index numbers of the triangle 
            # vertices. For each vertex n, if the index number of vertex n+1 (with
            # wraparound for the last vertex) is greater, then we split the face
            # that the edge (n, n+1) belongs to in a "positive" manner. Otherwise, 
            # we split the face in a "negative" manner. A positive split means that 
            # a diagonal will go from the pial vertex N to white vertex n+1. A
            # negative split will go from pial vertex N+1 to white vertex n. As a
            # result, around the complete prism formed by the two triangles, there
            # will be two face diagonals that ALWAYS meet at the WHITE vertex
            # with the HIGHEST index number (referred to as 'a'). With these two 
            # diagonals fixed, the order of the last diagonal depends on the 
            # condition B < C (+ve) or C < B (-ve). We check this using the 
            # flagsum variable, which will be 2 for B < C or 1 for C < B. Finally,
            # knowing how the last diagonal is arranged, there are exactly two 
            # ways of splitting the prism down, hardcoded at the top of this file. 
            # See http://www.alecjacobson.com/weblog/?p=1888. 


            # Test the sample points against the tetrahedra. We don't care about
            # double counting within the polyhedra (although in theory this 
            # shouldn't happen). Hull formation can fail due to geometric 
            # degeneracy so wrap it up in a try block 
            samps_in = np.zeros(v_samps.shape[0], dtype=bool)
            for tet in tets: 
                try: 
                    hull = Delaunay(hull_ps[tet,:])
                    samps_in |= (hull.find_simplex(v_samps) >= 0)  

                # Silent fail for geometric degeneracy, raise anything else 
                except QhullError:
                    continue  

                except Exception as e: 
                    raise e 

            # Don't write explicit zero
            if samps_in.any():
                vox_tri_samps[vidx,t] = samps_in.sum()

    return vox_tri_samps.tocsr()


def __meyer_worker(points, tris, edges, edge_lengths, worklist):
    """
    Woker function for _meyer_areas()

    Args: 
        points: Px3 array
        tris: Tx3 array of triangle indices into points 
        edges: Tx3x3 array of triangle edges 
        edge_lengths: Tx3 array of edge lengths 
        worklist: iterable object, point indices to process (indexing
            into the tris array)

    Returns: 
        PxT sparse CSR matrix, where element I,J is the area of triangle J
            belonging to vertx I 
    """

    # We pre-compute all triangle edges, in the following order:
    # e1-0, then e2-0, then e2-1. But we don't necessarily process
    # the edge lengths in this order, so we need to keep track of them
    vtx_tri_areas = sparse.dok_matrix((points.shape[0], tris.shape[0]),
        dtype=NP_FLOAT)

    # Iterate through each triangle containing each point 
    for pidx in worklist:
        tris_touched = (tris == pidx)

        for tidx in np.flatnonzero(tris_touched.any(1)):
            # We need to work out at which index within the triangle
            # this point sits: could be {0,1,2}, call it the cent_pidx
            # Edge pairs e1 and e2 are defined as including cent_pidx (order
            # irrelevant), then e3 is the remaining edge pair
            cent_pidx = np.flatnonzero(tris_touched[tidx,:]).tolist()
            e3 = TRI_FULL_SET.difference(cent_pidx)
            other_idx = list(e3)
            e1 = set(cent_pidx + [other_idx[0]])
            e2 = set(cent_pidx + [other_idx[1]])

            # Match the edge pairs to the order in which edges were calculated 
            # earlier 
            e1_idx, e2_idx, e3_idx = [ np.flatnonzero(
                [ e == ei for ei in TRI_EDGE_INDEXING ]
                ) for e in [e1, e2, e3] ] 

            # And finally load the edges in the correct order 
            L12 = edge_lengths[tidx,e3_idx]
            L01 = edge_lengths[tidx,e1_idx]
            L02 = edge_lengths[tidx,e2_idx]

            # Angles 
            alpha = (np.arccos((np.square(L01) + np.square(L02) - np.square(L12)) 
                        / (2*L01*L02)))
            beta  = (np.arccos((np.square(L01) + np.square(L12) - np.square(L02)) 
                        / (2*L01*L12)))
            gamma = (np.arccos((np.square(L02) + np.square(L12) - np.square(L01))
                        / (2*L02*L12)))
            angles = np.array([alpha, beta, gamma])

            # Area if not obtuse
            if not (angles > np.pi/2).any(): # Voronoi
                a = ((np.square(L01)/np.tan(gamma)) + (np.square(L02)/np.tan(beta))) / 8
            else: 
                # If obtuse, heuristic approach
                area_t = 0.5 * np.linalg.norm(np.cross(edges[tidx,0,:], edges[tidx,1,:]))
                if alpha > np.pi/2:
                    a = area_t / 2
                else:
                    a = area_t / 4

            vtx_tri_areas[pidx,tidx] = a 

    return vtx_tri_areas.tocsr()


def vtx_tri_weights(surf, cores=mp.cpu_count()):
    """
    Form a matrix of size (n_vertices x n_tris) where element (I,J) corresponds
    to the area of triangle J belonging to vertex I. 

    Areas are calculated according to the definition of A_mixed in "Discrete 
    Differential-Geometry Operators for Triangulated 2-Manifolds", M. Meyer, 
    M. Desbrun, P. Schroder, A.H. Barr.

    With thanks to Jack Toner for the original code from which this is adapted.

    Args: 
        surf: Surface object 
        cores: number of CPU cores to use, default max 

    Returns: 
        sparse CSR matrix, size (n_points, n_tris) where element I,J is the 
            area of triangle J belonging to vertx I 
    """

    points = surf.points 
    tris = surf.tris 
    edges = np.stack([points[tris[:,1],:] - points[tris[:,0],:],
                      points[tris[:,2],:] - points[tris[:,0],:],
                      points[tris[:,2],:] - points[tris[:,1],:]], axis=1)
    edge_lengths = np.linalg.norm(edges, axis=2)
    worker_func = functools.partial(__meyer_worker, points, tris, 
                                    edges, edge_lengths)

    if cores > 1: 
        worker_lists = utils._distributeObjects(range(surf.n_points), cores)
        with mp.Pool(cores) as p: 
            results = p.map(worker_func, worker_lists)

        # Flatten results back down 
        vtx_tri_weights = results[0]
        for r in results[1:]:
            vtx_tri_weights += r 

    else: 
        vtx_tri_weights = worker_func(range(surf.n_points))

    assert (vtx_tri_weights.data > 0).all(), 'Zero areas returned'
    return vtx_tri_weights 
