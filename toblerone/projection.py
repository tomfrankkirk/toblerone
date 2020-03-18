"""
Toblerone surface-volume projection functions
"""

import functools 
import itertools
import multiprocessing as mp 
import copy 
import warnings

import numpy as np 
from scipy import sparse 
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError 

from . import utils, pvestimation
from .classes import ImageSpace, Hemisphere, Surface


class Projector(object):
    """
    Use to perform projection between volume, surface and node space. 
    Creating a projector object may take some time whilst the consituent 
    matrices are prepared; once created any of the individual projections
    may be calculated directly from the object. 

    Args: 
        hemispheres: single, or iterable, of Hemisphere objects (order: L,R)
        spc: ImageSpace to project from/to 
        factor: voxel subdivision factor (default 10)
        cores: number of processor cores to use (default max)
    """

    def __init__(self, hemispheres, spc, factor=10, cores=mp.cpu_count()):

        print("Initialising projector (will take some time)")
        if not isinstance(hemispheres, Hemisphere):
            if not len(hemispheres) == 2: 
                raise RuntimeError("Either provide a single or iterable of 2 Hemisphere objects")
        else: 
            hemispheres = [hemispheres]
            
        self.hemis = hemispheres
        self.spc = spc 
        self.pvs = [] 
        self.__vox_tri_mats = [] 
        self.__vtx_tri_mats = []

        for hemi in hemispheres: 

            # If PV estimates are not present, then compute from scratch 
            if hasattr(hemi, 'pvs'): 
                self.pvs.append(hemi.pvs.reshape(-1,3))
            else: 
                supersample = np.ceil(spc.vox_size).astype(np.int8) 
                pvs, _ = pvestimation.cortex(hemi, spc, np.eye(4), supersample, 
                    cores, False)
                self.pvs.append(pvs.reshape(-1,3))

            # Transform surfaces voxel coordinates, check for partial coverage
            hemi.apply_transform(spc.world2vox)
            if ((hemi.outSurf.points.min(0) < -1).any() or
                (hemi.outSurf.points.max(0) > spc.size).any()): 
                warnings.warn("Surfaces not fully containined within reference" +
                    " space. Ensure they are in world-mm coordinates.")

            # Calculate the constituent matrices for projection with each hemi 
            midsurf = calc_midsurf(hemi.inSurf, hemi.outSurf)
            vox_tri = _vox_tri_weights(hemi.inSurf, hemi.outSurf, 
                spc, factor, cores)
            vtx_tri = _vtx_tri_weights(midsurf, cores)
            self.__vox_tri_mats.append(vox_tri)
            self.__vtx_tri_mats.append(vtx_tri)


    # Calculation of the projection matrices involves rescaling the constituent
    # matrices, so these proerties return copies to keep the originals private
    @property
    def vox_tri_mats(self): 
        return copy.deepcopy(self.__vox_tri_mats)


    @property
    def vtx_tri_mats(self): 
        return copy.deepcopy(self.__vtx_tri_mats)


    def flat_pvs(self):
        """
        Combine PV estimates from one or both hemispheres (if available) into 
        single map. 

        Returns: 
            (v x 3) array of PVs, columns arranged GM, WM, non-brain
        """
        if len(self.pvs) > 1:
            # Combine PV estimates from each hemisphere into single map 
            pvs = np.zeros((self.spc.size.prod(), 3))
            pvs[:,0] = np.minimum(1.0, self.pvs[0][:,0] + self.pvs[1][:,0])
            pvs[:,1] = np.minimum(1.0 - pvs[:,0], self.pvs[0][:,1] + self.pvs[1][:,1])
            pvs[:,2] = 1.0 - np.sum(pvs[:,0:2], axis=1)
            return pvs 
        else: 
            return self.pvs[0]


    def vol2surf_matrix(self, edge_correction=False):
        """
        Volume to surface projection matrix. 

        Args: 
            edge_correction: upweight signal from voxels less than 100% brain

        Returns: 
            sparse matrix sized (surface vertices x voxels). Surface vertices 
                are arranged L then R. 
        """

        proj_mats = [ assemble_vol2surf(vox_tri, vtx_tri) 
            for vox_tri, vtx_tri in zip(self.vox_tri_mats, self.vtx_tri_mats) ]
        v2s_mat = sparse.vstack(proj_mats, format="csr")

        if edge_correction: 
            brain_pv = self.flat_pvs()[:,:2].sum(1)
            brain = (brain_pv > 1e-3)
            upweight = np.zeros(brain_pv.shape)
            upweight[brain] = 1 / brain_pv[brain]
            v2s_mat.data *= np.take(upweight, v2s_mat.indices)

        return v2s_mat 


    def vol2node_matrix(self, edge_correction=True): 
        """
        Volume to node space projection matrix. 

        Args: 
            edge_correction: upweight signal from voxels less than 100% brain

        Returns: 
            sparse matrix sized ((surface vertices + voxels) x voxels)
        """

        v2s_mat = self.vol2surf_matrix(edge_correction)
        v2v_mat = sparse.eye(self.spc.size.prod())
        v2n_mat = sparse.vstack((v2s_mat, v2v_mat), format="csr")
        return v2n_mat

    def surf2vol_matrix(self):
        """
        Surface to volume projection matrix. 

        Returns: 
            sparse matrix sized (surface vertices x voxels)
        """

        gm_weights = []
        if len(self.hemis) == 1: 
            gm_weights.append(np.ones(self.spc.size.prod()))
        else: 
            # GM PV can be shared between both hemispheres, so rescale each row of
            # the s2v matrices by the proportion of all voxel-wise GM that belongs
            # to that hemisphere (eg, the GM could be shared 80:20 between the two)
            GM = self.pvs[0][:,0] + self.pvs[1][:,0]
            GM[GM == 0] = 1 
            gm_weights.append(self.pvs[0][:,0] / GM)
            gm_weights.append(self.pvs[1][:,0] / GM)

        proj_mats = []
        for vox_tri, vtx_tri, weights in zip(self.vox_tri_mats, self.vtx_tri_mats, gm_weights): 
            s2v_mat = assemble_surf2vol(vox_tri, vtx_tri).tocsc()
            s2v_mat.data *= np.take(weights, s2v_mat.indices)
            proj_mats.append(s2v_mat)

        pvs = self.flat_pvs()
        s2v_mat = sparse.hstack(proj_mats, format="csc")
        s2v_mat.data *= np.take(pvs[:,0], s2v_mat.indices)
        return s2v_mat  

    def node2vol_matrix(self): 
        """
        Node space to volume projection matrix. 

        Returns: 
            sparse matrix sized (voxels x (surface vertices + voxels))
        """

        pvs = self.flat_pvs()
        s2v_mat = self.surf2vol_matrix()
        v2v_mat = sparse.dia_matrix((pvs[:,1], 0), 
            shape=2*[self.spc.size.prod()])
        n2v_mat = sparse.hstack((s2v_mat, v2v_mat), format="csr")
        return n2v_mat

    def vol2surf(self, vdata, edge_correction=False):
        """
        Project data from volum to surface. 

        Args: 
            vdata: np.array, sized n_voxels in first dimension
            edge_correction: upweight voxels that are less than 100% brain
        
        Returns:
            np.array, sized n_vertices in first dimension 
        """

        if vdata.shape[0] != self.spc.size.prod(): 
            raise RuntimeError("vdata must have the same number of rows as" +
                " voxels in the reference ImageSpace")
        v2s_mat = self.vol2surf_matrix(edge_correction)
        return v2s_mat.dot(vdata)

    def surf2vol(self, sdata): 
        """
        Project data from surface to volume. 

        Args: 
            sdata: np.array sized n_vertices in first dimension (arranged L,R)

        Returns: 
            np.array, sized n_voxels in first dimension 
        """

        s2v_mat = self.surf2vol_matrix()
        if sdata.shape[0] != s2v_mat.shape[1]: 
            raise RuntimeError("sdata must have the same number of rows as" +
                " total surface nodes (were one or two hemispheres used?)")
        return s2v_mat.dot(sdata)

    def vol2node(self, vdata, edge_correction=True):
        """
        Project data from volume to node space. 

        Args: 
            vdata: np.array, sized n_voxels in first dimension 
        
        Returns: 
            np.array, sized (n_vertices + n_voxels) in first dimension.
                Surface vertices are arranged L then R. 
        """

        v2n_mat = self.vol2node_matrix(edge_correction)
        if vdata.shape[0] != v2n_mat.shape[1]: 
            raise RuntimeError("vdata must have the same number of rows as" +
                " nodes (voxels+vertices) in the reference ImageSpace")
        return v2n_mat.dot(vdata)

    def node2vol(self, ndata):
        """
        Project data from node space to volume.

        Args: 
            ndata: np.array, sized (n_vertices + n_voxels) in first dimension. 
                Surface data should be arranged L then R in the first dim. 

        Returns: 
            np.array, sized n_voxels in first dimension
        """

        n2v_mat = self.node2vol_matrix()
        if ndata.shape[0] != n2v_mat.shape[1]: 
            raise RuntimeError("ndata must have the same number of rows as" +
                " total nodes in ImageSpace (voxels+vertices)")
        return n2v_mat.dot(ndata)
        
def calc_midsurf(in_surf, out_surf):
    """
    Midsurface between two Surfaces
    """

    vec = out_surf.points - in_surf.points 
    points =  in_surf.points + (0.5 * vec)
    return Surface.manual(points, in_surf.tris)


def assemble_vol2surf(vox_tri, vtx_tri):
    """
    Combine (w/ normalisation) the vox_tri and vtx_tri matrices into vol2surf.
    """
    
    # Ensure each triangle's voxel weights sum to 1 
    # Ensure each vertices' triangle weights sum to 1 
    vox2tri = sparse_normalise(vox_tri, 0).T
    tri2vtx = sparse_normalise(vtx_tri, 1)
    vol2vtx = tri2vtx @ vox2tri
    return sparse_normalise(vol2vtx, 1)


def assemble_surf2vol(vox_tri, vtx_tri):
    """
    Combine (w/ normalisation) the vox_tri and vtx_tri matrices into surf2vol.
    """

    # Ensure each triangle's vertex weights sum to 1 
    # Ensure each voxel's triangle weights sum to 1
    vtx2tri = sparse_normalise(vtx_tri, 0).T
    tri2vox = sparse_normalise(vox_tri, 1)
    vtx2vox = tri2vox @ vtx2tri
    return sparse_normalise(vtx2vox, 1)


def sparse_normalise(mat, axis, threshold=1e-6): 
    """
    Normalise a sparse matrix so that all rows (axis=1) or columns (axis=0)
    sum to either 1 or zero. NB any rows or columns that sum to less than 
    threshold will be rounded to zeros.

    Args: 
        mat: sparse matrix to normalise 
        axis: dimension along which sums should equal 1 (0 for col, 1 for row)
        threshold: any row/col wuth sum < threshold will be set to zero  

    Returns: 
        sparse matrix. either CSR (axis 0) or CSC (axis 1)
    """

    # Make local copy - otherwise this function will modify the caller's copy 
    constructor = type(mat)
    mat = copy.deepcopy(mat)

    if axis == 0:
        matrix = mat.tocsr()
        norm = mat.sum(0).A.flatten()
    elif axis == 1: 
        matrix = mat.tocsc()
        norm = mat.sum(1).A.flatten()
    else: 
        raise RuntimeError("Axis must be 0 or 1")

    # Set threshold. Round any row/col below this to zeros 
    fltr = (norm > threshold)
    normalise = np.zeros(norm.size)
    normalise[fltr] = 1 / norm[fltr]
    matrix.data *= np.take(normalise, matrix.indices)

    # Sanity check
    sums = matrix.sum(axis).A.flatten()
    assert np.all(np.abs((sums[sums > 0] - 1)) < 1e-6), 'Did not normalise to 1'
    return constructor(matrix)


def _vox_tri_weights(in_surf, out_surf, spc, factor=10, cores=mp.cpu_count()):     
    """
    Form matrix of size (n_vox x n_tris), in which element (I,J) is the 
    fraction of samples from voxel I that are in triangle prism J. 

    Args: 
        in_surf: Surface object, inner surface of cortical ribbon
        out_surf: Surface object, outer surface of cortical ribbon
        spc: ImageSpace object within which to project 
        factor: voxel subdivision factor
        cores: number of cpu cores
        
    Returns: 
        vox_tri_weights: a scipy.sparse CSR matrix of shape
            (n_voxs, n_tris), in which each entry at index [I,J] gives the 
            number of samples from triangle prism J that are in voxel I. 
            NB this matrix is not normalised in any way!
    """

    n_tris = in_surf.tris.shape[0]
    worker = functools.partial(__vox_tri_weights_worker, in_surf=in_surf, 
        out_surf=out_surf, spc=spc, factor=factor)
    
    if cores > 1: 
        t_ranges = utils._distributeObjects(range(n_tris), cores)
        with mp.Pool(cores) as p: 
            vpmats = p.map(worker, t_ranges)
            
        vpmat = vpmats[0]
        for vp in vpmats[1:]:
            vpmat += vp
            
    else: 
        vpmat = worker(range(n_tris))
         
    return vpmat / (factor ** 3)


def __vox_tri_weights_worker(t_range, in_surf, out_surf, spc, factor):
    """
    Helper method for _vox_tri_weights(). 

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
        in_surf.tris.shape[0]))
    sampler = np.linspace(0,1, 2*factor + 1)[1:-1:2]
    sx, sy, sz = np.meshgrid(sampler, sampler, sampler)
    samples = np.vstack((sx.flatten(),sy.flatten(),sz.flatten())).T - 0.5

    for t in t_range: 

        # Load the triangle vertices around which to form the hull 
        hull = np.vstack((in_surf.points[in_surf.tris[t,:],:], 
                        out_surf.points[out_surf.tris[t,:],:]))

        # Hull formation can fail (due to triangles not being far enough 
        # apart). So try and do it first, if successful continue with 
        # the rest of the maths 
        try: 
            hull = Delaunay(hull)  

            # Get the neighbourhood of voxels that contains this hull 
            hood = hull_lims.astype(np.int16)
            hood = np.array(list(itertools.product(
                    range(*hood[:,0]), range(*hood[:,1]), range(*hood[:,2]))))
            fltr = np.all((hood > -1) & (hood < spc.size), 1)
            hood = hood[fltr,:]
            hood_vidx = np.ravel_multi_index(hood.T, spc.size)
            for vidx,ijk in zip(hood_vidx, hood):
                v_samps = ijk + samples
                samps_in = (hull.find_simplex(v_samps) >= 0).sum()

                # Don't write explicit zero
                if samps_in:
                    vox_tri_samps[vidx,t] = samps_in
                    
        # Silent fail for geometric degeneracy, raise anything else 
        except QhullError:
            continue  

        except Exception as e: 
            raise e 
                        
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
    EDGE_INDEXING = [{1,0}, {2,0}, {2,1}]
    FULL_SET = set(range(3))
    vtx_tri_areas = sparse.dok_matrix((points.shape[0], tris.shape[0]))

    # Iterate through each triangle containing each point 
    for pidx in worklist:
        tris_touched = (tris == pidx)

        for tidx in np.flatnonzero(tris_touched.any(1)):
            # We need to work out at which index within the triangle
            # this point sits: could be {0,1,2}, call it the cent_pidx
            # Edge pairs e1 and e2 are defined as including cent_pidx (order
            # irrelevant), then e3 is the remaining edge pair
            cent_pidx = np.flatnonzero(tris_touched[tidx,:]).tolist()
            e3 = FULL_SET.difference(cent_pidx)
            other_idx = list(e3)
            e1 = set(cent_pidx + [other_pidx[0]])
            e2 = set(cent_pidx + [other_pidx[1]])

            # Match the edge pairs to the order in which edges were calculated 
            # earlier 
            e1_idx, e2_idx, e3_idx = [ np.flatnonzero(
                [ e == ei for ei in EDGE_INDEXING ]
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
            if not np.any((angles > np.pi/2)): # Voronoi
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


def _vtx_tri_weights(surf, cores=mp.cpu_count()):
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
        worker_lists = utils._distributeObjects(range(points.shape[0]), cores)
        with mp.Pool(cores) as p: 
            results = p.map(worker_func, worker_lists)

        # Flatten results back down 
        vtx_tri_weights = results[0]
        for r in results[1:]:
            vtx_tri_weights += r 

    else: 
        vtx_tri_weights = worker_func(range(points.shape[0]))

    assert (vtx_tri_weights.data > 0).all(), 'Zero areas returned'
    return vtx_tri_weights 
