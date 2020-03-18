"""
Toblerone surface-volume projection functions
"""

import functools 
import itertools
import multiprocessing as mp 
import copy 
import warnings
from pdb import set_trace

from scipy import sparse 
import numpy as np 
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError 

from . import utils, estimators
from .classes import ImageSpace, Surface, Hemisphere


class Projector(object):

    def __init__(self, hemispheres, spc, factor=10, cores=mp.cpu_count()):

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

            if hasattr(hemi, 'pvs'): 
                self.pvs.append(hemi.pvs.reshape(-1,3))
            else: 
                supersample = np.ceil(spc.vox_size).astype(np.int8)
                pvs, _ = estimators._cortex(hemi, spc, np.eye(4), supersample, 
                    cores, False)
                self.pvs.append(pvs.reshape(-1,3))

            # Transform into voxel coordinates, check for partial coverage
            hemi.apply_transform(spc.world2vox)
            if ((hemi.outSurf.points.min(0) < -1).any() or
                (hemi.outSurf.points.max(0) > spc.size).any()): 
                warnings.warn("Surfaces not fully containined within reference" +
                    " space. Ensure they are in world-mm coordinates.")

            midsurf = calc_midsurf(hemi.inSurf, hemi.outSurf)
            vox_tri = _vox_tri_weights(hemi.inSurf, hemi.outSurf, spc, factor, cores)
            vtx_tri = _vtx_tri_weights(midsurf, cores)
            self.__vox_tri_mats.append(vox_tri)
            self.__vtx_tri_mats.append(vtx_tri)

    @property
    def vox_tri_mats(self): 
        return copy.deepcopy(self.__vox_tri_mats)

    @property
    def vtx_tri_mats(self): 
        return copy.deepcopy(self.__vtx_tri_mats)

    def flat_pvs(self):
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
        v2s_mat = self.vol2surf_matrix(edge_correction)
        v2v_mat = sparse.eye(self.spc.size.prod())
        v2n_mat = sparse.vstack((v2s_mat, v2v_mat), format="csr")
        return v2n_mat

    def surf2vol_matrix(self):
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
        pvs = self.flat_pvs()
        s2v_mat = self.surf2vol_matrix()
        v2v_mat = sparse.dia_matrix((pvs[:,1], 0), 
            shape=2*[self.spc.size.prod()])
        n2v_mat = sparse.hstack((s2v_mat, v2v_mat), format="csr")
        return n2v_mat

    def vol2surf(self, vdata, edge_correction=False):
        if vdata.shape[0] != self.spc.size.prod(): 
            raise RuntimeError("vdata must have the same number of rows as" +
                " voxels in the reference ImageSpace")
        v2s_mat = self.vol2surf_matrix(edge_correction)
        return v2s_mat.dot(vdata)

    def surf2vol(self, sdata): 
        s2v_mat = self.surf2vol_matrix()
        if sdata.shape[0] != s2v_mat.shape[1]: 
            raise RuntimeError("sdata must have the same number of rows as" +
                " total surface nodes (were one or two hemispheres used?)")
        return s2v_mat.dot(sdata)

    def vol2node(self, vdata, edge_correction=True):
        v2n_mat = self.vol2node_matrix(edge_correction)
        if vdata.shape[0] != v2n_mat.shape[1]: 
            raise RuntimeError("vdata must have the same number of rows as" +
                " nodes (voxels+vertices) in the reference ImageSpace")
        return v2n_mat.dot(vdata)

    def node2vol(self, ndata):
        n2v_mat = self.node2vol_matrix()
        if ndata.shape[0] != n2v_mat.shape[1]: 
            raise RuntimeError("ndata must have the same number of rows as" +
                " total nodes in ImageSpace (voxels+vertices)")
        return n2v_mat.dot(ndata)
        
def calc_midsurf(in_surf, out_surf):
    vec = out_surf.points - in_surf.points 
    points =  in_surf.points + (0.5 * vec)
    return Surface.manual(points, in_surf.tris)


def assemble_vol2surf(vox_tri, vtx_tri):
    # Ensure each triangle's voxel weights sum to 1 
    # Ensure each vertices' triangle weights sum to 1 
    vox2tri = sparse_normalise(vox_tri, 0).T
    tri2vtx = sparse_normalise(vtx_tri, 1)
    vol2vtx = tri2vtx @ vox2tri
    return sparse_normalise(vol2vtx, 1)



def vol2surf_weights(in_surf, out_surf, spc, factor=10, cores=mp.cpu_count()):
    """
    Weighting matrix used to project data from volumetric space to surface. 
    To use this matrix: weights.dot(data_to_project)

    Args: 
        in_surf: inner Surface of ribbon, in world mm coordinates 
        out_surf: outer Surface of ribbon, in world mm coordinates
        spc: ImageSpace in which data exists 
        factor: voxel subdivision factor (default 10)
        cores: number of processor cores (default max)

    Returns:   
        a scipy sparse CSR matrix of size (n_verts x n_voxs)
    """

    # Create copies of the provided surfaces and convert them into vox coords 
    in_surf = copy.deepcopy(in_surf)
    out_surf = copy.deepcopy(out_surf)
    mid_surf = calc_midsurf(in_surf, out_surf)

    # Mapping from voxels to triangles
    # Ensure each triangle's voxel weights sum to 1 
    vox_tri_weights = _vox_tri_weights(in_surf, out_surf, spc, factor, cores)

    # Mapping from triangles to vertices 
    # Ensure each vertices' triangle weights sum to 1 
    vtx_tri_weights = _vtx_tri_weights(mid_surf, cores)
    return assemble_vol2surf(vox_tri_weights, vtx_tri_weights)

       
def __vol2surf_worker(vertices, vox_tri_weights, vtx_tri_weights):
    """
    Worker function for vol2surf_weights(). 

    Args: 
        vertices: iterable of vertex numbers to process 
        vox_tri_weights: produced by _vox_tri_weights() 
        vtx_tri_weights: produced by _vtx_tri_weights()

    Returns: 
        CSR matrix of size (n_vertices x n_voxs)
    """

    weights = sparse.dok_matrix((vtx_tri_weights.shape[0], vox_tri_weights.shape[0]))  

    for vtx in vertices: 
        tri_weights = vtx_tri_weights[vtx,:]
        vox_weights = vox_tri_weights[:,tri_weights.indices]
        vox_weights.data *= tri_weights.data[vox_weights.tocsr().indices]
        u_voxs, at_inds = np.unique(vox_weights.indices, return_inverse=True)
        vtx_vox_weights = np.bincount(at_inds, weights=vox_weights.data)
        weights[vtx,u_voxs] = vtx_vox_weights

    return weights.tocsr()



def assemble_surf2vol(vox_tri, vtx_tri):
    # Ensure each triangle's vertex weights sum to 1 
    # Ensure each voxel's triangle weights sum to 1
    vtx2tri = sparse_normalise(vtx_tri, 0).T
    tri2vox = sparse_normalise(vox_tri, 1)
    vtx2vox = tri2vox @ vtx2tri
    return sparse_normalise(vtx2vox, 1)


def surf2vol_weights(in_surf, out_surf, spc, factor=10, cores=mp.cpu_count()):
    """
    Weights matrix used to project data from surface to volumetric space. NB 
    any registration transforms must be applied to the surfaces beforehand, 
    such that they are in world-mm coordinates for this function. 

    Args: 
        in_surf: inner Surface of ribbon, in world mm coordinates 
        out_surf: outer Surface of ribbon, in world mm coordinates
        spc: ImageSpace in which to project data
        factor: voxel subdivision factor (default 10)
        cores: number of processor cores (default max)

    Returns:   
        scipy sparse CSR matrix of size (n_voxs x n_verts)
    """

    # Create copies of the provided surfaces and convert them into vox coords 
    in_surf = copy.deepcopy(in_surf)
    out_surf = copy.deepcopy(out_surf)
    mid_surf = calc_midsurf(in_surf, out_surf)

    # Mapping from vertices to triangles - ensure each triangle's vertex 
    # weights sum to 1 
    vtx_tri_weights = _vtx_tri_weights(mid_surf, cores)

    # Mapping from triangles to voxels - ensure each voxel's triangle
    # weights sum to 1
    vox_tri_weights = _vox_tri_weights(in_surf, out_surf, spc, factor, cores)
    return assemble_surf2vol(_vox_tri_weights, vtx_tri_weights)

def __surf2vol_worker(voxs, vox_tri_weights, vtx_tri_weights):
    """
    Returns CSR matrix of size (n_vertices x n_verts)
    """

    # Weights matrix is sized (n_voxs x n_vertices)
    # On each row, the weights will be stored at the column indices of 
    # the relevant vertex numbers 
    weights = sparse.dok_matrix(
        (vox_tri_weights.shape[0], vtx_tri_weights.shape[0]))

    for vox in voxs:
        tri_weights = vox_tri_weights[vox,:]
        vtx_weights = vtx_tri_weights[:,tri_weights.indices]
        vtx_weights.data *= np.take(tri_weights.data, 
                                    vtx_weights.tocsr().indices)
        u_vtx, at_inds = np.unique(vtx_weights.indices, return_inverse=True)
        vox_vtx_weights = np.bincount(at_inds, weights=vtx_weights.data)
        weights[vox,u_vtx] = vox_vtx_weights    

    return weights.tocsr()


def sparse_normalise(mat, axis, threshold=1e-6): 
    """
    Normalise a sparse matrix so that all rows (axis=1) or columns (axis=0)
    sum to either 1 or zero. NB any rows or columns that sum to less than 
    threshold will be rounded to zeros.

    Args: 
        mat: sparse matrix to normalise 
        axis: dimension along which sums should equal 1 (0 for col, 1 for row)

    Returns: 
        sparse matrix. either CSR (axis 0) or CSC (axis 1)
    """

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
    assert np.all(np.abs((sums[sums > 0] - 1)) < 1e-6), 'did not normalise to 1'
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

    vox_tri_samps = sparse.dok_matrix((spc.size.prod(), 
        in_surf.tris.shape[0]))
    sampler = np.linspace(0,1, 2*factor + 1)[1:-1:2]
    sx, sy, sz = np.meshgrid(sampler, sampler, sampler)
    samples = np.vstack((sx.flatten(),sy.flatten(),sz.flatten())).T - 0.5

    for t in t_range: 
        hull = np.vstack((in_surf.points[in_surf.tris[t,:],:], 
                        out_surf.points[out_surf.tris[t,:],:]))
        hull_lims = np.round(np.vstack((hull.min(0), hull.max(0) + 1)))
        nhood = hull_lims.astype(np.int16)
        nhood = np.array(list(itertools.product(
                range(*nhood[:,0]), range(*nhood[:,1]), range(*nhood[:,2]))))
        fltr = np.all((nhood > -1) & (nhood < spc.size), 1)
        nhood = nhood[fltr,:]
        nhood_v = np.ravel_multi_index(nhood.T, spc.size)
                        
        try: 
            hull = Delaunay(hull)  
            
            for vidx,ijk in zip(nhood_v, nhood):
                v_samps = ijk + samples
                samps_in = (hull.find_simplex(v_samps) >= 0).sum()

                if samps_in:
                    vox_tri_samps[vidx,t] = samps_in
                    
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
    # e1-0, then e2-0, then e2-1
    EDGE_INDEXING = [{1,0}, {2,0}, {2,1}]
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
            other_pidx = np.flatnonzero(~tris_touched[tidx,:]).tolist()
            e1 = set(cent_pidx + [other_pidx[0]])
            e2 = set(cent_pidx + [other_pidx[1]])
            e3 = set(other_pidx)

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
                # If obtuse, heuristic 
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

    assert (vtx_tri_weights.data > 0).all(), 'zero areas returned'
    return vtx_tri_weights 
