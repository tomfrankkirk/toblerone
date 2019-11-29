from scipy.spatial import Delaunay
import numpy as np 
import functools 
import collections 
import itertools
import multiprocessing as mp 
import scipy.sparse as spmat
import nibabel 
from operator import iconcat
import warnings
from pdb import set_trace

from . import utils 
from .classes import  ImageSpace, Surface


def vol2surf_weights(ins, outs, spc, factor, cores):
    """
    Returns CSR matrix of size (n_vertices x n_voxs)
    """

    if (((outs.points.min(0) < 0).any()) 
        or ((outs.points.max(0) > spc.size).any())):
        warnings.warn("Surface not wholly contained within ImageSpace. " +
            "Is the surface in voxel coordinates?")

    tri_weights = _form_voxtri_mat(ins, outs, spc, factor, cores).tocsc()
    vtx_weights = _form_vtxtri_mat(ins, cores)

    worker = functools.partial(__vol2surf_worker, 
            tri_weights=tri_weights, vtx_weights=vtx_weights)

    if cores > 1: 
        vtx_ranges = utils._distributeObjects(range(vtx_weights.shape[0]), cores)

        with mp.Pool(cores) as p: 
            ws = p.map(worker, vtx_ranges)

        weights = ws[0]
        for w in ws[1:]:
            weights += w 

    else:
        weights = worker(range(vtx_weights.shape[0]))

    return weights


def __vol2surf_worker(vertices, tri_weights, vtx_weights):
    """
    Returns CSR matrix of size (n_vertices x n_voxs)
    """

    weights = spmat.dok_matrix((vtx_weights.shape[0], tri_weights.shape[0]), 
                dtype=np.float32)  

    for vtx in vertices: 
        tw = vtx_weights[vtx,:]
        voxw = tri_weights[:,tw.indices]
        uniq_voxs, at_inds = np.unique(voxw.indices, return_inverse=True)
        vtxw = np.bincount(at_inds, weights=voxw.data)
        weights[vtx,uniq_voxs] = (vtxw / vtxw.sum())

    return weights.tocsr()


def surf2vol_weights(ins, outs, spc, factor, cores):
    """
    Returns CSR matrix of size (n_voxs x n_vertices)
    """

    if (((outs.points.min(0) < 0).any()) 
        or ((outs.points.max(0) > spc.size).any())):
        warnings.warn("Surface not wholly contained within ImageSpace. " +
            "Is the surface in voxel coordinates?")

    tri_weights = _form_voxtri_mat(ins, outs, spc, factor, cores)
    vtx_weights = _form_vtxtri_mat(ins, cores).tocsc()
    voxs_nonzero = np.flatnonzero(tri_weights.sum(1) > 0)

    if cores > 1: 
        vox_ranges = [ voxs_nonzero[c] for c in 
            utils._distributeObjects(range(voxs_nonzero.size), cores) ]
        worker = functools.partial(__surf2vol_worker, 
            tri_weights=tri_weights, vtx_weights=vtx_weights)

        with mp.Pool(cores) as p: 
            ws = p.map(worker, vox_ranges)

        weights = ws[0]
        for w in ws[1:]:
            weights += w 

    else:
        weights = worker(voxs_nonzero)

    return weights

def __surf2vol_worker(voxs, tri_weights, vtx_weights):
    """
    Returns CSR matrix of size (n_voxs x n_vertices)
    """

    # Weights matrix is sized (n_voxs x n_vertices)
    # On each row, the weights will be stored at the column indices of 
    # the relevant vertex numbers 
    weights = spmat.dok_matrix((tri_weights.shape[0], vtx_weights.shape[0]), 
                dtype=np.float32)  

    for vox in voxs:
        tw = tri_weights[vox,:]
        vtw = vtx_weights[:,tw.indices]
        uniq_vtx, at_inds = np.unique(vtw.indices, return_inverse=True)
        voxw = np.bincount(at_inds, weights=vtw.data)
        weights[vox,uniq_vtx] = (voxw / voxw.sum())   

    return weights.tocsr()       

def _form_voxtri_mat(ins, outs, spc, factor, cores):     
    """
    Core method used by vol2surf and surf2vol projection functions

    Args: 
        ins: Surface object, inner surface of cortical ribbon
        outs: Surface object, outer surface of cortical ribbon
        spc: ImageSpace object within which to project 
        factor: int voxel subdivision factor
        cores: number of cpu cores
        
    Returns: 
        vox_tri_samps: a scipy.sparse DoK matrix (dictionary of keys format), 
            of shape (n_voxs, n_tris), in which each entry at index [I,J] 
            gives the number of samples from triangle J that are in voxel I. 
    """

    worker = functools.partial(__vtmat_worker, ins=ins, outs=outs,
        spc=spc, factor=factor)
    
    if cores > 1: 
        t_ranges = utils._distributeObjects(range(ins.tris.shape[0]), cores)
        with mp.Pool(cores) as p: 
            vtsamps = p.map(worker, t_ranges)
            
        vox_tri_samps = vtsamps[0]
        for vt in vtsamps[1:]:
            vox_tri_samps += vt
            
    else: 
        vox_tri_samps = worker(range(ins.tris.shape[0]))
        
    return vox_tri_samps

def __vtmat_worker(t_range, ins, outs, spc, factor):
    """
    Helper method for _form_voxtri_mat(), to be used in mp.Pool()
    """

    vox_tri_samps = spmat.dok_matrix((spc.size.prod(), 
        ins.tris.shape[0]), dtype=np.int32)
    sampler = np.linspace(0,1, 2*factor + 1)[1:-1:2]
    sx, sy, sz = np.meshgrid(sampler, sampler, sampler)
    samples = np.vstack((sx.flatten(),sy.flatten(),sz.flatten())).T - 0.5

    for t in t_range: 
        hull = np.vstack((ins.points[ins.tris[t,:],:], 
                        outs.points[outs.tris[t,:],:]))
        hull_lims = np.round(np.vstack((hull.min(0), hull.max(0) + 1)))
        nhood = hull_lims.astype(np.int32)
        nhood = np.array(list(itertools.product(
                range(*nhood[:,0]), range(*nhood[:,1]), range(*nhood[:,2]))))
        fltr = np.all((nhood > -1) & (nhood < spc.size), 1)
        nhood = nhood[fltr,:]
        nhood_v = np.ravel_multi_index(nhood.T, spc.size)
                        
        try: 
            hull = Delaunay(hull)  
            
            for vidx,ijk in zip(nhood_v, nhood):
                v_samps = np.array(ijk) + samples
                samps_in = (hull.find_simplex(v_samps) >= 0).sum()

                if samps_in:
                    vox_tri_samps[vidx,t] = samps_in
                    
        except Exception as e: 
            raise e 
                        
    return vox_tri_samps.tocsr()


def triangle_areas(ps, ts):
    return 0.5 * np.linalg.norm(np.cross(
        ps[ts[:,1],:] - ps[ts[:,0],:], 
        ps[ts[:,2],:] - ps[ts[:,0],:], axis=-1), axis=-1, ord=2)


def vertex_areas(ps, ts):
    tri_areas = triangle_areas(ps,ts)  
    vtx_areas = np.bincount(ts.flat, np.repeat(tri_areas, 3)) / 3 
    return vtx_areas


def _form_vtxtri_mat(surf, cores=mp.cpu_count()):
    """
    dfhjs
    Returns: 
        a sparse CSR matrix, shape (n_vtx, n_tris). Element (I,J)
        corresponds to the area of triangle J that belongs to vtx I.
    """

    n_vtx = surf.points.shape[0]
    tri_areas = triangle_areas(surf.points, surf.tris) / 3
    vtx_tri_areas = np.tile(tri_areas, (3,1)).T
    worker = functools.partial(__vtxtri_mat_worker, 
                surf=surf, vtx_tri_areas=vtx_tri_areas)

    if cores > 1: 
        vtx_ranges = utils._distributeObjects(range(n_vtx), cores)


        with mp.Pool(cores) as p: 
            vtx_tri_mats = p.map(worker, vtx_ranges)

        vtx_tri_mat = vtx_tri_mats[0]
        for vtmat in vtx_tri_mats[1:]:
            vtx_tri_mat += vtmat 

    else:
        vtx_tri_mat = worker(range(n_vtx))

    return vtx_tri_mat 

def __vtxtri_mat_worker(vtx_range, surf, vtx_tri_areas):
    n_vtx = surf.points.shape[0]
    n_tris = surf.tris.shape[0]
    vtx_tri_mat = spmat.dok_matrix((n_vtx,n_tris), dtype=np.float32)

    for vtx in vtx_range:
        inds = (surf.tris == vtx)
        tris = np.flatnonzero((inds.any(1)))
        vtx_tri_mat[vtx,tris] = vtx_tri_areas[inds]

    return vtx_tri_mat.tocsr()


# def vertex2tri(surf):
    
#     def _worker(x):
#         return x
    
#     with mp.Pool(8) as p: 
#         r = p.map(_worker, range(8))

#     return r 


# def _vol2tri_weights(ins, outs, spc, factor, cores):
#     """
#     Estimate weights used for projecting from volume to surface. 
#     Note that some triangles may not have any voxel weights. This is
#     because they represent an insignificant volume of cortex.

#     Args:
#         ins: Surface object, inner surface of cortical ribbon
#         outs: Surface object, outer surface of cortical ribbon
#         spc: ImageSpace object within which to project 
#         factor: int voxel subdivision factor
#         cores: number of cpu cores (default max-1)

#     Returns:
#         weights: list of np.arrays, length equal to number of triangles, 
#             for which the top row of each array are voxel indices and
#             the bottom row of each are the corresponding weights 
#     """
    
#     vtsamps = _form_voxtri_mat(ins, outs, spc, factor, cores)
#     vtsamps = vtsamps.tocsc()
#     weights = []
#     for t in range(ins.tris.shape[0]):
#         vt = vtsamps[:,t]
#         if not vt.size: 
#             print(t)
#         weights.append((vt.nonzero()[0], vt.data / vt.data.sum()))

#     return weights


# def vol2surf(img, ins, outs, spc, factor, cores=mp.cpu_count()-1):

#     s2r = np.eye(4)
#     img = nibabel.load(img)
#     data = img.get_fdata()

#     if not isinstance(spc, ImageSpace): spc = ImageSpace(spc)
#     if not isinstance(ins, Surface): ins = Surface(ins)
#     if not isinstance(outs, Surface): outs = Surface(outs)

#     # Need to find bridges - for which surfaces must be indexed first
#     # Which in turn means they need to be in voxel coords
#     surfs = [ins, outs]
#     bridges = np.union1d(*(s.find_bridges(spc, s2r) for s in surfs))
#     [ s.deindex(s2r) for s in surfs ]
#     overall = spc.world2vox @ s2r
#     [ s.applyTransform(overall) for s in surfs ]

#     if len(data.shape) != 1: 
#         data = data.reshape(-1)

#     out = np.zeros(ins.tris.shape[0], dtype=np.float32)
#     tri_weights = vol2surf_weights(ins, outs, spc, factor, cores)
#     vtx_weights = tri2vertex_weights(ins)



#     for idx,((ti,tw),(vi,vw)) in enumerate(zip(tri_weights,vtx_weights)): 
        
#         pass 

#     return out 

# def surf2vol(sdata, ins, outs, spc, factor, cores=mp.cpu_count()-1):

#     sdata = nibabel.load(sdata)

#     assert sdata.shape.size == ins.tris.shape[0]
#     out = np.zeros(spc.size.prod(), dtype=np.float32)
#     voxs, weights = surf2vol_weights(ins, outs, spc, factor, cores)
    
#     for v,(inds,we) in zip(voxs, weights):
#         out[v] = (sdata[inds] * we).sum()

#     return out 

# def surf2vol_weights(ins, outs, spc, factor, cores):
#     """
#     Estimate weights used for projecting from surface to volume

#     Args:
#         ins: Surface object, inner surface of cortical ribbon
#         outs: Surface object, outer surface of cortical ribbon
#         spc: ImageSpace object within which to project 
#         factor: int voxel subdivision factor
#         cores: number of cpu cores (default max-1)

#     Returns:
#         voxs: list of voxels 
#         weights: list of np.arrays, length equal to voxs, for which 
#             the top row of each array are triangle indices and the 
#             bottom row of each are corresponding triangle weights 
#     """
    
#     vtsamps = _form_voxtri_mat(ins, outs, spc, factor, cores)
#     voxs = np.unique(vtsamps.nonzero()[1])
#     vtsamps = vtsamps.tocsr()
#     weights = []
#     for v in voxs:
#         vt = vtsamps[v,:]
#         weights.append((vt.nonzero()[1], vt.data / vt.data.sum()))
    
#     return (voxs, weights)


