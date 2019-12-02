from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError 
import numpy as np 
import functools 
import itertools
import multiprocessing as mp 
import scipy.sparse as spmat
import nibabel 
import warnings
import copy 
from pdb import set_trace

from . import utils 
from .classes import  ImageSpace, Surface


def vol2surf(vdata, in_surf, out_surf, spc, factor, cores=mp.cpu_count()):
    """
    Project volumetric data to the cortical ribbon defined by inner/outer 
    surfaces. NB any registration transforms must be applied to the surfaces 
    beforehand, such that they are in world-mm coordinates for this function. 

    Args: 
        vdata: np.array of shape equal to spc.size, or flattened, to project
        in_surf: inner Surface of ribbon, in world mm coordinates 
        out_surf: outer Surface of ribbon, in world mm coordinates
        spc: ImageSpace in which data exists 
        factor: voxel subdivision factor 
        cores: number of processor cores (default max)

    Returns:   
        flat np.array of size equal to surf.n_vertices
    """

    if not vdata.size == spc.size.prod(): 
        raise RuntimeError("Size of vdata does not match size of ImageSpace")

    # Create copies of the provided surfaces and convert them into vox coords 
    in_surf = copy.deepcopy(in_surf)
    out_surf = copy.deepcopy(out_surf)
    [ s.applyTransform(spc.world2vox) for s in [in_surf, out_surf] ]

    v2s_weights = vol2surf_weights(in_surf, out_surf, spc, factor, cores)
    return v2s_weights.dot(vdata.reshape(-1))


def vol2surf_weights(in_surf, out_surf, spc, factor, cores=mp.cpu_count()):
    """
    Weighting matrix used to project data from volumetric space to surface. 
    To use this matrix: weights.dot(data_to_project)

    Args: 
        in_surf: inner Surface of ribbon, in world mm coordinates 
        out_surf: outer Surface of ribbon, in world mm coordinates
        spc: ImageSpace in which data exists 
        factor: voxel subdivision factor 
        cores: number of processor cores (default max)

    Returns:   
        a scipy sparse CSR matrix of size (n_verts x n_voxs)
    """

    # Create copies of the provided surfaces and convert them into vox coords 
    in_surf = copy.deepcopy(in_surf)
    out_surf = copy.deepcopy(out_surf)
    [ s.applyTransform(spc.world2vox) for s in [in_surf, out_surf] ]

    # Two step projection: from volume to prisms, then prisms to vertices
    voxprism_mat = _voxprism_mat(in_surf, out_surf, spc, factor, cores).tocsc()
    vtxtri_mat = _vtxtri_mat(in_surf, cores)

    n_vtx = in_surf.points.shape[0]
    worker = functools.partial(__vol2surf_worker, 
            voxprism_mat=voxprism_mat, vtxtri_mat=vtxtri_mat)

    if cores > 1: 
        vtx_ranges = utils._distributeObjects(range(n_vtx), cores)

        with mp.Pool(cores) as p: 
            ws = p.map(worker, vtx_ranges)

        weights = ws[0]
        for w in ws[1:]:
            weights += w 

    else:
        weights = worker(range(n_vtx))

    return weights


def __vol2surf_worker(vertices, voxprism_mat, vtxtri_mat):
    """
    Worker function for vol2surf_weights(). 

    Args: 
        vertices: iterable of vertex numbers to process 
        voxprism_mat: produced by _voxprism_mat() 
        vtxtri_mat: produced by _vtxtri_mat()

    Returns: 
        CSR matrix of size (n_vertices x n_voxs)
    """

    weights = spmat.dok_matrix((vtxtri_mat.shape[0], voxprism_mat.shape[0]), 
                dtype=np.float32)  

    for vtx in vertices: 
        tw = vtxtri_mat[vtx,:]
        voxw = voxprism_mat[:,tw.indices]
        uniq_voxs, at_inds = np.unique(voxw.indices, return_inverse=True)
        vtxw = np.bincount(at_inds, weights=voxw.data)
        weights[vtx,uniq_voxs] = (vtxw / vtxw.sum())

    return weights.tocsr()


def surf2vol(sdata, in_surf, out_surf, spc, factor, cores=mp.cpu_count()):
    """
    Project data defined on surface vertices to a volumetric space. NB any 
    registration transforms must be applied to the surfaces beforehand, 
    such that they are in world-mm coordinates for this function. 

    Args: 
        sdata: np.array of shape equal to surf.n_vertices, data to project
        in_surf: inner Surface of ribbon, in world mm coordinates 
        out_surf: outer Surface of ribbon, in world mm coordinates
        spc: ImageSpace in which to project data
        factor: voxel subdivision factor 
        cores: number of processor cores (default max)

    Returns:   
        flat np.array of size equal to spc.n_voxels
    """

    if not sdata.size == in_surf.points.shape[0]:
        raise RuntimeError("Size of sdata does not match surface")

    # Create copies of the provided surfaces and convert them into vox coords 
    in_surf = copy.deepcopy(in_surf)
    out_surf = copy.deepcopy(out_surf)
    [ s.applyTransform(spc.world2vox) for s in [in_surf, out_surf] ]

    s2v_weights = surf2vol_weights(in_surf, out_surf, spc, factor, cores).tocsc().T
    return s2v_weights.dot(sdata.reshape(-1))


def surf2vol_weights(in_surf, out_surf, spc, factor, cores=mp.cpu_count()):
    """
    Returns CSR matrix of size (n_voxs x n_vertices)
    """

    # Create copies of the provided surfaces and convert them into vox coords 
    in_surf = copy.deepcopy(in_surf)
    out_surf = copy.deepcopy(out_surf)
    [ s.applyTransform(spc.world2vox) for s in [in_surf, out_surf] ]

    tri_weights = _voxprism_mat(in_surf, out_surf, spc, factor, cores)
    vtx_weights = _vtxtri_mat(in_surf, cores).tocsc()
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
    Returns CSR matrix of size (n_vertices x n_vertices)
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


def _voxprism_mat(in_surf, out_surf, spc, factor, cores=mp.cpu_count()):     
    """
    Form matrix of size (n_vox x n_tris), in which element (I,J) is the 
    number of samples from triangle prism J that are inside voxel I. 
    WARNING: this is not normalised (the size of elements depends on factor)

    Args: 
        in_surf: Surface object, inner surface of cortical ribbon
        out_surf: Surface object, outer surface of cortical ribbon
        spc: ImageSpace object within which to project 
        factor: int voxel subdivision factor
        cores: number of cpu cores
        
    Returns: 
        vox_prism_mat: a scipy.sparse CSR matrix (compressed rows), of shape
            (n_voxs, n_tris), in which each entry at index [I,J] gives the 
            number of samples from triangle prism J that are in voxel I. 
            NB this matrix is not normalised in any way!
    """

    worker = functools.partial(__voxprism_mat_worker, in_surf=in_surf, out_surf=out_surf,
        spc=spc, factor=factor)
    
    if cores > 1: 
        t_ranges = utils._distributeObjects(range(in_surf.tris.shape[0]), cores)
        with mp.Pool(cores) as p: 
            vpmats = p.map(worker, t_ranges)
            
        vpmat = vpmats[0]
        for vt in vpmats[1:]:
            vpmat += vt
            
    else: 
        vpmat = worker(range(in_surf.tris.shape[0]))
        
    return vpmat


def __voxprism_mat_worker(t_range, in_surf, out_surf, spc, factor):
    """
    Helper method for _voxprism_mat(). 

    Args: 
        t_range: iterable of triangle numbers to process
        in_surf: inner surface of cortex, voxel coordinates
        out_surf: outer surface of cortex, voxel coordinates 
        spc: ImageSpace in which surfaces lie 
        factor: voxel subdivision factor

    Returns: 
        sparse CSR matrix of size (n_vox x n_tris)
    """

    vox_tri_samps = spmat.dok_matrix((spc.size.prod(), 
        in_surf.tris.shape[0]), dtype=np.int32)
    sampler = np.linspace(0,1, 2*factor + 1)[1:-1:2]
    sx, sy, sz = np.meshgrid(sampler, sampler, sampler)
    samples = np.vstack((sx.flatten(),sy.flatten(),sz.flatten())).T - 0.5

    for t in t_range: 
        hull = np.vstack((in_surf.points[in_surf.tris[t,:],:], 
                        out_surf.points[out_surf.tris[t,:],:]))
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
                    
        except QhullError:
            continue  

        except Exception as e: 
            raise e 
                        
    return vox_tri_samps.tocsr()


def _vtxtri_mat(surf, cores=mp.cpu_count()):
    """
    Form matrix of (n_vertices x n_tris) in which element (I,J) is the 
    area of triangle J that belongs to vertex I. WARNING: this is not 
    normalised (the areas associated with any vertex or triangle do not
    sum to 1). 

    Args: 
        surf: Surface object
        cores: number of cores to use 

    Returns:
        sparse CSR matrix. 
    """

    n_vtx = surf.points.shape[0]
    tri_areas = _triangle_areas(surf.points, surf.tris) / 3
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
    """
    Worker function for multiprocessing with _vtxtri_mat()

    Args: 
        vtx_range: Range object of vertices within the surface to process
        surf: Surface object 
        vtx_tri_areas: matrix of size (n_tris, 3), representing area of
            each triangle belonging to each of its vertices

    Returns: 
        sparse CSR matrix of shape (n_vertices, n_tris)
    """

    n_vtx = surf.points.shape[0]
    n_tris = surf.tris.shape[0]
    vtx_tri_mat = spmat.dok_matrix((n_vtx,n_tris), dtype=np.float32)

    for vtx in vtx_range:
        inds = (surf.tris == vtx)
        tris = np.flatnonzero((inds.any(1)))
        vtx_tri_mat[vtx,tris] = vtx_tri_areas[inds]

    return vtx_tri_mat.tocsr()


def _triangle_areas(ps, ts):
    return 0.5 * np.linalg.norm(np.cross(
        ps[ts[:,1],:] - ps[ts[:,0],:], 
        ps[ts[:,2],:] - ps[ts[:,0],:], axis=-1), axis=-1, ord=2)


def _vertex_areas(ps, ts):
    tri_areas = _triangle_areas(ps,ts)  
    vtx_areas = np.bincount(ts.flat, np.repeat(tri_areas, 3)) / 3 
    return vtx_areas



# def vol2prism_weights(in_surf, out_surf, spc, factor, cores=mp.cpu_count()):

#     voxtri_mat = _voxprism_mat(in_surf, out_surf, spc, factor, cores)
#     sums = voxtri_mat.sum(0).A.reshape(-1)
#     for abv in np.flatnonzero(sums):
#         voxtri_mat[abv,:] /= voxtri_mat[abv,:].sum()

#     set_trace()
#     fltr = (sums > 0)
#     voxtri_mat[fltr,:] /= sums[fltr][:,None]
#     set_trace()
#     return voxtri_mat

# def prism2vol_weights(): 
#     pass 