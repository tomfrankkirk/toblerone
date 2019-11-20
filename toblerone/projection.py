from scipy.spatial import Delaunay
import numpy as np 
import functools 
import collections 
import itertools
import multiprocessing as mp 
import scipy.sparse as spmat
import nibabel 
from pdb import set_trace

from . import utils 
from .classes import  ImageSpace, Surface

def vol2surf(img, ins, outs, spc, factor, cores=mp.cpu_count()-1):

    s2r = np.eye(4)
    img = nibabel.load(img)
    data = img.get_fdata()

    if not isinstance(spc, ImageSpace): spc = ImageSpace(spc)
    if not isinstance(ins, Surface): ins = Surface(ins)
    if not isinstance(outs, Surface): outs = Surface(outs)

    # Need to find bridges - for which surfaces must be indexed first
    # Which in turn means they need to be in voxel coords
    surfs = [ins, outs]
    bridges = np.union1d(*(s.find_bridges(spc, s2r) for s in surfs))
    [ s.deindex(s2r) for s in surfs ]
    overall = spc.world2vox @ s2r
    [ s.applyTransform(overall) for s in surfs ]

    if len(data.shape) != 1: 
        data = data.reshape(-1)

    out = np.zeros(ins.tris.shape[0], dtype=np.float32)
    weights = vol2surf_weights(ins, outs, spc, factor, cores)

    for widx,(inds,we) in enumerate(weights): 
        out[widx] = (data[inds] * we).sum()

    return out 

def surf2vol(sdata, ins, outs, spc, factor, cores=mp.cpu_count()-1):

    sdata = nibabel.load(sdata)

    assert sdata.shape.size == ins.tris.shape[0]
    out = np.zeros(spc.size.prod(), dtype=np.float32)
    voxs, weights = surf2vol_weights(ins, outs, spc, factor, cores)
    
    for v,(inds,we) in zip(voxs, weights):
        out[v] = (sdata[inds] * we).sum()

    return out 

def surf2vol_weights(ins, outs, spc, factor, cores):
    """
    Estimate weights used for projecting from surface to volume

    Args:
        ins: Surface object, inner surface of cortical ribbon
        outs: Surface object, outer surface of cortical ribbon
        spc: ImageSpace object within which to project 
        factor: int voxel subdivision factor
        cores: number of cpu cores (default max-1)

    Returns:
        voxs: list of voxels 
        weights: list of np.arrays, length equal to voxs, for which 
            the top row of each array are triangle indices and the 
            bottom row of each are corresponding triangle weights 
    """
    
    vtsamps = _form_voxtri_mat(ins, outs, spc, factor, cores)
    voxs = np.unique(vtsamps.nonzero()[1])
    vtsamps = vtsamps.tocsr()
    weights = []
    for v in voxs:
        vt = vtsamps[v,:]
        weights.append((vt.nonzero()[1], vt.data / vt.data.sum()))
    
    return (voxs, weights)


def vol2surf_weights(ins, outs, spc, factor, cores):
    """
    Estimate weights used for projecting from volume to surface. 
    Note that some triangles may not have any voxel weights. This is
    because they represent an insignificant volume of cortex.

    Args:
        ins: Surface object, inner surface of cortical ribbon
        outs: Surface object, outer surface of cortical ribbon
        spc: ImageSpace object within which to project 
        factor: int voxel subdivision factor
        cores: number of cpu cores (default max-1)

    Returns:
        weights: list of np.arrays, length equal to number of triangles, 
            for which the top row of each array are voxel indices and
            the bottom row of each are the corresponding weights 
    """
    
    vtsamps = _form_voxtri_mat(ins, outs, spc, factor, cores)
    vtsamps = vtsamps.tocsc()
    weights = []
    for t in range(vtsamps.shape[1]):
        vt = vtsamps[:,t]
        weights.append((vt.nonzero()[0], vt.data / vt.data.sum()))

    assert len(weights) == ins.tris.shape[0] 
    return weights
                     

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
        vox_tri_samps: a scipy.sparse CSR matrix (commpressed sparse row), 
            of shape (n_voxs, n_tris), in which each entry at index [I,J] 
            gives the number of samples from triangle J that are in voxel I. 
    """

    worker = functools.partial(__vtmat_worker, 
                               n_tris=ins.tris.shape[0], inps=ins.points, 
                               outps=outs.points, spc=spc, factor=factor)
    
    if cores > 1: 
        chunks = utils._distributeObjects(range(ins.tris.shape[0]), cores)
        chunks = [ ins.tris[c,:] for c in chunks ]   
        with mp.Pool(cores) as p: 
            vtsamps = p.map(worker, chunks)
            
        vox_tri_samps = (vtsamps[0]).tocsr()
        for vt in vtsamps[1:]:
            vox_tri_samps += vt.tocsr()
            
    else: 
        vox_tri_samps = worker(ins.tris)
        
    return vox_tri_samps

def __vtmat_worker(tris, n_tris, inps, outps, spc, factor):
    """
    Helper method for _form_voxtri_mat(), to be used in mp.Pool()
    """

    vox_tri_samps = spmat.dok_matrix((spc.size.prod(), n_tris), dtype=np.int16)
    sampler = np.linspace(0,1, 2*factor + 1)[1:-1:2]
    sx, sy, sz = np.meshgrid(sampler, sampler, sampler)
    samples = np.vstack((sx.flatten(),sy.flatten(),sz.flatten())).T - 0.5

    for tidx,t in enumerate(tris): 
        hull = np.vstack((inps[t,:], outps[t,:]))
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
                    vox_tri_samps[vidx,tidx] = samps_in
                    
        except: 
            pass 
                        
    return vox_tri_samps