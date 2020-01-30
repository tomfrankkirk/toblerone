"""
Toblerone surface-volume projection functions
"""
from pdb import set_trace
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError 
import numpy as np 
import functools 
import itertools
import multiprocessing as mp 
from scipy import sparse 
import copy 

from . import utils 
from .classes import  ImageSpace, Surface


def calc_midsurf(in_surf, out_surf):
    vec = out_surf.points - in_surf.points 
    points =  in_surf.points + (0.5 * vec)
    return Surface.manual(points, in_surf.tris)

def vol2surf(vdata, in_surf, out_surf, spc, factor=10, cores=mp.cpu_count()):
    """
    Project volumetric data to the cortical ribbon defined by inner/outer 
    surfaces. NB any registration transforms must be applied to the surfaces 
    beforehand, such that they are in world-mm coordinates for this function. 

    Args: 
        vdata: np.array of shape equal to spc.size, or flattened, to project
        in_surf: inner Surface of ribbon, in world mm coordinates 
        out_surf: outer Surface of ribbon, in world mm coordinates
        spc: ImageSpace in which data exists 
        factor: voxel subdivision factor (default 10)
        cores: number of processor cores (default max)

    Returns:   
        flat np.array of size equal to n_vertices
    """

    if not vdata.size == spc.size.prod(): 
        raise RuntimeError("Size of vdata does not match size of ImageSpace")

    v2s_weights = vol2surf_weights(in_surf, out_surf, spc, factor, cores)
    return v2s_weights.dot(vdata.reshape(-1))


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
    [ s.applyTransform(spc.world2vox) for s in [in_surf, out_surf] ]
    mid_surf = calc_midsurf(in_surf, out_surf)

    # Two step projection: from volume to prisms, then prisms to vertices
    vox2tri_mat = _vox2tri_mat(in_surf, out_surf, spc, factor, cores).tocsc()
    vtx2tri_mat = _vtx2tri_mat(mid_surf, cores).tocsr()

    # Normalise vtx2tri matrix so that per-triangle vertex areas sum to 1 
    norm = vtx2tri_mat.sum(0).A.flatten()
    fltr = (norm < 1e-3)
    norm[fltr] = 1
    vtx2tri_mat.data /= np.take(norm, vtx2tri_mat.indices)

    n_vtx = in_surf.points.shape[0]
    worker = functools.partial(__vol2surf_worker, 
            vox2tri_mat=vox2tri_mat, vtx2tri_mat=vtx2tri_mat)

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


def __vol2surf_worker(vertices, vox2tri_mat, vtx2tri_mat):
    """
    Worker function for vol2surf_weights(). 

    Args: 
        vertices: iterable of vertex numbers to process 
        vox2tri_mat: produced by _vox2tri_mat() 
        vtx2tri_mat: produced by _vtx2tri_mat()

    Returns: 
        CSR matrix of size (n_vertices x n_voxs)
    """

    weights = sparse.dok_matrix((vtx2tri_mat.shape[0], vox2tri_mat.shape[0]), 
                dtype=np.float32)  

    for vtx in vertices: 
        tri_weights = vtx2tri_mat[vtx,:]
        vox_weights = vox2tri_mat[:,tri_weights.indices]
        vox_weights.data *= tri_weights.data[vox_weights.tocsr().indices]
        u_voxs, at_inds = np.unique(vox_weights.indices, return_inverse=True)
        vtx_vox_weights = np.bincount(at_inds, weights=vox_weights.data)
        weights[vtx,u_voxs] = (vtx_vox_weights / vtx_vox_weights.sum())

    return weights.tocsr()


def surf2vol(sdata, in_surf, out_surf, spc, factor=10, cores=mp.cpu_count()):
    """
    Project data defined on surface vertices to a volumetric space. NB any 
    registration transforms must be applied to the surfaces beforehand, 
    such that they are in world-mm coordinates for this function. 

    Args: 
        sdata: np.array of shape equal to surf.n_vertices, data to project
        in_surf: inner Surface of ribbon, in world mm coordinates 
        out_surf: outer Surface of ribbon, in world mm coordinates
        spc: ImageSpace in which to project data
        factor: voxel subdivision factor (default 10)
        cores: number of processor cores (default max)

    Returns:   
        flat np.array of size equal to spc.n_voxels
    """

    if not sdata.size == in_surf.points.shape[0]:
        raise RuntimeError("Size of sdata does not match surface")

    s2v_weights = surf2vol_weights(in_surf, out_surf, spc, factor, cores)
    return s2v_weights.dot(sdata.reshape(-1))


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
    [ s.applyTransform(spc.world2vox) for s in [in_surf, out_surf] ]
    mid_surf = calc_midsurf(in_surf, out_surf)

    vox2tri_mat = _vox2tri_mat(in_surf, out_surf, spc, factor, cores)
    vtx2tri_mat = _vtx2tri_mat(mid_surf, cores).tocsc()

    # Normalise vtx2tri matrix so that per-vertex tri weights sum to 1
    norm = vtx2tri_mat.sum(1).A.flatten()
    fltr = (norm < 1e-3)
    norm[fltr] = 1 
    vtx2tri_mat.data /= np.take(norm, vtx2tri_mat.indices)

    voxs_nonzero = np.flatnonzero(vox2tri_mat.sum(1) > 0)
    worker = functools.partial(__surf2vol_worker, 
        vox2tri_mat=vox2tri_mat, vtx2tri_mat=vtx2tri_mat)

    if cores > 1: 
        vox_ranges = [ voxs_nonzero[c] for c in 
            utils._distributeObjects(range(voxs_nonzero.size), cores) ]

        with mp.Pool(cores) as p: 
            ws = p.map(worker, vox_ranges)

        weights = ws[0]
        for w in ws[1:]:
            weights += w 

    else:
        weights = worker(voxs_nonzero)
    
    return weights


def __surf2vol_worker(voxs, vox2tri_mat, vtx2tri_mat):
    """
    Returns CSR matrix of size (n_vertices x n_verts)
    """

    # Weights matrix is sized (n_voxs x n_vertices)
    # On each row, the weights will be stored at the column indices of 
    # the relevant vertex numbers 
    weights = sparse.dok_matrix((vox2tri_mat.shape[0], vtx2tri_mat.shape[0]), 
                dtype=np.float32)  

    for vox in voxs:
        tri_weights = vox2tri_mat[vox,:]
        vtx_weights = vtx2tri_mat[:,tri_weights.indices]
        vtx_weights.data *= tri_weights.data[vtx_weights.tocsr().indices]
        u_vtx, at_inds = np.unique(vtx_weights.indices, return_inverse=True)
        vox_vtx_weights = np.bincount(at_inds, weights=vtx_weights.data)
        weights[vox,u_vtx] = vox_vtx_weights    

    return weights.tocsr()     


def _vox2tri_mat(in_surf, out_surf, spc, factor=10, cores=mp.cpu_count()):     
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
        vox2tri_mat: a scipy.sparse CSR matrix (compressed rows), of shape
            (n_voxs, n_tris), in which each entry at index [I,J] gives the 
            number of samples from triangle prism J that are in voxel I. 
            NB this matrix is not normalised in any way!
    """

    n_tris = in_surf.tris.shape[0]
    worker = functools.partial(__vox2tri_mat_worker, in_surf=in_surf, 
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


def __vox2tri_mat_worker(t_range, in_surf, out_surf, spc, factor):
    """
    Helper method for _vox2tri_mat(). 

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
        in_surf.tris.shape[0]), dtype=np.int16)
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


# def _triangle_areas(ps, ts):
#     """Areas of triangles, vector of length n_tris"""
#     return 0.5 * np.linalg.norm(np.cross(
#         ps[ts[:,1],:] - ps[ts[:,0],:], 
#         ps[ts[:,2],:] - ps[ts[:,0],:], axis=-1), axis=-1, ord=2)


# def _simple_vertex_areas(ps, ts):
#     """Areas surrounding each vertex in surface, default 1/3 of each tri"""
     
#     tri_areas = _triangle_areas(ps,ts)  
#     vtx_areas = np.bincount(ts.flat, np.repeat(tri_areas, 3)) / 3 
#     return vtx_areas


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
    vtx_tri_areas = sparse.dok_matrix((points.shape[0], tris.shape[0]), 
        dtype=np.float32)

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


def _vtx2tri_mat(surf, cores=mp.cpu_count()):
    """
    Form a matrix of size (n_vertices x n_tris) where element (I,J) corresponds
    to the area of triangle J belonging to vertex I. 

    Areas are calculated according to the definition of A_mixed in "Discrete 
    Differential-Geometry Operators for Triangulated 2-Manifolds", M. Meyer, 
    M. Desbrun, P. Schroder, A.H. Barr.

    With thanks to Jack Toner for the original implementation from which this is 
    adapted. 

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
        vtx2tri_mat = results[0]
        for r in results[1:]:
            vtx2tri_mat += r 

    else: 
        vtx2tri_mat = worker_func(range(points.shape[0]))

    assert (vtx2tri_mat.data > 0).all(), 'zero areas returned'
    return vtx2tri_mat 
