"""Toblerone tests"""

import os.path as op
import pickle 
from pdb import set_trace
import os 
import multiprocessing
import sys 
import copy 

import numpy as np
from scipy import sparse
from numpy.lib.index_tricks import diag_indices
import nibabel
import trimesh

import toblerone
from toblerone import pvestimation
from toblerone.classes import Surface, Hemisphere
from toblerone import core, utils
from toblerone.ctoblerone import _cyfilterTriangles
from toblerone.pvestimation import estimators
from regtricks.application_helpers import sum_array_blocks
from toblerone.utils import NP_FLOAT, slice_sparse, sparse_normalise
from toblerone.__main__ import main
from toblerone.classes.image_space import reindexing_filter

cores = multiprocessing.cpu_count()

def get_testdir():
    return op.join(op.dirname(op.realpath(__file__)), 'testdata')


def test_indexing():
    td = get_testdir()
    surf = toblerone.Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    surf.index_on(spc, 1)
    surf.indexed.voxelised = surf.voxelise(spc, 1)

    truth = pickle.load(open(op.join(td, 'out_indexed.pkl'), 'rb'))
    truthspace = truth.indexed.space
    space = surf.indexed.space
    assocs = surf.indexed.assocs
    assert (np.array_equal(truth.indexed.assocs.indices, assocs.indices) and 
            (np.array_equal(truth.indexed.assocs.data, assocs.data)))
    assert np.all(space.bbox_origin == truthspace.bbox_origin)
    assert np.all(space.size == truthspace.size)
    assert np.all(space.offset == truthspace.offset)
    assert np.array_equal(surf.indexed.voxelised, truth.indexed.voxelised)


def test_enclosing_space(): 
    td = get_testdir()
    surf = toblerone.Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    surf.index_on(spc, 1)
    assert utils.space_encloses_surface(surf.indexed.space, surf.indexed.points_vox)

    spc2 = spc.resize([5,5,5], [5,5,5])
    surf2 = toblerone.Surface(op.join(td, 'out.surf.gii'))
    surf2.index_on(spc2, 1)
    assert not utils.space_encloses_surface(spc2, surf2.indexed.points_vox)
    assert surf.indexed.space == surf2.indexed.space

    surf = toblerone.Surface(op.join(td, 'out.surf.gii'))
    start = surf.points.min(0)
    end = surf.points.max(0)
    vsize = (start - end) / 10
    spc = toblerone.ImageSpace.create_axis_aligned(
           start, end, vsize 
    )
    surf.index_on(spc, 1)
    assert utils.space_encloses_surface(surf.indexed.space, surf.indexed.points_vox)

    


def test_vox_tri_intersection():

    # Generate N triangles for which all vertices are contained
    # within a unit voxel at the origin 
    N = int(1e6)
    ps = (np.random.rand(N,3,3) - 0.5).astype(NP_FLOAT)
    ts = np.arange(3*N).reshape(N,3).astype(np.int32)

    # Multiply out the first two vertices of each triangle,
    # which will push most of them outside the voxel
    ps[:,1:,:] *= np.random.randint(1, 10, (N,2))[:,:,None]
    ps = ps.reshape(-1,3)

    # Unit voxel, assert all intersect. 
    cent = np.zeros(3, dtype=NP_FLOAT)
    size = np.ones(3, dtype=NP_FLOAT)
    flags = _cyfilterTriangles(ts, ps, cent, size)
    assert flags.all(), 'Not all triangles intersect voxel'


def test_projection():
    td = get_testdir()
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    sdata = np.ones(hemi.inSurf.n_points, dtype=NP_FLOAT)
    vdata = np.ones(spc.size.prod(), dtype=NP_FLOAT)
    ndata = np.concatenate((sdata, vdata))
    projector = toblerone.projection.Projector(hemi, spc, factor=10)

    # volume to surface 
    v2s = projector.vol2surf(vdata, False)
    v2s_edge = projector.vol2surf(vdata, True)
    assert (v2s <= v2s_edge).all(), "edge correction did not increase signal"
    v2s = projector.vol2surf_matrix(False)
    v2s_edge = projector.vol2surf_matrix(True)
    assert (v2s_edge.data >= v2s.data).all(), 'egde correction: some weights should increase'

    # surface to volume 
    s2v = projector.surf2vol(sdata, False)
    s2v_pv = projector.surf2vol(sdata, True)
    assert (s2v_pv <= s2v).all(), "pv weighting did not reduce signal"
    s2v = projector.surf2vol_matrix(False)
    assert (s2v.sum(1).max() - 1) < 1e-6, 'total voxel weight > 1'
    s2v_pv = projector.surf2vol_matrix(False)
    assert (s2v_pv.data <= s2v.data).all(), 'pv weighting should reduce voxel weights'

    # volume to node 
    v2n = projector.vol2hybrid(vdata, False)
    v2n_edge = projector.vol2hybrid(vdata, True)
    assert (v2n <= v2n_edge).all(), "edge correction did not increase signal"
    v2n = projector.vol2hybrid_matrix(False)
    assert (v2n.sum(1).max() - 1) < 1e-6, 'total node weight > 1'
    v2n_edge = projector.vol2hybrid_matrix(True)
    assert (v2n_edge.sum(1).max() - 1) > 1e-6, 'edge correction: node should have weight > 1'

    # node to volume 
    n2v = projector.hybrid2vol(ndata, False)
    n2v_pv = projector.hybrid2vol(ndata, True)
    assert (n2v_pv <= n2v).all(), "pv weighting did not reduce signal"
    n2v = projector.hybrid2vol_matrix(False)
    assert (n2v.sum(1).max() - 1) < 1e-6, 'total voxel weight > 1'
    n2v_pv = projector.hybrid2vol_matrix(True)
    assert (n2v.sum(1).max() - 1) < 1e-6, 'total voxel weight > 1'


def test_projector_partial_fov(): 
    td = get_testdir()
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    spc = spc.resize([1,1,1], spc.size-2)
    projector = toblerone.projection.Projector(hemi, spc, cores=8)
    sdata = np.ones(hemi.inSurf.n_points, dtype=NP_FLOAT)
    vdata = np.ones(spc.size.prod(), dtype=NP_FLOAT)
    # ndata = np.concatenate((sdata, vdata))
    # proj = projector.surf2vol(sdata, True)

    adj = projector.adjacency_matrix()


def test_projector_rois():
    td = get_testdir()
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    fracs = nibabel.load(op.join(td, 'sph_fractions.nii.gz')).get_fdata()
    hemi.pvs = fracs.reshape(-1,3)
    puta = Surface.manual(0.25 * hemi.inSurf.points, 
                            hemi.inSurf.tris, 'L_Puta')
    rois = { 'L_Puta': puta }
    proj = toblerone.projection.Projector(hemi, spc, rois=rois, cores=8)

    ndata = np.ones(proj.n_nodes)
    ndata[-1] = 2 
    vdata = proj.hybrid2vol(ndata, True)
    spc.save_image(vdata, 'n2v.nii.gz')

    ndata = proj.vol2hybrid(vdata, True)
    print(ndata)


def test_subvoxels():

    supersampler = np.random.randint(2, 3, 3)
    subvox_size = 1.0 / supersampler
    vox_cent = np.random.randint(-10, 10, 3)
    subvox_cents = core._get_subvoxel_grid(supersampler) + vox_cent
    assert (np.abs(subvox_cents.mean(0) - vox_cent) < 1e-6).all()
    assert subvox_cents.shape[0] == supersampler.prod()
    for cidx, scent in enumerate(subvox_cents):
        corners = scent + ((core.SUBVOXCORNERS) * (subvox_size[None,:]))
        assert core._filterPoints(corners, scent, subvox_size).all()


def test_convert(): 
    td = get_testdir()
    s = Surface(op.join(td, 'in.surf.gii'))
    s.save('test.vtk')
    s2 = Surface('test.vtk')
    assert np.allclose(s.points, s2.points)
    os.remove('test.vtk')


def test_proj_properties():
    td = get_testdir()
    ins = Surface(op.join(td, 'in.surf.gii'))
    outs = Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    proj = toblerone.projection.Projector(hemi, spc)
    assert proj.n_hemis == 1 
    assert 'L' in proj.hemi_dict
    assert hemi.midsurface()
    assert proj['LPS']

    hemi2 = toblerone.Hemisphere(ins, outs, 'R')
    proj = toblerone.projection.Projector([hemi, hemi2], spc)
    assert proj.n_hemis == 2 
    assert proj['RWS']
    assert ('L' in proj.hemi_dict) & ('R' in proj.hemi_dict)
    for h,s in zip(proj.iter_hemis, ['L', 'R']):
        assert h.side == s 

    assert proj.n_surf_nodes == 2 * ins.n_points


def test_hemi_init():
    td = get_testdir()
    ins = Surface(op.join(td, 'in.surf.gii'))
    outs = Surface(op.join(td, 'out.surf.gii'))
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    hemi2 = toblerone.Hemisphere(ins, outs, 'L')
    assert id(hemi.inSurf.points) != id(hemi2.inSurf.points)


def test_surf_edges():
    td = get_testdir()
    ins = Surface(op.join(td, 'in.surf.gii'))
    e = ins.edges()


def test_adjacency():
    td = get_testdir()
    ins = Surface(op.join(td, 'in.surf.gii'))
    outs = Surface(op.join(td, 'out.surf.gii'))
    h = Hemisphere(ins, outs, 'L')
    for w in range(4):
        adj = h.adjacency_matrix(w)
        assert not (adj.data < 0).any(), 'negative value in adjacency matrix'

    try: 
        h.adjacency_matrix(-1)
    except Exception as e: 
        assert isinstance(e, ValueError), 'negative distance weight should give ValueError'     

    outs = Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    hemi2 = toblerone.Hemisphere(ins, outs, 'R')
    proj = toblerone.projection.Projector([hemi, hemi2], spc)
    n = proj.hemi_dict['L'].n_points

    for w in range(4):
        adj = proj.adjacency_matrix(w)
        assert not slice_sparse(adj, slice(0, n), slice(n, 2*n)).nnz
        assert not slice_sparse(adj, slice(n, 2*n), slice(0, n)).nnz


def test_mesh_laplacian():
    td = get_testdir()
    s = Surface(op.join(td, 'in.surf.gii'))
    outs = Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    hemi = toblerone.Hemisphere(s, outs, 'L')
    hemi2 = toblerone.Hemisphere(s, outs, 'R')
    proj = toblerone.projection.Projector([hemi, hemi2], spc)

    for w in range(4):
        lap = proj.mesh_laplacian(distance_weight=w)
        assert (lap[np.diag_indices(lap.shape[0])] < 0).min(), 'positive diagonal'
        n = proj.hemi_dict['L'].n_points
        assert not slice_sparse(lap, slice(0, n), slice(n, 2*n)).nnz
        assert not slice_sparse(lap, slice(n, 2*n), slice(0, n)).nnz
        assert not (lap[diag_indices(2*n)] > 0).any()


def test_cotan_laplacian(): 
    td = get_testdir()
    ins = Surface(op.join(td, 'in.surf.gii'))
    outs = Surface(op.join(td, 'out.surf.gii'))
    h = Hemisphere(ins, outs, 'L')
    h.cotangent_laplacian()


def cmdline_complete():
    fslanat = "/Users/tom/Data/pcasl2/1.anat"
    ref = fslanat + "/T1.nii.gz"
    fsdir = fslanat + "/fs"
    cmd = f""" -estimate-complete -ref {ref} -struct2ref I 
                -fslanat {fslanat} -fsdir {fsdir} -out delete"""
    sys.argv[1:] = cmd.split()
    main()     


def test_cortex():
    td = get_testdir()
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))

    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = Hemisphere(ins, outs, 'L')
    s2r = np.identity(4)
    supersampler = np.random.randint(3,6,3)
    fracs = estimators._cortex(hemi, spc, s2r, supersampler, 
        8, False)
    spc.save_image(fracs, f'{td}/fracs.nii.gz')

    # REFRESH the surfaces of the hemisphere before starting again - indexing! 
    hemi = Hemisphere(ins, outs, 'L')
    superfactor = 10
    spc_high = spc.resize_voxels(1.0/superfactor)
    voxelised = np.zeros(spc_high.size.prod(), dtype=NP_FLOAT)

    hemi.inSurf.index_on(spc_high)
    hemi.inSurf.indexed.voxelised = hemi.inSurf.voxelise(spc_high, 1)

    reindex_in = reindexing_filter(hemi.inSurf.indexed.space, spc_high)
    voxelised[reindex_in[1]] = -(hemi.inSurf.indexed.
                                    voxelised[reindex_in[0]]).astype(NP_FLOAT)

    hemi.outSurf.index_on(spc_high)
    hemi.outSurf.indexed.voxelised = hemi.outSurf.voxelise(spc_high, 1)
    reindex_out = reindexing_filter(hemi.outSurf.indexed.space, spc_high)
    voxelised[reindex_out[1]] += hemi.outSurf.indexed.voxelised[reindex_out[0]]

    voxelised = voxelised.reshape(spc_high.size)
    truth = sum_array_blocks(voxelised, 3 * [superfactor]) / superfactor**3
    spc.save_image(truth, f'{td}/truth.nii.gz')

    # truth = np.squeeze(nibabel.load(op.join(td, 'truth.nii.gz')).get_fdata())
    np.testing.assert_array_almost_equal(fracs[...,0], truth, 2)


def test_structure():
    td = get_testdir()
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    ins = Surface(op.join(td, 'in.surf.gii'), name='L')
    s2r = np.identity(4)

    fracs = pvestimation.structure(surf=op.join(td, 'in.surf.gii'), ref=spc, struct2ref=s2r, cores=1, struct=op.join(td, 'ref.nii.gz'))

    superfactor = 10
    spc_high = spc.resize_voxels(1.0/superfactor)
    voxelised = np.zeros(spc_high.size.prod(), dtype=NP_FLOAT)
    ins.index_on(spc_high)
    ins.indexed.voxelised = ins.voxelise(spc_high, 1)

    reindex_in = reindexing_filter(ins.indexed.space, spc_high)
    voxelised[reindex_in[1]] = (ins.indexed.
                                    voxelised[reindex_in[0]]).astype(NP_FLOAT)

    voxelised = voxelised.reshape(spc_high.size)
    truth = sum_array_blocks(voxelised, 3 * [superfactor]) / superfactor**3

    np.testing.assert_array_almost_equal(fracs, truth, 2)

    
def test_sparse_normalise():
    mat = sparse.random(5000, 5000, 0.1)
    thr = 1e-12
    for axis in range(2):
        normed = sparse_normalise(mat, axis, thr)
        sums = normed.sum(axis).A.flatten()
        assert (np.abs(sums[sums > 0] - 1) <= thr).all()


def test_projector_hdf5():
    td = get_testdir()
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    spc = op.join(td, 'ref.nii.gz')
    proj = toblerone.Projector(hemi, spc)
    proj.save('proj.h5')
    proj2 = toblerone.Projector.load('proj.h5')

    assert np.array_equiv(proj.pvs(), proj2.pvs())
    assert np.array_equiv(proj.spc, proj2.spc)
    assert np.array_equiv(proj.vox_tri_mats[0].data, 
                          proj2.vox_tri_mats[0].data)
    assert np.array_equiv(proj.vtx_tri_mats[0].data, 
                          proj2.vtx_tri_mats[0].data)

    os.remove('proj.h5')


def test_projector_cmdline():
    td = get_testdir()
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    spc = op.join(td, 'ref.nii.gz')

    cmd = f"""-prepare-projector -ref {spc} -LPS {outs} 
        -LWS {ins} -out proj -struct2ref I"""
    sys.argv[1:] = cmd.split()
    main()
    os.remove('proj.h5')


def cmdline(): 
    # sys.argv[1:] = ['-estimate-cortex']
    main()


def test_projector_cmdline():
    td = get_testdir()
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    spc = op.join(td, 'ref.nii.gz')

    cmd = f"""-prepare-projector -ref {spc} -LPS {outs} 
        -LWS {ins} -out proj -struct2ref I"""
    sys.argv[1:] = cmd.split()
    main()
    os.remove('proj.h5')


def test_prefer_convex_hull():

    # This was a sneaky bug. When estimating PVs for a voxel that intersects the surface, 
    # we should aim to pick the side that is more convex, so the convex hull volume will 
    # be more accurate. The below test case of a very small sphere that is contained within
    # a grid of 2,2,2 voxels is nasty because there are float rounding issues. On one side, 
    # a fold in the surface is detected and recursion is used. On the other side, no fold
    # is detected (due to float equality issues) and therefore a hull is used. It's important
    # that the two strategies yield similar results. 

    fov = 4
    vox_size = np.ones(3)
    spc = toblerone.ImageSpace.create_axis_aligned([0,0,0], (fov/vox_size).astype(int), vox_size)
    
    sph = trimesh.creation.icosphere(3, radius=1.0)
    sph.vertices += fov/2
    surf = Surface.manual(sph.vertices, sph.faces)

    # ground truth 
    factor = 10
    spc2 = spc.resize_voxels(1/factor)
    surf2 = copy.deepcopy(surf)
    surf2.index_on(spc2)
    surf2.indexed.voxelised = surf2.voxelise(spc2, 1)
    truth = np.zeros(spc2.size.prod())
    src_fltr, dest_fltr = reindexing_filter(surf2.indexed.space, spc2)
    truth[dest_fltr] = surf2.indexed.voxelised
    truth = sum_array_blocks(truth.reshape(spc2.size), 3*[factor]) / (factor**3)

    pvs = pvestimation.structure(spc, 'I', surf, cores=1)

    assert np.abs(truth - pvs).max() < 0.01, 'PVs for small sphere do not match truth'


if __name__ == "__main__":
    
    test_projector_cmdline()