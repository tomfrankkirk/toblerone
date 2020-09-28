"""Toblerone tests"""

import os.path as op 
import nibabel 
import pickle 
from nibabel import test
import numpy as np 
import multiprocessing
from pdb import set_trace
import os 

import toblerone
from toblerone import classes, projection, core 
from toblerone.ctoblerone import _cyfilterTriangles
from toblerone.pvestimation import estimators
from regtricks.application_helpers import sum_array_blocks
from toblerone.utils import NP_FLOAT

cores = multiprocessing.cpu_count()

def get_testdir():
    return op.join(op.dirname(op.realpath(__file__)), 'testdata')

def test_indexing():
    td = get_testdir()
    surf = toblerone.Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    surf.index_on(spc, np.identity(4), 1)

    truth = pickle.load(open(op.join(td, 'out_indexed.pkl'), 'rb'))
    truthspace = truth._index_space
    space = surf._index_space
    assert (np.array_equal(truth.assocs.indices, surf.assocs.indices) and 
            (np.array_equal(truth.assocs.data, surf.assocs.data)))
    assert np.all(space.bbox_origin == truthspace.bbox_origin)
    assert np.all(space.size == truthspace.size)
    assert np.all(space.offset == truthspace.offset)
    assert np.array_equal(surf.voxelised, truth.voxelised)

def test_cortex():
    td = get_testdir()
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))

    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = classes.Hemisphere(ins, outs, 'L')
    s2r = np.identity(4)
    supersampler = np.random.randint(3,6,3)
    fracs = estimators._cortex(hemi, spc, s2r, supersampler, 
        8, False)
    # spc.save_image(fracs, f'{td}/fracs.nii.gz')

    # REFRESH the surfaces of the hemisphere before starting again - indexing! 
    hemi = classes.Hemisphere(ins, outs, 'L')
    superfactor = 10
    spc_high = spc.resize_voxels(1.0/superfactor)
    voxelised = np.zeros(spc_high.size.prod(), dtype=NP_FLOAT)
    hemi.inSurf.index_on(spc_high, s2r)
    reindex_in = hemi.inSurf.reindexing_filter(spc_high)
    voxelised[reindex_in[1]] = -(hemi.inSurf.voxelised[reindex_in[0]]).astype(NP_FLOAT)
    hemi.outSurf.index_on(spc_high, s2r)
    reindex_out = hemi.outSurf.reindexing_filter(spc_high)
    voxelised[reindex_out[1]] += hemi.outSurf.voxelised[reindex_out[0]]
    voxelised = voxelised.reshape(spc_high.size)
    truth = sum_array_blocks(voxelised, 3 * [superfactor]) / superfactor**3
    # spc.save_image(truth, f'{td}/truth.nii.gz')

    # truth = np.squeeze(nibabel.load(op.join(td, 'truth.nii.gz')).get_fdata())
    np.testing.assert_array_almost_equal(fracs[...,0], truth, 2)

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
    sdata = np.ones(hemi.inSurf.points.shape[0], dtype=NP_FLOAT)
    vdata = np.ones(spc.size.prod(), dtype=NP_FLOAT)

    projector = toblerone.projection.Projector(hemi, spc, 10, 1)
    v2s = projector.vol2surf(vdata)
    v2s_edge = projector.vol2surf(vdata, True)
    assert (v2s <= v2s_edge).all(), "edge correction did not increase signal"
    s2v = projector.surf2vol(sdata)
    v2n = projector.vol2node(vdata)
    n2v = projector.node2vol(v2n)

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
    s = classes.Surface(op.join(td, 'in.surf.gii'))
    s.save('test.vtk')
    s2 = classes.Surface('test.vtk')
    assert np.allclose(s.points, s2.points)
    os.remove('test.vtk')

def test_adjacency():
    td = get_testdir()
    s = classes.Surface(op.join(td, 'in.surf.gii'))
    adj = s.adjacency_matrix()
    assert not (adj.data < 0).any(), 'negative value in adjacency matrix'

def test_mesh_laplacian():
    td = get_testdir()
    s = classes.Surface(op.join(td, 'in.surf.gii'))
    lap = s.mesh_laplacian()
    assert (lap[np.diag_indices(lap.shape[0])] < 0).min(), 'positive diagonal'

def test_lbo():
    td = get_testdir()
    s = classes.Surface(op.join(td, 'in.surf.gii'))
    for area in ['barycentric', 'voronoi', 'mayer']:
        lbo = s.laplace_beltrami(area)
        assert (lbo[np.diag_indices(lbo.shape[0])] < 0).min(), 'positive diag'

if __name__ == "__main__":
    test_lbo()
