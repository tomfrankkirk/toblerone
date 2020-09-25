"""Toblerone tests"""

import unittest
import os.path as op 
import nibabel 
import pickle 
import gzip
import numpy as np 
import multiprocessing
from pdb import set_trace
import os 

import toblerone
from toblerone import classes, projection, core 
from toblerone.pvestimation import estimators
from regtricks.application_helpers import sum_array_blocks

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
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = classes.Hemisphere(ins, outs, 'L')
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    s2r = np.identity(4)

    # superfactor = 10
    # spc_high = spc.resize_voxels(1.0/superfactor)
    # voxelised = np.zeros(spc_high.size.prod(), dtype=np.float32)
    # hemi.inSurf.index_on(spc_high, s2r)
    # reindex_in = hemi.inSurf.reindexing_filter(spc_high)
    # voxelised[reindex_in[1]] = -(hemi.inSurf.voxelised[reindex_in[0]]).astype(np.float32)
    # hemi.outSurf.index_on(spc_high, s2r)
    # reindex_out = hemi.outSurf.reindexing_filter(spc_high)
    # voxelised[reindex_out[1]] += hemi.outSurf.voxelised[reindex_out[0]]
    # voxelised = voxelised.reshape(spc_high.size)
    # voxelised = sum_array_blocks(voxelised, 3 * [superfactor]) / superfactor**3
    # spc.save_image(voxelised, f'{td}/voxelised.nii.gz')

    supersampler = 3 * [4]
    fracs = estimators._cortex(hemi, spc, s2r, supersampler, 
        8, False)

    spc.save_image(fracs, f'{td}/fracs.nii.gz')
    # truth = np.squeeze(nibabel.load(op.join(td, 'sph_fractions.nii.gz')).get_fdata())
    # np.testing.assert_array_almost_equal(fracs, truth, 1)


def test_projection():
    td = get_testdir()
    ins = op.join(td, 'in.surf.gii')
    outs = op.join(td, 'out.surf.gii')
    hemi = toblerone.Hemisphere(ins, outs, 'L')
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    sdata = np.ones(hemi.inSurf.points.shape[0], dtype=np.float32)
    vdata = np.ones(spc.size.prod(), dtype=np.float32)

    projector = toblerone.projection.Projector(hemi, spc, 10, 1)
    v2s = projector.vol2surf(vdata)
    v2s_edge = projector.vol2surf(vdata, True)
    assert (v2s <= v2s_edge).all(), "edge correction did not increase signal"
    s2v = projector.surf2vol(sdata)
    v2n = projector.vol2node(vdata)
    n2v = projector.node2vol(v2n)

def test_subvoxels():

    supersampler = np.random.randint(2, 10, 3)
    subvox_size = 1.0 / supersampler
    vox_cent = np.random.randint(-10, 10, 3)
    subvox_cents = core._get_subvoxel_grid(supersampler) + vox_cent
    assert (np.abs(subvox_cents.mean(0) - vox_cent) < 1e-6).all()
    for cidx, scent in enumerate(subvox_cents):
        corners = scent + ((core.SUBVOXCORNERS) * (subvox_size[None,:]))
        assert ((np.abs(corners - scent) - subvox_size/2) < 1e-6).all()

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
    test_subvoxels()
