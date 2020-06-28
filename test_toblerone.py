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
from toblerone import classes, projection
from toblerone.pvestimation import estimators

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
    supersampler = [2,2,2]

    fracs = estimators._cortex(hemi, spc, s2r, supersampler, 
        1, False)

    spc.save_image(fracs, 'fracs.nii.gz')
    truth = np.squeeze(nibabel.load(op.join(td, 'sph_fractions.nii.gz')).get_fdata())
    np.testing.assert_array_almost_equal(fracs, truth, 1)

def test_imagespace():
    td = get_testdir()
    spc = toblerone.ImageSpace(op.join(td,'sph_fractions.nii.gz'))
    sspc = spc.resize_voxels(0.5)

    assert np.all(spc.bbox_origin == sspc.bbox_origin)
    assert np.all(spc.FoV_size == sspc.FoV_size)

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


def test_convert(): 
    td = get_testdir()
    s = classes.Surface(op.join(td, 'in.surf.gii'))
    s.save('test.vtk')
    s2 = classes.Surface('test.vtk')
    assert np.allclose(s.points, s2.points)
    os.remove('test.vtk')

if __name__ == "__main__":
    test_convert()
