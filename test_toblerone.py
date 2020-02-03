"""Toblerone tests"""

import unittest
import os.path as op 
import nibabel 
import pickle 
import gzip
import numpy as np 
import multiprocessing
from pdb import set_trace

import toblerone
from toblerone import estimators, classes, projection

cores = multiprocessing.cpu_count()

def get_testdir():
    return op.join(op.dirname(op.realpath(__file__)), 'testdata')

def test_indexing():
    td = get_testdir()
    surf = toblerone.Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    surf.index_on(spc, np.identity(4))
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
    supersampler = np.ceil(spc.vox_size / 0.75)

    fracs, _ = estimators._cortex(hemi, spc, s2r, supersampler, 
        multiprocessing.cpu_count(), False)

    truth = np.squeeze(nibabel.load(op.join(td, 'sph_fractions.nii.gz')).get_fdata())
    np.testing.assert_array_almost_equal(fracs, truth, 1)

def test_imagespace():
    td = get_testdir()
    spc = toblerone.ImageSpace(op.join(td,'sph_fractions.nii.gz'))
    sspc = spc.supersample([2,2,2])

    assert np.all(spc.bbox_origin == sspc.bbox_origin)
    assert np.all(spc.FoV_size == sspc.FoV_size)

def test_projection():
    td = get_testdir()
    ins = toblerone.Surface(op.join(td, 'in.surf.gii'))
    outs = toblerone.Surface(op.join(td, 'out.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
    sdata = np.ones(ins.points.shape[0], dtype=np.float32)
    vdata = np.ones(spc.size.prod(), dtype=np.float32)
    sproj = projection.surf2vol_weights(ins, outs, spc, 5, 1)
    # assert (np.abs(1 - vdata[vdata > 0]) < 1e-6).all(), 'surf did not map to ones'


if __name__ == "__main__":
    test_projection()