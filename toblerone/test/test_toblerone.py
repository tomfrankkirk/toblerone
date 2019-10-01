"""Toblerone tests"""

import unittest
import os.path as op 
import nibabel 
import pickle 
import gzip
import numpy as np 
import multiprocessing

import toblerone
from toblerone import estimators

cores = multiprocessing.cpu_count()

def get_testdir():
    pwd = op.realpath(__file__)
    return op.abspath(op.join(pwd, '../../../testdata'))

def test_indexing():
    td = get_testdir()
    surf = toblerone.Surface(op.join(td, 'sph.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'sph_fractions.nii.gz'))
    surf.index_on(spc, np.identity(4))
    truth = pickle.load(gzip.open(op.join(td, 'sph_indexed.gz')))
    assert surf.assocs == truth

def test_sph():
    td = get_testdir()
    surf = toblerone.Surface(op.join(td, 'sph.surf.gii'))
    spc = toblerone.ImageSpace(op.join(td, 'sph_fractions.nii.gz'))
    s2r = np.identity(4)
    supersampler = np.ceil(spc.vox_size / 0.75)

    fracs = estimators._structure(surf, spc, 
        s2r, supersampler, False, multiprocessing.cpu_count())

    truth = np.squeeze(nibabel.load(op.join('sph_fractions.nii.gz')).get_fdata())
    assert np.testing.assert_array_almost_equal(fracs, truth, 1)

def test_struct():
    td = get_testdir()
    fracs = toblerone.estimate_structure(surf=op.join(td, 'L_Puta.surf.gii'), 
        ref=op.join(td, 'sph_fractions.nii.gz'), struct2ref='I')
    
    truth = np.squeeze(nibabel.load(
        op.join(td, 'L_Puta_fracs.nii.gz')).get_fdata())
    assert np.testing.assert_array_almost_equal(fracs, truth, 1)
