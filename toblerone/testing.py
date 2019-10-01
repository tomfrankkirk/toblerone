"""Toblerone tests"""

import unittest
import os.path as op 
import nibabel 
import pickle 
import gzip
import numpy as np 
import multiprocessing

from . import estimators
from .classes import ImageSpace, Surface
from .main import estimate_structure

cores = multiprocessing

class Toblerone_Tests(unittest.TestCase):

    def test_indexing(self):
        surf = Surface('sph.surf.gii')
        spc = ImageSpace('sph_fractions.nii.gz')
        surf.index_on(spc, np.identity(4))

        truth = pickle.load(gzip.open('sph_indexed.gz'))

        self.assertDictEqual(surf.assocs, truth)

    def test_sph(self):
        surf = Surface('sph.surf.gii')
        spc = ImageSpace('sph_fractions.nii.gz')
        s2r = np.identity(4)
        supersampler = np.ceil(spc.vox_size / 0.75)

        fracs = estimators._structure(surf, spc, 
            s2r, supersampler, False, multiprocessing.cpu_count())

        truth = np.squeeze(nibabel.load('sph_fractions.nii.gz').get_fdata())
        np.testing.assert_array_almost_equal(fracs, truth, 1)
        self.assertTrue(True)

    def test_struct(self):
        fracs = estimate_structure(surf='L_Puta.surf.gii', 
            ref='sph_fractions.nii.gz', struct2ref='I')
        
        truth = np.squeeze(nibabel.load('L_Puta_fracs.nii.gz').get_fdata())
        np.testing.assert_array_almost_equal(fracs, truth, 1)
        self.assertTrue(True)


def run_tests(): 

    print("Running Toblerone tests")
    testdir = op.realpath(__file__)
    testsuite = unittest.TestLoader().discover(testdir)
    unittest.TextTestRunner(verbosity=1).run(testsuite)

