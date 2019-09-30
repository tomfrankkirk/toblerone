"""Toblerone tests"""

import unittest
import os.path as op 
import nibabel 
import pickle 
import gzip
import toblerone
import numpy as np 
import multiprocessing

import toblerone.estimators

cores = multiprocessing

class Toblerone_Tests(unittest.TestCase):

    def test_indexing(self):
        surf = toblerone.Surface('sph.surf.gii')
        spc = toblerone.ImageSpace('sph_fractions.nii.gz')
        surf.index_on(spc, np.identity(4))

        truth = pickle.load(gzip.open('sph_indexed.gz'))

        self.assertDictEqual(surf.assocs, truth)

    # def test_sph(self):
    #     surf = toblerone.Surface('sph.surf.gii')
    #     spc = toblerone.ImageSpace('sph_fractions.nii.gz')
    #     s2r = np.identity(4)
    #     supersampler = np.ceil(spc.vox_size / 0.75)

    #     fracs = toblerone.estimators._structure(surf, spc, 
    #         s2r, supersampler, False, multiprocessing.cpu_count())

    #     truth = nibabel.load('sph_fractions.nii.gz').get_fdata()
    #     self.assertTrue(np.testing.assert_array_almost_equal(fracs, truth, 1))

