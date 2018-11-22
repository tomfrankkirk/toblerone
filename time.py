import unittest
import toblerone as t
import numpy as np 
import os
import os.path as op 
import sys 
import nibabel
import scipy.io as spio 
import pickle
import cProfile
import re 

def doToblerone(): 
    if False:
        ref = 'E:/HCP100/references/reference1.0.nii'
        LWS = 'E:/HCP100/103818/T1w/Native/103818.L.white.native.surf.gii'
        LPS = 'E:/HCP100/103818/T1w/Native/103818.L.pial.native.surf.gii'
        RWS = 'E:/HCP100/103818/T1w/Native/103818.R.white.native.surf.gii'
        RPS = 'E:/HCP100/103818/T1w/Native/103818.R.pial.native.surf.gii'

    else:
        # ref = '/Users/tom/Data/HCPExampleData/103818/T1w/Processed/ref1.0.nii'
        # LWS = '/Users/tom/Data/HCPExampleData/103818/T1w/Natpyive/103818.L.white.native.surf.gii'
        # LPS = '/Users/tom/Data/HCPExampleData/103818/T1w/Native/103818.L.pial.native.surf.gii'
        pass

    ref = 'testdata/perfusionNative1.nii'
    s2r = np.identity(4)
    outDir = 'testdata'
    outName = 'py_test_tob_1.0'
    t.estimatePVs(ref=ref, FSdir='testdata',
        struct2ref=s2r, outdir=outDir, name=outName, \
        saveassocs=True)


if __name__ == '__main__':
    doToblerone()