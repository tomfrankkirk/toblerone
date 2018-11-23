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
    ref = 'testdata/FS/aslref.nii'
        
    s2r = np.identity(4)
    t.estimatePVs(ref=ref, FSdir='testdata/FS',
        struct2ref=s2r, saveassocs=True)

if __name__ == '__main__':
    doToblerone()