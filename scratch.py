import numpy as np 
import multiprocessing
import collections 
import random 
import itertools
import functools
import toblerone as t 
import pickle
import sys 
import threading
import time 

import nibabel
import nibabel.freesurfer.io
import nibabel.nifti2


surf = '../FSonFSLData/FSoutput/surf/lh.pial'
ps, ts = tuple(nibabel.freesurfer.io.read_geometry(surf))
assocsPth = 'testdata/py_test_tob_1.0_assocs.pkl'

with open(assocsPth, 'rb') as f: 
    oldStamp, inAssocs, outAssocs = pickle.load(f)


assocs = inAssocs['L']
inLUT = list(inAssocs['L'].keys())
inAssocs = np.array(list(inAssocs['L'].values()))
# outLUT[h] = list(outAssocs[h].keys())
# outAssocs[h] = np.array(list(outAssocs[h].values()))

ps = ps.astype(np.float32)
tris = ts.astype(np.int32)

ts = tris[0:300,:]
xps = np.cross(ps[ts[:,2],:] - ps[ts[:,0],:], 
    ps[ts[:,1],:] - ps[ts[:,0],:], axis=1)

s = t.Patch(ps, ts, xps)


imgSize = np.array([64,64,28], dtype=np.int16)
pnt = np.random.rand(3).astype(np.float32)
ray = np.array([1,0,0], dtype=np.float32)

vijk = np.array([30, 30, 20], dtype=np.int32)
vC = np.array([10, 11, 12], dtype=np.float32)
vS = np.array([0.3, 0.4, 1], dtype=np.float32)


def tv1(): 
    x = t._pfilterTriangles(ts, ps, vC, vS)

def tv2(): 
    x = t._cfilterTriangles(ts, ps, vC, vS)

def tv3(): 
    x = t._cyfilterTriangles(ts, ps, vC, vS)


# Pure python method, find intersection of ray with many tris
def rt1():
    x = t._pTestManyRayTriangleIntersections(ts, ps, pnt, 0, 1)

# Ctyhon method: construct loop over tris in cython, then test each one
# using the C method 
def rt2():
    x = t._cytestManyRayTriangleIntersections(ts, ps, pnt, 0, 1)

# C method: construct loop and perform test in C only. 
def rt3():
    x = t._ctestManyRayTriangleIntersections(ts, ps, pnt, 0, 1)

# Pure python method. 
def rtp1():
    x = t._findRayTriPlaneIntersections(ps[ts[:,0]], 
        xps, pnt, ray)

def find3D1(): 
    x = t._findRayTriangleIntersections3D(pnt, ray, s)

def find3D2(): 
    x = t._cyfindRayTriangleIntersections3D(pnt, ray, s)


testpnts = 5 * np.random.rand(100,3).astype(np.float32)
def red(): 
    return t.redTest(testpnts, ps, ts, xps, pnt, False)