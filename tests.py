import unittest
import toblerone as t
import numpy as np 
import os
import os.path as op 
import sys 
import nibabel
import scipy.io as spio 
import pickle

vC = np.array([0, 0, 0], dtype=np.float32)
vS = np.array([2, 2, 2], dtype=np.float32)
NDF = 1

ps = np.array([
  [0, 0, 0],
  [1, 0, 0],
  [2, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [2, 1, 0],
  [0, 2, 0],
  [1, 2, 0],
  [2, 2, 0]], dtype=np.float32) 

ts = np.array([
  [0, 1, 3],
  [3, 4, 1],
  [1, 4, 2],
  [2, 4, 5],
  [6, 3, 4],
  [6, 4, 7],
  [7, 4, 5],
  [7, 5, 8]], dtype=np.int32) 

class Test_Toblerone(unittest.TestCase):

    def test_affineTransformPoints(self):
        a = np.zeros((4,4))
        a[tuple(range(4)), tuple(range(4))] = 1
        self.assertTrue(
            np.array_equal(t._affineTransformPoints(ps, a), ps))
        a[tuple(range(3)), tuple(range(3))] = 2
        self.assertTrue(
            np.array_equal(t._affineTransformPoints(ps, a), 2*ps))


    def test_filterPoints(self):
        points = np.array([
            [1, 1, 1],
            [1, 3, 1],
            [0, 0, 0],
            [-1, -1, -1], 
            [0, -1, 2]
        ])
        fltr = t._filterPoints(points, vC, vS)
        FLTR = np.array([1, 0, 1, 1, 0], dtype=bool)
        self.assertTrue(np.array_equal(fltr, FLTR))
    
    def test_getVoxelBounds(self):
        bounds = t._getVoxelBounds(vC, vS)
        BOUNDS = np.array([[-1.0, 1], [-1, 1], [-1, 1]])
        self.assertTrue(np.array_equal(bounds, BOUNDS))

    def test_getVoxelCorners(self):
        crns = t._generateVoxelCorners(vC, vS)
        CRNS = np.array([
            [-1, -1, -1,], 
            [1, -1, -1], 
            [-1, 1, -1], 
            [1, 1, -1], 
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1]
        ])
        self.assertTrue(np.array_equal(crns, CRNS))

    def test_dotVectorAndMatrix(self):
        v = np.array([0, 1, 0])
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        dp = t._dotVectorAndMatrix(v, m)
        DP = np.array([0, 1, 0])
        self.assertTrue(np.array_equal(DP, dp))

    def test_sub2ind(self):
        dims = (3,3)
        subs = ([0,2], [0,2])
        ind = t._sub2ind(dims, subs)
        self.assertTrue(np.all(ind == [0, 8]))

    def test_ind2sub(self):
        dims = (3,3)
        inds = (0,8)
        subs = t._ind2sub(dims, inds)
        self.assertTrue(np.all(subs[0] == [0,2]), np.all(subs[1] == [0,2]))

    def test_getVoxelSubscripts(self):
        dims = (4,4,4)
        inds = [0, 3, 63]
        subs = t._getVoxelSubscripts(inds, dims)
        self.assertTrue(np.array_equal(
            subs, np.array([[0,0,0], [0,0,3], [3,3,3]])
        ))

    def test_triangleVoxelIntersection(self):
        t1 = np.array([
            [-1.5, -1.5, -1.5], 
            [1.5, 1.5, 1.5],
            [-1.5, 1.5, 1.5]
        ])

        t2 = 4 + np.array([
            [-1.5, -1.5, -1.5], 
            [1.5, 1.5, 1.5],
            [-1.5, 1.5, 1.5]
        ])

        t3 = np.array([
            [0, -1, 0], 
            [1.5, 1.5, 1.5],
            [-2.5, 1.5, 1.5]
        ])

        for k in range(3): 
            self.assertTrue(t._ctestTriangleVoxelIntersection(vC, vS, \
                t1[[k-2,k-1,k],:]))

        for k in range(3): 
            self.assertFalse(t._ctestTriangleVoxelIntersection(vC, vS, \
                t2[[k-2,k-1,k],:]))

        for k in range(3): 
            self.assertTrue(t._ctestTriangleVoxelIntersection(vC, vS, \
                t3[[k-2,k-1,k],:]))


    def test_separatePointClouds(self):
        self.assertTrue(len(t._separatePointClouds(ts)) == 1)
        tris = np.vstack((ts[0,:], ts[7,:]))
        self.assertTrue(len(t._separatePointClouds(tris)) == 2)
        tris = np.vstack((ts[0,:], ts[5,:], ts[2,:]))
        self.assertTrue(len(t._separatePointClouds(tris)) == 2)
        self.assertTrue(len(t._separatePointClouds(ts[1:7])) == 1)

    

    def test_filterTriangles(self):
        vC = np.zeros(3, dtype=np.float32)
        vS = 0.25 * np.ones(3, dtype=np.float32)
        res = t._cyfilterTriangles(ts, ps, vC, vS)
        self.assertTrue(np.sum(res) == 1)
        vC = np.array([-1,-1,-1], dtype=np.float32)
        res = t._cyfilterTriangles(ts, ps, vC, vS)
        self.assertTrue(np.sum(res) == 0)



    def test_estimateFractions(self):

        refimg = nibabel.load('testdata/infractions.nii')
        mfracs = np.squeeze(refimg.get_fdata())
        imgSize = mfracs.shape

        ps, ts = nibabel.freesurfer.io.read_geometry('testdata/lh.white')
        world2vox = np.linalg.inv(refimg.affine)
        ps = t._affineTransformPoints(ps, world2vox).astype(np.float32)
        ts = ts.astype(np.int32)

        surf = t.Surface(ps, ts)
        supersampler = np.array([2,2,2])

        # Form the associations for this 
        assocspath = 'testdata/sphassocs.pkl'
        if op.isfile(assocspath):
            with open(assocspath, 'rb') as f:
                assocs = pickle.load(f)

        else: 
            assocs = t._formAssociations(surf, mfracs.shape)
            with open(assocspath, 'wb') as f:
                pickle.dump(assocs, f)

        surf.LUT = np.array(list(assocs.keys()), dtype=np.int32)
        surf.assocs = np.array(list(assocs.values()))
        
        for v in surf.LUT:

            ijk = np.array(t._ind2sub(imgSize, v))
            sln = mfracs[ijk[0], ijk[1], ijk[2]]
            f = t._estimateVoxelFraction(surf, ijk.astype(np.float32), 
                v, imgSize, supersampler)
            if (np.abs(sln - f) > 0.01):
                t._estimateVoxelFraction(surf, ijk.astype(np.float32), 
                    v, imgSize, supersampler)
                

    def test_toblerone(self):
        if False:
            ref = 'E:/HCP100/references/reference1.0.nii'
            LWS = 'E:/HCP100/103818/T1w/Native/103818.L.white.native.surf.gii'
            LPS = 'E:/HCP100/103818/T1w/Native/103818.L.pial.native.surf.gii'
            RWS = 'E:/HCP100/103818/T1w/Native/103818.R.white.native.surf.gii'
            RPS = 'E:/HCP100/103818/T1w/Native/103818.R.pial.native.surf.gii'

        ref = 'testdata/FS/aslref.nii'
            
        s2r = np.identity(4)
        t.estimatePVs(ref=ref, FSdir='testdata/FS',
            struct2ref=s2r, saveassocs=True)


if __name__ == '__main__':
    unittest.main()