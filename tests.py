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

    def test_triangleNormal(self):
        n = t._triangleNormal(ps[ts[0, :], :], NDF)
        self.assertTrue(np.array_equal(np.array([0, 0, -1]), n))
        n = t._triangleNormal(2 * ps[ts[0, :], :], NDF)
        self.assertTrue(np.array_equal(np.array([0, 0, -1]), n))
        n = t._triangleNormal(ps[ts[0, :], :], -NDF)
        self.assertTrue(np.array_equal(np.array([0, 0, 1]), n))

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

    def test_rebaseTriangles(self):
        (lPs, lTs) = t._rebaseTriangles(ps, ts[5:8,:])
        for k in range(3):
            self.assertTrue(
                np.array_equal(ps[ts[k+5,:],:], lPs[lTs[k,:],:])
            )

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

    def test_rayTriangleIntersection(self):
        for _ in range(100):
            start = np.random.rand(3)
            fltr = t._ctestRayTriangleIntersection(ps[ts[0,:],:], start, 0, 1)

    def test_RayTriangleIntersections2D(self):
        for _ in range(100):
            start = np.random.rand(3).astype(np.float32)
            fltr = t._ctestManyRayTriangleIntersections(ts, ps, start, 0, 1)
            fltr2 = t._vectorTestRayTriangleIntersection(ts, ps, start, 0, 1)
            self.assertTrue(np.array_equal(fltr, fltr2))
            self.assertTrue(np.sum(fltr) == 1)

    def test_separatePointClouds(self):
        self.assertTrue(len(t._separatePointClouds(ts)) == 1)
        tris = np.vstack((ts[0,:], ts[7,:]))
        self.assertTrue(len(t._separatePointClouds(tris)) == 2)
        tris = np.vstack((ts[0,:], ts[5,:], ts[2,:]))
        self.assertTrue(len(t._separatePointClouds(tris)) == 2)
        self.assertTrue(len(t._separatePointClouds(ts[1:7])) == 1)

    def test_findRayTriangleIntersections3D(self):
        for _ in range(100):
            start = 1 + 2*(np.random.rand(3) - 0.5)
            start[2] = 1 
            end = 1 + 2*(np.random.rand(3) - 0.5)
            end[2] = -1
            res = t._findRayTriangleIntersections3D(start, end - start, \
                ts, ps, NDF)
            self.assertTrue(res.shape == (1,))

    def test_filterTriangles(self):
        vC = np.zeros(3)
        vS = 0.25 * np.ones(3)
        res = t._cfilterTriangles(ts, ps, vC, vS)
        self.assertTrue(np.sum(res) == 1)
        vC = np.array([-1,-1,-1], dtype=np.float32)
        res = t._cfilterTriangles(ts, ps, vC, vS)
        self.assertTrue(np.sum(res) == 0)

    def test_findRayTriPlaneIntersections(self):
        start = np.array([0, 0, -1])
        ray = np.ones(3)
        res = t._findRayTriPlaneIntersections(ps, ts, start, ray, NDF)
        self.assertTrue(res.size == 8)


    def test_formAssociations(self): 
        imgPath = 'testdata/ref1.0.nii'
        surfPath = 'testdata/L.white.surf.gii'
        points, tris = tuple(map(lambda o: o.data, \
                    nibabel.load(surfPath).darrays))
        imgStruct = nibabel.load(imgPath)
        FoVsize = imgStruct.header['dim'][1:4]
        vox2world = imgStruct.affine
        points = t._affineTransformPoints(points, np.linalg.inv(vox2world))
        mat = spio.loadmat('testdata/massocs.mat', squeeze_me=True, struct_as_record=True)
        mAssocsRaw = mat['LinAssocs']
        massocs = []
        for e in list(mAssocsRaw):
            if type(e) is int: 
                massocs.append([e])
            else: 
                massocs.append(e.tolist())
        mLUT = mat['LinLUT'] - 1
        mLUT = np.array(np.unravel_index(mLUT, FoVsize, order='F'))

        # Finally assert the overall results are equal
        pAssocs = t._formAssociations(points, tris, FoVsize)
        # with open('testdata/passocs.pkl', 'rb') as f: 
        #     pLUT, pAssocs = pickle.load(f)
        pLUT = list(pAssocs.keys())
        pAssocs = list(pAssocs.values())
        mLUTc = np.ravel_multi_index(mLUT, FoVsize, order='C')
        assert np.all(np.isin(pLUT, mLUTc)) & (len(pLUT) == mLUTc.size)
        for i in range(99910, mLUTc.size): 
            pidx = np.argwhere(pLUT == mLUTc[i])[0][0]
            masc = np.array(massocs[i]) -1 
            pasc = pAssocs[pidx]
            if not (len(masc) == len(pasc)) & (np.all(np.isin(masc, pasc))):
                vijk = np.unravel_index(pLUT[pidx], FoVsize)
                diff = set(masc) - set(pasc)
                for tr in diff: 
                    t._ctestTriangleVoxelIntersection(vijk, np.ones(3), points[tris[tr,:],:])


    def test_toblerone(self):
        if False:
            ref = 'E:/HCP100/references/reference1.0.nii'
            LWS = 'E:/HCP100/103818/T1w/Native/103818.L.white.native.surf.gii'
            LPS = 'E:/HCP100/103818/T1w/Native/103818.L.pial.native.surf.gii'
            RWS = 'E:/HCP100/103818/T1w/Native/103818.R.white.native.surf.gii'
            RPS = 'E:/HCP100/103818/T1w/Native/103818.R.pial.native.surf.gii'

        ref = 'testdata/perfusionNative1.nii'
        FSDir = 'testdata'
            
        s2r = np.identity(4)
        outDir = 'testdata'
        t.toblerone(reference=ref, FSSubDir=FSDir, \
            struct2ref=s2r, outDir=outDir, \
            saveAssocs=True)


if __name__ == '__main__':
    unittest.main()