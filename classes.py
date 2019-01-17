import itertools
import copy 
import os.path as op
import collections
import multiprocessing
import functools
import argparse

import numpy as np 
import nibabel
import vtki
from vtk.util import numpy_support as vtknp

from . import pvcore
from . import toblerone


STRUCTURES = ['L_Accu', 'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 
    'L_Thal', 'R_Accu', 'R_Amyg', 'R_Caud', 'R_Hipp', 'R_Pall', 'R_Puta', 
    'R_Thal', 'BrStem']

TISSUES = ['GM', 'WM', 'CSF']


class Structure(object):

    def __init__(self, name, surfpath, space='world', struct=None):
        self.name = name 
        self.surf = Surface(surfpath, space=space, struct=struct)


# Class to contain an images voxel grid, including dimensions and vox2world transforms
class ImageSpace(object):

    def __init__(self, path):

        if not op.isfile(path):
            raise RuntimeError("Image does not exist")

        img = nibabel.load(path)
        self.imgSize = img.header['dim'][1:4]
        self.voxSize = img.header['pixdim'][1:4]
        self.vox2world = img.affine
        self.world2vox = np.linalg.inv(self.vox2world)



    def supersample(self, factor):
        """Produce a new image space which is a copy of the current space, 
        supersampled by a factor of (a,b,c) in each dimension 

        Args:
            factor: tuple/list of length 3, ints in each image dimension
        
        Returns: 
            new image space
        """

        if not len(factor) == 3:
            raise RuntimeError("Factor must have length 3")

        newSpace = copy.deepcopy(self)

        newSpace.imgSize = self.imgSize * factor
        newSpace.voxSize = self.voxSize / factor
        for r in range(3):
            newSpace.vox2world[r, 0:3] = newSpace.vox2world[r, 0:3] / factor[r]

        # The direction vector travelled in incrementing each index 
        # within the original grid by 1 is: 
        ijkvec = np.sum(self.vox2world[0:3,0:3], axis=1)

        # Origin of the src voxel grid is at (NB 1 is dist between vox cents)
        orig = self.vox2world[0:3,3] - (0.5 * ijkvec)

        # Which means the new grid has centre coord of vox (0,0,0) at: 
        offset = orig + (0.5 * ijkvec / factor)
        newSpace.vox2world[0:3,3] = offset 

        # Check the bounds of the new voxel grid we have created
        svertices = np.array(list(itertools.product([-0.5, newSpace.imgSize[0] - 0.5], 
            [-0.5, newSpace.imgSize[1] - 0.5], [-0.5, newSpace.imgSize[2] - 0.5])))
        rvertices = np.array(list(itertools.product([-0.5, self.imgSize[0] - 0.5], 
            [-0.5, self.imgSize[1] - 0.5], [-0.5, self.imgSize[2] - 0.5])))
        rvertices = pvcore._affineTransformPoints(rvertices, self.vox2world)
        svertices = pvcore._affineTransformPoints(svertices, newSpace.vox2world)
        assert np.all(np.abs(rvertices - svertices) < 1e-6)

        return newSpace


    def saveImage(self, data, path):

        if not np.all(data.shape[0:3] == self.imgSize):
            raise RuntimeError("Data size does not match image size")

        img = nibabel.Nifti2Image(data, self.vox2world)
        nibabel.save(img, path)



class Hemisphere(object): 
    """Class to encapsulate the white and pial surfaces of a hemisphere.
    """

    def __init__(self, inpath, outpath, side):

        self.side = side 
        self.inSurf = Surface(inpath) 
        self.outSurf = Surface(outpath)
        self.PVs = None 
        return


    def surfs(self):
        """Iterator over the inner/outer surfaces"""
        return [self.inSurf, self.outSurf]



class Surface(object):
    """Class to contain a surface's points, triangles and normals matrix. 
    Normals are calculated upon initialisation, with the appropriate norm 
    direction flag. LUT and associations are intially set to None but will be 
    updated later on. Using associations data, a surface can be cast to patch
    for a particular voxel which reduces computational complexity by 
    discarding unncessary triangles.
    """

    def __init__(self, path, space='world', struct=None):

        if not op.exists(path):
            raise RuntimeError("File {} does not exist".format(path))

        surfExt = op.splitext(path)[-1]
        if surfExt == '.gii':
            gft = nibabel.load(path).darrays
            ps, ts = tuple(map(lambda o: o.data, gft))
        elif surfExt == '.vtk':
            obj = vtki.PolyData(path)
            err = RuntimeError("Surface cannot be cast to triangle data")
            if obj.faces.size % 4:
                raise err
            ts = obj.faces.reshape(-1, 4)
            if not np.all(ts[:,0] == 3):
                raise err
            ps = obj.points
            ts = ts[:,1:]
        else: 
            ps, ts, meta = nibabel.freesurfer.io.read_geometry(path, 
                read_metadata=True)
            ps += meta['cras']

        if ps.shape[1] != 3: 
            raise RuntimeError("Points matrices should be p x 3")

        if ts.shape[1] != 3: 
            raise RuntimeError("Triangles matrices should be t x 3")

        if (np.max(ts) != ps.shape[0]-1) or (np.min(ts) != 0):
            raise RuntimeError("Incorrect points/triangle indexing")

        if space == 'first':
            
            if struct is None: 
                raise RuntimeError("Path to structural image required with FIRST surfs")

            structSpace = ImageSpace(struct)
            if np.linalg.det(structSpace.vox2world) > 0:
                
                # Flip X coords from [0, n-1] to [n-1, 0] by adding a shift vector
                N = structSpace.imgSize[0]
                ps[:,0] = (N-1) - ps[:,0]

            # Convert from FSL scaled voxel mm to struct voxel coords
            # Then to world mm coords
            ps /= structSpace.voxSize
            ps = pvcore._affineTransformPoints(ps, structSpace.vox2world)

        self.points = ps.astype(np.float32)
        self.tris = ts.astype(np.int32)
        self.xProds = None 
        self.voxelised = None 



    def calculateXprods(self):
        self.xProds = np.cross(
            self.points[self.tris[:,2],:] - self.points[self.tris[:,0],:], 
            self.points[self.tris[:,1],:] - self.points[self.tris[:,0],:], 
            axis=1)


    def applyTransform(self, transform):

        self.points = (pvcore._affineTransformPoints(
            self.points, transform).astype(np.float32))


    def formAssociations(self, FoVsize, cores):
        """Identify which triangles of a surface each voxel. This reduces the
        number of tests that must be performed in the main Toblerone algorithm.

        Args: 
            points: p x 3 matrix of surface nodes
            tris: t x 3 matrix of triangle node indices
            FoV: 1 x 3 vector of image dimensions (units of voxels) reqiured to
                fully enclose the surface

        Returns: 
            (associations, LUT) tuple of associations (a list of lists) and 
                a LUT used to index between associations table and vox index. 
        """

        # Check for negative coordinates: these should have been sripped. 
        if np.round(np.min(self.points)) < 0: 
            raise RuntimeError("formAssociations: negative coordinate found")

        if np.any(np.round(np.max(self.points, axis=0)) >= FoVsize): 
            raise RuntimeError("formAssociations: coordinate outside FoV")

        chunks = toblerone._distributeObjects(np.arange(self.tris.shape[0]), cores)
        workerFunc = functools.partial(toblerone._formAssociationsWorker, 
            self.tris, self.points, FoVsize)

        if cores > 1:
            with multiprocessing.Pool(cores) as p:
                allResults = p.map(workerFunc, chunks, chunksize=1)
        else:
            allResults = list(map(workerFunc, chunks))

        # Flatten results down from each worker. Iterate only over the keys
        # present in each dict. Use a default dict of empty [] to hold results
        associations = collections.defaultdict(list)
        for res in allResults: 
            for k in res.keys():
                associations[k] += res[k]

        # Convert back to dict and assert no empty entries 
        dct = dict(associations)
        assert all(map(len, dct.values()))

        self.assocs = np.array(list(dct.values()), dtype=object) 
        self.LUT = np.array(list(dct.keys()), dtype=np.int32)
        assert len(self.LUT) == len(self.assocs)


    def rebaseTriangles(self, triNums):
        """Re-express a patch of a larger surface as a new points and triangle matrix pair, indexed from 0. Useful for reducing computational complexity when working with a small
        patch of a surface where only a few nodes in the points 
        array are required by the triangles matrix. 

        Args: 
            triNums: t x 1 list of triangle numbers to rebase. 
        
        Returns: 
            (localPoints, localTris) tuple of re-indexed points/tris. 
        """

        localPoints = np.empty((0, 3), dtype=np.float32)
        localTris = np.zeros((len(triNums), 3), dtype=np.int32)
        pointsLUT = []

        for t in range(len(triNums)):
            for v in range(3):

                # For each vertex of each tri, check if we
                # have already processed it in the LUT
                vtx = self.tris[triNums[t],v]
                idx = np.argwhere(pointsLUT == vtx)

                # If not in the LUT, then add it and record that
                # as the new position. Write the missing vertex
                # into the local points array
                if not idx.size:
                    pointsLUT.append(vtx)
                    idx = len(pointsLUT) - 1
                    localPoints = np.vstack([localPoints, 
                        self.points[vtx,:]])

                # Update the local triangle
                localTris[t,v] = idx

        return (localPoints, localTris)


    def toPatch(self, voxIdx):
        """Return a patch object specific to a voxel given by linear index.
        Look up the triangles intersecting the voxel, and then load and rebase
        the points / xprods as required. 
        """

        triNums = self.assocs[self.LUT == voxIdx][0]
        (ps, ts) = self.rebaseTriangles(triNums)

        return Patch(ps, ts, self.xProds[triNums,:])

    
    def toPatchesForVoxels(self, voxIndices):
        """Return the patches for the voxels in voxel indices, flattened into 
        a single set of ps, ts and xprods. 

        If no patches exist for this list of voxels return None.
        """

        # Concatenate all triangle numbers for the voxels, using the
        # set constructor to strip out repeats
        vlists = self.assocs[np.isin(self.LUT, voxIndices)]

        if vlists.size:
            triNums = []
            [ triNums.extend(l) for l in vlists ]
            triNums = np.unique(triNums)

            return Patch(self.points, self.tris[triNums,:], 
                self.xProds[triNums,:])

        return None


    def shiftFoV(self, offset, FoVsize):

        self.points += offset
        if np.any(np.floor(self.points.min(axis=0)) < 0):
            raise RuntimeError("FoV offset does not remove negative coordinates")
        if np.any(np.ceil(self.points.max(axis=0)) >= FoVsize):
            raise RuntimeError("Full FoV does not contain all surface coordinates")



class Patch(Surface):
    """Subclass of Surface that represents a small patch of surface. 
    Points, triangles and xProds are all inherited from the parent surface. 
    This class should not be directly created but instead instantiated via
    the Surface.toPatch() / toPatchesForVoxels() methods. 
    """

    def __init__(self, points, tris, xProds):
        self.points = points 
        self.tris = tris
        self.xProds = xProds 
        

    def shrink(self, fltr):
        """Return a shrunk copy of the patch by applying the logical 
        filter fltr to the calling objects tris and xprods matrices
        """

        return Patch(self.points, self.tris[fltr,:], self.xProds[fltr,:])



class CommonParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('-ref', type=str, required=True)
        self.add_argument('-struct2ref', type=str, required=True) 
        self.add_argument('-flirt', action='store_true')
        self.add_argument('-struct', type=str, required=False)
        self.add_argument('-outdir', type=str, required=False)
        self.add_argument('-name', type=str, required=False)
        self.add_argument('-cores', type=int, required=False)


    def parse(self, args):
        return vars(super().parse_args(args))
