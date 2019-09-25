# Class definitions for the pvtools module, as follows: 
# 
# ImageSpace: image matrix, inc dimensions, voxel size, vox2world matrix and 
#     inverse, of an image. Used for resampling operations between different 
#     spaces and also for saving images into said space (eg, save PV estimates 
#     into the space of an image)
# Surface: the points and triangles of a surface, and various calculated
#     properties that are evaluated ahead of time to speed up later operations
# Hemisphere: a pair of surfaces, used specifically to represent one half 
#     of the cerebral cortex (referred to as inner and outer surfaces)
# Patch: a subcalss of Surface, representing a smaller portion of a surface,
#     used to reduce computational complexity of operations 
# CommonParser: a subclass of the library ArgumentParser object pre-configured
#     to parse arguments that are common to many pvtools functions 
    
import argparse
import itertools
import copy 
import os.path as op
import collections
import multiprocessing
import functools
import warnings
import operator

import numpy as np 
import nibabel
try:
    import vtki
except ImportError:
    import pyvista as vtki
from vtk.util import numpy_support as vtknp

from . import utils, core 

TISSUES = ['GM', 'WM', 'CSF']


class ImageSpace(object):
    """The voxel grid of an image. Attributes:  
    -size (dimensions)
    -vox_size (voxel size)
    -vox2world (voxel to world transformation)
    -world2voxel (inverse)

    Initialise by specifying a path to the image. 

    Two methods are provided: supersample (produce a super-resolution
    copy of the current space) and saveImage (save an image array into 
    the space represented by the calling object)
    """

    def __init__(self, path):

        if not op.isfile(path):
            raise RuntimeError("Image %s does not exist" % path)

        img = nibabel.load(path)
        self.size = img.header['dim'][1:4]
        self.vox_size = img.header['pixdim'][1:4]
        self.vox2world = img.affine
        self.world2vox = np.linalg.inv(self.vox2world)
        self.offset = None 


    @classmethod
    def save_like(cls, ref, data, path): 
        """Save data into the space of an existing image

        Args: 
            ref: path to image defining space to use 
            data: ndarray (of appropriate dimensions)
            path: path to write to 
        """
        
        spc = ImageSpace(ref)
        spc.saveImage(data, path)


    @property
    def FoVsize(self):
        return self.size * self.vox_size


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

        newSpace.size = (self.size * factor).round()
        newSpace.vox_size = self.vox_size / factor
        for r in range(3):
            newSpace.vox2world[0:3,r] /= factor[r]

        # Start at the vox centre of [0 0 0] on the original grid
        # Move in by 0.5 voxels along the diagonal direction vector of the voxel
        # Then move back out by 0.5 voxels of the NEW direction vector to get
        # the vox cent for [0 0 0] within the new grid
        orig = (self.vox2world[0:3,3] 
            - 0.5 * np.sum(self.vox2world[0:3,0:3], axis=1))
        new = orig + 0.5 * np.sum(newSpace.vox2world[0:3,0:3], axis=1)
        newSpace.vox2world[0:3,3] = new 

        # Check the bounds of the new voxel grid we have created
        svertices = np.array(list(itertools.product([-0.5, newSpace.size[0] - 0.5], 
            [-0.5, newSpace.size[1] - 0.5], [-0.5, newSpace.size[2] - 0.5])))
        rvertices = np.array(list(itertools.product([-0.5, self.size[0] - 0.5], 
            [-0.5, self.size[1] - 0.5], [-0.5, self.size[2] - 0.5])))
        rvertices = utils._affineTransformPoints(rvertices, self.vox2world)
        svertices = utils._affineTransformPoints(svertices, newSpace.vox2world)
        assert np.all(np.abs(rvertices - svertices) < 1e-6)

        return newSpace


    def saveImage(self, data, path):

        if not np.all(data.shape[0:3] == self.size):
            raise RuntimeError("Data size does not match image size")

        if data.dtype is np.dtype('bool'):
            data = data.astype(np.int8)

        if not (path.endswith('.nii') or path.endswith('.nii.gz')):
            path += '.nii.gz'

        nii = nibabel.nifti2.Nifti2Image(data, self.vox2world)
        nibabel.save(nii, path)

    @classmethod
    def minimal_enclosing(cls, surfs, reference, affine):
        """
        Return the minimal space required to enclose a set of surfaces. 
        This space will be based upon the reference, sharing its voxel 
        size and i,j,k unit vectors from the voxel2world matrix, but 
        will may have a different FoV. The offset of the voxel coord system
        relative to reference will be stored as the space.offset attribute

        Args: 
            surfs: singular or list of surface objects 
            reference: ImageSpace object or path to image to use 
            affine: 4x4 np.array, transformation INTO the reference space, 
                in world-world mm terms (ie, not a FLIRT scaled-voxel 
                matrix). See utils._FLIRT_to_world() for help. Pass None 
                to represent identity 

        Returns: 
            ImageSpace object, with a shifted origin and potentially different
                FoV relative to the reference. Subtract offset from coords in 
                this space to return them to original reference coords. 
        """

        if type(surfs) is not list: 
            slist = [surfs]
        else: 
            slist = surfs 

        if type(reference) is not ImageSpace: 
            space = ImageSpace(reference)
        else: 
            space = copy.deepcopy(reference)

        if affine is not None: 
            overall = space.world2vox @ affine 
        else: 
            overall = space.world2vox

        # Extract min and max vox coords in the reference space 
        min_max = np.zeros((2*len(slist), 3))
        for sidx,s in enumerate(slist):
            ps = utils._affineTransformPoints(s.points, overall)
            min_max[sidx*2,:] = ps.min(0)
            min_max[sidx*2 + 1,:] = ps.max(0)

        # Fix the offset relative to reference and minimal size 
        minFoV = min_max.min(0).round().astype(np.int16)
        maxFoV = min_max.max(0).round().astype(np.int16)
        size = maxFoV - minFoV + 1
        FoVoffset = -minFoV
    
        # Get a copy of the corresponding mm coords for checking later
        min_max_mm = utils._affineTransformPoints(np.array([minFoV, maxFoV]),
            space.vox2world)

        # Calculate new origin for the coordinate system and modify the 
        # vox2world matrix accordingly 
        space.size = size 
        space.vox2world[0:3,3] = min_max_mm[0,:]
        space.world2vox = np.linalg.inv(space.vox2world)
        space.offset = FoVoffset 

        check = utils._affineTransformPoints(min_max_mm, space.world2vox)
        if (np.any(check[0,:].round() < 0) or 
            np.any(check[1,:].round() > size - 1)): 
            raise RuntimeError("New space does not enclose surfaces")

        return space 

    def derives_from(self, parent):
        det1 = np.linalg.det(parent.vox2world[0:3,0:3])
        det2 = np.linalg.det(self.vox2world[0:3,0:3])
        v1 = parent.vox_size
        v2 = self.vox_size 
        offset = (self.offset is not None) 
        return all([ 
            np.abs(det1 - det2) < 1e-9, 
            np.all(np.abs(v1 - v2) < 1e-6), 
            offset ])

class Hemisphere(object): 
    """The white and pial surfaces of a hemisphere, and a repository to 
    store data when calculating tissue PVs from the fractions of each
    surface

    Args: 
        inpath: path to white surface
        outpath: path to pial surface 
        side: 'L' or 'R' 
    """

    def __init__(self, inpath, outpath, side):

        self.side = side 
        self.inSurf = Surface(inpath, name=side+'WS') 
        self.outSurf = Surface(outpath, name=side+'PS')
        self.PVs = None 
        return


    def surfs(self):
        """Iterator over the inner/outer surfaces"""
        return [self.inSurf, self.outSurf]

    def surf_dict(self):
        """Return surfs as dict with appropriate keys (eg LPS)"""
        return {self.side + 'WS': self.inSurf, 
            self.side+'PS': self.outSurf}

@utils.cascade_attributes
def ensure_derived_space(func):
    def ensured(self, *args):
        if not self.index_space.derives_from(args[0]):
            raise RuntimeError(
                "Target space is not derived from surface's current index space."+
                "Call surface.index_based_on with the target space first")
        return func(self, *args)
    return ensured 

class Surface(object):
    """Encapsulates a surface's points, triangles and associations data.
    Create either by passing a file path (as below) or use the static class 
    method Surface.manual() to directly pass points and triangles.
    
    Args: 
        path:   path to file (.gii/.vtk/.white/.pial)
        space:  'world' (default) or 'first'; coordinate system of surface
        struct: if in 'first' space, then path to structural image used by FIRST
    """

    def __init__(self, path, space='world', struct=None, name=None):

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
            if not 'cras' in meta:
                print('Warning: Could not load C_ras from surface', path)
                print('If true C_ras is non-zero then estimates will be inaccurate')
            else:
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
                N = structSpace.size[0]
                ps[:,0] = (N-1) - ps[:,0]

            # Convert from FSL scaled voxel mm to struct voxel coords
            # Then to world mm coords
            ps /= structSpace.vox_size
            ps = utils._affineTransformPoints(ps, structSpace.vox2world)

        self.points = ps.astype(np.float32)
        self.tris = ts.astype(np.int32)
        self.xProds = None 
        self.voxelised = None 
        self.name = name
        self.assocs = None 
        self.LUT = None  
        self.index_space = None 


    @classmethod
    def manual(cls, ps, ts, name=None):
        """Manual surface constructor using points and triangles arrays"""

        if (ps.shape[1] != 3) or (ts.shape[1] != 3):
            raise RuntimeError("ps, ts arrays must have N x 3 dimensions")

        s = cls.__new__(cls)
        s.points = ps.astype(np.float32)
        s.tris = ts.astype(np.int32)
        s.xProds = None 
        s.voxelised = None 
        s.name = name
        s.index_space = None 
        return s


    def save(self, path):
        
        if self.name is None: 
            warnings.warn("Surface has no name: will save as type 'Other'")
            self.name = 'Other'

        common = {'Description': 'Surface has been transformed into' + \
            'a reference image space for PV estimation'}

        m0 = {
            'GeometricType': 'Anatomical'
        }

        if self.name in ['LWS', 'LPS', 'RWS', 'RPS']:
            cortexdict = {
                side+surf+'S': {
                    'AnatomicalStructurePrimary': 
                        'CortexLeft' if side == 'L' else 'CortexRight', 
                    'AnatomicalStructureSecondary':
                        'GrayWhite' if surf == 'W' else 'Pial'
                }
                for (side,surf) in itertools.product(['L', 'R'], ['P', 'W'])
            }
            m0.update(cortexdict[self.name])   
        
        else:
            m0.update({'AnatomicalStructurePrimary': self.name})
        
        # Points matrix
        # 1 corresponds to NIFTI_XFORM_SCANNER_ANAT
        m0.update(common)
        ps = nibabel.gifti.GiftiDataArray(self.points, 
            intent='NIFTI_INTENT_POINTSET', 
            coordsys=nibabel.gifti.GiftiCoordSystem(1,1), 
            datatype='NIFTI_TYPE_FLOAT32', 
            meta=nibabel.gifti.GiftiMetaData.from_dict(m0))

        # Triangles matrix 
        m1 = {'TopologicalType': 'Closed'}
        m1.update(common)
        ts = nibabel.gifti.GiftiDataArray(self.tris, 
            intent='NIFTI_INTENT_TRIANGLE', 
            coordsys=nibabel.gifti.GiftiCoordSystem(0,0), 
            datatype='NIFTI_TYPE_INT32', 
            meta=nibabel.gifti.GiftiMetaData.from_dict(m1))

        img = nibabel.gifti.GiftiImage(darrays=[ps,ts])
        nibabel.save(img, path)

    @ensure_derived_space
    def output_pvs(self, in_space):
        pvs_curr = self.voxelised.astype(np.float32)
        pvs_curr[self.LUT] = self.fractions
        out = np.zeros(np.prod(in_space.size), dtype=np.float32)
        curr_inds, dest_inds = self.reindexing_filter(in_space)
        out[dest_inds] = pvs_curr[curr_inds]
        return out.reshape(in_space.size)

    def _estimate_fractions(self, supersampler, cores, ones, desc=''):
        if ones: 
            self.fractions = np.ones(self.LUT.size, dtype=bool) 
        else: 
            self.fractions = core._estimateFractions(self, 
                supersampler, desc, cores)

    def index_based_on(self, space, affine):
        """
        Index a surface to an ImageSpace. The space must enclose the surface 
        completely (see ImageSpace.minimal_enclosing()). The surface will be 
        transformed into voxel coordinates for the space, triangle/voxel 
        associations calculated and stored on the surface's LUT and assocs 
        attributes, surface normals calculated (triangle cross products) and
        finally voxelised within the space (binary mask of voxels contained in 
        the surface). See also Surface.reindex_for() method. 

        Args: 
            space: ImageSpace object large enough to contain the surface
            affine: 4x4 np.array representing transformation into the reference
                space, in world-world mm terms (not FLIRT scaled-voxels). See
                utils._FLIRT_to_world for help. Pass None to represent identity. 

        Updates: 
            self.points: converted into voxel coordinates for the space
            self.assocs: list of triangles intersecting each voxel of the space
            self.LUT: table of voxel indices to index self.assocs 
            self.xProds: triangle cross products, voxel coordinates
        """

        encl_space = ImageSpace.minimal_enclosing(self, space, affine)

        if affine is not None: 
            overall = encl_space.world2vox @ affine
        else: 
            overall = encl_space.world2vox

        self.applyTransform(overall)
        maxFoV = self.points.max(0).round()
        minFoV = self.points.min(0).round()
        if np.any(minFoV < -1) or np.any(maxFoV > encl_space.size -1):
            raise RuntimeError("Space should be large enough to enclose surface")

        self.index_space = encl_space 
        self.formAssociations(encl_space.size)
        self.calculateXprods()
        self.voxelise()


    @ensure_derived_space
    def reindex_LUT(self, space):
        src_inds, dest_inds = self.reindexing_filter(space)
        fltr = np.isin(src_inds, self.LUT, assume_unique=True)
        return dest_inds[fltr]


    @ensure_derived_space
    def reindex_for(self, dest_space):
        """ 
        Re-index a surface for a space that derives from the space for which
        the space is currently indexed. For example, if the current index 
        space was produced using ImageSpace.minimal_enclosing(), then it cannot
        be assumed to cover the same FoV as the space of the reference image
        that was used to generate it. This function can be used to update the 
        properties of that surface to match this original reference space. 

        Args: 
            dest_space: ImageSpace from which the current index_space derives

        Updates: 
            points, LUT, assocs, xProds
        """

        # Get the offset and size of the current index space 
        ref_space = self.index_space
        FoVoffset = ref_space.offset 
        size = self.index_space.size 

        # Get a filter for voxel of the current index space that are still 
        # present in the dest_space 
        ref_inds = np.arange(np.prod(size))
        ref_voxs_in_dest = np.array(np.unravel_index(ref_inds, size)).T
        ref_voxs_in_dest -= FoVoffset
        ref2dest_fltr = self.reindexing_filter(dest_space)

        # Update LUT: produce filter of ref voxel inds within the current LUT
        # Combine this filter with the ref2dest_fltr. Finally, map the voxel 
        # coords corresponding to accepted voxels in the LUT into dest_space 
        LUTfltr = np.isin(ref_inds, self.LUT)
        LUTfltr = np.logical_and(LUTfltr, ref2dest_fltr)
        newLUT = np.ravel_multi_index(ref_voxs_in_dest[LUTfltr,:].T, 
            dest_space.size)

        # Update associations table. Accept all those that correspond to 
        # accepted LUT entries 
        newassocs = self.assocs[np.isin(self.LUT, ref_inds[LUTfltr])]

        # Update the voxelisation mask. Convert accepted voxel coordinates 
        # into voxel indices within destination space. These correspond to
        # the values in the new voxelisation mask that should be updated
        # using the extracted values from the existing voxelisation mask 
        ref_inds_subspace = np.ravel_multi_index(
            ref_voxs_in_dest[ref2dest_fltr,:].T, dest_space.size)
        newvoxelised = np.zeros(dest_space.size, dtype=bool).flatten()
        newvoxelised[ref_inds_subspace] = self.voxelised[ref2dest_fltr]

        # Lastly update the points: map them back to world mm coordinates 
        # and use the world2vox of the dest space 
        self.applyTransform(ref_space.vox2world)
        self.applyTransform(dest_space.world2vox)

        self.LUT = newLUT
        self.assocs = newassocs
        self.voxelised = newvoxelised
        self.index_space = dest_space
        self.offset = None

    @ensure_derived_space
    def reindexing_filter(self, dest_space, as_bool=False):
        """
        Filter of voxels in the current index space that lie within 
        dest_space. Use for extracting PV estimates from index space back to
        the space from which the index space derives. 

        Args: 
            dest_space: ImageSpace from which current index_space derives
            as_bool: output results as logical filters instead of indices

        Returns: 
            (src_inds, dest_inds) arrays of equal length, flat indices into 
            arrays of size index_space.size and dest_space.size respectively, 
            mapping voxels from source to destination positions 
        """

        # Get the offset and size of the current index space 
        src_space = self.index_space
        offset = src_space.offset 
        size = self.index_space.size 

        # List voxel indices in the current index space 
        # List corresponding voxel coordinates in the destination space 
        # curr2dest_fltr selects voxel indices from the current space that 
        # are also contained within the destination space 
        inds_in_src = np.arange(np.prod(size))
        voxs_in_src = np.array(np.unravel_index(inds_in_src, size)).T
        voxs_in_dest = voxs_in_src - offset
        fltr = np.logical_and(np.all(voxs_in_dest > - 1, 1), 
            np.all(voxs_in_dest < dest_space.size, 1))
        
        src_inds = np.ravel_multi_index(voxs_in_src[fltr,:].T, size)
        dest_inds = np.ravel_multi_index(voxs_in_dest[fltr,:].T, 
            dest_space.size)

        if as_bool: 
            src_fltr = np.zeros(np.prod(size), dtype=bool)
            src_fltr[src_inds] = 1 
            dest_fltr = np.zeros(np.prod(dest_space.size), dtype=bool)
            dest_fltr[dest_inds] = 1
            return (src_fltr, dest_fltr)

        return src_inds, dest_inds


    def calculateXprods(self):
        """Calculate and store surface element normals. Must be called prior to 
        estimateFractions()
        """

        self.xProds = np.cross(
            self.points[self.tris[:,2],:] - self.points[self.tris[:,0],:], 
            self.points[self.tris[:,1],:] - self.points[self.tris[:,0],:], 
            axis=1)

    
    def flipXprods(self):
        """Flip surface element normals"""

        self.xProds = -1 * self.xProds 


    def applyTransform(self, transform):
        """Apply affine transformation (4x4 array) to surface coordinates"""

        self.points = (utils._affineTransformPoints(
            self.points, transform).astype(np.float32))


    def formAssociations(self, size):
        """Identify which triangles of a surface intersect each voxel. This 
        reduces the number of operations that need be performed later. The 
        results will be stored on the surface object (ie, self)

        Args: 
            points: p x 3 matrix of surface nodes
            tris: t x 3 matrix of triangle node indices
            FoV: 1 x 3 vector of image dimensions (units of voxels) reqiured to
                fully enclose the surface

        Returns: 
            None, but associations (a list of lists) and LUT (used to index 
                between the associations list and vox index) are set on the 
                calling object. 
        """

        # Check for negative coordinates: these should have been sripped. 
        if np.round(np.min(self.points)) < 0: 
            raise RuntimeError("formAssociations: negative coordinate found")

        if np.any(np.round(np.max(self.points, axis=0)) >= size): 
            raise RuntimeError("formAssociations: coordinate outside FoV")

        cores = multiprocessing.cpu_count()
        chunks = utils._distributeObjects(range(self.tris.shape[0]), cores)
        workerFunc = functools.partial(core._formAssociationsWorker, 
            self.tris, self.points, size)

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


    @ensure_derived_space
    def find_bridges(self, space=None):
        if space is None: 
            space = self.index_space 
        counts = np.array([ len(x) for x in self.assocs ])
        if space is self.index_space:
            return self.LUT[counts > 1]
        else: 
            newLUT = self.reindex_LUT(space)
            return newLUT[counts > 1]


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

        # Load lists of tri numbers for each voxel index 
        vlists = self.assocs[np.isin(self.LUT, voxIndices, assume_unique=True)]

        if vlists.size:

            # Flatten the triangle numbers for all these voxels into single list
            triNums = functools.reduce(operator.iconcat, vlists, [])
            triNums = np.unique(triNums)

            return Patch(self.points, self.tris[triNums,:], 
                self.xProds[triNums,:])

        else: 
            return None

    
    def voxelise(self):
        """Voxelise (create binary in/out mask) a surface within a voxel grid
        of size size. Surface coordinates must be in 0-indexed voxel units, 
        as will the voxel grid be interpreted (ie, 0 : size - 1 in xyz). 
        Method is defined as static on the class for compataibility with 
        multiprocessing.Pool().

        Args: 
            None, but surface must have been indexed for a space first 

        Updates: 
            Surface.voxelised: flat boolean mask of voxels contained
                 within the surface. Use reshape(size) as required. 
        """

        try: 

            # Project rays along the largest dimension to classify as many voxels
            # at once as we can 
            size = self.index_space.size 
            dim = np.argmax(size)
            mask = np.zeros(np.prod(size), dtype=bool)
            otherdims = [0,1,2]
            otherdims.remove(dim)
            d1, d2 = tuple(otherdims)
            startPoint = np.zeros(3, dtype=np.float32)

            # The lazy way of working out strides through the voxel grid, 
            # regardless of what dimension we are projecting rays along. Define
            # a single ray of voxels, convert to linear indices and calculate 
            # the stride from that. 
            allIJKs = np.vstack(np.unravel_index(self.LUT, size)).T
            rayIJK = np.zeros((size[dim], 3), dtype=np.int16)
            rayIJK[:,dim] = np.arange(0, size[dim])
            rayIJK[:,d1] = allIJKs[0,d1]
            rayIJK[:,d2] = allIJKs[0,d2]
            linearInds = np.ravel_multi_index((rayIJK[:,0], rayIJK[:,1], 
                rayIJK[:,2]), size)
            stride = linearInds[1] - linearInds[0]

            # We now cycle through each voxel in the LUT, check which ray it 
            # lies on, calculate the indices of all other voxels on this ray, 
            # load the surface patches for all these voxels and find ray 
            # intersections to classify the voxel centres. Finally, we remove
            # the entire set of ray voxels from our copy of the LUT and repeat
            LUT = copy.copy(self.LUT)
            while LUT.size: 

                # Where does the ray that passes through this voxel start?
                # Load all other voxels on this ray
                startPoint[[d1,d2]] = [allIJKs[0,d1], allIJKs[0,d2]]
                startInd = np.ravel_multi_index(startPoint.astype(np.int32), size)
                voxRange = np.arange(startInd, startInd + (size[dim])*stride, 
                    stride)

                # Romeve those which are present in the LUT / allIJKs arrays
                keep = np.isin(LUT, voxRange, assume_unique=True, invert=True)
                LUT = LUT[keep]
                allIJKs = allIJKs[keep,:]
                
                # Load patches along this ray, we can assert that at least 
                # one patch must be returned. Find intersections 
                patches = self.toPatchesForVoxels(voxRange)
                assert patches is not None, 'No patches returned for voxel in LUT'
                intersectionMus = core._findRayTriangleIntersections2D(
                    startPoint, patches, dim)

                if not intersectionMus.size:
                    continue
                
                # If intersections were found, perform a parity test. 
                # Any ray should make an even number of intersections
                # as it crosses from -ve to +ve infinity
                if (intersectionMus.shape[0] % 2):
                    raise RuntimeError("voxelise: odd number of intersections" + 
                    " found. Does the FoV cover the full extents of the surface?")

                # Calculate points of intersection along the ray. 
                sorted = np.argsort(intersectionMus)
                intDs = startPoint[dim] + (intersectionMus[sorted])

                # Assignment. All voxels before the first point of intersection
                # are outside. The mask is already zeroed for these. All voxels
                # between point 1 and n could be in or out depending on parity
                for i in range(1, len(sorted)+1):

                    # Starting from infinity, all points between an odd numbered
                    # intersection and the next even one are inside the mask 
                    # Points beyond the last intersection are outside the mask
                    if ((i % 2) & ((i+1) <= len(sorted))):
                        indices = ((rayIJK[:,dim] > intDs[i-1]) 
                            & (rayIJK[:,dim] < intDs[i]))
                        mask[voxRange[indices]] = 1

            self.voxelised = mask 

        except Exception as e:
            print("Error voxelising surface.")
            raise e 




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
    """Preconfigured subclass of ArgumentParser to parse arguments that
    are common across pvtools functions. To use, instantiate an object, 
    then call add_argument to add in the arguments unique to the particular
    function in which it is being used, then finally call parse_args as 
    normal. 
    """

    def __init__(self):
        super().__init__()
        self.add_argument('-ref', type=str, required=True)
        self.add_argument('-struct2ref', type=str, required=True) 
        self.add_argument('-flirt', action='store_true', required=False)
        self.add_argument('-struct', type=str, required=False)
        self.add_argument('-cores', type=int, required=False)
        self.add_argument('-out', type=str, required=False)
        self.add_argument('-super', nargs='+', required=False)


    def parse(self, args):
        return vars(super().parse_args(args))
