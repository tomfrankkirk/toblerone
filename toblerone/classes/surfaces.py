"""
Tobleroe surface-related classes

Surface: the points and triangles of a surface, and various calculated
    properties that are evaluated ahead of time to speed up later operations
Hemisphere: a pair of surfaces, used specifically to represent one half 
    of the cerebral cortex (referred to as inner and outer surfaces)
Patch: a subcalss of Surface, representing a smaller portion of a surface,
    used to reduce computational complexity of operations 
"""

import itertools
import os.path as op 
import collections 
import operator
import functools
import copy 
import warnings 

import numpy as np 
import multiprocessing
import nibabel 
import pyvista 


from toblerone import utils, core 
from .image_space import ImageSpace

@utils.cascade_attributes
def ensure_derived_space(func):
    """
    Decorator for Surface functions that require ImageSpace arguments. 
    Internally, Surface objecs store information indexed to a minimal 
    enclosing voxel grid (referred to as the self.index_grid) based on 
    some arbitrary ImageSpace. When interacting with other ImageSpaces, 
    this function ensures that two grids are compatible with each other.
    """

    def ensured(self, *args):
        if not args: 
            raise RuntimeError("Function must be called with ImageSpace argument")
        if not self._index_space: 
            raise RuntimeError("Surface must be indexed prior to using this function" + 
            "Call surface.index_on()")
        if not self._index_space.derives_from(args[0]):
            raise RuntimeError(
                "Target space is not derived from surface's current index space."+
                "Call surface.index_on with the target space first")
        return func(self, *args)
    return ensured 



class Surface(object):
    """
    Encapsulates a surface's points, triangles and associations data.
    Create either by passing a file path (as below) or use the static class 
    method Surface.manual() to directly pass points and triangles.
    
    Args: 
        path:   path to file (.gii/.vtk/.white/.pial)
        space:  'world' (default) or 'first'; coordinate system of surface
        struct: if in 'first' space, then path to structural image used by FIRST
        name: optional, can be useful for progress bars 
    """

    def __init__(self, path, space='world', struct=None, name=None):

        if not op.exists(path):
            raise RuntimeError("File {} does not exist".format(path))

        if (path.count('first')) and ('space' == 'world'):
            print("Warning: surface seems to be from FIRST but space was set" +
                " as 'world'. See the docs.")

        surfExt = op.splitext(path)[-1]
        if surfExt == '.gii':
            gft = nibabel.load(path).darrays
            ps, ts = tuple(map(lambda o: o.data, gft))
        elif surfExt == '.vtk':
            obj = pyvista.PolyData(path)
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
            ps = utils.affineTransformPoints(ps, structSpace.vox2world)

        self.points = ps.astype(np.float32)
        self.tris = ts.astype(np.int32)
        self.xProds = None 
        self.voxelised = None 
        self.name = name
        self.assocs = None 
        self._index_space = None 


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
        s._index_space = None 
        return s


    def save(self, path):
        """Save surface as GIFTI file at path"""

        if not path.endswith('.surf.gii'):
            if path.endswith('.gii'):
                path.replace('.gii', '.surf.gii')
            else: 
                path += '.surf.gii'
        
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
    def output_pvs(self, space):
        """
        Express PVs in the voxel grid of space. Space must derive from the 
        surface's current index_space. 
        """

        pvs_curr = self.voxelised.astype(np.float32)
        pvs_curr[self.assocs_keys] = self.fractions
        out = np.zeros(np.prod(space.size), dtype=np.float32)
        curr_inds, dest_inds = self.reindexing_filter(space)
        out[dest_inds] = pvs_curr[curr_inds]
        return out.reshape(space.size)


    def _estimate_fractions(self, supersampler, cores, ones, desc=''):
        """
        Estimate interior/exterior fractions within current index_space. 

        Args: 
            supersampler: list/tuple of 3 ints, subdivision factor in each dim
            cores: number of multiprocessing cores to use 
            ones: debug tool, write ones in all voxels within assocs_keys 
            desc: for use with progress bar 
        """

        if self._index_space is None: 
            raise RuntimeError("Surface must be indexed first")

        if ones: 
            self.fractions = np.ones(self.assocs_keys.size, dtype=bool) 
        else: 
            self.fractions = core._estimateFractions(self, 
                supersampler, desc, cores)


    def index_on(self, space, struct2ref, cores=multiprocessing.cpu_count()):
        """
        Index a surface to an ImageSpace. The space must enclose the surface 
        completely (see ImageSpace.minimal_enclosing()). The surface will be 
        transformed into voxel coordinates for the space, triangle/voxel 
        associations calculated and stored on the surface's assocs 
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
            self.assocs: dict of triangles intersecting each voxel of the space
            self.assocs_keys: voxel indices that are keys into assocs dict  
            self.xProds: triangle cross products, voxel coordinates
        """

        # Smallest possible ImageSpace, based on space, that fully encloses surf 
        encl_space = ImageSpace.minimal_enclosing(self, space, struct2ref)
        if struct2ref is not None: 
            overall = encl_space.world2vox @ struct2ref
        else: 
            overall = encl_space.world2vox

        # Map surface points into this space via struct2ref
        self.applyTransform(overall)
        maxFoV = self.points.max(0).round()
        minFoV = self.points.min(0).round()
        if np.any(minFoV < -1) or np.any(maxFoV > encl_space.size -1):
            raise RuntimeError("Space should be large enough to enclose surface")

        # Update surface attributes
        self._index_space = encl_space 
        self.form_associations(cores)
        self.calculateXprods()
        self.voxelise()


    # @ensure_derived_space
    # def reindex_for(self, dest_space):
    #     """ 
    #     Re-index a surface for a space that derives from the space for which
    #     the space is currently indexed. For example, if the current index 
    #     space was produced using ImageSpace.minimal_enclosing(), then it cannot
    #     be assumed to cover the same FoV as the space of the reference image
    #     that was used to generate it. This function can be used to update the 
    #     properties of that surface to match this original reference space. 

    #     Args: 
    #         dest_space: ImageSpace from which the current index_space derives

    #     Updates: 
    #         points, LUT, assocs, xProds
    #     """

    #     # Get the offset and size of the current index space 
    #     ref_space = self._index_space
    #     FoVoffset = ref_space.offset 
    #     size = self._index_space.size 

    #     # Get a filter for voxel of the current index space that are still 
    #     # present in the dest_space 
    #     ref_inds = np.arange(np.prod(size))
    #     ref_voxs_in_dest = np.array(np.unravel_index(ref_inds, size)).T
    #     ref_voxs_in_dest -= FoVoffset
    #     ref2dest_fltr = self.reindexing_filter(dest_space)

    #     # Update LUT: produce filter of ref voxel inds within the current LUT
    #     # Combine this filter with the ref2dest_fltr. Finally, map the voxel 
    #     # coords corresponding to accepted voxels in the LUT into dest_space 
    #     LUTfltr = np.isin(ref_inds, self.LUT)
    #     LUTfltr = np.logical_and(LUTfltr, ref2dest_fltr)
    #     newLUT = np.ravel_multi_index(ref_voxs_in_dest[LUTfltr,:].T, 
    #         dest_space.size)

    #     # Update associations table. Accept all those that correspond to 
    #     # accepted LUT entries 
    #     newassocs = self.assocs[np.isin(self.LUT, ref_inds[LUTfltr])]

    #     # Update the voxelisation mask. Convert accepted voxel coordinates 
    #     # into voxel indices within destination space. These correspond to
    #     # the values in the new voxelisation mask that should be updated
    #     # using the extracted values from the existing voxelisation mask 
    #     ref_inds_subspace = np.ravel_multi_index(
    #         ref_voxs_in_dest[ref2dest_fltr,:].T, dest_space.size)
    #     newvoxelised = np.zeros(dest_space.size, dtype=bool).flatten()
    #     newvoxelised[ref_inds_subspace] = self.voxelised[ref2dest_fltr]

    #     # Lastly update the points: map them back to world mm coordinates 
    #     # and use the world2vox of the dest space 
    #     self.applyTransform(ref_space.vox2world)
    #     self.applyTransform(dest_space.world2vox)

    #     self.LUT = newLUT
    #     self.assocs = newassocs
    #     self.voxelised = newvoxelised
    #     self._index_space = dest_space
    #     self.offset = None


    @ensure_derived_space
    def reindexing_filter(self, dest_space, as_bool=False):
        """
        Filter of voxels in the current index space that lie within 
        dest_space. Use for extracting PV estimates from index space back to
        the space from which the index space derives. NB dest_space must 
        derive from the surface's current index_space 

        Args: 
            dest_space: ImageSpace from which current index_space derives
            as_bool: output results as logical filters instead of indices
                (note they will be of different size in this case)

        Returns: 
            (src_inds, dest_inds) arrays of equal length, flat indices into 
            arrays of size index_space.size and dest_space.size respectively, 
            mapping voxels from source to corresponding destination positions 
        """

        # Get the offset and size of the current index space 
        src_space = self._index_space
        offset = src_space.offset 
        size = self._index_space.size 

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


    @ensure_derived_space
    def reindex_LUT(self, space):
        """Return a copy of LUT indices expressed in another space"""

        src_inds, dest_inds = self.reindexing_filter(space)
        fltr = np.in1d(src_inds, self.assocs_keys, assume_unique=True)
        return dest_inds[fltr]


    @ensure_derived_space
    def find_bridges(self, space): 
        """
        Find voxels within space that are intersected by this surface 
        multiple times
        """

        group_counts = np.array([len(core._separatePointClouds(self.tris[a,:]))
            for a in self.assocs.values() ])
        bridges = self.assocs_keys[group_counts > 1]
        if space is self._index_space:
            return bridges 
        else: 
            src_inds, dest_inds = self.reindexing_filter(space)
            fltr = np.in1d(src_inds, bridges, assume_unique=True)
            return dest_inds[fltr]


    def calculateXprods(self):
        """Calculate and store surface element normals"""

        self.xProds = np.cross(
            self.points[self.tris[:,2],:] - self.points[self.tris[:,0],:], 
            self.points[self.tris[:,1],:] - self.points[self.tris[:,0],:], 
            axis=1)


    def applyTransform(self, transform):
        """Apply affine transformation (4x4 array) to surface coordinates"""

        self.points = (utils.affineTransformPoints(
            self.points, transform).astype(np.float32))


    def form_associations(self, cores=multiprocessing.cpu_count()):
        """
        Identify which triangles of a surface intersect each voxel. This 
        reduces the number of operations that need be performed later. The 
        results will be stored on the surface object (ie, self)

        Returns: 
            None, but associations (a list of lists) and assocs_keys (used to index 
                between the associations list and vox index) are set on the 
                calling object. 
        """

        size = self._index_space.size 

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
        self.assocs = dct 
        self.assocs_keys = np.array(list(self.assocs.keys()), dtype=np.int32)


    def rebaseTriangles(self, tri_inds):
        """
        Re-express a patch of a larger surface as a new points and triangle
        matrix pair, indexed from 0. Useful for reducing computational 
        complexity when working with a small patch of a surface where only 
        a few nodes in the points array are required by the triangles matrix. 

        Args: 
            tri_inds: t x 1 list of triangle numbers to rebase. 
        
        Returns: 
            (points, tris) tuple of re-indexed points/tris. 
        """

        points = np.empty((0, 3), dtype=np.float32)
        tris = np.empty((len(tri_inds), 3), dtype=np.int32)
        pointsLUT = []

        for t in range(len(tri_inds)):
            for v in range(3):

                # For each vertex of each tri, check if we
                # have already processed it in the LUT
                vtx = self.tris[tri_inds[t],v]
                idx = np.argwhere(pointsLUT == vtx)

                # If not in the LUT, then add it and record that
                # as the new position. Write the missing vertex
                # into the local points array
                if not idx.size:
                    pointsLUT.append(vtx)
                    idx = len(pointsLUT) - 1
                    points = np.vstack([points, self.points[vtx,:]])

                # Update the local triangle
                tris[t,v] = idx

        return (points, tris)


    def to_patch(self, vox_idx):
        """
        Return a patch object specific to a voxel given by linear index.
        Look up the triangles intersecting the voxel, and then load and rebase
        the points / surface normals as required. 
        """

        tri_nums = self.assocs[vox_idx]
        (ps, ts) = self.rebaseTriangles(tri_nums)

        return Patch(ps, ts, self.xProds[tri_nums,:])

    
    def to_patches(self, vox_inds):
        """
        Return the patches for the voxels in voxel indices, flattened into 
        a single set of ps, ts and xprods. 

        If no patches exist for this list of voxels return None.
        """

        # Load lists of tri numbers for each voxel index 
        voxs = np.intersect1d(self.assocs_keys, vox_inds, assume_unique=True)

        if voxs.size:

            # Flatten the triangle numbers for all these voxels into single list
            tri_nums = functools.reduce(operator.iconcat, 
                [self.assocs[v] for v in voxs], []) 
            tri_nums = np.unique(tri_nums)

            return Patch(self.points, self.tris[tri_nums,:], 
                self.xProds[tri_nums,:])

        else: 
            return None


    
    def voxelise(self):
        """
        Voxelise (create binary in/out mask) a surface within a voxel grid
        of size size. Surface coordinates must be in 0-indexed voxel units, 
        as will the voxel grid be interpreted (ie, 0 : size - 1 in xyz). 
        Method is defined as static on the class for compataibility with 
        multiprocessing.Pool(). NB surface must have been indexed first 

        Updates: 
            self.voxelised: flat boolean mask of voxels contained
                 within the surface. Use reshape(size) as required. 
        """

        try: 

            # Project rays along the largest dimension to classify as many voxels
            # at once as we can 
            size = self._index_space.size 
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
            allIJKs = np.array(np.unravel_index(self.assocs_keys, size)).T
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
            LUT = copy.deepcopy(self.assocs_keys)
            while LUT.size: 

                # Where does the ray that passes through this voxel start?
                # Load all other voxels on this ray
                startPoint[[d1,d2]] = [allIJKs[0,d1], allIJKs[0,d2]]
                startInd = np.ravel_multi_index(startPoint.astype(np.int32), size)
                voxRange = np.arange(startInd, startInd + (size[dim])*stride, 
                    stride)

                # Romeve those which are present in the LUT / allIJKs arrays
                keep = np.in1d(LUT, voxRange, assume_unique=True, invert=True)
                LUT = LUT[keep]
                allIJKs = allIJKs[keep,:]
                
                # Load patches along this ray, we can assert that at least 
                # one patch must be returned. Find intersections 
                patches = self.to_patches(voxRange)
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
    """
    Subclass of Surface that represents a small patch of surface. 
    Points, triangles and xProds are all inherited from the parent surface. 
    This class should not be directly created but instead instantiated via
    the Surface.to_patch() / to_patches() methods. 
    """

    def __init__(self, points, tris, xProds):
        self.points = points 
        self.tris = tris
        self.xProds = xProds 
        

    def shrink(self, fltr):
        """
        Return a shrunk copy of the patch by applying the logical 
        filter fltr to the calling objects tris and xprods matrices
        """

        return Patch(self.points, self.tris[fltr,:], self.xProds[fltr,:])


class Hemisphere(object): 
    """
    The white and pial surfaces of a hemisphere, and a repository to 
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