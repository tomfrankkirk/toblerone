"""
ImageSpace: image matrix, inc dimensions, voxel size, vox2world matrix and
inverse, of an image. Inherits most methods and properties from 
regtricks.ImageSpace. 
"""

import copy 
import warnings

import numpy as np 
from regtricks import ImageSpace as BaseSpace

from toblerone import utils

class ImageSpace(BaseSpace):
    """
    Voxel grid of an image, ignoring actual image data. 

    Args: 
        reference: path to image file, or regtricks ImageSpace object 
    
    Attributes: 
        size: array of voxel counts in each dimension 
        vox_size: array of voxel size in each dimension 
        vox2world: 4x4 affine to transform voxel coords -> world
        world2vox: inverse of above 
        self.offset: private variable used for derived spaces 
    """


    def __init__(self, reference):
        if type(reference) is str:
            super().__init__(reference)
        else: 
            if not type(reference) is BaseSpace:
                raise ValueError("Reference must be a path or regtricks ImageSpace")
            for k,v in vars(reference).items():
                setattr(self, k, copy.deepcopy(v))         
        self.offset = None



    @classmethod
    def minimal_enclosing(cls, surfs, reference):
        """
        Return the minimal space required to enclose a set of surfaces. 
        This space will be based upon the reference, sharing its voxel 
        size and i,j,k unit vectors from the voxel2world matrix, but 
        will may have a different FoV. The offset of the voxel coord system
        relative to reference will be stored as the space.offset attribute

        Args: 
            surfs: singular or list of surface objects 
            reference: ImageSpace object or path to image to use 

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

        # Extract min and max vox coords in the reference space 
        min_max = np.empty((2*len(slist), 3))
        for sidx,s in enumerate(slist):
            ps = utils.affine_transform(s.points, space.world2vox)
            min_max[sidx*2,:] = ps.min(0)
            min_max[sidx*2 + 1,:] = ps.max(0)

        # Fix the offset relative to reference and minimal size 
        min_vox = np.floor(min_max.min(0)).astype(np.int16)
        max_vox = np.ceil(min_max.max(0)).astype(np.int16)
        # min_vox = np.clip(min_vox, 0, space.size-1)
        # max_vox = np.clip(max_vox, 0, space.size-1)
        size = max_vox - min_vox + 1
    
        # Get a copy of the corresponding mm coords for checking later
        min_max_mm = utils.affine_transform(np.array([min_vox, max_vox]),
            space.vox2world)

        # Calculate new origin for the coordinate system and modify the 
        # vox2world matrix accordingly 
        space.size = size 
        space.vox2world[0:3,3] = min_max_mm[0,:]
        space.offset = - min_vox 

        check = utils.affine_transform(min_max_mm, space.world2vox)
        if (np.any(check[0,:].round() < 0) or 
            np.any(check[1,:].round() > size - 1)): 
            raise RuntimeError("New space does not enclose surfaces")

        new_voxs, old_voxs = reindexing_filter(space, reference)
        if not (new_voxs.size and old_voxs.size): 
            warnings.warn(f"Surfaces {[s.name for s in slist]} do not intersect the reference voxel grid")

        space.parent = reference

        return space 


    def derives_from(self, parent):
        """
        Logical test whether this ImageSpace was derived from another. 
        "Derived" means sharing i,j,k unit vectors and having their origins
        shifted by an integer multiple of voxels relative to each other. 
        """

        det1 = np.linalg.det(parent.vox2world[0:3,0:3])
        det2 = np.linalg.det(self.vox2world[0:3,0:3])
        offset_mm = parent.vox2world[0:3,3] - self.vox2world[0:3,3] 
        offset_mm2 = parent.vox2world[0:3,3] @ self.offset 
        return ((np.abs(det1 - det2) < 1e-9) and np.all(np.abs(offset_mm - offset_mm2) < 1e9))


def reindexing_filter(src_space, dest_space, as_bool=False):
    """
    Filter of voxels in the source space that lie within destination
    space. Use for extracting PV estimates from index space back to
    the space from which the index space derives. NB dest_space must 
    derive from the surface's current index_space 

    Args: 
        src_space: ImageSpace data is currently in 
        dest_space: ImageSpace data will be mapped into. Must derive from 
                    src_space. 
        as_bool: output results as logical filters instead of indices
            (note they will be of different size in this case)

    Returns: 
        (src_inds, dest_inds) arrays of equal length, flat indices into 
        arrays of size src_space.size and dest_space.size respectively, 
        mapping voxels from source to corresponding destination positions 
    """

    if not src_space.derives_from(dest_space):
        raise ValueError("src_space must derive from dest_space")

    # We need offset and size of source space compared to dest 
    offset = src_space.offset 
    size = src_space.size 

    # src_orig = src_space.vox2world[:3,3]
    # dest_orig = dest_space.vox2world[:3,3]
    # shift_mm = dest_orig - src_orig 
    # shift_vox = shift_mm / src_space.vox_size

    # List voxel indices in the current index space 
    # List corresponding voxel coordinates in the destination space 
    # curr2dest_fltr selects voxel indices from the current space that 
    # are also contained within the destination space 
    # inds_in_src = np.arange(np.prod(size))
    # voxs_in_src = np.array(np.unravel_index(inds_in_src, size)).T
    voxs_in_src = np.moveaxis(np.indices(size), 0, 3).reshape(-1,3)
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
