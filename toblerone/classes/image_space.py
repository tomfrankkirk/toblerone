"""
ImageSpace: image matrix, inc dimensions, voxel size, vox2world matrix and
inverse, of an image. Used for resampling operations between different 
spaces and also for saving images into said space (eg, save PV estimates 
into the space of an image)
"""

import os.path as op 
import copy 
import warnings

import nibabel
import numpy as np 
from regtools import ImageSpace as BaseSpace

from .. import utils

class ImageSpace(BaseSpace):
    """
    Voxel grid of an image, ignoring actual image data. 

    Args: 
        path: path to image file
    
    Attributes: 
        size: array of voxel counts in each dimension 
        vox_size: array of voxel size in each dimension 
        vox2world: 4x4 affine to transform voxel coords -> world
        world2vox: inverse of above 
        self.offset: private variable used for derived spaces 
    """


    def __init__(self, path):
        super().__init__(path)
        self.offset = None


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
        min_max = np.empty((2*len(slist), 3))
        for sidx,s in enumerate(slist):
            ps = utils.affineTransformPoints(s.points, overall)
            min_max[sidx*2,:] = ps.min(0)
            min_max[sidx*2 + 1,:] = ps.max(0)

        # Fix the offset relative to reference and minimal size 
        minFoV = min_max.min(0).round().astype(np.int16)
        maxFoV = min_max.max(0).round().astype(np.int16)
        size = maxFoV - minFoV + 1
        FoVoffset = -minFoV
    
        # Get a copy of the corresponding mm coords for checking later
        min_max_mm = utils.affineTransformPoints(np.array([minFoV, maxFoV]),
            space.vox2world)

        # Calculate new origin for the coordinate system and modify the 
        # vox2world matrix accordingly 
        space.size = size 
        space.vox2world[0:3,3] = min_max_mm[0,:]
        space.offset = FoVoffset 

        check = utils.affineTransformPoints(min_max_mm, space.world2vox)
        if (np.any(check[0,:].round() < 0) or 
            np.any(check[1,:].round() > size - 1)): 
            raise RuntimeError("New space does not enclose surfaces")

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