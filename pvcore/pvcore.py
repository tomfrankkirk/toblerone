import nibabel
import os.path as op
import numpy as np
import copy
import itertools

# Class definitions -----------------------------------------------------------

# Class to contain an images voxel grid, including dimensions and vox2world transforms
class ImageSpace():

    def __init__(self, imgSize, voxSize, vox2world):
        """Use the static method .fromfile() to initialise"""
        self.imgSize = imgSize
        self.voxSize = voxSize
        self.vox2world = vox2world
        self.world2vox = np.linalg.inv(vox2world)
        

    @staticmethod
    def fromfile(path):
        """Load the image at path and read in header matrix, image size 
        and voxel size
        """

        if not op.isfile(path):
            raise RuntimeError("Image does not exist")

        img = nibabel.load(path)
        return ImageSpace(
            img.header['dim'][1:4], 
            img.header['pixdim'][1:4], 
            img.affine
        )


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

        newSize = self.imgSize * factor
        newVoxSize = self.voxSize / factor
        newVox2World = copy.deepcopy(self.vox2world)
        for r in range(3):
            newVox2World[r, 0:3] = newVox2World[r, 0:3] / factor[r]

        # The direction vector travelled in incrementing each index 
        # within the original grid by 1 is: 
        ijkvec = np.sum(self.vox2world[0:3,0:3], axis=1)

        # Origin of the src voxel grid is at (NB 1 is dist between vox cents)
        orig = self.vox2world[0:3,3] - (0.5 * ijkvec)

        # Which means the new grid has centre coord of vox (0,0,0) at: 
        offset = orig + (0.5 * ijkvec / factor)
        newVox2World[0:3,3] = offset 

        # Check the bounds of the new voxel grid we have created
        svertices = np.array(list(itertools.product([-0.5, newSize[0] - 0.5], 
            [-0.5, newSize[1] - 0.5], [-0.5, newSize[2] - 0.5])))
        rvertices = np.array(list(itertools.product([-0.5, self.imgSize[0] - 0.5], 
            [-0.5, self.imgSize[1] - 0.5], [-0.5, self.imgSize[2] - 0.5])))
        rvertices = affineTransformPoints(rvertices, self.vox2world)
        svertices = affineTransformPoints(svertices, newVox2World)
        assert np.all(np.abs(rvertices -svertices) < 1e-6)

        return ImageSpace(newSize, newVoxSize, newVox2World)


    def saveImage(self, data, path):

        if not np.all(data.shape[0:3] == self.imgSize):
            raise RuntimeError("Data size does not match image size")

        img = nibabel.Nifti2Image(data, self.vox2world)
        nibabel.save(img, path)


# Function definitions --------------------------------------------------------


def addSuffixToFilename(suffix, fname):
    """Add suffix to filename, whilst preserving original extension"""
    root = copy.copy(fname)
    ext = ''
    while '.' in root:
        root, e = op.splitext(root)
        ext = e + ext 
    
    return root + suffix + ext 


def affineTransformPoints(points, affine):
    """Apply affine transformation to set of points.

    Args: 
        points: n x 3 matrix of points to transform
        affine: 4 x 4 matrix for transformation

    Returns: 
        transformed copy of points 
    """

    # Add 1s on the 4th column, transpose and multiply, 
    # then re-transpose and drop 4th column  
    transfd = np.ones((points.shape[0], 4))
    transfd[:,0:3] = points
    transfd = np.matmul(affine, transfd.T).astype(np.float32)
    return (transfd[0:3,:]).T


def loadSurfsToDict(FSdir):
    sdir = op.realpath(op.join(FSdir, 'surf'))

    if not op.isdir(sdir):
        raise RuntimeError("Subject's surf directory does not exist")

    surfs = {}    
    for s in ['LWS', 'LPS', 'RWS', 'RPS']:
        snames = {'L': 'lh', 'R': 'rh'}
        exts = {'W': '.white', 'P': '.pial'}
        surfs[s] = op.join(sdir, snames[s[0]] + exts[s[1]])

    if not all(map(op.isfile, surfs.values())):
        raise RuntimeError("One of the subject's surfaces does not exist")

    return surfs


def coordinatesForGrid(ofSize):
    I, J, K = np.unravel_index(np.arange(np.prod(ofSize)), ofSize)
    cents = np.vstack((I.flatten(), J.flatten(), K.flatten())).T
    return cents.astype(np.int32)


def maskVolumes(fourDarray, mask):

    if not (fourDarray.shape[0:3] == mask.shape):
        raise RuntimeError("Mask dimensions do not match volumes")
    
    for n in range(fourDarray.shape[-1]):
        fourDarray[:,:,:,n] = (fourDarray[:,:,:,n] * mask)

    return fourDarray