import nibabel
import os.path as op
import numpy as np
import copy
import itertools


# Class definitions -----------------------------------------------------------

# Class to contain an images voxel grid, including dimensions and vox2world transforms
class ImageSpace(object):

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
        rvertices = _affineTransformPoints(rvertices, self.vox2world)
        svertices = _affineTransformPoints(svertices, newVox2World)
        assert np.all(np.abs(rvertices -svertices) < 1e-6)

        return ImageSpace(newSize, newVoxSize, newVox2World)


    def saveImage(self, data, path):

        if not np.all(data.shape[0:3] == self.imgSize):
            raise RuntimeError("Data size does not match image size")

        img = nibabel.Nifti2Image(data, self.vox2world)
        nibabel.save(img, path)


# Function definitions --------------------------------------------------------


def _addSuffixToFilename(suffix, fname):
    """Add suffix to filename, whilst preserving original extension"""
    root = copy.copy(fname)
    ext = ''
    while '.' in root:
        root, e = op.splitext(root)
        ext = e + ext 
    
    return root + suffix + ext 


def _affineTransformPoints(points, affine):
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


def _loadSurfsToDict(FSdir):
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


def _coordinatesForGrid(ofSize):
    I, J, K = np.unravel_index(np.arange(np.prod(ofSize)), ofSize)
    cents = np.vstack((I.flatten(), J.flatten(), K.flatten())).T
    return cents.astype(np.int32)


def _maskVolumes(fourDarray, mask):

    if not (fourDarray.shape[0:3] == mask.shape):
        raise RuntimeError("Mask dimensions do not match volumes")
    
    for n in range(fourDarray.shape[-1]):
        fourDarray[:,:,:,n] = (fourDarray[:,:,:,n] * mask)

    return fourDarray



def _adjustFLIRT(source, reference, transform):
    """Adjust a FLIRT transformation matrix into a true world-world 
    transform. Required as FSL matrices are encoded in a specific form 
    such that they can only be applied alongside the requisite images (extra
    information is required from those images). With thanks to Martin Craig
    and Tim Coalson. See: https://github.com/Washington-University/workbench/blob/9c34187281066519e78841e29dc14bef504776df/src/Nifti/NiftiHeader.cxx#L168 
    https://github.com/Washington-University/workbench/blob/335ad0c910ca9812907ea92ba0e5563225eee1e6/src/Files/AffineFile.cxx#L144

    Args: 
        source: path to source image, the image to be deformed 
        reference: path to reference image, the target of the transform
        transform: affine matrix produced by FLIRT from src to ref 

    Returns: 
        complete transformation matrix between the two. 
    """

    # Local function to read out an FSL-specific affine matrix from an image
    def __getFSLspace(imgPth):
        obj = nibabel.load(imgPth)
        if obj.header['dim'][0] < 3:
            raise RuntimeError("Volume has less than 3 dimensions" + \
                 "cannot resolve space")

        sform = obj.affine
        det = np.linalg.det(sform[0:4, 0:4])
        ret = np.identity(4)
        pixdim = obj.header['pixdim'][1:4]
        for d in range(3):
            ret[d,d] = pixdim[d]

        # Check the xyzt field to find the spatial units. 
        xyzt =str(obj.header['xyzt_units'])
        if xyzt == '01': 
            multi = 1000
        elif xyzt == '10':
            multi = 1 
        elif xyzt =='11':
            multi = 1e-3
        else: 
            raise RuntimeError("Unknown units")

        if det > 0:
            ret[0,0] = -pixdim[0]
            ret[0,3] = (obj.header['dim'][1] - 1) * pixdim[0]

        ret = ret * multi
        ret[3,3] = 1
        return ret

    # Main function
    srcSpace = __getFSLspace(source)
    refSpace = __getFSLspace(reference)

    refObj = nibabel.load(reference)
    refAff = refObj.affine 
    srcObj = nibabel.load(source)
    srcAff = srcObj.affine 

    outAff = np.matmul(np.matmul(
        np.matmul(refAff, np.linalg.inv(refSpace)),
        transform), srcSpace)
    return np.matmul(outAff, np.linalg.inv(srcAff))

