# Resampling tricks 
# Essentially a python implementation of the approach used in FSL applywarp, 
# ie, upsample the image to an interdmediate space and then sum across super-
# resolution voxels to get new output values 

import numpy as np 
import scipy.ndimage
import nibabel

from .classes import ImageSpace

def _resampleImage(data, srcSpace, destSpace, src2dest):
    """Resample array data onto destination space, applying affine transformation
    at the same time

    Args: 
        data: array to resample
        scrSpace: ImageSpace in which the data currently exists 
        destSpace: ImageSpace onto which it should be resampled
        src2dest: 4x4 transformation matrix to apply during resampling

    Returns: 
        array of size destSpace.imgSize
    """

    # Transform the destination grid into world coordinates, aligned with src
    destvox2src = np.matmul(np.linalg.inv(src2dest), destSpace.vox2world)
    destvox2src = np.matmul(np.linalg.inv(srcSpace.vox2world), destvox2src)

    # Interpolate. 
    out = scipy.ndimage.affine_transform(data, destvox2src, 
        output_shape=destSpace.imgSize, mode='constant', order=4)

    # Due to the spline interpolants, the resampled output can go outside
    # the original min,max of the input data 
    out = np.maximum(out, data.min())
    out = np.minimum(out, data.max())

    return out 


def _superResampleImage(source, factor, destSpace, src2dest):
    """Resample an image onto a new space, applying an affine transformation
    at the same time. The source image transformed onto an upsampled copy of 
    the destination space first and then block-summed back down to the required
    resolution. 

    Args:
        source: path to image to resample
        factor: iterable length 3, extent to upsample in each dim 
        destSpace: an ImageSpace representing the destination space
        src2dest: affine transformation matrix (4x4) between source and reference

    Returns: 
        an array of the dimensions given by destSpace.imgSize
    """

    # Load input data and read in the space for it
    srcSpace = ImageSpace(source)
    data = nibabel.load(source).get_fdata().astype(np.float32)

    # Create a supersampled version of the space 
    superSpace = destSpace.supersample(factor)

    # Resample onto this new grid, applying transform at the same time. 
    # Then sum the array blocks and divide by the size of each block to 
    # get the mean value within each block. This is the final output. 
    resamp = _resampleImage(data, srcSpace, superSpace, src2dest)
    resamp = _sumArrayBlocks(resamp, factor) / np.prod(factor)

    return resamp


def _sumArrayBlocks(array, factor):
    """Sum sub-arrays of a larger array, each of which is sized according to factor. 
    The array is split into smaller subarrays of size given by factor, each of which 
    is summed, and the results returned in a new array, shrunk accordingly. 

    Args:
        array: n-dimensional array of data to sum
        factor: n-length tuple, size of sub-arrays to sum over

    Returns:
        array of size array.shape/factor, each element containing the sum of the 
            corresponding subarray in the input
    """

    outshape = [ int(s/f) for (s,f) in zip(array.shape, factor) ]
    out = np.copy(array)

    for dim in range(3):
        newshape = [0] * 4

        for d in range(3):
            if d < dim: 
                newshape[d] = outshape[d]
            elif d == dim: 
                newshape[d+1] = factor[d]
                newshape[d] = outshape[d]
            else: 
                newshape[d+1] = array.shape[d]

        newshape = newshape + list(array.shape[3:])
        out = np.sum(out.reshape(newshape), axis=dim+1)

    return out 
