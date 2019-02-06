# Core functions and utilities used throughout pvtools

import sys
import copy
import itertools
import shutil
import os.path as op
import os
import subprocess

import numpy as np
import nibabel
import scipy.ndimage

from pvtools.classes import ImageSpace, TISSUES
from pvtools import fileutils


BAR_FORMAT = '{l_bar}{bar} {elapsed} | {remaining}'


def _clipArray(arr, mini=0.0, maxi=1.0):
    """Clip array values into range [mini, maxi], default [0 1]"""

    arr[arr < mini] = mini 
    arr[arr > maxi] = maxi 
    return arr 


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


def _coordinatesForGrid(ofSize):
    """Produce N x 3 array of all voxel indices (eg [10, 18, 2]) within
    a grid of size ofSize, 0-indexed and in integer form. 
    """

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


def _sysprint(txt):
    sys.stdout.write(txt)
    sys.stdout.flush()


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



def _mergeCortexAndSubcorticalPVs(subcortPath, cortexPath, reference, outpath):
    """Take subcort PVs in intermediate space and Tob estimates, 
    sum across grid and output results in reference space"""

    refSpace = ImageSpace(reference)
    intSpace = ImageSpace(subcortPath)
    cortex = nibabel.load(cortexPath).get_fdata().astype(np.float32)
    subcortex = nibabel.load(subcortPath).get_fdata().astype(np.float32)

    # Calculate the factor multiple between spaces. Assert no remainder, 
    # then force into ints. 
    factor = [ i/r for (i,r) in zip(intSpace.imgSize, refSpace.imgSize) ]
    if not all(map(lambda f: f%1 == 0, factor)):
        raise RuntimeError("Intermediate space is not an integer multiple of reference")
    factor = [ int(f) for f in factor ]

    # Cortex mask: accept all and(GM, CSF) from Toblerone
    # Apply mask to the different sets of estimates. 
    mask = np.logical_or(cortex[:,:,:,0], cortex[:,:,:,2]) 
    outdir = op.dirname(cortexPath)
    intSpace.saveImage(mask.astype(np.int8), op.join(outdir, 'mask.nii.gz'))

    for (a, m, f) in zip([cortex, subcortex], [mask, ~mask], 
        [cortexPath, subcortPath]):
        a = _maskVolumes(a, m)
        intSpace.saveImage(a, fileutils._addSuffixToFilename('_masked', f))

    # Combine estimates in intermediate space using the mask
    print("Combining estimates in intermediate space")
    summed = cortex + subcortex

    # Sum array blocks and divide by blocksize to get mean within each
    print("Summing estimates back into reference space")
    outpvs = _sumArrayBlocks(summed, factor) / np.prod(factor)

    # Rescale by sum of each voxels estimates to get range [0 1]
    # (small float rounding errors expected otherwise)
    divisors = np.sum(outpvs, axis=3)
    for d in range(3):
        outpvs[:,:,:,d] = outpvs[:,:,:,d] / divisors

    print("Saving final output at:", outpath)
    refSpace.saveImage(outpvs, outpath)


def _estimateIntermediatePVs(subcorts, surfs, reference, struct2ref, 
    savepaths, tob):
    """Estimate PVs in intermediate space by resampling subcortical estimates
    and running Toblerone on surfaces.
    
    Args:
        subcorts:   dict with keys GM/WM/CSF for paths to PV estimates for 
                        each of these tissues respectively, in structural 
                        space (ie, FAST)
        surfs:      dict with keys RWS/RPS/LPS/LWS for paths to each of these
                        surfaces
        reference:  path to reference image
        struct2ref: structural to reference affine transform, in world/world
                        coordinates (ie, adjusted-FLIRT)
        savepaths:  two path names, where to save subcortical and cortex 
                        respectively output. 
    """

    # Load reference image information, create intermediate space as 
    # supersampled copy
    refSpace = ImageSpace(reference)
    factor = np.maximum(np.floor(refSpace.voxSize), [1,1,1]).astype(np.int8)
    intSpace = refSpace.supersample(factor)
    print("Intermediate factor set at", factor)

    # Resample subcortical estimates and flatten into single image. 
    _sysprint("Resampling subcortical estimates...")
    subcortPVs = np.stack((_superResampleImage(subcorts[t], (2,2,2), 
        intSpace, struct2ref) for t in TISSUES ), axis=3) 
    intSpace.saveImage(subcortPVs, savepaths[0])   
    _sysprint("Done.\n") 
    
    # Run Toblerone to estimate cortex PVs. Use the subcortical image as
    # the reference for the intermediate space 
    print("Estimating within cortex")
    if tob: 
        pass
        # cortex.estimatePVs(LWS=surfs['LWS'], LPS=surfs['LPS'], 
        #     RWS=surfs['RWS'], RPS=surfs['RPS'], ref=savepaths[0], 
        #     struct2ref=struct2ref, name=savepaths[1])



def getVoxList(imgSize, FoVoffset, FoVsize):
    """Single list of linear voxel indices for all voxels lying within 
    a grid of size imgSize, contained within a potentially larger grid 
    of size FoVsize, in which the respective origins are shifted by 
    FoVoffst. 

    Args: 
        imgSize: the size of the grid for which voxels indices are needed
        FoVoffset: the offset (3-vector) between voxel [0,0,0] of the grid
            represented by FoVoffset and voxel [0,0,0] of imgSize. 
        FoVsize: the size of the (potentially) larger voxel grid within which
            the smaller grid lies (they are at least the same size)

    Returns: 
        list of linear voxel indices referenced to the grid FoVsize
    """

    voxSubs = _coordinatesForGrid(imgSize)
    voxSubs = voxSubs + FoVoffset 
    return np.ravel_multi_index((voxSubs[:,0], voxSubs[:,1], voxSubs[:,2]),
        FoVsize)



def _shellCommand(cmd):   
    """Convenience function for calling shell commands"""

    try: 
        ret = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, 
            shell=True)
        if ret.returncode:
            print("Non-zero return code")
            raise RuntimeError()
    except Exception as e:
        print("Error when executing cmd:", cmd)
        raise e


def _runFreeSurfer(struct, dir):
    """Args: 
        struct: path to structural image 
        dir: path to directory in which a subject directory entitled
            'fs' will be created and FS run within
    """

    struct = op.abspath(struct)
    pwd = os.getcwd()
    os.chdir(dir)
    cmd = 'recon-all -i {} -all -subjid fs -sd .'.format(struct)
    print("Calling FreeSurfer on", struct)
    print("This will take ~10 hours")
    _shellCommand(cmd)
    os.chdir(pwd)


def _runFIRST(struct, dir):
    """Args: 
        struct: path to structural image 
        dir: path to directory in which FIRST will be run
    """

    fileutils.weak_mkdir(dir)
    nameroot, _ = fileutils.splitExts(struct)
    struct = op.abspath(struct)
    pwd = os.getcwd()
    os.chdir(dir)
    cmd = 'run_first_all -i {} -o {}'.format(struct, nameroot)
    print("Calling FIRST on", struct)
    _shellCommand(cmd)
    os.chdir(pwd)


def _runFAST(struct, dir):
    """Args: 
        struct: path to structural image 
        dir: path to directory in which FAST will be run
    """

    fileutils.weak_mkdir(dir)
    struct = op.abspath(struct)
    pwd = os.getcwd()
    newstruct = op.abspath(op.join(dir, op.split(struct)[1]))
    shutil.copy(struct, newstruct)
    os.chdir(dir)
    cmd = 'fast {}'.format(newstruct)
    print("Calling FAST on", struct)
    _shellCommand(cmd)
    os.chdir(pwd)