import sys
import itertools
import copy
import multiprocessing
import os
import os.path as op
import subprocess
import warnings
import shutil

from pvtools import toblerone as cortex
from pvtools import pvcore
import nibabel
import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.ndimage

TISSUES = ['GM', 'WM', 'CSF']

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
        output_shape=destSpace.imgSize, mode='constant', order=3)

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
    srcSpace = pvcore.ImageSpace.fromfile(source)
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

    refSpace = pvcore.ImageSpace.fromfile(reference)
    intSpace = pvcore.ImageSpace.fromfile(subcortPath)
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
    mask = np.logical_and(cortex[:,:,:,0], cortex[:,:,:,2]) 
    outdir = op.dirname(cortexPath)
    intSpace.saveImage(mask.astype(np.int8), op.join(outdir, 'mask.nii.gz'))

    for (a, m, f) in zip([cortex, subcortex], [mask, ~mask], 
        [cortexPath, subcortPath]):
        a = pvcore._maskVolumes(a, m)
        intSpace.saveImage(a, pvcore._addSuffixToFilename('_masked', f))

    # Combine estimates in intermediate space using the mask
    print("Combing estimates in intermediate space")
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
    savepaths):
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
    refSpace = pvcore.ImageSpace.fromfile(reference)
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
    print("Estimating within cortex (Toblerone)")
    cortex.estimatePVs(LWS=surfs['LWS'], LPS=surfs['LPS'], 
        RWS=surfs['RWS'], RPS=surfs['RPS'], ref=savepaths[0], 
        struct2ref=struct2ref, name=savepaths[1])



def _shellCommand(cmd):   
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
    pwd = os.getcwd()
    os.chdir(dir)
    cmd = 'recon-all -i {} -all -subjid fs -sd .'.format(struct)
    print("Preparing to call FreeSurfer: this will take ~10 hours\n")
    _shellCommand(cmd)
    os.chdir(pwd)


def _estimateStructuralPVs(struct, savepaths):
    """Run FAST and FreeSurfer on the given structural image, saving the
    output into respective folders within the given directory. 

    Args:
        struct:     path to structural image
        savepaths:  dict with keys GM/WM/CSF, paths to save images. 
        debug:      don't run commands 
    
    Returns: 
        each tool, with the following keys:
        subcorts: GM, WM, CSF: paths to PV estimates for these tissues
        surfs: LWS/LPS/RWS/RPS: Left/Right White/Pial surfaces.
    """

    # Required to find FAST's output later on
    _, sname = op.split(struct)
    while '.' in sname:
        sname, _ = op.splitext(sname)

    # Move into the output dir, run FAST, move back afterwards
    pwd = os.getcwd()
    fastdir = op.dirname(savepaths['GM'])

    if not op.isdir(fastdir):
        os.mkdir(fastdir)

    os.chdir(fastdir)
    _sysprint("Running FAST...")
    newfname = op.split(struct)[-1]
    shutil.copyfile(struct, newfname)
    cmd = 'fast ' + newfname 
    _shellCommand(cmd)
    _sysprint("Done.\n")

    # Convert between FAST's naming convention and explicit tissue names
    oldname = lambda n: \
        op.join(sname + '_pve_{}.nii.gz'.format(n))
    for (n,t) in zip([1,2,0], TISSUES):
        shutil.copyfile(oldname(n), savepaths[t])

    # spc = pvcore.ImageSpace.fromfile(savepaths['GM'])
    # wm = nibabel.load(savepaths['WM']).get_fdata()
    # gm = nibabel.load(savepaths['GM']).get_fdata()
    # csf = 1 - (wm + gm)
    # spc.saveImage(csf, savepaths['CSF'])

    os.chdir(pwd)


def merge_with_surface():
    pass 


def estimate_all(**kwargs):

    debug = ('debug' in kwargs)

    # Check args
    for a in ['ref', 'struct', 'struct2ref']:
        if (a not in kwargs) or (not op.isfile(kwargs[a])):
            raise RuntimeError(a +" not given or non-existent")
        else: 
            kwargs[a] = op.realpath(kwargs[a])

    # If outdir not given, then use the same as the reference
    if 'outdir' in kwargs:
        outdir = kwargs['outdir']
    else:
        outdir = op.dirname(op.realpath(kwargs['ref']))

    pvdir = op.join(outdir, 'pvtools')
    intdir = op.join(pvdir, 'intermediate')
    for d in [pvdir, intdir]:
        if not op.isdir(d):
            os.mkdir(d)

    # Prepare output directorxy and run structural (FAST/FS) estimation
    if not op.isdir(pvdir):
        os.mkdir(pvdir)

    fastnames = { t: op.join(pvdir, 'fast', t + '.nii.gz') 
        for t in TISSUES }
    if not debug and not kwargs.get('noFAST'): 
        _estimateStructuralPVs(kwargs['struct'], fastnames)

    if not kwargs.get('noFS'):
        _runFreeSurfer(kwargs['struct'], pvdir)

    surfs = pvcore._loadSurfsToDict(op.join(pvdir, 'fs'))

    # Load struct2func transform, and adjust appropriately
    struct2ref = np.loadtxt(kwargs['struct2ref'])
    if kwargs.get('flirt'):
        print("Adjusting FLIRT matrix")
        struct2ref = pvcore._adjustFLIRT(kwargs['struct'], kwargs['ref'], 
            struct2ref)

    # Prepare intermediate directory and resample FAST / run Toblerone within 
    intsubcort = op.join(intdir, 'subcort.nii.gz')
    intctx = op.join(intdir, 'cortex.nii.gz')
    if not debug:
        _estimateIntermediatePVs(fastnames, surfs, kwargs['ref'], struct2ref, 
            (intsubcort, intctx))

    # Merge subcort/cortex estimates in intermediate space
    outname = pvcore._addSuffixToFilename('_pvs', kwargs['ref'])
    _mergeCortexAndSubcorticalPVs(intsubcort, intctx, kwargs['ref'], 
        outname)

    print("Saving final output:", outname)
    