import os.path as op
import multiprocessing
import warnings
import functools
import copy
import shutil
import time 

import numpy as np
import tqdm

from toblerone import core, estimators, utils, resampling
from toblerone.classes import ImageSpace, Hemisphere
from toblerone.classes import Surface, CommonParser

# Simply apply a function to list of arguments.
# Used for multiprocessing shell commands. 
def apply_func(func, args):
    func(*args)


def timer(func):
    """Timing decorator, prints duration in minutes"""

    @functools.wraps(func)
    def timed_function(*args, **kwargs):
        t1 = time.time()
        out = func(*args, **kwargs)
        t2 = time.time()
        print("Elapsed time: %.1f minutes" % ((t2-t1)//60))
        return out 
    
    return timed_function


def enforce_and_load_common_arguments(func):
    """Decorator to enforce and pre-processes common arguments in a 
    kwargs dict that are used across multiple functions. Note
    some function-specific checking is still required. This intercepts the
    kwargs dict passed to the caller, does some checking and modification 
    in place, and then returns to the caller. The following args are handled:

    Required args:
        ref: path to a reference image in which to operate 
        struct2ref: path to file or 4x4 array representing transformation
            between structural space (that of the surfaces) and reference. 
            If given as 'I', identity matrix will be used. 

    Optional args: 
        anat: a fsl_anat directory (created/augmented by make_surf_anat_dir)
        flirt: bool denoting that the struct2ref is a FLIRT transform.
            This means it requires special treatment. If set, then it will be
            pre-processed in place by this function, and then the flag will 
            be set back to false when the kwargs dict is returned to the caller
        struct: if FLIRT given, then the path to the structural image used
            for surface generation is required for said special treatment
        cores: maximum number of cores to parallelise tasks across 
            (default is N-1)

    Returns: 
        a modified copy of kwargs dictionary passed to the caller
    """
    
    @functools.wraps(func)
    def enforcer(**kwargs):

        # Reference image path 
        if not kwargs.get('ref'):
            raise RuntimeError("Path to reference image must be given")

        if not op.isfile(kwargs['ref']):
            raise RuntimeError("Reference image does not exist")

        # If given a anat_dir we can load the structural image in 
        if kwargs.get('anat'):
            if not utils.check_surf_anat_dir_complete(kwargs['anat']):
                raise RuntimeError("anat is not complete: it must contain" + 
                    "fast, fs and first subdirectories")

            s = op.join(kwargs['anat'], 'T1.nii.gz')
            if not op.isfile(s):
                raise RuntimeError("Could not find T1.nii.gz in the anat dir")

            kwargs['struct'] = s
 
        # Structural to reference transformation. Either as array or path
        # to file containing matrix
        if not any([type(kwargs.get('struct2ref')) is str, 
            type(kwargs.get('struct2ref')) is np.ndarray]):
            raise RuntimeError("struct2ref transform must be given (either path", 
                "or np.array object)")

        else:
            s2r = kwargs['struct2ref']

            if (type(s2r) is str): 
                if s2r == 'I':
                    matrix = np.identity(4)
                else:
                    _, matExt = op.splitext(kwargs['struct2ref'])

                    try: 
                        if matExt in ['.txt', '.mat']:
                            matrix = np.loadtxt(kwargs['struct2ref'], 
                                dtype=np.float32)
                        elif matExt in ['.npy', 'npz', '.pkl']:
                            matrix = np.load(kwargs['struct2ref'])
                        else: 
                            matrix = np.fromfile(kwargs['struct2ref'], 
                                dtype=np.float32)

                    except Exception as e:
                        warnings.warn("""Could not load struct2ref matrix. 
                            File should be any type valid with numpy.load().""")
                        raise e 

                kwargs['struct2ref'] = matrix

        if not kwargs['struct2ref'].shape == (4,4):
            raise RuntimeError("struct2ref must be a 4x4 matrix")

        # If FLIRT transform we need to do some clever preprocessing
        # We then set the flirt flag to false again (otherwise later steps will 
        # repeat the tricks and end up reverting to the original - those steps don't
        # need to know what we did here, simply that it is now world-world again)
        if kwargs.get('flirt'):
            if not kwargs.get('struct'):
                raise RuntimeError("If using a FLIRT transform, the path to the \
                    structural image must also be given")
            kwargs['struct2ref'] = utils._adjustFLIRT(kwargs['struct'], kwargs['ref'], 
                kwargs['struct2ref'])
            kwargs['flirt'] = False 

        # Processor cores
        if not kwargs.get('cores'):
            kwargs['cores'] = max([multiprocessing.cpu_count() - 1, 1])

        # Supersampling factor
        sup = kwargs.get('super')
        if sup is not None: 
            try: 
                if (type(sup) is list) and (len(sup) == 3): 
                    sup = np.array([int(s) for s in sup])
                else: 
                    sup = int(sup[0])
                    sup = np.array([sup for _ in range(3)])

                if type(sup) is not np.ndarray: 
                    raise RuntimeError() 
            except:
                raise RuntimeError("-super must be a value or list of 3" + 
                    " values of int type")
        
            kwargs['super'] = sup.astype(np.int8)
            print("Using manual supersampling factor", kwargs['super'])

        return kwargs

    def enforced(**kwargs):
        kwargs = enforcer(**kwargs)
        return func(**kwargs)

    return enforced


@timer
@enforce_and_load_common_arguments
def estimate_all(**kwargs):
    """Estimate PVs for cortex and all structures identified by FIRST within 
    a reference image space. Use FAST to fill in non-surface PVs. 
    All arguments are kwargs.

    Required args: 
        ref: path to reference image for which PVs are required
        struct2ref: path to np or text file, or np.ndarray obj, denoting affine
                registration between structural (surface) and reference space.
                Use 'I' for identity. 
        anat: path to anat dir directory (created by make_surf_anat_dir)

    Optional args: 
        flirt: bool denoting struct2ref is FLIRT transform. If so, set struct
        struct: path to structural image from which surfaces were derived
        cores: number of cores to use (default N-1)
 
    Returns: 
        (pvs, transformed) both dictionaries. 
        pvs contains the PVs associated with each individual structure and 
            also the overall combined result ('stacked')
        transformed contains copies of each surface transformed into ref space
    """

    print("Estimating PVs for", kwargs['ref'])

    # If not provided with a anat dir, then create 
    if 'anat' not in kwargs:
        raise RuntimeError("An anat dir must be given. See make_surf_anat_dir")

    # We should now have a complete anat, so form paths to fsdir, 
    # fastdir, firstdir accordingly. 
    kwargs['fastdir'] = kwargs['anat']
    kwargs['fsdir'] = op.join(kwargs['anat'], 'fs')
    kwargs['firstdir'] = op.join(kwargs['anat'], 'first_results')

    for k in ['fast', 'fs', 'first']:
        print("Using {} data in: {}".format(k, kwargs[k+'dir']))
   
    # Resample FASTs to reference space. Then redefine CSF as 1-(GM+WM)
    fasts = utils._loadFASTdir(kwargs['fastdir'])
    output = { t: resample(fasts[t], kwargs['ref'], kwargs['struct2ref'])
        for t in ['FAST_WM', 'FAST_GM'] } 
    output['FAST_CSF'] = 1 - (output['FAST_WM'] + output['FAST_GM'])
        
    # Process subcortical structures first. 
    FIRSTsurfs = utils._loadFIRSTdir(kwargs['firstdir'])
    structures = [ Surface(surf, 'first', kwargs['struct'], name) 
        for name, surf in FIRSTsurfs.items() ]
    print("The following structures will be estimated:", flush=True)
    [ print(s.name, end=' ') for s in structures ]
    print('Cortex')
    desc = ' Subcortical structures'
    estimator = functools.partial(estimate_structure_wrapper, 
        **kwargs)

    results = []
    if kwargs['cores'] > 1:
        with multiprocessing.Pool(kwargs['cores']) as p: 
            for _, r in tqdm.tqdm(enumerate(p.imap(estimator, structures)), 
                total=len(structures), desc=desc, 
                bar_format=core.BAR_FORMAT, ascii=True):
                    results.append(r)

    else: 
        for _, r in tqdm.tqdm(enumerate(map(estimator, structures)), 
            total=len(structures), desc=desc, 
            bar_format=core.BAR_FORMAT, ascii=True):
                results.append(r)

    if kwargs.get('savesurfs'):
        transformed = {} 
    else: 
        transformed = None 

    for key, (pvs, trans) in zip(FIRSTsurfs.keys(), results):
        output.update({key: pvs})
        if kwargs.get('savesurfs'): 
            transformed.update({key: trans})

    # Now do the cortex, then stack the whole lot 
    ctx, ctxmask, trans = estimate_cortex(**kwargs)
    for i,t in enumerate(['_GM', '_WM', '_nonbrain']):
        output['cortex' + t] = (ctx[:,:,:,i])
    if trans: 
        transformed.update(trans)

    output['cortexmask'] = ctxmask
    output['stacked'] = stack_images(output)

    return output, transformed


def estimate_structure_wrapper(surf, **kwargs):
    """Convenience method for parallel processing"""
    return estimate_structure(surf=surf, **kwargs)


@enforce_and_load_common_arguments
def estimate_structure(**kwargs):
    """Estimate PVs for a structure defined by a single surface. 
    All arguments are kwargs.
    
    Required args: 
        ref: path to reference image for which PVs are required
        struct2ref: path to np or text file, or np.ndarray obj, denoting affine
                registration between structural (surface) and reference space.
                Use 'I' for identity. 
        surf: path to surface (see space argument below)

    Optional args: 
        space: space in which surface is defined: default is 'world' (mm coords),
            for FIRST surfaces set 'first' (FSL convention). 
        struct: path to structural image from which surfaces were derived
        cores: number of cores to use (default N-1)
 
    Returns: 
        (pvs, transformed) PV image and transformed surface object. 
    """

    # Check we either have a surface object or path to one 
    if not 'surf' in kwargs:
        raise RuntimeError("surf kwarg must be a Surface object or path to one")

    if type(kwargs['surf']) is str: 
        surf = Surface(kwargs['surf'], kwargs['space'], kwargs['struct'], 
            op.split(kwargs['surf'])[1])
    
    elif type(kwargs['surf']) is not Surface: 
        raise RuntimeError("surf kwarg must be a Surface object or path to one")

    else: 
        surf = kwargs['surf']

    # Load reference space, set supersampler
    refSpace = ImageSpace(kwargs['ref'])
    supersampler = kwargs.get('super')
    if supersampler is None:
        supersampler = np.ceil(refSpace.voxSize).astype(np.int8) + 1

    # Apply registration and save copies if reqd 
    surf.applyTransform(kwargs['struct2ref'])
    transformed = None
    if kwargs.get('savesurfs'):
        transformed = copy.deepcopy(surf)

    # Apply transformation to voxel space 
    surf.applyTransform(refSpace.world2vox)

    return (estimators._structure(refSpace, 1, supersampler, surf), 
        transformed)


@enforce_and_load_common_arguments
def estimate_cortex(**kwargs):
    """Estimate PVs for L/R cortex. All arguments are kwargs.

    Required args: 
        ref: path to reference image for which PVs are required
        struct2ref: path to np or text file, or np.ndarray obj, denoting affine
                registration between structural (surface) and reference space.
                Use 'I' for identity. 

        One of: 
        fsdir: path to a FreeSurfer subject directory, from which L/R 
            white/pial surfaces will be loaded 
        LWS/LPS/RWS/RPS: individual paths to the individual surfaces,
            eg LWS = Left White surface, RPS = Right Pial surace
            To estimate for a single hemisphere, only provide surfaces
            for that side. 

    Optional args: 
        space: space in which surfaces are in (world/first)
        flirt: bool denoting struct2ref is FLIRT transform. If so, set struct
        struct: path to structural image from which surfaces were derived
        cores: number of cores to use (default N-1)
        stack: stack the estimates for GM/WM/non-brain into a 4D NIFTI, 
            in that order 
 
    Returns: 
        (pvs, mask, transformed) all dictionaries. 
        pvs contains the PVs associated with each individual structure and 
            also the overall combined result ('stacked')
        mask is a binary mask of voxels intersecting the cortex
        transformed contains copies of each surface transformed into ref space
    """

    if not any([
        kwargs.get('fsdir') is not None, 
        any([ kwargs.get(s) is not None 
            for s in ['LWS', 'LPS', 'RWS', 'RPS'] ]) ]):
        raise RuntimeError("Either a fsdir or paths to LWS/LPS etc"
            "must be given.")

    if not kwargs.get('cores'):
        kwargs['cores'] = max([multiprocessing.cpu_count() - 1, 1])

    # If subdir given, then get all the surfaces out of the surf dir
    # If individual surface paths were given they will already be in scope
    if kwargs.get('fsdir'):
        surfdict = utils._loadSurfsToDict(kwargs['fsdir'])
        kwargs.update(surfdict)

    # What hemispheres are we working with?
    sides = []
    if all([ kwargs.get(s) is not None for s in ['LPS', 'LWS'] ]): 
        sides.append('L')

    if all([ kwargs.get(s) is not None for s in ['RPS', 'RWS'] ]): 
        sides.append('R')

    if not sides:
        raise RuntimeError("At least one hemisphere (eg LWS/LPS required")

    # Load reference ImageSpace object
    # Form the final transformation matrix to bring the surfaces to 
    # the same world (mm, not voxel) space as the reference image
    refSpace = ImageSpace(kwargs['ref'])
    if kwargs.get('verbose'): 
        np.set_printoptions(precision=3, suppress=True)
        print("Final surface-to-reference (world) transformation:\n", 
            kwargs['struct2ref'])

    # Transforms: surface -> reference -> reference voxels
    # then calculate cross prods for each surface element 
    hemispheres = [ Hemisphere(kwargs[s+'WS'], kwargs[s+'PS'], s) 
        for s in sides ] 
    surfs = [ s for h in hemispheres for s in h.surfs() ]
    for s in surfs: 
        s.applyTransform(kwargs['struct2ref'])
    
    # Grab transformed copies of the surfaces before going to voxel space
    surfdict = {}
    ( surfdict.update(h.surf_dict) for h in hemispheres )
    transformed = { (k,copy.deepcopy(s)) for (k,s) in surfdict.items() }
    
    for s in surfs:
        s.applyTransform(refSpace.world2vox)

    # Set supersampler and estimate. 
    supersampler = kwargs.get('super')
    if supersampler is None:
        supersampler = np.ceil(refSpace.voxSize).astype(np.int8) + 1

    outPVs, cortexMask = estimators._cortex(hemispheres, refSpace, 
        supersampler, kwargs['cores'])

    return (outPVs, cortexMask, transformed)


def resample(src, ref, src2ref=np.identity(4), flirt=False):
    """Resample an image via upsampling to an intermediate space followed
    by summation back down to reference space. Wrapper for superResampleImage()
    
    Args:
        src: path to source image 
        ref: path to reference image, onto which src will be resampled 
        out: path to save output 
        src2ref: 4x4 affine transformation between src and ref 
    """
   
    if flirt:
        src2ref = utils._adjustFLIRT(src, ref, src2ref)

    refSpace = ImageSpace(ref)
    factor = np.ceil(refSpace.voxSize).astype(np.int8)
    return resampling._superResampleImage(src, factor, refSpace, src2ref)


def stack_images(images):
    """Combine the results of estimate_all() into overall PV maps
    for each tissue. Note that the below logic is entirely specific 
    to the surfaces produced by FreeSurfer, FIRST and how they may be
    combined with FAST estimates. If you're going off-piste anywhere else
    then you probably DON'T want to re-use this logic. 

    Args: 
        dictionary of PV maps, keyed as follows: all FIRST subcortical
        structures named by their FIST convention (eg L_Caud); FAST's estimates
        named as FAST_CSF/WM/GM; cortex estimates as cortex_GM/WM/non_brain

    Returns: 
        single 4D array of PVs, arranged GM/WM/non-brain in the 4th dim
    """

    # Copy the dict of images as we are going to make changes and dont want 
    # to play with the caller's copy. Pop unwanted images
    images = copy.copy(images)
    if 'cortexmask' in images: 
        images.pop('cortexmask')
        images.pop('BrStem')
    
    # Pop out FAST's estimates  
    csf = images.pop('FAST_CSF').flatten()
    wm = images.pop('FAST_WM').flatten()
    gm = images.pop('FAST_GM')
    shape = list(gm.shape[0:3]) + [3]

    # Pop the cortex estimates and initialise output as all CSF
    ctxgm = images.pop('cortex_GM').flatten()
    ctxwm = images.pop('cortex_WM').flatten()
    ctxnon = images.pop('cortex_nonbrain').flatten()
    ctx = np.vstack((ctxgm, ctxwm, ctxnon)).T
    out = np.zeros_like(ctx)
    out[:,2] = 1

    # Then write in Toblerone's filled cortex estimates from the 
    # cortex inwards 
    mask = np.logical_or(ctx[:,0], ctx[:,1])
    out[mask,:] = ctx[mask,:]

    # Overwrite using FAST's CSF estimate. Where FAST has suggested a higher
    # CSF estimate than currently exists, and the voxel is not within the pure
    # cortex, accept FAST's estimate. Then update the other tissues estimates
    # in these voxels in the order GM, then WM. This is because we always have 
    # higher confidence in a GM estimate than WM. 
    ctxmask = (ctx[:,0] > 0)
    updates = np.logical_and(csf > out[:,2], ~ctxmask)
    tmpwm = out[updates,1]
    tmpgm = out[updates,0]
    out[updates,2] = csf[updates]
    out[updates,0] = np.minimum(tmpgm, 1 - out[updates,2])
    out[updates,1] = np.minimum(tmpwm, 1 - (out[updates,0] + out[updates,2]))

    # Sanity check: total tissue PV in each vox should sum to 1
    assert np.all(np.abs(out.sum(1) - 1) < 1e-6)

    # Prepare the WM weights to be used when writing in PVs from subcortical 
    # structures. Lots of tricks are needed to avoid zero- or small-divisions. 
    # These are taken from FAST's WM/CSF estimates. 
    div = utils._clipArray(wm+csf)
    divmask = (div > 0)
    wmweights = np.zeros_like(wm)
    wmweights[divmask] = wm[divmask] / div[divmask] 
    wmweights = utils._clipArray(wmweights)

    # For each subcortical structure, create a mask of the voxels which it 
    # relates to. Write in the PVs as GM
    for s in images.values():
        smask = (s.flatten() > 0)
        out[smask,0] = np.minimum(1, out[smask,0] + s.flatten()[smask])

        # And then for the remainders, ie, 1-PV, distribute it amongst
        # WM and CSF in the proportion of these tissue that FAST assigned. 
        out[smask,1] = np.maximum(0, wmweights[smask] * (1-out[smask,0]))
        out[smask,2] = np.maximum(0, 1-np.sum(out[smask,0:2], axis=1))
        
    # Final sanity check, then rescaling so all voxels sum to unity. 
    sums = out.sum(1)
    assert np.all(np.abs(sums - 1) < 1e-6)
    out = out / sums[:,None]

    return out.reshape(shape)
