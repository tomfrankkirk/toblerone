import os.path as op
import multiprocessing
import warnings
import functools
import copy
import shutil
import time 
import subprocess
import nibabel

import numpy as np
import tqdm

from toblerone import core, estimators, utils, resampling
from toblerone.classes import ImageSpace, Hemisphere
from toblerone.classes import Surface, CommonParser


# Simply apply a function to list of arguments.
# Used for multiprocessing shell commands. 
def apply_func(func, args):
    func(*args)


def cascade_attributes(decorator):
    """
    Overrride default decorator behaviour to preserve docstrings etc
    of decorated functions - functools.wraps didn't seem to work. 
    See https://stackoverflow.com/questions/6394511/python-functools-wraps-equivalent-for-classes
    """

    def new_decorator(original):
        wrapped = decorator(original)
        wrapped.__name__ = original.__name__
        wrapped.__doc__ = original.__doc__
        wrapped.__module__ = original.__module__
        return wrapped
    return new_decorator


@cascade_attributes
def timer(func):
    """Timing decorator, prints duration in minutes"""

    def timed_function(*args, **kwargs):
        t1 = time.time()
        out = func(*args, **kwargs)
        t2 = time.time()
        print("Elapsed time: %.1f minutes" % ((t2-t1)//60))
        return out 
    
    return timed_function


@cascade_attributes
def enforce_and_load_common_arguments(func):
    """
    Decorator to enforce and pre-processes common arguments in a 
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
    
    def enforcer(**kwargs):

        # Reference image path 
        if not kwargs.get('ref'):
            raise RuntimeError("Path to reference image must be given")

        if not op.isfile(kwargs['ref']):
            raise RuntimeError("Reference image does not exist")

        # If given a anat_dir we can load the structural image in 
        if kwargs.get('anat'):
            if not utils.check_anat_dir(kwargs['anat']):
                raise RuntimeError("anat is not complete: it must contain" + 
                    "fast, fs and first subdirectories")

            kwargs['fastdir'] = kwargs['anat']
            kwargs['fsdir'] = op.join(kwargs['anat'], 'fs')
            kwargs['firstdir'] = op.join(kwargs['anat'], 'first_results')

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
            kwargs['struct2ref'] = utils._FLIRT_to_world(kwargs['struct'], kwargs['ref'], 
                kwargs['struct2ref'])
            kwargs['flirt'] = False 

        # Processor cores
        if not kwargs.get('cores'):
            kwargs['cores'] = multiprocessing.cpu_count()

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
    """
    Estimate PVs for cortex and all structures identified by FIRST within 
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
        cores: number of cores to use 
 
    Returns: 
        (pvs, transformed) both dictionaries. 
        pvs contains the PVs associated with each individual structure and 
            also the overall combined result ('stacked')
        transformed contains copies of each surface transformed into ref space
    """

    print("Estimating PVs for", kwargs['ref'])

    # If anat dir then various subdirs are loaded by @enforce_common_args
    # If not then direct load below 
    if not bool(kwargs.get('anat')):
        if not bool(kwargs.get('fsdir')):
            if not all([ bool(kwargs.get(k)) for k in ['LWS','LPS','RWS','RPS'] ]):
                raise RuntimeError("If fsdir not given, " + 
                    "provide paths for LWS,LPS,RWS,RPS")
        
        if not (bool(kwargs.get('fastdir')) and bool(kwargs.get('firstdir'))):
            raise RuntimeError("If not using anat dir, fastdir/firstdir required")
   
    # Resample FASTs to reference space. Then redefine CSF as 1-(GM+WM)
    fasts = utils._loadFASTdir(kwargs['fastdir'])
    output = { t: resample(fasts[t], kwargs['ref'], kwargs['struct2ref'])
        for t in ['FAST_WM', 'FAST_GM'] } 
    output['FAST_CSF'] = 1 - (output['FAST_WM'] + output['FAST_GM'])
        
    # Process subcortical structures first. 
    FIRSTsurfs = utils._loadFIRSTdir(kwargs['firstdir'])
    structures = [ Surface(surf, 'first', kwargs['struct'], name) 
        for name, surf in FIRSTsurfs.items() ]
    print("Structures found: ", end=' ')
    [ print(s.name, end=' ') for s in structures ]
    print('Cortex')
    desc = 'Subcortical structures'

    # To estimate against each subcortical structure, we apply the following
    # partial func to each using a map() call. Carry kwargs from this func 
    estimator = functools.partial(estimate_structure_wrapper, **kwargs)

    # This is equivalent to a map(estimator, structures) call
    # All the extra stuff (tqdm etc) is used for progress bar
    results = [ pv for _, pv in 
        tqdm.tqdm(enumerate(map(estimator, structures)), 
        total=len(structures), desc=desc, bar_format=core.BAR_FORMAT, 
        ascii=True) ] 

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
    stacked = stack_images(output)
    output['all_GM'] = stacked[:,:,:,0]
    output['all_WM'] = stacked[:,:,:,1]
    output['all_nonbrain'] = stacked[:,:,:,2]
    output['all_stacked'] = stacked

    return output, transformed


def estimate_structure_wrapper(surf, **kwargs):
    """Convenience method for parallel processing"""
    return estimate_structure(surf=surf, **kwargs)


@enforce_and_load_common_arguments
def estimate_structure(**kwargs):
    """
    Estimate PVs for a structure defined by a single surface. 
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
        cores: number of cores to use 
 
    Returns: 
        (pvs, transformed) PV image and transformed surface object. 
    """

    # Check we either have a surface object or path to one 
    if not bool(kwargs.get('surf')):
        raise RuntimeError("surf kwarg must be a Surface object or path to one")

    if type(kwargs['surf']) is str: 
        surf = Surface(kwargs['surf'], kwargs['space'], kwargs['struct'], 
            op.split(kwargs['surf'])[1])
    
    elif type(kwargs['surf']) is not Surface: 
        raise RuntimeError("surf kwarg must be a Surface object or path to one")

    else: 
        surf = kwargs['surf']

        
    # Load reference space, set supersampler
    ref_space = ImageSpace(kwargs['ref'])
    encl_space = ImageSpace.minimal_enclosing(surf, ref_space, kwargs['struct2ref'])

    supersampler = kwargs.get('super')
    if supersampler is None:
        supersampler = np.ceil(ref_space.vox_size / 0.75).astype(np.int8)

    # Index the suface to the enclosing space 
    surf.index_for(encl_space, kwargs['struct2ref'])
    pvs_encl_space = estimators._structure(kwargs['cores'], supersampler, 
        bool(kwargs.get('ones')), surf)

    # Output transformed copies of surfaces if requested 
    transformed = None
    if kwargs.get('savesurfs'):
        transformed = copy.copy(surf)
        transformed.applyTransform(surf.index_space.vox2world)

    # Extract PVs in the reference space 
    encl_inds, ref_inds = surf.reindexing_filter(ref_space)
    pvs = np.zeros(np.prod(ref_space.size), dtype=np.float32)
    pvs[ref_inds] = pvs_encl_space.flatten()[encl_inds]

    return (pvs.reshape(ref_space.size), transformed)


@enforce_and_load_common_arguments
def estimate_cortex(**kwargs):
    """
    Estimate PVs for L/R cortex. All arguments are kwargs.

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
        cores: number of cores to use 
        stack: stack the estimates for GM/WM/non-brain into a 4D NIFTI, 
            in that order 
 
    Returns: 
        (pvs, mask, transformed) all dictionaries. 
        pvs contains the PVs associated with each named structure and 
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
    hemispheres = [ Hemisphere(kwargs[s+'WS'], kwargs[s+'PS'], s) 
        for s in sides ] 
    surfs = [ s for h in hemispheres for s in h.surfs() ]
    ref_space = ImageSpace(kwargs['ref'])
    encl_space = ImageSpace.minimal_enclosing(surfs, ref_space, kwargs['struct2ref'])
    for surf in surfs: 
        surf.index_for(encl_space, kwargs['struct2ref'])
    
    # Grab transformed copies of the surfaces before going to voxel space
    surfdict = {}
    ( surfdict.update(h.surf_dict) for h in hemispheres )
    transformed = {}
    if bool(kwargs.get('savesurfs')):
        for k,s in surfdict.items():
            srf = copy.copy(s)
            srf.applyTransform(srf.index_space.vox2world)
            transformed[k] = srf
    
    # Set supersampler and estimate. 
    supersampler = kwargs.get('super')
    if supersampler is None:
        supersampler = np.ceil(ref_space.vox_size / 0.75).astype(np.int8)

    pvs_encl, ctx_mask_encl = estimators._cortex(hemispheres, 
        supersampler, kwargs['cores'], bool(kwargs.get('ones')))

    col_shape = np.prod(ref_space.size)
    shape_4D = (*ref_space.size, 3)
    shape_3D = ref_space.size 
    encl_inds, ref_inds = surfs[0].reindexing_filter(ref_space)
    pvs = np.zeros((col_shape, 3), dtype=np.float32)
    ctx_mask = np.zeros(col_shape, dtype=bool)
    pvs[ref_inds,:] = pvs_encl.reshape(-1, 3)[encl_inds,:]
    ctx_mask[ref_inds] = ctx_mask_encl.flatten()[encl_inds]

    return (pvs.reshape(shape_4D), ctx_mask.reshape(shape_3D), transformed)


def resample(src, ref, src2ref=None, flirt=False):
    """
    Resample an image via upsampling to an intermediate space followed
    by summation back down to reference space. Wrapper for superResampleImage()
    
    Args:
        src: path to source image 
        ref: path to reference image, onto which src will be resampled 
        src2ref: 4x4 affine transformation between src and ref, default
            is None, to represent identity transform 
        flirt: bool, if affine is a FLIRT matrix 
    """
   
    if flirt:
        assert src2ref is not None, 'Default src2ref cannot be FLIRT'
        src2ref = utils._FLIRT_to_world(src, ref, src2ref)

    if src2ref is None: 
        src2ref = np.identity(4)

    ref_space = ImageSpace(ref)
    src_space = ImageSpace(src)
    factor = np.ceil(ref_space.vox_size).astype(np.int8)
    data = nibabel.load(src).get_fdata().astype(np.float32)

    return resampling._superResampleImage(data, src_space, ref_space, src2ref, factor)


def stack_images(images):
    """
    Combine the results of estimate_all() into overall PV maps
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


def fsl_fs_anat(**kwargs):
    """
    Run fsl_anat (FAST & FIRST) and augment output with FreeSurfer

    Args: 
        anat: (optional) path to existing fsl_anat dir to augment
        struct: (optional) path to T1 NIFTI to create a fresh fsl_anat dir
        out: output path (default alongside input, named input.anat)
    """

    # We are either adding to an existing dir, or we are creating 
    # a fresh one 
    if (not bool(kwargs.get('anat'))) and (not bool(kwargs.get('struct'))):
        raise RuntimeError("Either a structural image or a path to an " + 
            "existing fsl_anat dir must be given")

    if kwargs.get('struct') and (not op.isfile(kwargs['struct'])):
        raise RuntimeError("No struct image given, or does not exist")

    if kwargs.get('anat') and (not op.isdir(kwargs['anat'])):
        raise RuntimeError("fsl_anat dir does not exist")

    debug = bool(kwargs.get('debug'))
    anat_exists = bool(kwargs.get('anat'))
    struct = kwargs['struct']

    # Run fsl_anat if needed. Either use user-supplied name or default
    if not anat_exists:
        outname = kwargs.get('out')
        if not outname:
            outname = utils._splitExts(kwargs['struct'])[0]
            outname = op.dirname(kwargs['struct']) + outname
        print("Preparing an fsl_anat dir at %s" % outname)
        if outname.endswith('.anat'):
            outname = outname[:-5]
        cmd = 'fsl_anat -i {} -o {}'.format(struct, outname)
        subprocess.run(cmd, shell=True)
        outname += '.anat'

    else:
        outname = kwargs['anat']
    
    # Run the surface steps if reqd. 
    # Check the fullfov T1 exists within anat_dir
    if not op.isdir(op.join(outname, 'fs', 'surf')):
        fullfov = op.join(outname, 'T1_fullfov.nii.gz')

        if not op.isfile(fullfov):
            raise RuntimeError("Could not find T1_fullfov.nii.gz within anat_dir %s" 
                % outname)

        print("Adding FreeSurfer to fsl_anat dir at %s" % outname)
        utils._runFreeSurfer(fullfov, outname, debug)

    if not utils.check_anat_dir(outname): 
        raise RuntimeError("fsl_anat dir should be complete with surfaces") 

    print("fsl_anat dir at %s is now complete with surfaces" % outname)
    return outname 