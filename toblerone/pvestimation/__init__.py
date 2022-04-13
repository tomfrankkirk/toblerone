"""PV estimation functions"""

import os.path as op 
import functools
import copy
import nibabel 

import numpy as np 
import tqdm
import regtricks as rt 

from toblerone.pvestimation import estimators
from toblerone import utils, core
from toblerone.classes import ImageSpace, Surface

@utils.enforce_and_load_common_arguments
def cortex(ref, struct2ref, **kwargs):
    """
    Estimate PVs for L/R cortex. To estimate for a single hemisphere, 
    provide only surfaces for that side. 

    Args: 
        ref (str/regtricks ImageSpace): voxel grid in which to estimate PVs. 
        struct2ref (str/np.array/rt.Registration): registration between space 
            of surface and reference [see `-flirt` and `-stuct`], use 'I' for identity. 
        fsdir (str): path to a FreeSurfer subject directory. 
        LWS/LPS/RWS/RPS (str): individual paths to the surfaces, 
            eg LWS = Left White surface, RPS = Right Pial surace. 

    Other parameters: 
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct. 
        struct (str): path to structural image from which surfaces were derived. 
        cores (int): number of cores to use, default 8. 
        supersample (int/array): single or 3 values, supersampling factor. 
 
    Returns: 
        (np.array) PVs arranged GM/WM/non-brain in 4th dim.
    """

    if not any([
        kwargs.get('fsdir') is not None, 
        any([ kwargs.get(s) is not None 
            for s in ['LWS', 'LPS', 'RWS', 'RPS'] ]) ]):
        raise RuntimeError("Either a fsdir or paths to LWS/LPS etc"
            "must be given.")

    hemispheres = utils.load_surfs_to_hemispheres(**kwargs)

    # Either create local copy of ImageSpace object or init from path 
    if isinstance(ref, ImageSpace):
        ref_space = copy.deepcopy(ref)
    else:
        ref_space = ImageSpace(ref)

    # Set supersampler and estimate. 
    if kwargs.get('supersample') is None:
        supersampler = np.maximum(np.floor(ref_space.vox_size.round(1) 
                                / 0.75), 1).astype(np.int32)
    else: 
        supersampler = kwargs.get('supersample') * np.ones(3)

    pvs = estimators._cortex(hemispheres, ref_space, struct2ref,
        supersampler, kwargs['cores'], bool(kwargs.get('ones')))

    return pvs

@utils.enforce_and_load_common_arguments
def structure(ref, struct2ref, surf, **kwargs):
    """
    Estimate PVs for a structure defined by a single surface. 
    
    Args: 
        ref (str/regtricks ImageSpace): voxel grid in which to estimate PVs. 
        struct2ref (str/np.array/rt.Registration): registration between space 
            of surface and reference (see -flirt and -stuct). Use 'I' for identity. 
        surf (str/`Surface`): path to surface (see coords argument below)

    Other parameters: 
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct. 
        coords (str): convention by which surface is defined: default is 'world' 
            (mm coords), for FIRST surfaces set as 'fsl' and provide struct argument 
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8 
        supersample (int/array): single or 3 values, supersampling factor
 
    Returns: 
        (np.array) PV image, sized equal to reference space 
    """

    coords = kwargs.get('coords', 'world')
    if coords == 'fsl' and not kwargs.get('struct'):
        raise RuntimeError("Structural image must be supplied for FIRST surfs")

    if type(surf) is str: 
        surf = Surface(surf, name=op.split(surf)[1])
        if kwargs.get('coords', 'world') == 'fsl':
            struct_spc = ImageSpace(kwargs['struct'])
            surf = surf.transform(struct_spc.FSL2world)

    elif type(surf) is not Surface: 
        raise RuntimeError("surf kwarg must be a Surface object or path to one")
        
    # Either create local copy of ImageSpace object or init from path 
    if isinstance(ref, ImageSpace):
        ref_space = copy.deepcopy(ref)
    else:
        ref_space = ImageSpace(ref)

    if kwargs.get('supersample') is None:
        supersampler = np.maximum(np.floor(ref_space.vox_size.round(1) 
                                / 0.75), 1).astype(np.int32)
    else: 
        supersampler = kwargs.get('supersample') * np.ones(3)

    pvs = estimators._structure(surf, ref_space, struct2ref, 
        supersampler, bool(kwargs.get('ones')), kwargs['cores'])

    return pvs


def __structure_wrapper(surf, **kwargs):
    """Convenience method for parallel processing"""
    return structure(surf=surf, **kwargs)


@utils.enforce_and_load_common_arguments
def complete(ref, struct2ref, **kwargs):
    """
    Estimate PVs for cortex and all structures identified by FIRST within 
    a reference image space. Use FAST to fill in non-surface PVs. 

    Args: 
        ref (str/regtricks.ImageSpace): voxel grid in which to estimate PVs. 
        struct2ref (str/np.array/regtricks.Registration): registration between space 
            of surface and reference (see -flirt and -stuct). Use 'I' for identity. 
        fslanat: path to fslanat directory. This REPLACES firstdir/fastdir/struct. 
        firstdir (str): FIRST directory in which .vtk surfaces are located
        fastdir (str): FAST directory in which _pve_0/1/2 are located 
        struct (str): path to structural image from which FIRST surfaces were dervied
        fsdir (str): FreeSurfer subject directory, OR: 
        LWS/LPS/RWS/RPS (str): paths to individual surfaces (L/R white/pial)

    Other parameters: 
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct. 
        coords (str): convention by which surface is defined: default is 'world' 
            (mm coords), for FIRST surfaces set as 'fsl' and provide struct argument 
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8 
        supersample (int/array): single or 3 values, supersampling factor

    Returns: 
        (dict) keyed for each structure and also flattened ('stacked')
    """

    print("Estimating PVs for", ref.file_name)

    # If anat dir then various subdirs are loaded by @enforce_common_args
    # If not then direct load below 
    if not bool(kwargs.get('fsdir')):
        if not all([ bool(kwargs.get(k)) for k in ['LWS','LPS','RWS','RPS'] ]):
            raise RuntimeError("If fsdir not given, " + 
                "provide paths for LWS,LPS,RWS,RPS")

    if not bool(kwargs.get('fslanat')):
        if not (bool(kwargs.get('fastdir')) and bool(kwargs.get('firstdir'))):
            raise RuntimeError("If not using fslanat dir, fastdir/firstdir required")
   
    # Resample FASTs to reference space. Then redefine CSF as 1-(GM+WM)
    fast_paths = utils._loadFASTdir(kwargs['fastdir'])
    fast_spc = fast_paths['FAST_GM']
    fast = np.stack([
        nibabel.load(fast_paths[f'FAST_{p}']).get_fdata() for p in ['GM', 'WM']
    ], axis=-1)
    fasts_transformed = rt.Registration(struct2ref).apply_to_array(fast, fast_spc, ref)
    output = dict(FAST_GM=fasts_transformed[...,0], FAST_WM=fasts_transformed[...,1])
    output['FAST_CSF'] = np.maximum(0, 1 - (output['FAST_WM'] + output['FAST_GM']))
        
    # Process subcortical structures first. 
    FIRSTsurfs = utils._loadFIRSTdir(kwargs['firstdir'])
    subcortical = []
    struct_spc = ImageSpace(kwargs['struct'])
    for name, surf in FIRSTsurfs.items(): 
        s = Surface(surf, name)
        s = s.transform(struct_spc.FSL2world)
        subcortical.append(s)
    
    disp = "Structures found: " + ", ".join([ 
        s.name for s in subcortical ] + ['Cortex'])
    print(disp)

    # To estimate against each subcortical structure, we apply the following
    # partial func to each using a map() call. Carry kwargs from this func 
    desc = 'Subcortical structures'
    estimator = functools.partial(__structure_wrapper, 
                                  ref=ref, struct2ref=struct2ref, **kwargs)

    # This is equivalent to a map(estimator, subcortical) call
    # All the extra stuff (tqdm etc) is used for progress bar
    results = [ pv for _, pv in 
        tqdm.tqdm(enumerate(map(estimator, subcortical)), 
        total=len(subcortical), desc=desc, bar_format=core.BAR_FORMAT, 
        ascii=True) ] 

    output.update(dict(zip([s.name for s in subcortical], results)))

    # Now do the cortex, then stack the whole lot 
    ctx  = cortex(ref=ref, struct2ref=struct2ref, **kwargs)
    for i,t in enumerate(['_GM', '_WM', '_nonbrain']):
       output['cortex' + t] = (ctx[:,:,:,i])

    stacked = estimators.stack_images(
        {k:v for k,v in output.items() if k != 'BrStem'})
    output['GM'] = stacked[:,:,:,0]
    output['WM'] = stacked[:,:,:,1]
    output['nonbrain'] = stacked[:,:,:,2]
    output['stacked'] = stacked

    return output