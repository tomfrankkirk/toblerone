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
from toblerone.classes import ImageSpace, Surface, Hemisphere

@utils.enforce_and_load_common_arguments
def cortex(ref, struct2ref, **kwargs):
    """
    Estimate PVs for L/R cortex. All arguments are kwargs. To estimate for 
    a single hemisphere, provide only surfaces for that side. 

    Required args: 
        ref (str/regtricks ImageSpace): voxel grid in which to estimate PVs. 
        struct2ref (str/np.array/rt.Registration): registration between space 
            of surface and reference. Use 'I' for identity. 
        fsdir (str): path to a FreeSurfer subject directory. 
        LWS/LPS/RWS/RPS (str): individual paths to the surfaces,
            eg LWS = Left White surface, RPS = Right Pial surace. 

    Optional args: 
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct. 
        struct (str): path to structural image from which surfaces were derived. 
        cores (int): number of cores to use, default 8. 
        supersample (int/array): single or 3 values, supersampling factor. 
 
    Returns: 
        (np.array), 4D, size equal to the reference image, with the PVs arranged 
            GM/WM/non-brain in 4th dim. 
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
    if np.all([ (kwargs.get(s) is not None) for s in ['LPS', 'LWS'] ]): 
        sides.append('L')

    if np.all([ kwargs.get(s) is not None for s in ['RPS', 'RWS'] ]): 
        sides.append('R')

    if not sides:
        raise RuntimeError("At least one hemisphere (eg LWS/LPS required")

    # Load reference ImageSpace object
    hemispheres = [ Hemisphere(kwargs[s+'WS'], kwargs[s+'PS'], s) 
        for s in sides ] 

    # Either create local copy of ImageSpace object or init from path 
    if isinstance(ref, ImageSpace):
        ref_space = copy.deepcopy(ref)
    else:
        ref_space = ImageSpace(ref)

    # Set supersampler and estimate. 
    if kwargs.get('supersample') is None:
        supersampler = np.ceil(ref_space.vox_size.round(1) 
                                / 0.75).astype(np.int8)
    else: 
        supersampler = kwargs.get('supersample') * np.ones(3)

    pvs = estimators._cortex(hemispheres, ref_space, struct2ref,
        supersampler, kwargs['cores'], bool(kwargs.get('ones')))

    return pvs

@utils.enforce_and_load_common_arguments
def structure(ref, struct2ref, **kwargs):
    """
    Estimate PVs for a structure defined by a single surface. 
    All arguments are kwargs.
    
    Required args: 
        ref (str/regtricks ImageSpace): voxel grid in which to estimate PVs. 
        struct2ref (str/np.array/rt.Registration): registration between space 
            of surface and reference. Use 'I' for identity. 
        surf (str): path to surface (see space argument below)

    Optional args: 
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct. 
        space (str): space in which surface is defined: default is 'world' (mm coords),
            for FIRST surfaces set as 'first' and provide struct argument 
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8 
        supersample (int/array): single or 3 values, supersampling factor
 
    Returns: 
        (np.array) PV image, sized equal to reference space 
    """

    # Check we either have a surface object or path to one 
    if not bool(kwargs.get('surf')):
        raise RuntimeError("surf kwarg must be a Surface object or path to one")

    if not kwargs.get('space'): 
        kwargs['space'] = 'world'

    if kwargs['space'] == 'first' and not kwargs.get('struct'):
        raise RuntimeError("Structural image must be supplied for FIRST surfs")

    if type(kwargs['surf']) is str: 

        surf = Surface(kwargs['surf'], kwargs['space'], kwargs.get('struct'), 
            op.split(kwargs['surf'])[1])
    
    elif type(kwargs['surf']) is not Surface: 
        raise RuntimeError("surf kwarg must be a Surface object or path to one")

    else: 
        surf = kwargs['surf']
        
    # Either create local copy of ImageSpace object or init from path 
    if isinstance(ref, ImageSpace):
        ref_space = copy.deepcopy(ref)
    else:
        ref_space = ImageSpace(ref)

    if kwargs.get('supersample') is None:
        supersampler = np.ceil(ref_space.vox_size.round(1) 
                                / 0.75).astype(np.int8)
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
    All arguments are kwargs.

    Required args: 
        ref (str/regtricks ImageSpace): voxel grid in which to estimate PVs. 
        struct2ref (str/np.array/rt.Registration): registration between space 
            of surface and reference. Use 'I' for identity. 
        anat: path to augmented fsl_anat directory (see -fsl_fs_anat command).
            This REPLACES fsdir, firstdir, fastdir, LPS/RPS etc args 

    Alternatvies to anat argument (all required): 
        fsdir (str): FreeSurfer subject directory, OR: 
        LWS/LPS/RWS/RPS (str): paths to individual surfaces (L/R white/pial)
        firstdir (str): FIRST directory in which .vtk surfaces are located
        fastdir (str): FAST directory in which _pve_0/1/2 are located 
        struct (str): path to structural image from which surfaces were dervied

    Optional args: 
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct. 
        space (str): space in which surface is defined: default is 'world' (mm coords),
            for FIRST surfaces set as 'first' and provide struct argument 
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8 
        supersample (int/array): single or 3 values, supersampling factor

    Returns: 
        (dict) PVs associated with each individual structure and 
            also the overall combined result ('stacked')
    """

    print("Estimating PVs for", ref.file_name)

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
    structures = [ Surface(surf, 'first', kwargs['struct'], name) 
        for name, surf in FIRSTsurfs.items() ]
    print("Structures found: ", end=' ')
    [ print(s.name, end=' ') for s in structures ]
    print('Cortex')
    desc = 'Subcortical structures'

    # To estimate against each subcortical structure, we apply the following
    # partial func to each using a map() call. Carry kwargs from this func 
    estimator = functools.partial(__structure_wrapper, 
                                  ref=ref, struct2ref=struct2ref, **kwargs)

    # This is equivalent to a map(estimator, structures) call
    # All the extra stuff (tqdm etc) is used for progress bar
    results = [ pv for _, pv in 
        tqdm.tqdm(enumerate(map(estimator, structures)), 
        total=len(structures), desc=desc, bar_format=core.BAR_FORMAT, 
        ascii=True) ] 

    output.update(dict(zip([s.name for s in structures], results)))

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