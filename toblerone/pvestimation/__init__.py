"""PV estimation functions"""

import os.path as op 
import functools

import numpy as np 
import tqdm
import regtricks as rt 

from toblerone.pvestimation import estimators
from toblerone import utils, core
from toblerone.classes import ImageSpace, Surface, Hemisphere

@utils.enforce_and_load_common_arguments
def cortex(**kwargs):
    """
    Estimate PVs for L/R cortex. All arguments are kwargs.

    Required args: 
        ref (str): path to reference image for which PVs are required
        struct2ref (str/np.array): registration between structural 
            (surface) and reference space. Use 'I' for identity. 
        fsdir (str): path to a FreeSurfer subject directory
        LWS/LPS/RWS/RPS (str): individual paths to the surfaces,
            eg LWS = Left White surface, RPS = Right Pial surace
            To estimate for a single hemisphere, only provide surfaces
            for that side. 

    Optional args: 
        flirt (bool): denoting struct2ref is FLIRT transform. If so, set struct
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8 
        supersample (int/array): single or 3 values, supersampling factor
 
    Returns: 
        (np.array), 4D, size equal to the reference image, with the PVs arranged 
            GM/WM/non-brain in 4th dim
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

    ref_space = ImageSpace(kwargs['ref'])

    # Set supersampler and estimate. 
    if kwargs.get('supersample') is None:
        supersampler = np.ceil(ref_space.vox_size / 0.75).astype(np.int8)
    else: 
        supersampler = kwargs.get('supersample') * np.ones(3)

    pvs = estimators._cortex(hemispheres, ref_space, kwargs['struct2ref'],
        supersampler, kwargs['cores'], bool(kwargs.get('ones')))

    return pvs

@utils.enforce_and_load_common_arguments
def structure(**kwargs):
    """
    Estimate PVs for a structure defined by a single surface. 
    All arguments are kwargs.
    
    Required args: 
        ref (str): path to reference image for which PVs are required
        struct2ref (str/np.array): registration between structural 
            (surface) and reference space. Use 'I' for identity. 
        surf (str): path to surface (see space argument below)

    Optional args: 
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
        
    ref_space = ImageSpace(kwargs['ref'])

    if kwargs.get('supersample') is None:
        supersampler = np.ceil(ref_space.vox_size / 0.75).astype(np.int8)
    else: 
        supersampler = kwargs.get('supersample') * np.ones(3)

    pvs = estimators._structure(surf, ref_space, kwargs['struct2ref'], 
        supersampler, bool(kwargs.get('ones')), kwargs['cores'])

    return pvs


def __structure_wrapper(surf, **kwargs):
    """Convenience method for parallel processing"""
    return structure(surf=surf, **kwargs)


@utils.enforce_and_load_common_arguments
def complete(**kwargs):
    """
    Estimate PVs for cortex and all structures identified by FIRST within 
    a reference image space. Use FAST to fill in non-surface PVs. 
    All arguments are kwargs.

    Required args: 
        ref (str): path to reference image for which PVs are required
        struct2ref (str/np.array): registration between structural 
            (surface) and reference space. Use 'I' for identity. 
        anat (str): path to fsl_fs_anat dir (see toblerone -fsl_fs_anat) 
        fsdir (str): path to a FreeSurfer subject directory
        LWS/LPS/RWS/RPS (str): individual paths to the surfaces,
            eg LWS = Left White surface, RPS = Right Pial surace
        firstdir (str): path to FSL FIRST directory 
        fastdir (str): path to FSL FAST results directory 

    Optional args: 
        space (str): space in which surface is defined: default is 'world' (mm coords),
            for FIRST surfaces set as 'first' and provide struct argument 
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8 
        supersample (int/array): single or 3 values, supersampling factor

    Returns: 
        (dict) PVs associated with each individual structure and 
            also the overall combined result ('stacked')
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
    s2r = rt.Registration(kwargs['struct2ref'])
    output = { t: s2r.apply_to_image(fasts[t], kwargs['ref'], superlevel=2).get_data()
        for t in ['FAST_WM', 'FAST_GM'] }
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
    estimator = functools.partial(__structure_wrapper, **kwargs)

    # This is equivalent to a map(estimator, structures) call
    # All the extra stuff (tqdm etc) is used for progress bar
    results = [ pv for _, pv in 
        tqdm.tqdm(enumerate(map(estimator, structures)), 
        total=len(structures), desc=desc, bar_format=core.BAR_FORMAT, 
        ascii=True) ] 

    output.update(dict(zip([s.name for s in structures], results)))

    # Now do the cortex, then stack the whole lot 
    ctx  = cortex(**kwargs)
    for i,t in enumerate(['_GM', '_WM', '_nonbrain']):
       output['cortex' + t] = (ctx[:,:,:,i])

    stacked = estimators.stack_images(
        {k:v for k,v in output.items() if k != 'BrStem'})
    output['GM'] = stacked[:,:,:,0]
    output['WM'] = stacked[:,:,:,1]
    output['nonbrain'] = stacked[:,:,:,2]
    output['stacked'] = stacked

    return output