# Functions to estimate the PVs of the cerebral cortex and subcortical 
# structures. These are wrappers around the core Toblerone functions 
# that handle the aggregate various pieces of information into overall results

import warnings
import copy 

import numpy as np

from toblerone.utils import STRUCTURES, NP_FLOAT

def _cortex(hemispheres, space, struct2ref, supersampler, cores, ones):
    """Estimate the PVs of the cortex. 

    Args: 
        hemispheres: either a single, or iterable list of, Hemisphere objects.
        space: an ImageSpace within which to operate.
        struct2ref: np.array affine transformation into reference space. 
        supersampler: supersampling factor (3-vector) to use for estimation.
        cores: number of processor cores to use.
        ones: debug tool, write ones in all voxels containing triangles.

    Returns: 
        4D array, size equal to the reference image, with the PVs arranged 
            GM/WM/non-brain in 4th dim
    """

    # Create our own local copy of inputs 
    loc_hemispheres = copy.deepcopy(hemispheres)

    if not isinstance(loc_hemispheres, list):
        loc_hemispheres = [loc_hemispheres]

    for h in loc_hemispheres: 
        h.inSurf, h.outSurf = [ s.transform(struct2ref) for s in h.surfs ]

    surfs = [ s for h in loc_hemispheres for s in h.surfs ]
    for s in surfs: 
        s.index_on(space, cores)
        s.indexed.voxelised = s.voxelise(space, cores)

    # Estimate PV fractions for each surface
    for h in loc_hemispheres:
        if np.any(np.max(np.abs(h.inSurf.points)) > 
            np.max(np.abs(h.outSurf.points))):
            warnings.warn("Inner surface vertices appear to be further" + 
                " from the origin than the outer vertices. Are the surfaces in" + 
                " the correct order?")
        
        for s, d in zip(h.surfs, ['in', 'out']):
            descriptor = "{} cortex {}".format(h.side, d)
            s._estimate_fractions(supersampler, cores, ones, descriptor)

    # Merge the voxelisation results with PVs
    for h in loc_hemispheres:
        in_pvs = h.inSurf.output_pvs(space).flatten()
        out_pvs = h.outSurf.output_pvs(space).flatten()

        # Combine estimates from each surface into whole hemi PV estimates
        hemiPVs = np.zeros((np.prod(space.size), 3), dtype=NP_FLOAT)
        hemiPVs[:,1] = in_pvs 
        hemiPVs[:,0] = np.maximum(0.0, out_pvs - in_pvs)
        hemiPVs[:,2] = 1.0 - (hemiPVs[:,0:2].sum(1))
        h.PVs = hemiPVs

    # Merge the hemispheres, giving priority to GM, then WM, then CSF.
    # Do nothing if just one hemi
    if len(loc_hemispheres) == 1:
        outPVs = loc_hemispheres[0].PVs

    else:
        h1, h2 = loc_hemispheres
        outPVs = np.zeros((np.prod(space.size), 3), dtype=NP_FLOAT)
        outPVs[:,0] = np.minimum(1.0, h1.PVs[:,0] + h2.PVs[:,0])
        outPVs[:,1] = np.minimum(1.0 - outPVs[:,0],
            h1.PVs[:,1] + h2.PVs[:,1])
        outPVs[:,2] = 1.0 - outPVs[:,0:2].sum(1)

    # Sanity checks
    if np.any(outPVs > 1.0): 
        raise RuntimeError("PV exceeds 1")

    if np.any(outPVs < 0.0):
        raise RuntimeError("Negative PV returned")

    if not np.all(outPVs.sum(1) == 1.0):
        raise RuntimeError("PVs do not sum to 1")

    return outPVs.reshape((*space.size, 3))


def _structure(surf, space, struct2ref, supersampler, ones, cores):
    """
    Estimate the PVs of a structure denoted by a single surface. Note
    that the results should be interpreted simply as "fraction of each 
    voxel lying within the structure", and it is ambiguous as to what tissue
    lies outside the structure

    Args: 
        surf: Surface object 
        space: ImageSpace to estimate within 
        struct2ref: np.array affine transformation into reference space. 
        supersampler: supersampling factor (3-vector) to use for estimation
        ones: debug tool, write ones in voxels containing triangles 
        cores: number of processor cores to use

    Returns: 
        an array of size refSpace.size containing the PVs. 
    """

    # Create our own local copy of inputs 
    loc_surf = copy.deepcopy(surf)
    loc_surf = loc_surf.transform(struct2ref)
    loc_surf.index_on(space, cores)
    loc_surf.indexed.voxelised = loc_surf.voxelise(space, cores)
    loc_surf._estimate_fractions(supersampler, cores, ones)
    
    return loc_surf.output_pvs(space)


def stack_images(images):
    """
    Combine the results of estimate_complete() into overall PV maps
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

    # The logic is as follows: 
    # Initialise everything as non-brain
    # Write in PV estimates from the cortical surfaces. This sets the cortical GM
    # and also all subcortical tissue as WM 
    # Layer on top FAST's CSF estimates as non-brain. This is because surface methods
    # dom't pick up ventricular or mid-brain CSF, so wherever FAST has found more CSF
    # than Toblerone we will take that higher estimate 
    # Layer in subcortical GM from each individual FIRST structure (except brain stem). 
    # After adding in each structure's GM, recalculate CSF as either the existing amount, 
    # or reduce the CSF estimate if the total (GM + CSF) > 1
    # If there is any remainder unassigned in the voxel, set that as WM

    # To summarise, the tissues are stacked up as: 
    # All CSF 
    # Cortical GM
    # Then within the subcortex only: 
        # All subcortical volume set as WM 
        # Subcortical CSF fixed using FAST 
        # Add in subcortical GM for each structure, reducing CSF if required 
        # Set the remainder 1 - (GM+CSF) as WM in voxels that were updated 

    # Copy the dict of images as we are going to make changes and dont want 
    # to play with the caller's copy. Pop unwanted images
    reqd_keys = STRUCTURES + [ 'FAST_GM', 'FAST_WM', 'FAST_CSF', 
        'cortex_GM', 'cortex_WM', 'cortex_nonbrain' ]
    reqd_keys.remove('BrStem')
    for k in reqd_keys:
        if k not in images.keys():
            raise RuntimeError(f"Did not find '{k}' key in images dict")
    
    # Pop out FAST's estimates  
    csf = images.pop('FAST_CSF').flatten()
    wm = images.pop('FAST_WM').flatten()
    gm = images.pop('FAST_GM')
    shape = (*gm.shape[0:3], 3)

    # Squash small FAST CSF values 
    csf[csf < 0.01] = 0 

    # Pop the cortex estimates and initialise output as all CSF
    ctxgm = images.pop('cortex_GM').flatten()
    ctxwm = images.pop('cortex_WM').flatten()
    ctxnon = images.pop('cortex_nonbrain').flatten()
    ctx = np.vstack((ctxgm, ctxwm, ctxnon)).T
    out = np.zeros_like(ctx)
    out[:,2] = 1

    # Then write in Toblerone's cortex estimates from all voxels
    # that contain either WM or GM (on the ctx image)
    mask = np.logical_or(ctx[:,0], ctx[:,1])
    out[mask,:] = ctx[mask,:]

    # Layer in FAST's CSF estimates (to get mid-brain and ventricular CSF). 
    # Where FAST has suggested a higher CSF estimate than currently exists, 
    # and the voxel does not intersect the cortical ribbon, accept FAST's 
    # estimate. Then update the WM estimates, reducing where necessary to allow
    # for the greater CSF volume
    GM_threshold = 0.01 
    ctxmask = (ctx[:,0] > GM_threshold)
    to_update = np.logical_and(csf > out[:,2], ~ctxmask)
    tmpwm = out[to_update,1]
    out[to_update,2] = np.minimum(csf[to_update], 1 - out[to_update,0])
    out[to_update,1] = np.minimum(tmpwm, 1 - (out[to_update,2] + out[to_update,0]))

    # Sanity checks: total tissue PV in each vox should sum to 1
    assert (out[to_update,0] <= GM_threshold).all(), 'Some update voxels have GM'
    assert (np.abs(out.sum(1) - 1) < 1e-6).all(), 'Voxel PVs do not sum to 1'
    assert (out > -1e-6).all(), 'Negative PV found'

    # For each subcortical structure, create a mask of the voxels which it 
    # relates to. The following operations then apply only to those voxels 
    # All subcortical structures interpreted as pure GM 
    # Update CSF to ensure that GM + CSF in those voxels < 1 
    # Finally, set WM as the remainder in those voxels.
    for k,s in images.items():
        smask = (s.flatten() > 0)
        out[smask,0] = np.minimum(1, out[smask,0] + s.flatten()[smask])
        out[smask,2] = np.minimum(out[smask,2], 1 - out[smask,0])
        out[smask,1] = np.maximum(1 - (out[smask,0] + out[smask,2]), 0)
        assert (out > -1e-6).all(), f'Negative found after {k} layer'
        assert (np.abs(out.sum(1) - 1) < 1e-6).all(), f'PVs sum > 1 after {k} layer'

    # Final sanity check, then rescaling so all voxels sum to unity. 
    # assert (out > -1e-6).all()
    out[out < 0] = 0 
    sums = out.sum(1)
    assert (np.abs(out.sum(1) - 1) < 1e-6).all(), 'Voxel PVs do not sum to 1'
    out = out / sums[:,None]

    return out.reshape(shape)
