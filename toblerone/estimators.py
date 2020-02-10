# Functions to estimate the PVs of the cerebral cortex and subcortical 
# structures. These are wrappers around the core Toblerone functions 
# that handle the aggregate various pieces of information into overall results

import warnings
import copy 

import numpy as np

from . import core, utils 
from .classes import Hemisphere, Surface, ImageSpace

def _cortex(hemispheres, space, struct2ref, supersampler, cores, ones):
    """Estimate the PVs of the cortex. 

    Args: 
        hemispheres: either a single, or iterable list of, Hemisphere objects 
        space: an ImageSpace within which to operate
        struct2ref: 4x4 affine transformation from struct to surface space
        supersampler: supersampling factor (3-vector) to use for estimation
        cores: number of processor cores to use
        ones: debug tool, write ones in all voxels containing triangles

    Returns: 
        (PVs, mask) 
            PVs is 4D array with the PVs arranged GM/WM/non-brain in 4th dim
            mask is boolean mask denoting intersection with any cortical surf
    """

    # Create our own local copy of inputs 
    loc_hemispheres = copy.deepcopy(hemispheres)

    if not isinstance(loc_hemispheres, list):
        loc_hemispheres = [loc_hemispheres]
        
    surfs = [ s for h in loc_hemispheres for s in h.surfs ]

    for s in surfs: 
        s.index_on(space, struct2ref, cores)

    # Estimate PV fractions for each surface
    for h in loc_hemispheres:
        if np.any(np.max(np.abs(h.inSurf.points)) > 
            np.max(np.abs(h.outSurf.points))):
            warnings.warn("Inner surface vertices appear to be further",\
                "from the origin than the outer vertices. Are the surfaces in",\
                "the correct order?")
        
        for s, d in zip(h.surfs, ['in', 'out']):
            descriptor = "{} cortex {}".format(h.side, d)
            s._estimate_fractions(supersampler, cores, ones, descriptor)

    # Merge the voxelisation results with PVs
    for h in loc_hemispheres:
        in_pvs = h.inSurf.output_pvs(space).flatten()
        out_pvs = h.outSurf.output_pvs(space).flatten()

        # Combine estimates from each surface into whole hemi PV estimates
        hemiPVs = np.zeros((np.prod(space.size), 3), dtype=np.float32)
        hemiPVs[:,1] = in_pvs 
        hemiPVs[:,0] = np.maximum(0.0, out_pvs - in_pvs)
        hemiPVs[:,2] = 1.0 - np.sum(hemiPVs[:,0:2], axis=1)
        h.PVs = hemiPVs

    # Merge the hemispheres, giving priority to GM, then WM, then CSF.
    # Do nothing if just one hemi
    if len(loc_hemispheres) == 1:
        outPVs = loc_hemispheres[0].PVs

    else:
        h1, h2 = loc_hemispheres
        outPVs = np.zeros((np.prod(space.size), 3), dtype=np.float32)
        outPVs[:,0] = np.minimum(1.0, h1.PVs[:,0] + h2.PVs[:,0])
        outPVs[:,1] = np.minimum(1.0 - outPVs[:,0],
            h1.PVs[:,1] + h2.PVs[:,1])
        outPVs[:,2] = 1.0 - np.sum(outPVs[:,0:2], axis=1)

    # Sanity checks
    if np.any(outPVs > 1.0): 
        raise RuntimeError("PV exceeds 1")

    if np.any(outPVs < 0.0):
        raise RuntimeError("Negative PV returned")

    if not np.all(outPVs.sum(1) == 1.0):
        raise RuntimeError("PVs do not sum to 1")

    # Form the surface mask (3D logical) as any voxel containing GM or 
    # intersecting the cortex (these definitions should always be equivalent)
    ctxMask = np.zeros((outPVs.shape[0], 1), dtype=bool)
    for h in loc_hemispheres:
        for s in h.surfs: 
            ctxMask[s.reindex_LUT(space)] = True
    ctxMask[outPVs[:,0] > 0] = True 

    # Reshape images back into 4D or 3D images
    outPVs = outPVs.reshape((*space.size, 3))
    ctxMask = ctxMask.reshape(space.size)

    return outPVs, ctxMask


def _structure(surf, space, struct2ref, supersampler, ones, cores):
    """
    Estimate the PVs of a structure denoted by a single surface. Note
    that the results should be interpreted simply as "fraction of each 
    voxel lying within the structure", and it is ambiguous as to what tissue
    lies outside the structure

    Args: 
        surf: Surface object 
        space: ImageSpace to estimate within 
        supersampler: supersampling factor (3-vector) to use for estimation
        ones: debug tool, write ones in voxels containing triangles 
        cores: number of processor cores to use

    Returns: 
        an array of size refSpace.size containing the PVs. 
    """

    # Create our own local copy of inputs 
    loc_surf = copy.deepcopy(surf)

    loc_surf.index_on(space, struct2ref, cores)
    loc_surf._estimate_fractions(supersampler, cores, ones)
    
    return loc_surf.output_pvs(space)