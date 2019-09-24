# Functions to estimate the PVs of the cerebral cortex and subcortical 
# structures. These are wrappers around the core Toblerone functions 
# that handle the aggregate various pieces of information into overall results

import functools
import multiprocessing
import warnings
import tqdm

import numpy as np

from . import core, utils 
from .classes import Hemisphere, Surface, Patch


def _cortex(hemispheres, supersampler, cores, ones):
    """Estimate the PVs of the cortex. 

    Args: 
        hemispheres: a list of Hemisphere objects (one or two), each 
            containing the appropriate surface objects (in voxel coords)
        refSpace: an ImageSpace within which PVs are required
        supersampler: supersampling factor (3-vector) to use for estimation
        cores: number of processor cores to use
        ones: debug tool, write ones in all voxels containing triangles

    Returns: 
        (PVs, mask) 
            PVs is 4D array with the PVs arranged GM/WM/non-brain in 4th dim
            mask is boolean mask denoting intersection with any cortical surf
    """

    surfs = [ s for h in hemispheres for s in h.surfs() ]
    ref_space = surfs[0].index_space 
    if not all([ref_space is s.index_space for s in surfs]):
        raise RuntimeError("Surface must share common index_space")

    if ones:
        sz = (*ref_space.size, 3)        
        outPVs = np.zeros((np.prod(ref_space.size), 3), dtype=np.bool)
        ctxMask = np.zeros(np.prod(ref_space.size), dtype=np.bool)
        for s in surfs:
            outPVs[s.LUT,0] = 1 
            ctxMask[s.LUT] = 1 

        outPVs = outPVs.reshape(sz)
        ctxMask = ctxMask.reshape(ref_space.size)

    else: 

        # Estimate PV fractions for each surface
        for h in hemispheres:
            if np.any(np.max(np.abs(h.inSurf.points)) > 
                np.max(np.abs(h.outSurf.points))):
                raise RuntimeWarning("Inner surface vertices appear to be further",\
                    "from the origin than the outer vertices. Are the surfaces in",\
                    "the correct order?")
            
            for s, d in zip(h.surfs(), ['in', 'out']):
                descriptor = "{} cortex {}".format(h.side, d)
                f = core._estimateFractions(s, supersampler, descriptor, cores)
                s.fractions = f 

        # Merge the voxelisation results with PVs
        for h in hemispheres:
            inFractions = (h.inSurf.voxelised).astype(np.float32)
            outFractions = (h.outSurf.voxelised).astype(np.float32)
            inFractions[h.inSurf.LUT] = h.inSurf.fractions
            outFractions[h.outSurf.LUT] = h.outSurf.fractions

            # Combine estimates from each surface into whole hemi PV estimates
            hemiPVs = np.zeros((np.prod(ref_space.size), 3), dtype=np.float32)
            hemiPVs[:,1] = inFractions 
            hemiPVs[:,0] = np.maximum(0.0, outFractions - inFractions)
            hemiPVs[:,2] = 1.0 - np.sum(hemiPVs[:,0:2], axis=1)
            h.PVs = hemiPVs

        # Merge the hemispheres, giving priority to GM, then WM, then CSF.
        # Do nothing if just one hemi
        if len(hemispheres) == 1:
            outPVs = hemispheres[0].PVs

        else:
            h1, h2 = hemispheres
            outPVs = np.zeros((np.prod(ref_space.size), 3), dtype=np.float32)
            outPVs[:,0] = np.minimum(1.0, h1.PVs[:,0] + h2.PVs[:,0])
            outPVs[:,1] = np.minimum(1.0 - outPVs[:,0],
                h1.PVs[:,1] + h2.PVs[:,1])
            outPVs[:,2] = 1.0 - np.sum(outPVs[:,0:2], axis=1)

        # Sanity checks
        if np.any(outPVs > 1.0): 
            raise RuntimeError("PV exceeds 1")

        if np.any(outPVs < 0.0):
            raise RuntimeError("Negative PV returned")

        if not np.all(np.sum(outPVs, axis=1) == 1.0):
            raise RuntimeError("PVs do not sum to 1")

        # Form the surface mask (3D logical) as any voxel containing GM or 
        # intersecting the cortex (these definitions should always be equivalent)
        ctxMask = np.zeros((outPVs.shape[0], 1), dtype=bool)
        for h in hemispheres:
            for s in h.surfs(): 
                ctxMask[s.LUT] = True
        ctxMask[outPVs[:,0] > 0] = True 

        # Reshape images back into 4D or 3D images
        outPVs = np.reshape(outPVs, (*ref_space.size, 3))
        ctxMask = np.reshape(ctxMask, tuple(ref_space.size[0:3]))

    return outPVs, ctxMask


def _structure(cores, supersampler, ones, surf):
    """Estimate the PVs of a structure denoted by a single surface. Note
    that the results should be interpreted simply as "fraction of each 
    voxel lying within the structure", and it is ambiguous as to what tissue
    lies outside the structure

    Args: 
        cores: number of processor cores to use
        supersampler: supersampling factor (3-vector) to use for estimation
        ones: debug tool, write ones in voxels containing triangles 
        surf: a surface that has been indexed; PVs will be estimated within 
            this space. 

    Returns: 
        an array of size refSpace.size containing the PVs. 
    """

    if not surf.index_space:
        raise RuntimeError("Surface has not been indexed into a space." + 
            "See Surface.index_for()")

    size = surf.index_space.size 

    if ones: 
        outPVs = np.zeros(np.prod(size), dtype=np.bool)
        outPVs[surf.LUT] = 1 
        outPVs = outPVs.reshape(size)

    else:
        desc = '' 
        fractions = core._estimateFractions(surf, supersampler, desc, cores)
        outPVs = surf.voxelised.astype(np.float32)
        outPVs[surf.LUT] = fractions 
        outPVs = outPVs.reshape(size)

    return outPVs