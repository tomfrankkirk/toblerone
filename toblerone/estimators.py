# Functions to estimate the PVs of the cerebral cortex and subcortical 
# structures. These are wrappers around the core Toblerone functions 
# that handle the aggregate various pieces of information into overall results

import functools
import multiprocessing
import warnings
import tqdm

import numpy as np

from . import core 
from .classes import Hemisphere, Surface, Patch


def _cortex(hemispheres, refSpace, supersampler, cores, ones):
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
    FoVoffset, FoVsize = core._determineFullFoV(surfs, refSpace)

    for s in surfs:
        s.shiftFoV(FoVoffset, FoVsize)
        s.formAssociations(FoVsize, cores)
        s.calculateXprods()

    if ones:
        sz = np.concatenate((refSpace.imgSize, [3]))
        outPVs = np.zeros((np.prod(refSpace.imgSize), 3), dtype=np.bool)
        ctxMask = np.zeros(np.prod(refSpace.imgSize), dtype=np.bool)
        for s in surfs:
            outPVs[s.LUT,0] = 1 
            ctxMask[s.LUT] = 1 

        outPVs = outPVs.reshape(sz)
        ctxMask = ctxMask.reshape(refSpace.imgSize)

    else: 
        # Voxelise the surfaces (ie, no PVs), then store the results 
        # as the 'voxelised' attr of the surfaces 
        print('Voxelising')
        voxelise = functools.partial(core.voxelise, FoVsize)
        fills = []
        if cores > 1:
            with multiprocessing.Pool(min([cores, len(surfs)])) as p: 
                for _, r in enumerate(p.imap(voxelise, surfs)):
                    fills.append(r)
        else: 
            for surf in surfs:
                fills.append(voxelise(surf))

        [ setattr(s, 'voxelised', f) for (s,f) in zip(surfs, fills) ]

        # Estimate PV fractions for each surface
        for h in hemispheres:
            if np.any(np.max(np.abs(h.inSurf.points)) > 
                np.max(np.abs(h.outSurf.points))):
                raise RuntimeWarning("Inner surface vertices appear to be further",\
                    "from the origin than the outer vertices. Are the surfaces in",\
                    "the correct order?")
            
            for s, d in zip(h.surfs(), ['in', 'out']):
                descriptor = " {} cortex {}".format(h.side, d)
                f = core._estimateFractions(s, FoVsize, supersampler, 
                    descriptor, cores)
                s.fractions = f 

        # Merge the voxelisation results with PVs
        for h in hemispheres:
            inFractions = (h.inSurf.voxelised).astype(np.float32)
            outFractions = (h.outSurf.voxelised).astype(np.float32)
            inFractions[h.inSurf.LUT] = h.inSurf.fractions
            outFractions[h.outSurf.LUT] = h.outSurf.fractions

            # Combine estimates from each surface into whole hemi PV estimates
            hemiPVs = np.zeros((np.prod(FoVsize), 3), dtype=np.float32)
            hemiPVs[:,1] = inFractions 
            hemiPVs[:,0] = np.maximum(0.0, outFractions - inFractions)
            hemiPVs[:,2] = 1.0 - np.sum(hemiPVs[:,0:2], axis=1)

            # And write into the hemisphere object. 
            h.PVs = hemiPVs

        # Merge the hemispheres, giving priority to GM, then WM, then CSF.
        if len(hemispheres) == 1:
            outPVs = hemispheres[0].PVs
        else:
            h1, h2 = hemispheres
            outPVs = np.zeros((np.prod(FoVsize), 3), dtype=np.float32)
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
        outPVs = np.reshape(outPVs, (*FoVsize, 3))
        ctxMask = np.reshape(ctxMask, tuple(FoVsize[0:3]))

        # Extract the output within the FoV of the reference image
        outPVs = outPVs[ FoVoffset[0] : FoVoffset[0] + refSpace.imgSize[0],
            FoVoffset[1] : FoVoffset[1] + refSpace.imgSize[1],
            FoVoffset[2] : FoVoffset[2] + refSpace.imgSize[2], : ]
        ctxMask = ctxMask[ FoVoffset[0] : FoVoffset[0] + refSpace.imgSize[0],
            FoVoffset[1] : FoVoffset[1] + refSpace.imgSize[1],
            FoVoffset[2] : FoVoffset[2] + refSpace.imgSize[2] ]

    return outPVs, ctxMask


def _structure(refSpace, cores, supersampler, ones, surf):
    """Estimate the PVs of a structure denoted by a single surface. Note
    that the results should be interpreted simply as "fraction of each 
    voxel lying within the structure", and it is ambiguous as to what tissue
    lies outside the structure

    Args: 
        refSpace: an ImageSpace within which PVs are required
        cores: number of processor cores to use
        supersampler: supersampling factor (3-vector) to use for estimation
        ones: debug tool, write ones in voxels containing triangles 

    Returns: 
        an array of size refSpace.imgSize containing the PVs. 
    """

    surf.calculateXprods()
    FoVoffset, FoVsize = core._determineFullFoV([surf], refSpace)
    surf.shiftFoV(FoVoffset, FoVsize)
    surf.formAssociations(FoVsize, cores)

    if not surf.LUT.size:
        warnings.warn("Surface {} does not lie within reference space"
            .format(surf.name))

    if ones: 
        sz = refSpace.imgSize
        outPVs = np.zeros(np.prod(sz), dtype=np.bool)
        outPVs[surf.LUT] = 1 
        outPVs = outPVs.reshape(sz)

    else:
        surf.voxelised = core.voxelise(FoVsize, surf)
        desc = '' 
        fractions = core._estimateFractions(surf, FoVsize, 
            supersampler, desc, cores)

        outPVs = surf.voxelised.astype(np.float32)
        outPVs[surf.LUT] = fractions 
        outPVs = outPVs.reshape(*FoVsize)

        # Extract the output within the FoV of the reference image
        outPVs = outPVs[ FoVoffset[0] : FoVoffset[0] + refSpace.imgSize[0], \
            FoVoffset[1] : FoVoffset[1] + refSpace.imgSize[1], \
            FoVoffset[2] : FoVoffset[2] + refSpace.imgSize[2] ]

    return outPVs