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


def _cortex(hemispheres, refSpace, supersampler, cores, zeros):
    """Estimate the PVs of the cortex. 

    Args: 
        hemispheres: a list of Hemisphere objects (one or two), each 
            containing the appropriate surface objects (in voxel coords)
        refSpace: an ImageSpace within which PVs are required
        supersampler: supersampling factor (3-vector) to use for estimation
        cores: number of processor cores to use
        zeros: debug tool, write zeros everywhere

    Returns: 
        (PVs, mask) 
            PVs is 4D array with the PVs arranged GM/WM/non-brain in 4th dim
            mask is boolean mask denoting intersection with any cortical surf
    """

    if zeros:
        sz = np.concatenate((refSpace.imgSize, [3]))
        outPVs = np.zeros(sz, dtype=np.float32)
        mask = np.zeros(sz[0:3])
        return (outPVs, mask)

    surfs = [ s for h in hemispheres for s in h.surfs() ]
    FoVoffset, FoVsize = core._determineFullFoV(surfs, refSpace)

    for s in surfs:
        s.shiftFoV(FoVoffset, FoVsize)
        s.formAssociations(FoVsize, cores)
        s.calculateXprods()

    # Check surface normals and flip if required
    # if not core._checkSurfaceNormals(surfs[0], FoVsize):
    #     print("Inverse surface normals detected, flipping orientation")
    #     (s.flipXprods() for s in surfs)

    # Prepare for estimation. Generate list of voxels to process:
    # Start with grid, add offset, then flatten to linear indices. 
    voxList = core._getVoxList(refSpace.imgSize, FoVoffset, FoVsize)
    
    # Fill in whole voxels (ie, no PVs), then match the results of the map
    # to respective surfaces.
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

    # Estimate fractions for each surface
    for h in hemispheres:
        if np.any(np.max(np.abs(h.inSurf.points)) > 
            np.max(np.abs(h.outSurf.points))):
            raise RuntimeWarning("Inner surface vertices appear to be further",\
                "from the origin than the outer vertices. Are the surfaces in",\
                "the correct order?")
        
        for s, d in zip(h.surfs(), ['in', 'out']):
            descriptor = " {} cortex {}".format(h.side, d)
            s.flist = np.intersect1d(voxList, s.LUT).astype(np.int32)
            f = core._estimateFractions(s, FoVsize, supersampler, 
                s.flist, descriptor, cores)
            s.fractions = f 

    for h in hemispheres:
        inFractions = (h.inSurf.voxelised).astype(np.float32)
        outFractions = (h.outSurf.voxelised).astype(np.float32)
        inFractions[h.inSurf.flist] = h.inSurf.fractions
        outFractions[h.outSurf.flist] = h.outSurf.fractions

        # Combine estimates from each surface into whole hemi PV estimates
        hemiPVs = np.zeros((np.prod(FoVsize), 3), dtype=np.float32)
        hemiPVs[:,1] = inFractions 
        hemiPVs[:,0] = np.maximum(0.0, outFractions - inFractions)
        hemiPVs[:,2] = 1.0 - np.sum(hemiPVs[:,0:2], axis=1)

        # And write into the hemisphere object. 
        h.PVs = hemiPVs

    # Merge the fill masks by giving priority to GM, then WM, then CSF.
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
    outPVs = np.reshape(outPVs, (FoVsize[0], FoVsize[1], \
        FoVsize[2], 3))
    ctxMask = np.reshape(ctxMask, tuple(FoVsize[0:3]))

    # Extract the output within the FoV of the reference image
    outPVs = outPVs[ FoVoffset[0] : FoVoffset[0] + refSpace.imgSize[0],
        FoVoffset[1] : FoVoffset[1] + refSpace.imgSize[1],
        FoVoffset[2] : FoVoffset[2] + refSpace.imgSize[2], : ]
    ctxMask = ctxMask[ FoVoffset[0] : FoVoffset[0] + refSpace.imgSize[0],
        FoVoffset[1] : FoVoffset[1] + refSpace.imgSize[1],
        FoVoffset[2] : FoVoffset[2] + refSpace.imgSize[2] ]

    return outPVs, ctxMask


def _structure(refSpace, cores, supersampler, zeros, surf):
    """Estimate the PVs of a structure denoted by a single surface. Note
    that the results should be interpreted simply as "fraction of each 
    voxel lying within the structure", and it is ambiguous as to what tissue
    lies outside the structure

    Args: 
        refSpace: an ImageSpace within which PVs are required
        cores: number of processor cores to use
        supersampler: supersampling factor (3-vector) to use for estimation
        zeros: debug tool, write zeros everywhere 

    Returns: 
        an array of size refSpace.imgSize containing the PVs. 
    """

    if zeros:
        sz = refSpace.imgSize
        outPVs = np.zeros(sz, dtype=np.int32)
        return outPVs

    surf.calculateXprods()
    FoVoffset, FoVsize = core._determineFullFoV([surf], refSpace)
    surf.shiftFoV(FoVoffset, FoVsize)
    surf.formAssociations(FoVsize, cores)

    # Check surface normals and flip if required
    # if not core._checkSurfaceNormals(surf, FoVsize):
    #     print("Inverse surface normals detected, flipping orientation")
    #     surf.flipXprods()

    # Prepare for estimation. Generate list of voxels to process:
    # Start with grid, add offset, then flatten to linear indices. 
    voxList = core._getVoxList(refSpace.imgSize, FoVoffset, FoVsize)
    vlist = np.intersect1d(voxList, surf.LUT).astype(np.int32)
    if not vlist.size:
        warnings.warn("Surface {} does not lie within reference image"
            .format(surf.name))

    surf.voxelised = core.voxelise(FoVsize, surf)
    desc = '' 
    fractions = core._estimateFractions(surf, FoVsize, 
        supersampler, vlist, desc, cores)

    outPVs = surf.voxelised.astype(np.float32)
    outPVs[vlist] = fractions 
    outPVs = outPVs.reshape(FoVsize[0], FoVsize[1], FoVsize[2])

    # Extract the output within the FoV of the reference image
    outPVs = outPVs[ FoVoffset[0] : FoVoffset[0] + refSpace.imgSize[0], \
        FoVoffset[1] : FoVoffset[1] + refSpace.imgSize[1], \
        FoVoffset[2] : FoVoffset[2] + refSpace.imgSize[2] ]

    return outPVs