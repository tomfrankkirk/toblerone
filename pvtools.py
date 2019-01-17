import os
import os.path as op
import argparse
import itertools
import multiprocessing
import warnings
import functools

import nibabel
import numpy as np
import tqdm

from pvtools import toblerone
from pvtools import pvcore
from .classes import ImageSpace, Hemisphere, Structure
from .classes import Surface, CommonParser, STRUCTURES
from pvtools import estimators 
from pvtools import fileutils

def estimate_all(**kwargs):

    kwargs = enforce_and_load_common_arguments(**kwargs)

    if kwargs.get('outdir'):
        fileutils.weak_mkdir(kwargs.get('outdir'))
    else: 
        kwargs['outdir'] = op.join(op.dirname(kwargs['ref']), 'pvtools')
        fileutils.weak_mkdir(kwargs['outdir'])

    # TODO: we could run these in parallel
    # Use a logical map to determine which need running, 
    # then prepare func calls and args, and run on pool

    # Run first if not given a FS dir
    if not kwargs.get('FSdir'):
        kwargs['FSdir'] = op.join(kwargs.get('outdir'), 'fs')
        fileutils.weak_mkdir(kwargs['FSdir'])
        pvcore._runFreeSurfer(kwargs['struct'], kwargs['FSdir'])

    # Run first if not given a first dir
    if not kwargs.get('firstdir'):
        kwargs['firstdir'] = op.join(kwargs['outdir'], 'first')
        fileutils.weak_mkdir(kwargs['firstdir'])
        pvcore._runFIRST(kwargs['struct'], kwargs['firstdir'])      

    # Process subcortical structures first. 
    FIRSTsurfs = fileutils.loadFIRSTdir(kwargs['firstdir'])
    structures = [ Structure(n, s, 'first', kwargs['struct']) 
        for n, s in FIRSTsurfs.items() ]
    print("The following structures will be estimated:")
    [ print(s.name, end=' ') for s in structures ]
    results = []
    desc = 'Subcortical structures'
    estimator = functools.partial(estimate_structure_wrapper, 
        **kwargs)

    # if kwargs['cores'] > 1:
    #     with multiprocessing.Pool(kwargs['cores']) as p: 
    #         for r in tqdm.tqdm(p.imap(estimator, structures), 
    #             total=len(structures), desc=desc, 
    #             bar_format=toblerone.BAR_FORMAT, ascii=True):
    #             results.append(r)

    # else: 
    #     for idx in tqdm.trange(len(structures), desc=desc, 
    #         bar_format= toblerone.BAR_FORMAT, ascii=True):
    #         results.append(estimator(substruct=structures[idx]))

    # # Save each individual output. 
    # refSpace = ImageSpace(kwargs['ref'])
    # for s, r in zip(structures, results):
    #     outpath = op.join(kwargs['outdir'], s.name + '_pvs.nii.gz')
    #     refSpace.saveImage(r, outpath)

    # And now do the cortex
    estimate_cortex(**kwargs)




def estimate_structure_wrapper(substruct, **kwargs):
    return estimate_structure(substruct=substruct, **kwargs)


def estimate_structure(**kwargs):

    kwargs = enforce_and_load_common_arguments(**kwargs)
    # Check we either have a substruct or surfpath
    if not any([
        kwargs.get('substruct') is not None, 
        kwargs.get('surf') is not None]):
        raise RuntimeError("A path to a surface must be given.")

    # if not kwargs.get('space'):
    #     raise RuntimeError("Surface coordinate space must be provided (world/first")

    if kwargs.get('substruct') is None:
        # We will create a struct using the surf path 
        surfname = op.splitext(op.split(kwargs['surf'])[1])[0]
        substruct = Structure(surfname, kwargs['surf'], kwargs.get('space'), 
            kwargs['struct'])
        
    else: 
        substruct = kwargs['substruct']

    refSpace = ImageSpace(kwargs['ref'])
    supersampler = np.ceil(refSpace.voxSize).astype(np.int8)

    overall = np.matmul(refSpace.world2vox, kwargs['struct2ref'])
    substruct.surf.applyTransform(overall)

    return estimators.structure(refSpace, 1, supersampler, substruct)



def estimate_cortex(**kwargs):

    """Estimate partial volumes on the cortical ribbon"""
    kwargs = enforce_and_load_common_arguments(**kwargs)
    if not any([
        kwargs.get('FSdir') is not None, 
        any([ kwargs.get(s) is not None 
            for s in ['LWS', 'LPS', 'RWS', 'RPS'] ]) ]):
        raise RuntimeError("Either a FSdir or paths to LWS/LPS etc"
            "must be given.")

    if kwargs.get('cores') is not None:
        cores = kwargs['cores']
    else: 
        cores = multiprocessing.cpu_count() - 1

    # If subdir given, then get all the surfaces out of the surf dir
    # If individual surface paths were given they will already be in scope
    if kwargs.get('FSdir'):
        surfdict = fileutils._loadSurfsToDict(kwargs['FSdir'])
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
    # then calc cross prods 
    hemispheres = [ Hemisphere(kwargs[s+'WS'], kwargs[s+'PS'], s) 
        for s in sides ]    
    surfs = [ s for h in hemispheres for s in h.surfs() ]
    overall = np.matmul(refSpace.world2vox, kwargs['struct2ref'])
    for s in surfs:
        s.applyTransform(overall)
        s.calculateXprods()

    # Set supersampler and estimate. 
    supersampler = np.ceil(refSpace.voxSize).astype(np.int8)
    outPVs, cortexMask = estimators.cortex(hemispheres, refSpace, supersampler, cores)

    return (outPVs, cortexMask)



def resample(src, ref, out, aff=np.identity(4), flirt=False):

    
    if flirt:
        src2ref = pvcore._adjustFLIRT(src, ref, aff)

    refSpace = ImageSpace(ref)
    factor = np.ceil(refSpace.voxSize).astype(np.int8)
    resamp = pvcore._superResampleImage(ref, factor, refSpace, src2ref)
    refSpace.saveImage(resamp, out)




def enforce_and_load_common_arguments(**kwargs):
    
    # Reference image path 
    if not kwargs.get('ref'):
        raise RuntimeError("Path to reference image must be given")

    if not op.isfile(kwargs['ref']):
        raise RuntimeError("Reference image does not exist")

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
                    if matExt == '.txt':
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
    if kwargs.get('flirt'):
        if not kwargs.get('struct'):
            raise RuntimeError("If using a FLIRT transform, the path to the \
                structural image must also be given")
        if not op.isfile(kwargs['struct']):
            raise RuntimeError("Structural image does not exist")
        kwargs['struct2ref'] = pvcore._adjustFLIRT(kwargs['struct'], kwargs['ref'], 
            kwargs['struct2ref'])

    if not kwargs.get('cores'):
        kwargs['cores'] = max([multiprocessing.cpu_count() - 1, 1])

    return kwargs