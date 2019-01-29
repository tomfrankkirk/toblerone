import os
import os.path as op
import argparse
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import warnings
import functools
import copy

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

    if not kwargs.get('outdir'):
        kwargs['outdir'] = op.join(op.dirname(kwargs['ref']), 'pvtools')
        fileutils.weak_mkdir(kwargs['outdir'])

    # Run first if not given a FS dir
    processes = []
    if not kwargs.get('FSdir'):
        kwargs['FSdir'] = op.join(kwargs.get('outdir'), 'fs')
        proc = multiprocessing.Process(
            target=pvcore._runFreeSurfer, 
            args=(kwargs['struct'], kwargs['outdir']))
        processes.append(proc)

    # Run first if not given a first dir
    if not kwargs.get('firstdir'):
        kwargs['firstdir'] = op.join(kwargs['outdir'], 'first')
        fileutils.weak_mkdir(kwargs['firstdir'])
        proc = multiprocessing.Process(
            target=pvcore._runFIRST, 
            args=(kwargs['struct'], kwargs['firstdir']))
        processes.append(proc)

    # Run FAST if not given a FAST dir
    if not kwargs.get('fastdir'):
        kwargs['fastdir'] = op.join(kwargs['outdir'], 'fast')
        fileutils.weak_mkdir(kwargs['fastdir'])
        proc = multiprocessing.Process(
            target=pvcore._runFAST, 
            args=(kwargs['struct'], kwargs['fastdir']))
        processes.append(proc)

    if len(processes) <= kwargs['cores']:
        [ p.start() for p in processes ]
        [ p.join() for p in processes ]
    else:
        for p in processes: 
            p.start()
            p.join()
      
    for kw in ['FSdir', 'firstdir', 'fastdir']:
        print("Using {}: {}".format(kw, kwargs[kw]))
   

    # Resample FASTs to reference space. 
    fasts = fileutils._loadFASTdir(kwargs['fastdir'])

    # Redefine FAST CSF
    GM = nibabel.load(fasts['FAST_GM']).get_fdata()
    WM = nibabel.load(fasts['FAST_WM']).get_fdata()
    CSF = 1 - (GM + WM)
    spc = ImageSpace(fasts['FAST_GM'])
    spc.saveImage(CSF, fasts['FAST_CSF'])

    for t, f in fasts.items():
        outpath = op.join(kwargs['outdir'], t + '.nii.gz')
        resample(f, kwargs['ref'], outpath, kwargs['struct2ref'])

    # Process subcortical structures first. 
    FIRSTsurfs = fileutils._loadFIRSTdir(kwargs['firstdir'])
    structures = [ Structure(n, s, 'first', kwargs['struct']) 
        for n, s in FIRSTsurfs.items() ]
    print("The following structures will be estimated:", flush=True)
    [ print(s.name, end=' ') for s in structures ]
    print('Cortex')
    desc = ' Subcortical structures'
    estimator = functools.partial(estimate_structure_wrapper, 
        **kwargs)

    results = []
    if kwargs['cores'] > 1:
        with multiprocessing.Pool(kwargs['cores']) as p: 
            for _, r in tqdm.tqdm(enumerate(p.imap(estimator, structures)), 
                total=len(structures), desc=desc, 
                bar_format=toblerone.BAR_FORMAT, ascii=True):
                    results.append(r)

    else: 
        for _, r in tqdm.tqdm(enumerate(map(estimator, structures)), 
            total=len(structures), desc=desc, 
            bar_format=toblerone.BAR_FORMAT, ascii=True):
                results.append(r)

    output = { k: o for (k,o) in zip(FIRSTsurfs.keys(), results) }

    # Now do the cortex
    ctx, ctxmask = estimate_cortex(**kwargs)
    output['cortex'] = ctx 
    output['cortexmask'] = ctxmask

    # Finally, flatten individual results onto single volume. 

    return output 


def stack_images(images):

    images = copy.deepcopy(images)
    
    ctx = images.pop('cortex')
    shape = ctx.shape
    ctx = ctx.reshape(-1,3)
    out = np.zeros_like(ctx)
    out[:,2] = 1

    csf = images.pop('FAST_CSF').flatten()
    wm = images.pop('FAST_WM').flatten()
    gm = images.pop('FAST_GM').flatten()

    # Method 1
    mask = np.logical_or(ctx[:,0], ctx[:,1])
    brain = np.logical_or(mask, csf>0.05)
    out[mask,:] = ctx[mask,:]

    out[:,2] = np.maximum(csf[:], out[:,2])
    out[:,1] = np.maximum(0, 1 - np.sum(out[:,0:4:2], axis=1))

    # out[brain,:] = (out[brain,:] / (np.sum(out[brain,:], axis=1)[:,None]))
    # assert np.all(np.abs(np.sum(out[brain,:], axis=1) - 1) < 1e-6)

    for s in images.values():
        smask = (s.flatten() > 0)
        out[smask,0] = np.minimum(1, out[smask,0] + s.flatten()[smask])
        out[smask,1] = np.maximum(0, 1 - out[smask,0])

    # out = out / (np.maximum(1, np.sum(out, axis=1))[:,None])

    # Method 2
    # mask = np.logical_and(wm, gm).flatten()
    # out[:,2] = csf.flatten() 
    # for s in images.values():
    #     out[:,0] = np.minimum(1 - out[:,2], out[:,0] + s.flatten())
    # out[:,0] = np.minimum(1 - out[:,2], out[:,0] + ctx[:,:,:,0].flatten())
    # mask = np.logical_and(mask, ctx[:,:,:,1].flatten())
    # out[:,1][mask] = (1 - np.sum(out, axis=1))[mask]
    # out = out / (np.maximum(1, np.sum(out, axis=1))[:,None])

    return out.reshape(shape)



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

    if not kwargs.get('cores'):
        kwargs['cores'] = max([multiprocessing.cpu_count() - 1, 1])

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
    outPVs, cortexMask = estimators.cortex(hemispheres, refSpace, 
        supersampler, kwargs['cores'])

    return (outPVs, cortexMask)



def resample(src, ref, out, src2ref=np.identity(4), flirt=False):
   
    if flirt:
        src2ref = pvcore._adjustFLIRT(src, ref, src2ref)

    refSpace = ImageSpace(ref)
    factor = np.ceil(refSpace.voxSize).astype(np.int8)
    resamp = pvcore._superResampleImage(src, factor, refSpace, src2ref)
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
    # We then set the flirt flag to false again (otherwise later steps will 
    # repeat the tricks and end up reverting to the original - those steps don't
    # need to know what we did here, simply that it is now world-world again)
    if kwargs.get('flirt'):
        if not kwargs.get('struct'):
            raise RuntimeError("If using a FLIRT transform, the path to the \
                structural image must also be given")
        kwargs['struct2ref'] = pvcore._adjustFLIRT(kwargs['struct'], kwargs['ref'], 
            kwargs['struct2ref'])
        kwargs['flirt'] = False 

    if not kwargs.get('cores'):
        kwargs['cores'] = max([multiprocessing.cpu_count() - 1, 1])

    return kwargs