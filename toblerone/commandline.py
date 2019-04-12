# Command line interface for toblerone
# The following functions are exposed to the __main__.py file and are
# called when the module is invoked eg python3 -m pvtools

import argparse
import sys 
import os.path as op
import os

import numpy as np

from . import main, utils
from .classes import CommonParser, ImageSpace


def estimate_cortex_cmd(*args):
    """Estimate PVs for L/R cortex.

    Required args: 
        -ref: path to reference image for which PVs are required
        -struct2ref: path to np or text file, or np.ndarray obj, denoting affine
                registration between structural (surface) and reference space.
                Use 'I' for identity. 

        One of: 
        -fsdir: path to a FreeSurfer subject directory, from which L/R 
            white/pial surfaces will be loaded 
        -LWS/LPS/RWS/RPS: individual paths to the individual surfaces,
            eg LWS = Left White surface, RPS = Right Pial surace
            To estimate for a single hemisphere, only provide surfaces
            for that side. 

    Optional args: 
        -flirt: bool denoting struct2ref is FLIRT transform. If so, set struct
        -struct: path to structural image from which surfaces were derived
        -cores: number of cores to use (default N-1)
        -out: path to save output (default alongside ref, using same basename)
        -savesurfs: save copies of each surface in reference space. 
            HIGHLY recommended to check registration quality. 
        -hard: perform simple segmentation based on voxel centres (no PVs, 
            useful for masking)
    """
    

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser()
    parser.add_argument('-fsdir', type=str, required=False)
    parser.add_argument('-LWS', type=str, required=False)
    parser.add_argument('-LPS', type=str, required=False)
    parser.add_argument('-RWS', type=str, required=False)        
    parser.add_argument('-RPS', type=str, required=False)
    parser.add_argument('-hard', action='store_true')
    parser.add_argument('-stack', action='store_true', required=False)
    kwargs = parser.parse(args)

    # Estimation
    PVs, mask, transformed = main.estimate_cortex(**kwargs)

    # Output 
    if not kwargs.get('out'):
        kwargs['out'] = utils._default_output_path(kwargs['ref'], 
            kwargs['ref'])

    outPath = utils._addSuffixToFilename('_cortex_pvs', kwargs['out'])
    maskPath = utils._addSuffixToFilename('_cortexmask', kwargs['out'])

    refSpace = ImageSpace(kwargs['ref'])
    print("Saving output to", kwargs['outdir'])
    refSpace.saveImage(mask, maskPath)

    if kwargs.get('stack'):
        refSpace.saveImage(PVs, outPath)    
    else:
        for i,t in enumerate(['_GM', '_WM', '_nonbrain']):
            refSpace.saveImage(PVs[:,:,:,i], 
            utils._addSuffixToFilename(t, outPath))

    if kwargs.get('savesurfs'):
        assert transformed is not None 
        sbase = utils._default_output_path(kwargs['ref'], 
            kwargs['ref'], ext=False)
        print('Saving transformed surfaces to', op.dirname(sbase))
        for k, s in transformed.items():
            sname = utils._addSuffixToFilename('_'+k, sbase) + '.surf.gii'
            s.save(sname)


def resample_cmd(*args):

    parser = CommonParser()

    parser.add_argument('-src', type=str, required=True)
    parser.add_argument('-aff', type=str, required=False)

    kwargs = parser.parse(args)

    if kwargs['flirt'] and not kwargs.get('aff'):
        raise RuntimeError("Flirt flag set but no affine transform supplied")

    src2ref = kwargs.get('aff')
    if not src2ref:
        src2ref = np.identity(4)

    main.resample(**kwargs)


def estimate_structure_cmd(*args):
    """Estimate PVs for a structure defined by a single surface. 
    
    Required args: 
        -ref: path to reference image for which PVs are required
        -struct2ref: path to np or text file, or np.ndarray obj, denoting affine
                registration between structural (surface) and reference space.
                Use 'I' for identity. 
        -surf: path to surface (see space argument below)

    Optional args: 
        space: space in which surface is defined: default is 'world' (mm coords),
            for FIRST surfaces set 'first' (FSL convention). 
        -flirt: bool denoting struct2ref is FLIRT transform. If so, set struct
        -struct: path to structural image from which surfaces were derived
        -cores: number of cores to use (default N-1)
        -out: path to save output (default alongside ref)
        -savesurfs: save copies of each surface in reference space. 
            HIGHLY recommended to check quality of registration. 
    """

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser()
    parser.add_argument('-surf', type=str, required=True)
    parser.add_argument('-space', type=str, default='world', required=True)
    kwargs = parser.parse(args)

    if kwargs.get('out') is None:
        surfname = utils._splitExts(kwargs['surf'])[0]
        kwargs['out'] = utils._default_output_path(kwargs['ref'], 
            kwargs['ref'], '_%s_pvs' % surfname)

    # Estimate
    PVs, transformed = main.estimate_structure(**kwargs)

    # Output
    refSpace = ImageSpace(kwargs['ref'])
    refSpace.saveImage(PVs, kwargs['out'])

    if kwargs.get('savesurfs'):
        print('Saving transformed surfaces to', op.dirname(kwargs['out']))
        assert transformed is not None 
        fname = op.join(op.dirname(kwargs['out']), 
            utils._splitExts(kwargs['ref'])[0] + '_%s.surf.gii' % 
            transformed.name)
        transformed.save(fname)



def estimate_all_cmd(*args):
    """Estimate PVs for cortex and all structures identified by FIRST within 
    a reference image space. Use FAST to fill in non-surface PVs. 
    
    Required args: 
        -ref: path to reference image for which PVs are required
        -struct2ref: path to np or text file, or np.ndarray obj, denoting affine
                registration between structural (surface) and reference space.
                Use 'I' for identity. 
        -pvdir: path to pvtools directory (created by make_pvtools_dir)

    Optional args: 
        -flirt: bool denoting struct2ref is FLIRT transform. If so, set struct
        -struct: path to structural image from which surfaces were derived
        -cores: number of cores to use (default N-1)
        -out: path to save output (default alongside ref)
        -stack: stack PVs into 4D NIFTI, arranged GM/WM/non-brain
        -savesurfs: save copies of each surface in reference space. 
            HIGHLY recommended to check quality of registration. 
 
    """
    
    # parse stuff here
    parser = CommonParser()
    parser.add_argument('-pvdir', type=str, required=False)
    parser.add_argument('-stack', action='store_true', required=False)
    kwargs = parser.parse(args)
    
    # Unless we have been given prepared pvdir, we will provide the path
    # to the next function to create one
    if type(kwargs.get('pvdir')) is str:
        if not op.isdir(kwargs.get('pvdir')):
            raise RuntimeError("pvdir %s does not exist" % kwargs['pvdir'])
    else: 
        raise RuntimeError("pvdir must be provided (run make_pvtools_dir()")

    output, transformed = main.estimate_all(**kwargs)

    # Output paths. If given an -out argument of the form path/name then we use
    # path as the output directory and name as the basic filename. Otherwise we
    # use the pvdir for output and the reference as basic filename. 
    outdir = ''
    namebase = ''
    ext = '.nii.gz'
    if kwargs.get('out'):
        outdir = op.split(kwargs['out'])[0]
        namebase = utils._splitExts(kwargs['out'])[0]
    
    if not namebase:
        namebase = utils._splitExts(kwargs['ref'])[0]

    if not outdir: 
        outdir = kwargs['pvdir']

    # Make output dirs if they do not exist. 
    intermediatedir = op.join(outdir, namebase + '_intermediate')
    utils._weak_mkdir(outdir)
    utils._weak_mkdir(intermediatedir)

    # Load the reference image space and save the various outputs. 
    # 'stacked' goes in the outdir, all others go in outdir/intermediate 
    refSpace = ImageSpace(kwargs['ref'])
    for k, o in output.items():
        if k == 'stacked':
            path = op.join(outdir, namebase + ext)
            if kwargs.get('stack'): 
                refSpace.saveImage(o, 
                    utils._addSuffixToFilename('_'+k, path))
            else:
                for i,t in enumerate(['_GM', '_WM', '_nonbrain']):
                    refSpace.saveImage(o[:,:,:,i], 
                        utils._addSuffixToFilename(t, path))
        else: 
            path = op.join(intermediatedir, namebase + ext)
            refSpace.saveImage(o, 
                utils._addSuffixToFilename('_' + k, path))

    if kwargs.get('savesurfs'):
        assert transformed is not None 
        print('Saving transformed surfaces to', intermediatedir)
        for k, s in transformed.items():
            sname = op.join(intermediatedir, 
                namebase + '_%s.surf.gii' % s.name)
            s.save(sname)


def make_pvtools_dir_cmd(*args): 
    """Create a pvtools directory from a T1 and brain-extracted T1 image. 
    Runs FreeSurfer, FIRST and FAST in parallel using specified number of cores

    Args: 
        -struct: path to T1 
        -struct_brain: path to brain-extracted T1
        -path: will default to location of T1
        -cores: number of cores to use in parallel
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('struct', type=str, required=True)
    parser.add_argument('struct_brain', type=str, required=True)
    parser.add_argument('path', type=str, required=True)
    parsed = parser.parse_args(args)

    main.make_pvtools_dir(**parsed)