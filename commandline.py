import argparse
import sys 
import os.path as op
import os

import numpy as np

from . import pvtools
from .classes import ImageSpace, CommonParser
from . import fileutils


def estimate_cortex_cmd(*args):

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser()
    parser.add_argument('-out', type=str, required=False)
    parser.add_argument('-fsdir', type=str, required=False)
    parser.add_argument('-LWS', type=str, required=False)
    parser.add_argument('-LPS', type=str, required=False)
    parser.add_argument('-RWS', type=str, required=False)        
    parser.add_argument('-RPS', type=str, required=False)
    parser.add_argument('-hard', action='store_true')
    parser.add_argument('-stack', action='store_true', required=False)
    parser.add_argument('-saveassocs', action='store_true', required=False)
    kwargs = parser.parse(args)

    # Preparation
    if not kwargs.get('out'):
        kwargs['out'] = fileutils.default_output_path(kwargs['ref'], 
            kwargs['ref'])

    outPath = fileutils._addSuffixToFilename('_cortex_pvs', kwargs['out'])
    maskPath = fileutils._addSuffixToFilename('_cortexmask', kwargs['out'])

    # Estimation
    PVs, mask, transformed = pvtools.estimate_cortex(**kwargs)

    # Output
    refSpace = ImageSpace(kwargs['ref'])
    print("Saving output to", kwargs['outdir'])
    refSpace.saveImage(mask, maskPath)

    if kwargs.get('stack'):
        refSpace.saveImage(PVs, outPath)    
    else:
        for i,t in enumerate(['_GM', '_WM', '_nonbrain']):
            refSpace.saveImage(PVs[:,:,:,i], 
            fileutils._addSuffixToFilename(t, outPath))

    if kwargs.get('savesurfs'):
        assert transformed is not None 
        sbase = fileutils.default_output_path(kwargs['ref'], 
            kwargs['ref'], ext=False)
        print('Saving transformed surfaces to', op.dirname(sbase))
        for k, s in transformed.items():
            sname = fileutils._addSuffixToFilename('_'+k, sbase) + '.surf.gii'
            s.save(sname)


def resample_cmd(*args):

    parser = CommonParser()

    parser.add_argument('-ref', type=str, required=True)
    parser.add_argument('-src', type=str, required=True)
    parser.add_argument('-out', type=str, required=True)
    parser.add_argument('-aff', type=str, required=False)
    parser.add_argument('-flirt', action='store_true')

    kwargs = parser.parse(args)

    if kwargs['flirt'] and not kwargs.get('aff'):
        raise RuntimeError("Flirt flag set but no affine transform supplied")

    src2ref = kwargs.get('aff')
    if not src2ref:
        src2ref = np.identity(4)

    pvtools.resample(**kwargs)


def estimate_structure_cmd(*args):

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser()
    parser.add_argument('-surf', type=str, required=True)
    parser.add_argument('-space', type=str, default='world', required=True)
    kwargs = parser.parse(args)

    if kwargs.get('out') is None:
        surfname = fileutils.splitExts(kwargs['surf'])[0]
        kwargs['out'] = fileutils.default_output_path(kwargs['ref'], 
            kwargs['ref'], '_%s_pvs' % surfname)

    # Estimate
    PVs, transformed = pvtools.estimate_structure(**kwargs)

    # Output
    refSpace = ImageSpace(kwargs['ref'])
    refSpace.saveImage(PVs, kwargs['out'])

    if kwargs.get('savesurfs'):
        print('Saving transformed surfaces to', op.dirname(kwargs['out']))
        assert transformed is not None 
        fname = op.join(op.dirname(kwargs['out']), 
            fileutils.splitExts(kwargs['ref'])[0] + '_%s.surf.gii' % 
            transformed.name)
        transformed.save(fname)



def estimate_all_cmd(*args):
    
    # parse stuff here
    parser = CommonParser()
    parser.add_argument('-struct_brain', type=str, required=False)
    parser.add_argument('-pvdir', type=str, required=False)
    parser.add_argument('-stack', action='store_true', required=False)
    kwargs = parser.parse(*args)
    
    # Unless we have been given prepared pvdir, we will provide the path
    # to the next function to create one
    if type(kwargs.get('pvdir')) is str:

        if not op.isdir(kwargs.get('pvdir')):
            raise RuntimeError("pvdir %s does not exist" % kwargs['pvdir'])

    else:
        kwargs['pvdir'] = fileutils.default_output_path(
            kwargs['struct'], kwargs['struct'], '_pvtools', False)

    output, transformed = pvtools.estimate_all(**kwargs)

    # Output paths. If given an -out argument of the form path/name then we use
    # path as the output directory and name as the basic filename. Otherwise we
    # use the pvdir for output and the reference as basic filename. 
    outdir = ''
    namebase = ''
    ext = '.nii.gz'
    if kwargs.get('out'):
        outdir = op.split(kwargs['out'])[0]
        namebase = fileutils.splitExts(kwargs['out'])[0]
    
    if not namebase:
        namebase = fileutils.splitExts(kwargs['ref'])[0]

    if not outdir: 
        outdir = kwargs['pvdir']

    # Make output dirs if they do not exist. 
    intermediatedir = op.join(outdir, namebase + '_intermediate')
    fileutils.weak_mkdir(outdir)
    fileutils.weak_mkdir(intermediatedir)

    # Load the reference image space and save the various outputs. 
    # 'stacked' goes in the outdir, all others go in outdir/intermediate 
    refSpace = ImageSpace(kwargs['ref'])
    for k, o in output.items():
        if k == 'stacked':
            path = op.join(outdir, namebase + ext)
            if kwargs.get('stack'): 
                refSpace.saveImage(o, 
                    fileutils._addSuffixToFilename('_'+k, path))
            else:
                for i,t in enumerate(['_GM', '_WM', '_nonbrain']):
                    refSpace.saveImage(o[:,:,:,i], 
                        fileutils._addSuffixToFilename(t, path))
        else: 
            path = op.join(intermediatedir, namebase + ext)
            refSpace.saveImage(o, 
                fileutils._addSuffixToFilename('_' + k, path))

    if kwargs.get('savesurfs'):
        assert transformed is not None 
        print('Saving transformed surfaces to', intermediatedir)
        for k, s in transformed.items():
            sname = op.join(intermediatedir, 
                namebase + '_%s.surf.gii' % s.name)
            s.save(sname)


            