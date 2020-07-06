# Command line interface for toblerone
# The following functions are exposed to the __main__.py file and are
# called when the module is invoked eg python3 -m toblerone

import argparse
import sys 
import os.path as op
import os

import numpy as np

from toblerone import utils, pvestimation
from toblerone.classes import CommonParser, ImageSpace, Surface

suffix = (
"""
Tom Kirk, thomas.kirk@eng.ox.ac.uk
Institute of Biomedical Engineering / Wellcome Centre for Integrative Neuroimaging
University of Oxford, 2018
""")


def estimate_cortex_cmd(*args):
    """
    Estimate PVs for L/R cortex.

    Required args: 
        -ref: path to reference image for which PVs are required
        -struct2ref: path to np or text file, or np.ndarray obj, denoting affine
            registration between structural image used to produce surfaces 
            and reference. Use 'I' for identity, if using FLIRT also set -flirt 
        -fsdir: path to a FreeSurfer subject directory, from which L/R 
            white/pial surfaces will be loaded, OR: 
        -LWS/LPS/RWS/RPS: individual paths to the individual surfaces,
            eg LWS = Left White surface, RPS = Right Pial surace
            To estimate for a single hemisphere, only provide surfaces
            for that side. 

    Optional args: 
        -flirt: bool denoting struct2ref is FLIRT transform. If so, set -struct
        -struct: path to structural image from which surfaces were derived
        -cores: number of cores to use 
        -out: directory to save output within (default alongside ref)
        -ones: perform simple segmentation based on voxel centres (debug)
        -supersample: voxel subdision factor 
    """
    

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser()
    parser.add_argument('-fsdir', type=str, required=False)
    parser.add_argument('-LWS', type=str, required=False)
    parser.add_argument('-LPS', type=str, required=False)
    parser.add_argument('-RWS', type=str, required=False)        
    parser.add_argument('-RPS', type=str, required=False)
    parser.add_argument('-ones', action='store_true')
    parser.add_argument('-stack', action='store_true', required=False)
    parser.add_argument('-supersample', type=int)
    kwargs = parser.parse(args)

    # Estimation
    PVs = pvestimation.cortex(**kwargs)

    # Output 
    ext = '.nii.gz'
    if not kwargs.get('out'):
        namebase = op.splitext(utils._splitExts(kwargs['ref'])[0])[0]
        outdir = op.join(op.dirname(kwargs['ref']), namebase + '_cortexpvs')
    else: 
        outdir = kwargs['out']

    utils._weak_mkdir(outdir)
    refSpace = ImageSpace(kwargs['ref'])

    print('Saving output at', outdir)
    p = op.join(outdir, 'stacked' + ext)
    refSpace.save_image(PVs, p)
    for i,t in enumerate(['GM', 'WM', 'nonbrain']):
        p = op.join(outdir, t + ext)
        refSpace.save_image(PVs[:,:,:,i], p)


def estimate_structure_cmd(*args):
    """
    Estimate PVs for a structure defined by a single surface. 
    
    Required args: 
        -ref: path to reference image for which PVs are required
        -struct2ref: path to np or text file, or np.ndarray obj, denoting affine
            registration between structural image used to produce surfaces 
            and reference. Use 'I' for identity, if using FLIRT also set -flirt 
        -surf: path to surface (see space argument below)
        -space: space in which surface is defined: default is 'world' (mm coords),
            for FIRST surfaces set 'first' (FSL convention). 

    Optional args: 
        -flirt: bool denoting struct2ref is FLIRT transform. If so, set -struct
        -struct: path to structural image from which surfaces were derived
        -cores: number of cores to use 
        -out: path to save output (default alongside ref)
        -ones: perform simple segmentation based on voxel centres (debug)
        -supersample: voxel subdision factor 

    """

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser()
    parser.add_argument('-surf', type=str, required=True)
    parser.add_argument('-space', type=str, default='world', required=True)
    parser.add_argument('-ones', action='store_true')
    parser.add_argument('-supersample', type=int)
    kwargs = parser.parse(args)

    ext = '.nii.gz'
    if not kwargs.get('out'):
        namebase = op.splitext(utils._splitExts(kwargs['ref'])[0])[0]
        sname = op.splitext(utils._splitExts(kwargs['surf'])[0])[0]
        outdir = op.dirname(kwargs['ref'])
        kwargs['out'] = op.join(outdir, '%s_%s_pvs%s' % (namebase, sname, ext))
    else: 
        if not kwargs['out'].endswith(ext):
            kwargs['out'] += ext 

    # Estimate
    PVs = pvestimation.structure(**kwargs)

    # Output
    print('Saving output at', kwargs['out'])
    refSpace = ImageSpace(kwargs['ref'])
    refSpace.save_image(PVs, kwargs['out'])



def estimate_complete_cmd(*args):
    """
    Estimate PVs for cortex and all structures identified by FIRST within 
    a reference image space. Use FAST to fill in non-surface PVs. 
    
    Required args: 
        -ref: path to reference image for which PVs are required
        -struct2ref: path to np or text file, or np.ndarray obj, denoting affine
            registration between structural image used to produce surfaces 
            and reference. Use 'I' for identity, if using FLIRT also set -flirt 
        -anat: path to augmented fsl_anat directory (see -fsl_fs_anat). This
            REPLACES -fsdir, -firstdir, -fastdir, -LPS/RPS etc 

    Alternatvies to anat argument (all required): 
        -fsdir: FreeSurfer subject directory, OR: 
        -LWS/-LPS/-RWS/-RPS paths to individual surfaces (L/R white/pial)
        -firstdir: FIRST directory in which .vtk surfaces are located
        -fastdir: FAST directory in which _pve_0/1/2 are located 
        -struct: path to structural image from which surfaces were dervied

    Optional args: 
        -flirt: bool denoting struct2ref is FLIRT transform. If so, set -struct
        -cores: number of cores to use
        -out: directory to save output within (default alongside ref)
        -ones: perform simple segmentation based on voxel centres (debug)
        -supersample: voxel subdision factor 

    """
    
    parser = CommonParser()
    parser.add_argument('-anat', type=str, required=False)
    parser.add_argument('-stack', action='store_true', required=False)
    parser.add_argument('-fsdir', type=str, required=False)
    parser.add_argument('-firstdir', type=str, required=False)
    parser.add_argument('-fastdir', type=str, required=False)
    parser.add_argument('-LWS', type=str, required=False)
    parser.add_argument('-LPS', type=str, required=False)
    parser.add_argument('-RWS', type=str, required=False)        
    parser.add_argument('-RPS', type=str, required=False)
    parser.add_argument('-ones', action='store_true')
    parser.add_argument('-supersample', type=int)
    kwargs = parser.parse(args)
    
    # Unless we have been given prepared anat dir, we will provide the path
    # to the next function to create one
    if type(kwargs.get('anat')) is str:
        if not op.isdir(kwargs.get('anat')):
            raise RuntimeError("anat dir %s does not exist" % kwargs['anat'])
    else: 
        if not all([ 
            (('fastdir' in kwargs) and ('firstdir' in kwargs)),
            (('fsdir' in kwargs) or (('LPS' in kwargs) and ('RPS' in kwargs))) ]): 
            raise RuntimeError("Either separate -firstdir, -fsdir and -fastdir"+
                " must be provided, or an -anat dir must be provided")

    output = pvestimation.complete(**kwargs)

    # Output paths. If given -out then use that as output, otherwise
    # save alongside reference image 
    ext = '.nii.gz'
    if not kwargs.get('out'):
        namebase = op.splitext(utils._splitExts(kwargs['ref'])[0])[0]
        outdir = op.join(op.dirname(kwargs['ref']), namebase + '_surfpvs')
    else: 
        outdir = kwargs['out']

    # Make output dirs if they do not exist. 
    intermediatedir = op.join(outdir, 'intermediate_pvs')
    utils._weak_mkdir(outdir)
    utils._weak_mkdir(intermediatedir)

    # Load the reference image space and save the various outputs. 
    # 'stacked' goes in the outdir, all others go in outdir/intermediate 
    refSpace = ImageSpace(kwargs['ref'])
    print('Saving output at', outdir)
    for k, o in output.items():
        if k in ['stacked', 'GM', 'WM', 'nonbrain']:
            path = op.join(outdir, k + ext)
        else: 
            path = op.join(intermediatedir, k + ext)
        refSpace.save_image(o, path)


def fsl_fs_anat_cmd(*args):
    """
    Run fsl_anat (FAST & FIRST) and augment output with FreeSurfer

    Args: 
        -anat: (optional) path to existing fsl_anat dir to augment
        -struct: (optional) path to T1 NIFTI to create a fresh fsl_anat dir
        -out: output path (default alongside input, named input.anat)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-anat', type=str, required=False)
    parser.add_argument('-struct', type=str, required=False)
    parser.add_argument('-out', type=str, required=False)
    kwargs = vars(parser.parse_args(args))
    utils.fsl_fs_anat(**kwargs)

def convert_surface_cmd(*args):
    """
    Convert a surface file (.white/.pial/.vtk/.surf.gii). NB FreeSurfer files
    will have the c_ras offset automatically applied during conversion. 
    """
    
    insurf = Surface(args[0])
    insurf.save(args[1])

