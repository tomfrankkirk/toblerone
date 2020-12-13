# Command line interface for toblerone
# The following functions are exposed to the __main__.py file and are
# called when the module is invoked eg python3 -m toblerone

import argparse
import os.path as op
import os

from toblerone import utils, pvestimation, projection
from toblerone.classes import CommonParser, ImageSpace, Surface

suffix = (
"""
Tom Kirk, thomas.kirk@eng.ox.ac.uk
Institute of Biomedical Engineering / Wellcome Centre for Integrative Neuroimaging,
University of Oxford, 2018
""")


def estimate_cortex():

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser('ref', 'struct2ref', 'fsdir', 'LPS', 'RPS', 'RWS', 
        'LWS', 'flirt', 'struct', 'cores', 'out', 'ones', 'super', 
        description="Estimate PVs for L/R cortical hemispheres")
    kwargs = vars(parser.parse_args())

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


def estimate_structure():

    # Parse the common arguments and store as kwargs
    # Then run the parser specific to this function and add those in
    parser = CommonParser('ref', 'struct2ref', 'flirt', 'struct', 'super', 'out',
        description="Estimate PVs for a structure defined by a single surface.")

    parser.add_argument('-surf', type=str, required=True,
        help="path to surface (see -space argument below)")
    parser.add_argument('-coords', required=True,
        help=("""coordinates in which surface is defined, either 'world' 
            (mm coords) or 'fsl' (FSL convention, eg FIRST surfaces)"""))
    kwargs = vars(parser.parse_args())

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



def estimate_complete():
    """
    Estimate PVs for cortex and all structures identified by FIRST within 
    a reference image space. Use FAST to fill in non-surface PVs. 
    """
    
    parser = CommonParser('ref', 'struct2ref', 'fslanat', 'fsdir', 'firstdir',
        'fastdir', 'LPS', 'LWS', 'RPS', 'RWS', 'ones', 'super', 'cores', 'out', 
        'flirt', 'struct', 
        description=("Estimate PVs for cortex and all structures identified "
        "by FIRST within a reference image space. Use FAST to fill in "
        "non-surface PVs"))

    kwargs = vars(parser.parse_args())
    
    # Unless we have been given prepared fslanat dir, we will provide the path
    # to the next function to create one
    if type(kwargs.get('fslanat')) is str:
        if not op.isdir(kwargs.get('fslanat')):
            raise RuntimeError("fslanat dir %s does not exist" % kwargs['fslanat'])
    else: 
        if not all([ 
            (('fastdir' in kwargs) and ('firstdir' in kwargs)),
            (('LPS' in kwargs) and ('RPS' in kwargs)) ]): 
            raise RuntimeError("Either separate -firstdir and -fastdir"+
                " must be provided, or an -fslanat dir must be provided")

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


def convert_surface():

    parser = argparse.ArgumentParser(description=
            """Convert a surface file (.white/.pial/.vtk/.surf.gii). 
            NB FreeSurfer files will have the c_ras offset automatically 
            applied during conversion.""")

    parser.add_argument('surf', help='path to surface')
    parser.add_argument('-coords', default='world',
        help=("""coordinates in which surface is defined, either 'world' 
            (mm coords) or 'fsl' (FSL convention, eg FIRST surfaces)"""))
    parser.add_argument('-struct', 
        help=("""if -coords is 'fsl', provide a path to the structural
        image used to derive the surface to apply a conversion to 
        world-mm coordinates"""))        
    parser.add_argument('out', help='path to save output, with extension')
    parsed = parser.parse_args()

    if parsed.coords == 'fsl' and parsed.struct:
        insurf = Surface(parsed.surf, 'fsl', parsed.struct)
    else: 
        insurf = Surface(parsed.surf)
    insurf.save(parsed.out)


def prepare_projector():
    """
    CLI for making a Projector
    """

    parser = CommonParser('ref', 'fsdir', 'LPS', 'LWS', 'RPS', 'RWS', 
        'cores', 'ones', 'out',
        description=("Prepare a projector for a reference voxel grid and set "
            "of surfaces, and save in HDF5 format. This is a pre-processing "
            "step for performing surface-based analysis of volumetric data."))

    parser.add_argument('-superfactor', type=int,
        help="voxel supersampling factor, default 2x voxel size")
    args = parser.parse_args()

    # Set up the hemispheres, reference ImageSpace, and prepare projector.
    hemispheres = utils.load_surfs_to_hemispheres(**vars(args))
    spc = ImageSpace(args.ref)
    proj = projection.Projector(hemispheres, spc)

    # Add default .h5 extension if needed, make outdir, save. 
    outdir, outname = op.split(args.out)
    outbase, outext = op.splitext(outname)
    if not outext: outext = '.h5'
    if outdir: os.makedirs(outdir, exist_ok=True)
    out = op.join(outdir, outbase + outext)
    proj.save(out)
