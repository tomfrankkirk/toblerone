# Command line interface for toblerone
# The following functions are exposed to the __main__.py file and are
# called when the module is invoked eg python3 -m toblerone

import os.path as op
import os

import regtricks as rt 

from toblerone import utils, pvestimation, projection
from toblerone.classes import CommonParser, ImageSpace, Surface


def estimate_cortex():
    """
    CLI for estimating PVs from cortex (either L,R, or both)
    """

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

    os.makedirs(outdir, exist_ok=True)
    refSpace = ImageSpace(kwargs['ref'])

    print('Saving output at', outdir)
    p = op.join(outdir, 'stacked' + ext)
    refSpace.save_image(PVs, p)
    for i,t in enumerate(['GM', 'WM', 'nonbrain']):
        p = op.join(outdir, t + ext)
        refSpace.save_image(PVs[:,:,:,i], p)


def estimate_structure():
    """
    CLI for estimating PVs from a single surface
    """

    parser = CommonParser('ref', 'struct2ref', 'flirt', 'struct', 'super', 'out',
        'surf', 'coords',
        description="Estimate PVs for a structure defined by a single surface.")
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
    CLI for estimating PVs for L/R cortex and subcortex
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
            raise RuntimeError("Either separate firstdir, fastdir and struct"+
                " must be provided, or an fslanat dir must be provided")

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
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(intermediatedir, exist_ok=True)

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
    """
    CLI for converting surface formats 
    """

    parser = CommonParser('surf', 'coords', 'struct', 'out',
            description="""Convert a surface file (.white/.pial/.vtk/.surf.gii). 
            NB FreeSurfer files will have the c_ras offset automatically 
            applied during conversion.""")
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

    parser = CommonParser('ref', 'struct2ref', 'flirt', 'struct', 'fsdir', 
        'LPS', 'LWS', 'RPS', 'RWS', 'cores', 'ones', 'out', 'super',
        description=("Prepare a projector for a reference voxel grid and set "
            "of surfaces, and save in HDF5 format. This is a pre-processing "
            "step for performing surface-based analysis of volumetric data."))

    args = parser.parse_args()

    if args.flirt: 
        struct2ref = rt.Registration.from_flirt(args.struc2ref, args.struct, 
                                args.ref).src2ref
    elif args.struct2ref == "I":
        struct2ref = rt.Registration.identity().src2ref
    else: 
        struct2ref = rt.Registration(args.struct2ref).src2ref

    # Set up the hemispheres, reference ImageSpace, and prepare projector.
    spc = ImageSpace(args.ref)
    hemispheres = utils.load_surfs_to_hemispheres(**vars(args))
    hemispheres = [ h.transform(struct2ref) for h in hemispheres ]
    proj = projection.Projector(hemispheres, spc, args.super, 
        args.cores, args.ones)

    # Add default .h5 extension if needed, make outdir, save. 
    outdir, outname = op.split(args.out)
    outbase, outext = op.splitext(outname)
    if not outext: outext = '.h5'
    if outdir: os.makedirs(outdir, exist_ok=True)
    out = op.join(outdir, outbase + outext)
    proj.save(out)
