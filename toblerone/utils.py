"""Toblerone utility functions"""

import os.path as op
import os 
import glob
import subprocess 
import sys
import shutil
import warnings 
import time
import multiprocessing

import numpy as np 
import nibabel
from fsl.wrappers import fsl_anat

STRUCTURES = ['L_Accu', 'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 
                'L_Thal', 'R_Accu', 'R_Amyg', 'R_Caud', 'R_Hipp', 'R_Pall', 'R_Puta', 
                'R_Thal', 'BrStem']


def cascade_attributes(decorator):
    """
    Overrride default decorator behaviour to preserve docstrings etc
    of decorated functions - functools.wraps didn't seem to work. 
    See https://stackoverflow.com/questions/6394511/python-functools-wraps-equivalent-for-classes
    """

    def new_decorator(original):
        wrapped = decorator(original)
        wrapped.__name__ = original.__name__
        wrapped.__doc__ = original.__doc__
        wrapped.__module__ = original.__module__
        return wrapped
    return new_decorator


def _mp_call_attribute(obj, method_name, args=None):
    if args: 
        return getattr(obj, method_name)(args)
    else: 
        return getattr(obj, method_name)


def check_anat_dir(dir):
    """Check that dir contains output from FIRST, FAST and FreeSurfer"""

    return all([
        op.isdir(op.join(dir, 'fs', 'surf')), 
        op.isdir(op.join(dir, 'first_results')), 
        op.isfile(op.join(dir, 'T1_fast_pve_0.nii.gz')), 
    ])


def _loadFIRSTdir(dir_path):
    """Load surface paths from a FIRST directory into a dict, accessed by the
    standard keys used by FIRST (eg 'BrStem'). The function will attempt to 
    load every available surface found in the directory using the standard
    list as reference (see FIRST documentation) but no errors will be raised
    if a particular surface is not found
    """

    files = glob.glob(op.join(dir_path, '*.vtk'))
    if not files:
        raise RuntimeError("FIRST directory %s is empty" % dir_path)

    surfs = {}
    for f in files: 
        fname = op.split(f)[1]
        for s in STRUCTURES:
            if s in fname:
                surfs[s] = f 

    return surfs


def _loadFASTdir(dir):
    """Load the PV image paths for WM,GM,CSF from a FAST directory into a 
    dict, accessed by the keys FAST_GM for GM etc
    """

    if not op.isdir(dir):
        raise RuntimeError("FAST directory does not exist")
    
    paths = {}
    files = glob.glob(op.join(dir, '*.nii.gz'))
    channels = { '_pve_{}'.format(c): t for (c,t)
        in enumerate(['CSF', 'GM', 'WM']) }
        
    for f in files: 
        fname = op.split(f)[1]
        for c, t in channels.items():
            if c in fname: 
                paths['FAST_' + t] = f
    
    if len(paths) != 3:
        raise RuntimeError("Could not load 3 PV maps from FAST directory")

    return paths


def _loadSurfsToDict(fsdir):
    """Load the left/right white/pial surface paths from a FS directory into 
    a dictionary, accessed by the keys LWS/LPS/RPS/RWS
    """

    sdir = op.realpath(op.join(fsdir, 'surf'))

    if not op.isdir(sdir):
        raise RuntimeError("Subject's surf directory does not exist")

    surfs = {}    
    for s in ['LWS', 'LPS', 'RWS', 'RPS']:
        snames = {'L': 'lh', 'R': 'rh'}
        exts = {'W': '.white', 'P': '.pial'}
        potentials = glob.glob(op.join(sdir, '*{}*'.format(snames[s[0]] + exts[s[1]])))
        if len(potentials) > 0:
            best = None  
            for p in potentials: 
                if p.endswith('.H') or p.endswith('.K') or p.endswith('.preparc'):
                    continue
                try: 
                    nibabel.freesurfer.io.read_geometry(p)
                    best = p 
                    break 
                except Exception as e: 
                    pass 
            if p is not None: 
                print("Found multiple potential {}, using: {}".format(s, best))
                surfs[s] = p 
            else: 
                raise RuntimeError("Could not find a suitable surface for", s)
        else: 
            surfs[s] = potentials[0]

    if not all(map(op.isfile, surfs.values())):
        raise RuntimeError("One of the subject's surfaces does not exist")

    return surfs


def _addSuffixToFilename(suffix, fname):
    """Add suffix to filename, whilst preserving original extension, eg:
    'file.ext1.ext2' + '_suffix' -> 'file_suffix.ext1.ext2'
    """

    head = op.split(fname)[0]
    fname, ext = _splitExts(fname)   
    return op.join(head, fname + suffix + ext)


def _addPrefixToFilename(prefix, fname):
    """Add prefix to filename, whilst preserving original extension, eg:
    'prefix_' + file.ext1.ext2' -> 'prefix_file_suffix.ext1.ext2'
    """

    head = op.split(fname)[0]
    fname, ext = _splitExts(fname)   
    return op.join(head, prefix + fname + ext)


def _splitExts(fname):
    """Split all extensions off a filename, eg:
    'file.ext1.ext2' -> ('file', '.ext1.ext2')
    """

    fname = op.split(fname)[1]
    ext = ''
    while '.' in fname:
        fname, e = op.splitext(fname)
        ext = e + ext 
    
    return fname, ext


def _weak_mkdir(dir):
    """Create a directory if it does not already exist"""

    if not op.isdir(dir):
        os.makedirs(dir)


def _shellCommand(cmd):   
    """Convenience function for calling shell commands"""

    try: 
        ret = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, 
            shell=True)
        if ret.returncode:
            print("Non-zero return code")
            raise RuntimeError()
    except Exception as e:
        print("Error when executing cmd:", cmd)
        raise e


def _runFreeSurfer(struct, dir, debug=False):
    """Args: 
        struct: path to structural image 
        dir: path to directory in which a subject directory entitled
            'fs' will be created and FS run within
    """

    struct = op.abspath(struct)
    pwd = os.getcwd()
    os.chdir(dir)
    cmd = 'recon-all -i {} -all -subjid fs -sd .'.format(struct)
    if debug: cmd += ' -dontrun'
    print("Calling FreeSurfer on", struct)
    print("This will take ~10 hours")
    _shellCommand(cmd)
    os.chdir(pwd)


def _runFIRST(struct, dir):
    """Args: 
        struct: path to structural image 
        dir: path to directory in which FIRST will be run
    """

    _weak_mkdir(dir)
    nameroot, _ = _splitExts(struct)
    struct = op.abspath(struct)
    pwd = os.getcwd()
    os.chdir(dir)
    cmd = 'run_first_all -i {} -o {}'.format(struct, nameroot)
    print("Calling FIRST on", struct)
    _shellCommand(cmd)
    os.chdir(pwd)


def _runFAST(struct, dir):
    """Args: 
        struct: path to structural image 
        dir: path to directory in which FAST will be run
    """

    _weak_mkdir(dir)
    struct = op.abspath(struct)
    pwd = os.getcwd()
    newstruct = op.abspath(op.join(dir, op.split(struct)[1]))
    shutil.copy(struct, newstruct)
    os.chdir(dir)
    cmd = 'fast {}'.format(newstruct)
    print("Calling FAST on", struct)
    _shellCommand(cmd)
    os.chdir(pwd)


# Local function to read out an FSL-specific affine matrix from an image
def _getFSLspace(imgPth):
    obj = nibabel.load(imgPth)
    if obj.header['dim'][0] < 3:
        raise RuntimeError("Volume has less than 3 dimensions" +
                                "cannot resolve space")

    sform = obj.affine
    det = np.linalg.det(sform[0:4, 0:4])
    ret = np.identity(4)
    pixdim = obj.header['pixdim'][1:4]
    for d in range(3):
        ret[d,d] = pixdim[d]

    # Check the xyzt field to find the spatial units. 
    xyzt = str(obj.header['xyzt_units'])
    if xyzt == '01': 
        multi = 1000
    elif xyzt == '10':
        multi = 1 
    elif xyzt == '11':
        multi = 1e-3
    else: 
        multi = 1
        warnings.warn("Assuming mm units for transform")

    if det > 0:
        ret[0,0] = -pixdim[0]
        ret[0,3] = (obj.header['dim'][1] - 1) * pixdim[0]

    ret = ret * multi
    ret[3,3] = 1
    return ret


def _FLIRT_to_world(source, reference, transform):
    """Adjust a FLIRT transformation matrix into a true world-world 
    transform. Required as FSL matrices are encoded in a specific form 
    such that they can only be applied alongside the requisite images (extra
    information is required from those images). With thanks to Martin Craig
    and Tim Coalson. See: https://github.com/Washington-University/workbench/blob/9c34187281066519e78841e29dc14bef504776df/src/Nifti/NiftiHeader.cxx#L168 
    https://github.com/Washington-University/workbench/blob/335ad0c910ca9812907ea92ba0e5563225eee1e6/src/Files/AffineFile.cxx#L144

    Args: 
        source: path to source image, the image to be deformed 
        reference: path to reference image, the target of the transform
        transform: affine matrix produced by FLIRT from src to ref 

    Returns: 
        complete transformation matrix between the two. 
    """

    # 'Space' refers to conversion of world -> scaled voxel coordinates, 
    # with no shift for origin 
    Ss = _getFSLspace(source)
    Rs = _getFSLspace(reference)

    # 'Affine' refers to the voxel -> world conversion matrix 
    Ra = nibabel.load(reference).affine
    Sa = nibabel.load(source).affine

    # Work backwards (R->L) to interpret these 
    out = Ra @ np.linalg.inv(Rs) @ transform @ Ss @ np.linalg.inv(Sa)
    return out


def _world_to_FLIRT(source, reference, transform):
    """
    Inverse of _FLIRT_to_world: convert a world-world transform to FLIRT matrix
    """

    # 'Space' refers to conversion of world -> scaled voxel coordinates, 
    # with no shift for origin 
    Ss = _getFSLspace(source)
    Rs = _getFSLspace(reference)

    # 'Affine' refers to the voxel -> world conversion matrix 
    Ra = nibabel.load(reference).affine
    Sa = nibabel.load(source).affine

    # Work backwards (R->L) to interpret these 
    out = Rs @ np.linalg.inv(Ra) @ transform @ Sa @ np.linalg.inv(Ss)
    return out



def affineTransformPoints(points, affine):
    """Apply affine transformation to set of points.

    Args: 
        points: n x 3 matrix of points to transform
        affine: 4 x 4 matrix for transformation

    Returns: 
        transformed copy of points 
    """

    if len(points.shape) != 2:
        if points.size != 3: 
            raise RuntimeError("Points must be n x 3 or 3-vector")
        points = points[None,:]

    # Add 1s on the 4th column, transpose and multiply, 
    # then re-transpose and drop 4th column  
    transfd = np.ones((points.shape[0], 4))
    transfd[:,0:3] = points
    transfd = np.matmul(affine, transfd.T).astype(np.float32)
    return np.squeeze(transfd[0:3,:].T)


def _coordinatesForGrid(ofSize):
    """Produce N x 3 array of all voxel indices (eg [10, 18, 2]) within
    a grid of size ofSize, 0-indexed and in integer form. 
    """

    I, J, K = np.unravel_index(np.arange(np.prod(ofSize)), ofSize)
    cents = np.vstack((I.flatten(), J.flatten(), K.flatten())).T
    return cents.astype(np.int32)


def _distributeObjects(objs, ngroups):
    """Distribute a set of objects into n groups.
    For preparing chunks before multiprocessing.pool.map
    
    Returns a set of ranges, each of which are index numbers for 
    the original set of objs 
    """

    chunkSize = np.floor(len(objs) / ngroups).astype(np.int32)
    chunks = [] 

    for n in range(ngroups):
        if n != ngroups - 1: 
            chunks.append(range(n * chunkSize, (n+1) * chunkSize))
        else:
            chunks.append(range(n * chunkSize, len(objs)))

    assert sum(map(len, chunks)) == len(objs), \
        "Distribute objects error: not all objects distributed"        

    return chunks 


def _clipArray(arr, mini=0.0, maxi=1.0):
    """Clip array values into range [mini, maxi], default [0 1]"""

    arr[arr < mini] = mini 
    arr[arr > maxi] = maxi 
    return arr 


def fsl_fs_anat(**kwargs):
    """
    Run fsl_anat (FAST & FIRST) and augment output with FreeSurfer

    Args: 
        anat: (optional) path to existing fsl_anat dir to augment
        struct: (optional) path to T1 NIFTI to create a fresh fsl_anat dir
        out: output path (default alongside input, named input.anat)
    """

    # We are either adding to an existing dir, or we are creating 
    # a fresh one 
    if (not bool(kwargs.get('anat'))) and (not bool(kwargs.get('struct'))):
        raise RuntimeError("Either a structural image or a path to an " + 
            "existing fsl_anat dir must be given")

    if kwargs.get('struct') and (not op.isfile(kwargs['struct'])):
        raise RuntimeError("No struct image given, or does not exist")

    if kwargs.get('anat') and (not op.isdir(kwargs['anat'])):
        raise RuntimeError("fsl_anat dir does not exist")

    debug = bool(kwargs.get('debug'))
    anat_exists = bool(kwargs.get('anat'))
    struct = kwargs['struct']

    # Run fsl_anat if needed. Either use user-supplied name or default
    if not anat_exists:
        outname = kwargs.get('out')
        if not outname:
            outname = _splitExts(kwargs['struct'])[0]
            outname = op.dirname(kwargs['struct']) + outname
        print("Preparing an fsl_anat dir at %s" % outname)
        if outname.endswith('.anat'):
            outname = outname[:-5]
        fsl_anat(struct, outname)
        outname += '.anat'

    else:
        outname = kwargs['anat']
    
    # Run the surface steps if reqd. 
    # Check the fullfov T1 exists within anat_dir
    if not op.isdir(op.join(outname, 'fs', 'surf')):
        fullfov = op.join(outname, 'T1_fullfov.nii.gz')

        if not op.isfile(fullfov):
            raise RuntimeError("Could not find T1_fullfov.nii.gz within anat_dir %s" 
                    % outname)

        print("Adding FreeSurfer to fsl_anat dir at %s" % outname)
        _runFreeSurfer(fullfov, outname, debug)

    if not check_anat_dir(outname): 
        raise RuntimeError("fsl_anat dir should be complete with surfaces") 

    print("fsl_anat dir at %s is now complete with surfaces" % outname)
    return outname 


@cascade_attributes
def enforce_and_load_common_arguments(func):
    """
    Decorator to enforce and pre-processes common arguments in a 
    kwargs dict that are used across multiple functions. Note
    some function-specific checking is still required. This intercepts the
    kwargs dict passed to the caller, does some checking and modification 
    in place, and then returns to the caller. The following args are handled:

    Required args:
        ref: path to a reference image in which to operate 
        struct2ref: path to file or 4x4 array representing transformation
            between structural space (that of the surfaces) and reference. 
            If given as 'I', identity matrix will be used. 

    Optional args: 
        anat: a fsl_anat directory (created/augmented by make_surf_anat_dir)
        flirt: bool denoting that the struct2ref is a FLIRT transform.
            This means it requires special treatment. If set, then it will be
            pre-processed in place by this function, and then the flag will 
            be set back to false when the kwargs dict is returned to the caller
        struct: if FLIRT given, then the path to the structural image used
            for surface generation is required for said special treatment
        cores: maximum number of cores to parallelise tasks across 
            (default is N-1)

    Returns: 
        a modified copy of kwargs dictionary passed to the caller
    """
    
    def enforcer(**kwargs):

        # Reference image path 
        if not kwargs.get('ref'):
            raise RuntimeError("Path to reference image must be given")

        if (type(kwargs['ref']) is str) and not op.isfile(kwargs['ref']):
            raise RuntimeError("Reference image %s does not exist" % kwargs['ref'])

        # If given a anat_dir we can load the structural image in 
        if kwargs.get('anat'):
            if not check_anat_dir(kwargs['anat']):
                raise RuntimeError("anat is not complete: it must contain " 
                    "fast, fs and first subdirectories")

            kwargs['fastdir'] = kwargs['anat']
            kwargs['fsdir'] = op.join(kwargs['anat'], 'fs')
            kwargs['firstdir'] = op.join(kwargs['anat'], 'first_results')

            # If no struct image given, try and pull it out from the anat dir
            # But, if it has been cropped relative to original T1, then give
            # warning (as we will not be able to convert FLIRT to world-world)
            if not kwargs.get('struct'): 
                if kwargs.get('flirt'):
                    matpath = glob.glob(op.join(kwargs['anat'], '*nonroi2roi.mat'))[0]
                    nonroi2roi = np.loadtxt(matpath)
                    if np.any(np.abs(nonroi2roi[0:3,3])):
                        print("Warning: T1 was cropped relative to T1_orig within" + 
                            " fsl_fs_anat dir.\n Please ensure the struct2ref FLIRT" +
                            " matrix is referenced to T1, not T1_orig.")

                s = op.join(kwargs['anat'], 'T1.nii.gz')
                kwargs['struct'] = s
                if not op.isfile(s):
                    raise RuntimeError("Could not find T1.nii.gz in the anat dir")

 
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
                        if matExt in ['.txt', '.mat']:
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
            kwargs['struct2ref'] = _FLIRT_to_world(kwargs['struct'], kwargs['ref'], 
                kwargs['struct2ref'])
            kwargs['flirt'] = False 

        # Processor cores
        if not kwargs.get('cores'):
            kwargs['cores'] = multiprocessing.cpu_count()

        # Supersampling factor
        sup = kwargs.get('super')
        if sup is not None: 
            try: 
                if (type(sup) is list) and (len(sup) == 3): 
                    sup = np.array([int(s) for s in sup])
                else: 
                    sup = int(sup[0])
                    sup = np.array([sup for _ in range(3)])

                if type(sup) is not np.ndarray: 
                    raise RuntimeError() 
            except:
                raise RuntimeError("-super must be a value or list of 3" + 
                    " values of int type")
        
            kwargs['super'] = sup.astype(np.int8)
            print("Using manual supersampling factor", kwargs['super'])

        return kwargs

    def common_args_enforced(**kwargs):
        enforced = enforcer(**kwargs)
        return func(**enforced)

    return common_args_enforced

