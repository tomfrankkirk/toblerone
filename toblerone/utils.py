# Utility functions for Toblerone 
# Mix of file/path related funcs and numerical tools 

import os.path as op
import os 
import copy
import glob
import subprocess 
import sys
import shutil 

import numpy as np 
import nibabel

STRUCTURES = ['L_Accu', 'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 
    'L_Thal', 'R_Accu', 'R_Amyg', 'R_Caud', 'R_Hipp', 'R_Pall', 'R_Puta', 
    'R_Thal', 'BrStem']

def check_surf_anat_dir_complete(dir):
    """Check that dir contains output from FIRST, FAST and FreeSurfer"""

    return all([
        op.isdir(op.join(dir, 'fs', 'surf')), 
        op.isdir(op.join(dir, 'first_results')), 
        op.isfile(op.join(dir, 'T1_fast_pve_0.nii.gz')), 
    ])



def _loadFIRSTdir(dir):
    """Load surface paths from a FIRST directory into a dict, accessed by the
    standard keys used by FIRST (eg 'BrStem'). The function will attempt to 
    load every available surface found in the directory using the standard
    list as reference (see FIRST documentation) but no errors will be raised
    if a particular surface is not found
    """

    if not op.isdir(dir):
        raise RuntimeError("FIRST directory does not exist")

    surfs = {}
    files = glob.glob(op.join(dir, '*.vtk'))

    for f in files: 
        fname = op.split(f)[1]
        for s in STRUCTURES:
            if s in fname:
                surfs[s] = f 
        
    if not len(surfs):
        raise RuntimeError("No surfaces were found")

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
        surfs[s] = op.join(sdir, snames[s[0]] + exts[s[1]])

    if not all(map(op.isfile, surfs.values())):
        raise RuntimeError("One of the subject's surfaces does not exist")

    return surfs


def _default_output_path(dir, fname, suffix='', ext=True):
    """Produce a default path from a dir, filename, optionally adding
    a suffix and preserving extensions of given filename. 

    Args: 
        dir: directory to serve as path root
        fname: file for the basename, eg file1.txt -> file1
        suffix: to add onto the file eg _suff -> file1_suff
        ext: bool, preserve the extension of the fname 

    Returns: 
        path
    """
    
    if op.isfile(dir):
        dir = op.dirname(dir)
    fname = op.split(fname)[1]
    fname, fexts = _splitExts(fname)
    name = _addSuffixToFilename(suffix, fname)
    out = op.join(dir, name)
    if ext:
        out = out + fexts 
    return out 


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
        os.mkdir(dir)


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


def _adjustFLIRT(source, reference, transform):
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

    # Local function to read out an FSL-specific affine matrix from an image
    def __getFSLspace(imgPth):
        obj = nibabel.load(imgPth)
        if obj.header['dim'][0] < 3:
            raise RuntimeError("Volume has less than 3 dimensions" + \
                 "cannot resolve space")

        sform = obj.affine
        det = np.linalg.det(sform[0:4, 0:4])
        ret = np.identity(4)
        pixdim = obj.header['pixdim'][1:4]
        for d in range(3):
            ret[d,d] = pixdim[d]

        # Check the xyzt field to find the spatial units. 
        xyzt =str(obj.header['xyzt_units'])
        if xyzt == '01': 
            multi = 1000
        elif xyzt == '10':
            multi = 1 
        elif xyzt =='11':
            multi = 1e-3
        else: 
            raise RuntimeError("Unknown units")

        if det > 0:
            ret[0,0] = -pixdim[0]
            ret[0,3] = (obj.header['dim'][1] - 1) * pixdim[0]

        ret = ret * multi
        ret[3,3] = 1
        return ret

    # Main function
    srcSpace = __getFSLspace(source)
    refSpace = __getFSLspace(reference)

    refObj = nibabel.load(reference)
    refAff = refObj.affine 
    srcObj = nibabel.load(source)
    srcAff = srcObj.affine 

    outAff = np.matmul(np.matmul(
        np.matmul(refAff, np.linalg.inv(refSpace)),
        transform), srcSpace)
    return np.matmul(outAff, np.linalg.inv(srcAff))


def _affineTransformPoints(points, affine):
    """Apply affine transformation to set of points.

    Args: 
        points: n x 3 matrix of points to transform
        affine: 4 x 4 matrix for transformation

    Returns: 
        transformed copy of points 
    """

    # Add 1s on the 4th column, transpose and multiply, 
    # then re-transpose and drop 4th column  
    transfd = np.ones((points.shape[0], 4))
    transfd[:,0:3] = points
    transfd = np.matmul(affine, transfd.T).astype(np.float32)
    return (transfd[0:3,:]).T


def _coordinatesForGrid(ofSize):
    """Produce N x 3 array of all voxel indices (eg [10, 18, 2]) within
    a grid of size ofSize, 0-indexed and in integer form. 
    """

    I, J, K = np.unravel_index(np.arange(np.prod(ofSize)), ofSize)
    cents = np.vstack((I.flatten(), J.flatten(), K.flatten())).T
    return cents.astype(np.int32)


def _distributeObjects(objs, ngroups):
    """Distribute a set of objects into n groups.
    For preparing chunks before multiprocessing.pool.map"""

    chunkSize = np.floor(len(objs) / ngroups).astype(np.int32)
    chunks = [] 

    for n in range(ngroups):
        if n != ngroups - 1: 
            chunks.append(objs[n * chunkSize : (n+1) * chunkSize])
        else:
            chunks.append(objs[n * chunkSize :])

    assert sum(map(len, chunks)) == len(objs), \
        "Distribute objects error: not all objects distributed"        

    return chunks 


def _clipArray(arr, mini=0.0, maxi=1.0):
    """Clip array values into range [mini, maxi], default [0 1]"""

    arr[arr < mini] = mini 
    arr[arr > maxi] = maxi 
    return arr 