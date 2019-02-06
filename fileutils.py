# File and folder utilities for pvtools

import os.path as op
import os 
import copy
import glob

from .classes import STRUCTURES


def _check_pvdir(pvdir):
    for d in ['fast', 'fs', 'first']:
        p = op.join(pvdir, d)
        if not op.isdir(p):
            print("pvdir is not complete (missing %s)" % p)
            return False

    return True


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


def default_output_path(dir, fname, suffix='', ext=True):
    """Produce a default path from a dir, filename, optionally adding
    a suffix and preserving extensions. 

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
    fname, fexts = splitExts(fname)
    name = _addSuffixToFilename(suffix, fname)
    out = op.join(dir, name)
    if ext:
        out = out + fexts 
    return out 


def _addSuffixToFilename(suffix, fname):
    """Add suffix to filename, whilst preserving original extension, eg:
    'file.ext1.ext2' + '_suffix' -> 'file_suffix.ext1.ext2'"""
    head = op.split(fname)[0]
    fname, ext = splitExts(fname)   
    return op.join(head, fname + suffix + ext)

def _addPrefixToFilename(prefix, fname):
    """Add prefix to filename, whilst preserving original extension, eg:
    'prefix_' + file.ext1.ext2' -> 'prefix_file_suffix.ext1.ext2'"""
    head = op.split(fname)[0]
    fname, ext = splitExts(fname)   
    return op.join(head, prefix + fname + ext)


def splitExts(fname):
    """Split all extensions off a filename, eg:
    'file.ext1.ext2' -> ('file', '.ext1.ext2')
    """
    fname = op.split(fname)[1]
    ext = ''
    while '.' in fname:
        fname, e = op.splitext(fname)
        ext = e + ext 
    
    return fname, ext

def weak_mkdir(dir):
    """Create a directory if it does not already exist"""
    if not op.isdir(dir):
        os.mkdir(dir)
