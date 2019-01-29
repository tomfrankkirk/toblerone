import os.path as op
import os 
import copy
import glob

from .classes import STRUCTURES

def _loadFIRSTdir(dir):

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


def _loadSurfsToDict(FSdir):
    sdir = op.realpath(op.join(FSdir, 'surf'))

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


def _addSuffixToFilename(suffix, fname):
    """Add suffix to filename, whilst preserving original extension"""
    fname, ext = splitExts(fname)   
    return fname + suffix + ext 


def splitExts(fname):
    fname = op.split(fname)[1]
    ext = ''
    while '.' in fname:
        fname, e = op.splitext(fname)
        ext = e + ext 
    
    return fname, ext

def weak_mkdir(dir):
    if not op.isdir(dir):
        os.mkdir(dir)
