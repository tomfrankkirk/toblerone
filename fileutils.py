import os.path as op
import os 
import copy
import glob

from .classes import STRUCTURES

def loadFIRSTdir(dir):

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
