"""Toblerone utility functions"""

import os.path as op
import glob
import warnings 
import multiprocessing
import copy 

import numpy as np 
import regtricks as rt
from scipy.sparse.linalg import eigs

NP_FLOAT = np.float32

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


def check_anat_dir(dir):
    """Check that dir contains output from FIRST and FAST"""

    return all([
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
    snames = {'L': 'lh', 'R': 'rh'}
    exts = {'W': 'white', 'P': 'pial'}
    for s in ['LWS', 'LPS', 'RWS', 'RPS']:
        path = op.join(fsdir, 'surf', '%s.%s' % (snames[s[0]], exts[s[1]]))
        if not op.exists(path):
            raise RuntimeError("Could not find a file for %s in %s" % (s,fsdir))
        surfs[s] = path 
    return surfs


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


def affine_transform(points, affine):
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
    transfd = np.matmul(affine, transfd.T).astype(NP_FLOAT)
    return np.squeeze(transfd[0:3,:].T)


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


@cascade_attributes
def enforce_and_load_common_arguments(func):
    """
    Decorator to enforce and pre-processes common arguments in a 
    kwargs dict that are used across multiple functions. Note
    some function-specific checking is still required. This intercepts the
    kwargs dict passed to the caller, does some checking and modification 
    in place, and then returns to the caller. The following args are handled:

    Required args:
        ref: path to a reference image in which to operate. 
        struct2ref: path/np.array/Registration representing transformation
            between structural space (that of the surfaces) and reference. 
            If given as 'I', identity matrix will be used. 

    Optional args: 
        fslanat: a fslanat directory. 
        flirt: bool denoting that the struct2ref is a FLIRT transform.
            This means it requires special treatment. If set, then it will be
            pre-processed in place by this function, and then the flag will 
            be set back to false when the kwargs dict is returned to the caller
        struct: if FLIRT given, then the path to the structural image used
            for surface generation is required for said special treatment. 
        cores: maximum number of cores to parallelise tasks across 
            (default is N-1). 

    Returns: 
        a modified copy of kwargs dictionary passed to the caller. 
    """
    
    def enforcer(ref, struct2ref, **kwargs):

        # Reference image path 
        if (not isinstance(ref, rt.ImageSpace)): 
            ref = rt.ImageSpace(ref)

        # If given a fslanat dir we can load the structural image in 
        if kwargs.get('fslanat'):
            if not check_anat_dir(kwargs['fslanat']):
                raise RuntimeError("fslanat is not complete: it must contain " 
                    "FAST output and a first_results subdirectory")

            kwargs['fastdir'] = kwargs['fslanat']
            kwargs['firstdir'] = op.join(kwargs['fslanat'], 'first_results')

            # If no struct image given, try and pull it out from the anat dir
            # But, if it has been cropped relative to original T1, then give
            # warning (as we will not be able to convert FLIRT to world-world)
            if not kwargs.get('struct'): 
                if kwargs.get('flirt'):
                    matpath = glob.glob(op.join(kwargs['fslanat'], '*nonroi2roi.mat'))[0]
                    nonroi2roi = np.loadtxt(matpath)
                    if np.any(np.abs(nonroi2roi[0:3,3])):
                        print("Warning: T1 was cropped relative to T1_orig within" + 
                            " fslanat dir.\n Please ensure the struct2ref FLIRT" +
                            " matrix is referenced to T1, not T1_orig.")

                s = op.join(kwargs['fslanat'], 'T1.nii.gz')
                kwargs['struct'] = s
                if not op.isfile(s):
                    raise RuntimeError("Could not find T1.nii.gz in the fslanat dir")

 
        # Structural to reference transformation. Either as array, path
        # to file containing matrix, or regtricks Registration object 
        if not any([type(struct2ref) is str, type(struct2ref) is np.ndarray,
                    type(struct2ref) is rt.Registration ]):
            raise RuntimeError("struct2ref transform must be given (either path,", 
                " np.array or regtricks Registration object)")

        else:
            s2r = struct2ref

            if (type(s2r) is str): 
                if s2r == 'I':
                    matrix = np.identity(4)
                else:
                    _, matExt = op.splitext(s2r)

                    try: 
                        if matExt in ['.txt', '.mat']:
                            matrix = np.loadtxt(s2r, 
                                dtype=NP_FLOAT)
                        elif matExt in ['.npy', 'npz', '.pkl']:
                            matrix = np.load(s2r)
                        else: 
                            matrix = np.fromfile(s2r, 
                                dtype=NP_FLOAT)

                    except Exception as e:
                        warnings.warn("""Could not load struct2ref matrix. 
                            File should be any type valid with numpy.load().""")
                        raise e 

                struct2ref = matrix

        # If FLIRT transform we need to do some clever preprocessing
        # We then set the flirt flag to false again (otherwise later steps will 
        # repeat the tricks and end up reverting to the original - those steps don't
        # need to know what we did here, simply that it is now world-world again)
        if kwargs.get('flirt'):
            if not kwargs.get('struct'):
                raise RuntimeError("If using a FLIRT transform, the path to the"
                    " structural image must also be given")
            
            struct2ref = rt.Registration.from_flirt(struct2ref, 
                                        kwargs['struct'], ref).src2ref
            kwargs['flirt'] = False 
        elif isinstance(struct2ref, rt.Registration): 
            struct2ref = struct2ref.src2ref 
        
        assert isinstance(struct2ref, np.ndarray), 'should have cast struc2ref to np.array'

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

        return ref, struct2ref, kwargs

    def common_args_enforced(ref, struct2ref, *args, **kwargs):
        ref, struct2ref, enforced = enforcer(ref, struct2ref, **kwargs)
        return func(ref, struct2ref, *args, **enforced)

    return common_args_enforced


def sparse_normalise(mat, axis, threshold=1e-6): 
    """
    Normalise a sparse matrix so that all rows (axis=1) or columns (axis=0)
    sum to either 1 or zero. NB any rows or columns that sum to less than 
    threshold will be rounded to zeros.

    Args: 
        mat: sparse matrix to normalise 
        axis: dimension for which sum should equal 1 (1 for row, 0 for col)
        threshold: any row/col with sum < threshold will be set to zero  

    Returns: 
        sparse matrix, same format as input. 
    """

    # Make local copy - otherwise this function will modify the caller's copy 
    constructor = type(mat)
    mat = copy.deepcopy(mat)

    if axis == 0:
        matrix = mat.tocsr()
    elif axis == 1: 
        matrix = mat.tocsc()
    else: 
        raise ValueError("Axis must be 0 or 1")

    # Set threshold. Round any row/col below this to zeros 
    norm = mat.sum(axis).A.flatten()
    fltr = (norm > threshold)
    normalise = np.zeros(norm.size)
    normalise[fltr] = 1 / norm[fltr]
    matrix.data *= np.take(normalise, matrix.indices)

    # Sanity check
    sums = matrix.sum(axis).A.flatten()
    assert np.abs((sums[sums > 0] - 1)).max() < (1e2 * threshold), 'Did not normalise to 1'
    return constructor(matrix)


def is_symmetric(a, tol=1e-9): 
    return not (np.abs(a - a.T) > tol).max()


def is_nsd(a):
    return not (eigs(a)[0] > 0).any()


def calc_midsurf(in_surf, out_surf):
    """
    Midsurface between two Surfaces
    """
    from .classes import Surface
    vec = out_surf.points - in_surf.points 
    points =  in_surf.points + (0.5 * vec)
    return Surface.manual(points, in_surf.tris)


def calculateXprods(points, tris):
    """
    Normal vectors for points,triangles array. 
    For triangle vertices ABC, this is calculated as (C - A) x (B - A). 
    """

    return np.cross(
        points[tris[:,2],:] - points[tris[:,0],:], 
        points[tris[:,1],:] - points[tris[:,0],:], 
        axis=1)


def slice_sparse(mat, slice0, slice1):
    """
    Slice a block out of a sparse matrix, ie mat[slice0,slice1]. 
    Scipy sparse matrices do not support slicing in this manner (unlike numpy)

    Args: 
        mat (sparse): of any form 
        slice0 (bool,array): mask to apply on axis 0 (rows)
        slice1 (bool,array): mask to apply on axis 1 (columns)

    Returns: 
        CSR matrix
    """
    
    out = mat.tocsc()[:,slice1]
    return out.tocsr()[slice0,:]


def rebase_triangles(points, tris, tri_inds):
    """
    Re-express a patch of a larger surface as a new points and triangle
    matrix pair, indexed from 0. Useful for reducing computational 
    complexity when working with a small patch of a surface where only 
    a few nodes in the points array are required by the triangles matrix. 

    Args: 
        points (np.array): surface vertices, P x 3
        tris (np.array): surface triangles, T x 3
        tri_inds (np.array): row indices into triangles array, to rebase
    
    Returns: 
        (points, tris) tuple of re-indexed points/tris. 
    """

    ps = np.empty((0, 3), dtype=NP_FLOAT)
    ts = np.empty((len(tri_inds), 3), dtype=np.int32)
    pointsLUT = []

    for t in range(len(tri_inds)):
        for v in range(3):

            # For each vertex of each tri, check if we
            # have already processed it in the LUT
            vtx = tris[tri_inds[t],v]
            idx = np.argwhere(pointsLUT == vtx)

            # If not in the LUT, then add it and record that
            # as the new position. Write the missing vertex
            # into the local points array
            if not idx.size:
                pointsLUT.append(vtx)
                idx = len(pointsLUT) - 1
                ps = np.vstack([ps, points[vtx,:]])

            # Update the local triangle
            ts[t,v] = idx

    return (ps, ts)


def space_encloses_surface(space, points_vox):

    if np.round(np.min(points_vox)) < 0: 
        return False 
    if (np.round(np.max(points_vox, axis=0)) >= space.size).any(): 
        return False 
    return True 


def load_surfs_to_hemispheres(**kwargs):

    from .classes import Hemisphere
    # If subdir given, then get all the surfaces out of the surf dir
    # If individual surface paths were given they will already be in scope
    if kwargs.get('fsdir'):
        surfdict = _loadSurfsToDict(kwargs['fsdir'])
        kwargs.update(surfdict)

    # What hemispheres are we working with?
    sides = []
    if np.all([ (kwargs.get(s) is not None) for s in ['LPS', 'LWS'] ]): 
        sides.append('L')

    if np.all([ kwargs.get(s) is not None for s in ['RPS', 'RWS'] ]): 
        sides.append('R')

    if not sides:
        raise RuntimeError("At least one hemisphere (eg LWS/LPS) required")

    hemispheres = [ Hemisphere(kwargs[s+'WS'], kwargs[s+'PS'], s) 
        for s in sides ] 

    return hemispheres