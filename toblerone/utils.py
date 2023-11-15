"""Toblerone utility functions"""

import copy
import glob
import itertools
import os.path as op

import numpy as np
import regtricks as rt
import trimesh
from scipy import sparse

from . import icosphere

NP_FLOAT = np.float32

FIRST_STRUCTURES = [
    "L_Accu",
    "L_Amyg",
    "L_Caud",
    "L_Hipp",
    "L_Pall",
    "L_Puta",
    "L_Thal",
    "R_Accu",
    "R_Amyg",
    "R_Caud",
    "R_Hipp",
    "R_Pall",
    "R_Puta",
    "R_Thal",
    "BrStem",
]

FREESURFER_SUBCORT_LUT = {
    # Left hemisphere
    2: "WM",  # Cerebral WM
    # 3: "GM", # Cerebral GM - ignored because surfaces will be used
    4: "CSF",  # Ventricle
    5: "CSF",  # Ventricle
    10: "L_Thal",
    11: "L_Caud",
    12: "L_Puta",
    13: "L_Pall",
    17: "L_Hipp",
    18: "L_Amyg",
    26: "L_Accu",
    28: "WM",  # Left ventral DC
    30: "WM",  # Left vessel
    31: "WM",  # Left choroid plexus
    # Right hemisphere
    41: "WM",  # Cerebral WM
    # 42: "GM", # Cerebral GM - ignored because surfaces will be used
    43: "CSF",  # Ventricle
    44: "CSF",  # Ventricle
    49: "R_Thal",
    50: "R_Caud",
    51: "R_Puta",
    52: "R_Pall",
    53: "R_Hipp",
    54: "R_Amyg",
    58: "R_Accu",
    60: "WM",  # Right ventral DC
    62: "WM",  # Right vessel
    63: "WM",  # Right choroid plexus
    # Neither hemi
    24: "CSF",
    77: "WM",  # WM hyper-intensity
    85: "WM",  # Optic chiasm
    251: "WM",  # CC
    252: "WM",  # CC
    253: "WM",  # CC
    254: "WM",  # CC
    255: "WM",  # CC
    # Left cerebellum
    # 7: "WM",
    # 8: "GM",
    # Right cerebellum
    # 46: "WM",
    # 47: "GM",
    # 16: "WM",  # Brainstem
}


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

    return all(
        [
            op.isdir(op.join(dir, "first_results")),
            op.isfile(op.join(dir, "T1_fast_pve_0.nii.gz")),
        ]
    )


def _loadFIRSTdir(dir_path, sides=["L", "R"]):
    """Load surface paths from a FIRST directory into a dict, accessed by the
    standard keys used by FIRST (eg 'BrStem'). The function will attempt to
    load every available surface found in the directory using the standard
    list as reference (see FIRST documentation) but no errors will be raised
    if a particular surface is not found
    """

    files = glob.glob(op.join(dir_path, "*.vtk"))
    if not files:
        raise RuntimeError("FIRST directory %s is empty" % dir_path)

    surfs = {}
    for f in files:
        fname = op.split(f)[1]
        for s in FIRST_STRUCTURES:
            for side in sides:
                if s in fname and (fname.count(f"-{side}_")):
                    surfs[s] = f

    return surfs


def _loadFASTdir(d):
    """Load the PV image paths for WM,GM,CSF from a FAST directory into a
    dict, accessed by the keys FAST_GM for GM etc
    """

    if not op.isdir(d):
        raise RuntimeError("FAST directory does not exist")

    paths = {}
    files = glob.glob(op.join(d, "*.nii.gz"))
    channels = {"_pve_{}".format(c): t for (c, t) in enumerate(["CSF", "GM", "WM"])}

    for f in files:
        fname = op.split(f)[1]
        for c, t in channels.items():
            if c in fname:
                paths["FAST_" + t] = f

    if len(paths) != 3:
        raise RuntimeError("Could not load 3 PV maps from FAST directory")

    return paths


def _loadSurfsToDict(fsdir, sides=["L", "R"]):
    """Load the left/right white/pial surface paths from a FS directory into
    a dictionary, accessed by the keys LWS/LPS/RPS/RWS
    """

    sdir = op.realpath(op.join(fsdir, "surf"))

    if not op.isdir(sdir):
        raise RuntimeError("Subject's surf directory does not exist")

    surfs = {}
    snames = {"L": "lh", "R": "rh"}
    exts = {"WS": "white", "PS": "pial", "SS": "sphere"}
    for side, surf in itertools.product(sides, exts.keys()):
        path = op.join(fsdir, "surf", "%s.%s" % (snames[side], exts[surf]))
        if not op.exists(path):
            raise RuntimeError(f"Could not file at {path}")
        surfs[side + surf] = path
    return surfs


def _splitExts(fname):
    """Split all extensions off a filename, eg:
    'file.ext1.ext2' -> ('file', '.ext1.ext2')
    """

    fname = op.split(fname)[1]
    ext = ""
    while "." in fname:
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
        points = points[None, :]

    # Add 1s on the 4th column, transpose and multiply,
    # then re-transpose and drop 4th column
    transfd = np.ones((points.shape[0], 4))
    transfd[:, 0:3] = points
    transfd = np.matmul(affine, transfd.T).astype(NP_FLOAT)
    return np.squeeze(transfd[0:3, :].T)


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
            chunks.append(range(n * chunkSize, (n + 1) * chunkSize))
        else:
            chunks.append(range(n * chunkSize, len(objs)))

    assert sum(map(len, chunks)) == len(
        objs
    ), "Distribute objects error: not all objects distributed"

    return chunks


def cast_ref(ref):
    if not isinstance(ref, rt.ImageSpace):
        ref = rt.ImageSpace(ref)
    return ref


def cast_struct2ref(struct2ref, flirt=False, ref=None, struct=None):
    """Preprocessing for scripts requring struct2ref arg"""

    if isinstance(struct2ref, str) and struct2ref == "I":
        struct2ref = np.eye(4)
    if not isinstance(struct2ref, rt.Registration):
        struct2ref = rt.Registration(struct2ref)
    if flirt:
        struct2ref = rt.Registration.from_flirt(struct2ref.src2ref, struct, ref)
    return struct2ref


def cast_sides(sides):
    """Preprocessing for scripts requring struct2ref arg"""

    if not isinstance(sides, (list, set)):
        raise ValueError("sides should be a set or list")
    return [s.upper() for s in sides]


def cast_supr(supr, ref):
    """Preprocessing for scripts requring struct2ref arg"""

    if supr is None:
        supr = np.maximum(np.floor(ref.vox_size.round(1) / 0.75), 1)
    try:
        supr = np.asanyarray(supr)
        supr = (np.ones(3) * supr).astype(int)
        return supr
    except:
        raise RuntimeError("could not cast supr to numeric value")


def check_spherical(vertices):
    """Check radius of sphere to within 0.1%. Note centre is assumed
    to be at origin."""
    r = np.linalg.norm(vertices, axis=1, ord=2)
    rmin = r.min()
    if (r.max() - rmin) > (rmin / 1000):
        raise RuntimeError("Sphere is not genuinely spherical")


def barycentric_coordinates(points, tris, mesh):
    """Calculate barycentric coordinates for a set of points known to lie within particular mesh triangles.

    Args:
        points (array): p x 3 array of coordinates
        tris (array): p vector of indices for mesh triangles
        mesh (Trimesh): mesh object from which barycentric weights are calculated.

    Returns:
        (array): p x 3, where the first value in each row is the weight assigned to the corresponding triangle's first vertex, etc
    """

    # triangle ABC with a point P inside it
    # calculate vectors from each vertex to each point
    ap, bp, cp = [points - mesh.vertices[mesh.faces[tris, i]] for i in range(3)]

    # 3 subtriangles ABP, ACP, BCP
    # Calculate the area of each
    Apbc = 0.5 * np.linalg.norm(np.cross(bp, cp), ord=2, axis=-1)
    Aapc = 0.5 * np.linalg.norm(np.cross(ap, cp), ord=2, axis=-1)
    Aabp = 0.5 * np.linalg.norm(np.cross(ap, bp), ord=2, axis=-1)

    # Weights are defined by subtri area / total area
    areas = mesh.area_faces[tris]
    weights = np.array((Apbc, Aapc, Aabp)).T / areas[:, None]

    # Weights should sum to unity
    if not np.allclose(weights.sum(1), 1, atol=1e-6):
        raise RuntimeError("Barycentric weights do not sum to 1")

    return weights


def interpolate_barycentric(weights, vertices, from_triangles):
    """Interpolate into a triangular mesh using barycentric weights. Note only the triangles that are known to correspond to the weights should be provided, not the complete mesh triangles array.

    Args:
        weights (array): p x 3 barycentric weights
        vertices (array): v x 3 triangular mesh vertices
        from_triangles (array): p x 3 triangles corresponding to each row of weights

    Returns:
        (array): p x 3 of interpolated vertex coordinates
    """

    out = np.zeros_like(weights)

    for idx in range(3):
        out += vertices[from_triangles[:, idx]] * weights[:, idx, None]

    return out


def find_sphere_centre(vertices):
    """Calculate sphere centre by taking bounding box"""
    mi = vertices.min(0)
    ma = vertices.max(0)
    return 0.5 * (mi + ma)


def round_voxel_coordinates(a):
    """Rounds an array with values of 0.5 going towards zero"""
    high = a.max(0)
    low = a.min(0)
    rhigh = np.rint(np.nextafter(high, high - 1)).astype(int)
    rlow = np.rint(np.nextafter(low, low + 1)).astype(int)
    return rlow, rhigh


# Rescale to unit radius
def make_unit_radius(tri_mesh):
    verts = tri_mesh.vertices
    norm = np.linalg.norm(verts, ord=2, axis=1)
    verts = verts / norm[:, None]
    if not np.allclose(np.linalg.norm(verts, ord=2, axis=1), 1, atol=1e-9):
        raise ValueError("Did not scale sphere to unit radius")

    return trimesh.Trimesh(vertices=verts, faces=tri_mesh.faces)


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
    fltr = norm > threshold
    normalise = np.zeros(norm.size)
    normalise[fltr] = 1 / norm[fltr]
    matrix.data *= np.take(normalise, matrix.indices)

    # Sanity check
    sums = matrix.sum(axis).A.flatten()
    assert np.allclose(sums[sums > 0], 1, atol=1e-4), "Did not normalise to 1"
    return constructor(matrix)


def laplacian_is_valid(a, tol=1e-5):
    a = a.tocsr()
    if not a.nnz:
        raise ValueError("Laplacian is empty")

    # Check that the diagonal is zero
    dia = sparse.diags(a.diagonal())
    if (a.diagonal() > 0).any():
        raise ValueError("Diagonal of laplacian is positive")

    # Check evenly-weighted
    if (np.abs(a.sum(1)) > tol).any():
        raise ValueError("Laplacian row sum non-zero")

    # Check all off diagnonal elements are positive
    if ((a - dia).data < 0).any():
        raise ValueError("Laplacian has negative off-diagonal elements")

    # Check symmetric
    if (np.abs(a - a.T).data > tol).any():
        raise ValueError("Laplacian is not symmetric")

    return True


def calculateXprods(points, tris):
    """
    Normal vectors for points,triangles array.
    For triangle vertices ABC, this is calculated as (C - A) x (B - A).
    """

    return np.cross(
        points[tris[:, 2], :] - points[tris[:, 0], :],
        points[tris[:, 1], :] - points[tris[:, 0], :],
        axis=1,
    )


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

    out = mat.tocsc()[:, slice1]
    return out.tocsr()[slice0, :]


def mask_projection_matrix(matrix, row_mask, col_mask):
    """
    Mask a sparse projection matrix, whilst preserving total signal intensity.
    For example, if the mask implies discarding voxels from a vol2surf matrix,
    upweight the weights of the remaining voxels to account for the discarded
    voxels. Rows represent the target domain and columns the source.

    Args:
        matrix (sparse): of any form, shape (N,M)
        row_mask (np.array,bool): flat vector of size N, rows to retain in matrix
        col_mask (np.array,bool): flat vector of size M, cols to retain in matrix

    Returns:
        CSR matrix, shape (sum(row_mask), sum(col_mask))
    """

    if (row_mask.dtype.kind != "b") or (col_mask.dtype.kind != "b"):
        raise ValueError("Row and column masks must be boolean arrays")

    if not (row_mask.shape[0] == row_mask.size == matrix.shape[0]):
        raise ValueError("Row mask size does not match matrix shape")

    if not (col_mask.shape[0] == col_mask.size == matrix.shape[1]):
        raise ValueError("Column mask size does not match matrix shape")

    # Masking by rows is easy because they represent the output domain - ie,
    # we can just drop the rows and ignore.
    matrix = matrix[row_mask, :]

    # Masking by cols is harder because we seek to preserve the overall signal
    # intensity from source to output. So for every bit of source signal that
    # goes missing, we want to upscale the remainder so that the overall
    # magnitude of output remains the same.
    matrix_masked = matrix[:, col_mask]

    orig_weights = matrix.sum(1).A.flatten()
    new_weights = matrix_masked.sum(1).A.flatten()
    valid = new_weights > 0
    sf = np.zeros_like(orig_weights)
    sf[valid] = orig_weights[valid] / new_weights[valid]

    matrix_csc = matrix_masked.tocsc()
    matrix_csc.data = matrix_csc.data * np.take(sf, matrix_csc.indices)
    matrix_new = matrix_csc.tocsr()

    final_weights = matrix_new.sum(1).A.flatten()
    if not np.allclose(final_weights[valid], orig_weights[valid]):
        raise RuntimeError("Masked matrix did not re-scale correctly")

    return matrix_new


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
            vtx = tris[tri_inds[t], v]
            idx = np.argwhere(pointsLUT == vtx)

            # If not in the LUT, then add it and record that
            # as the new position. Write the missing vertex
            # into the local points array
            if not idx.size:
                pointsLUT.append(vtx)
                idx = len(pointsLUT) - 1
                ps = np.vstack([ps, points[vtx, :]])

            # Update the local triangle
            ts[t, v] = idx

    return (ps, ts)


def space_encloses_surface(space, surface):
    points_vox = affine_transform(surface.points, space.world2vox)
    low, high = round_voxel_coordinates(points_vox)
    if (low < 0).any():
        return False
    elif (high >= space.size).any():
        return False
    else:
        return True


def load_surfs_to_hemispheres(surf_dict):
    from .classes import Hemisphere, Surface

    # What hemispheres are we working with?
    sides = []
    if all([surf_dict.get(s) is not None for s in ["LPS", "LWS"]]):
        sides.append("L")

    if all([surf_dict.get(s) is not None for s in ["RPS", "RWS"]]):
        sides.append("R")

    if not sides:
        raise RuntimeError("At least one hemisphere (eg LWS/LPS) required")

    hemispheres = [
        Hemisphere(
            insurf=surf_dict[s + "WS"],
            outsurf=surf_dict[s + "PS"],
            sphere=surf_dict.get(s + "SS"),
            side=s,
        )
        for s in sides
    ]

    return hemispheres


def stack_images(images):
    """
    Combine the results of estimate_complete() into overall PV maps
    for each tissue. Note that the below logic is entirely specific
    to the surfaces produced by FreeSurfer, FIRST and how they may be
    combined with FAST estimates. If you're going off-piste anywhere else
    then you probably DON'T want to re-use this logic.

    Args:
        dictionary of PV maps, keyed as follows: all FIRST subcortical
        structures named by their FIST convention (eg L_Caud); FAST's estimates
        named as FAST_CSF/WM/GM; cortex estimates as cortex_GM/WM/non_brain

    Returns:
        single 4D array of PVs, arranged GM/WM/non-brain in the 4th dim
    """

    # The logic is as follows:
    # Initialise everything as non-brain
    # Write in PV estimates from the cortical surfaces. This sets the cortical GM
    # and also all subcortical tissue as WM
    # Layer on top FAST's CSF estimates as non-brain. This is because surface methods
    # dom't pick up ventricular or mid-brain CSF, so wherever FAST has found more CSF
    # than Toblerone we will take that higher estimate
    # Layer in subcortical GM from each individual FIRST structure (except brain stem).
    # After adding in each structure's GM, recalculate CSF as either the existing amount,
    # or reduce the CSF estimate if the total (GM + CSF) > 1
    # If there is any remainder unassigned in the voxel, set that as WM

    # To summarise, the tissues are stacked up as:
    # All CSF
    # Cortical GM
    # Then within the subcortex only:
    # All subcortical volume set as WM
    # Subcortical CSF fixed using FAST
    # Add in subcortical GM for each structure, reducing CSF if required
    # Set the remainder 1 - (GM+CSF) as WM in voxels that were updated

    # Pop out nonbrain estimates
    csf = images.pop("nonbrain")
    shape = (*csf.shape[0:3], 3)
    csf = csf.flatten()

    # Squash small FAST CSF values
    csf[csf < 0.01] = 0

    # Pop the cortex estimates and initialise output as all CSF
    ctxgm = images.pop("cortex_GM").flatten()
    ctxwm = images.pop("cortex_WM").flatten()
    ctxnon = images.pop("cortex_nonbrain").flatten()
    ctx = np.vstack((ctxgm, ctxwm, ctxnon)).T
    out = np.zeros_like(ctx)
    out[:, 2] = 1

    # Then write in Toblerone's cortex estimates from all voxels
    # that contain either WM or GM (on the ctx image)
    mask = np.logical_or(ctx[:, 0], ctx[:, 1])
    out[mask, :] = ctx[mask, :]

    # Layer in FAST's CSF estimates (to get mid-brain and ventricular CSF).
    # Where FAST has suggested a higher CSF estimate than currently exists,
    # and the voxel does not intersect the cortical ribbon, accept FAST's
    # estimate. Then update the WM estimates, reducing where necessary to allow
    # for the greater CSF volume
    GM_threshold = 0.01
    ctxmask = ctx[:, 0] > GM_threshold
    to_update = np.logical_and(csf > out[:, 2], ~ctxmask)
    tmpwm = out[to_update, 1]
    out[to_update, 2] = np.minimum(csf[to_update], 1 - out[to_update, 0])
    out[to_update, 1] = np.minimum(tmpwm, 1 - (out[to_update, 2] + out[to_update, 0]))

    # Sanity checks: total tissue PV in each vox should sum to 1
    assert (out[to_update, 0] <= GM_threshold).all(), "Some update voxels have GM"
    assert (np.abs(out.sum(1) - 1) < 1e-6).all(), "Voxel PVs do not sum to 1"
    assert (out > -1e-6).all(), "Negative PV found"

    # All remaining keys are assumed to be subcortical structures
    # For each subcortical structure, create a mask of the voxels which it
    # relates to. The following operations then apply only to those voxels
    # All subcortical structures interpreted as pure GM
    # Update CSF to ensure that GM + CSF in those voxels < 1
    # Finally, set WM as the remainder in those voxels.
    for k, s in images.items():
        smask = s.flatten() > 0
        out[smask, 0] = np.minimum(1, out[smask, 0] + s.flatten()[smask])
        out[smask, 2] = np.minimum(out[smask, 2], 1 - out[smask, 0])
        out[smask, 1] = np.maximum(1 - (out[smask, 0] + out[smask, 2]), 0)
        assert (out > -1e-6).all(), f"Negative found after {k} layer"
        assert (np.abs(out.sum(1) - 1) < 1e-6).all(), f"PVs sum > 1 after {k} layer"

    # Final sanity check, then rescaling so all voxels sum to unity.
    # assert (out > -1e-6).all()
    out[out < 0] = 0
    sums = out.sum(1)
    assert (np.abs(out.sum(1) - 1) < 1e-6).all(), "Voxel PVs do not sum to 1"
    out = out / sums[:, None]

    return out.reshape(shape)
