import functools
import logging
import multiprocessing as mp
import os.path as op

import nibabel as nib
import numpy as np
import regtricks as rt
from tqdm import tqdm

from toblerone import core
from toblerone import surface_estimators as estimators
from toblerone import utils
from toblerone.classes import ImageSpace, Surface
from toblerone.projection import Projector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pvs_cortex_freesurfer(
    *,
    ref=None,
    struct2ref=None,
    sides=["L", "R"],
    LWS=None,
    LPS=None,
    LSS=None,
    RWS=None,
    RPS=None,
    RSS=None,
    fsdir=None,
    resample=32492,
    ones=False,
    cores=mp.cpu_count(),
    supr=None,
    flirt=False,
    struct=None,
):
    """
    Estimate PVs for L/R cortex. To estimate for a single hemisphere,
    provide only surfaces for that side.

    Args:
        ref (str/regtricks ImageSpace): voxel grid in which to estimate PVs.
        struct2ref (str/np.array/rt.Registration): registration between space
            of surface and reference [see `-flirt` and `-stuct`], use 'I' for identity.
        fsdir (str): path to a FreeSurfer subject directory, overrides LWS etc
        LWS/LPS/RWS/RPS (str): individual paths to the surfaces,
            eg LWS = Left White surface, RPS = Right Pial surace.

    Other parameters:
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct.
        struct (str): path to structural image from which surfaces were derived.
        cores (int): number of cores to use, default 8.
        supr (int/array): single or 3 values, supersampling factor.

    Returns:
        (np.array) PVs arranged GM/WM/non-brain in 4th dim.
    """

    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)
    sides = utils.cast_sides(sides)
    supr = utils.cast_supr(supr, ref)

    if fsdir:
        surfs = utils._loadSurfsToDict(fsdir, sides)
    else:
        surfs = dict(LWS=LWS, LPS=LPS, LSS=LSS, RWS=RWS, RPS=RPS, RSS=RSS)

    hemispheres = utils.load_surfs_to_hemispheres(surfs)
    if resample:
        hemispheres = [h.resample_geometry(resample) for h in hemispheres]

    pvs = estimators.cortex(hemispheres, ref, struct2ref, supr, cores, ones)

    return pvs


def pvs_structure(
    *,
    ref=None,
    struct2ref=None,
    surf=None,
    ones=False,
    cores=mp.cpu_count(),
    supr=None,
    flirt=False,
    struct=None,
    coords="world",
):
    """
    Estimate PVs for a structure defined by a single surface.

    Args:
        ref (str/regtricks ImageSpace): voxel grid in which to estimate PVs.
        struct2ref (str/np.array/rt.Registration): registration between space
            of surface and reference (see -flirt and -stuct). Use 'I' for identity.
        surf (str/`Surface`): path to surface (see coords argument below)

    Other parameters:
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct.
        coords (str): convention by which surface is defined: default is 'world'
            (mm coords), for FIRST surfaces set as 'fsl' and provide struct argument
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8
        supr (int/array): single or 3 values, supersampling factor

    Returns:
        (np.array) PV image, sized equal to reference space
    """

    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)
    supr = utils.cast_supr(supr, ref)

    if coords == "fsl" and not struct:
        raise RuntimeError("Structural image must be supplied for FIRST surfs")

    if type(surf) is str:
        surf = Surface(surf, name=op.split(surf)[1])
        if coords == "fsl":
            struct_spc = ImageSpace(struct)
            surf = surf.transform(struct_spc.FSL2world)

    elif type(surf) is not Surface:
        raise RuntimeError("surf kwarg must be a Surface object or path to one")

    pvs = estimators.structure(surf, ref, struct2ref, supr, ones, cores)

    return pvs


def _pvs_structure_wrapper(surf, **kwargs):
    """Convenience method for parallel processing"""
    return pvs_structure(surf=surf, **kwargs)


def pvs_subcortex_freesurfer(
    *,
    ref=None,
    struct2ref=None,
    fsdir=None,
    cores=mp.cpu_count(),
    flirt=False,
    struct=None,
):
    """Extract subcortical GM and non-brain PVs from FreeSurfer volumetric aseg segmentation"""

    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)

    # Load aseg and initialise a COPY of the FS LUT
    aseg = op.join(fsdir, "mri/aseg.mgz")
    aseg_mgz = nib.load(aseg)
    avol = aseg_mgz.get_fdata().astype(np.int32)
    lut = {**utils.FREESURFER_SUBCORT_LUT}

    # Drop BrStem and cerebellum from LUT
    lut = {k: v for k, v in lut.items() if k not in [16, 7, 8, 46, 47]}

    # Drop all WM and GM from LUT
    lut = {k: v for k, v in lut.items() if v not in ["WM", "GM"]}

    # Label all CSF as zero (there are many labels) and drop from LUT
    for k, v in {**lut}.items():
        if v == "CSF":
            avol[avol == k] = 0
            lut.pop(k)

    # The LUT now contains only deep GM structures, map them
    # into reference space
    output = []
    for k in lut.keys():
        output.append((avol == k))

    output = np.stack(output, axis=-1)
    output = struct2ref.apply_to_array(output, aseg_mgz, ref, order=1, cores=cores)
    output = {v: output[..., idx] for idx, v in enumerate(lut.values())}

    nb = avol == 0
    nb = struct2ref.apply_to_array(nb, aseg_mgz, ref, order=1)
    output["nonbrain"] = nb

    return output


def pvs_subcortex_fsl(
    *,
    ref=None,
    struct2ref=None,
    firstdir=None,
    fastdir=None,
    cores=mp.cpu_count(),
    ones=False,
    supr=None,
    flirt=False,
    struct=None,
):
    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)
    supr = utils.cast_supr(supr, ref)

    fast_paths = utils._loadFASTdir(fastdir)
    struct_spc = ImageSpace(fast_paths["FAST_GM"])

    # Resample FASTs to reference space. Then redefine CSF as 1-(GM+WM)
    fast = np.stack(
        [nib.load(fast_paths[f"FAST_{p}"]).get_fdata() for p in ["GM", "WM"]], axis=-1
    )
    fasts_transformed = struct2ref.apply_to_array(fast, struct_spc, ref, order=1)
    output = {"nonbrain": np.clip(1 - fasts_transformed.sum(-1), 0, 1)}

    # Process subcortical structures first.
    FIRSTsurfs = utils._loadFIRSTdir(firstdir)
    FIRSTsurfs.pop("BrStem", None)
    subcortical = []
    for name, surf in FIRSTsurfs.items():
        s = Surface(surf, name)
        subcortical.append(s.transform(struct_spc.FSL2world))

    # To estimate against each subcortical structure, we apply the following
    # partial func to each using a map() call. Carry kwargs from this func
    desc = "Subcortical structures"
    estimator = functools.partial(
        _pvs_structure_wrapper,
        ref=ref,
        struct2ref=struct2ref,
        cores=1,
        ones=ones,
        supr=supr,
    )

    iterator = functools.partial(
        tqdm, total=len(subcortical), desc=desc, bar_format=core.BAR_FORMAT, ascii=True
    )

    with mp.Pool(cores) as p:
        results = [r for r in iterator(p.imap(estimator, subcortical))]
    output.update(dict(zip([s.name for s in subcortical], results)))

    return output


def pvs_freesurfer_fsl(
    *,
    ref=None,
    struct2ref=None,
    sides=["L", "R"],
    fsdir=None,
    firstdir=None,
    fastdir=None,
    ones=False,
    cores=mp.cpu_count(),
    supr=None,
    flirt=False,
    struct=None,
):
    """
    Estimate PVs for cortex and all structures identified by FIRST within
    a reference image space. Use FAST to fill in non-surface PVs.

    Args:
        ref (str/regtricks.ImageSpace): voxel grid in which to estimate PVs.
        struct2ref (str/np.array/regtricks.Registration): registration between space
            of surface and reference (see -flirt and -stuct). Use 'I' for identity.
        firstdir (str): FIRST directory in which .vtk surfaces are located
        fastdir (str): FAST directory in which _pve_0/1/2 are located
        struct (str): path to structural image from which FIRST surfaces were dervied
        fsdir (str): FreeSurfer subject directory, OR:
        LWS/LPS/RWS/RPS (str): paths to individual surfaces (L/R white/pial)

    Other parameters:
        flirt (bool): denoting struct2ref is FLIRT transform; if so, set struct.
        coords (str): convention by which surface is defined: default is 'world'
            (mm coords), for FIRST surfaces set as 'fsl' and provide struct argument
        struct (str): path to structural image from which surfaces were derived
        cores (int): number of cores to use, default 8
        supr (int/array): single or 3 values, supersampling factor

    Returns:
        (dict) keyed for each structure and also flattened ('stacked')
    """

    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)
    sides = utils.cast_sides(sides)
    supr = utils.cast_supr(supr, ref)

    # Resample FASTs to reference space. Then redefine CSF as 1-(GM+WM)
    fast_paths = utils._loadFASTdir(fastdir)
    fast_spc = fast_paths["FAST_GM"]
    fast = np.stack(
        [nib.load(fast_paths[f"FAST_{p}"]).get_fdata() for p in ["GM", "WM"]], axis=-1
    )
    fasts_transformed = struct2ref.apply_to_array(fast, fast_spc, ref, cores=1)
    output = dict(FAST_GM=fasts_transformed[..., 0], FAST_WM=fasts_transformed[..., 1])
    output["nonbrain"] = np.maximum(0, 1 - (output["FAST_WM"] + output["FAST_GM"]))

    # Process subcortical structures first.
    FIRSTsurfs = utils._loadFIRSTdir(firstdir, sides)
    subcortical = []
    struct_spc = ImageSpace(struct)
    for name, surf in FIRSTsurfs.items():
        s = Surface(surf, name)
        s = s.transform(struct_spc.FSL2world)
        subcortical.append(s)

    # To estimate against each subcortical structure, we apply the following
    # partial func to each using a map() call. Carry kwargs from this func
    desc = "Subcortical structures"
    estimator = functools.partial(
        _pvs_structure_wrapper, ref=ref, struct2ref=struct2ref, cores=1, ones=ones
    )

    iterator = functools.partial(
        tqdm, total=len(subcortical), desc=desc, bar_format=core.BAR_FORMAT, ascii=True
    )

    with mp.Pool(cores) as p:
        results = [r for r in iterator(p.imap(estimator, subcortical))]
    output.update(dict(zip([s.name for s in subcortical], results)))

    # Now do the cortex, then stack the whole lot
    ctx = pvs_cortex_freesurfer(
        ref=ref,
        struct2ref=struct2ref,
        sides=sides,
        fsdir=fsdir,
        ones=ones,
        cores=cores,
        supr=supr,
        flirt=flirt,
        struct=struct,
    )
    for i, t in enumerate(["_GM", "_WM", "_nonbrain"]):
        output["cortex" + t] = ctx[:, :, :, i]

    # Stack the structure specific images into flat tissue maps
    stacked = utils.stack_images({k: v for k, v in output.items() if k != "BrStem"})
    output["GM"] = stacked[:, :, :, 0]
    output["WM"] = stacked[:, :, :, 1]
    output["nonbrain"] = stacked[:, :, :, 2]
    output["stacked"] = stacked

    return output


def pvs_freesurfer(
    *,
    ref=None,
    struct2ref=None,
    fsdir=None,
    sides=["L", "R"],
    ones=False,
    cores=mp.cpu_count(),
    supr=None,
    flirt=False,
    struct=None,
):
    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)
    sides = utils.cast_sides(sides)
    supr = utils.cast_supr(supr, ref)

    output = pvs_subcortex_freesurfer(
        ref=ref, struct2ref=struct2ref, fsdir=fsdir, cores=cores
    )

    # Only keep structures that match against sides (or nonbrain)
    output = {k: v for k, v in output.items() if ((k[0] in sides) or (k == "nonbrain"))}

    # Now do the cortex, then stack the whole lot
    ctx = pvs_cortex_freesurfer(
        ref=ref,
        struct2ref=struct2ref,
        fsdir=fsdir,
        sides=sides,
        ones=ones,
        cores=cores,
        supr=supr,
    )
    for i, t in enumerate(["_GM", "_WM", "_nonbrain"]):
        output["cortex" + t] = ctx[:, :, :, i]

    # Stack the structure specific images into flat tissue maps
    stacked = utils.stack_images({k: v for k, v in output.items() if k != "BrStem"})
    output["GM"] = stacked[:, :, :, 0]
    output["WM"] = stacked[:, :, :, 1]
    output["nonbrain"] = stacked[:, :, :, 2]
    output["stacked"] = stacked

    return output


def projector_freesurfer(
    *,
    ref=None,
    struct2ref=None,
    fsdir=None,
    resample=32492,
    sides=["L", "R"],
    ones=False,
    cores=mp.cpu_count(),
    flirt=False,
    struct=None,
):
    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)
    sides = utils.cast_sides(sides)

    surfs = utils._loadSurfsToDict(fsdir, sides)
    hemispheres = utils.load_surfs_to_hemispheres(surfs)
    if resample:
        logger.info(f"Resampling hemispheres to {resample} vertices")
        hemispheres = [h.resample_geometry(resample) for h in hemispheres]

    subcort_pvs = pvs_subcortex_freesurfer(
        ref=ref, struct2ref=struct2ref, fsdir=fsdir, cores=cores
    )

    nb_pvs = subcort_pvs.pop("nonbrain")
    p = Projector(
        ref,
        struct2ref,
        hemispheres=hemispheres,
        roi_pvs=subcort_pvs,
        nonbrain_pvs=nb_pvs,
        cores=cores,
        ones=ones,
    )

    return p


def projector_freesurfer_fsl(
    *,
    ref=None,
    struct2ref=None,
    sides=["L", "R"],
    fsdir=None,
    resample=32492,
    firstdir=None,
    fastdir=None,
    ones=False,
    cores=mp.cpu_count(),
    flirt=False,
    struct=None,
):
    # Cast inputs to expected types
    ref = utils.cast_ref(ref)
    struct2ref = utils.cast_struct2ref(struct2ref, flirt, ref, struct)
    sides = utils.cast_sides(sides)

    surfs = utils._loadSurfsToDict(fsdir, sides)
    hemispheres = utils.load_surfs_to_hemispheres(surfs)
    if resample:
        hemispheres = [h.resample_geometry(resample) for h in hemispheres]

    subcort_pvs = pvs_subcortex_fsl(
        ref=ref,
        struct2ref=struct2ref,
        fastdir=fastdir,
        firstdir=firstdir,
        cores=cores,
        ones=ones,
    )

    nb_pvs = subcort_pvs.pop("nonbrain")
    p = Projector(
        ref,
        struct2ref,
        hemispheres=hemispheres,
        roi_pvs=subcort_pvs,
        nonbrain_pvs=nb_pvs,
        cores=cores,
        ones=ones,
    )

    return p


def projector_print_pvs(*, projector=None):
    p = Projector.load(projector)
    return p.spc.make_nifti(p.pvs())
