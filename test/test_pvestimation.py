"""Toblerone tests"""

import os.path as op
import sys

sys.path.insert(0, op.abspath(op.join(__file__, "../..")))

import copy
import tempfile

import numpy as np
from regtricks.application_helpers import sum_array_blocks

import toblerone as tob
from toblerone import core, icosphere, utils
from toblerone.classes.image_space import reindexing_filter


def get_testdir():
    return op.dirname(__file__)


def test_voxelise():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    surf = tob.Surface.manual(ps, ts)
    spc = tob.ImageSpace.create_axis_aligned([-1, -1, -1], [2, 2, 2], [1, 1, 1])

    surf.index_on(spc, 1)
    vox = surf.voxelise(spc, 1)
    assert vox.all(), "all voxels centres lie within sphere"

    surf2 = tob.Surface.manual(ps, ts)
    spc2 = spc.resize_voxels(0.05)
    surf2.index_on(spc2, 1)
    vox2 = surf2.voxelise(spc2, 1)
    vol = 8 * vox2.sum() / vox2.size
    assert np.allclose(
        vol, 4 * np.pi / 3, rtol=0.05
    ), "voxelised volume not equal to ground truth"


def test_prefer_convex_hull():
    """This was a sneaky bug. When estimating PVs for a voxel that intersects the surface,
    we should aim to pick the side that is more convex, so the convex hull volume will
    be more accurate. The below test case of a very small sphere that is contained within
    a grid of 2,2,2 voxels is nasty because there are float rounding issues. On one side,
    a fold in the surface is detected and recursion is used. On the other side, no fold
    is detected (due to float equality issues) and therefore a hull is used. It's important
    that the two strategies yield similar results."""

    fov = 4
    vox_size = np.ones(3)
    spc = tob.ImageSpace.create_axis_aligned(
        [0, 0, 0], (fov / vox_size).astype(int), vox_size
    )

    ps, ts = icosphere.icosphere(nr_verts=1000)
    ps += fov / 2
    surf = tob.Surface.manual(ps, ts)

    # ground truth
    factor = 10
    spc2 = spc.resize_voxels(1 / factor)
    surf2 = copy.deepcopy(surf)
    surf2.index_on(spc2)
    surf2.indexed.voxelised = surf2.voxelise(spc2, 1)
    truth = np.zeros(spc2.size.prod())
    src_fltr, dest_fltr = reindexing_filter(surf2.indexed.space, spc2)
    truth[dest_fltr] = surf2.indexed.voxelised
    truth = sum_array_blocks(truth.reshape(spc2.size), 3 * [factor]) / (factor**3)

    pvs = tob.scripts.pvs_structure(
        ref=spc, struct2ref=np.eye(4), surf=surf, supr=5, cores=1
    )

    assert np.allclose(pvs, truth, rtol=0.05), "PVs for small sphere do not match truth"


def test_cortex():
    spc = tob.ImageSpace.create_axis_aligned([-2, -2, -2], [4, 4, 4], [1, 1, 1])
    ps, ts = icosphere.icosphere(nr_verts=1000)
    sph = tob.Surface.manual(ps, ts)
    ins = tob.Surface.manual(1 * ps, ts)
    outs = tob.Surface.manual(2 * ps, ts)

    s2r = np.identity(4)
    supr = np.random.randint(3, 6, 3)
    fracs = tob.scripts.pvs_cortex_freesurfer(
        LWS=ins, LPS=outs, LSS=sph, ref=spc, struct2ref=s2r, supr=supr
    )

    wm = fracs[..., 1].sum()
    gm = fracs[..., 0].sum()

    wm_true = 4 * np.pi * (1**3) / 3
    gm_true = (4 * np.pi * (2**3) / 3) - wm_true

    assert np.allclose(wm, wm_true, rtol=0.05)
    assert np.allclose(gm, gm_true, rtol=0.05)

    # Symmetry of results in central voxels
    wm = fracs[..., 1][fracs[..., 1] > 0]
    assert np.allclose(wm[0], wm, rtol=0.01)

    gm = fracs[1:3, 1:3, 1:3, 0]
    assert np.allclose(gm[0, 0, 0], gm, rtol=0.01)

    csf = fracs[1:3, 1:3, 1:3, 2]
    assert np.allclose(csf[0, 0, 0], 0, rtol=0.01)


def test_structure():
    spc = tob.ImageSpace.create_axis_aligned([-1, -1, -1], [2, 2, 2], [1, 1, 1])
    ps, ts = icosphere.icosphere(nr_verts=1000)
    sph = tob.Surface.manual(ps, ts)
    s2r = np.identity(4)

    fracs = tob.scripts.pvs_structure(surf=sph, ref=spc, struct2ref=s2r, cores=1)
    true = 4 * np.pi / 3
    assert np.allclose(fracs.sum(), true, rtol=0.01), "pvs do not sum to ground truth"
    assert np.allclose(
        fracs[0, 0, 0], fracs, rtol=0.01
    ), "pvs not symmetrically distributed across sphere"


def test_matrix_masking():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    ins = tob.Surface.manual(1.5 * ps, ts)
    outs = tob.Surface.manual(1.9 * ps, ts)
    sph = tob.Surface.manual(ps, ts)
    subcort = tob.Surface.manual(0.5 * ps, ts)

    spc = tob.ImageSpace.create_axis_aligned([-2, -2, -2], [8, 8, 8], [0.5, 0.5, 0.5])

    hemi = tob.Hemisphere(ins, outs, sph, "L")
    roi_pvs = {
        "L_subcort": tob.scripts.pvs_structure(
            ref=spc, struct2ref=np.eye(4), surf=subcort, cores=1
        )
    }
    proj = tob.Projector(spc, np.eye(4), hemi, roi_pvs)

    smask = proj.cortex_thickness() > 0.1
    wmmask = (proj.pvs()[..., 1] > 0.1).flatten()
    rmask = np.ones(proj.n_rois, dtype=bool)
    nmask = np.concatenate((smask, wmmask, rmask))
    vmask = proj.hybrid2vol(nmask, edge_scale=False).astype(bool)

    vones = np.ones(vmask.sum())
    sones = np.ones(smask.sum())
    nones = np.ones(nmask.sum())
    eps = 1e-6

    m1 = proj.vol2surf_matrix(False, vol_mask=vmask, surf_mask=smask)
    m2 = utils.slice_sparse(proj.vol2surf_matrix(False), smask, vmask)
    x = m1 @ vones
    y = m2 @ vones
    assert (x[x > 0] >= y[x > 0]).all()
    assert np.allclose(x[x > 0], 1)

    m1 = proj.vol2surf_matrix(True, vol_mask=vmask, surf_mask=smask)
    m2 = utils.slice_sparse(proj.vol2surf_matrix(True), smask, vmask)
    x = m1 @ vones
    y = m2 @ vones
    assert (x[x > 0] >= y[x > 0]).all()
    assert (x[x > 0] >= 1 - eps).all()

    m1 = proj.surf2vol_matrix(False, surf_mask=smask, vol_mask=vmask)
    m2 = utils.slice_sparse(proj.surf2vol_matrix(False), vmask, smask)
    x = m1 @ sones
    y = m2 @ sones
    assert (x[x > 0] >= y[y > 0]).all()
    assert np.allclose(x[x > 0], 1)

    m1 = proj.surf2vol_matrix(True, surf_mask=smask, vol_mask=vmask)
    m2 = utils.slice_sparse(proj.surf2vol_matrix(True), vmask, smask)
    x = m1 @ sones
    y = m2 @ sones
    assert (x[x > 0] >= y[y > 0]).all()
    assert (x[x > 0] <= 1 + eps).all()

    m1 = proj.vol2hybrid_matrix(False, vol_mask=vmask, node_mask=nmask)
    m2 = utils.slice_sparse(proj.vol2hybrid_matrix(False), nmask, vmask)
    x = m1 @ vones
    y = m2 @ vones
    assert (x[x > 0] >= y[x > 0]).all()
    assert np.allclose(x[x > 0], 1)

    m1 = proj.vol2hybrid_matrix(True, vol_mask=vmask, node_mask=nmask)
    m2 = utils.slice_sparse(proj.vol2hybrid_matrix(True), nmask, vmask)
    x = m1 @ vones
    y = m2 @ vones
    assert (x[x > 0] >= x[x > 0]).all()
    assert (x[x > 0] >= 1 - eps).all()

    m1 = proj.hybrid2vol_matrix(False, node_mask=nmask)
    m2 = proj.hybrid2vol_matrix(False)[:, nmask]
    x = m1 @ nones
    y = m2 @ nones
    assert (x[x > 0] >= y[y > 0]).all()
    assert np.allclose(x[x > 0], 1)

    m1 = proj.hybrid2vol_matrix(True, node_mask=nmask)
    m2 = proj.hybrid2vol_matrix(True)[:, nmask]
    x = m1 @ nones
    y = m2 @ nones
    assert (x[x > 0] >= y[y > 0]).all()
    assert (x[x > 0] <= 1 + eps).all()


def test_projector_brain():
    td = op.join(get_testdir(), "testdata/brain")

    spc = tob.ImageSpace(op.join(td, "T1_fast_pve_0.nii.gz"))
    spc = spc.resize_voxels(3)

    def save_and_load_projector(proj):
        with tempfile.TemporaryDirectory() as d:
            fname = op.join(d, "proj.h5")
            p.save(fname)
            tob.Projector.load(fname)

    def test_projector_functions(proj):
        # Test PV calculations
        cpvs = proj.cortex_pvs()
        assert np.allclose(cpvs.sum(-1), 1)
        spvs = proj.subcortex_pvs()
        pvs = proj.pvs()
        assert np.allclose(cpvs.sum(-1), 1)

        # Surf to vol
        eps = 1e-5
        x = np.ones(proj.n_surf_nodes)
        y_noedge = proj.surf2vol(x, False)
        y_edge = proj.surf2vol(x, True)
        fltr = y_noedge > 0
        assert np.alltrue(y_noedge >= 0)
        assert np.allclose(y_noedge[fltr], 1)
        assert np.alltrue(y_edge <= y_noedge + eps)

        # Vol to surf
        x = np.ones(proj.n_vol_nodes)
        y_noedge = proj.vol2surf(x, False)
        y_edge = proj.vol2surf(x, True)
        fltr = y_noedge > 0
        assert np.alltrue(y_noedge >= 0)
        assert np.allclose(y_noedge[fltr], 1)
        assert np.alltrue(y_edge + eps >= y_noedge)

        # Hybrid to vol
        x = np.concatenate(
            [
                60 * np.ones(proj.n_surf_nodes),
                20 * np.ones(proj.n_vox),
                60 * np.ones(proj.n_rois),
            ]
        )
        y_noedge = proj.hybrid2vol(x, False)
        y_edge = proj.hybrid2vol(x, True)
        fltr = y_noedge > 0
        assert np.alltrue(y_noedge >= 0)
        assert np.allclose(y_noedge[fltr].min(), 20)
        assert np.allclose(y_noedge[fltr].max(), 60)
        assert np.alltrue(y_edge <= y_noedge + eps)

        # Vol to hybrid
        x = 60 * pvs[..., 0] + 20 * pvs[..., 1]
        y_noedge = proj.vol2hybrid(x.flatten(), False)
        y_edge = proj.vol2hybrid(x.flatten(), True)
        fltr = y_edge > 0
        assert np.alltrue(y_noedge >= 0)
        assert np.alltrue(y_edge + eps >= y_noedge)

    LWS = op.join(td, "10k/lh_white10k.surf.gii")
    LPS = op.join(td, "10k/lh_pial10k.surf.gii")
    LSS = op.join(td, "10k/sph_10k.surf.gii")
    RWS = op.join(td, "10k/rh_white10k.surf.gii")
    RPS = op.join(td, "10k/rh_pial10k.surf.gii")
    LHEMI = tob.Hemisphere(LWS, LPS, LSS, "L")
    RHEMI = tob.Hemisphere(RWS, RPS, LSS, "R")

    for hemis in [[LHEMI], [RHEMI], [LHEMI, RHEMI]]:
        # Test with surfaces and ventricular CSF
        p = tob.Projector(spc, np.eye(4), hemis)
        save_and_load_projector(p)
        test_projector_functions(p)


def test_pvs_subcortex_freesurfer():
    td = get_testdir()
    td = op.join(td, "testdata/brain")

    spc = tob.ImageSpace(op.join(td, "T1_fast_pve_0.nii.gz"))
    spc = spc.resize_voxels(3)
    subcort_pvs = tob.scripts.pvs_subcortex_freesurfer(
        ref=spc, struct2ref=np.eye(4), fsdir="/Users/thomaskirk/Data/singlePLDpcASL/fs"
    )


def test_pvs_subcortex_fsl():
    td = get_testdir()
    td = op.join(td, "testdata/brain")
    spc = tob.ImageSpace(op.join(td, "T1_fast_pve_0.nii.gz"))
    spc = spc.resize_voxels(3)

    fastdir = td
    firstdir = op.join(td, "first_results")
    s2r = np.eye(4)

    subcort_pvs = tob.scripts.pvs_subcortex_fsl(
        ref=spc, struct2ref=s2r, firstdir=firstdir, fastdir=fastdir
    )


if __name__ == "__main__":
    test_projector_brain()
