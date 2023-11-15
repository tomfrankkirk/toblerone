import os.path as op
import sys

sys.path.insert(0, op.abspath(op.join(__file__, "../..")))

import os

import numpy as np
import regtricks as rt

import toblerone as tob
from toblerone import core, icosphere, utils

SPC = rt.ImageSpace.create_axis_aligned([-2, -2, -2], [10, 10, 10], [0.4, 0.4, 0.4])
ps, ts = icosphere.icosphere(nr_verts=1000)
LWS = tob.Surface.manual(1.5 * ps, ts, "LWS")
LPS = tob.Surface.manual(1.9 * ps, ts, "LPS")
LSS = tob.Surface.manual(ps, ts, "LSS")
SUBCORT = tob.Surface.manual(0.5 * ps, ts, "subcort")


def test_vox_tri():
    vox_tri = core.vox_tri_weights(LWS, LPS, SPC, 5, 1)
    assert np.all(vox_tri.sum(0) > 0)


def test_sph_projector():
    for s in [LWS, LPS, LSS, SUBCORT]:
        assert tob.utils.space_encloses_surface(SPC, s)

    hemi = tob.Hemisphere(LWS, LPS, LSS, "L")
    roi_pvs = {
        "L_subcort": tob.scripts.pvs_structure(
            ref=SPC, struct2ref=np.eye(4), surf=SUBCORT
        )
    }

    p = tob.Projector(SPC, np.eye(4), [hemi], roi_pvs=roi_pvs)
    assert np.all(
        p.surf2vol_matrix(True).sum(0) > 0
    ), "all surface vertices should map to a voxel"
    assert p.n_hemis == 1, "projector should only have one hemisphere"
    assert "L" in p.hemi_dict, "projector should contain L hemisphere"
    assert p["LPS"], "projector should expose dict access"
    assert p.n_nodes == SPC.n_vox + LSS.n_points + 1


def test_save_and_load_projector_hdf5():
    hemi = tob.Hemisphere(LWS, LPS, LSS, "R")
    proj = tob.Projector(SPC, np.eye(4), [hemi])
    proj.save("proj.h5")
    proj2 = tob.Projector.load("proj.h5")

    assert np.array_equiv(proj.pvs(), proj2.pvs()), "pvs were not preserved"
    assert np.array_equiv(proj.spc, proj2.spc), "image spaces were not preserved"
    assert np.array_equiv(
        proj._Projector__vox_tri_mats["R"].data,
        proj2._Projector__vox_tri_mats["R"].data,
    ), "voxel-triangle matrices were not preserved"
    assert np.array_equiv(
        proj._Projector__vtx_tri_mats["R"].data,
        proj2._Projector__vtx_tri_mats["R"].data,
    ), "vertex-triangle matrices were not preserved"

    os.remove("proj.h5")


def test_hemisphere_laplacian():
    hemi = tob.Hemisphere(LWS, LPS, LSS, "L")
    hemi2 = tob.Hemisphere(LWS, LPS, LSS, "R")
    proj = tob.Projector(SPC, np.eye(4), [hemi, hemi2])

    lap = proj.mesh_laplacian()

    assert utils.laplacian_is_valid(lap)
    assert (lap[np.diag_indices(lap.shape[0])] <= 0).all(), "positive diagonal"
    n = proj.hemi_dict["L"].n_points
    assert not utils.slice_sparse(
        lap, slice(0, n), slice(n, 2 * n)
    ).nnz, "projector laplacian not block diagnonal"
    assert not utils.slice_sparse(
        lap, slice(n, 2 * n), slice(0, n)
    ).nnz, "projector laplacian not block diagonal"


def test_adjacency():
    hemi = tob.Hemisphere(LWS, LPS, LSS, "L")
    adj = hemi.adjacency_matrix()
    assert not (adj.data < 0).any(), "negative value in adjacency matrix"


# def test_projector_rois():
#     td = get_testdir()
#     ins = op.join(td, 'in.surf.gii')
#     outs = op.join(td, 'out.surf.gii')
#     hemi = toblerone.Hemisphere(ins, outs, 'L')
#     spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
#     fracs = nibabel.load(op.join(td, 'sph_fractions.nii.gz')).get_fdata()
#     hemi.pvs = fracs.reshape(-1,3)
#     puta = Surface.manual(0.5 * hemi.inSurf.points,
#                             hemi.inSurf.tris, 'L_Puta')
#     rois = { 'L_Puta': puta }
#     proj = toblerone.projection.Projector(hemi, spc, rois=rois, cores=8)

#     ndata = np.ones(proj.n_nodes)
#     ndata[-1] = 2
#     vdata = proj.hybrid2vol(ndata, True)
#     assert np.allclose(vdata.max(), 2)

#     ndata = proj.vol2hybrid(vdata, True)
#     assert np.allclose(ndata.max(), 2)


# def test_projector_partial_fov():
#     # FIXME: this is a defunct test right now
#     td = get_testdir()
#     ins = op.join(td, 'in.surf.gii')
#     outs = op.join(td, 'out.surf.gii')
#     hemi = toblerone.Hemisphere(ins, outs, 'L')
#     spc = toblerone.ImageSpace(op.join(td, 'ref.nii.gz'))
#     spc = spc.resize([1,1,1], spc.size-2)
#     projector = toblerone.projection.Projector(hemi, spc, cores=8)
#     sdata = np.ones(hemi.inSurf.n_points, dtype=NP_FLOAT)
#     vdata = np.ones(spc.n_vox, dtype=NP_FLOAT)
#     # ndata = np.concatenate((sdata, vdata))
#     # proj = projector.surf2vol(sdata, True)

#     adj = projector.adjacency_matrix()


def test_projection():
    hemi = tob.Hemisphere(LWS, LPS, LSS, "L")
    projector = tob.Projector(SPC, np.eye(4), [hemi])

    sdata = np.ones(hemi.inSurf.n_points)
    vdata = np.ones(SPC.n_vox)
    ndata = np.concatenate((sdata, vdata))

    eps = 1e-6
    # volume to surface
    v2s = projector.vol2surf(vdata, False)
    v2s_edge = projector.vol2surf(vdata, True)
    assert (v2s <= v2s_edge + eps).all(), "edge correction did not increase signal"
    v2s = projector.vol2surf_matrix(False)
    v2s_edge = projector.vol2surf_matrix(True)
    assert (
        v2s_edge.data >= v2s.data
    ).all(), "egde correction: some weights should increase"

    # surface to volume
    s2v = projector.surf2vol(sdata, False)
    s2v_pv = projector.surf2vol(sdata, True)
    assert (s2v_pv <= s2v + eps).all(), "pv weighting did not reduce signal"
    s2v = projector.surf2vol_matrix(False)
    assert (s2v.sum(1).max() - 1) < eps, "total voxel weight > 1"
    s2v_pv = projector.surf2vol_matrix(False)
    assert (s2v_pv.data <= s2v.data).all(), "pv weighting should reduce voxel weights"

    # volume to node
    v2n = projector.vol2hybrid(vdata, False)
    v2n_edge = projector.vol2hybrid(vdata, True)
    assert (v2n <= v2n_edge + eps).all(), "edge correction did not increase signal"
    v2n = projector.vol2hybrid_matrix(False)
    assert (v2n.sum(1).max() - 1) < eps, "total node weight > 1"
    v2n_edge = projector.vol2hybrid_matrix(True)
    assert (
        v2n_edge.sum(1).max() - 1
    ) > eps, "edge correction: node should have weight > 1"

    # node to volume
    n2v = projector.hybrid2vol(ndata, False)
    n2v_pv = projector.hybrid2vol(ndata, True)
    assert (n2v_pv <= n2v + eps).all(), "pv weighting did not reduce signal"
    n2v = projector.hybrid2vol_matrix(False)
    assert (n2v.sum(1).max() - 1) < eps, "total voxel weight > 1"
    n2v_pv = projector.hybrid2vol_matrix(True)
    assert (n2v.sum(1).max() - 1) < eps, "total voxel weight > 1"


if __name__ == "__main__":
    test_projector_hdf5()
