import os.path as op
import sys

sys.path.insert(0, op.abspath(op.join(__file__, "../..")))

import os

import numpy as np
from scipy import sparse

import toblerone as tob
from toblerone import core, icosphere, utils
from toblerone.ctoblerone import _cyfilterTriangles


def test_rounding():
    a = np.array([[-1.5, -1.2, -0.8], [-0.5, 0, 0.5], [0.5, 1.1, 1.5]])
    out = utils.round_voxel_coordinates(a)
    print(out)


def test_surface_create():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    surf = tob.Surface.manual(ps, ts)
    assert np.allclose(ps, surf.points), "surface vertices are not equal"
    assert np.allclose(ts, surf.tris), "surface triangles are not equal"


# TODO create new spaces by subdivision of voxels and check same result holds
def test_spc_encloses_surface():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    surf = tob.Surface.manual(ps, ts)
    spc = tob.ImageSpace.create_axis_aligned([-1, -1, -1], [2, 2, 2], [1, 1, 1])
    assert utils.space_encloses_surface(
        spc, surf
    ), "space should exactly enclose surface"

    spc = spc.resize_voxels(1 / np.random.randint(1, 10, 3))
    assert utils.space_encloses_surface(
        spc, surf
    ), "space should exactly enclose surface"

    spc = tob.ImageSpace.create_axis_aligned([-2, -2, -2], [4, 4, 4], [1, 1, 1])
    assert utils.space_encloses_surface(spc, surf), "space should enclose surface"

    spc = tob.ImageSpace.create_axis_aligned([-0.5, -0.5, -0.5], [1, 1, 1], [1, 1, 1])
    assert not utils.space_encloses_surface(
        spc, surf
    ), "space should not enclose surface"


def test_minimal_enclosing_spc():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    surf = tob.Surface.manual(ps, ts)
    spc = tob.ImageSpace.create_axis_aligned([-1, -1, -1], [2, 2, 2], [1, 1, 1])
    assert utils.space_encloses_surface(
        spc, surf
    ), "space should exactly enclose surface"

    spc2 = tob.ImageSpace.minimal_enclosing([surf], spc)
    assert utils.space_encloses_surface(
        spc2, surf
    ), "enclosing space should enclose surface"

    spc2 = tob.ImageSpace.minimal_enclosing([surf, surf], spc)
    assert utils.space_encloses_surface(
        spc2, surf
    ), "minimal_enclosing() should accept a list of surfaces"

    assert spc == spc2, "minimal enclosing space should be same as original space"


def test_vox_tri_intersection():
    # Generate N triangles for which all vertices are contained
    # within a unit voxel at the origin
    N = int(1e6)
    ps = (np.random.rand(N, 3, 3) - 0.5).astype(utils.NP_FLOAT)
    ts = np.arange(3 * N).reshape(N, 3).astype(np.int32)

    # Multiply out the first two vertices of each triangle,
    # which will push most of them outside the voxel
    ps[:, 1:, :] *= np.random.randint(1, 10, (N, 2))[:, :, None]
    ps = ps.reshape(-1, 3)

    # Unit voxel, assert all intersect.
    cent = np.zeros(3, dtype=utils.NP_FLOAT)
    size = np.ones(3, dtype=utils.NP_FLOAT)
    flags = _cyfilterTriangles(ts, ps, cent, size)
    assert flags.all(), "Not all triangles intersect voxel"


def test_subvoxels():
    vox_size = np.ones(3)
    vox_cent = np.random.randint(-10, 10, 3)

    supr = np.random.randint(2, 3, 3)
    subvox_size = 1.0 / supr
    subvox_cents = core._get_subvoxel_grid(supr) + vox_cent
    assert subvox_cents.shape[0] == supr.prod()

    mean_cent = subvox_cents.mean(0)
    assert np.allclose(mean_cent, vox_cent), "subvoxels not evenly distributed"

    for cidx, scent in enumerate(subvox_cents):
        corners = scent + ((core.SUBVOXCORNERS) * (subvox_size[None, :]))
        assert core._filterPoints(
            corners, scent, subvox_size
        ).all(), "all subvoxel corners should be contained within subvoxel"
        assert core._filterPoints(
            corners, vox_cent, vox_size
        ).all(), "all subvoxel corners should be contained within voxel"


def test_write_read_surface_vtk():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    s = tob.Surface.manual(ps, ts)
    s.save("test.vtk")
    s2 = tob.Surface("test.vtk")
    assert np.allclose(s.points, s2.points)
    os.remove("test.vtk")


def test_indexing():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    surf = tob.Surface.manual(ps, ts)
    spc = tob.ImageSpace.create_axis_aligned([-1, -1, -1], [2, 2, 2], [1, 1, 1])
    surf.index_on(spc, 1)
    assert utils.space_encloses_surface(surf.indexed.space, surf)
    assert surf.indexed.space == spc, "indexed space should be same as original space"

    spc2 = spc.resize([-1, -1, -1], [4, 4, 4])
    surf2 = tob.Surface.manual(ps, ts)
    surf2.index_on(spc2, 1)
    assert (
        surf2.indexed.space == surf.indexed.space
    ), "indexed space should be same as original space"

    spc3 = spc.resize([1, 1, 1], [1, 1, 1])
    surf3 = tob.Surface.manual(ps, ts)
    surf3.index_on(spc3, 1)
    assert not utils.space_encloses_surface(spc3, surf3)
    assert (
        surf2.indexed.space == surf.indexed.space
    ), "indexed space should be same as original space"


def test_sparse_normalise():
    mat = sparse.random(5000, 5000, 0.1)
    thr = 1e-12
    for axis in range(2):
        normed = utils.sparse_normalise(mat, axis, thr)
        sums = normed.sum(axis).A.flatten()
        assert (np.abs(sums[sums > 0] - 1) <= thr).all()


def test_hemi_init():
    ps, ts = icosphere.icosphere(nr_verts=1000)
    ins = tob.Surface.manual(ps, ts)
    sph = tob.Surface.manual(ps, ts)
    outs = tob.Surface.manual(ps * 2, ts)
    hemi = tob.Hemisphere(ins, outs, sph, "L")
    hemi2 = tob.Hemisphere(ins, outs, sph, "L")
    assert id(hemi.inSurf) != id(hemi2.inSurf)
    assert np.allclose(hemi.outSurf.tris, hemi.outSurf.tris)


if __name__ == "__main__":
    test_spc_encloses_surface()
