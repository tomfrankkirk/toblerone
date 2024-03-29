"""
Surface-related classes
"""

import copy
import functools
import itertools
import multiprocessing as mp
import os.path as op
import warnings
from types import SimpleNamespace

import nibabel
import numpy as np
import pyvista
import regtricks as rt
import vtk
from scipy import sparse
from trimesh import Trimesh, proximity

from .. import core, icosphere, utils
from ..utils import NP_FLOAT
from .image_space import BaseSpace, ImageSpace, reindexing_filter

MP_THRESHOLD = 1000


@utils.cascade_attributes
def ensure_derived_space(func):
    """
    Decorator for Surface functions that require ImageSpace arguments.
    Internally, Surface objecs store information indexed to a minimal
    enclosing voxel grid (referred to as the self.index_grid) based on
    some arbitrary ImageSpace. When interacting with other ImageSpaces,
    this function ensures that two grids are compatible with each other.
    """

    def ensured(self, *args):
        if (not args) or (not isinstance(args[0], BaseSpace)):
            raise RuntimeError("Function must be called with ImageSpace argument first")
        if self.indexed is None:
            raise RuntimeError(
                "Surface must be indexed prior to using this function"
                + "Call surface.index_on()"
            )
        if not self.indexed.space.derives_from(args[0]):
            raise RuntimeError(
                "Target space is not derived from surface's current index space."
                + "Call surface.index_on with the target space first"
            )
        return func(self, *args)

    return ensured


@utils.cascade_attributes
def assert_indexed(func):
    """
    Decorator to ensure surface has been indexed prior to calling function.
    """

    def asserted(self, *args, **kwargs):
        if self.indexed is None:
            raise RuntimeError("Surface must be indexed before calling ", func.__name__)
        return func(self, *args, **kwargs)

    return asserted


class Surface(object):
    """
    Encapsulates a surface's points, triangles and associations data.
    Create either by passing a file path (as below) or use the static class
    method Surface.manual() to directly pass points and triangles.

    Args:
        path (str): path to file (GIFTI/FS binary/pyvista compatible)
        name (str): optional, for progress bars
    """

    def __init__(self, path, name=None, shift_cras=True):
        if not op.exists(path):
            raise RuntimeError("File {} does not exist".format(path))

        # Can we get an extension please, makes life easier
        surfExt = op.splitext(path)[-1]

        # GIFTI via nibabel
        if surfExt == ".gii":
            try:
                gft = nibabel.load(path).darrays
                ps, ts = gft[0].data, gft[1].data

            except Exception as e:
                print(
                    f"""Could not load {path} as .gii. Is it a surface
                    GIFTI (.surf.gii)?"""
                )
                raise e

        # VTK via vtk (ie python vtk library)
        elif surfExt == ".vtk":
            try:
                reader = vtk.vtkGenericDataObjectReader()
                reader.SetFileName(path)
                reader.Update()

                ps = np.array(reader.GetOutput().GetPoints().GetData())
                ts = np.array(reader.GetOutput().GetPolys().GetData())

                # tris array is returned as a single vector eg
                # [3 a b c 3 a b c] where 3 represents triangle faces
                # so it actually has FOUR columns, the first of which
                # is all 3...
                if ts.size % 4:
                    raise ValueError(
                        f"VTK file does not appear to be triangle data (first poly has {ts[0]} faces"
                    )
                ts = ts.reshape(-1, 4)
                if (ts[:, 0] != 3).any():
                    raise ValueError(
                        f"VTK file does not appear to be triangle data (first poly has {ts[0,0]} faces"
                    )
                ts = ts[:, 1:]

            except Exception as e:
                print(
                    f"""Could not load {path} as .vtk. Is it a triangle
                    VTK?"""
                )
                raise e

        else:
            # FS files don't have a proper extension (binary)
            # FreeSurfer via nibabel
            try:
                ps, ts, meta = nibabel.freesurfer.io.read_geometry(
                    path, read_metadata=True
                )
                if shift_cras:
                    ps += meta["cras"]

            # Maybe FreeSurfer didn't work, try anything else via pyvista
            except Exception as e:
                try:
                    poly = pyvista.read(path)
                    ps = np.array(poly.points)
                    ts = poly.faces.reshape(-1, 4)[:, 1:]

                except Exception as e:
                    print("Could not load surface via pyvista")
                    raise e

        if ps.shape[1] != 3:
            raise RuntimeError("Points matrices should be p x 3")

        if ts.shape[1] != 3:
            raise RuntimeError("Triangles matrices should be t x 3")

        if (np.max(ts) != ps.shape[0] - 1) or (np.min(ts) != 0):
            raise RuntimeError("Incorrect points/triangle indexing")

        # TODO could check closed and winding consistent here?

        self.points = ps.astype(NP_FLOAT)
        self.tris = ts.astype(np.int32)
        self.indexed = None
        self.name = name
        self._use_mp = self.tris.shape[0] > MP_THRESHOLD

    def __repr__(self):
        from textwrap import dedent

        return dedent(
            f"""\
            Surface with {self.n_points} points and {self.tris.shape[0]} triangles. 
            min (X,Y,Z):  {self.points.min(0)}
            mean (X,Y,Z): {self.points.mean(0)}
            max (X,Y,Z):  {self.points.max(0)}
            """
        )

    @classmethod
    def manual(cls, ps, ts, name="<manually created surface>"):
        """Manual surface constructor using points and triangles arrays"""

        if (ps.shape[1] != 3) or (ts.shape[1] != 3):
            raise RuntimeError("ps, ts arrays must have N x 3 dimensions")

        if ts.min() > 0:
            raise RuntimeError("ts array should be 0-indexed")

        s = cls.__new__(cls)
        s.points = copy.deepcopy(ps.astype(NP_FLOAT))
        s.tris = copy.deepcopy(ts.astype(np.int32))
        s.indexed = None
        s.name = name
        s._use_mp = s.tris.shape[0] > MP_THRESHOLD
        return s

    @property
    def n_points(self):
        return self.points.shape[0]

    def save_metric(self, data, path):
        """
        Save vertex-wise data as a .func.gii at path
        """

        if not self.n_points == data.shape[0]:
            raise RuntimeError("Incorrect data shape")

        if not path.endswith(".func.gii"):
            print("appending .func.gii extension")
            path += ".func.gii"

        gii = nibabel.GiftiImage()
        gii.add_gifti_data_array(nibabel.gifti.GiftiDataArray(data.astype(NP_FLOAT)))
        nibabel.save(gii, path)

    def save(self, path):
        """
        Save surface as .surf.gii (default), .white/.pial at path.
        """

        if path.endswith(".vtk"):
            # Faces must be an array of polygons with 3 in first
            # columns to denote triangle data
            faces = 3 * np.ones((self.tris.shape[0], 4), dtype=int)
            faces[:, 1:] = self.tris
            m = pyvista.PolyData(self.points, faces)
            m.save(path)

        elif path.count(".gii"):
            if not path.endswith(".surf.gii"):
                if path.endswith(".gii"):
                    path.replace(".gii", ".surf.gii")
                else:
                    path += ".surf.gii"

            if self.name is None:
                self.name = "Other"

            m0 = {"GeometricType": "Anatomical"}

            if self.name in ["LWS", "LPS", "RWS", "RPS"]:
                cortexdict = {
                    sd
                    + sf
                    + "S": {
                        "AnatomicalStructurePrimary": "CortexLeft"
                        if sd == "L"
                        else "CortexRight",
                        "AnatomicalStructureSecondary": "GrayWhite"
                        if sd == "W"
                        else "Pial",
                    }
                    for (sd, sf) in itertools.product(["L", "R"], ["P", "W"])
                }
                m0.update(cortexdict[self.name])

            else:
                m0.update({"AnatomicalStructurePrimary": self.name})

            # Points matrix
            # 1 corresponds to NIFTI_XFORM_SCANNER_ANAT
            ps = nibabel.gifti.GiftiDataArray(
                self.points,
                intent="NIFTI_INTENT_POINTSET",
                coordsys=nibabel.gifti.GiftiCoordSystem(1, 1),
                datatype="NIFTI_TYPE_FLOAT32",
                meta=nibabel.gifti.GiftiMetaData.from_dict(m0),
            )

            # Triangles matrix
            m1 = {"TopologicalType": "Closed"}
            ts = nibabel.gifti.GiftiDataArray(
                self.tris,
                intent="NIFTI_INTENT_TRIANGLE",
                coordsys=nibabel.gifti.GiftiCoordSystem(0, 0),
                datatype="NIFTI_TYPE_INT32",
                meta=nibabel.gifti.GiftiMetaData.from_dict(m1),
            )

            img = nibabel.gifti.GiftiImage(darrays=[ps, ts])
            nibabel.save(img, path)

        else:
            if not (path.endswith(".white") or path.endswith(".pial")):
                warnings.warn("Saving as FreeSurfer binary")
            nibabel.freesurfer.write_geometry(path, self.points, self.tris)

    @ensure_derived_space
    def output_pvs(self, space):
        """
        Express PVs in the voxel grid of space. Space must derive from that
        which the surface was indexed with (see ImageSpace.derives_from()).
        """

        pvs_curr = self.indexed.voxelised.astype(NP_FLOAT)
        pvs_curr[self.indexed.assocs_keys] = self.fractions
        out = np.zeros(np.prod(space.size), dtype=NP_FLOAT)
        curr_inds, dest_inds = reindexing_filter(self.indexed.space, space)
        out[dest_inds] = pvs_curr[curr_inds]
        return out.reshape(space.size)

    @assert_indexed
    def _estimate_fractions(self, supr, cores, ones, desc=""):
        """
        Estimate interior/exterior fractions within current index_space.

        Args:
            supr: list/tuple of 3 ints, subdivision factor in each dim
            cores: number of multiprocessing cores to use
            ones: debug tool, write ones in all voxels within assocs_keys
            desc: for use with progress bar
        """

        cores = cores if self._use_mp else 1
        if ones:
            self.fractions = np.ones(self.indexed.assocs_keys.size, dtype=NP_FLOAT)
        else:
            self.fractions = core._estimateFractions(self, supr, desc, cores)

    def points_within_space(self, spc):
        if not isinstance(spc, ImageSpace):
            spc = ImageSpace(spc)
        ps_vox = self.transform(spc.world2vox).points
        ps_vox = ps_vox.round()
        return np.logical_and((ps_vox > 0).all(-1), (ps_vox <= spc.size - 1).all(-1))

    def index_on(self, space, cores=mp.cpu_count()):
        """
        Index a surface to an ImageSpace. This is a pre-processing step
        for PV estimation and produces a set of state variables that
        are specific to the space that the surface has been indexed on.
        Accordingly, these are all stored on the surface.indexed
        attribute to keep them clear of the original surface attributes.

        Args:
            space (ImageSpace): voxel grid to index against.
            cores (int): CPU cores for multiprocessing.

        Returns:
            None, but the attribute ``self.indexed`` is set (see notes)

        Notes:
            ``self.indexed.points_vox``: vertices converted into voxel coordinates for the space\n
            ``self.indexed.assocs``: vertex/voxel associations as sparse CSR bool matrix of size (voxs, tris)\n
            ``self.indexed.assocs_keys``: flat voxel indices that intersect surface\n
            ``self.indexed.xprods``: surface triangle cross products in voxel coordinates\n
            ``self.indexed.space``: the minimal enclosing ``ImageSpace`` used for indexing associations\n
                (NB this is not necessarily the same as the input space)
        """

        # Smallest possible ImageSpace, based on space, that fully encloses surf.
        # This is necessary to deal with partial coverage (ie, the voxel grid
        # does not enclose the surface, which in turn causes problems with
        # voxelisation). As such, the indexing space may not be the same as the
        # space passed in by the caller, but it will "derive from" that space.
        # Other functions that relate to ImageSpaces use the @ensure_derived_space
        # decorator to check this holds.
        encl_space = ImageSpace.minimal_enclosing(self, space)
        assert utils.space_encloses_surface(encl_space, self)

        # Calculate associations and cross products
        cores = cores if self._use_mp else 1
        points_vox = utils.affine_transform(self.points, encl_space.world2vox)
        points_vox = points_vox.round(5)
        assocs, assocs_keys = core.form_associations(
            points_vox, self.tris, encl_space, cores
        )
        if not assocs_keys.size:
            warnings.warn(
                f"Surface {self.name} does not intersect the reference voxel grid"
            )
        xprods = utils.calculateXprods(points_vox, self.tris)

        # All the results of indexing are stored using the namedtuple
        # structure under the attribue self.indexed
        self.indexed = SimpleNamespace(
            points_vox=points_vox,
            space=encl_space,
            assocs=assocs,
            assocs_keys=assocs_keys,
            xprods=xprods,
            minimal_enclosing_space=encl_space,
        )

    @ensure_derived_space
    def voxelise(self, space, cores=mp.cpu_count()):
        """
        Voxelise a surface within an ImageSpace. This requires the surface to
        have been indexed on the same ImageSpace first.

        Args:
            space (ImageSpace): voxel grid to test against surface
            cores (int): CPU cores to use

        Returns
            (bool): flat array of voxels contained within surface
        """

        if self.indexed is None:
            raise RuntimeError("Surface must be indexed before calling this function")

        # We will project rays along the longest dimension and split the
        # grid along the first of the other two.
        size = self.indexed.space.size
        dim = np.argmax(size)
        other_dims = list({0, 1, 2} - {dim})

        # Get the voxel IJKs for every voxel intersecting the surface
        # Group these according to their ray number, and then strip out
        # repeats to get the minimal set of rays that need to be projected
        # For voxel IJK, and projection along dimension Y, the ray number
        # is given by I,K within a grid of size XZ. Ray D1D2 is an Nx2 array
        # holding the unique ray coords in dimensions XZ.
        allIJKs = np.array(np.unravel_index(self.indexed.assocs_keys, size)).T
        ray_numbers = np.ravel_multi_index(allIJKs[:, other_dims].T, size[other_dims])
        _, uniq_rays = np.unique(ray_numbers, return_index=True)
        rayD1D2 = allIJKs[uniq_rays, :]
        rayD1D2 = rayD1D2[:, other_dims]

        worker = functools.partial(core._voxelise_worker, self)
        cores = cores if self._use_mp else 1
        if cores > 1:
            # Share out the rays and subset of the overall dimension range
            # amongst pool workers
            sub_ranges = utils._distributeObjects(range(size[other_dims[0]]), cores)
            subD1D2s = [
                rayD1D2[(rayD1D2[:, 0] >= sub.start) & (rayD1D2[:, 0] < sub.stop), :]
                for sub in sub_ranges
            ]

            # Check tasks were shared out completely, zip them together for pool
            assert sum([s.shape[0] for s in subD1D2s]) == rayD1D2.shape[0]
            assert sum([len(s) for s in sub_ranges]) == size[other_dims[0]]
            worker_args = zip(sub_ranges, subD1D2s)

            with mp.Pool(cores) as p:
                submasks = p.starmap(worker, worker_args)
            src_mask = np.concatenate(submasks, axis=other_dims[0])

        else:
            src_mask = worker(range(size[other_dims[0]]), rayD1D2)

        if not src_mask.any():
            warnings.warn(f"Voxelisation of {self.name}: no voxels filled")
        src_mask = src_mask.flatten()

        # If the space provided is not the same as the space used for indexing
        # (because we reduced the FoV when indexing), we need to map results
        # from the indexing space into the destination space. See index_on()
        # for more info.
        # Else, we can just return the mask as-is.
        if space != self.indexed.space:
            src_inds, dest_inds = reindexing_filter(self.indexed.space, space, False)
            mask = np.zeros(space.size.prod(), dtype=bool)
            mask[dest_inds] = src_mask[src_inds]
        else:
            mask = src_mask

        return src_mask

    @ensure_derived_space
    def reindex_LUT(self, space):
        """Return a copy of LUT indices expressed in another space"""

        src_inds, dest_inds = reindexing_filter(self.indexed.space, space)
        fltr = np.in1d(src_inds, self.indexed.assocs_keys, assume_unique=True)
        return dest_inds[fltr]

    @ensure_derived_space
    def find_bridges(self, space):
        """
        Find voxels within space that are intersected by this surface
        multiple times

        Args:
            space: ImageSpace, or path to image, in which to find bridge voxels
                NB the surface must have been indexed in this space already
                (see Surface.index_on)

        Returns:
            array of linear voxel indices
        """

        group_counts = np.array(
            [
                len(
                    core._separatePointClouds(
                        self.tris[self.indexed.assocs[v, :].indices, :]
                    )
                )
                for v in self.indexed.assocs_keys
            ]
        )
        bridges = self.indexed.assocs_keys[group_counts > 1]

        if space == self.indexed.space:
            return bridges

        else:
            src_inds, dest_inds = reindexing_filter(self.indexed.space, space)
            fltr = np.in1d(src_inds, bridges, assume_unique=True)
            return dest_inds[fltr]

    def transform(self, transform):
        """Apply affine transformation to surface vertices, return new Surface"""

        if isinstance(transform, rt.Registration):
            transform = transform.src2ref
        points = utils.affine_transform(self.points, transform).astype(NP_FLOAT)
        return Surface.manual(points, self.tris, self.name)

    def to_patch(self, vox_idx):
        """
        Return a patch object specific to a voxel given by linear index.
        Look up the triangles intersecting the voxel, and then load and rebase
        the points / surface normals as required.
        """

        tri_nums = self.indexed.assocs[vox_idx, :].indices
        (ps, ts) = utils.rebase_triangles(self.indexed.points_vox, self.tris, tri_nums)
        return Patch(ps, ts, self.indexed.xprods[tri_nums, :])

    def to_patches(self, vox_inds):
        """
        Return the patches for the voxels in voxel indices, flattened into
        a single set of ps, ts and xprods.

        If no patches exist for this list of voxels return None.
        """

        tri_nums = self.indexed.assocs[vox_inds, :].indices

        if tri_nums.size:
            tri_nums = np.unique(tri_nums)
            return Patch(
                self.indexed.points_vox,
                self.tris[tri_nums, :],
                self.indexed.xprods[tri_nums, :],
            )

        else:
            return None

    def to_polydata(self):
        """Return pyvista polydata object for this surface"""

        tris = 3 * np.ones((self.tris.shape[0], self.tris.shape[1] + 1), np.int32)
        tris[:, 1:] = self.tris
        return pyvista.PolyData(self.points, tris)

    def adjacency_matrix(self):
        """
        Adjacency matrix for the points of this surface, as a scipy sparse
        matrix of size P x P, with 1 denoting a shared edge between points.

        """

        edge_pairs = np.array(
            [
                np.concatenate((self.tris[:, 0], self.tris[:, 1], self.tris[:, 2])),
                np.concatenate((self.tris[:, 1], self.tris[:, 2], self.tris[:, 0])),
            ]
        )
        row, col = edge_pairs
        weights = np.ones(row.size, dtype=np.int32)
        adj = sparse.coo_matrix(
            (weights, (row, col)), shape=(self.n_points, self.n_points)
        )

        return adj.tocsr()

    def mesh_laplacian(self):
        """
        Mesh Laplacian operator for this surface, as a scipy sparse matrix
        of size n_points x n_points. Elements on the diagonal are negative
        and off-diagonal elements are positive. All neighbours are weighted
        with value 1 (ie, equal weighting ignoring distance).

        Returns:
            sparse CSR matrix
        """

        # The diagonal is the negative sum of other elements
        adj = self.adjacency_matrix()
        adj = np.around(
            adj, 9
        )  # FIXME: this is less than ideal - would be better to round to s.f.
        dia = adj.sum(1).A.flatten()
        laplacian = sparse.dia_matrix((dia, 0), shape=(adj.shape), dtype=np.float32)
        laplacian = adj - laplacian

        assert utils.laplacian_is_valid(laplacian)
        return laplacian

    def edges(self):
        """
        Edge matrix, sized as follows (tris, 3, 3), where the second dimension
        contains the edges defined as (v1 - v0), (v2 - v0), (v2 - v1), and
        the final dimension contains the edge components in XYZ.
        """
        edge_defns = [list(e) for e in core.TRI_EDGE_INDEXING]
        edges = np.stack(
            [
                self.points[self.tris[:, e[0]], :] - self.points[self.tris[:, e[1]], :]
                for e in edge_defns
            ],
            axis=1,
        )

        return edges


class Patch(Surface):
    """
    Subclass of Surface that represents a small patch of surface.
    Points, triangles and xprods are all inherited from the parent surface.
    This class should not be directly created but instead instantiated via
    the Surface.to_patch() / to_patches() methods.
    """

    def __init__(self, points, tris, xprods):
        self.points = points
        self.tris = tris
        self.xprods = xprods

    def shrink(self, fltr):
        """
        Return a shrunk copy of the patch by applying the logical
        filter fltr to the calling objects tris and xprods matrices
        """

        return Patch(self.points, self.tris[fltr, :], self.xprods[fltr, :])


class Hemisphere(object):
    """
    The white and pial surfaces of a hemisphere, and a repository to
    store data when calculating tissue PVs from the fractions of each
    surface

    Args:
        inpath: path to white surface
        outpath: path to pial surface
        side: 'L' or 'R'
    """

    def __init__(self, insurf, outsurf, sphere, side):
        if not side in ["L", "R"]:
            raise ValueError("Side must be either 'L' or 'R'")
        self.side = side

        # Create surfaces from path or make our own copy
        if not isinstance(insurf, Surface):
            self.inSurf = Surface(insurf, name=side + "WS")
        else:
            self.inSurf = copy.deepcopy(insurf)
        if not isinstance(outsurf, Surface):
            self.outSurf = Surface(outsurf, name=side + "PS")
        else:
            self.outSurf = copy.deepcopy(outsurf)

        if not isinstance(sphere, Surface):
            self.sphere = Surface(sphere, name=side + "SS", shift_cras=False)
            utils.check_spherical(self.sphere.points)
        else:
            self.sphere = copy.deepcopy(sphere)

        if (self.inSurf.tris != self.outSurf.tris).any() or (
            self.inSurf.tris != self.sphere.tris
        ).any():
            raise ValueError("All surfaces must have same triangles array")

        self.PVs = None
        return

    @property
    def surfs(self):
        """Iterator over the inner/outer surfaces, not including sphere"""

        return [self.inSurf, self.outSurf]

    @property
    def surf_dict(self):
        """Return surfs as dict with appropriate keys (eg LPS)"""

        keys = [self.side + n for n in ["WS", "SS", "PS"]]
        vals = [self.inSurf, self.sphere, self.outSurf]
        return dict(zip(keys, vals))

    def transform(self, mat):
        """
        Apply affine transformation to each surface. Returns a new Hemisphre.
        """

        surfs = [s.transform(mat) for s in self.surfs]
        return Hemisphere(*surfs, self.sphere, side=self.side)

    def midsurface(self):
        """Midsurface between inner and outer cortex"""

        vec = self.outSurf.points - self.inSurf.points
        points = self.inSurf.points + (0.5 * vec)
        return Surface.manual(points, self.inSurf.tris)

    def thickness(self):
        vec = self.outSurf.points - self.inSurf.points
        return np.linalg.norm(vec, ord=2, axis=-1)

    def adjacency_matrix(self):
        """
        Adjacency matrix of any cortical surface (they necessarily share
        the same triagulation, which is checked during initialisation).

        """

        return self.inSurf.adjacency_matrix()

    @property
    def n_points(self):
        """Number of vertices on either cortical surface"""
        return self.inSurf.n_points

    def mesh_laplacian(self):
        """
        Mesh Laplacian on cortical midsurface.

        Returns:
            sparse CSR matrix
        """

        return self.midsurface().mesh_laplacian()

    def resample_geometry(self, n_verts):
        """Resample geometry to a different vertex resolution

        Args:
            white (str): white surface
            pial (str): pial surface
            sph (str): sphere relating to white/pial
            n_verts (int): target vertex resolution

        (white, pial) Surface objects

        """

        sph_mid = utils.find_sphere_centre(self.sphere.points)
        if not np.allclose(sph_mid, 0, atol=1e-2):
            raise RuntimeError("Input sphere not centred on origin")

        sph_in = Trimesh(vertices=self.sphere.points, faces=self.sphere.tris)
        sph_in = utils.make_unit_radius(sph_in)

        # New sphere at target output resolution
        sph_out = icosphere.icosphere(nr_verts=int(n_verts))
        sph_mid = utils.find_sphere_centre(sph_out[0])
        if not np.allclose(sph_mid, 0, atol=1e-2):
            raise RuntimeError("Input sphere not centred on origin")
        utils.check_spherical(sph_out[0])

        sph_out = Trimesh(vertices=sph_out[0], faces=sph_out[1])
        sph_out = utils.make_unit_radius(sph_out)

        if not (sph_in.is_watertight and sph_in.is_winding_consistent):
            raise RuntimeError("Input sphere not closed or winding consistent")

        if not (sph_out.is_watertight and sph_out.is_winding_consistent):
            raise RuntimeError("Output sphere not closed or winding consistent")

        # For each vertex of the output sphere, find the
        # closest point and triangle on the input sphere
        (
            nr_pt,
            _,
            nr_t,
        ) = proximity.closest_point(sph_in, sph_out.vertices)

        # Express the closest points in barycentric coordinates,
        # then use these to reconstruct the output vertices from
        # the input vertices.
        weights = utils.barycentric_coordinates(nr_pt, nr_t, sph_in)
        test = utils.interpolate_barycentric(
            weights, sph_in.vertices, sph_in.faces[nr_t]
        )

        # Check the quality of the interpolation
        dist = test - nr_pt
        if not np.allclose(dist, 0, atol=1e-6):
            raise RuntimeError("Barycentric weights do not reconstruct sphere")

        # Now we can re-use the same weights to interpolate
        # the white and pial surfaces. The new triangles are defined
        # by the output sphere
        white_out, pial_out = [
            Surface.manual(
                utils.interpolate_barycentric(weights, s.points, sph_in.faces[nr_t]),
                sph_out.faces,
                s.name,
            )
            for s in [self.inSurf, self.outSurf]
        ]

        sph_out = Surface.manual(sph_out.vertices, sph_out.faces, name=self.sphere.name)
        return Hemisphere(white_out, pial_out, sph_out, side=self.side)
