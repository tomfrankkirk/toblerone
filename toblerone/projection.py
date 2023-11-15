"""
Projection between surface, volume and hybrid spaces 
"""

import multiprocessing as mp
import os
from collections import defaultdict
from textwrap import dedent

import h5py
import numpy as np
from scipy import sparse

from toblerone import surface_estimators, utils
from toblerone.classes import Hemisphere, ImageSpace, Surface
from toblerone.core import vox_tri_weights, vtx_tri_weights

SIDES = ["L", "R"]


class Projector(object):
    """
    Use to perform projection between volume, surface and node space.
    Creating a projector object may take some time whilst the consituent
    matrices are prepared; once created any of the individual projections
    may be calculated directly from the object.

    Node space ordering: L hemisphere surface, R hemisphere surface,
    brain voxels voxels in linear index order, ROIs in alphabetical
    order according to their dictionary key (see below)

    WARNING: surfaces must be in alignment before calling this function - ie,
    apply all registrations beforehand.

    Args:
        hemispheres (list/Hemisphere): single or list (L/R) of Hemisphere objects.
        spc (str/ImageSpace): path for, or ImageSpace object, for voxel grid
            to project from/to
        rois (dict): ROIs; keys are ROI name and values
            are volumetric PV maps representing ROI fraction.
        factor (int): voxel subdivision factor (default 3x voxel size)
        cores (int): number of processor cores to use (default max)
        ones (bool): debug tool, whole voxel PV assignment.
    """

    def __init__(
        self,
        ref,
        struct2ref,
        hemispheres,
        roi_pvs={},
        nonbrain_pvs=None,
        cores=mp.cpu_count(),
        ones=False,
    ):
        if not isinstance(ref, ImageSpace):
            self.spc = ImageSpace(ref)
        else:
            self.spc = ref

        if isinstance(hemispheres, Hemisphere):
            hemispheres = [hemispheres]

        self.hemi_dict = {h.side: h for h in hemispheres}
        self.roi_pvs = {}
        for k, v in roi_pvs.items():
            if not (k.startswith("L_") or k.startswith("R_")):
                raise ValueError(
                    "All subcortical ROIs must have keys starting with L_ or R_"
                )

            if not np.all(v.shape == self.spc.size):
                raise ValueError(
                    f"PVs for ROI {k} do not have same shape "
                    f"as reference space {v.shape} vs {self.spc.size}"
                )
            if k[0] in self.hemi_dict.keys():
                self.roi_pvs[k] = v

        self.__cortex_pvs = {}
        self.__vox_tri_mats = {}
        self.__vtx_tri_mats = {}

        ncores = cores if hemispheres[0].inSurf._use_mp else 1
        supr = np.maximum(np.floor(self.spc.vox_size.round(1) / 0.75), 1).astype(
            np.int32
        )

        for s, h in self.hemi_dict.items():
            h = h.transform(struct2ref)
            self.hemi_dict[s] = h
            hpvs = surface_estimators.cortex(
                h,
                self.spc,
                np.eye(4),
                supr=supr,
                cores=cores,
                ones=ones,
            )
            self.__cortex_pvs[s] = hpvs

        # Mask using the cortex PVs to remove irrelevant subcortex
        # structures (used for enforcing single hemisphere)
        if nonbrain_pvs is None:
            self.nonbrain_pvs = self.cortex_pvs()[..., -1]
        else:
            if not np.all(nonbrain_pvs.shape == self.spc.size):
                raise ValueError(
                    f"Nonbrain PVs do not have same shape as reference "
                    f"space {nonbrain_pvs.shape} vs {self.spc.size}"
                )
            ctx_mask = self.cortex_pvs()[..., :2].any(-1)
            nonbrain_pvs[~ctx_mask] = 1
            self.nonbrain_pvs = nonbrain_pvs

        factor = 2 * np.ceil(self.spc.vox_size).astype(int)
        self._assemble_vtx_vox_mats(factor, ncores, ones)

    def _assemble_vtx_vox_mats(self, factor, ncores, ones):
        for hemi in self.iter_hemis:
            # Calculate the constituent matrices for projection with each hemi
            midsurf = hemi.midsurface()
            vox_tri = vox_tri_weights(
                *hemi.surfs,
                self.spc,
                factor,
                cores=ncores,
                ones=ones,
                descriptor=f"{hemi.side} prisms",
            )
            vtx_tri = vtx_tri_weights(midsurf, ncores)
            self.__vox_tri_mats[hemi.side] = vox_tri
            self.__vtx_tri_mats[hemi.side] = vtx_tri

    def save(self, path):
        """Save Projector in HDF5 format.

        A projector can be re-used for multiple analyses, assuming the reference
        image space and cortical surfaces remain in alignment for all data.

        Args:
            path (str): path to write out with .h5 extension
        """

        ctype = "gzip"

        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with h5py.File(path, "w") as f:
            # Save properties of the reference ImageSpace: vox2world, size
            # and filename
            f.create_dataset("ref_spc_vox2world", data=self.spc.vox2world)
            f.create_dataset("ref_spc_size", data=self.spc.size)
            if self.spc.fname:
                f.create_dataset(
                    "ref_spc_fname",
                    data=np.array(self.spc.fname.encode("utf-8")),
                    dtype=h5py.string_dtype("utf-8"),
                )

            f.create_dataset("nonbrain_pvs", data=self.nonbrain_pvs)

            # Each hemisphere is a group within the file (though there may
            # only be 1)
            for h in self.iter_hemis:
                side = h.side
                g = f.create_group(f"{side}_hemi")
                g.create_dataset(
                    f"{side}_pvs", data=self.__cortex_pvs[h.side], compression=ctype
                )

                # Sparse matrices cannot be save in HDF5, so convert them
                # to COO and then save as a 3 x N array, where the top row
                # is row indices, second is columns, and last is data.
                voxtri = self.__vox_tri_mats[h.side].tocoo()
                voxtri = np.vstack((voxtri.row, voxtri.col, voxtri.data))
                g.create_dataset(f"{side}_vox_tri", data=voxtri, compression=ctype)

                # Same again: top row is row indices, then cols, then data
                vtxtri = self.__vtx_tri_mats[h.side].tocoo()
                vtxtri = np.vstack((vtxtri.row, vtxtri.col, vtxtri.data))
                g.create_dataset(f"{side}_vtx_tri", data=vtxtri, compression=ctype)

                # Finally, save the surfaces of each hemisphere, named
                # as LPS,RPS,LWS,RWS.
                for k, s in h.surf_dict.items():
                    g.create_dataset(f"{k}_tris", data=s.tris, compression=ctype)
                    g.create_dataset(f"{k}_points", data=s.points, compression=ctype)

            # Save ROI pvs
            if self.roi_pvs:
                g = f.create_group("roi_pvs")
                for k, v in self.roi_pvs.items():
                    g.create_dataset(k, data=v, compression=ctype)

    @classmethod
    def load(cls, path):
        """Load Projector from path in HDF5 format.

        This is useful for performing repeated analyses with the same voxel
        grid and cortical surfaces.

        Args:
            path (str): path to load from
        """

        with h5py.File(path, "r") as f:
            p = cls.__new__(cls)

            # Recreate the reference ImageSpace first
            p.spc = ImageSpace.manual(f["ref_spc_vox2world"][()], f["ref_spc_size"][()])
            if "ref_spc_fname" in f:
                fname = f["ref_spc_fname"][()]
                if isinstance(fname, bytes):
                    fname = fname.decode("utf-8")
                p.spc.fname = fname
            n_vox = p.spc.n_vox

            # Now read out hemisphere specific properties
            p.__cortex_pvs = {}
            p.__vox_tri_mats = {}
            p.__vtx_tri_mats = {}
            p.hemi_dict = {}
            p.roi_pvs = {}
            p.nonbrain_pvs = f["nonbrain_pvs"][()]

            for s in SIDES:
                hemi_key = f"{s}_hemi"
                if hemi_key in f:
                    # Read out the surfaces, create the Hemisphere
                    ins, outs, sph = [
                        Surface.manual(
                            f[hemi_key][f"{s}{n}S_points"][()],
                            f[hemi_key][f"{s}{n}S_tris"][()],
                            f"{s}{n}S",
                        )
                        for n in ["W", "P", "S"]
                    ]
                    p.hemi_dict[s] = Hemisphere(ins, outs, sph, s)

                    # Read out the PVs array for the hemi
                    p.__cortex_pvs[s] = f[hemi_key][f"{s}_pvs"][()]

                    # Recreate the sparse voxtri and vtxtri matrices.
                    # They are stored as a 3 x N array, where top row
                    # is row indices, second is column, then data
                    voxtri = f[hemi_key][f"{s}_vox_tri"][()]
                    assert voxtri.shape[0] == 3, "expected 3 rows"
                    voxtri = sparse.coo_matrix(
                        (voxtri[2, :], (voxtri[0, :], voxtri[1, :])),
                        shape=(n_vox, ins.tris.shape[0]),
                    )
                    p.__vox_tri_mats[s] = voxtri.tocsr()

                    # Same convention as above
                    vtxtri = f[hemi_key][f"{s}_vtx_tri"][()]
                    assert vtxtri.shape[0] == 3, "expected 3 rows"
                    vtxtri = sparse.coo_matrix(
                        (vtxtri[2, :], (vtxtri[0, :], vtxtri[1, :])),
                        shape=(ins.n_points, ins.tris.shape[0]),
                    )
                    p.__vtx_tri_mats[s] = vtxtri.tocsr()

            if "roi_pvs" in f:
                g = f["roi_pvs"]
                for k in sorted(g.keys()):
                    p.roi_pvs[k] = g[k][()]

            return p

    def __repr__(self):
        sides = ",".join(list(self.hemi_dict.keys()))
        nrois = self.n_rois
        nnodes = self.n_nodes
        spc = "\n".join(repr(self.spc).splitlines()[1:])
        disp = dedent(
            f"""\
        Projector for {sides} hemispheres, {nrois} ROIS, and {nnodes} total nodes.
        Reference voxel grid:"""
        )
        return disp + "\n" + spc

    @property
    def iter_hemis(self):
        """Iterator over hemispheres of projector, in L/R order"""
        for s in SIDES:
            if s in self.hemi_dict:
                yield self.hemi_dict[s]

    @property
    def n_hemis(self):
        """Number of hemispheres (1/2) in projector"""
        return len(self.hemi_dict)

    # Direct access to the underlying surfaces via keys LPS, RWS etc.
    def __getitem__(self, surf_key):
        side = surf_key[0]
        return self.hemi_dict[side].surf_dict[surf_key]

    @property
    def n_surf_nodes(self):
        """Number of surface vertices in projector (one or both hemis)"""
        return sum([h.n_points for h in self.iter_hemis])

    @property
    def n_nodes(self):
        """Number of nodes in projector (surface, voxels, ROIs)"""
        return sum([self.spc.n_vox, self.n_surf_nodes, len(self.roi_pvs)])

    @property
    def n_vol_nodes(self):
        """Number of voxels in reference ImageSpace"""
        return self.spc.n_vox

    @property
    def n_vox(self):
        """Number of voxels in reference ImageSpace"""
        return self.spc.n_vox

    @property
    def n_rois(self):
        """Number of ROIs"""
        return len(self.roi_pvs)

    @property
    def roi_names(self):
        """List of names for ROIs"""
        return list(self.roi_pvs.keys())

    def adjacency_matrix(self):
        """Adjacency matrix for all surface vertices of projector.

        If there are two hemispheres present, the matrix indices will
        be arranged L,R.

        Returns:
            sparse CSR matrix, square sized (n vertices)
        """

        mats = []
        for hemi in self.iter_hemis:
            midsurf = hemi.midsurface()
            a = midsurf.adjacency_matrix().tolil()
            verts_vox = utils.affine_transform(
                midsurf.points, self.spc.world2vox
            ).round()
            verts_in_spc = ((verts_vox >= 0) & (verts_vox < self.spc.size)).all(-1)
            a[~verts_in_spc, :] = 0
            a[..., ~verts_in_spc] = 0
            assert utils.is_symmetric(a)
            mats.append(a)

        return sparse.block_diag(mats, format="csr")

    def mesh_laplacian(self):
        """Mesh Laplacian matrix for all surface vertices of projector.

        If there are two hemispheres present, the matrix indices will be
        arranged L/R.

        Returns:
            sparse CSR matrix
        """

        mats = [h.mesh_laplacian() for h in self.iter_hemis]
        return sparse.block_diag(mats, format="csr")

    def cortex_pvs(self):
        """Cortical PVs for all hemispheres of Projector.

        Returns:
            np.array, same shape as reference space, arranged GM, WM,
                non-brain in 4th dim.
        """
        if len(self.__cortex_pvs) > 1:
            # Combine PV estimates from each hemisphere into single map
            pvs = np.zeros((*self.spc.size, 3))
            pvs[..., 0] = np.minimum(
                1.0, self.__cortex_pvs["L"][..., 0] + self.__cortex_pvs["R"][..., 0]
            )
            pvs[..., 1] = np.minimum(
                1.0 - pvs[..., 0],
                self.__cortex_pvs["L"][..., 1] + self.__cortex_pvs["R"][..., 1],
            )
            pvs[..., 2] = 1.0 - pvs[..., 0:2].sum(-1)
            return pvs
        else:
            pvs = next(iter(self.__cortex_pvs.values()))
            return pvs

    def cortex_thickness(self):
        return np.concatenate([h.thickness() for h in self.iter_hemis])

    def subcortex_pvs(self):
        """Flattened 3D array of interior/exterior PVs for all ROIs.

        Returns:
            np.array, same shape as ``self.ref_spc``
        """

        if self.roi_pvs:
            pvs = np.stack(self.roi_pvs.values(), axis=-1)
            return np.clip(pvs.sum(-1).reshape(self.spc.size), 0, 1)
        else:
            return np.zeros(self.spc.size)

    def pvs(self):
        """Flattened 4D array of PVs for cortex, subcortex and ROIs.

        Returns:
            np.array, same shape as reference space, arranged GM, WM,
                non-brain in 4th dim.
        """

        # We may or may not have ROIs present. If not, use
        # a default dict that returns empty PV arrays to trick the
        # stacking code to work properly (it expects all FIRST ROIs to
        # be present)
        cpvs = self.cortex_pvs()
        to_stack = defaultdict(lambda: np.zeros(cpvs.shape[:3]))

        to_stack.update(
            {
                "cortex_GM": cpvs[..., 0],
                "cortex_WM": cpvs[..., 1],
                "cortex_nonbrain": cpvs[..., 2],
                "nonbrain": self.nonbrain_pvs,
                **self.roi_pvs,
            }
        )

        return utils.stack_images(to_stack)

    def brain_mask(self, pv_threshold=0.1):
        """Boolean mask of brain voxels, in reference ImageSpace

        Args:
            pv_threshold (float): minimum brain PV (WM+GM) to include

        Returns:
            np.array, same shape as reference space, boolean dtype
        """

        pvs = self.cortex_pvs()
        return pvs[..., :2].sum(-1) > pv_threshold

    def vol2surf_matrix(self, edge_scale, vol_mask=None, surf_mask=None):
        """
        Volume to surface projection matrix.

        Args:
            edge_scale (bool):  upweight signal from voxels that are not
                                100% brain to account for 'missing' signal.
                                Set True for quantities that scale with PVE
                                (eg perfusion), set False otherwise
                                (eg time quantities)

            vol_mask (np.array,bool): bool mask of voxels to include
            surf_mask (np.array,bool): bool mask of surface vertices to include

        Returns:
            sparse CSR matrix, sized (surface vertices x voxels). Surface vertices
                are arranged L then R.
        """

        if surf_mask is None:
            surf_mask = np.ones(self.n_surf_nodes, dtype=bool)

        if vol_mask is None:
            vol_mask = np.ones(self.n_vox, dtype=bool)

        if not (vol_mask.shape[0] == vol_mask.size == self.spc.n_vox):
            raise ValueError(
                f"Vol mask incorrectly sized ({vol_mask.shape})"
                f" for reference ImageSpace ({self.spc.n_vox})"
            )

        if not (surf_mask.shape[0] == surf_mask.size == self.n_surf_nodes):
            raise ValueError(
                f"Surf mask incorrectly sized ({surf_mask.shape})"
                f" for projector surfaces ({self.n_surf_nodes})"
            )

        proj_mats = [
            assemble_vol2surf(self.__vox_tri_mats[h.side], self.__vtx_tri_mats[h.side])
            for h in self.iter_hemis
        ]
        v2s_mat = sparse.vstack(proj_mats, format="csr")
        assert v2s_mat.sum(1).max() < 1 + 1e-6

        if edge_scale:
            brain_pv = self.cortex_pvs().reshape(-1, 3)[..., :2].sum(1)
            brain = brain_pv > 1e-3
            upweight = np.ones(brain_pv.shape)
            upweight[brain] = 1 / brain_pv[brain]
            v2s_mat.data *= np.take(upweight, v2s_mat.indices)

        if not (vol_mask.all() and surf_mask.all()):
            v2s_mat = utils.mask_projection_matrix(
                v2s_mat, row_mask=surf_mask, col_mask=vol_mask
            )

        return v2s_mat

    def vol2hybrid_matrix(self, edge_scale, vol_mask=None, node_mask=None):
        """
        Volume to node space projection matrix.

        Args:
            edge_scale (bool):  upweight signal from voxels that are not
                                100% brain to account for 'missing' signal.
                                Set True for quantities that scale with PVE
                                (eg perfusion), set False otherwise
                                (eg time quantities)

            vol_mask (np.array,bool): bool mask of voxels to include
            node_mask (np.array,bool): bool mask of nodes to include

        Returns:
            sparse CSR matrix, sized (nodes x voxels)
        """

        if node_mask is None:
            node_mask = np.ones(self.n_nodes, dtype=bool)

        if vol_mask is None:
            vol_mask = np.ones(self.spc.n_vox, dtype=bool)

        if not (vol_mask.shape[0] == vol_mask.size == self.spc.n_vox):
            raise ValueError(
                f"Vol mask incorrectly sized ({vol_mask.shape})"
                f" for reference ImageSpace ({self.spc.n_vox})"
            )

        if not (node_mask.shape[0] == node_mask.size == self.n_nodes):
            raise ValueError(
                f"Node mask incorrectly sized ({node_mask.shape})"
                f" for projector nodes ({self.n_nodes})"
            )

        surf_mask = node_mask[: self.n_surf_nodes]
        v2s_mat = self.vol2surf_matrix(
            edge_scale, vol_mask=vol_mask, surf_mask=surf_mask
        )

        wm_mask = node_mask[self.n_surf_nodes : self.n_nodes - self.n_rois]
        v2v_mat = sparse.eye(self.spc.n_vox)

        if edge_scale:
            brain_pv = self.pvs().reshape(-1, 3)
            brain_pv = brain_pv[..., :2].sum(1)
            brain = brain_pv > 1e-3
            upweight = np.ones(brain_pv.shape)
            upweight[brain] = 1 / brain_pv[brain]
            v2v_mat.data *= upweight

        v2v_mat = utils.slice_sparse(v2v_mat, wm_mask, vol_mask)

        if self.roi_pvs:
            # mapping from voxels to ROIs - weighted averaging
            v2r_mat = np.stack([r.flatten() for r in self.roi_pvs.values()], axis=0)
            v2r_mat = sparse.csr_matrix(v2r_mat)
            v2r_mat = utils.sparse_normalise(v2r_mat, 1)
            assert v2r_mat.sum(1).max() < 1 + 1e-6
            if edge_scale:
                v2r_mat.data *= np.take(upweight, v2r_mat.indices)

            roi_mask = node_mask[-self.n_rois :]
            v2r_mat = utils.mask_projection_matrix(
                v2r_mat, row_mask=roi_mask, col_mask=vol_mask
            )

            v2n_mat = sparse.vstack((v2s_mat, v2v_mat, v2r_mat), format="csr")

        else:
            v2n_mat = sparse.vstack((v2s_mat, v2v_mat), format="csr")

        return v2n_mat

    def surf2vol_matrix(self, edge_scale, vol_mask=None, surf_mask=None):
        """
        Surface to volume projection matrix.

        Args:
            edge_scale (bool): downweight signal in voxels that are not 100% brain:
                               set True for data that scales with PVE (eg perfusion),
                               set False for data that does not (eg time quantities).

            vol_mask (np.array,bool): bool mask of voxels to include
            surf_mask (np.array,bool): bool mask of surface vertices to include

        Returns:
            sparse CSC matrix, sized (surface vertices x voxels)
        """

        if surf_mask is None:
            surf_mask = np.ones(self.n_surf_nodes, dtype=bool)

        if vol_mask is None:
            vol_mask = np.ones(self.n_vox, dtype=bool)

        if not (vol_mask.shape[0] == vol_mask.size == self.spc.n_vox):
            raise ValueError(
                f"Vol mask incorrectly sized ({vol_mask.shape})"
                f" for reference ImageSpace ({self.spc.n_vox})"
            )

        if not (surf_mask.shape[0] == surf_mask.size == self.n_surf_nodes):
            raise ValueError(
                f"Surf mask incorrectly sized ({surf_mask.shape})"
                f" for projector surfaces ({self.n_surf_nodes})"
            )

        # If voxels are shared by both hemispheres, split the relative
        # weighting according to the GM PV of each hemisphere. This is
        # not a PV weighting - just deciding which hemi contributes more
        # to the signal.
        if self.n_hemis == 2:
            gm_sum = self.__cortex_pvs["L"][..., 0] + self.__cortex_pvs["R"][..., 0]
            denom = np.ones_like(gm_sum)
            fltr = gm_sum > 0
            denom[fltr] = 1 / gm_sum[fltr]
            l_weight = np.clip(self.__cortex_pvs["L"][..., 0] * denom, 0, 1)
            r_weight = np.clip(self.__cortex_pvs["R"][..., 0] * denom, 0, 1)
            weights = {"L": l_weight, "R": r_weight}
        else:
            weights = {next(self.iter_hemis).side: np.ones(self.spc.n_vox)}

        proj_mats = []
        for h in self.iter_hemis:
            x = assemble_surf2vol(
                self.__vox_tri_mats[h.side], self.__vtx_tri_mats[h.side]
            ).tocsc()
            x.data *= np.take(weights[h.side], x.indices)
            proj_mats.append(x)

        s2v_mat = sparse.hstack(proj_mats, format="csc")
        s2v_mat = utils.sparse_normalise(s2v_mat, axis=1)

        if edge_scale:
            pvs = self.cortex_pvs().reshape(-1, 3)
            s2v_mat.data *= np.take(pvs[..., 0], s2v_mat.indices)

        if not (vol_mask.all() and surf_mask.all()):
            s2v_mat = utils.mask_projection_matrix(
                s2v_mat, row_mask=vol_mask, col_mask=surf_mask
            )

        s2v_mat.data = np.clip(s2v_mat.data, 0, 1)
        assert s2v_mat.sum(1).max() < 1 + 1e-6
        return s2v_mat

    def hybrid2vol_matrix(self, edge_scale, vol_mask=None, node_mask=None):
        """
        Node space to volume projection matrix.

        Args:
            edge_scale (bool): downweight signal in voxels that are not 100% brain:
                               set True for data that scales with PVE (eg perfusion),
                               set False for data that does not (eg time quantities).

            vol_mask (np.array,bool): bool mask of voxels to include
            node_mask (np.array,bool): bool mask of nodes to include

        Returns:
            sparse CSR matrix, sized (voxels x nodes)
        """

        # Assemble the matrices corresponding to cortex and subcortex individually.
        # Regardless of whether the overall projection should be edge_scaled or not,
        # we want the balance between cortex and subcortex to be determined by the
        # weight of GM and WM, which is why we scale both matrices accordingly.

        # If the final result should be edge_scaled, then we simply stack and return
        # the result. If the final result should not be scaled, then we normalise
        # so that each row sums to 1 to get a weighted-average projection. In both
        # cases, the weighting given to cortex/subcortex within the projection is
        # determined by GM and WM PVs, only the scaling of the final matrix changes.

        if node_mask is None:
            node_mask = np.ones(self.n_nodes, dtype=bool)

        if vol_mask is None:
            vol_mask = np.ones(self.spc.n_vox, dtype=bool)

        if not (vol_mask.shape[0] == vol_mask.size == self.spc.n_vox):
            raise ValueError(
                f"Vol mask incorrectly sized ({vol_mask.shape})"
                f" for reference ImageSpace ({self.spc.n_vox})"
            )

        if not (node_mask.shape[0] == node_mask.size == self.n_nodes):
            raise ValueError(
                f"Node mask incorrectly sized ({node_mask.shape})"
                f" for projector nodes ({self.n_nodes})"
            )

        surf_mask = node_mask[: self.n_surf_nodes]
        wm_mask = node_mask[self.n_surf_nodes : self.n_nodes - self.n_rois]

        cpvs = self.cortex_pvs().reshape(-1, 3)
        pvs = self.pvs().reshape(-1, 3)

        if self.roi_pvs:
            # subcortical GM PVs, stacked across ROIs
            spvs = np.stack([r.flatten() for r in self.roi_pvs.values()], axis=1)

            # The sum of ROI GM and cortex GM can be greater than 1,
            # in which case we downweight until they sum to 1 again. This
            # doesn't apply to voxels with GM sum less than 1
            sgm_sum = spvs.sum(-1)
            rescale = np.maximum(1, cpvs[..., 0] + sgm_sum)

            # mapping from subcortial ROIs to voxels is just the PV matrix
            spvs = spvs / rescale[..., None]
            r2v_mat = sparse.csr_matrix(spvs)
            roi_mask = node_mask[-self.n_rois :]
            r2v_mat = utils.mask_projection_matrix(
                r2v_mat, row_mask=vol_mask, col_mask=roi_mask
            )

            # mappings from surface to voxel and from ROIs to voxels
            s2v_mat = self.surf2vol_matrix(edge_scale=True).tocsc()
            s2v_mat.data /= np.take(rescale, s2v_mat.indices)
            s2v_mat = utils.mask_projection_matrix(
                s2v_mat, row_mask=vol_mask, col_mask=surf_mask
            )

            v2v_mat = sparse.dia_matrix(
                (pvs[:, 1], 0), shape=2 * [self.spc.n_vox]
            ).tocsr()
            v2v_mat = utils.mask_projection_matrix(
                v2v_mat, row_mask=vol_mask, col_mask=wm_mask
            )

            n2v_mat = sparse.hstack((s2v_mat, v2v_mat, r2v_mat), format="csr")

        else:
            s2v_mat = self.surf2vol_matrix(edge_scale=True)
            s2v_mat = utils.mask_projection_matrix(
                s2v_mat, row_mask=vol_mask, col_mask=surf_mask
            )

            v2v_mat = sparse.dia_matrix(
                (cpvs[:, 1], 0), shape=2 * [self.spc.n_vox]
            ).tocsr()
            v2v_mat = utils.mask_projection_matrix(
                v2v_mat, row_mask=vol_mask, col_mask=wm_mask
            )

            n2v_mat = sparse.hstack((s2v_mat, v2v_mat), format="csr")

        assert n2v_mat.sum(1).max() < 1 + 1e-6
        if not edge_scale:
            n2v_mat = utils.sparse_normalise(n2v_mat, 1)

        return n2v_mat

    def vol2surf(self, vdata, edge_scale):
        """
        Project data from volum to surface.

        Args:
            vdata (np.array):   sized n_voxels in first dimension
            edge_scale (bool):  upweight signal from voxels that are not
                                100% brain to account for 'missing' signal.
                                Set True for quantities that scale with PVE
                                (eg perfusion), set False otherwise
                                (eg time quantities)

        Returns:
            np.array, sized n_vertices in first dimension
        """

        if vdata.shape[0] != self.spc.n_vox:
            raise RuntimeError(
                "vdata must have the same number of rows as"
                + " voxels in the reference ImageSpace"
            )
        v2s_mat = self.vol2surf_matrix(edge_scale)
        return v2s_mat.dot(vdata)

    def surf2vol(self, sdata, edge_scale):
        """
        Project data from surface to volume.

        Args:
            sdata (np.array):  sized n_vertices in first dimension (arranged L,R)
            edge_scale (bool): downweight signal in voxels that are not 100% brain:
                               set True for data that scales with PVE (eg perfusion),
                               set False for data that does not (eg time quantities).

        Returns:
            np.array, sized n_voxels in first dimension
        """

        s2v_mat = self.surf2vol_matrix(edge_scale)
        if sdata.shape[0] != s2v_mat.shape[1]:
            raise RuntimeError(
                "sdata must have the same number of rows as"
                + " total surface nodes (were one or two hemispheres used?)"
            )
        return s2v_mat.dot(sdata)

    def vol2hybrid(self, vdata, edge_scale):
        """
        Project data from volume to node space.

        Args:
            vdata (np.array):   sized n_voxels in first dimension
            edge_scale (bool):  upweight signal from voxels that are not
                                100% brain to account for 'missing' signal.
                                Set True for quantities that scale with PVE
                                (eg perfusion), set False otherwise
                                (eg time quantities)

        Returns:
            np.array, sized (n_vertices + n_voxels) in first dimension. Surface vertices are arranged L then R.
        """

        v2n_mat = self.vol2hybrid_matrix(edge_scale)
        if vdata.shape[0] != v2n_mat.shape[1]:
            raise RuntimeError(
                "vdata must have the same number of rows as"
                + " nodes (voxels+vertices) in the reference ImageSpace"
            )
        return v2n_mat.dot(vdata)

    def hybrid2vol(self, ndata, edge_scale):
        """
        Project data from node space to volume.

        Args:
            ndata (np.array):  sized (n_vertices + n_voxels) in first dimension,
                                   Surface data should be arranged L, R in the first dim.
            edge_scale (bool): downweight signal in voxels that are not 100% brain:
                               set True for data that scales with PVE (eg perfusion),
                               set False for data that does not (eg time quantities).
        Returns:
            np.array, sized n_voxels in first dimension
        """

        n2v_mat = self.hybrid2vol_matrix(edge_scale)
        if ndata.shape[0] != n2v_mat.shape[1]:
            raise RuntimeError(
                "ndata must have the same number of rows as"
                + " total nodes in ImageSpace (voxels+vertices)"
            )
        return n2v_mat.dot(ndata)


def assemble_vol2surf(vox_tri, vtx_tri):
    """
    Combine with normalisation the vox_tri and vtx_tri matrices into vol2surf.
    """

    # Ensure each triangle's voxel weights sum to 1
    # Ensure each vertices' triangle weights sum to 1
    vox2tri = utils.sparse_normalise(vox_tri, 0).T
    tri2vtx = utils.sparse_normalise(vtx_tri, 1)
    vol2vtx = tri2vtx @ vox2tri
    return utils.sparse_normalise(vol2vtx, 1)


def assemble_surf2vol(vox_tri, vtx_tri):
    """
    Combine with normalisation the vox_tri and vtx_tri matrices into surf2vol.
    """

    # Ensure each triangle's vertex weights sum to 1
    # Ensure each voxel's triangle weights sum to 1
    vtx2tri = utils.sparse_normalise(vtx_tri, 0).T
    tri2vox = utils.sparse_normalise(vox_tri, 1)
    vtx2vox = tri2vox @ vtx2tri
    return utils.sparse_normalise(vtx2vox, 1)
