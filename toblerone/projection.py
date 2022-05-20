"""
Projection between surface, volume and hybrid spaces 
"""

import multiprocessing as mp 
import copy
import os
from textwrap import dedent

import numpy as np 
from scipy import sparse 
import h5py

from toblerone import utils 
from toblerone.pvestimation import estimators
from toblerone.classes import Hemisphere, Surface, ImageSpace
from toblerone.core import vtx_tri_weights, vox_tri_weights

SIDES = ['L', 'R']

class Projector(object):
    """
    Use to perform projection between volume, surface and node space. 
    Creating a projector object may take some time whilst the consituent 
    matrices are prepared; once created any of the individual projections
    may be calculated directly from the object. 

    Node space ordering: L hemisphere surface, R hemisphere surface, 
    subcortical voxels in linear index order, subcortical ROIs in alphabetical 
    order according to their dictionary key (see below)

    Args: 
        hemispheres (list/Hemisphere): single or list (L/R) of Hemisphere objects.
            Note that the surfaces of the hemispheres must be in alignment with 
            the reference space (ie, apply any transformations beforehand).
        spc (str/ImageSpace): path for, or ImageSpace object, for voxel grid 
            to project from/to 
        rois (dict): subcortical ROIs; keys are ROI name and values 
            are paths to surfaces or Surface objects for the ROIs themselves. 
        factor (int): voxel subdivision factor (default 3x voxel size)
        cores (int): number of processor cores to use (default max)
        ones (bool): debug tool, whole voxel PV assignment. 
    """

    def __init__(self, hemispheres, spc, rois=None, factor=None,
                cores=mp.cpu_count(), ones=False):

        print("Initialising projector (will take some time)")
        if not isinstance(hemispheres, Hemisphere):
            if len(hemispheres) == 2:
                if (any([ h.side not in SIDES for h in hemispheres ])
                    and (hemispheres[0].side == hemispheres[1].side)):
                    raise ValueError("Hemisphere objects must have 'L' and 'R' sides") 
        
        else: 
            if not isinstance(hemispheres, Hemisphere): 
                raise ValueError("Projector must be initialised with 1 or 2 Hemispheres")
            side = hemispheres.side[0]
            if not side in SIDES: 
                raise ValueError("Hemisphere must have 'L' or 'R' side")
            hemispheres = [hemispheres]
            
        self.hemi_dict = { h.side: copy.deepcopy(h) for h in hemispheres }
        if not isinstance(spc, ImageSpace): 
            spc = ImageSpace(spc)
        self.spc = spc 
        self._hemi_pvs = [] 
        self.vox_tri_mats = [] 
        self.vtx_tri_mats = []
        self._roi_pvs = {}

        if factor is None:
            factor = np.ceil(3 * spc.vox_size)
        factor = (factor * np.ones(3)).astype(np.int32)
        ncores = cores if hemispheres[0].inSurf._use_mp else 1 
        supersampler = np.maximum(np.floor(spc.vox_size.round(1)/0.75), 
                                            1).astype(np.int32)         

        if rois is not None: 
            for name in sorted(rois.keys()): 
                assert isinstance(rois[name], Surface)
                rpvs = estimators._structure(rois[name], spc, np.eye(4), 
                                             supersampler, ones=ones, cores=cores)
                self._roi_pvs[name] = rpvs 

        for hemi in self.iter_hemis: 

            # If PV estimates are not present, then compute from scratch 
            if hasattr(hemi, 'pvs'): 
                # print("WARNING: PVs should ideally be recalculated from scratch")
                self._hemi_pvs.append(hemi.pvs.reshape(-1,3))
            else: 
                supersampler = np.maximum(np.floor(spc.vox_size.round(1)/0.75), 
                                            1).astype(np.int32)                
                pvs = estimators._cortex(hemi, spc, np.eye(4), supersampler, 
                                        cores=ncores, ones=ones)
                self._hemi_pvs.append(pvs.reshape(-1,3))

        self._assemble_vtx_vox_mats(factor, ncores, ones)


    def _assemble_vtx_vox_mats(self, factor, ncores, ones): 

        for side,hemi in self.hemi_dict.items(): 
            # Calculate the constituent matrices for projection with each hemi 
            midsurf = hemi.midsurface()
            vox_tri = vox_tri_weights(*hemi.surfs, self.spc, factor, 
                                      cores=ncores, ones=ones, 
                                      descriptor=f'{side} prisms')
            vtx_tri = vtx_tri_weights(midsurf, ncores)
            self.vox_tri_mats.append(vox_tri)
            self.vtx_tri_mats.append(vtx_tri)


    def save(self, path):
        """Save Projector in HDF5 format. 

        A projector can be re-used for multiple analyses, assuming the reference 
        image space and cortical surfaces remain in alignment for all data. 

        Args: 
            path (str): path to write out with .h5 extension 
        """

        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with h5py.File(path, 'w') as f:    

            # Save properties of the reference ImageSpace: vox2world, size
            # and filename 
            f.create_dataset('ref_spc_vox2world', data=self.spc.vox2world)
            f.create_dataset('ref_spc_size', data=self.spc.size)
            if self.spc.fname: 
                f.create_dataset('ref_spc_fname', data=np.array(
                    self.spc.fname.encode("utf-8")), 
                    dtype=h5py.string_dtype('utf-8'))

            # Each hemisphere is a group within the file (though there may 
            # only be 1)
            for idx,h in enumerate(self.iter_hemis): 

                side = h.side 
                g = f.create_group(f"{side}_hemi")
                g.create_dataset(f"{side}_pvs", data=self._hemi_pvs[idx], 
                                                compression='gzip')

                # Sparse matrices cannot be save in HDF5, so convert them 
                # to COO and then save as a 3 x N array, where the top row
                # is row indices, second is columns, and last is data. 
                voxtri = self.vox_tri_mats[idx].tocoo()
                voxtri = np.vstack((voxtri.row, voxtri.col, voxtri.data)) 
                g.create_dataset(f"{side}_vox_tri", data=voxtri, 
                                                    compression='gzip')

                # Same again: top row is row indices, then cols, then data 
                vtxtri = self.vtx_tri_mats[idx].tocoo()
                vtxtri = np.vstack((vtxtri.row, vtxtri.col, vtxtri.data))
                g.create_dataset(f"{side}_vtx_tri", data=vtxtri, 
                                                    compression='gzip')

                # Finally, save the surfaces of each hemisphere, named
                # as LPS,RPS,LWS,RWS. 
                for k,s in h.surf_dict.items(): 
                    g.create_dataset(f"{k}_tris", data=s.tris, 
                                                  compression='gzip')
                    g.create_dataset(f"{k}_points", data=s.points, 
                                                    compression='gzip')

            # Save subcortical ROI pvs 
            if self._roi_pvs: 
                g = f.create_group("subcortical_pvs")
                for k,v in self._roi_pvs.items(): 
                    g.create_dataset(k, data=v, compression='gzip')

    
    @classmethod
    def load(cls, path):
        """Load Projector from path in HDF5 format. 
        
        This is useful for performing repeated analyses with the same voxel 
        grid and cortical surfaces.

        Args: 
            path (str): path to load from 
        """
        
        with h5py.File(path, 'r') as f: 
            p = cls.__new__(cls)

            # Recreate the reference ImageSpace first 
            p.spc = ImageSpace.manual(f['ref_spc_vox2world'][()],
                                    f['ref_spc_size'][()])
            if 'ref_spc_fname' in f: 
                fname = f['ref_spc_fname'][()]
                if isinstance(fname, bytes): 
                    fname = fname.decode('utf-8')
                p.spc.fname = fname 
            n_vox = p.spc.size.prod()

            # Now read out hemisphere specific properties 
            p._hemi_pvs = [] 
            p.vox_tri_mats = [] 
            p.vtx_tri_mats = []
            p.hemi_dict = {} 
            p._roi_pvs = {}

            for s in SIDES: 
                hemi_key = f"{s}_hemi"
                if hemi_key in f: 

                    # Read out the surfaces, create the Hemisphere 
                    ins, outs = [ Surface.manual(
                        f[hemi_key][f'{s}{n}S_points'][()], 
                        f[hemi_key][f'{s}{n}S_tris'][()], f'{s}{n}S')
                        for n in ['W', 'P'] ]
                    p.hemi_dict[s] = Hemisphere(ins, outs, s)

                    # Read out the PVs array for the hemi 
                    p._hemi_pvs.append(f[hemi_key][f"{s}_pvs"][()])

                    # Recreate the sparse voxtri and vtxtri matrices. 
                    # They are stored as a 3 x N array, where top row 
                    # is row indices, second is column, then data 
                    voxtri = f[hemi_key][f"{s}_vox_tri"][()]
                    assert voxtri.shape[0] == 3, 'expected 3 rows'
                    voxtri = sparse.coo_matrix(
                        (voxtri[2,:], (voxtri[0,:], voxtri[1,:])),
                        shape=(n_vox, ins.tris.shape[0]))
                    p.vox_tri_mats.append(voxtri.tocsr())

                    # Same convention as above
                    vtxtri = f[hemi_key][f"{s}_vtx_tri"][()]
                    assert vtxtri.shape[0] == 3, 'expected 3 rows'
                    vtxtri = sparse.coo_matrix(
                        (vtxtri[2,:], (vtxtri[0,:], vtxtri[1,:])),
                        shape=(ins.n_points, ins.tris.shape[0]))
                    p.vtx_tri_mats.append(vtxtri.tocsr())

            if "subcortical_pvs" in f: 
                g = f["subcortical_pvs"]
                for k in sorted(g.keys()): 
                    p._roi_pvs[k] = g[k][()]

            return p 


    def __repr__(self):
        sides = ",".join(list(self.hemi_dict.keys()))
        nverts = sum([ h.n_points for h in self.iter_hemis ])
        spc = "\n".join(repr(self.spc).splitlines()[1:])
        disp = dedent(f"""\
        Projector for {sides} hemispheres, with {nverts} total vertices.
        Reference voxel grid:""")
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
        return sum([ h.n_points for h in self.iter_hemis ])


    @property 
    def n_nodes(self): 
        """Number of nodes in projector (surface, voxels, subcortical ROIs)"""
        return sum([ self.spc.size.prod(), self.n_surf_nodes, len(self._roi_pvs) ])


    @property 
    def n_subcortical_nodes(self):
        """Number of subcortical ROIs"""
        return len(self._roi_pvs)

    
    @property
    def roi_names(self):
        """List of names for subcortical ROIs"""
        return list(self._roi_pvs.keys())


    def adjacency_matrix(self, distance_weight=0):
        """Adjacency matrix for all surface vertices of projector.

        If there are two hemispheres present, the matrix indices will 
        be arranged L,R.  

        Args: 
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all egdes are unity), whereas positive
                values will weight edges by 1 / d^n, where d is geometric 
                distance between vertices. 

        Returns: 
            sparse CSR matrix, square sized (n vertices)
        """ 

        mats = []
        for hemi in self.iter_hemis: 
            midsurf = hemi.midsurface() 
            a = midsurf.adjacency_matrix(distance_weight).tolil()
            verts_vox = utils.affine_transform(midsurf.points, self.spc.world2vox).round()
            verts_in_spc = ((verts_vox >= 0) & (verts_vox < self.spc.size)).all(-1)
            a[~verts_in_spc,:] = 0 
            a[:,~verts_in_spc] = 0 
            assert utils.is_symmetric(a)
            mats.append(a)

        return sparse.block_diag(mats, format="csr")


    def mesh_laplacian(self, distance_weight=0):
        """Mesh Laplacian matrix for all surface vertices of projector. 

        If there are two hemispheres present, the matrix indices will be 
        arranged L/R. 

        Args: 
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all egdes are unity), whereas positive
                values will weight edges by 1 / d^n, where d is geometric 
                distance between vertices. 

        Returns: 
            sparse CSR matrix 
        """

        mats = [ h.mesh_laplacian(distance_weight) for h in self.iter_hemis ]
        return sparse.block_diag(mats, format="csr")

    def cotangent_laplacian(self): 
        """Cotangent Laplacian matrix for all surface vertices of projector. 

        If there are two hemispheres present, the matrix indices will be 
        arranged L/R. 

        Returns: 
            sparse CSR matrix, square with size (n_vertices)
        """

        mats = [ h.cotangent_laplacian() for h in self.iter_hemis ]
        return sparse.block_diag(mats, format="csr")

    def cortex_pvs(self):
        """Single Vx3 array of cortex PVs for all hemispheres of Projector. 

        Returns: 
            np.array, same shape as reference space, arranged GM, WM, 
                non-brain in 4th dim. 
        """
        if len(self._hemi_pvs) > 1:
            # Combine PV estimates from each hemisphere into single map 
            pvs = np.zeros((self.spc.size.prod(), 3))
            pvs[:,0] = np.minimum(1.0, self._hemi_pvs[0][:,0] + self._hemi_pvs[1][:,0])
            pvs[:,1] = np.minimum(1.0 - pvs[:,0], self._hemi_pvs[0][:,1] + self._hemi_pvs[1][:,1])
            pvs[:,2] = 1.0 - pvs[:,0:2].sum(1)
            return pvs.reshape(*self.spc.size, 3)
        else: 
            return self._hemi_pvs[0].reshape(*self.spc.size, 3)


    def subcortex_pvs(self):
        """Flattened 3D array of interior/exterior PVs for all ROIs.
        
        Returns: 
            np.array, same shape as ``self.ref_spc``
        """

        if self._roi_pvs:
            pvs = np.stack(list(self._roi_pvs.values()), axis=-1)
            return np.clip(pvs.sum(-1).reshape(self.spc.size), 0, 1)
        else: 
            return np.zeros(self.spc.size)


    def pvs(self):
        """Flattened 4D array of PVs for cortex, subcortex and ROIs. 

        Returns: 
            np.array, same shape as reference space, arranged GM, WM, 
                non-brain in 4th dim. 
        """

        pvs = np.zeros((self.spc.size.prod(), 3))
        cpvs = self.cortex_pvs()
        spvs = self.subcortex_pvs().flatten()
        pvs[...] = cpvs[...].reshape(-1,3)

        # Assign subcort GM from cortical CSF,
        sgm_reassigned = np.minimum(pvs[...,2], spvs)
        pvs[...,2] -= sgm_reassigned
        pvs[...,0] += sgm_reassigned

        # Assign subcort GM from cortical WM
        sgm_reassigned = spvs - sgm_reassigned
        sgm_reassigned = np.minimum(pvs[...,1], sgm_reassigned)
        pvs[...,1] -= sgm_reassigned
        pvs[...,0] += sgm_reassigned

        return pvs.reshape(*self.spc.size, 3)


    def vol2surf_matrix(self, edge_scale):
        """
        Volume to surface projection matrix. 

        Args: 
            edge_scale (bool):  upweight signal from voxels that are not
                                100% brain to account for 'missing' signal. 
                                Set True for quantities that scale with PVE 
                                (eg perfusion), set False otherwise 
                                (eg time quantities)

        Returns: 
            sparse CSR matrix, sized (surface vertices x voxels). Surface vertices 
                are arranged L then R. 
        """

        proj_mats = [ assemble_vol2surf(vox_tri, vtx_tri) 
            for vox_tri, vtx_tri in zip(self.vox_tri_mats, self.vtx_tri_mats) ]
        v2s_mat = sparse.vstack(proj_mats, format="csr")

        if edge_scale: 
            brain_pv = self.cortex_pvs().reshape(-1,3)[:,:2].sum(1)
            brain = (brain_pv > 1e-3)
            upweight = np.ones(brain_pv.shape)
            upweight[brain] = 1 / brain_pv[brain]
            v2s_mat.data *= np.take(upweight, v2s_mat.indices)

        return v2s_mat 


    def vol2hybrid_matrix(self, edge_scale): 
        """
        Volume to node space projection matrix. 

        Args: 
            edge_scale (bool):  upweight signal from voxels that are not
                                100% brain to account for 'missing' signal. 
                                Set True for quantities that scale with PVE 
                                (eg perfusion), set False otherwise 
                                (eg time quantities)

        Returns: 
            sparse CSR matrix, sized ((surface vertices + voxels) x voxels)
        """

        v2s_mat = self.vol2surf_matrix(edge_scale)
        v2v_mat = sparse.eye(self.spc.size.prod())

        if edge_scale: 
            brain_pv = self.pvs().reshape(-1,3)
            brain_pv = brain_pv[:,:2].sum(1)
            brain = (brain_pv > 1e-3)
            upweight = np.ones(brain_pv.shape)
            upweight[brain] = 1 / brain_pv[brain]
            v2v_mat.data *= upweight

        if self._roi_pvs: 

            # mapping from voxels to subcortical ROIs - weighted averaging 
            v2r_mat = np.stack(
                [ r.flatten() for r in self._roi_pvs.values() ], axis=0)
            v2r_mat = sparse.csr_matrix(v2r_mat)
            v2r_mat = utils.sparse_normalise(v2r_mat, 1)            
            if edge_scale: 
                v2r_mat.data *= np.take(upweight, v2r_mat.indices)
            v2n_mat = sparse.vstack((v2s_mat, v2v_mat, v2r_mat), format="csr")

        else: 
            v2n_mat = sparse.vstack((v2s_mat, v2v_mat), format="csr")

        return v2n_mat


    def surf2vol_matrix(self, edge_scale):
        """
        Surface to volume projection matrix. 

        Args: 
            edge_scale (bool): downweight signal in voxels that are not 100% brain: 
                               set True for data that scales with PVE (eg perfusion),
                               set False for data that does not (eg time quantities). 

        Returns: 
            sparse CSC matrix, sized (surface vertices x voxels)
        """

        proj_mats = []

        if edge_scale: 
            gm_weights = []
            if len(self.hemi_dict) == 1: 
                gm_weights.append(np.ones(self.spc.size.prod()))
            else: 
                # GM PV can be shared between both hemispheres, so rescale each row of
                # the s2v matrices by the proportion of all voxel-wise GM that belongs
                # to that hemisphere (eg, the GM could be shared 80:20 between the two)
                GM = self._hemi_pvs[0][:,0] + self._hemi_pvs[1][:,0]
                GM[GM == 0] = 1 
                gm_weights.append(self._hemi_pvs[0][:,0] / GM)
                gm_weights.append(self._hemi_pvs[1][:,0] / GM)

            for vox_tri, vtx_tri, weights in zip(self.vox_tri_mats, 
                                                 self.vtx_tri_mats, gm_weights): 
                s2v_mat = assemble_surf2vol(vox_tri, vtx_tri).tocsc()
                s2v_mat.data *= np.take(weights, s2v_mat.indices)
                proj_mats.append(s2v_mat)

            pvs = self.cortex_pvs().reshape(-1,3)
            s2v_mat = sparse.hstack(proj_mats, format="csc")
            s2v_mat.data *= np.take(pvs[:,0], s2v_mat.indices)

        else: 
            for vox_tri, vtx_tri in zip(self.vox_tri_mats, 
                                        self.vtx_tri_mats): 
                s2v_mat = assemble_surf2vol(vox_tri, vtx_tri).tocsc()
                proj_mats.append(s2v_mat)

            # If voxels are shared by both hemispheres, split the relative 
            # weighting according to the GM PV of each hemisphere. This is 
            # not a PV weighting - just deciding which hemi contributes more
            # to the signal. 
            if self.n_hemis == 2: 
                gm_sum = (self._hemi_pvs[0][...,0] + self._hemi_pvs[1][...,0])
                gm_sum[gm_sum > 0] = 1 / gm_sum[gm_sum > 0]
                l_weight = np.clip(self._hemi_pvs[0][...,0] * gm_sum, 0, 1)
                r_weight = np.clip(self._hemi_pvs[1][...,0] * gm_sum, 0, 1)
                weights = [l_weight, r_weight]
            else: 
                weights = [np.ones(proj_mats[0].shape[0])]

            for p,w in zip(proj_mats, weights): 
                p.data *= np.take(w, p.indices)

            s2v_mat = sparse.hstack(proj_mats, format="csc")

        pvs = self.cortex_pvs().reshape(-1,3)
        s2v_mat = sparse.hstack(proj_mats, format="csc")
        if edge_scale:
            s2v_mat.data *= np.take(pvs[:,0], s2v_mat.indices)
        s2v_mat.data = np.clip(s2v_mat.data, 0, 1)
        return s2v_mat  


    def hybrid2vol_matrix(self, edge_scale): 
        """
        Node space to volume projection matrix. 

        Args: 
            edge_scale (bool): downweight signal in voxels that are not 100% brain: 
                               set True for data that scales with PVE (eg perfusion),
                               set False for data that does not (eg time quantities). 

        Returns: 
            sparse CSR matrix, sized (voxels x (surface vertices + voxels))
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
         
        if self._roi_pvs: 

            pvs = self.pvs().reshape(-1,3)
            cpvs = self.cortex_pvs().reshape(-1,3)

            # subcortical GM PVs, stacked across ROIs 
            spvs = np.stack(
                [ r.flatten() for r in self._roi_pvs.values() ], axis=1)

            # The sum of subcortical GM and cortex GM can be greater than 1, 
            # in which case we downweight until they sum to 1 again. This 
            # doesn't apply to voxels with GM sum less than 1 
            sgm_sum = spvs.sum(-1)
            rescale = np.maximum(1, cpvs[...,0] + sgm_sum)

            # mapping from subcortial ROIs to voxels is just the PV matrix 
            spvs = spvs / rescale[:,None]
            r2v_mat = sparse.csr_matrix(spvs)

            # mappings from surface to voxel and from subcortical nodes to voxels 
            # nb subcortical nodes are just voxels themselves!
            s2v_mat = self.surf2vol_matrix(edge_scale=True).tocsc()
            s2v_mat.data /= np.take(rescale, s2v_mat.indices)
            v2v_mat = sparse.dia_matrix((pvs[:,1], 0), 
                shape=2*[self.spc.size.prod()])
            n2v_mat = sparse.hstack((s2v_mat, v2v_mat, r2v_mat), format="csr")

        else: 
            cpvs = self.cortex_pvs().reshape(-1,3)
            s2v_mat = self.surf2vol_matrix(edge_scale=True)
            v2v_mat = sparse.dia_matrix((cpvs[:,1], 0), 
                shape=2*[self.spc.size.prod()])
            n2v_mat = sparse.hstack((s2v_mat, v2v_mat), format="csr")

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

        if vdata.shape[0] != self.spc.size.prod(): 
            raise RuntimeError("vdata must have the same number of rows as" +
                " voxels in the reference ImageSpace")
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
            raise RuntimeError("sdata must have the same number of rows as" +
                " total surface nodes (were one or two hemispheres used?)")
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
            raise RuntimeError("vdata must have the same number of rows as" +
                " nodes (voxels+vertices) in the reference ImageSpace")
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
            raise RuntimeError("ndata must have the same number of rows as" +
                " total nodes in ImageSpace (voxels+vertices)")
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
