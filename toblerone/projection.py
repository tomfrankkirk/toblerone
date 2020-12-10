"""
Toblerone surface-volume projection functions
"""

import multiprocessing as mp 
import copy
from toblerone.utils import sparse_normalise 
import warnings

import numpy as np 
from scipy import sparse 

from toblerone import utils 
from toblerone.pvestimation import estimators
from toblerone.classes import Hemisphere, Surface
from toblerone.core import vtx_tri_weights, vox_tri_weights

SIDES = ['L', 'R']

# TODO: save projector in hdf5 format?

class Projector(object):
    """
    Use to perform projection between volume, surface and node space. 
    Creating a projector object may take some time whilst the consituent 
    matrices are prepared; once created any of the individual projections
    may be calculated directly from the object. 

    Args: 
        hemispheres: single or list of two (L/R) Hemisphere objects 
        spc: ImageSpace to project from/to 
        factor: voxel subdivision factor (default 10)
        cores: number of processor cores to use (default max)
    """

    def __init__(self, hemispheres, spc, factor=10, cores=mp.cpu_count(), 
                 ones=False):

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
            
        self.hemis = { h.side: copy.deepcopy(h) for h in hemispheres }
        self.spc = spc 
        self.pvs = [] 
        self.__vox_tri_mats = [] 
        self.__vtx_tri_mats = []
        ncores = cores if hemispheres[0].inSurf._use_mp else 1 

        for hemi in self.iter_hemis: 

            # If PV estimates are not present, then compute from scratch 
            if hasattr(hemi, 'pvs'): 
                self.pvs.append(hemi.pvs.reshape(-1,3))
            else: 
                supersample = np.ceil(spc.vox_size).astype(np.int8) 
                pvs = estimators._cortex(hemi, spc, np.eye(4), supersample, 
                    ncores, ones)
                self.pvs.append(pvs.reshape(-1,3))

            # Calculate the constituent matrices for projection with each hemi 
            midsurf = hemi.midsurface()
            vox_tri = vox_tri_weights(*hemi.surfs, spc, factor, ncores, ones)
            vtx_tri = vtx_tri_weights(midsurf, ncores)
            self.__vox_tri_mats.append(vox_tri)
            self.__vtx_tri_mats.append(vtx_tri)

    
    @property
    def iter_hemis(self):
        """Iterator over hemispheres of projector, in L/R order"""
        for s in SIDES:
            if s in self.hemis:
                yield self.hemis[s]

    @property
    def n_hemis(self):
        return len(self.hemis)


    # Direct access to the underlying surfaces via keys LPS, RWS etc. 
    def __getitem__(self, surf_key):
        side = surf_key[0]
        return self.hemis[side].surf_dict[surf_key]


    # Calculation of the projection matrices involves rescaling the constituent
    # matrices, so these proerties return copies to keep the originals private
    @property
    def vox_tri_mats(self): 
        return copy.deepcopy(self.__vox_tri_mats)


    @property
    def vtx_tri_mats(self): 
        return copy.deepcopy(self.__vtx_tri_mats)


    @property
    def n_surf_points(self):
        return sum([ h.n_points for h in self.iter_hemis ])


    def adjacency_matrix(self, distance_weight=0):
        """
        Overall adjacency matrix for all surface vertices of projector. 
        If there are two hemispheres present, the matrix indices will 
        be arranged L,R.  

        Args: 
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all egdes are unity), whereas positive
                values will weight edges by 1 / d^n, where d is geometric 
                distance between vertices. 
        """ 

        mats = [ h.adjacency_matrix(distance_weight) for h in self.iter_hemis ]
        return sparse.block_diag(mats, format="csr")


    def mesh_laplacian(self, distance_weight=0):
        """
        Overall mesh Laplacian matrix for all surface vertices of projector. 
        If there are two hemispheres present, the matrix indices will be 
        arranged L/R. 

        Args: 
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all egdes are unity), whereas positive
                values will weight edges by 1 / d^n, where d is geometric 
                distance between vertices. 
        """

        mats = [ h.mesh_laplacian(distance_weight) for h in self.iter_hemis ]
        return sparse.block_diag(mats, format="csr")


    def flat_pvs(self):
        """
        Combine PV estimates from one or both hemispheres (if available) into 
        single map. 

        Returns: 
            (v x 3) array of PVs, columns arranged GM, WM, non-brain
        """
        if len(self.pvs) > 1:
            # Combine PV estimates from each hemisphere into single map 
            pvs = np.zeros((self.spc.size.prod(), 3))
            pvs[:,0] = np.minimum(1.0, self.pvs[0][:,0] + self.pvs[1][:,0])
            pvs[:,1] = np.minimum(1.0 - pvs[:,0], self.pvs[0][:,1] + self.pvs[1][:,1])
            pvs[:,2] = 1.0 - pvs[:,0:2].sum(1)
            return pvs 
        else: 
            return self.pvs[0]


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
            sparse matrix sized (surface vertices x voxels). Surface vertices 
                are arranged L then R. 
        """

        proj_mats = [ assemble_vol2surf(vox_tri, vtx_tri) 
            for vox_tri, vtx_tri in zip(self.vox_tri_mats, self.vtx_tri_mats) ]
        v2s_mat = sparse.vstack(proj_mats, format="csr")

        if edge_scale: 
            brain_pv = self.flat_pvs()[:,:2].sum(1)
            brain = (brain_pv > 1e-3)
            upweight = np.ones(brain_pv.shape)
            upweight[brain] = 1 / brain_pv[brain]
            v2s_mat.data *= np.take(upweight, v2s_mat.indices)

        return v2s_mat 


    def vol2node_matrix(self, edge_scale): 
        """
        Volume to node space projection matrix. 

        Args: 
            edge_scale (bool):  upweight signal from voxels that are not
                                100% brain to account for 'missing' signal. 
                                Set True for quantities that scale with PVE 
                                (eg perfusion), set False otherwise 
                                (eg time quantities)

        Returns: 
            sparse matrix sized ((surface vertices + voxels) x voxels)
        """

        v2s_mat = self.vol2surf_matrix(edge_scale)
        v2v_mat = sparse.eye(self.spc.size.prod())
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
            sparse matrix sized (surface vertices x voxels)
        """

        proj_mats = []

        if edge_scale: 
            gm_weights = []
            if len(self.hemis) == 1: 
                gm_weights.append(np.ones(self.spc.size.prod()))
            else: 
                # GM PV can be shared between both hemispheres, so rescale each row of
                # the s2v matrices by the proportion of all voxel-wise GM that belongs
                # to that hemisphere (eg, the GM could be shared 80:20 between the two)
                GM = self.pvs[0][:,0] + self.pvs[1][:,0]
                GM[GM == 0] = 1 
                gm_weights.append(self.pvs[0][:,0] / GM)
                gm_weights.append(self.pvs[1][:,0] / GM)

            for vox_tri, vtx_tri, weights in zip(self.vox_tri_mats, 
                                                 self.vtx_tri_mats, gm_weights): 
                s2v_mat = assemble_surf2vol(vox_tri, vtx_tri).tocsc()
                s2v_mat.data *= np.take(weights, s2v_mat.indices)
                proj_mats.append(s2v_mat)

            pvs = self.flat_pvs()
            s2v_mat = sparse.hstack(proj_mats, format="csc")
            s2v_mat.data *= np.take(pvs[:,0], s2v_mat.indices)

        else: 
            for vox_tri, vtx_tri in zip(self.vox_tri_mats, 
                                        self.vtx_tri_mats): 
                s2v_mat = assemble_surf2vol(vox_tri, vtx_tri).tocsc()
                proj_mats.append(s2v_mat)

            s2v_mat = sparse.hstack(proj_mats, format="csc")

        pvs = self.flat_pvs()
        s2v_mat = sparse.hstack(proj_mats, format="csc")
        if edge_scale:
            s2v_mat.data *= np.take(pvs[:,0], s2v_mat.indices)
        return s2v_mat  


    def node2vol_matrix(self, edge_scale): 
        """
        Node space to volume projection matrix. 

        Args: 
            edge_scale (bool): downweight signal in voxels that are not 100% brain: 
                               set True for data that scales with PVE (eg perfusion),
                               set False for data that does not (eg time quantities). 

        Returns: 
            sparse CSR matrix sized (voxels x (surface vertices + voxels))
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

        pvs = self.flat_pvs()
        s2v_mat = self.surf2vol_matrix(edge_scale=True)
        v2v_mat = sparse.dia_matrix((pvs[:,1], 0), 
            shape=2*[self.spc.size.prod()])
        n2v_mat = sparse.hstack((s2v_mat, v2v_mat), format="csr")

        if not edge_scale: 
            n2v_mat = sparse_normalise(n2v_mat, 1)
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


    def vol2node(self, vdata, edge_scale):
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
            np.array, sized (n_vertices + n_voxels) in first dimension.
                Surface vertices are arranged L then R. 
        """

        v2n_mat = self.vol2node_matrix(edge_scale)
        if vdata.shape[0] != v2n_mat.shape[1]: 
            raise RuntimeError("vdata must have the same number of rows as" +
                " nodes (voxels+vertices) in the reference ImageSpace")
        return v2n_mat.dot(vdata)


    def node2vol(self, ndata, edge_scale):
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

        n2v_mat = self.node2vol_matrix(edge_scale)
        if ndata.shape[0] != n2v_mat.shape[1]: 
            raise RuntimeError("ndata must have the same number of rows as" +
                " total nodes in ImageSpace (voxels+vertices)")
        return n2v_mat.dot(ndata)
        

def assemble_vol2surf(vox_tri, vtx_tri):
    """
    Combine (w/ normalisation) the vox_tri and vtx_tri matrices into vol2surf.
    """
    
    # Ensure each triangle's voxel weights sum to 1 
    # Ensure each vertices' triangle weights sum to 1 
    vox2tri = utils.sparse_normalise(vox_tri, 0).T
    tri2vtx = utils.sparse_normalise(vtx_tri, 1)
    vol2vtx = tri2vtx @ vox2tri
    return utils.sparse_normalise(vol2vtx, 1)


def assemble_surf2vol(vox_tri, vtx_tri):
    """
    Combine (w/ normalisation) the vox_tri and vtx_tri matrices into surf2vol.
    """

    # Ensure each triangle's vertex weights sum to 1 
    # Ensure each voxel's triangle weights sum to 1
    vtx2tri = utils.sparse_normalise(vtx_tri, 0).T
    tri2vox = utils.sparse_normalise(vox_tri, 1)
    vtx2vox = tri2vox @ vtx2tri
    return utils.sparse_normalise(vtx2vox, 1)
