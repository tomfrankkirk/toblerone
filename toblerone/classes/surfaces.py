"""
Surface-related classes
"""

import itertools
import os.path as op 
from types import SimpleNamespace
import functools
import warnings 
import multiprocessing as mp 
import copy 

import numpy as np 
import nibabel 
from scipy import sparse

try: 
    import pyvista
    import meshio 
    _VTK_ENABLED = True 
except ImportError as e: 
    warnings.warn("Could not import meshio/pyvista: these are required to"
        " read/write VTK surfaces. VTK requires Python <=3.7 (as of May 2020)")
    _VTK_ENABLED = False 

from .image_space import ImageSpace, BaseSpace
from .. import utils, core
from ..utils import NP_FLOAT, calc_midsurf, is_symmetric

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
            raise RuntimeError("Surface must be indexed prior to using this function" + 
            "Call surface.index_on()")
        if not self.indexed.space.derives_from(args[0]):
            raise RuntimeError(
                "Target space is not derived from surface's current index space."+
                "Call surface.index_on with the target space first")
        return func(self, *args)
    return ensured 


@utils.cascade_attributes
def assert_indexed(func): 
    """
    Decorator to ensure surface has been indexed prior to calling function. 
    """
    def asserted(self, *args, **kwargs):
        if self.indexed is None: 
            raise RuntimeError("Surface must be indexed before calling ",
                                func.__name__)
        return func(self, *args, **kwargs)
    return asserted


class Surface(object):
    """
    Encapsulates a surface's points, triangles and associations data.
    Create either by passing a file path (as below) or use the static class 
    method Surface.manual() to directly pass points and triangles.
    
    Args: 
        path:   path to file (.gii/FS binary/meshio compatible)
        coords:  'world' (default) or 'fsl'; coordinate system of surface
        struct: if in 'fsl' coords, then path to structural image used by FIRST
        name: optional, can be useful for progress bars 
    """

    def __init__(self, path, name=None):

        if not op.exists(path):
            raise RuntimeError("File {} does not exist".format(path))

        if path.endswith('.vtk') and (not _VTK_ENABLED):
            raise NotImplementedError("VTK/meshio must be available to "
                "save VTK surfaces (requires Python <=3.7")    

        surfExt = op.splitext(path)[-1]
        if surfExt == '.gii':
            gft = nibabel.load(path).darrays
            ps, ts = gft[0].data, gft[1].data

        else: 
            try: 
                ps, ts, meta = nibabel.freesurfer.io.read_geometry(path, 
                    read_metadata=True)
                if not 'cras' in meta:
                    print('Warning: Could not load C_ras from surface', path)
                    print('If true C_ras is non-zero then estimates will be inaccurate')
                else:
                    ps += meta['cras']

            except Exception as e: 
                try:
                    mesh = meshio.read(path)
                    ps = np.array(mesh.points)
                    ts = mesh.cells[0].data
                    
                except Exception as e: 
                    try: 
                        poly = pyvista.read(path)
                        ps = np.array(poly.points)
                        ts = poly.faces.reshape(-1,4)[:,1:]

                    except Exception as e: 
                        print("Could not load surface as GIFTI, FS binary or"
                            " meshio/pyvista format")
                        raise e 

        if ps.shape[1] != 3: 
            raise RuntimeError("Points matrices should be p x 3")

        if ts.shape[1] != 3: 
            raise RuntimeError("Triangles matrices should be t x 3")

        if (np.max(ts) != ps.shape[0]-1) or (np.min(ts) != 0):
            raise RuntimeError("Incorrect points/triangle indexing")

        self.points = ps.astype(NP_FLOAT)
        self.tris = ts.astype(np.int32)
        self.indexed = None 
        self.name = name
        self._use_mp = (self.tris.shape[0] > MP_THRESHOLD)

    def __repr__(self):

        from textwrap import dedent
        return dedent(f"""\
            Surface with {self.n_points} points and {self.tris.shape[0]} triangles. 
            min (X,Y,Z):  {self.points.min(0)}
            mean (X,Y,Z): {self.points.mean(0)}
            max (X,Y,Z):  {self.points.max(0)}
            """)


    @classmethod
    def manual(cls, ps, ts, name=None):
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
        s._use_mp = (s.tris.shape[0] > MP_THRESHOLD)
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

        if not path.endswith('.func.gii'):
            print("appending .func.gii extension")
            path += '.func.gii'

        gii = nibabel.GiftiImage()
        gii.add_gifti_data_array(
            nibabel.gifti.GiftiDataArray(data.astype(NP_FLOAT)))
        nibabel.save(gii, path)


    def save(self, path):
        """
        Save surface as .surf.gii (default), .vtk or .white/.pial at path.
        """

        if path.endswith('.vtk'):
            if not _VTK_ENABLED:
                raise NotImplementedError("VTK/meshio must be available to "
                    "save VTK surfaces (requires Python 3.7")
            mesh = meshio.Mesh(self.points, [ ("triangle", t[None,:]) 
                                                for t in self.tris ])
            mesh.write(path)

        elif path.count('.gii'): 
            if not path.endswith('.surf.gii'):
                if path.endswith('.gii'):
                    path.replace('.gii', '.surf.gii')
                else: 
                    path += '.surf.gii'
            
            if self.name is None: 
                self.name = 'Other'

            m0 = {'GeometricType': 'Anatomical'}

            if self.name in ['LWS', 'LPS', 'RWS', 'RPS']:
                cortexdict = {
                    sd + sf + 'S': {
                        'AnatomicalStructurePrimary': 
                            'CortexLeft' if sd == 'L' else 'CortexRight', 
                        'AnatomicalStructureSecondary':
                            'GrayWhite' if sd == 'W' else 'Pial'
                    }
                    for (sd, sf) in itertools.product(['L', 'R'], ['P', 'W'])
                }
                m0.update(cortexdict[self.name])   
            
            else:
                m0.update({'AnatomicalStructurePrimary': self.name})
            
            # Points matrix
            # 1 corresponds to NIFTI_XFORM_SCANNER_ANAT
            ps = nibabel.gifti.GiftiDataArray(self.points, 
                intent='NIFTI_INTENT_POINTSET', 
                coordsys=nibabel.gifti.GiftiCoordSystem(1,1),  
                datatype='NIFTI_TYPE_FLOAT32', 
                meta=nibabel.gifti.GiftiMetaData.from_dict(m0))

            # Triangles matrix 
            m1 = {'TopologicalType': 'Closed'}
            ts = nibabel.gifti.GiftiDataArray(self.tris, 
                intent='NIFTI_INTENT_TRIANGLE', 
                coordsys=nibabel.gifti.GiftiCoordSystem(0,0), 
                datatype='NIFTI_TYPE_INT32', 
                meta=nibabel.gifti.GiftiMetaData.from_dict(m1))

            img = nibabel.gifti.GiftiImage(darrays=[ps,ts])
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
        curr_inds, dest_inds = self.reindexing_filter(space)
        out[dest_inds] = pvs_curr[curr_inds]
        return out.reshape(space.size)


    @assert_indexed
    def _estimate_fractions(self, supersampler, cores, ones, desc=''):
        """
        Estimate interior/exterior fractions within current index_space. 

        Args: 
            supersampler: list/tuple of 3 ints, subdivision factor in each dim
            cores: number of multiprocessing cores to use 
            ones: debug tool, write ones in all voxels within assocs_keys 
            desc: for use with progress bar 
        """
        
        cores = cores if self._use_mp else 1
        if ones: 
            self.fractions = np.ones(self.indexed.assocs_keys.size, dtype=bool) 
        else: 
            self.fractions = core._estimateFractions(self, supersampler, 
                                                        desc, cores)


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

        Updates: 
            self.indexed (SimpleNamespace): with the following attributes
                indexed.points_vox:  converted into voxel coordinates for the space
                indexed.assocs:      sparse CSR bool matrix of size (voxs, tris)
                indexed.assocs_keys: voxel indices that contain surface 
                indexed.xprods:      triangle cross products, voxel coordinates
                indexed.space:       the minimal enclosing ImageSpace used for indexing
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

        # Map surface points into this space
        points_vox = utils.affine_transform(self.points, encl_space.world2vox)
        assert utils.space_encloses_surface(encl_space, points_vox)

        # Calculate associations and cross products
        cores = cores if self._use_mp else 1 
        assocs, assocs_keys = core.form_associations(points_vox, self.tris, 
                                                        encl_space, cores)
        xprods = utils.calculateXprods(points_vox, self.tris)

        # All the results of indexing are stored using the namedtuple 
        # structure under the attribue self.indexed
        self.indexed = SimpleNamespace(
            points_vox=points_vox,
            space=encl_space, 
            assocs=assocs,
            assocs_keys=assocs_keys, 
            xprods=xprods,
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
        other_dims = list({0,1,2} - {dim})
        
        # Get the voxel IJKs for every voxel intersecting the surface
        # Group these according to their ray number, and then strip out
        # repeats to get the minimal set of rays that need to be projected
        # For voxel IJK, and projection along dimension Y, the ray number
        # is given by I,K within a grid of size XZ. Ray D1D2 is an Nx2 array
        # holding the unique ray coords in dimensions XZ. 
        allIJKs = np.array(np.unravel_index(self.indexed.assocs_keys, size)).T
        ray_numbers = np.ravel_multi_index(allIJKs[:,other_dims].T, 
            size[other_dims])
        _, uniq_rays = np.unique(ray_numbers, return_index=True)
        rayD1D2 = allIJKs[uniq_rays,:]
        rayD1D2 = rayD1D2[:,other_dims]

        worker = functools.partial(core._voxelise_worker, self)
        cores = cores if self._use_mp else 1 
        if cores > 1: 

            # Share out the rays and subset of the overall dimension range
            # amongst pool workers
            sub_ranges = utils._distributeObjects(range(size[other_dims[0]]), cores)
            subD1D2s = [ 
                rayD1D2[(rayD1D2[:,0] >= sub.start) & (rayD1D2[:,0] < sub.stop),:]
                for sub in sub_ranges ] 

            # Check tasks were shared out completely, zip them together for pool
            assert sum([s.shape[0] for s in subD1D2s]) == rayD1D2.shape[0]
            assert sum([len(s) for s in sub_ranges]) == size[other_dims[0]]
            worker_args = zip(sub_ranges, subD1D2s)

            with mp.Pool(cores) as p: 
                submasks = p.starmap(worker, worker_args)
            src_mask = np.concatenate(submasks, axis=other_dims[0])

        else: 
            src_mask = worker(range(size[other_dims[0]]), rayD1D2)

        assert src_mask.any(), 'no voxels filled'
        src_mask = src_mask.flatten()

        # If the space provided is not the same as the space used for indexing 
        # (because we reduced the FoV when indexing), we need to map results 
        # from the indexing space into the destination space. See index_on()
        # for more info. 
        # Else, we can just return the mask as-is. 
        if space != self.indexed.space: 
            src_inds, dest_inds = self.reindexing_filter(space, False)
            mask = np.zeros(space.size.prod(), dtype=bool)
            mask[dest_inds] = src_mask[src_inds]    
        else: 
            mask = src_mask 

        return src_mask 
    

    @ensure_derived_space
    def reindexing_filter(self, dest_space, as_bool=False):
        """
        Filter of voxels in the current index space that lie within 
        dest_space. Use for extracting PV estimates from index space back to
        the space from which the index space derives. NB dest_space must 
        derive from the surface's current index_space 

        Args: 
            dest_space: ImageSpace from which current index_space derives
            as_bool: output results as logical filters instead of indices
                (note they will be of different size in this case)

        Returns: 
            (src_inds, dest_inds) arrays of equal length, flat indices into 
            arrays of size index_space.size and dest_space.size respectively, 
            mapping voxels from source to corresponding destination positions 
        """

        # Get the offset and size of the current index space 
        src_space = self.indexed.space
        offset = src_space.offset 
        size = src_space.size 

        # List voxel indices in the current index space 
        # List corresponding voxel coordinates in the destination space 
        # curr2dest_fltr selects voxel indices from the current space that 
        # are also contained within the destination space 
        inds_in_src = np.arange(np.prod(size))
        voxs_in_src = np.array(np.unravel_index(inds_in_src, size)).T
        voxs_in_dest = voxs_in_src - offset
        fltr = np.logical_and(np.all(voxs_in_dest > - 1, 1), 
            np.all(voxs_in_dest < dest_space.size, 1))
        
        src_inds = np.ravel_multi_index(voxs_in_src[fltr,:].T, size)
        dest_inds = np.ravel_multi_index(voxs_in_dest[fltr,:].T, 
            dest_space.size)

        if as_bool: 
            src_fltr = np.zeros(np.prod(size), dtype=bool)
            src_fltr[src_inds] = 1 
            dest_fltr = np.zeros(np.prod(dest_space.size), dtype=bool)
            dest_fltr[dest_inds] = 1
            return (src_fltr, dest_fltr)

        return src_inds, dest_inds


    @ensure_derived_space
    def reindex_LUT(self, space):
        """Return a copy of LUT indices expressed in another space"""

        src_inds, dest_inds = self.reindexing_filter(space)
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

        group_counts = np.array([len(core._separatePointClouds(
            self.tris[self.indexed.assocs[v,:].indices,:])) 
            for v in self.indexed.assocs_keys ])
        bridges = self.indexed.assocs_keys[group_counts > 1]

        if space == self.indexed.space:
            return bridges 

        else: 
            src_inds, dest_inds = self.reindexing_filter(space)
            fltr = np.in1d(src_inds, bridges, assume_unique=True)
            return dest_inds[fltr]


    def transform(self, transform):
        """Apply affine transformation to surface vertices, return new Surface"""

        points = utils.affine_transform(
                    self.points, transform).astype(NP_FLOAT)
        return Surface.manual(points, self.tris, self.name)


    def to_patch(self, vox_idx):
        """
        Return a patch object specific to a voxel given by linear index.
        Look up the triangles intersecting the voxel, and then load and rebase
        the points / surface normals as required. 
        """

        tri_nums = self.indexed.assocs[vox_idx,:].indices
        (ps, ts) = utils.rebase_triangles(self.indexed.points_vox, self.tris, tri_nums)
        return Patch(ps, ts, self.indexed.xprods[tri_nums,:])

    
    def to_patches(self, vox_inds):
        """
        Return the patches for the voxels in voxel indices, flattened into 
        a single set of ps, ts and xprods. 

        If no patches exist for this list of voxels return None.
        """
        
        tri_nums = self.indexed.assocs[vox_inds,:].indices

        if tri_nums.size:
            tri_nums = np.unique(tri_nums)
            return Patch(self.indexed.points_vox, self.tris[tri_nums,:], 
                self.indexed.xprods[tri_nums,:])

        else: 
            return None


    def to_polydata(self):
        """Return pyvista polydata object for this surface"""

        tris = 3 * np.ones((self.tris.shape[0], self.tris.shape[1]+1), np.int32)
        tris[:,1:] = self.tris 
        return pyvista.PolyData(self.points, tris)


    def adjacency_matrix(self, distance_weight=0):
        """
        Adjacency matrix for the points of this surface, as a scipy sparse
        matrix of size P x P, with 1 denoting a shared edge between points. 
        
        Args:
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all egdes are unity), whereas positive
                values will weight edges by 1 / d^n, where d is geometric 
                distance between vertices. 
        """

        if not (isinstance(distance_weight, int) and (distance_weight >= 0)):
            raise ValueError("distance_weight must be int >= 0")

        edge_pairs = np.array([
            np.concatenate((self.tris[:,0], self.tris[:,1], self.tris[:,2])), 
            np.concatenate((self.tris[:,1], self.tris[:,2], self.tris[:,0]))
        ])
        row, col = edge_pairs

        if distance_weight > 0: 
            weights = np.linalg.norm(self.points[row,:] - self.points[col,:], 
                                        ord=2, axis=-1).flatten()
            weights = 1 / (weights ** distance_weight)
        else:
            weights = np.ones(row.size, dtype=np.int32)

        adj = sparse.coo_matrix((weights, (row, col)), shape=(self.n_points, self.n_points))

        if distance_weight == 0: 
            assert adj.max() == 1
        assert is_symmetric(adj), 'Adjacency should be symmetric'

        return adj.tocsr()


    def mesh_laplacian(self, distance_weight=0):
        """
        Mesh Laplacian operator for this surface, as a scipy sparse matrix 
        of size n_points x n_points. Elements on the diagonal are negative 
        and off-diagonal elements are positive. All neighbours are weighted 
        with value 1 (ie, equal weighting ignoring distance). 

        Args: 
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all values are unity), whereas positive
                values will weight elements by 1 / d^n, where d is geometric 
                distance between vertices. 

        Returns: 
            sparse CSR matrix
        """

        # The diagonal is the negative sum of other elements 
        adj = self.adjacency_matrix(distance_weight)
        adj = np.around(adj, 12)
        dia = adj.sum(1).A.flatten()
        laplacian = sparse.dia_matrix((dia, 0), shape=(adj.shape), dtype=np.float32)
        laplacian = adj - laplacian

        assert np.abs(laplacian.sum(1)).max() < 1e-4, 'Unweighted laplacian'
        assert utils.is_nsd(laplacian), 'Not negative semi-definite'
        assert utils.is_symmetric(laplacian), 'Not symmetric'
        return laplacian


    def discriminated_laplacian(self, spc, distance_weight=0, in_weight=100): 

        if not (isinstance(distance_weight, int) and (distance_weight >= 0)):
            raise ValueError("distance_weight must be int >= 0")

        # Some vertices may not be in the voxel grid, so give them a placeholder
        # voxel index of -1 
        vtx_vox = self.transform(spc.world2vox).points.round().astype(np.int32)
        vtx_in_spc = ((vtx_vox >= 0) & (vtx_vox < spc.size)).all(-1)
        vtx_vox_idx = -1 * np.ones(vtx_in_spc.shape[0], dtype=np.int32)
        vtx_vox_idx[vtx_in_spc] = np.ravel_multi_index(vtx_vox[vtx_in_spc,:].T, spc.size)

        # All the edge vectors of the surface 
        edge_pairs = np.array([
            np.concatenate((self.tris[:,0], self.tris[:,1], self.tris[:,2])), 
            np.concatenate((self.tris[:,1], self.tris[:,2], self.tris[:,0]))
        ])
        start, end = edge_pairs

        # Inverse distance weighting of power n 
        if distance_weight > 0: 
            dists = np.linalg.norm(self.points[start,:] - self.points[end,:], 
                                        ord=2, axis=-1).flatten()
            dists = 1 / (dists ** distance_weight)
        else:
            dists = np.ones(start.size, dtype=np.int32)

        # If the start/end of an edge is the same voxel, then full weight 
        # if different voxels, downweight it
        in_weights = in_weight * np.ones_like(dists)
        in_weights[vtx_vox_idx[start] != vtx_vox_idx[end]] = 1 
        weights = dists * in_weights

        # Knock out edges where at least one vertex isn't in the voxel grid 
        fltr = (vtx_vox_idx[start] >= 0) & (vtx_vox_idx[end] >= 0)
        start = start[fltr]
        end = end[fltr]
        weights = weights[fltr]
        adj = sparse.coo_matrix((weights, (start, end)), 
                shape=(self.n_points, self.n_points))

        assert is_symmetric(adj), 'Adjacency should be symmetric'

        # The diagonal is the negative sum of other elements 
        adj = np.around(adj, 9)
        dia = adj.sum(1).A.flatten()
        laplacian = sparse.dia_matrix((dia, 0), shape=(adj.shape), dtype=np.float32)
        laplacian = adj - laplacian

        assert np.abs(laplacian.sum(1)).max() < 1e-3, 'Unweighted laplacian'
        # FIXME: the eigs test required by nsd crashes for in_weight > 10 
        # assert utils.is_nsd(laplacian), 'Not negative semi-definite'
        assert utils.is_symmetric(laplacian), 'Not symmetric'

        return laplacian.tocsr()


    # def laplace_beltrami(self, cores=mp.cpu_count()):
    #     """
    #     Laplace-Beltrami operator for this surface, as a scipy sparse matrix
    #     of size n_points x n_points. Elements on the diagonal are negative 
    #     and off-diagonal elements are positive. 

    #     Areas are calculated according to the definition of A_mixed in "Discrete 
    #     Differential-Geometry Operators for Triangulated 2-Manifolds", M. Meyer, 
    #     M. Desbrun, P. Schroder, A.H. Barr.

    #     Args: 
    #         cores (int): CPU cores to use.

    #     Returns: 
    #         sparse matrix 
    #     """

    #     ncores = cores if self._use_mp else 1 
    #     M = core.vtx_tri_weights(self, ncores)
    #     M = sparse.diags(np.squeeze(M.sum(1).A))

    #     L = igl.cotmatrix(self.points, self.tris)
    #     lbo = (M.power(-1)).dot(L)
    #     assert (np.abs(lbo.sum(1).A) < 1e-2).all(), 'Unweighted LBO matrix'
    #     return lbo


    def edges(self):
        """
        Edge matrix, sized as follows (tris, 3, 3), where the second dimension
        contains the edges defined as (v1 - v0), (v2 - v0), (v2 - v1), and 
        the final dimension contains the edge components in XYZ.  
        """
        edge_defns = [ list(e) for e in core.TRI_EDGE_INDEXING ]
        edges = np.stack([
            self.points[self.tris[:,e[0]],:] - self.points[self.tris[:,e[1]],:] 
            for e in edge_defns
        ], axis=1)

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

        return Patch(self.points, self.tris[fltr,:], self.xprods[fltr,:])


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

    def __init__(self, insurf, outsurf, side):

        if not side in ['L', 'R']:
            raise ValueError("Side must be either 'L' or 'R'")
        self.side = side 

        # Create surfaces from path or make our own copy 
        if not isinstance(insurf, Surface):
            self.inSurf = Surface(insurf, name=side+'WS') 
        else: 
            self.inSurf = copy.deepcopy(insurf)
        if not isinstance(insurf, Surface):
            self.outSurf = Surface(outsurf, name=side+'PS')
        else: 
            self.outSurf = copy.deepcopy(outsurf)    
            
        if (self.inSurf.tris != self.outSurf.tris).any(): 
            raise ValueError("Both surfaces must have same triangles array")

        self.PVs = None 
        return

    @property
    def surfs(self):
        """Iterator over the inner/outer surfaces"""

        return [self.inSurf, self.outSurf]

    @property
    def surf_dict(self):
        """Return surfs as dict with appropriate keys (eg LPS)"""

        return { self.side + 'WS': self.inSurf, 
                 self.side + 'PS': self.outSurf}

    def transform(self, mat):
        """
        Apply affine transformation to each surface. Returns a new Hemisphre.
        """

        surfs = [ s.transform(mat) for s in self.surfs ]
        return Hemisphere(*surfs, self.side)

    def midsurface(self):
        """Midsurface between inner and outer cortex"""
        return calc_midsurf(self.inSurf, self.outSurf)

    def adjacency_matrix(self, distance_weight=0):
        """
        Adjacency matrix of any cortical surface (they necessarily share
        the same triagulation, which is checked during initialisation). 

        Args:
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all egdes are unity), whereas positive
                values will weight edges by 1 / d^n, where d is geometric 
                distance between vertices. 
        """

        return self.inSurf.adjacency_matrix(distance_weight)

    @property
    def n_points(self):
        """Number of vertices on either cortical surface"""
        return self.inSurf.n_points

    def mesh_laplacian(self, distance_weight=0):
        """
        Mesh Laplacian on cortical midsurface. 

        Args: 
            distance_weight (int): apply inverse distance weighting, default 
                0 (do not weight, all egdes are unity), whereas positive
                values will weight edges by 1 / d^n, where d is geometric 
                distance between vertices. 
        """

        return self.midsurface().mesh_laplacian(distance_weight)

    def discriminated_laplacian(self, spc, distance_weight=0, in_weight=100):
        return self.midsurface().discriminated_laplacian(spc, distance_weight, in_weight)
