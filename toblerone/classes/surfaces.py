"""
Surface-related classes
"""

import itertools
import os.path as op 
import functools
import warnings 
import multiprocessing as mp 
import copy 

import numpy as np 
import nibabel 
import igl
from numpy.lib.arraysetops import isin 
from scipy import sparse
try: 
    import pyvista
    import meshio 
    _VTK_ENABLED = True 
except ImportError as e: 
    warnings.warn("Could not import meshio/pyvista: these are required to"
        " read/write VTK surfaces. VTK requires Python <=3.7 (as of May 2020)")
    _VTK_ENABLED = False 


from .image_space import ImageSpace
from .. import utils, core
from ..utils import NP_FLOAT

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
        if not args: 
            raise RuntimeError("Function must be called with ImageSpace argument")
        if not self._index_space: 
            raise RuntimeError("Surface must be indexed prior to using this function" + 
            "Call surface.index_on()")
        if not self._index_space.derives_from(args[0]):
            raise RuntimeError(
                "Target space is not derived from surface's current index space."+
                "Call surface.index_on with the target space first")
        return func(self, *args)
    return ensured 


class Surface(object):
    """
    Encapsulates a surface's points, triangles and associations data.
    Create either by passing a file path (as below) or use the static class 
    method Surface.manual() to directly pass points and triangles.
    
    Args: 
        path:   path to file (.gii/FS binary/meshio compatible)
        space:  'world' (default) or 'first'; coordinate system of surface
        struct: if in 'first' space, then path to structural image used by FIRST
        name: optional, can be useful for progress bars 
    """

    def __init__(self, path, space='world', struct=None, name=None):

        if not op.exists(path):
            raise RuntimeError("File {} does not exist".format(path))

        if path.endswith('.vtk') and (not _VTK_ENABLED):
            raise NotImplementedError("VTK/meshio must be available to "
                "save VTK surfaces (requires Python <=3.7")    

        if (path.count('first')) and (space == 'world'):
            print("Warning: surface seems to be from FIRST but space was set" +
                " as 'world'. See the docs.")

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

        if space == 'first':
            
            if struct is None: 
                raise RuntimeError("Path to structural image required with FIRST surfs")

            # Convert from FSL coordinates to structural voxel coords 
            struct_spc = ImageSpace(struct)
            ps /= struct_spc.vox_size

            # Flip the X dimension if reqd (according to FSL convention)
            # Remap from 0, 1, ... N-2, N-1 to N-1, N-2, ..., 1, 0
            if np.linalg.det(struct_spc.vox2world) > 0:
                ps[:,0] = ((struct_spc.size[0]) - 1) - ps[:,0]

            # Finally, convert from voxel coords to world mm 
            ps = utils.affineTransformPoints(ps, struct_spc.vox2world)

        self.points = ps.astype(NP_FLOAT)
        self.tris = ts.astype(np.int32)
        self.xProds = None 
        self.voxelised = None 
        self.name = name
        self.assocs = None 
        self._index_space = None 
        self._use_mp = (self.tris.shape[0] > 1000)

    def __repr__(self):

        from textwrap import dedent
        return dedent(f"""\
            Surface with {self.points.shape[0]} points and {self.tris.shape[0]} triangles. 
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
        s.xProds = None 
        s.voxelised = None 
        s.name = name
        s._index_space = None 
        s._use_mp = (s.tris.shape[0] > 1000)
        return s
    

    def save_metric(self, data, path):
        """
        Save vertex-wise data as a .func.gii at path
        """

        if not self.points.shape[0] == data.size:
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

            common = {'Description': 'Surface has been transformed into' +
                      'a reference image space for PV estimation'}

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
            m0.update(common)
            ps = nibabel.gifti.GiftiDataArray(self.points, 
                intent='NIFTI_INTENT_POINTSET', 
                coordsys=nibabel.gifti.GiftiCoordSystem(1,1),  
                datatype='NIFTI_TYPE_FLOAT32', 
                meta=nibabel.gifti.GiftiMetaData.from_dict(m0))

            # Triangles matrix 
            m1 = {'TopologicalType': 'Closed'}
            m1.update(common)
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
        Express PVs in the voxel grid of space. Space must derive from the 
        surface's current index_space. 
        """

        pvs_curr = self.voxelised.astype(NP_FLOAT)
        pvs_curr[self.assocs_keys] = self.fractions
        out = np.zeros(np.prod(space.size), dtype=NP_FLOAT)
        curr_inds, dest_inds = self.reindexing_filter(space)
        out[dest_inds] = pvs_curr[curr_inds]
        return out.reshape(space.size)


    def _estimate_fractions(self, supersampler, cores, ones, desc=''):
        """
        Estimate interior/exterior fractions within current index_space. 

        Args: 
            supersampler: list/tuple of 3 ints, subdivision factor in each dim
            cores: number of multiprocessing cores to use 
            ones: debug tool, write ones in all voxels within assocs_keys 
            desc: for use with progress bar 
        """

        if self._index_space is None: 
            raise RuntimeError("Surface must be indexed first")

        if ones: 
            self.fractions = np.ones(self.assocs_keys.size, dtype=bool) 
        else: 
            self.fractions = \
                core._estimateFractions(self, supersampler, desc, cores)


    def index_on(self, space, struct2ref, cores=mp.cpu_count()):
        """
        Index a surface to an ImageSpace. The space must enclose the surface 
        completely (see ImageSpace.minimal_enclosing()). The surface will be 
        transformed into voxel coordinates for the space, triangle/voxel 
        associations calculated and stored on the surface's assocs 
        attributes, surface normals calculated (triangle cross products) and
        finally voxelised within the space (binary mask of voxels contained in 
        the surface). See also Surface.reindex_for() method. 

        Args: 
            space (ImageSpace): containing the surface
            struct2ref (np.array): transformation into the reference
                space, in world-world mm terms (not FLIRT convention)

        Updates: 
            self.points: converted into voxel coordinates for the space
            self.assocs: sparse CSR bool matrix of size (voxs, tris)
            self.assocs_keys: voxel indices that contain surface 
            self.xProds: triangle cross products, voxel coordinates
        """

        # Smallest possible ImageSpace, based on space, that fully encloses surf 
        encl_space = ImageSpace.minimal_enclosing(self, space, struct2ref)
        if struct2ref is not None: 
            overall = encl_space.world2vox @ struct2ref
        else: 
            overall = encl_space.world2vox

        # Map surface points into this space via struct2ref
        self.applyTransform(overall)
        maxFoV = self.points.max(0).round()
        minFoV = self.points.min(0).round()
        if np.any(minFoV < -1) or np.any(maxFoV > encl_space.size -1):
            raise RuntimeError("Space should be large enough to enclose surface")

        # Update surface attributes
        self._index_space = encl_space 
        self.form_associations(cores)
        self.calculateXprods()
        self.voxelise(cores)


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
        src_space = self._index_space
        offset = src_space.offset 
        size = self._index_space.size 

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
        fltr = np.in1d(src_inds, self.assocs_keys, assume_unique=True)
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
            self.tris[self.assocs[v,:].indices,:])) 
            for v in self.assocs_keys ])
        bridges = self.assocs_keys[group_counts > 1]

        if space is self._index_space:
            return bridges 

        else: 
            src_inds, dest_inds = self.reindexing_filter(space)
            fltr = np.in1d(src_inds, bridges, assume_unique=True)
            return dest_inds[fltr]


    def calculateXprods(self):
        """Calculate and store surface element normals"""

        self.xProds = np.cross(
            self.points[self.tris[:,2],:] - self.points[self.tris[:,0],:], 
            self.points[self.tris[:,1],:] - self.points[self.tris[:,0],:], 
            axis=1)


    def applyTransform(self, transform):
        """Apply affine transformation (4x4 array) to surface coordinates"""

        self.points = (utils.affineTransformPoints(
            self.points, transform).astype(NP_FLOAT))


    def form_associations(self, cores=mp.cpu_count()):
        """
        Identify which triangles of a surface intersect each voxel. This 
        reduces the number of operations that need be performed later. The 
        results will be stored on the surface object (ie, self)

        Returns: 
            None, but associations (sparse CSR matrix of size (voxs, tris)
            and assocs_keys (array of voxel indices containint the surface)
            will be set on the calling object. 
        """

        size = self._index_space.size 

        # Check for negative coordinates: these should have been sripped. 
        if np.round(np.min(self.points)) < 0: 
            raise RuntimeError("formAssociations: negative coordinate found")

        if np.any(np.round(np.max(self.points, axis=0)) >= size): 
            raise RuntimeError("formAssociations: coordinate outside FoV")

        workerFunc = functools.partial(core._formAssociationsWorker, 
            self.tris, self.points, size)

        if (cores > 1) and (self._use_mp):
            chunks = utils._distributeObjects(range(self.tris.shape[0]), cores)
            with mp.Pool(cores) as p:
                worker_assocs = p.map(workerFunc, chunks, chunksize=1)

            assocs = worker_assocs[0]
            for a in worker_assocs[1:]:
                assocs += a 

        else:
            assocs = workerFunc(range(self.tris.shape[0]))

        # Assocs keys is a list of all voxels touched by triangles
        self.assocs = assocs 
        self.assocs_keys = np.flatnonzero(assocs.sum(1).A)


    def rebaseTriangles(self, tri_inds):
        """
        Re-express a patch of a larger surface as a new points and triangle
        matrix pair, indexed from 0. Useful for reducing computational 
        complexity when working with a small patch of a surface where only 
        a few nodes in the points array are required by the triangles matrix. 

        Args: 
            tri_inds: t x 1 list of triangle numbers to rebase. 
        
        Returns: 
            (points, tris) tuple of re-indexed points/tris. 
        """

        points = np.empty((0, 3), dtype=NP_FLOAT)
        tris = np.empty((len(tri_inds), 3), dtype=np.int32)
        pointsLUT = []

        for t in range(len(tri_inds)):
            for v in range(3):

                # For each vertex of each tri, check if we
                # have already processed it in the LUT
                vtx = self.tris[tri_inds[t],v]
                idx = np.argwhere(pointsLUT == vtx)

                # If not in the LUT, then add it and record that
                # as the new position. Write the missing vertex
                # into the local points array
                if not idx.size:
                    pointsLUT.append(vtx)
                    idx = len(pointsLUT) - 1
                    points = np.vstack([points, self.points[vtx,:]])

                # Update the local triangle
                tris[t,v] = idx

        return (points, tris)


    def to_patch(self, vox_idx):
        """
        Return a patch object specific to a voxel given by linear index.
        Look up the triangles intersecting the voxel, and then load and rebase
        the points / surface normals as required. 
        """

        tri_nums = self.assocs[vox_idx,:].indices
        (ps, ts) = self.rebaseTriangles(tri_nums)

        return Patch(ps, ts, self.xProds[tri_nums,:])

    
    def to_patches(self, vox_inds):
        """
        Return the patches for the voxels in voxel indices, flattened into 
        a single set of ps, ts and xprods. 

        If no patches exist for this list of voxels return None.
        """
        
        tri_nums = self.assocs[vox_inds,:].indices

        if tri_nums.size:
            tri_nums = np.unique(tri_nums)
            return Patch(self.points, self.tris[tri_nums,:], 
                self.xProds[tri_nums,:])

        else: 
            return None


    def voxelise(self, cores=mp.cpu_count()):
        """
        Voxelise surface within its current index space. A flat boolean 
        mask will be stored on the calling object as .voxelised. 

        Args:
            cores: number of cores to use 
        """

        if self._index_space is None:
            raise RuntimeError("Surface must be indexed for voxelisation")

        # We will project rays along the longest dimension and split the 
        # grid along the first of the other two. 
        size = self._index_space.size 
        dim = np.argmax(size)
        other_dims = list({0,1,2} - {dim})
        other_size = size[other_dims]
        
        # Get the voxel IJKs for every voxel intersecting the surface
        # Group these according to their ray number, and then strip out
        # repeats to get the minimal set of rays that need to be projected
        # For voxel IJK, and projection along dimension Y, the ray number
        # is given by I,K within a grid of size XZ. Ray D1D2 is an Nx2 array
        # holding the unique ray coords in dimensions XZ. 
        allIJKs = np.array(np.unravel_index(self.assocs_keys, size)).T
        ray_numbers = np.ravel_multi_index(allIJKs[:,other_dims].T, 
            size[other_dims])
        _, uniq_rays = np.unique(ray_numbers, return_index=True)
        rayD1D2 = allIJKs[uniq_rays,:]
        rayD1D2 = rayD1D2[:,other_dims]

        worker = functools.partial(core._voxelise_worker, self)

        if (cores > 1) and (self._use_mp): 

            # Share out the rays and subset of the overall dimension range
            # amongst pool workers
            sub_ranges = utils._distributeObjects(
                range(size[other_dims[0]]), cores)
            subD1D2s = [ 
                rayD1D2[(rayD1D2[:,0] >= sub.start) & (rayD1D2[:,0] < sub.stop),:]
                for sub in sub_ranges ] 

            # Check tasks were shared out completely, zip them together for pool
            assert sum([s.shape[0] for s in subD1D2s]) == rayD1D2.shape[0]
            assert sum([len(s) for s in sub_ranges]) == size[other_dims[0]]
            worker_args = zip(sub_ranges, subD1D2s)

            with mp.Pool(cores) as p: 
                submasks = p.starmap(worker, worker_args)
            mask = np.concatenate(submasks, axis=other_dims[0])

        else: 
            mask = worker(range(size[other_dims[0]]), rayD1D2)

        assert mask.any(), 'no voxels filled'
        self.voxelised = mask.reshape(-1)

    def to_polydata(self):
        """Return pyvista polydata object for this surface"""

        tris = 3 * np.ones((self.tris.shape[0], self.tris.shape[1]+1), np.int32)
        tris[:,1:] = self.tris 
        return pyvista.PolyData(self.points, tris)

    def adjacency_matrix(self):
        """
        Adjacency matrix for the points of this surface, as a scipy sparse
        matrix of size P x P, with 1 denoting a shared edge between points. 
        """

        return igl.adjacency_matrix(self.tris)

    def mesh_laplacian(self, distance_weight=None):
        """
        Mesh Laplacian operator for this surface, as a scipy sparse matrix 
        of size n_points x n_points. Elements on the diagonal are negative 
        and off-diagonal elements are positive. All neighbours are weighted 
        with value 1 (ie, equal weighting ignoring distance). 

        Args: 
            distance_weight (int): apply inverse distance weighting, default 
                None (do not weight, all values are unity), whereas positive
                values will weight elements by 1 / d^n, where d is Euclidean 
                distance between vertices. 

        Returns: 
            sparse CSR matrix
        """

        adj = self.adjacency_matrix().tocsr().astype(NP_FLOAT)

        if distance_weight is not None: 
            if not (isinstance(distance_weight, int) and (distance_weight > 0)):
                raise ValueError("distance_weight must be a positive int")
            
            # Find all connected pairs of vertices in the upper triangle
            # of the adjacency matrix. By symmetry of the adjacency matrix, 
            # we don't need to consider the lower triangle. We use some 
            # sneaky tricks on the CSR indexing of the adjacency matrix
            # to work out which columns are non-zero 
            pairs = []
            for row in range(adj.shape[0]):
                for col in adj[row,row:].indices + row:
                    pairs.append((row,col))

            # Find the distance between connected pairs 
            pairs = np.array(pairs)
            dists = np.linalg.norm(self.points[pairs[:,0],:]
                                    - self.points[pairs[:,1],:], axis=1, ord=2)

            # Apply inverse distance weighting (eg, 1 / d^2), write
            # them back into the matrix (symmetry again)
            weight = 1 / (dists ** distance_weight)
            adj[pairs[:,0], pairs[:,1]] = weight 
            adj[pairs[:,1], pairs[:,0]] = weight 

        # The diagonal is the negative sum of other elements 
        dia = adj.sum(1).A.flatten()
        laplacian = sparse.dia_matrix((dia, 0), shape=(adj.shape))
        laplacian = adj - laplacian

        assert np.abs(laplacian.sum(1)).max() < 1e-6, 'Unweighted laplacian'
        assert utils.is_nsd(laplacian), 'Not negative semi-definite'
        assert utils.is_symmetric(laplacian), 'Not symmetric'
        return laplacian

    def laplace_beltrami(self, area='mayer', cores=mp.cpu_count()):
        """
        Laplace-Beltrami operator for this surface, as a scipy sparse matrix
        of size n_points x n_points. Elements on the diagonal are negative 
        and off-diagonal elements are positive. 

        Args: 
            area (string): 'barycentric', 'voronoi', 'mayer'. Area calculation
                used for mass matrix, default is 'mayer'. 
        """

        if area == 'barycentric':
            M = igl.massmatrix(self.points, self.tris, 
                    igl.MASSMATRIX_TYPE_BARYCENTRIC)
        elif area == 'voronoi':
            M = igl.massmatrix(self.points, self.tris, 
                    igl.MASSMATRIX_TYPE_VORONOI)
        elif area == 'mayer':
            ncores = cores if self._use_mp else 1 
            M = core.vtx_tri_weights(self, ncores)
            M = sparse.diags(np.squeeze(M.sum(1).A))
        else: 
            raise ValueError("Area must be barycentric/voronoi/mayer.")

        L = igl.cotmatrix(self.points, self.tris)
        lbo = (M.power(-1)).dot(L)
        assert (np.abs(lbo.sum(1).A) < 1e-2).all(), 'Unweighted LBO matrix'
        return lbo

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
    Points, triangles and xProds are all inherited from the parent surface. 
    This class should not be directly created but instead instantiated via
    the Surface.to_patch() / to_patches() methods. 
    """

    def __init__(self, points, tris, xProds):
        self.points = points 
        self.tris = tris
        self.xProds = xProds 
        

    def shrink(self, fltr):
        """
        Return a shrunk copy of the patch by applying the logical 
        filter fltr to the calling objects tris and xprods matrices
        """

        return Patch(self.points, self.tris[fltr,:], self.xProds[fltr,:])


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

    def __init__(self, inpath, outpath, side=''):

        self.side = side 
        self.inSurf = Surface(inpath, name=side+'WS') 
        self.outSurf = Surface(outpath, name=side+'PS')
        self.PVs = None 
        return

    @classmethod
    def manual(cls, insurf, outsurf, side):
        """
        Manual hemisphere constructor
        """

        if not (isinstance(insurf, Surface) and isinstance(outsurf, Surface)):
            raise RuntimeError("Initialise with surface objects")

        h = cls.__new__(cls)
        h.inSurf = copy.deepcopy(insurf)
        h.outSurf = copy.deepcopy(outsurf)
        h.PVs = None 
        h.side = side 
        return h 

    @property
    def surfs(self):
        """Iterator over the inner/outer surfaces"""
        return [self.inSurf, self.outSurf]

    def surf_dict(self):
        """Return surfs as dict with appropriate keys (eg LPS)"""
        return {self.side + 'WS': self.inSurf, 
            self.side+'PS': self.outSurf}

    
    def apply_transform(self, mat):
        [ s.applyTransform(mat) for s in self.surfs ]
