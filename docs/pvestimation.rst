.. _pvestimation-index:

Partial volume estimation
==============================

Surface-based partial volumes are calculated as the volume of intersection between a voxel grid and a closed surface that is contained within it. Some voxels will partially intersect the voxel; the interior fractions is the *partial volume*. The voxel grid of interest is referred to as the *reference grid*. 

Partial volume estimation is performed on a surface-by-surface basis. For a structure defined by a single surface (eg thalamus), the function :func:`toblerone.pvestimation.structure` will return a single array containing values in the range [0,1] where 1 denotes that a voxel is fully contained within the surface. 

For a structure defined by multiple surfaces (cortex), partial volumes can be estimated by performing estimation for each individual surface in turn and then combining the results. The function :func:`toblerone.pvestimation.cortex` does this with the pial and white surfaces, and then takes the voxel-wise difference between the two as grey matter. Currently the cortex is the only mulit-surface structure supported, though others could be processed by using :func:`toblerone.pvestimation.structure` on the individual surfaces and combining the results as appropriate. 

PV estimation functions 
----------------------------------

Toblerone estimates PVs across the brain using surface segmentations (for example, those from FreeSurfer and FSL FIRST). The following functions are available in :mod:`toblerone.pvestimation`: 

   * :func:`toblerone.pvestimation.cortex`: estimate PVs for one/both hemispheres of the cortex
   * :func:`toblerone.pvestimation.structure`: estimate PVs for a structure delineated by a single surface, eg thalamus 
   * :func:`toblerone.pvestimation.complete`: estimate PVs across the whole brain, including subcortical structures identified by FSL FIRST 

The latter function can be used for a direct replacement for conventional PV estimation tools such as FSL FIRST as it provides complete whole-brain PV estimates for GM, WM and non-brain tissue. 

Example usage 
---------------------

Partial volume estimation for the cortex via FreeSurfer: 

.. code-block:: python 

   import toblerone as tob 

   fsdir = '/path/to/freesurfer_subject_dir'
   ref = '/some/reference_image.nii.gz'

   # Identity transform for registration of surfaces to reference image  
   pvs = tob.pvestimation.cortex(fsdir=fsdir, ref=ref, struct2ref='I')

Partial volume estimation for a structure with a single surface: 

.. code-block:: python 

   import toblerone as tob 

   # NB it is assumed that the surface is in world-mm coordinates, 
   # if not use surface.transform() to get it so. 
   surf = '/path/to/surface.surf.gii'
   ref = '/some/reference_image.nii.gz'

   # Identity transform for registration of surfaces to reference image 
   pvs = tob.pvestimation.structure(surf=surf, ref=ref, struct2ref='I')

Partial volume estimation for the whole brain via FreeSurfer and fsl_anat:

.. code-block:: python 

   import toblerone as tob 

   fsdir = '/path/to/freesurfer_subject_dir'
   fslanat = '/path/to/fsl_anat_dir'
   ref = '/some/reference_image.nii.gz'

   # Identity transform for registration of surfaces to reference image  
   pvs = tob.pvestimation.complete(fsdir=fsdir, fslanat=fslanat, ref=ref, struct2ref='I')

Note also that toblerone can be used at the terminal. Type ``toblerone -h`` for help. 

PVEc with oxasl
-------------------

Toblerone has been integrated with ``oxasl`` to provide PV correction via the spatial Variational Bayesian (spatial-VB) inference. See `oxasl <https://github.com/ibme-qubic/oxasl>`_ and `oxasl_surfpvc <https://github.com/ibme-qubic/oxasl_surfpvc>`_. 
