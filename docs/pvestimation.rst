.. _pvestimation-index:

Partial volume estimation
==============================

Surface-based partial volumes are defined as the volume of intersection between a voxel grid and a closed surface that is contained within it. Some voxels will partially intersect the voxel; the interior/exterior fractions are *partial volumes*. The voxel grid of interest is referred to as the *reference grid*. 

Partial volume estimation is performed on a surface-by-surface basis. For a structure defined by a single surface (eg thalamus), the function toblerone.pvestimation.structure will return a single array containing values in the range [0,1] where 1 denotes that a voxel is fully contained within the surface. 

For a structure defined by multiple surfaces (cortex), partial volumes can be estimated by performing estimation for each individual surface in turn and then combining the results. The function toblerone.pvestimation.cortex does this with the pial and white surfaces, and then takes the voxel-wise difference between the two as grey matter. Currently the cortex is the only mulit-surface structure supported, though others could be processed by using toblerone.pvestimation.surface on the individual surfaces and combining the results as necessary. 

PV estimation functions 
----------------------------------

Toblerone can estimate partial volumes (PVs) across the brain using surface segmentations (for example, those from FreeSurfer and FSL FIRST). This can be used as a direct replacement for tools such as FSL FAST. 
The following functions are available in :mod:`toblerone.pvestimation`: 

   * :func:`toblerone.pvestimation.estimation.cortex`: estimate PVs for one/both hemispheres of the cortex
   * :func:`toblerone.pvestimation.estimation.structure`: estimate PVs for a structure delineated by a single surface, eg thalamus 
   * :func:`toblerone.pvestimation.estimation.all`: estimate PVs across the whole brain, including subcortical structures identified by FSL FIRST 


PVEc with oxasl
-------------------

Toblerone has been integrated with `oxasl` to provide PV correction via the spatial Variational Bayesian method. See [oxasl](#https://github.com/ibme-qubic/oxasl) and [oxasl_surfpvc](#https://github.com/ibme-qubic/oxasl_surfpvc). 

.. automodule:: toblerone.pvestimation
   :members:
   :undoc-members:
   :show-inheritance: