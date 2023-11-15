.. _projection-index:

Projection 
=============

Spaces of data representation
-------------------------------
Toblerone can project data between different *spaces*, as explained below. 

1. **Volume**: the conventional analysis space for MRI data, this can be any 3D voxel grid as defined by a NIFTI header, for example. This space is best for the study of the subcortex. 
2. **Surface**: particularly well-suited to the study of the cortex, this is a 2D topological sphere (ie, closed surface) defined by a cortical surface reconstruction (ie, the output of FreeSurfer)
3. **Hybrid**: this is the union of the volume and surface spaces into a single data representation. It includes all voxels of interest for the subcortex, all surface vertices for the cortex, and (optionally) any subcortical regions of interest (ROI). This concept is very similar, but not indetical, to the HCP's concept of *grayordinates*, which is embodied in the CIFTI file format. 

The Projector object 
----------------------
Any projection of data between spaces is handled by a :class:`~toblerone.projection.Projector` object. A :class:`~toblerone.projection.Projector` encapsulates the geometric relationship between the surfaces of the cortex (one or both hemispheres) and *one specific voxel grid*. Once initialised (a slow process, due to the need to perform many geometric computations), a :class:`~toblerone.projection.Projector` can be used to project any data that is in alignment with the cortical surfaces and exists in the correct voxel grid. This means that you can re-use a :class:`~toblerone.projection.Projector` for multiple acquisitions, if the data is in alignment. 

.. warning:: 
   Toblerone **does not perform registration** between cortical surfaces and volumetric data. When creating a :class:`~toblerone.projection.Projector`, it is assumed that the surfaces are in alignment 
   with the data to be projected. If you have the necessary registration matrices, you can apply them to the surfaces using the :func:`~toblerone.classes.surfaces.Surface.transform()` method before creating a :class:`~toblerone.projection.Projector`. This is preferred to transforming the data itself, which creates interpolation artefacts. 

.. warning:: 
   A :class:`~toblerone.projection.Projector` is only valid for the voxel grid with which it is created. If data from another grid is to be projected (eg, an acquisition with a different resolution), then a new :class:`~toblerone.projection.Projector` must be created for that voxel grid. 

The basic steps of creating and using a :class:`~toblerone.projection.Projector` are as follows: 

1. Create the :class:`~toblerone.projection.Projector` for the cortical surfaces and voxel grid of interest. This usually takes around 10 minutes.  
2. It is strongly recommended at this point to save the projector. This avoids having to re-create it from scratch at a later date.
3. Choose the correct projection method on the :class:`~toblerone.projection.Projector`: :func:`~toblerone.projection.Projector.vol2surf`, :func:`~toblerone.projection.Projector.surf2vol()`, :func:`~toblerone.projection.Projector.hybrid2vol()`, :func:`~toblerone.projection.Projector.vol2hybrid()`
4. Call the appropriate method with the vector of data to be projected. 

**Surface projection** is achieved with the :func:`~toblerone.projection.Projector.vol2surf()` and :func:`~toblerone.projection.Projector.surf2vol()` methods. 

**Hybrid projection** is achieved with the :func:`~toblerone.projection.Projector.vol2hybrid()` and :func:`~toblerone.projection.Projector.hybrid2vol()` methods. 

.. note::
   Edge scaling is optional for all projection methods, in all spaces. It reflects the presence of PVE in creating *missing* signal in voxels that are less than 100% brain tissue. 

   In the surface to volume direction, edge scaling is on by default and will reduce the final voxel signal where the brain tissue PV in that voxel is less than 100%. 

   In the volume to surface direction, edge scaling is off by default and will (if enabled) increase the final surface signal where the corresponding voxels have brain tissue PVs less than 100%. This accounts for the *missing* signal that would have been acquired if there were no PVE. NB in this direction, edge scaling is a poorly-conditioned operation that will amplify noise. 


Example usage 
--------------- 

Initialisation and saving of a projector for a single cortical hemisphere: 

.. code-block:: python 

   import toblerone as tob 

   # Create a hemisphere object for left surfaces, side must be specified
   # NB surfaces could also be GIFTI 
   LWS = '/path/to/lh.white'
   LPS = '/path/to/lh.pial'
   LSS = '/path/to/lh.sphere'
   lhemi = tob.Hemisphere(LWS, LPS, LSS, side='L')

   # If any registration of the surfaces to the reference grid is required, 
   # it must be done now 
   lhemi = lhemi.transform(some_registration)

   # Create the projector and save to file 
   ref = '/some/reference_image.nii.gz'   
   proj = toblerone.Projector(lhemi, spc)
   proj.save('/path/to/save.h5')

Load a projector and use to project data between surface and volume spaces: 

.. code-block:: python 

   import toblerone as tob 
   import numpy as np 

   # Load existing projector 
   proj = tob.Projector.load('/path/to/load.h5')

   # Simulate some volumetric data for the reference voxel grid 
   # and project onto the surface 
   vol_data = np.random.rand(*proj.spc.size)
   vol_on_surf = proj.vol2surf(vol_data.flatten(), edge_scale=False)

   # Simulate some surface data and project into the volume 
   surf_data = np.random.rand(proj.n_surf_nodes)
   surf_in_vol = proj.surf2vol(surf_data, edge_scale=False)