
Toblerone
=====================================

Surface-based analysis tools 

Installation
-----------------

Cython needs to be installed prior to running pip: 
```
python -m pip install cython
python -m pip install toblerone
```


Partial volume estimation 
-----------------------------


Toblerone can estimate partial volumes (PVs) across the brain using surface segmentations (for example, those from FreeSurfer and FSL FIRST). 
The following functions are available in :mod:`toblerone.pvestimation`: 

   * :func:`toblerone.pvestimation.estimation.cortex`: estimate PVs for one/both hemispheres of the cortex
   * :func:`toblerone.pvestimation.estimation.structure`: estimate PVs for a structure delineated by a single surface, eg thalamus 
   * :func:`toblerone.pvestimation.estimation.all`: estimate PVs across the whole brain, including subcortical structures identified by FSL FIRST 


Projection 
-------------

This is work in progress. 

.. toctree::
   :maxdepth: 2

   PV estimation <pvestimation>
   API index <modules> 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
