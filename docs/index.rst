
Toblerone
=====================================

Surface-based analysis tools 

Installation
-----------------

Cython and numpy need to be installed prior to running pip: 

.. code-block:: python

    python -m pip install cython numpy
    python -m pip install toblerone


Usage from command line
-------------------------

Run ``toblerone -h`` to see available commands and help info. 


Usage from Python: PV estimation
----------------------------------

Toblerone can estimate partial volumes (PVs) across the brain using surface segmentations (for example, those from FreeSurfer and FSL FIRST). This can be used as a direct replacement for tools such as FSL FAST. 
The following functions are available in :mod:`toblerone.pvestimation`: 

   * :func:`toblerone.pvestimation.estimation.cortex`: estimate PVs for one/both hemispheres of the cortex
   * :func:`toblerone.pvestimation.estimation.structure`: estimate PVs for a structure delineated by a single surface, eg thalamus 
   * :func:`toblerone.pvestimation.estimation.all`: estimate PVs across the whole brain, including subcortical structures identified by FSL FIRST 


Parial volume correction 
----------------------------

Toblerone has been integrated with `oxasl` to provide PV correction via the spatial Variational Bayesian method. See [oxasl](#https://github.com/ibme-qubic/oxasl) and [oxasl_surfpvc](#https://github.com/ibme-qubic/oxasl_surfpvc). 

Usage from Python: PV estimation
------------------------------------

This is work in progress. 

.. toctree::
   :maxdepth: 2

   PV estimation <pvestimation>
   Projection <projection>
   API index <modules> 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
