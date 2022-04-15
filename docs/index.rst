
Toblerone
=====================================

Surface-based analysis tools for neuroimaging

Installation
-----------------

Cython and numpy must be installed *prior* to toblerone itself: 

.. code-block:: bash

    python -m pip install cython numpy
    python -m pip install toblerone

Usage 
-----------

Toblerone has two broad use cases:

1. :ref:`Partial volume estimation <pvestimation-index>` using surface segmentations 
2. :ref:`Projection <projection-index>` of data between volume, surface and hybrid spaces. 

Use from within Python scripts is recommended as it allows for better customisation of options. Toblerone can also be used at the terminal, run ``toblerone -h`` to see the available commands.

Citation 
----------- 

If you use Toblerone in your work, please include the following citations: 

T. F. Kirk, T. S. Coalson, M. S. Craig and M. A. Chappell, “Toblerone: Surface-Based Partial Volume Estimation,” *IEEE Transactions on Medical Imaging*, vol. 39, no. 5, pp. 1501-1510, May 2020, `doi: 10.1109/TMI.2019.2951080. <https://ieeexplore.ieee.org/document/8892523>`_

T. F. Kirk, M. S. Craig and M. A. Chappell, "Unified surface and volumetric projection of physiological imaging data", *bioRxiv*, 2022, `doi: 10.1101/2022.01.28.477071. <https://www.biorxiv.org/content/10.1101/2022.01.28.477071v1>`_

.. toctree::
   :maxdepth: 1
   :hidden:

   pvestimation
   projection
   API reference <modules>
   Index <genindex>
