
Toblerone
=====================================

Surface-based analysis tools for neuroimaging

Installation
-----------------

Cython and numpy must be installed *prior* to running pip: 

.. code-block:: python

    python -m pip install cython numpy
    python -m pip install toblerone

Usage 
-----------

Toblerone has two broad use cases:

1. :ref:`Partial volume estimation <pvestimation-index>` using surface segmentations 
2. :ref:`Projection <projection-index>` of data between volume, surface and hybrid spaces. 

Run ``toblerone -h`` to see available commands and help info. Toblerone can also be used within python scripts.

Citation 
----------- 

If you use Toblerone in your work, please include the following citation: 

T. F. Kirk, T. S. Coalson, M. S. Craig and M. A. Chappell, “Toblerone: Surface-Based Partial Volume Estimation,” in IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1501-1510, May 2020, `doi: 10.1109/TMI.2019.2951080. <https://ieeexplore.ieee.org/document/8892523>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   pvestimation
   projection
   API reference <modules>
   Index <genindex>


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
