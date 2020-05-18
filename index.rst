.. Toblerone documentation master file, created by
   sphinx-quickstart on Mon May 18 18:43:29 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Toblerone
===============
Surface-based analysis tools 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <static/installation> 
   Quickstart <static/quickstart>
   Overview <static/overview>
   API <static/api>

Citing
------- 
If you use Toblerone in your work or would like to know more about how it works, please cite/refer to the paper: 

T. F. Kirk, T. S. Coalson, M. S. Craig and M. A. Chappell, *Toblerone: Surface-Based Partial Volume Estimation*, in IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1501-1510, May 2020, doi: 10.1109/TMI.2019.2951080.


License 
--------
BSD 

Acknowledgements 
----------------
Funding. 

The ray-triangle intersection test is based upon Tim Coalson's code (https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.cxx#L510), itself an adaptation of the PNPOLY test (https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html).

The triangle-voxel intersection test is a direct port of Tomas Akenine-Moller's code (http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt).

