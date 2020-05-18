# Toblerone

### Surface-based partial volume estimation 

<!-- ## Contents
1. [Installation](#installation)
2. [Usage](#usage) -->

## Installation

#### From pip: 
Both Cython and Numpy are required **prior** to installation via pip (_they will not be installed automatically by pip_). FSL is also required (for fslpy). Then: 

```bash
pip install toblerone
```

To check the installation, type `toblerone` at the command line. 

## Usage

### Python scripting

The python interface provides more control over the individual functions and also allows you to assemble your own PV estimation framework if the standard ones are not suitable. 

- `toblerone.pvestimation.structure` Estimate PVs arising from a structure defined by a single surface. Structures defined by multiple surfaces can be processed by estimating on each individual surface and subtracting the results (as is done in `estimate_cortex`). This returns an array of size equal to the reference image space, where the value in each position corresponds to the proportion of that voxel that is *interior* to the structure. 

- `toblerone.pvestimation.cortex` Estimate PVs within the cortical ribbon (either one or both hemispheres). If both hemispheres are requested (default) then the results from each will be combined into a single image. Returns an array of size equal to the reference, extended to three volumes in the 4th dimension (eg X Y Z 3), where the three values in each position give the GM, WM and non-brain (CSF + background + everything else) PVs respectively.  

#### Example usage (for the cortex)

```python 
import toblerone 

# FreeSurfer native surfaces
LWS = 'path/to/lh.white'
LPS = 'path/to/lh.pial'
RPS = 'path/to/rh.pial'
RWS = 'path/to/rh.white'
ref = 'path/to/functional.nii.gz'

# NB if using a FSL FLIRT matrix also set 'flirt=True' and 
# provide a path to structural image below
struct2ref = 'registration_structural_to_reference.mat'

# This returns a 4D array, sized equal to the reference image, 
# with the PVs arranged GM, WM and non-brain in the last dimension 
pvs = toblerone.pvestimation.cortex(LWS=LWS, LPS=LPS, RPS=RPS, RWS=RWS, struct2ref=struct2ref, ref=ref)

```

### Command line 

Run `toblerone` without arguments for usage information. 

## Documentation 

Full documentation to follow. In the meantime, all functions have docstrings that can be accessed in IPython, eg: 
```python 
import python
toblerone.pvestimation.structure?
```

## Acknowledgements
The ray-triangle intersection test is based upon Tim Coalson's code (https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.cxx#L510), itself an adaptation of the PNPOLY test (https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html).

The triangle-voxel intersection test is a direct port of Tomas Akenine-Moller's code (http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt).

Martin Craig generously did the vast-majority of trouble shooting and setup for this module. 

## License
TBC

## Contact 
Tom Kirk, 2018. 

Institute of Biomedical Engineering, University of Oxford. 

thomas.kirk@eng.ox.ac.uk