# Toblerone

### Surface-based partial volume estimation 

## Contents
1. [Installation](#installation)
2. [Usage](#usage)

## Installation

#### From source

#### From pip: 
`pip install toblerone`

#### From source 
Requires cython and numpy. 

Clone the git repository (https://github.com/tomfrankkirk/toblerone) and cd into it.
Install the python package: `pip install .` (cython will run automatically). 
Pip may display a warning about the main script not being on your `$PATH`. If you would like to run Toblerone as a command-line tool then follow the instructions given in the warning. 

To check the installation, type `toblerone` at the command line. 

## Usage

#### Python scripting

The python interface provides more control over the individual functions and also allows you to assemble your own PV estimation framework (if the standard ones listed below are not suitable). The following are available at the module level (ie, `toblerone.estimate_all`)

- `fsl_fs_anat` Pre-processing step for neuroimaging applications. Runs FSL's `fsl_anat` script (cropping, re-orienting, bias-field correction, brain extraction, tissue segmentation via FAST and subcortical structure segmentation via FIRST) on a T1 image and then augments the output with FreeSurfer's `recon-all` to get cortical surfaces. The output of this (referred to as an `anat_dir`) can then be passed as an input to other functions.  

- `estimate_structure` Estimate PVs arising from a structure defined by a single surface. Structures defined by multiple surfaces can be processed by estimating on each individual surface and subtracting the results (as is done in `estimate_cortex`). This returns an array of size equal to the reference image space, where the value in each position corresponds to the proportion of that voxel that is *interior* to the structure. 

- `estimate_cortex` Estimate PVs within the cortical ribbon (either one or both hemispheres). If both hemispheres are requested (default) then the results from each will be combined into a single image. Returns an array of size equal to the reference, extended to three volumes in the 4th dimension (eg X Y Z 3), where the three values in each position give the GM, WM and non-brain (CSF + background + everything else) PVs respectively.  

#### Command line 

Docs to come. 

## Acknowledgements
The ray-triangle intersection test is based upon Tim Coalson's code (https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.cxx#L510), itself an adaptation of the PNPOLY test (https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html).

The FLIRT matrix adjustment code was supplied by Tim Coalson and Martin Craig, adapted from (https://github.com/Washington-University/workbench/blob/9c34187281066519e78841e29dc14bef504776df/src/Nifti/NiftiHeader.cxx#L168) and (https://github.com/Washington-University/workbench/blob/335ad0c910ca9812907ea92ba0e5563225eee1e6/src/Files/AffineFile.cxx#L144)

The triangle-voxel intersection test is a direct port of Tomas Akenine-Moller's code (http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt).

Martin Craig generously did the vast-majority of trouble shooting and setup for this module. 

## License

## Contact 
Tom Kirk, 2018. 

Institute of Biomedical Engineering, University of Oxford. 

thomas.kirk@eng.ox.ac.uk