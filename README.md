# Toblerone

### Surface-based partial volume estimation 

## Contents
1. [Installation](#installation)
2. [Usage](#usage)

## Installation

#### From source
Clone this git repository. 

Build the python package `python3 setup.py sdist`

Install the package built in `dist/` via pip (activate environment first if desired) `pip3 install dist/*`

Pip may display a warning about the main script not being on your `$PATH`. If you would like to run Toblerone as a command-line tool then follow the instructions given in the warning. Alternatively, you can access the command-line via `python3 -m toblerone [args]`

To check the installation you can run `toblerone -tests`
## Usage

#### Python scripting


#### Command line 


## Acknowledgements
The ray-triangle intersection test is based upon Tim Coalson's code (https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.cxx#L510), itself an adaptation of the PNPOLY test (https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html).

The FLIRT matrix adjustment code was supplied by Tim Coalson and Martin Craig, adapted from (https://github.com/Washington-University/workbench/blob/9c34187281066519e78841e29dc14bef504776df/src/Nifti/NiftiHeader.cxx#L168) and (https://github.com/Washington-University/workbench/blob/335ad0c910ca9812907ea92ba0e5563225eee1e6/src/Files/AffineFile.cxx#L144)

The triangle-voxel intersection test is a direct port of Tomas Akenine-Moller's code (http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt).

Martin Craig generously did almost all the trouble-shooting and module creation for this code. 

With thanks to all the above.

## License

## Contact 
Tom Kirk, 2018. 

Institute of Biomedical Engineering, University of Oxford. 

thomas.kirk@eng.ox.ac.uk