# Toblerone

### Surface-based partial volume estimation 

## Contents
1. [Usage](#usage)

## Usage

#### Python scripting
The python module provides the greatest level of control and flexibility. 
```python
import toblerone

## Cortex 
# Path to image for which PVs are required
ref = 'path/to/reference/image.nii.gz'
fsdir = '/path/to/freesurfer/subjectdir'

# Registration between structural and reference image (eg with FLIRT)
# NB if flirt must also provide path to structural image used for the registration
s2r = 'path/to/struct2reference_registration_FLIRT'
struct = 'path/to/struct/image.nii.gz'

# pvs is a 4D array with the tissues arranged GM/WM/non-brain in the 4th dimension
pvs, _, _ = toblerone.estimate_cortex(ref=ref, fsdir=fsdir, 
    struct2ref=s2r, struct=struct, flirt=True)

# A convenience method for saving the results in the space of the reference
toblerone.classes.ImageSpace.save_like(ref, pvs, 'output/path')

## Subcortical structure
# direct control via points and triangle arrays
ps = # a Px3 array of surface vertices, in world mm coordinates 
ts = # a Tx3 array of surface triangles

# Create a surface object using the ps,ts, with a name
# Pas 'I' as struct2ref to denote identity transform (ie, do nothing)
surf = toblerone.classes.Surface.manual(ps, ts, 'name')
pvs = toblerone.estimate_structure(ref=ref, struct2ref='I', surf=surf)
```

All arguments must be provided as keyword arguments eg `function(arg=val)` or as flags eg `function(flag=True)`.


#### Command line 
Some of the high-level functions are available at the command-line. Documentation is built-in. 
```bash
$ python3 -m toblerone
```

There are two PV estimation functions: 

estimate_cortex
estimate_structure

And there is one script 
estimate_all



## Acknowledgements
The ray-triangle intersection test is based upon Tim Coalson's code (https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.cxx#L510), itself an adaptation of the PNPOLY test (https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html).

The FLIRT matrix adjustment code was supplied by Tim Coalson and Martin Craig, adapted from (https://github.com/Washington-University/workbench/blob/9c34187281066519e78841e29dc14bef504776df/src/Nifti/NiftiHeader.cxx#L168) and (https://github.com/Washington-University/workbench/blob/335ad0c910ca9812907ea92ba0e5563225eee1e6/src/Files/AffineFile.cxx#L144)

The triangle-voxel intersection test is a direct port of Tomas Akenine-Moller's code (http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt).

With thanks to all the above.

## License

## Contact 
Tom Kirk, 2018. 

Institute of Biomedical Engineering, University of Oxford. 

thomas.kirk@eng.ox.ac.uk