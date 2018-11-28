# Toblerone

### Partial volume estimation on the cortical ribbon

## About
Toblerone estimates partial volumes on the cortical ribbon using cortical surfaces produced by tools such as FreeSurfer. It works on both Windows and UNIX platforms as a Python module or command-line script.

## Contents
1. [Usage](#usage)
2. [Acknowledgements](#acknowledgements)
3. [License](#license)
4. [Contact](#contact)

## Usage
```python
# As a command line tool (run without args for help)
$ python3 toblerone --ref path/to/reference --FSdir path/to/subject/directory 
--struct2ref path/to/structural/functional/transform

# Within a python script. Note estimatePVs() is the only function required. 
import toblerone

# Produce PV estimates for left hemisphere only
LPS = '/path/to/left/pial/surface.surf.gii' 
LWS = '/path/to/left/white/surface.surf.gii'
ref = '/path/to/reference/image.nii'
reg = some_matrix     # or a path to an array file
toblerone.estimatePVs(LWS=LWS, LPS=LPS, ref=ref, struct2ref=reg)

# Produce PV estimates for both hemispheres from a FS subject directory, with registration from structural to functional via FLIRT
subdir = '/path/to/subject/directory'
ref = '/path/to/reference/image.nii'
reg = '/path/to/FLIRT/matrix'
struct = '/path/to/structural/image.nii'   # Required with FLIRT matrix
toblerone.estimatePVs(FSdir=subdir, ref=ref, struct2ref=reg, flirt=True, struct=struct)
```

Within a script, all arguments should be provided as keyword arguments eg `function(arg=val)` or as flags eg `function(flag=True)`.

Toblerone can run with either two surfaces (eg left white, left pial) from the same hemisphere or all four surfaces from both hemispheres. Surfaces can either be specified individually or collectively from a FreeSurfer subject directory.


#### Required arguments

`ref` Path to reference image for which PV estimates are required

`LWS/LPS` Left hemishphere, White or Pial Surface. Paths to files in GIFTI or FS binary format. 

`RWS/RPS` Right hemisphere, White or Pial Surface. 

`FSdir` Alternative to `LWS/LPS/RWS/RPS`. Path to a FreeSurfer subject directory, eg `/freesurfer/subjects/01`. This will load both surfaces for both hemispheres. 

`struct2ref` Registration between structural image (from which surfaces were produced) and reference. If in a script this can be a numpy array; otherwise a path is valid for both command-line and scripts. 
If FSL's FLIRT was used for registration then set the `flirt` flag and give a path to the structural image using the `structural` argument. NB 'structural' *does not* refer to the 256x256x256 isotropic image `mri/orig.mgz` produced by FreeSurfer. 


#### Optional arguments 

`outdir` Output directory path. Will be created if it does not already exist. Defaults to the same directory as the reference image if not provided. 

`name` Output file name. Defaults to the reference filename with the suffix `_tob`. 

`flirt` Flag to signify `strcut2ref` is a FLIRT transformation, set with `struct`.

`struct` Path to structural image. Required when using FLIRT transform. 

`hard` Flag; do not estimate PVs. Perform whole-voxel tissue assignment based on position relative to cortex

`nostack` Flag; save estimates for each tissue as separate images (rather than the default of a single 4D volume, arranged GM, WM and non-brain)

`nosave` Flag; do not save PV estimates to files and return numpy array objects to the calling function. Useful when running within a script.

#### Outputs

The following are either saved to files or returned from the call to `estimatePVs()`.

PV estimates: 4D image of partial volume estimates in the range [0,1] for each tissue in each voxel. Dimensions 1:3 are the spatial dimensions of the reference image and dimension 4 is tissue type, arranged as GM, WM and CSF in indices 1:3. 

Surface mask: 3D logical mask showing voxels that at least partially intersect the cortex. These are the voxels of highest confidence, all others have an estimate assigned purely on their position relative to the cortex. **No brain masking is performed, it is left to the user to do this as they see fit**.

<p>
<figure align="center">
    <img src="doc/tobCSF.PNG" width="40%">
    <img src="doc/fastCSF.PNG" width="40%">
    <br>
    <figcaption><i>L: Toblerone's CSF estimate. All voxels exterior to the cortex are labelled 100% CSF. R: FAST's CSF estimate, implictly masked as the input was a masked brain image. </i></figcaption>
</figure>
</p>


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