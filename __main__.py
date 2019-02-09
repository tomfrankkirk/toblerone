import sys

from .commandline import estimate_all_cmd, estimate_cortex_cmd
from .commandline import estimate_structure_cmd, resample_cmd


if __name__ == '__main__':

    suffix = """
Tom Kirk, thomas.kirk@eng.ox.ac.uk
Institute of Biomedical Engineering / Wellcome Centre for Integrative Neuroimaging
University of Oxford, 2018"""




    usage_main = """
PVTOOLS     Tools for estimating partial volumes

Usage (preface all with "python3 -m pvtools"):

-estimate_all           estimate PVs across the brain, using FAST for subcortical 
                            and FreeSurfer/Toblerone for cortical tissues

-estimate_cortex        estimate PVs for the cortex using FreeSurfer/Toblerone

-estimate_structure     estimate PVs for a subcortical structure defined by a
                            single surface

-resample               resample an image onto a reference via integrative method, 
                            applying affine transformation

"""



    usage_all = """
PVTOOLS -estimate-all	  Run FreeSurfer and FSL's FAST & Toblerone to estimate PVs for a reference 
image. Requires installations of FreeSurfer and FSL tools to be available on the system $PATH. 

Usage:

$ python3 -m pvtools -estimate-all -struct STRUCT -ref REF -struct2ref S2R [-dir DIR -flirt]

Required arguments: 
    -struct         path to structural T1-weighted image from which PVs are
                        estimated
    -ref            path to reference image for which PVs are required 
    -struct2ref	    path to 4x4 affine transformation matrix (text-like file) describing transformation 
                        from structural to reference image. If using a FLIRT matrix set the -flirt 
                        flag as well.

Optional arguments: 
    -dir            where to save output folder (default is in same location as ref)
    -flirt          to signify the -struct2ref argument was produced by FSL's FLIRT

Overview:

Runs FreeSurfer on the structural image to produce cortical surfaces. 
Runs FAST on the structural image to produce volumetric PV estimates for subcortical tissue. 
Resamples FAST estimates to an upsampled copy of reference image space, applying the -struct2ref 
transformation matrix in the process, to give subcortical PVs.
Runs Toblerone to estimate cortical PVs in this upsampled reference space. 
Combines the two sets of estimates into a single output in an appropriate manner. 

The structural and intermediate space files will be saved in their own folders within the output directory. 
The overall PV estimates file will have the same filename as the reference with the suffix '_pvs'. 

"""


    usage_cortex = """
PVTOOLS -estimate-cortex    PV estimation for the cortex via Toblerone 

PVs are estimated by considering the intersection between the surfaces of the cortex and the voxels of 
a reference image. Use the -FS flag to run FreeSurfer first to produce surfaces (requires installation)

Required arguments: 
    -ref           path to reference image for which to estimate PVs
    -fsdir         path to a FreeSurfer subject directory; surfaces will be loaded from the /surf dir. 
                        Alternative to LWS/LPS/RWS/RPS, in .gii or .white/.pial format
    -LWS, -LPS    paths to left hemisphere white and pial surfaces respectively 
    -RWS, -RPS    as above for right hemisphere
    -struct2ref    path to structural (from which surfaces were produced) to reference transform.
                        Set '--struct2ref I' for identity transform. NB if this is a FSL FLIRT
                        transform then set the --flirt flag

Optional arguments:
    -name          output filename (ext optional). Defaults to reference filename with suffix _tob  
    -outdir        output directory. Defaults to directory containing the reference image   
    -flirt         flag, signifying that the --struct2ref transform was produced by FSL's FLIRT
                        If set, then a path to the structural image from which surfaces were 
                        produced must also be given     
    -struct        path to structural image (ie, what FreeSurfer was run on)
    -hard          don't estimate PVs, instead simply assign whole-voxel tissue volumes based on position
                        relative to surfaces
    -nostack       don't stack each tissue estimates into single image, save each separately 
    -saveassocs    save triangle/voxel associations data (debug tool)
    -cores         number of (logical) cores to use, default is maximum available - 1
    

File formats:
    Surfaces are loaded either via the nibabel.freesurferio (.white/pial) or freesurfer.load (.gii)
    modules. 
    Images are loaded via the nibabel.load module (.nii/.nii.gz)
    Transformation matrices are loaded via np.fromtxt(), np.load() or np.fromfile() functions. 
"""

    usage_structure = """
BLAH
"""

    usage_resample = """
PVTOOLS resample    resample image via integrative process 

Usage:

python3 -m pvtools resample -src SOURCE -ref REF [-aff AFFINE=I -flirt]

Required arguments: 
    -src    path to image to be resampled
    -ref    path to image that defines reference space onto which src will be mapped
    -out    path to save output

Optional arguments
    -aff    path to text-like file for affine transformation from SOURCE to REFERENCE space.
                Defaults to identity matrix. 
    -flirt  set if AFFINE is a FLIRT transformation 
"""




    usages = [usage_all, usage_cortex, usage_structure, usage_resample]
    funcs = [estimate_all_cmd, estimate_cortex_cmd,
        estimate_structure_cmd, resample_cmd]
    names = ['-estimate_all', '-estimate_cortex', '-estimate_structure' '-resample']

    args = sys.argv

    if len(args) == 1:
        print(usage_main + suffix)

    if len(args) > 1:
        name = args[1]

        if len(args) > 2:
            fargs = args[2:]
        else:
            fargs = []

        matched = False
        for f, n, u in zip(funcs, names, usages):
            if name == n:
                matched = True 
                if fargs:
                    f(*fargs)
                else:
                    print(u + suffix)

        if not matched:
            print("Unrecognised command")
            print(usage_main + suffix)