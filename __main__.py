import sys 

if __name__ == '__main__':

    suffix = "\nTom Kirk, thomas.kirk@eng.ox.ac.uk, 2018 \n"

    usage_main = """
PVTOOLS     Tools for estimating partial volumes

Usage (run either option with no arguments for more information):

$ pvtools -estimate-all             estimate PVs for both cortical and subcortical tissues, 
                                        using FAST and FreeSurfer/Toblerone 

$ pvtools -estimate-cortex          estimate PVs for the cortex using FreeSurfer/Toblerone

$ pvtools -resample                 resample an image via upsampling and integration, apply
                                        optional affine transformation

"""

    usage_estimate = """
PVTOOLS -estimate-all	  Run FreeSurfer and FSL's FAST & Toblerone to estimate PVs for a reference 
image. Requires installations of FreeSurfer and FSL tools to be available on the system $PATH. 

Usage:

$ pvtools -estimate-all -struct STRUCT -ref REF -struct2ref S2R [-dir DIR -flirt]

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

    args = sys.argv

    if len(args) == 1:
        print(usage_main + suffix)

    if len(args) == 2:
    
        if args[1] == '-estimate-all':
            print(usage_estimate + suffix)

        elif args[1] == '-merge-with-surface':
            print(usage_merge + suffix)

        elif args[1] == '-resample-image':
            pass 

        else:
            print("PV tools: unrecognised command")
