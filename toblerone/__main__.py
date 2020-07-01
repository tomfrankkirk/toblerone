import sys

from toblerone.commandline import (estimate_complete_cmd, estimate_cortex_cmd, 
                                   estimate_structure_cmd, convert_surface_cmd, 
                                   fsl_fs_anat_cmd, suffix)

def main():

    usage_main = ("""
TOBLERONE     Surface-based analysis tools

Usage:

-estimate_complete      estimate PVs across the brain, for both cortical and subcortical
                            structures

-estimate_cortex        estimate PVs for the cortex

-estimate_structure     estimate PVs for a structure defined by a single surface 

-fsl_fs_anat            run fsl_anat and augment output with FreeSurfer (pre-processing
                            step for other Toblerone functions)

-convert_surface        convert a surface file (.white/.pial/.vtk/.surf.gii). Note that FS 
                            surfaces will have the C_ras shift applied automatically.     
""")


    funcs = [estimate_complete_cmd, estimate_cortex_cmd,
        estimate_structure_cmd, fsl_fs_anat_cmd, convert_surface_cmd]
    names = ['-estimate_complete', '-estimate_cortex', '-estimate_structure', 
        '-fsl_fs_anat', '-convert_surface']

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
        for f, n in zip(funcs, names):
            if name == n:
                matched = True 
                if fargs:
                    f(*fargs)
                else:
                    print(f.__doc__ + suffix)

        if not matched:
            print("Unrecognised command")
            print(usage_main + suffix)

if __name__ == '__main__':
    main()
