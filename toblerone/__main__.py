import sys

from toblerone.commandline import estimate_all_cmd, estimate_cortex_cmd
from toblerone.commandline import estimate_structure_cmd, resample_cmd 
from toblerone.commandline import fsl_surf_anat_cmd

def main():

    suffix = (
"""
Tom Kirk, thomas.kirk@eng.ox.ac.uk
Institute of Biomedical Engineering / Wellcome Centre for Integrative Neuroimaging
University of Oxford, 2018
""")




    usage_main = ("""
PVTOOLS     Tools for estimating partial volumes

Usage:

-estimate_all           estimate PVs across the brain, for both cortical and subcortical
                            structures

-estimate_cortex        estimate PVs for the cortex

-estimate_structure     estimate PVs for a structure defined by a single surface 

-fsl_surf_anat          run fsl_anat and augment output with FreeSurfer (pre-processing
                            step for other Toblerone functions)
""")


    funcs = [estimate_all_cmd, estimate_cortex_cmd,
        estimate_structure_cmd, fsl_surf_anat_cmd]
    names = ['-estimate_all', '-estimate_cortex', '-estimate_structure' '-fsl_surf_anat']

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
                    print("\n", f.__doc__ + suffix)

        if not matched:
            print("Unrecognised command")
            print(usage_main + suffix)

if __name__ == '__main__':
    main()
