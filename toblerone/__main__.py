import sys

from toblerone.commandline import estimate_all_cmd, estimate_cortex_cmd
from toblerone.commandline import estimate_structure_cmd, resample_cmd 

def main():

    suffix = (
"""
Tom Kirk, thomas.kirk@eng.ox.ac.uk
Institute of Biomedical Engineering / Wellcome Centre for Integrative Neuroimaging
University of Oxford, 2018
""")




    usage_main = """(
PVTOOLS     Tools for estimating partial volumes

Usage (preface all with "python3 -m pvtools"):

-estimate_all           estimate PVs across the brain, using FAST for subcortical 
                            and FreeSurfer/Toblerone for cortical tissues

-estimate_cortex        estimate PVs for the cortex using FreeSurfer/Toblerone

-estimate_structure     estimate PVs for a subcortical structure defined by a
                            single surface

-resample               resample an image onto a reference via integrative method, 
                            applying affine transformation

""")


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
