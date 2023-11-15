from toblerone import scripts
import numpy as np
from toblerone.__main__ import main
import sys
import toblerone as tob

if __name__ == "__main__":
    ref = "/Users/thomaskirk/Data/singlePLDpcASL/asltc.nii.gz"
    fs = "/Users/thomaskirk/Data/singlePLDpcASL/fs"
    first = "/Users/thomaskirk/Data/singlePLDpcASL/T1.anat/first_results"
    fast = "/Users/thomaskirk/Data/singlePLDpcASL/T1.anat"
    struct = "/Users/thomaskirk/Data/singlePLDpcASL/T1.anat/T1.nii.gz"
    s2r = "I"
    ones = False

    sides = ["L"]

    # via python

    # scripts.pvs_cortex-freesurfer(ref=ref, struct2ref=s2r, sides=sides, fsdir=fs, ones=True)

    # scripts.pvs_freesurfer(ref=ref, struct2ref=s2r, fsdir=fs, sides=sides, ones=True)

    # scripts.pvs_freesurfer_fsl(ref=ref, struct2ref=s2r, fsdir=fs, fastdir=fast, firstdir=first, struct=struct, sides=sides, ones=True)

    # scripts.projector_freesurfer(ref=ref, struct2ref=s2r, sides=sides, fsdir=fs, ones=True)

    # scripts.projector_freesurfer_fsl(ref=ref, struct2ref=s2r, fsdir=fs, fastdir=fast, firstdir=first, ones=True, sides=sides)

    # scripts.pvs_subcortex_fsl(ref=ref, struct2ref=s2r, firstdir=first, fastdir=fast, ones=True)

    # scripts.pvs_subcortex_freesurfer(ref=ref, struct2ref=s2r, fsdir=fs)

    # via command line

    sides = "L"
    ones = "-ones"

    # cmd = f"""-pvs-cortex-freesurfer -ref {ref} -struct2ref {s2r}
    #             -fsdir {fs} {ones} -out scratch/pvs-cortex"""
    # sys.argv[1:] = cmd.split()
    # main()

    # cmd = f"""-pvs-freesurfer -ref {ref} -struct2ref {s2r} -fsdir {fs}
    #          {ones} -sides {sides} -out scratch/pvs-fs"""
    # sys.argv[1:] = cmd.split()
    # main()

    # cmd = f"""-pvs-freesurfer-fsl -ref {ref} -struct2ref {s2r}
    #             -fsdir {fs} -fastdir {fast} -firstdir {first} -struct {struct} -sides {sides}
    #             {ones} -out scratch/pvs-fs-fsl"""
    # sys.argv[1:] = cmd.split()
    # main()

    # cmd = f"""-projector-freesurfer -ref {ref} -struct2ref {s2r}
    #             -fsdir {fs} -sides {sides} -out scratch/projector-fs"""
    # sys.argv[1:] = cmd.split()
    # main()

    # cmd = f"""-projector-freesurfer-fsl -ref {ref} -struct2ref {s2r} -fsdir
    #         {fs} -fastdir {fast} -firstdir {first} {ones}
    #           -out scratch/projector-fs-fsl"""
    # sys.argv[1:] = cmd.split()
    # main()

    # cmd = f"""-pvs-subcortex-fsl -ref {ref} -struct2ref {s2r} -firstdir {first}
    #          -fastdir {fast} {ones} -out scratch/pvs-subcortex-fsl"""
    # sys.argv[1:] = cmd.split()
    # main()

    # cmd = f"""-pvs-subcortex-freesurfer -ref {ref} -struct2ref {s2r}
    #          -fsdir {fs} -out scratch/pvs-subcortex-fs"""
    # sys.argv[1:] = cmd.split()
    # main()
