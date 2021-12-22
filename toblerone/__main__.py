import sys
import argparse
from textwrap import dedent

from toblerone import commandline
from toblerone._version import __version__

suffix = (
f"""
version {__version__}
Tom Kirk,
Institute of Biomedical Engineering / Wellcome Centre for Integrative Neuroimaging,
University of Oxford, 2018
""")


def main():

    args = sys.argv

    parser = argparse.ArgumentParser(prog='toblerone', 
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=suffix,
        usage='toblerone -command-name <options>',
        description=dedent("TOBLERONE Surface-based analysis tools. Run any command with -h for help."))
        
    parser.add_argument('-estimate-complete', action='store_true',
        help=("estimate PVs across the brain, for both cortical and "
            "subcortical structures."))

    parser.add_argument('-estimate-cortex', action='store_true',
        help=("estimate PVs for L/R cortical hemispheres."))

    parser.add_argument('-estimate-structure', action='store_true',
        help=("estimate PVs for a structure defined by a single surface."))

    parser.add_argument('-convert-surface', action='store_true',
        help=("convert a surface file (.white/.pial/.vtk/.surf.gii). Note that"
            " FS surfaces will have the C_ras shift applied automatically.")) 

    parser.add_argument('-prepare-projector', action='store_true',
        help="prepare a projector for surface-based analysis of volumetric data.")

    cmd_name = args[1:2]
    if not cmd_name: 
        parser.print_help()
        return 

    parsed = parser.parse_args(cmd_name)
    for attr in vars(parsed):
        if hasattr(commandline, attr) and (getattr(parsed, attr) is True):
            sys.argv[1:] = args[2:]
            getattr(commandline, attr)()


if __name__ == '__main__':
    main()
