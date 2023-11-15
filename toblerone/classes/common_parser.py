"""
CommonParser: a subclass of the library ArgumentParser object pre-configured
to parse arguments that are common to many pvtools functions
"""

import argparse
import multiprocessing as mp


class CommonParser(argparse.ArgumentParser):
    """
    Preconfigured subclass of ArgumentParser to parse arguments that
    are common across pvtools functions. To use, instantiate an object,
    then call add_argument to add in the arguments unique to the particular
    function in which it is being used, then finally call parse_args as
    normal.
    """

    def __init__(self, args_to_add, **kwargs):
        from ..__main__ import suffix

        super().__init__(
            prog="toblerone",
            epilog=suffix,
            usage="toblerone -command-name <options>",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **kwargs
        )

        general = self.add_argument_group("general arguments")
        if "ref" in args_to_add:
            general.add_argument(
                "-ref",
                required=True,
                help="path to reference image that defines voxel grid",
            )

        if "struct2ref" in args_to_add:
            general.add_argument(
                "-struct2ref",
                required=True,
                help="""path to registration to align anatomical inputs with reference
                image. Use 'I' to denote identity transform.""",
            )

        if "surf" in args_to_add:
            general.add_argument(
                "-surf",
                type=str,
                help="path to surface (see -coords argument below)",
            )

        if "coords" in args_to_add:
            general.add_argument(
                "-coords",
                choices=["fsl", "world"],
                default="world",
                help="""coordinates in which surface is defined, either 'world' 
                (mm coords) or 'fsl' (FSL convention, eg FIRST surfaces)""",
            )

        if "flirt" in args_to_add:
            general.add_argument(
                "-flirt",
                action="store_true",
                help="flag to set if struct2ref transform was produced by FSL FLIRT",
            )

        if "struct" in args_to_add:
            general.add_argument(
                "-struct",
                help=("""image from which anatomic inputs were derived"""),
            )

        if "projector" in args_to_add:
            general.add_argument("-projector", help="path to projector", required=True)

        if "out" in args_to_add:
            general.add_argument("-out", required=True, help="path to save output at")

        fsgroup = self.add_argument_group("cortical surfaces")
        if "fsdir" in args_to_add:
            fsgroup.add_argument(
                "-fsdir",
                help="path to FreeSurfer subject directory, from which /surf "
                "will be loaded, or provide -LPS/LWS/RPS/RWS",
            )

        if "LWS" in args_to_add:
            fsgroup.add_argument(
                "-LWS",
                help="alternative to -fsdir, path to left white surface",
            )

        if "LPS" in args_to_add:
            fsgroup.add_argument(
                "-LPS",
                help="alternative to -fsdir, path to left pial surface",
            )

        if "LSS" in args_to_add:
            fsgroup.add_argument(
                "-LSS",
                help="alternative to -fsdir, path to left spherical surface",
            )

        if "RWS" in args_to_add:
            fsgroup.add_argument(
                "-RWS",
                help="alternative to -fsdir, path to right white surface",
            )

        if "RPS" in args_to_add:
            fsgroup.add_argument(
                "-RPS",
                help="alternative to -fsdir, path to right pial surface",
            )

        if "RSS" in args_to_add:
            fsgroup.add_argument(
                "-RSS",
                help="alternative to -fsdir, path to right spherical surface",
            )

        if "sides" in args_to_add:
            fsgroup.add_argument(
                "-sides",
                choices=["L", "R"],
                default=["L", "R"],
                nargs="+",
                help="cortical hemispheres to process, only works with -fsdir",
            )

        if "resample" in args_to_add:
            fsgroup.add_argument(
                "-resample",
                type=int,
                default=32492,
                metavar="N",
                help="resample cortical surfaces to N vertices (default 32492)",
            )

        anatgroup = self.add_argument_group("subcortical surfaces and segmentations")
        if "firstdir" in args_to_add:
            anatgroup.add_argument(
                "-firstdir",
                type=str,
                help=("""directory with FSL FIRST outputs"""),
            )

        if "fastdir" in args_to_add:
            anatgroup.add_argument(
                "-fastdir",
                type=str,
                help="""directory with FSL FAST outputs""",
            )

        misc = self.add_argument_group("other arguments")
        if "supr" in args_to_add:
            misc.add_argument(
                "-super",
                dest="supr",
                nargs="+",
                type=int,
                metavar="N",
                help="""voxel subdivision factor, a single value 
                    or one for each dimension""",
            )

        if "cores" in args_to_add:
            misc.add_argument(
                "-cores",
                type=int,
                default=mp.cpu_count(),
                metavar="N",
                help="number of CPU cores to use (default is max available)",
            )

        if "ones" in args_to_add:
            misc.add_argument(
                "-ones",
                action="store_true",
                help="debug tool (whole voxel PV assignment)",
            )
