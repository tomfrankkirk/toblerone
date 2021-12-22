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

    def __init__(self, *args_to_add, **kwargs):
        from ..__main__ import suffix

        super().__init__(prog='toblerone', epilog=suffix,
            usage='toblerone -command-name <options>', 
            formatter_class=argparse.RawDescriptionHelpFormatter, 
            **kwargs)

        general = self.add_argument_group("general arguments")
        if 'ref' in args_to_add: 
            general.add_argument('-ref', required=True, 
                help="path to reference image that defines voxel grid")

        if 'surf' in args_to_add:
            general.add_argument('-surf', type=str, required=True,
                help="path to surface (see -coords argument below)")

        if 'coords' in args_to_add:
            general.add_argument('-coords', required=True, 
                choices=['fsl', 'world'], default='world',
                help="""coordinates in which surface is defined, either 'world' 
                (mm coords) or 'fsl' (FSL convention, eg FIRST surfaces)""")
            
        if 'struct2ref' in args_to_add: 
            general.add_argument('-struct2ref', required=True,
                help="""path to registration to align surfaces with reference
                image. Use 'I' to denote identity transform.""")
    
        if 'flirt' in args_to_add: 
            general.add_argument('-flirt', action='store_true', required=False,
                help="set if struct2ref transform was produced by FSL FLIRT")

        if 'struct' in args_to_add: 
            general.add_argument('-struct', required=False, 
                help=("""if -struct2ref is FLIRT transform, or -firstdir has 
                    been set, provide a path to the structural image used 
                    to generate the surfaces"""))

        if 'out' in args_to_add: 
            general.add_argument('-out', required=True, 
                help="path to save output at")            



        fsgroup = self.add_argument_group("cortical surfaces")
        if 'fsdir' in args_to_add: 
            fsgroup.add_argument('-fsdir', required=False,
                help="path to FreeSurfer subject directory, from which /surf "
                "will be loaded, or provide -LPS/LWS/RPS/RWS")

        if 'LWS' in args_to_add: 
            fsgroup.add_argument('-LWS', required=False,
                help="alternative to -fsdir, path to left white surface")

        if 'LPS' in args_to_add: 
            fsgroup.add_argument('-LPS', required=False,
                help="alternative to -fsdir, path to left pial surface")

        if 'RWS' in args_to_add: 
            fsgroup.add_argument('-RWS', required=False,
                help="alternative to -fsdir, path to right white surface")

        if 'RPS' in args_to_add: 
            fsgroup.add_argument('-RPS', required=False,
                help="alternative to -fsdir, path to right pial surface")


        anatgroup = self.add_argument_group("subcortical surfaces and segmentations")
        if 'fslanat' in args_to_add: 
            anatgroup.add_argument('-fslanat', type=str, required=False, 
                help="path to fslanat dir (replaces firstdir/fastdir)")

        if 'firstdir' in args_to_add:
            anatgroup.add_argument('-firstdir', type=str, required=False, 
                help=("""replaces -fslanat, path to FSL FIRST directory 
                    (all .vtk surfaces will be loaded)"""))
        
        if 'fastdir' in args_to_add: 
            anatgroup.add_argument('-fastdir', type=str, required=False,
                help="""replaces -fslanat, path to directory with
                    FAST outputs. Note that -struct2ref transform will 
                    also be applied""")


        misc = self.add_argument_group("other arguments")
        if 'super' in args_to_add: 
            misc.add_argument('-super', nargs='+', required=False,
                type=int, help="""voxel subdivision factor, a single value 
                    or one for each dimension""")

        if 'cores' in args_to_add: 
            misc.add_argument('-cores', type=int, required=False, 
                default=mp.cpu_count(), 
                help="number of CPU cores to use (default is max available)")

        if 'ones' in args_to_add: 
            misc.add_argument('-ones', action='store_true', 
                help="debug tool (whole voxel PV assignment)")
