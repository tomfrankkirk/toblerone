"""
CommonParser: a subclass of the library ArgumentParser object pre-configured
to parse arguments that are common to many pvtools functions
"""

import argparse

class CommonParser(argparse.ArgumentParser):
    """
    Preconfigured subclass of ArgumentParser to parse arguments that
    are common across pvtools functions. To use, instantiate an object, 
    then call add_argument to add in the arguments unique to the particular
    function in which it is being used, then finally call parse_args as 
    normal. 
    """

    def __init__(self):
        super().__init__()
        self.add_argument('-ref', type=str, required=True)
        self.add_argument('-struct2ref', type=str, required=True) 
        self.add_argument('-flirt', action='store_true', required=False)
        self.add_argument('-struct', type=str, required=False)
        self.add_argument('-cores', type=int, required=False)
        self.add_argument('-out', type=str, required=False)
        self.add_argument('-super', nargs='+', required=False)


    def parse(self, args):
        return vars(super().parse_args(args))



