# Installation 

### From pip

Numpy and Cython must be already installed (_pip cannot install these automatically_). FSL is also required (for fslpy). Then: 
```bash
pip install toblerone
``` 

### From source 

Clone the git respository and then run `python setup.py build sdist` to build a pip-installable package in the `dist` directory. The Cython extensions can be built with `python setup.py build_ext --inplace`

To check the installation, type `toblerone` at the command line. 