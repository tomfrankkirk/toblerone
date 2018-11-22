#!/usr/bin/env python
import os
import sys
import glob

from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools.extension import Extension

import numpy

Description = """/
Toblerone
"""

extensions = []
compile_args = []
link_args = []

if sys.platform.startswith('win'):
    zlib = "zlib"
    extra_inc = "src/compat"
    compile_args.append('/EHsc')
elif sys.platform.startswith('darwin'):
    link_args.append("-stdlib=libc++")
    zlib = "z"
    extra_inc = "."

# T1 map generation extension
extensions.append(Extension("ctoblerone",
                 sources=['ctoblerone.pyx', 
                          'extern/tribox.c', 
                          'extern/ctoblerone.c'],
                 include_dirs=['extern', numpy.get_include()],
                 language="c", extra_compile_args=compile_args, extra_link_args=link_args))

# setup parameters
setup(name='toberone',
      cmdclass={'build_ext': build_ext},
      version="0.0.1",
      description="Toblerone",
      long_description=Description,
      author='',
      author_email='',
      packages=[],
      setup_requires=['Cython'],
      install_requires=[],
      ext_modules=cythonize(extensions),
)
