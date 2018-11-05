#!/usr/bin/env python
"""
Example cython wrapper
"""
import sys

from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

extensions = []
compile_args = []
libs = []
includes = ['.', 'src', numpy.get_include()]

if sys.platform.startswith('win'):
    compile_args.append('/EHsc')
else:
    libs.append('m')
    #if sys.platform.startswith('darwin'):
    #    link_args.append("-stdlib=libc++")
  
extensions.append(
    Extension("wrapper",
              sources=['wrapper.pyx', 'src/wrapper.c'],
              include_dirs=includes, libraries=libs,
              language="c", extra_compile_args=compile_args))

# setup parameters
setup(name='cython_template',
      cmdclass={'build_ext': build_ext},
      version="0.0.1",
      description="Cython template",
      author='Martin Craig',
      setup_requires=['Cython'],
      entry_points={
        'console_scripts' : [
            "oxasl_ve=oxasl_ve.api:main",
        ],
      },
      ext_modules=cythonize(extensions))
