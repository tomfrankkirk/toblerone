"""
Setup script for toblerone
"""
import os
import subprocess
import re
import io
import sys

from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension

PACKAGE_NAME = 'toblerone'
ROOTDIR = os.path.abspath(os.path.dirname(__file__))

def get_filetext(filename):
    """ Get the text of a local file """
    with io.open(os.path.join(ROOTDIR, filename), encoding='utf-8') as f:
        return f.read()

def git_version():
    """ Get the full and python standardized version from Git tags (if possible) """
    try:
        # Full version includes the Git commit hash
        full_version = subprocess.check_output('git describe --dirty', shell=True).decode("utf-8").strip(" \n")

        # Python standardized version in form major.minor.patch.post<build>
        version_regex = re.compile(r"v?(\d+\.\d+\.\d+(-\d+)?).*")
        match = version_regex.match(full_version)
        if match:
            std_version = match.group(1).replace("-", ".post")
        else:
            raise RuntimeError("Failed to parse version string %s" % full_version)
        return full_version, std_version
        
    except:
        # Any failure, return None. We may not be in a Git repo at all
        return None, None

def git_timestamp():
    """ Get the last commit timestamp from Git (if possible)"""
    try:
        return subprocess.check_output('git log -1 --format=%cd', shell=True).decode("utf-8").strip(" \n")
    except:
        # Any failure, return None. We may not be in a Git repo at all
        return None

def update_metadata(version_str, timestamp_str):
    """ Update the version and timestamp metadata in the module _version.py file """
    with io.open(os.path.join(ROOTDIR, PACKAGE_NAME, "_version.py"), "w", encoding='utf-8') as f:
        f.write("__version__ = '%s'\n" % version_str)
        f.write("__timestamp__ = '%s'\n" % timestamp_str)

def get_requirements():
    """ Get a list of all entries in the requirements file """
    with io.open(os.path.join(ROOTDIR, 'requirements.txt'), encoding='utf-8') as f:
        return [l.strip() for l in f.readlines()]

def get_version():
    """ Get the current version number (and update it in the module _version.py file if necessary)"""
    version, timestamp = git_version()[1], git_timestamp()

    if version is not None and timestamp is not None:
        # We got the metadata from Git - update the version file
        update_metadata(version, timestamp)
    else:
        # Could not get metadata from Git - use the version file if it exists
        try:
            with io.open(os.path.join(ROOTDIR, PACKAGE_NAME, '_version.py'), encoding='utf-8') as f:
                md = f.read()
                match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", md, re.M)
                if match:
                    version = match.group(1)
                else:
                    raise ValueError("Stored version could not be parsed")
        except (IOError, ValueError):
            version = "unknown"
            update_metadata(version, "unknown")
    return version

def get_extensions():
    """ Build Cython extensions """

    from Cython.Build import cythonize
    import numpy

    extensions = []
    compile_args = []
    link_args = []

    if sys.platform.startswith('win'):
        compile_args.append('/EHsc')

    # C extension for fast triangle/box intersection
    extensions.append(
        Extension("toblerone.ctoblerone",
                  sources=[
                      'toblerone/ctoblerone/ctoblerone.pyx', 
                      'src/ctoblerone.c',
                      'src/tribox.c'
                  ],
                  include_dirs=['src', numpy.get_include()],
                  language="c", 
                  extra_compile_args=compile_args, 
                  extra_link_args=link_args)
    )
    return cythonize(extensions, compiler_directives={'language_level': 3})

if __name__ == '__main__':

    from Cython.Distutils import build_ext
    setup(name=PACKAGE_NAME,
        version=get_version(),
        description="Surface-based analysis tools",
        long_description=get_filetext('README.md'),
        long_description_content_type='text/markdown',
        author='Tom Kirk',
        author_email='tomfrankkirk@gmail.com',
        license='BSD-3-clause', 
        url='https://github.com/tomfrankkirk/toblerone',
        setup_requires=['numpy', 'cython'],
        install_requires=get_requirements(),
        packages=find_packages(),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': build_ext},
        entry_points={
            'console_scripts' : [
                'toblerone=toblerone.__main__:main',
            ],
        },
    )
