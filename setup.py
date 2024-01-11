from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError

from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy as np
import os
import subprocess as sp
from pathlib import Path

opj = os.path.join

path = str(Path(__file__).parent.absolute())

compile_opts = {
    'extra_compile_args' : ['-shared', '-std=c99', '-g', '-Wall'],
    'extra_link_args' : ['-Wl,-rpath,' + opj(path, 'lib')]}

compile_opts_mat = {
    'extra_compile_args' : ['-shared', '-std=c99', '-g', '-Wall'],
    'extra_link_args' : ['-Wl,-rpath,' + opj(path, 'lib')]}

compile_opts_map = {
    'extra_compile_args' : ['-shared', '-std=c99', '-g', '-Wall'],
    'extra_link_args' : ['-Wl,-rpath,' + opj(path, 'lib')]}

compiler_directives = {'language_level' : 3}

def presrc():
    '''Create .so library from C files.'''
    if sp.call('make', shell=True) != 0:
        raise CompileError('Failure in the C compile source-prep.')

class CustomSrc(build_ext):
    def run(self):
        presrc()
        return build_ext.run(self)

# Mostly taken from pixell. The main change is the cython requirement.
requirements =  ['numpy>=1.20.0',
                 'astropy>=2.0',
                 'setuptools>=39',
                 'h5py>=2.7',
                 'scipy>=1.0',
                 'python_dateutil>=2.7',
                 'cython>=3.0.0',
                 'healpy>=1.13',
                 'matplotlib>=2.0',
                 'pyyaml>=5.0',
                 'Pillow>=5.3.0',
                 'pytest-cov>=2.6',
                 'coveralls>=1.5',
                 'pytest>=4.6',
                 'ducc0>=0.31.0']

cmdclass = {'build_ext': CustomSrc}

ext_modules = [Extension('optweight.alm_c_utils',
                        [opj(path, 'cython', 'alm_c_utils.pyx')],
                         libraries=['optweight_c_utils'],
                         library_dirs=[opj(path, 'lib')],
                         include_dirs=[opj(path, 'include'),
                                       np.get_include()],
                         **compile_opts),
               Extension('optweight.mat_c_utils',
                         [opj(path, 'cython', 'mat_c_utils.pyx')],
                         libraries=['optweight_c_utils'],
                         library_dirs=[opj(path, 'lib')],
                         include_dirs=[opj(path, 'include'),
                                       np.get_include()],
                         **compile_opts_mat),
               Extension('optweight.map_c_utils',
                         [opj(path, 'cython', 'map_c_utils.pyx')],
                         libraries=['optweight_c_utils'],
                         library_dirs=[opj(path, 'lib')],
                         include_dirs=[opj(path, 'include'),
                                       np.get_include()],
                         **compile_opts_map)]

setup(name='optweight',
      packages=['optweight'],
      version='0.0.3',
      install_requires=requirements,
      cmdclass=cmdclass,
      ext_modules=cythonize(ext_modules, annotate=True,
                            compiler_directives=compiler_directives))
