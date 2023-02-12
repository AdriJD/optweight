from numpy.distutils.core import setup, Extension, build_src
from distutils.errors import DistutilsError
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
    #'extra_link_args' : ['-Wl,-rpath,' + opj(path, 'lib'), '-L$(MKLROOT)/lib/intel64', '-Wl,--no-as-needed', '-lmkl_intel_lp64', '-lmkl_sequential', '-lmkl_core', '-lpthread']}

compile_opts_mat = {
    'extra_compile_args' : ['-shared', '-std=c99', '-g', '-Wall', '-fopenmp', '-Ofast'],
    'extra_link_args' : ['-fopenmp']}
#    'extra_compile_args' : ['-shared', '-std=c99', '-g', '-Wall', '-openmp' '-O3' '-ffast-math'],
#    'extra_link_args' : ['-openmp']}


compiler_directives = {'language_level' : 3}

def presrc():
    '''Create .so library from C files.'''
    if sp.call('make', shell=True) != 0:
        raise DistutilsError('Failure in the C compile source-prep.')

class CustomSrc(build_src.build_src):
    def run(self):
        presrc()
        return build_src.build_src.run(self)

cmdclass = {'build_src': CustomSrc}

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
                         **compile_opts_mat)]

setup(name='optweight',
      packages=['optweight'],
      cmdclass=cmdclass,
      ext_modules=cythonize(ext_modules, annotate=True,
                            compiler_directives=compiler_directives))
