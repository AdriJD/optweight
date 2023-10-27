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
      version='0.0.2',
      cmdclass=cmdclass,
      ext_modules=cythonize(ext_modules, annotate=True,
                            compiler_directives=compiler_directives))
