from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path

opj = os.path.join

path = str(Path(__file__).parent.absolute())

compile_opts = {
    'extra_compile_args': ['-shared', '-std=c99', '-g', '-Wall'],
    'extra_link_args' : ['-Wl,-rpath,' + opj(path, 'lib')]}

compiler_directives = {'language_level' : 3}

ext_modules = [Extension('optweight.alm_c_utils',
                        [opj(path, 'cython', 'alm_c_utils.pyx')],
                         libraries=['optweight_c_utils'],
                         library_dirs=[opj(path, 'lib')],
                         include_dirs=[opj(path, 'include'),
                                       np.get_include()],
                         **compile_opts)]

setup(name='optweight',
      packages=['optweight'],
      ext_modules=cythonize(ext_modules,
                            compiler_directives=compiler_directives))
