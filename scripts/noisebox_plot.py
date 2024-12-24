import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils, enmap
from enlib import cg

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils, noise_utils

opj = os.path.join

basedir = '/home/adriaand/project/actpol/20201210_noisebox'
imgdir = opj(basedir, 'img')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'

arrays = ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150', 'pa4_f150',
          'pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150',
          'ar1_f150', 'ar2_f220', 'planck_f090', 'planck_f150', 'planck_f220']
bins = [100, 111, 124, 137, 153, 170, 189, 210, 233, 260, 289, 321, 357, 397,
        441, 490, 545, 606, 674, 749, 833, 926, 1029, 1144, 1272, 1414, 1572,
        1748, 1943, 2161, 2402, 2671, 2969, 3301, 3670, 4081, 4537, 5044, 5608,
        6235, 6931, 7706, 8568, 9525, 10590, 11774, 13090, 14554, 16180, 17989]

noisebox = enmap.read_fits(opj(metadir, 'noisebox_f150_night.fits'))
# Sum over arrays.
#noisebox = np.sum(noisebox, axis=1)
#noisebox = np.sum(noisebox[:,:-3,...], axis=1) # No Planck.

# Plot all maps for each array. Up to some lmax perhaps.
for aidx, array in enumerate(arrays):
    if not 'f150' in array:
        continue
    for bidx in range(len(bins)):
        lbin = bins[bidx]
        plot = enplot.plot(noisebox[:,aidx,bidx], colorbar=True, ticks=10)
        enplot.write(opj(imgdir, 'noisebox_{}_{}'.format(array, lbin)), plot)

