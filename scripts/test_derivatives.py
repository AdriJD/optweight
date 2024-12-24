import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import time

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils, enmap, sharp
from enlib import cg

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils
from optweight import noise_utils, alm_utils, mat_utils, alm_c_utils

opj = os.path.join
np.random.seed(39)

lmax = 2000

basedir = '/home/adriaand/project/actpol/20210416_test_derivatives'
imgdir = opj(basedir, 'img')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))

# Note that is of course wrong for icov.
icov = enmap.downgrade(icov, 4, op=np.mean)

ainfo = sharp.alm_info(lmax)
alm = np.empty(ainfo.nelem, dtype=np.complex64)
curvedsky.map2alm(icov[0], alm, ainfo=ainfo)

omap = enmap.zeros((2,) + icov.shape[-2:], wcs=icov.wcs, dtype=icov.dtype)
curvedsky.alm2map(alm, omap, ainfo=ainfo, deriv=True)

plot = enplot.plot(omap, colorbar=True, grid=False)
enplot.write(opj(imgdir, f'omap'), plot)

mag = np.sqrt(omap[0] ** 2 + omap[1] ** 2)
plot = enplot.plot(mag, colorbar=True, grid=False)
enplot.write(opj(imgdir, f'magnitude'), plot)

