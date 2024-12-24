import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils, sharp, enmap
from enlib import cg

from optweight import sht, map_utils, mat_utils, solvers, operators, preconditioners

opj = os.path.join

basedir = '/home/adriaand/project/actpol/20220514_pcg_planck'
imgdir = opj(basedir, 'img')
odir = opj(basedir, 'out')

# load planck otput alm
alm_wiener = hp.read_alm(opj(odir, 'alm_out.fits'), hdu=(1, 2, 3))
alm_constr = hp.read_alm(opj(odir, 'alm_constr.fits'), hdu=(1, 2, 3))
alm_in = hp.read_alm(opj(odir, 'alm_in.fits'), hdu=(1, 2, 3))
alm_icov = hp.read_alm(opj(odir, 'alm_icov.fits'), hdu=(1, 2, 3))

lmax = hp.Alm.getlmax(alm_wiener.shape[-1])
ainfo = sharp.alm_info(lmax)

# construct a slice of the map
omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/5, -np.pi/5],[-np.pi, np.pi]], lmax, dims=(alm_wiener.shape[0],))

# alm2map
omap_wiener = curvedsky.alm2map(alm_wiener, omap.copy(), ainfo=ainfo)
omap_constr = curvedsky.alm2map(alm_constr, omap.copy(), ainfo=ainfo)
omap_in = curvedsky.alm2map(alm_in, omap.copy(), ainfo=ainfo)
omap_icov = curvedsky.alm2map(alm_icov, omap.copy(), ainfo=ainfo)

patch = np.radians([[-40, 0], [40,-180]])        
omap_wiener = enmap.submap(omap_wiener, patch)
omap_constr = enmap.submap(omap_constr, patch)
omap_icov = enmap.submap(omap_icov, patch)
omap_in = enmap.submap(omap_in, patch)

# plot
plot = enplot.plot(omap_wiener, colorbar=True, font_size=25, ticks=20, range='250:7', downgrade=4,
                   grid_color='00000000')
enplot.write(opj(imgdir, 'omap_wiener'), plot)

plot = enplot.plot(omap_constr, colorbar=True, font_size=25, ticks=20, range='250:7', downgrade=4,
                   grid_color='00000000')
enplot.write(opj(imgdir, 'omap_constr'), plot)

plot = enplot.plot(omap_in, colorbar=True, font_size=25, ticks=20, range='250:7', downgrade=4,
                   grid_color='00000000')
enplot.write(opj(imgdir, 'omap_in'), plot)

plot = enplot.plot(omap_icov, colorbar=True, font_size=25, ticks=20, range='10000:13000', downgrade=4,
                   grid_color='00000000')
enplot.write(opj(imgdir, 'omap_icov'), plot)
