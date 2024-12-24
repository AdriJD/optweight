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
np.random.seed(39)

lmax = 2000
dpi = 300
margin = 0.05 

basedir = '/home/adriaand/project/actpol/20201206_pcg_mask_test'
imgdir = opj(basedir, 'img')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')
areadir = '/home/adriaand/project/actpol/mapdata/area/'

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))
#icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest')
icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest')

mask_bool = icov_pix > 1e-3
icov_pix[~mask_bool] = 0
mask = mask_bool.astype(np.float64)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=dpi)
    im = ax.imshow(mask[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'mask_{}'.format(pidx)))
    plt.close(fig)

cov_pix = np.power(icov_pix, -1, where=mask_bool, out=icov_pix.copy())

ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

# Load beam.
b_ell = hp.gauss_beam(np.radians(1.3 / 60), lmax=lmax, pol=True)
b_ell = np.ascontiguousarray(b_ell[:,:3].T)

# Preprare spectrum. Input file is Dls in uk^2.
cov_ell = np.zeros((3, 3, lmax + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
    skiprows=1, usecols=[1, 2, 3, 4]) #  TT, EE, BB, TE.
c_ell = c_ell.T

cov_ell[0,0,2:] = c_ell[0,:lmax-1] 
cov_ell[0,1,2:] = c_ell[3,:lmax-1] 
cov_ell[1,0,2:] = c_ell[3,:lmax-1] 
cov_ell[1,1,2:] = c_ell[1,:lmax-1] 
cov_ell[2,2,2:] = c_ell[2,:lmax-1] 

alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
for pidx in range(alm.shape[0]):
    hp.almxfl(alm[pidx], b_ell[pidx], inplace=True)
# Draw map-based noise and add to alm.
noise = map_utils.rand_map_pix(cov_pix)
signal = noise.copy() * 0
alm_signal = alm.copy()
alm_noise = alm.copy()
sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=False)

sht.alm2map(alm_noise, noise, ainfo, minfo, [0,2], adjoint=False)
sht.alm2map(alm_signal, signal, ainfo, minfo, [0,2], adjoint=False)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(noise[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'noise_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(signal[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'signal_{}'.format(pidx)))
    plt.close(fig)

# Yt M Y.
noise *= mask
alm_noise_1 = alm_noise.copy()
sht.map2alm(noise, alm_noise_1, minfo, ainfo, [0,2], adjoint=True)

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
noise_enmap = curvedsky.alm2map(alm_noise, omap.copy())

omap = curvedsky.alm2map(alm_noise_1, omap)
for pidx in range(3):
    plot = enplot.plot(omap[pidx] - noise_enmap[pidx], colorbar=True, font_size=50, grid=False)
    enplot.write(opj(imgdir, 'noise_mask_adjoint_{}'.format(pidx)), plot)

# Yt W M Y.
alm_noise_2 = alm_noise.copy()
sht.map2alm(noise, alm_noise_2, minfo, ainfo, [0,2], adjoint=False)

omap = curvedsky.alm2map(alm_noise_2, omap)
for pidx in range(3):
    plot = enplot.plot(omap[pidx] - noise_enmap[pidx], colorbar=True, font_size=50, grid=False)
    enplot.write(opj(imgdir, 'noise_mask_normal_{}'.format(pidx)), plot)

# Yt M W Y.
#sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=True)
sht.alm2map(alm_noise, noise, ainfo, minfo, [0,2], adjoint=True)
noise *= mask

alm_noise_3 = alm_noise.copy()
sht.map2alm(noise, alm_noise_3, minfo, ainfo, [0,2], adjoint=True)

omap = curvedsky.alm2map(alm_noise_3, omap)
for pidx in range(3):
    plot = enplot.plot(omap[pidx] - noise_enmap[pidx], colorbar=True, font_size=50, grid=False)
    enplot.write(opj(imgdir, 'noise_mask_normal_adjoint_{}'.format(pidx)), plot)


