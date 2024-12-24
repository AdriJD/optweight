'''
Create map-based noise (i.e. with band-limit that exceeds lmax) and see if you can
estimate cov that produces sims with same power spectrum as sim.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy
import os
import time

import healpy as hp
from pixell import curvedsky, enplot, utils, enmap, sharp

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils
from optweight import noise_utils, alm_utils, mat_utils, alm_c_utils, wavtrans

opj = os.path.join
np.random.seed(39)

odir = '/home/adriaand/project/actpol/20210419_est_noise_pix_fullsky'
imgdir = opj(odir, 'img')

os.makedirs(imgdir, exist_ok=True)

lmax = 500
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

ainfo = sharp.alm_info(lmax)

cov_pix = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,))
cov_pix[:] = 10
cov_ell = np.ones(lmax + 1) * 10

rand_map = cov_pix.copy()
rand_map[:] = np.random.randn(*rand_map.shape) * np.sqrt(cov_pix)

rand_alm = np.empty((1, ainfo.nelem), dtype=np.complex128)
rand_alm = curvedsky.map2alm(rand_map, rand_alm, ainfo)
rand_map_bl = curvedsky.alm2map(rand_alm, rand_map.copy(), ainfo=ainfo)

#rand_alm = curvedsky.rand_alm(cov_ell, ainfo=ainfo)
#rand_alm = rand_alm[np.newaxis,:]
#rand_map = curvedsky.alm2map(rand_alm, rand_map, ainfo=ainfo)

pix_areas = enmap.pixsizemap(rand_map.shape, rand_map.wcs)

#rand_alm = np.empty((1, ainfo.nelem), dtype=np.complex128)
#rand_alm = curvedsky.map2alm(rand_map, rand_alm, ainfo)

n_ell_in = ainfo.alm2cl(rand_alm[:,None,:], rand_alm[None,:,:])

#cov_pix_est = rand_map * rand_map
cov_pix_est = rand_map_bl * rand_map_bl

alm = curvedsky.map2alm(cov_pix_est, rand_alm.copy(), ainfo)
b_ell = hp.gauss_beam(np.radians(2), lmax=lmax)
alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
cov_pix_est = curvedsky.alm2map(alm, cov_pix_est, ainfo=ainfo)
cov_pix_est[cov_pix_est < 0] = 0

cov_pix_est /= pix_areas 
cov_pix_est /= (lmax + 1) ** 2 / 4 / np.pi

rand_map_est = rand_map.copy()
rand_map_est[:] = np.random.randn(*rand_map_est.shape) * np.sqrt(cov_pix_est)

#rand_map_est *= np.sqrt(pix_areas)
#rand_map_est *= np.sqrt((lmax + 1) ** 2 / 4 / np.pi)

curvedsky.map2alm(rand_map_est, alm, ainfo)
n_ell_out = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])

fig, axs = plt.subplots(dpi=300, constrained_layout=True, nrows=2)
axs[0].plot(ells, n_ell_in[0,0], label='n_ell_in')
axs[0].plot(ells, n_ell_out[0,0], label='n_ell_out')
axs[0].legend()
axs[1].plot(ells, n_ell_out[0,0] / n_ell_in[0,0])
fig.savefig(opj(imgdir, 'n_ell_out'))
plt.close(fig)


# Repeat with function.

rand_alm2 = np.empty((1, ainfo.nelem), dtype=np.complex128)
minfo = map_utils.get_gauss_minfo(2*lmax) 
rand_map_gl = np.zeros((1, minfo.npix))
sht.alm2map(rand_alm, rand_map_gl, ainfo, minfo, 0)

#kernel_ell = np.ones(lmax+1)
#kernel_ell[:150] = 0
kernel_ell = None

cov_pix_est2 = noise_utils.estimate_cov_pix(rand_map_gl, minfo, kernel_ell=kernel_ell, lmax=lmax)
rand_map_gl_draw = rand_map_gl.copy()
rand_map_gl_draw[:] = np.random.randn(*rand_map_gl_draw.shape) * np.sqrt(cov_pix_est2)
sht.map2alm(rand_map_gl_draw, rand_alm2, minfo, ainfo, 0)
n_ell_out2 = ainfo.alm2cl(rand_alm2[:,None,:], rand_alm2[None,:,:])

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.plot(ells, n_ell_in[0,0], label='n_ell_in')
ax.plot(ells, n_ell_out[0,0], label='n_ell_out')
ax.plot(ells, n_ell_out2[0,0], label='n_ell_out2')
ax.legend()
fig.savefig(opj(imgdir, 'n_ell_out2'))
plt.close(fig)

