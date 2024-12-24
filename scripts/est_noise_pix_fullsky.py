'''
Estimate noise on full sky to check for biases.
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


odir = '/home/adriaand/project/actpol/20210407_est_noise_pix_fullsky'
imgdir = opj(odir, 'img')

os.makedirs(imgdir, exist_ok=True)

lmax = 1000
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

ainfo = sharp.alm_info(lmax)
imap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,))

#imap[:] = np.random.randn(*imap.shape)
alm, ainfo = curvedsky.rand_alm(np.ones(lmax+1), return_ainfo=True)
imap = curvedsky.alm2map(alm, imap, ainfo=ainfo)

#imap[:,:400:,:] = 0
#imap[:,-400:,:] = 0

for pidx in range(imap.shape[0]):
    plot = enplot.plot(imap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'imap_in_{pidx}'), plot)

alm = np.empty((imap.shape[0], ainfo.nelem), dtype=np.complex128)
curvedsky.map2alm(imap, alm, ainfo)
imap = curvedsky.alm2map(alm, imap, ainfo=ainfo)

for pidx in range(imap.shape[0]):
    plot = enplot.plot(imap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'imap_bl_{pidx}'), plot)

#b_ell = hp.gauss_beam(np.radians(1), lmax=lmax)
#b_ell = np.ones_like(b_ell)
#alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
curvedsky.map2alm(imap, alm, ainfo)
n_ell = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])

#imap = curvedsky.alm2map(alm, imap, ainfo=ainfo)

#for pidx in range(imap.shape[0]):
#    plot = enplot.plot(imap[pidx], colorbar=True, grid=False)
 #   enplot.write(opj(imgdir, f'imap__{pidx}'), plot)

# lamb = 1.7
# #lmin = 25
# lmin = 1000
# lmax_w = 1000
# #lmax_j = 900
# lmax_j = None

# w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
#     lamb, lmax_w, j0=None, lmin=lmin, return_j=True, lmax_j=lmax_j)

# fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
# for widx in range(w_ell.shape[0]):
#     ax.plot(w_ell[widx], label='Phi')
# ax.set_ylabel('Wavelet kernel')
# ax.set_xlabel('Multipole')
# ax.set_xscale('log')
# fig.savefig(opj(imgdir, 'kernels_log'))
# plt.close(fig)

# fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
# for widx in range(w_ell.shape[0]):
#     ax.plot(w_ell[widx], label='Phi')
# ax.set_ylabel('Wavelet kernel')
# ax.set_xlabel('Multipole')
# fig.savefig(opj(imgdir, 'kernels'))
# plt.close(fig)

# wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)

# # Test roundtrip
# alm_rt, _ = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell, ainfo=ainfo)
# n_ell_rt = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
# omap_rt = curvedsky.alm2map(alm_rt, imap.copy(), ainfo=ainfo)

# for pidx in range(imap.shape[0]):
#     plot = enplot.plot(omap_rt[pidx], colorbar=True, grid=False)
#     enplot.write(opj(imgdir, f'omap_rt_{pidx}'), plot)

# wav_sm = wavtrans.Wav(2)

# for widx, (wlm, winfo) in enumerate(zip(wlms, winfos)):

#     wmap = enmap.zeros(imap.shape, wcs=imap.wcs)
#     curvedsky.alm2map(wlm, wmap, ainfo=winfo)

#     wmap *= wmap

#     curvedsky.map2alm(wmap, wlm, winfo)
#     b_ell = hp.gauss_beam(np.radians(10), lmax=winfo.lmax)
#     alm_c_utils.lmul(wlm, b_ell, winfo, inplace=True)

#     minfo = sharp.map_info_gauss_legendre(winfo.lmax+1, nphi=2*winfo.lmax+1)
#     omap_gl = np.empty((imap.shape[0], minfo.npix))
#     sht.alm2map(wlm, omap_gl, winfo, minfo, 0)
#     omap_gl = mat_utils.get_near_psd(omap_gl)

#     wav_sm.add((widx, widx), omap_gl, minfo)

#     omap_w = curvedsky.alm2map(wlm, imap.copy(), ainfo=winfo)
    
#     for pidx in range(imap.shape[0]):
#         plot = enplot.plot(omap_w[pidx], colorbar=True, grid=False)
#         enplot.write(opj(imgdir, f'omap_w_{widx}_{pidx}'), plot)

# rand_alm = alm_utils.rand_alm_wav(wav_sm, ainfo, w_ell, [0,2])
# n_ell_out_w = ainfo.alm2cl(rand_alm[:,None,:], rand_alm[None,:,:])

#imap_2 = curvedsky.make_projectable_map_by_pos(
#    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], 2 * lmax, dims=(1,))
#curvedsky.alm2map(alm, imap_2, ainfo=ainfo)

#imap_2 *= imap_2
#imap = imap_2
imap *= imap

for pidx in range(imap.shape[0]):
    plot = enplot.plot(imap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'imap_sq_{pidx}'), plot)

b_ell = hp.gauss_beam(np.radians(10), lmax=lmax)

curvedsky.map2alm(imap, alm, ainfo)
alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
curvedsky.alm2map(alm, imap, ainfo=ainfo)

imap[imap < 0] = 0

for pidx in range(imap.shape[0]):
    plot = enplot.plot(imap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'imap_sm_{pidx}'), plot)

pix_scaling = enmap.pixsizemap(imap.shape, imap.wcs).copy()
#pix_scaling /= pix_scaling[pix_scaling.shape[0]//2,0]

rand_omap = enmap.zeros(imap.shape, imap.wcs)
rand_omap[:] = np.random.randn(*rand_omap.shape) * np.sqrt(imap / pix_scaling)
#rand_omap[:] = np.random.randn(*rand_omap.shape) * np.sqrt(imap / np.sqrt(pix_scaling) / lmax * 4 * np.pi)
#rand_omap[:] = np.random.randn(*rand_omap.shape) * np.sqrt(imap)

for pidx in range(imap.shape[0]):
    plot = enplot.plot(rand_omap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'rand_omap_{pidx}'), plot)

alm = np.empty((rand_omap.shape[0], ainfo.nelem), dtype=np.complex128)
curvedsky.map2alm(rand_omap, alm, ainfo)
n_ell_out = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.plot(n_ell_out[0,0], label='n_ell_out')
ax.plot(n_ell[0,0], label='n_ell')
#ax.plot(n_ell_rt[0,0], ls=':', color='black', label='n_ell_rt')
#ax.plot(n_ell_out_w[0,0], ls=':', color='red', label='n_ell_out_w')
#ax.set_xscale('log')
ax.legend()
fig.savefig(opj(imgdir, 'n_ell_out'))
plt.close(fig)


