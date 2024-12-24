'''
Draw 2 types of noise alms: uncorrelated pixel noise and wavelet-based noise.
'''
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

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils
from optweight import noise_utils, alm_utils, mat_utils

opj = os.path.join
np.random.seed(39)

lmax = 5000

basedir = '/home/adriaand/project/actpol/20210119_act_sim_wavelet'
imgdir = opj(basedir, 'img')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')
areadir = '/home/adriaand/project/actpol/mapdata/area/'

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))
icov = icov.astype(np.float64)
icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest')
icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=0.3)
mask = icov_pix != 0
mask_pix = mask.astype(np.float64)

# determine area of mask, approx...
# Area of rings.
theta_min = np.min(minfo.theta)
theta_max = np.max(minfo.theta)
area = - 2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
area *= np.sum(mask) / mask.size
fsky = area / 4 / np.pi
print('area', area)
print('fsky', fsky)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'icov_real_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(np.log10(np.abs(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'icov_real_log_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(mask[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'mask_{}'.format(pidx)))
    plt.close(fig)

cov_pix = np.power(icov_pix, -1, where=mask, out=icov_pix.copy())

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(
        cov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'cov_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(
        np.log10(np.abs(cov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'cov_log_{}'.format(pidx)))
    plt.close(fig)

arrays = ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150', 'pa4_f150',
          'pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150',
          'ar1_f150', 'ar2_f220', 'planck_f090', 'planck_f150', 'planck_f220']
bins = [100, 111, 124, 137, 153, 170, 189, 210, 233, 260, 289, 321, 357, 397,
        441, 490, 545, 606, 674, 749, 833, 926, 1029, 1144, 1272, 1414, 1572,
        1748, 1943, 2161, 2402, 2671, 2969, 3301, 3670, 4081, 4537, 5044, 5608,
        6235, 6931, 7706, 8568, 9525, 10590, 11774, 13090, 14554, 16180, 17989]

# load kernels
lamb = 1.7
lmin = 100

#w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
#    lamb, lmax, j0=None, lmin=lmin, return_j=True, lmax_j=2500)
w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax, j0=None, lmin=lmin, return_j=True)

print('lmaxs', lmaxs)
print('js', j_scales)

ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels_me'))
plt.close(fig)

noisebox = enmap.read_fits(opj(metadir, 'noisebox_f150_night.fits'))
# Sum over arrays.
noisebox[0,-3:,...] = np.sqrt(noisebox[1,-3:,...] ** 2 + noisebox[2,-3:,...] ** 2)
noisebox = np.sum(noisebox, axis=1)
#noisebox = np.sum(noisebox[:,:-3,...], axis=1) # No Planck.

t0 = time.time()
icov_wav = noise_utils.noisebox2wavmat(noisebox.copy(), bins, w_ell, offsets=[0]) 
print(time.time() - t0)
              

in_ell_iso = map_utils.get_ivar_ell(icov_wav, w_ell)

fig, axs = plt.subplots(dpi=300, constrained_layout=True, nrows=3)
for pidx in range(3):
    axs[pidx].plot(ells, in_ell_iso[pidx,pidx])
fig.savefig(opj(imgdir, 'in_ell_iso'))
plt.close(fig)

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

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(cov_ell[idxs])
fig.savefig(opj(imgdir, 'cov_ell'))
plt.close(fig)

cov_ell[...,1:] /= dells[1:]

icov_ell = np.ones((3, 3, lmax + 1))
for lidx in range(icov_ell.shape[-1]):
    if lidx < 2:
        # Set monopole and dipole to zero.
        icov_ell[:,:,lidx] = 0
    else:
        icov_ell[:,:,lidx] = np.linalg.inv(cov_ell[:,:,lidx])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(icov_ell[idxs])
fig.savefig(opj(imgdir, 'icov_ell'))
plt.close(fig)

# Draw signal alms.
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
for pidx in range(alm.shape[0]):
    hp.almxfl(alm[pidx], b_ell[pidx], inplace=True)

# Draw wavelet-based noise alms.
cov_wav = mat_utils.wavmatpow(icov_wav, -1)
t0 = time.time()
alm_noise_wav = alm_utils.rand_alm_wav(cov_wav, ainfo, w_ell, [0, 2])
print('Wavelet noise', time.time() - t0)

# Draw pixel-based noise alms.
t0 = time.time()
alm_noise_pix = alm_utils.rand_alm_pix(cov_pix, ainfo, minfo, [0, 2])
print('Pixel noise', time.time() - t0)

# Draw map-based noise and add to alm.
#noise = map_utils.rand_map_pix(cov_pix)
#alm_signal = alm.copy()
#alm_noise = alm.copy()
#sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=False)

# Apply mask.
noise = np.zeros_like(icov_pix)
sht.alm2map(alm_noise_wav, noise, ainfo, minfo, [0,2], adjoint=False)
noise *= mask
sht.map2alm(noise, alm_noise_wav, minfo, ainfo, [0,2], adjoint=False)

sht.alm2map(alm_noise_pix, noise, ainfo, minfo, [0,2], adjoint=False)
noise *= mask
sht.map2alm(noise, alm_noise_pix, minfo, ainfo, [0,2], adjoint=False)

nl_wav = ainfo.alm2cl(alm_noise_wav[:,None,:], alm_noise_wav[None,:,:])
nl_pix = ainfo.alm2cl(alm_noise_pix[:,None,:], alm_noise_pix[None,:,:])

# Save numpy arrays.
np.save(opj(basedir, 'n_ell_wav'), nl_wav)
np.save(opj(basedir, 'n_ell_pix'), nl_pix)
np.save(opj(basedir, 's_ell'), cov_ell)
np.save(opj(basedir, 'fsky'), np.asarray([fsky]))

exit()

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(nl_wav[idxs], label='wav')
    axs[idxs].plot(nl_pix[idxs], label='pix')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'n_ell'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(ells, dells * nl_wav[idxs], label='wav')
    axs[idxs].plot(ells, dells * nl_pix[idxs], label='pix')
    axs[idxs].plot(ells, dells * cov_ell[idxs], label='signal')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'tot_ell'))
plt.close(fig)

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
    
# Plot alms
omap_noise_wav = curvedsky.alm2map(alm_noise_wav, omap.copy())
omap_noise_pix = curvedsky.alm2map(alm_noise_pix, omap.copy())

for pidx in range(alm.shape[0]):

    plot = enplot.plot(omap_noise_wav[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, 'alm_noise_wav_{}'.format(pidx)), plot)

    plot = enplot.plot(omap_noise_pix[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, 'alm_noise_pix_{}'.format(pidx)), plot)
