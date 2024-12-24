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

lmax = 5800

basedir = '/home/adriaand/project/actpol/20201202_pcg_act_wavelet'
imgdir = opj(basedir, 'img')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')
areadir = '/home/adriaand/project/actpol/mapdata/area/'

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))
icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest')

# Set too small values to zero.
#mask = icov_pix > 1e-4
mask = icov_pix > 1e-3 # Use for lmax=500
#mask = icov_pix > 0.5 # Use for lmax=500
icov_pix[~mask] = 0

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
#cov_pix[~mask] = 0

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
lamb = 1.5
lmin = 100
#lmin = lmax

w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax, j0=None, lmin=lmin, return_j=True)
print('lmaxs', lmaxs)
print('js', j_scales)

# NOTE NOTE
#w_ell = np.ones((1, lmax+1))

ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
#ax.set_xscale('log')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels_me'))
plt.close(fig)

noisebox = enmap.read_fits(opj(metadir, 'noisebox_f150_night.fits'))
# Sum over arrays.
noisebox = np.sum(noisebox, axis=1)
#noisebox = np.sum(noisebox[:,:-3,...], axis=1) # No Planck.

# NOTE NOTE
#noisebox[:] = noisebox[:,-1,:,:][:,np.newaxis,:,:]
#noisebox *= (bins[-1]/10800) ** 2
#noisebox *= (bins[-1]/10800)
#noisebox[:] = np.mean(noisebox, axis=1)[:,np.newaxis,:,:]

t0 = time.time()
#icov_wav = noise_utils.noisebox2wavmat(noisebox.copy(), bins, w_ell)
icov_wav = noise_utils.noisebox2wavmat(noisebox.copy(), bins, w_ell, offsets=[0]) # NOTE
print(time.time() - t0)

wavelet_matrix = np.einsum('ij,kj->ikj', w_ell, w_ell, optimize=True)
wavelet_matrix *= (2 * ells + 1) / 4 / np.pi

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    for wpidx in range(w_ell.shape[0]):
        ax.plot(wavelet_matrix[widx,wpidx], label='{},{}'.format(widx,wpidx))
#ax.set_xscale('log')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'wavelet_matrix'))
plt.close(fig)

for index in icov_wav.indices:
    for pidx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(icov_wav.maps[tuple(index)][pidx].reshape(
            (icov_wav.minfos[tuple(index)].nrow, icov_wav.minfos[tuple(index)].nphi[0])))
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(imgdir, 'icov_wav_{}_{}_{}'.format(index[0], index[1], pidx)))
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

# Draw alms.
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
for pidx in range(alm.shape[0]):
    hp.almxfl(alm[pidx], b_ell[pidx], inplace=True)
# Draw map-based noise and add to alm.
noise = map_utils.rand_map_pix(cov_pix)
alm_signal = alm.copy()
alm_noise = alm.copy()
sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=False)
nl = ainfo.alm2cl(alm_noise[:,None,:], alm_noise[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(nl[idxs])
fig.savefig(opj(imgdir, 'n_ell'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(ells, dells * (cov_ell[idxs] + nl[idxs]))
    axs[idxs].plot(ells, dells * nl[idxs])
    axs[idxs].plot(ells, dells * cov_ell[idxs])
fig.savefig(opj(imgdir, 'tot_ell'))
plt.close(fig)

alm += alm_noise
# Apply mask.
sht.alm2map(alm, noise, ainfo, minfo, [0,2], adjoint=False)
noise *= mask
sht.map2alm(noise, alm, minfo, ainfo, [0,2], adjoint=False)

icov_noise_wav = operators.WavMatVecAlm(ainfo, icov_wav, w_ell, [0,2])
icov_noise_pix = operators.PixMatVecAlm(ainfo, icov_pix, minfo, [0,2])

# Plot result
t0 = time.time()
print('start')
alm_icov_wav = icov_noise_wav(alm)
print('icov_wav : {}'.format(time.time() - t0))
t0 = time.time()
alm_icov_pix = icov_noise_pix(alm)
print('icov_pix : {}'.format(time.time() - t0))
alm_signal_icov_wav = icov_noise_wav(alm_signal)
alm_signal_icov_pix = icov_noise_pix(alm_signal)

cl_icov_wav = ainfo.alm2cl(alm_icov_wav[:,None,:], alm_icov_wav[None,:,:])
cl_icov_pix = ainfo.alm2cl(alm_icov_pix[:,None,:], alm_icov_pix[None,:,:])
cl_signal_icov_wav = ainfo.alm2cl(alm_signal_icov_wav[:,None,:], alm_signal_icov_wav[None,:,:])
cl_signal_icov_pix = ainfo.alm2cl(alm_signal_icov_pix[:,None,:], alm_signal_icov_pix[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(dells * cl_icov_wav[idxs], label='icov_wav')
    axs[idxs].plot(dells * cl_icov_pix[idxs], label='icov_pix')
    axs[idxs].plot(dells * cl_signal_icov_wav[idxs], label='signal icov_wav', lw=0.5)
    axs[idxs].plot(dells * cl_signal_icov_pix[idxs], label='signal icov_pix', lw=0.5)
    if idxs[0] == idxs[1]:
        axs[idxs].set_yscale('log')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'c_ell_full'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(dells * cl_signal_icov_pix[idxs], label='signal icov_pix', lw=0.5)
    axs[idxs].plot(dells * cl_signal_icov_wav[idxs], label='signal icov_wav', lw=0.5)
    if idxs[0] == idxs[1]:
        axs[idxs].set_yscale('log')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'c_ell_signal_only_full'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(dells * cl_icov_wav[idxs], label='icov_wav')
    axs[idxs].plot(dells * cl_signal_icov_wav[idxs], label='signal icov_wav', lw=0.5)
    if idxs[0] == idxs[1]:
        axs[idxs].set_yscale('log')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'c_ell_wav_only_full'))
plt.close(fig)

# Load areas
area_deep56 = enmap.read_map(opj(areadir, 'deep56.fits'))
omap_d56 = enmap.zeros((3,) + area_deep56.shape[-2:], area_deep56.wcs)

alm_icov_wav_in = icov_noise_wav(alm)
alm_icov_pix_in = icov_noise_pix(alm)
alm_signal_icov_wav_in = icov_noise_wav(alm_signal)
alm_signal_icov_pix_in = icov_noise_pix(alm_signal)

curvedsky.alm2map(alm_icov_wav_in, omap_d56, ainfo=ainfo)
alm_icov_wav = curvedsky.map2alm(omap_d56, ainfo=ainfo)
curvedsky.alm2map(alm_icov_pix_in, omap_d56, ainfo=ainfo)
alm_icov_pix = curvedsky.map2alm(omap_d56, ainfo=ainfo)
curvedsky.alm2map(alm_signal_icov_wav_in, omap_d56, ainfo=ainfo)
alm_signal_icov_wav = curvedsky.map2alm(omap_d56, ainfo=ainfo)
curvedsky.alm2map(alm_signal_icov_pix_in, omap_d56, ainfo=ainfo)
alm_signal_icov_pix = curvedsky.map2alm(omap_d56, ainfo=ainfo)

cl_icov_wav = ainfo.alm2cl(alm_icov_wav[:,None,:], alm_icov_wav[None,:,:])
cl_icov_pix = ainfo.alm2cl(alm_icov_pix[:,None,:], alm_icov_pix[None,:,:])
cl_signal_icov_wav = ainfo.alm2cl(alm_signal_icov_wav[:,None,:], alm_signal_icov_wav[None,:,:])
cl_signal_icov_pix = ainfo.alm2cl(alm_signal_icov_pix[:,None,:], alm_signal_icov_pix[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(dells * cl_icov_wav[idxs], label='icov_wav')
    axs[idxs].plot(dells * cl_icov_pix[idxs], label='icov_pix')
    axs[idxs].plot(dells * cl_signal_icov_wav[idxs], label='signal icov_wav', lw=0.5)
    axs[idxs].plot(dells * cl_signal_icov_pix[idxs], label='signal icov_pix', lw=0.5)
    if idxs[0] == idxs[1]:
        axs[idxs].set_yscale('log')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'c_ell_deep56'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(dells * cl_signal_icov_pix[idxs], label='signal icov_pix', lw=0.5)
    axs[idxs].plot(dells * cl_signal_icov_wav[idxs], label='signal icov_wav', lw=0.5)
    if idxs[0] == idxs[1]:
        axs[idxs].set_yscale('log')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'c_ell_signal_only_deep56'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(dells * cl_icov_wav[idxs], label='icov_wav')
    axs[idxs].plot(dells * cl_signal_icov_wav[idxs], label='signal icov_wav', lw=0.5)
    if idxs[0] == idxs[1]:
        axs[idxs].set_yscale('log')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'c_ell_wav_only_deep56'))
plt.close(fig)

# Plot alms
omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
omap = curvedsky.alm2map(alm_signal, omap)
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:10')
enplot.write(opj(imgdir, 'alm_signal'), plot)

omap = curvedsky.alm2map(alm, omap)
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:10')
enplot.write(opj(imgdir, 'alm_in'), plot)

omap = curvedsky.alm2map(alm_icov_wav_in, omap)
for pidx in range(3):
    plot = enplot.plot(omap[pidx], colorbar=True, font_size=50, grid=False)
    #plot = enplot.plot(omap[pidx], font_size=50, grid=False)
    enplot.write(opj(imgdir, 'alm_icov_noise_{}'.format(pidx)), plot)

omap = curvedsky.alm2map(alm_icov_pix_in, omap)
for pidx in range(3):
    plot = enplot.plot(omap[pidx], colorbar=True, font_size=50, grid=False)
    enplot.write(opj(imgdir, 'alm_icov_noise_pix_{}'.format(pidx)), plot)

omap = curvedsky.alm2map(alm_signal_icov_wav_in, omap)
for pidx in range(3):
    plot = enplot.plot(omap[pidx], colorbar=True, font_size=50, grid=False)
    #plot = enplot.plot(omap[pidx], font_size=50, grid=False)
    enplot.write(opj(imgdir, 'alm_icov_noise_signal_{}'.format(pidx)), plot)

omap = curvedsky.alm2map(alm_signal_icov_pix_in, omap)
for pidx in range(3):
    plot = enplot.plot(omap[pidx], colorbar=True, font_size=50, grid=False)
    enplot.write(opj(imgdir, 'alm_icov_noise_signal_pix_{}'.format(pidx)), plot)

