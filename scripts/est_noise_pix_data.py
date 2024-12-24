'''
Load real data maps, and look at wavelet maps.
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

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

lmax = 1000
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

imapdir = '/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019'
maskdir = '/projects/ACT/zatkins/sync/20201207/masks/masks_20200723'
odir = '/home/adriaand/project/actpol/20210402_est_noise_pix_data'
imgdir = opj(odir, 'img')

os.makedirs(imgdir, exist_ok=True)

# Load map and mask
imap = enmap.read_fits(opj(imapdir, 's15_deep56_pa3_f150_nohwp_night_3pass_4way_set0_map_srcfree.fits'))
imap -= enmap.read_fits(opj(imapdir, 's15_deep56_pa3_f150_nohwp_night_3pass_4way_set1_map_srcfree.fits'))

mask = enmap.read_fits(opj(maskdir, 'deep56.fits'))
mask = enmap.project(mask, imap.shape, imap.wcs, order=1)
#mask[mask!=1] = 0
mask_bool = mask != 0

# NOTE NOTE
std = np.std(imap[:,mask_bool])
imap[:] = np.random.randn(*imap.shape)
imap *= std

lims = [500, 200, 200]

for pidx in range(imap.shape[0]):

    plot = enplot.plot(imap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'imap_in_{pidx}'), plot)

imap *= mask

for pidx in range(imap.shape[0]):

    plot = enplot.plot(imap[pidx], colorbar=True, grid=False, range=lims[pidx])
    enplot.write(opj(imgdir, f'imap_mask_{pidx}'), plot)

imap[...,~mask_bool] = np.nan

# Downgrade map by factor 2
imap = enmap.downgrade(imap, 2, op=np.nanmean)
mask = enmap.downgrade(mask, 2, op=np.mean)
mask_dg = ~np.isnan(imap)
imap[~mask_dg] = 0

mask_band = scipy.ndimage.distance_transform_edt(~mask_dg) < 200
mask_band *= (scipy.ndimage.distance_transform_edt(mask_dg) < 20)
#mask_band = mask_band[:,:,::-1]
mask_band = enmap.enmap(mask_band, wcs=imap.wcs, dtype=np.float32)
mask_band = mask_band[:1]

ainfo = sharp.alm_info(lmax)
b_ell = hp.gauss_beam(np.radians(2), lmax=lmax)

alm = np.empty((1, ainfo.nelem), dtype=np.complex64)
curvedsky.map2alm(mask_band, alm, ainfo)
alm_c_utils.lmul(alm, b_ell.astype(np.float32), ainfo, inplace=True)
curvedsky.alm2map(alm, mask_band, ainfo=ainfo)

mask_band /= np.max(mask_band)

plot = enplot.plot(mask_band, colorbar=True, grid=False)
enplot.write(opj(imgdir, f'mask_band_0'), plot)

# Plot map
for pidx in range(imap.shape[0]):

    plot = enplot.plot(imap[pidx], colorbar=True, grid=False, range=lims[pidx])
    enplot.write(opj(imgdir, f'imap_dg_{pidx}'), plot)

# map2alm
ainfo = sharp.alm_info(lmax)
alm = np.empty((3, ainfo.nelem), dtype=np.complex64)
curvedsky.map2alm(imap, alm, ainfo)

n_ell = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(n_ell[idxs])
    if idxs[0] == idxs[1]:
        ax.set_yscale('log')
    ax.set_xscale('log')
fig.savefig(opj(imgdir, 'n_ell'))
plt.close(fig)

omap = np.empty_like(imap)
curvedsky.alm2map(alm, omap, ainfo=ainfo)

for pidx in range(imap.shape[0]):

    plot = enplot.plot(omap[pidx], colorbar=True, grid=False, range=lims[pidx])
    enplot.write(opj(imgdir, f'omap_{pidx}'), plot)

# alm2w
lamb = 1.7
#lamb = 10
lmin = 25
#lmin = 999
#lmax_w = 5000
#lmax_j = 4500
lmax_w = 1000
lmax_j = 900

w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax_w, j0=None, lmin=lmin, return_j=True, lmax_j=lmax_j)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
ax.set_xscale('log')
fig.savefig(opj(imgdir, 'kernels_log'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels'))
plt.close(fig)

alm_trunc, ainfo_trunc = alm_utils.trunc_alm(alm, ainfo, lmax_w)
w_ell = w_ell.astype(np.float32)

#wav = wavtrans.alm2wav(alm_trunc, ainfo_trunc, [0,2], w_ell, wav=None, adjoint=False, lmaxs=None)
wlms, winfos = alm_utils.alm2wlm_axisym(alm_trunc, ainfo_trunc, w_ell)

# Test roundtrip
alm_rt, _ = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell, ainfo=ainfo)
omap_rt = curvedsky.alm2map(alm_rt, imap.copy(), ainfo=ainfo)

for pidx in range(imap.shape[0]):
    plot = enplot.plot(omap_rt[pidx], colorbar=True, grid=False, range=lims[pidx])
    enplot.write(opj(imgdir, f'omap_rt_{pidx}'), plot)


#for widx in range(wav.shape[0]):

wav_sm = wavtrans.Wav(2, dtype=np.float32)

for widx, (wlm, winfo) in enumerate(zip(wlms, winfos)):

    omap = np.empty_like(imap)
    curvedsky.alm2map(wlm, omap, ainfo=winfo)

    for pidx in range(imap.shape[0]):

        plot = enplot.plot(omap[pidx], colorbar=True, grid=False)
        enplot.write(opj(imgdir, f'wav_{widx}_{pidx}'), plot)

        plot = enplot.plot(omap[pidx] ** 2, colorbar=True, grid=False)
        enplot.write(opj(imgdir, f'wavsq_{widx}_{pidx}'), plot)

    if widx < 4:
        w_ell_j, _, _ = wlm_utils.get_sd_kernels(
            lamb, winfo.lmax, j0=None, lmin=25, return_j=True)
    else:
        w_ell_j, _, _ = wlm_utils.get_sd_kernels(
            lamb, winfo.lmax, j0=None, lmin=lmin, return_j=True)

    w_ell_j = w_ell_j.astype(np.float32)
    wlm_sq = np.empty((3, winfo.nelem), dtype=np.complex64)
    curvedsky.map2alm(omap ** 2, wlm_sq, winfo, spin=0)

    ##
    #b_ell = hp.gauss_beam(np.radians(1), lmax=winfo.lmax)
    #alm_c_utils.lmul(wlm_sq, b_ell.astype(np.float32), winfo, inplace=True)
    #minfo = sharp.map_info_gauss_legendre(winfo.lmax+1, nphi=2*winfo.lmax+1)
    #omap_gl = np.empty((3, minfo.npix), np.float32)
    #sht.alm2map(wlm_sq, omap_gl, winfo, minfo, 0)
    #omap_gl = mat_utils.get_near_psd(omap_gl)

    ##
    wlms_j, winfos_j = alm_utils.alm2wlm_axisym(wlm_sq, winfo, w_ell_j)

    for widx_j, (wlm_j, winfo_j) in enumerate(zip(wlms_j, winfos_j)):
        omap_j = np.empty_like(imap)
        curvedsky.alm2map(wlm_j, omap_j, ainfo=winfo_j, spin=0)

        # mask omap_j around edge of map.
        if widx_j > 3:
            omap_j *= mask_band
            # map2lalm tinto wlm_sm
            curvedsky.map2alm(omap_j, wlm_j, winfo_j, spin=0)

        for pidx in range(imap.shape[0]):
            
            plot = enplot.plot(omap_j[pidx], colorbar=True, grid=False)
            enplot.write(opj(imgdir, f'wavwavsq_{widx}_{widx_j}_{pidx}'), plot)

    wlm_sm, _ = alm_utils.wlm2alm_axisym(wlms_j[:3], winfos_j[:3], w_ell_j[:3], ainfo=winfo)
    omap = np.empty_like(imap)
    curvedsky.alm2map(wlm_sm, omap, ainfo=winfo, spin=0)

    for pidx in range(imap.shape[0]):

        plot = enplot.plot(omap[pidx], colorbar=True, grid=False)
        enplot.write(opj(imgdir, f'wav_sm_{widx}_{pidx}'), plot)

    # minfo = sharp.map_info_gauss_legendre(winfo.lmax+1, nphi=2*winfo.lmax+1)
    # omap_gl = np.empty((3, minfo.npix), np.float32)
    # sht.alm2map(wlm_sm, omap_gl, winfo, minfo, 0)
    # omap_gl = mat_utils.get_near_psd(omap_gl)
    # #omap_gl[omap_gl < 0] = 0
    ##

    #wav_sm.add((widx, widx), omap_gl, minfo)

exit()

rand_alm = alm_utils.rand_alm_wav(wav_sm, ainfo, w_ell, [0,2])

omap_rand = curvedsky.alm2map(rand_alm, imap.copy(), ainfo=ainfo)

for pidx in range(imap.shape[0]):

    plot = enplot.plot(omap_rand[pidx], colorbar=True, grid=False, range=lims[pidx])
    enplot.write(opj(imgdir, f'omap_rand_{pidx}'), plot)

omap_rand *= mask

for pidx in range(imap.shape[0]):

    plot = enplot.plot(omap_rand[pidx], colorbar=True, grid=False, range=lims[pidx])
    enplot.write(opj(imgdir, f'omap_rand_masked_{pidx}'), plot)

curvedsky.map2alm(omap_rand, rand_alm, ainfo)

n_ell_out = ainfo.alm2cl(rand_alm[:,None,:], rand_alm[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(n_ell_out[idxs])
    axs[idxs].plot(n_ell[idxs], ls=':')
    if idxs[0] == idxs[1]:
        ax.set_yscale('log')
    ax.set_xscale('log')
fig.savefig(opj(imgdir, 'n_ell_out'))
plt.close(fig)


