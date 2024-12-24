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
from optweight import noise_utils, alm_utils, mat_utils, alm_c_utils, wavtrans, type_utils

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def rolling_average(arr, window_size):

    if arr.ndim == 1:
        arr = arr[np.newaxis,:]

    out = np.zeros(arr.shape, arr.dtype)
    window = np.ones(window_size) / window_size

    for idxs in np.ndindex(arr.shape[:-1]):
        out[idxs] = np.convolve(arr[idxs], window, mode='same')
    return out

opj = os.path.join
np.random.seed(201)
#np.random.seed(1)

lmax = 2000
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

imapdir = '/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019'
maskdir = '/projects/ACT/zatkins/sync/20201207/masks/masks_20200723'
odir = '/home/adriaand/project/actpol/20210420_est_noise_pix_data'
imgdir = opj(odir, 'img')

os.makedirs(imgdir, exist_ok=True)

# Load map and mask
#imap = enmap.read_fits(opj(imapdir, 's15_deep56_pa3_f150_nohwp_night_3pass_4way_set0_map_srcfree.fits'))
#imap -= enmap.read_fits(opj(imapdir, 's15_deep56_pa3_f150_nohwp_night_3pass_4way_set1_map_srcfree.fits'))

imap = enmap.read_fits(opj(imapdir, 's19_cmb_pa5_f150_nohwp_day_1pass_2way_set0_map.fits'))
imap -= enmap.read_fits(opj(imapdir, 's19_cmb_pa5_f150_nohwp_day_1pass_2way_set1_map.fits'))

#imap = enmap.read_fits(opj(imapdir, 's19_cmb_pa4_f220_nohwp_night_1pass_4way_set0_map.fits'))
#imap -= enmap.read_fits(opj(imapdir, 's19_cmb_pa4_f220_nohwp_night_1pass_4way_set1_map.fits'))

#ivar = enmap.read_fits(opj(imapdir, 's19_cmb_pa5_f150_nohwp_day_1pass_2way_set0_ivar.fits'))

#mask = enmap.read_fits(opj(maskdir, 'deep56.fits'))
mask = enmap.read_fits('/home/adriaand/project/actpol/maps/dr6sims/masks/v1/BN_bottomcut.fits')
mask = enmap.project(mask, imap.shape, imap.wcs, order=1)

imap = enmap.downgrade(imap, 4, op=np.nanmean)
mask = enmap.downgrade(mask, 4, op=np.mean)
#ivar = enmap.downgrade(ivar, 4, op=np.sum)

#print(ivar.shape)

ainfo = sharp.alm_info(lmax)
alm = np.empty((imap.shape[0], ainfo.nelem), dtype=type_utils.to_complex(imap.dtype))
mask_large = enmap.grow_mask(mask.astype(bool), np.radians(0.5))
mask_large = mask_large.astype(np.float32)
alm = curvedsky.map2alm(mask_large, alm, ainfo)
b_ell = hp.gauss_beam(np.radians(0.5), lmax=ainfo.lmax)
alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
mask_large = curvedsky.alm2map(alm, mask_large, ainfo=ainfo)
imap *= mask_large

#print(ivar.shape)
#ivar[ivar < ivar.max() * 0.02] = 0
#plot = enplot.plot(ivar, colorbar=True, grid=False)
#enplot.write(opj(imgdir, f'ivar'), plot)

#imap[:,ivar == 0] = 0

#imap = enmap.inpaint(imap, ~(mask != 0))

mask_gl, minfo = map_utils.enmap2gauss(mask, 2*lmax, order=1)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(map_utils.view_2d(mask_gl, minfo))
colorbar(im)
fig.savefig(opj(imgdir, 'mask_gl'))
plt.close(fig)

#features = mask_gl < 0.5
#features = features.astype(np.float32)

#minfo_features = map_utils.get_gauss_minfo(lmax)
#features = map_utils.gauss2gauss(features, minfo, minfo_features, order=1)

features = None
minfo_features = None


sht.map2alm(features, alm[:1], minfo_features, ainfo, 0)
b_ell_f = hp.gauss_beam(np.radians(30), lmax=ainfo.lmax)
alm_c_utils.lmul(alm, b_ell_f, ainfo, inplace=True)
sht.alm2map(alm[:1], features, ainfo, minfo_features, 0)

features /= features.max()

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(map_utils.view_2d(features, minfo_features))
colorbar(im)
fig.savefig(opj(imgdir, 'features'))
plt.close(fig)


#icov_gl, _ = map_utils.enmap2gauss(ivar, 2*lmax, order=1, area_pow=1)

#icov_gl = map_utils.round_icov_matrix(icov_gl[np.newaxis,:], rtol=1e-3)

#mask_dg = ~np.isnan(imap)
#imap[~mask_dg] = 0

#mask[mask!=1] = 0
#mask_bool = mask != 0
#imap *= mask

lims = [500, 200, 200]

#for pidx in range(imap.shape[0]):
#    plot = enplot.plot(imap[pidx], colorbar=True, grid=False, range=lims[pidx])
#    enplot.write(opj(imgdir, f'imap_in_{pidx}'), plot)

ainfo = sharp.alm_info(lmax)
#minfo = map_utils.get_enmap_minfo(imap.shape, imap.wcs, 2 * lmax)

alm = np.empty((imap.shape[0], ainfo.nelem), dtype=type_utils.to_complex(imap.dtype))
curvedsky.map2alm(imap, alm, ainfo)

n_ell_unmasked = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])

alm_imap = alm.copy()

# alm2w
#lamb = 1.2
#lamb = 1.4
lamb = 1.3
lmin = 10
lmax_w = lmax
lmax_j = lmax-100

w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax_w, j0=None, lmin=lmin, return_j=True, lmax_j=lmax_j)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels'))
ax.set_xscale('log')
fig.savefig(opj(imgdir, 'kernels_log'))
plt.close(fig)

t0 = time.time()
cov_wav = noise_utils.estimate_cov_wav(alm, ainfo, w_ell, [0, 2], diag=True,
                                       features=features, minfo_features=minfo_features)
print(time.time() - t0)

#alm_draw = alm_utils.rand_alm_wav(cov_wav, ainfo, w_ell, [0, 2])
#print(time.time() - t0)

sqrt_cov_wav = operators.WavMatVecWav(cov_wav, power=0.5, inplace=True)
print(time.time() - t0)
wav_uni = noise_utils.unit_var_wav(cov_wav.get_minfos_diag(), (3,), np.float32)
print(time.time() - t0)
rand_wav = sqrt_cov_wav(wav_uni)
print(time.time() - t0)
alm_draw, _ = wavtrans.wav2alm(rand_wav, alm.copy(), ainfo, [0, 2], w_ell)
print(time.time() - t0)


omap_gl = np.empty((alm.shape[0], minfo.npix), dtype=np.float32)
sht.alm2map(alm, omap_gl, ainfo, minfo, [0, 2])
omap_gl *= mask_gl
sht.map2alm(omap_gl, alm, minfo, ainfo, [0, 2])

n_ell = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
n_ell_draw = ainfo.alm2cl(alm_draw[:,None,:], alm_draw[None,:,:])

imap_masked = curvedsky.alm2map(alm, imap.copy(), ainfo=ainfo)
for pidx in range(imap_masked.shape[0]):
    plot = enplot.plot(imap_masked[pidx], colorbar=True, grid=False, range=lims[pidx], font_size=100)
    enplot.write(opj(imgdir, f'imap_masked_{pidx}'), plot)

#mask_gl[mask_gl != 0] = 1
omap_gl = np.empty((alm_draw.shape[0], minfo.npix), dtype=np.float32)
sht.alm2map(alm_draw, omap_gl, ainfo, minfo, [0, 2])
omap_gl *= mask_gl
sht.map2alm(omap_gl, alm_draw, minfo, ainfo, [0, 2])

map_draw = curvedsky.alm2map(alm_draw, imap.copy(), ainfo=ainfo)
#map_draw *= mask
#alm_draw_masked = curvedsky.map2alm(map_draw, alm_draw, ainfo)

n_ell_draw_masked = ainfo.alm2cl(alm_draw[:,None,:], alm_draw[None,:,:])

for pidx in range(map_draw.shape[0]):
    plot = enplot.plot(map_draw[pidx], colorbar=True, grid=False, range=lims[pidx], font_size=100)
    enplot.write(opj(imgdir, f'map_draw_{pidx}'), plot)


# fig, ax = plt.subplots(dpi=300, constrained_layout=True)
# ax.plot(ells, n_ell[0,0], label='n_ell')
# ax.plot(ells, n_ell_draw[0,0], label='n_ell_draw')
# ax.plot(ells, n_ell_draw_masked[0,0], label='n_ell_draw_masked')
# ax.legend()
# fig.savefig(opj(imgdir, 'n_ell_out'))
# plt.close(fig)


# fig, ax = plt.subplots(dpi=300, constrained_layout=True)
# ax.plot(ells, n_ell[0,0], label='n_ell')
# ax.plot(ells, n_ell_draw[0,0], label='n_ell_draw')
# ax.plot(ells, n_ell_draw_masked[0,0], label='n_ell_draw_masked')
# ax.legend()
# ax.set_yscale('log')
# fig.savefig(opj(imgdir, 'n_ell_out_log'))
# plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    ax.plot(n_ell[idxs], label='n_ell')
    ax.plot(n_ell_draw[idxs], label='n_ell_draw')
    ax.plot(n_ell_draw_masked[idxs], label='n_ell_draw_masked')
    ax.set_xscale('log')
axs[0,2].legend()
fig.savefig(opj(imgdir, 'n_ell_out'))

for idxs, ax in np.ndenumerate(axs):
    if idxs[0] == idxs[1]:
        ax.set_yscale('log')
fig.savefig(opj(imgdir, 'n_ell_out_log'))
plt.close(fig)

fig, axs = plt.subplots(nrows=3, dpi=300, constrained_layout=True, sharex=True)
for aidx, ax in enumerate(axs):
    ax.plot(n_ell_draw_masked[aidx,aidx], label='sim')
    ax.plot(n_ell[aidx,aidx], label='input', lw=0.5)
    ax.plot(n_ell_draw[aidx,aidx], label='n_ell_draw', lw=0.5)

for ax in axs:
    ax.set_xscale('log')
axs[2].set_xlabel(r'$\ell$')
axs[0].set_ylabel(r'$C_{\ell}^{TT}$')
axs[1].set_ylabel(r'$C_{\ell}^{EE}$')
axs[2].set_ylabel(r'$C_{\ell}^{BB}$')
axs[2].legend()
fig.savefig(opj(imgdir, 'n_ell_out_auto'))

for ax in axs:
    ax.set_yscale('log')
fig.savefig(opj(imgdir, 'n_ell_out_log_auto'))
plt.close(fig)


# apply icov_ell to input noise map
#n_ell_unmasked_sm = rolling_average(n_ell_unmasked, 20)
n_ell_unmasked_sm = n_ell_unmasked * np.eye(3)[:,:,np.newaxis]

#icov_gl_ext = np.zeros((3, minfo.npix), dtype=icov_gl.dtype)
#icov_gl_ext[:] = icov_gl[np.newaxis,:]

#sqrticov_pix = operators.PixMatVecAlm(ainfo, icov_gl_ext, minfo, [0,2], power=0.5, inplace=False, use_weights=True)
#sqrtcov_pix = operators.PixMatVecAlm(ainfo, icov_gl_ext, minfo, [0,2], power=-0.5, inplace=False, use_weights=True)

icov_ell = operators.EllMatVecAlm(ainfo, n_ell_unmasked_sm, power=-0.5)
cov_ell = operators.EllMatVecAlm(ainfo, n_ell_unmasked_sm, power=0.5)

#alm_flat = sqrticov_pix(alm_imap)
alm_flat = icov_ell(alm_imap)

imap_flat = curvedsky.alm2map(alm_flat, imap.copy(), ainfo=ainfo)
for pidx in range(imap_flat.shape[0]):
    plot = enplot.plot(imap_flat[pidx], colorbar=True, grid=False, font_size=100)
    enplot.write(opj(imgdir, f'imap_flat_{pidx}'), plot)

#imap_unflat = curvedsky.alm2map(sqrtcov_pix(cov_ell(alm_flat)), imap.copy(), ainfo=ainfo)
#for pidx in range(imap_unflat.shape[0]):
#    plot = enplot.plot(imap_unflat[pidx], colorbar=True, grid=False, font_size=100)
#    enplot.write(opj(imgdir, f'imap_unflat_{pidx}'), plot)

# estimate cov
# draw
cov_wav_flat = noise_utils.estimate_cov_wav(alm_flat, ainfo, w_ell, [0, 2], diag=True,
                                            features=features, minfo_features=minfo_features)

n_ell_draw_masked_flat = np.zeros_like(n_ell)

niter = 1
for idx in range(niter):
    alm_draw_flat = alm_utils.rand_alm_wav(cov_wav_flat, ainfo, w_ell, [0, 2])

    omap_flat = curvedsky.alm2map(alm_draw_flat, imap.copy(), ainfo=ainfo)
    if idx == 0:
        for pidx in range(omap_flat.shape[0]):
            plot = enplot.plot(omap_flat[pidx], colorbar=True, grid=False, font_size=100)
            enplot.write(opj(imgdir, f'omap_flat_{pidx}'), plot)

    n_ell_draw_flat = ainfo.alm2cl(alm_draw_flat[:,None,:], alm_draw_flat[None,:,:])

    alm_draw = cov_ell(alm_draw_flat)
    #alm_draw = sqrtcov_pix(alm_draw)

    #alm_draw = cov_ell(alm_draw_flat)

    omap_gl = np.empty((alm_draw.shape[0], minfo.npix), dtype=np.float32)
    sht.alm2map(alm_draw, omap_gl, ainfo, minfo, [0, 2])
    omap_gl *= mask_gl
    sht.map2alm(omap_gl, alm_draw, minfo, ainfo, [0, 2])

    map_draw = curvedsky.alm2map(alm_draw, imap.copy(), ainfo=ainfo)
#map_draw *= mask
#alm_draw_masked = curvedsky.map2alm(map_draw, alm_draw, ainfo)

    #n_ell_draw_masked_flat = ainfo.alm2cl(alm_draw[:,None,:], alm_draw[None,:,:])
    n_ell_draw_masked_flat += ainfo.alm2cl(alm_draw[:,None,:], alm_draw[None,:,:])
n_ell_draw_masked_flat /= niter

for pidx in range(map_draw.shape[0]):
    plot = enplot.plot(map_draw[pidx], colorbar=True, grid=False, range=lims[pidx], font_size=100)
    enplot.write(opj(imgdir, f'map_draw_flat_{pidx}'), plot)

fig, axs = plt.subplots(nrows=3, dpi=300, constrained_layout=True, sharex=True)
for aidx, ax in enumerate(axs):
    ax.plot(n_ell[aidx,aidx], label='data', lw=0.5)
    ax.plot(n_ell_draw_masked[aidx,aidx], label='sim without flattening', lw=0.5)
    ax.plot(n_ell_draw_masked_flat[aidx,aidx], label='sim with flattening', lw=0.5)
    #ax.plot(n_ell_draw_flat[aidx,aidx], label='n_ell_draw_flat', lw=0.5)
for ax in axs:
    ax.set_xscale('log')
axs[2].set_xlabel(r'$\ell$')
axs[0].set_ylabel(r'$C_{\ell}^{TT}$')
axs[1].set_ylabel(r'$C_{\ell}^{EE}$')
axs[2].set_ylabel(r'$C_{\ell}^{BB}$')
axs[2].legend()
fig.savefig(opj(imgdir, 'n_ell_out_auto'))

for ax in axs:
    ax.set_yscale('log')
fig.savefig(opj(imgdir, 'n_ell_out_log_auto_flat'))
plt.close(fig)


fig, axs = plt.subplots(nrows=3, dpi=300, constrained_layout=True, sharex=True)
for aidx, ax in enumerate(axs):
    #ax.plot(n_ell[aidx,aidx], label='input', lw=0.5)
    ax.plot(n_ell_draw_masked[aidx,aidx] / n_ell[aidx,aidx], label='sim without flattening', color='C1', lw=0.5)
    ax.plot(n_ell_draw_masked_flat[aidx,aidx] / n_ell[aidx,aidx], label='sim with flattening', lw=0.5, color='C2')
    #ax.plot(n_ell_draw_flat[aidx,aidx], label='n_ell_draw_flat', lw=0.5)
for ax in axs:
    ax.set_xscale('log')
axs[2].set_xlabel(r'$\ell$')
axs[0].set_ylabel(r'$C_{\ell}^{TT}$')
axs[1].set_ylabel(r'$C_{\ell}^{EE}$')
axs[2].set_ylabel(r'$C_{\ell}^{BB}$')
for ax in axs:
    ax.axhline(y=1, ls=':', color='black', alpha=0.8)

axs[2].legend()
fig.savefig(opj(imgdir, 'n_ell_out_auto_ratio'))

for ax in axs:
    ax.set_yscale('log')
fig.savefig(opj(imgdir, 'n_ell_out_log_auto_flat_ratio'))
plt.close(fig)

fig, axs = plt.subplots(nrows=3, ncols=3, dpi=300, constrained_layout=True, sharex=True)
for idxs, ax in np.ndenumerate(axs):

    ax.plot(n_ell[idxs], label='data', lw=0.5)
    ax.plot(n_ell_draw_masked[idxs], label='sim without flattening', lw=0.5)
    ax.plot(n_ell_draw_masked_flat[idxs], label='sim with flattening', lw=0.5)

    ax.set_xscale('log')

axs[0,2].legend()
fig.savefig(opj(imgdir, 'n_ell_out_flat'))

for idxs, ax in np.ndenumerate(axs):
    if idxs[0] == idxs[1]:
        ax.set_yscale('log')
fig.savefig(opj(imgdir, 'n_ell_out_log_flat'))
plt.close(fig)
