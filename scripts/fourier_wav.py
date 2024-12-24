import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import time

from scipy.interpolate import interp1d
import healpy as hp
from pixell import enmap, enplot, sharp, curvedsky
from optweight import (noise_utils, wavtrans, mat_utils, wlm_utils, map_utils, dft,
                       solvers, preconditioners, alm_utils, sht, alm_c_utils, operators)

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20230206_fourier_wav'
os.makedirs(imgdir, exist_ok=True)

lmax = 500

def f2wav(fmap, wav, modlmap, w_ell_call):

    for widx in range(len(w_ell_call)):

        kernel = w_ell_call[widx](modlmap)
        dft.irfft(fmap * kernel, wav[widx:widx+1])

def wav2f(wav, fmap, modlmap, w_ell_call):

    fmap *= 0 

    for widx in range(len(w_ell_call)):
        
        kernel = w_ell_call[widx](modlmap)
        tmp = fmap * 0
        dft.rfft(wav[widx:widx+1], tmp)
        fmap += tmp * kernel        

def digitize(a):
    """Turn a smooth array with values between 0 and 1 into an on/off array
    that approximates it."""
    f = np.round(np.cumsum(a))
    return np.concatenate([[1], f[1:] != f[:-1]])

def plot_fourier(fmap, lwcs, **kwargs):
    
    fmap_shift = fmap.copy()
    fmap_shift = np.fft.fftshift(fmap_shift, axes=[-2])
    fmap_shift = enmap.ndmap(fmap_shift, lwcs)
    return enplot.plot(fmap_shift, **kwargs)

ells = np.arange(lmax + 1)

lamb = 1.6
lmin = 100
lmax_w = lmax
lmax_j = lmax - 50
w_ell, _ = wlm_utils.get_sd_kernels(lamb, lmax_w, lmin=lmin, lmax_j=lmax_j)
nwav = w_ell.shape[0]

#ells_fine = np.linspace(0, lmax + 1, 10 * lmax)
ells_fine = np.linspace(0, lmax + 1, 1 * lmax)
w_ell_fine = np.zeros((nwav, ells_fine.size))
for widx in range(nwav):
    w_ell_fine[widx] = interp1d(ells, w_ell[widx], fill_value='extrapolate')(ells_fine)
ells = ells_fine
w_ell = w_ell_fine

d_ell = np.ones_like(w_ell)

for idx in range(0, w_ell.shape[0], 2):
    d_ell[idx] = digitize(w_ell[idx])
    
    if idx != 0:
        d_ell[idx,0] = 0

d_ell[1::2] *= (1 - (np.sum(d_ell[::2,:], axis=0)[np.newaxis,:] > 0.5).astype(int))
for idx in range(1, w_ell.shape[0], 2):
    d_ell[idx,w_ell[idx] < 1e-5] = 0

fig, axs = plt.subplots(dpi=300, nrows=nwav, sharex=True, constrained_layout=True)
for idx in range(nwav):
    axs[idx].plot(ells, w_ell[idx])
fig.savefig(opj(imgdir, 'w_ell'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=nwav, sharex=True, constrained_layout=True)
for idx in range(nwav):
    axs[idx].plot(ells, d_ell[idx])
fig.savefig(opj(imgdir, 'd_ell'))
plt.close(fig)

w_ell_call = []
d_ell_call = []
for idx in range(nwav):
    if idx == nwav - 1:
        fill_value = (0., 1.)
    else:
        fill_value = (0., 0.)
    w_ell_call.append(interp1d(ells, w_ell[idx], kind='linear', #kind='nearest', 
                               bounds_error=False, fill_value=fill_value))
    d_ell_call.append(interp1d(ells, d_ell[idx], kind='nearest', #kind='nearest', 
                               bounds_error=False, fill_value=fill_value))

imap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1)
imap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/4, -np.pi/3],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1) # NOTE CUTSKY

modlmap = dft.modlmap_real(imap.shape, imap.wcs, dtype=np.float64)
modlmap_full = enmap.modlmap(imap.shape, imap.wcs)
lwcs = dft.lwcs_real(imap.shape, imap.wcs)

#imap[:] = np.random.randn(*imap.shape)
cov = np.zeros((1, 1) + modlmap_full.shape)
cov[0,0,:,:] = (1 + (np.maximum(modlmap_full, 10) / 2000) ** -3.)
imap[:] = enmap.rand_map(imap.shape, imap.wcs, cov, scalar=False, seed=None, pixel_units=False, iau=False, spin=[0,2])



omap = np.zeros_like(imap)

plot = enplot.plot(imap, colorbar=True, ticks=30)
enplot.write(opj(imgdir, 'imap'), plot)


print(modlmap.max())

kernel_tot = np.zeros_like(modlmap)
for widx in range(len(w_ell_call)):
    kernel = w_ell_call[widx](modlmap)
    kernel = enmap.samewcs(kernel, lwcs)
    #plot = enplot.plot(kernel, colorbar=True, ticks=100)
    plot = plot_fourier(kernel, lwcs, colorbar=True, ticks=100)
    enplot.write(opj(imgdir, f'kernel_w_ell_{widx}'), plot)
    kernel_tot += kernel ** 2

plot = enplot.plot(kernel_tot, colorbar=True, ticks=100, min=0.999, max=1.001)
enplot.write(opj(imgdir, f'kernel_w_ell_tot'), plot)

kernel_tot = np.zeros_like(modlmap)
for widx in range(len(w_ell_call)):
    kernel = d_ell_call[widx](modlmap)
    kernel = enmap.samewcs(kernel, lwcs)
    #plot = enplot.plot(kernel, colorbar=True, ticks=100)
    plot = plot_fourier(kernel, lwcs, colorbar=True, ticks=100)
    enplot.write(opj(imgdir, f'kernel_d_ell_{widx}'), plot)
    kernel_tot += kernel ** 2

plot = enplot.plot(kernel_tot, colorbar=True, ticks=100, min=0.999, max=1.001)
enplot.write(opj(imgdir, f'kernel_d_ell_tot'), plot)

for tag, x_ell_call in zip(['w_ell', 'd_ell'], [w_ell_call, d_ell_call]):

    # Test roundtrip. F' wav2f f2wav F m
    wav = np.zeros((nwav,) + imap.shape[-2:])
    fmap = np.zeros(imap.shape[:-1] + (imap.shape[-1] // 2 + 1,), np.complex128)
    dft.rfft(imap, fmap)
    f2wav(fmap, wav, modlmap, x_ell_call)

    for idx in range(nwav):
        plot = enplot.plot(enmap.samewcs(wav[idx], imap.wcs), colorbar=True, ticks=30)
        enplot.write(opj(imgdir, f'wav_{idx}_{tag}'), plot)

    wav2f(wav, fmap, modlmap, x_ell_call)
    dft.irfft(fmap, omap)
    
    plot = enplot.plot(omap, colorbar=True, ticks=30)
    enplot.write(opj(imgdir, f'omap_{tag}'), plot)

    plot = enplot.plot(omap - imap, colorbar=True, ticks=30)
    enplot.write(opj(imgdir, f'diff_{tag}'), plot)

wav_in = wav.copy()
wav_out = wav * 0

for tag, x_ell_call in zip(['w_ell', 'd_ell'], [w_ell_call, d_ell_call]):

    # Test orthonomality. f2wav wav2f wav

    for widx in range(nwav):

        kernel = x_ell_call[widx](modlmap)
        kernel = enmap.samewcs(kernel, lwcs)

        for widx_p in range(nwav):
            
            kernel_p = x_ell_call[widx_p](modlmap)
            kernel_p = enmap.samewcs(kernel_p, lwcs)

            if widx_p == widx:
                vmin, vmax = 0.999, 1.001
            else:
                vmin, vmax = -0.001, 0.001

            #plot = enplot.plot(kernel * kernel_p, colorbar=True, ticks=100, min=vmin, max=vmax)
            plot = plot_fourier(kernel * kernel_p, lwcs, colorbar=True, ticks=100, min=vmin, max=vmax)
            enplot.write(opj(imgdir, f'kernelmat_{widx}_{widx_p}_{tag}'), plot)
            

    for widx in range(nwav):
        plot = enplot.plot(enmap.samewcs(wav_in[widx], imap.wcs), colorbar=True, ticks=30)
        enplot.write(opj(imgdir, f'wav_in_{widx}_{tag}'), plot)

    wav2f(wav_in, fmap, modlmap, x_ell_call)
    f2wav(fmap, wav_out, modlmap, x_ell_call)
    
    for widx in range(nwav):
        plot = enplot.plot(enmap.samewcs(wav_out[widx], imap.wcs), colorbar=True, ticks=30)
        enplot.write(opj(imgdir, f'wav_out_{widx}_{tag}'), plot)

        plot = enplot.plot(enmap.samewcs(wav_out[widx] - wav_in[widx], imap.wcs), colorbar=True, ticks=30)
        enplot.write(opj(imgdir, f'wav_diff_{widx}_{tag}'), plot)

for tag, x_ell_call in zip(['w_ell', 'd_ell'], [w_ell_call, d_ell_call]):

    # Test N^-1 N.
    wav = np.zeros((nwav,) + imap.shape[-2:])
    fmap = np.zeros(imap.shape[:-1] + (imap.shape[-1] // 2 + 1,), np.complex128)
    dft.rfft(imap, fmap)
    f2wav(fmap, wav, modlmap, x_ell_call)
    wav2f(wav, fmap, modlmap, x_ell_call)
    dft.irfft(fmap, omap)
    dft.rfft(omap, fmap)
    f2wav(fmap, wav, modlmap, x_ell_call)
    wav2f(wav, fmap, modlmap, x_ell_call)
    dft.irfft(fmap, omap)

    plot = enplot.plot(omap - imap, colorbar=True, ticks=30)
    enplot.write(opj(imgdir, f'diff_npn_{tag}'), plot)


# minfo = map_utils.get_cc_minfo(lmax)
# #minfo = map_utils.get_gauss_minfo(lmax)

# imap = np.random.randn(1, minfo.npix)

# fmap = np.zeros(imap.shape[:-1] + (imap.shape[-1] // 2 + 1,), np.complex128)
        
# dft.rfft(imap, fmap)
# omap = np.zeros_like(imap)
# dft.irfft(fmap, omap)

# diff = omap - imap

# for pidx in range(imap.shape[0]):
#     fig, ax = plt.subplots(dpi=300)
#     im = ax.imshow(map_utils.view_2d(imap[pidx], minfo), interpolation='none')
#     fig.colorbar(im, ax=ax)
#     fig.savefig(opj(imgdir, 'imap_{}'.format(pidx)))
#     plt.close(fig)

# for pidx in range(omap.shape[0]):
#     fig, ax = plt.subplots(dpi=300)
#     im = ax.imshow(map_utils.view_2d(omap[pidx], minfo), interpolation='none')
#     fig.colorbar(im, ax=ax)
#     fig.savefig(opj(imgdir, 'omap_{}'.format(pidx)))
#     plt.close(fig)

# for pidx in range(diff.shape[0]):
#     fig, ax = plt.subplots(dpi=300)
#     im = ax.imshow(map_utils.view_2d(diff[pidx], minfo), interpolation='none')
#     fig.colorbar(im, ax=ax)
#     fig.savefig(opj(imgdir, 'diff_{}'.format(pidx)))
#     plt.close(fig)

        

