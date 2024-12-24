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
                       solvers, preconditioners, alm_utils, sht, alm_c_utils, operators,
                       fkernel)

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20230319_fourier_wav'
os.makedirs(imgdir, exist_ok=True)

lmax = 501

lamb = 1.4
lmin = 50
lmax_j = lmax - 50

w_ell, _ = wlm_utils.get_sd_kernels(
    lamb, lmax, j0=None, lmin=lmin, lmax_j=lmax_j)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels_me'))
plt.close(fig)


imap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1)
modlmap = dft.modlmap_real(imap.shape, imap.wcs)

print(modlmap.max())

fkernels = fkernel.get_sd_kernels_fourier(modlmap, lamb, j0=None,
                                          lmin=lmin, lmax_j=lmax_j,
                                          digital=False)
nwav = fkernels.shape[0]

imap_delta = imap.copy()
imap_delta[:,imap.shape[-2]//2,imap.shape[-1]//2] = 1
ny, nx = imap.shape[-2:]
#fmap = np.zeros((1, ny, nx//2+1), dtype=np.complex128)
#ly, lx = dft.laxes_real(imap.shape, imap.wcs)
fmap = np.ones((1, ny, nx//2+1), dtype=np.complex128)
#fmap *= np.exp(1j * 2 * np.pi * (ly * (ny // 2) / ny + lx * (nx // 2) / nx))
#fmap *= np.exp(1j * 2 * np.pi * (ly * (ny // 2) + lx * (nx // 2)))
#print(ly.shape)

freqs = np.fft.fftfreq(imap.shape[-2])
fmap[0,:,:] *= np.exp(1j * 2 * np.pi * (freqs * (ny // 2)))[None,:]
freqs = np.fft.fftfreq(imap.shape[-1] // 2 + 1)
fmap[0,:,:] *= np.exp(1j * 2 * np.pi * (freqs * (nx // 4 + 2)))[:,None]
#fmap *= np.exp(1j * 2 * np.pi * (ly * np.pi / 2 + lx * np.pi))
#fmap = np.fft.fftshift(fmap, axes=-2)

# Note
imap_delta[0,np.random.randint(0, ny, 10), np.random.randint(0, nx, 10)] = 1
dft.rfft(imap_delta, fmap)


lwcs = dft.lwcs_real(imap.shape, imap.wcs)

for widx in range(nwav):
 
    plot = enplot.plot(enmap.enmap(fkernels[widx], lwcs), colorbar=True, ticks=100, quantile=0)
    enplot.write(opj(imgdir, f'fkernel_{widx}'), plot)


    #dft.rfft(imap_delta, fmap)
    arr2plot = fkernels[widx] * fmap
#    arr2plot = fmap

    dft.irfft(arr2plot, imap)

    #arr2plot /= arr2plot.max()

    #plot = enplot.plot(np.log10(np.abs(imap)), colorbar=True, ticks=30, min=-5, max=0,
    #                   sub="-10:10,-10:10")
    plot = enplot.plot(imap, colorbar=True, ticks=30, quantile=0)#, sub="-20:20,-20:20")
    enplot.write(opj(imgdir, f'kernel_{widx}'), plot)

fkernels = fkernel.get_sd_kernels_fourier(modlmap, lamb, j0=None,
                                          lmin=lmin, lmax_j=lmax_j,
                                          digital=True, oversample=30)
nwav = fkernels.shape[0]

#imap_delta = imap.copy()
#imap_delta[:,imap.shape[-2]//2,imap.shape[-1]//2] = 1
#ny, nx = imap.shape[-2:]
#fmap = np.zeros((1, ny, nx//2+1), dtype=np.complex128)

for widx in range(nwav):

    plot = enplot.plot(enmap.enmap(fkernels[widx], lwcs), colorbar=True, ticks=100, quantile=0)
    enplot.write(opj(imgdir, f'fkernel_digital_{widx}'), plot)

    #dft.rfft(imap_delta, fmap)
    arr2plot = fkernels[widx] * fmap

    dft.irfft(arr2plot, imap)

    #arr2plot /= arr2plot.max()

    #plot = enplot.plot(np.log10(np.abs(imap)), colorbar=True, ticks=30, min=-5, max=0,
    #                   sub="-10:10,-10:10")
    plot = enplot.plot(imap, colorbar=True, ticks=30, quantile=0)# , sub="-20:20,-20:20")
    enplot.write(opj(imgdir, f'kernel_digital_{widx}'), plot)
