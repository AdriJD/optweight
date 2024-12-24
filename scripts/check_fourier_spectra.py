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

imgdir = '/home/adriaand/project/actpol/20230318_fourier_spec'
os.makedirs(imgdir, exist_ok=True)

lmax = 500
ainfo = sharp.alm_info(lmax)
ells = np.arange(lmax + 1)

# Approximate band.
imap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/4, -np.pi/3],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1) # NOTE CUTSKY

modlmap = dft.modlmap_real(imap.shape, imap.wcs, dtype=np.float64)
modlmap_full = enmap.modlmap(imap.shape, imap.wcs)
lwcs = dft.lwcs_real(imap.shape, imap.wcs)

cov = np.zeros((1, 1) + modlmap_full.shape)
cov[0,0,:,:] = (1 + (np.maximum(modlmap_full, 10) / 2000) ** -3.)
imap[:] = enmap.rand_map(imap.shape, imap.wcs, cov, scalar=False, seed=None, pixel_units=False,
                         iau=False, spin=[0,2])

ps1d, fs = enmap.lbin(enmap.enmap(cov, wcs=imap.wcs))

fmap = np.zeros(imap.shape[:-1] + (imap.shape[-1] // 2 + 1,), np.complex128)
mask = enmap.ones(imap.shape, imap.wcs)
mask_apod = enmap.apod(mask, 50)

dft.rfft(imap * mask, fmap)

ps1d_real, fs_real = dft.calc_ps1d(fmap, imap.wcs, modlmap)
ps1d_real /= np.mean(mask ** 2) / enmap.pixsize(imap.shape, imap.wcs)

alm = curvedsky.map2alm(imap, ainfo=ainfo)
c_ell = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
pmap = enmap.pixsizemap(mask.shape, mask.wcs)
w2 = np.sum((mask ** 2)*pmap) / np.pi / 4. 
c_ell /= w2

# Now try to whiten the data.
fmap_white = dft.fmul(fmap, fmat1d=ps1d_real ** -0.5, ells=fs_real, modlmap=modlmap)
ps1d_white_real, _ = dft.calc_ps1d(fmap_white, imap.wcs, modlmap)
ps1d_white_real /= np.mean(mask ** 2) / enmap.pixsize(imap.shape, imap.wcs)


fig, ax = plt.subplots(dpi=300)
ax.axhline(1, color='black', lw=0.5)
ax.plot(fs, ps1d[0,0], label='input flat')
ax.plot(fs_real, ps1d_real[0], label='me flat')
ax.plot(fs_real, ps1d_white_real[0], label='me white')
ax.plot(ells, c_ell[0,0], label='sht')
ax.set_ylim(bottom=0.1)
ax.set_yscale('log')
ax.legend()
fig.savefig(opj(imgdir, 'spectra'))
plt.close()


