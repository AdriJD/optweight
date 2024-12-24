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
from nawrapper import maptools

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils, noise_utils

opj = os.path.join
np.random.seed(39)

lmax = 5000

basedir = '/home/adriaand/project/actpol/20201215_wavelet_pinv'
imgdir = opj(basedir, 'img')

# load kernels
lamb = 1.7
lmin = 100

w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax, j0=None, lmin=lmin, return_j=True)
print('lmaxs', lmaxs)
print('js', j_scales)
w_ell = w_ell[:-1,:]

ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels_me'))
plt.close(fig)

wavelet_matrix = np.einsum('ij,kj->ikj', w_ell, w_ell, optimize=True)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    for wpidx in range(w_ell.shape[0]):
        if widx == wpidx:
            color = 'C{}'.format(widx)
        else:
            color = 'gray'
        ax.plot(wavelet_matrix[widx,wpidx], color=color)
ax.set_ylabel('Wavelet matrix')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'wav_mat'))
plt.close(fig)

inv_wavelet_matrix = np.transpose(wavelet_matrix.copy(), (2, 0, 1))
inv_wavelet_matrix = np.linalg.pinv(inv_wavelet_matrix)
inv_wavelet_matrix = np.transpose(inv_wavelet_matrix, (1, 2, 0))

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    for wpidx in range(w_ell.shape[0]):
        if widx == wpidx:
            color = 'C{}'.format(widx)
        else:
            color = 'gray'
        ax.plot(inv_wavelet_matrix[widx,wpidx], color=color)
ax.set_ylabel('Wavelet matrix')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'inv_wav_mat'))
plt.close(fig)

prod = np.einsum('ijl, jkl -> ikl', inv_wavelet_matrix, wavelet_matrix)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    for wpidx in range(w_ell.shape[0]):
        if widx == wpidx:
            color = 'C{}'.format(widx)
        else:
            color = 'gray'
        ax.plot(prod[widx,wpidx], color=color)
ax.set_ylabel('Wavelet matrix')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'prod'))
plt.close(fig)

identity = np.einsum('il, ijl, jl -> l', w_ell, inv_wavelet_matrix, w_ell)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
ax.plot(identity)
ax.set_ylabel('Wavelet matrix')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'identity'))
plt.close(fig)

w_ell2 = np.einsum('il, ijl, jkl -> kl', w_ell, inv_wavelet_matrix, wavelet_matrix)
fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell2[widx])
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'w_ell2'))
plt.close(fig)

pinv = np.einsum('ijl, jl-> il', inv_wavelet_matrix, w_ell)
for widx in range(w_ell.shape[0]):
    for wpidx in range(w_ell.shape[0]):
        if widx == wpidx:
            color = 'C{}'.format(widx)
        else:
            color = 'gray'
        ax.plot(pinv[widx,wpidx], color=color)
ax.set_ylabel('Wavelet matrix')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'pinv'))
plt.close(fig)

for widx in range(w_ell.shape[0]):
    np.testing.assert_array_almost_equal(pinv[widx], w_ell[widx])

