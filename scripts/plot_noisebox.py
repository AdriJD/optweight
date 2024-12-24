import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import os

from pixell import enmap, enplot
from pys2let import axisym_wav_l

import wlm_utils

opj = os.path.join

mapdir = '/home/adriaand/project/actpol/20201029_noisebox'
imgdir = opj(mapdir, 'img')

arrays = ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150', 'pa4_f150',
          'pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150',
          'ar1_f150', 'ar2_f220', 'planck_f090', 'planck_f150', 'planck_f220']
bins = [100, 111, 124, 137, 153, 170, 189, 210, 233, 260, 289, 321, 357, 397,
        441, 490, 545, 606, 674, 749, 833, 926, 1029, 1144, 1272, 1414, 1572,
        1748, 1943, 2161, 2402, 2671, 2969, 3301, 3670, 4081, 4537, 5044, 5608,
        6235, 6931, 7706, 8568, 9525, 10590, 11774, 13090, 14554, 16180, 17989]

# load kernels
lmax = 6000
#lamb = 2
lamb = 1.5
spin = 0 

w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax, j0=None, lmin=100, return_j=True)
print('lmaxs', lmaxs)
print('js', j_scales)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_xscale('log')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels_me'))
plt.close(fig)


noisebox = enmap.read_fits(opj(mapdir, 'noisebox_f150_night.fits'))
wcs = noisebox.wcs
# Stokes, array, multipole, ny, nx.
print(noisebox.shape)

# Sum over arrays.
noisebox = np.sum(noisebox, axis=1)

# Extend noisebox to l=0.
noisebox_ext = np.zeros(noisebox.shape[:1] + (noisebox.shape[1] + 1,) + noisebox.shape[2:])
noisebox_ext[:,1:,:,:] = noisebox
noisebox_ext[:,0,:,:] = noisebox[:,0,:,:]
bins_ext = np.zeros(len(bins) + 1)
bins_ext[0] = 0
bins_ext[1:] = bins

# Interpolate noisebox over ell 
cs = CubicSpline(bins_ext, noisebox_ext, axis=1)

lmax = 6000
ells = np.arange(lmax + 1)

noisebox_full = np.zeros(noisebox.shape[:1] + (lmax + 1,) + noisebox.shape[2:])
noisebox_full[...] = cs(ells)
noisebox_full = enmap.enmap(noisebox_full, wcs=wcs, copy=False)

plot = enplot.plot(noisebox_full[0,10,:,:], colorbar=True, grid=20)
enplot.write(opj(imgdir, 'noisebox_full'), plot)


# k^JJ N_l
w_ell_trunc = np.ascontiguousarray(w_ell[:,:lmax+1])

wavelet_matrix = np.einsum('ij,kj->ikj', w_ell_trunc, w_ell_trunc, optimize=True)

print(noisebox_full.shape)
print(w_ell_trunc.shape)
print(w_ell_trunc)

#or normalize w_ell?
w_ell_trunc /= np.sqrt(np.sum(w_ell_trunc, axis=1))[:,np.newaxis]

noisebox_jj = np.einsum('ij, kj, ljno -> likno', w_ell_trunc, w_ell_trunc, noisebox_full, optimize=True)
noisebox_jj = enmap.enmap(noisebox_jj, wcs=wcs, copy=False)
# norm_jj = np.einsum('ij, kj -> ik', w_ell_trunc, w_ell_trunc, optimize=True)
# fig, ax = plt.subplots(dpi=300)
# im = ax.imshow(norm_jj)
# fig.colorbar(im, ax=ax)
# fig.savefig(opj(imgdir, 'norm_jj'))
# plt.close(fig)


# norm_jj = np.linalg.inv(norm_jj)

# fig, ax = plt.subplots(dpi=300)
# im = ax.imshow(norm_jj)
# fig.colorbar(im, ax=ax)
# fig.savefig(opj(imgdir, 'inorm_jj'))
# plt.close(fig)

# noisebox_jj = np.einsum('ij, pjklm -> piklm', norm_jj, noisebox_jj, optimize=True)

print(noisebox_jj[0,7,7,::50,::50])

midy = 120
midx = 350

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(noisebox_jj[0,:,:,midy,midx])
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'n_jj'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
im = ax.imshow(np.log10(np.abs(noisebox_jj[0,:,:,midy,midx])))
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'n_jj_log'))
plt.close(fig)



fig, ax = plt.subplots(dpi=300, constrained_layout=True)
for pidx in range(3):
    #n_w = np.dot(noisebox_jj[pidx,:,:,midy,midx], np.ones(len(lmaxs)))
    mat = noisebox_jj[pidx,:,:,midy,midx]
    #matr = mat.ravel()
    mat[np.tril_indices(len(lmaxs), k=-1)] = 0
    n_w = np.dot(mat, np.ones(len(lmaxs)))
    for widx in range(w_ell.shape[0]):


        #n_ell_w = np.ones(lmax+1) * noisebox_jj[pidx,widx,widx,midy,midx]
        n_ell_w = np.ones(lmax+1) * n_w[widx]
        lmax_w = lmaxs[widx]
        try:
            lmin_w = lmaxs[widx-1]
        except IndexError:
            lmin_w = 0
        n_ell_w[lmax_w+1:] = np.nan
        n_ell_w[:lmin_w+1] = np.nan
        
        color = 'C{}'.format(pidx)
        ax.plot(ells, n_ell_w, color=color)
        ax.plot(ells, noisebox_full[pidx,:,midy,midx], color=color)

fig.savefig(opj(imgdir, 'n_ell_jj'))
plt.close(fig)

for pidx in range(3):
    for widx in range(w_ell.shape[0]):
        #plot = enplot.plot(noisebox_jj[pidx,widx,widx,:,:], colorbar=True, grid=20)
        #enplot.write(opj(imgdir, 'n_diag_{}_{}'.format(widx, pidx)), plot)

        # Determine lmin.
        j_scale = j_scales[widx]
        lmin_w = wlm_utils.j_scale_to_lmin(j_scale, lamb)
        if widx == 0:
            lmin_w = 0
        #lmax_w = wlm_utils.j_scale_to_lmax(j_scale, lamb)
        lmax_w = lmaxs[widx]

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        im = ax.imshow(noisebox_jj[pidx,widx,widx,:,:], origin='lower')
        #fig.colorbar(im, ax=ax, shrink=0.3)
        ax.text(0.987, 0.05,
                r'{} $\leq \ell \leq$ {}'.format(lmin_w, lmax_w),
                horizontalalignment='right', verticalalignment='bottom',
                color='black', transform=ax.transAxes, 
                bbox=dict(facecolor='white', edgecolor='black', pad=5.0))
        ax.axis('off')
        fig.savefig(opj(imgdir, 'n_diag_{}_{}'.format(widx, pidx)), bbox_inches='tight')
        plt.close(fig)

#exit()

for bidx in range(len(bins)):
    for pidx in range(3):
        #plot = enplot.plot(np.sum(noisebox[pidx,:,bidx,:,:], axis=0), colorbar=True, grid=20)
        plot = enplot.plot(noisebox[pidx,bidx,:,:], colorbar=True, grid=20)
        enplot.write(opj(imgdir, 'test_{}_{}'.format(bidx, pidx)), plot)





# loop over kernels

# weight each pixel by kernel

# plot map

# end loop

