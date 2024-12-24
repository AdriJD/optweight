import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import os

from pixell import enmap, enplot, sharp

from optweight import wlm_utils, map_utils, operators, sht, noise_utils, mat_utils

opj = os.path.join

mapdir = '/home/adriaand/project/actpol/20201029_noisebox'
imgdir = '/home/adriaand/project/actpol/20201111_noisebox/img'

arrays = ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150', 'pa4_f150',
          'pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150',
          'ar1_f150', 'ar2_f220', 'planck_f090', 'planck_f150', 'planck_f220']
bins = [100, 111, 124, 137, 153, 170, 189, 210, 233, 260, 289, 321, 357, 397,
        441, 490, 545, 606, 674, 749, 833, 926, 1029, 1144, 1272, 1414, 1572,
        1748, 1943, 2161, 2402, 2671, 2969, 3301, 3670, 4081, 4537, 5044, 5608,
        6235, 6931, 7706, 8568, 9525, 10590, 11774, 13090, 14554, 16180, 17989]

# load kernels
lmax = 2000
lamb = 1.5
spin = 0 
lmin = 100

w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax, j0=None, lmin=lmin, return_j=True)
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

#print(noisebox.shape[-2:])
#print(enmap.extent(noisebox.shape[-2:], noisebox.wcs))
fsky_patch = np.prod(enmap.extent(noisebox.shape[-2:], noisebox.wcs)) / 4 / np.pi
print(fsky_patch)


wcs = noisebox.wcs
# Stokes, array, multipole, ny, nx.
print(noisebox.shape)

## Sum over arrays.
#noisebox = np.sum(noisebox, axis=1)
noisebox = noisebox[:,-2,...]

# Compare this to old method.
icov_wav = noise_utils.noisebox2wavmat(noisebox.copy(), bins, w_ell)
#print(icov_wav.maps)
#print(icov_wav.minfos)


# Extend noisebox to l=0.
noisebox_ext = np.zeros(noisebox.shape[:1] + (noisebox.shape[1] + 1,) + noisebox.shape[2:])
noisebox_ext[:,1:,:,:] = noisebox
noisebox_ext[:,0,:,:] = noisebox[:,0,:,:]
bins_ext = np.zeros(len(bins) + 1)
bins_ext[0] = 0
bins_ext[1:] = bins

# Interpolate noisebox over ell 
cs = CubicSpline(bins_ext, noisebox_ext, axis=1)

ells = np.arange(lmax + 1)
dells = ells * (ells + 1) / 2 / np.pi

noisebox_full = np.zeros(noisebox.shape[:1] + (lmax + 1,) + noisebox.shape[2:])
noisebox_full[...] = cs(ells)
noisebox_full = enmap.enmap(noisebox_full, wcs=wcs, copy=False)

fig, axs = plt.subplots(dpi=300, nrows=3, constrained_layout=True, sharex=True)
for yidx in range(0, noisebox_full.shape[-2], 10):
    for xidx in range(0, noisebox_full.shape[-1], 10):
        axs[0].plot(ells, noisebox_full[0,:,yidx,xidx], color='C0', alpha=0.5)
        axs[1].plot(ells, noisebox_full[1,:,yidx,xidx], color='C1', alpha=0.5)
        axs[2].plot(ells, noisebox_full[2,:,yidx,xidx], color='C2', alpha=0.5)
fig.savefig(opj(imgdir, 'icov_ell_in'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=3, constrained_layout=True, sharex=True, sharey=True)
for yidx in range(0, noisebox_full.shape[-2], 20):
    for xidx in range(0, noisebox_full.shape[-1], 20):
        axs[0].plot(ells[:3000], noisebox_full[0,:3000,yidx,xidx] ** -0.5, color='C0', alpha=0.1)
        axs[1].plot(ells[:3000], noisebox_full[1,:3000,yidx,xidx] ** -0.5, color='C1', alpha=0.1)
        axs[2].plot(ells[:3000], noisebox_full[2,:3000,yidx,xidx] ** -0.5, color='C2', alpha=0.1)
axs[2].set_xscale('log')
axs[2].set_xlim(100, 3000)
axs[2].set_ylim(0, 200)
fig.savefig(opj(imgdir, 'sqrt_n_ell_in'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=3, constrained_layout=True, sharex=True, sharey=True)
for yidx in range(0, noisebox_full.shape[-2], 20):
    for xidx in range(0, noisebox_full.shape[-1], 20):
        axs[0].plot(ells[:3000], noisebox_full[0,:3000,yidx,xidx] ** -1, color='C0', alpha=0.1)
        axs[1].plot(ells[:3000], noisebox_full[1,:3000,yidx,xidx] ** -1, color='C1', alpha=0.1)
        axs[2].plot(ells[:3000], noisebox_full[2,:3000,yidx,xidx] ** -1, color='C2', alpha=0.1)
axs[2].set_xscale('log')
axs[2].set_xlim(100, 3000)
axs[2].set_ylim(0, 4e4)
fig.savefig(opj(imgdir, 'n_ell_in'))
plt.close(fig)

#NDEGSQ = 441253
#noisebox_full /= 4 * np.pi / (NDEGSQ * 60 ** 2)
noisebox_full /= 4 * np.pi / (10800 ** 2)

fig, axs = plt.subplots(dpi=300, nrows=3, constrained_layout=True, sharex=True, sharey=True)
for yidx in range(0, noisebox_full.shape[-2], 20):
    for xidx in range(0, noisebox_full.shape[-1], 20):
        axs[0].plot(ells[:1000], noisebox_full[0,:1000,yidx,xidx] ** -1, color='C0', alpha=0.1)
        axs[1].plot(ells[:1000], noisebox_full[1,:1000,yidx,xidx] ** -1, color='C1', alpha=0.1)
        axs[2].plot(ells[:1000], noisebox_full[2,:1000,yidx,xidx] ** -1, color='C2', alpha=0.1)
axs[2].set_xscale('log')
axs[2].set_xlim(100, 1000)
fig.savefig(opj(imgdir, 'n_ell_out'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=3, constrained_layout=True, sharex=True, sharey=True)
for yidx in range(0, noisebox_full.shape[-2], 20):
    for xidx in range(0, noisebox_full.shape[-1], 20):
        axs[0].plot(ells[:1000], dells[:1000] * noisebox_full[0,:1000,yidx,xidx] ** -1, color='C0', alpha=0.1)
        axs[1].plot(ells[:1000], dells[:1000] * noisebox_full[1,:1000,yidx,xidx] ** -1, color='C1', alpha=0.1)
        axs[2].plot(ells[:1000], dells[:1000] * noisebox_full[2,:1000,yidx,xidx] ** -1, color='C2', alpha=0.1)
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlim(1, 1000)
axs[2].set_ylim(1e-5, 10)
fig.savefig(opj(imgdir, 'n_ell_dell_out'))
plt.close(fig)

# k^JJ N_l
w_ell_trunc = np.ascontiguousarray(w_ell[:,:lmax+1])
wavelet_matrix = np.einsum('ij,kj->ikj', w_ell_trunc, w_ell_trunc, optimize=True)

# SUM NOISEBOX OVER Kjj' TO GET PIX VAR PER jj'

#print(noisebox_full.shape)
#print(w_ell_trunc.shape)
#print(w_ell_trunc)

#or normalize w_ell?
#w_ell_trunc /= np.sqrt(np.sum(w_ell_trunc, axis=1))[:,np.newaxis]

#noisebox_jj = np.einsum('ij, kj, ljno -> likno', w_ell_trunc, w_ell_trunc, noisebox_full, optimize=True)
#noisebox_jj = enmap.enmap(noisebox_jj, wcs=wcs, copy=False)

# Convert to cov_pix using legendre formula
#cov_pix = np.ones((1, minfo.npix)) * np.sum(n_ell * (2 * ells + 1)) / 4 / np.pi

prefactor = (2 * ells + 1) / 4 / np.pi
noisebox_jj = np.einsum('ijk, k, lkmn -> lijmn', wavelet_matrix, prefactor, noisebox_full, optimize=True)
noisebox_jj = enmap.enmap(noisebox_jj, wcs=wcs, copy=False)

# Scale pixels by pixel pix_area * (lmax + 1) ** 2 / 4 / pi
#cov_pix[:,start:end] /= (minfo.weight[tidx] * (lmax + 1) ** 2 / 4 / np.pi)
pix_areas = enmap.pixsizemap(noisebox_jj.shape[-2:], wcs)
print(pix_areas.shape)
print(noisebox_jj.shape)
noisebox_jj *= pix_areas / ((lmax + 1) ** 2 / 4 / np.pi)

# Now we have icov_pix per j j' per pixel.

# Plot for single pixel:
midy = 120
midx = 350

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(noisebox_jj[0,:,:,midy,midx])
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'icov_jj'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
im = ax.imshow(np.log10(np.abs(noisebox_jj[0,:,:,midy,midx])))
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'icov_jj_log'))
plt.close(fig)

# draw noise from 1/icov per j j' and get N_ells to see if they make sense?
for jidx in range(j_scales.size):
    # Determine lmax for j
    
    for jpidx in range(jidx, min(jidx+2, j_scales.size)): 

        # Interpolate onto GL map
        icov_pix, minfo = map_utils.enmap2gauss(
            noisebox_jj[:,jidx,jpidx,:,:], 2 * lmaxs[jidx], area_pow=1, 
            mode='nearest')

        print(icov_pix.shape)

        #mask = icov_pix < 1e-3
        icov_pix[icov_pix < 1e-3] = 0
        cov_pix = mat_utils.matpow(icov_pix, -1)

        # Plot maps
        for pidx in range(3):
            fig, ax = plt.subplots(dpi=300, constrained_layout=True)
            im = ax.imshow(
                icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
            fig.colorbar(im, ax=ax, shrink=0.3)
            fig.savefig(opj(imgdir, 'icov_{}_{}_{}'.format(jidx, jpidx, pidx)), bbox_inches='tight')
            plt.close(fig)

            minfo_new = icov_wav.minfos[jidx,jpidx]
            fig, ax = plt.subplots(dpi=300, constrained_layout=True)
            im = ax.imshow(
                icov_wav.maps[jidx,jpidx][pidx].reshape(minfo_new.nrow, minfo_new.nphi[0]), interpolation='none')
            fig.colorbar(im, ax=ax, shrink=0.3)
            fig.savefig(opj(imgdir, 'icov_new_{}_{}_{}'.format(jidx, jpidx, pidx)), bbox_inches='tight')
            plt.close(fig)

                        
        # Draw noise from maps
        rand_map = map_utils.rand_map_pix(cov_pix)

        # Plot noise
        for pidx in range(3):
            fig, ax = plt.subplots(dpi=300, constrained_layout=True)
            #im = ax.imshow(np.log10(np.abs(
            #    rand_map[pidx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
            im = ax.imshow(rand_map[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')

            fig.colorbar(im, ax=ax, shrink=0.3)
            fig.savefig(opj(imgdir, 'noise_{}_{}_{}'.format(jidx, jpidx, pidx)), bbox_inches='tight')
            plt.close(fig)
        
        # determine fsky
        fsky = np.sum(icov_pix.astype(bool), axis=1) / minfo.npix
        fsky *= fsky_patch
        print(fsky_patch)
        print(fsky)

        # map2alm
        ainfo = sharp.alm_info(lmax=lmaxs[jidx])
        alm = np.zeros((3, ainfo.nelem), dtype=np.complex128)
        sht.map2alm(rand_map, alm, minfo, ainfo, [0, 2])

        # take spectra
        n_ell = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
        n_ell /= fsky[:,np.newaxis]
        ells = np.arange(lmaxs[jidx] + 1)
        dells = ells * (ells + 1) / 2 / np.pi
        fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
        for idxs, ax in np.ndenumerate(axs):
            axs[idxs].plot(ells, dells * n_ell[idxs])
        fig.savefig(opj(imgdir, 'n_ell_{}_{}'.format(jidx, jpidx)))
        plt.close(fig)
        

        # Save maps 

# draw rand alms

# weight alm 
        
        
#        mask = noisebox_jj[0,jidx,jdidx,:,:] == 0
#        sqrt_cov = np.zeros(noisebox_jj[0,jidx,jdidx,:,:].shape)
#        sqrt_cov[:,mask] = noisebox_jj[:,jidx,jpidx,mask] ** -0.5
        

        
