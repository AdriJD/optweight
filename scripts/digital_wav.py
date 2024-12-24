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
from optweight import (noise_utils, wavtrans, mat_utils, wlm_utils, map_utils,
                       solvers, preconditioners, alm_utils, sht, alm_c_utils, operators)

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20230201_digital_wav'

os.makedirs(imgdir, exist_ok=True)

def digitize(a):
    """Turn a smooth array with values between 0 and 1 into an on/off array
    that approximates it."""
    f = np.round(np.cumsum(a))
    return np.concatenate([[1], f[1:] != f[:-1]])

#def digitize2(a, fact=4):
#    """Turn a smooth array with values between 0 and 1 into an on/off array
#    that approximates it."""

#    ells = np.arange(a.size)
#    a = interp1d(np.arange(ells), a)(ells[::fact])

#    f = np.round(np.cumsum(a))
#    out = np.concatenate([[1], f[1:] != f[:-1]])
    


lmax = 1000
ells = np.arange(lmax + 1)

# Get wavelet kernels and estimate wavelet covariance.
lamb = 1.6
lmin = 100
lmax_w = lmax
# If lmax <= 5400, lmax_j will usually be lmax-100; else, capped at 5300
# so that white noise floor is described by a single (omega) wavelet
#lmax_j = min(max(lmax - 100, lmin), 5300)
lmax_j = lmax - 300
w_ell, _ = wlm_utils.get_sd_kernels(lamb, lmax_w, lmin=lmin, lmax_j=lmax_j)

# NOTE:
w_ell = np.ones((1, lmax+1))

nwav = w_ell.shape[0]

d_ell = np.ones_like(w_ell)

for idx in range(0, w_ell.shape[0], 2):
    d_ell[idx] = digitize(w_ell[idx])
    
    if idx != 0:
        d_ell[idx,0] = 0

d_ell[1::2] *= (1 - (np.sum(d_ell[::2,:], axis=0)[np.newaxis,:] > 0.5).astype(int))

print(np.sum(d_ell[::2,:], axis=0))

for idx in range(1, w_ell.shape[0], 2):
#    #d_ell[idx,(d_ell[idx-1] == 1) & (d_ell[idx-1] == 1)] = 0
    d_ell[idx,w_ell[idx] < 1e-5] = 0

    


wr = np.zeros((nwav, 3 * lmax))
dr = np.zeros((nwav, 3 * lmax))
radii = np.linspace(0, np.pi, 3 * lmax)
for idx in range(nwav):
    wr[idx] = hp.bl2beam(w_ell[idx], radii)
    dr[idx] = hp.bl2beam(d_ell[idx], radii)



ells_super = np.linspace(0, lmax, 50 * lmax)
d_ell_super = interp1d(ells, d_ell, axis=-1, kind='nearest')(ells_super)

#for idx in range(0, w_ell.shape[0], 2):
#    d_ell[idx,d_ell[idx] < 0.99] = 0
#for idx in range(1, w_ell.shape[0], 2):
#    d_ell[idx,d_ell[idx] > 0.01] = 1

fig, axs = plt.subplots(dpi=300, nrows=nwav, sharex=True, constrained_layout=True, squeeze=False)
#for idx in range(w_ell.shape[0]):
for idx in range(nwav):
    axs[idx,idx].plot(ells, w_ell[idx])
    #ax.set_xscale('log')
#    axs[idx,idx].set_xlim(0, 100)
fig.savefig(opj(imgdir, 'w_ell'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=nwav, sharex=True, constrained_layout=True, squeeze=False)
#for idx in range(w_ell.shape[0]):
for idx in range(nwav):
    axs[idx,idx].plot(ells_super, d_ell_super[idx])
#    axs[idx,idx].set_xlim(0, 100)
#ax.set_xscale('log')
fig.savefig(opj(imgdir, 'd_ell'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=nwav, sharex=True, constrained_layout=True, squeeze=False)
for idx in range(nwav):
    axs[idx,idx].plot(radii, wr[idx])
fig.savefig(opj(imgdir, 'wr'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=nwav, sharex=True, constrained_layout=True, squeeze=False)
for idx in range(nwav):
    axs[idx,idx].plot(radii, wr[idx])
    axs[idx,idx].plot(radii, dr[idx])
fig.savefig(opj(imgdir, 'dr'))
plt.close(fig)



# Draw isotropic white noise.
n_ell = np.ones(lmax + 1) * np.eye(3)[:,:,None]
alm, ainfo = curvedsky.rand_alm(n_ell, return_ainfo=True)

#minfo = map_utils.get_gauss_minfo(2 * lmax)
#noise = np.zeros((3, minfo.npix))
#sht.alm2map(alm, noise, ainfo, minfo, [0,2], adjoint=False)


cov_wav = noise_utils.estimate_cov_wav(alm, ainfo, w_ell, [0, 2], diag=False,
                                       fwhm_fact=5)

for idx in cov_wav.indices:
    print(idx)
    cov_wav.maps[tuple(idx)] = cov_wav.maps[tuple(idx)][0,:,0,:]
cov_wav.preshape = (3, 3,)
    

# NOTE
for idx in cov_wav.indices:
    print(idx)
    cov_wav.maps[tuple(idx)][:] = 1

#icov_wav = wavmatpow(cov_wav, -1)
cov_op = operators.WavMatVecAlm(ainfo, cov_wav, w_ell, [0, 2], power=1, adjoint=True)
icov_op = operators.WavMatVecAlm(ainfo, cov_wav, w_ell, [0, 2], power=-1, adjoint=False)

alm_nnp = cov_op(icov_op(alm))
alm_npn = icov_op(cov_op(alm))

nl_in = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
nl_nnp = ainfo.alm2cl(alm_nnp[:,None,:], alm_nnp[None,:,:])
nl_npn = ainfo.alm2cl(alm_npn[:,None,:], alm_npn[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(enmap.smooth_spectrum(nl_in[idxs], width=50), label='in', color='C0')
    axs[idxs].plot(enmap.smooth_spectrum(nl_nnp[idxs], width=50), label='nnp', color='C1')
    axs[idxs].plot(enmap.smooth_spectrum(nl_npn[idxs], width=50), label='npn', color='C2')

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))

omap_in = omap.copy()
curvedsky.alm2map(alm, omap_in, ainfo=ainfo)
plot = enplot.plot(omap_in, colorbar=True, grid=False)
enplot.write(opj(imgdir, 'alm_in'), plot)

curvedsky.alm2map(alm_nnp, omap, ainfo=ainfo)
plot = enplot.plot(omap - omap_in, colorbar=True, grid=False)
enplot.write(opj(imgdir, 'alm_nnp'), plot)

curvedsky.alm2map(alm_npn, omap, ainfo=ainfo)
plot = enplot.plot(omap - omap_in, colorbar=True, grid=False)
enplot.write(opj(imgdir, 'alm_npn'), plot)


cov_op = operators.WavMatVecAlm(ainfo, cov_wav, d_ell, [0, 2], power=1, adjoint=True)
icov_op = operators.WavMatVecAlm(ainfo, cov_wav, d_ell, [0, 2], power=-1, adjoint=False)

alm_nnp = cov_op(icov_op(alm))
alm_npn = icov_op(cov_op(alm))

nl_nnp = ainfo.alm2cl(alm_nnp[:,None,:], alm_nnp[None,:,:])
nl_npn = ainfo.alm2cl(alm_npn[:,None,:], alm_npn[None,:,:])

for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(enmap.smooth_spectrum(nl_nnp[idxs], width=50), label='nnp_d', color='C3')
    axs[idxs].plot(enmap.smooth_spectrum(nl_npn[idxs], width=50), label='npn_d', color='C4')
axs[1,0].legend(frameon=True)
fig.savefig(opj(imgdir, 'n_ell'))
plt.close(fig)

curvedsky.alm2map(alm_nnp, omap, ainfo=ainfo)
plot = enplot.plot(omap - omap_in, colorbar=True, grid=False)
enplot.write(opj(imgdir, 'alm_nnp_d'), plot)

curvedsky.alm2map(alm_npn, omap, ainfo=ainfo)
plot = enplot.plot(omap - omap_in, colorbar=True, grid=False)
enplot.write(opj(imgdir, 'alm_npn_d'), plot)

