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

from optweight import (sht, map_utils, solvers, operators, preconditioners, wlm_utils, noise_utils, wavtrans)

opj = os.path.join
np.random.seed(39)

npol = 3

#lmax = 5000
#lmax = 2000
lmax = 3400

#basedir = '/home/adriaand/project/actpol/20201206_pcg_act_wavelet'
basedir = '/home/adriaand/project/actpol/20201208_pcg_act_wavelet_pinv'
imgdir = opj(basedir, 'img')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')
areadir = '/home/adriaand/project/actpol/mapdata/area/'

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))

icov = icov.astype(np.float64)

# Point sources.
#npoint = 1000
#mask_ps = enmap.ones(icov.shape[-2:], wcs=icov.wcs)
#for i in range(npoint):
#    idx_y = np.random.randint(low=0, high=icov.shape[-2])
#    idx_x = np.random.randint(low=0, high=icov.shape[-1])
#    mask_ps[idx_y,idx_x] = 0
#mask_ps = maptools.apod_C2(mask_ps, 0.1)

icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest')
#psm_pix, _ = map_utils.enmap2gauss(mask_ps, 2 * lmax, mode='nearest')

#icov_pix *= psm_pix

# Set too small values to zero.
#mask = icov_pix > 1e-4
#for pidx in range(3):
#    print(np.median(icov_pix[pidx,icov_pix[pidx] != 0]))
#    print(np.mean(icov_pix[pidx,icov_pix[pidx] != 0]))

#mask = icov_pix > 1e-3 # Use for lmax=500
#mask = icov_pix > 0.5 # Use for lmax=500
#icov_pix[~mask] = 0

icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=0.3)
mask = icov_pix != 0
mask_pix = mask.astype(np.float64)

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
#lamb = 1.5
lamb = 1.7
lmin = 100
#lmin = lmax

#w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
#    lamb, lmax, j0=None, lmin=lmin, return_j=True)
#w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
#    lamb, lmax, j0=None, lmin=lmin, return_j=True, lmax_j=2500)
w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax, j0=None, lmin=lmin, return_j=True, lmax_j=3500)


print('lmaxs', lmaxs)
print('js', j_scales)
#print(w_ell[-1,::100])
#w_ell = w_ell[:-1,:]

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

noisebox[0,-3:,...] = np.sqrt(noisebox[1,-3:,...] ** 2 + noisebox[2,-3:,...] ** 2)
noisebox = np.sum(noisebox, axis=1)
#noisebox = np.sum(noisebox[:,:-3,...], axis=1) # No Planck.

# NOTE
#noisebox = noisebox.astype(np.float32)

t0 = time.time()
icov_wav = noise_utils.noisebox2wavmat(noisebox.copy(), bins, w_ell, offsets=[0]) 
print(time.time() - t0)
print(icov_wav.dtype)              

# NOTE REMOVE THIS
#icov_wav.maps[0,0][0] /= 20
#icov_wav.maps[1,1][0] /= 20
#icov_wav.maps[2,2][0] /= 10
#icov_wav.maps[3,3][0] /= 10

icov_name = f'icov_wav_{lmax}'
wavtrans.write_wav(opj(basedir, icov_name), icov_wav, extra={'w_ell': w_ell})


# Determine itau per icov_wav map
# Plot spectrum...
# Will result in f_ell, can be operator just like beam..

#itaus = np.zeros((w_ell.shape[0], 3))
#for jidx in range(w_ell.shape[0]):
#    itaus[jidx] = np.diag(map_utils.get_isotropic_ivar(icov_wav.maps[jidx,jidx],
#                                     icov_wav.minfos[jidx,jidx]))

#n_ell_iso = np.einsum('ij, ik -> jk', itaus, w_ell ** 2)

in_ell_iso = map_utils.get_ivar_ell(icov_wav, w_ell)

fig, axs = plt.subplots(dpi=300, constrained_layout=True, nrows=3)
for pidx in range(3):
    axs[pidx].plot(ells, in_ell_iso[pidx,pidx])
fig.savefig(opj(imgdir, 'in_ell_iso'))
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
#alm = alm.astype(np.complex64)
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

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
    
niter = 35

#stypes = ['pcg_harm', 'pcg_pinv']
#stypes = ['pcg_pinv', 'pcg_harm']
stypes = ['pcg_pinv']
#stypes = ['pcg_harm']
errors = np.zeros((len(stypes), niter + 1))
chisqs = np.zeros_like(errors)
residuals = np.zeros_like(errors)
times = np.zeros_like(errors)
ps_c_ell = np.zeros((niter, npol, npol, lmax + 1))

for sidx, stype in enumerate(stypes):
    
    if stype == 'cg' or stype == 'cg_scaled':
        prec = None
    elif stype == 'pcg_harm':
        prec = 'harmonic'
        mmask_pix = mask_pix
    elif stype == 'pcg_pinv':
        prec = 'pinv_wav'
        mmask_pix = mask_pix
        #mmask_pix = None
    print('sidx', sidx)

    t0 = time.time()
    #solver = solvers.CGWiener.from_arrays_wav(alm, ainfo, icov_ell, icov_wav, w_ell, 
    #                                          b_ell=b_ell, mask_pix=mmask_pix, minfo_mask=minfo,
    #                                          prec=prec, icov_pix=icov_pix, minfo_icov_pix=minfo,
    #                                          use_prec_masked=True)
    solver = solvers.CGWiener.from_arrays_wav(alm, ainfo, icov_ell, icov_wav, w_ell, 
                                              b_ell=b_ell, mask_pix=mmask_pix, minfo_mask=minfo, 
                                              draw_constr=True)
    prec_pinv = preconditioners.PseudoInvPreconditionerWav(
        ainfo, icov_ell, icov_wav, w_ell, [0, 2], mask_pix=mask_pix, 
        minfo_mask=minfo, b_ell=b_ell)

    prec_masked_cg = preconditioners.MaskedPreconditionerCG(
        ainfo, icov_ell, [0, 2], mask_pix.astype(bool), minfo, lmax=2500,
        nsteps=15, lmax_r_ell=None)

    prec_masked_mg = preconditioners.MaskedPreconditioner(
        ainfo, icov_ell[0:1,0:1], 0, mask_pix[0].astype(bool), minfo,
        min_pix=10000, n_jacobi=1, lmax_r_ell=6000)

    solver.add_preconditioner(prec_pinv)
    solver.add_preconditioner(prec_masked_cg)
    solver.init_solver()
    for idx in range(7):
        solver.step()
        print(solver.i, solver.err)

    solver.reset_preconditioner()
    solver.add_preconditioner(prec_pinv)
    solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
    solver.b_vec = solver.b0
    solver.init_solver(x0=solver.x)

    #t0 = time.time()
    #solver.A(alm.copy())
    #print('A(x)', time.time() - t0)

    #t0 = time.time()
    #solver.M(alm.copy())
    #print('M(x)', time.time() - t0)

    #exit()

    # omap = curvedsky.alm2map(solver.M(alm), omap)
    # for pidx in range(alm.shape[0]):

    #     plot = enplot.plot(omap[pidx], colorbar=True, grid=False)
    #     enplot.write(opj(imgdir, 'alm_prec_{}'.format(pidx)), plot)

    # omap = curvedsky.alm2map(solver.M.pcov_noise(alm), omap)
    # for pidx in range(alm.shape[0]):

    #     plot = enplot.plot(omap[pidx], colorbar=True, grid=False)
    #     enplot.write(opj(imgdir, 'alm_pcov_noise_{}'.format(pidx)), plot)

    # icov_noise = operators.WavMatVecAlm(
    #     ainfo, icov_wav, w_ell, [0, 2], adjoint=False)

    # omap = curvedsky.alm2map(icov_noise(alm), omap)
    # for pidx in range(alm.shape[0]):

    #     plot = enplot.plot(omap[pidx], colorbar=True, grid=False)
    #     enplot.write(opj(imgdir, 'alm_icov_noise_{}'.format(pidx)), plot)

    # plot maps inside cov_noise.
    # for index in icov_noise.m_wav.indices:
    #     print(index)
    #     for pidx in range(3):
    #         m = icov_noise.m_wav.maps[tuple(index)][pidx]
    #         minfo_i = icov_noise.m_wav.minfos[tuple(index)]
    #         fig, ax = plt.subplots(dpi=300)
    #         #im = ax.imshow(np.log10(np.abs(
    #         #    m.reshape(minfo_i.nrow, minfo_i.nphi[0]))), interpolation='none')
    #         im = ax.imshow(
    #             m.reshape(minfo_i.nrow, minfo_i.nphi[0]), interpolation='none')

    #         fig.colorbar(im, ax=ax)
    #         fig.savefig(opj(imgdir, 'icov_noise_{}_{}_{}'.format(index[0], index[1], pidx)))
    #         plt.close(fig)

    # cov_noise = operators.WavMatVecAlm(
    #     ainfo, icov_wav, w_ell, [0, 2], power=-1, adjoint=False)

    # omap = curvedsky.alm2map(cov_noise(alm), omap)
    # for pidx in range(alm.shape[0]):

    #     plot = enplot.plot(omap[pidx], colorbar=True, grid=False)
    #     enplot.write(opj(imgdir, 'alm_cov_noise_{}'.format(pidx)), plot)

    # # plot maps inside cov_noise.
    # for index in cov_noise.m_wav.indices:
    #     print(index)
    #     for pidx in range(3):
    #         m = cov_noise.m_wav.maps[tuple(index)][pidx,pidx]
    #         minfo_i = cov_noise.m_wav.minfos[tuple(index)]
    #         fig, ax = plt.subplots(dpi=300)
    #         im = ax.imshow(np.log10(np.abs(
    #             m.reshape(minfo_i.nrow, minfo_i.nphi[0]))), interpolation='none')
    #         fig.colorbar(im, ax=ax)
    #         fig.savefig(opj(imgdir, 'cov_noise_{}_{}_{}'.format(index[0], index[1], pidx)))
    #         plt.close(fig)


    errors[sidx,0] = np.nan
    chisqs[sidx,0] = solver.get_chisq()
    residuals[sidx,0] = solver.get_residual()

    b_copy = solver.b.copy()

    print('|b| :', np.sqrt(solver.dot(b_copy, b_copy)))

    for idx in range(niter):
        t0 = time.time()
        solver.step()
        t1 = time.time()
        error = solver.err
        chisq = solver.get_chisq()
        residual = solver.get_residual()
        t_solve = t1 - t0

        errors[sidx,idx+1] = error
        chisqs[sidx,idx+1] = chisq
        residuals[sidx,idx+1] = residual
        times[sidx,idx+1] = t_solve
        ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])

        print(solver.i, error, chisq / alm.size / 2, residual, t_solve)

# Save all arrays.
np.save(opj(imgdir, 'residuals'), residuals)
np.save(opj(imgdir, 'chisqs'), chisqs)
np.save(opj(imgdir, 'errors'), errors)
np.save(opj(imgdir, 'times'), times)

# Plot alms

#omap = curvedsky.alm2map(alm_signal, omap)
#plot = enplot.plot(omap, colorbar=True, grid=False, range='250:10')
#enplot.write(opj(imgdir, 'alm_signal'), plot)

#omap = curvedsky.alm2map(alm, omap)
#plot = enplot.plot(omap, colorbar=True, grid=False, range='250:10')
#enplot.write(opj(imgdir, 'alm_in'), plot)

fig, axs = plt.subplots(ncols=npol, nrows=npol, dpi=300, constrained_layout=True, squeeze=False)
for ax in axs.ravel():
    ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, niter)])
for idx in range(niter):
    for aidxs, ax in np.ndenumerate(axs):
        axs[aidxs].plot(ells, dells * ps_c_ell[idx,aidxs[0],aidxs[1]],
                        lw=0.5)
for aidxs, ax in np.ndenumerate(axs):
    axs[aidxs].plot(ells, dells * cov_ell[aidxs[0],aidxs[1]], lw=0.5, color='black', ls=':')
axs[0,0].set_ylim(0, 1e4)
fig.savefig(opj(imgdir, 'ps_c_ell'))
plt.close(fig)

for idxs, ax in np.ndenumerate(axs):
    if idxs[0] == idxs[1]:
        ax.set_yscale('log')
        ax.set_ylim(ax.get_ylim())
        ax.plot(ells, dells * nl[idxs[0],idxs[1]] / b_ell[0] ** 2, lw=1, color='black')
#axs[0,0].set_ylim(10, 1e6)
axs[0,0].set_ylim(0.1, 1e4)
fig.savefig(opj(imgdir, 'ps_c_ell_log'))
plt.close(fig)

# Plot result
omap = curvedsky.alm2map(solver.get_wiener(), omap)
plot = enplot.plot(omap, colorbar=True, grid=False, range='250:10', downgrade=4)
enplot.write(opj(imgdir, 'alm_out'), plot)

omap_b_out = curvedsky.alm2map(solver.A(solver.x), omap.copy())
omap_b_in = curvedsky.alm2map(b_copy, omap.copy())

for pidx in range(alm.shape[0]):

    plot = enplot.plot(omap_b_out[pidx], colorbar=True, grid=False, downgrade=4)
    enplot.write(opj(imgdir, 'b_out_{}'.format(pidx)), plot)

    plot = enplot.plot(omap_b_in[pidx], colorbar=True, grid=False, downgrade=4)
    enplot.write(opj(imgdir, 'b_{}'.format(pidx)), plot)
