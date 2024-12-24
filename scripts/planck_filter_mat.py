import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils
from enlib import cg

from optweight import sht, map_utils, mat_utils, solvers, operators, preconditioners, filters

opj = os.path.join
np.random.seed(39)

def get_planck_b_ell(rimo_file, lmax):
    '''
    Return b_ell.

    Parameters
    ----------
    rimo_file : str
        Path to RIMO beam file.
    lmax : int
        Truncate to this lmax.
    
    Returns
    -------
    b_ell : (npol, nell)
    '''
    
    with fits.open(rimo_file) as hdul:        
        b_ell_T = hdul[1].data['T']
        b_ell_E = hdul[1].data['E']
        b_ell_B = hdul[1].data['B']
    
    b_ell = np.zeros((3, lmax+1))
    if b_ell_T.size < (lmax + 1):
        lmax = b_ell_T.size - 1
    
    b_ell[0,:lmax+1] = b_ell_T[:lmax+1]
    b_ell[1,:lmax+1] = b_ell_E[:lmax+1]
    b_ell[2,:lmax+1] = b_ell_B[:lmax+1]

    return b_ell

lmax = 2500

basedir = '/home/adriaand/project/actpol/20220201_pcg_planck'
maskdir = '/home/adriaand/project/actpol/20201009_pcg_planck/meta'
imgdir = opj(basedir, 'img_new')

utils.mkdir(imgdir)

# Load II, IQ, IU, QQ, QU, UU cov.
cov = hp.read_map(opj(maskdir, 'HFI_SkyMap_100_2048_R3.01_full.fits'), field=(4, 5, 6, 7, 8, 9))
cov *= 1e12 # Convert from K^2 to uK^2.

cov, minfo = map_utils.healpix2gauss(cov, 2*lmax, area_pow=-1)
cov_pix = np.zeros((3, 3, cov.shape[-1]))
cov_pix[0,0] = cov[0]
cov_pix[0,1] = cov[1]
cov_pix[0,2] = cov[2]
cov_pix[1,0] = cov[1]
cov_pix[1,1] = cov[3]
cov_pix[1,2] = cov[4]
cov_pix[2,0] = cov[2]
cov_pix[2,1] = cov[4]
cov_pix[2,2] = cov[5]

for idx in range(3):
    for jdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(cov_pix[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(imgdir, 'cov_real_{}_{}'.format(idx, jdx)))
        plt.close(fig)

for idx in range(3):
    for jdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(cov_pix[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(imgdir, 'cov_real_log_{}_{}'.format(idx, jdx)))
        plt.close(fig)

icov_pix = mat_utils.matpow(cov_pix, -1)
icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=1e-2)

for idx in range(3):
    for jdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(icov_pix[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(imgdir, 'icov_{}_{}'.format(idx, jdx)))
        plt.close(fig)

mask_I = hp.read_map(opj(maskdir, 'COM_Mask_Likelihood-temperature-100-hm2_2048_R3.00.fits'), field=0)
mask_I, _ = map_utils.healpix2gauss(mask_I[np.newaxis,:], 2*lmax, area_pow=0)
mask_I[mask_I>=0.1] = 1
mask_I[mask_I<0.1] = 0

#icov_pix[0,0] *= mask_I[0]
#icov_pix[0,1] *= mask_I[0]
#icov_pix[0,2] *= mask_I[0]
#icov_pix[1,0] *= mask_I[0]
#icov_pix[1,1] *= mask_I[0]
#icov_pix[1,2] *= mask_I[0]
#icov_pix[2,0] *= mask_I[0]
#icov_pix[2,1] *= mask_I[0]
#icov_pix[2,2] *= mask_I[0]

# Load beam.
b_ell = get_planck_b_ell(opj(maskdir, 'BeamWf_HFI_R3.01', 'Bl_TEB_R3.01_fullsky_100x100.fits'), lmax)

#b_ell = np.ones_like(b_ell) # NOTE

# Preprare spectrum. Input file is Dls in uk^2.
c_ell = np.loadtxt(
    opj(maskdir, 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'),
    skiprows=1, usecols=[1, 2, 3, 4]) #  TT, TE, EE, BB.

c_ell = c_ell.T
lmax_c_ell = c_ell.shape[-1] - 1
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi
#cov_ell = np.zeros((3, 3, lmax + 1))
cov_ell = np.zeros((3, 3, lmax + 1)) + np.eye(3)[:,:,None] * 1e-5 # NOTE NOTE NOTE
lmax_arr = lmax
if lmax_c_ell < lmax:
    lmax_arr = lmax_c_ell
cov_ell[0,0,2:lmax_arr+1] = c_ell[0,:lmax_arr-1] 
cov_ell[0,1,2:lmax_arr+1] = c_ell[1,:lmax_arr-1] 
cov_ell[1,0,2:lmax_arr+1] = c_ell[1,:lmax_arr-1] 
cov_ell[1,1,2:lmax_arr+1] = c_ell[2,:lmax_arr-1] 
cov_ell[2,2,2:lmax_arr+1] = c_ell[3,:lmax_arr-1] 

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
        #icov_ell[:,:,lidx] = np.linalg.inv(cov_ell[:,:,lidx])
        icov_ell[:,:,lidx] = np.linalg.pinv(cov_ell[:,:,lidx])

# Draw alms.
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
for pidx in range(alm.shape[0]):
    hp.almxfl(alm[pidx], b_ell[pidx], inplace=True)
# Draw map-based noise and add to alm.
noise = map_utils.rand_map_pix(cov_pix)
alm_noise = alm.copy()
sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=False)
nl = ainfo.alm2cl(alm_noise[:,None,:], alm_noise[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(nl[idxs])
fig.savefig(opj(imgdir, 'n_ell'))
plt.close(fig)

alm += alm_noise

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))

# Apply mask;
#sht.alm2map(alm, noise, ainfo, minfo, [0,2], adjoint=False)
#noise *= mask_I
#sht.map2alm(noise, alm, minfo, ainfo, [0,2], adjoint=False)

niter_cg = 7
niter_mg = 15
niter = niter_cg + niter_mg

ps_c_ell = np.zeros((niter, 3, 3, lmax + 1))

# solver = solvers.CGWiener.from_arrays(alm, ainfo, icov_ell, icov_pix, minfo, b_ell=b_ell,
#                                       draw_constr=False, mask_pix=mask_I * np.ones(3)[:,np.newaxis],
#                                       swap_bm=True)

# prec_pinv = preconditioners.PseudoInvPreconditioner(
#     ainfo, icov_ell, icov_pix, minfo, [0, 2], b_ell=b_ell)

# prec_harm = preconditioners.HarmonicPreconditioner(
#     ainfo, icov_ell, b_ell=b_ell, icov_pix=icov_pix, minfo=minfo)

# prec_masked_cg = preconditioners.MaskedPreconditionerCG(
#     ainfo, icov_ell, [0, 2], mask_I[0].astype(bool), minfo, lmax=1300, # NOTE
#     nsteps=15, lmax_r_ell=None)

# prec_masked_mg = preconditioners.MaskedPreconditioner(
#     ainfo, icov_ell[0:1,0:1], 0, mask_I[0].astype(bool), minfo,
#     min_pix=10000, n_jacobi=1, lmax_r_ell=6000)

# solver.add_preconditioner(prec_pinv)
# #solver.add_preconditioner(prec_harm)
# solver.add_preconditioner(prec_masked_cg)
# solver.init_solver()
# for idx in range(niter_cg):
#     t0 = time.time()
#     solver.step()
#     dt = time.time() - t0
#     print(solver.i, solver.err, dt)
#     ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])

# solver.reset_preconditioner()
# solver.add_preconditioner(prec_pinv)
# #solver.add_preconditioner(prec_harm)
# solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
# solver.b_vec = solver.b0
# solver.init_solver(x0=solver.x)

theory_cls = {'TT' : cov_ell[0,0], 'EE' : cov_ell[1,1], 'BB' : cov_ell[2,2], 'TE' : cov_ell[0,1]}

odict = filters.cg_pix_filter_old(alm, theory_cls, b_ell, lmax,
                      icov_pix=icov_pix, mask_bool=mask_I * np.ones(3)[:,np.newaxis], minfo=minfo,
                      include_te=True, niter=15, ainfo=ainfo,
                      benchmark=5, verbose=True, err_tol=1e-15, rtol_icov=1e-2,
                      order=1, lmax_masked_cg=1300)
solver = odict['solver']


#for idx in range(niter_cg, niter_cg + niter_mg):
    
#    t0 = time.time()
#    solver.step()
#    dt = time.time() - t0
#    print(solver.i, solver.err, dt)
#    ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])

# fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True, squeeze=False)
# for ax in axs.ravel():
#     ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, niter)])
# for idx in range(niter):
#     for aidxs, ax in np.ndenumerate(axs):
#         axs[aidxs].plot(ells, dells * ps_c_ell[idx,aidxs[0],aidxs[1]],
#                         lw=0.5)
# for aidxs, ax in np.ndenumerate(axs):
#     axs[aidxs].plot(ells, dells * cov_ell[aidxs[0],aidxs[1]], lw=0.5, color='black', ls=':')
# axs[0,0].set_ylim(0, 1e4)
# fig.savefig(opj(imgdir, 'ps_c_ell'))
# plt.close(fig)

# for idxs, ax in np.ndenumerate(axs):
#     if idxs[0] == idxs[1]:
#         ax.set_yscale('log')
#         ax.set_ylim(ax.get_ylim())
#         ax.plot(ells, dells * nl[idxs[0],idxs[1]] / b_ell[0] ** 2, lw=1, color='black')
# #axs[0,0].set_ylim(10, 1e6)
# axs[0,0].set_ylim(0.1, 1e4)
# fig.savefig(opj(imgdir, 'ps_c_ell_log'))
# plt.close(fig)

# Plot alms
omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
omap = curvedsky.alm2map(alm, omap)
#omap *= mask_I.reshape(minfo.nrow, minfo.nphi[0])
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=2)
enplot.write(opj(imgdir, 'alm_in'), plot)

# Plot result
omap = curvedsky.alm2map(solver.get_wiener(), omap)
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=2)
enplot.write(opj(imgdir, 'alm_out'), plot)

omap = curvedsky.alm2map(odict['ialm'], omap)
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, downgrade=2)
enplot.write(opj(imgdir, 'alm_icov'), plot)

omap_b_out = curvedsky.alm2map(solver.A(solver.x), omap.copy())
omap_b_in = curvedsky.alm2map(solver.b0, omap.copy())

for pidx in range(alm.shape[0]):

    plot = enplot.plot(omap_b_out[pidx], colorbar=True, grid=False, font_size=50, downgrade=2)
    enplot.write(opj(imgdir, 'b_out_{}'.format(pidx)), plot)

    plot = enplot.plot(omap_b_in[pidx], colorbar=True, grid=False, font_size=50, downgrade=2)
    enplot.write(opj(imgdir, 'b_{}'.format(pidx)), plot)
