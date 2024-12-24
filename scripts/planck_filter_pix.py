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

from optweight import sht, map_utils, mat_utils, solvers, operators, preconditioners

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
    b_ell[0] = b_ell_T[:lmax+1]
    b_ell[1] = b_ell_E[:lmax+1]
    b_ell[2] = b_ell_B[:lmax+1]

    return b_ell

#lmax = 1500
lmax = 2500
#lmax = 1000

basedir = '/home/adriaand/project/actpol/20230222_pcg_planck'
maskdir = '/home/adriaand/project/actpol/20201009_pcg_planck/meta'
imgdir = opj(basedir, 'img_new_full')

utils.mkdir(imgdir)

# Load II, IQ, IU, QQ, QU, UU cov.
cov = hp.read_map(opj(maskdir, 'HFI_SkyMap_100_2048_R3.01_full.fits'), field=(4, 5, 6, 7, 8, 9))
cov *= 1e12 # Convert from K^2 to uK^2.

cov, minfo = map_utils.healpix2gauss(cov, 2*lmax, area_pow=-1)


# NOTE
#cov /= 4
#cov[0,0] /= 32

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

# NOTE try thresholding low *cov* values.
#cov_pix = map_utils.round_icov_matrix(cov_pix, rtol=1e-1, threshold=True)
#cov_pix = map_utils.round_icov_matrix(cov_pix, rtol=0.25, threshold=True)

# Set to 0 for img_new_full
#cov_pix = map_utils.round_icov_matrix(cov_pix, rtol=1e-1, threshold=True)
#cov_pix *= np.eye(3)[:,:,None]

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
#icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=1e-2, threshold=True)
# NOTE
#icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=1e-1, threshold=True)

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


# NOTE NOTE
#b_ell = b_ell ** 0.5
print(b_ell[0,::50])
#exit()
#b_ell[:,2000:] = (b_ell[:,2000])[:,None]
#b_ell = np.ones_like(b_ell)
#b_ell *= hp.gauss_beam(np.radians(2/60), lmax=lmax)[None,:]

# Preprare spectrum. Input file is Dls in uk^2.
c_ell = np.loadtxt(
    opj(maskdir, 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'),
    skiprows=1, usecols=[1, 2, 3, 4]) #  TT, TE, EE, BB.
c_ell = c_ell.T
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi
cov_ell = np.zeros((3, 3, lmax + 1))
cov_ell[0,0,2:] = c_ell[0,:lmax-1]
cov_ell[0,1,2:] = c_ell[1,:lmax-1]
cov_ell[1,0,2:] = c_ell[1,:lmax-1]
cov_ell[1,1,2:] = c_ell[2,:lmax-1]
cov_ell[2,2,2:] = c_ell[3,:lmax-1]

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(cov_ell[idxs])
fig.savefig(opj(imgdir, 'cov_ell'))
plt.close(fig)

cov_ell[...,1:] /= dells[1:]

icov_ell = np.ones((3, 3, lmax + 1))
for lidx in range(icov_ell.shape[-1]):
    #if lidx < 2:
    #    # Set monopole and dipole to zero.
    #    icov_ell[:,:,lidx] = 0
    #else:
    icov_ell[:,:,lidx] = np.linalg.pinv(cov_ell[:,:,lidx])

# Draw alms.
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
alm_signal = alm.copy()
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

imap = np.zeros((3, minfo.npix))
sht.alm2map(alm, imap, ainfo, minfo, [0, 2])

# NOTE
imap *= mask_I
#alm_tmp = alm.copy()
#sht.map2alm(imap, alm_tmp, minfo, ainfo, [0,2], adjoint=False)
#for pidx in range(alm.shape[0]):
#    hp.almxfl(alm_tmp[pidx], b_ell[pidx], inplace=True)
#imap_tmp = imap.copy()
#sht.alm2map(alm_tmp, imap_tmp, ainfo, minfo, [0,2], adjoint=False)
#imap[:,mask_I[0] == 0] = imap_tmp[:,mask_I[0] == 0]



imap += noise

# NOTE
#imap *= mask_I

# NOTE
#imap = map_utils.inpaint_nearest(imap, mask_I.astype(bool), minfo)


#alm += alm_noise

# Apply mask;
#sht.alm2map(alm, noise, ainfo, minfo, [0,2], adjoint=False)
#noise *= mask_I
#sht.map2alm(noise, alm, minfo, ainfo, [0,2], adjoint=False)

niter_cg = 7
niter_mg = 15
#niter_cg = 15
#niter_mg = 25

niter = niter_cg + niter_mg

ps_c_ell = np.zeros((niter, 3, 3, lmax + 1))


# NOTE
#minfo_ea = map_utils.get_equal_area_gauss_minfo(
#    2 * lmax, gl_band=0.4, ratio_pow=1.)
#icov_pix_ea = map_utils.gauss2map(icov_pix, minfo, minfo_ea, order=1, area_pow=1)
#mask_I_ea = map_utils.gauss2map(mask_I, minfo, minfo_ea, order=1, area_pow=0)
#mask_I_ea[mask_I_ea < 0.5] = 0
#mask_I_ea[mask_I_ea >= 0.5] = 1
#imap_ea = map_utils.gauss2map(imap, minfo, minfo_ea, order=1, area_pow=0)
#icov_pix = icov_pix_ea
#mask_I = mask_I_ea
#imap = imap_ea
#minfo = minfo_ea




solver = solvers.CGWienerMap.from_arrays(imap, minfo, ainfo, icov_ell, icov_pix, draw_constr=False,
                                         b_ell=b_ell, mask_pix=mask_I * np.ones(3)[:,np.newaxis],
                                         spin=[0, 2], swap_bm=True)

#solver = solvers.CGWienerMap.from_arrays(imap, minfo, ainfo, icov_ell, icov_pix, draw_constr=False,
#                                         b_ell=b_ell, mask_pix=mask_I_ea * np.ones(3)[:,np.newaxis],
#                                         spin=[0, 2], swap_bm=True, minfo_mask=minfo_ea)

# NOTE
#imap_prime = solver.proj(alm_signal) + noise
#diff = imap_prime - imap

#for pidx in range(3):
#    fig, ax = plt.subplots(dpi=300, figsize=(10, 8))
#    im = ax.imshow(diff[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
#    fig.colorbar(im, ax=ax)
#    fig.savefig(opj(imgdir, 'diff_{}'.format(pidx)))
#    plt.close(fig)



#solver.imap = imap
#solver.b_vec = solver.get_b_vec(solver.imap)
#solver.b0 = solver.b_vec.copy()


#solver = solvers.CGWienerMap.from_arrays(imap_ea, minfo_ea, ainfo, icov_ell, icov_pix_ea,
#                                         draw_constr=False,
#                                         b_ell=b_ell, mask_pix=mask_I_ea * np.ones(3)[:,np.newaxis],
#                                         spin=[0, 2], swap_bm=False)


# Using this vs unmasked icov_pix seems to do a little bit better.
# I think the only thing is that the pinv preconditiner can invert the
# masked ivar.

# If swap_bm is set, you don't want to mask for some reason.
icov_pix_prec = icov_pix.copy()
#icov_pix_prec[0,0] *= mask_I[0]
#icov_pix_prec[0,1] *= mask_I[0]
#icov_pix_prec[0,2] *= mask_I[0]
#icov_pix_prec[1,0] *= mask_I[0]
#icov_pix_prec[1,1] *= mask_I[0]
#icov_pix_prec[1,2] *= mask_I[0]
#icov_pix_prec[2,0] *= mask_I[0]
#icov_pix_prec[2,1] *= mask_I[0]
#icov_pix_prec[2,2] *= mask_I[0]

# icov_pix_prec_ea = icov_pix_ea.copy()
# icov_pix_prec_ea[0,0] *= mask_I_ea[0]
# icov_pix_prec_ea[0,1] *= mask_I_ea[0]
# icov_pix_prec_ea[0,2] *= mask_I_ea[0]
# icov_pix_prec_ea[1,0] *= mask_I_ea[0]
# icov_pix_prec_ea[1,1] *= mask_I_ea[0]
# icov_pix_prec_ea[1,2] *= mask_I_ea[0]
# icov_pix_prec_ea[2,0] *= mask_I_ea[0]
# icov_pix_prec_ea[2,1] *= mask_I_ea[0]
# icov_pix_prec_ea[2,2] *= mask_I_ea[0]


#prec_pinv = preconditioners.PseudoInvPreconditioner(
#    ainfo, icov_ell, icov_pix, minfo, [0, 2], b_ell=b_ell)

prec_pinv = preconditioners.PseudoInvPreconditioner(
  ainfo, icov_ell, icov_pix_prec, minfo, [0, 2], b_ell=b_ell)

prec_harm = preconditioners.HarmonicPreconditioner(
  ainfo, icov_ell, b_ell=b_ell, icov_pix=icov_pix, minfo=minfo)

#prec_masked_cg = preconditioners.MaskedPreconditionerCG(
#  ainfo, icov_ell, [0, 2], mask_I[0].astype(bool), minfo, lmax=1500, # NOTE
#  nsteps=15, lmax_r_ell=None)

prec_masked_cg = preconditioners.MaskedPreconditionerCG(
  ainfo, icov_ell, [0, 2], mask_I[0].astype(bool), minfo, lmax=None, # NOTE
  nsteps=25, lmax_r_ell=None)

#prec_masked_cg = preconditioners.MaskedPreconditionerCG(
#  ainfo, icov_ell, [0, 2], mask_I[0].astype(bool), minfo, lmax=None, # NOTE
#  nsteps=15, lmax_r_ell=None)


#prec_masked_cg = preconditioners.MaskedPreconditionerCG(
#  ainfo, icov_ell, [0, 2], mask_I[0].astype(bool), minfo, lmax=1500, # NOTE
#  nsteps=15, lmax_r_ell=None)

#prec_masked_mg = preconditioners.MaskedPreconditioner(
#  ainfo, icov_ell[0:1,0:1], 0, mask_I[0].astype(bool), minfo,
#  min_pix=10000, n_jacobi=1, lmax_r_ell=6000)

prec_masked_mg = preconditioners.MaskedPreconditioner(
  ainfo, icov_ell[0:1,0:1], 0, mask_I[0].astype(bool), minfo,
  min_pix=1000, n_jacobi=1, lmax_r_ell=6000)

#prec_masked_mg = preconditioners.MaskedPreconditioner(
#  ainfo, icov_ell[0:1,0:1], 0, mask_I[0].astype(bool), minfo,
#  min_pix=1000, n_jacobi=3, lmax_r_ell=6000)

#prec_masked_mg_pol = preconditioners.MaskedPreconditioner(
#  ainfo, icov_ell, [0, 2], mask_I[0].astype(bool), minfo,
#  min_pix=1000, n_jacobi=1, lmax_r_ell=6000)

# prec_pinv = preconditioners.PseudoInvPreconditioner(
#     ainfo, icov_ell, icov_pix_prec_ea, minfo_ea, [0, 2], b_ell=b_ell)

# prec_harm = preconditioners.HarmonicPreconditioner(
#     ainfo, icov_ell, b_ell=b_ell, icov_pix=icov_pix_ea, minfo=minfo)

# prec_masked_cg = preconditioners.MaskedPreconditionerCG(
#     ainfo, icov_ell, [0, 2], mask_I[0].astype(bool), minfo, lmax=1300, # NOTE
#     nsteps=15, lmax_r_ell=None)

# prec_masked_mg = preconditioners.MaskedPreconditioner(
#     ainfo, icov_ell[0:1,0:1], 0, mask_I[0].astype(bool), minfo,
#     min_pix=10000, n_jacobi=1, lmax_r_ell=6000)

solver.add_preconditioner(prec_pinv)
#solver.add_preconditioner(prec_harm)
solver.add_preconditioner(prec_masked_cg)
#solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
solver.init_solver()
for idx in range(niter_cg):
    t0 = time.time()
    solver.step()
    dt = time.time() - t0
    print(solver.i, solver.err, solver.get_qform(), dt)
    ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])

solver.reset_preconditioner()


#def prec_masked_mg_pol(alm):
#    out = np.zeros_like(alm)
#    out[0] = prec_masked_mg(alm[0:1].copy())
#    return out

#p_op = lambda alm : alm - solver.A(prec_masked_mg_pol(alm))
#mp_op = lambda alm : prec_pinv(p_op(alm))

#def mp_op_full(alm):
#    #tmp = prec_masked_cg(alm)
#    tmp = prec_masked_mg_pol(alm)
#    return prec_pinv(alm - solver.A(tmp)) + tmp
#solver.add_preconditioner(mp_op_full)
# NOTE
solver.add_preconditioner(prec_pinv)
#solver.add_preconditioner(prec_harm)
solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
#solver.add_preconditioner(prec_masked_cg)
#solver.add_preconditioner(prec_masked_mg_pol)

#solver.add_preconditioner(prec_pinv)
#solver.add_preconditioner(prec_harm)
#solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
solver.b_vec = solver.b0
solver.init_solver(x0=solver.x)

for idx in range(niter_cg, niter_cg + niter_mg):

    t0 = time.time()
    solver.step()
    dt = time.time() - t0
    print(solver.i, solver.err, solver.get_qform(), dt)
    ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True, squeeze=False)
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

# Plot alms
omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
#omap = curvedsky.alm2map(alm, omap)
#omap *= mask_I.reshape(minfo.nrow, minfo.nphi[0])
#plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=2)
#enplot.write(opj(imgdir, 'alm_in'), plot)

alm_imap = alm.copy()
sht.map2alm(imap, alm_imap, minfo, ainfo, [0,2], adjoint=False)
omap = curvedsky.alm2map(alm_imap, omap)
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=4)
enplot.write(opj(imgdir, 'alm_in'), plot)

# Plot result
omap = curvedsky.alm2map(solver.get_wiener(), omap)
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=4)
enplot.write(opj(imgdir, 'alm_out'), plot)

omap = curvedsky.alm2map(solver.get_icov(), omap)
for pidx in range(omap.shape[0]):
    plot = enplot.plot(omap[pidx], colorbar=True, font_size=50, grid=False, downgrade=4)
    enplot.write(opj(imgdir, f'alm_icov_{pidx}'), plot)

omap_b_out = curvedsky.alm2map(solver.A(solver.x), omap.copy())
omap_b_in = curvedsky.alm2map(solver.b0, omap.copy())

for pidx in range(alm.shape[0]):

    plot = enplot.plot(omap_b_out[pidx], colorbar=True, grid=False, font_size=50, downgrade=4)
    enplot.write(opj(imgdir, 'b_out_{}'.format(pidx)), plot)

    plot = enplot.plot(omap_b_in[pidx], colorbar=True, grid=False, font_size=50, downgrade=4)
    enplot.write(opj(imgdir, 'b_{}'.format(pidx)), plot)

    plot = enplot.plot(omap_b_out[pidx] - omap_b_in[pidx],
                       colorbar=True, grid=False, font_size=50, downgrade=4)
    enplot.write(opj(imgdir, 'b_diff_{}'.format(pidx)), plot)
