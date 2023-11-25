import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils, enmap

from optweight import sht
from optweight import map_utils
from optweight import solvers
from optweight import operators
from optweight import preconditioners
from optweight import mat_utils

opj = os.path.join
np.random.seed(39)

t_only = False
if t_only:
    npol = 1
else:
    npol = 3


#lmax = 5000
#lmax = 2000
lmax = 3000

scaled = True
deflate_cg = True
deflate_mg = True

#basedir = '/home/adriaand/project/actpol/20201115_pcg_act'
basedir = '/home/adriaand/project/actpol/20230228_pcg_act'
#basedir = '/mnt/home/aduivenvoorden/project/actpol/20230725_pcg_act'
# img_adef1_scaled_new means with adef1 incorporated in precondition.py
imgdir = opj(basedir, 'img_adef2_scaled_new')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')
maskdir = '/home/adriaand/project/actpol/20211206_pcg_act/mask'

utils.mkdir(imgdir)

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))

icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest', order=1)
mask_icov = enmap.read_map(opj(maskdir, 'BN_bottomcut_no_apo_dg4.fits'))
mask_icov, _ = map_utils.enmap2gauss(mask_icov, minfo, mode='nearest', order=1)
mask_icov[mask_icov < 0.1] = 0
mask_icov[mask_icov >= 0.1] = 1

if t_only:
    icov_pix = icov_pix[0:1]

mask = np.zeros(icov_pix.shape, dtype=bool)
mask[:] = mask_icov
 
icov_pix_unmasked = icov_pix.copy()
icov_pix[~mask] = 0

for pidx in range(icov_pix.shape[0]):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(mask[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'mask_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(icov_pix.shape[0]):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'icov_real_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(icov_pix.shape[0]):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(np.log10(np.abs(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'icov_real_log_{}'.format(pidx)))
    plt.close(fig)

#cov_pix = np.power(icov_pix, -1, where=mask, out=icov_pix.copy())
#cov_pix = mat_utils.matpow(icov_pix, -1, return_diag=True)
cov_pix = mat_utils.matpow(icov_pix_unmasked, -1, return_diag=True)

for pidx in range(cov_pix.shape[0]):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(
        np.log10(np.abs(cov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'cov_{}'.format(pidx)))
    plt.close(fig)

# Load beam.
b_ell = np.zeros((icov_pix.shape[0], lmax+1))
b_ell[:] = hp.gauss_beam(np.radians(1.3 / 60), lmax=lmax, pol=False)

# Preprare spectrum. Input file is Dls in uk^2.
cov_ell = np.zeros((icov_pix.shape[0], icov_pix.shape[0], lmax + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
    skiprows=1, usecols=[1, 2, 3, 4]) #  TT, EE, BB, TE.
c_ell = c_ell.T
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

cov_ell[0,0,2:] = c_ell[0,:lmax-1] 
if not t_only:
    cov_ell[0,1,2:] = c_ell[3,:lmax-1] 
    cov_ell[1,0,2:] = c_ell[3,:lmax-1] 
    cov_ell[1,1,2:] = c_ell[1,:lmax-1] 
    cov_ell[2,2,2:] = c_ell[2,:lmax-1] 

fig, axs = plt.subplots(ncols=npol, nrows=npol, dpi=300, constrained_layout=True, squeeze=False)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(cov_ell[idxs])
fig.savefig(opj(imgdir, 'cov_ell'))
plt.close(fig)

cov_ell[...,1:] /= dells[1:]

if t_only:
    cov_ell = cov_ell[0:1,0:1,:]

icov_ell = np.ones_like(cov_ell)
for lidx in range(icov_ell.shape[-1]):
    if lidx < 2:
        # Set monopole and dipole to zero.
        icov_ell[:,:,lidx] = 0
    else:
        icov_ell[:,:,lidx] = np.linalg.pinv(cov_ell[:,:,lidx])

fig, axs = plt.subplots(ncols=npol, nrows=npol, dpi=300, constrained_layout=True, squeeze=False)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(icov_ell[idxs])
fig.savefig(opj(imgdir, 'icov_ell'))
plt.close(fig)

# Draw alms.
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
print(alm.dtype)
alm = alm.astype(np.complex64)
alm_signal = alm.copy()
for pidx in range(alm.shape[0]):
    hp.almxfl(alm[pidx], b_ell[pidx], inplace=True)

# Draw map-based noise and add to alm.
noise = map_utils.rand_map_pix(cov_pix)
imap = np.zeros_like(noise)
sht.alm2map(alm, imap, ainfo, minfo, [0,2], adjoint=False)
imap *= mask
imap += noise

alm_data = alm.copy()
sht.map2alm(imap * mask, alm_data, minfo, ainfo, [0,2], adjoint=False)

alm_noise = alm.copy()
sht.map2alm(noise * mask, alm_noise, minfo, ainfo, [0,2], adjoint=False)
nl = ainfo.alm2cl(alm_noise[:,None,:], alm_noise[None,:,:])

fig, axs = plt.subplots(ncols=npol, nrows=npol, dpi=300, constrained_layout=True, squeeze=False)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(nl[idxs])
fig.savefig(opj(imgdir, 'n_ell'))
plt.close(fig)

fig, axs = plt.subplots(ncols=npol, nrows=npol, dpi=300, constrained_layout=True, squeeze=False)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(ells, dells * (cov_ell[idxs] + nl[idxs]))
    axs[idxs].plot(ells, dells * nl[idxs])
    axs[idxs].plot(ells, dells * cov_ell[idxs])
fig.savefig(opj(imgdir, 'tot_ell'))
plt.close(fig)

#alm += alm_noise

# NOTE
#alm_x = alm.copy()

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))

# Apply mask.
#sht.alm2map(alm, noise, ainfo, minfo, [0,2], adjoint=False)
#noise *= mask 
#sht.map2alm(noise, alm, minfo, ainfo, [0,2], adjoint=False)

# Solve for Wiener filtered alms.

niter_cg = 8 
niter_mg = 20
niter = niter_cg + niter_mg

errors = np.zeros((niter + 1))
rel_errors = np.zeros(niter + 1)
chisqs = np.zeros_like(errors)
residuals = np.zeros_like(errors)
ps_c_ell = np.zeros((niter, npol, npol, lmax + 1))


x0 = None
mask_pix = mask.astype(np.float32)

np.random.seed(30)

if scaled:
    sfilt = mat_utils.matpow(b_ell, -0.5)
else:
    sfilt = None

solver = solvers.CGWienerMap.from_arrays(imap, minfo, ainfo, icov_ell, icov_pix,
                                         draw_constr=True,
                                         b_ell=b_ell, mask_pix=mask_pix,
                                         spin=[0, 2], swap_bm=False, sfilt=sfilt)


prec_pinv = preconditioners.PseudoInvPreconditioner(
    ainfo, icov_ell, icov_pix, minfo, [0, 2], b_ell=b_ell, sfilt=sfilt)

prec_harm = preconditioners.HarmonicPreconditioner(
    ainfo, icov_ell, b_ell=b_ell, icov_pix=icov_pix, minfo=minfo, sfilt=sfilt)

prec_masked_cg = preconditioners.MaskedPreconditionerCG(
    ainfo, icov_ell, [0, 2], mask_pix[0].astype(bool), minfo, lmax=3000,
    nsteps=15, lmax_r_ell=None, sfilt=sfilt)

prec_masked_mg = preconditioners.MaskedPreconditioner(
    ainfo, icov_ell[0:1,0:1], 0, mask_pix[0].astype(bool), minfo,
    min_pix=10000, n_jacobi=1, lmax_r_ell=6000, sfilt=sfilt[0:1,0:1])


if deflate_cg:

    # A-DEF1
    # def prec_cg_adef1(alm):
    #     q = prec_masked_cg(alm)
    #     out = prec_pinv(alm - solver.A(q))
    #     out += q
    #     return out
   
    # solver.add_preconditioner(prec_cg_adef1)

    solver.add_preconditioner(
        preconditioners.get_2level_prec(prec_pinv, prec_masked_cg, solver,
                                        'ADEF-2', sel_masked=None))
else:

    solver.add_preconditioner(prec_pinv)
    solver.add_preconditioner(prec_masked_cg)

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))


solver.init_solver(x0=x0)
for idx in range(niter_cg):
    solver.step()
    print(solver.i, solver.err)

    #omap = curvedsky.alm2map(solver.get_wiener(), omap)
    #plot = enplot.plot(omap, colorbar=True, font_size=150, grid=False, range='250:10', downgrade=4)
    #enplot.write(opj(imgdir, f'alm_x_{idx:02d}'), plot)

    ps_c_ell[idx,...] = ainfo.alm2cl(solver.get_wiener()[:,None,:], solver.get_wiener()[None,:,:])


solver.reset_preconditioner()

if deflate_mg:
    # def prec_masked_mg_pol(alm):
    #     out = np.zeros_like(alm)
    #     out[0] = prec_masked_mg(alm[0:1].copy())
    #     return out

    # def prec_mg_adef1(alm):
    #     q = prec_masked_mg_pol(alm)
    #     out = prec_pinv(alm - solver.A(q))
    #     out += q
    #     return out

    # solver.add_preconditioner(prec_mg_adef1)
    solver.add_preconditioner(
        preconditioners.get_2level_prec(prec_pinv, prec_masked_mg, solver,
                                        'ADEF-2', sel_masked=np.s_[0]))

else:
    solver.add_preconditioner(prec_pinv)
    solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])

solver.b_vec = solver.b0
solver.init_solver(x0=solver.x)


#NOTE
#solver.b = solver.A(alm_x.copy())
#solver.r = solver.b

errors[0] = np.nan
#chisqs[sidx,0] = solver.get_chisq()
residuals[0] = solver.get_residual()
#rel_errors[sidx,0] = np.sqrt(solver.dot(alm_x - solver.x, alm_x - solver.x))

b_copy = solver.b.copy()

print('|b| :', np.sqrt(solver.dot(b_copy, b_copy)))

#if aidx == 0:
#    niter_real = 10
#else:
#    niter_real = niter

#for idx in range(niter):
#for idx in range(niter_real):
for idx in range(niter_cg, niter_cg + niter_mg):
    t0 = time.time()
    solver.step()

    #omap = curvedsky.alm2map(solver.get_wiener(), omap)
    #plot = enplot.plot(omap, colorbar=True, font_size=150, grid=False, range='250:10', downgrade=4)
    #enplot.write(opj(imgdir, f'alm_x_{idx:02d}'), plot)

    dt = time.time() - t0
    error = solver.err
    #errors[idx+1] = error
    #rel_error = np.sqrt(solver.dot(alm_x - solver.x, alm_x - solver.x))
    #rel_errors[sidx,idx+1] = rel_error
    #chisq = solver.get_chisq()
    chisq = 0
    #chisqs[sidx,idx+1] = chisq
    #residual = solver.get_residual()
    residual = 0
    #residuals[sidx,idx+1] = residual
    ps_c_ell[idx,...] = ainfo.alm2cl(solver.get_wiener()[:,None,:], solver.get_wiener()[None,:,:])
    print(solver.i, error, chisq / alm.size / 2, residual, dt)#rel_error, dt)


# Save all arrays.
#np.save(opj(imgdir, 'residuals'), residuals)
#np.save(opj(imgdir, 'chisqs'), chisqs)
#np.save(opj(imgdir, 'errors'), errors)
#np.save(opj(imgdir, 'cov_ell'), cov_ell)
#np.save(opj(imgdir, 'n_ell'), nl)
#np.save(opj(imgdir, 'ps_c_ell'), ps_c_ell)

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

fig, axs = plt.subplots(nrows=npol, dpi=300, constrained_layout=True, sharex=True)
for ax in axs.ravel():
    ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, niter)])
for pidx in range(npol):
    axs[pidx].plot(ells, dells * nl[pidx,pidx] / b_ell[0] ** 2, lw=1, color='black', label=r'noise, $N_{\ell}/B_{\ell}^2$ ')

for idx in range(niter):
    for pidx in range(npol):
        axs[pidx].plot(ells, dells * ps_c_ell[idx,pidx,pidx],
                        lw=0.5)
for pidx in range(3):
    axs[pidx].plot(ells, dells * cov_ell[pidx,pidx], lw=0.5, color='black', ls=':', label=r'signal, $S_{\ell}$')
axs[0].set_ylim(0, 7000)
axs[1].set_ylim(0, 50)
axs[2].set_ylim(0, 0.15)
axs[0].set_ylabel(r'$D^{TT}_{\ell}$ [$\mathrm{\mu K_{CMB}}^2$]')
axs[1].set_ylabel(r'$D^{EE}_{\ell}$ [$\mathrm{\mu K_{CMB}}^2$]')
axs[2].set_ylabel(r'$D^{BB}_{\ell}$ [$\mathrm{\mu K_{CMB}}^2$]')
axs[2].set_xlabel(r'Multipole, $\ell$')
axs[0].legend(frameon=False)

fig.savefig(opj(imgdir, 'ps_c_ell_diag'))
plt.close(fig)


# fig, axs = plt.subplots(nrows=4, dpi=300, sharex=True, figsize=(4, 6), 
#                         constrained_layout=True)
# for sidx, stype in enumerate(stypes):
#     axs[0].plot(errors[sidx], label=stype)
#     #axs[1].plot(chisqs[sidx], label=stype)
#     axs[2].plot(residuals[sidx], label=stype)
#     #axs[3].plot(rel_errors[sidx], label=stype)
# axs[0].set_ylabel('error')
# axs[1].set_ylabel('chisq')
# axs[2].set_ylabel('residual')
# axs[3].set_ylabel('rel. error')
# axs[0].legend()
# for ax in axs:
#     ax.set_yscale('log')
# axs[2].set_xlabel('steps')
# fig.savefig(opj(imgdir, 'stats'))
# plt.close(fig)

# Plot alms
omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))

omap = curvedsky.alm2map(alm_signal, omap)
plot = enplot.plot(omap, colorbar=True, font_size=150, grid=False, range='250:10', downgrade=4)
enplot.write(opj(imgdir, 'alm_signal'), plot)

omap = curvedsky.alm2map(alm, omap)
plot = enplot.plot(omap, colorbar=True, font_size=150, grid=False, range='250:10', downgrade=4)
enplot.write(opj(imgdir, 'alm_in'), plot)

omap = curvedsky.alm2map(alm_data, omap)
plot = enplot.plot(omap, colorbar=True, font_size=150, grid=False, range='250:10', downgrade=4)
enplot.write(opj(imgdir, 'alm_data'), plot)

# Plot result
omap = curvedsky.alm2map(solver.get_wiener(), omap)
plot = enplot.plot(omap, colorbar=True, font_size=150, grid=False, range='250:10', downgrade=4)
enplot.write(opj(imgdir, 'alm_out'), plot)

omap = curvedsky.alm2map(solver.get_icov(), omap)
for pidx in range(omap.shape[0]):
    plot = enplot.plot(omap[pidx], colorbar=True, font_size=150, grid=False, downgrade=4)
    enplot.write(opj(imgdir, f'alm_icov_{pidx}'), plot)

omap_b_out = curvedsky.alm2map(solver.A(solver.x), omap.copy())
omap_b_in = curvedsky.alm2map(b_copy, omap.copy())
#omap_x_in = curvedsky.alm2map(alm_x, omap.copy())

for pidx in range(omap_b_out.shape[0]):
    plot = enplot.plot(omap_b_out[pidx], colorbar=True, grid=False, font_size=150, downgrade=4)
    enplot.write(opj(imgdir, f'b_out_{pidx}'), plot)

    plot = enplot.plot(omap_b_in[pidx], colorbar=True, grid=False, font_size=150, downgrade=4)
    enplot.write(opj(imgdir, f'b_{pidx}'), plot)

    plot = enplot.plot(omap_b_in[pidx] - omap_b_out[pidx], colorbar=True, grid=False, font_size=150, downgrade=4)
    enplot.write(opj(imgdir, f'b_diff_{pidx}'), plot)

#plot = enplot.plot(omap_x_in, colorbar=True, grid=False, font_size=150, downgrade=4, range='250:10')
#enplot.write(opj(imgdir, 'x_in'), plot)

#for pidx in range(3):
#    plot = enplot.plot(omap_x_in[pidx] - omap[pidx], colorbar=True, grid=False, font_size=150, downgrade=4)
#    enplot.write(opj(imgdir, f'x_diff_{pidx}'), plot)
