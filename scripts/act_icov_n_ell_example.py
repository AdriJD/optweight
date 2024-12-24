'''
Example script that Wiener filters a simulated sky using the 
ACT + Plack coadd ivar map as inverse noise variance.
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import healpy as hp
from pixell import curvedsky, enplot, enmap

from optweight import sht
from optweight import map_utils, mat_utils
from optweight import solvers

opj = os.path.join
np.random.seed(39)

lmax = 5000

###### Replace with your paths.

# Output dirs.
basedir = '/home/adriaand/project/actpol/20201115_pcg_act'
imgdir = opj(basedir, 'img')

# Ivar and spectra paths.
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')

#raise ValueError('Replace paths with your own paths')

######

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))
icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest', order=1)
icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=1e-2)
mask = icov_pix != 0

## Set too small values to zero. Note that this varies a little with lmax!
##mask = icov_pix > 1e-4 # Seems to work well for lmax=5000.
#mask = icov_pix > 1e-3 # Use for lmax=500.
#icov_pix[~mask] = 0

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

cov_pix = np.power(icov_pix, -1, where=mask, out=icov_pix.copy())

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(
        np.log10(np.abs(cov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'cov_{}'.format(pidx)))
    plt.close(fig)

# Make up a reasonable beam.
b_ell = hp.gauss_beam(np.radians(1.3 / 60), lmax=lmax, pol=True)
b_ell = np.ascontiguousarray(b_ell[:,:3].T)

# Prepare spectrum. Input file is from CAMB so Dls in uk^2.
cov_ell = np.zeros((3, 3, lmax + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
    skiprows=1, usecols=[1, 2, 3, 4]) #  TT, EE, BB, TE.
c_ell = c_ell.T
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

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

# Invert to get inverse signal cov.
#icov_ell = np.ones((3, 3, lmax + 1))
#for lidx in range(icov_ell.shape[-1]):
#    if lidx < 2:
#        # Set monopole and dipole to zero.
#        icov_ell[:,:,lidx] = 0
#    else:
#        icov_ell[:,:,lidx] = np.linalg.inv(cov_ell[:,:,lidx])
#for lidx in range(icov_ell.shape[-1]):
    #if lidx < 2:
    #    # Set monopole and dipole to zero.
    #    icov_ell[:,:,lidx] = 0
    #else:
#    icov_ell[:,:,lidx] = np.linalg.pinv(cov_ell[:,:,lidx])
icov_ell = mat_utils.matpow(cov_ell, -1)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(icov_ell[idxs])
fig.savefig(opj(imgdir, 'icov_ell'))
plt.close(fig)

cov_noise_ell = np.ones((3, lmax+1))
#cov_noise_ell[:,1:] *= (150 * np.nan_to_num(1 / ells[1:]) ** 0.4 + 1) 
cov_noise_ell[:,1:] *= (1500 * np.nan_to_num(1 / ells[1:])) ** 2 + 1
cov_noise_ell[:,0] = cov_noise_ell[:,1]
cov_noise_ell /= cov_noise_ell[0,-1]
cov_noise_ell = mat_utils.full_matrix(cov_noise_ell)
icov_noise_ell = mat_utils.matpow(cov_noise_ell, -1)
print(cov_noise_ell[0])
print(cov_noise_ell.shape)
print(icov_noise_ell.shape)
print(icov_noise_ell[0])
#exit()
#icov_noise_ell = None

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(cov_noise_ell[idxs])
fig.savefig(opj(imgdir, 'cov_noise_ell'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(cov_noise_ell[idxs])
    axs[idxs].set_yscale('log')
fig.savefig(opj(imgdir, 'cov_noise_ell_log'))
plt.close(fig)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(icov_noise_ell[idxs])
fig.savefig(opj(imgdir, 'icov_noise_ell'))
plt.close(fig)

# Draw alms.
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
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
omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))

# Apply mask.
sht.alm2map(alm, noise, ainfo, minfo, [0,2], adjoint=False)
noise *= mask
sht.map2alm(noise, alm, minfo, ainfo, [0,2], adjoint=False)

# Solve for Wiener filtered alms.
niter = 35
#stypes = ['cg', 'cg_scaled', 'pcg_harm', 'pcg_pinv']
stypes = ['pcg_harm']
errors = np.zeros((len(stypes), niter + 1))
chisqs = np.zeros_like(errors)
residuals = np.zeros_like(errors)
ps_c_ell = np.zeros((niter, 3, 3, lmax + 1))

for sidx, stype in enumerate(stypes):
    
    if stype == 'cg' or stype == 'cg_scaled':
        prec = None
    elif stype == 'pcg_harm':
        prec = 'harmonic'
    elif stype == 'pcg_pinv':
        prec = 'pinv'
    print('sidx', sidx)

    x0 = None
    
    if stype == 'cg_scaled':
        solver = solvers.CGWienerScaled.from_arrays(alm, ainfo, icov_ell, icov_pix, minfo, b_ell=b_ell,
                                                    draw_constr=False, prec=None, x0=x0)
    else:
        solver = solvers.CGWiener.from_arrays(alm, ainfo, icov_ell, icov_pix, minfo, b_ell=b_ell,
                                              draw_constr=False, prec=prec, x0=x0,
                                              icov_noise_flat_ell=icov_noise_ell)

    b_copy = solver.b.copy()

    # Note, calculating all this metainfo is quite expensive. To time the solver only, you
    # can remove all lines below except for the solver.step() call.
    errors[sidx,0] = np.nan
    chisqs[sidx,0] = solver.get_chisq()
    residuals[sidx,0] = solver.get_residual()

    if sidx == 0:
        print('|b| :', np.sqrt(solver.dot(b_copy, b_copy)))

    for idx in range(niter):
        solver.step()
        error = solver.err
        errors[sidx,idx+1] = error
        chisq = solver.get_chisq()
        chisqs[sidx,idx+1] = chisq
        residual = solver.get_residual()
        residuals[sidx,idx+1] = residual
        ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])
        print(solver.i, error, chisq / alm.size / 2, residual)

# Save all arrays.
np.save(opj(imgdir, 'residuals'), residuals)
np.save(opj(imgdir, 'chisqs'), chisqs)
np.save(opj(imgdir, 'errors'), errors)
np.save(opj(imgdir, 'cov_ell'), cov_ell)
np.save(opj(imgdir, 'n_ell'), nl)
np.save(opj(imgdir, 'ps_c_ell'), ps_c_ell)

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for ax in axs.ravel():
    ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, niter)])
for idx in range(niter):
    for aidxs, ax in np.ndenumerate(axs):
        axs[aidxs].plot(ells, dells * ps_c_ell[idx,aidxs[0],aidxs[1]],
                        lw=0.5)
fig.savefig(opj(imgdir, 'ps_c_ell'))
plt.close(fig)

fig, axs = plt.subplots(nrows=3, dpi=300, sharex=True, figsize=(4, 6), 
                        constrained_layout=True)
for sidx, stype in enumerate(stypes):
    axs[0].plot(errors[sidx], label=stype)
    axs[1].plot(chisqs[sidx], label=stype)
    axs[2].plot(residuals[sidx], label=stype)
axs[0].set_ylabel('error')
axs[1].set_ylabel('chisq')
axs[2].set_ylabel('residual')
axs[0].legend()
for ax in axs:
    ax.set_yscale('log')
axs[2].set_xlabel('steps')
fig.savefig(opj(imgdir, 'stats'))
plt.close(fig)

# Plot alms
omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))

omap = curvedsky.alm2map(alm_signal, omap)
plot = enplot.plot(omap, colorbar=True, font_size=100, grid=False, range='250:10')
enplot.write(opj(imgdir, 'alm_signal'), plot)

omap = curvedsky.alm2map(alm, omap)
plot = enplot.plot(omap, colorbar=True, font_size=100, grid=False, range='250:10')
enplot.write(opj(imgdir, 'alm_in'), plot)

# Plot result
omap = curvedsky.alm2map(solver.get_wiener(), omap)
plot = enplot.plot(omap, colorbar=True, font_size=100, grid=False, range='250:10')
enplot.write(opj(imgdir, 'alm_out'), plot)

omap_b_out = curvedsky.alm2map(solver.A(solver.x), omap.copy())
omap_b_in = curvedsky.alm2map(b_copy, omap.copy())

for pidx in range(alm.shape[0]):

    plot = enplot.plot(omap_b_out[pidx], colorbar=True, grid=False, font_size=100)
    enplot.write(opj(imgdir, 'b_out_{}'.format(pidx)), plot)

    plot = enplot.plot(omap_b_in[pidx], colorbar=True, grid=False, font_size=100)
    enplot.write(opj(imgdir, 'b_{}'.format(pidx)), plot)
