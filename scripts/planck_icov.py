import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils
from enlib import cg

from optweight import sht
from optweight import map_utils
from optweight import solvers
from optweight import operators
from optweight import preconditioners

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

lmax = 2500

basedir = '/home/adriaand/project/actpol/20201009_pcg_planck'
maskdir = opj(basedir, 'meta')
imgdir = opj(basedir, 'img')
#imgdir = '/home/adriaand/project/actpol/20201021_pcg_gg/convergence'

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


# NOTE
#cov_pix[0,1] = 0
#cov_pix[0,2] = 0
#cov_pix[1,0] = 0
#cov_pix[1,2] = 0
#cov_pix[2,0] = 0
#cov_pix[2,1] = 0

icov_pix = np.ascontiguousarray(np.linalg.inv(cov_pix.transpose(2, 0, 1)).transpose(1, 2, 0))
for idx in range(3):
    for jdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(icov_pix[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(imgdir, 'icov_{}_{}'.format(idx, jdx)))
        plt.close(fig)

# Load mask.
mask = True
if mask:
    #mask_I = hp.read_map(opj(maskdir, 'COM_Mask_CMB-Inpainting-Mask-Int_2048_R3.00.fits'), field=0)
    #mask_P = hp.read_map(opj(maskdir, 'COM_Mask_CMB-Inpainting-Mask-Pol_2048_R3.00.fits'), field=0)
    mask_I = hp.read_map(opj(maskdir, 'COM_Mask_Likelihood-temperature-100-hm2_2048_R3.00.fits'), field=0)

    print('fsky', np.sum(mask_I) / mask_I.size)
    #mask_P = mask_I.copy()
    mask_I, _ = map_utils.healpix2gauss(mask_I[np.newaxis,:], 2*lmax, area_pow=0)
    #mask_P, _ = map_utils.healpix2gauss(mask_P[np.newaxis,:], 2*lmax, area_pow=0)

    # NOTE
    mask_I[mask_I < 0.1] = 0

    icov_pix[0,0] *= mask_I[0]
    icov_pix[0,1] *= mask_I[0]
    icov_pix[0,2] *= mask_I[0]
    icov_pix[1,0] *= mask_I[0]
    icov_pix[1,1] *= mask_I[0]
    icov_pix[1,2] *= mask_I[0]
    icov_pix[2,0] *= mask_I[0]
    icov_pix[2,1] *= mask_I[0]
    icov_pix[2,2] *= mask_I[0]

cov_pix_eig = np.ascontiguousarray(np.transpose(icov_pix, (2, 0, 1)))
mask_zero = (cov_pix_eig[:,0,0] == 0) & (cov_pix_eig[:,1,1] == 0) & (cov_pix_eig[:,2,2]== 0)
cov_pix_eig[~mask_zero,:,:] = np.linalg.inv(cov_pix_eig[~mask_zero,:,:])
cov_pix_eig = np.ascontiguousarray(np.transpose(cov_pix_eig, (1, 2, 0)))

for idx in range(3):
    for jdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(cov_pix_eig[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(imgdir, 'cov_eig_log_{}_{}'.format(idx, jdx)))
        plt.close(fig)

for idx in range(3):
    for jdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(cov_pix_eig[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(imgdir, 'cov_eig_{}_{}'.format(idx, jdx)))
        plt.close(fig)


# Load beam.
b_ell = get_planck_b_ell(opj(maskdir, 'BeamWf_HFI_R3.01', 'Bl_TEB_R3.01_fullsky_100x100.fits'), lmax)
#b_ell = np.ascontiguousarray((hp.gauss_beam(np.radians(5), lmax=lmax, pol=True)[:,:-1]).T)
#b_ell = np.ones_like(b_ell)

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
#cov_ell[1,1,2:] = c_ell[2,:lmax-1] 
#cov_ell[2,2,2:] = c_ell[2,:lmax-1] * 0.5

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
sht.alm2map(alm, noise, ainfo, minfo, [0,2], adjoint=False)
if mask:
    noise *= mask_I
sht.map2alm(noise, alm, minfo, ainfo, [0,2], adjoint=False)

# Solve for Wiener filtered alms.
#for aidx, alpha in enumerate(np.logspace(5, 0, num=10)):
alpha = 1
aidx = 1

#icov_signal = operators.callable_matvec_pow_ell_alm(
#    ainfo, icov_ell, 1, inplace=False)
#sqrt_cov_signal = operators.callable_matvec_pow_ell_alm(
#    ainfo, icov_ell, -0.5, inplace=False)
#icov_noise = operators.callable_matvec_pow_pix_alm(
#    ainfo, icov_pix / alpha, minfo, [0, 2], 1, inplace=False)
#beam = operators.callable_matvec_pow_ell_alm(
#    ainfo, b_ell, 1, inplace=False)

#if bidx == 0:
#   x0 = None
#else:
#   x0 = solver.x

#x0 = alm.copy()
#denom = cov_ell + nl
#denom[0,0,:2] = 1
#denom[1,1,:2] = 1
#denom[2,2,:2] = 1
#x0 = operators.matvec_pow_ell_alm(x0, ainfo, cov_ell, 1, inplace=True)
#x0 = operators.matvec_pow_ell_alm(x0, ainfo, denom, -1, inplace=True)

#def draw_inoise(icov_pix, minfo, ainfo):
#    noise = map_utils.rand_map_pix(icov_pix / alpha)
#    alm_noise = np.zeros((noise.shape[0], ainfo.nelem), dtype=np.complex128)
#    sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=False)
#    return alm_noise

#constrained = False
#np.random.seed(100)
#if constrained:
#    rand_isignal = curvedsky.rand_alm(icov_ell, return_ainfo=False)
#    rand_inoise = draw_inoise(icov_pix, minfo, ainfo)
#else:
#    rand_isignal = None
#    rand_inoise = None

#tau = np.asarray([np.mean(cov_pix[0,0]), np.mean(cov_pix[1,1]), np.mean(cov_pix[2,2])]) * np.eye(3) * beta 
#itau = np.linalg.inv(tau)
#itau = itau[:,:,np.newaxis]

#M = operators.callable_matvec_pow_ell_alm(
#    ainfo, icov_ell + itau, -1, inplace=False)

#solver = solvers.CGWiener(
#    alm, icov_signal, icov_noise, beam=beam, M=M,
#    rand_isignal=rand_isignal, rand_inoise=rand_inoise, x0=x0)
#solver = solvers.CGWienerScaled(sqrt_cov_signal,
#                                 alm, icov_signal, icov_noise, beam=beam,
#                                 rand_isignal=rand_isignal, rand_inoise=rand_inoise, x0=x0)

#idx_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
#           30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,
#           500,550,600,650,700,749]

#tau = np.asarray([71**-2, 45**-2, 45**-2])
#tau = None

#itau = map_utils.get_isotropic_ivar(icov_pix, minfo)
#M = preconditioners.pinv_preconditioner(
#            icov_ell, ainfo, itau, icov_pix, minfo, b_ell=b_ell)


niter = 150
#stypes = ['cg', 'cg_scaled', 'pcg_harm', 'pcg_pinv']
stypes = ['pcg_pinv']
errors = np.zeros((len(stypes), niter + 1))
chisqs = np.zeros_like(errors)
residuals = np.zeros_like(errors)

for sidx, stype in enumerate(stypes):
    
    if stype == 'cg' or stype == 'cg_scaled':
        prec = None
    elif stype == 'pcg_harm':
        prec = 'harmonic'
    elif stype == 'pcg_pinv':
        prec = 'pinv'
    print('sidx', sidx)

    x0 = None
    
    #for aidx, alpha in enumerate([1e5, 1e4, 1e3, 1e2, 1e1, 1e1, 1e0, 1e0]):
    for aidx, alpha in enumerate([1]):

        if aidx > 0:
            x0 = solver.x

        if stype == 'cg_scaled':
            solver = solvers.CGWienerScaled.from_arrays(alm, ainfo, icov_ell, icov_pix, minfo, b_ell=b_ell,
                                                        draw_constr=False, prec=None, x0=x0)
        else:
            #solver = solvers.CGWiener.from_arrays(alm, ainfo, icov_ell, icov_pix, minfo, b_ell=b_ell,
            #                                      draw_constr=True, prec=prec, x0=x0)
            solver = solvers.CGWiener.from_arrays(alm, ainfo, icov_ell, icov_pix / alpha, minfo, b_ell=b_ell,
                                                  draw_constr=False, prec=prec, x0=x0)

        errors[sidx,0] = np.nan
        chisqs[sidx,0] = solver.get_chisq()
        residuals[sidx,0] = solver.get_residual()

        b_copy = solver.b.copy()
        
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
            print(solver.i, error, chisq / alm.size / 2, residual)

            # Plot result
            # if idx in idx_plot:
            #     omap = curvedsky.alm2map(solver.get_wiener(), omap)
            #     plot = enplot.plot(omap, colorbar=True, grid=20, font_size=50, range='250:5')
            #     enplot.write(opj(imgdir, 'alm_out_{}'.format(idx)), plot)



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
omap = curvedsky.alm2map(alm, omap)
#omap *= mask_I.reshape(minfo.nrow, minfo.nphi[0])
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5')
enplot.write(opj(imgdir, 'alm_in'), plot)

# Plot result
omap = curvedsky.alm2map(solver.get_wiener(), omap)
plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5')
enplot.write(opj(imgdir, 'alm_out'), plot)

omap_b_out = curvedsky.alm2map(solver.A(solver.x), omap.copy())
omap_b_in = curvedsky.alm2map(b_copy, omap.copy())

for pidx in range(alm.shape[0]):

    plot = enplot.plot(omap_b_out[pidx], colorbar=True, grid=False, font_size=50)
    enplot.write(opj(imgdir, 'b_out_{}'.format(pidx)), plot)

    plot = enplot.plot(omap_b_in[pidx], colorbar=True, grid=False, font_size=50)
    enplot.write(opj(imgdir, 'b_{}'.format(pidx)), plot)
