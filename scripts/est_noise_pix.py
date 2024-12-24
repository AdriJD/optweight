'''
Draw noise from cov_pix, try to reconstruct.
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import time

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils, enmap, sharp
from enlib import cg

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils
from optweight import noise_utils, alm_utils, mat_utils, alm_c_utils

opj = os.path.join
np.random.seed(39)

lmax = 500

basedir = '/home/adriaand/project/actpol/20210321_est_noise_pix'
imgdir = opj(basedir, 'img')
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')
areadir = '/home/adriaand/project/actpol/mapdata/area/'

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

icov = enmap.read_fits(opj(metadir, 'act_s08_s18_cmb_f150_night_ivar.fits'))
icov = icov.astype(np.float64)
#icov = curvedsky.make_projectable_map_by_pos(
#    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(3,))

icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest', order=1)
icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=0.3)


#icov_pix[:] = 1
# nOTE
#icov_pix[:] = np.mean(icov_pix)

mask = icov_pix != 0
mask_pix = mask.astype(np.float64)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    #im = ax.imshow(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='bicubic')
    #im = ax.matshow(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))
    colorbar(im)
    fig.savefig(opj(imgdir, 'icov_real_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(np.log10(np.abs(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))),
                       interpolation='none')
    #im = ax.imshow(np.log10(np.abs(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))),
    #                   interpolation='bicubic')
    #im = ax.matshow(np.log10(np.abs(icov_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]))))

    colorbar(im)
    fig.savefig(opj(imgdir, 'icov_real_log_{}'.format(pidx)))
    plt.close(fig)
# Get cov_pix
cov_pix = mat_utils.matpow(icov_pix, -1)

cov_pix[0,1] = 0
cov_pix[0,2] = 0
cov_pix[1,2] = 0
cov_pix[1,0] = 0
cov_pix[2,0] = 0
cov_pix[2,1] = 0

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(cov_pix[pidx,pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    colorbar(im)
    fig.savefig(opj(imgdir, 'cov_real_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(np.log10(np.abs(cov_pix[pidx,pidx].reshape(minfo.nrow, minfo.nphi[0]))),
                   interpolation='none', vmin=0, vmax=2)
    colorbar(im)
    fig.savefig(opj(imgdir, 'cov_real_log_{}'.format(pidx)))
    plt.close(fig)

# Draw noise
noise_pix = map_utils.rand_map_pix(cov_pix)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(noise_pix[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
    colorbar(im)
    fig.savefig(opj(imgdir, 'noise_pix_{}'.format(pidx)))
    plt.close(fig)

ainfo = sharp.alm_info(lmax)
alm_noise = np.zeros((noise_pix.shape[0], ainfo.nelem), dtype=np.complex128)
noise_pix_bndl = np.zeros_like(noise_pix)

#sht.map2alm(noise_pix, alm_noise, minfo, ainfo, [0,2], adjoint=False)
#sht.alm2map(alm_noise, noise_pix_bndl, ainfo, minfo, [0,2], adjoint=False)

#for pidx in range(3):
#    fig, ax = plt.subplots(dpi=300)
#    im = ax.imshow(noise_pix_bndl[pidx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
#    colorbar(im)
#    fig.savefig(opj(imgdir, 'noise_pix_bndl_{}'.format(pidx)))
#    plt.close(fig)

# NOTE
#noise_pix_bndl[0] /= np.std(noise_pix_bndl[0])
#noise_pix_bndl[1] /= np.std(noise_pix_bndl[1])
#noise_pix_bndl[2] /= np.std(noise_pix_bndl[2])

#print(np.var(noise_pix_bndl[0]))
#print(np.var(noise_pix_bndl[1]))
#print(np.var(noise_pix_bndl[2]))
noise_pix_bndl = noise_pix

# Square map.
cov_pix_est = np.einsum('il, kl -> ikl', noise_pix_bndl, noise_pix_bndl, optimize=True)

#cov_pix_est = np.zeros((3, 3, minfo.npix))
#cov_pix_est[0,0] = noise_pix_bndl[0] ** 2
#cov_pix_est[1,1] = noise_pix_bndl[1] ** 2
#cov_pix_est[2,2] = noise_pix_bndl[2] ** 2

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(cov_pix_est[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(cov_pix_est[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_log_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)

# Determine mean variance inside mask
#mean_var = np.mean(cov_pix_est[0,0,mask[0]])
#cov_pix_est[:,:,~mask[0]] = mean_var * 100

#weight = minfo.weight[np.nonzero(minfo.theta > (np.pi / 2))[0][0]]
#weight_ratio = minfo.weight / weight

#cov_pix_est_sm = cov_pix_est_sm.reshape((3,3,minfo.nrow,minfo.nphi[0]))
#cov_pix_est_sm /= weight_ratio[np.newaxis,np.newaxis,:,np.newaxis]
#cov_pix_est_sm = cov_pix_est_sm.reshape((3,3,minfo.nrow*minfo.nphi[0]))

#cov_pix_est = cov_pix_est.reshape((3,3,minfo.nrow,minfo.nphi[0]))
#cov_pix_est /= weight_ratio[np.newaxis,np.newaxis,:,np.newaxis]
#cov_pix_est = cov_pix_est.reshape((3,3,minfo.nrow*minfo.nphi[0]))

#cov_pix_est[0,0] = map_utils.inpaint_nearest(cov_pix_est[0,0], mask[0], minfo)
#cov_pix_est[1,1] = map_utils.inpaint_nearest(cov_pix_est[1,1], mask[1], minfo)
#cov_pix_est[2,2] = map_utils.inpaint_nearest(cov_pix_est[2,2], mask[2], minfo)

mask_nxn = np.einsum('il, kl -> ikl', mask, mask, optimize=True)
cov_pix_est = map_utils.inpaint_nearest(cov_pix_est, mask_nxn, minfo)

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(cov_pix_est[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_mask_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(cov_pix_est[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_mask_log_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)

# Smooth cov_est
cov_pix_est_sm = np.zeros_like(cov_pix_est)
b_ell = hp.gauss_beam(np.radians(3), lmax=lmax)

# NOTE NTOE
#b_ell = np.ones_like(b_ell)

for pidx in range(3):
    for pjdx in range(3):
        
        # Only use spin 0 transforms..
        alm = alm_noise[0].copy()
        sht.map2alm(cov_pix_est[pidx,pjdx], alm, minfo, ainfo, 0, adjoint=False)
        alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
        sht.alm2map(alm, cov_pix_est_sm[pidx,pjdx], ainfo, minfo, 0, adjoint=False)

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(cov_pix_est_sm[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_sm_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(cov_pix_est_sm[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]))), 
                       interpolation='none')
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_sm_log_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)


# Remove inpainted stuff.

#cov_pix_est_sm[0,0,~mask[0]] = 0
#cov_pix_est_sm[1,1,~mask[1]] = 0
#cov_pix_est_sm[2,2,~mask[2]] = 0
cov_pix_est_sm[~mask_nxn] = 0

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(cov_pix_est_sm[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_sm_nomask_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)

for pidx in range(3):
    for pjdx in range(3):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(np.log10(np.abs(cov_pix_est_sm[pidx,pjdx].reshape(minfo.nrow, minfo.nphi[0]))), 
                       interpolation='none', vmin=0, vmax=2)
        colorbar(im)
        fig.savefig(opj(imgdir, 'cov_est_sm_nomask_log_{}_{}'.format(pidx, pjdx)))
        plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow((cov_pix[pidx,pidx] / cov_pix_est_sm[pidx,pidx]).reshape(minfo.nrow, minfo.nphi[0]),
                   interpolation='none', vmin=0.5, vmax=1.5)
    colorbar(im)
    fig.savefig(opj(imgdir, 'cov_ratio_{}'.format(pidx)))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow((cov_pix[pidx,pidx] / cov_pix_est_sm[pidx,pidx]).reshape(minfo.nrow, minfo.nphi[0]),
                   interpolation='none', vmin=0.9, vmax=1.1)
    colorbar(im)
    fig.savefig(opj(imgdir, 'cov_ratio_zoom_{}'.format(pidx)))
    plt.close(fig)

print(np.mean(cov_pix[0,0,mask[0]] / cov_pix_est_sm[0,0,mask[0]]))
print(np.mean(cov_pix[1,1,mask[1]] / cov_pix_est_sm[1,1,mask[1]]))
print(np.mean(cov_pix[2,2,mask[2]] / cov_pix_est_sm[2,2,mask[2]]))
