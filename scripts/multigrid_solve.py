import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import healpy as hp
from pixell import curvedsky, sharp, curvedsky, enplot, enmap

from optweight import multigrid
from optweight import map_utils, sht, alm_c_utils, alm_utils, mat_utils

opj = os.path.join

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')
maskdir = '/home/adriaand/analysis/actpol/20210504_noisesims_wav'
imgdir = '/home/adriaand/project/actpol/20211124_multigrid'

lmax = 1250
#lmax = 2250
#lmax = 20
#lmax = 200
ells = np.arange(lmax + 1)
dells = ells * (ells + 1) / 2 / np.pi
npol = 3
#icov_ell = np.ones((npol, npol, lmax+1))
#icov_ell *= np.asarray([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])[:,:,np.newaxis]
#icov_ell *= np.asarray([[1, 0., 0.], [0., 1, 0.], [0., 0, 1]])[:,:,np.newaxis]
# Approximate icov_ell.

#icov_ell *= np.asarray([[1, 0, 0], [0, 10, 0], [0, 0, 100]])[:,:,np.newaxis]

#icov_ell *= 1e-4 * (ells + 1) ** 2
#icov_ell[0,0,:10] /= 3
#icov_ell *= 1e-4 * (ells + 1) ** 1
#icov_ell *= hp.gauss_beam(2 * np.pi / lmax, lmax)
#icov_ell[:,:,:2] = 0
#icov_ell[:,:,:1] = 0


cov_ell = np.zeros((3, 3, lmax + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
    skiprows=1, usecols=[1, 2, 3, 4]) #  TT, EE, BB, TE.
c_ell = c_ell.T

cov_ell[0,0,2:] = c_ell[0,:lmax-1] 
#cov_ell[0,1,2:] = c_ell[3,:lmax-1] 
#cov_ell[1,0,2:] = c_ell[3,:lmax-1] 
#cov_ell[1,1,2:] = c_ell[1,:lmax-1] 
#cov_ell[2,2,2:] = c_ell[2,:lmax-1] 
#cov_ell[1,1,2:] = c_ell[1,:lmax-1] 
#cov_ell[2,2,2:] = c_ell[1,:lmax-1] 
cov_ell[1,1,2:] = c_ell[0,:lmax-1] #* (ells[2:] ** -1.) * 1e5
cov_ell[2,2,2:] = c_ell[0,:lmax-1] #* (ells[2:] ** -1.) * 1e5

cov_ell[...,1:] /= dells[1:]

icov_ell = mat_utils.matpow(cov_ell, -1)
#icov_ell = np.zeros_like(cov_ell)
#for lidx in range(2, icov_ell.shape[-1]):
#    icov_ell[:,:,lidx] = np.linalg.inv(cov_ell[:,:,lidx])


fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(icov_ell[idxs])

#icov_ell[1,1,:300] = icov_ell[1,1,300] * (ells[:300] / 299) ** 2
#icov_ell[1,1,:80] = icov_ell[1,1,80] * (ells[:80] / 79) ** 2
#icov_ell[2,2,:] = icov_ell[1,1,:]

for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(icov_ell[idxs])
    if idxs[0] == idxs[1]:
        ax.set_yscale('log')
        #ax.set_ylim(1e-4, 1e8)
        ax.plot(ells, dells * 1e-2)
    ax.set_xscale('log')

fig.savefig(opj(imgdir, 'icov_ell'))
plt.close(fig)

#exit()
#map_utils.get_gauss_minfo(2 * lmax, theta_min=1, theta_max=2)
#map_utils.get_gauss_minfo(2 * lmax)

minfo = map_utils.get_gauss_minfo(2 * lmax)

# Mask north and south cap.
mask = np.ones((minfo.npix), dtype=bool)
mask_2d = map_utils.view_2d(mask, minfo)
print(mask_2d.shape)

np.random.seed(10)

#mask[np.random.randint(0, minfo.npix, 50000)] = False
#mask_2d[0:5,:] = False
#mask_2d[50:,:] = False
#mask_2d[20:35,:] = False
#mask_2d[:40,:] = False
#mask_2d[80:110,:] = False
#mask_2d[500:610,:] = False

#mask_2d[500:710,:] = False

# Works
#mask_2d[:] = False
#mask_2d[300:1000,:] = True
#mask_2d[500:1700,:] = True

mask_enmap = enmap.read_map(opj(maskdir, 'mask_dg2.fits'))
mask, _ = map_utils.enmap2gauss(mask_enmap, minfo, order=1, area_pow=0, destroy_input=False,
                   mode='nearest')
mask[mask<0.9] = 0
mask = mask.astype(bool)

#mask_2d[1000:1210,:] = False
#mask_2d[500:1500,:] = False

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(map_utils.view_2d(mask, minfo))
colorbar(im)
fig.savefig(opj(imgdir, 'mask_gl'))
plt.close(fig)

#min_pix = 1000
min_pix = 1500
#min_pix = 20
levels = multigrid.get_levels(mask, minfo, icov_ell, min_pix=min_pix)

for lidx, level in enumerate(levels):
    
    for pidx in range(3):

        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(map_utils.equal_area_gauss_copy_2d(
            level.mask_unobs[pidx], level.minfo))
        colorbar(im)
        fig.savefig(opj(imgdir, f'mask_gl_level{lidx}_{pidx}'))
        plt.close(fig)
    
minfo_reduced = levels[0].minfo
mask_reduced = ~levels[0].mask_unobs[0]

# Create map.
cov_ell = mat_utils.matpow(icov_ell, -1)
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
#imap = np.ones((npol, minfo.npix))
imap = np.ones((npol, minfo_reduced.npix))
#sht.alm2map(alm, imap, ainfo, minfo, [0, 2])
sht.alm2map(alm, imap, ainfo, minfo_reduced, [0, 2])

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    #im = ax.imshow(map_utils.view_2d(imap[pidx], minfo), vmin=-400, vmax=400)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap[pidx], minfo_reduced),
                   vmin=-400, vmax=400)
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_gl_{pidx}'))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    #im = ax.imshow(map_utils.view_2d(mask * imap[pidx], minfo), vmin=-400, vmax=400)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(mask_reduced * imap[pidx], minfo_reduced),
                   vmin=-400, vmax=400)
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_masked_{pidx}'))
    plt.close(fig)

imap_g = levels[0].g_op(imap)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    #im = ax.imshow(map_utils.view_2d(imap_g[pidx], minfo))
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap_g[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_g_{pidx}'))
    plt.close(fig)

#imap = levels[0].g_op_full(imap)

#print(imap[0][~mask])
#omap = multigrid.v_cycle(levels, imap * ~mask)
#omap = multigrid.v_cycle(levels, imap * mask)
#omap = multigrid.v_cycle(levels, imap * mask_shrunk + np.ones_like(imap) * ~mask_shrunk)
#omap = multigrid.v_cycle(levels, imap_g, n_jacobi=3)
omap = multigrid.v_cycle(levels, imap_g, n_jacobi=5)

# NOTE NOTE NOTE
#omap *= np.prod(mask.shape)

#print(omap[0][~mask])

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    #im = ax.imshow(map_utils.view_2d(omap[pidx], minfo))
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(omap[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'omap_unmasked_{pidx}'))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    #im = ax.imshow(map_utils.view_2d(omap[pidx] * ~mask + imap[pidx] * mask, minfo))
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(
        omap[pidx] * ~mask_reduced + imap[pidx] * mask_reduced, minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'omap_gl_{pidx}'))
    plt.close(fig)

#omap_enmap = curvedsky.make_projectable_map_by_pos(
#    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], 4 * lmax, dims=(3,))

omap_enmap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(3,))
imap_enmap = omap_enmap.copy()

alm_plot = alm.copy()
#sht.map2alm(imap, alm_plot, minfo, ainfo, [0,2])
sht.map2alm(imap, alm_plot, minfo_reduced, ainfo, [0,2])
curvedsky.alm2map(alm_plot, imap_enmap, ainfo=ainfo)

for pidx in range(3):
    #plot = enplot.plot(imap_enmap[pidx], colorbar=True, grid=False, min=-400, max=400, font_size=100)
    plot = enplot.plot(imap_enmap[pidx], colorbar=True, grid=False, quantile=0)
    enplot.write(opj(imgdir, 'imap_{}'.format(pidx)), plot)

#sht.map2alm(omap * ~mask + imap * mask, alm_plot, minfo, ainfo, [0,2])
#sht.map2alm(omap * ~mask, alm_plot, minfo, ainfo, [0,2])
sht.map2alm(omap * ~mask_reduced, alm_plot, minfo_reduced, ainfo, [0,2])
curvedsky.alm2map(alm_plot, omap_enmap, ainfo=ainfo)

for pidx in range(3):
    #plot = enplot.plot(omap_enmap[pidx], colorbar=True, grid=False, min=-400, max=400, font_size=100)
    plot = enplot.plot(omap_enmap[pidx], colorbar=True, grid=False, quantile=0)
    enplot.write(opj(imgdir, 'omap_{}'.format(pidx)), plot)

sht.map2alm(omap * ~mask_reduced + imap * mask_reduced, alm_plot, minfo_reduced, ainfo, [0,2])
curvedsky.alm2map(alm_plot, omap_enmap, ainfo=ainfo)

for pidx in range(3):
    plot = enplot.plot(imap_enmap[pidx] - omap_enmap[pidx], colorbar=True, grid=False, quantile=0, font_size=100)
    #plot = enplot.plot(omap_enmap[pidx], colorbar=True, grid=False, quantile=0)
    enplot.write(opj(imgdir, 'diff_{}'.format(pidx)), plot)
    

