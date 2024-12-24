import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import healpy as hp
from pixell import curvedsky, sharp, curvedsky, enplot, utils

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

imgdir = '/home/adriaand/project/actpol/20211125_reduced_gl'
utils.mkdir(imgdir)

lmax = 100

# minfo = map_utils.get_gauss_minfo(2 * lmax)

# print(np.sum(minfo.weight) * minfo.nphi[0])

# theta = minfo.theta
# weight = minfo.weight
# nphi = minfo.nphi
# idx = np.searchsorted(theta, np.pi / 2, side="left")
# area_pix_0 = weight[idx] #/ nphi[idx]

# nphi_reduced = np.zeros_like(nphi)
# weight_reduced = np.zeros_like(weight)

# for tidx in range(theta.size):
    
#     area_pix = weight[tidx] #/ nphi[tidx]    
#     #ratio = area_pix / area_pix_0
#     if tidx < 0.25 * theta.size or tidx > 0.75 * theta.size:
#         ratio = (area_pix / area_pix_0) ** 0.85
#         #ratio = (area_pix / area_pix_0) ** 1
#     else:
#         ratio = 1
#     #ratio = 1
#     #print(ratio)
#     nphi_reduced[tidx] = max(2, int(np.round(minfo.nphi[tidx] * ratio)))
#     #print(nphi_reduced[tidx])
#     weight_reduced[tidx] = area_pix * nphi[tidx] / nphi_reduced[tidx]
#     #print(area_pix * nphi_reduced[tidx])
#     #if not (0.99999999 < weight_reduced[tidx] / weight[tidx] < 1.00000001):
#     #    print(weight_reduced[tidx], weight[tidx])
# #    print(nphi_reduced[tidx], nphi[tidx])

#     #m = lmax // 2
#     m = 10
#     print(np.sqrt(m ** 2 - 2 * m * np.cos(theta[tidx])) - lmax * np.sin(theta[tidx]))

#     print(nphi_reduced[tidx], nphi[tidx])
#     #print(weight_reduced[tidx])

# #print(minfo.offsets)
# #print(minfo.phi0)
# minfo_reduced = sharp.map_info(minfo.theta, nphi=nphi_reduced, weight=weight_reduced)
# #minfo_reduced = sharp.map_info(theta=minfo.theta, nphi=minfo.nphi, weight=minfo.weight, phi0=minfo.phi0,
# #                               offsets=minfo.offsets, stride=minfo.stride)
# #minfo_reduced = map_utils.copy_minfo(minfo)
# print(minfo_reduced.npix)

# print(minfo_reduced.nphi)

minfo_reduced = map_utils.get_equal_area_gauss_minfo(2 * lmax, ratio_pow=0.85, gl_band=np.pi / 4)

ells = np.arange(lmax + 1)
npol = 3
icov_ell = np.ones((npol, npol, lmax+1))
icov_ell *= np.asarray([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])[:,:,np.newaxis]
icov_ell *= 1e-4 * (ells + 1) ** 2
icov_ell[:,:,:2] = 0

np.random.seed(10)
cov_ell = mat_utils.matpow(icov_ell, -1)
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
imap = np.ones((npol, minfo_reduced.npix))
omap = imap.copy() * 0

imap_enmap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], 4 *  lmax, dims=(3,))
curvedsky.alm2map(alm, imap_enmap, ainfo=ainfo)
for pidx in range(3):
    plot = enplot.plot(imap_enmap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, 'imap_{}'.format(pidx)), plot)


#print(alm)
sht.alm2map(alm, imap, ainfo, minfo_reduced, [0, 2])

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap[pidx], minfo_reduced),
                   vmin=-400, vmax=400)
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_reduced_{pidx}'))
    plt.close(fig)

# Also project alms to normal GL grid.
minfo_gl = map_utils.get_gauss_minfo(2 * lmax)
imap_gl = np.zeros((3, minfo_gl.npix))
sht.alm2map(alm, imap_gl, ainfo, minfo_gl, [0, 2])

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.view_2d(imap_gl[pidx], minfo_gl),
                   vmin=-400, vmax=400)
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_gl_{pidx}'))
    plt.close(fig)

imap_reduced_interp = map_utils.gauss2map(imap_gl, minfo_gl, minfo_reduced, order=1)
for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap_reduced_interp[pidx], minfo_reduced),
                   vmin=-400, vmax=400)
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_reduced_interp_{pidx}'))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(
        imap_reduced_interp[pidx] - imap[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_reduced_diff_{pidx}'))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap[pidx], minfo_reduced),
                   vmin=-400, vmax=400)
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_reduced_{pidx}'))
    plt.close(fig)



sht.map2alm(imap, alm, minfo_reduced, ainfo, [0, 2])
#sht.alm2map(alm, imap, ainfo, minfo, [0, 2])
#sht.map2alm(imap, alm, minfo, ainfo, [0, 2])

omap_enmap = imap_enmap.copy()
curvedsky.alm2map(alm, omap_enmap, ainfo=ainfo)
for pidx in range(3):
    plot = enplot.plot(omap_enmap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, 'omap_{}'.format(pidx)), plot)

for pidx in range(3):
    plot = enplot.plot(imap_enmap[pidx] - omap_enmap[pidx], colorbar=True, grid=False, quantile=0)
    enplot.write(opj(imgdir, 'diff_{}'.format(pidx)), plot)

#print(alm)
#sht.alm2map(alm, omap, ainfo, minfo_reduced, [0, 2])

#print(imap)
#print(omap)


