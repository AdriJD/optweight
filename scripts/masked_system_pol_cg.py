import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from pixell import sharp, utils, curvedsky
from optweight import sht, operators, map_utils, alm_c_utils, mat_utils, multigrid

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

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20211207_masked_solver_pol_cg'
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')

utils.mkdir(imgdir)

lmax = 2000
ainfo = sharp.alm_info(lmax)
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

cov_ell = np.zeros((2, 2, lmax + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
                   skiprows=1, usecols=[2, 3]) #  TT, EE, BB, TE.

c_ell = c_ell.T
cov_ell[0,0,2:] = c_ell[0,2:lmax+1] 
cov_ell[1,1,2:] = c_ell[1,2:lmax+1] 
#cov_ell[1,1,2:] = c_ell[0,2:lmax+1]
#cov_ell[1,1,2:] = c_ell[0,2:lmax+1] 
cov_ell[...,1:] /= dells[None, None, 1:]

r_ell = multigrid.lowpass_filter(lmax) ** 1.
#r_ell = np.ones(lmax+1)
print(r_ell[::50])
# NOTE
#cov_ell /= r_ell ** 2
#print(cov_ell)

icov_ell = mat_utils.matpow(cov_ell, -1)

# Create mask
minfo = sharp.map_info_gauss_legendre(lmax + 1, 2 * lmax + 1)
mask = np.zeros((1, minfo.npix))
mask = map_utils.view_2d(mask, minfo)
#mask[0,3:7,2:17] = 1

mask[:] = 1
mask[0,:500,] = 0
mask[0,900:,] = 0
mask[0,np.random.randint(0, lmax, 20), np.random.randint(0, lmax, 20)] = 1
mask = map_utils.view_1d(mask, minfo)
mask = mask.astype(bool)

minfo_reduced = map_utils.get_equal_area_gauss_minfo(
    2 * lmax, ratio_pow=1, gl_band=0)
mask_reduced = map_utils.gauss2map(
    mask, minfo, minfo_reduced, order=1)

def g_op(imap):

    alm = np.zeros((2, ainfo.nelem), dtype=np.complex128)
    imap = imap * ~mask[0]
    sht.map2alm(imap, alm, minfo, ainfo, 2, adjoint=True)
    alm = alm_c_utils.lmul(alm, icov_ell, ainfo, inplace=False)
    sht.alm2map(alm, imap, ainfo, minfo, 2)
    imap = imap * ~mask[0]
    return imap

def g_op_reduced(imap):

    alm = np.zeros((2, ainfo.nelem), dtype=np.complex128)
    imap = imap * ~mask_reduced[0]
    sht.map2alm(imap, alm, minfo_reduced, ainfo, 2, adjoint=True)
    #alm = alm_c_utils.lmul(alm, icov_ell, ainfo, inplace=False)
    alm = alm_c_utils.lmul(alm, icov_ell * r_ell ** 2, ainfo, inplace=False)
    sht.alm2map(alm, imap, ainfo, minfo_reduced, 2)
    imap = imap * ~mask_reduced[0]
    return imap

def g_op_reduced_pinv(imap):

    alm = np.zeros((2, ainfo.nelem), dtype=np.complex128)
    imap = imap * ~mask_reduced[0]
    sht.map2alm(imap, alm, minfo_reduced, ainfo, 2, adjoint=False)
    alm = alm_c_utils.lmul(alm, cov_ell, ainfo, inplace=False)
    #alm = alm_c_utils.lmul(alm, cov_ell / r_ell ** 2, ainfo, inplace=False)
    #alm = alm_c_utils.lmul(alm, cov_ell / r_ell, ainfo, inplace=False)
    sht.alm2map(alm, imap, ainfo, minfo_reduced, 2, adjoint=True)
    imap = imap * ~mask_reduced[0]
    return imap

alm = curvedsky.rand_alm(cov_ell, ainfo=ainfo).astype(np.complex128)
#imap = np.zeros((2, minfo.npix))
imap = np.zeros((2, minfo_reduced.npix))
sht.alm2map(alm, imap, ainfo, minfo_reduced, 2)

#omap = g_op(imap)
omap = g_op_reduced(imap)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    #im = ax.imshow(map_utils.view_2d(imap[pidx], minfo))
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_{pidx}'))
    plt.close(fig)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(omap[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'omap_{pidx}'))
    plt.close(fig)

#b_vec = imap * ~mask
b_vec = omap

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(b_vec[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'b_{pidx}'))
    plt.close(fig)

test_map = np.zeros_like(imap)
test_map[0,0] = 1
test_omap = g_op_reduced(test_map)
diag_g = test_omap[0,0]
print(diag_g)

test_map = np.zeros_like(imap)
test_map[1,0] = 1
test_omap = g_op_reduced(test_map)
diag_g = test_omap[1,0]
print(diag_g)

def dot(vec1, vec2):
    return np.sum(vec1 * vec2)
#def prec(imap):
#    return ~mask[0] * imap / diag_g 
#prec = lambda x : x
prec = g_op_reduced_pinv

cg = utils.CG(g_op_reduced, b_vec, dot=dot, M=prec)

for idx in range(10):
    cg.step()
    print(idx, cg.err)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(cg.x[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'x_{pidx}'))
    plt.close(fig)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(cg.x[pidx] - imap[pidx] * ~mask_reduced[0], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'diff_{pidx}'))
    plt.close(fig)


imap_pinv = g_op_reduced_pinv(omap)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap_pinv[pidx], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'imap_pinv_{pidx}'))
    plt.close(fig)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.equal_area_gauss_copy_2d(imap_pinv[pidx] - imap[pidx] * ~mask_reduced[0], minfo_reduced))
    colorbar(im)
    fig.savefig(opj(imgdir, f'diff_pinv_{pidx}'))
    plt.close(fig)
