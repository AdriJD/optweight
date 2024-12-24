import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from pixell import sharp, utils, curvedsky
from optweight import sht, operators, map_utils, alm_c_utils, mat_utils

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

imgdir = '/home/adriaand/project/actpol/20211207_masked_solver_pol'
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')

utils.mkdir(imgdir)

lmax = 10
ainfo = sharp.alm_info(lmax)
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

cov_ell = np.zeros((2, 2, lmax + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
                   skiprows=1, usecols=[2, 3]) #  TT, EE, BB, TE.

c_ell = c_ell.T
cov_ell[0,0,2:] = c_ell[0,2:lmax+1] 
cov_ell[1,1,2:] = c_ell[0,2:lmax+1] 
#cov_ell[1,1,2:] = c_ell[0,:lmax//2-1] 
cov_ell[...,1:] /= dells[None, None, 1:]

icov_ell = mat_utils.matpow(cov_ell, -1)

# Create mask
minfo = sharp.map_info_gauss_legendre(lmax + 1, 2 * lmax + 1)
mask = np.zeros((1, minfo.npix))
mask = map_utils.view_2d(mask, minfo)
#mask[0,3:7,2:17] = 1

mask[:] = 0
mask[0,:3,] = 1
mask[0,7:,] = 1
mask[0,np.random.randint(0, lmax, 20), np.random.randint(0, lmax, 20)] = 1
mask = map_utils.view_1d(mask, minfo)
mask = mask.astype(bool)

# NOTE
#mask[:] = False

# NOTE
#icov_ell[0,0] = 1
#icov_ell[1,1] = 1

# define operation
# def g_op(imap):

#     alm = np.zeros((1, ainfo.nelem), dtype=np.complex128)
#     imap *= ~mask[0]
#     sht.map2alm(imap, alm, minfo, ainfo, 0, adjoint=True)
#     #alm_c_utils.lmul(alm, icov_ell, ainfo, inplace=True)
#     alm = alm_c_utils.lmul(alm, icov_ell, ainfo, inplace=False)
#     sht.alm2map(alm, imap, ainfo, minfo, 0)
#     imap *= ~mask[0]
#     return imap

def g_op_alm(alm_real):
    
    alm = alm_real.view(np.complex128)
    omap = np.zeros((2, minfo.npix))
    sht.alm2map(alm, omap, ainfo, minfo, 2)
    omap *= ~mask
    sht.map2alm(omap, alm, minfo, ainfo, 2, adjoint=False)
    alm = alm_c_utils.lmul(alm, icov_ell, ainfo, inplace=True)
    sht.alm2map(alm, omap, ainfo, minfo, 2)
    omap *= ~mask
    sht.map2alm(omap, alm, minfo, ainfo, 2, adjoint=False)
    return alm.view(np.float64)

def m_op_alm(alm_real):
    
    alm = alm_real.view(np.complex128)
    omap = np.zeros((2, minfo.npix))
    sht.alm2map(alm, omap, ainfo, minfo, 2, adjoint=True)
    omap *= ~mask
    sht.map2alm(omap, alm, minfo, ainfo, 2)
    alm = alm_c_utils.lmul(alm, cov_ell, ainfo, inplace=False)
    sht.alm2map(alm, omap, ainfo, minfo, 2, adjoint=True)
    omap *= ~mask
    sht.map2alm(omap, alm, minfo, ainfo, 2)
    return alm.view(np.float64)
    
alm = curvedsky.rand_alm(cov_ell, ainfo=ainfo).astype(np.complex128)

assert np.allclose(alm.view(np.float64).view(np.complex128), alm)

omap = np.zeros((2, minfo.npix))
sht.alm2map(alm, omap, ainfo, minfo, 2)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.view_2d(omap[pidx], minfo))
    colorbar(im)
    fig.savefig(opj(imgdir, f'alm_in_{pidx}'))
    plt.close(fig)

alm = g_op_alm(alm.view(np.float64)).view(np.complex128)

sht.alm2map(alm, omap, ainfo, minfo, 2)

for pidx in range(2):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.view_2d(omap[pidx], minfo))
    colorbar(im)
    fig.savefig(opj(imgdir, f'alm_g_{pidx}'))
    plt.close(fig)


# op2mat
mat = operators.op2mat(g_op_alm, 2 * ainfo.nelem * 2, np.float64, input_shape=(2, ainfo.nelem * 2))

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(map_utils.view_2d(mask, minfo)[0])
colorbar(im)
fig.savefig(opj(imgdir, 'mask'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(mat)
colorbar(im)
fig.savefig(opj(imgdir, 'mat'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.log(np.abs(mat)))
colorbar(im)
fig.savefig(opj(imgdir, 'mat_log'))
plt.close(fig)

print(np.diag(mat))

pinvmat = np.linalg.pinv(mat)
fig, ax = plt.subplots(dpi=300)
im = ax.imshow(pinvmat)
colorbar(im)
fig.savefig(opj(imgdir, 'pinv_mat'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.log(np.abs(pinvmat)))
colorbar(im)
fig.savefig(opj(imgdir, 'pinv_mat_log'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.dot(pinvmat, mat))
colorbar(im)
fig.savefig(opj(imgdir, 'prod'))
plt.close(fig)

m_mat = operators.op2mat(m_op_alm, 2 * ainfo.nelem * 2, np.float64, input_shape=(2, ainfo.nelem * 2))

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(m_mat)
colorbar(im)
fig.savefig(opj(imgdir, 'm_mat'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.log(np.abs(m_mat)))
colorbar(im)
fig.savefig(opj(imgdir, 'm_mat_log'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.dot(m_mat, mat))
colorbar(im)
fig.savefig(opj(imgdir, 'm_prod'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.log(np.abs(np.dot(m_mat, mat))))
colorbar(im)
fig.savefig(opj(imgdir, 'm_prod_log'))
plt.close(fig)
