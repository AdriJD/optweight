import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from pixell import sharp
from optweight import sht, operators, map_utils, alm_c_utils

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

imgdir = '/home/adriaand/project/actpol/20211108_masked_solver'
metadir = '/home/adriaand/project/actpol/20201029_noisebox'
specdir = opj(metadir, 'spectra')

lmax = 7
ainfo = sharp.alm_info(lmax // 2)
ells = np.arange(lmax // 2 + 1)
dells = ells * (ells + 1)  / 2 / np.pi

cov_ell = np.zeros((1, 1, lmax // 2 + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
    skiprows=1, usecols=[1]) #  TT, EE, BB, TE.

cov_ell[0,0,2:] = c_ell[:lmax//2-1] 
cov_ell[0,0,1:] /= dells[1:]
icov_ell = cov_ell.copy()
icov_ell[:,:,2:] = 1 / cov_ell[:,:,2:]

# Create mask
minfo = sharp.map_info_gauss_legendre(lmax + 1, 2 * lmax + 1)
mask = np.zeros((1, minfo.npix))
mask = map_utils.view_2d(mask, minfo)
mask[0,3:7,2:17] = 1
mask[0,np.random.randint(0, lmax, 20), np.random.randint(0, lmax, 20)] = 1
mask = map_utils.view_1d(mask, minfo)
mask = mask.astype(bool)

# define operation
def g_op(imap):

    alm = np.zeros((1, ainfo.nelem), dtype=np.complex128)
    imap *= ~mask[0]
    sht.map2alm(imap, alm, minfo, ainfo, 0, adjoint=True)
    #alm_c_utils.lmul(alm, icov_ell, ainfo, inplace=True)
    alm = alm_c_utils.lmul(alm, icov_ell, ainfo, inplace=False)
    sht.alm2map(alm, imap, ainfo, minfo, 0)
    imap *= ~mask[0]
    return imap

# op2mat
mat = operators.op2mat(g_op, minfo.npix, np.float64)

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

print(np.diag(mat))

pinvmat = np.linalg.pinv(mat)
fig, ax = plt.subplots(dpi=300)
im = ax.imshow(pinvmat)
colorbar(im)
fig.savefig(opj(imgdir, 'pinv_mat'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
#for idx in range(minfo.npix):
#for idx in range(100,101):
ax.plot(np.diag(mat), color='C0', alpha=0.5)
fig.savefig(opj(imgdir, 'profiles'))
plt.close(fig)

from optweight import operators, mat_utils

ainfo = sharp.alm_info(lmax)
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi
spin = [0, 2]

cov_ell = np.zeros((3, 3, lmax + 1))
c_ell = np.loadtxt(opj(specdir, 'planck_2018_lensedCls.dat'),
                   skiprows=1, usecols=[1, 2, 3, 4]) #  TT, EE, BB, TE.
c_ell = c_ell.T

cov_ell[0,0,2:] = c_ell[0,:lmax-1]
cov_ell[1,1,2:] = c_ell[1,:lmax-1]
cov_ell[2,2,2:] = c_ell[2,:lmax-1]
#cov_ell[1,1,2:] = c_ell[0,:lmax-1]
#cov_ell[2,2,2:] = c_ell[0,:lmax-1]
cov_ell[0,1,2:] = c_ell[3,:lmax-1]
cov_ell[1,0,2:] = c_ell[3,:lmax-1]
cov_ell[:,:,1:] /= dells[1:]

icov_ell = mat_utils.matpow(cov_ell, -1)

mask_bool = np.ones((3, minfo.npix), dtype=bool)
#mask_bool *= mask
op = operators.PixEllPixMatVecMap(mask_bool, icov_ell, minfo, spin)
#op_flat = lambda x : op(x.reshape(3, minfo.npix)).reshape(-1)

print(minfo.npix, icov_ell.shape)
mat = operators.op2mat(op, 3 * minfo.npix, np.float64, input_shape=(3, minfo.npix))

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(mat)
colorbar(im)
fig.savefig(opj(imgdir, 'mat_pol'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.log10(np.abs(mat)), vmin=-4, vmax=4)
colorbar(im)
fig.savefig(opj(imgdir, 'mat_pol_log'))
plt.close(fig)

print(np.diag(mat)[::1])

pinvmat = np.linalg.pinv(mat)
print(pinvmat)
fig, ax = plt.subplots(dpi=300)
im = ax.imshow(pinvmat)
colorbar(im)
fig.savefig(opj(imgdir, 'pinv_mat_pol'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.log10(np.abs(pinvmat)))
colorbar(im)
fig.savefig(opj(imgdir, 'pinv_mat_pol_log'))
plt.close(fig)
