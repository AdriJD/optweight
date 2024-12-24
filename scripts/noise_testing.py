import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os

from pixell import sharp
from optweight import sht, map_utils

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20201111_noisebox/img'

lmax = 500
ainfo = sharp.alm_info(lmax=lmax)
#minfo, theta_arc_len = map_utils.get_gauss_minfo(2 * lmax, return_arc_len=True)
minfo, theta_arc_len = map_utils.get_gauss_minfo(3 * lmax, return_arc_len=True)

# Start with n_ell
ells = np.arange(lmax + 1)
n_ell = np.ones(lmax + 1) * 10
#n_ell = np.linspace(200, 210, num=lmax + 1)

# Convert to cov_pix using legendre formula
cov_pix = np.ones((1, minfo.npix)) * np.sum(n_ell * (2 * ells + 1)) / 4 / np.pi
print('cov_pix', cov_pix)

# Weight by pixel size
nphi = minfo.nphi[0]
dphi = 2 * np.pi / nphi
theta_ring = np.zeros(nphi)

print('theta_arc_len', theta_arc_len, np.sum(theta_arc_len))
print(minfo.npix)

tot_area = 0
for tidx, theta in enumerate(minfo.theta):
    theta_ring[:] = theta
    start = tidx * nphi
    end = start + nphi

    area_gauss = np.sin(theta) * dphi * theta_arc_len[tidx]
    #cov_pix[:,start:end] *= area_gauss * minfo.npix
    #cov_pix[:,start:end] /= (area_gauss * minfo.npix / 8 / np.pi) # this works a bit
    #cov_pix[:,start:end] /= (area_gauss * minfo.npix)
    #cov_pix[:,start:end] /= (area_gauss)
    #cov_pix[:,start:end] /= minfo.weight[tidx]
#    cov_pix[:,start:end] /= (minfo.weight[tidx] * minfo.npix / 8 / np.pi)
    cov_pix[:,start:end] /= (minfo.weight[tidx] * (lmax + 1) ** 2 / 4 / np.pi)

    #tot_area += area_gauss * nphi
    #tot_area += minfo.weight[tidx] * nphi

    print(area_gauss, minfo.weight[tidx], minfo.npix, nphi)

#cov_pix *= tot_area / minfo.npix * 2
#cov_pix *= 4 * np.pi / minfo.npix * 2
#cov_pix *= np.sum(minfo.weight) * nphi / minfo.npix * 2
#cov_pix *= np.sum(minfo.weight) 

print(tot_area)
print('cov_pix', cov_pix)

# draw noise maps from cov
n_mc = 100
c_ell = np.zeros_like(n_ell)
for ridx in range(n_mc):
    
    m_rand = map_utils.rand_map_pix(cov_pix)

    # turn into alms
    alm = np.zeros((1, ainfo.nelem), np.complex128)
    sht.map2alm(m_rand, alm, minfo, ainfo, 0)

    # take power spectra
    #c_ell += ainfo.alm2cl(alm)[0]
    c_ell += hp.alm2cl(alm)[0]

c_ell /= n_mc

# compare to n_ell
print(n_ell)
print(c_ell)

fig, ax = plt.subplots(dpi=300)
ax.plot(ells, n_ell, label='in')
ax.plot(ells, c_ell, label='out')
fig.savefig(opj(imgdir, 'noise_test'))
plt.close(fig)

# Now we do the same for icov.

inv_n_ell = 1 / n_ell

# Convert to cov_pix using legendre formula
icov_pix = np.ones((1, minfo.npix)) * np.sum(inv_n_ell * (2 * ells + 1)) / 4 / np.pi
print('icov_pix', icov_pix)

# Weight by pixel size
nphi = minfo.nphi[0]
dphi = 2 * np.pi / nphi

for tidx, theta in enumerate(minfo.theta):

    start = tidx * nphi
    end = start + nphi
    icov_pix[:,start:end] *= minfo.weight[tidx] / ((lmax + 1) ** 2 / 4 / np.pi)

# draw noise maps from cov
n_mc = 100
c_ell = np.zeros_like(n_ell)
for ridx in range(n_mc):
    
    m_rand = map_utils.rand_map_pix(1 / icov_pix)
    # turn into alms
    alm = np.zeros((1, ainfo.nelem), np.complex128)
    sht.map2alm(m_rand, alm, minfo, ainfo, 0)

    # take power spectra
    #c_ell += ainfo.alm2cl(alm)[0]
    c_ell += hp.alm2cl(alm)[0]

c_ell /= n_mc

fig, ax = plt.subplots(dpi=300)
ax.plot(ells, 1 / inv_n_ell, label='in')
ax.plot(ells, c_ell, label='out')
ax.legend()
fig.savefig(opj(imgdir, 'inverse_noise_test'))
plt.close(fig)
