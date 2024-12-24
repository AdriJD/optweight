import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import healpy as hp
from pixell import curvedsky, sharp

import sht
import map_utils
import alm_utils

opj = os.path.join
np.random.seed(39)

basedir = '/home/adriaand/project/actpol/20201009_pcg_planck'
imgdir = opj(basedir, 'n_ell')

lmax = 100

minfo, theta_arc_len = map_utils.get_gauss_minfo(2 * lmax, return_arc_len=True)
ainfo = sharp.alm_info(lmax=lmax)

#cov_pix = np.ones((1, 1, minfo.npix)) # Will produce different noise at pole.
cov_pix = np.zeros((3, 3, minfo.npix)) 
alm_mono = np.zeros((1, ainfo.nelem), dtype=np.complex128)
alm_mono[0,0] = 1
sht.alm2map(alm_mono, cov_pix[0,0], ainfo, minfo, 0)
cov_pix[1,1] = cov_pix[0,0]
cov_pix[2,2] = cov_pix[0,0]

thetas = minfo.theta
nphi = minfo.nphi[0]
dphi = 2 * np.pi / nphi

for tidx, theta in enumerate(thetas):
    start = tidx * nphi
    end = start + nphi
    area_gauss = np.sin(theta) * dphi * theta_arc_len[tidx]
    cov_pix[...,start:end] /= area_gauss

icov_pix = cov_pix.transpose(2, 0, 1)
icov_pix = np.linalg.inv(icov_pix)
icov_pix = np.ascontiguousarray(icov_pix.transpose(1, 2, 0))

nl = np.zeros((3, 3, lmax+1))
nmc = 500
for i in range(nmc):
    alm = alm_utils.rand_alm_pix(cov_pix, ainfo, minfo, [0, 2])

    nl += ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
nl /= nmc

alpha = np.sqrt(np.sum(map_utils.inv_qweight_map(icov_pix[0,0], minfo) ** 2) / np.sum(map_utils.inv_qweight_map(icov_pix[0,0], minfo)))

print(nl)
print(np.mean(nl[0,0]))
print(alpha ** -2)

alpha_ee = np.sqrt(np.sum(map_utils.inv_qweight_map(icov_pix[1,1], minfo) ** 2) / np.sum(map_utils.inv_qweight_map(icov_pix[1,1], minfo)))
alpha_bb = np.sqrt(np.sum(map_utils.inv_qweight_map(icov_pix[2,2], minfo) ** 2) / np.sum(map_utils.inv_qweight_map(icov_pix[2,2], minfo)))

print(alpha_ee ** -2)
print(alpha_bb ** -2)
