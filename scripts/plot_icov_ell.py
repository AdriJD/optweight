import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils
from enlib import cg

from optweight import sht, map_utils, mat_utils, solvers, operators, preconditioners

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
#lmax = 1000

basedir = '/home/adriaand/project/actpol/20230614_icov_ell'
maskdir = '/home/adriaand/project/actpol/20201009_pcg_planck/meta'
imgdir = opj(basedir, 'img')

utils.mkdir(imgdir)

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

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(cov_ell[idxs])
fig.savefig(opj(imgdir, 'cov_ell'))
plt.close(fig)

cov_ell[...,1:] /= dells[1:]

icov_ell = np.ones((3, 3, lmax + 1))
for lidx in range(icov_ell.shape[-1]):
    icov_ell[:,:,lidx] = np.linalg.pinv(cov_ell[:,:,lidx])

ellsq = ells ** 2
lnorm = 500

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    axs[idxs].plot(icov_ell[idxs])
fig.savefig(opj(imgdir, 'icov_ell'))
plt.close(fig)

b_ell = get_planck_b_ell(opj(maskdir, 'BeamWf_HFI_R3.01', 'Bl_TEB_R3.01_fullsky_100x100.fits'), lmax)

#isqrt_b_ell = mat_utils.matpow(b_ell, -0.5)
isqrt_b_ell = mat_utils.matpow(b_ell, -0.8)
icov_ell_scaled = np.einsum('abk, bck, cdk -> adk', isqrt_b_ell, icov_ell, isqrt_b_ell)

fig, axs = plt.subplots(nrows=3, dpi=300, constrained_layout=True)
for pidx in range(3):
    axs[pidx].plot(icov_ell[pidx,pidx])
    axs[pidx].plot(icov_ell_scaled[pidx,pidx])
    axs[pidx].plot(ellsq * (icov_ell[pidx,pidx][lnorm] / lnorm ** 2))
    axs[pidx].set_yscale('log')
fig.savefig(opj(imgdir, 'icov_ell_scaled'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.plot(b_ell[1] ** -0.5)
ax.plot(icov_ell[1,1] / ellsq)
ax.set_yscale('log')
fig.savefig(opj(imgdir, 'scalings'))
plt.close(fig)

radii = np.linspace(0, np.radians(10), 1000)

fig, axs = plt.subplots(nrows=3, dpi=300, constrained_layout=True)
for pidx in range(3):
    axs[pidx].plot(radii, hp.bl2beam(icov_ell[pidx,pidx], radii))
fig.savefig(opj(imgdir, 'icov_ell_r'))
plt.close(fig)
