import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from pixell import curvedsky, sharp
import healpy as hp
from pys2let import analysis_axisym_lm_wav, synthesis_axisym_lm_wav, axisym_wav_l

import sht
import map_utils

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20201019_wavelets/img'

lmax = 100
B = 2
J_min = 1
spin = 0 

# Create white noise alms
cov_ell = np.ones((1, lmax + 1))
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

# split into wavelets
f_wav_lm, f_scal_lm = analysis_axisym_lm_wav(alm[0], B, lmax+1, J_min)

fig, ax = plt.subplots(dpi=300)
ax.plot(hp.alm2cl(alm[0]))
fig.savefig(opj(imgdir, 'cl_alm_in'))
plt.close(fig)

# plot wavelets...
fig, ax = plt.subplots(dpi=300)
ax.plot(hp.alm2cl(f_scal_lm))
fig.savefig(opj(imgdir, 'cl_scal'))
plt.close(fig)

for idx in range(f_wav_lm.shape[-1]):
    fig, ax = plt.subplots(dpi=300)
    ax.plot(hp.alm2cl(f_wav_lm[:,idx]))
    fig.savefig(opj(imgdir, 'cl_wav_{}'.format(idx)))
    plt.close(fig)

minfo = map_utils.get_gauss_minfo(2 * lmax)
omap = np.ones((f_wav_lm.shape[-1] + 1, minfo.npix))
sht.alm2map(f_scal_lm, omap[0], ainfo, minfo, spin)
for idx in range(f_wav_lm.shape[-1]):
    sht.alm2map(np.ascontiguousarray(f_wav_lm[:,idx]), omap[idx+1], ainfo, minfo, spin)

nmaps = omap.shape[0]
fig, axs = plt.subplots(dpi=300, nrows=nmaps//2, ncols=2, constrained_layout=True)
for aidx, ax in enumerate(axs.ravel()):
    ax.imshow(omap[aidx].reshape(minfo.nrow, minfo.nphi[0]))
fig.savefig(opj(imgdir, 'omaps'))
plt.close(fig)

# turn back into alms
alm_out = synthesis_axisym_lm_wav(f_wav_lm, f_scal_lm, B, lmax+1, J_min)

fig, ax = plt.subplots(dpi=300)
ax.plot(hp.alm2cl(alm_out))
fig.savefig(opj(imgdir, 'cl_alm_out'))
plt.close(fig)

f_scal_lm_out = np.zeros(ainfo.nelem, dtype=np.complex128)
sht.map2alm(omap[0], f_scal_lm_out, minfo, ainfo, spin)
f_wav_lm_out = np.zeros((f_wav_lm.shape[-1], ainfo.nelem), dtype=np.complex128)
for idx in range(f_wav_lm.shape[-1]):
    sht.map2alm(omap[idx+1], f_wav_lm_out[idx], minfo, ainfo, spin)

# Plot power spectra of output alms.
fig, ax = plt.subplots(dpi=300)
ax.plot(hp.alm2cl(f_scal_lm_out))
fig.savefig(opj(imgdir, 'cl_scal_out'))
plt.close(fig)

for idx in range(f_wav_lm.shape[-1]):
    fig, ax = plt.subplots(dpi=300)
    ax.plot(hp.alm2cl(f_wav_lm_out[idx,:]))
    fig.savefig(opj(imgdir, 'cl_wav_out_{}'.format(idx)))
    plt.close(fig)

scal_l, wav_l = axisym_wav_l(B, lmax+1, J_min)

fig, ax = plt.subplots(dpi=300)
ax.plot(scal_l, label='Phi')
for idx in range(wav_l.shape[-1]):
    ax.plot(wav_l[:,idx], label='Psi_{}'.format(idx))
ax.legend()
fig.savefig(opj(imgdir, 'kernels'))
plt.close(fig)

ells = np.arange(scal_l.size)
fig, ax = plt.subplots(dpi=300)
ax.plot((scal_l ** 2) + np.sum(wav_l ** 2, axis=1))
ax.plot(scal_l ** 2)
ax.plot(wav_l ** 2)
fig.savefig(opj(imgdir, 'sq_sum'))
plt.close(fig)


lmaxs = []
def find_lmax(arr):
    return max(0, arr.size - np.argmax(arr[::-1] > 0) - 1)
lmaxs.append(find_lmax(scal_l))

for idx in range(wav_l.shape[-1]):
    lmaxs.append(find_lmax(wav_l[:,idx]))

minfos = []
ainfos = []
omaps = []
alms_out = []
for lmax in lmaxs:
    print(lmax)
    minfo = map_utils.get_gauss_minfo(2 * lmax)
    minfos.append(minfo)
    omaps.append(np.zeros((1, minfo.npix)))
    ainfo_tmp = sharp.alm_info(lmax=lmax)
    ainfos.append(ainfo_tmp)
    alms_out.append(np.zeros((1, ainfo_tmp.nelem), dtype=np.complex128))
    

print([omap.shape for omap in omaps])
print(alm.shape)

sht.alm2map(f_scal_lm, omaps[0], ainfo, minfos[0], spin)
for idx in range(f_wav_lm.shape[-1]):
    sht.alm2map(np.ascontiguousarray(f_wav_lm[:,idx]), omaps[idx+1][np.newaxis,:], ainfo, minfos[idx+1], spin)

nmaps = len(omaps)
fig, axs = plt.subplots(dpi=300, nrows=nmaps//2, ncols=2, constrained_layout=True)
for aidx, ax in enumerate(axs.ravel()):
    ax.imshow(omaps[aidx].reshape(minfos[aidx].nrow, minfos[aidx].nphi[0]))
fig.savefig(opj(imgdir, 'omaps_res'))
plt.close(fig)

sht.map2alm(omaps[0], alms_out[0], minfos[0], ainfos[0], spin)
for idx in range(f_wav_lm.shape[-1]):
    sht.map2alm(omaps[idx+1], alms_out[idx+1], minfos[idx+1], ainfos[idx+1], spin)

fig, ax = plt.subplots(dpi=300)
for idx in range(len(alms_out)):
    ax.plot(hp.alm2cl(alms_out[idx][0]))

ax.plot(hp.alm2cl(f_scal_lm), color='black', ls=':')
for idx in range(f_wav_lm.shape[-1]):
    ax.plot(hp.alm2cl(f_wav_lm[:,idx]), color='black', ls=':')
ax.set_xlim([0, lmax+1])
fig.savefig(opj(imgdir, 'cl_out'.format(idx)))
plt.close(fig)
