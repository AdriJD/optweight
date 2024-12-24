'''
Estimate noise on full sky to check for biases.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy
import os
import time

import healpy as hp
from pixell import curvedsky, enplot, utils, enmap, sharp

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils
from optweight import noise_utils, alm_utils, mat_utils, alm_c_utils, wavtrans

opj = os.path.join
np.random.seed(39)

odir = '/home/adriaand/project/actpol/20210408_est_noise_pix_fullsky'
imgdir = opj(odir, 'img')

os.makedirs(imgdir, exist_ok=True)

lmax = 500
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

ainfo = sharp.alm_info(lmax)

cov_ell = np.ones(lmax + 1) * 5
#cov_ell = np.ones(lmax + 1) * 5 * np.linspace(10, 1, num=lmax+1)
#cov_ell = np.ones(lmax + 1) * 5 * (1 / ((ells + 0.5) / 10))
#cov_ell = np.ones(lmax + 1)
#cov_ell[:100] = 0
#cov_ell[300:] = 0

# Can I from N_ell create a N_pix that gives the correct power spectrum..
rand_map = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,))
pix_areas = enmap.pixsizemap(rand_map.shape, rand_map.wcs)
cov_pix = np.sum(cov_ell) / pix_areas / (lmax + 1)

plot = enplot.plot(cov_pix, colorbar=True, grid=False)
enplot.write(opj(imgdir, f'cov_pix'), plot)

rand_map[:] = np.random.randn(*rand_map.shape) * np.sqrt(cov_pix)

for pidx in range(rand_map.shape[0]):
    plot = enplot.plot(rand_map[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'rand_map_{pidx}'), plot)

plot = enplot.plot(pix_areas, colorbar=True, grid=False)
enplot.write(opj(imgdir, 'pix_areas'), plot)


alm = np.empty((rand_map.shape[0], ainfo.nelem), dtype=np.complex128)
curvedsky.map2alm(rand_map, alm, ainfo)
n_ell = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])


print(np.sum(cov_ell * (ells + 1)))
print(np.sum(n_ell[0,0] * (ells + 1)))

print(np.sum(cov_ell))
print(np.sum(n_ell[0,0]))
print(np.mean(n_ell[0,0]))

# Now try to draw map from cov_ell and get cov_pix.
rand_alm = curvedsky.rand_alm(cov_ell, ainfo=ainfo)
rand_alm = rand_alm[np.newaxis,:]

rand_alm_pix = curvedsky.alm2map(rand_alm, rand_map.copy(), ainfo=ainfo)

for pidx in range(rand_alm_pix.shape[0]):
    plot = enplot.plot(rand_alm_pix[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'rand_alm_pix_{pidx}'), plot)


# Cut away half of map
#rand_alm_pix[:,:,500:1500] = 0
#rand_alm = curvedsky.map2alm(rand_alm_pix, rand_alm, ainfo)

# Get true draw.
n_ell_draw = ainfo.alm2cl(rand_alm[:,None,:], rand_alm[None,:,:])

rand_alm_pix = curvedsky.alm2map(rand_alm, rand_map.copy(), ainfo=ainfo)
rand_alm_pix *= rand_alm_pix

# NOTE
alm = curvedsky.map2alm(rand_alm_pix, rand_alm.copy(), ainfo)
b_ell = hp.gauss_beam(np.radians(2), lmax=lmax)
alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
cov_pix_est = curvedsky.alm2map(alm, rand_alm_pix.copy(), ainfo=ainfo)
cov_pix_est[cov_pix_est < 0] = 0
#cov_pix_est = rand_alm_pix

for pidx in range(cov_pix_est.shape[0]):
    plot = enplot.plot(cov_pix_est[pidx], colorbar=True, grid=False, mask=0, mask_tol=100)
    enplot.write(opj(imgdir, f'cov_pix_est_before_{pidx}'), plot)

cov_pix_est /= pix_areas 
cov_pix_est /= (lmax + 1) ** 2 / 4 / np.pi

for pidx in range(cov_pix_est.shape[0]):
    plot = enplot.plot(cov_pix_est[pidx], colorbar=True, grid=False, mask=0, mask_tol=100)
    enplot.write(opj(imgdir, f'cov_pix_est_{pidx}'), plot)

rand_map_est = rand_map.copy()
rand_map_est[:] = np.random.randn(*rand_map_est.shape) * np.sqrt(cov_pix_est)

for pidx in range(rand_map_est.shape[0]):
    plot = enplot.plot(rand_map_est[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'rand_map_est_{pidx}'), plot)

curvedsky.map2alm(rand_map_est, alm, ainfo)
n_ell_out = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])

#n_ell_out /= (lmax + 1) ** 2 / 4 / np.pi

fig, axs = plt.subplots(dpi=300, constrained_layout=True, nrows=2)
axs[0].plot(ells, n_ell[0,0], label='n_ell')
axs[0].plot(ells, n_ell_out[0,0], label='n_ell_out')
axs[0].legend()
axs[1].plot(ells, n_ell_out[0,0] / n_ell[0,0])
fig.savefig(opj(imgdir, 'n_ell_out'))
plt.close(fig)

fig, axs = plt.subplots(dpi=300, nrows=2)
axs[0].plot(np.std(rand_map[0,10:-10,:], axis=-1), label='rand_map')
axs[1].plot(np.std(rand_map_est[0,10:-10,:], axis=-1), label='rand_map_est')
axs[0].legend()
axs[1].legend()
fig.savefig(opj(imgdir, 'noise_profiles'))
plt.close(fig)


#Try without smoothing but by averaging sims.
rand_alm_pix_i = rand_map.copy() * 0
for idx in range(10):
    rand_alm_i = curvedsky.rand_alm(cov_ell, ainfo=ainfo)
    rand_alm_i = rand_alm_i[np.newaxis,:]
    rand_alm_pix_i += curvedsky.alm2map(rand_alm_i, rand_map.copy(), ainfo=ainfo) ** 2
rand_alm_pix_i /= 10
cov_pix_est_av = rand_alm_pix_i

cov_pix_est_av /= pix_areas 
cov_pix_est_av /= (lmax + 1) ** 2 / 4 / np.pi

for pidx in range(cov_pix_est.shape[0]):
    plot = enplot.plot(cov_pix_est_av[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f'cov_pix_est_av_{pidx}'), plot)


rand_map_est_av = rand_map.copy()
rand_map_est_av[:] = np.random.randn(*rand_map_est.shape) * np.sqrt(cov_pix_est_av)
alm_i = curvedsky.map2alm(rand_map_est_av, alm.copy(), ainfo)
n_ell_out_i = ainfo.alm2cl(alm_i[:,None,:], alm_i[None,:,:])

fig, axs = plt.subplots(dpi=300, constrained_layout=True, nrows=2)
axs[0].plot(ells, n_ell[0,0], label='n_ell')
axs[0].plot(ells, n_ell_out[0,0], label='n_ell_out')
axs[0].plot(ells, n_ell_out_i[0,0], label='n_ell_out_i')
axs[0].legend()
axs[1].plot(ells, n_ell_out[0,0] / n_ell[0,0])
axs[1].plot(ells, n_ell_out_i[0,0] / n_ell[0,0])
fig.savefig(opj(imgdir, 'n_ell_i_out'))
plt.close(fig)

# Now split noise alm into wlms, determine cov for each, draw from each cov, get wlms, get alm, compare to cov_ell.
# alm2w
lamb = 1.5
#lamb = 1.2
#lamb = 1.7
#lamb = 3
lmin = 5
lmax_w = lmax
lmax_j = lmax - 10

w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
    lamb, lmax_w, j0=None, lmin=lmin, return_j=True, lmax_j=lmax_j)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
ax.set_xscale('log')
fig.savefig(opj(imgdir, 'kernels_log'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
for widx in range(w_ell.shape[0]):
    ax.plot(w_ell[widx], label='Phi')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels'))
plt.close(fig)

wlms, winfos = alm_utils.alm2wlm_axisym(rand_alm, ainfo, w_ell)

rand_alm_wav = np.zeros_like(rand_alm)

for widx, (wlm, winfo) in enumerate(zip(wlms, winfos)):

    cov_j = curvedsky.alm2map(wlm, rand_alm_pix.copy(), ainfo=winfo)
    cov_j *= cov_j

    alm_j = curvedsky.map2alm(cov_j, wlm.copy(), winfo)
    b_ell = hp.gauss_beam(np.radians(2), lmax=winfo.lmax)
    alm_c_utils.lmul(alm_j, b_ell, winfo, inplace=True)
    curvedsky.alm2map(alm_j, cov_j, ainfo=winfo)
    cov_j[cov_j < 0] = 0
    
    cov_j /= pix_areas 
    #cov_j /= (lmax + 1) ** 2 / 4 / np.pi # prob replace by sum w_ell or something like that.
#    cov_j /= np.sum(w_ell[widx]) ** 2 / 4 / np.pi
#    cov_j /= np.sum(w_ell[widx] * (2 * ells + 1)) / 4 / np.pi
    cov_j /= np.sum(w_ell[widx] ** 2 * (2 * ells + 1)) / 4 / np.pi 

    for pidx in range(cov_j.shape[0]):
        plot = enplot.plot(cov_j[pidx], colorbar=True, grid=False)
        enplot.write(opj(imgdir, f'cov_{widx}_{pidx}'), plot)

    rand_map_j = rand_map.copy()
    #np.random.seed(10)
    rand_map_j[:] = np.random.randn(*rand_map_j.shape) * np.sqrt(cov_j)
    wlm_rand = curvedsky.map2alm(rand_map_j, wlm.copy(), winfo)

    alm_utils.wlm2alm_axisym([wlm_rand], [winfo], w_ell[widx:widx+1,:],
                             alm=rand_alm_wav, ainfo=ainfo)

n_ell_wav = ainfo.alm2cl(rand_alm_wav[:,None,:], rand_alm_wav[None,:,:])

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.plot(ells, n_ell[0,0], label='n_ell')
ax.plot(ells, n_ell_out[0,0], label='n_ell_out')
ax.plot(ells, n_ell_draw[0,0], label='n_ell_draw')
ax.plot(ells, n_ell_wav[0,0], label='n_ell_wav')
ax.plot(ells, cov_ell, label='cov_ell')
#ax.set_yscale('log')
ax.legend()
fig.savefig(opj(imgdir, 'n_ell_out_wav'))
plt.close(fig)

print(np.sum(n_ell_wav[0,0]))

# Also plot off-diaginal cov elements...
for widx_a, (wlm_a, winfo_a) in enumerate(zip(wlms, winfos)):
    for widx_b, (wlm_b, winfo_b) in enumerate(zip(wlms, winfos)):

        if widx_b < widx_a:
            continue
        if widx_b > widx_a + 1:
            continue

        cov_a = curvedsky.alm2map(wlm_a, rand_alm_pix.copy(), ainfo=winfo_a)
        cov_b = curvedsky.alm2map(wlm_b, rand_alm_pix.copy(), ainfo=winfo_b)
        cov = cov_a * cov_b

        alm_j = curvedsky.map2alm(cov, wlm_a.copy(), winfo_a)
        b_ell = hp.gauss_beam(np.radians(2), lmax=winfo_a.lmax)
        alm_c_utils.lmul(alm_j, b_ell, winfo_a, inplace=True)
        curvedsky.alm2map(alm_j, cov, ainfo=winfo_a)

        cov /= pix_areas 
        #cov /= np.sum(w_ell[widx_a] * w_ell[widx_b] * (2 * ells + 1)) / 4 / np.pi 
        cov /= np.sum(((w_ell[widx_a] + w_ell[widx_b]) / 2) **2 * (2 * ells + 1)) / 4 / np.pi 

        for pidx in range(cov.shape[0]):
            plot = enplot.plot(cov[pidx], colorbar=True, grid=False, quantile=0.25)
            enplot.write(opj(imgdir, f'cov_{widx_a}_{widx_b}_{pidx}'), plot)

rand_maps = [np.random.randn(*rand_map_j.shape) for i in range(len(wlms))]

rand_alm_wav_cross = np.zeros_like(rand_alm)
for widx in range(len(wlms)):

    if widx == len(wlms) - 1:
        break

    cov_a = curvedsky.alm2map(wlms[widx], rand_alm_pix.copy(), ainfo=winfos[widx])
    cov_b = curvedsky.alm2map(wlms[widx+1], rand_alm_pix.copy(), ainfo=winfos[widx+1])
    
    cov_auto = cov_a * cov_a
    cov_cross = cov_a * cov_b

    alm_j = curvedsky.map2alm(cov_auto, wlms[widx].copy(), winfos[widx])
    b_ell = hp.gauss_beam(np.radians(2), lmax=winfos[widx].lmax)
    alm_c_utils.lmul(alm_j, b_ell, winfos[widx], inplace=True)
    curvedsky.alm2map(alm_j, cov_auto, ainfo=winfos[widx])
    cov_auto[cov_auto < 0] = 0

    cov_auto /= pix_areas 
    cov_auto /= np.sum(w_ell[widx] ** 2 * (2 * ells + 1)) / 4 / np.pi 

    alm_j = curvedsky.map2alm(cov_cross, wlms[widx].copy(), winfos[widx])
    b_ell = hp.gauss_beam(np.radians(2), lmax=winfos[widx].lmax)
    alm_c_utils.lmul(alm_j, b_ell, winfos[widx], inplace=True)
    curvedsky.alm2map(alm_j, cov_cross, ainfo=winfos[widx])
    #cov_cross[cov_cross < 0] = 0

    cov_cross /= pix_areas 
    cov_cross /= np.sum(((w_ell[widx] + w_ell[widx+1]) / 2) **2 * (2 * ells + 1)) / 4 / np.pi 
    
    sqrt_cov_cross = cov_cross / np.sqrt(cov_auto) / 2
    sqrt_cov_cross[~np.isfinite(sqrt_cov_cross)] = 0
    rand_map_j = rand_map.copy()
    #np.random.seed(10)
    #rand_map_j[:] = np.random.randn(*rand_map_j.shape) * (np.sqrt(cov_auto) + sqrt_cov_cross)
    rand_map_j[:] = rand_maps[widx] * np.sqrt(cov_auto)
    rand_map_j += rand_maps[widx+1] * sqrt_cov_cross
    if widx > 0:
        rand_map_j += rand_maps[widx-1] * sqrt_cov_cross
    wlm_rand = curvedsky.map2alm(rand_map_j, wlms[widx].copy(), winfos[widx])

    
    alm_utils.wlm2alm_axisym([wlm_rand], [winfos[widx]], w_ell[widx:widx+1,:],
                             alm=rand_alm_wav_cross, ainfo=ainfo)

n_ell_wav_cross = ainfo.alm2cl(rand_alm_wav_cross[:,None,:], rand_alm_wav_cross[None,:,:])

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.plot(ells, n_ell[0,0], label='n_ell')
ax.plot(ells, n_ell_out[0,0], label='n_ell_out')
ax.plot(ells, n_ell_draw[0,0], label='n_ell_draw')
ax.plot(ells, n_ell_wav[0,0], label='n_ell_wav')
ax.plot(ells, n_ell_wav_cross[0,0], label='n_ell_wav_cross')
ax.plot(ells, cov_ell, label='cov_ell')
ax.set_yscale('log')
ax.legend()
fig.savefig(opj(imgdir, 'n_ell_out_wav_cross'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
ax.plot(ells, n_ell_wav[0,0] / cov_ell)
fig.savefig(opj(imgdir, 'ratio'))
plt.close(fig)
