import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from pixell import curvedsky, sharp, enmap
import healpy as hp
from pys2let import analysis_axisym_lm_wav, synthesis_axisym_lm_wav, axisym_wav_l

import sht
import map_utils
import alm_utils

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20201019_wavelets/img/presentation'
hitsdir = '/home/adriaand/project/actpol/20200429_spinmap/run14/pa1_f150_s13'

cmap = 'RdBu_r'

#lmax = 6000
lmax = 1000
B = 3
J_min = 5
spin = 0 

scal_l, wav_l = axisym_wav_l(B, lmax + 1, J_min)

fig, ax = plt.subplots(dpi=300, figsize=(4,2), constrained_layout=True)
ax.plot(scal_l, label='Phi')
for idx in range(wav_l.shape[-1]):
    ax.plot(wav_l[:,idx], label='Psi_{}'.format(idx))
#ax.legend()
#ax.set_xscale('log')
ax.set_ylabel('Wavelet kernel')
ax.set_xlabel('Multipole')
fig.savefig(opj(imgdir, 'kernels'))
plt.close(fig)

# Load deep5 hits
div = enmap.read_fits(opj(hitsdir, 'div_deep5_0.fits'), sel=np.s_[0,0,:])
div = div[np.newaxis,:]

# Convert to Gauss-legendre
div_gauss, minfo = map_utils.enmap2gauss(div, lmax=2*lmax, area_pow=-1)
minfo_in = minfo
div_gauss /= div_gauss.max()
mask = div_gauss != 0
div_gauss[div_gauss < 0.00001] = 0.00001
div_gauss[div_gauss < 0.01] = 0.01 * np.sqrt(div_gauss[div_gauss < 0.01])
div_gauss[~mask] = 0
print(div_gauss.shape)

# Draw alms from n_ell
n_ell = np.ones((1, lmax+1))
n_ell[:,:50] = 100000
n_ell[:,:100] = 1000
n_ell[:,:200] = 100
n_ell[:,200:1000] = 10
n_ell[:,1000:] = 1

n_ell_white = np.ones_like(n_ell) * 300

alm, ainfo = curvedsky.rand_alm(n_ell, return_ainfo=True)
alm_white = curvedsky.rand_alm(n_ell_white, return_ainfo=False)

# alm2map in region -> noise map
omap = np.zeros(div_gauss.shape)
omap_white = np.zeros(div_gauss.shape)
print(omap.shape)
print(alm.shape)
sht.alm2map(alm, omap, ainfo, minfo, 0, adjoint=False)
sht.alm2map(alm_white, omap_white, ainfo, minfo, 0, adjoint=False)

# scale noise maps by hits
omap[div_gauss != 0] /= np.sqrt(div_gauss[div_gauss != 0])
omap += omap_white
omap[div_gauss == 0] = 0

omap_in = omap.copy()

omap_2d_gauss = omap[0].reshape(minfo.nrow, minfo.nphi[0])
div_2d_gauss = div_gauss[0].reshape(minfo.nrow, minfo.nphi[0])

def center(omap_2d, minfo):
    #start = None
    #end = None
    #for pidx in range(minfo.nphi[0]):
    #    if not np.any(omap_2d[:,pidx]):
    #        end = pidx
    #        break
    #print(end)
    #if end:
    nphi = minfo.nphi[0]
    phis = 2 * np.pi / nphi * np.arange(nphi)
    end = np.where(phis >= np.radians(20))[0][0]
    omap_2d = np.roll(omap_2d, +end, axis=1)
    start = np.where(phis >= np.radians(34))[0][0]
    #for pidx in range(minfo.nphi[0]):
    #    if np.any(omap_2d[:,pidx]):
    #        start = pidx
    #         break
    return omap_2d, start

omap_2d_gauss, start = center(omap_2d_gauss, minfo)
div_2d_gauss, _ = center(div_2d_gauss, minfo)

fig, ax = plt.subplots(dpi=300)
#arr_plot = omap_2d_gauss[:,start:].copy()
arr_plot = omap_2d_gauss[:,:start].copy()
arr_plot_in = arr_plot.copy()
vmin = arr_plot.min() / 8
vmax = arr_plot.max() / 8
#arr_plot[arr_plot == 0] = np.nan
im = ax.imshow(arr_plot, vmin=vmin, vmax=vmax, cmap=cmap)
fig.colorbar(im, ax=ax)
#ax.set_axis_off()
fig.savefig(opj(imgdir, 'alm_in'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
im = ax.imshow(1/div_2d_gauss[:,:start], cmap=cmap)
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'div'))
plt.close(fig)

# map2alm noisemap
sht.map2alm(omap, alm, minfo, ainfo, 0, adjoint=False)

# alm 2 wavelets
ells = np.arange(lmax + 1)
#q_ell = np.sqrt(4 * np.pi / (2 * ells + 1))
scal_l, wav_l = axisym_wav_l(B, lmax+1, J_min)
w_ell = np.zeros((wav_l.shape[1] + 1, wav_l.shape[0]))
w_ell[0] = scal_l
w_ell[1:] = wav_l.T
#w_ell /= q_ell

#f_wav_lm_old, f_scal_lm_old = analysis_axisym_lm_wav(alm[0], B, lmax+1, J_min)

#print(f_wav_lm_old.shape, hp.Alm.getlmax(f_wav_lm_old.shape[0]))

print('start')
#f_wav_lm, f_scal_lm = analysis_axisym_lm_wav(alm[0], B, lmax+1, J_min)
wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)
print('end')

for winfo in winfos:
    print(winfo.lmax)


#f_wav_lm = np.zeros((alm.shape[1], wav_l.shape[1]), np.complex128)
#for widx, wlm in enumerate(wlms[1:]):
#    f_wav_lm[:wlm.shape[1],widx] = wlm
#f_scal_lm = np.zeros(alm.shape[1], dtype=np.complex128)
#f_scal_lm[:wlms[0].size] = wlms[0]

#print(f_wav_lm_old)
#print(f_wav_lm)
#print('nerts')
#print(f_scal_lm_old)
#print(f_scal_lm)

#print(f_wav_lm_old.dtype)
#print(f_wav_lm.dtype)

#np.testing.assert_array_almost_equal(f_wav_lm, f_wav_lm_old)
#mask = f_scal_lm != f_scal_lm_old
#print(f_scal_lm[mask][-100:])
#print(f_scal_lm_old[mask][-100:])


#np.testing.assert_array_almost_equal(f_scal_lm, f_scal_lm_old)

#f_wav_lm, f_scal_lm = analysis_axisym_lm_wav(alm[0], B, lmax+1, J_min)

# alm2map for all wavelets -> plot.



#lmaxs = []
#def find_lmax(arr):
#    return max(0, arr.size - np.argmax(arr[::-1] > 0) - 1)
#lmaxs.append(find_lmax(scal_l))

#for idx in range(wav_l.shape[-1]):
#    lmaxs.append(find_lmax(wav_l[:,idx]))

minfos = []
#ainfos = []
omaps = []
#alms_out = []
#for lmax in lmaxs:
for winfo in winfos:
    #print('lmax', lmax)
    lmax = winfo.lmax
    minfo_tmp = map_utils.get_gauss_minfo(2 * lmax)
    minfos.append(minfo_tmp)
    omaps.append(np.zeros((1, minfo_tmp.npix)))
    #ainfo_tmp = sharp.alm_info(lmax=lmax)
    #ainfos.append(ainfo_tmp)
    #alms_out.append(np.zeros((1, ainfo_tmp.nelem), dtype=np.complex128))


#sht.alm2map(f_scal_lm, omaps[0], ainfo, minfos[0], spin)
#for idx in range(f_wav_lm.shape[-1]):
#    sht.alm2map(np.ascontiguousarray(f_wav_lm[:,idx]), omaps[idx+1][np.newaxis,:], ainfo, minfos[idx+1], spin)

for idx in range(len(wlms)):
    sht.alm2map(wlms[idx], omaps[idx][np.newaxis,:], winfos[idx], minfos[idx], spin)

for oidx in range(len(omaps)):
    omap_2d_tmp = omaps[oidx].reshape(minfos[oidx].nrow, minfos[oidx].nphi[0])
    omap_2d_tmp, start = center(omap_2d_tmp, minfos[oidx])

    theta = minfos[oidx].theta
    start_y = np.where(theta > np.radians(80))[0][0]
    end_y = np.where(theta > np.radians(100))[0][0]

    fig, ax = plt.subplots(dpi=300)
    arr_plot = omap_2d_tmp[start_y:end_y,:start].copy()
    #arr_plot -= np.mean([arr_plot[0,0], arr_plot[-1,0], arr_plot[0,-1], arr_plot[0,0]])
    arr_plot -= np.mean(arr_plot[0,:])
    vmin = arr_plot.min() / 4
    vmax = arr_plot.max() / 4
    arr_plot[arr_plot == 0] = np.nan
    im = ax.imshow(arr_plot, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)
    #ax.set_axis_off()
    fig.savefig(opj(imgdir, 'omap_{}'.format(oidx)))
    plt.close(fig)


alm_out, ainfo_out = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell)
lmax = ainfo_out.lmax
print(lmax)
#minfo = map_utils.get_gauss_minfo(2 * lmax)
minfo = minfo_in
omap = np.zeros((1, minfo.npix))

sht.alm2map(alm_out, omap, ainfo_out, minfo, spin)

omap_2d = omap.reshape(minfo.nrow, minfo.nphi[0])
omap_2d, start = center(omap_2d, minfo)

#theta = minfo.theta
#start_y = np.where(theta > np.radians(80))[0][0]
#end_y = np.where(theta > np.radians(100))[0][0]

fig, ax = plt.subplots(dpi=300)
#arr_plot = omap_2d[start_y:end_y,:start].copy()
arr_plot = omap_2d[:,:start].copy()
vmin = arr_plot.min() / 8
vmax = arr_plot.max() / 8
arr_plot[arr_plot == 0] = np.nan
im = ax.imshow(arr_plot, vmin=vmin, vmax=vmax, cmap=cmap)
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'alm_out'))
plt.close(fig)


alm_in = alm_out.copy()
sht.map2alm(omap_in, alm_in, minfo, ainfo_out, spin)
sht.alm2map(alm_in, omap_in, ainfo_out, minfo, spin)

diff_2d = (omap - omap_in).reshape(minfo.nrow, minfo.nphi[0])
diff_2d, start = center(diff_2d, minfo)


fig, ax = plt.subplots(dpi=300)
#arr_plot = diff_2d.copy()
arr_plot = diff_2d[:,:start].copy()
im = ax.imshow(arr_plot, cmap=cmap)#, vmin=0.9, vmax=1.1)
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'alm_diff'))
plt.close(fig)
    

