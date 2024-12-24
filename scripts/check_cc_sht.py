import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from pixell import curvedsky, sharp

from optweight import sht, map_utils

opj = os.path.join

imgdir = '/home/adriaand/project/actpol/20230219_check_cc_sht'

lmax = 999
spin = [0, 2]
ells = np.arange(lmax + 1)

cov_ell = np.ones((3, 3, lmax + 1)) * np.eye(3)[:,:,np.newaxis]
#cov_ell *= (1 + (np.maximum(ells, 10) / 2000) ** -3.)
cov_ell[1,1,:2] = 0
cov_ell[2,2,:2] = 0

alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
alm_out = np.zeros_like(alm)

# Create GL map.
nrings = lmax + 1
nphi = 2 * lmax + 1
minfo = sharp.map_info_gauss_legendre(nrings, nphi)

omap = np.zeros((3, minfo.npix))
omap_out = omap.copy()

sht.alm2map(alm, omap, ainfo, minfo, spin)
sht.map2alm(omap, alm_out, minfo, ainfo, spin)
sht.alm2map(alm_out, omap_out, ainfo, minfo, spin)

print(np.allclose(alm_out, alm))
print(np.allclose(omap_out, omap))

# Create CC map.
nrings = 2 * lmax + 1
#nrings = 1 * lmax + 1
nphi = 2 * lmax + 1
minfo = sharp.map_info_clenshaw_curtis(nrings, nphi)

omap = np.zeros((3, minfo.npix))
omap_out = omap.copy()

sht.alm2map(alm, omap, ainfo, minfo, spin)
sht.map2alm(omap, alm_out, minfo, ainfo, spin)
sht.alm2map(alm_out, omap_out, ainfo, minfo, spin)

print(np.allclose(alm_out, alm))
print(np.allclose(omap_out, omap))

# Create 0.5 CC map.
#nrings = 1 * lmax + 1
#nrings = int(1.25 * lmax + 1)
nrings = int(1.5 * lmax + 1)
nphi = 2 * lmax + 1
minfo = sharp.map_info_clenshaw_curtis(nrings, nphi)

omap = np.zeros((3, minfo.npix))
omap_out = omap.copy()

sht.alm2map(alm, omap, ainfo, minfo, spin)
sht.map2alm(omap, alm_out, minfo, ainfo, spin)
sht.alm2map(alm_out, omap_out, ainfo, minfo, spin)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.view_2d(omap[pidx], minfo))
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, f'omap_{pidx}'))
    plt.close(fig)

for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(map_utils.view_2d(omap_out[pidx] - omap[pidx], minfo), vmin=-0.01, vmax=0.01)
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, f'diff_{pidx}'))
    plt.close(fig)

#print(alm_out)
#print(alm)
#print(alm_out-alm)


print(np.allclose(alm_out, alm))
print(np.allclose(omap_out, omap))
