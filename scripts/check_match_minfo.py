import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import unittest
import numpy as np
from scipy.special import roots_legendre
import os
import tempfile
import pathlib

from pixell import sharp, enmap, curvedsky
import h5py

from optweight import map_utils, sht, wavtrans

opj = os.path.join

odir = '/home/adriaand/project/actpol/20230321_minfo_check'
os.makedirs(odir, exist_ok=True)

np.random.seed(11)

# Create cut sky enmap geometry                                                                                                                                                                                             
#ny, nx = 360, 720
#ny, nx = 360, 720
#ny, nx = 36, 72
ny, nx = 37, 72
res = [np.pi / (ny - 1), 2 * np.pi / nx]
dec_cut = np.radians([-60, 30])
shape, wcs = enmap.band_geometry(dec_cut, res=res, proj='car')

#shape, wcs = enmap.fullsky_geometry(res, shape=(ny, nx))

# NOTE
#wcs.wcs.cdelt[1] *= -1

# NOTE
#wcs.wcs.cdelt[0] *= -1

assert curvedsky.get_minfo(shape, wcs, quad=True)

#minfo = map_utils.get_enmap_minfo(shape, wcs, 2 * lmax, mtype='CC')                                                                                                                                                        
minfo = map_utils.match_enmap_minfo(shape, wcs)

print('here')

#print(shape)                                                                                                                                                                                                               
#print(wcs)                                                                                                                                                                                                                 

ntheta = shape[-2]
theta = enmap.pix2sky(shape, wcs, [np.arange(ntheta),np.zeros(ntheta)])[0]
#print(np.degrees(np.pi / 2 - theta))                                                                                                                                                                                       
#print(np.pi / 2 - theta)                                                                                                                                                                                                   

#print(minfo.theta.size)                                                                                                                                                                                                    
print(shape)

print(np.degrees(minfo.theta))
#print(np.degrees(minfo.theta))
print(minfo.offsets)
print(minfo.npix)
print(minfo.stride)

#assert False                                                                                                                                                                                                               

# Now get test alm                                                                                                                                                                                                          
# alm2map onto enmap                                                                                                                                                                                                        
# map2alm pixell, map2alm with minfo.                                                                                                                                                                                       
lmax = 18
cov_ell = np.ones((1, lmax + 1))
alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
omap = enmap.zeros(shape, wcs)
omap_me = np.zeros((1, minfo.npix))


curvedsky.alm2map(alm, omap, ainfo=ainfo)
fig, ax = plt.subplots(dpi=300)
ax.imshow(omap)
fig.savefig(opj(odir, 'omap_pixell'))
plt.close(fig)

sht.alm2map(alm, omap_me, ainfo, minfo, 0)
fig, ax = plt.subplots(dpi=300)
ax.imshow(map_utils.view_2d(omap_me, minfo)[0])
fig.savefig(opj(odir, 'omap_me'))
plt.close(fig)

print(np.allclose(map_utils.view_2d(omap_me, minfo)[0], omap))

#print(omap.shape)
#print(omap_me.shape)

alm_out = alm.copy() * 0
#sht.map2alm(omap_me, alm_out, minfo, ainfo, 0)
curvedsky.map2alm(omap, alm_out, ainfo=ainfo)

#print(np.abs(alm_out - alm).max())
#print(alm_out[::10])

print(np.allclose(alm, alm_out))

exit()

#print(omap)
#print(omap.shape)

alm_pixell = curvedsky.map2alm(omap, alm.copy(), ainfo=ainfo)
print(alm_pixell)

alm_me = alm.copy()
omap_me = np.asarray(omap.reshape(1, minfo.npix))
#assert False                                                                                                                                                                                                               
sht.map2alm(omap_me, alm_me, minfo, ainfo, 0)
print(alm_me)
