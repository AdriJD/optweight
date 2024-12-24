import os
import numpy as np
import time

from pixell import sharp, curvedsky

from optweight import sht

opj = os.path.join

odir = '/home/adriaand/analysis/actpol/20210128_libsharp_pixell/intel_fast_math_car_enmap'
tag = 'l1'
grid = 'car_enmap'

numthreads = os.getenv('OMP_NUM_THREADS')

spin = [0, 2]
niter = 2
lmaxs = np.linspace(1000, 10000, 10, dtype=int)

timings = np.ones((len(lmaxs), 2, niter)) * np.nan

def lmax2nside(lmax):
    
    for n in range(20):
        nside = 2 ** n
        if 3 * nside >= lmax:
            return nside
    raise ValueError()

for lidx, lmax in enumerate(lmaxs):

    if lmax > 5000 and int(numthreads) < 10:
        continue

    ainfo = sharp.alm_info(lmax)
    alm = np.zeros((3, ainfo.nelem), dtype=np.complex128)

    if grid == 'gl':
        nrings = lmax + 1
        nphi = 2 * lmax + 1    
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)
        omap = np.zeros((3, minfo.npix))
    elif grid == 'healpix':
        nside = lmax2nside(lmax)
        print('lmax : {}, nside {}'.format(lmax, nside))
        minfo = sharp.map_info_healpix(nside)
        omap = np.zeros((3, minfo.npix))
    elif grid == 'car':
        #minfo = sharp.map_info_clenshaw_curtis(2 * lmax + 1, 4 * lmax + 1)
        minfo = sharp.map_info_clenshaw_curtis(2 * lmax + 1, 4 * lmax)
        omap = np.zeros((3, minfo.npix))
        print(omap.size)
    elif grid == 'car_enmap':
        omap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
        print(omap.size)

    for idx in range(niter):

        if grid == 'car_enmap':
            t0 = time.time()
            curvedsky.alm2map(alm, omap, ainfo=ainfo, spin=spin, direct=True)
            timings[lidx,0,idx] = time.time() - t0
            t0 = time.time()
            curvedsky.map2alm(omap, alm, ainfo=ainfo, lmax=lmax, spin=spin, direct=True)
            timings[lidx,1,idx] = time.time() - t0
                        
        else:
            t0 = time.time()
            sht.alm2map(alm, omap, ainfo, minfo, spin)
            timings[lidx,0,idx] = time.time() - t0
            t0 = time.time()
            sht.map2alm(omap, alm, minfo, ainfo, spin)
            timings[lidx,1,idx] = time.time() - t0

        print(lmax, idx, timings[lidx,:,idx])

np.save(opj(odir, 'lmaxs.npy'), lmaxs)
np.save(opj(odir, 'timings_{}_{}.npy'.format(tag, numthreads)), timings)

#print('alm2map, lmax : {}, mean : {}, std : {}'
#      .format(lmax, np.mean(timings[0]), np.std(timings[0])))
#print('map2alm, lmax : {}, mean : {}, std : {}'
#      .format(lmax, np.mean(timings[1]), np.std(timings[1])))

