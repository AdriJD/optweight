import numpy as np

from pixell import enmap, sharp, curvedsky

from optweight import noise_utils, operators

ny = 50
nx = 100
shape = (ny, nx)
res = [np.pi / (ny - 1), 2 * np.pi / nx]
#shape, wcs = enmap.fullsky_geometry(res=res, shape=shape)
shape, wcs = enmap.band_geometry(np.pi/2-1e-10, res=res, shape=shape)

omap = curvedsky.make_projectable_map_by_pos(
    [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], 10, dims=(1,), oversample=4)
print(omap.shape)
shape = omap.shape[-2:]
wcs = omap.wcs
print(shape)

lmax = 30
bins = np.arange(lmax + 1)
icov_ell = np.ones((1, lmax + 1)) * 4 * np.pi / (10800 ** 2)
#icov_ell *= np.linspace(1, 2, num=lmax+1)
icov_ell *= np.cos(0.2 * np.arange(lmax+1)) ** 2

#w_ell = np.zeros((lmax + 1, lmax + 1))
#for ell in range(lmax + 1):
#    w_ell[ell,ell] = 1
ells_bins = np.arange(0, lmax+1, 1)

w_ell = np.zeros((ells_bins.size, lmax + 1))
for lidx, ell_bin in enumerate(ells_bins):
    print(ell_bin)
    try:
        end = ells_bins[lidx+1]
    except IndexError:
        end = lmax+1
    w_ell[lidx,ell_bin:end] = 1
print(w_ell)

noisebox = enmap.zeros((1, lmax + 1) + shape, wcs)
noisebox[:] = icov_ell[:,:,np.newaxis,np.newaxis]

icov_wav = noise_utils.noisebox2wavmat(noisebox, bins, w_ell, offsets=[0])

#for index in icov_wav.minfos:
#    print(icov_wav.minfos[index].nrow)


ainfo = sharp.alm_info(lmax=lmax)
spin = 0
alm = np.ones((1, ainfo.nelem), dtype=np.complex128)
icov_noise = operators.WavMatVecAlm(ainfo, icov_wav, w_ell, spin)

alm_out = icov_noise(alm)
#print(w_ell)
#print(icov_ell)
print(alm_out)
print(ainfo.lmul(alm, icov_ell[np.newaxis,:,:] /  (4 * np.pi / (10800 ** 2))))
print(alm_out / ainfo.lmul(alm, icov_ell[np.newaxis,:,:] /  (4 * np.pi / (10800 ** 2))))

#from optweight import wavtrans

#wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell)

#alm_out = np.zeros_like(alm)
#wavtrans.wav2alm(wav, alm_out, ainfo, spin, w_ell)

#np.testing.assert_array_almost_equal(alm_out, alm)
#print(alm_out)
#print(alm)

