import unittest
import numpy as np

from pixell import sharp, curvedsky, enmap

from optweight import operators
from optweight import wavtrans
from optweight import sht
from optweight import alm_utils
from optweight import noise_utils

class TestOperators(unittest.TestCase):
    
    def test_MatVecAlm_init(self):

        self.assertRaises(TypeError, operators.MatVecAlm)

    def test_EllMatVecAlm(self):

        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3, 3))
        icov[0,0] = 2
        icov[1,1] = 4
        icov[2,2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= 2
        alm_exp[1] *= 4
        alm_exp[2] *= 9

        matvec_alm = operators.EllMatVecAlm(ainfo, icov)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_EllMatVecAlm_sqrt(self):

        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3, 3))
        icov[0,0] = 2
        icov[1,1] = 4
        icov[2,2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= np.sqrt(2)
        alm_exp[1] *= np.sqrt(4)
        alm_exp[2] *= np.sqrt(9)

        matvec_alm = operators.EllMatVecAlm(ainfo, icov, power=0.5)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_ell_alm_diag(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3))
        icov[0] = 2
        icov[1] = 4
        icov[2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= 2
        alm_exp[1] *= 4
        alm_exp[2] *= 9

        matvec_alm = operators.EllMatVecAlm(ainfo, icov)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_ell_alm_diag_sqrt(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3))
        icov[0] = 2
        icov[1] = 4
        icov[2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= np.sqrt(2)
        alm_exp[1] *= np.sqrt(4)
        alm_exp[2] *= np.sqrt(9)

        matvec_alm = operators.EllMatVecAlm(ainfo, icov, power=0.5)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_ell_alm_sqrt_inplace(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3, 3))
        icov[0,0] = 2
        icov[1,1] = 4
        icov[2,2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= np.sqrt(2)
        alm_exp[1] *= np.sqrt(4)
        alm_exp[2] *= np.sqrt(9)

        matvec_alm = operators.EllMatVecAlm(ainfo, icov, power=0.5, inplace=True)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
        np.testing.assert_array_equal(alm, alm_exp)
        self.assertTrue(np.shares_memory(alm, alm_out))

    def test_matvec_pow_pix_alm(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])
        spin = [0, 2]
        power = 1

        minfo = sharp.map_info_gauss_legendre(3) # (lmax + 1).

        # This function should do Yt icov Y. So if we make icov = W, where
        # W are the quadrature weights, the functions should do Yt W Y = 1.

        icov = np.zeros((3, 3, minfo.nrow, minfo.nphi[0]))
        icov[0,0,:,:] = minfo.weight[:,np.newaxis]
        icov[1,1,:,:] = minfo.weight[:,np.newaxis]
        icov[2,2,:,:] = minfo.weight[:,np.newaxis]
        icov = icov.reshape((3, 3, minfo.npix))

        alm_exp = alm.copy()

        matvec_alm = operators.PixMatVecAlm(
            ainfo, icov, minfo, spin, power=power, inplace=False)
        alm_out = matvec_alm(alm)

        #alm_out = operators.matvec_pow_pix_alm(
        #    alm, ainfo, icov, minfo, spin, power, inplace=False)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_pix_alm_sqrt(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])
        spin = [0, 2]
        power = 0.5

        minfo = sharp.map_info_gauss_legendre(3) # (lmax + 1).

        # This function should do Yt icov Y. So if we make icov = W^2, where
        # W are the quadrature weights, the functions should do Yt sqrt(W^2) Y = 1.

        icov = np.zeros((3, 3, minfo.nrow, minfo.nphi[0]))
        icov[0,0,:,:] = minfo.weight[:,np.newaxis] ** 2
        icov[1,1,:,:] = minfo.weight[:,np.newaxis] ** 2
        icov[2,2,:,:] = minfo.weight[:,np.newaxis] ** 2
        icov = icov.reshape((3, 3, minfo.npix))

        alm_exp = alm.copy()

        matvec_alm = operators.PixMatVecAlm(
            ainfo, icov, minfo, spin, power=power, inplace=False)
        alm_out = matvec_alm(alm)

        #alm_out = operators.matvec_pow_pix_alm(
        #    alm, ainfo, icov, minfo, spin, power, inplace=False)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
    
    def test_WavMatVecAlm(self):

        lmax = 3
        ainfo = sharp.alm_info(lmax=lmax)
        spin = 0
        w_ell = np.zeros((2, lmax + 1))
        lmaxs = [1, lmax]
        w_ell[0,:lmaxs[0]+1] = 1
        w_ell[1,lmaxs[0]+1:] = 2
        m_wav = wavtrans.Wav(2)

        # Add first map.
        minfo1 = sharp.map_info_gauss_legendre(lmaxs[0] + 1)
        m_arr1 = np.ones((1, minfo1.npix)) * 10
        index1 = (0, 0)
        m_wav.add(index1, m_arr1, minfo1)

        # Add second map.
        minfo2 = sharp.map_info_gauss_legendre(lmaxs[1] + 1)
        m_arr2 = np.ones((1, minfo2.npix)) * 20
        index2 = (1, 1)
        m_wav.add(index2, m_arr2, minfo2)
        
        icov_wav = operators.WavMatVecAlm(ainfo, m_wav, w_ell, spin)

        alm = np.ones((1, ainfo.nelem), dtype=np.complex128)
        alm_out = icov_wav(alm)
        
        # Do manual computation.
        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)
        omap1 = m_wav.maps[0,0].copy()
        sht.alm2map(wlms[0], omap1, winfos[0], m_wav.minfos[0,0], spin, adjoint=False)
        omap2 = m_wav.maps[1,1].copy()
        sht.alm2map(wlms[1], omap2, winfos[1], m_wav.minfos[1,1], spin, adjoint=False)
        
        omap1 *= m_wav.maps[0,0]
        omap2 *= m_wav.maps[1,1]
                
        sht.map2alm(omap1, wlms[0], m_wav.minfos[0,0], winfos[0], spin, adjoint=True)
        sht.map2alm(omap2, wlms[1], m_wav.minfos[1,1], winfos[1], spin, adjoint=True)

        alm_exp, _ = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell, alm=None, ainfo=ainfo)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
        self.assertFalse(np.shares_memory(alm, alm_out))

    def test_WavMatVecAlm_pol(self):

        lmax = 3
        npol = 3
        ainfo = sharp.alm_info(lmax=lmax)
        spin = [0, 2]
        w_ell = np.zeros((2, lmax + 1))
        lmaxs = [2, lmax]
        w_ell[0,:lmaxs[0]+1] = 1
        w_ell[1,lmaxs[0]+1:] = 2
        m_wav = wavtrans.Wav(2, preshape=(npol,))

        # Add first map.
        minfo1 = sharp.map_info_gauss_legendre(lmaxs[0] + 1)
        m_arr1 = np.ones((npol, minfo1.npix)) * 10
        index1 = (0, 0)
        m_wav.add(index1, m_arr1, minfo1)

        # Add second map.
        minfo2 = sharp.map_info_gauss_legendre(lmaxs[1] + 1)
        m_arr2 = np.ones((npol, minfo2.npix)) * 20
        index2 = (1, 1)
        m_wav.add(index2, m_arr2, minfo2)
        
        icov_wav = operators.WavMatVecAlm(ainfo, m_wav, w_ell, spin)

        alm = np.ones((npol, ainfo.nelem), dtype=np.complex128)
        alm_out = icov_wav(alm)
        
        # Do manual computation.
        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)
        omap1 = m_wav.maps[0,0].copy()
        sht.alm2map(wlms[0], omap1, winfos[0], m_wav.minfos[0,0], spin, adjoint=False)
        omap2 = m_wav.maps[1,1].copy()
        sht.alm2map(wlms[1], omap2, winfos[1], m_wav.minfos[1,1], spin, adjoint=False)
        
        omap1 *= m_wav.maps[0,0]
        omap2 *= m_wav.maps[1,1]
                
        sht.map2alm(omap1, wlms[0], m_wav.minfos[0,0], winfos[0], spin, adjoint=True)
        sht.map2alm(omap2, wlms[1], m_wav.minfos[1,1], winfos[1], spin, adjoint=True)

        alm_exp, _ = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell, alm=None, ainfo=ainfo)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_WavMatVecAlm_noisebox(self):
        
        # Create noisebox with equal icov spectrum for each
        # pixel, compare wavelet weighting to N^-1_ell weighting.
        ny = 50
        nx = 100
        omap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], 10, dims=(1,), oversample=4)
        shape = omap.shape[-2:]
        wcs = omap.wcs

        lmax = 30
        bins = np.arange(lmax + 1)
        # Create oscillating spectrum. Scale by conversion factor
        # to get muK ^-2 arcmin^-2.
        icov_ell = np.ones((1, lmax + 1)) * 4 * np.pi / (10800 ** 2)
        icov_ell *= np.cos(np.arange(lmax+1)) ** 2

        # Create wavelets kernels of Delta_ell = 1 for exact comparison
        # to direct N^-1_ell weighting.
        ells_bins = np.arange(0, lmax+1, 1)
        w_ell = np.zeros((ells_bins.size, lmax + 1))
        for lidx, ell_bin in enumerate(ells_bins):
            try:
                end = ells_bins[lidx+1]
            except IndexError:
                end = lmax+1
            w_ell[lidx,ell_bin:end] = 1

        noisebox = enmap.zeros((1, lmax + 1) + shape, wcs)
        noisebox[:] = icov_ell[:,:,np.newaxis,np.newaxis]

        icov_wav = noise_utils.noisebox2wavmat(noisebox, bins, w_ell, offsets=[0])

        ainfo = sharp.alm_info(lmax=lmax)
        spin = 0
        alm = np.ones((1, ainfo.nelem), dtype=np.complex128)
        icov_noise = operators.WavMatVecAlm(ainfo, icov_wav, w_ell, spin)

        alm_out = icov_noise(alm)
        alm_out_exp =  ainfo.lmul(alm, icov_ell[np.newaxis,:,:])
        alm_out_exp /= (4 * np.pi / (10800 ** 2))
        np.testing.assert_allclose(alm_out, alm_out_exp, rtol=1e-2)

    def test_WavMatVecAlm_noisebox_flat(self):
        
        # Create noisebox with equal icov spectrum for each
        # pixel, compare wavelet weighting to N^-1_ell weighting.
        ny = 50
        nx = 100
        omap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], 10, dims=(1,), oversample=4)
        shape = omap.shape[-2:]
        wcs = omap.wcs

        lmax = 30
        bins = np.arange(lmax + 1)
        # Create almost flat spectrum. Scale by conversion factor
        # to get muK ^-2 arcmin^-2.
        icov_ell = np.ones((1, lmax + 1)) * 4 * np.pi / (10800 ** 2)
        icov_ell *= np.linspace(1, 1.5, num=lmax+1)

        # Create wavelets kernels of Delta_ell = 3. Should approximately 
        # be the same as direct N^-1_ell weighting.
        ells_bins = np.arange(0, lmax+1, 3)
        w_ell = np.zeros((ells_bins.size, lmax + 1))
        for lidx, ell_bin in enumerate(ells_bins):
            try:
                end = ells_bins[lidx+1]
            except IndexError:
                end = lmax+1
            w_ell[lidx,ell_bin:end] = 1

        noisebox = enmap.zeros((1, lmax + 1) + shape, wcs)
        noisebox[:] = icov_ell[:,:,np.newaxis,np.newaxis]

        icov_wav = noise_utils.noisebox2wavmat(noisebox, bins, w_ell, offsets=[0])

        ainfo = sharp.alm_info(lmax=lmax)
        spin = 0
        alm = np.ones((1, ainfo.nelem), dtype=np.complex128)
        icov_noise = operators.WavMatVecAlm(ainfo, icov_wav, w_ell, spin)

        alm_out = icov_noise(alm)
        alm_out_exp =  ainfo.lmul(alm, icov_ell[np.newaxis,:,:])
        alm_out_exp /= (4 * np.pi / (10800 ** 2))

        np.testing.assert_allclose(alm_out, alm_out_exp, rtol=5e-2)
