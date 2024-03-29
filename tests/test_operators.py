import unittest
import numpy as np

from pixell import curvedsky, enmap

from optweight import operators
from optweight import wavtrans
from optweight import sht
from optweight import alm_utils
from optweight import noise_utils
from optweight import dft
from optweight import map_utils

class TestOperators(unittest.TestCase):
    
    def test_MatVecAlm_init(self):

        self.assertRaises(TypeError, operators.MatVecAlm)

    def test_EllMatVecAlm(self):

        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = curvedsky.alm_info(lmax=2)
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
        ainfo = curvedsky.alm_info(lmax=2)
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
        ainfo = curvedsky.alm_info(lmax=2)
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
        ainfo = curvedsky.alm_info(lmax=2)
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
        ainfo = curvedsky.alm_info(lmax=2)
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
        np.testing.assert_array_almost_equal(alm, alm_exp)
        self.assertTrue(np.shares_memory(alm, alm_out))

    def test_matvec_pow_pix_alm(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = curvedsky.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])
        spin = [0, 2]
        power = 1

        minfo = map_utils.MapInfo.map_info_gauss_legendre(3, 5) # (lmax + 1).

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

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_pix_alm_sqrt(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = curvedsky.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])
        spin = [0, 2]
        power = 0.5

        minfo = map_utils.MapInfo.map_info_gauss_legendre(3, 5) # (lmax + 1).

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

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
    
    def test_WavMatVecAlm(self):

        lmax = 3
        ainfo = curvedsky.alm_info(lmax=lmax)
        spin = 0
        w_ell = np.zeros((2, lmax + 1))
        lmaxs = [1, lmax]
        w_ell[0,:lmaxs[0]+1] = 1
        w_ell[1,lmaxs[0]+1:] = 2
        m_wav = wavtrans.Wav(2)

        # Add first map.
        minfo1 = map_utils.MapInfo.map_info_gauss_legendre(
            lmaxs[0] + 1, 2 * lmaxs[0] + 1)
        m_arr1 = np.ones((1, minfo1.npix)) * 10
        index1 = (0, 0)
        m_wav.add(index1, m_arr1, minfo1)

        # Add second map.
        minfo2 = map_utils.MapInfo.map_info_gauss_legendre(
            lmaxs[1] + 1, 2 * lmaxs[1] + 1)
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
        ainfo = curvedsky.alm_info(lmax=lmax)
        spin = [0, 2]
        w_ell = np.zeros((2, lmax + 1))
        lmaxs = [2, lmax]
        w_ell[0,:lmaxs[0]+1] = 1
        w_ell[1,lmaxs[0]+1:] = 2
        m_wav = wavtrans.Wav(2, preshape=(npol,))

        # Add first map.
        minfo1 = map_utils.MapInfo.map_info_gauss_legendre(
            lmaxs[0] + 1, 2 * lmaxs[0] + 1)
        m_arr1 = np.ones((npol, minfo1.npix)) * 10
        index1 = (0, 0)
        m_wav.add(index1, m_arr1, minfo1)

        # Add second map.
        minfo2 = map_utils.MapInfo.map_info_gauss_legendre(
            lmaxs[1] + 1, 2 * lmaxs[1] + 1)
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

        oversample = 4
        lmax_map = 10
        shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax_map),
                                                 2 * np.pi / (2 * oversample * lmax_map + 1)])
        omap = enmap.zeros((1,) + shape, wcs)

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

        ainfo = curvedsky.alm_info(lmax=lmax)
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

        oversample = 4
        lmax_map = 10
        shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax_map),
                                                 2 * np.pi / (2 * oversample * lmax_map + 1)])
        omap = enmap.zeros((1,) + shape, wcs)
        
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

        ainfo = curvedsky.alm_info(lmax=lmax)
        spin = 0
        alm = np.ones((1, ainfo.nelem), dtype=np.complex128)
        icov_noise = operators.WavMatVecAlm(ainfo, icov_wav, w_ell, spin)

        alm_out = icov_noise(alm)
        alm_out_exp =  ainfo.lmul(alm, icov_ell[np.newaxis,:,:])
        alm_out_exp /= (4 * np.pi / (10800 ** 2))

        np.testing.assert_allclose(alm_out, alm_out_exp, rtol=5e-2)

    def test_op2mat(self):

        # Test identity matrix with complex input.
        op = lambda x : x
        dtype = np.complex64
        nrow = 3

        mat = operators.op2mat(op, nrow, dtype)
        mat_exp = np.eye(3, dtype=np.complex64)

        np.testing.assert_allclose(mat, mat_exp)
        
        # More complicated matrix.
        mat_exp = np.asarray([[1, 2, 3], [4, 5, 6]])
        op = lambda x : np.dot(mat_exp, x)

        mat = operators.op2mat(op, 2, mat_exp.dtype, ncol=3)
        np.testing.assert_allclose(mat, mat_exp)
        
    def test_op2map_sht(self):

        # Example of how to deal with complex alms.
        lmax = 3
        ainfo = curvedsky.alm_info(lmax)
        minfo = map_utils.MapInfo.map_info_gauss_legendre(lmax + 1, 2 * lmax + 1)

        def alm2map_real(alm_real):
            alm = alm_real.view(np.complex64)
            omap = np.zeros(minfo.npix, dtype=np.float32)
            sht.alm2map(alm, omap, ainfo, minfo, 0)
            return omap

        mat = operators.op2mat(alm2map_real, minfo.npix, np.float32, ncol=2 * ainfo.nelem)

        alm_test = (np.ones(ainfo.nelem) + np.arange(ainfo.nelem) * 1j).astype(np.complex64)
        alm_test_real = alm_test.view(np.float32)
        omap = np.dot(mat, alm_test_real)
        omap_exp = np.zeros(minfo.npix, dtype=np.float32)
        sht.alm2map(alm_test, omap_exp, ainfo, minfo, 0)

        np.testing.assert_allclose(omap, omap_exp, rtol=1e-5)

    def test_PixEllPixMatVecMap(self):
        
        # This function should do M_pix Y X_ell Yt M_pix. So if we make 
        # M_pix = W, where W are the quadrature weights, and X_ell = 1, 
        # the functions should do W Y Yt W = W.

        lmax = 2
        minfo = map_utils.MapInfo.map_info_gauss_legendre(lmax + 1, 2 * lmax + 1)
        spin = [0, 2]

        m_pix = np.zeros((3, 3, minfo.nrow, minfo.nphi[0]))
        m_pix[0,0,:,:] = minfo.weight[:,np.newaxis]
        m_pix[1,1,:,:] = minfo.weight[:,np.newaxis]
        m_pix[2,2,:,:] = minfo.weight[:,np.newaxis]
        m_pix = m_pix.reshape((3, 3, minfo.npix))

        x_ell = np.ones((3, 3, lmax + 1)) * np.eye(3)[:,:,np.newaxis]
        
        op = operators.PixEllPixMatVecMap(m_pix, x_ell, minfo, spin)

        # Give W^-1 as map, we should get ones (for spin 0 at least).
        imap = np.ones((3, minfo.npix)) * (1 / m_pix[0,0])        
        imap_copy = imap.copy()
        omap = op(imap)
        omap_exp = np.ones((1, minfo.npix))
        np.testing.assert_allclose(omap[0:1], omap_exp)
        np.testing.assert_allclose(imap[0:1], imap_copy[0:1])
        self.assertFalse(np.shares_memory(imap, omap))

        # Inplace.
        op = operators.PixEllPixMatVecMap(m_pix, x_ell, minfo, spin, inplace=True)

        # Give W^-1 as map, we should get ones (for spin 0 at least).
        omap = op(imap)
        omap_exp = np.ones((1, minfo.npix))
        np.testing.assert_allclose(omap[0:1], omap_exp)
        np.testing.assert_allclose(imap[0:1], omap_exp)
        self.assertTrue(np.shares_memory(imap, omap))

    def test_FMatVecAlm(self):

        # This function should do Yt W F^(-1) M F Y. So if we make 
        # M = 1, we should get an identity operation.

        lmax = 10
        spin = [0, 2]
        ainfo = curvedsky.alm_info(lmax)
        minfo = map_utils.MapInfo.map_info_clenshaw_curtis(2 * lmax + 1, 2 * lmax + 1)

        # Create smaller patch for 2D power spectrum.
        ny = 6
        nx = 11
        shape, wcs = enmap.geometry([0,0], res=10, deg=True, shape=(ny, nx))
        lwcs = dft.lwcs_real(shape, wcs)
        m_k = enmap.ones((3, minfo.nrow, int(minfo.nphi[0]) // 2 + 1), 
                         lwcs)
        
        alm = np.arange(1, 1 + 3 * ainfo.nelem).reshape(3, ainfo.nelem)
        alm = alm.astype(np.complex128)
        alm[:,lmax+1:] += 1j * alm[:,lmax+1:]
        # Set ell=0, 1 pol elements to zero.
        alm[1:,:2] = 0
        alm[1:,lmax+1] = 0

        op = operators.FMatVecAlm(ainfo, m_k, minfo, spin, power=2)
        alm_out = op(alm)

        np.testing.assert_allclose(alm_out, alm)

    def test_add_operators(self):
        
        op_1 = lambda x : x + 2
        op_2 = lambda x : x + 10

        op_add = operators.add_operators(op_1, op_2)
        vec = np.ones((3, 10))
        out = op_add(vec)
        out_exp = np.ones_like(vec) * 14

        np.testing.assert_array_equal(out, out_exp)

        # Now with slicing.
        slice_2 = np.s_[:1]

        op_add = operators.add_operators(
            op_1, op_2, slice_2=slice_2)
        out = op_add(vec)
        out_exp = np.ones_like(vec)
        out_exp[0] = 14
        out_exp[1] = 3
        out_exp[2] = 3

        np.testing.assert_array_equal(out, out_exp)

        # Different slicing.
        slice_2 = np.s_[0]

        op_add = operators.add_operators(
            op_1, op_2, slice_2=slice_2)
        out = op_add(vec)
        out_exp = np.ones_like(vec)
        out_exp[0] = 14
        out_exp[1] = 3
        out_exp[2] = 3

        np.testing.assert_array_equal(out, out_exp)

    def test_PixMatVecMap(self):
        
        m_pix = np.arange(12).reshape(2, 2, 3)
        power = 1
        op = operators.PixMatVecMap(m_pix, power)

        imap = np.arange(6).reshape(2, 3)
        imap_copy = imap.copy()
        omap_exp = np.einsum('ijk, jk -> ik', m_pix, imap)
        omap = op(imap)
        np.testing.assert_allclose(omap, omap_exp)
        np.testing.assert_allclose(imap, imap_copy)
        self.assertFalse(np.shares_memory(imap, omap))

    def test_PixMatVecMap_diag(self):
        
        m_pix = np.arange(6).reshape(2, 3)
        power = 1
        op = operators.PixMatVecMap(m_pix, power)

        imap = m_pix + 1
        imap_copy = imap.copy()
        omap_exp = m_pix * imap
        omap = op(imap)
        np.testing.assert_allclose(omap, omap_exp)
        np.testing.assert_allclose(imap, imap_copy)
        self.assertFalse(np.shares_memory(imap, omap))

    def test_PixMatVecMap_inplace(self):

        m_pix = np.arange(12).reshape(2, 2, 3)
        power = 1
        op = operators.PixMatVecMap(m_pix, power, inplace=True)

        imap = np.arange(6).reshape(2, 3)
        imap_copy = imap.copy()
        omap_exp = np.einsum('ijk, jk -> ik', m_pix, imap)
        omap = op(imap)
        np.testing.assert_allclose(imap, omap_exp)
        np.testing.assert_allclose(omap, omap_exp)
        self.assertTrue(np.shares_memory(imap, omap))
        
        m_pix = np.arange(6).reshape(2, 3)
        power = 1
        op = operators.PixMatVecMap(m_pix, power, inplace=True)

        imap = m_pix + 1
        imap_copy = imap.copy()
        omap_exp = m_pix * imap
        omap = op(imap)
        np.testing.assert_allclose(imap, omap_exp)
        np.testing.assert_allclose(omap, omap_exp)
        self.assertTrue(np.shares_memory(imap, omap))

    def test_YMatVecAlm(self):

        lmax = 4
        spin = [0, 2]

        ainfo = curvedsky.alm_info(lmax)
        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        imap = np.zeros((3, minfo.npix))
        omap_exp = imap.copy()
        sht.alm2map(alm, omap_exp, ainfo, minfo, spin)        

        op = operators.YMatVecAlm(ainfo, minfo, spin)
        omap = op(alm)

        np.testing.assert_allclose(omap, omap_exp)

    def test_YMatVecAlm_inplace(self):

        lmax = 4
        spin = [0, 2]

        ainfo = curvedsky.alm_info(lmax)
        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        imap = np.zeros((3, minfo.npix))
        omap_exp = imap.copy()
        omap = imap.copy()
        sht.alm2map(alm, omap_exp, ainfo, minfo, spin)        

        op = operators.YMatVecAlm(ainfo, minfo, spin)
        omap2 = op(alm, omap=omap)

        np.testing.assert_allclose(omap, omap_exp)
        self.assertTrue(np.shares_memory(omap, omap2))
        
    def test_YMatVecAlm_qweight(self):

        lmax = 4
        spin = [0, 2]

        ainfo = curvedsky.alm_info(lmax)
        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        imap = np.zeros((3, minfo.npix))
        omap_exp = imap.copy()
        sht.alm2map(alm, omap_exp, ainfo, minfo, spin, adjoint=True)

        op = operators.YMatVecAlm(ainfo, minfo, spin, qweight=True)
        omap = op(alm)

        np.testing.assert_allclose(omap, omap_exp)

    def test_YTWMatVecMap(self):

        lmax = 4
        spin = [0, 2]

        ainfo = curvedsky.alm_info(lmax)
        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        imap = np.zeros((3, minfo.npix))    
        sht.alm2map(alm, imap, ainfo, minfo, spin)        

        oalm_exp = np.zeros_like(alm)
        sht.map2alm(imap, oalm_exp, minfo, ainfo, spin)

        op = operators.YTWMatVecMap(minfo, ainfo, spin)
        oalm = op(imap)

        np.testing.assert_allclose(oalm, oalm_exp)

    def test_YTWMatVecMap_inplace(self):

        lmax = 4
        spin = [0, 2]

        ainfo = curvedsky.alm_info(lmax)
        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        imap = np.zeros((3, minfo.npix))    
        sht.alm2map(alm, imap, ainfo, minfo, spin)        

        oalm_exp = np.zeros_like(alm)
        oalm = np.zeros_like(alm)
        sht.map2alm(imap, oalm_exp, minfo, ainfo, spin)

        op = operators.YTWMatVecMap(minfo, ainfo, spin)
        oalm2 = op(imap, oalm=oalm)

        np.testing.assert_allclose(oalm, oalm_exp)
        self.assertTrue(np.shares_memory(oalm, oalm2))

    def test_YTWMatVecMap_qweight(self):

        lmax = 4
        spin = [0, 2]

        ainfo = curvedsky.alm_info(lmax)
        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        imap = np.zeros((3, minfo.npix))    
        sht.alm2map(alm, imap, ainfo, minfo, spin)        

        oalm_exp = np.zeros_like(alm)
        sht.map2alm(imap, oalm_exp, minfo, ainfo, spin, adjoint=True)

        op = operators.YTWMatVecMap(minfo, ainfo, spin, qweight=False)
        oalm = op(imap)

        np.testing.assert_allclose(oalm, oalm_exp)

    def test_FMatVecF(self):

        ny = 6
        nx = 10
        fmap = np.full((3, ny, nx // 2 + 1), 2, np.complex128)
        
        fmat2d = np.full((ny, nx // 2 + 1), 2.)
        op = operators.FMatVecF(fmat2d)
                
        fmap_out = op(fmap)
        fmap_exp = fmap * 2
        np.testing.assert_allclose(fmap_out, fmap_exp)

        # Power
        op = operators.FMatVecF(fmat2d, power=2)
                
        fmap_out = op(fmap)
        fmap_exp = fmap * 4
        np.testing.assert_allclose(fmap_out, fmap_exp)

        # Power inplace.
        op = operators.FMatVecF(fmat2d, power=2, inplace=True)
                
        fmap_out = op(fmap)
        self.assertTrue(np.shares_memory(fmap, fmap_out))
        np.testing.assert_allclose(fmap, fmap_exp)        

        # npol matrix.
        fmat2d = fmat2d * np.asarray([1, 2, 3])[:,np.newaxis,np.newaxis]
        fmap = np.full((3, ny, nx // 2 + 1), 1.)
        op = operators.FMatVecF(fmat2d, power=2)
        fmap_exp = fmap.copy()
        fmap_exp[0] *= 4
        fmap_exp[1] *= 16
        fmap_exp[2] *= 36
        fmap_out = op(fmap)
        np.testing.assert_allclose(fmap_out, fmap_exp)        

        # npol, npol matrix.
        fmat2d = np.full((3, 3, ny, nx // 2 + 1), 1.)
        fmat2d *= (np.eye(3) * [1, 2, 3])[:,:,np.newaxis,np.newaxis]
        op = operators.FMatVecF(fmat2d, power=2)
        fmap_exp = fmap.copy()
        fmap_exp[0] *= 1
        fmap_exp[1] *= 4
        fmap_exp[2] *= 9
        fmap_out = op(fmap)
        np.testing.assert_allclose(fmap_out, fmap_exp)        
