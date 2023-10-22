import unittest
import numpy as np
from scipy.special import roots_legendre
import os
import tempfile
import pathlib

from pixell import sharp, enmap, curvedsky
import h5py

from optweight import map_utils, sht, wavtrans

class TestMapUtils(unittest.TestCase):

    def test_mapinfo_init(self):

        ntheta = 5
        theta = np.linspace(0, np.pi, num=ntheta, endpoint=True)
        nphi = np.asarray([3, 6, 10, 6, 3])
        weight = np.ones(ntheta)
        phi0 = np.zeros(ntheta)
        stride = np.ones(ntheta)
        offsets = np.asarray([0, 3, 9, 19, 25])

        minfo = map_utils.MapInfo(theta, weight, nphi=nphi, phi0=phi0, offsets=offsets,
                                  stride=stride)

        self.assertFalse(np.shares_memory(theta, minfo.theta))
        self.assertFalse(np.shares_memory(weight, minfo.weight))
        self.assertFalse(np.shares_memory(nphi, minfo.nphi))
        self.assertFalse(np.shares_memory(phi0, minfo.phi0))
        self.assertFalse(np.shares_memory(stride, minfo.stride))
        self.assertFalse(np.shares_memory(offsets, minfo.offsets))

        self.assertEqual(minfo.nrow, ntheta)
        self.assertEqual(minfo.npix, 28)

        np.testing.assert_allclose(minfo.theta, theta)
        np.testing.assert_allclose(minfo.weight, weight)
        np.testing.assert_allclose(minfo.nphi, nphi)
        np.testing.assert_allclose(minfo.phi0, phi0)
        np.testing.assert_allclose(minfo.stride, stride)
        np.testing.assert_allclose(minfo.offsets, offsets)

        self.assertEqual(minfo.theta.dtype, np.float64)
        self.assertEqual(minfo.weight.dtype, np.float64)
        self.assertEqual(minfo.phi0.dtype, np.float64)
        self.assertEqual(minfo.nphi.dtype, np.uint64)
        self.assertEqual(minfo.stride.dtype, np.int64)
        self.assertEqual(minfo.offsets.dtype, np.uint64)

    def test_mapinfo_init_alt(self):

        ntheta = 5
        theta = np.linspace(0, np.pi, num=ntheta, endpoint=True)
        weight = np.ones(ntheta)

        minfo = map_utils.MapInfo(theta, weight)

        # Defaults to 0 phi samples, so empty map.
        self.assertEqual(minfo.nrow, ntheta)
        self.assertEqual(minfo.npix, 0)

        # Again, with 2 phi pixels per ring.
        minfo = map_utils.MapInfo(theta, weight, nphi=2)

        self.assertEqual(minfo.nrow, ntheta)
        self.assertEqual(minfo.npix, 2 * ntheta)

        np.testing.assert_allclose(minfo.stride, np.ones(ntheta))
        np.testing.assert_allclose(minfo.phi0, np.zeros(ntheta))        

        offsets_exp = np.asarray([0, 2, 4, 6, 8])
        np.testing.assert_allclose(minfo.offsets, offsets_exp)

    def test_map_info_healpix(self):

        nside = 4
        minfo = map_utils.MapInfo.map_info_healpix(nside)

        self.assertEqual(minfo.npix, 12 * nside ** 2)
        self.assertAlmostEqual(np.sum(minfo.weight * minfo.nphi), 4 * np.pi)
        self.assertAlmostEqual(np.sum(minfo.nphi), 12 * nside ** 2)        

    def test_map_info_gauss_legendre(self):

        ntheta = 4
        nphi = 2
        minfo = map_utils.MapInfo.map_info_gauss_legendre(ntheta, nphi=nphi)

        self.assertAlmostEqual(np.sum(minfo.weight * minfo.nphi), 4 * np.pi)
        
    def test_map_info_clenshaw_curtis(self):

        ntheta = 4
        nphi = 2
        minfo = map_utils.MapInfo.map_info_clenshaw_curtis(ntheta, nphi=nphi)

        theta_exp = np.linspace(0, np.pi, num=ntheta, endpoint=True)
        self.assertAlmostEqual(np.sum(minfo.weight * minfo.nphi), 4 * np.pi)
        np.testing.assert_allclose(minfo.theta, theta_exp)

    def test_map_info_fejer1(self):

        ntheta = 4
        nphi = 2
        minfo = map_utils.MapInfo.map_info_fejer1(ntheta, nphi=nphi)

        theta_exp = np.linspace(0, np.pi, num=ntheta + 1, endpoint=True)[:-1]
        theta_exp += np.pi / ntheta / 2
        
        self.assertAlmostEqual(np.sum(minfo.weight * minfo.nphi), 4 * np.pi)
        np.testing.assert_allclose(minfo.theta, theta_exp)

    def test_map_info_fejer2(self):

        ntheta = 4
        nphi = 2
        minfo = map_utils.MapInfo.map_info_fejer2(ntheta, nphi=nphi)

        theta_exp = np.linspace(0, np.pi, num=ntheta + 2, endpoint=True)[1:-1]
        
        self.assertAlmostEqual(np.sum(minfo.weight * minfo.nphi), 4 * np.pi)
        np.testing.assert_allclose(minfo.theta, theta_exp)
        
    def test_get_gauss_minfo(self):

        lmax = 5
        minfo = map_utils.get_gauss_minfo(lmax)

        # We expect support for polynomial in cos(theta) of order lmax,
        # so lmax = 2 n - 1 -> n = 3.
        n_theta = int(lmax / 2) + 1
        n_phi = lmax + 1
        ct, ct_w = roots_legendre(n_theta)

        self.assertTrue(np.all(minfo.nphi == lmax + 1))
        np.testing.assert_array_almost_equal(minfo.theta, np.arccos(ct)[::-1])
        np.testing.assert_array_almost_equal(
            minfo.weight, ct_w[::-1] * 2 * np.pi / n_phi)
        np.testing.assert_array_almost_equal(minfo.phi0, np.zeros(n_theta))

        offsets_exp = np.arange(n_theta) * n_phi
        np.testing.assert_array_almost_equal(minfo.offsets, offsets_exp)
        np.testing.assert_array_almost_equal(minfo.stride, np.ones(n_theta))

    def test_get_gauss_minfo_cutsky(self):

        lmax = 5
        theta_min = np.pi / 3
        theta_max = 2 * np.pi / 3
        minfo = map_utils.get_gauss_minfo(lmax, theta_min=theta_min, theta_max=theta_max)

        n_theta = 1
        n_phi = lmax + 1
        ct, ct_w = roots_legendre(int(lmax / 2) + 1)
        ct = np.asarray(ct[1])
        ct_w = np.asarray(ct_w[1])

        self.assertTrue(np.all(minfo.nphi == lmax + 1))
        np.testing.assert_array_almost_equal(minfo.theta, np.asarray(np.arccos(ct)))
        np.testing.assert_array_almost_equal(
            minfo.weight, ct_w * 2 * np.pi / n_phi)
        np.testing.assert_array_almost_equal(minfo.phi0, np.zeros(n_theta))

        offsets_exp = np.arange(n_theta) * n_phi
        np.testing.assert_array_almost_equal(minfo.offsets, offsets_exp)
        np.testing.assert_array_almost_equal(minfo.stride, np.ones(n_theta))

    def test_get_gauss_minfo_cutsky_arclen(self):

        lmax = 5
        theta_min = np.pi / 3
        theta_max = 2 * np.pi / 3
        minfo, arc_len = map_utils.get_gauss_minfo(
            lmax, theta_min=theta_min, theta_max=theta_max, return_arc_len=True)

        n_theta = 1
        n_phi = lmax + 1
        ct, ct_w = roots_legendre(int(lmax / 2) + 1)

        arc_len_exp = (np.pi - 2 * np.arccos(ct[-1])) / 2

        np.testing.assert_array_almost_equal(arc_len, np.asarray(arc_len_exp))

    def test_get_cc_minfo(self):

        lmax = 5
        minfo = map_utils.get_cc_minfo(lmax)

        # We expect support for polynomial in cos(theta) of order lmax,
        n_theta = lmax + 1
        n_phi = lmax + 1

        self.assertTrue(np.all(minfo.nphi == lmax + 1))
        theta_exp = np.linspace(0, np.pi, n_theta)
        np.testing.assert_allclose(minfo.theta, theta_exp, atol=1e-7)
        np.testing.assert_array_almost_equal(minfo.phi0, np.zeros(n_theta))

        offsets_exp = np.arange(n_theta) * n_phi
        np.testing.assert_array_almost_equal(minfo.offsets, offsets_exp)
        np.testing.assert_array_almost_equal(minfo.stride, np.ones(n_theta))

    def test_get_equal_area_gauss_minfo(self):

        lmax = 9
        minfo = map_utils.get_equal_area_gauss_minfo(lmax)

        # We expect support for polynomial in cos(theta) of order lmax,
        # so lmax = 2 n - 1 -> n = 5.
        n_theta = int(lmax / 2) + 1
        n_phi = lmax + 1
        ct, ct_w = roots_legendre(n_theta)

        np.testing.assert_array_almost_equal(minfo.theta, np.arccos(ct)[::-1])
        np.testing.assert_array_almost_equal(minfo.phi0, np.zeros(n_theta))

        # Test if areas are equal.
        np.testing.assert_allclose(minfo.weight / minfo.weight[2], np.ones(n_theta),
                                   rtol=0.12)

        # And rings are not.
        self.assertTrue(minfo.nphi[0] < minfo.nphi[2])
        self.assertTrue(minfo.nphi[1] < minfo.nphi[2])
        # Symmetric?
        self.assertTrue(minfo.nphi[0] == minfo.nphi[4])
        self.assertTrue(minfo.nphi[1] == minfo.nphi[3])

        # Test if offsets and nphi match up.
        self.assertEqual(minfo.offsets[-1] + minfo.nphi[-1], minfo.npix)
        np.testing.assert_array_equal(np.ediff1d(minfo.offsets), minfo.nphi[:-1])

        # If changing gl_band, we should get GL rings in wider band.
        minfo = map_utils.get_equal_area_gauss_minfo(lmax, gl_band=np.pi/4)

        self.assertTrue(minfo.nphi[0] < minfo.nphi[2])
        self.assertTrue(minfo.nphi[1] == minfo.nphi[2])

        self.assertTrue(minfo.nphi[0] == minfo.nphi[4])
        self.assertTrue(minfo.nphi[1] == minfo.nphi[3])

        # If changing ratio_pow to 0 we should get back GL.
        minfo = map_utils.get_equal_area_gauss_minfo(lmax, ratio_pow=0)
        minfo_gl = map_utils.get_gauss_minfo(lmax)
        self.assertTrue(map_utils.minfo_is_equiv(minfo, minfo_gl))

    def test_enmap2gauss_fullsky(self):

        lmax = 100
        spin = 0

        # Create random enmap with a low bandlimit.
        cov_ell = np.ones((1, lmax + 1))
        cov_ell[:,5:] = 0
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
        omap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=6)
        curvedsky.alm2map_cyl(alm, omap, ainfo=ainfo, spin=spin)

        m_gl, minfo = map_utils.enmap2gauss(omap, 2 * lmax)

        # map2alm into second GL map.
        nrings = lmax + 1
        nphi = 2 * lmax + 1

        minfo_2 = sharp.map_info_gauss_legendre(nrings, nphi)
        m_gl_exp = np.zeros((1, minfo_2.npix))
        sht.alm2map(alm, m_gl_exp, ainfo, minfo_2, spin)

        # This really is not a great method, hard to get correct.
        np.testing.assert_array_almost_equal(m_gl, m_gl_exp, decimal=3)

    def test_enmap2gauss_fullsky_power(self):

        lmax = 100
        spin = 0
        area_pow = 3

        # Create random enmap with a low bandlimit.
        cov_ell = np.ones((1, lmax + 1))
        cov_ell[:,5:] = 0
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
        omap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=6)
        curvedsky.alm2map_cyl(alm, omap, ainfo=ainfo, spin=spin)

        m_gl, minfo = map_utils.enmap2gauss(omap, 2 * lmax, area_pow=area_pow)

        # map2alm into second GL map.
        nrings = lmax + 1
        nphi = 2 * lmax + 1

        minfo_2 = sharp.map_info_gauss_legendre(nrings, nphi)
        m_gl_exp = np.zeros((1, minfo_2.npix))
        sht.alm2map(alm, m_gl_exp, ainfo, minfo_2, spin)

        # Scale expected map by (area_out / area_in)^1.
        area_in_map = enmap.pixsizemap(
            omap.shape, omap.wcs, separable="auto", broadcastable=False)

        m_gl_exp = m_gl_exp.reshape(1, minfo_2.nrow, minfo_2.nphi[0])

        for tidx in range(m_gl_exp.shape[1]):
            # Determine closest ring in input map.
            dec = np.pi / 2 - minfo_2.theta[tidx]
            pix = enmap.sky2pix(omap.shape, omap.wcs, [[dec], [0]])
            pidx_y = round(pix[0,0])
            pidx_x = round(pix[1,0])
            area_in = area_in_map[pidx_y,pidx_x]
            m_gl_exp[:,tidx,:] *= (minfo_2.weight[tidx] / area_in) ** area_pow

        m_gl_exp = m_gl_exp.reshape((1, minfo_2.npix))

        # Again, kind of terrible. First and last ring areas are 30% different.
        np.testing.assert_array_almost_equal(m_gl/m_gl_exp, np.ones_like(m_gl),
                                             decimal=0)

    def test_get_arc_len(self):

        # Theta range [2, 15].
        # Thetas: 3, 5, 10, 14.
        # Arc lengths: 2, 3.5, 4.5, 3.

        # Give unsorted array.
        thetas = np.asarray([10, 3, 5, 14])
        arc_lens = map_utils.get_arc_len(thetas, 2, 15)
        arc_lens_exp = np.asarray([4.5, 2, 3.5, 3])

        np.testing.assert_almost_equal(arc_lens_exp, arc_lens)

    def test_get_arc_len_err(self):

        thetas = np.asarray([10, 10, 5, 14])
        self.assertRaises(ValueError, map_utils.get_arc_len, thetas, 2, 15)

        thetas = np.asarray([10, 20, 5, 14])
        self.assertRaises(ValueError, map_utils.get_arc_len, thetas, 2, 15)

        thetas = np.asarray([10, 1, 5, 14])
        self.assertRaises(ValueError, map_utils.get_arc_len, thetas, 2, 15)

    def test_inv_qweight_map(self):

        lmax = 3
        npol = 3
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)

        m = np.ones((npol, minfo.npix))
        m = m.reshape(npol, minfo.nrow, minfo.nphi[0])
        m[:,:,:] = minfo.weight[np.newaxis,:,np.newaxis]
        m = m.reshape(npol, minfo.npix)

        out = map_utils.inv_qweight_map(m, minfo, inplace=False)

        np.testing.assert_array_almost_equal(out, np.ones_like(m))

    def test_rand_map(self):

        cov_pix = np.ones((2, 2, 3))
        draw1 = map_utils.rand_map_pix(cov_pix)

        self.assertEqual(draw1.shape, (2, 3))

        draw2 = map_utils.rand_map_pix(cov_pix)

        self.assertRaises(
            AssertionError, np.testing.assert_array_almost_equal, draw1, draw2)

    def test_rand_map_diag(self):

        cov_pix = np.zeros((2, 2, 3))
        cov_pix[0,0,:] = [1, 2, 3]
        cov_pix[1,1,:] = [4, 5, 6]

        cov_pix_diag = np.ones((2, 3))
        cov_pix_diag[0] = cov_pix[0,0]
        cov_pix_diag[1] = cov_pix[1,1]

        np.random.seed(1)
        draw1 = map_utils.rand_map_pix(cov_pix)
        np.random.seed(1)
        draw2 = map_utils.rand_map_pix(cov_pix_diag)

        np.testing.assert_array_almost_equal(draw1, draw2)

    def test_rand_map_dtype(self):

        cov_pix = np.ones((2, 2, 3), dtype=np.float32)
        draw = map_utils.rand_map_pix(cov_pix)
        self.assertEqual(draw.dtype, cov_pix.dtype)

        cov_pix = np.zeros((2, 2, 3), dtype=np.float32)
        draw = map_utils.rand_map_pix(cov_pix)
        self.assertEqual(draw.dtype, cov_pix.dtype)

    def test_get_isotropic_ivar(self):

        lmax = 3
        npol = 3
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)

        icov_pix = np.ones((npol, npol, minfo.npix))
        icov_pix[0,0] = 10
        # Off-diagonal should be ignored.
        icov_pix[0,1] = 2
        icov_pix[1,0] = 2

        itau = map_utils.get_isotropic_ivar(icov_pix, minfo)

        icov_pix = icov_pix.reshape((npol, npol, minfo.nrow, minfo.nphi[0]))
        icov_pix[:,:,:,:] /= minfo.weight[np.newaxis,np.newaxis,:,np.newaxis]
        icov_pix = icov_pix.reshape((npol, npol, minfo.npix))

        itau_exp = np.zeros((npol, npol))
        itau_exp[0,0] = np.sum(icov_pix[0,0] ** 2) / np.sum(icov_pix[0,0])
        itau_exp[1,1] = np.sum(icov_pix[1,1] ** 2) / np.sum(icov_pix[1,1])
        itau_exp[2,2] = np.sum(icov_pix[2,2] ** 2) / np.sum(icov_pix[2,2])

        np.testing.assert_array_almost_equal(itau, itau_exp)

    def test_get_isotropic_ivar_diag(self):

        lmax = 3
        npol = 3
        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)

        icov_pix = np.ones((npol, minfo.npix))
        icov_pix[0] = 10

        itau = map_utils.get_isotropic_ivar(icov_pix, minfo)

        icov_pix = icov_pix.reshape((npol, minfo.nrow, minfo.nphi[0]))
        icov_pix[:,:,:] /= minfo.weight[np.newaxis,:,np.newaxis]
        icov_pix = icov_pix.reshape((npol, minfo.npix))

        itau_exp = np.zeros((npol, npol))
        itau_exp[0,0] = np.sum(icov_pix[0] ** 2) / np.sum(icov_pix[0])
        itau_exp[1,1] = np.sum(icov_pix[1] ** 2) / np.sum(icov_pix[1])
        itau_exp[2,2] = np.sum(icov_pix[2] ** 2) / np.sum(icov_pix[2])

        np.testing.assert_array_almost_equal(itau, itau_exp)

    def test_get_ivar_ell(self):

        lmax = 3
        npol = 3
        minfo = sharp.map_info_gauss_legendre(lmax + 1)

        icov_pix = np.ones((npol, minfo.npix))

        icov_wav = wavtrans.Wav(2)

        # Add first map.
        icov_wav.add((0,0), icov_pix, minfo)

        # Second map.
        m_arr = np.ones((3, minfo.npix))
        icov_wav.add((1,1), icov_pix * 2, minfo)

        w_ell = np.zeros((2, lmax + 1))
        w_ell[0] = [1, 1, 0, 0]
        w_ell[1] = [0, 0, 1, 1]

        ivar_ell = map_utils.get_ivar_ell(icov_wav, w_ell)

        self.assertEqual(ivar_ell.shape, (npol, npol, lmax + 1))

        icov_pix = icov_pix.reshape((npol, minfo.nrow, minfo.nphi[0]))
        icov_pix[:,:,:] /= minfo.weight[np.newaxis,:,np.newaxis]
        icov_pix = icov_pix.reshape((npol, minfo.npix))

        amp_exp = np.sum(icov_pix[0] ** 2) / np.sum(icov_pix[0])
        ivar_ell_exp = np.zeros((npol, npol, lmax+1))
        ivar_ell_exp[0,0] = np.asarray([1, 1, 2, 2]) * amp_exp
        ivar_ell_exp[1,1] = np.asarray([1, 1, 2, 2]) * amp_exp
        ivar_ell_exp[2,2] = np.asarray([1, 1, 2, 2]) * amp_exp

        np.testing.assert_array_almost_equal(ivar_ell, ivar_ell_exp)

    def test_get_ivar_ell_err(self):

        lmax = 3
        w_ell = np.zeros((2, lmax + 1))
        w_ell[0] = [1, 1, 0, 0]
        w_ell[1] = [0, 0, 1, 1]

        # Not (npol, npol).
        icov_wav = wavtrans.Wav(2, preshape=(1,2))
        self.assertRaises(ValueError, map_utils.get_ivar_ell, icov_wav, w_ell)

        # Too many leading dimensions.
        icov_wav = wavtrans.Wav(2, preshape=(2, 2, 2))
        self.assertRaises(ValueError, map_utils.get_ivar_ell, icov_wav, w_ell)

    def test_rand_wav(self):

        npol = 3

        cov_wav = wavtrans.Wav(2)

        # Add first map.
        lmax = 3
        minfo1 = sharp.map_info_gauss_legendre(lmax + 1)
        cov_pix = np.ones((npol, npol, minfo1.npix))
        cov_wav.add((0,0), cov_pix, minfo1)

        # Second map.
        lmax = 4
        minfo2 = sharp.map_info_gauss_legendre(lmax + 1)
        cov_pix = np.ones((npol, npol, minfo2.npix))
        cov_wav.add((1,1), cov_pix, minfo2)

        rand_wav = map_utils.rand_wav(cov_wav)

        self.assertEqual(rand_wav.shape, (2,))
        self.assertEqual(rand_wav.maps[0].shape, (npol, minfo1.npix))
        self.assertEqual(rand_wav.maps[1].shape, (npol, minfo2.npix))

    def test_round_icov_matrix(self):

        npol = 3
        npix = 4

        icov_pix = np.ones((npol, npol, npix))

        icov_pix_round = map_utils.round_icov_matrix(icov_pix)

        self.assertFalse(np.shares_memory(icov_pix_round, icov_pix))
        np.testing.assert_array_equal(icov_pix_round, icov_pix)

        # I expect that rows and columns that hit zero diagonal
        # are set to zero too
        icov_pix[1,1,2] = 1e-3
        icov_pix_exp = np.ones_like(icov_pix)
        icov_pix_exp[:,1,2] = 0
        icov_pix_exp[1,:,2] = 0
        icov_pix_round = map_utils.round_icov_matrix(icov_pix)

        np.testing.assert_array_equal(icov_pix_round, icov_pix_exp)

        # Small off-diagonal element should not triger update.
        icov_pix[:] = 1
        icov_pix[1,0,2] = 1e-3
        icov_pix_exp = icov_pix.copy()
        icov_pix_round = map_utils.round_icov_matrix(icov_pix)

        np.testing.assert_array_equal(icov_pix_round, icov_pix_exp)

        # Matrix that is completely zeros should work.
        icov_pix[:] = 0
        icov_pix_exp = icov_pix.copy()
        icov_pix_round = map_utils.round_icov_matrix(icov_pix)

        np.testing.assert_array_equal(icov_pix_round, icov_pix_exp)

        #assert False

    def test_round_icov_matrix_diag(self):

        # Test if function also works if (npol, npix) array is given.
        npol = 3
        npix = 4

        icov_pix = np.ones((npol, npix))

        icov_pix_round = map_utils.round_icov_matrix(icov_pix)

        self.assertFalse(np.shares_memory(icov_pix_round, icov_pix))
        np.testing.assert_array_equal(icov_pix_round, icov_pix)

        # I expect that rows and columns that hit zero diagonal
        # are set to zero too
        icov_pix[1,2] = 1e-3
        icov_pix_exp = np.ones_like(icov_pix)
        icov_pix_exp[1,2] = 0
        icov_pix_round = map_utils.round_icov_matrix(icov_pix)

        np.testing.assert_array_equal(icov_pix_round, icov_pix_exp)

    def test_round_icov_matrix_threshold(self):

        # Test for threshold option.

        npol = 1
        npix = 4
        icov_pix = np.ones((npol, npix))

        icov_pix[0,2] = 1e-3

        rtol = 1e-2
        val_exp = rtol * np.median(icov_pix)

        icov_pix_exp = np.ones_like(icov_pix)
        icov_pix_exp[0,2] = val_exp
        icov_pix_round = map_utils.round_icov_matrix(
            icov_pix, rtol=rtol, threshold=True)

        np.testing.assert_allclose(icov_pix_round, icov_pix_exp)

    def test_round_icov_matrix_err(self):

        npol = 3
        npix = 4

        icov_pix = np.ones((npol, 2, npix))
        self.assertRaises(ValueError, map_utils.round_icov_matrix, icov_pix)

        icov_pix = np.ones((npol, npol, npol, npix))
        self.assertRaises(ValueError, map_utils.round_icov_matrix, icov_pix)

        icov_pix = np.ones((npix))
        self.assertRaises(ValueError, map_utils.round_icov_matrix, icov_pix)

    def test_select_mask_edge(self):

        lmax = 6
        minfo = map_utils.get_gauss_minfo(lmax)
        mask = np.asarray([[0, 1, 1, 0, 1, 1, 0],
                           [1, 1, 1, 0, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        mask = mask.reshape(-1)

        edges = map_utils.select_mask_edge(mask, minfo)

        edges_exp = np.asarray([[0, 1, 1, 0, 1, 1, 0],
                                [1, 1, 1, 0, 1, 0, 1],
                                [0, 0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0]], dtype=bool)

        edges_exp = edges_exp.reshape(-1)
        np.testing.assert_array_equal(edges, edges_exp)

    def test_select_mask_edge_2d(self):

        lmax = 6
        minfo = map_utils.get_gauss_minfo(lmax)
        mask1 = np.asarray([[0, 1, 1, 0, 1, 1, 0],
                            [1, 1, 1, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        mask2 = np.asarray([[1, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 1, 1, 1],
                            [0, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0]], dtype=bool)

        mask = np.zeros((2, minfo.npix), dtype=bool)
        mask[0] = mask1.ravel()
        mask[1] = mask2.ravel()

        edges = map_utils.select_mask_edge(mask, minfo)

        edges_exp1 = np.asarray([[0, 1, 1, 0, 1, 1, 0],
                                 [1, 1, 1, 0, 1, 0, 1],
                                 [0, 0, 0, 0, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        edges_exp2 = np.asarray([[1, 0, 1, 0, 1, 1, 0],
                                 [0, 1, 1, 0, 1, 0, 1],
                                 [0, 0, 1, 0, 1, 1, 1],
                                 [1, 0, 0, 0, 0, 0, 0]], dtype=bool)

        edges_exp = np.zeros((2, minfo.npix), dtype=bool)
        edges_exp[0] = edges_exp1.ravel()
        edges_exp[1] = edges_exp2.ravel()

        np.testing.assert_array_equal(edges, edges_exp)

    def test_inpaint_nearest(self):

        lmax = 6
        minfo = map_utils.get_gauss_minfo(lmax)
        mask = np.asarray([[0, 1, 1, 0, 1, 1, 0],
                           [1, 1, 1, 0, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        mask = mask.reshape(-1)

        imap = np.asarray([[0, 2, 3, 0, 4, 5, 0],
                           [10, 1, 3, 0, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0]])
        imap = imap.reshape(-1)

        omap = map_utils.inpaint_nearest(imap, mask, minfo)

        omap_exp = np.asarray([[2, 2, 3, 3, 4, 5, 5],
                               [10, 1, 3, 3, 1, 1, 1],
                               [10, 1, 3, 1, 1, 1, 1],
                               [10, 1, 3, 1, 1, 1, 1]])
        omap_exp = omap_exp.reshape(-1)

        np.testing.assert_array_equal(omap, omap_exp)

    def test_inpaint_nearest_2d(self):

        lmax = 6
        minfo = map_utils.get_gauss_minfo(lmax)

        mask1 = np.asarray([[0, 1, 1, 0, 1, 1, 0],
                            [1, 1, 1, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        mask2 = np.asarray([[1, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 1, 1, 1],
                            [0, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0]], dtype=bool)

        mask = np.zeros((2, minfo.npix), dtype=bool)
        mask[0] = mask1.ravel()
        mask[1] = mask2.ravel()

        imap1 = np.asarray([[0, 2, 3, 0, 4, 5, 0],
                            [10, 1, 3, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0]])
        imap2 = np.asarray([[2, 2, 3, 0, 4, 5, 0],
                            [0, 1, 3, 0, 1, 1, 1],
                            [0, 0, 6, 0, 1, 1, 1],
                            [5, 0, 0, 0, 0, 0, 0]])

        imap = np.zeros((2, minfo.npix))
        imap[0] = imap1.reshape(-1)
        imap[1] = imap2.reshape(-1)

        omap = map_utils.inpaint_nearest(imap, mask, minfo)

        omap_exp1 = np.asarray([[2, 2, 3, 3, 4, 5, 5],
                               [10, 1, 3, 3, 1, 1, 1],
                               [10, 1, 3, 1, 1, 1, 1],
                               [10, 1, 3, 1, 1, 1, 1]])
        omap_exp2 = np.asarray([[2, 2, 3, 3, 4, 5, 5],
                               [2, 1, 3, 3, 1, 1, 1],
                               [5, 1, 6, 6, 1, 1, 1],
                               [5, 5, 6, 6, 1, 1, 1]])
        omap_exp = np.zeros((2, minfo.npix))
        omap_exp[0] = omap_exp1.reshape(-1)
        omap_exp[1] = omap_exp2.reshape(-1)

        np.testing.assert_array_equal(omap, omap_exp)

    def test_view_2d(self):

        lmax = 6
        minfo = map_utils.get_gauss_minfo(lmax)

        imap = np.zeros((2, 1, 2, minfo.npix))

        omap = map_utils.view_2d(imap, minfo)
        self.assertTrue(omap.shape == (2, 1, 2, minfo.nrow, minfo.nphi[0]))
        self.assertTrue(np.shares_memory(omap, imap))

    def test_view_1d(self):

        lmax = 6
        minfo = map_utils.get_gauss_minfo(lmax)

        imap = np.zeros((2, 1, 2, minfo.nrow, minfo.nphi[0]))

        omap = map_utils.view_1d(imap, minfo)
        self.assertTrue(omap.shape == (2, 1, 2, minfo.npix))
        self.assertTrue(np.shares_memory(omap, imap))

        # Transposed input cannot be flattened without copy.
        imap = np.zeros((minfo.nrow, minfo.nphi[0], 2))
        imap.reshape(2, minfo.nrow, minfo.nphi[0])
        self.assertRaises(ValueError, map_utils.view_1d, imap, minfo)

    def test_equal_area_gauss_copy_2d(self):

        lmax = 9
        minfo = map_utils.get_equal_area_gauss_minfo(lmax)

        imap = np.ones((2, minfo.npix), dtype=bool)

        imap[0,map_utils.get_ring_slice(0, minfo)] = [False, False, True, True]
        imap[0,map_utils.get_ring_slice(2, minfo)] = False
        imap[0,map_utils.get_ring_slice(4, minfo)] = False

        omap = map_utils.equal_area_gauss_copy_2d(imap, minfo)

        np.testing.assert_array_equal(
            omap[0,0], np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool))
        np.testing.assert_array_equal(
            omap[0,1], np.ones(10, dtype=bool))
        np.testing.assert_array_equal(
            omap[0,2], np.zeros(10, dtype=bool))
        np.testing.assert_array_equal(
            omap[0,3], np.ones(10, dtype=bool))
        np.testing.assert_array_equal(
            omap[0,4], np.zeros(10, dtype=bool))

        np.testing.assert_array_equal(omap[1], np.ones((5, 10), dtype=bool))

        # Test if normal GL map stays unchanged.
        lmax = 9
        minfo = map_utils.get_gauss_minfo(lmax)

        imap = np.ones((2, minfo.npix), dtype=np.float64)
        imap[:] = np.random.randn(*imap.shape)

        omap = map_utils.equal_area_gauss_copy_2d(imap, minfo)

        np.testing.assert_almost_equal(omap, imap.reshape(2, 5, 10))

    def test_get_ring_slice(self):

        lmax = 9
        minfo = map_utils.get_gauss_minfo(lmax)

        ring_slice = map_utils.get_ring_slice(2, minfo)
        ring_slice_exp = slice(minfo.offsets[2], minfo.offsets[2] + minfo.nphi[2], 1)
        self.assertEqual(ring_slice, ring_slice_exp)

        minfo = map_utils.get_equal_area_gauss_minfo(lmax)
        ring_slice = map_utils.get_ring_slice(2, minfo)
        ring_slice_exp = slice(minfo.offsets[2], minfo.offsets[2] + minfo.nphi[2], 1)
        self.assertEqual(ring_slice, ring_slice_exp)

        self.assertRaises(IndexError, map_utils.get_ring_slice, 10, minfo)

    def test_gauss2gauss(self):

        lmax = 20
        minfo = map_utils.get_gauss_minfo(lmax)

        imap = np.ones((2, minfo.nrow, minfo.nphi[0]))
        imap *= np.cos(minfo.theta[:,np.newaxis])
        imap = imap.reshape(2, minfo.npix)

        lmax_out = 15
        minfo_out = map_utils.get_gauss_minfo(lmax_out)

        omap = map_utils.gauss2gauss(imap, minfo, minfo_out, order=3)

        omap_exp = np.ones((2, minfo_out.nrow, minfo_out.nphi[0]))
        omap_exp *= np.cos(minfo_out.theta[:,np.newaxis])
        omap_exp = omap_exp.reshape(2, minfo_out.npix)

        np.testing.assert_allclose(omap, omap_exp, rtol=1e-3)

    def test_gauss2gauss_area_pow(self):

        area_pow = 1
        lmax = 6
        minfo = map_utils.get_gauss_minfo(lmax)

        # Scale input by area, should cancel in output.
        imap = np.ones((2, minfo.nrow, minfo.nphi[0]))
        imap *= minfo.weight[:,np.newaxis]
        imap = imap.reshape(2, minfo.npix)

        lmax_out = 12
        minfo_out = map_utils.get_gauss_minfo(lmax_out)

        omap = map_utils.gauss2gauss(
            imap, minfo, minfo_out, order=3, area_pow=area_pow)

        omap_exp = np.ones((2, minfo_out.nrow, minfo_out.nphi[0]))
        omap_exp *= minfo_out.weight[:,np.newaxis]
        omap_exp = omap_exp.reshape(2, minfo_out.npix)

        np.testing.assert_allclose(omap, omap_exp)

    def test_gauss2map(self):

        lmax = 20
        minfo = map_utils.get_gauss_minfo(lmax)

        imap = np.ones((2, minfo.nrow, minfo.nphi[0]))
        imap *= np.cos(minfo.theta[:,np.newaxis])
        imap = imap.reshape(2, minfo.npix)

        lmax_out = 15
        minfo_out = map_utils.get_equal_area_gauss_minfo(lmax_out)

        omap = map_utils.gauss2map(imap, minfo, minfo_out, order=3)

        omap_exp = np.ones((2, minfo_out.npix))
        for tidx in range(minfo_out.theta.size):
            ring_slice = map_utils.get_ring_slice(tidx, minfo_out)
            omap_exp[:,ring_slice] *= np.cos(minfo_out.theta[tidx])

        np.testing.assert_allclose(omap, omap_exp, rtol=1e-3)

    def test_gauss2map_area_pow(self):

        area_pow = 1
        lmax = 20
        minfo = map_utils.get_gauss_minfo(lmax)

        # Scale input by area, should cancel in output.
        imap = np.ones((2, minfo.nrow, minfo.nphi[0]))
        imap *= minfo.weight[:,np.newaxis]
        imap = imap.reshape(2, minfo.npix)

        lmax_out = 15
        minfo_out = map_utils.get_equal_area_gauss_minfo(lmax_out)

        omap = map_utils.gauss2map(imap, minfo, minfo_out, order=3,
                                                area_pow=area_pow)

        omap_exp = np.ones((2, minfo_out.npix))
        for tidx in range(minfo_out.theta.size):
            ring_slice = map_utils.get_ring_slice(tidx, minfo_out)
            omap_exp[:,ring_slice] *= minfo_out.weight[tidx]

        np.testing.assert_allclose(omap, omap_exp, rtol=1e-3)

    def test_minfo2lmax(self):

        lmax = 5
        minfo = map_utils.get_gauss_minfo(2 * lmax)

        self.assertEqual(map_utils.minfo2lmax(minfo), lmax)

    def test_minfo_is_equiv(self):

        lmax_1 = 5
        minfo_1 = map_utils.get_gauss_minfo(2 * lmax_1)

        minfo_2 = map_utils.get_gauss_minfo(2 * lmax_1)

        lmax_3 = 4
        minfo_3 = map_utils.get_gauss_minfo(2 * lmax_3)

        self.assertTrue(map_utils.minfo_is_equiv(minfo_1, minfo_1))
        self.assertTrue(map_utils.minfo_is_equiv(minfo_1, minfo_2))
        self.assertFalse(map_utils.minfo_is_equiv(minfo_1, minfo_3))

class TestMapUtilsIO(unittest.TestCase):

    def setUp(self):

        # Get location of this script.
        self.path = pathlib.Path(__file__).parent.absolute()

    def test_read_write_minfo(self):

        lmax = 5
        minfo = map_utils.get_gauss_minfo(lmax)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'minfo')

            map_utils.write_minfo(filename, minfo)

            self.assertTrue(os.path.isfile(filename + '.hdf5'))

            minfo_read = map_utils.read_minfo(filename + '.hdf5')

            self.assertTrue(map_utils.minfo_is_equiv(minfo, minfo_read))

    def test_read_write_map(self):

        lmax = 5
        npol = 2
        minfo = map_utils.get_gauss_minfo(lmax)
        imap = np.ones((npol, minfo.npix), dtype=np.float32)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'testmap')

            map_utils.write_map(filename, imap, minfo)

            self.assertTrue(os.path.isfile(filename + '.hdf5'))

            omap, minfo_read = map_utils.read_map(filename + '.hdf5')

            self.assertTrue(map_utils.minfo_is_equiv(minfo, minfo_read))
            np.testing.assert_allclose(omap, imap)

    def test_read_write_map_symm(self):

        lmax = 5
        npol = 2
        minfo = map_utils.get_gauss_minfo(lmax)
        imap = np.ones((npol, npol, minfo.npix), dtype=np.float32)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'testmap')

            map_utils.write_map(filename, imap, minfo, symm_axes=[0,1])

            self.assertTrue(os.path.isfile(filename + '.hdf5'))

            # Check if only upper-triangular elements were stored.
            with h5py.File(filename + '.hdf5', 'r') as hfile:

                omap_on_disk = hfile['map'][()]
                self.assertEqual(omap_on_disk.shape, (3, minfo.npix))
                self.assertEqual(hfile.attrs['symm_axis'], 0)

            omap, minfo_read = map_utils.read_map(filename + '.hdf5')

            self.assertTrue(map_utils.minfo_is_equiv(minfo, minfo_read))
            np.testing.assert_allclose(omap, imap)

    def test_read_write_map_symm_flat(self):

        lmax = 0
        npol = 2
        ncomp = 3
        minfo = map_utils.get_gauss_minfo(lmax)
        # Create symmetric matrix.
        imap = np.arange(ncomp * npol * ncomp * npol * minfo.npix, dtype=np.float32)
        imap = imap.reshape(ncomp, npol, ncomp, npol, minfo.npix)
        imap += np.transpose(imap, axes=(2, 3, 0, 1, 4))

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'testmap')

            map_utils.write_map(filename, imap, minfo, symm_axes=[[0, 1], [2, 3]])

            self.assertTrue(os.path.isfile(filename + '.hdf5'))

            # Check if only upper-triangular elements were stored.
            with h5py.File(filename + '.hdf5', 'r') as hfile:

                omap_on_disk = hfile['map'][()]
                self.assertEqual(omap_on_disk.shape, (21, minfo.npix))
                self.assertEqual(hfile.attrs['symm_axis'], 0)

            omap, minfo_read = map_utils.read_map(filename + '.hdf5')

            self.assertTrue(map_utils.minfo_is_equiv(minfo, minfo_read))
            np.testing.assert_allclose(omap, imap)

    def test_get_enmap_minfo(self):

        lmax = 180

        # Create cut sky enmap geometry
        ny, nx = 360, 720
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec_cut = np.radians([-60, 30])
        shape, wcs = enmap.band_geometry(dec_cut, res=res, proj='car')

        minfo = map_utils.get_enmap_minfo(shape, wcs, 2 * lmax)

        # Test if all thetas are inside band, +/- 1 degree because of low res.
        self.assertTrue(np.all((
            minfo.theta > np.radians(59)) & (minfo.theta < np.radians(151))))

        # Redo with 10 deg pad
        minfo = map_utils.get_enmap_minfo(shape, wcs, 2 * lmax, pad=np.radians(10))
        self.assertTrue(np.all((
            minfo.theta > np.radians(49)) & (minfo.theta < np.radians(161))))
        self.assertTrue(49 < np.degrees(minfo.theta.min()) < 52)
        self.assertTrue(159 < np.degrees(minfo.theta.max()) < 161)

        # Redo with large pad.
        minfo = map_utils.get_enmap_minfo(shape, wcs, 2 * lmax, pad=np.radians(90))
        self.assertTrue(np.all((
            minfo.theta > np.radians(0)) & (minfo.theta < np.radians(180))))
        self.assertTrue(0 < np.degrees(minfo.theta.min()) < 1)
        self.assertTrue(179 < np.degrees(minfo.theta.max()) < 180)

    def test_get_enmap_minfo_cc(self):

        # Same but now with CC grid.

        lmax = 180

        # Create cut sky enmap geometry
        ny, nx = 360, 720
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec_cut = np.radians([-60, 30])
        shape, wcs = enmap.band_geometry(dec_cut, res=res, proj='car')

        minfo = map_utils.get_enmap_minfo(shape, wcs, 2 * lmax, mtype='CC')

        # Test if all thetas are inside band, +/- 1 degree because of low res.
        self.assertTrue(np.all((
            minfo.theta > np.radians(59)) & (minfo.theta < np.radians(151))))

        # Redo with 10 deg pad
        minfo = map_utils.get_enmap_minfo(shape, wcs, 2 * lmax, pad=np.radians(10),
                                          mtype='CC')
        self.assertTrue(np.all((
            minfo.theta > np.radians(49)) & (minfo.theta < np.radians(161))))
        self.assertTrue(49 < np.degrees(minfo.theta.min()) < 52)
        self.assertTrue(159 < np.degrees(minfo.theta.max()) < 161)

        # Redo with large pad.
        minfo = map_utils.get_enmap_minfo(shape, wcs, 2 * lmax, pad=np.radians(90),
                                          mtype='CC')
        self.assertTrue(np.all((
            minfo.theta > np.radians(0)) & (minfo.theta < np.radians(180))))
        self.assertTrue(0 < np.degrees(minfo.theta.min()) < 1)
        self.assertTrue(179 < np.degrees(minfo.theta.max()) < 180)

    def test_match_enmap_minfo(self):

        # Check if we can just feed the enmap to libsharp without interpolation.
        # And without having to copy the data like pixell does.

        # Full sky CC CAR map first.
        ny, nx = 37, 72
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        #shape, wcs = enmap.band_geometry(dec_cut, res=res, proj='car')

        shape, wcs = enmap.fullsky_geometry(res, shape=(ny, nx))
        minfo = map_utils.match_enmap_minfo(shape, wcs)

        lmax = int(np.floor((ny - 1) / 2))
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

    def test_match_enmap_minfo_f1(self):

        # Full sky f1 CAR map first.
        ny, nx = 37, 72
        res = [np.pi / (ny - 0), 2 * np.pi / nx]

        shape, wcs = enmap.fullsky_geometry(res, shape=(ny, nx),
                                            variant="fejer1")
        minfo = map_utils.match_enmap_minfo(shape, wcs, mtype='fejer1')

        lmax = int(np.floor((ny - 1) / 2))
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

    def test_match_enmap_minfo_cutsky(self):

        # Same but with cut-sky map. Can only do alm2map test now.
        ny, nx = 37, 72
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec_cut = np.radians([-60, 30])
        shape, wcs = enmap.band_geometry(dec_cut, res=res, proj='car')

        minfo = map_utils.match_enmap_minfo(shape, wcs)

        lmax = int(np.floor((ny - 1) / 2))
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

    def test_match_enmap_minfo_cutsky_f1(self):

        # Same but with cut-sky map. Can only do alm2map test now.
        ny, nx = 37, 72
        res = [np.pi / (ny - 0), 2 * np.pi / nx]
        dec_cut = np.radians([-60, 30])
        shape, wcs = enmap.band_geometry(
            dec_cut, res=res, proj='car', variant='fejer1')

        minfo = map_utils.match_enmap_minfo(shape, wcs, mtype='fejer1')

        lmax = int(np.floor((ny - 1) / 2))
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

    def test_match_enmap_minfo_variations(self):

        # Check all the cdelt variations.
        ny, nx = 37, 72
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res, shape=(ny, nx))

        lmax = int(np.floor((ny - 1) / 2))
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        # ************
        # cdelt_x * -1
        # ************
        wcs.wcs.cdelt[0] *= -1
        minfo = map_utils.match_enmap_minfo(shape, wcs)

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

        # ************
        # cdelt_y * -1
        # ************
        wcs.wcs.cdelt[0] *= -1
        wcs.wcs.cdelt[1] *= -1

        minfo = map_utils.match_enmap_minfo(shape, wcs)

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

        # **************************
        # cdelt_x * -1, cdelt_y * -1
        # **************************
        wcs.wcs.cdelt[0] *= -1

        minfo = map_utils.match_enmap_minfo(shape, wcs)

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

    def test_match_enmap_minfo_variations_f1(self):

        # Check all the cdelt variations.
        ny, nx = 37, 72
        res = [np.pi / (ny - 0), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res, shape=(ny, nx),
                                            variant='fejer1')

        lmax = int(np.floor((ny - 1) / 2))
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        # ************
        # cdelt_x * -1
        # ************
        wcs.wcs.cdelt[0] *= -1
        minfo = map_utils.match_enmap_minfo(shape, wcs, mtype='fejer1')

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

        # ************
        # cdelt_y * -1
        # ************
        wcs.wcs.cdelt[0] *= -1
        wcs.wcs.cdelt[1] *= -1

        minfo = map_utils.match_enmap_minfo(shape, wcs, mtype='fejer1')

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

        # **************************
        # cdelt_x * -1, cdelt_y * -1
        # **************************
        wcs.wcs.cdelt[0] *= -1

        minfo = map_utils.match_enmap_minfo(shape, wcs, mtype='fejer1')

        omap = enmap.zeros((1,) + shape, wcs).reshape(1, minfo.npix)
        omap_exp = enmap.zeros((1,) + shape, wcs)

        # Compare alm2map to pixell.
        sht.alm2map(alm, omap, ainfo, minfo, 0)
        curvedsky.alm2map(alm, omap_exp, ainfo=ainfo)

        np.testing.assert_allclose(omap.reshape(*omap_exp.shape), omap_exp)

        # Compare to input alm.
        alm_out = alm.copy()
        sht.map2alm(omap, alm_out, minfo, ainfo, 0)
        np.testing.assert_allclose(alm_out, alm)

    def test_minfo2wcs(self):

        lmax = 10
        minfo = map_utils.get_cc_minfo(lmax)
        wcs = map_utils.minfo2wcs(minfo)

        self.assertAlmostEqual(wcs.wcs.cdelt[0] * (lmax + 1), -360.)
        self.assertAlmostEqual(wcs.wcs.cdelt[1] * (lmax + 1), -180.)

        # Test if GL map raises error.
        minfo_gl = map_utils.get_gauss_minfo(lmax)
        self.assertRaises(ValueError, map_utils.minfo2wcs, minfo_gl)

    def test_fmul_pix(self):

        lmax = 10
        minfo = map_utils.get_cc_minfo(lmax)
        imap = np.ones((2, minfo.npix))

        ny = minfo.nrow
        nx = minfo.nphi[0]

        fmat2d = np.full((ny, nx // 2 + 1), 2, dtype=np.complex128)
        omap = map_utils.fmul_pix(imap, minfo, fmat2d)

        omap_exp = imap * 2
        np.testing.assert_allclose(omap, omap_exp)
