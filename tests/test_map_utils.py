import unittest
import numpy as np
from scipy.special import roots_legendre

from pixell import sharp, enmap, curvedsky

from optweight import map_utils, sht, wavtrans

class TestMapUtils(unittest.TestCase):

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

    def test_round_icov_matrix_err(self):
        
        npol = 3
        npix = 4

        icov_pix = np.ones((npol, 2, npix))
        self.assertRaises(ValueError, map_utils.round_icov_matrix, icov_pix)

        icov_pix = np.ones((npol, npol, npol, npix))
        self.assertRaises(ValueError, map_utils.round_icov_matrix, icov_pix)

        icov_pix = np.ones((npix))
        self.assertRaises(ValueError, map_utils.round_icov_matrix, icov_pix)
