import unittest
import numpy as np

from pixell import sharp

from optweight import map_utils

class TestMapUtils(unittest.TestCase):

    def test_get_arc_len(self):
        
        # Theta range [2, 15].
        # Thetas: 3, 5, 10, 14.
        # Arc lengths: 2, 3.5, 4.5, 3.

        # Give unsorted array.
        thetas = np.asarray([10, 3, 5, 14])
        arc_lens = map_utils.get_arc_len(thetas, 2, 15)
        print(np.sort(arc_lens))
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
        icov_pix[:,:,:] /= minfo.weight[np.newaxis,:,np.newaxis]
        icov_pix = icov_pix.reshape((npol, npol, minfo.npix))
        
        itau_exp = np.zeros((npol, npol))
        itau_exp[0,0] = np.sum(icov_pix[0,0] ** 2) / np.sum(icov_pix[0,0])
        itau_exp[1,1] = np.sum(icov_pix[1,1] ** 2) / np.sum(icov_pix[1,1])
        itau_exp[2,2] = np.sum(icov_pix[2,2] ** 2) / np.sum(icov_pix[2,2])

        np.testing.assert_array_almost_equal(itau, itau_exp)
