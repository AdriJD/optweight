import unittest
import numpy as np

from pixell import curvedsky, sharp

from optweight import sht

class TestSHT(unittest.TestCase):

    def test_alm2map(self):

        lmax = 4
        spin = 0
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
        alm_out = np.zeros_like(alm)

        nrings = lmax + 1
        nphi = 2 * lmax + 1

        minfo = sharp.map_info_gauss_legendre(nrings, nphi)
        omap = np.zeros((1, minfo.npix))

        sht.alm2map(alm, omap, ainfo, minfo, spin)
        sht.map2alm(omap, alm_out, minfo, ainfo, spin)

        np.testing.assert_array_almost_equal(alm_out, alm)

    def test_map2alm(self):

        lmax = 4
        spin = 0
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        nrings = lmax + 1
        nphi = 2 * lmax + 1

        minfo = sharp.map_info_gauss_legendre(nrings, nphi)
        imap = np.zeros((1, minfo.npix))
        omap = np.zeros_like(imap)

        # Create signal band-limited at lmax.
        sht.alm2map(alm, imap, ainfo, minfo, spin)        
                
        sht.map2alm(imap, alm, minfo, ainfo, spin)
        sht.alm2map(alm, omap, ainfo, minfo, spin)

        np.testing.assert_array_almost_equal(imap, omap)

    def test_alm2map_cut(self):

        lmax = 5
        spin = 0
        cov_ell = np.ones((1, lmax + 1))
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        nrings = lmax + 1
        nphi = 2 * lmax + 1

        minfo_full = sharp.map_info_gauss_legendre(nrings, nphi)

        # Only keep rings in the middle.
        theta_mask = [False, False, True, True, True, False]
        
        theta = minfo_full.theta[theta_mask]
        weight = minfo_full.weight[theta_mask]
        stride = minfo_full.stride[theta_mask]
        offsets = minfo_full.offsets[theta_mask]
        offsets -= offsets[0]

        minfo = sharp.map_info(theta, nphi=nphi, phi0=0, offsets=offsets,
                               stride=stride, weight=weight)

        imap_full = np.zeros((1, minfo_full.npix))
        omap = np.zeros((1, minfo.npix))

        # Create signal band-limited at lmax.
        sht.alm2map(alm, imap_full, ainfo, minfo_full, spin)        

        # Extract nonzero rings from input map.
        omap_exp = np.zeros((1, minfo.npix))
        omap_exp = omap_exp.reshape((1, minfo.nrow, minfo.nphi[0]))
        imap_full = imap_full.reshape((1, minfo_full.nrow, minfo_full.nphi[0]))
        omap_exp[...] = imap_full[:,theta_mask,:]
        omap_exp = omap_exp.reshape(1, minfo.npix)
        
        # alm2map on cut sky.
        sht.alm2map(alm, omap, ainfo, minfo, spin)

        np.testing.assert_array_almost_equal(omap, omap_exp)

    def test_map2alm_err(self):

        lmax = 4
        spin = 0
        ainfo = sharp.alm_info(lmax)
        alm = np.zeros((1, ainfo.nelem), dtype=np.complex128)

        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)
        imap = np.zeros((1, minfo.npix))

        # Wrong spin.
        self.assertRaises(ValueError, sht.map2alm, imap, alm, minfo, ainfo, 5)

        # Wrong npix.
        imap_wrong = np.zeros((1, minfo.npix + 1))
        self.assertRaises(ValueError, sht.map2alm, imap_wrong, alm, minfo, ainfo, spin)

        # Wrong nelem.
        alm_wrong = np.zeros((1, ainfo.nelem + 1), dtype=alm.dtype)
        self.assertRaises(ValueError, sht.map2alm, imap, alm_wrong, minfo, ainfo, spin)

    def test_alm2map_err(self):

        lmax = 4
        spin = 0
        ainfo = sharp.alm_info(lmax)
        alm = np.zeros((1, ainfo.nelem), dtype=np.complex128)

        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)
        omap = np.zeros((1, minfo.npix))

        # Wrong spin.
        self.assertRaises(ValueError, sht.alm2map, alm, omap, ainfo, minfo, 5)

        # Wrong npix.
        omap_wrong = np.zeros((1, minfo.npix + 1))
        self.assertRaises(ValueError, sht.alm2map, alm, omap_wrong, ainfo, minfo, spin)

        # Wrong nelem.
        alm_wrong = np.zeros((1, ainfo.nelem + 1), dtype=alm.dtype)
        self.assertRaises(ValueError, sht.alm2map, alm_wrong, omap, ainfo, minfo, spin)

    def test_default_spin(self):
        
        self.assertEqual(sht.default_spin((10,)), 0)
        self.assertEqual(sht.default_spin((1, 10)), 0)
        self.assertEqual(sht.default_spin((2, 10,)), 2)
        self.assertEqual(sht.default_spin((3, 10,)), [0, 2])                

    def test_default_spin_err(self):
        
        self.assertRaises(ValueError, sht.default_spin, (2, 2, 10))
        self.assertRaises(ValueError, sht.default_spin, (4, 10))
