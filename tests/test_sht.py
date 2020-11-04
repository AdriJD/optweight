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

        nrings = int(np.floor(2 * lmax / 2)) + 1
        nphi = 2 * lmax + 1

        minfo = sharp.map_info_gauss_legendre(nrings, nphi)
        omap = np.zeros((1, minfo.npix))

        sht.alm2map(alm, omap, ainfo, minfo, spin)
        sht.map2alm(omap, alm_out, minfo, ainfo, spin)

        np.testing.assert_array_almost_equal(alm_out, alm)
