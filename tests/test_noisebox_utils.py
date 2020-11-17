import unittest
import numpy as np

from pixell import enmap, wcsutils

from optweight import noisebox_utils

class TestNoiseBoxUtils(unittest.TestCase):

    def test_prepare_noisebox(self):

        npol = 3
        nbins = 4
        ny = 2
        nx = 5

        noisebox = enmap.ones((npol, nbins, ny, nx))
        bins = np.asarray([10, 20, 30, 40])
        lmax = 25
        noisebox_out = noisebox_utils.prepare_noisebox(
            noisebox, bins, lmax)
        
        noisebox_exp = enmap.ones(
            (npol, lmax + 1, ny, nx), wcs=noisebox.wcs)
        noisebox_exp *= (10800) ** 2 / 4 / np.pi

        np.testing.assert_array_almost_equal(
            np.asarray(noisebox_out), np.asarray(noisebox_exp))
        self.assertTrue(wcsutils.is_compatible(
            noisebox_out.wcs, noisebox_exp.wcs))

    def test_prepare_noisebox_err(self):

        npol = 3
        nbins = 4
        narray = 2
        ny = 2
        nx = 5

        noisebox = enmap.ones((npol, narray, nbins, ny, nx))
        bins = np.asarray([10, 20, 30, 40])
        lmax = 25
        self.assertRaises(ValueError, noisebox_utils.prepare_noisebox,
            noisebox, bins, lmax)



