import unittest
import numpy as np

from pixell import enmap, wcsutils

from optweight import noisebox_utils

class TestNoiseBoxUtils(unittest.TestCase):

    def test_noisebox2wavmat(self):
        
        npol = 1
        nbins = 4
        ny = 20
        nx = 50

        shape, wcs = enmap.geometry([0, 0], res=0.1, shape=(ny, nx))
        noisebox = enmap.ones((npol, nbins, ny, nx), wcs=wcs)
        bins = np.asarray([10, 20, 30, 40])
        lmax = 19
        w_ell = np.zeros((3, lmax+1))
        w_ell[0,:2] = 1
        w_ell[1,2:5] = 1
        w_ell[2,5:] = 1

        icov_wav = noisebox_utils.noisebox2wavmat(
            noisebox, bins, w_ell, offsets=[-1, 0, 1])

        # Check if I get the right amount of maps.
        indices_exp = np.zeros((7, 2))
        indices_exp[0] = [0,0]
        indices_exp[1] = [0,1]
        indices_exp[2] = [1,0]
        indices_exp[3] = [1,1]
        indices_exp[4] = [1,2]
        indices_exp[5] = [2,1]
        indices_exp[6] = [2,2]

        np.testing.assert_array_equal(icov_wav.indices, indices_exp)

        # Check if minfos of maps make sense.
        lmaxs = np.asarray([1, 4, lmax])
        nphi_exp = 2 * lmaxs + 1
        self.assertEqual(icov_wav.minfos[0,0].nphi[0], nphi_exp[0])
        self.assertEqual(icov_wav.minfos[0,1].nphi[0], nphi_exp[1])
        self.assertEqual(icov_wav.minfos[1,0].nphi[0], nphi_exp[0])
        self.assertEqual(icov_wav.minfos[1,1].nphi[0], nphi_exp[1])
        self.assertEqual(icov_wav.minfos[1,2].nphi[0], nphi_exp[2])
        self.assertEqual(icov_wav.minfos[2,1].nphi[0], nphi_exp[1])
        self.assertEqual(icov_wav.minfos[2,2].nphi[0], nphi_exp[2])

        # Check if maps make sense. 
        # I start with icov power spectra of 1 / (uk ^ 2 armcin ^ 2).
        # These are converted to uk ^ -2.
        # These are then converted to uk ^ -2 per pixel. 
        # Finally, these are converted to the Gauss-Legendre map.
        
        pix_area_enmap = enmap.pixsizemap(shape, wcs)

        icov_exp = np.ones_like(pix_area_enmap)
        # To go to uK ^ -2:
        icov_exp = 1 * (10800) ** 2 / 4 / np.pi 
        # To go to uK ^ -2 per pixel:
        # Since the icov power spectrum in this test is flat,
        # the multiplication with the wavelet kernel and the 
        # normalization cancel and thus we only have to scale
        # by the pixel area.
        icov_exp *= pix_area_enmap 
        
        # Then there comes another correction of area_GL / area_enmap, which is 
        # hard to test without rewriting the function again. So I'll divide
        # the expected map and the real map by their pixel size, those then
        # should be **approximately** equal.

        map_ans = icov_wav.maps[2,2].reshape(
            1, icov_wav.minfos[2,2].nrow, icov_wav.minfos[2,2].nphi[0])
        map_ans /= icov_wav.minfos[2,2].weight[np.newaxis,:,np.newaxis]

        icov_exp /= pix_area_enmap

        np.testing.assert_allclose(map_ans[0,10,0], icov_exp[10,0], rtol=0.005)
                
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



