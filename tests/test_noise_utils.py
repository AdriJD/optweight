import unittest
import numpy as np

from pixell import enmap, wcsutils, curvedsky

from optweight import noise_utils, map_utils, wlm_utils

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

        icov_wav = noise_utils.noisebox2wavmat(
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
        # To go to uK^-2:
        icov_exp = 1 * (10800) ** 2 / 4 / np.pi 
        # To go to uK^-2 per pixel:
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

        np.testing.assert_allclose(map_ans[0,10,0], icov_exp[10,0],
                                   rtol=0.005)
                
    def test_prepare_noisebox(self):

        npol = 3
        nbins = 4
        ny = 2
        nx = 5

        noisebox = enmap.ones((npol, nbins, ny, nx))
        bins = np.asarray([10, 20, 30, 40])
        lmax = 25
        noisebox_out = noise_utils.prepare_noisebox(
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
        self.assertRaises(ValueError, noise_utils.prepare_noisebox,
            noisebox, bins, lmax)

    def test_band_limit_gauss_beam(self):

        lmax = 10
        b_ell = noise_utils._band_limit_gauss_beam(lmax)

        self.assertTrue(b_ell.shape == (lmax + 1,))
        self.assertAlmostEqual(b_ell[-1] / 0.019927032253073726, 1)

    def test_band_limit_gauss_beam_fwhm(self):

        lmax = 10
        fwhm = np.radians(10)
        b_ell = noise_utils._band_limit_gauss_beam(lmax, fwhm=fwhm)

        self.assertTrue(b_ell.shape == (lmax + 1,))
        self.assertAlmostEqual(b_ell[-1] / 0.73923778, 1)

    def test_norm_cov_est(self):

        lmax = 3
        ells = np.arange(lmax + 1)
        npol = 3
        minfo = map_utils.get_gauss_minfo(2 * lmax) 

        kernel_ell = np.ones(lmax + 1)
        kernel_ell[0] = 0

        cov_pix_in = np.ones((npol, npol, minfo.npix))
        cov_pix = noise_utils.norm_cov_est(cov_pix_in, minfo, kernel_ell)

        self.assertFalse(np.shares_memory(cov_pix_in, cov_pix))

        cov_pix_exp = cov_pix_in.reshape((npol, npol, minfo.theta.size, minfo.nphi[0]))
        cov_pix_exp /= minfo.weight[np.newaxis,np.newaxis,:,np.newaxis]
        cov_pix_exp /= np.sum(kernel_ell ** 2 * (2 * ells + 1))
        cov_pix_exp *= 4 * np.pi
        cov_pix_exp = cov_pix_exp.reshape((npol, npol, minfo.npix))

        np.testing.assert_allclose(cov_pix, cov_pix_exp)        

    def test_norm_cov_est_inplace(self):

        lmax = 3
        ells = np.arange(lmax + 1)
        npol = 3
        minfo = map_utils.get_gauss_minfo(2 * lmax) 

        kernel_ell = np.ones(lmax + 1)
        kernel_ell[0] = 0

        cov_pix_in = np.ones((npol, npol, minfo.npix))

        cov_pix_exp = cov_pix_in.copy().reshape((npol, npol, minfo.theta.size, minfo.nphi[0]))
        cov_pix_exp /= minfo.weight[np.newaxis,np.newaxis,:,np.newaxis]
        cov_pix_exp /= np.sum(kernel_ell ** 2 * (2 * ells + 1))
        cov_pix_exp *= 4 * np.pi
        cov_pix_exp = cov_pix_exp.reshape((npol, npol, minfo.npix))

        cov_pix = noise_utils.norm_cov_est(cov_pix_in, minfo, kernel_ell, inplace=True)

        self.assertTrue(np.shares_memory(cov_pix_in, cov_pix))
        np.testing.assert_allclose(cov_pix_in, cov_pix_exp)        

    def test_minimum_w_ell_lambda(self):

        lmax = 100
        lmin = 50
        lmax_j = 75

        lamb = noise_utils.minimum_w_ell_lambda(lmax, lmin, lmax_j)

        _, _, js = wlm_utils.get_sd_kernels(lamb, lmax, lmin=lmin,
                                            lmax_j=lmax_j, return_j=True) 

        self.assertTrue(len(js) == 3)
            
    def test_minimum_w_ell_lambda_err(self):

        lmax = 100
        lmin = 50
        lmax_j = 51

        self.assertRaises(ValueError, noise_utils.minimum_w_ell_lambda, lmax, lmin, lmax_j)        
    
    def test_unit_var_wav(self):

        minfos = np.asarray([map_utils.MapInfo.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [3, 3, 4]])
        preshape = (2, 3)
        dtype = np.float32

        wav_unit = noise_utils.unit_var_wav(minfos, preshape, dtype)
        map_0 = wav_unit.maps[0]
        map_1 = wav_unit.maps[1]
        map_2 = wav_unit.maps[2]

        self.assertEqual(wav_unit.dtype, dtype)
        self.assertEqual(wav_unit.preshape, preshape)
        self.assertEqual(wav_unit.ndim, 1)
        self.assertEqual(wav_unit.shape, (3,))

        # Running againg with no seed should give different maps.
        wav_unit = noise_utils.unit_var_wav(minfos, preshape, dtype)        
        self.assertRaises(AssertionError, np.testing.assert_allclose,
                          wav_unit.maps[0], map_0)
        self.assertRaises(AssertionError, np.testing.assert_allclose,
                          wav_unit.maps[1], map_1)
        self.assertRaises(AssertionError, np.testing.assert_allclose,
                          wav_unit.maps[2], map_2)

        
        wav_unit = noise_utils.unit_var_wav(minfos, preshape, dtype, seed=1)        
        map_0 = wav_unit.maps[0]
        map_1 = wav_unit.maps[1]
        map_2 = wav_unit.maps[2]

        # Running againg with same seed should give same maps.
        wav_unit = noise_utils.unit_var_wav(minfos, preshape, dtype, seed=1)        
        np.testing.assert_allclose(wav_unit.maps[0], map_0)
        np.testing.assert_allclose(wav_unit.maps[1], map_1)
        np.testing.assert_allclose(wav_unit.maps[2], map_2)

        rng = np.random.default_rng(1)
        wav_unit = noise_utils.unit_var_wav(minfos, preshape, dtype, seed=rng)        
        map_0 = wav_unit.maps[0]
        map_1 = wav_unit.maps[1]
        map_2 = wav_unit.maps[2]

        # Running againg with same generator should five different maps.
        wav_unit = noise_utils.unit_var_wav(minfos, preshape, dtype, seed=rng)        
        self.assertRaises(AssertionError, np.testing.assert_allclose,
                          wav_unit.maps[0], map_0)
        self.assertRaises(AssertionError, np.testing.assert_allclose,
                          wav_unit.maps[1], map_1)
        self.assertRaises(AssertionError, np.testing.assert_allclose,
                          wav_unit.maps[2], map_2)

        # Running againg with fresh generator should give same maps.
        rng = np.random.default_rng(1)
        wav_unit = noise_utils.unit_var_wav(minfos, preshape, dtype, seed=rng)        
        np.testing.assert_allclose(wav_unit.maps[0], map_0)
        np.testing.assert_allclose(wav_unit.maps[1], map_1)
        np.testing.assert_allclose(wav_unit.maps[2], map_2)

    def test_muKarcmin_to_n_ell(self):

        noise_level = 20 # muK arcmin.
        amp_exp = (noise_level * np.pi / 60 / 180) ** 2
        amp = noise_utils.muKarcmin_to_n_ell(noise_level)
        self.assertAlmostEqual(amp_exp / amp, 1)

    def test_get_white_icov_pix(self):

        lmax = 3
        minfo = map_utils.get_gauss_minfo(2 * lmax)
        noise_level = 20 # muK arcmin.
        amp_exp = (noise_level * np.pi / 60 / 180) ** 2
        
        icov_pix = noise_utils.get_white_icov_pix(noise_level, minfo)

        self.assertEqual(icov_pix.shape, (1, minfo.npix))
        self.assertEqual(icov_pix.dtype, np.float64)

        weight_map = np.ones((minfo.nrow, minfo.nphi[0]))
        weight_map *= minfo.weight[:,np.newaxis]
        weight_map = weight_map.reshape(1, minfo.npix)
        
        icov_pix_exp = np.ones((1, minfo.npix)) * weight_map / amp_exp

        np.testing.assert_allclose(icov_pix, icov_pix_exp)

        # Try 2d array.
        noise_level = np.asarray([20, 40, 80])
        icov_pix_exp = np.ones((3, minfo.npix)) * weight_map / amp_exp
        icov_pix_exp /= np.asarray([1, 4, 16])[:,np.newaxis]
        icov_pix = noise_utils.get_white_icov_pix(noise_level, minfo)
        
        np.testing.assert_allclose(icov_pix, icov_pix_exp)

        # Try output array.
        out = np.zeros((3, minfo.npix))
        icov_pix = noise_utils.get_white_icov_pix(noise_level, minfo, out=out)
        
        np.testing.assert_allclose(icov_pix, icov_pix_exp)
        self.assertTrue(np.shares_memory(icov_pix, out))        
        
        self.assertRaises(ValueError, noise_utils.get_white_icov_pix,
                          noise_level, minfo, out=out, dtype=np.float32)
        out = np.ones((3, 3, minfo.npix))
        self.assertRaises(ValueError, noise_utils.get_white_icov_pix,
                          noise_level, minfo, out=out)
        noise_level = np.ones((2, 2))
        self.assertRaises(ValueError, noise_utils.get_white_icov_pix,
                          noise_level, minfo)
        
        
