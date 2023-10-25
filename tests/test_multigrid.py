import unittest
import numpy as np

import healpy as hp
from pixell import curvedsky

from optweight import multigrid
from optweight import map_utils, sht, alm_c_utils, alm_utils, mat_utils

class TestMultiGrid(unittest.TestCase):

    def test_lowpass_filter(self):

        lmax = 10
        r_ell = multigrid.lowpass_filter(lmax)

        self.assertAlmostEqual(r_ell[0], 1)
        self.assertAlmostEqual(r_ell[lmax // 2] ** 2, 0.05)

    def test_get_levels(self):

        lmax = 22
        ells = np.arange(lmax + 1)
        npol = 3
        spin = [0, 2]
        icov_ell = np.ones((npol, npol, lmax+1))
        icov_ell *= np.asarray([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])[:,:,np.newaxis]
        # Approximate icov_ell.
        icov_ell *= 1e-4 * (ells + 1) ** 2
        icov_ell *= hp.gauss_beam(2 * np.pi / lmax, lmax)
        icov_ell[:,:,:2] = 0

        minfo = map_utils.get_gauss_minfo(2 * lmax)

        # Mask north and south cap.
        mask = np.ones((minfo.npix), dtype=bool)
        mask_2d = map_utils.view_2d(mask, minfo)
        mask_2d[0:5,:] = False
        mask_2d[21:,:] = False

        min_pix = 10
        levels = multigrid.get_levels(mask, minfo, icov_ell, spin, min_pix=min_pix,
                                      lmax_r_ell=lmax)

        self.assertEqual(len(levels), 3)

        self.assertTrue(np.any(np.sum(levels[2].mask_unobs, axis=1) <= min_pix))

        self.assertEqual(levels[0].mask_unobs.shape, (npol, levels[0].minfo.npix))
        self.assertEqual(levels[1].mask_unobs.shape, (npol, levels[1].minfo.npix))
        self.assertEqual(levels[2].mask_unobs.shape, (npol, levels[2].minfo.npix))

        self.assertEqual(levels[0].mask_unobs.dtype, bool)
        self.assertEqual(levels[1].mask_unobs.dtype, bool)
        self.assertEqual(levels[2].mask_unobs.dtype, bool)

        self.assertEqual(map_utils.minfo2lmax(levels[0].minfo), lmax)
        self.assertEqual(map_utils.minfo2lmax(levels[1].minfo), lmax // 2)
        self.assertEqual(map_utils.minfo2lmax(levels[2].minfo), lmax // 4)

        # Test the smoothers on each level.
        map0 = levels[0].smoother(np.ones((3, levels[0].minfo.npix)))
        map1 = levels[1].smoother(np.ones((3, levels[1].minfo.npix)))
        map2 = levels[2].smoother(np.ones((3, levels[2].minfo.npix)))

        # First 2 smoothers are constant scalings.
        np.testing.assert_allclose(map0 / map0[:,0:1], np.ones((3, map0.shape[-1])))
        np.testing.assert_allclose(map1 / map1[:,0:1], np.ones((3, map1.shape[-1])))
        self.assertEqual(levels[0].pinv_g, None)
        self.assertEqual(levels[1].pinv_g, None)

        # Last smoother is dense.
        # Test if pseudo inverse of G matrix is acting as inverse of G.
        cov_ell = mat_utils.matpow(icov_ell, -1)
        alm, ainfo = curvedsky.rand_alm(cov_ell[:,:,:levels[2].lmax+1], return_ainfo=True)
        test_map_in = np.ones((npol, levels[2].minfo.npix))
        sht.alm2map(alm, test_map_in, ainfo, levels[2].minfo, [0, 2])
        test_map_transformed = test_map_in * levels[2].mask_unobs
        test_map_transformed = levels[2].g_op(test_map_transformed)
        # Not very accurate, condition number of pinv G is large.
        test_map_transformed = np.dot(
            levels[2].pinv_g, test_map_transformed[levels[2].mask_unobs])
        np.testing.assert_allclose(test_map_transformed, test_map_in[levels[2].mask_unobs],
                                   rtol=1e-3)

    def test_get_levels_err(self):

        lmax = 50
        npol = 3
        spin = [0, 2]
        icov_ell = np.ones((npol, npol, lmax+1))
        icov_ell *= np.asarray([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])[:,:,np.newaxis]

        minfo = map_utils.get_gauss_minfo(2 * lmax)

        # Wrong dtype.
        mask = np.ones((minfo.npix), dtype=float)
        self.assertRaises(ValueError, multigrid.get_levels, mask, minfo, icov_ell, spin)

        # Wrong shape.
        mask = np.ones((minfo.npix), dtype=bool)
        icov_ell = icov_ell[0,0]
        self.assertRaises(ValueError, multigrid.get_levels, mask, minfo, icov_ell, spin)

    def test_coarsen(self):

        lmax = 5
        npol = 3
        spin = [0, 2]
        ells = np.arange(lmax + 1)

        # Create 2 masks.
        minfo_in = map_utils.get_gauss_minfo(2 * lmax)
        mask_in = np.ones((3, minfo_in.npix), dtype=bool)
        mask_in[0,np.random.randint(0, minfo_in.npix, size=50)] = False
        mask_in[1,np.random.randint(0, minfo_in.npix, size=50)] = False
        mask_in[2,np.random.randint(0, minfo_in.npix, size=50)] = False

        minfo_out = map_utils.get_gauss_minfo(2 * lmax // 2)
        mask_out = map_utils.gauss2gauss(
            mask_in.astype(np.float32), minfo_in, minfo_out, order=1)
        mask_out[(mask_out > 0) & (mask_out < 1)] = 1. 
        mask_out = mask_out.astype(bool)

        # Create map.
        cov_ell = np.ones((npol, npol, lmax+1))
        cov_ell *= np.asarray([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])[:,:,np.newaxis]
        cov_ell[:,:,1:] *= ells[1:] ** -2.
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)
        imap = np.ones((npol, minfo_in.npix))
        sht.alm2map(alm, imap, ainfo, minfo_in, [0, 2])

        # Create 2 levels.
        level_in = multigrid.Level(mask_in, minfo_in, cov_ell, spin)
        lmax_out = map_utils.minfo2lmax(minfo_out)
        level_out = multigrid.Level(mask_out, minfo_out,
                                    cov_ell[:,:,:lmax_out+1], spin)
        omap = multigrid.coarsen(imap, level_in, level_out, spin)

        # Do manual coarsen in a way that keeps the alm at full size.
        omap_exp = imap * ~mask_in
        alm_tmp = alm.copy()

        sht.map2alm(omap_exp, alm_tmp, minfo_in, ainfo, [0, 2])
        r_ell = hp.gauss_beam(2 * np.pi / lmax_out, ainfo.lmax)
        r_ell[lmax_out+1:] = 0
        alm_c_utils.lmul(alm_tmp, r_ell, ainfo, inplace=True)
        omap_exp = np.zeros((3, minfo_out.npix))
        sht.alm2map(alm_tmp, omap_exp, ainfo, minfo_out, [0, 2])
        omap_exp *= ~mask_out

        np.testing.assert_allclose(omap, omap_exp)

    def test_interpolate(self):

        lmax = 5        
        npol = 3
        spin = [0, 2]
        ells = np.arange(lmax + 1)

        # Create 2 masks.
        minfo_out = map_utils.get_gauss_minfo(2 * lmax)
        mask_out = np.ones((3, minfo_out.npix), dtype=bool)

        # Large mask to make sure that there are still masked pixel in downgraded version.
        mask_out[0,int(0.2 * minfo_out.npix):int(0.8 * minfo_out.npix)] = False
        mask_out[1,int(0.3 * minfo_out.npix):int(0.9 * minfo_out.npix)] = False
        mask_out[2,int(0.4 * minfo_out.npix):int(1 * minfo_out.npix)] = False        
        
        minfo_in = map_utils.get_gauss_minfo(2 * lmax / 2)
        mask_in = map_utils.gauss2gauss(
            mask_out.astype(np.float32), minfo_out, minfo_in, order=1)
        mask_in[(mask_in > 0) & (mask_in < 1)] = 1. 
        mask_in = mask_in.astype(bool)

        lmax_in = map_utils.minfo2lmax(minfo_in)

        # Create map.
        cov_ell = np.ones((npol, npol, lmax+1))
        cov_ell *= np.asarray([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])[:,:,np.newaxis]
        cov_ell[:,:,1:] *= ells[1:] ** -2.
        alm, ainfo = curvedsky.rand_alm(cov_ell[:,:,:lmax_in+1], return_ainfo=True)
        imap = np.ones((npol, minfo_in.npix))
        sht.alm2map(alm, imap, ainfo, minfo_in, [0, 2])

        # Create 2 levels. Note, switched compared to previous test.
        level_out = multigrid.Level(mask_out, minfo_out, cov_ell, spin)
        level_in = multigrid.Level(mask_in, minfo_in,
                                    cov_ell[:,:,:lmax_in+1], spin)
        omap = multigrid.interpolate(imap, level_in, level_out, spin)

        # Do manual interpolate.
        omap_exp = imap * ~mask_in
        alm_tmp = alm.copy()

        sht.map2alm(omap_exp, alm_tmp, minfo_in, ainfo, [0, 2], adjoint=True)
        r_ell = hp.gauss_beam(2 * np.pi / lmax_in, ainfo.lmax)
        alm_c_utils.lmul(alm_tmp, r_ell, ainfo, inplace=True)
        omap_exp = np.zeros((3, minfo_out.npix))
        sht.alm2map(alm_tmp, omap_exp, ainfo, minfo_out, [0, 2], adjoint=True)
        omap_exp *= ~mask_out

        np.testing.assert_allclose(omap, omap_exp)

    def test_v_cycle(self):

        #lmax = 15
        lmax = 22
        ells = np.arange(lmax + 1)
        npol = 3
        spin = [0, 2]
        icov_ell = np.ones((npol, npol, lmax+1))
        icov_ell *= np.asarray([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])[:,:,np.newaxis]
        # Approximate icov_ell.
        icov_ell *= 1e-4 * (ells + 1) ** 2
        icov_ell[:,:,:2] = 0

        minfo = map_utils.get_gauss_minfo(2 * lmax)

        # Mask strip in middle.
        mask = np.ones((minfo.npix), dtype=bool)
        mask_2d = map_utils.view_2d(mask, minfo)
        mask_2d[8:12,:] = False

        #min_pix = 150
        min_pix = 10
        levels = multigrid.get_levels(mask, minfo, icov_ell, spin, min_pix=min_pix,
                                      lmax_r_ell=lmax)

        # Create map.
        cov_ell = mat_utils.matpow(icov_ell, -1)
        # We only expect solver to do well with large scales.
        cov_ell[:,:,5:] = 0
        alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

        minfo_reduced = levels[0].minfo
        imap = np.ones((npol, minfo_reduced.npix))
        sht.alm2map(alm, imap, ainfo, minfo_reduced, [0, 2])

        # Apply the G operation, let multigrid find approximate inverse.
        imap_g = levels[0].g_op(imap)
        omap = multigrid.v_cycle(levels, imap_g, spin, n_jacobi=3)

        # Lots of noisy outliers, so only test median and median absolute deviation.
        diff = (imap[levels[0].mask_unobs] - omap[levels[0].mask_unobs]) / \
              np.abs(imap[levels[0].mask_unobs]) 
        self.assertLess(np.abs(np.median(diff)), 0.1)
        mad = np.median(np.abs(diff - np.median(diff)))
        self.assertLess(mad, 0.3)



