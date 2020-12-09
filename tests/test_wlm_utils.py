import unittest
import numpy as np

from optweight import wlm_utils

class TestWlmUtils(unittest.TestCase):

    def test_schwarz(self):

        ts = np.linspace(-2, 2, num=10)
        
        s_out = wlm_utils.schwarz(ts)
        
        s_exp = np.zeros_like(ts)
        mask = (ts >= -1) & (ts <= 1)
        s_exp[(ts > -1) & (ts < 1)] = np.exp(-1 / (1 - ts[mask] ** 2))
        
        np.testing.assert_array_almost_equal(s_out, s_exp)

    def test_schwarz_overflow(self):

        ts = np.asarray([-1e40, 1e40])        
        s_out = wlm_utils.schwarz(ts)
        
        np.testing.assert_array_almost_equal(s_out, np.zeros_like(ts))

    def test_s_lambda(self):

        lamb = 3
        ts = np.asarray([0, 0.1, 1 / lamb, 0.5, 1, 2])

        out = wlm_utils.s_lambda(ts, lamb)

        # Out should be zero for ts <= 1 / lambda
        # Out should be zero for ts >= 1.

        self.assertAlmostEqual(out[0], 0)
        self.assertAlmostEqual(out[1], 0)
        self.assertAlmostEqual(out[2], 0)
        # Schwarz(- 0.5)
        self.assertAlmostEqual(out[3], np.exp(-1 / (1 - 0.5 ** 2)))
        self.assertAlmostEqual(out[4], 0)
                
    def test_s_lambda_err(self):
        
        lamb = -1
        ts = np.zeros(1)
        self.assertRaises(ValueError, wlm_utils.s_lambda, ts, lamb)

    def test_k_lambda(self):
        
        lamb = 3
        ts = np.asarray([0, 0.1, 1 / lamb, 0.5, 1, 2])

        out = wlm_utils.k_lambda(ts, lamb)
        
        # One for t < 1 / lambda, zero for t > 1.
        self.assertAlmostEqual(out[0], 1)
        self.assertAlmostEqual(out[1], 1)
        self.assertAlmostEqual(out[2], 1)
        # Schwarz(- 0.5)
        self.assertTrue(out[3] < 1)
        self.assertTrue(out[3] > 0)
        self.assertAlmostEqual(out[4], 0)

    def test_phi_ell(self):
        
        ells = np.arange(20)
        j0_scale = 2
        lamb = 2

        # We want support in [0, lambda^j0] = [0, 4].
        phi_ell = wlm_utils.phi_ell(ells, lamb, j0_scale)
        
        self.assertAlmostEqual(phi_ell[0], 1)
        self.assertAlmostEqual(phi_ell[1], 1)
        self.assertTrue(np.all(phi_ell[ells < 4] > 0))
        self.assertTrue(np.all(phi_ell[ells > 4] == 0))

    def test_phi_ell_j0_0(self):
        
        lamb = 3
        lmax = 30
        j0_scale = 0
        ells = np.arange(lmax + 1)

        # We want support in [0, lambda^j0] = [0, 1].
        phi_ell = wlm_utils.phi_ell(ells, lamb, j0_scale)
        
        self.assertAlmostEqual(phi_ell[0], 1)
        self.assertTrue(np.all(phi_ell[1:] == 0))

    def test_psi_ell(self):
        
        ells = np.arange(20)
        j_scale = 2
        lamb = 2

        # We want support in [lambda^(j-1), lambda^(j+1)] = [2, 8].
        psi_ell = wlm_utils.psi_ell(ells, lamb, j_scale)
        
        self.assertTrue(np.all(psi_ell[ells < 2] == 0))
        self.assertTrue(np.all(psi_ell[ells > 8] == 0))
        self.assertTrue(np.all(psi_ell[(ells >= 2) & (ells <= 8)] >= 0))

    def test_lmax_to_j_scale(self):
        
        lamb = 3
        lmax = 10
        
        # We expect ceil(log3(lmax)) = 3.
        j_scale_exp = 3
        j_scale = wlm_utils.lmax_to_j_scale(lmax, lamb)

        self.assertEqual(j_scale, j_scale_exp)
        
    def test_j_scale_to_lmax(self):        
        
        lamb = 3
        j_scale = 3

        lmax = wlm_utils.j_scale_to_lmax(j_scale, lamb)
        
        # We do not expect 10 from above test, but instead 
        # 3^(3 + 1) = 81
        lmax_exp = 81
        self.assertEqual(lmax, lmax_exp)

    def test_j_scale_to_lmax_float(self):        
        
        lamb = 1.3
        j_scale = 3

        lmax = wlm_utils.j_scale_to_lmax(j_scale, lamb)
        
        # We expect rounded up: so 1.3^(3 + 1) = 3.
        lmax_exp = 3
        self.assertEqual(lmax, lmax_exp)

    def test_get_sd_kernels(self):

        lamb = 3
        lmax = 30

        w_ell, lmaxs = wlm_utils.get_sd_kernels(lamb, lmax)

        self.assertEqual(w_ell.shape, (5, lmax + 1))

        # See Fig. 1 in (1211.1680). 
        lmaxs_exp = np.asarray([3, 9, 27, lmax, lmax])        
        np.testing.assert_array_equal(lmaxs, lmaxs_exp)
        
        # Check for Eq. 9 in that same paper.
        np.testing.assert_array_almost_equal(np.sum(w_ell ** 2, axis=0),
                                             np.ones(lmax + 1))

    def test_get_sd_kernels_accuracy(self):

        lamb = 3
        lmax = 6000

        w_ell, lmaxs = wlm_utils.get_sd_kernels(lamb, lmax)
        
        np.testing.assert_array_almost_equal(np.sum(w_ell ** 2, axis=0),
                                             np.ones(lmax + 1), decimal=10)

    def test_get_sd_kernels_lmin(self):

        lamb = 3
        lmax = 30
        lmin = 4

        w_ell, lmaxs = wlm_utils.get_sd_kernels(lamb, lmax, lmin=lmin)

        self.assertEqual(w_ell.shape, (4, lmax + 1))

        # See Fig. 1 in (1211.1680). 
        lmaxs_exp = np.asarray([9, 27, lmax, lmax])        
        np.testing.assert_array_equal(lmaxs, lmaxs_exp)
        
        # Check for Eq. 9 in that same paper.
        np.testing.assert_array_almost_equal(np.sum(w_ell ** 2, axis=0),
                                             np.ones(lmax + 1))

    def test_get_sd_kernels_j0(self):

        lamb = 3
        lmax = 30
        j0 = 1

        w_ell, lmaxs = wlm_utils.get_sd_kernels(lamb, lmax, j0=j0)

        self.assertEqual(w_ell.shape, (4, lmax + 1))

        # See Fig. 1 in (1211.1680). 
        lmaxs_exp = np.asarray([9, 27, lmax, lmax])        
        np.testing.assert_array_equal(lmaxs, lmaxs_exp)
        
        # Check for Eq. 9 in that same paper.
        np.testing.assert_array_almost_equal(np.sum(w_ell ** 2, axis=0),
                                             np.ones(lmax + 1))

    def test_get_sd_kernels_err(self):

        lamb = 3
        lmax = 30
        j0 = 1
        lmin = 10

        # Cannot have both lmin and j0.
        self.assertRaises(
            ValueError, wlm_utils.get_sd_kernels, lamb, lmax, lmin=10, j0=j0)

        # Cannot have negative lmin.
        self.assertRaises(
            ValueError, wlm_utils.get_sd_kernels, lamb, lmax, lmin=-10)

        # Cannot have negative j0.
        self.assertRaises(
            ValueError, wlm_utils.get_sd_kernels, lamb, lmax, j0=-10)

        # Cannot have negative j0 > jmax.
        self.assertRaises(
            ValueError, wlm_utils.get_sd_kernels, lamb, lmax, j0=10)
    
    def test_get_sd_kernels_return_j(self):

        lamb = 3
        lmax = 30
        lmin = 4

        w_ell, lmaxs, j_scales = wlm_utils.get_sd_kernels(
            lamb, lmax, lmin=lmin, return_j=True)

        self.assertEqual(w_ell.shape, (4, lmax + 1))

        # See Fig. 1 in (1211.1680). 
        j_scales_exp = np.asarray([1, 2, 3, 4])
        np.testing.assert_array_equal(j_scales, j_scales_exp)
