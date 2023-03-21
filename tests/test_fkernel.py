import unittest
import numpy as np

from pixell import curvedsky

from optweight import fkernel
from optweight import wlm_utils
from optweight import dft

class TestWlmUtils(unittest.TestCase):

    def test_digitize_1d(self):
        
        arr = np.ones(10)
        out_exp = np.ones(arr.size, dtype=bool)
        out = fkernel.digitize_1d(arr)
        np.testing.assert_array_equal(out, out_exp)

        arr = np.zeros(10)
        out_exp = np.zeros(arr.size, dtype=bool)
        out = fkernel.digitize_1d(arr)
        np.testing.assert_array_equal(out, out_exp)
        
        # Test if integral is approx. conserved.
        x = np.linspace(0, np.pi / 3, 2000)                
        arr = np.sin(x)
        arr /= arr.max()
        out = fkernel.digitize_1d(arr)
        
        self.assertAlmostEqual(np.sum(arr), sum(out), delta=sum(arr) * 1e-3)

    def test_digitize_1d_err(self):
        
        arr = np.ones(10) * 2
        self.assertRaises(ValueError, fkernel.digitize_1d, arr)

        arr = np.ones(10) - 1.1
        self.assertRaises(ValueError, fkernel.digitize_1d, arr)

    def test_digitize_kernels(self):
        
        lamb = 3
        lmax = 30

        w_ell, lmaxs = wlm_utils.get_sd_kernels(lamb, lmax)
        d_ell, ells = fkernel.digitize_kernels(w_ell, upsamp_fact=10) 

        self.assertEqual(d_ell.dtype, bool)
        self.assertEqual(d_ell.shape, (w_ell.shape[0], 10 * (lmax + 1)))
        np.testing.assert_allclose(ells, np.linspace(0, lmax+1, 10 * (lmax + 1)))

        # Make sure kernels do not overlap.
        out_exp = np.ones(10 * (lmax + 1))
        np.testing.assert_array_equal(np.sum(d_ell, axis=0), out_exp)

    def test_digitize_kernels_err(self):
        
        # Overlapping non-neighbouring kernels should crash.
        w_ell = np.ones((3, 5))
        self.assertRaises(ValueError, fkernel.digitize_kernels, w_ell)

    def test_get_sd_kernels_fourier(self):
        
        lamb = 3
        lmax = 30

        imap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1)

        modlmap = dft.modlmap_real(imap.shape, imap.wcs, dtype=np.float64)
        fkernels = fkernel.get_sd_kernels_fourier(
            modlmap, lamb, digital=False)

        self.assertEqual(fkernels.shape, (5,) + modlmap.shape)

        # Make sure inner product is one.
        out_exp = np.ones(modlmap.shape)
        np.testing.assert_allclose(np.sum(fkernels ** 2, axis=0), out_exp)        
        
    def test_get_sd_kernels_fourier_digital(self):
        
        lamb = 3
        lmax = 30

        imap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1)

        modlmap = dft.modlmap_real(imap.shape, imap.wcs, dtype=np.float64)
        fkernels = fkernel.get_sd_kernels_fourier(
            modlmap, lamb, digital=True, oversample=10)

        self.assertEqual(fkernels.dtype, bool)
        self.assertEqual(fkernels.shape, (5,) + modlmap.shape)
        
        # Make sure kernels do not overlap.
        out_exp = np.ones(modlmap.shape)
        np.testing.assert_array_equal(np.sum(fkernels, axis=0), out_exp)        

