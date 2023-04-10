import unittest
import numpy as np

from pixell import curvedsky

from optweight import fkernel
from optweight import wlm_utils
from optweight import dft
from optweight import wavtrans

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

        ly, lx = dft.laxes_real(imap.shape, imap.wcs)
        modlmap = dft.laxes2modlmap(ly, lx, dtype=np.float64)
        fkernels = fkernel.get_sd_kernels_fourier(
            ly, lx, lamb, digital=False)

        self.assertEqual(len(fkernels), 5)
        self.assertEqual(fkernels.dtype, np.float64)

    def test_get_sd_kernels_fourier_inner(self):

        lamb = 3
        lmax = 30

        imap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1)

        ly, lx = dft.laxes_real(imap.shape, imap.wcs)
        modlmap = dft.laxes2modlmap(ly, lx, dtype=np.float64)
        fkernels = fkernel.get_sd_kernels_fourier(
            ly, lx, lamb, digital=False)

        nwav = 5
        self.assertEqual(len(fkernels), 5)

        # Make sure inner product is one.
        nwav = len(fkernels)
        fkernels_arr = fkernels.to_full_array()

        self.assertEqual(fkernels_arr.shape, (nwav,) + modlmap.shape)
        out_exp = np.ones(modlmap.shape)
        np.testing.assert_allclose(np.sum(fkernels_arr ** 2, axis=0), out_exp)

    def test_get_sd_kernels_fourier_digital_inner(self):

        lamb = 3
        lmax = 30

        imap = curvedsky.make_projectable_map_by_pos(
            [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(1,), oversample=1)

        ly, lx = dft.laxes_real(imap.shape, imap.wcs)
        modlmap = dft.laxes2modlmap(ly, lx, dtype=np.float64)
        fkernels = fkernel.get_sd_kernels_fourier(
            ly, lx, lamb, digital=True, oversample=10)

        nwav = 5
        self.assertEqual(len(fkernels), 5)
        self.assertEqual(fkernels.dtype, bool)

        # Make sure inner product is one.
        nwav = len(fkernels)
        fkernels_arr = fkernels.to_full_array()

        self.assertEqual(fkernels_arr.shape, (nwav,) + modlmap.shape)
        out_exp = np.ones(modlmap.shape)
        np.testing.assert_allclose(np.sum(fkernels_arr ** 2, axis=0), out_exp)

    def test_find_last_nonzero_idx(self):

        arr = np.asarray([0, 1, 0])
        idx_exp = 1
        idx = fkernel._find_last_nonzero_idx(arr)
        self.assertEqual(idx, idx_exp)

        arr = np.asarray([0, 1, 1])
        idx_exp = 2
        idx = fkernel._find_last_nonzero_idx(arr)
        self.assertEqual(idx, idx_exp)

        arr = np.asarray([1, 0, 0])
        idx_exp = 0
        idx = fkernel._find_last_nonzero_idx(arr)
        self.assertEqual(idx, idx_exp)

        arr = np.asarray([0, 0, 0])
        self.assertRaises(ValueError, fkernel._find_last_nonzero_idx, arr)

    def test_find_last_nonzero_idx_err(self):

        arr = np.asarray([0, 0, 0])
        self.assertRaises(ValueError, fkernel._find_last_nonzero_idx, arr)

        arr = np.ones((1, 2))
        self.assertRaises(ValueError, fkernel._find_last_nonzero_idx, arr)

    def test_find_kernel_slice(self):

        arr = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(arr)
        slice_x_exp = slice(0, 3)
        slice_ypos_exp = slice(0, 3)
        slice_yneg_exp = slice(-2, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

        arr = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [1, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(arr)
        slice_x_exp = slice(0, 3)
        slice_ypos_exp = slice(0, 3)
        slice_yneg_exp = slice(-3, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

        arr = np.asarray([[0, 1, 1, 1],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [1, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(arr)
        slice_x_exp = slice(0, 4)
        slice_ypos_exp = slice(0, 3)
        slice_yneg_exp = slice(-3, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)


        arr = np.asarray([[1, 0, 0, 0],  # monopole
                          [0, 0, 0, 0],  # positive 1
                          [0, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # negative 3
                          [0, 0, 0, 0],  # negative 2
                          [0, 0, 0, 0]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(arr)
        slice_x_exp = slice(0, 1)
        slice_ypos_exp = slice(0, 1)
        slice_yneg_exp = slice(0, 0)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)


        arr = np.asarray([[0, 0, 0, 0],  # monopole
                          [0, 0, 0, 0],  # positive 1
                          [0, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # negative 3
                          [0, 0, 0, 0],  # negative 2
                          [1, 0, 0, 0]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(arr)
        slice_x_exp = slice(0, 1)
        slice_ypos_exp = slice(0, 2)
        slice_yneg_exp = slice(-1, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

        # Try with odd input.
        arr = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [1, 0, 0, 0],  # positive 3
                          [0, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(arr)
        slice_x_exp = slice(0, 3)
        slice_ypos_exp = slice(0, 4)
        slice_yneg_exp = slice(-3, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

        arr = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # positive 3
                          [1, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(arr)
        slice_x_exp = slice(0, 3)
        slice_ypos_exp = slice(0, 4)
        slice_yneg_exp = slice(-3, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

    def test_find_kernel_slice_optimize(self):

        arr = np.asarray([[0],  # monopole
                          [1],  # positive 1
                          [0],  # positive 2
                          [0],  # negative 2
                          [1]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(
            arr, optimize_len=True)

        # Without optimization we would get
        # [0] # mono
        # [1] # pos 1
        # [1] # neg 1

        # But 3 is not an optimal fftlen. It will become 4.
        # [0] # mono
        # [1] # pos 1
        # [0] # neg 2
        # [1] # neg 1

        slice_x_exp = slice(0, 1)
        slice_ypos_exp = slice(0, 2)
        slice_yneg_exp = slice(-2, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

        # Even input size.
        arr = np.asarray([[0],  # monopole
                          [1],  # positive 1
                          [0],  # positive 2
                          [0],  # negative 3
                          [0],  # negative 2
                          [1]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(
            arr, optimize_len=True)

        # Without optimization we would get
        # [0] # mono
        # [1] # pos 1
        # [1] # neg 1

        # But 3 is not an optimal fftlen. It will become 4.
        # [0] # mono
        # [1] # pos 1
        # [0] # neg 2
        # [1] # neg 1

        slice_x_exp = slice(0, 1)
        slice_ypos_exp = slice(0, 2)
        slice_yneg_exp = slice(-2, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

        # Even input with no cut.
        arr = np.asarray([[0],  # monopole
                          [1],  # positive 1
                          [0],  # positive 2
                          [1],  # negative 3
                          [0],  # negative 2
                          [1]]) # negative 1

        (slice_ypos, slice_yneg), slice_x = fkernel.find_kernel_slice(
            arr, optimize_len=True)

        slice_x_exp = slice(0, 1)
        slice_ypos_exp = slice(0, 3)
        slice_yneg_exp = slice(-3, None)

        self.assertEqual(slice_x, slice_x_exp)
        self.assertEqual(slice_ypos, slice_ypos_exp)
        self.assertEqual(slice_yneg, slice_yneg_exp)

    def test_find_kernel_slice_err(self):

        arr = np.asarray([[0, 0],
                          [0, 0]])

        self.assertRaises(ValueError, fkernel.find_kernel_slice, arr)

        arr = np.ones((1, 2, 2))

        self.assertRaises(ValueError, fkernel.find_kernel_slice, arr)

        arr = np.ones((1, 2, 2)) * 1e-7

        self.assertRaises(ValueError, fkernel.find_kernel_slice, arr)

    def test_fkernel_init(self):

        arr = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        slice_x = slice(0, 3)
        slice_ypos = slice(0, 3)
        slice_yneg = slice(-2, None)

        ly = np.asarray([0, 1, 2, -2, -1])
        lx = np.asarray([0, 1, 2])

        fk_arr = np.asarray([[0, 1, 1],  # monopole
                             [1, 1, 0],  # positive 1
                             [1, 0, 0],  # positive 2
                             [0, 1, 0],  # negative 2
                             [0, 1, 1]]) # negative 1

        fk = fkernel.FKernel(fk_arr, ly, lx, slices_y=(slice_ypos, slice_yneg),
                             slice_x=slice_x, shape_full=arr.shape)

        lmap_y = ly[:,np.newaxis]
        lmap_x = lx[np.newaxis,:]
        modlmap_exp = np.sqrt(lmap_y ** 2 + lmap_x ** 2)
        np.testing.assert_allclose(fk.modlmap(), modlmap_exp)

        self.assertEqual(fk.dtype, fk_arr.dtype)

        np.testing.assert_allclose(fk.to_full_array(), arr)

    def test_fkernelset(self):

        # Kernel 1.
        arr1 = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        slice_x = slice(0, 3)
        slice_ypos = slice(0, 3)
        slice_yneg = slice(-2, None)

        ly = np.asarray([0, 1, 2, -2, -1])
        lx = np.asarray([0, 1, 2])

        fk_arr1 = np.asarray([[0, 1, 1],  # monopole
                              [1, 1, 0],  # positive 1
                              [1, 0, 0],  # positive 2
                              [0, 1, 0],  # negative 2
                              [0, 1, 1]]) # negative 1
        
        fk1 = fkernel.FKernel(fk_arr1, ly, lx, slices_y=(slice_ypos, slice_yneg),
                             slice_x=slice_x, shape_full=arr1.shape)
        
        # Kernel 2.
        arr2 = np.asarray([[1, 0, 0, 0],  # monopole
                           [0, 0, 0, 0],  # positive 1
                           [0, 0, 0, 0],  # positive 2
                           [0, 0, 0, 0],  # negative 3
                           [0, 0, 0, 0],  # negative 2
                           [0, 0, 0, 0]]) # negative 1

        slice_x = slice(0, 1)
        slice_ypos = slice(0, 1)
        slice_yneg = slice(-1, -1)

        ly = np.asarray([0])
        lx = np.asarray([0])

        fk_arr2 = np.asarray([[1]])

        fk2 = fkernel.FKernel(fk_arr2, ly, lx, slices_y=(slice_ypos, slice_yneg),
                             slice_x=slice_x, shape_full=arr2.shape)

        fks = fkernel.FKernelSet()
        fks[0] = fk1
        fks[1] = fk2

        self.assertEqual(len(fks), 2)
        self.assertEqual(fks.dtype, np.int64)
        self.assertTrue(np.shares_memory(fks[0].fkernel, fk_arr1))
        self.assertTrue(np.shares_memory(fks[1].fkernel, fk_arr2))

        full_arr = fks.to_full_array()

        np.testing.assert_allclose(full_arr[0], arr1)
        np.testing.assert_allclose(full_arr[1], arr2)

        # Changing dtype should make copy.
        fks = fks.astype(np.float32)
        self.assertEqual(fks.dtype, np.float32)
        self.assertEqual(fks[0].dtype, np.float32)
        self.assertEqual(fks[0].fkernel.dtype, np.float32)
        self.assertEqual(fks[1].dtype, np.float32)
        self.assertEqual(fks[1].fkernel.dtype, np.float32)
        
        self.assertFalse(np.shares_memory(fks[0].fkernel, fk_arr1))
        self.assertFalse(np.shares_memory(fks[1].fkernel, fk_arr2))

        np.testing.assert_allclose(full_arr[0], arr1)
        np.testing.assert_allclose(full_arr[1], arr2)
