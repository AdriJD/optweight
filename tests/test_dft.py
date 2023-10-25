import unittest
import numpy as np

from pixell import enmap, fft

from optweight import dft

class TestSHT(unittest.TestCase):

    def test_rfft(self):
        
        # Test roundtrip.
        ny = 10
        nx = 11
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec = np.radians([-60, 40])
        shape, wcs = enmap.band_geometry(dec, res=res, shape=(ny, nx), dims=(2,))
        imap = enmap.ones(shape, wcs)

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        
        dft.rfft(imap, fmap)
        fmap_copy = fmap.copy()

        omap = np.zeros_like(imap)
        dft.irfft(fmap, omap)
        
        np.testing.assert_allclose(imap, omap)
        # Test if c2r transform does not overwrite input.
        np.testing.assert_allclose(fmap, fmap_copy)

    def test_rfft_sp(self):
        
        # Test roundtrip with single precision.
        ny = 10
        nx = 11
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec = np.radians([-60, 40])
        shape, wcs = enmap.band_geometry(dec, res=res, shape=(ny, nx), dims=(2,))
        imap = enmap.ones(shape, wcs, dtype=np.float32)

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex64)
        
        dft.rfft(imap, fmap)
        fmap_copy = fmap.copy()

        omap = np.zeros_like(imap)
        dft.irfft(fmap, omap)
        
        np.testing.assert_allclose(imap, omap)
        np.testing.assert_allclose(fmap, fmap_copy)

        self.assertEqual(imap.dtype, np.float32)
        self.assertEqual(omap.dtype, np.float32)
        self.assertEqual(fmap.dtype, np.complex64)

    def test_rfft_err(self):
        
        ny = 10
        nx = 11
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec = np.radians([-60, 40])
        shape, wcs = enmap.band_geometry(dec, res=res, shape=(ny, nx), dims=(2,))
        imap = enmap.ones(shape, wcs)

        # Wrong dtype.
        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex64)
        
        self.assertRaises(TypeError, dft.rfft, imap, fmap)

        fmap_copy = fmap.copy()
        omap = np.zeros_like(imap)

        self.assertRaises(TypeError, dft.irfft, fmap, omap)

    def test_rfft_numpy(self):
        
        # Compare to numpy.
        ny = 10
        nx = 11
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec = np.radians([-60, 40])
        shape, wcs = enmap.band_geometry(dec, res=res, shape=(ny, nx), dims=(2,))

        imap = enmap.ones(shape, wcs)
        imap += np.random.randn(*imap.shape)

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        fmap_exp = np.fft.rfft2(imap, s=imap.shape[-2:], norm='ortho')        
        dft.rfft(imap, fmap)

        np.testing.assert_allclose(fmap, fmap_exp)

    def test_irfft(self):
        
        # Test roundtrip.
        ny = 10
        nx = 11
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec = np.radians([-60, 40])
        shape, wcs = enmap.band_geometry(dec, res=res, shape=(ny, nx), dims=(2,))
        imap = enmap.zeros(shape, wcs)

        fmap = np.ones(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)        
        fmap[0, 3, 5] = 2 + 4j

        print(fmap.shape, imap.shape)

        dft.irfft(fmap, imap)

        fmap_out = fmap * 0
        dft.rfft(imap, fmap_out)
        
        np.testing.assert_allclose(fmap, fmap_out)

    def test_allocate_fmap(self):
        
        ny = 10
        nx = 11
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        dec = np.radians([-60, 40])
        shape, wcs = enmap.band_geometry(dec, res=res, shape=(ny, nx), dims=(2,))
        
        dtype = np.float32
        omap = dft.allocate_fmap(shape, dtype)

        ny_exp = shape[-2]
        nx_exp = shape[-1] // 2 + 1

        self.assertEqual(omap.shape, (2, ny_exp, nx_exp))
        self.assertEqual(omap.dtype, np.complex64)
        self.assertTrue(np.all(omap == 0))

        omap = dft.allocate_fmap(shape, dtype, fill_value=1)
        self.assertTrue(np.all(omap == 1))

        dtype = np.float64
        omap = dft.allocate_fmap(shape, dtype)
        self.assertEqual(omap.dtype, np.complex128)
        
    def test_laxes_real(self):
        
        ny = 6
        nx = 11
        shape, wcs = enmap.geometry([0,0], res=0.1, deg=True, shape=(ny, nx))

        ly, lx = dft.laxes_real(shape, wcs)

        ly_exp = np.fft.fftfreq(ny, np.radians(0.1)) * 2 * np.pi
        lx_exp = np.fft.rfftfreq(nx, -np.radians(0.1)) * 2 * np.pi
        
        np.testing.assert_allclose(ly, ly_exp, atol=1e-3, rtol=1e-5)
        np.testing.assert_allclose(lx, lx_exp, atol=1e-3, rtol=1e-5)

    def test_lmap_real(self):
        
        ny = 6
        nx = 11
        shape, wcs = enmap.geometry([0,0], res=0.1, deg=True, shape=(ny, nx))

        ly_exp = np.fft.fftfreq(ny, np.radians(0.1)) * 2 * np.pi
        lx_exp = np.fft.rfftfreq(nx, -np.radians(0.1)) * 2 * np.pi
        lmap_exp = np.zeros((2, ny, nx // 2 + 1))
        lmap_exp[0] = ly_exp[:,np.newaxis]
        lmap_exp[1] = lx_exp[np.newaxis,:]
            
        lmap = dft.lmap_real(shape, wcs)

        self.assertEqual(lmap.shape, (2, ny, nx // 2 + 1))
        np.testing.assert_allclose(lmap, lmap_exp, atol=1e-3, rtol=1e-5)
        self.assertEqual(lmap.dtype, np.float64)
    
        lmap = dft.lmap_real(shape, wcs, dtype=np.float32)
        self.assertEqual(lmap.dtype, np.float32)
        np.testing.assert_allclose(
            lmap, lmap_exp.astype(np.float32), atol=1e-3, rtol=1e-5)

    def test_modlmap_real(self):
        
        ny = 6
        nx = 11
        shape, wcs = enmap.geometry([0,0], res=0.1, deg=True, shape=(ny, nx))

        ly_exp = np.fft.fftfreq(ny, np.radians(0.1)) * 2 * np.pi
        lx_exp = np.fft.rfftfreq(nx, -np.radians(0.1)) * 2 * np.pi
        lmap_exp = np.zeros((2, ny, nx // 2 + 1))
        lmap_exp[0] = ly_exp[:,np.newaxis]
        lmap_exp[1] = lx_exp[np.newaxis,:]
        lmod_exp = np.zeros((ny, nx // 2 + 1))
        lmod_exp = np.sqrt(lmap_exp[0] ** 2 + lmap_exp[1] ** 2)

        lmod = dft.modlmap_real(shape, wcs)

        np.testing.assert_allclose(lmod, lmod_exp, atol=1e-3, rtol=1e-5)

        lmod = dft.modlmap_real(shape, wcs, dtype=np.float32)
        self.assertEqual(lmod.dtype, np.float32)
        np.testing.assert_allclose(
            lmod, lmod_exp.astype(np.float32), atol=1e-3, rtol=1e-5)
        
    def test_lwcs_real(self):

        ny = 6
        nx = 11
        shape, wcs = enmap.geometry([0,0], res=0.1, deg=True, shape=(ny, nx))
        
        lwcs = dft.lwcs_real(shape, wcs)

        # Note that wcs order is x, y. Center pix y is 6 // 2 + 1 = 4.
        np.testing.assert_allclose(lwcs.wcs.crpix, np.asarray([0., 4.]))
        np.testing.assert_allclose(lwcs.wcs.crval, np.asarray([0., 0.]))        
        cdelt_exp = 2 * np.pi / np.radians([1.1, 0.6])

        np.testing.assert_allclose(lwcs.wcs.cdelt, cdelt_exp, atol=1e-3, rtol=1e-5)

    def test_lbin(self):
        
        ny = 6
        nx = 11
        shape, wcs = enmap.geometry([0,0], res=0.1, deg=True, shape=(ny, nx))

        ly_exp = np.fft.fftfreq(ny, np.radians(0.1)) * 2 * np.pi
        lx_exp = np.fft.rfftfreq(nx, -np.radians(0.1)) * 2 * np.pi
        lmap_exp = np.zeros((2, ny, nx // 2 + 1))
        lmap_exp[0] = ly_exp[:,np.newaxis]
        lmap_exp[1] = lx_exp[np.newaxis,:]
        lmod_exp = np.zeros((ny, nx // 2 + 1))
        lmod_exp = np.sqrt(lmap_exp[0] ** 2 + lmap_exp[1] ** 2)

        fmap = np.ones((2, ny, nx // 2 + 1))
        
        fbin, bins = dft.lbin(fmap, lmod_exp, bsize=500)

        fbin_exp = np.ones((2, 4))
        bins_exp = np.asarray([163.636364, 747.392518, 1282.901329, 1731.966898])

        np.testing.assert_allclose(bins, bins_exp)
        np.testing.assert_allclose(fbin, fbin_exp)

    def test_cl2flat_roundtrip(self):
                
        ny = 6
        nx = 11
        shape, wcs = enmap.geometry([0,0], res=0.1, deg=True, shape=(ny, nx))
        modlmap = dft.modlmap_real(shape, wcs, dtype=np.float64)

        ells = np.linspace(modlmap.min(), modlmap.max(), 100)
        c_ell = np.zeros((1, 1, ells.size))
        c_ell[0,0,:] = (1 + (np.maximum(ells, 10) / 2000) ** -3.)

        out_exp = np.zeros((1, 1) + modlmap.shape)
        out_exp[:] = (1 + (np.maximum(modlmap, 10) / 2000) ** -3.)
        
        out = dft.cl2flat(c_ell, ells, modlmap)
        
        # This is not expected to be perfect match because the interpolation
        # done by cl2flat.
        np.testing.assert_allclose(out, out_exp, rtol=1e-2)

    def test_contract_fxg(self):
        
        # Computing an inner product of 2d Fourier coefficients from the output of rfft
        # is a bit non-trivial. Here we copmare to inner product from normal fft coefficients.

        # Odd nx, even ny.
        ny = 4
        nx = 5
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=())

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        self.assertAlmostEqual(ans, ans_exp)

        # Even nx, even ny.
        ny = 4
        nx = 4
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=())

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        self.assertAlmostEqual(ans, ans_exp)

        # Odd nx, odd ny.
        ny = 5
        nx = 5
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=())

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        self.assertAlmostEqual(ans, ans_exp)

        # Even nx, odd ny.
        ny = 5
        nx = 4
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=())

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        self.assertAlmostEqual(ans, ans_exp)

    def test_contract_fxg_2d(self):
        
        # Same as above, but now with (2, ny, ny) input maps.

        # Odd nx, even ny.
        ny = 4
        nx = 5
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=(2,))

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        # Even nx, even ny.
        ny = 4
        nx = 4
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=(2,))

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        # Odd nx, odd ny.
        ny = 5
        nx = 5
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=(2,))

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        self.assertAlmostEqual(ans, ans_exp)

        # Even nx, odd ny.
        ny = 5
        nx = 4
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=(2,))

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex128)

        imap_f = np.random.randn(*shape)
        imap_g = np.random.randn(*shape)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        self.assertAlmostEqual(ans, ans_exp)

    def test_contract_fxg_sp(self):
                
        # Single precision.
        ny = 4
        nx = 5
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=())

        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex64)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex64)

        imap_f = np.random.randn(*shape).astype(np.float32)
        imap_g = np.random.randn(*shape).astype(np.float32)
        
        dft.rfft(imap_f, fmap)
        dft.rfft(imap_g, gmap)

        fmap2 = np.fft.fft2(imap_f, norm='ortho')
        gmap2 = np.fft.fft2(imap_g, norm='ortho')

        ans_exp = np.real(np.sum(fmap2 * np.conj(gmap2)))
        ans = dft.contract_fxg(fmap, gmap)

        self.assertAlmostEqual(ans, ans_exp, places=5)

    def test_contract_fxg_err(self):
                
        ny = 4
        nx = 5
        res = [np.pi / (ny - 1), 2 * np.pi / nx]
        shape, wcs = enmap.fullsky_geometry(res=res, shape=(ny, nx), dims=())

        # Different shapes.
        fmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 1,), np.complex64)
        gmap = np.zeros(shape[:-1] + (shape[-1] // 2 + 2,), np.complex64)

        self.assertRaises(ValueError, dft.contract_fxg, fmap, gmap)

    def test_slice_fmap(self):
        
        arr = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # positive 3
                          [0, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        slice_x = slice(0, 3)
        slice_ypos = slice(0, 3)
        slice_yneg = slice(-2, None)

        out_exp = np.asarray([[0, 1, 1],  
                              [1, 1, 0],  
                              [1, 0, 0],  
                              [0, 1, 0],  
                              [0, 1, 1]])

        out = dft.slice_fmap(arr, (slice_ypos, slice_yneg), slice_x)
        np.testing.assert_allclose(out, out_exp)

    def test_slice_fmap_3d(self):
        
        arr = np.zeros((2, 7, 4))
        arr[0] = np.asarray([[0, 1, 1, 0],  # monopole
                             [1, 1, 0, 0],  # positive 1
                             [1, 0, 0, 0],  # positive 2
                             [0, 0, 0, 0],  # positive 3
                             [0, 0, 0, 0],  # negative 3
                             [0, 1, 0, 0],  # negative 2
                             [0, 1, 1, 0]]) # negative 1
        arr[1] = arr[0] * 2

        slice_x = slice(0, 3)
        slice_ypos = slice(0, 3)
        slice_yneg = slice(-2, None)

        out_exp = np.zeros((2, 5, 3))
        out_exp[0] = np.asarray([[0, 1, 1],  
                                 [1, 1, 0],  
                                 [1, 0, 0],  
                                 [0, 1, 0],  
                                 [0, 1, 1]])
        out_exp[1] = out_exp[0] * 2
        
        out = dft.slice_fmap(arr, (slice_ypos, slice_yneg), slice_x)
        np.testing.assert_allclose(out, out_exp)

    def test_slice_fmap_laxes(self):
        
        arr = np.asarray([[0, 1, 1, 0],  # monopole
                          [1, 1, 0, 0],  # positive 1
                          [1, 0, 0, 0],  # positive 2
                          [0, 0, 0, 0],  # positive 3
                          [0, 0, 0, 0],  # negative 3
                          [0, 1, 0, 0],  # negative 2
                          [0, 1, 1, 0]]) # negative 1

        slice_x = slice(0, 3)
        slice_ypos = slice(0, 3)
        slice_yneg = slice(-2, None)

        ly = np.asarray([0, 1, 2, 3, -3, -2, -1])
        lx = np.asarray([0, 1, 2, 3])
        laxes = (ly, lx)

        out_exp = np.asarray([[0, 1, 1],  
                              [1, 1, 0],  
                              [1, 0, 0],  
                              [0, 1, 0],  
                              [0, 1, 1]])

        ly_exp = np.asarray([0, 1, 2, -2, -1])
        lx_exp = np.asarray([0, 1, 2])

        out, (ly_out, lx_out) = dft.slice_fmap(arr, (slice_ypos, slice_yneg), slice_x,
                             laxes=laxes)
        np.testing.assert_allclose(out, out_exp)
        np.testing.assert_allclose(ly_out, ly_exp)
        np.testing.assert_allclose(lx_out, lx_exp)

    def test_get_optimal_fftlen(self):
        
        n_in = 0
        n_out_exp = 0
        n_out = dft.get_optimal_fftlen(n_in)

        self.assertEqual(n_out, n_out_exp)

        n_in = 1
        n_out_exp = 2
        n_out = dft.get_optimal_fftlen(n_in)

        self.assertEqual(n_out, n_out_exp)

        n_in = 1
        n_out_exp = 1
        n_out = dft.get_optimal_fftlen(n_in, even=False)

        self.assertEqual(n_out, n_out_exp)

        n_in = 3
        n_out_exp = 4
        n_out = dft.get_optimal_fftlen(n_in, even=False)

        n_in = 4
        n_out_exp = 4
        n_out = dft.get_optimal_fftlen(n_in, even=False)

        self.assertEqual(n_out, n_out_exp)
