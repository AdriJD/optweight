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

        dft.irfft(fmap, imap)

        fmap_out = fmap * 0
        dft.rfft(imap, fmap_out)
        
        np.testing.assert_allclose(fmap, fmap_out)
        
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
        cdelt_exp[0] *= -1
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
        bins_exp = np.asarray([0., 591.45389248, 1105.18002676, 1514.60855085])

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
