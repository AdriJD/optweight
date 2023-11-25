import unittest

import numpy as np
from pixell import curvedsky

from optweight import lensing, alm_utils

class TestSHT(unittest.TestCase):

    def test_LensAlm_init(self):

        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        # Try with just phi.
        plm = np.ones((1, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        # Try with phi and Omega.
        plm = np.ones((2, ainfo_lens.nelem), dtype=np.complex128)
        plm[1] *= 2
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

    def test_get_slices(self):

        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)
        plm = np.ones((1, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        tslice, pslice = lens_op.get_slices(1)
        self.assertEqual(tslice, slice(0, 1))
        self.assertEqual(pslice, None)        

        tslice, pslice = lens_op.get_slices(2)
        self.assertEqual(tslice, None)
        self.assertEqual(pslice, slice(0, 2))        

        tslice, pslice = lens_op.get_slices(3)
        self.assertEqual(tslice, slice(0, 1))
        self.assertEqual(pslice, slice(1, 3))        

        self.assertRaises(ValueError, lens_op.get_slices, 4)
        
    def test_LensAlm_lens_1d(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((1, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))

    def test_LensAlm_lens_2d(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))

    def test_LensAlm_lens_sp(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex64)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))

        # Try inplace.
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo, inplace=True)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex64)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertTrue(np.shares_memory(alm, alm_lensed))
        
    def test_LensAlm_lens_inplace(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo,
                                  inplace=True)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertTrue(np.shares_memory(alm, alm_lensed))

    def test_LensAlm_lens_pol(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        # Try 1d and 2d.
        alm = np.ones((1, ainfo.nelem), dtype=np.complex128)    
        alm_lensed = lens_op.lens(alm)        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        self.assertFalse(np.shares_memory(alm, alm_lensed))
        
        alm = np.ones((2, ainfo.nelem), dtype=np.complex128)
        alm[:,:2] = 0
        alm[:,ainfo.lmax+1] = 0        
        alm_lensed = lens_op.lens(alm)        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))               
        
    def test_LensAlm_lens_adjoint_1d(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens_adjoint(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))               
        
    def test_LensAlm_lens_adjoint_2d(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens_adjoint(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))               

    def test_LensAlm_lens_adjoint_sp(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex64)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens_adjoint(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))

        # Try inplace.
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo, inplace=True)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex64)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens_adjoint(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertTrue(np.shares_memory(alm, alm_lensed))
        
    def test_LensAlm_lens_adjoint_pol(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        # Try 1d and 2d.
        alm = np.ones((1, ainfo.nelem), dtype=np.complex128)    
        alm_lensed = lens_op.lens_adjoint(alm)        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        self.assertFalse(np.shares_memory(alm, alm_lensed))
        
        alm = np.ones((2, ainfo.nelem), dtype=np.complex128)
        alm[:,:2] = 0
        alm[:,ainfo.lmax+1] = 0        
        alm_lensed = lens_op.lens_adjoint(alm)        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)        
        self.assertFalse(np.shares_memory(alm, alm_lensed))               
        
    def test_LensAlm_lens_adjoint_inplace(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo,
                                  inplace=True)

        alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
        alm[1:3,:2] = 0
        alm[1:3,ainfo.lmax+1] = 0
        
        alm_lensed = lens_op.lens_adjoint(alm)
        
        np.testing.assert_allclose(alm[0], alm_lensed[0], rtol=1e-06)
        np.testing.assert_allclose(alm[1], alm_lensed[1], rtol=1e-06)
        np.testing.assert_allclose(alm[2], alm_lensed[2], rtol=1e-06)        
        self.assertTrue(np.shares_memory(alm, alm_lensed))               

    def test_LensAlm_err(self):
                
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        plm = np.zeros((2, ainfo_lens.nelem), dtype=np.complex128)
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        # Wrong npol.
        alm = np.ones((4, ainfo.nelem), dtype=np.complex128)        
        self.assertRaises(ValueError, lens_op.lens, alm)
        self.assertRaises(ValueError, lens_op.lens_adjoint, alm)
        
        # Wrong nelem.
        alm = np.ones((4, ainfo_lens.nelem), dtype=np.complex128)        
        self.assertRaises(ValueError, lens_op.lens, alm)        
        self.assertRaises(ValueError, lens_op.lens_adjoint, alm)
        
    def test_LensAlm_adjoint(self):

        # Use this trick to test if adjoint is really adjoint:
        # https://sepwww.stanford.edu/sep/prof/pvi/conj/paper_html/node9.html#:~:
        # text=The%20dot%20product%20test%20is,\are%20adjoint%20to%20each%20other
        # .&text=your%20program%20for-,.,scalars%20that%20should%20be%20equal.&te
        # xt=(\unless%20the%20random%20numbers%20do%20something%20miraculous)
        
        rng = np.random.default_rng(0)
        
        ainfo_lens = curvedsky.alm_info(10)
        ainfo = curvedsky.alm_info(5)

        # ell**2 (ell + 1)**2 / 2 / pi PhiPhi power spetrum from CAMB
        # is approx 1e-6.
        # So phi is approx 1e-3. plm is sqrt(ell * (ell + 1)) plm.
        cov_phi_ell = np.ones((2, ainfo_lens.lmax+1)) * 1e-6
        cov_phi_ell[:,:2] = 0
        ells = np.arange(ainfo_lens.lmax + 1)
        wpp = ells ** 2 * (ells + 1.) ** 2 / 2 / np.pi
        cov_phi_ell[:,1:] /= wpp[1:]
        plm = alm_utils.rand_alm(cov_phi_ell, ainfo_lens, rng)
                
        lens_op = lensing.LensAlm(plm, ainfo_lens, ainfo)

        # Create some random sky alms.
        cov_ell = np.ones((3, ainfo.lmax+1))
        alm_x = alm_utils.rand_alm(cov_ell, ainfo, rng)
        alm_y = alm_utils.rand_alm(cov_ell, ainfo, rng)        

        alm_x_lens = lens_op.lens(alm_x)
        alm_y_lens_adj = lens_op.lens_adjoint(alm_y)

        xthx_T = alm_utils.contract_almxblm(alm_y_lens_adj[0], alm_x[0])
        yhyt_T = alm_utils.contract_almxblm(alm_y[0], alm_x_lens[0])
        
        self.assertAlmostEqual(xthx_T, yhyt_T)

        xthx_P = alm_utils.contract_almxblm(alm_y_lens_adj[1:3], alm_x[1:3])
        yhyt_P = alm_utils.contract_almxblm(alm_y[1:3], alm_x_lens[1:3])
        
        self.assertAlmostEqual(xthx_P, yhyt_P)
