import unittest
import numpy as np

import healpy as hp
from pixell import curvedsky

from optweight import preconditioners
from optweight import map_utils, sht, alm_c_utils, alm_utils, mat_utils

class TestPreconditioners(unittest.TestCase):

    def test_maskedpreconditioner(self):

        lmax = 10
        ainfo = curvedsky.alm_info(lmax)
        spin = 0
        minfo = map_utils.get_gauss_minfo(2 * lmax)
        mask_bool = np.ones((1, minfo.npix), dtype=bool)

        ells = np.arange(lmax + 1)
        icov_ell = np.ones((1, lmax + 1)) * ells ** 2

        mask_bool_2d = map_utils.view_2d(mask_bool, minfo)
        mask_bool_2d[0,:3] = False
        mask_bool_2d[0,7:] = False

        prec = preconditioners.MaskedPreconditioner(
            ainfo, icov_ell, spin, mask_bool, minfo, n_jacobi=3, min_pix=10,
            lmax_r_ell=lmax)

        self.assertEqual(prec.npol, 1)
        self.assertEqual(prec.n_jacobi, 3)

        # Test calling the operator.
        alm = np.ones((1, ainfo.nelem), dtype=np.complex128)
        alm_out = prec(alm)
        self.assertTrue(np.all(np.isfinite(alm_out)))
        self.assertFalse(np.shares_memory(alm, alm_out))
        self.assertEqual(alm.dtype, alm_out.dtype)

        alm = np.ones((1, ainfo.nelem), dtype=np.complex64)
        alm_out_sp = prec(alm)
        self.assertTrue(np.all(np.isfinite(alm_out_sp)))
        self.assertFalse(np.shares_memory(alm, alm_out_sp))
        self.assertEqual(alm.dtype, alm_out_sp.dtype)

        np.testing.assert_allclose(alm_out / alm_out_sp, np.ones_like(alm),
                                   rtol=3e-4)

    def test_get_itau_ell_harm(self):
        
        nell = 4
        npol = 3
        w_ell = np.zeros((2, nell), dtype=np.float32)
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.1
        
        icov_ell = np.ones((npol, nell), dtype=np.float32)
        itau_ell = preconditioners.get_itau_ell_harm(icov_ell, w_ell)
        
        self.assertEqual(itau_ell.shape, (npol, npol, nell))
        self.assertEqual(itau_ell.dtype, np.float32)
        
        itau_ell_exp = np.asarray([1, 1, 0.01, 0.01]) * np.eye(npol)[:,:,np.newaxis]
        itau_ell_exp.astype(np.float32)

        np.testing.assert_allclose(itau_ell, itau_ell_exp)

        # Full input matrix, should still only use diagonal.
        icov_ell = np.ones((npol, npol, nell), dtype=np.float32)
        itau_ell = preconditioners.get_itau_ell_harm(icov_ell, w_ell)
        np.testing.assert_allclose(itau_ell, itau_ell_exp)
