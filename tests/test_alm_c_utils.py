import unittest
import numpy as np

from pixell import sharp

from optweight import alm_c_utils

class TestAlmCUtils(unittest.TestCase):

    def test_trunc_alm_dp(self):
        
        alm = np.zeros((2, 6), dtype=np.complex128)
        alm[0] = np.asarray([1, 2, 3, 4, 5, 6])
        alm[1] = np.asarray([1, 2, 3, 4, 5, 6]) * 2

        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 1
        alm_new_exp = np.zeros((2, 3), dtype=np.complex128)
        alm_new_exp[0] = np.asarray([1, 2, 4])
        alm_new_exp[1] = np.asarray([1, 2, 4]) * 2

        alm_new = np.zeros_like(alm_new_exp)
        alm_c_utils.trunc_alm(alm, alm_new, 2, 1)

        np.testing.assert_equal(alm_new, alm_new_exp)

    def test_trunc_alm_sp(self):
        
        alm = np.zeros((2, 6), dtype=np.complex64)
        alm[0] = np.asarray([1, 2, 3, 4, 5, 6])
        alm[1] = np.asarray([1, 2, 3, 4, 5, 6]) * 2

        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 1
        alm_new_exp = np.zeros((2, 3), dtype=np.complex64)
        alm_new_exp[0] = np.asarray([1, 2, 4])
        alm_new_exp[1] = np.asarray([1, 2, 4]) * 2

        alm_new = np.zeros_like(alm_new_exp)
        alm_c_utils.trunc_alm(alm, alm_new, 2, 1)

        np.testing.assert_equal(alm_new, alm_new_exp)

    def test_trunc_alm_err(self):
        
        alm = np.zeros((2, 6), dtype=np.complex128)
        alm[0] = np.asarray([1, 2, 3, 4, 5, 6])
        alm[1] = np.asarray([1, 2, 3, 4, 5, 6]) * 2

        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 1
        alm_new_exp = np.zeros((2, 3), dtype=np.complex128)
        alm_new_exp[0] = np.asarray([1, 2, 4])
        alm_new_exp[1] = np.asarray([1, 2, 4]) * 2

        alm_new = np.zeros_like(alm_new_exp)

        # lmax_out > lmax_in
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 3)

        # Wrong alm size.
        alm = np.zeros((3, 6), dtype=np.complex128)
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 1)

        # Wrong alm size 2.
        alm = np.zeros((2, 6), dtype=np.complex128)
        alm_new = np.zeros((3, 3), dtype=np.complex128)
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 1)

        # Wrong alm size 3.
        alm_new = np.zeros((2, 3), dtype=np.complex128)
        alm = np.zeros((2, 5), dtype=np.complex128)
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 1)

        # Wrong alm size 2.
        alm = np.zeros((2, 6), dtype=np.complex128)
        alm_new = np.zeros((2, 2), dtype=np.complex128)
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 1)

        # Wrong dtype 1.
        alm = np.zeros((2, 6), dtype=np.complex128)
        alm_new = np.zeros((2, 3), dtype=np.complex64)
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 1)

        # Wrong dtype 2.
        alm = np.zeros((2, 6), dtype=np.complex64)
        alm_new = np.zeros((2, 3), dtype=np.complex128)
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 1)

        # Wrong dtype 3.
        alm = np.zeros((2, 6), dtype=np.complex256)
        alm_new = np.zeros((2, 3), dtype=np.complex256)
        self.assertRaises(ValueError, alm_c_utils.trunc_alm, alm, alm_new, 2, 1)

    def test_lmul_dp(self):
        
        lmax = 4
        ncomp = 3

        ainfo = sharp.alm_info(4)
        alm = np.arange(ncomp * ainfo.nelem, dtype=np.complex128)
        alm = alm.reshape(ncomp, ainfo.nelem)
        alm *= 1 + 1j

        alm_out = np.zeros_like(alm)

        lmat = np.arange(ncomp * ncomp * (lmax + 1), dtype=float)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 1)

        alm_c_utils.lmul(alm, lmat, ainfo, alm_out=alm_out)
        alm_out_exp = ainfo.lmul(alm, lmat)
        np.testing.assert_array_almost_equal(alm_out, alm_out_exp)

        # Diag mat.
        lmat = np.arange(ncomp * (lmax + 1), dtype=float)
        lmat = lmat.reshape(ncomp, lmax + 1)

        alm_c_utils.lmul(alm, lmat, ainfo, alm_out=alm_out)
        alm_out_exp = ainfo.lmul(alm, lmat * np.eye(3)[:,:,np.newaxis])
        np.testing.assert_array_almost_equal(alm_out, alm_out_exp)

        # 1d mat.
        lmat = np.arange((lmax + 1), dtype=float)

        alm_c_utils.lmul(alm, lmat, ainfo, alm_out=alm_out)
        alm_out_exp = ainfo.lmul(alm, lmat)
        np.testing.assert_array_almost_equal(alm_out, alm_out_exp)

    def test_lmul_inplace_dp(self):
        
        lmax = 4
        ncomp = 3

        ainfo = sharp.alm_info(4)
        alm = np.arange(ncomp * ainfo.nelem, dtype=np.complex128)
        alm = alm.reshape(ncomp, ainfo.nelem)
        alm *= 1 + 1j

        lmat = np.arange(ncomp * ncomp * (lmax + 1), dtype=float)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 1)

        alm_out_exp = ainfo.lmul(alm, lmat)
        alm_c_utils.lmul(alm, lmat, ainfo, inplace=True)
        np.testing.assert_array_almost_equal(alm, alm_out_exp)

        # Diag mat.
        lmat = np.arange(ncomp * (lmax + 1), dtype=float)
        lmat = lmat.reshape(ncomp, lmax + 1)

        alm_out_exp = ainfo.lmul(alm, lmat * np.eye(3)[:,:,np.newaxis])
        alm_c_utils.lmul(alm, lmat, ainfo, inplace=True)
        np.testing.assert_array_almost_equal(alm, alm_out_exp)

    def test_lmul_sp(self):
        
        lmax = 4
        ncomp = 3

        ainfo = sharp.alm_info(4)
        alm = np.arange(ncomp * ainfo.nelem, dtype=np.complex64)
        alm = alm.reshape(ncomp, ainfo.nelem)
        alm *= 1 + 1j

        alm_out = np.zeros_like(alm)

        lmat = np.arange(ncomp * ncomp * (lmax + 1), dtype=np.float32)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 1)

        alm_c_utils.lmul(alm, lmat, ainfo, alm_out=alm_out)
        alm_out_exp = ainfo.lmul(alm, lmat)
        np.testing.assert_array_almost_equal(alm_out, alm_out_exp)

        # Diag mat.
        lmat = np.arange(ncomp * (lmax + 1), dtype=np.float32)
        lmat = lmat.reshape(ncomp, lmax + 1)

        alm_c_utils.lmul(alm, lmat, ainfo, alm_out=alm_out)
        alm_out_exp = ainfo.lmul(alm, lmat * np.eye(3)[:,:,np.newaxis])
        np.testing.assert_array_almost_equal(alm_out, alm_out_exp)

        # 1d mat.
        lmat = np.arange((lmax + 1), dtype=np.float32)

        alm_c_utils.lmul(alm, lmat, ainfo, alm_out=alm_out)
        alm_out_exp = ainfo.lmul(alm, lmat)
        np.testing.assert_array_almost_equal(alm_out, alm_out_exp)

    def test_lmul_inplace_sp(self):
        
        lmax = 4
        ncomp = 3

        ainfo = sharp.alm_info(4)
        alm = np.arange(ncomp * ainfo.nelem, dtype=np.complex64)
        alm = alm.reshape(ncomp, ainfo.nelem)
        alm *= 1 + 1j

        lmat = np.arange(ncomp * ncomp * (lmax + 1), dtype=np.float32)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 1)

        alm_out_exp = ainfo.lmul(alm, lmat)
        alm_c_utils.lmul(alm, lmat, ainfo, inplace=True)
        np.testing.assert_array_almost_equal(alm, alm_out_exp)

        # Diag mat.
        lmat = np.arange(ncomp * (lmax + 1), dtype=np.float32)
        lmat = lmat.reshape(ncomp, lmax + 1)

        alm_out_exp = ainfo.lmul(alm, lmat * np.eye(3)[:,:,np.newaxis])
        alm_c_utils.lmul(alm, lmat, ainfo, inplace=True)
        np.testing.assert_array_almost_equal(alm, alm_out_exp)

    def test_lmul_err(self):

        lmax = 4
        ncomp = 3

        ainfo = sharp.alm_info(4)
        alm = np.arange(ncomp * ainfo.nelem, dtype=np.complex128)
        alm = alm.reshape(ncomp, ainfo.nelem)
        alm *= 1 + 1j

        alm_out = np.zeros_like(alm)

        # Wrong shape.
        lmat = np.arange(ncomp * ncomp * (lmax + 2), dtype=np.float64)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 2)

        self.assertRaises(ValueError, alm_c_utils.lmul, alm, lmat, ainfo,
                          alm_out=alm_out)

        # Wrong alm_out.
        alm_out_wrong = np.zeros((ainfo.nelem))
        lmat = np.arange(ncomp * ncomp * (lmax + 1), dtype=np.float64)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 1)

        self.assertRaises(ValueError, alm_c_utils.lmul, alm, lmat, ainfo,
                          alm_out=alm_out_wrong)

        # Wrong dtype.
        lmat = np.arange(ncomp * ncomp * (lmax + 1), dtype=np.float32)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 1)

        self.assertRaises(ValueError, alm_c_utils.lmul, alm, lmat, ainfo,
                          alm_out=alm_out)

        # Wrong alm layout.
        ainfo = sharp.alm_info(4, layout='rect')
        alm = np.arange(ncomp * ainfo.nelem, dtype=np.complex128)
        alm = alm.reshape(ncomp, ainfo.nelem)
        alm *= 1 + 1j

        lmat = np.arange(ncomp * ncomp * (lmax + 1), dtype=np.float64)
        lmat = lmat.reshape(ncomp, ncomp, lmax + 1)

        self.assertRaises(ValueError, alm_c_utils.lmul, alm, lmat, ainfo,
                          alm_out=alm_out)

