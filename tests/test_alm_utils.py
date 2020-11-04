import unittest
import numpy as np

from pixell import sharp

from optweight import alm_utils

class TestAlmUtils(unittest.TestCase):

    def test_trunc_alm(self):
        
        alm = np.asarray([1, 2, 3, 4, 5, 6])
        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 1
        alm_new_exp = np.asarray([1, 2, 4])
        
        alm_new, ainfo_new = alm_utils.trunc_alm(alm, ainfo, lmax_new)

        np.testing.assert_equal(alm_new, alm_new_exp)
        
    def test_trunc_alm_2d(self):

        alm = np.zeros((2, 6))
        alm[0] = np.asarray([1, 2, 3, 4, 5, 6])
        alm[1] = np.asarray([1, 2, 3, 4, 5, 6]) * 2

        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 1
        alm_new_exp = np.zeros((2, 3))
        alm_new_exp[0] = np.asarray([1, 2, 4])
        alm_new_exp[1] = np.asarray([1, 2, 4]) * 2
        
        alm_new, ainfo_new = alm_utils.trunc_alm(alm, ainfo, lmax_new)

        np.testing.assert_equal(alm_new, alm_new_exp)

    def test_trunc_alm_err(self):
        
        alm = np.asarray([1, 2, 3, 4, 5, 6])
        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 3
        
        self.assertRaises(ValueError, alm_utils.trunc_alm, alm, ainfo, lmax_new)

    def test_alm2wlm_axisym(self):
        
        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5

        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)

        wlms_exp = [np.asarray([1, 2, 5]), 
                    0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10])]
        
        np.testing.assert_array_almost_equal(wlms[0], wlms_exp[0])
        np.testing.assert_array_almost_equal(wlms[1], wlms_exp[1])

    def test_wlm2alm_axisym(self):
        
        ainfo_exp = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = -1
        alm_exp = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)

        wlms = [np.asarray([1, 2, 5]), 
                -1 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10])]
        winfos = [sharp.alm_info(lmax=1), sharp.alm_info(lmax=3)]
        
        alm, ainfo = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell)

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_wlm2alm_axisym_2d(self):
        
        ainfo_exp = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = -1
        alm_exp = np.zeros((2, 10), dtype=np.complex128)
        alm_exp[0] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex128)
        alm_exp[1] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex128) * 2
        
        wlms = [np.asarray([[1, 2, 5], [2, 4, 10]]), 
                -1 * np.asarray([[0, 0, 3, 4, 0, 6, 7, 8, 9, 10],
                                 [0, 0, 6, 8, 0, 12, 14, 16, 18, 20]])]
        winfos = [sharp.alm_info(lmax=1), sharp.alm_info(lmax=3)]
        
        alm, ainfo = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell)

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_wlm2alm_axisym_err(self):
        
        ainfo = sharp.alm_info(lmax=2)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = -1

        wlms = [np.asarray([1, 2, 5]), 
                -1 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10])]
        winfos = [sharp.alm_info(lmax=1), sharp.alm_info(lmax=3)]
        
        self.assertRaises(ValueError, alm_utils.wlm2alm_axisym, wlms, winfos, w_ell,
                          **dict(ainfo=ainfo))

        ainfo = sharp.alm_info(lmax=3, stride=2)

        self.assertRaises(NotImplementedError, alm_utils.wlm2alm_axisym, wlms, winfos, w_ell,
                          **dict(ainfo=ainfo))

    def test_alm2wlm_axisym_roundtrip(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = -1

        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)
        
        alm_out, ainfo_out = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell)

        np.testing.assert_array_almost_equal(alm_out, alm)



