import unittest
import numpy as np

from pixell import sharp

from optweight import alm_utils, wavtrans, map_utils

class TestAlmUtils(unittest.TestCase):

    def test_trunc_alm(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 1
        alm_new_exp = np.asarray([1, 2, 4], dtype=np.complex128)

        alm_new, ainfo_new = alm_utils.trunc_alm(alm, ainfo, lmax_new)

        np.testing.assert_equal(alm_new, alm_new_exp)

    def test_trunc_alm_2d(self):

        alm = np.zeros((2, 6), dtype=np.complex128)
        alm[0] = np.asarray([1, 2, 3, 4, 5, 6])
        alm[1] = np.asarray([1, 2, 3, 4, 5, 6]) * 2

        ainfo = sharp.alm_info(lmax=2)
        lmax_new = 1
        alm_new_exp = np.zeros((2, 3), dtype=np.complex128)
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
        w_ell = np.zeros((2, 4), dtype=np.float32)
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5

        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)

        wlms_exp = [np.asarray([1, 2, 5]),
                    0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10])]

        np.testing.assert_array_almost_equal(wlms[0], wlms_exp[0])
        np.testing.assert_array_almost_equal(wlms[1], wlms_exp[1])

        self.assertIs(wlms[0].dtype.type, np.complex64)
        self.assertIs(wlms[1].dtype.type, np.complex64)

    def test_alm2wlm_axisym_nd(self):

        ainfo = sharp.alm_info(lmax=3)
        alm = np.zeros((2, 3, ainfo.nelem), dtype=np.complex64)
        alm[:,:] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        w_ell = np.zeros((2, 4), dtype=np.float32)
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5

        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)

        wlms_exp = []
        wlm_0 = np.zeros((2, 3, 3), dtype=alm.dtype)
        wlm_0[:,:] = np.asarray([1, 2, 5])
        wlm_1 = np.zeros((2, 3, ainfo.nelem), dtype=alm.dtype)
        wlm_1[:,:] = 0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10])

        wlms_exp.append(wlm_0)
        wlms_exp.append(wlm_1)

        np.testing.assert_array_almost_equal(wlms[0], wlms_exp[0])
        np.testing.assert_array_almost_equal(wlms[1], wlms_exp[1])

        self.assertIs(wlms[0].dtype.type, np.complex64)
        self.assertIs(wlms[1].dtype.type, np.complex64)

    def test_alm2wlm_axisym_lmaxs(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4), dtype=np.float32)
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5

        lmaxs = np.asarray([2, 3])

        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell, lmaxs=lmaxs)

        wlms_exp = [np.asarray([1, 2, 0, 5, 0, 0]),
                    0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10])]

        np.testing.assert_array_almost_equal(wlms[0], wlms_exp[0])
        np.testing.assert_array_almost_equal(wlms[1], wlms_exp[1])

        self.assertIs(wlms[0].dtype.type, np.complex64)
        self.assertIs(wlms[1].dtype.type, np.complex64)

    def test_alm2wlm_axisym_err(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 5), dtype=np.float32) # Does not match lmax.
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5

        self.assertRaises(ValueError, alm_utils.alm2wlm_axisym, alm, ainfo, w_ell)

    def test_wlm2alm_axisym(self):

        ainfo_exp = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4), dtype=np.float32)
        w_ell[0,:2] = 1
        w_ell[1,2:] = -1
        alm_exp = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)

        wlms = [np.asarray([1, 2, 5], dtype=np.complex64),
                -1 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10], dtype=np.complex64)]
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
        wlms = [w.astype(np.complex128) for w in wlms]
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
        w_ell = np.zeros((2, 4), dtype=np.float32)
        w_ell[0,:2] = 1
        w_ell[1,2:] = -1

        wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell)

        alm_out, ainfo_out = alm_utils.wlm2alm_axisym(wlms, winfos, w_ell)

        np.testing.assert_array_almost_equal(alm_out, alm)

    def test_add_to_alm(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)

        blm = np.asarray([1, 2, 3, 4, 5j, 6j, 7j, 8j, 9j, 10j], dtype=np.complex64)
        binfo = sharp.alm_info(lmax=3)

        alm_utils.add_to_alm(alm, blm, ainfo, binfo)

        alm_exp = np.asarray(
            [2, 4, 6, 8, 5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j, 9 + 9j, 10 + 10j],
            dtype=np.complex64)

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_add_to_alm_diff_size(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)

        blm = np.asarray([1, 2, 3, 4j, 5j, 6j], dtype=np.complex64)
        binfo = sharp.alm_info(lmax=2)

        alm_utils.add_to_alm(alm, blm, ainfo, binfo)

        alm_exp = np.asarray([2, 4, 6, 4, 5 + 4j, 6 + 5j, 7, 8 + 6j, 9, 10],
                             dtype=np.complex64)

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_add_to_alm_overwrite(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)

        blm = np.asarray([1, 2, 3, 4j, 5j, 6j], dtype=np.complex64)
        binfo = sharp.alm_info(lmax=2)

        alm_utils.add_to_alm(alm, blm, ainfo, binfo, overwrite=True)

        alm_exp = np.asarray([1, 2, 3, 4, 4j, 5j, 7, 6j, 9, 10],
                             dtype=np.complex64)

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_add_to_alm_err(self):

        alm = np.asarray([1, 2, 3, 4j, 5j, 6j], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=2)

        blm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        binfo = sharp.alm_info(lmax=3)

        self.assertRaises(ValueError, alm_utils.add_to_alm, alm, blm, ainfo, binfo)

    def test_rand_alm_pix(self):

        lmax = 3
        ainfo = sharp.alm_info(lmax=lmax)

        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        npol = 3
        spin = [0, 2]
        cov_pix = np.ones((npol, npol, minfo.npix)) * np.eye(npol)[:,:,np.newaxis]

        rand_alm = alm_utils.rand_alm_pix(cov_pix, ainfo, minfo, spin)

        self.assertEqual(rand_alm.shape, (npol, ainfo.nelem))
        self.assertEqual(rand_alm.dtype, np.complex128)
        self.assertTrue(np.any(rand_alm))

    def test_rand_alm_wav(self):

        lmax = 3
        ainfo = sharp.alm_info(lmax=lmax)

        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = map_utils.MapInfo.map_info_gauss_legendre(nrings, nphi)

        npol = 3
        spin = [0, 2]

        w_ell = np.zeros((2, lmax + 1))
        w_ell[0] = [1, 1, 0, 0]
        w_ell[1] = [0, 0, 1, 1]
                         
        cov_wav = wavtrans.Wav(2)

        # Add first map.
        m_arr = np.ones((npol, minfo.npix))
        index = (0, 0)

        cov_wav.add(index, m_arr, minfo)

        # Add second map.
        index = (1, 1)
        cov_wav.add(index, m_arr, minfo)

        rand_alm = alm_utils.rand_alm_wav(cov_wav, ainfo, w_ell, spin)

        self.assertEqual(rand_alm.shape, (npol, ainfo.nelem))
        self.assertEqual(rand_alm.dtype, np.complex128)
        self.assertTrue(np.any(rand_alm))

    def test_contract_almxblm(self):
        
        alm = np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j, 4j])
        blm = np.asarray([2, 2, 2, 2, 3, 3, 3, -4j, -4j, 5])

        ans_exp = -40

        ans = alm_utils.contract_almxblm(alm, blm)

        self.assertEqual(ans, ans_exp)

    def test_contract_almxblm_2d(self):
        
        alm = np.ones((3, 10), dtype=np.complex128)
        blm = np.ones((3, 10), dtype=np.complex128)

        alm *= np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j, 4j])
        blm *= np.asarray([2, 2, 2, 2, 3, 3, 3, -4j, -4j, 5])

        ans_exp = -120

        ans = alm_utils.contract_almxblm(alm, blm)

        self.assertEqual(ans, ans_exp)

    def test_contract_almxblm_3d(self):
        
        alm = np.ones((2, 3, 10), dtype=np.complex128)
        blm = np.ones((2, 3, 10), dtype=np.complex128)

        alm *= np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j, 4j])
        blm *= np.asarray([2, 2, 2, 2, 3, 3, 3, -4j, -4j, 5])

        ans_exp = -240

        ans = alm_utils.contract_almxblm(alm, blm)

        self.assertEqual(ans, ans_exp)

    def test_contract_almxblm_err(self):
        
        alm = np.ones((10), dtype=np.complex128)
        blm = np.ones((11), dtype=np.complex128)

        self.assertRaises(ValueError, alm_utils.contract_almxblm, alm, blm)
