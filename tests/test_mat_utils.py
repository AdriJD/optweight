import unittest
import numpy as np

from pixell import sharp

from optweight import mat_utils, wavtrans

class TestMatUtils(unittest.TestCase):

    def test_matpow(self):

        mat = np.zeros((2, 2, 3))

        mat[0,0] = [1,2,3]
        mat[1,1] = [1,2,3]
        mat[1,0] = 0.1
        mat[0,1] = 0.1

        mat_out = mat_utils.matpow(mat, 0.5)

        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,0], mat_out[...,0]), mat[...,0])
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,1], mat_out[...,1]), mat[...,1])
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,2], mat_out[...,2]), mat[...,2])

        # Test if square root is symmetric. Required, see astro-ph/0608007.
        np.testing.assert_array_almost_equal(
            mat_out, np.transpose(mat_out, (1, 0, 2)))

    def test_matpow_minus(self):

        mat = np.zeros((2, 2, 3))

        mat[0,0] = [1,2,3]
        mat[1,1] = [1,2,3]
        mat[1,0] = 0.1
        mat[0,1] = 0.1

        mat_out = mat_utils.matpow(mat, -0.5)

        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,0], mat_out[...,0]), 
            np.linalg.inv(mat[...,0]))
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,1], mat_out[...,1]), 
            np.linalg.inv(mat[...,1]))
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,2], mat_out[...,2]),
            np.linalg.inv(mat[...,2]))

        # Test if minus square root is symmetric. Required, see astro-ph/0608007.
        np.testing.assert_array_almost_equal(
            mat_out, np.transpose(mat_out, (1, 0, 2)))

        # Test if return_diag does not nothing in this case
        # (like it should because input is dense).
        mat_out2 = mat_utils.matpow(mat, -0.5, return_diag=True)        

        np.testing.assert_allclose(mat_out, mat_out2)

    def test_matpow_diag(self):

        mat = np.zeros((2, 3))

        mat[0] = [1,2,3]
        mat[1] = [1,2,3]

        mat_out = mat_utils.matpow(mat, 0.5)

        mat_out_exp = np.eye(2)[:,:,np.newaxis] * np.sqrt(mat)
        np.testing.assert_array_almost_equal(mat_out, mat_out_exp)

        # Test return_diag.
        mat_out = mat_utils.matpow(mat, 0.5, return_diag=True)

        mat_out_exp = np.sqrt(mat)
        np.testing.assert_array_almost_equal(mat_out, mat_out_exp)

    def test_full_matrix(self):

        mat = np.zeros((2, 2, 3))

        mat[0,0] = [1,2,3]
        mat[1,1] = [1,2,3]

        mat_out_exp = mat.copy()
        mat_out = mat_utils.full_matrix(mat)

        np.testing.assert_array_almost_equal(mat_out, mat_out_exp)

    def test_full_matrix_diag(self):

        mat = np.zeros((2, 3))

        mat[0] = [1,2,3]
        mat[1] = [1,2,3]

        mat_out = mat_utils.full_matrix(mat)
        mat_out_exp = np.eye(2)[:,:,np.newaxis] * mat

        np.testing.assert_array_almost_equal(mat_out, mat_out_exp)

    def test_full_matrix_err(self):

        mat = np.zeros((3))

        self.assertRaises(ValueError, mat_utils.full_matrix, mat)

    def test_wavmatpow(self):

        power = 0.5
        lmax = 3
        npol = 3
        ainfo = sharp.alm_info(lmax=lmax)
        spin = [0, 2]
        w_ell = np.zeros((2, lmax + 1))
        lmaxs = [2, lmax]
        w_ell[0,:lmaxs[0]+1] = 1
        w_ell[1,lmaxs[0]+1:] = 2
        m_wav = wavtrans.Wav(2)

        # Add first map.
        minfo1 = sharp.map_info_gauss_legendre(lmaxs[0] + 1)
        m_arr1 = np.ones((npol, npol, minfo1.npix)) + 10 * \
                 np.eye(3)[:,:,np.newaxis]
        index1 = (0, 0)
        m_wav.add(index1, m_arr1, minfo1)

        # Add second map.
        minfo2 = sharp.map_info_gauss_legendre(lmaxs[1] + 1)
        m_arr2 = np.ones((npol, minfo2.npix)) + 20 * \
                 np.eye(3)[:,:,np.newaxis]
        index2 = (1, 1)
        m_wav.add(index2, m_arr2, minfo2)

        m_wav_sqrt = mat_utils.wavmatpow(m_wav, power)
                
        np.testing.assert_allclose(
            np.einsum('ijk, jlk -> ilk',
                      m_wav_sqrt.maps[0,0], m_wav_sqrt.maps[0,0]),
            m_wav.maps[0,0])

        np.testing.assert_allclose(
            np.einsum('ijk, jlk -> ilk',
                      m_wav_sqrt.maps[1,1], m_wav_sqrt.maps[1,1]),
            m_wav.maps[1,1])

        # Test if matpow kwarg does nothing in this case.
        m_wav_sqrt2 = mat_utils.wavmatpow(m_wav, power, return_diag=True)
        np.testing.assert_allclose(m_wav_sqrt.maps[0,0], m_wav_sqrt2.maps[0,0])
        np.testing.assert_allclose(m_wav_sqrt.maps[1,1], m_wav_sqrt2.maps[1,1])

    def test_get_near_psd(self):

        npol = 3
        nell = 4
        mat = np.ones((npol, npol, 4)) * np.eye(3)[:,:,np.newaxis]        

        # PSD input should stay unchanged.
        out = mat_utils.get_near_psd(mat)
        np.testing.assert_allclose(out, mat)

        # Try non-PSD input.
        mat = np.ones((npol, npol, 4))
        mat[0,2,0] = 1.1
        mat[2,0,0] = 1.1

        out = mat_utils.get_near_psd(mat)

        e, _ = np.linalg.eig(out[:,:,0])
        self.assertTrue(np.all(e >= 0))

        out_exp = mat.copy()
        out_exp[:,:,0] = np.asarray([[1.05, 1, 1.05],[1, 1, 1],[1.05, 1, 1.05]])
        np.testing.assert_allclose(out, out_exp)

    def test_get_near_psd_sd(self):

        npol = 3
        nell = 4
        mat = np.ones((npol, npol, 4)) * np.eye(3)[:,:,np.newaxis]        

        # Try 32 bit non-PSD input.
        mat = np.ones((npol, npol, 4), dtype=np.float32)
        mat[0,2,0] = 1.1
        mat[2,0,0] = 1.1

        out = mat_utils.get_near_psd(mat)

        out_exp = mat.copy()
        out_exp[:,:,0] = np.asarray([[1.05, 1, 1.05],[1, 1, 1],[1.05, 1, 1.05]])
        np.testing.assert_allclose(out, out_exp)

        self.assertEqual(out.dtype, np.float32)
