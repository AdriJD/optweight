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

        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])

    def test_matpow_inplace(self):

        mat = np.zeros((2, 2, 3))

        mat[0,0] = [1,2,3]
        mat[1,1] = [1,2,3]
        mat[1,0] = 0.1
        mat[0,1] = 0.1

        mat_in = mat.copy()
        mat_out = mat_utils.matpow(mat, 0.5, inplace=True)
        self.assertTrue(np.shares_memory(mat, mat_out))
        
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,0], mat_out[...,0]), mat_in[...,0])
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,1], mat_out[...,1]), mat_in[...,1])
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,2], mat_out[...,2]), mat_in[...,2])

        # Test if square root is symmetric. Required, see astro-ph/0608007.
        np.testing.assert_array_almost_equal(
            mat_out, np.transpose(mat_out, (1, 0, 2)))

        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])

    def test_matpow_axes(self):

        mat = np.ones((3, 2, 2))
        mat *= np.eye(2)[np.newaxis,:,:] * 2

        mat_out_exp = np.ones_like(mat)
        mat_out_exp *= np.eye(2)[np.newaxis,:,:] * 0.5
        
        mat_out = mat_utils.matpow(mat, -1, axes=[-2, -1])

        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])
        
        # Inplace.
        mat_out = mat_utils.matpow(mat, -1, axes=[-2, -1], inplace=True)
        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])
        self.assertTrue(np.shares_memory(mat_out, mat))

        # 2d Matrix.
        mat = np.ones((2, 2))
        mat *= np.eye(2) * 2

        mat_out_exp = np.ones_like(mat)
        mat_out_exp *= np.eye(2) * 0.5
        
        mat_out = mat_utils.matpow(mat, -1, axes=[-2, -1])

        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])

        # Inplace.
        mat_out = mat_utils.matpow(mat, -1, axes=[-2, -1], inplace=True)
        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])
        self.assertTrue(np.shares_memory(mat_out, mat))

    def test_matpow_axes_2d(self):

        mat = np.ones((6, 6, 5))
        mat *= np.eye(6)[:,:,np.newaxis] * 2
        mat = mat.reshape(2, 3, 2, 3, 5)

        mat_out_exp = mat.copy() * 0.25
        
        mat_out = mat_utils.matpow(mat, -1, axes=[[0, 1], [2, 3]])
        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])

        # Inplace.
        mat_out = mat_utils.matpow(mat, -1, axes=[[0, 1], [2, 3]], inplace=True)
        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])
        self.assertTrue(np.shares_memory(mat_out, mat))

        # Different order.
        mat = np.ones((5, 6, 6))
        mat *= np.eye(6)[np.newaxis, :,:] * 2
        mat = mat.reshape(5, 2, 3, 2, 3)

        mat_out_exp = mat.copy() * 0.25
        
        mat_out = mat_utils.matpow(mat, -1, axes=[[1, 2], [3, 4]])
        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])

        # Inplace.
        mat_out = mat_utils.matpow(mat, -1, axes=[[1, 2], [3, 4]], inplace=True)
        np.testing.assert_allclose(mat_out, mat_out_exp)
        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])
        self.assertTrue(np.shares_memory(mat_out, mat))

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

        self.assertTrue(mat_out.flags['C_CONTIGUOUS'])

    def test_matpow_diag_pinv(self):
        
        # Check if non PSD diagonal matrix is consistent.
        mat_full = np.ones((3, 3, 1))
        mat_full[:,:,0] = [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
        
        mat_diag = np.ones((3, 1))
        mat_diag[1] = 0

        mat_full_out = mat_utils.matpow(mat_full, -1)
        mat_diag_out = mat_utils.matpow(mat_diag, -1)

        np.testing.assert_allclose(mat_full_out, mat_diag_out)
        
        mat_diag_out_diag = mat_utils.matpow(mat_diag, -1, return_diag=True)        
        np.testing.assert_allclose(mat_diag_out_diag, mat_diag)

        # Try matrix with negative eigenvalue
        mat_full = np.ones((3, 3, 1))
        mat_full[:,:,0] = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        
        mat_diag = np.ones((3, 1))
        mat_diag[1] = -1

        mat_full_out = mat_utils.matpow(mat_full, -1)
        mat_diag_out = mat_utils.matpow(mat_diag, -1)

        np.testing.assert_allclose(mat_full_out, mat_diag_out)

        mat_diag_out_diag = mat_utils.matpow(mat_diag, -1, return_diag=True)        
        mat_diag_out_diag_exp = np.ones((3, 1))
        mat_diag_out_diag_exp[1] = 0
        np.testing.assert_allclose(mat_diag_out_diag, mat_diag_out_diag_exp)

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
                
        map0_square = np.einsum('ijk, jlk -> ilk',
                             m_wav_sqrt.maps[0,0], m_wav_sqrt.maps[0,0])

        map1_square = np.einsum('ijk, jlk -> ilk',
                             m_wav_sqrt.maps[1,1], m_wav_sqrt.maps[1,1])
        
        np.testing.assert_allclose(map0_square, m_wav.maps[0,0])
        np.testing.assert_allclose(map1_square, m_wav.maps[1,1])

        # Test if matpow kwarg does nothing in this case.
        m_wav_sqrt2 = mat_utils.wavmatpow(m_wav, power, return_diag=True)
        np.testing.assert_allclose(m_wav_sqrt.maps[0,0], m_wav_sqrt2.maps[0,0])
        np.testing.assert_allclose(m_wav_sqrt.maps[1,1], m_wav_sqrt2.maps[1,1])

        # Inplace.
        m_wav2 = wavtrans.Wav(2)
        m_wav2.add(index1, m_arr1, minfo1)
        m_wav2.add(index2, m_arr2, minfo2)
        
        m_wav_sqrt3 = mat_utils.wavmatpow(m_wav2, power, inplace=True)
        np.testing.assert_allclose(m_wav_sqrt3.maps[0,0], m_wav_sqrt.maps[0,0])
        np.testing.assert_allclose(m_wav_sqrt3.maps[1,1], m_wav_sqrt.maps[1,1])
        np.testing.assert_allclose(m_wav_sqrt3.maps[0,0], m_wav2.maps[0,0])
        np.testing.assert_allclose(m_wav_sqrt3.maps[1,1], m_wav2.maps[1,1])

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
        # We can tolerate some negative values as long as they are close to zero.
        self.assertTrue(np.all(e >= 0) | np.all(np.abs(e[e<0]) < 1e-14))

        out_exp = mat.copy()
        out_exp[:,:,0] = np.asarray([[1.05, 1, 1.05],[1, 1, 1],[1.05, 1, 1.05]])
        np.testing.assert_allclose(out, out_exp)
        self.assertFalse(np.shares_memory(out, mat))

        # Inplace.
        out = mat_utils.get_near_psd(mat, inplace=True)
        np.testing.assert_allclose(out, out_exp)
        self.assertTrue(np.shares_memory(out, mat))

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
        np.testing.assert_allclose(out, out_exp, rtol=1e-6)

        self.assertEqual(out.dtype, np.float32)

    def test_symm2triu(self):
        
        # Use a non-symmetric matrix for testing.
        mat = np.arange(9, dtype=np.float32).reshape(3, 3)
        
        mat_triu = mat_utils.symm2triu(mat, [0, 1])

        mat_triu_exp = np.asarray([0, 1, 2, 4, 5, 8], dtype=np.float32)

        np.testing.assert_allclose(mat_triu, mat_triu_exp)
        self.assertEqual(mat_triu.dtype, mat.dtype)
        self.assertTrue(mat_triu.flags['C_CONTIGUOUS'])
        self.assertFalse(np.shares_memory(mat_triu, mat))

    def test_symm2triu_nd(self):
        
        # Use a non-symmetric matrix for testing.
        mat = np.ones((3, 2, 2, 4), dtype=np.float32)
        mat[:] = np.arange(4, dtype=np.float32).reshape(2, 2)[np.newaxis,:,:,np.newaxis]
                
        mat_triu = mat_utils.symm2triu(mat, [1, 2])

        mat_triu_exp = np.ones((3, 3, 4), dtype=np.float32)
        mat_triu_exp *= np.asarray([0, 1, 3], dtype=np.float32)[np.newaxis,:,np.newaxis]

        np.testing.assert_allclose(mat_triu, mat_triu_exp)
        self.assertEqual(mat_triu.dtype, mat.dtype)
        self.assertTrue(mat_triu.flags['C_CONTIGUOUS'])
        self.assertFalse(np.shares_memory(mat_triu, mat))

        # Another with negative indices.
        mat_triu = mat_utils.symm2triu(mat, [-2, -3])
        np.testing.assert_allclose(mat_triu, mat_triu_exp)
        self.assertEqual(mat_triu.dtype, mat.dtype)
        self.assertTrue(mat_triu.flags['C_CONTIGUOUS'])
        self.assertFalse(np.shares_memory(mat_triu, mat))

        mat_triu = mat_utils.symm2triu(mat, [-3, -2])
        np.testing.assert_allclose(mat_triu, mat_triu_exp)
        self.assertEqual(mat_triu.dtype, mat.dtype)
        self.assertTrue(mat_triu.flags['C_CONTIGUOUS'])
        self.assertFalse(np.shares_memory(mat_triu, mat))
        
        # Another with mixed indices.
        mat_triu = mat_utils.symm2triu(mat, [-3, 2])
        np.testing.assert_allclose(mat_triu, mat_triu_exp)
        self.assertEqual(mat_triu.dtype, mat.dtype)
        self.assertTrue(mat_triu.flags['C_CONTIGUOUS'])
        self.assertFalse(np.shares_memory(mat_triu, mat))

    def test_symm2triu_err(self):
        
        # Use a non-symmetric matrix for testing.
        mat = np.ones((3, 2, 2, 4), dtype=np.float32)
        mat[:] = np.arange(4, dtype=np.float32).reshape(2, 2)[np.newaxis,:,:,np.newaxis]

        # Non adjacent.
        self.assertRaises(ValueError, mat_utils.symm2triu, mat, [0, 2])

        # Not NxN.
        mat = np.ones((3, 2, 1, 4), dtype=np.float32)
        self.assertRaises(ValueError, mat_utils.symm2triu, mat, [0, 2])

    def test_triu2symm(self):

        mat_triu = np.asarray([0, 1, 2, 4, 5, 8], dtype=np.float32)        

        mat_symm = mat_utils.triu2symm(mat_triu, axis=0)
        mat_symm_exp = np.asarray([[0, 1, 2], [1, 4, 5], [2, 5, 8]],
                                  dtype=np.float32)

        np.testing.assert_allclose(mat_symm, mat_symm_exp)
        self.assertEqual(mat_symm.dtype, mat_triu.dtype)
        self.assertTrue(mat_symm.flags['C_CONTIGUOUS'])
        self.assertFalse(np.shares_memory(mat_symm, mat_triu))

    def test_triu2symm_nd(self):

        mat_triu = np.ones((3, 3, 4), dtype=np.float32)
        mat_triu[:] = np.asarray([0, 1, 2])[np.newaxis,:,np.newaxis]

        mat_symm = mat_utils.triu2symm(mat_triu, axis=1)

        mat_symm_exp = np.ones((3, 2, 2, 4), dtype=np.float32)
        mat_symm_exp *= np.asarray([[0, 1], [1, 2]], 
                                   dtype=np.float32)[np.newaxis,:,:,np.newaxis]

        np.testing.assert_allclose(mat_symm, mat_symm_exp)
        self.assertEqual(mat_symm.dtype, mat_triu.dtype)
        self.assertTrue(mat_symm.flags['C_CONTIGUOUS'])
        self.assertFalse(np.shares_memory(mat_symm, mat_triu))

    def test_triu2symm_err(self):

        # Wrong number of elements.
        mat_triu = np.asarray([0, 1, 2, 4, 5], dtype=np.float32)        

        self.assertRaises(ValueError, mat_utils.triu2symm, mat_triu, axis=0)

    def test_atleast_nd(self):
        
        mat = np.ones((3, 3), dtype=np.float32)

        mat_out = mat_utils.atleast_nd(mat, 2)        
        self.assertEqual(mat_out.shape, mat.shape)
        self.assertTrue(np.shares_memory(mat_out, mat))

        mat_out = mat_utils.atleast_nd(mat, 3)        
        self.assertEqual(mat_out.shape, (1, 3, 3))
        self.assertTrue(np.shares_memory(mat_out, mat))

        mat_out = mat_utils.atleast_nd(mat, 4)        
        self.assertEqual(mat_out.shape, (1, 1, 3, 3))
        self.assertTrue(np.shares_memory(mat_out, mat))

        mat_out = mat_utils.atleast_nd(mat, 1)
        self.assertEqual(mat_out.shape, (3, 3))
        self.assertTrue(np.shares_memory(mat_out, mat))

        mat_out = mat_utils.atleast_nd(mat, 4, append=True)        
        self.assertEqual(mat_out.shape, (3, 3, 1, 1))
        self.assertTrue(np.shares_memory(mat_out, mat))
    
    def test_flattened_view(self):
        
        mat = np.arange(120)
        mat = mat.reshape(3, 4, 10)
        
        mat_view, flat_ax = mat_utils.flattened_view(mat, [[0, 1]],
                                                     return_flat_axes=True)
        mat_exp = mat.reshape(12, 10)

        self.assertEqual(mat_view.shape, (12, 10))
        self.assertTrue(np.shares_memory(mat_view, mat))
        np.testing.assert_array_equal(mat_view, mat_exp)        
        self.assertEqual(flat_ax, [0])

        # 2 inputs to axes.
        mat = mat.reshape(3, 4, 5, 2)        
        mat_view, flat_ax = mat_utils.flattened_view(mat, [[0, 1], [2, 3]],
                                            return_flat_axes=True)

        self.assertEqual(mat_view.shape, (12, 10))
        self.assertTrue(np.shares_memory(mat_view, mat))
        np.testing.assert_array_equal(mat_view, mat_exp)
        self.assertEqual(flat_ax, [0, 1])
        
        # Weird order.
        mat_view, flat_ax = mat_utils.flattened_view(mat, [[2, 3], [1, 0]], 
                                                     return_flat_axes=True)

        self.assertEqual(mat_view.shape, (12, 10))
        self.assertTrue(np.shares_memory(mat_view, mat))
        np.testing.assert_array_equal(mat_view, mat_exp)
        self.assertEqual(flat_ax, [0, 1])

        # Negative axes.
        mat_view, flat_ax = mat_utils.flattened_view(mat, [[0, 1], [-2, -1]],
                                                     return_flat_axes=True)
        self.assertEqual(mat_view.shape, (12, 10))
        self.assertTrue(np.shares_memory(mat_view, mat))
        np.testing.assert_array_equal(mat_view, mat_exp)
        self.assertEqual(flat_ax, [0, 1])

        # Repeats.
        self.assertRaises(ValueError, mat_utils.flattened_view, mat, [[0, 0], [2, 3]])

        # Non-contiguous axes.
        mat_view, flat_ax = mat_utils.flattened_view(mat, [[0, 2], [1, 3]], 
                                                     return_flat_axes=True)
        mat_exp = mat.reshape(15, 8)
        self.assertEqual(mat_view.shape, (15, 8))
        self.assertTrue(np.shares_memory(mat_view, mat))
        np.testing.assert_array_equal(mat_view, mat_exp)
        self.assertEqual(flat_ax, [0, 1])

        # 5D case.
        mat = mat.reshape(3, 2, 2, 5, 2)        
        mat_view, flat_ax = mat_utils.flattened_view(mat, [[0, 1], [3, 4]],
                                            return_flat_axes=True)
        mat_exp = mat.reshape(6, 2, 10)

        self.assertEqual(mat_view.shape, (6, 2, 10))
        self.assertTrue(np.shares_memory(mat_view, mat))
        np.testing.assert_array_equal(mat_view, mat_exp)
        self.assertEqual(flat_ax, [0, 2])
        
    def test_eigpow(self):
        
        mat = np.ones((3, 3, 10), dtype=np.float32)
        mat *= np.eye(3)[:,:,np.newaxis] * 2
        mat_exp = mat * 0.25

        mat_out = mat_utils._eigpow(mat, -1, [0, 1], np.float64)
        np.testing.assert_allclose(mat_out, mat_exp)
        self.assertEqual(mat_out.dtype, mat.dtype)
        self.assertFalse(np.shares_memory(mat_out, mat))

        mat_out = mat_utils._eigpow(mat, -1, [0, 1], np.float64, chunksize=4)
        np.testing.assert_allclose(mat_out, mat_exp)
        self.assertEqual(mat_out.dtype, mat.dtype)
        self.assertFalse(np.shares_memory(mat_out, mat))

        mat_out = mat_utils._eigpow(mat, -1, [0, 1], np.float64, inplace=True)
        np.testing.assert_allclose(mat_out, mat_exp)
        self.assertEqual(mat_out.dtype, mat.dtype)
        self.assertTrue(np.shares_memory(mat_out, mat))        

        # Now try a different shape. Can't be flattened without copy, but 
        # should still allow inplace operations.
        mat = np.ones((2, 3, 3, 5), dtype=np.float32)
        mat *= np.eye(3)[np.newaxis,:,:,np.newaxis] * 2
        mat_exp = mat * 0.25

        mat_out = mat_utils._eigpow(mat, -1, [1, 2], np.float64)
        np.testing.assert_allclose(mat_out, mat_exp)
        self.assertEqual(mat_out.dtype, mat.dtype)
        self.assertFalse(np.shares_memory(mat_out, mat))

        mat_out = mat_utils._eigpow(mat, -1, [1, 2], np.float64, chunksize=4)
        np.testing.assert_allclose(mat_out, mat_exp)
        self.assertEqual(mat_out.dtype, mat.dtype)
        self.assertFalse(np.shares_memory(mat_out, mat))

        mat_out = mat_utils._eigpow(mat, -1, [1, 2], np.float64, inplace=True)
        np.testing.assert_allclose(mat_out, mat_exp)
        self.assertEqual(mat_out.dtype, mat.dtype)
        self.assertTrue(np.shares_memory(mat_out, mat))        
