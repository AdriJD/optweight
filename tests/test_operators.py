import unittest
import numpy as np

from pixell import sharp

from optweight import operators

class TestOperators(unittest.TestCase):
    
    def test_MatVecAlm_init(self):

        self.assertRaises(TypeError, operators.MatVecAlm)

    def test_EllMatVecAlm(self):

        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3, 3))
        icov[0,0] = 2
        icov[1,1] = 4
        icov[2,2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= 2
        alm_exp[1] *= 4
        alm_exp[2] *= 9

        matvec_alm = operators.EllMatVecAlm(ainfo, icov)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_EllMatVecAlm_sqrt(self):

        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3, 3))
        icov[0,0] = 2
        icov[1,1] = 4
        icov[2,2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= np.sqrt(2)
        alm_exp[1] *= np.sqrt(4)
        alm_exp[2] *= np.sqrt(9)

        matvec_alm = operators.EllMatVecAlm(ainfo, icov, power=0.5)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_ell_alm_diag(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3))
        icov[0] = 2
        icov[1] = 4
        icov[2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= 2
        alm_exp[1] *= 4
        alm_exp[2] *= 9

        matvec_alm = operators.EllMatVecAlm(ainfo, icov)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_ell_alm_diag_sqrt(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3))
        icov[0] = 2
        icov[1] = 4
        icov[2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= np.sqrt(2)
        alm_exp[1] *= np.sqrt(4)
        alm_exp[2] *= np.sqrt(9)

        matvec_alm = operators.EllMatVecAlm(ainfo, icov, power=0.5)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_ell_alm_sqrt_inplace(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])

        icov = np.zeros((3, 3, 3))
        icov[0,0] = 2
        icov[1,1] = 4
        icov[2,2] = 9

        alm_exp = alm.copy()
        alm_exp[0] *= np.sqrt(2)
        alm_exp[1] *= np.sqrt(4)
        alm_exp[2] *= np.sqrt(9)

        matvec_alm = operators.EllMatVecAlm(ainfo, icov, power=0.5, inplace=True)
        alm_out = matvec_alm(alm)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
        np.testing.assert_array_equal(alm, alm_exp)
        self.assertTrue(np.shares_memory(alm, alm_out))

    def test_matvec_pow_pix_alm(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])
        spin = [0, 2]
        power = 1

        minfo = sharp.map_info_gauss_legendre(3) # (lmax + 1).

        # This function should do Yt icov Y. So if we make icov = W, where
        # W are the quadrature weights, the functions should do Yt W Y = 1.

        icov = np.zeros((3, 3, minfo.nrow, minfo.nphi[0]))
        icov[0,0,:,:] = minfo.weight[:,np.newaxis]
        icov[1,1,:,:] = minfo.weight[:,np.newaxis]
        icov[2,2,:,:] = minfo.weight[:,np.newaxis]
        icov = icov.reshape((3, 3, minfo.npix))

        alm_exp = alm.copy()

        matvec_alm = operators.PixMatVecAlm(
            ainfo, icov, minfo, spin, power=power, inplace=False)
        alm_out = matvec_alm(alm)

        #alm_out = operators.matvec_pow_pix_alm(
        #    alm, ainfo, icov, minfo, spin, power, inplace=False)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)

    def test_matvec_pow_pix_alm_sqrt(self):
        
        alm = np.zeros((3, 6), dtype=np.complex128)
        ainfo = sharp.alm_info(lmax=2)
        alm[0] = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm[1] = np.asarray([0, 0, 6, 0, 10j, 12])
        alm[2] = np.asarray([0, 0, 9, 0, 15j, 18])
        spin = [0, 2]
        power = 0.5

        minfo = sharp.map_info_gauss_legendre(3) # (lmax + 1).

        # This function should do Yt icov Y. So if we make icov = W^2, where
        # W are the quadrature weights, the functions should do Yt sqrt(W^2) Y = 1.

        icov = np.zeros((3, 3, minfo.nrow, minfo.nphi[0]))
        icov[0,0,:,:] = minfo.weight[:,np.newaxis] ** 2
        icov[1,1,:,:] = minfo.weight[:,np.newaxis] ** 2
        icov[2,2,:,:] = minfo.weight[:,np.newaxis] ** 2
        icov = icov.reshape((3, 3, minfo.npix))

        alm_exp = alm.copy()

        matvec_alm = operators.PixMatVecAlm(
            ainfo, icov, minfo, spin, power=power, inplace=False)
        alm_out = matvec_alm(alm)

        #alm_out = operators.matvec_pow_pix_alm(
        #    alm, ainfo, icov, minfo, spin, power, inplace=False)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
    
    def test_matpow(self):

        mat = np.zeros((2, 2, 3))

        mat[0,0] = [1,2,3]
        mat[1,1] = [1,2,3]
        mat[1,0] = 0.1
        mat[0,1] = 0.1

        mat_out = operators._matpow(mat, 0.5)

        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,0], mat_out[...,0]), mat[...,0])
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,1], mat_out[...,1]), mat[...,1])
        np.testing.assert_array_almost_equal(
            np.dot(mat_out[...,2], mat_out[...,2]), mat[...,2])

    def test_matpow_diag(self):

        mat = np.zeros((2, 3))

        mat[0] = [1,2,3]
        mat[1] = [1,2,3]

        mat_out = operators._matpow(mat, 0.5)

        mat_out_exp = np.eye(2)[:,:,np.newaxis] * np.sqrt(mat)

        np.testing.assert_array_almost_equal(mat_out, mat_out_exp)

    def test_full_matrix(self):

        mat = np.zeros((2, 2, 3))

        mat[0,0] = [1,2,3]
        mat[1,1] = [1,2,3]

        mat_out_exp = mat.copy()
        mat_out = operators._full_matrix(mat)

        np.testing.assert_array_almost_equal(mat_out, mat_out_exp)

    def test_full_matrix_diag(self):

        mat = np.zeros((2, 3))

        mat[0] = [1,2,3]
        mat[1] = [1,2,3]

        mat_out = operators._full_matrix(mat)
        mat_out_exp = np.eye(2)[:,:,np.newaxis] * mat

        np.testing.assert_array_almost_equal(mat_out, mat_out_exp)

    def test_full_matrix_err(self):

        mat = np.zeros((3))

        self.assertRaises(ValueError, operators._full_matrix, mat)
