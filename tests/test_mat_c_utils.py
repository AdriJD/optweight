import unittest
import numpy as np

from pixell import utils

from optweight import mat_c_utils

class TestMatCUtils_32(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dtype = np.float32
        cls.ctype = np.complex64
        cls.rtol = 1e-5
    
    def test_eigpow_c(self):

        # First single matrix.
        imat = np.asarray([[5, -4, 0], [-4, 5, 0], [0, 0, 9]], dtype=self.dtype)
        imat_copy = imat.copy()
        omat_exp = np.asarray([[2, -1, 0], [-1, 2, 0], [0, 0, 3]], dtype=self.dtype)

        omat = mat_c_utils.eigpow(imat, 0.5)
        np.testing.assert_allclose(imat, imat_copy)
        np.testing.assert_allclose(omat, omat_exp, rtol=self.rtol)

        omat = mat_c_utils.eigpow(imat, 1)
        np.testing.assert_allclose(omat, imat, rtol=self.rtol)
        self.assertFalse(np.shares_memory(imat, omat))
        
        # Now several matrices.
        imat = np.ones((3, 3, 2), dtype=self.dtype)
        imat *= np.asarray([[5, -4, 0], [-4, 5, 0], [0, 0, 9]])[:,:,np.newaxis]
        imat[:,:,1] *= 2
        imat_copy = imat.copy()        
        omat_exp = np.ones((3, 3, 2), dtype=self.dtype)
        omat_exp *= np.asarray([[2, -1, 0], [-1, 2, 0], [0, 0, 3]])[:,:,np.newaxis]
        omat_exp[:,:,1] *= np.sqrt(2)
        
        omat = mat_c_utils.eigpow(imat, 0.5)
        np.testing.assert_allclose(imat, imat_copy)
        np.testing.assert_allclose(omat, omat_exp, rtol=self.rtol)

        omat = mat_c_utils.eigpow(imat, 1)
        np.testing.assert_allclose(omat, imat, rtol=self.rtol)
        self.assertFalse(np.shares_memory(imat, omat))
                         
    def test_eigpow_c_err(self):

        # Wrong dim.
        imat = np.zeros((1, 3, 3, 2), dtype=self.dtype)
        self.assertRaises(ValueError, mat_c_utils.eigpow, imat, 1)

        # Wrong shape.
        imat = np.zeros((3, 2, 2), dtype=self.dtype)
        self.assertRaises(ValueError, mat_c_utils.eigpow, imat, 1)

        # Wrong dtype.
        imat = np.zeros((3, 2, 2), dtype=self.ctype)
        self.assertRaises(ValueError, mat_c_utils.eigpow, imat, 1)
        
    def test_eigpow_c_lim(self):

        # Test default lim0.
        too_small_value = np.finfo(self.dtype).tiny
        imat = np.asarray(
            [[too_small_value, 0, 0],
             [0, too_small_value, 0],
             [0, 0, too_small_value]],
             dtype=self.dtype)
        omat_exp = np.zeros_like(imat)
        omat = mat_c_utils.eigpow(imat, 1)
            
        np.testing.assert_allclose(omat, omat_exp, rtol=self.rtol)

        # Test default lim.
        imat = np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1e-10]], dtype=self.dtype)
        omat_exp = np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=self.dtype)        
        omat = mat_c_utils.eigpow(imat, 1)
        
        np.testing.assert_allclose(omat, omat_exp, rtol=self.rtol)

        # Test with single negative eigenvalue.
        imat = np.asarray(
            [[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=self.dtype)
        omat_exp = np.asarray(
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=self.dtype)        
        omat = mat_c_utils.eigpow(imat, -1)
        
        np.testing.assert_allclose(omat, omat_exp, rtol=self.rtol)
        
class TestMatCUtils_64(TestMatCUtils_32):

    @classmethod
    def setUpClass(cls):
        cls.dtype = np.float64
        cls.ctype = np.complex128
        cls.rtol = 1e-7
        
