import unittest
import numpy as np

from optweight import type_utils

class TestTypeUtils(unittest.TestCase):

    def test_to_complex(self):

        self.assertIs(type_utils.to_complex(np.float32), np.complex64)
        self.assertIs(type_utils.to_complex(np.float64), np.complex128)
        self.assertIs(type_utils.to_complex(np.float128), np.complex256)

        self.assertRaises(ValueError, type_utils.to_complex, np.complex128)

    def test_to_real(self):

        self.assertIs(type_utils.to_real(np.complex64), np.float32)
        self.assertIs(type_utils.to_real(np.complex128), np.float64)
        self.assertIs(type_utils.to_real(np.complex256), np.float128)

        self.assertRaises(ValueError, type_utils.to_real, np.float64)

    def test_is_seq_of_seq(self):
        
        self.assertTrue(type_utils.is_seq_of_seq([[1], [2]]))
        self.assertFalse(type_utils.is_seq_of_seq([[1], 2]))
        self.assertTrue(type_utils.is_seq_of_seq(((1,), (2,))))
        self.assertTrue(type_utils.is_seq_of_seq(np.ones((2,2))))
