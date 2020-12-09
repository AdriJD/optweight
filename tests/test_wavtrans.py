import unittest
import numpy as np

from pixell import sharp

from optweight import wavtrans, sht

class TestWavTrans(unittest.TestCase):

    def test_wav_init_vec(self):
        
        indices = np.asarray([3, 4, 6])
        minfos = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [3, 3, 4]])
        preshape = (3,)
        wavvec = wavtrans.Wav(1, indices=indices, minfos=minfos, preshape=preshape)

        self.assertEqual(wavvec.preshape, preshape)
        np.testing.assert_array_almost_equal(wavvec.maps[3], 
                                             np.zeros((3, 4 * 7)))
        np.testing.assert_array_almost_equal(wavvec.maps[4], 
                                             np.zeros((3, 4 * 7)))
        np.testing.assert_array_almost_equal(wavvec.maps[6], 
                                             np.zeros((3, 5 * 9)))
        
        self.assertEqual(wavvec.minfos[3], minfos[0])
        self.assertEqual(wavvec.minfos[4], minfos[1])
        self.assertEqual(wavvec.minfos[6], minfos[2])

        self.assertEqual(wavvec.ndim, 1)
        self.assertEqual(wavvec.dtype, np.float64)
        self.assertEqual(wavvec.shape, (7,))

        self.assertRaises(KeyError, lambda : wavvec.maps[0])
        self.assertRaises(KeyError, lambda : wavvec.minfos[0])

    def test_wav_diag_vec(self):
        
        indices = np.asarray([3, 4, 6])
        minfos = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [3, 3, 4]])
        preshape = (3,)
        wavvec_orig = wavtrans.Wav(1, indices=indices, minfos=minfos, preshape=preshape)

        wavvec_orig.maps[3][:] = 3
        wavvec_orig.maps[4][:] = 4
        wavvec_orig.maps[6][:] = 6

        wavvec = wavvec_orig.diag()

        np.testing.assert_array_almost_equal(wavvec.maps[3], 
                                             3 * np.ones((3, 4 * 7)))
        np.testing.assert_array_almost_equal(wavvec.maps[4], 
                                             4 * np.ones((3, 4 * 7)))
        np.testing.assert_array_almost_equal(wavvec.maps[6], 
                                            6 *  np.ones((3, 5 * 9)))

        self.assertEqual(wavvec.ndim, 1)
        self.assertEqual(wavvec.dtype, np.float64)
        self.assertEqual(wavvec.shape, (7,))

        self.assertRaises(KeyError, lambda : wavvec.maps[0])
        self.assertRaises(KeyError, lambda : wavvec.minfos[0])

        # I expect copies of maps and minfos.
        self.assertFalse(np.shares_memory(wavvec.maps[3], wavvec_orig.maps[3]))
        self.assertFalse(np.shares_memory(wavvec.maps[4], wavvec_orig.maps[4]))
        self.assertFalse(np.shares_memory(wavvec.maps[6], wavvec_orig.maps[6]))
        
        self.assertNotEqual(wavvec.minfos[3], minfos[0])
        self.assertNotEqual(wavvec.minfos[4], minfos[1])
        self.assertNotEqual(wavvec.minfos[6], minfos[2])
        
        # Test if new minfos contain the same info.
        np.testing.assert_almost_equal(wavvec.minfos[3].theta,
                                       wavvec_orig.minfos[3].theta)
        np.testing.assert_almost_equal(wavvec.minfos[3].nphi,
                                       wavvec_orig.minfos[3].nphi)
        np.testing.assert_almost_equal(wavvec.minfos[3].phi0,
                                       wavvec_orig.minfos[3].phi0)
        np.testing.assert_almost_equal(wavvec.minfos[3].offsets,
                                       wavvec_orig.minfos[3].offsets)
        np.testing.assert_almost_equal(wavvec.minfos[3].stride,
                                       wavvec_orig.minfos[3].stride)
        np.testing.assert_almost_equal(wavvec.minfos[3].weight,
                                       wavvec_orig.minfos[3].weight)

    def test_wav_init_vec_add(self):
        
        wavvec = wavtrans.Wav(1)

        self.assertEqual(wavvec.ndim, 1)
        self.assertEqual(wavvec.dtype, np.float64)
        self.assertEqual(wavvec.shape, (0,))

        self.assertEqual(wavvec.maps, {})
        self.assertEqual(wavvec.minfos, {})
        np.testing.assert_array_equal(wavvec.indices, np.zeros((0, 1), dtype=int))

        # Add first map.
        minfo = sharp.map_info_gauss_legendre(3)
        m_arr = np.ones((1, minfo.npix))
        index = 10

        wavvec.add(index, m_arr, minfo)

        self.assertEqual(wavvec.preshape, (1,))
        self.assertEqual(wavvec.ndim, 1)
        self.assertEqual(wavvec.dtype, np.float64)
        self.assertEqual(wavvec.shape, (11,))

        np.testing.assert_array_equal(wavvec.maps[10], m_arr)
        self.assertEqual(wavvec.minfos[10], minfo)
        np.testing.assert_array_equal(wavvec.indices, np.asarray([[10]]))

        # Add second map.
        index = 5

        wavvec.add(index, m_arr, minfo)

        self.assertEqual(wavvec.ndim, 1)
        self.assertEqual(wavvec.dtype, np.float64)
        self.assertEqual(wavvec.shape, (11,))

        np.testing.assert_array_equal(wavvec.maps[5], m_arr)
        np.testing.assert_array_equal(wavvec.maps[10], m_arr)
        self.assertEqual(wavvec.minfos[5], minfo)
        self.assertEqual(wavvec.minfos[10], minfo)
        np.testing.assert_array_equal(wavvec.indices, np.asarray([[5, 10]]).T)

        # Overwrite first map.
        index = 10
        minfo_2 = sharp.map_info_gauss_legendre(5)
        m_arr_2 = np.ones((1, minfo_2.npix))

        wavvec.add(index, m_arr_2, minfo_2)

        self.assertEqual(wavvec.ndim, 1)
        self.assertEqual(wavvec.dtype, np.float64)
        self.assertEqual(wavvec.shape, (11,))

        np.testing.assert_array_equal(wavvec.maps[5], m_arr)
        np.testing.assert_array_equal(wavvec.maps[10], m_arr_2)
        self.assertEqual(wavvec.minfos[5], minfo)
        self.assertEqual(wavvec.minfos[10], minfo_2)
        np.testing.assert_array_equal(wavvec.indices, np.asarray([[5, 10]]).T)

    def test_wav_init_vec_dtype(self):
        
        wavvec = wavtrans.Wav(1, dtype=np.float64)

        # Add map with wrong dtype.
        minfo = sharp.map_info_gauss_legendre(3)
        m_arr = np.ones((1, minfo.npix), dtype=np.float32)
        index = 10

        self.assertRaises(ValueError, wavvec.add, index, m_arr, minfo)

    def test_wav_init_vec_add_preshape(self):
        
        wavvec = wavtrans.Wav(1, preshape=(3,))

        # Add first map.
        minfo = sharp.map_info_gauss_legendre(3)
        m_arr = np.ones((2, minfo.npix))
        index = 10

        wavvec.add(index, m_arr, minfo)

        # First map is allowed to redefine preshape
        self.assertEqual(wavvec.preshape, (2,))

        # Second map is not allowed to do that.
        m_arr = np.ones((3, minfo.npix))
        index = 5
        self.assertRaises(ValueError, wavvec.add, index, m_arr, minfo)

        # But this should work.
        m_arr = np.ones((2, minfo.npix))
        wavvec.add(index, m_arr, minfo)
        np.testing.assert_array_equal(wavvec.maps[5], m_arr)

    def test_wav_init_mat(self):
        
        indices = np.asarray([(0,0), (0,1), (1,1)])
        minfos = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [3, 3, 4]])
        preshape = (3,)
        wavmat = wavtrans.Wav(2, indices=indices, minfos=minfos, preshape=preshape)

        self.assertEqual(wavmat.preshape, preshape)
        np.testing.assert_array_almost_equal(wavmat.maps[0,0], 
                                             np.zeros((3, 4 * 7)))
        np.testing.assert_array_almost_equal(wavmat.maps[0,1], 
                                             np.zeros((3, 4 * 7)))
        np.testing.assert_array_almost_equal(wavmat.maps[1,1], 
                                             np.zeros((3, 5 * 9)))
        
        self.assertEqual(wavmat.minfos[0,0], minfos[0])
        self.assertEqual(wavmat.minfos[0,1], minfos[1])
        self.assertEqual(wavmat.minfos[1,1], minfos[2])

        self.assertEqual(wavmat.ndim, 2)
        self.assertEqual(wavmat.dtype, np.float64)
        self.assertEqual(wavmat.shape, (2, 2))

        self.assertRaises(KeyError, lambda : wavmat.maps[1,0])
        self.assertRaises(KeyError, lambda : wavmat.minfos[1,0])

    def test_wav_diag_mat(self):
        
        indices = np.asarray([(0,0), (0,1), (1,1)])
        minfos = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [3, 3, 4]])
        preshape = (3,)
        wavmat_orig = wavtrans.Wav(2, indices=indices, minfos=minfos, preshape=preshape)

        wavmat_orig.maps[0,0][:] = 3
        wavmat_orig.maps[0,1][:] = 4
        wavmat_orig.maps[1,1][:] = 6

        wavdiag = wavmat_orig.diag()

        np.testing.assert_array_almost_equal(wavdiag.maps[0], 
                                             3 * np.ones((3, 4 * 7)))
        np.testing.assert_array_almost_equal(wavdiag.maps[1], 
                                             6 * np.ones((3, 5 * 9)))

        self.assertEqual(wavdiag.ndim, 1)
        self.assertEqual(wavdiag.dtype, np.float64)
        self.assertEqual(wavdiag.shape, (2,))

        # I expect copies of maps and minfos.
        self.assertFalse(np.shares_memory(wavdiag.maps[0], wavmat_orig.maps[0,0]))
        self.assertFalse(np.shares_memory(wavdiag.maps[1], wavmat_orig.maps[1,1]))
        
        self.assertNotEqual(wavdiag.minfos[0], minfos[0])
        self.assertNotEqual(wavdiag.minfos[1], minfos[2])
        
        # Test if new minfos contain the same info.
        np.testing.assert_almost_equal(wavdiag.minfos[0].theta,
                                       wavmat_orig.minfos[0,0].theta)
        np.testing.assert_almost_equal(wavdiag.minfos[0].nphi,
                                       wavmat_orig.minfos[0,0].nphi)
        np.testing.assert_almost_equal(wavdiag.minfos[0].phi0,
                                       wavmat_orig.minfos[0,0].phi0)
        np.testing.assert_almost_equal(wavdiag.minfos[0].offsets,
                                       wavmat_orig.minfos[0,0].offsets)
        np.testing.assert_almost_equal(wavdiag.minfos[0].stride,
                                       wavmat_orig.minfos[0,0].stride)
        np.testing.assert_almost_equal(wavdiag.minfos[0].weight,
                                       wavmat_orig.minfos[0,0].weight)

        np.testing.assert_almost_equal(wavdiag.minfos[1].theta,
                                       wavmat_orig.minfos[1,1].theta)
        np.testing.assert_almost_equal(wavdiag.minfos[1].nphi,
                                       wavmat_orig.minfos[1,1].nphi)
        np.testing.assert_almost_equal(wavdiag.minfos[1].phi0,
                                       wavmat_orig.minfos[1,1].phi0)
        np.testing.assert_almost_equal(wavdiag.minfos[1].offsets,
                                       wavmat_orig.minfos[1,1].offsets)
        np.testing.assert_almost_equal(wavdiag.minfos[1].stride,
                                       wavmat_orig.minfos[1,1].stride)
        np.testing.assert_almost_equal(wavdiag.minfos[1].weight,
                                       wavmat_orig.minfos[1,1].weight)

    def test_wav_init_mat_add(self):
        
        wavmat = wavtrans.Wav(2)

        self.assertEqual(wavmat.ndim, 2)
        self.assertEqual(wavmat.dtype, np.float64)
        self.assertEqual(wavmat.shape, (0,0))

        self.assertEqual(wavmat.maps, {})
        self.assertEqual(wavmat.minfos, {})
        np.testing.assert_array_equal(wavmat.indices, np.zeros((0, 2), dtype=int))

        # Add first map.
        minfo = sharp.map_info_gauss_legendre(3)
        m_arr = np.ones((1, minfo.npix))
        index = (10, 10)

        wavmat.add(index, m_arr, minfo)

        self.assertEqual(wavmat.preshape, (1,))
        self.assertEqual(wavmat.ndim, 2)
        self.assertEqual(wavmat.dtype, np.float64)
        self.assertEqual(wavmat.shape, (11, 11))

        np.testing.assert_array_equal(wavmat.maps[10,10], m_arr)
        self.assertEqual(wavmat.minfos[10,10], minfo)
        np.testing.assert_array_equal(wavmat.indices, np.asarray([[10,10]]))

        # Add second map.
        index = (5, 5)

        wavmat.add(index, m_arr, minfo)

        self.assertEqual(wavmat.ndim, 2)
        self.assertEqual(wavmat.dtype, np.float64)
        self.assertEqual(wavmat.shape, (11, 11))

        np.testing.assert_array_equal(wavmat.maps[5,5], m_arr)
        np.testing.assert_array_equal(wavmat.maps[10,10], m_arr)
        self.assertEqual(wavmat.minfos[5,5], minfo)
        self.assertEqual(wavmat.minfos[10,10], minfo)
        indices_exp = np.zeros((2, 2), dtype=int)
        indices_exp[0] = [5, 5]
        indices_exp[1] = [10, 10]
        np.testing.assert_array_equal(wavmat.indices, indices_exp)

        # Overwrite first map.
        index = (10, 10)
        minfo_2 = sharp.map_info_gauss_legendre(5)
        m_arr_2 = np.ones((1, minfo_2.npix))

        wavmat.add(index, m_arr_2, minfo_2)

        self.assertEqual(wavmat.ndim, 2)
        self.assertEqual(wavmat.dtype, np.float64)
        self.assertEqual(wavmat.shape, (11, 11))

        np.testing.assert_array_equal(wavmat.maps[5,5], m_arr)
        np.testing.assert_array_equal(wavmat.maps[10,10], m_arr_2)
        self.assertEqual(wavmat.minfos[5,5], minfo)
        self.assertEqual(wavmat.minfos[10,10], minfo_2)
        np.testing.assert_array_equal(wavmat.indices, indices_exp)

    def test_wav_init_err(self):
        
        minfos = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [3, 3, 4]])
        preshape = (3,)

        # Negative indices.
        indices = np.asarray([-1, 4, 6])
        self.assertRaises(
            ValueError, wavtrans.Wav, 1,
            **dict(indices=indices, minfos=minfos, preshape=preshape))

        # Float indices.
        indices = np.asarray([1, 4, 6], dtype=float)
        self.assertRaises(
            ValueError, wavtrans.Wav, 1, 
            **dict(indices=indices, minfos=minfos, preshape=preshape))

        # 3d indices.
        indices = np.asarray([[0, 2, 1], [2, 1, 1], [1, 4, 6]])
        self.assertRaises(
            NotImplementedError, wavtrans.Wav, 3, 
            **dict(indices=indices, minfos=minfos, preshape=preshape))

        # Different shapes.
        indices = np.asarray([4, 6, 7, 9])
        self.assertRaises(
            IndexError, wavtrans.Wav, 1, 
            **dict(indices=indices, minfos=minfos, preshape=preshape))

    def test_alm2wav(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5
        spin = 0

        wlms_exp = [np.asarray([1, 2, 5], dtype=np.complex64), 
                    0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10],
                                     dtype=np.complex64)]

        winfos_exp = [sharp.alm_info(lmax=1), sharp.alm_info(lmax=3)]
        minfos_exp = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [1, 3]])
        maps_exp = [np.zeros(minfos_exp[idx].npix, dtype=np.float32) for idx in [0, 1]]
        sht.alm2map(wlms_exp[0], maps_exp[0], winfos_exp[0], minfos_exp[0], spin)
        sht.alm2map(wlms_exp[1], maps_exp[1], winfos_exp[1], minfos_exp[1], spin)
        
        wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell)

        self.assertEqual(wav.shape, (2,))        
        np.testing.assert_array_almost_equal(wav.maps[0], maps_exp[0])
        np.testing.assert_array_almost_equal(wav.maps[1], maps_exp[1])

    def test_alm2wav_pol(self):

        alm = np.zeros((3, 10), dtype=np.complex128)
        alm[0] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex128)
        alm[1] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex128) * 2
        alm[2] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex128) * 3
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:3] = 1
        w_ell[1,3:] = 0.5
        spin = [0, 2]

        wlms_exp = []
        wlms_exp.append(np.asarray([[1, 2, 3, 5, 6, 8],
                                    [2, 4, 6, 10, 12, 16],
                                    [3, 6, 9, 15, 18, 24]],
                                   dtype=np.complex128))
        wlms_exp.append(0.5 * np.asarray([[0, 0, 0, 4, 0, 0, 7, 0, 9, 10],
                                          [0, 0, 0, 8, 0, 0, 14, 0, 18, 20],
                                          [0, 0, 0, 12, 0, 0, 21, 0, 27, 30]],
                                         dtype=np.complex128))

        winfos_exp = [sharp.alm_info(lmax=2), sharp.alm_info(lmax=3)]
        minfos_exp = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [2, 3]])
        maps_exp = [np.zeros((3, minfos_exp[idx].npix), dtype=np.float64) 
                    for idx in [0, 1]]

        sht.alm2map(wlms_exp[0], maps_exp[0], winfos_exp[0], minfos_exp[0], spin)
        sht.alm2map(wlms_exp[1], maps_exp[1], winfos_exp[1], minfos_exp[1], spin)
        
        wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell)

        self.assertEqual(wav.shape, (2,))        
        np.testing.assert_array_almost_equal(wav.maps[0], maps_exp[0])
        np.testing.assert_array_almost_equal(wav.maps[1], maps_exp[1])

    def test_alm2wav_wav(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5
        spin = 0

        wlms_exp = [np.asarray([1, 2, 5], dtype=np.complex64), 
                    0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10],
                                     dtype=np.complex64)]

        winfos_exp = [sharp.alm_info(lmax=1), sharp.alm_info(lmax=3)]
        minfos_exp = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [1, 3]])
        maps_exp = [np.zeros(minfos_exp[idx].npix, dtype=np.float32) for idx in [0, 1]]
        sht.alm2map(wlms_exp[0], maps_exp[0], winfos_exp[0], minfos_exp[0], spin)
        sht.alm2map(wlms_exp[1], maps_exp[1], winfos_exp[1], minfos_exp[1], spin)
        
        # Provide wav and check if maps are updated inplace.
        wav_in = wavtrans.Wav(1, indices=np.arange(2), minfos=minfos_exp,
                              preshape=(), dtype=np.float32)

        wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell, wav=wav_in)

        np.testing.assert_array_almost_equal(wav_in.maps[0], maps_exp[0])
        np.testing.assert_array_almost_equal(wav_in.maps[1], maps_exp[1])
        
    def test_alm2wav_lmaxs(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5
        spin = 0

        lmaxs = np.asarray([2, 3])

        wlms_exp = [np.asarray([1, 2, 0, 5, 0, 0], dtype=np.complex64), 
                    0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10],
                                     dtype=np.complex64)]

        winfos_exp = [sharp.alm_info(lmax=2), sharp.alm_info(lmax=3)]
        minfos_exp = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [2, 3]])
        maps_exp = [np.zeros(minfos_exp[idx].npix, dtype=np.float32) for idx in [0, 1]]
        sht.alm2map(wlms_exp[0], maps_exp[0], winfos_exp[0], minfos_exp[0], spin)
        sht.alm2map(wlms_exp[1], maps_exp[1], winfos_exp[1], minfos_exp[1], spin)
        
        wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell, lmaxs=lmaxs)

        self.assertEqual(wav.shape, (2,))        
        np.testing.assert_array_almost_equal(wav.maps[0], maps_exp[0])
        np.testing.assert_array_almost_equal(wav.maps[1], maps_exp[1])

    def test_alm2wav_adjoint(self):

        alm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex64)
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5
        spin = 0

        wlms_exp = [np.asarray([1, 2, 5], dtype=np.complex64), 
                    0.5 * np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10],
                                     dtype=np.complex64)]

        winfos_exp = [sharp.alm_info(lmax=1), sharp.alm_info(lmax=3)]
        minfos_exp = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [1, 3]])
        maps_exp = [np.zeros(minfos_exp[idx].npix, dtype=np.float32) for idx in [0, 1]]
        sht.alm2map(wlms_exp[0], maps_exp[0], winfos_exp[0],
                    minfos_exp[0], spin, adjoint=True)
        sht.alm2map(wlms_exp[1], maps_exp[1], winfos_exp[1],
                    minfos_exp[1], spin, adjoint=True)
        
        wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell, adjoint=True)

        self.assertEqual(wav.shape, (2,))
        np.testing.assert_array_almost_equal(wav.maps[0], maps_exp[0])
        np.testing.assert_array_almost_equal(wav.maps[1], maps_exp[1])

    def test_wav2alm(self):

        alm = np.zeros((3, 10), dtype=np.complex128)
        alm[0] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.complex128)
        alm[1] = np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10], dtype=np.complex128) * 2
        alm[2] = np.asarray([0, 0, 3, 4, 0, 6, 7, 8, 9, 10], dtype=np.complex128) * 3
        ainfo = sharp.alm_info(lmax=3)
        w_ell = np.zeros((2, 4))
        w_ell[0,:3] = 1
        w_ell[1,3:] = 1
        spin = [0, 2]

        wlms_exp = []
        wlms_exp.append(np.asarray([[1, 2, 3, 5, 6, 8],
                                    [0, 0, 6, 0, 12, 16],
                                    [0, 0, 9, 0, 18, 24]],
                                   dtype=np.complex128))
        wlms_exp.append(np.asarray([[0, 0, 0, 4, 0, 0, 7, 0, 9, 10],
                                    [0, 0, 0, 8, 0, 0, 14, 0, 18, 20],
                                    [0, 0, 0, 12, 0, 0, 21, 0, 27, 30]],
                                   dtype=np.complex128))

        winfos_exp = [sharp.alm_info(lmax=2), sharp.alm_info(lmax=3)]
        minfos_exp = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [2, 3]])
        maps_exp = [np.zeros((3, minfos_exp[idx].npix), dtype=np.float64) 
                    for idx in [0, 1]]

        sht.alm2map(wlms_exp[0], maps_exp[0], winfos_exp[0], minfos_exp[0], spin)
        sht.alm2map(wlms_exp[1], maps_exp[1], winfos_exp[1], minfos_exp[1], spin)
        
        wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell)

        alm_out = np.zeros_like(alm)
        wavtrans.wav2alm(wav, alm_out, ainfo, spin, w_ell)

        np.testing.assert_array_almost_equal(alm_out, alm)

    def test_wav2alm_adjoint(self):

        ainfo = sharp.alm_info(lmax=3)
        alm = np.zeros((1, ainfo.nelem), dtype=np.complex128)
        minfos_exp = np.asarray([sharp.map_info_gauss_legendre(
            lmax + 1, 2 * lmax + 1) for lmax in [1, 3]])
        w_ell = np.zeros((2, 4))
        w_ell[0,:2] = 1
        w_ell[1,2:] = 0.5
        spin = 0
        
        wav = wavtrans.Wav(1, indices=np.arange(2), minfos=minfos_exp)
        
        wav.maps[0] += 1
        wav.maps[1] += 2

        wavtrans.wav2alm(wav, alm, ainfo, spin, w_ell, adjoint=True)

        ainfo_1 = sharp.alm_info(lmax=1)
        ainfo_2 = sharp.alm_info(lmax=3)

        alm_1 = np.zeros((1, ainfo_1.nelem), dtype=np.complex128)
        alm_2 = np.zeros((1, ainfo_2.nelem), dtype=np.complex128)

        sht.map2alm(wav.maps[0], alm_1, wav.minfos[0], ainfo_1, spin, adjoint=True)
        sht.map2alm(wav.maps[1], alm_2, wav.minfos[1], ainfo_2, spin, adjoint=True)

        alm_1 = ainfo_1.lmul(alm_1, w_ell[0])
        alm_2 = ainfo_2.lmul(alm_2, w_ell[1])

        alm_exp = alm_2
        alm_exp[0,0] += alm_1[0,0]
        alm_exp[0,1] += alm_1[0,1]
        alm_exp[0,4] += alm_1[0,2]
        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_preshape2npol(self):

        preshape = ()
        self.assertEqual(wavtrans.preshape2npol(preshape), 1)

        preshape = (2,)
        self.assertEqual(wavtrans.preshape2npol(preshape), 2)

        preshape = (3, 3)
        self.assertEqual(wavtrans.preshape2npol(preshape), 3)
        
        self.assertRaises(ValueError, wavtrans.preshape2npol, (2, 1))

        self.assertRaises(ValueError, wavtrans.preshape2npol, (2, 2, 2))
