import numpy as np
from abc import ABC, abstractmethod

from pixell import sharp

from optweight import sht, wavtrans, alm_utils, type_utils, alm_c_utils, mat_utils

class MatVecAlm(ABC):
    '''Template for all matrix-vector operators working on alm-input.'''

    def __call__(self, alm):
        return self.call(alm)

    @abstractmethod
    def call(self, alm):
        pass

class EllMatVecAlm(MatVecAlm):
    '''
    Calculate M^p alm for M diagonal in the harmonic domain and
    positive semi-definite symmetric in other axes.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    m_ell : (npol, npol, nell) array or (npol, nell) array
        M matrix, either symmetric but dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    power : int, float
        Power of matrix.
    inplace : bool, optional
        Perform operation in place.

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''

    def __init__(self, ainfo, m_ell, power=1, inplace=False):

        m_ell = mat_utils.full_matrix(m_ell)

        if power != 1:
            m_ell = mat_utils.matpow(m_ell, power)

        self.m_ell = m_ell
        self.ainfo = ainfo
        self.inplace = inplace

    def call(self, alm):
        '''
        Apply the operator to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        if self.inplace:
            return alm_c_utils.lmul(alm, self.m_ell, self.ainfo, inplace=True)
        else:
            return alm_c_utils.lmul(alm, self.m_ell, self.ainfo, alm_out=alm.copy())

class PixMatVecAlm(MatVecAlm):
    '''
    Calculate M^p alm for M diagonal in the pixel domain and
    positive semi-definite symmetric in other axes.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    m_pix : (npol, npol, npix) array or (npol, npix) array
        Matrix diagonal in pixel domain, either dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    minfo : sharp.map_info object
        Metainfo for pixelization of the M matrix.
    spin : int, array-like
        Spin values for spherical harmonic transforms, should be
        compatible with npol.
    power : int, float
        Power of matrix.
    inplace : bool, optional
        Perform operation in place.
    adjoint : bool, optional
        If set, calculate Yt W M W Y instead of Yt M Y.
    use_weights : bool, optional
        If set, use integration weights: Yt W M Y or Yt M W Y for adjoint.

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''

    def __init__(self, ainfo, m_pix, minfo, spin, power=1, inplace=False,
                 adjoint=False, use_weights=False):

        m_pix = mat_utils.full_matrix(m_pix)
        if power != 1:
            m_pix = mat_utils.matpow(m_pix, power)

        self.m_pix = m_pix
        self.ainfo = ainfo
        self.minfo = minfo
        self.spin = spin
        self.inplace = inplace
        self.adjoint = adjoint
        self.use_weights = use_weights

    def call(self, alm):
        '''
        Apply the operator to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        if self.inplace:
            out = alm
        else:
            out = alm.copy()

        npol = alm.shape[0]
        omap = np.zeros((npol, self.m_pix.shape[-1]),
                        dtype=type_utils.to_real(alm.dtype))

        sht.alm2map(alm, omap, self.ainfo, self.minfo, self.spin,
                    adjoint=self.adjoint)
        imap = np.einsum('ijk, jk -> ik', self.m_pix, omap, optimize=True)

        if self.use_weights:
            adjoint = self.adjoint
        else:
            adjoint = not self.adjoint
        sht.map2alm(imap, out, self.minfo, self.ainfo, self.spin,
                    adjoint=adjoint)

        return out

class WavMatVecAlm(MatVecAlm):
    '''
    Apply wavelet block matrix to input alm.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    m_wav : wavtrans.Wav object
        Wavelet block matrix.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    spin : int, array-like
        Spin values for spherical harmonic transforms, should be
        compatible with npol.
    power : int, float
        Power of matrix.
    adjoint : bool, optional
        If set, calculate Kt Yt W M W Y K instead of Kt Yt M Y K.

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''
    #@profile
    def __init__(self, ainfo, m_wav, w_ell, spin, power=1, adjoint=False):

        self.ainfo = ainfo
        self.w_ell = w_ell
        self.spin = spin
        self.adjoint = adjoint

        if power != 1:
            self.m_wav = mat_utils.wavmatpow(m_wav, power)
        else:
            self.m_wav = m_wav

        # Create wavelet vector that has correct preshape.
        npol = wavtrans.preshape2npol(self.m_wav.preshape)
        self.v_wav = wavtrans.Wav(1, preshape=(npol,), 
                                  dtype=self.m_wav.dtype)

        for jidx in range(m_wav.shape[0]):
            
            minfo = self.m_wav.minfos[jidx,jidx]
            m_arr = np.zeros(self.v_wav.preshape + (minfo.npix,),
                             dtype=self.v_wav.dtype)
            self.v_wav.add([jidx], m_arr, minfo)
                            
        winfos = {}
        for index in self.v_wav.minfos:
            minfo = self.v_wav.minfos[index]
            # Note, only valid for Gauss Legendre pixels.
            # We use nphi to support maps with cuts in theta.
            lmax = (minfo.nphi[0] - 1) // 2
            winfos[index] = sharp.alm_info(lmax=lmax)

        self.winfos = winfos

        self.alm_dtype = type_utils.to_complex(self.v_wav.dtype)

    #@profile
    def call(self, alm):
        '''
        Apply the operator to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        alm_out = np.zeros_like(alm)
        wavtrans.alm2wav(alm, self.ainfo, self.spin, self.w_ell,
                         wav=self.v_wav, adjoint=self.adjoint)

        for jidx in range(self.m_wav.shape[0]):

            for jpidx in range(self.m_wav.shape[1]):

                try:
                    map_mat = self.m_wav.maps[jidx,jpidx]
                except KeyError:
                    continue

                map_vec = self.v_wav.maps[jpidx]

                # If icov is (3, 3, npix) this should be a
                # matrix-vector product.
                if map_mat.ndim == 3:
                    map_prod = np.einsum(
                        'ijk, jk -> ik', map_mat, map_vec, optimize=True)
                elif map_mat.ndim == 2:
                    map_prod = map_mat * map_vec

                minfo = self.v_wav.minfos[jpidx]
                winfo = self.winfos[jpidx]
                wlm = np.zeros(alm.shape[:-1] + (winfo.nelem,),
                               dtype=self.alm_dtype)

                sht.map2alm(map_prod, wlm, minfo, winfo, self.spin,
                            adjoint=not self.adjoint)

                alm_utils.wlm2alm_axisym([wlm], [winfo], self.w_ell[jidx:jidx+1,:],
                                         alm=alm_out, ainfo=self.ainfo)

        return alm_out
