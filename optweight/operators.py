import numpy as np
from abc import ABC, abstractmethod

from pixell import utils, sharp

from optweight import sht, wavtrans, alm_utils

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

        m_ell = _full_matrix(m_ell)
        if power != 1:
            m_ell = _matpow(m_ell, power)

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
            out = alm
        else:
            out = alm.copy()

        return self.ainfo.lmul(alm, self.m_ell, out=out)

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

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''
    
    def __init__(self, ainfo, m_pix, minfo, spin, power=1, inplace=False,
                 adjoint=False):

        m_pix = _full_matrix(m_pix)
        if power != 1:
            m_pix = _matpow(m_pix, power)

        self.m_pix = m_pix
        self.ainfo = ainfo
        self.minfo = minfo
        self.spin = spin
        self.inplace = inplace
        self.adjoint = adjoint

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
        omap = np.zeros((npol, self.m_pix.shape[-1]))

        sht.alm2map(alm, omap, self.ainfo, self.minfo, self.spin,
                    adjoint=self.adjoint)
        imap = np.einsum('ijk, jk -> ik', self.m_pix, omap, optimize=True)
        sht.map2alm(imap, out, self.minfo, self.ainfo, self.spin,
                    adjoint=not self.adjoint)

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
    adjoint : bool, optional
        If set, calculate Kt Yt W M W Y K instead of Kt Yt M Y K.

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''
    
    def __init__(self, ainfo, m_wav, w_ell, spin, adjoint=False):

        self.ainfo = ainfo
        self.m_wav = m_wav
        self.w_ell = w_ell
        self.spin = spin
        self.adjoint = adjoint
        self.v_wav = m_wav.diag()

        winfos = {}
        for index in self.v_wav.minfos:
            minfo = self.v_wav.minfos[index]
            # Note, only valid for Gauss Legendre pixels.
            # We use nphi to support maps with cuts in theta.
            lmax = (minfo.nphi[0] - 1) // 2
            winfos[index] = sharp.alm_info(lmax=lmax)

        self.winfos = winfos

        if self.v_wav.dtype == np.float64:
            self.alm_dtype = np.complex128
        elif self.v_wav.dtype == np.float32:
            self.alm_dtype = np.complex64
            
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
                
def _full_matrix(mat):
    '''
    If needed, expand matrix diagonal to full-sized matrix.

    Parameters
    ----------
    mat : (npol, npol, N) array or (npol, N) array
        Matrix, dense or diagonal in first two axes.

    Returns
    -------
    mat_full : (npol, npol, N) array
        Full-sized matrix.

    Raises
    ------
    ValueError
        If input array dimensions are not understood.
    '''

    if mat.ndim not in (2, 3):
        raise ValueError(
            'Matrix dimension : {} not supported. Requires dim = 2 or 3.'.
            format(mat.ndim))

    if mat.ndim == 2:
        # Diagonal of matrix, expand to full matrix. 
        npol = mat.shape[0]
        mat = np.eye(npol)[:,:,np.newaxis] * mat

    return mat
        
def _matpow(mat, power):
    '''
    Raise matrix to a given power.

    Parameters
    ----------
    mat : (npol, npol, N) array or (npol, N) array
        Matrix, either symmetric but dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    power : int, float
        Power of matrix.
    
    Returns
    -------
    mat_out : (npol, npol, N) array
        Output matrix.
    '''

    mat = _full_matrix(mat)

    if power == 1:
        return mat
        
    mat = np.ascontiguousarray(np.transpose(mat, (2, 0, 1)))
    mat = utils.eigpow(mat, power)
    mat = np.ascontiguousarray(np.transpose(mat, (1, 2, 0)))

    return mat
