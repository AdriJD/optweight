import numpy as np
from abc import ABC, abstractmethod

from pixell import sharp

from optweight import sht, wavtrans, alm_utils, type_utils, alm_c_utils, mat_utils, map_utils

class MatVecAlm(ABC):
    '''Template for all matrix-vector operators working on alm input.'''

    def __call__(self, alm):
        return self.call(alm)

    @abstractmethod
    def call(self, alm):
        pass

class MatVecMap(ABC):
    '''Template for all matrix-vector operators working on pixelized maps as input.'''

    def __call__(self, imap):
        return self.call(imap)

    @abstractmethod
    def call(self, imap):
        pass

class MatVecWav(ABC):
    '''Template for all matrix-vector operators working on wavelet input.'''

    def __call__(self, wav):
        return self.call(wav)

    @abstractmethod
    def call(self, wav):
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
    power : int, float, optional
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
    power : int, float, optional
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

class EllWavEllMatVecAlm(MatVecAlm):
    '''
    Calculate X_ell^q M_wav^p X_ell^q alm for X and M diagonal in the multipole and 
    wavelet-pixel domain, respectively. Both matrices should be positive semi-definite
    symmetric in other axes.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    m_wav : wavtrans.Wav object
        Wavelet block matrix.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    x_ell : (npol, npol, nell) array or (npol, nell) array
        A matrix, either symmetric but dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    spin : int, array-like
        Spin values for spherical harmonic transforms, should be
        compatible with npol.
    power_m : int, float, optional
        Power of M matrix.
    power_x : int, float, optional
        Power of X matrix.
    adjoint : bool, optional
        If set, calculate Kt Yt W M W Y K instead of Kt Yt M Y K.

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''

    def __init__(self, ainfo, m_wav, w_ell, x_ell, spin, power_m=1, power_x=1,
                 adjoint=False):

        self.x_ell_op = EllMatVecAlm(ainfo, x_ell, power=power_x, inplace=False)
        self.m_wav_op = WavMatVecAlm(ainfo, m_wav, w_ell, spin, power=power_m,
                                     adjoint=adjoint)
        
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

        alm = self.x_ell_op(alm)
        alm = self.m_wav_op(alm)
        alm = self.x_ell_op(alm)

        return alm
        
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
    power : int, float, optional
        Power of matrix.
    adjoint : bool, optional
        If set, calculate Kt Yt W M W Y K instead of Kt Yt M Y K.

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''

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

class EllPixEllMatVecAlm(MatVecAlm):
    '''
    Calculate X_ell^q M_pix^p X_ell^q alm for X and M diagonal in the multipole and 
    pixel domain, respectively. Both matrices should be positive semi-definite symmetric
    in other axes.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    m_pix : (npol, npol, npix) array or (npol, npix) array
        Matrix diagonal in pixel domain, either dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    x_ell : (npol, npol, nell) array or (npol, nell) array
        A matrix, either symmetric but dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    minfo : sharp.map_info object
        Metainfo for pixelization of the M matrix.
    spin : int, array-like
        Spin values for spherical harmonic transforms, should be
        compatible with npol.
    power_m : int, float, optional
        Power of M matrix.
    power_x : int, float, optional
        Power of X matrix.
    inplace : bool, optional
        Perform operation in place.
    adjoint : bool, optional
        If set, calculate X Yt W M W Y X instead of X Yt M Y X.
    use_weights : bool, optional
        If set, use integration weights: X Yt W M Y X or X Yt M W Y X for adjoint.

    Methods
    -------
    call(alm) : Apply the operator to a set of alms.
    '''

    def __init__(self, ainfo, m_pix, x_ell, minfo, spin, power_m=1, power_x=1,
                 inplace=False, adjoint=False, use_weights=False):

        m_pix = mat_utils.full_matrix(m_pix)
        x_ell = mat_utils.full_matrix(x_ell)
        if power_m != 1:
            m_pix = mat_utils.matpow(m_pix, power_m)
        if power_x != 1:
            x_ell = mat_utils.matpow(x_ell, power_x)

        self.m_pix = m_pix
        self.x_ell = x_ell
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

        alm_c_utils.lmul(alm, self.x_ell, self.ainfo, inplace=True)

        sht.alm2map(alm, omap, self.ainfo, self.minfo, self.spin,
                    adjoint=self.adjoint)
        imap = np.einsum('ijk, jk -> ik', self.m_pix, omap, optimize=True)

        if self.use_weights:
            adjoint = self.adjoint
        else:
            adjoint = not self.adjoint
        sht.map2alm(imap, out, self.minfo, self.ainfo, self.spin,
                    adjoint=adjoint)

        alm_c_utils.lmul(out, self.x_ell, self.ainfo, inplace=True)

        return out

class WavMatVecWav(MatVecWav):
    '''
    Apply wavelet block matrix to input wavelet vector.

    Parameters
    ----------
    m_wav : wavtrans.Wav object
        Wavelet block matrix.
    power : int, float, optional
        Power of matrix.
    inplace : bool, optional
        Perform operation in place.
    op : str, optional
        Operation (pased to np.einsum) used for each block. E.g 
        "ijk, jk -> ik" to reproduce the standard behaviour.

    Methods
    -------
    call(wav) : Apply the operator to a wavelet vector.
    '''

    def __init__(self, m_wav, power=1, inplace=False, op=None):

        if m_wav.ndim != 2:
            raise ValueError(f'Expected m_wav.ndim = 2, got : {w_wav.ndim}')

        if power != 1:
            self.m_wav = mat_utils.wavmatpow(m_wav, power, return_diag=True)
        else:
            self.m_wav = m_wav
        self.inplace = inplace

        if not self.inplace:
            raise NotImplementedError()

        self.op = op

    def call(self, wav):
        '''
        Apply the operator to a vector of wavelet maps.

        Parameters
        ----------
        wav : wavtrans.Wav object
            Input wavelet maps

        Returns
        -------
        out : wavtrans.Wav object
            Output from matrix-vector operation.
        '''

        if wav.ndim != 1:
            raise ValueError(f'Expected wav.ndim = 1, got : {wav.ndim}')
        
        for jidx in range(self.m_wav.shape[0]):
            for jpidx in range(self.m_wav.shape[1]):

                try:
                    map_mat = self.m_wav.maps[jidx,jpidx]
                except KeyError:
                    continue

                map_vec = wav.maps[jpidx]

                if self.op is None:
                    # Default behaviour.
                    if map_mat.ndim == 3:
                        map_prod = np.einsum('ijk, jk -> ik', map_mat, 
                            map_vec, out=map_vec, optimize=True)
                    elif map_mat.ndim == 2:
                        map_vec *= map_mat
                else:
                    map_prod = np.einsum(self.op, map_mat, map_vec, out=map_vec, 
                                         optimize=True)
        return wav

class PixEllPixMatVecMap(MatVecMap):
    '''
    Calculate M_pix^p X_ell^q M_pix^p map for X and M diagonal in the multipole and 
    pixel domain, respectively. Both matrices should be positive semi-definite symmetric
    in other axes.

    Parameters
    ----------
    m_pix : (npol, npol, npix) array or (npol, npix) array, or None.
        Matrix diagonal in pixel domain, either dense in first two axes or diagonal,
        in which case only the diagonal elements are needed. Can also be set to None,
        in which case the matrix is ignored.
    x_ell : (npol, npol, nell) array or (npol, nell) array
        A matrix, either symmetric but dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    minfo : sharp.map_info object
        Metainfo for pixelization of input map and the M matrix.
    spin : int, array-like
        Spin values for spherical harmonic transforms, should be
        compatible with npol.
    power_m : int, float, optional
        Power of M matrix.
    power_x : int, float, optional
        Power of X matrix.
    inplace : bool, optional
        Perform operation in place.
    adjoint : bool, optional
        If set, calculate (M_pix W Y X_ell Yt W M_pix) instead of 
        (M_pix Y X_ell Yt M_pix).
    use_weights : bool, optional
        If set, use integration weights: (M_pix Y X_ell Yt W M_pix) or
        (M_pix W Y X_ell Yt M_pix) for adjoint.
    lmax : int, optional
        Max multipole for internal harmonic operations. Will be inferred from
        minfo if not provided.

    Methods
    -------
    call(imap) : Apply the operator to a map.

    Notes
    -----
    Ignoring the powers, this calculates (M_pix Y X_ell Yt M_pix) map, so note
    the lack of integration weights.
    '''

    def __init__(self, m_pix, x_ell, minfo, spin, power_m=1, power_x=1, 
                 inplace=False, adjoint=False, use_weights=False, lmax=None):

        if m_pix is not None:
            if power_m != 1:
                m_pix = mat_utils.matpow(m_pix, power_m, return_diag=True)

        if power_x != 1:
            x_ell = mat_utils.matpow(x_ell, power_x, return_diag=True)

        self.m_pix = m_pix
        self.x_ell = x_ell
        self.npol = x_ell.shape[0]
        if lmax is None:
            lmax = map_utils.minfo2lmax(minfo)
        self.ainfo = sharp.alm_info(lmax)
        self.minfo = minfo
        self.spin = spin
        self.inplace = inplace
        self.adjoint = adjoint
        self.use_weights = use_weights

    def call(self, imap):
        '''
        Apply the operator to input map.

        Parameters
        ----------
        imap : (npol, npix) array
            Input map.

        Returns
        -------
        out : (npol, npix) array
            Output from matrix-vector operation.
        '''

        assert imap.shape == (self.npol, self.minfo.npix), (
            f'Wrong imap shape: {imap.shape} != {(self.npol, self.minfo.npix)}')

        if self.inplace:
            out = imap
        else:
            out = imap.copy()

        alm = np.zeros((self.npol, self.ainfo.nelem),
                       dtype=type_utils.to_complex(imap.dtype))

        if self.m_pix is not None:
            if self.m_pix.ndim == 3:
                np.einsum('abp, bp -> ap', self.m_pix, out, out=out, optimize=True)
            else:
                out *= self.m_pix
        sht.map2alm(out, alm, self.minfo, self.ainfo, self.spin,
                    adjoint=self.adjoint is self.use_weights)
        alm_c_utils.lmul(alm, self.x_ell, self.ainfo, inplace=True)
        sht.alm2map(alm, out, self.ainfo, self.minfo, self.spin,
                    adjoint=self.adjoint)
        if self.m_pix is not None:
            if self.m_pix.ndim == 3:
                np.einsum('abp, bp -> ap', self.m_pix, out, out=out, optimize=True)
            else:
                out *= self.m_pix

        return out

def op2mat(op, nrow, dtype, ncol=None, input_shape=None):
    '''
    Convert a linear operator into a matrix representation.

    Parameters
    ----------
    op : callable
        Linear operation. Takes in 1D vector of length ncol and produces 1d
        vector of length nrow.
    nrow : int
        Number of rows of matrix.
    dtype : type
        Dtype of input vector and matrix.
    ncol : int
        Number of colums of matrix, if different from nrow.
    input_shape : tuple, optional
        Reshape ncol input vector into this shape before applying operation.

    Returns
    -------
    mat : (nrow, ncol) array
        Matrix representation.
    '''

    ncol = nrow if ncol is None else ncol    
    uvec = np.zeros(ncol, dtype=dtype)
    if input_shape is not None:
        uvec_shaped = uvec.reshape(input_shape)
    else:
        uvec_shaped = uvec
    mat = np.zeros((nrow, ncol), dtype=dtype)

    for idx in range(ncol):
        uvec[idx] = 1
        mat[:,idx] = op(uvec_shaped).reshape(-1)
        uvec[:] = 0

    return mat

def add_operators(op_1, op_2, slice_2=None):
    '''
    Return calling function that acts as sum of two operators.

    Arguments
    ---------
    op_1 : callable
        First operator.
    op_2 : callable
        Second operator.
    slice_2 : slice or tuple of slices, optional
        Slice into input vector needed for operator 2.

    Returns
    -------
    op_add : callable
        Function that takes in same input as both functions
    '''
    
    if slice_2 is None:
        slice_2 = slice(None)

    def add(ivec):
        
        # Copying is probably redundant, but we don't know if
        # operators work in-place or not.
        ovec = op_1(ivec.copy())
        ovec[slice_2] += op_2(ivec[slice_2].copy())

        return ovec

    return add
