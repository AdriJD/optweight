import numpy as np

from pixell import utils

from optweight import operators, mat_utils

class HarmonicPreconditioner(operators.MatVecAlm):
    '''
    Harmonic preconditioner: M = (C^-1 + B^2 itau)^-1.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse covariance matrix, either symmetric but dense in first two axes
        or diagonal, in which case only the diagonal elements are needed.
    itau : (npol, npol) or (npol, npol, nell) array
        Isotropic noise (co)variance.
    b_ell : (npol, nell) array, optional
        Beam window function.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''

    def __init__(self, ainfo, icov_ell, itau, b_ell=None):

        npol, nell = icov_ell.shape[-2:]

        if itau.ndim == 2:
            itau = itau[:,:,np.newaxis]

        if b_ell is None:
            b_ell = np.ones((npol, nell))

        if np.count_nonzero(itau - itau * np.eye(npol)[:,:,np.newaxis]):
            # itau != symmetric, so b_ell and itau will not necessarily commute.
            b_ell = b_ell * np.eye(npol)[:,:,np.newaxis]
            op = icov_ell + np.einsum('ijl, jkl, kol -> iol', b_ell, itau, b_ell)        
        else:
            op = icov_ell + itau * b_ell ** 2

        self.preconditioner = operators.EllMatVecAlm(ainfo, op, -1, inplace=False)
            
    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        return self.preconditioner(alm)

class PseudoInvPreconditioner(operators.MatVecAlm):
    '''
    Implementation of the pseudo-inverse preconditioner from Seljebotn et al.,
    2017 (1710.00621).

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse signal covariance, If diagonal, only the diagonal suffices.
    itau : (npol, npol) or (npol) array or float
        Isotropic noise (co)variance.
    icov_pix : (npol, npol, npix) or (npol, npix) array
        Inverse noise covariance. If diagonal, only the diagonal suffices.
    minfo : sharp.map_info object
        Metainfo for inverse noise covariance.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    b_ell : (npol, nell) array, optional
        Beam window function.
    cov_pix : (npol, npol, npix) or (npol, npix) array, optional
        Noise covariance. If diagonal, only the diagonal suffices.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''

    def __init__(self, ainfo, icov_ell, itau, icov_pix, minfo, spin,
                 b_ell=None, cov_pix=None):

        npol, nell = icov_ell.shape[-2:]

        if itau.ndim == 2:
            itau = itau[:,:,np.newaxis]

        if np.count_nonzero(itau - itau * np.eye(npol)[:,:,np.newaxis]):
            raise NotImplementedError(
                'Pinv precondition cannot handle off-diagonal itau elements for now.')

        if b_ell is None:
            b_ell = np.ones((icov_ell.shape[0], icov_ell.shape[-1]))

        if cov_pix is None:
            cov_pix = mat_utils.matpow(icov_pix, -1)

        self.harmonic_prec = operators.EllMatVecAlm(
            ainfo, icov_ell + itau * b_ell ** 2, -1, inplace=True)
        
        self.icov_signal = operators.EllMatVecAlm(
            ainfo, icov_ell, inplace=True)

        self.ivar_noise_iso = operators.EllMatVecAlm(
            ainfo, itau * b_ell, inplace=True)

        self.pcov_noise = operators.PixMatVecAlm(
            ainfo, cov_pix, minfo, spin, inplace=True, adjoint=True)

    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        alm = alm.copy()

        alm = self.harmonic_prec(alm)

        alm_signal = alm.copy()
        alm_noise = alm

        alm_signal = self.icov_signal(alm_signal)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.pcov_noise(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)

        alm = alm_noise
        alm += alm_signal

        alm = self.harmonic_prec(alm)

        return alm

class PseudoInvPreconditionerWav(operators.MatVecAlm):
    '''
    Adaptation of the pseudo-inverse preconditioner from Seljebotn et al.,
    2017 (1710.00621) to a wavelet-based noise model.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse signal covariance, If diagonal, only the diagonal suffices.
    itau_ell : (npol, npol, nell) array
        Isotropic noise (co)variance.
    icov_wav : wavtrans.Wav object
        Wavelet block matrix representing the inverse noise covariance.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    mask_pix = (npol, npix) array, optional
        Pixel mask.
    minfo_mask : sharp.map_info object, optional
        Metainfo for pixel mask covariance.
    b_ell : (npol, nell) array, optional
        Beam window function.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.

    '''
    
    def __init__(self, ainfo, icov_ell, itau_ell, icov_wav, w_ell, spin,
                 mask_pix=None, minfo_mask=None, b_ell=None):

        if itau_ell.ndim != 3:
            raise ValueError(
                'Wrong dimensions itau_ell : expected 3, got {}'.
                format(itau_ell.ndim))

        if b_ell is None:
            b_ell = np.ones((icov_ell.shape[0], icov_ell.shape[-1]))

        if mask_pix is None:
            self.imask = lambda alm: alm
        else:            
            self.imask = operators.PixMatVecAlm(
                ainfo, mask_pix, minfo_mask, spin, power=-1,
                use_weights=True)

        self.harmonic_prec = operators.EllMatVecAlm(
            ainfo, icov_ell + itau_ell * b_ell ** 2, -1, inplace=True)
        
        self.icov_signal = operators.EllMatVecAlm(
            ainfo, icov_ell, inplace=True)

        self.beam = operators.EllMatVecAlm(
            ainfo, b_ell, inplace=True)

        self.ivar_noise_iso = operators.EllMatVecAlm(
            ainfo, itau_ell, inplace=True)

        self.pcov_noise = operators.WavMatVecAlm(
            ainfo, icov_wav, w_ell, spin, power=-1, adjoint=True)
    #@profile
    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        alm = alm.copy()

        alm = self.harmonic_prec(alm)

        alm_signal = alm.copy()
        alm_noise = alm

        alm_signal = self.icov_signal(alm_signal)
        alm_noise = self.beam(alm_noise)
        alm_noise = self.imask(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.pcov_noise(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.imask(alm_noise)
        alm_noise = self.beam(alm_noise)

        alm = alm_noise
        alm += alm_signal

        alm = self.harmonic_prec(alm)

        return alm
