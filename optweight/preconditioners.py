import numpy as np

from pixell import utils

from optweight import sht
from optweight import operators

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
    itau : (npol, npol) or (npol) array or float
        Isotropic noise (co)variance.
    b_ell : (npol, nell) array, optional
        Beam window function.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''

    def __init__(self, ainfo, icov_ell, itau, b_ell=None):

        if itau.ndim == 2:
            itau = itau[:,:,np.newaxis]

        if b_ell is None:
            b_ell = np.ones((icov_ell.shape[0], icov_ell.shape[-1]))

        self.preconditioner = operators.EllMatVecAlm(
            ainfo, icov_ell + itau * b_ell ** 2, -1, inplace=False)
            
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
    b_ell : (npol, nell) array, optional
        Beam window function.
    cov_pix : (npol, npol, npix) or (npol, npix) array, optional
        Noise covariance. If diagonal, only the diagonal suffices.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''

    def __init__(self, ainfo, icov_ell, itau, icov_pix, minfo,
                 b_ell=None, cov_pix=None):

        if itau.ndim == 2:
            itau = itau[:,:,np.newaxis]

        if b_ell is None:
            b_ell = np.ones((icov_ell.shape[0], icov_ell.shape[-1]))

        if cov_pix is None:
            
            cov_pix = operators._matpow(icov_pix, -1)

        # Init all necessary operators.
        self.harmonic_prec = operators.EllMatVecAlm(
            ainfo, icov_ell + itau * b_ell ** 2, -1, inplace=True)
        
        self.icov_signal = operators.EllMatVecAlm(
            ainfo, icov_ell, inplace=True)

        self.ivar_noise_iso = operators.EllMatVecAlm(
            ainfo, itau * b_ell, inplace=True)

        self.pcov_noise = operators.PixMatVecAlm(
            ainfo, cov_pix, minfo, [0, 2], inplace=True, adjoint=True)

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

    # alm = operators.matvec_pow_ell_alm(
    #     alm, ainfo, icov_ell + itau * b_ell ** 2,
    #     -1, inplace=False)

    # # alm_sig = icov_ell alm
    # alm_sig = operators.matvec_pow_ell_alm(
    #     alm, ainfo, icov_ell, 1, inplace=False)

    # # alm_noise = (a^2 B N+ B) alm
    # alm_noise = operators.matvec_pow_ell_alm(
    #     alm, ainfo, itau * b_ell, 1, inplace=False)    

    # alm_noise = operators.matvec_pow_pix_alm(
    #     alm_noise, ainfo, cov_pix, minfo, [0, 2], 1,
    #     inplace=True, adjoint=True)

    # alm_noise = operators.matvec_pow_ell_alm(
    #     alm_noise, ainfo, itau * b_ell, 1, inplace=True)

    # # alm = alm_sig + alm_noise
    # alm = alm_sig
    # alm += alm_noise

    # # alm = (a^2 B^2 + icov_ell)^-1 alm 
    # alm = operators.matvec_pow_ell_alm(
    #     alm, ainfo, icov_ell + itau * b_ell ** 2,
    #     -1, inplace=False)

    # return alm



def harmonic_preconditioner(icov_ell, ainfo, itau, b_ell=None):
    '''
    Hoi ik ben een docstring
    '''
    if itau.ndim == 2:
        itau = itau[:,:,np.newaxis]

    if b_ell is None:
        preconditioner = operators.callable_matvec_pow_ell_alm(
            ainfo, icov_ell + itau, -1, inplace=False)
    else:
        preconditioner = operators.callable_matvec_pow_ell_alm(
            ainfo, icov_ell + itau * b_ell ** 2, -1, inplace=False)

    return preconditioner

def pinv_preconditioner(icov_ell, ainfo, itau, icov_pix, minfo, b_ell=None):

    return lambda alm: _pinv_preconditioner(
        alm, icov_ell, ainfo, itau, icov_pix, minfo, b_ell=b_ell)

def _pinv_preconditioner(alm, icov_ell, ainfo, itau, icov_pix, minfo,
                         b_ell=None, cov_pix=None):

    if itau.ndim == 2:
        itau = itau[:,:,np.newaxis]

    if b_ell is None:
        b_ell = np.ones((icov_ell.shape[0], icov_ell.shape[-1]))
        
    # You could precompute (a^2 B^2 + icov_ell)^-1 and (icov_pix)^-1

    # Calculate sqrt(tau)
    #alpha = utils.eigpow(itau, 0.5)

    #print('itau', itau)
    
    #tau = utils.eigpow(itau[:,:,0], -1)
    #tau = tau[:,:,np.newaxis]
    #itau = tau # NOTE NOTE NOTE
    
    #itau[1,1] = itau[0,0]    
    #itau[2,2] *= 1.3
    #itau = 0.5 * itau
    #itau = 0.6853702784139802 * itau

    if cov_pix is None:
        cov_pix = np.ascontiguousarray(np.transpose(icov_pix, (2, 0, 1)))
        #mask = cov_pix[:,0,0] < 1e-20
        #mask *= cov_pix[:,1,1] < 1e-20
        #mask *= cov_pix[:,2,2] < 1e-20
        #mask = (cov_pix[:,0,0] < 1e-20) & (cov_pix[:,1,1] < 1e-20) & (cov_pix[:,2,2] < 1e-20)
        mask = (cov_pix[:,0,0] == 0) & (cov_pix[:,1,1] == 0) & (cov_pix[:,2,2]== 0)

        # This works
        #cov_pix[~mask,:,:] = np.linalg.inv(cov_pix[~mask,:,:])

        cov_pix = utils.eigpow(cov_pix, -1)
        #cov_pix[mask] = np.eye(3)[np.newaxis,:,:] * 1e40
        ##mat = utils.eigpow(mat, power)
        cov_pix = np.ascontiguousarray(np.transpose(cov_pix, (1, 2, 0)))
    
    #print(cov_pix.max())
    #print(cov_pix.min())
    #print(cov_pix.mean())

    # alm = (a^2 B^2 + icov_ell)^-1 alm
    alm = operators.matvec_pow_ell_alm(
        alm, ainfo, icov_ell + itau * b_ell ** 2,
        -1, inplace=False)

    # alm_sig = icov_ell alm
    alm_sig = operators.matvec_pow_ell_alm(
        alm, ainfo, icov_ell, 1, inplace=False)

    # alm_noise = (a^2 B N+ B) alm
    alm_noise = operators.matvec_pow_ell_alm(
        alm, ainfo, itau * b_ell, 1, inplace=False)    

    #alm_noise = operators.matvec_pow_pix_alm(
    #    alm_noise, ainfo, icov_pix, minfo, [0, 2], -1,
    #    inplace=True, adjoint=True)
    alm_noise = operators.matvec_pow_pix_alm(
        alm_noise, ainfo, cov_pix, minfo, [0, 2], 1,
        inplace=True, adjoint=True)

    alm_noise = operators.matvec_pow_ell_alm(
        alm_noise, ainfo, itau * b_ell, 1, inplace=True)

    # alm = alm_sig + alm_noise
    alm = alm_sig
    alm += alm_noise

    # alm = (a^2 B^2 + icov_ell)^-1 alm 
    alm = operators.matvec_pow_ell_alm(
        alm, ainfo, icov_ell + itau * b_ell ** 2,
        -1, inplace=False)

    return alm
