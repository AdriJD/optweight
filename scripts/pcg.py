import numpy as np

from pixell import curvedsky, utils
import healpy as hp

from sht import alm2map, map2alm
import map_utils

def a_matrix(alm, icov_pix, icov_ell, minfo, ainfo, spin, b_ell=None):
    '''
    Apply the A = (S^-1 + N^-1) matrix to input alm.

    Parameters
    ----------
    alm : (npol, nelem) complex array.
        Input alms.
    icov_pix : (npol, npol, npix) array
        Map that gives diagonal N^{-1} in pixel space.
    icov_ell : (npol, npol, nell) : array
        S^{-1} diagonal in multipole space.
    minfo : sharp.map_info object
        Map info for N^-1 map.
    ainfo : sharp.alm_info object
        alm info for input alm.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.  
    b_ell : (npol, nell) array, optional
        Beam window functions.

    Returns
    -------
    Aalm : (npol, nelem) complex array
        Output vector.
    '''

    # First apply inverse signal cov.
    alm_sig = ainfo.lmul(alm, icov_ell, out=alm.copy())

    # Then apply inverse noise cov.
    alm_sig += b_vec(alm, icov_pix, minfo, ainfo, spin, b_ell=b_ell)

    return alm_sig

def b_vec(alm, icov_pix, minfo, ainfo, spin, b_ell=None):
    '''
    Apply N^-1 to input alm to form the b vector.

    Parameters
    ----------
    alm : (npol, nelem) complex array.
        Input alms.
    icov_pix : (npol, npol, npix) array
        Map that gives diagonal N^{-1} in pixel space.
    minfo : sharp.map_info object
        Map info for N^-1 map.
    ainfo : sharp.alm_info object
        alm info for input alm.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.    
    b_ell : (npol, nell) array, optional
        Beam window functions.

    Returns
    -------
    alm_pix : (npol, nelem) complex array
        N^-1 alm.
    '''

    npol = alm.shape[0] if alm.ndim == 2 else 1
    if npol == 1 and alm.ndim == 1:
        alm = alm[np.newaxis,:]

    alm_pix = alm.copy()
    omap = np.zeros((npol, icov_pix.shape[-1]))
    #if b_ell is not None:
    #    for pidx in range(npol):
    #        hp.almxfl(alm_pix[pidx], b_ell[pidx], inplace=True)
    alm2map(alm, omap, ainfo, minfo, spin, adjoint=False)
    imap = np.einsum('ijk, jk -> ik', icov_pix, omap, optimize=True)
    map2alm(imap, alm_pix, minfo, ainfo, spin, adjoint=True)    
    if b_ell is not None:
        for pidx in range(npol):
            hp.almxfl(alm_pix[pidx], b_ell[pidx], inplace=True)

    return alm_pix

def b_vec_constr(alm, icov_pix, icov_ell, minfo, ainfo, spin, b_ell=None):
    '''
    Form the RHS vector used for constrained realisations: 

    b = N^-1 alm + Yt Np^-1/2 w1 + C^-1/2 w2,

    where w1 is unit variate map and w2 is unit variate alm.

    Parameters
    ----------
    alm : (npol, nelem) complex array.
        Input alms.
    icov_pix : (npol, npol, npix) array
        Map that gives diagonal N^{-1} in pixel space.
    icov_ell : (npol, npol, nell) : array
        S^{-1} diagonal in multipole space.
    minfo : sharp.map_info object
        Map info for N^-1 map.
    ainfo : sharp.alm_info object
        alm info for input alm.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.    
    b_ell : (npol, nell) array, optional
        Beam window functions.

    Returns
    -------
    alm_constr : (npol, nelem) complex array
        Output vector.
    '''

    npol = alm.shape[0] if alm.ndim == 2 else 1
    if npol == 1 and alm.ndim == 1:
        alm = alm[np.newaxis,:]

    alm_constr = b_vec(alm, icov_pix, minfo, ainfo, spin, b_ell=b_ell)

    w1 = map_utils.rand_map_pix(icov_pix)
    # See execute_helper in sharp.pyx, you can get it to add to input alm.
    alm_w1 = np.zeros_like(alm)
    map2alm(w1, alm_w1, minfo, ainfo, spin, adjoint=True)    
    if b_ell is not None:
        for pidx in range(npol):
            hp.almxfl(alm_w1[pidx], b_ell[pidx], inplace=True)

    alm_w2 = curvedsky.rand_alm(icov_ell, ainfo=ainfo, return_ainfo=False)

    alm_constr += alm_w1
    alm_constr += alm_w2

    return alm_constr

def contract_almxblm(alm, blm):
    '''
    Return sum_lm alm x conj(blm), i.e. the sum of the Hadamard product of two
    sets of spherical harmonic coefficients corresponding to real fields.

    Parameters
    ---------
    alm : (..., nelem) complex array
        Healpix ordered (m-major) alm array.
    blm : (..., nelem) complex array
        Healpix ordered (m-major) alm array.

    Returns
    -------
    had_sum : float
        Sum of Hadamard product (real valued).

    Raises
    ------
    ValueError
        If input arrays have different shapes.
    '''

    if blm.shape != alm.shape:
        raise ValueError('Shape alm ({}) != shape blm ({})'.format(alm.shape, blm.shape))

    lmax = hp.Alm.getlmax(alm.shape[-1])
    blm = np.conj(blm)
    csum = complex(np.tensordot(alm, blm, axes=alm.ndim))    
    had_sum = 2 * np.real(csum)

    # We need to subtract the m=0 elements once.
    had_sum -= np.real(np.sum(alm[...,:lmax+1] * blm[...,:lmax+1]))

    return had_sum    
