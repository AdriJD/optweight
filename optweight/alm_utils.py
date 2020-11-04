import numpy as np

import healpy as hp
from pixell import sharp

from optweight import map_utils
from optweight import sht

def contract_almxblm(alm, blm):
    '''
    Return sum_lm alm x conj(blm), i.e. the sum of the Hadamard product of two
    sets of spherical harmonic coefficients corresponding to real fields.

    Parameters
    ---------
    alm : (..., nelem) complex array
        Healpix-ordered (m-major) alm array.
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

def alm2wlm_axisym(alm, ainfo, w_ell):
    '''
    Convert SH coeffients into wavelet coefficients.

    Parameters
    ----------
    alm : (..., nelem) array
        Input alm array
    ainfo : sahrp.alm_info object
        Metainfo for input alms.
    w_ell : (nwav, nell) array
        Wavelet kernels.

    Returns
    -------
    wlms : (nwav) list of (..., nelem') arrays
        Output wavelets with lmax possibly varing for each wlm.
    winfos : (nwav) list of sharp.alm_info objects
        Metainfo for each wavelet.

    Notes
    -----
    Wavelet kernels are defined as w^x_lm = sum_ell w^x_ell alm.
    '''
    
    if w_ell.ndim == 1:
        w_ell = w_ell[np.newaxis,:]
    nwav = w_ell.shape[0]

    lmax = ainfo.lmax

    wlms = []
    winfos = []
    for idx in range(nwav):

        # Determine lmax of each wavelet.
        lmax_w = lmax - np.argmax(w_ell[idx,::-1] > 0)

        wlm, winfo = trunc_alm(alm, ainfo, lmax_w)        
        winfo.lmul(wlm, w_ell[idx], out=wlm)

        wlms.append(wlm)
        winfos.append(winfo)

    return wlms, winfos

def wlm2alm_axisym(wlms, winfos, w_ell, alm=None, ainfo=None):
    '''
    Convert wavelet coeffients into alm coefficients.

    Parameters
    ----------
    wlms : (nwav) list of (..., nelem') arrays
        Input wavelets with lmax possibly varing for each wlm.
    winfos : (nwav) list of sharp.alm_info object
        Metainfo for each wavelet.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    alm : (..., nelem) array, optional
        Output alm array, results will be added to array!
    ainfo : sahrp.alm_info object, optional
        Metainfo for output alms.

    Returns
    -------
    alm : (..., nelem) array
        Output alm array
    ainfo : sahrp.alm_info object
        Metainfo for output alms.

    Raises
    ------
    ValueError
        If lmax alm < lmax wavelets
    NotImplementedError
        If ainfo has stride != 1 and no output alm is given.
    
    Notes
    -----
    Wavelet kernels are defined as alm = sum_x w^x_lm * w^x_ell.
    '''

    lmax_w = np.max([winfo.lmax for winfo in winfos])
    
    if ainfo is None:        
        ainfo = sharp.alm_info(lmax=lmax_w)
    else:
        if ainfo.lmax < lmax_w:
            raise ValueError('lmax alm {} < lmax wavelets {}'.
                             format(ainfo.lmax, lmax_w))

    if alm is None:      
        if ainfo.stride != 1:
            raise NotImplementedError('Cannot create alm for ainfo with stride != 1')
        alm = np.zeros(wlms[0].shape[:-1] + (ainfo.nelem,), dtype=wlms[0].dtype)

    stride = ainfo.stride
    w_ell = w_ell.astype(wlms[0].dtype)

    for widx in range(len(wlms)):
        winfo = winfos[widx]
        wlm = wlms[widx]            
        stride_wlm = winfo.stride
        lmax_wlm = winfo.lmax

        for m in range(winfo.mmax + 1):

            start_alm = ainfo.lm2ind(m, m)        
            end_alm = ainfo.lm2ind(lmax_wlm, m)
            start_wlm = winfo.lm2ind(m, m)
            end_wlm = winfo.lm2ind(lmax_wlm, m)
            
            slice_alm = np.s_[...,start_alm:end_alm+stride:stride]
            slice_wlm = np.s_[...,start_wlm:end_wlm+stride_wlm:stride_wlm]

            alm[slice_alm] += wlm[slice_wlm] * w_ell[widx,m:lmax_wlm+1]
                
    return alm, ainfo

def trunc_alm(alm, ainfo, lmax):
    '''
    Truncate alm to lmax.

    Parameters
    ----------
    alm : array
        Input alm.
    ainfo : sharp.alm_info object
        Meta info input alm.
    lmax : int
        New lmax.

    Returns
    -------
    alm_trunc : array
        Output alm.
    ainfo_trunc : sharp.alm_info object
        Metainfo output alm.

    Raises
    ------
    ValueError
        If new lmax > old lmax.
    '''
    
    lmax_old = ainfo.lmax
    if lmax > lmax_old:
        raise ValueError('New lmax {} exceeds old lmax {}'.format(lmax, lmax_old))

    if lmax_old == int(ainfo.nelem / (ainfo.mmax + 1)) - 1:
        layout = 'rect'
    else:
        layout = 'tri'
        
    mmax = min(ainfo.mmax, lmax)
    stride = ainfo.stride

    ainfo_trunc = sharp.alm_info(
        lmax=lmax, mmax=mmax, stride=stride, layout=layout)

    alm_trunc = np.zeros(alm.shape[:-1] + (ainfo_trunc.nelem,), dtype=alm.dtype)

    for m in range(mmax + 1):
        start_trunc = ainfo_trunc.lm2ind(m, m)
        start_old = ainfo.lm2ind(m, m)
        end_trunc = ainfo_trunc.lm2ind(lmax, m)
        end_old = ainfo.lm2ind(lmax, m)
        
        slice_trunc = np.s_[...,start_trunc:end_trunc+stride:stride]
        slice_old = np.s_[...,start_old:end_old+stride:stride]

        alm_trunc[slice_trunc] = alm[slice_old]

    return alm_trunc, ainfo_trunc

def rand_alm_pix(cov_pix, ainfo, minfo, dtype=np.complex128):
    '''
    Draw random alm from covariance diagonal in pixel domain.

    Parameters
    ----------
    cov_pix : (npol, npol, npix) or (npol, npix) array
        Covariance diagonal in pixel space.
    ainfo : sharp.alm_info object
        Metainfo for output alms.
    minfo : sharp.map_info object
        Metainfo specifying pixelization of covariance.
    dtype : type
        dtype of output alms.

    Returns
    -------
    rand_alm : (npol, nelem) array
        Draw from covariance.
    '''
    
    noise = map_utils.rand_map_pix(cov_pix)
    alm_noise = np.zeros((noise.shape[0], ainfo.nelem), dtype=dtype)
    sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=False)

    return alm_noise
    
