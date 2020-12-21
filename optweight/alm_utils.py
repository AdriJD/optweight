import numpy as np

import healpy as hp
from pixell import sharp

from optweight import map_utils, sht, type_utils, wavtrans

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
        raise ValueError('Shape alm ({}) != shape blm ({})'.
                         format(alm.shape, blm.shape))

    lmax = hp.Alm.getlmax(alm.shape[-1])
    blm = np.conj(blm)
    csum = complex(np.tensordot(alm, blm, axes=alm.ndim))
    had_sum = 2 * np.real(csum)

    # We need to subtract the m=0 elements once.
    had_sum -= np.real(np.sum(alm[...,:lmax+1] * blm[...,:lmax+1]))

    return had_sum
#@profile
def alm2wlm_axisym(alm, ainfo, w_ell, lmaxs=None):
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
    lmaxs : (nwav) array lmax values, optional
        Max multipole for each wavelet.

    Returns
    -------
    wlms : (nwav) list of (..., nelem') arrays
        Output wavelets with lmax possibly varing for each wlm.
    winfos : (nwav) list of sharp.alm_info objects
        Metainfo for each wavelet.

    Raises
    ------
    ValueError
        If lmax of alm and w_ell do not match.

    Notes
    -----
    Wavelet kernels are defined as w^x_lm = sum_ell w^x_ell alm.
    '''

    if w_ell.ndim == 1:
        w_ell = w_ell[np.newaxis,:]
    nwav = w_ell.shape[0]

    lmax = ainfo.lmax
    if w_ell.shape[1] != lmax + 1:
        raise ValueError('lmax alm : {} and lmax of w_ell : {} do not match'
                         .format(lmax, w_ell.shape[1]))

    wlms = []
    winfos = []
    for idx in range(nwav):

        # Determine lmax of each wavelet.
        if lmaxs is not None:
            lmax_w = lmaxs[idx]
        else:
            lmax_w = lmax - np.argmax(w_ell[idx,::-1] > 0)

        wlm, winfo = trunc_alm(alm, ainfo, lmax_w)
        winfo.lmul(wlm, w_ell[idx], out=wlm)

        wlms.append(wlm)
        winfos.append(winfo)

    return wlms, winfos
#@profile
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
#@profile
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

def rand_alm_pix(cov_pix, ainfo, minfo, spin):
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
    spin : int, array-like
        Spin values for transform, should be compatible with npol.

    Returns
    -------
    rand_alm : (npol, nelem) array
        Draw from covariance.
    '''

    dtype = type_utils.to_complex(cov_pix.dtype)
    noise = map_utils.rand_map_pix(cov_pix)
    alm_noise = np.zeros((noise.shape[0], ainfo.nelem), dtype=dtype)

    sht.map2alm(noise, alm_noise, minfo, ainfo, spin, adjoint=False)

    return alm_noise

def rand_alm_wav(cov_wav, ainfo, w_ell, spin):
    '''
    Draw random alm from covariance diagonal in wavelet domain.

    Parameters
    ----------
    cov_wav : (nwav, nwav) wavtrans.Wav object.
        Block diagonal wavelet covariance matrix.
    ainfo : sharp.alm_info object
        Metainfo for output alms.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.

    Returns
    -------
    rand_alm : (npol, nelem) array
        Draw from covariance.
    '''

    npol = wavtrans.preshape2npol(cov_wav.preshape)

    alm_dtype = type_utils.to_complex(cov_wav.dtype)
    alm_shape = (npol, ainfo.nelem)
    rand_alm = np.zeros(alm_shape, dtype=alm_dtype)

    for jidx in range(cov_wav.shape[0]):

        cov_pix = cov_wav.maps[jidx,jidx]
        minfo = cov_wav.minfos[jidx, jidx]

        rand_map = map_utils.rand_map_pix(cov_pix)

        # Note, only valid for Gauss Legendre pixels.
        # We use nphi to support maps with cuts in theta.
        lmax = (minfo.nphi[0] - 1) // 2
        winfo = sharp.alm_info(lmax=lmax)

        wlm = np.zeros(alm_shape[:-1] + (winfo.nelem,),
                       dtype=alm_dtype)

        sht.map2alm(rand_map, wlm, minfo, winfo, spin)

        wlm2alm_axisym([wlm], [winfo], w_ell[jidx:jidx+1,:],
                       alm=rand_alm, ainfo=ainfo)

    return rand_alm

def add_to_alm(alm, blm, ainfo, binfo):
    '''
    In-place addition of blm coefficients to alm coefficients.

    Parameters
    ----------
    alm : (..., nelem) array
        Base SH coeffcients.
    blm : (..., nelem') array
        SH coefficients to be added to alm.
    ainfo : sharp.alm_info object
        Metainfo for alm.
    binfo : sharp.alm_info object
        Metainfo for blm.

    Returns
    -------
    alm : (..., nelem) array
        Result of addition.

    Raises
    ------
    ValueError
        If lmax or mmax of blm exceeds that of alm.
    '''

    if binfo.lmax > ainfo.lmax:
        raise ValueError('lmax blm exceeds that of alm : {} > {}'
                         .format(binfo.lmax, ainfo.lmax))

    if binfo.mmax > ainfo.mmax:
        raise ValueError('mmax blm exceeds that of alm : {} > {}'
                         .format(binfo.mmax, ainfo.mmax))

    for m in range(binfo.mmax + 1):

        start_alm = ainfo.lm2ind(m, m)
        end_alm = ainfo.lm2ind(binfo.lmax, m)
        start_blm = binfo.lm2ind(m, m)
        end_blm = binfo.lm2ind(binfo.lmax, m)

        slice_alm = np.s_[...,start_alm:end_alm+ainfo.stride:ainfo.stride]
        slice_blm = np.s_[...,start_blm:end_blm+binfo.stride:binfo.stride]

        alm[slice_alm] += blm[slice_blm]

    return alm
