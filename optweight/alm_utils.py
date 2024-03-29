import numpy as np

import healpy as hp
from pixell import curvedsky

from optweight import (map_utils, sht, type_utils, wavtrans, alm_c_utils,
                       mat_utils, operators)

def contract_almxblm(alm, blm):
    '''
    Return sum_lm alm x conj(blm), i.e. the sum of the Hadamard product of two
    sets of spherical harmonic coefficients corresponding to real fields.

    Parameters
    ----------
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
    winfos : (nwav) list of pixell.curvedsky.alm_info objects
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
        alm_c_utils.lmul(wlm, w_ell[idx,:winfo.lmax+1], winfo, inplace=True)

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
    winfos : (nwav) list of pixell.curvedsky.alm_info object
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
        ainfo = curvedsky.alm_info(lmax=lmax_w)
    else:
        if ainfo.lmax < lmax_w:
            raise ValueError('lmax alm {} < lmax wavelets {}'.
                             format(ainfo.lmax, lmax_w))

    if alm is None:
        if ainfo.stride != 1:
            raise NotImplementedError('Cannot create alm for ainfo with stride != 1')
        alm = np.zeros(wlms[0].shape[:-1] + (ainfo.nelem,), dtype=wlms[0].dtype)

    stride = ainfo.stride    
    lmax_a = ainfo.lmax

    for widx in range(len(wlms)):
        winfo = winfos[widx]
        wlm = wlms[widx]
        stride_wlm = winfo.stride
        lmax_wlm = winfo.lmax

        if stride == 1 and stride_wlm == 1:
            # Fast method.
            alm_c_utils.wlm2alm(w_ell[widx], wlm, alm, lmax_wlm, lmax_a)

        else:
            # Slow method.
            for m in range(winfo.mmax + 1):

                start_alm = ainfo.lm2ind(m, m)
                end_alm = ainfo.lm2ind(lmax_wlm, m)
                start_wlm = winfo.lm2ind(m, m)
                end_wlm = winfo.lm2ind(lmax_wlm, m)

                w_ells = w_ell[widx,m:lmax_wlm+1].astype(wlms[widx].dtype) 

                slice_alm = np.s_[...,start_alm:end_alm+stride:stride]
                slice_wlm = np.s_[...,start_wlm:end_wlm+stride_wlm:stride_wlm]

                alm[slice_alm] += wlm[slice_wlm] * w_ells

    return alm, ainfo

def trunc_alm(alm, ainfo, lmax):
    '''
    Truncate alm to lmax.

    Parameters
    ----------
    alm : array
        Input alm.
    ainfo : pixell.curvedsky.alm_info object
        Meta info input alm.
    lmax : int
        New lmax.

    Returns
    -------
    alm_trunc : array
        Output alm.
    ainfo_trunc : pixell.curvedsky.alm_info object
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

    ainfo_trunc = curvedsky.alm_info(
        lmax=lmax, mmax=mmax, stride=stride, layout=layout)

    alm_trunc = np.zeros(alm.shape[:-1] + (ainfo_trunc.nelem,), dtype=alm.dtype)

    if layout == 'tri' and ainfo.mmax == ainfo.lmax:
        alm_c_utils.trunc_alm(alm, alm_trunc, ainfo.lmax, lmax)

    else:
        # Slower but more general version.
        for m in range(mmax + 1):
            start_trunc = ainfo_trunc.lm2ind(m, m)
            start_old = ainfo.lm2ind(m, m)
            end_trunc = ainfo_trunc.lm2ind(lmax, m)
            end_old = ainfo.lm2ind(lmax, m)

            slice_trunc = np.s_[...,start_trunc:end_trunc+stride:stride]
            slice_old = np.s_[...,start_old:end_old+stride:stride]

            alm_trunc[slice_trunc] = alm[slice_old]

    return alm_trunc, ainfo_trunc

def unit_var_alm(ainfo, preshape, seed, out=None, dtype=np.complex128):
    '''
    Draw unit-variance Gaussian random alm.

    Parameters
    ----------
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for output alms.
    preshape : tuple
        Leading dimensions of output array.
    seed : int or np.random._generator.Generator object
        Seed for np.random.seed.
    out : (pre,) + (nelem,) complex array, optional
        If provided, use this array for output.
    dtype : type
        Output dtype. Pick between complex64 and 128.

    Returns
    -------
    out : (pre,) + (nelem,) array
        Output alms.

    Raises
    ------
    ValueError
        If dtype is not supported
        If out array does not match expected shape or dtype.
    '''

    if not dtype in (np.complex64, np.complex128):
        raise ValueError(f'{dtype=} not supported')
    
    oshape = preshape + (ainfo.nelem,)
    if out is not None:
        if out.shape != oshape or out.dtype != dtype:
            raise ValueError(
                f'{oshape=}, {dtype=} != {out.shape=}, {out.dtype=}')
    else:
        out = np.zeros(oshape, dtype=dtype)
    out = mat_utils.atleast_nd(out, 2)
    shape_nd = out.shape    
    # Unlike pixell we do not draw in ell-major order. We don't really
    # need low-lmax draws and high-lmax draws to agree and we will
    # often draw 3-d alms for which that argument does not hold anyway.
    rng = np.random.default_rng(seed)
    rtype = type_utils.to_real(out.dtype)
    out = out.reshape(-1)
    out = out.view(rtype)
    
    out[:] = np.reshape(
        rng.normal(
            scale=1 / np.sqrt(2), size=out.size).astype(rtype, copy=False),
        out.shape)
    out = out.view(dtype)
    out = out.reshape(shape_nd)
    out[...,:ainfo.lmax+1].imag = 0
    out[...,:ainfo.lmax+1].real *= np.sqrt(2)

    out = out.reshape(oshape)
    
    return out

def rand_alm(cov_ell, ainfo, seed, out=None, dtype=np.complex128):
    '''
    Draw alm from covariance matrix.

    Parameters
    ----------
    cov_ell : (npol, npol, nell) array or (npol, nell) array
        Covariance matrix, either symmetric but dense in first two axes
        or diagonal, in which case only the diagonal elements are needed.
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for output alms.
    seed : int or np.random._generator.Generator object
        Seed for np.random.seed.
    out : (pre,) + (nelem,) complex array, optional
        If provided, use this array for output.    
    dtype : type
        Output dtype. Pick between complex64 and 128.

    Returns
    -------
    draw : (npol, nelem) array
        Random alms. 
    '''

    cov_ell = mat_utils.atleast_nd(cov_ell, 2)
    
    rand_alm = unit_var_alm(
        ainfo, (cov_ell.shape[0],), seed, out=out, dtype=dtype)
    return operators.EllMatVecAlm(
        ainfo, cov_ell, power=0.5, inplace=True)(rand_alm)
    
def rand_alm_pix(cov_pix, ainfo, minfo, spin, seed, adjoint=False):
    '''
    Draw random alm from covariance diagonal in pixel domain.

    Parameters
    ----------
    cov_pix : (npol, npol, npix) or (npol, npix) array
        Covariance diagonal in pixel space.
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for output alms.
    minfo : map_utils.MapInfo object
        Metainfo specifying pixelization of covariance.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    seed : int or np.random._generator.Generator object
        Seed for np.random.seed.    
    adjoint : bool, optional
        If set, calculate Yt N^0.5 r, instead of Yt W N^0.5 r, where
        r ~ N(0,1), N^0.5 is square root of pixel-based noise model and
        W are the SH quadrature weights.

    Returns
    -------
    rand_alm : (npol, nelem) array
        Draw from covariance.
    '''

    dtype = type_utils.to_complex(cov_pix.dtype)
    noise = map_utils.rand_map_pix(cov_pix, seed)
    alm_noise = np.zeros((noise.shape[0], ainfo.nelem), dtype=dtype)

    sht.map2alm(noise, alm_noise, minfo, ainfo, spin, adjoint=adjoint)

    return alm_noise

def rand_alm_wav(cov_wav, ainfo, w_ell, spin, seed, adjoint=False):
    '''
    Draw random alm from covariance diagonal in wavelet domain.

    Parameters
    ----------
    cov_wav : (nwav, nwav) wavtrans.Wav object.
        Block diagonal wavelet covariance matrix.
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for output alms.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    seed : int or np.random._generator.Generator object
        Seed for np.random.seed.        
    adjoint : bool, optional
        Compute Kt Yt N^0.5 rand_pix instead of Kt Yt W N^0.5 rand_pix.

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

        rand_map = map_utils.rand_map_pix(cov_pix, seed)

        # Note, only valid for Gauss Legendre pixels.
        # We use nphi to support maps with cuts in theta.
        lmax = (minfo.nphi[0] - 1) // 2
        winfo = curvedsky.alm_info(lmax=lmax)

        wlm = np.zeros(alm_shape[:-1] + (winfo.nelem,),
                       dtype=alm_dtype)

        sht.map2alm(rand_map, wlm, minfo, winfo, spin, adjoint=adjoint)

        wlm2alm_axisym([wlm], [winfo], w_ell[jidx:jidx+1,:],
                       alm=rand_alm, ainfo=ainfo)

    return rand_alm

def add_to_alm(alm, blm, ainfo, binfo, overwrite=False):
    '''
    In-place addition of blm coefficients to alm coefficients.

    Parameters
    ----------
    alm : (..., nelem) array
        Base SH coeffcients.
    blm : (..., nelem') array
        SH coefficients to be added to alm.
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for alm.
    binfo : pixell.curvedsky.alm_info object
        Metainfo for blm.
    overwrite : bool, optional
        If set, do not add but overwrite alm with blm.

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

        if overwrite:
            alm[slice_alm] = blm[slice_blm]
        else:
            alm[slice_alm] += blm[slice_blm]

    return alm

def ainfo_is_equiv(ainfo_1, ainfo_2):
    '''
    Test whether two alm info objects are equivalent.

    Parameters
    ----------
    ainfo_1 : pixell.curvedsky.alm_info object
        First alm info object.
    ainfo_2 : pixell.curvedsky.alm_info object
        Second alm info object.

    Returns
    -------
    is_equiv : bool
        True if equivalent, False if not.
    '''

    is_equiv = True
    attributes = ['lmax', 'mmax', 'stride', 'nelem', 'mstart']

    for attr in attributes:
        try:
            np.testing.assert_allclose(
                getattr(ainfo_1, attr), getattr(ainfo_2, attr))
        except AssertionError:
            is_equiv = False
            break

    return is_equiv
    
