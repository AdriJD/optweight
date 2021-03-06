import numpy as np
from scipy.interpolate import CubicSpline

from pixell import enmap, sharp
import healpy as hp

from optweight import wavtrans, map_utils, mat_utils, type_utils, wlm_utils, sht

def estimate_cov_wav(alm, ainfo, w_ell, spin, diag=False, features=None,
                     minfo_features=None):
    '''
    Estimate wavelet-based covariance matrix given noise alms.
    
    Parameters
    ----------
    alm : (npol, nelem) complex array
        Noise alms.
    ainfo : sharp.alm_info object
        Metainfo input alms.
    w_ell : (nwav, nell) array
        Wavelet kernels.    
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    diag : bool, optional
        If set, only estimate elements diagonal in pol.
    features : (npix) array, optional
        Feature map with edges etc. that need to be smoothed less 
        when estimating variance.
    minfo_features : sharp.map_info object, optional
        Metainfo for features map.

    Returns
    -------
    cov_wav : wavtrans.Wav object
        (nwav, nwav) diagonal block covariance matrix.
    '''

    if w_ell.ndim == 1:
        w_ell = w_ell[np.newaxis,:]

    noise_wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell)
    cov_wav = wavtrans.Wav(2, dtype=type_utils.to_real(alm.dtype))

    for jidx in range(w_ell.shape[0]):
            
        # No off-diagonal elements for now.
        index = (jidx, jidx)
        minfo = noise_wav.minfos[jidx]
        
        if features is not None:
            features_j = map_utils.gauss2gauss(features, minfo_features,
                                              minfo, order=1)
        else:
            features_j = None

        cov_pix = estimate_cov_pix(noise_wav.maps[jidx], minfo,
                                   kernel_ell=w_ell[jidx], diag=diag,
                                   features=features_j)
        
        cov_wav.add(index, cov_pix, minfo)

    return cov_wav

def estimate_cov_pix(imap, minfo, features=None, diag=False, fwhm=None, 
                     lmax=None, kernel_ell=None):
    '''
    Estimate noise covariance (uncorelated between pixels, possible correlated 
    between components).

    Arguments
    ---------
    imap : (npol, npix)
        Input noise map.
    minfo : sharp.map_info object
        Metainfo input map.
    features : (npix) array, optional
        Feature map with edges etc. that need to be smoothed less.
    diag : bool, optional
        If set, only estimate elements diagonal in pol.
    fwhm : float, optional
        Specify the FWHM (in radians) of the Gaussian used for smoothing,
        default is 2 * pi / lmax. If features is given, this parameter is ignored.
    lmax : int, optional
        If set, assume that input noise map has this harmonic band limit.
    kernel_ell : (nell) array, optional
        Assume that this ell-based filter has been applied to the input noise map.

    Returns
    -------
    cov_pix : (npol, npol, npix) or (npol, npix)
        Estimated covariance matrix, only diagonal if diag is set.
    '''
    
    if imap.ndim == 1:
        imap = imap[np.newaxis,:]

    if features is not None:
        if features.ndim == 1:
            features = features[np.newaxis,:]
        
    if diag:
        cov_pix = imap ** 2
    else:
        cov_pix = np.einsum('il, kl -> ikl', imap, imap, optimize=True)

    if features is not None:
        lmax_w = map_utils.minfo2lmax(minfo)        
        _, w_ell = minimum_w_ell_lambda(lmax_w, lmax_w // 2, (4 * lmax_w) // 5,
                                        return_w_ell=True)
        b_ell = None
    else: 
        b_ell = _band_limit_gauss_beam(map_utils.minfo2lmax(minfo), fwhm=fwhm)
        lmax_w = None
        w_ell = None

    # Loop over upper triangle or diagonal.
    npol = cov_pix.shape[0]
    elements = range(npol) if diag else zip(*np.triu_indices(npol))

    for idx in elements:
        # Only using spin 0 is an approximation, cov elements are not all spin-0.
        if features is not None:
            smooth_locally(cov_pix[idx], minfo, w_ell, features, 0, inplace=True)
        else:
            map_utils.lmul_pix(cov_pix[idx], b_ell, minfo, 0, inplace=True)

    if not diag:
        # Fill lower triangle.
        for idx, jdx in zip(*np.tril_indices(npol, k=-1)):
            cov_pix[idx,jdx] = cov_pix[jdx,idx]
                    
    if lmax or kernel_ell is not None:
        if kernel_ell is None:
            # Assume flat kernel in this case.
            kernel_ell = np.ones(lmax + 1)
        cov_pix = norm_cov_est(cov_pix, minfo, kernel_ell=kernel_ell, inplace=True)

    cov_pix = mat_utils.get_near_psd(cov_pix)
    
    return cov_pix

def norm_cov_est(cov_est, minfo, kernel_ell, inplace=False):
    '''
    Scale covariance map such that draws convolved with provided filter
    have the correct power, see estimate_cov_pix.

    Arguments
    ---------
    cov_est : (npol, npol, npix) or (npol, npix) array
        Input map, if diagonal only diagonal suffices.
    minfo : sharp.map_info object
        Metainfo input map
    kernel_ell : (nell) array
        The support in multipole of the noise map from which input covariance 
        was derived.
    inplace : bool, optional
        If set, normalize input map inplace.

    Returns
    -------
    omap : (npol, npol, npix) or (npol, npix) array
        Covariance map.
    '''

    ells = np.arange(kernel_ell.size)

    omap = map_utils.inv_qweight_map(cov_est, minfo, inplace=inplace)    
    omap /= np.sum(kernel_ell ** 2 * (2 * ells + 1)) / 4 / np.pi 

    return omap

def noisebox2wavmat(noisebox, bins, w_ell, offsets=[-1, 0, 1],
                    rtol_icov=0.1):
    '''
    Convert noisebox to block wavelet matrix.
    
    Parameters
    ----------
    noisebox : (npol, nbins, ny, nx) enmap
        Generalized map containing a power spectrum per pixel.
    bins : (nbins) array
        Multipole bins of power spectra.
    w_ell : (nwav, nell) array
        Wavelet kernels.    
    offsets : array-like of int, optional
        Diagonals of block matrix to consider. Positive numbers
        refer to the kth upper diagonal, negative numbers to the 
        kth lower diagonal.
    rtol_icov : float, optional
        Inverse covariance pixels below this factor times the median
        of nonzero pixels are set to zero.

    Returns
    -------
    icov_wav : wavtrans.Wav object
        Inverse covariance wavelet matrix.
        
    Notes
    -----
    Units of noisebox power spectra are assumed to be (uK arcmin)^-2,
    which matches the noisebox of the DR5 release.
    '''
    
    lmax = w_ell.shape[1] - 1
    ells = np.arange(lmax + 1)
    w_ell = np.ascontiguousarray(w_ell[:,:lmax+1])
    j_scales = np.arange(w_ell.shape[0])
    n_j = j_scales.size

    wcs = noisebox.wcs
    pix_areas = enmap.pixsizemap(noisebox.shape[-2:], wcs)

    # Interpolate and unit conversion.
    noisebox_full = prepare_noisebox(noisebox, bins, lmax)

    wavelet_matrix = np.einsum('ij,kj->ikj', w_ell, w_ell, optimize=True)
    wavelet_matrix *= (2 * ells + 1) / 4 / np.pi

    # Determine lmax of all wavelet kernels.
    lmaxs = np.zeros(j_scales.size, dtype=int)
    for jidx in j_scales:
        lmaxs[jidx] = lmax - np.argmax(w_ell[jidx,::-1] > 0)

    icov_wav = wavtrans.Wav(2, dtype=noisebox_full.dtype)

    for jidx in j_scales:
        for offset in offsets:
            
            jpidx = jidx + offset
            if jpidx < 0 or jpidx >= n_j:
                continue         
            index = (jidx, jpidx)

            # Convert to icov_pix using legendre formula.
            icov_pix = np.einsum('j, ijkl -> ikl', wavelet_matrix[index], 
                                 noisebox_full, optimize=True)
            icov_pix = enmap.enmap(icov_pix, wcs=wcs, copy=False)
           
            sum_kernel = np.sum(wavelet_matrix[index])
            if sum_kernel != 0:
                # If zero, icov_pix is zero too so its okay to skip this.
                icov_pix *= pix_areas / sum_kernel
            
            # Column determines lmax -> enables matrix-vector operations.
            icov_pix, minfo = map_utils.enmap2gauss(
                icov_pix, 2 * lmaxs[jpidx], area_pow=1, mode='nearest')

            icov_pix = map_utils.round_icov_matrix(
                icov_pix, rtol=rtol_icov)
            
            icov_wav.add(index, icov_pix, minfo)

    return icov_wav

def prepare_noisebox(noisebox, bins, lmax):
    '''
    Scale noisebox from (uK arcmin)^-2 to uK^-2 and interpolate
    to all multipoles.

    Arguments
    ---------
    noisebox : (npol, nbins, ny, nx) enmap
        Generalized map containing a power spectrum per pixel.
    bins : (nbins) array
        Multipole bins of power spectra.
    lmax : int
        Maximum multipole of output.

    Returns
    -------
    noisebox_full : (npol, nell, ny, nx) enmap
        Scaled and interpolated noisebox.
    '''

    if noisebox.ndim != 4:
        raise ValueError('Noisebox should be 4-D, not {}-D'.
                         format(noisebox.ndim))

    shape = noisebox.shape
    wcs = noisebox.wcs
    if bins[0] != 0:
        # Extend noisebox to ell = 0. Assuming white noise.
        noisebox_ext = np.zeros(
            shape[:1] + (shape[1] + 1,) + shape[2:])
        noisebox_ext[:,1:,:,:] = noisebox
        noisebox_ext[:,0,:,:] = noisebox[:,0,:,:]

        bins_ext = np.zeros(len(bins) + 1)
        bins_ext[0] = 0
        bins_ext[1:] = bins

    else:
        noisebox_ext = noisebox.copy()
        bins_ext = bins

    # Scale icov from (uK arcmin)^-2 to uK^-2. 
    # 10800 corresponds to l = 180 / arcmin = 180 / (1 / 60),
    # i.e. number of modes on the sky for an arcmin resolution.
    noisebox_ext /= 4 * np.pi / (10800 ** 2)

    ells = np.arange(lmax + 1)
    noisebox_full = np.zeros(shape[:1] + (lmax + 1,) + shape[2:])

    cs = CubicSpline(bins_ext, noisebox_ext, axis=1)
    noisebox_full[...] = cs(ells)
    noisebox_full = enmap.enmap(noisebox_full, wcs=wcs, copy=False)

    return noisebox_full

def _band_limit_gauss_beam(lmax, fwhm=None):
    '''
    Return Gaussian beam window function with band limit roughly at lmax.

    Arguments
    ---------
    lmax : int
        Bandlimit
    fwhm : float, optional
        FWHM in radians to overwrite default.
    
    Returns
    -------
    b_ell : (lmax+1) array
        Gaussian beam window function
    '''
    
    if fwhm is None:
        fwhm = 2 * np.pi / lmax 

    return hp.gauss_beam(fwhm, lmax=lmax)

def univariate_wav(minfos, preshape, dtype, seed=None):
    '''
    Create wavelet block vector with maps filled with univariate Gaussian
    noise 
    
    Arguments
    ---------
    minfos : (ndiag) array-like of sharp.map_info objects
        Map info objects describing each wavelet map.
    preshape : tuple
        First dimensions of the maps, i.e. map.shape = preshape + (npix,)
    dtype : type
        Dtype of maps.
    seed : int, optional
        Seed for np.random.seed.

    Returns
    -------
    wav_uni : wavtrans.Wav object
        Block vector with univariate noise maps.
    '''

    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(minfos))
    wav_uni = wavtrans.Wav(1, preshape=preshape, dtype=dtype)

    for widx in indices:
        
        minfo = minfos[widx]
        shape = preshape + (minfo.npix,)
        m_arr = np.random.randn(*shape).astype(dtype)
        
        wav_uni.add(widx, m_arr, minfo)

    return wav_uni

def minimum_w_ell_lambda(lmax, lmin, lmax_j, delta_lamb=0.04, return_w_ell=False):
    '''
    Find the minimum lambda parameter (see wlm_utils) that will result in 
    only a single wavelet between lmin and lmax_j.

    Parameters
    ----------
    lmax : int
        Maximum mulltipole for wavelet kernels.
    lmin : int, optional
        Multipole after which the first kernel (phi) ends.
    lmax_j : int, optional
        Multipole after which the second to last multipole ends.
    delta_lamb : float, optional
        Search for lambda using these intervals.
    return_w_ell : bool, optional
        If set, return wavelet kernels corresponding to lambda.

    Returns
    -------
    lamb : float
        Minumum lambda value.
    w_ell : (nwav, nell) array
        Wavelet kernels, if return_w_ell is set.

    Raises
    ------
    ValueError
        If no value can be found.
    '''

    lamb = 1.01
    w_ell, _, js = wlm_utils.get_sd_kernels(lamb, lmax, lmin=lmin,
                                        lmax_j=lmax_j, return_j=True) 
    while len(js) > 3 and lamb < 10.:
        lamb += delta_lamb
        w_ell, _, js = wlm_utils.get_sd_kernels(lamb, lmax, lmin=lmin,
                                            lmax_j=lmax_j, return_j=True) 

    if len(js) != 3:
        raise ValueError(f'Did not found suitable lambda, terminated at '
                         f'lambda = {lamb} giving len(js) = {len(js)}')

    if return_w_ell:
        return lamb, w_ell
    else:
        return lamb

def smooth_locally(imap, minfo, w_ell, features, spin, inplace=False):
    '''
    Perform smoothing that varies in strength accross the map.

    Arguments
    ---------
    imap : (1, npix)
        Input map.
    minfo : sharp.map_info object
        Metainfo input map.
    features : (1, npix) array
        Feature map with edges etc. that need to be smoothed less.
    w_ell : (nwav, nell) array
        Wavelet kernels.    
    spin : int
        Spin value for SH transforms.
    inplace : bool
        Perform computation inplace.

    returns
    -------
    omap : npol, npix)
        Smooth output map.

    Raises
    ------
    ValueError
        If nwav != 3.
        If input maps are not 1d.
    '''

    nwav = w_ell.shape[0]
    if nwav != 3:
        raise ValueError(f'Expected nwav = 3, got {nwav}')

    if imap.ndim == 1:
        imap = imap[np.newaxis,:]

    if features.ndim == 1:
        features = features[np.newaxis,:]
    
    if imap.ndim != 2 or imap.shape[0] != 1:
        raise ValueError(f'Input map shape : {imap.shape} not supported')

    if features.ndim != 2 or features.shape[0] != 1:
        raise ValueError(f'Features map shape : {features.shape} not supported')

    if not inplace:
        omap = np.empty(imap.shape, imap.dtype)
    else:
        omap = imap

    lmax = map_utils.minfo2lmax(minfo)        
    ainfo = sharp.alm_info(lmax)
    alm = np.empty((1, ainfo.nelem), dtype=type_utils.to_complex(imap.dtype))

    sht.map2alm(imap, alm, minfo, ainfo, spin)
    wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell)
    wav.maps[1] *= map_utils.gauss2gauss(features, minfo, wav.minfos[1], order=1)
    wav.maps[2] *= 0
    wavtrans.wav2alm(wav, alm, ainfo, spin, w_ell)
    sht.alm2map(alm, omap, ainfo, minfo, spin)

    return omap
