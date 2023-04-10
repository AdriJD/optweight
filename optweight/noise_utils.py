import numpy as np
from scipy.interpolate import CubicSpline

from pixell import enmap, sharp
import healpy as hp

from optweight import wavtrans, map_utils, mat_utils, type_utils, wlm_utils, sht, dft

def estimate_cov_wav(alm, ainfo, w_ell, spin, diag=False, wav_template=None,
                     fwhm_fact=2):
    '''
    Estimate wavelet-based covariance matrix given noise alms.
    
    Parameters
    ----------
    alm : (ncomp, npol, nelem) complex array
        Noise alms.
    ainfo : sharp.alm_info object
        Metainfo input alms.
    w_ell : (nwav, nell) array
        Wavelet kernels.    
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    diag : bool, optional
        If set, only estimate elements diagonal in pol.
    wav_template : wavtrans.Wav object, optional
        (nwav) wavelet vector used for alm2wav operation, used as 
        template for cut sky wavelet maps. Will determine minfos
        of output wavelet matrix.
    fwhm_fact : scalar or callable, optional, optional
        Factor determining smoothing scale at each wavelet scale:
        FWHM = fact * pi / lmax, where lmax is the max wavelet ell.
        Can also be a function specifying this factor for a given
        ell. Function must accept a single scalar ell value and 
        return one.
        
    Returns
    -------
    cov_wav : wavtrans.Wav object
        (nwav, nwav) diagonal block covariance matrix.
    '''

    if w_ell.ndim == 1:
        w_ell = w_ell[np.newaxis,:]

    # Subtract monopole, i.e. the mean of the map, before estimating variance.
    alm_mono = alm[...,0,0].copy()
    alm[...,0,0] = 0

    noise_wav = wavtrans.alm2wav(alm, ainfo, spin, w_ell, wav=wav_template)

    # Insert monopole back in.
    alm[...,0,0] = alm_mono

    cov_wav = wavtrans.Wav(2, dtype=type_utils.to_real(alm.dtype))

    # Get fwhm_fact(l) callable.
    def _fwhm_fact(l):
        if callable(fwhm_fact):
            return fwhm_fact(l)
        else:
            return fwhm_fact

    for jidx in range(w_ell.shape[0]):
        # No off-diagonal elements for now.
        index = (jidx, jidx)
        minfo = noise_wav.minfos[jidx]
        
        _lmax = map_utils.minfo2lmax(minfo)
        fwhm = _fwhm_fact(_lmax) * np.pi / _lmax
        cov_pix = estimate_cov_pix(noise_wav.maps[jidx], minfo,
                                   kernel_ell=w_ell[jidx], diag=diag,
                                   fwhm=fwhm)
        cov_wav.add(index, cov_pix, minfo)

    return cov_wav

def estimate_cov_fwav(fmap, fkernelset, wav_template, diag=False,
                     fwhm_fact=2):
    '''
    Estimate wavelet-based covariance matrix given noise 2D fourier
    coefficients.
    
    Parameters
    ----------
    fmap : (..., nly, nlx) complex array
        Input 2D Fourier map.
    fkernels : fkernel.FKernelSet
        Fourier wavelet kernels. 
    wav_template : wavtrans.Wav object
        (nwav) wavelet vector used for f2wav operation, used as 
        template for cut sky wavelet maps. Will determine minfos
        of output wavelet matrix.
    diag : bool, optional
        If set, only estimate elements diagonal in pol.
    fwhm_fact : scalar or callable, optional, optional
        Factor determining smoothing scale at each wavelet scale:
        FWHM = fact * pi / lmax, where lmax is the max wavelet ell.
        Can also be a function specifying this factor for a given
        ell. Function must accept a single scalar ell value and 
        return one.
        
    Returns
    -------
    cov_wav : wavtrans.Wav object
        (nwav, nwav) diagonal block covariance matrix.
    '''
    
    #Subtract monopole, i.e. the mean of the map, before estimating variance.
    fmap_mono = fmap[...,0,0].copy()
    fmap[...,0,0] = 0

    noise_wav = wavtrans.f2wav(fmap, wav_template, fkernelset)

    # Insert monopole back in.
    fmap[...,0,0] = fmap_mono

    cov_wav = wavtrans.Wav(2, dtype=type_utils.to_real(fmap.dtype))

    # Get fwhm_fact(l) callable.
    def _fwhm_fact(l):
        if callable(fwhm_fact):
            return fwhm_fact(l)
        else:
            return fwhm_fact

    #for jidx, fkern in range(fkernelset.shape[0]):
    for jidx, fkern in fkernelset:
        index = (jidx, jidx)
        minfo = noise_wav.minfos[jidx]
        
        #_lmax = int(np.max(modlmap[fkernelset[jidx] > 1e-6]))
        modlmap = fkern.modlmap()
        #_lmax = int(np.max(modlmap[fkern.fkernel > 1e-6]))
        _lmax = fkern.lmax
        #_lmax = int(np.max(fkernel.modlmap() > 1e-6]))

        fwhm = _fwhm_fact(_lmax) * np.pi / _lmax
        cov_pix = estimate_cov_pix(noise_wav.maps[jidx], minfo, diag=diag,
                                   fwhm=fwhm, flatsky=True, modlmap=modlmap)
        # Correct for kernel shape.
        cov_pix /= np.mean(np.abs(fkern.fkernel) ** 2)
        
        cov_wav.add(index, cov_pix, minfo)

    return cov_wav

def estimate_cov_pix(imap, minfo, diag=False, fwhm=None, lmax=None,
                     kernel_ell=None, flatsky=False, modlmap=None):
    '''
    Estimate noise covariance (uncorelated between pixels, possible correlated 
    between polarizations and other components).

    Arguments
    ---------
    imap : (ncomp, npol, npix)
        Input noise map.
    minfo : sharp.map_info object
        Metainfo input map.
    diag : bool, optional
        If set, only estimate elements diagonal in pol.
    fwhm : float, optional
        Specify the FWHM (in radians) of the Gaussian used for smoothing,
        default is 2 * pi / lmax.
    lmax : int, optional
        If set, assume that input noise map has this harmonic band limit.
    kernel_ell : (nell) array, optional
        Assume that this ell-based filter has been applied to the input noise map.
    flatsky : bool, optional
        If set, do flatsky smoothing.
    modlmap : (nly, nlx) array, optional
        Map of absolute wavenumbers. Only needed when `flatsky=True`.
    
    Returns
    -------
    cov_pix : (ncomp, npol, ncomp, npol, npix) or (ncomp, ncomp, npol, npix)
        Estimated covariance matrix, only diagonal in pol if diag is set.

    Notes
    -----
    There are two scenarios: 1) the input noise map is generated in the pixel
    domain or, 2) the input noise map has undergone a band-limiting transformation 
    (i.e. alm2map, so the noise is generated in the spherical harmonic domain 
    before it is transformed to the map domain). Estimating the variance for
    case 1 is straightforward, we simply square and smooth the input map. For
    case 2, we first square and smooth and then divide by the pixel area and 
    the number of modes that are included below the band limit. See 
    "norm_cov_est" which is used when either lmax or kernel_ell are provided.
    '''
    
    imap = mat_utils.atleast_nd(imap, 3)
    ncomp, npol, npix = imap.shape

    if imap.ndim != 3:
        raise ValueError(f'imap is not (ncomp, npol, npix), but {imap.shape}')
            
    if diag:
        cov_pix = np.einsum('abc, dbc -> adbc', imap, imap, optimize=True)
    else:
        cov_pix = np.einsum('abc, dec -> abdec', imap, imap, optimize=True)

    print(np.degrees(fwhm))
    b_ell = _band_limit_gauss_beam(map_utils.minfo2lmax(minfo), fwhm=fwhm)
    if flatsky:
        fb = dft.cl2flat(b_ell, np.arange(b_ell.shape[-1]), modlmap)

    if not diag:
        cov_pix = cov_pix.reshape(ncomp * npol, ncomp * npol, npix)

    for idxs in np.ndindex(cov_pix.shape[:-1]):

        if idxs[0] <= idxs[1]:
            # Upper triangular part.
            # Only using spin 0 is an approximation, cov elements are not all spin-0.
            if fwhm > 0:
                if not flatsky:
                    map_utils.lmul_pix(cov_pix[idxs], b_ell, minfo, 0, inplace=True)
                else:
                    map_utils.fmul_pix(cov_pix[idxs], minfo, fmat2d=fb, inplace=True)
        else:
            # Fill lower triangular. Works because idxs loop is in row-major order.
            cov_pix[idxs] = cov_pix[(idxs[1],idxs[0])+idxs[2:]]

    if lmax or kernel_ell is not None:
        if kernel_ell is None:
            # Assume flat kernel in this case.
            kernel_ell = np.ones(lmax + 1)
        cov_pix = norm_cov_est(cov_pix, minfo, kernel_ell=kernel_ell, inplace=True)

    if not diag:
        cov_pix = mat_utils.get_near_psd(cov_pix, axes=[0,1], inplace=True)
        cov_pix = cov_pix.reshape(ncomp, npol, ncomp, npol, npix)
    else:
        cov_pix[cov_pix < 0] = 0

    return cov_pix

def norm_cov_est(cov_est, minfo, kernel_ell, inplace=False):
    '''
    Scale covariance map such that draws convolved with provided filter
    have the correct power, see estimate_cov_pix.

    Arguments
    ---------
    cov_est : (..., npix) array
        Input maps.
    minfo : sharp.map_info object
        Metainfo input map
    kernel_ell : (nell) array
        The support in multipole of the noise map from which input covariance 
        was derived.
    inplace : bool, optional
        If set, normalize input map inplace.

    Returns
    -------
    omap : (..., npix) array
        Covariance maps.
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

def unit_var_wav(minfos, preshape, dtype, seed=None):
    '''
    Create wavelet block vector with maps filled with unit-variance Gaussian
    noise 
    
    Arguments
    ---------
    minfos : (ndiag) array-like of sharp.map_info objects
        Map info objects describing each wavelet map.
    preshape : tuple
        First dimensions of the maps, i.e. map.shape = preshape + (npix,)
    dtype : type
        Dtype of maps.
    seed : int or np.random._generator.Generator object, optional
        Seed for np.random.seed.

    Returns
    -------
    wav_uni : wavtrans.Wav object
        Block vector with unit-variance noise maps.
    '''

    rng = np.random.default_rng(seed)

    indices = np.arange(len(minfos))
    wav_uni = wavtrans.Wav(1, preshape=preshape, dtype=dtype)

    for widx in indices:
        
        minfo = minfos[widx]
        shape = preshape + (minfo.npix,)
        m_arr = rng.normal(size=shape).astype(dtype)
        
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

def muKarcmin_to_n_ell(noise_level):
    '''
    Convert a noise level in micro K arcmin to to the amplitude
    of the corresponding full-sky, flat noise power spectrum in
    micro K squared.

    Parameters
    ----------
    noise_level : float
        Noise level in uK arcmin.

    Returns
    -------
    amp : float
        Amplitude of power spectrum.
    '''

    return (noise_level * (np.pi / 60 / 180)) ** 2
