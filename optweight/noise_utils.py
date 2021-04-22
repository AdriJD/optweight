import numpy as np
from scipy.interpolate import CubicSpline

from pixell import enmap
import healpy as hp

from optweight import wavtrans, map_utils, mat_utils, type_utils

def estimate_cov_wav(alm, ainfo, w_ell, spin, diag=False):
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
        
        cov_pix = estimate_cov_pix(noise_wav.maps[jidx], minfo,
                                   kernel_ell=w_ell[jidx], diag=diag)
        
        cov_wav.add(index, cov_pix, minfo)

    return cov_wav

def estimate_cov_pix(imap, minfo, mask=None, diag=False, fwhm=None, 
                     lmax=None, kernel_ell=None):
    '''
    Estimate noise covariance (uncorelated between pixels, possible correlated 
    between components).

    Arguments
    ---------
    imap : (npol, npix)
        Input noise map.
    minfo : sharp.map_info object
        Metainfo input mask.
    mask : (npol, npix) or (npix) bool array, optional
        Mask, True for unmasked (good) data.
    diag : bool, optional
        If set, only estimate elements diagonal in pol.
    fwhm : float, optional
        Specify the FWHM (in radians) of the Gaussian used for smoothing,
        default is 2 * pi / lmax. If mask is given, this parameter is ignored.
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

    if mask:
        if mask.ndim == 1:
            mask = mask[np.newaxis,:]
        raise NotImplementedError
        
    if diag:
        cov_pix = imap ** 2
    else:
        cov_pix = np.einsum('il, kl -> ikl', imap, imap, optimize=True)

    if mask:
        # Determine w_ell.
        pass
        b_ell = None
    else: 
        b_ell = _band_limit_gauss_beam(map_utils.minfo2lmax(minfo), fwhm=fwhm)

    # Loop over upper triangle or diagonal.
    npol = cov_pix.shape[0]
    elements = range(npol) if diag else zip(*np.triu_indices(npol))

    for idx in elements:
        # Only using spin 0 is an approximation, cov elements are not all spin-0.
        if mask:
            pass
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

def univariate_wav(minfos, preshape, dtype):
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

    Returns
    -------
    wav_uni : wavtrans.Wav object
        Block vector with univariate noise maps.
    '''

    indices = np.arange(len(minfos))
    wav_uni = Wav(1, preshape=preshape, dtype=dtype)

    for widx in indices:
        
        minfo = minfos[widx]
        m_arr = np.random.randn(preshape + (minfo.npix,)).astype(dtype)
        
        wav_uni.add(index, m_arr, minfo)

    return wav_uni
