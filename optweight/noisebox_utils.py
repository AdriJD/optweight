import numpy as np
from scipy.interpolate import CubicSpline

from pixell import enmap

from optweight import wavtrans, map_utils

def noisebox2wavmat(noisebox, bins, w_ell, offsets=[-1, 0, 1]):
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
