import numpy as np
from scipy.interpolate import interp1d

from optweight import wlm_utils

def digitize_1d(arr):
    '''
    Turn a smooth array with values between 0 and 1 into an on/off array
    that approximates it.

    Parameters
    ----------
    arr : (N) array
        1D input array containing smooth input.

    Returns
    -------
    out : (N) bool array
        Digital approximation of input.    

    Notes
    -----
    Adapted from pixell.wavelets. Integral of normalized input is 
    conserved in limit of large array.
    '''
    
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError('Input values do not lie between 0 and 1.')

    out = np.round(np.cumsum(arr))
    out = np.concatenate([out[0:1], out[1:] != out[:-1]])
    return out.astype(bool)

def digitize_kernels(w_ell, upsamp_fact=10):
    '''
    Approximate smooth kernels as on/off array and possibly upsample in
    ell.

    Parameters
    ----------
    w_ell : (nwav, nell) array
        Wavelet kernels. Non-neighbouring kernels should not overlap.
    upsamp_fact : int
        Upsample the kernels by this factor in ell.

    Returns
    -------
    d_ell : (nwav, nell * upsamp_fact) bool array
        Digital wavelet kernels.
    ells_upsamp : (nell * upsamp_fact) array
        Upsampled multipoles.

    Raises
    ------
    ValueError
        If non-neighbours input wavelet kernels overlap.
    '''

    nwav = w_ell.shape[0]
    
    err = ValueError('Input `w_ell` has overlapping non-neighbouring kernels')
    EPS = 1e-8
    if nwav > 2 and not np.all(np.prod(w_ell[::2], axis=0) < EPS):
        raise err
    if nwav > 3 and not np.all(np.prod(w_ell[1::2], axis=0) < EPS):    
        raise err

    lmax = w_ell.shape[-1] - 1
    ells = np.arange(0, lmax + 1, 1)
    ells_fine = np.linspace(0, lmax + 1, upsamp_fact * ells.size)

    # Interpolate smooth input.
    w_ell_fine = np.zeros((nwav, ells_fine.size))
    d_ell_fine = np.ones_like(w_ell_fine, dtype=bool)
    for widx in range(nwav):
        w_ell_fine[widx] = interp1d(ells, w_ell[widx], fill_value='extrapolate',
                                    kind='linear')(ells_fine)
    # Digitize even kernels.
    for idx in range(0, nwav, 2):
        d_ell_fine[idx] = digitize_1d(w_ell_fine[idx])

    # Populate odd kernels.
    d_ell_fine[1::2] *= np.logical_xor(
        True, np.sum(d_ell_fine[::2,:], axis=0)[np.newaxis,:] > 0.5)
 
    for idx in range(1, nwav, 2):
        d_ell_fine[idx,w_ell_fine[idx] < 1e-5] = 0

    return d_ell_fine, ells_fine
        
def w_ell2fkernels(w_ell, ells, modlmap, interp_kind):
    '''
    Turn set of 1d wavelet kernels into maps of 2D fourier coefficients.
        
    Parameters
    ----------
    w_ell : (nwav, nell) array
        Wavelet kernels. Non-neighbouring kernels should not overlap.
    ells : (nell) array
        Multiploles corresponding to `w_ell`, allowed to be non-integer.
    modlmap : (nly, nlx) array
        Map of ell (= sqrt(kx^2 + ky^2)) parameters of 2D Fourier coefficients.
    interp_kind : str
        Interpolation method, see `scipy.interpolate.interp1d`.

    Returns
    -------    
    fkernels : (nwav, nly, nlx) array
        Output kernels.
    '''

    nwav = w_ell.shape[0]

    fkernels = np.zeros((nwav,) + modlmap.shape)

    for idx in range(nwav):
        if idx == nwav - 1:
            fill_value = (0., 1.)
        else:
            fill_value = (0., 0.)
        cs = interp1d(ells, w_ell[idx], kind=interp_kind,
                      bounds_error=False, fill_value=fill_value)
        fkernels[idx,:,:] = cs(modlmap)

    return fkernels

def get_sd_kernels_fourier(modlmap, lamb, j0=None, lmin=None, jmax=None,
                           lmax_j=None, digital=True, oversample=10):
    '''
    2D representation of the Scale-Discrete wavelet kernels for 2D Fourier wavelet
    transforms.

    Parameters
    ----------
    modlmap : (nly, nlx) array
        Map of ell (= sqrt(kx^2 + ky^2)) parameters of 2D Fourier coefficients.
    lamb : float
        Lambda parameter specifying the width of the kernels.
    j0 : int, optional
        Minimum J scale used, i.e. the phi wavelet.
    lmin : int, optional
        Multipole after which the first kernel (phi) ends, alternative to j0.
    jmax : int, optional
        Maximum J scale used, i.e.g the omega wavelet.
    lmax_j : int, optional
        Multipole after which the second to last multipole ends, alternative
        to jmax.
    digital : bool
        Produce digital (on/off) version of the kernels.
    oversample : int
        Oversample the SD kernels by this factor with respect to the Fourier ell
        spacing. Only used when `digital` is set.
    
    Returns
    -------
    fkernels : (nwav, nly, nlx) array
        Output kernels. If `digital` was set, this will be a boolean array.
    '''

    lmax = int(np.ceil(modlmap.max()))

    w_ell, _ = wlm_utils.get_sd_kernels(
        lamb, lmax, j0=j0, lmin=lmin, jmax=jmax, lmax_j=lmax_j)

    if digital:
        delta_ell = np.mean([np.abs(modlmap[0,0] - modlmap[0,1]),
                             np.abs(modlmap[0,0] - modlmap[1,0])])
        upsamp_fact = max(1, int(np.round(oversample / delta_ell)))
        w_ell, ells = digitize_kernels(w_ell, upsamp_fact=upsamp_fact)
    else:
        ells = np.arange(w_ell.shape[-1])

    if digital:
        dtype = bool
    else:
        dtype = w_ell.dtype

    interp_kind = 'nearest'
    return w_ell2fkernels(
        w_ell, ells, modlmap, interp_kind).astype(dtype, copy=False)
