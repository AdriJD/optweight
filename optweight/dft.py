'''
Routines for real-to-complex and complex-to-real FFTs. Adapted from pixell and mnms.
'''
import numpy as np

from pixell import fft, enmap, wcsutils

def rfft(emap, fmap, normalize=True, adjoint=False):
    '''
    Real-to-complex FFT.

    Parameters
    ----------
    emap : (..., ny, nx) ndmap
        Map to transform.
    fmap : (..., ny, nx//2+1) complex ndmap
        Output buffer.
    normalize : bool, optional
        The FFT normalization, by default True. If True, normalize
        using pixel number. If in ['phy', 'phys', 'physical'],
        normalize by sky area.
    adjoint : bool, optional
        Compute adjoint of the complex-to-real FFT.
    '''

    fmap = fft.rfft(emap, fmap, axes=[-2, -1])
    norm = 1

    if normalize:
        norm /= np.prod(emap.shape[-2:]) ** 0.5
    if normalize in ["phy","phys","physical"]:
        if adjoint:
            norm /= emap.pixsize() ** 0.5
        else:
            norm *= emap.pixsize() ** 0.5
    if norm != 1:
        fmap *= norm

def irfft(fmap, omap, normalize=True, adjoint=False, destroy_input=False):
    '''
    Complex-to-real FFT.

    Parameters
    ----------
    fmap : (..., ny, nx//2+1) complex ndmap
        Input Fourier coefficients.
    omap : (..., ny, nx) ndmap
        Output buffer.
    normalize : bool, optional
        The FFT normalization, by default True. If True, normalize
        using pixel number. If in ['phy', 'phys', 'physical'],
        normalize by sky area.
    adjoint : bool, optional
        Compute the adjoint of the real-to-complex FFT.
    destroy_input : bool, optional
        If set, input `fmap` array might be overwritten.
    '''

    if not destroy_input:
        fmap = fmap.copy()
    omap = fft.irfft(fmap, omap, axes=[-2, -1], normalize=False)
    norm = 1

    if normalize:
        norm /= np.prod(omap.shape[-2:]) ** 0.5
    if normalize in ["phy","phys","physical"]:
        if adjoint:
            norm *= fmap.pixsize() ** 0.5
        else:
            norm /= fmap.pixsize() ** 0.5
    if norm != 1:
        omap *= norm

def laxes_real(shape, wcs):
    '''
    Compute ell_x and ell_y axes corresponding to a given enmap geometry.

    Arguments
    ---------
    shape : tuple
        Shape of geometry.
    wcs : astropy.wcs.WCS object
        WCS object describing geometry.

    Returns
    -------
    ly : (nly) array
        Wavenumbers in y direction.
    lx : (nlx) array
        Wavenumbers in x direction.
    '''

    step = enmap.extent(shape, wcs, signed=True, method='auto') / shape[-2:]
    ly = np.fft.fftfreq(shape[-2], step[0]) * 2 * np.pi
    lx = np.fft.rfftfreq(shape[-1], step[1]) * 2 * np.pi

    return ly, lx

def lmap_real(shape, wcs, dtype=np.float64):
    '''
    Return maps of all the wavenumbers corresponding to a given enmap geometry.

    Arguments
    ---------
    shape : tuple
        Shape of geometry.
    wcs : astropy.wcs.WCS object
        WCS object describing geometry.
    dtype : type, optional
        Type of ouput array.

    Returns
    -------
    lmap : (2, nly, nlx) array
       Maps of ell_y and ell_x wavenumbers.
    '''

    ly, lx = laxes_real(shape, wcs)
    lmap = np.empty((2, ly.size, lx.size), dtype=dtype)
    lmap[0] = ly[:,np.newaxis]
    lmap[1] = lx[np.newaxis,:]

    return lmap

def modlmap_real(shape, wcs, dtype=np.float64):
    '''
    Return a map of all the abs wavenumbers in the fourier transform
    of a map with the given shape and wcs.

    Arguments
    ---------
    shape : tuple
        Shape of geometry.
    wcs : astropy.wcs.WCS object
        WCS object describing geometry.
    dtype : type, optional
        Type of ouput array.

    Returns
    -------
    lmod : (nly, nlx) array
       Map of absolute wavenumbers.
    '''

    slmap = lmap_real(shape, wcs, dtype=dtype)
    lmod = np.sum(slmap ** 2, axis=0) ** 0.5

    return lmod

def lwcs_real(shape, wcs):
    '''
    Build world coordinate system for Fourier space (with reality symmetry) given
    enmap geometry.

    Arguments
    ---------
    shape : tuple
        Shape of geometry.
    wcs : astropy.wcs.WCS object
        WCS object describing geometry.

    Returns
    -------
    lwcs : astropy.wcs.WCS object
        WCS object describing Fourier space geometry.
    '''

    lres = 2 * np.pi / enmap.extent(shape, wcs, signed=True)
    ny = shape[-2]
    
    return wcsutils.explicit(crpix=[0,ny//2+1], crval=[0,0], cdelt=lres[::-1])

def lbin(fmap, lmod, bsize=None):
    '''
    Bin Fourier-space map into radial bins.
    
    Arguments
    ---------
    fmap : (..., nly, nlx) array.
        Input 2D Fourier map.
    lmod : (nly, nlx) array
        Map of absolute wavenumbers.
    bsize : float, optional
        Radial bin size. Defaults to resolution in ell of input map.

    Returns
    -------
    fbin : (..., nbin) array
        Radially binned input.
    bins : (nbin) array
        Bins.
    '''

    if bsize is None:
        bsize = min(abs(lmod[0,1]), abs(lmod[1,0]))

    return enmap._bin_helper(fmap, lmod, bsize)
