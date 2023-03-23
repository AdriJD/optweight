'''
Routines for real-to-complex and complex-to-real FFTs. Adapted from pixell and mnms.
But differ at some critical point, mainly in the defintion of the flat sky lx.
'''
import numpy as np
from scipy.interpolate import interp1d

from pixell import fft, enmap, wcsutils

from optweight import type_utils, mat_utils

def rfft(imap, fmap, normalize=True, adjoint=False):
    '''
    Real-to-complex FFT.

    Parameters
    ----------
    imap : (..., ny, nx) ndmap
        Map to transform.
    fmap : (..., ny, nx//2+1) complex ndmap
        Output buffer.
    normalize : bool, optional
        The FFT normalization, by default True. If True, normalize
        using pixel number. If in ['phy', 'phys', 'physical'],
        normalize by sky area.
    adjoint : bool, optional
        Compute adjoint of the complex-to-real FFT.

    Raises
    ------
    ValueError
        If input and output map have inconsistent shapes.
    '''

    ny, nx = imap.shape[-2:]
    if fmap.shape != imap.shape[:-2] + (ny, nx // 2 + 1):
        raise ValueError(
            f'Inconsistent shapes: imap : {imap.shape}, fmap : {fmap.shape}')

    fmap = fft.rfft(imap, fmap, axes=[-2, -1])
    norm = 1

    if normalize:
        norm /= np.prod(imap.shape[-2:]) ** 0.5
    if normalize in ["phy","phys","physical"]:
        if adjoint:
            norm /= imap.pixsize() ** 0.5
        else:
            norm *= imap.pixsize() ** 0.5
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

    Raises
    ------
    ValueError
        If input and output map have inconsistent shapes.
    '''

    ny, nx = omap.shape[-2:]
    if fmap.shape != omap.shape[:-2] + (ny, nx // 2 + 1):
        raise ValueError(
            f'Inconsistent shapes: fmap : {fmap.shape}, omap : {omap.shape}')

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

def allocate_fmap(shape, dtype, fill_value=0):
    '''
    Allocate an array suitable for the output of rfft.

    Parameters
    ----------
    shape : tuple
        Input shape of enmap.
    dtype : type, optional
        Type of input map.
    fill_value : scalar, optional
        Value to fill new array with.

    Returns
    -------
    omap : (..., ny, nx//2+1) complex array
        New array for 2D Fourier coefficients.
    '''
    
    preshape = shape[:-2]
    ny, nx = shape[-2:]

    return np.full(preshape + (ny, nx // 2 + 1), fill_value, 
                   dtype=type_utils.to_complex(dtype))

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

    Notes
    -----
    This definition differs from the one used in pixell and mnms. We 
    attempt no correction for sky curvature. Our `ly` only matches up 
    with the harmonic `m` on the equator of a cylindrical pixelization.
    '''

    step = np.radians(wcs.wcs.cdelt[::-1])
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
    modlmap : (nly, nlx) array
       Map of absolute wavenumbers.
    '''

    slmap = lmap_real(shape, wcs, dtype=dtype)
    modlmap = np.sum(slmap ** 2, axis=0) ** 0.5

    return modlmap

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

def lbin(fmap, modlmap, bsize=None):
    '''
    Bin Fourier-space map into radial bins.
    
    Arguments
    ---------
    fmap : (..., nly, nlx) array.
        Input 2D Fourier map.
    modlmap : (nly, nlx) array
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
        bsize = min(abs(modlmap[0,1]), abs(modlmap[1,0]))

    return enmap._bin_helper(fmap, modlmap, bsize)

def fmul(fmap, fmat2d=None, fmat1d=None, ells=None, modlmap=None,
         out=None):
    '''
    Compute f'[...,i,ly,lx] = m[i,j,ly,lx] f[...,j,ly,lx] matrix
    multiplication.
    
    Parameters
    ----------
    fmap : (..., nly, nlx) complex array.
        Input 2D Fourier map.
    fmat2d : (npol, npol, nly, nlx) or (npol, nly, nlx) array, optional
        Matrix, if diagonal only the diagal suffices.
    fmat1d : (npol, npol, nell) or (npol, nly, nell) array, optional
        Matrix, if diagonal only the diagal suffices.
    ells : (nell) array, optional
        Array with multipoles, can be non-integer, needed for `fmat1d`.
    modlmap : (nly, nlx) array
        Map of absolute wavenumbers, needed for `fmat1d`.
    out : (..., nly, nlx) array, optional
        Output array.

    Returns
    -------
    out : (..., nly, nlx) complex array
        Result from matrix multiplication.
    '''
    
    if fmat2d is not None and fmat1d is not None:
        raise ValueError('Cannot have both fmat2d and fmat1d')

    if fmat1d is not None:
        fmat2d = cl2flat(fmat1d, ells, modlmap)
        
    return fmul_2d(fmap, fmat2d, out=out)

def fmul_2d(fmap, fmat2d, out=None):
    '''
    Compute f'[...,i,ly,lx] = m[i,j,ly,lx] f[...,j,ly,lx] matrix
    multiplication.
    
    Parameters
    ----------
    fmap : (npol, nly, nlx) complex array
        Input array.
    fmat2d : (npol, npol, nly, nlx) or (npol, nly, nlx) array
        Matrix, if diagonal only the diagal suffices.
    out : (npol, nly, nlx) complex array, optional
        Output array. Will be overwritten!

    Returns
    -------
    out : (npol, nly, nlx) complex array
        Output array.
    '''
    
    fmap = mat_utils.atleast_nd(fmap, 3)
    npol = fmap.shape[0]
    nly, nlx = fmap.shape[-2:]

    if out is None:
        out = fmap.copy()
    else:
        out[:] = fmap 

    if fmat2d.ndim in (2, 3):
        out *= fmat2d
    else:
        out = np.einsum('ablk, blk -> alk', fmat2d, out,
                        out=out, optimize=True)
    
    return out

def cl2flat(c_ell, ells, modlmap):
    '''
    Interpolate a 1d function of multipole to a 2D map of Fourier
    coefficients.

    Parameters
    ----------    
    c_ell : (..., nell) array
        Input power spectrum.
    ells : (nell) array
        Multipoles, can be non-integer.
    modlmap : (nly, nlx) array
        Map of absolute wavenumbers.

    Returns
    -------
    out : (npol, nly, nlx) complex array
        2D output array.    
    
    Raises
    ------
    ValueError
        If input has to be extrapolated more than 5 ell bins.
    '''

    # Extrapolate the input to the output using nearest neighbor.
    # This makes sure we don't change signs etc and turn a PSD
    # matrix into a non-PSD matrix. A bit ugly, but better safe
    # than sorry.

    lmin_out = modlmap.min()
    lmax_out = modlmap.max()

    ell_start, ell_end = [], []
    if ells[0] > lmin_out:
        ell_start = np.asarray([lmin_out])
        c_ell_start = c_ell[...,0]
        c_ell = np.concatenate((c_ell_start[...,np.newaxis], c_ell), axis=-1)
    if ells[-1] < lmax_out:
        ell_end = np.asarray([lmax_out])
        c_ell_end = c_ell[...,-1]
        c_ell = np.concatenate((c_ell, c_ell_end[...,np.newaxis]), axis=-1)
    ells = np.concatenate((ell_start, ells, ell_end))        

    out = np.zeros(c_ell.shape[:-1] + modlmap.shape,
                   dtype=type_utils.to_complex(c_ell.dtype))
    cs = interp1d(ells, c_ell, kind='linear', assume_sorted=True,
                  bounds_error=True)

    return cs(modlmap)

def calc_ps1d(fmap, wcs, modlmap, fmap2=None, bsize=None):
    '''
    Calculate 1D power spectrum from set of Fourier
    coefficients.
    
    Parameters
    ----------
    fmap : (..., nly, nlx) complex array
        Input 2D Fourier map.
    wcs : astropy.wcs.WCS object
        WCS object describing geometry of original map.
    modlmap : (nly, nlx) array
        Map of absolute wavenumbers.
    fmap2 : (..., nly, nlx) complex array, optional
        Second 2D Fourier map for cross-correlation.
    bsize : float, optional
        Radial bin size. Defaults to resolution in ell of input map.
    
    Returns
    -------
    ps1d : (..., nbin) array
        Radially binned 1D power spectrum.
    bins : (nbin) array
        Bins.
    '''

    fmap = enmap.enmap(fmap, wcs=wcs, copy=False)
    if fmap2:
        fmap2 = enmap.enmap(fmap2, wcs=wcs, copy=False)

    ps2d = enmap.calc_ps2d(fmap, harm2=fmap2)
    return lbin(ps2d, modlmap, bsize=bsize)
        
