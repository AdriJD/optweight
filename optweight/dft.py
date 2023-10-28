'''
Routines for real-to-complex and complex-to-real FFTs. Adapted from pixell and mnms.
But differ at some critical point, mainly in the defintion of the flat sky lx.
'''
import numpy as np
from scipy.interpolate import interp1d

import ducc0
from pixell import enmap, wcsutils

from optweight import type_utils, mat_utils

def rfft(imap, fmap):
    '''
    Real-to-complex FFT.

    Parameters
    ----------
    imap : (..., ny, nx) ndmap
        Map to transform.
    fmap : (..., ny, nx//2+1) complex ndmap
        Output buffer.

    Raises
    ------
    ValueError
        If input and output map have inconsistent shapes.
    '''

    ny, nx = imap.shape[-2:]
    if fmap.shape != imap.shape[:-2] + (ny, nx // 2 + 1):
        raise ValueError(
            f'Inconsistent shapes: imap : {imap.shape}, fmap : {fmap.shape}')
    if fmap.dtype != type_utils.to_complex(imap.dtype):
        raise TypeError(
            f'imap.dtype : {imap.dtype} and fmap.dtype : {fmap.dtype} do not match')
    
    ducc0.fft.r2c(np.asarray(imap), axes=[-2, -1], inorm=1, out=np.asarray(fmap),
                  nthreads=0)

def irfft(fmap, omap):
    '''
    Complex-to-real FFT.

    Parameters
    ----------
    fmap : (..., ny, nx//2+1) complex ndmap
        Input Fourier coefficients.
    omap : (..., ny, nx) ndmap
        Output buffer.

    Raises
    ------
    ValueError
        If input and output map have inconsistent shapes.
    '''

    ny, nx = omap.shape[-2:]
    if fmap.shape != omap.shape[:-2] + (ny, nx // 2 + 1):
        raise ValueError(
            f'Inconsistent shapes: fmap : {fmap.shape}, omap : {omap.shape}')
    if fmap.dtype != type_utils.to_complex(omap.dtype):
        raise TypeError(
            f'omap.dtype : {omap.dtype} and fmap.dtype : {fmap.dtype} do not match')

    ducc0.fft.c2r(np.asarray(fmap), axes=[-2, -1], forward=False, inorm=1,
                  lastsize=omap.shape[-1], out=np.asarray(omap), nthreads=0)
    
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

def allocate_map(fshape, dtype, fill_value=0):
    '''
    Allocate an array suitable for the output of irfft.

    Parameters
    ----------
    shape : tuple
        Input shape of fmap: (..., ly, lx).
    dtype : type, optional
        Type of input fmap.
    fill_value : scalar, optional
        Value to fill new array with.

    Returns
    -------
    omap : (..., ly, 2 * (lx -1) + 1) complex array
        New array.

    Notes
    -----
    This always results in an array with an odd-length. From the
    shape of the fmap it is unknown if the input map was even or odd
    in the x direction.
    '''
    
    preshape = fshape[:-2]
    ly, lx = fshape[-2:]

    return np.full(preshape + (ly, 2 * (lx - 1) + 1), fill_value, 
                   dtype=type_utils.to_real(dtype))

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

def laxes2lmap(ly, lx, dtype=np.float64):
    '''
    Return maps of all the wavenumbers corresponding to a given enmap geometry.

    Arguments
    ---------
    ly : (nly) array
        Wavenumbers in y direction.
    lx : (nlx) array
        Wavenumbers in x direction.

    Returns
    -------
    lmap : (2, nly, nlx) array
       Maps of ell_y and ell_x wavenumbers.
    '''

    lmap = np.empty((2, ly.size, lx.size), dtype=dtype)
    lmap[0] = ly[:,np.newaxis]
    lmap[1] = lx[np.newaxis,:]

    return lmap

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
    return laxes2lmap(ly, lx, dtype=dtype)

def laxes2modlmap(ly, lx, dtype=np.float64):
    '''
    Return a map of all the abs wavenumbers in the fourier transform
    of a map with the given shape and wcs.

    Arguments
    ---------
    ly : (nly) array
        Wavenumbers in y direction.
    lx : (nlx) array
        Wavenumbers in x direction.
    dtype : type, optional
        Type of ouput array.

    Returns
    -------
    modlmap : (nly, nlx) array
       Map of absolute wavenumbers.
    '''
    
    lmap = laxes2lmap(ly, lx, dtype=dtype)
    return np.sum(lmap ** 2, axis=0) ** 0.5
    
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

    ly, lx = laxes_real(shape, wcs)
    return laxes2modlmap(ly, lx, dtype=dtype)

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

    lres = 2 * np.pi / (np.radians(wcs.wcs.cdelt[::-1]) * shape[-2:])
    lres[-1] = abs(lres[-1])
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

    cs = interp1d(ells, c_ell, kind='linear', assume_sorted=True,
                  bounds_error=True)

    return cs(modlmap).astype(c_ell.dtype, copy=False)

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
        
def contract_fxg(fmap, gmap):
    '''
    Return sum_{ly lx} f_{ly lx} x conj(f_{ly lx}), i.e. the sum of the Hadamard
    product of two sets of 2D Fourier coefficients corresponding to real fields.

    Parameters
    ----------
    fmap : (..., nly, nlx) complex array
        Input 2D Fourier map.
    gmap : (..., nly, nlx) complex array
        Input 2D Fourier map.

    Returns
    -------
    had_sum : float
        Sum of Hadamard product (real valued).

    Raises
    ------
    ValueError
        If input arrays have different shapes.    
    '''
    
    if fmap.shape != gmap.shape:
        raise ValueError(
            f'Shape fmap ({fmap.shape}) != shape gmap ({gmap.shape})')

    gmap = np.conj(gmap)
    csum = complex(np.tensordot(fmap, gmap, axes=fmap.ndim))
    had_sum = 2 * csum
    
    had_sum -= np.sum(fmap[...,:,0] * gmap[...,:,0])

    # If nx is even we also have to subtract the last column.
    # How to determine if input nx was odd or even? We need to 
    # check if the last element of the first row is real.
    # If ny is also even, we can additionally check if the 
    # last element of the middle row is also even.

    nx_even = np.all(np.isreal(fmap[...,0,-1]))
    ny = fmap.shape[-2]
    if ny % 2 == 0:        
        nx_even &= np.all(np.isreal(fmap[...,ny//2,-1]))

    if nx_even:        
        had_sum -= np.sum(fmap[...,:,-1] * gmap[...,:,-1])

    return np.real(had_sum)

def slice_fmap(fmap, slices_y, slice_x, laxes=None):
    '''
    Return map of 2D fourier coefficients that is cut out from input map.

    Parameters
    ----------
    fmap : (..., ny, nx) array
        Input 2D Fourier coefficients.
    slices_y : tuple of slices
        Two slices that slice select nonzero positive frequences and nonzero
        negaive frequencies.
    slice_x : slice
        Slice with nonzero elements in the x direction
    laxes : tuple of (ny) array and (nx) array, optional
        ly and lx coordinates of input coefficients.

    Returns
    -------
    fmap_out : (ny', nx') array
        Output coefficients. New C-contigous array (i.e. not a view of input).
    laxes_out : tuple of (ny') array and (nx') array
        ly and lx coordinates of output. Only if `laxes` was provided.
    '''

    pos_part = fmap[...,slices_y[0],slice_x]
    neg_part = fmap[...,slices_y[1],slice_x]

    out = np.concatenate((pos_part, neg_part), axis=-2)
    
    if laxes is not None:
        ly, lx = laxes
        lx_new = lx[slice_x]
        ly_new = np.concatenate((ly[slices_y[0]], ly[slices_y[1]]))
        out = out, (ly_new, lx_new)

    return out

def add_to_fmap(fmap_large, fmap_small, slices_y, slice_x):
    '''
    In-place addition of `fmap_small` to `fmap_large`.

    Parameters
    ----------
    fmap_large : (..., ny, nx) array
        Base 2D Fourier coefficients.
    fmap_small : (..., ny', nx') array
        2D Fourier coefficients to be added.
    slices_y : tuple of slices
        Two slices that select nonzero positive frequences and nonzero
        negaive frequencies. See fkernel.find_kernel_slice.
    slice_x : slice
        Slice with nonzero elements in the x direction.
    
    Returns
    -------
    fmap_large : (..., ny, nx) array
        Input array with addition.    
    '''

    fmap_large[...,slices_y[0],slice_x] += fmap_small[...,slices_y[0],:]
    fmap_large[...,slices_y[1],slice_x] += fmap_small[...,slices_y[1],:]

    return fmap_large

def get_optimal_fftlen(len_min, even=True):
    '''
    Compute optimal array length for FFT given a minumum length.

    Paramters
    ---------
    len_min : int
        Minumum length.
    even : bool, optional
        Demand even optimal lengths (for real FFTs).

    Returns
    -------
    len_opt : int
        Optimal length

    Notes
    -----
    This assumes we want input sizes that can be factored as 2^a 3^b 5^c 7^d.
    Adapted from ksw.
    '''
    
    if len_min == 0:
        return len_min
    if len_min == 1 and even:
        return 2

    max_a = int(np.ceil(np.log(len_min) / np.log(2)))
    max_b = int(np.ceil(np.log(len_min) / np.log(3)))
    max_c = int(np.ceil(np.log(len_min) / np.log(5)))
    max_d = int(np.ceil(np.log(len_min) / np.log(7)))       

    len_opt = 2 ** max_a # Reasonable starting point.
    for a in range(max_a):
        for b in range(max_b):
            for c in range(max_c):
                for d in range(max_d):
                    fftlen = 2 ** a * 3 ** b * 5 ** c * 7 ** d
                    if even and fftlen % 2:
                        continue
                    if fftlen < len_min:
                        continue
                    if fftlen == len_min:
                        len_opt = fftlen
                        break
                    if fftlen < len_opt:
                        len_opt = fftlen

    return len_opt
