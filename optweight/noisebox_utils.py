import numpy as np
from scipy.interpolate import CubicSpline

from pixell import enmap

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
        raise ValueError('Noisebox should be 4D, not {}'.
                         format(noisebox.ndim))

    shape = noisebox.shape
    wcs = noisebox.wcs
    if bins[0] != 0:
        # Extend noisebox to elll = 0. Assuming white noise.
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
