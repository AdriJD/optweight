import numpy as np

from pixell import enmap, sharp

#@profile
def map2alm(imap, alm, minfo, ainfo, spin, adjoint=False):
    '''
    Wrapper around pixell's libsharp wrapper that does not
    require enmaps.

    Parameters
    ----------
    imap : (npol, npix) array
        Input map.
    alm : (npol, nelem) complex array
        Output alm array, will be overwritten.
    minfo : sharp.map_info object
        Map info for input map.
    ainfo : sharp.alm_info object
        alm info for output alm.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    adjoint : bool, optional
        If set, compute adjoint synthesis: Yt, so map2alm without 
        theta integration weights.    

    Raises
    ------
    ValueError
        If spin value is larger than lmax.
    '''

    if np.asarray(spin).max() > ainfo.lmax:
        raise ValueError('Spin exceeds lmax')

    if imap.ndim == 1:
        imap = imap[np.newaxis,:]
    if alm.ndim == 1:
        alm = alm[np.newaxis,:]

    npol = imap.shape[0]

    if adjoint:
        job_type = 2
    else:
        job_type = 0

    for s, i1, i2 in enmap.spin_helper(spin, npol):    
        sharp.execute(job_type, ainfo, alm[i1:i2,:], minfo, imap[i1:i2,:], spin=s)

#@profile
def alm2map(alm, omap, ainfo, minfo, spin, adjoint=False):
    '''
    Wrapper around pixell's libsharp wrapper that does not
    require enmaps.

    Parameters
    ----------
    alm : (npol, nelem) complex array
        Input alm array.
    omap : (npol, npix) array
        Output map, will be overwritten.
    ainfo : sharp.alm_info object
        alm info for inpu alm.
    minfo : sharp.map_info object
        Map info for output map.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    adjoint : bool
        If set, compute adjoint analysis: WY, so alm2map with integration weights.

    Raises
    ------
    ValueError
        If spin value is larger than lmax.
    '''

    if np.asarray(spin).max() > ainfo.lmax:
        raise ValueError('Spin exceeds lmax')

    if omap.ndim == 1:
        omap = omap[np.newaxis,:]
    if alm.ndim == 1:
        alm = alm[np.newaxis,:]

    npol = omap.shape[0]

    if adjoint:
        job_type = 3
    else:
        job_type = 1

    for s, i1, i2 in enmap.spin_helper(spin, npol):    
        sharp.execute(job_type, ainfo, alm[i1:i2,:], minfo, omap[i1:i2,:], spin=s)
        
