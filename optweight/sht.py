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
        If npix does not match size map.
        If nelem does not match size alm.
    '''

    if np.asarray(spin).max() > ainfo.lmax:
        raise ValueError('Spin exceeds lmax')
    if imap.shape[-1] != minfo.npix:
        raise ValueError(
            f'Wrong map size, got {imap.shape[-1]} expected {minfo.npix}')
    if alm.shape[-1] != ainfo.nelem:
        raise ValueError(
            f'Wrong alm size, got {alm.shape[-1]} expected {ainfo.nelem}')
    
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
        If npix does not match size map.
        If nelem does not match size alm.
    '''

    if np.asarray(spin).max() > ainfo.lmax:
        raise ValueError('Spin exceeds lmax')

    if np.asarray(spin).max() > ainfo.lmax:
        raise ValueError('Spin exceeds lmax')
    if omap.shape[-1] != minfo.npix:
        raise ValueError(
            f'Wrong map size, got {omap.shape[-1]} expected {minfo.npix}')
    if alm.shape[-1] != ainfo.nelem:
        raise ValueError(
            f'Wrong alm size, got {alm.shape[-1]} expected {ainfo.nelem}')
    
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
        
def default_spin(shape):
    '''
    Infer spin from alm/map shape.

    shape : tuple
        Shape of map or alm array.

    Returns
    -------
    spin : int, list
        Spin argument for SH transforms. 

    Raises
    ------
    ValueError
        If spin cannot be determined (npol > 3 or len(shape) > 2).

    Notes
    -----
    Shape is assumed to be (npol, nelem) or (npol, npix). For npol=0, 2 and 3
    we default to spin=0, 2, [0, 2], respectively.
    '''

    if len(shape) > 2:
        raise ValueError(f'Cannot determine default spin for alm/map with more than '
                         'one leading dimension. Got shape : {shape}')
    if len(shape) == 1 or shape[0] == 1:
        return 0
    elif shape[0] == 2:
        return 2
    elif shape[0] == 3:
        return [0, 2]
    else:
        raise ValueError(f'Cannot determine default spin value for shape : {shape}')
