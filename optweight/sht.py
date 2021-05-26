import numpy as np

from pixell import enmap, sharp
from optweight import mat_utils

def map2alm(imap, alm, minfo, ainfo, spin, adjoint=False):
    '''
    Wrapper around pixell's libsharp wrapper that does not
    require enmaps.

    Parameters
    ----------
    imap : (ntrans, npol, npix) array
        Input map.
    alm : (ntrans, npol, nelem) complex array
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
        If leading dimensions of alm and map do not match.
    '''

    if np.asarray(spin).max() > ainfo.lmax:
        raise ValueError('Spin exceeds lmax')
    if imap.shape[-1] != minfo.npix:
        raise ValueError(
            f'Wrong map size, got {imap.shape[-1]} expected {minfo.npix}')
    if alm.shape[-1] != ainfo.nelem:
        raise ValueError(
            f'Wrong alm size, got {alm.shape[-1]} expected {ainfo.nelem}')

    imap = mat_utils.atleast_nd(imap, 3)
    alm = mat_utils.atleast_nd(alm, 3)

    if alm.shape[:-1] != imap.shape[:-1]:
        raise ValueError(f'Mismatch leading dimension of map and alm :'
                         f'{imap.shape[:-1]} and {alm.shape[:-1]}')

    if adjoint:
        job_type = 2
    else:
        job_type = 0

    ntrans, npol = imap.shape[:-1]
    for tidx in range(ntrans):
        for s, i1, i2 in enmap.spin_helper(spin, npol):    
            sharp.execute(job_type, ainfo, alm[tidx,i1:i2,:],
                          minfo, imap[tidx,i1:i2,:], spin=s)

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
    
    omap = mat_utils.atleast_nd(omap, 3)
    alm = mat_utils.atleast_nd(alm, 3)

    if alm.shape[:-1] != omap.shape[:-1]:
        raise ValueError(f'Mismatch leading dimension of map and alm :'
                         f'{omap.shape[:-1]} and {alm.shape[:-1]}')

    if adjoint:
        job_type = 3
    else:
        job_type = 1

    ntrans, npol = omap.shape[:-1]
    for tidx in range(ntrans):
        for s, i1, i2 in enmap.spin_helper(spin, npol):    
            sharp.execute(job_type, ainfo, alm[tidx,i1:i2,:],
                          minfo, omap[tidx,i1:i2,:], spin=s)
        
def default_spin(shape):
    '''
    Infer spin from alm/map shape.

    Parameters
    ----------
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
