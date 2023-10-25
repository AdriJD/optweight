import os
import multiprocessing
import numpy as np

from pixell import enmap
import ducc0

from optweight import mat_utils, map_utils, map_c_utils

def map2alm(imap, alm, minfo, ainfo, spin, adjoint=False):
    '''
    Wrapper around ducc's adjoint_synthesis. Computes YtW or Yt.

    Parameters
    ----------
    imap : (..., npol, npix) array
        Input map(s).
    alm : (..., npol, nelem) complex array
        Output alm array(s), will be overwritten.
    minfo : map_utils.MapInfo object
        Map info for input map.
    ainfo : pixell.curvedsky.alm_info object
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

    npol = imap.shape[-2]
    preshape = imap.shape[:-2]

    theta = minfo.theta.astype(np.float64, copy=False)
    nphi = minfo._nphi.astype(np.uint64, copy=False)
    phi0 = minfo.phi0.astype(np.float64, copy=False)
    mstart = ainfo.mstart.astype(np.uint64, copy=False)
    ringstart = minfo._offsets.astype(np.uint64, copy=False)
    lstride = int(ainfo.stride)
    pixstride = int(minfo.stride[0])
    lmax = int(ainfo.lmax)
    mmax = int(ainfo.mmax)

    # Default to OMP_NUM_THREADS. If not availble, use cpu_count.
    try:
        nthreads = int(os.environ["OMP_NUM_THREADS"])
    except KeyError:
        nthreads = multiprocessing.cpu_count()

    for idxs in np.ndindex(preshape):
        for s, i1, i2 in enmap.spin_helper(spin, npol):
            if adjoint:
                map_tmp = np.asarray(imap[idxs][i1:i2,:])
            else:
                map_tmp = np.asarray(imap[idxs][i1:i2,:]).copy()
                # Inplace multiplication with weight.
                map_c_utils.apply_ringweight(map_tmp, minfo)

            ducc0.sht.experimental.adjoint_synthesis(
                map=map_tmp,
                alm=alm[idxs][i1:i2,:],
                theta=theta,
                lmax=lmax,
                nphi=nphi,
                phi0=phi0,
                mstart=mstart,
                ringstart=ringstart,
                lstride=lstride,
                pixstride=pixstride,
                spin=s,
                mmax=mmax,
                mode='STANDARD',
                theta_interpol=False,
                nthreads=nthreads)


def alm2map(alm, omap, ainfo, minfo, spin, adjoint=False):
    '''
    Wrapper around ducc's synthesis. Computes Y or WY.

    Parameters
    ----------
    alm : (ntrans, npol, nelem) complex array
        Input alm array.
    omap : (ntrans, npol, npix) array
        Output map, will be overwritten.
    ainfo : pixell.curvedsky.alm_info object
        alm info for inpu alm.
    minfo : map_utils.MapInfo object
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
    npol = omap.shape[-2]
    preshape = omap.shape[:-2]

    theta = minfo.theta.astype(np.float64, copy=False)
    nphi = minfo._nphi.astype(np.uint64, copy=False)
    phi0 = minfo.phi0.astype(np.float64, copy=False)
    mstart = ainfo.mstart.astype(np.uint64, copy=False)
    ringstart = minfo._offsets.astype(np.uint64, copy=False)
    lstride = int(ainfo.stride)
    pixstride = int(minfo.stride[0])
    lmax = int(ainfo.lmax)
    mmax = int(ainfo.mmax)

    # Default to OMP_NUM_THREADS. If not availble, use cpu_count.
    try:
        nthreads = int(os.environ["OMP_NUM_THREADS"])
    except KeyError:
        nthreads = multiprocessing.cpu_count()

    for idxs in np.ndindex(preshape):
        for s, i1, i2 in enmap.spin_helper(spin, npol):
            map_slice = np.asarray(omap[idxs][i1:i2,:])

            ducc0.sht.experimental.synthesis(
                alm=alm[idxs][i1:i2,:],
                map=map_slice,
                theta=theta,
                lmax=lmax,
                nphi=nphi,
                phi0=phi0,
                mstart=mstart,
                ringstart=ringstart,
                lstride=lstride,
                pixstride=pixstride,
                spin=s,
                mmax=mmax,
                mode='STANDARD',
                theta_interpol=False,
                nthreads=nthreads)

            if adjoint:
                # Inplace multiplication with weight.
                map_c_utils.apply_ringweight(map_slice, minfo)
                
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
                         f'one leading dimension. Got shape : {shape}')
    if len(shape) == 1 or shape[0] == 1:
        return 0
    elif shape[0] == 2:
        return 2
    elif shape[0] == 3:
        return [0, 2]
    else:
        raise ValueError(f'Cannot determine default spin value for shape : {shape}')
