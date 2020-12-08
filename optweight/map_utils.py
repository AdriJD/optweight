import numpy as np

from pixell import enmap, sharp, utils, wcsutils
import healpy as hp

def enmap2gauss(imap, lmax, order=3, area_pow=0, destroy_input=False,
                mode='constant'):
    '''
    Interpolate input enmap to Gauss-Legendre grid.

    Parameters
    ----------
    imap : (..., ny, nx) enmap
        Input map(s)
    lmax : int
        Band limit supported by Gauss-Legendre grid.
    order : int, optional
        Order of spline interpolation
    area_pow : float, optional
        How the quantity in the map scales with area, i.e.
        (area_pix_out / area_pix_in)^area_power. Useful for e.g. variance maps
        that need a correction for pixel area difference (area_pow = -1).
    destroy_input : bool, optional
        If set, input is filtered inplace.
    mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        How the input array is extended beyond its boundaries, see 
        scipy.ndimage.map_coordinates.

    Returns
    -------
    omap : (..., npix) array
        Output map(s).
    map_info : sharp.map_info object
        Metadata of output map.

    Raises
    ------
    NotImplementedError
        If enmap is not cilindrical.
    '''
    
    if not wcsutils.is_cyl(imap.wcs):
        raise NotImplementedError('Non-cilindrical enmaps not supported')

    ny, nx = imap.shape[-2:]
    dec_range = enmap.pix2sky(
        imap.shape, imap.wcs, [[0, ny-1], [0, 0]], safe=False)[0]
    ra_range = enmap.pix2sky(
        imap.shape, imap.wcs, [[0, 0], [0, nx-1]], safe=False)[1]

    theta_range = np.pi / 2 - dec_range
    
    # I want to use modulo pi except that I want to keep pi as pi.
    if not 0 <= theta_range[0] <= np.pi:
        theta_range[0] = theta_range[0] % np.pi
    if not 0 <= theta_range[1] <= np.pi:
        theta_range[1] = theta_range[1] % np.pi

    theta_min = min(theta_range)
    theta_max = max(theta_range)
    minfo = get_gauss_minfo(
        lmax, theta_min=theta_min, theta_max=theta_max)

    if order > 1:
        imap = utils.interpol_prefilter(
            imap, order=order, inplace=destroy_input)

    omap = np.zeros(imap.shape[:-2] + (minfo.npix,), dtype=imap.dtype)

    thetas, phis = _get_gauss_coords(minfo)
    nphi = phis.size
    pos = np.zeros((2, nphi))
    pos[1,:] = phis

    if area_pow != 0:
        area_in = enmap.pixsizemap(
            imap.shape, imap.wcs, separable="auto", broadcastable=False) 

    for tidx, theta in enumerate(thetas):
        pos[0,:] = np.pi / 2 - theta
        start = tidx * nphi
        end = start + nphi

        omap[...,start:end] = imap.at(pos, order=order, mask_nan=False,
                                      prefilter=False, mode=mode)

        if area_pow != 0:
            area_gauss = minfo.weight[tidx]
            omap[...,start:end] *= area_gauss ** area_pow
            omap[...,start:end] *= area_in.at(
                pos, order=order, mask_nan=False, prefilter=False, 
                mode='nearest') ** -area_pow

    return omap, minfo

def healpix2gauss(imap, lmax, nest=False, area_pow=0):
    '''
    Interpolate input HEALPix map to Gauss-Legendre grid.

    Parameters
    ----------
    imap : (..., npix) array
        Input HEALPix maps.
    lmax : int
        Band limit supported by Gauss-Legendre grid.
    nest : bool, optional
        If set, input is assumed to be in NESTED ordering instead of RING.
    area_pow : float, optional
        How the quantity in the map scales with area, i.e.
        (area_pix_out / area_pix_in)^area_power. Useful for e.g. variance maps
        that need a correction for pixel area difference (area_pow = -1).

    Returns
    -------
    omap : (..., npix) array
        Output map(s).
    map_info : sharp.map_info object
        Metadata of output map.
    '''

    # We assume that HEALPix maps are full-sky.
    minfo, theta_arc_len = get_gauss_minfo(
        lmax, theta_min=None, theta_max=None, return_arc_len=True)
    omap = np.zeros(imap.shape[:-1] + (minfo.npix,), dtype=imap.dtype)

    thetas, phis = _get_gauss_coords(minfo)
    nphi = phis.size
    theta_ring = np.zeros(nphi)

    if area_pow != 0:
        area_hp = hp.nside2pixarea(hp.npix2nside(imap.shape[-1]))
        dphi = 2 * np.pi / nphi

    for tidx, theta in enumerate(thetas):
        theta_ring[:] = theta
        start = tidx * nphi
        end = start + nphi

        for index in np.ndindex(imap.shape[:-1]):
            omap[index,start:end] = hp.get_interp_val(
                imap[index], theta_ring, phis, nest=nest, lonlat=False)

        if area_pow != 0:
            area_gauss = np.sin(theta) * dphi * theta_arc_len[tidx]
            scal_fact = (area_gauss / area_hp) ** area_pow
            omap[...,start:end] *= scal_fact

    return omap, minfo

def _get_gauss_coords(map_info):
    '''
    Return coordinate arrays for map_info.

    Parameters
    ----------
    map_info : sharp.map_info object
        Metadata of Gauss-Legendre map.

    Returns
    -------
    thetas : (ntheta) array
        Co-latitude coordinates in radians (0 at north pole, pi at south pole).
    phi : (nphi) array
        Longitude coordinates for each ring.
    '''

    # I know that each ring has nphi phis and also that the phi coordinates
    # increase from phi0 to 2pi + phi0 in each ring.
    thetas = map_info.theta
    nphi = map_info.nphi[0]
    phi0 = map_info.phi0[0]
    phis = np.linspace(phi0, 2 * np.pi + phi0 , nphi, endpoint=False)

    return thetas, phis

def get_gauss_minfo(lmax, theta_min=None, theta_max=None, return_arc_len=False):
    '''
    Compute map_info metadata for a Gauss-Legendre grid.

    Parameters
    ----------
    lmax : int
        Band limit supported by Gauss-Legendre grid.
    theta_min : float, optional
        Minimum colattitute of grid in radians.
    theta_max : float, optional
        Maximum colattitute of grid in radians.
    return_arc_len : bool, optional
        If set, return arc length of each theta ring.

    Returns
    -------
    map_info : sharp.map_info object
        metadata of Gausss-Legendre grid.
    arc_lengths : (ntheta) array
        If return_arc_len is set: arc length of rings along the theta direction.
    '''

    nrings = int(np.floor(lmax / 2)) + 1
    nphi = lmax + 1
    map_info = sharp.map_info_gauss_legendre(nrings, nphi)

    if return_arc_len:
        arc_len = get_arc_len(map_info.theta, 0, np.pi)

    if theta_min or theta_max:
        theta_min = theta_min if theta_min else 0
        theta_max = theta_max if theta_max else np.pi
        thetas = map_info.theta
        theta_mask = np.logical_and(thetas >= theta_min, thetas <= theta_max)

        theta = map_info.theta[theta_mask]
        weight = map_info.weight[theta_mask]
        stride = map_info.stride[theta_mask]
        offsets = map_info.offsets[theta_mask]
        offsets -= offsets[0]

        # Create new map_info corresponding to truncated theta range.
        map_info = sharp.map_info(theta, nphi=nphi, phi0=0, offsets=offsets,
                                  stride=stride, weight=weight)
        if return_arc_len:
            arc_len = arc_len[theta_mask]

    ret = map_info
    if return_arc_len:
        return map_info, + arc_len,

    return ret

def get_arc_len(theta, theta_min, theta_max):
    '''
    Get arc length corresponding to each theta coordinate.

    Parameters
    ----------
    theta : (ntheta) array
        Angles in radians.
    theta_min : float
        Minimum value of theta domain.
    theta_max : float
        Maximum value of theta domain.

    Returns
    -------
    arc_lengths : (ntheta) array
        Arc length of each pixel.

    Raises
    ------
    ValueError
        If theta values are not unique.
        If theta values exceed theta domain.
    '''

    idx_sort = np.argsort(theta)
    idx_isort = np.argsort(idx_sort)

    theta = theta[idx_sort]

    if not np.array_equal(np.unique(theta), theta):
        raise ValueError('Theta values have to be unique.')

    if theta.min() <= theta_min:
        raise ValueError('theta_min >= theta.min(): {} >= {}'
                         .format(theta_min, theta.min()))
    if theta.max() >= theta_max:
        raise ValueError('theta_max <= theta.max(): {} <= {}'
                         .format(theta_max, theta.max()))

    dtheta = np.diff(theta)

    theta_edge = np.zeros(theta.size + 1)
    theta_edge[0] = theta_min
    theta_edge[-1] = theta_max
    theta_edge[1:-1] = theta[:-1] + dtheta / 2

    arc_lengths = np.diff(theta_edge)

    return arc_lengths[idx_isort]

def rand_map_pix(cov_pix):
    '''
    Draw random Gaussian realisation from covariance.

    Parameters
    ----------
    cov_pix : (npol, npol, npix) or (npol, npix) array
        Per-pixel covariance matrix.

    returns
    -------
    rand_map : (npol, npix) array
        Random draw.
    '''
    
    npol, npix = cov_pix.shape[-2:]

    # Draw (npol, npix) unit variates for map.
    uv = np.random.randn(npol * npix).reshape(npol, npix)

    if cov_pix.ndim == 2:
        cov_pix_sqrt = np.sqrt(cov_pix)
        uv *= cov_pix_sqrt

    elif cov_pix.ndim == 3:
        # We use SVD (=eigenvalue decomposition for positive-semi-definite matrices)
        # because some pixels are zero and (i)cov is positive *semi*-definite
        # instead of *definite*, so Cholesky won't work.    
        cov_pix_t = np.transpose(cov_pix, (2, 0, 1))
        cov_pix_sqrt_t = utils.eigpow(cov_pix_t, 0.5)
        cov_pix_sqrt = np.ascontiguousarray(np.transpose(cov_pix_sqrt_t, (1, 2, 0)))    
        uv = np.einsum('ijk, jk -> ik', cov_pix_sqrt, uv, optimize=True)

    return uv

def inv_qweight_map(imap, minfo, inplace=False):
    '''
    Calculate W^-1 m where W is a diagonal matrix with quadrature weights
    in the pixel domain and m is a set of input maps.

    Parameters
    ----------
    imap : (..., npix) array
        Input maps.
    minfo : sharp.map_info object
        Metainfo for pixelization of input maps.    
    ainfo : sharp.alm_info object
        Metainfo for internally used alms.
    inplace : bool, optional
        Perform operation in place.

    Returns
    -------
    out : (..., npix) array
        Output from matrix-vector operation.
    '''

    dim_in = imap.ndim

    if inplace:
        out = imap
    else:
        out = imap.copy()

    if dim_in == 1:
        out = out[np.newaxis,:]

    for ridx in range(minfo.nrow):
        iweight = 1 / minfo.weight[ridx]
        stride = minfo.stride[ridx]
        start = minfo.offsets[ridx]
        end = start + (minfo.nphi[ridx]) * stride
        out[...,start:end:stride] *= iweight

    if dim_in == 1:
        imap = out[0]

    return out

def get_isotropic_ivar(icov_pix, minfo):
    '''
    Compute the isotropic inverse variance for an inverse 
    covariance matrix diagonal in pixel space.

    Parameters
    ----------
    icov_pix : (npol, npol, npix) or (npol, npix) array
        Inverse covariance matrix.
    minfo : sharp.map_info object
        Metainfo spefying pixelization of covariance matrix.
    
    Returns
    -------
    var_iso : (npol, npol) array
        Isotropic variance. Note that only the diagonal is calculated.

    Notes
    -----
    See eq. A1 in Seljebotn et al. (1710.00621).
    '''

    wcov = inv_qweight_map(icov_pix, minfo, inplace=False)
    
    # Set off-diagonal elements to zero, I only understand diagonal for now.
    numerator = np.sum(wcov ** 2, axis=-1)
    denominator = np.sum(wcov, axis=-1)

    if icov_pix.ndim == 2:
        return np.diag(numerator / denominator)

    elif icov_pix.ndim == 3:
        return np.diag(np.diag(numerator) / np.diag(denominator))

def get_ivar_ell(icov_wav, w_ell):
    '''
    Compute the inverse variance spectrum for an inverse 
    covariance matrix diagonal in wavelet space.

    Parameters
    ----------
    icov_wav : wavetrans.Wav object
        Inverse covariance matrix.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    
    Returns
    -------
    ivar_ell : (npol, npol, nell) array
        Inverse variance spectrum. Note that only the diagonal 
        is calculated.

    Raises
    ------
    ValueError
        If npol cannot be determined from preshape of wavelet object.
    '''

    preshape = icov_wav.preshape
    if len(preshape) > 2 or len(set(preshape)) > 1:
        raise ValueError('Could not determine npol from preshape : {}'
                         .format(preshape))
    if len(preshape) == 0:
        npol = 1
    else:
        npol = preshape[0]
                    
    itaus = np.zeros((3, 3, w_ell.shape[0]))

    for jidx in range(w_ell.shape[0]):
        itaus[:,:,jidx] = get_isotropic_ivar(
            icov_wav.maps[jidx,jidx], icov_wav.minfos[jidx,jidx])

    ivar_ell = np.einsum('ijk, kl -> ijl', itaus, w_ell ** 2,
                         optimize=True)
    
    return ivar_ell

