'''
A collection of functions for dealing with Gauss-Legendre pixelated maps.
'''
import numpy as np
from scipy.interpolate import NearestNDInterpolator, RectBivariateSpline

from pixell import enmap, sharp, utils, wcsutils
import healpy as hp

from optweight import wavtrans, sht, mat_utils, type_utils, alm_c_utils

def view_2d(imap, minfo):
    '''
    Expand last dimension to 2D, return view.

    Parameters
    ----------
    imap : (..., npix) array
        Input map.
    minfo : sharp.map_info object
        Metainfo for map.

    Returns
    -------
    omap : (..., nx, nx) array
        View of input map.

    Raises
    ------
    ValueError
        If view cannot be return without making a copy.
    '''

    omap = imap.view()

    try:
        omap.shape = (imap.shape[:-1] + (minfo.nrow, minfo.nphi[0]))
    except AttributeError:
        raise ValueError("Cannot return (..., ny, nx) view without copy.")
        
    return omap

def view_1d(imap, minfo):
    '''
    Return 1D view of map with last dimension expanded to 2D.

    Parameters
    ----------
    imap : (..., ny, nx) array
        Input map.
    minfo : sharp.map_info object
        Metainfo for map.

    Returns
    -------
    omap : (..., npix) array
        View of input map.

    Raises
    ------
    ValueError
        If view cannot be return without making a copy.
    '''

    omap = imap.view()

    try:
        omap.shape = (imap.shape[:-2] + (minfo.npix,))
    except AttributeError:
        raise ValueError("Cannot return (..., npix) view without copy.")
        
    return omap

#@profile
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
        that need a correction for pixel area difference use area_pow = -1 and
        for inverse variance maps use area_pow = 1.
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
        If enmap is not cylindrical.
    '''
    
    minfo = get_enmap_minfo(imap.shape, imap.wcs, lmax)

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
            # Linear interpolation should be good enough for pixel areas.
            omap[...,start:end] *= area_in.at(
                pos, order=1, mask_nan=False, prefilter=False, 
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

def gauss2gauss(imap, minfo_in, minfo_out, order=3, area_pow=0):
    '''
    Interpolate one Gauss-Legendre map to another at different resolution.

    imap : (..., npix) array
        Input map.
    minfo_in : sharp.map_info object
        Metainfo input map.
    minfo_out : sharp.map_info object
        Metainfo output map.
    order : int
        Order of the spline used for interpolation.
    area_pow : float, optional
        How the quantity in the map scales with area, i.e.
        (area_pix_out / area_pix_in)^area_power. Useful for e.g. variance maps
        that need a correction for pixel area difference use area_pow = -1 and
        for inverse variance maps use area_pow = 1.
    
    Returns
    -------
    omap : (..., npix_out) array
        Interpolated output map.
    '''

    omap = np.zeros(imap.shape[:-1] + (minfo_out.npix,), dtype=imap.dtype)

    thetas_in, phis_in = _get_gauss_coords(minfo_in)
    thetas_out, phis_out = _get_gauss_coords(minfo_out)

    imap = view_2d(imap, minfo_in)
    omap = view_2d(omap, minfo_out)

    if area_pow != 0:
        imap = imap * (minfo_in.weight ** -area_pow)[:,np.newaxis]
    
    # Loop over flattened leading dims.
    for idx in np.ndindex(imap.shape[:-2]):
        
        rbs = RectBivariateSpline(thetas_in, phis_in, imap[idx], kx=order, ky=order)
        omap[idx] = rbs(thetas_out, phis_out)

    if area_pow != 0:
        omap *= (minfo_out.weight ** area_pow)[:,np.newaxis]
    
    return view_1d(omap, minfo_out)

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
    uv = np.random.randn(npol, npix).astype(cov_pix.dtype)

    if cov_pix.ndim == 2:
        cov_pix_sqrt = np.sqrt(cov_pix)
        uv *= cov_pix_sqrt

    elif cov_pix.ndim == 3:
        # We use SVD (=eigenvalue decomposition for positive-semi-definite matrices)
        # because some pixels are zero and (i)cov is positive *semi*-definite
        # instead of *definite*, so Cholesky won't work.    
        cov_pix_sqrt = mat_utils.matpow(cov_pix, 0.5)
        uv = np.einsum('ijk, jk -> ik', cov_pix_sqrt, uv, optimize=True)

    return uv

def inv_qweight_map(imap, minfo, inplace=False, qweight=False):
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
    qweight : bool
        If set, calculate W m instead.

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
        if qweight:
            iweight = minfo.weight[ridx]
        else:
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
    icov_wav : (nwav, nwav) wavtrans.Wav object
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

    npol = wavtrans.preshape2npol(icov_wav.preshape)
   
    itaus = np.zeros((3, 3, w_ell.shape[0]))

    for jidx in range(w_ell.shape[0]):
        itaus[:,:,jidx] = get_isotropic_ivar(
            icov_wav.maps[jidx,jidx], icov_wav.minfos[jidx,jidx])

    ivar_ell = np.einsum('ijk, kl -> ijl', itaus, w_ell ** 2,
                         optimize=True)
    
    return ivar_ell

def rand_wav(cov_wav):
    '''
    Draw random Gaussian realisation from wavelet-based 
    block diagonal covariance matrix.

    Parameters
    ----------
    cov_wav : (nwav, nwav) wavtrans.Wav object
        Block-diagonal covariance matrix.

    returns
    -------
    rand_wav : (nwav) wavtrans.Wav object
        Wavelet vector containing the random draw.
    '''
    
    rand_wav = wavtrans.Wav(1, dtype=cov_wav.dtype)

    for jidx in range(cov_wav.shape[0]):
        
        cov_pix = cov_wav.maps[jidx,jidx]
        rand_wav.add(np.asarray([jidx]), rand_map_pix(cov_pix),
                     cov_wav.minfos[jidx,jidx])
                                 
    return rand_wav

def round_icov_matrix(icov_pix, rtol=1e-2):
    '''
    Set too small values in inverse covariance matrix to zero.

    Parameters
    ----------
    icov_pix : (npol, npol, npix) or (npol, npix) array
        Inverse covariance matrix.
    rtol : float, optional
        Elements below rtol times the median of nonzero elements 
        are set to zero.

    Returns
    -------
    icov_pix_round : (npol, npol, npix) or (npol, npix) array
        Rounded inverse covariance matrix.

    Raises
    ------
    ValueError
        If dimensionality of matrix is not understood.
    '''

    ndim = icov_pix.ndim
    if not (ndim == 2 or ndim == 3):
        raise ValueError(
            'Wrong dimensionality of icov_pix : {}, expected 2 or 3'.
            format(ndim))

    npol = icov_pix.shape[0]
    if ndim == 3 and icov_pix.shape[1] != npol:
        raise ValueError('Expected (npol, npol, npix) matrix, got : {}'.
                         format(icov_pix.shape))

    icov_pix = icov_pix.copy()
    
    # Loop over diagonal.
    for pidx in range(npol):

        index = (pidx, pidx) if ndim == 3 else pidx
        
        mask_nonzero = icov_pix[index] > 0
        if np.sum(mask_nonzero) == 0:
            continue

        median = np.median(icov_pix[index][mask_nonzero])
        mask = icov_pix[index] < rtol * median
        
        if ndim == 2:
            icov_pix[index][mask] = 0
        else:
            icov_pix[:,pidx,mask] = 0
            icov_pix[pidx,:,mask] = 0

    return icov_pix

def get_enmap_minfo(shape, wcs, lmax):
    '''
    Compute map_info metadata for a Gauss-Legendre grid
    given a cylindrical enmap shape and wcs.

    Parameters
    ----------
    shape : the shape of the enmap geometry
    wcs : the wcs object of the enmap geometry
    lmax : int
        Band limit supported by Gauss-Legendre grid.

    Returns
    -------
    map_info : sharp.map_info object
        metadata of Gausss-Legendre grid.
    '''

    if not wcsutils.is_cyl(wcs):
        raise NotImplementedError('Non-cylindrical enmaps not supported')

    ny, nx = shape[-2:]
    dec_range = enmap.pix2sky(
        shape, wcs, [[0, ny-1], [0, 0]], safe=False)[0]
    ra_range = enmap.pix2sky(
        shape, wcs, [[0, 0], [0, nx-1]], safe=False)[1]

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

    return minfo

def select_mask_edge(mask, minfo):
    '''
    Return boolean mask that is True for all pixels that are on the edge of
    the mask.

    Parameters
    ----------
    mask : (..., npix) bool array
        False underneath the mask, True otherwise.
    minfo : sharp.map_info object
        Metainfo for pixelization of input maps.

    Returns
    -------
    mask_edge : (..., npix) bool array
        True for pixels closer or equal than radius away from mask edge.

    Raises
    ------
    ValueError
        If map is not cylindrical.
    '''

    mask = mask.astype(bool)
    mask_2d = mask.reshape(mask.shape[:-1] + (minfo.nrow, minfo.nphi[0]))
    
    # Detect edge by convolution.
    # First left and right, we wrap around the phi direction.
    edges_lr = mask_2d ^ np.roll(mask_2d, -1, axis=-1)
    edges_lr |= mask_2d ^ np.roll(mask_2d, 1, axis=-1)

    # Then up and down. Complication: we don't want to wrap around theta.
    mask_shift_up = np.roll(mask_2d, -1, axis=-2)
    mask_shift_up[...,-1,:] = mask_2d[...,-1,:]

    mask_shift_down = np.roll(mask_2d, 1, axis=-2)
    mask_shift_down[...,0,:] = mask_2d[...,0,:]

    edges_ud = mask_2d ^ mask_shift_up
    edges_ud |= mask_2d ^ mask_shift_down

    edges = (edges_lr | edges_ud) & mask_2d
    
    return edges.reshape(mask.shape[:-1] + (minfo.npix,))    

def inpaint_nearest(imap, mask, minfo):
    '''
    Inpaint regions under mask using nearest neighbour.

    Arguments
    ---------    
    imap : (..., npix)
        Input map.
    mask : (..., npix) or (npix) bool array
        Mask, False for bad areas.
    minfo : sharp.map_info object
        Metainfo for map and mask.

    Returns
    -------
    omap : (npol, npix)
        Inpainted map.

    Raises
    ------
    ValueError
        If mask shape is not supported.

    Notes
    -----
    Does not wrap around in phi direction.
    '''

    shape_in = imap.shape

    if imap.ndim == 1:
        imap = imap[np.newaxis,:]

    if mask.ndim == 1:
        mask = mask[np.newaxis,:]
    
    preshape = imap.shape[:-1]
    if not (mask.shape[:-1] == preshape or mask.shape[:-1] == (1,)):
        raise ValueError(f'Leading dimensions mask should be (1,) or match '
                         f'those of imap : {preshape}, got {mask.shape[:-1]}')

    # Flatten all leading dimensions and call them npol.
    imap = imap.reshape(np.prod(imap.shape[:-1]), imap.shape[-1])
    mask = mask.reshape(np.prod(mask.shape[:-1]), mask.shape[-1])
    npol = imap.shape[0]

    omap = imap.copy()

    edges = select_mask_edge(mask, minfo)

    pix = np.mgrid[:minfo.nrow,:minfo.nphi[0]]
    pix = pix.reshape(2, minfo.nrow * minfo.nphi[0])
    
    pix_edge = [pix[:,edges[midx]] for midx in range(mask.shape[0])]
    pix_masked = [pix[:,~mask[midx]] for midx in range(mask.shape[0])]

    for pidx in range(npol):

        if mask.shape[0] == npol:
            midx = pidx
        else:
            midx = 0
            
        map_edge = imap[pidx][edges[midx]]
        ndi = NearestNDInterpolator(pix_edge[midx].T, map_edge)
        
        omap[pidx,~mask[midx]] = ndi(pix_masked[midx][0], pix_masked[midx][1])

    return omap.reshape(shape_in)

def lmul_pix(imap, lmat, minfo, spin, inplace=False, adjoint=False):
    '''
    Convert map to spherical harmonic domain, apply matrix and convert back.

    Arguments
    ---------
    imap : (npol, npix)
        Input map.
    lmat : (npol, npol, nell) or (npol, nell) array
        Matrix in multipole domain, if diagonal only the diaganal suffices.
    minfo : sharp.map_info object
        Metainfo input map.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.    
    inplace : bool, optional
        If set, use input map array for output.

    Returns
    -------
    omap : (npol, npix)
        Output map.
    '''

    if imap.ndim == 1:
        imap = imap[np.newaxis,:]
    npol = imap.shape[0]

    lmax = lmat.shape[-1] - 1
    ainfo = sharp.alm_info(lmax)
    alm = np.zeros((npol, ainfo.nelem), dtype=type_utils.to_complex(imap.dtype))

    sht.map2alm(imap, alm, minfo, ainfo, spin, adjoint=False)
    alm_c_utils.lmul(alm, lmat, ainfo, inplace=True)

    if inplace:
        omap = imap
    else:
        omap = np.zeros_like(imap)

    sht.alm2map(alm, omap, ainfo, minfo, spin, adjoint=False)
    
    return omap

def minfo2lmax(minfo):
    '''
    Determine lmax from map info assuming GL pixelization.

    Arguments
    ---------
    minfo : sharp.map_info object
        Metainfo of map

    Returns
    -------
    lmax : int
        Max multipole supported by pixelization.
    '''
    
    return int(0.5 * (minfo.nphi[0] - 1))

def copy_minfo(minfo):
    '''
    Return copy (new instance) of map info object.

    Arguments
    ---------
    minfo : sharp.map_info object
        Object to be copied.

    Returns
    -------
    minfo_copy : sharp.map_info object
        Copy.
    '''

    return sharp.map_info(theta=minfo.theta, nphi=minfo.nphi,
                          phi0=minfo.phi0, offsets=minfo.offsets,
                          stride=minfo.stride, weight=minfo.weight)

