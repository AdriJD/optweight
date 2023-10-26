'''
A collection of functions to deal with maps defined as sets of iso-latitude
rings.
'''
import numpy as np
from scipy.interpolate import (NearestNDInterpolator, RectBivariateSpline,
                               interp1d)
from scipy.special import roots_legendre
import os

from pixell import enmap, utils, wcsutils, curvedsky
import healpy as hp
import h5py
import ducc0

from optweight import (wavtrans, sht, mat_utils, type_utils, alm_c_utils,
                       dft, map_c_utils)

class MapInfo():
    '''
    Map info object that mimics the map_info object used for libsharp.
    Contains geometry information needed for spherical harmonic transforms.

    Parameters
    ----------
    theta : (ntheta) array
        Co-latitudes.
    weight : (ntheta) array
        Quadrature per pixel per ring.    
    nphi : int or (ntheta) int array, optional
        Number of samples per ring.
    phi0 : float or (ntheta) array, optional
        First longitude coordinate per ring.
    offsets : (ntheta) int array, optional
        The pixel index of the first pixel per ring.
    stride : int or (ntheta) int array, optional
        Stride per ring.

    Attributes
    ----------
    theta : (ntheta) array
        Co-latitudes.
    weight : (ntheta) array
        Quadrature per pixel per ring.    
    nphi : int or (ntheta) int array, optional
        Number of samples per ring.
    phi0 : float or (ntheta) array, optional
        First longitude coordinate per ring.
    offsets : (ntheta) int array, optional
        The pixel index of the first pixel per ring.
    stride : int or (ntheta) int array, optional
        Stride per ring.
    nrow : int
        ntheta.
    npix : int
        Number of pixels in map.
    '''
    
    def __init__(self, theta, weight, nphi=0, phi0=0, offsets=None, stride=None):
        
        self.theta = np.array(theta, dtype=np.float64)
        if self.theta.ndim != 1:
            raise ValueError(f'theta array must be 1d, got {self.theta.ndim}')

        self.nrow = self.theta.size
        
        self.weight = np.array(weight, dtype=np.float64)
        if self.weight.shape != (self.nrow,):
            raise ValueError(
                f'weight must be ({self.nrow},) got {self.weight.shape}')
        
        self._nphi = np.array(nphi, dtype=np.uint64)
        if not self._nphi.shape in ((), (self.nrow,)):
            raise ValueError(
                f'nphi must be 0d or ({self.nrow},) got {self._nphi.shape}')
        if self._nphi.ndim == 0:
            self._nphi = np.full(self.nrow, self._nphi, dtype=self._nphi.dtype)

        self.phi0 = np.array(phi0, dtype=np.float64)
        if not self.phi0.shape in ((), (self.nrow,)):
            raise ValueError(
                f'phi0 must be 0d or ({self.nrow},) got {self.phi0.shape}')
        if self.phi0.ndim == 0:
            self.phi0 = np.full(self.nrow, self.phi0, dtype=self.phi0.dtype)

        if offsets is None:
            self._offsets = np.concatenate(
                [[0],np.cumsum(self.nphi[:-1])]).astype(np.uint64)
        else:
            self._offsets = np.array(offsets, dtype=np.uint64)
        if self._offsets.shape != (self.nrow,):
            raise ValueError(
                f'offsets must be ({self.nrow},) got {self.offseets.shape}')

        if stride is None:
            self.stride = np.ones(self.nrow, np.int64)
        else:
            self.stride = np.array(stride, dtype=np.int64)
        if not self.stride.shape in ((), (self.nrow,)):
            raise ValueError(
            f'stride must be 0d or ({self.nrow},) got {self.stride.shape}')
        if self.stride.ndim == 0:
            self.stride = np.full(
                self.nrow, self.stride, dtype=self.stride.dtype)
            
        self.npix = int(np.sum(self.nphi))
            
    @property
    def nphi(self):
        '''Return signed int version.'''
        return self._nphi.astype(np.int64)

    @property
    def offsets(self):
        '''Return signed int version.'''
        return self._offsets.astype(np.int64)
        
    @classmethod
    def map_info_healpix(cls, nside):
        '''
        Construct a new map_info object for the HEALPix pixelization in the RING
        scheme.

        Parameters
        ----------
        nside : int
            NSIDE parameter.

        Returns
        -------
        minfo : map_utils.MapInfo object
            Map_info object.
        '''

        HB = ducc0.healpix.Healpix_Base(nside, 'RING')
        hdict = HB.sht_info()
        theta = hdict['theta']
        nphi = hdict['nphi']
        phi0 = hdict['phi0']
        offsets = hdict['ringstart']
        stride = 1
        
        npix = 12 * nside ** 2
        weight = 4 * np.pi / npix
        weight = np.full(theta.size, weight)
        
        return cls(theta, weight, nphi=nphi, phi0=phi0, offsets=offsets,
                   stride=stride)
        
    @classmethod
    def map_info_gauss_legendre(cls, nrings, nphi, phi0=0,
                                stride_lon=None, stride_lat=None):
        '''
        Construct a new map_info object for the Gauss-Legendre pixelization.

        Parameters
        ----------
        nrings : int
            Number of rings.
        nphi : int or (ntheta) int array
            Number of samples per ring.
        phi0 : float or (ntheta) array, optional
            First longitude coordinate per ring.
        stride_lon : int or (ntheta) int array, optional
            Stride per ring.
        stride_lat : int or (ntheta) int array, optional
            The pixel index of the first pixel per ring.

        Returns
        -------
        minfo : map_utils.MapInfo object
            Map_info object.
        '''

        theta, weight = roots_legendre(nrings)
        theta = np.ascontiguousarray(np.arccos(theta)[::-1])
        weight = np.ascontiguousarray(weight[::-1] * 2 * np.pi)        
        weight /= nphi

        if stride_lat is not None:
            offsets = np.arange(theta.size) * stride_lat
        else:
            offsets = None
        return cls(theta, weight, nphi=nphi, phi0=phi0, offsets=offsets,
                    stride=stride_lon)

    @classmethod
    def map_info_clenshaw_curtis(cls, nrings, nphi, phi0=0,
                                 stride_lon=None, stride_lat=None):
        '''
        Construct a new map_info object for the Clenshaw-Curtis pixelization.

        Parameters
        ----------
        nrings : int
            Number of rings.
        nphi : int or (ntheta) int array
            Number of samples per ring.
        phi0 : float or (ntheta) array, optional
            First longitude coordinate per ring.
        stride_lon : int or (ntheta) int array, optional
            Stride per ring.
        stride_lat : int or (ntheta) int array, optional
            The pixel index of the first pixel per ring.

        Returns
        -------
        minfo : map_utils.MapInfo object
            Map_info object.
        '''
        
        theta = np.linspace(0, np.pi, num=nrings, endpoint=True)
        weight = ducc0.sht.experimental.get_gridweights(
            geometry='CC', ntheta=nrings)
        weight /= nphi

        if stride_lat is not None:
            offsets = np.arange(theta.size) * stride_lat
        else:
            offsets = None
        return cls(theta, weight, nphi=nphi, phi0=phi0, offsets=offsets,
                    stride=stride_lon)

    @classmethod
    def map_info_fejer1(cls, nrings, nphi, phi0=0,
                        stride_lon=None, stride_lat=None):
        '''
        Construct a new map_info object for the Fejer1 pixelization.

        Parameters
        ----------
        nrings : int
            Number of rings.
        nphi : int or (ntheta) int array
            Number of samples per ring.
        phi0 : float or (ntheta) array, optional
            First longitude coordinate per ring.
        stride_lon : int or (ntheta) int array, optional
            Stride per ring.
        stride_lat : int or (ntheta) int array, optional
            The pixel index of the first pixel per ring.

        Returns
        -------
        minfo : map_utils.MapInfo object
            Map_info object.
        '''
        
        theta = np.linspace(0, np.pi, num=nrings + 1, endpoint=True)[:-1]
        theta += np.pi / nrings / 2
        weight = ducc0.sht.experimental.get_gridweights(
            geometry='F1', ntheta=nrings)
        weight /= nphi

        if stride_lat is not None:
            offsets = np.arange(theta.size) * stride_lat
        else:
            offsets = None
        return cls(theta, weight, nphi=nphi, phi0=phi0, offsets=offsets,
                    stride=stride_lon)

    @classmethod
    def map_info_fejer2(cls, nrings, nphi, phi0=0,
                        stride_lon=None, stride_lat=None):
        '''
        Construct a new map_info object for the Fejer2 pixelization.

        Parameters
        ----------
        nrings : int
            Number of rings.
        nphi : int or (ntheta) int array
            Number of samples per ring.
        phi0 : float or (ntheta) array, optional
            First longitude coordinate per ring.
        stride_lon : int or (ntheta) int array, optional
            Stride per ring.
        stride_lat : int or (ntheta) int array, optional
            The pixel index of the first pixel per ring.

        Returns
        -------
        minfo : map_utils.MapInfo object
            Map_info object.
        '''
        
        theta = np.linspace(0, np.pi, num=nrings + 2, endpoint=True)[1:-1]
        weight = ducc0.sht.experimental.get_gridweights(
            geometry='F2', ntheta=nrings)
        weight /= nphi

        if stride_lat is not None:
            offsets = np.arange(theta.size) * stride_lat
        else:
            offsets = None
        return cls(theta, weight, nphi=nphi, phi0=phi0, offsets=offsets,
                    stride=stride_lon)

def get_fullsky_geometry(minfo):
    '''
    Take a cut-sky map_info object and return a map_info object that
    spans the full sky and shares overlapping theta ring coordinates.
    
    Parameters
    ----------
    minfo : map_utils.MapInfo object
        Metainfo describing input geometry.

    Raises
    ------
    ValueError
        If input is not GL, CC, F1 or F2.
        Input input maps has < 3 rings.
    
    Notes
    -----
    No guarantee that order of rings, nphi, phi0 offsets and strides are
    shared between input and output geometries.    
    '''

    if minfo.theta.size < 3:
        raise ValueError(f'Need atleast 3 rings, got {minfo.theta.size=}')
    
    diff = np.diff(minfo.theta)
    nphi = minfo.nphi[0]

    if not all(minfo.nphi == nphi):
        raise ValueError('Geometries with varying nphi not supported')
    print(diff, minfo.theta.size)
    if not np.allclose(diff, diff[0]):
        # Assume GL. Use emperical relation to get nrings.        
        nrings = int(np.round(np.min(np.pi / np.abs(diff) - 0.5)))
        minfo_fullsky = MapInfo.map_info_gauss_legendre(nrings, nphi)
        
    else:
        # Assume either CC, F1 or F2.
        dtheta = diff[0]
        diff_norm = (np.pi - minfo.theta.max()) / dtheta
        remain = np.round(np.remainder(diff_norm, 1), 1)
        
        if remain == 0.0:
            # CC, could also be F2, but we cannot distinguish between those.
            nrings = int(np.round(np.pi / dtheta + 1))
            minfo_fullsky = MapInfo.map_info_clenshaw_curtis(nrings, nphi)
            
        elif remain == 0.5:
            # Fejer 1.
            nrings = int(np.round(np.pi / dtheta ))
            minfo_fullsky = MapInfo.map_info_fejer1(nrings, nphi)
            
        else:
            raise ValueError(f'Cannot determine mtype, {remain=}')
                
    # Perform checks to make sure the input thetas are a subset of output.
    theta_sorted = np.sort(minfo.theta)
    theta_fullsky_sorted = np.sort(minfo_fullsky.theta)

    # Find index of first theta ring in fullsky thetas.
    idx_start = np.argwhere(np.isclose(theta_fullsky_sorted, theta_sorted[0]))
    if idx_start.size != 1:
        raise ValueError('theta_sort[0] of input not in fullsky output')
    idx_start = idx_start[0][0]
    if not np.allclose(theta_fullsky_sorted[idx_start:idx_start+minfo.nrow],
                       theta_sorted):
        raise ValueError('theta input does not overlap with fullsky output')
    
    return minfo_fullsky        
        
def view_2d(imap, minfo):
    '''
    Expand last dimension to 2D, return view.

    Parameters
    ----------
    imap : (..., npix) array
        Input map.
    minfo : map_utils.MapInfo object
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
    minfo : map_utils.MapInfo object
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

def equal_area_gauss_copy_2d(imap, minfo):
    '''
    Return 2D copy (for plotting and debugging) of equal_area_gauss grid 
    using nearest-neighbor interpolation (or any map with unequal elements per ring).

    Parameters
    ----------
    imap : (..., npix) array
        Input map.
    minfo : map_utils.MapInfo object
        Metainfo for map.

    Returns
    -------
    omap : (..., ny, nx) array
        Output map.
    '''
    
    idx_max = np.argmax(minfo.nphi)
        
    omap = np.zeros(imap.shape[:-1] + (minfo.nrow, minfo.nphi[idx_max]),
                     dtype=imap.dtype)
    
    # Determine phi coordinates in omap.
    nphi_omap = minfo.nphi[idx_max]
    phi0_omap = minfo.phi0[idx_max]
    phis_omap = np.linspace(phi0_omap, 2 * np.pi + phi0_omap , nphi_omap,
                            endpoint=False)
                            
    # Loop over thetas
    for tidx in range(minfo.nrow):

        # Determine phi coords in this ring of imap.
        nphi = minfo.nphi[tidx]
        phi0 = minfo.phi0[tidx]
        phis = np.linspace(phi0, 2 * np.pi + phi0 , nphi, endpoint=False)

        # Determine slice into to reduced gauss map (because it's not 2d)
        ring_slice = get_ring_slice(tidx, minfo)

        # Loop over outer dims
        for index in np.ndindex(imap.shape[:-1]):
    
            fp = interp1d(phis, imap[index][ring_slice], kind='nearest',
                          fill_value='extrapolate')

            omap[index][tidx,:] = fp(phis_omap).astype(omap.dtype, copy=False)

    return omap

def get_ring_slice(ring_idx, minfo):
    '''
    Return slice object that would return view of i'th ring of a map.

    Parameters
    ----------
    ring_idx : int
       Ring index.
    minfo : map_utils.MapInfo object
        Metainfo for map.

    Returns
    -------
    ring_slice : slice object
        Slice object such that map[ring_slice] gives you the i'th ring.
    '''

    return slice(int(minfo.offsets[ring_idx]),
                 int(minfo.offsets[ring_idx] + minfo.nphi[ring_idx]),
                 int(minfo.stride[ring_idx]))    

def enmap2gauss(imap, lmax, order=3, area_pow=0, destroy_input=False,
                mode='constant'):
    '''
    Interpolate input enmap to Gauss-Legendre grid.

    Parameters
    ----------
    imap : (..., ny, nx) enmap
        Input map(s)
    lmax : int or map_utils.MapInfo object
        Band limit supported by Gauss-Legendre grid or map_info object that
        describes output geometry.
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
    map_info : map_utils.MapInfo object
        Metadata of output map.

    Raises
    ------
    NotImplementedError
        If enmap is not cylindrical.
    '''
    
    if isinstance(lmax, MapInfo):
        minfo = lmax
    else:
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
    map_info : map_utils.MapInfo object
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
    minfo_in : map_utils.MapInfo object
        Metainfo input map.
    minfo_out : map_utils.MapInfo object
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
        area_pow = float(area_pow)
        imap = imap * (minfo_in.weight ** -area_pow)[:,np.newaxis]
    
    # Loop over flattened leading dims.
    for idx in np.ndindex(imap.shape[:-2]):
        
        rbs = RectBivariateSpline(thetas_in, phis_in, imap[idx], kx=order, ky=order)
        omap[idx] = rbs(thetas_out, phis_out)

    if area_pow != 0:
        omap *= (minfo_out.weight ** area_pow)[:,np.newaxis]
    
    return view_1d(omap, minfo_out)

def gauss2map(imap, minfo_in, minfo_out, order=3, area_pow=0):
    '''
    Interpolate one Gauss-Legendre map to an equal-area GL map (or a generic 
    ducc compatible map) at different resolution.

    imap : (..., npix) array
        Input map.
    minfo_in : map_utils.MapInfo object
        Metainfo input map.
    minfo_out : map_utils.MapInfo object
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

    thetas_in, phis_in = _get_gauss_coords(minfo_in)

    # This is a bit slow, but at least very general.
    # RectBivariateSpline needs stricly ascending coordinates.
    theta_idxs = np.argsort(thetas_in)
    phi_idxs = np.argsort(phis_in)

    omap = np.zeros(imap.shape[:-1] + (minfo_out.npix,), dtype=imap.dtype)    

    if area_pow != 0:
        area_pow = float(area_pow)
        imap = imap.copy()

    imap_2d = view_2d(imap, minfo_in)
    if area_pow != 0:
        imap_2d *= (minfo_in.weight ** -area_pow)[:,np.newaxis]

    # Loop over out dims
    for index in np.ndindex(imap.shape[:-1]):

        map2rbs = imap_2d[index][theta_idxs,:]
        map2rbs = map2rbs[:,phi_idxs]
        rbs = RectBivariateSpline(thetas_in[theta_idxs], phis_in[phi_idxs],
                                  map2rbs, kx=order, ky=order)

        # Loop over thetas.
        for tidx in range(minfo_out.theta.size):
           
            nphi = minfo_out.nphi[tidx]
            phi0 = minfo_out.phi0[tidx]

            # ADD STRIDE ....
            phis = np.linspace(phi0, 2 * np.pi + phi0, nphi, endpoint=False)

            ring_slice = get_ring_slice(tidx, minfo_out)
            omap[index][ring_slice] = rbs(minfo_out.theta[tidx], phis)
            
            if area_pow != 0:
                omap[index][ring_slice] *= (minfo_out.weight[tidx] ** area_pow)

    return omap
    
def _get_gauss_coords(map_info):
    '''
    Return coordinate arrays for map_info.

    Parameters
    ----------
    map_info : map_utils.MapInfo object
        Metadata of Gauss-Legendre map.

    Returns
    -------
    thetas : (ntheta) array
        Co-latitude coordinates in radians (0 at north pole, pi at south pole).
    phi : (nphi) array
        Longitude coordinates for each ring.
    '''

    if not np.all(map_info.nphi == map_info.nphi[0]):
        raise ValueError('Input map_info nphi is not the same for all rings')
    if not np.all(map_info.stride == map_info.stride[0]):
        raise ValueError('Input map_info phi stride is not the same for all rings')

    # I know that each ring has nphi phis and also that the phi coordinates
    # increase/decrease at same rate for each ring.
    thetas = map_info.theta
    nphi = map_info.nphi[0]
    phi0 = map_info.phi0[0]
    phis = np.linspace(phi0, phi0 + (2 * np.pi * map_info.stride[0]), nphi, endpoint=False)
    phis = np.mod(phis, 2 * np.pi)

    return thetas, phis

def get_minfo(mtype, nrings, nphi, theta_min=None, theta_max=None, return_arc_len=False):
    '''
    Compute map_info metadata for a rectangular grid.

    Parameters
    ----------
    mtype : str
        Pick between "GL" (Gauss Legendre), "CC" (Clenshaw Curtis), F1
        or F2 (Fejer 1 and 2).
    nrings : int
        Number of theta rings on full sky.
    nphi : int
        Number of nphi points for all rings.
    theta_min : float, optional
        Minimum colattitute of grid in radians.
    theta_max : float, optional
        Maximum colattitute of grid in radians.
    return_arc_len : bool, optional
        If set, return arc length of each theta ring.

    Returns
    -------
    map_info : map_utils.MapInfo object
        metadata of grid.
    arc_lengths : (ntheta) array
        If return_arc_len is set: arc length of rings along the theta direction.
    '''

    if mtype == 'GL':
        map_info = MapInfo.map_info_gauss_legendre(nrings, nphi)
    elif mtype == 'CC':
        map_info = MapInfo.map_info_clenshaw_curtis(nrings, nphi)
    elif mtype == 'F1':
        map_info = MapInfo.map_info_fejer1(nrings, nphi)
    elif mtype == 'F2':
        map_info = MapInfo.map_info_fejer2(nrings, nphi)        
    else:
        raise ValueError(f'mtype : {mtype} not supported')

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
        map_info = MapInfo(theta, nphi=nphi, phi0=0, offsets=offsets,
                           stride=stride, weight=weight)
        if return_arc_len:
            arc_len = arc_len[theta_mask]

    ret = map_info
    if return_arc_len:
        return map_info, + arc_len,

    return ret    

def get_cc_minfo(lmax, theta_min=None, theta_max=None, return_arc_len=False):
    '''
    Compute map_info metadata for a Clenshaw Curtis grid.

    Parameters
    ----------
    lmax : int
        Band limit supported by grid.
    theta_min : float, optional
        Minimum colattitute of grid in radians.
    theta_max : float, optional
        Maximum colattitute of grid in radians.
    return_arc_len : bool, optional
        If set, return arc length of each theta ring.

    Returns
    -------
    map_info : map_utils.MapInfo object
        metadata of Gausss-Legendre grid.
    arc_lengths : (ntheta) array
        If return_arc_len is set: arc length of rings along the theta direction.
    '''

    nrings = lmax + 1
    nphi = lmax + 1

    return get_minfo('CC', nrings, nphi, theta_min=theta_min, theta_max=theta_max,
                     return_arc_len=return_arc_len)

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
    map_info : map_utils.MapInfo object
        metadata of Gausss-Legendre grid.
    arc_lengths : (ntheta) array
        If return_arc_len is set: arc length of rings along the theta direction.
    '''

    nrings = int(np.floor(lmax / 2)) + 1
    nphi = lmax + 1

    return get_minfo('GL', nrings, nphi, theta_min=theta_min, theta_max=theta_max,
                     return_arc_len=return_arc_len)

def get_equal_area_minfo(minfo, orig_band=np.pi/8, ratio_pow=1.):
    '''
    Convert input minfo into equal area version.

    Parameters
    ----------
    minfo : map_utils.MapInfo object
        Metadata of input grid.
    orig_band : float, optional
        Angle in radians away from equator that determines the band 
        in which the grid is set to the original input grid. 
        Set to 0 to turn off.
    ratio_pow : float, optional
        if set to value < 1, tolerate smaller area pixels close to pole.
        Returns to input if set to 0. Number of phi samples per ring 
        is given by nphi_gl * (area_pix_on_ring / area_pix_equator)^ratio_pow.

    Returns
    -------
    map_info : map_utils.MapInfo object
        metadata of grid.    
    '''

    assert orig_band >= 0, ('orig_band must be >= 0, got {gl_band}')

    # Use pixel area on equator as reference. This assumes this ring is included.
    # Make sure this ring is actually included.
    delta_theta = np.min(np.abs(np.diff(minfo.theta)))
    if not np.any(np.abs(minfo.theta - (np.pi / 2)) < 2 * delta_theta):
        raise ValueError('Input minfo has to include equator.')

    idx_ref = np.searchsorted(minfo.theta, np.pi / 2, side="left")
    area_pix_ref = minfo.weight[idx_ref]

    nphi_reduced = np.zeros_like(minfo.nphi)
    weight_reduced = np.zeros_like(minfo.weight)

    for tidx in range(minfo.nrow):

        area_pix = minfo.weight[tidx]

        if (np.pi / 2 - orig_band) < minfo.theta[tidx] < (np.pi / 2 + orig_band):
            ratio = 1            
        else:
            ratio = (area_pix / area_pix_ref) ** ratio_pow

        nphi_reduced[tidx] = max(2, int(np.round(minfo.nphi[tidx] * ratio)))
        weight_reduced[tidx] = area_pix * minfo.nphi[tidx] / nphi_reduced[tidx]

    return MapInfo(minfo.theta, nphi=nphi_reduced, weight=weight_reduced)

def get_equal_area_gauss_minfo(lmax, theta_min=None, theta_max=None, 
                               gl_band=np.pi/8, ratio_pow=1.):
    '''
    Compute map_info metadata for a Gauss-Legendre grid that has
    samples per ring reduced to get approximate equal area pixels.

    Parameters
    ----------
    lmax : int
        Band limit supported by Gauss-Legendre grid.
    theta_min : float, optional
        Minimum colattitute of grid in radians.
    theta_max : float, optional
        Maximum colattitute of grid in radians.
    gl_band : float, optional
        Angle in radians away from equator that determines the band 
        in which the grid is set to the normal Gauss-Legendre grid. 
        Set to 0 to turn off.
    ratio_pow : float, optional
        if set to value < 1, tolerate smaller area pixels close to pole.
        Returns to normal GL if set to 0. Number of phi samples per ring 
        is given by nphi_gl * (area_pix_on_ring / area_pix_equator)^ratio_pow.

    Returns
    -------
    map_info : map_utils.MapInfo object
        metadata of grid.

    Notes
    -----
    Kept for backwards compatibility, see `get_equal_area_minfo` for general function.
    '''

    assert gl_band >= 0, ('gl_band must be >= 0, got {gl_band}')

    minfo = get_gauss_minfo(lmax, theta_min=theta_min, theta_max=theta_max)

    return get_equal_area_minfo(minfo, orig_band=gl_band, ratio_pow=ratio_pow)

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
    minfo : map_utils.MapInfo object
        Metainfo for pixelization of input maps.    
    ainfo : pixell.curvedsky.alm_info object
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

    if qweight:
        custom_weight = None
    else:
        custom_weight = 1 / minfo.weight
        
    preshape = out.shape[:-1]
    for idxs in np.ndindex(preshape):
        map_c_utils.apply_ringweight(
            out[idxs], minfo, custom_weight=custom_weight)
    if dim_in == 1:
        imap = out[0]

    return out

def get_isotropic_ivar(icov_pix, minfo, mask=None):
    '''
    Compute the isotropic inverse variance for an inverse 
    covariance matrix diagonal in pixel space.

    Parameters
    ----------
    icov_pix : (npol, npol, npix) or (npol, npix) array
        Inverse covariance matrix.
    minfo : map_utils.MapInfo object
        Metainfo specifying pixelization of covariance matrix.
    mask : (npol, npix) array, optional
        Sky mask, usually 1 for observed pixels, 0 for unobserved, but 
        can be apodized.
    
    Returns
    -------
    ivar_iso : (npol, npol) array
        Isotropic inverse variance. Note that only the diagonal is
        calculated.

    Notes
    -----
    See eq. A1 in Seljebotn et al. (1710.00621). If you have N_pix^-1
    then N_{ell m ell' m'} = Yt N_pix^-1 Y. ivar_iso is the scalar that 
    approximates 1 / N_{ell m ell' m'}.
    '''

    wcov = inv_qweight_map(icov_pix, minfo, inplace=False)

    if mask is not None:
        if wcov.ndim == 2:
            wcov *= mask
        else:
            mask = mat_utils.atleast_nd(mask, 2)
            sqrt_mask = np.sqrt(mask)
            wcov *= np.einsum('ap, bp -> abp', sqrt_mask, sqrt_mask)

    # Set off-diagonal elements to zero, I only understand diagonal for now.
    numerator = np.sum(wcov ** 2, axis=-1)
    denominator = np.sum(wcov, axis=-1)

    if icov_pix.ndim == 2:
        return np.diag(numerator / denominator)

    elif icov_pix.ndim == 3:
        return np.diag(np.diag(numerator) / np.diag(denominator))

def get_ivar_ell(icov_wav, w_ell, mask=None, minfo_mask=None):
    '''
    Compute the inverse variance spectrum for an inverse 
    covariance matrix diagonal in wavelet space.

    Parameters
    ----------
    icov_wav : (nwav, nwav) wavtrans.Wav object
        Inverse covariance matrix.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    mask : (npol, npix) array, optional
        Sky mask, usually 1 for observed pixels, 0 for unobserved.
    minfo_mask : map_utils.MapInfo object, optional
        Meta info for mask, only required if mask if given.
    
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

    itaus = np.zeros((npol, npol, w_ell.shape[0]))

    for jidx in range(w_ell.shape[0]):
        
        if mask is not None:
            mask_j = gauss2gauss(mask, minfo_mask, icov_wav.minfos[jidx,jidx],
                                 order=1, area_pow=0)
        else:
            mask_j = None

        itaus[:,:,jidx] = get_isotropic_ivar(
            icov_wav.maps[jidx,jidx], icov_wav.minfos[jidx,jidx], mask=mask_j)

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

def round_icov_matrix(icov_pix, rtol=1e-2, threshold=False):
    '''
    Set too small values in inverse covariance matrix to zero.

    Parameters
    ----------
    icov_pix : (npol, npol, npix) or (npol, npix) array
        Inverse covariance matrix.
    rtol : float, optional
        Elements below rtol times the median of nonzero elements 
        are set to zero.
    threshold : bool, optional
        If set, set values to lowest allowed value instead of 0.

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

        if threshold: 
            val = rtol * median
        else:
            val = 0

        if ndim == 2:
            icov_pix[index][mask] = val
        else:
            icov_pix[:,pidx,mask] = val
            icov_pix[pidx,:,mask] = val

    return icov_pix

def get_enmap_minfo(shape, wcs, lmax, pad=None, mtype='GL'):
    '''
    Compute map_info metadata for a ducc grid given a CAR enmap
    shape and wcs.

    Parameters
    ----------
    shape : tuple 
        The shape of the enmap geometry
    wcs : astropy.wcs.WCS object
        The wcs object of the enmap geometry
    lmax : int
        Band limit supported by Gauss-Legendre grid.
    pad : float, optional
        Amount of extra space [radians] in the theta (collatitude) 
        direction above and below region.
    mtype : str
        Pick between "GL" (Gauss Legendre), "CC" (Clenshaw Curtis).

    Returns
    -------
    map_info : map_utils.MapInfo object
        metadata of grid.

    Raises
    ------
    NotImplementedError
        If enmap is not cylindrical.
    ValueError
        If mtype is not understood.
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
    if pad is not None:
        if pad < 0: raise NotImplementedError(
                f'Only positive padding implemented, got pad : {pad}')
        theta_min = max(0, theta_min - pad)
        theta_max = min(np.pi, theta_max + pad)

    if mtype == 'GL':
        return get_gauss_minfo(lmax, theta_min=theta_min, theta_max=theta_max)
    elif mtype == 'CC':
        return get_cc_minfo(lmax, theta_min=theta_min, theta_max=theta_max)
    else:
        raise ValueError(f'mtype : {mtype} not supported')

def match_enmap_minfo(shape, wcs, mtype='CC'):
    '''
    Given a enmap in the Clenshaw-Curtis CAR variant, find the matching 
    map_info object for ducc.

    Parameters
    ----------
    shape : tuple
        The shape of the enmap geometry.
    wcs : astropy.wcs.WCS object
        The wcs object of the enmap geometry
    mtype : string
        The CAR variant. Right now only "CC" and "fejer1" supported.

    Returns
    -------
    map_info : map_utils.MapInfo object
        metadata of grid.

    Raises
    ------
    NotImplementedError
        If mtype is not understood.
    ValueError
        If input map does not span the whole sphere in RA.
        If the input geometry does not confom to the Clenshaw-Curtis or
        fejer1 rules.

    Notes
    -----
    Different from the methods in pixell.curvedsky in the sense that we
    don't modifiy the input map, e.g. flipping it horizontally, vertically,
    but instead modify the ducc's map info object if we need to flip something.
    '''

    if mtype not in ('CC', 'fejer1'):
        raise NotImplementedError('Only "CC" or "fejer1" `mtype` implemented for now.')
    if mtype == 'fejer1':
        yo = 0
    elif mtype == 'CC':
        yo = 1
    
    # Determine number of y and x pixels on the full sky.
    nx   = utils.nint(np.abs(360 / wcs.wcs.cdelt[0]))
    if nx != shape[-1]:
        raise ValueError(f'Input needs to span full range in phi, {shape[-1]}, {nx} ')
    
    # If CC, add 1 because Clenshaw Curtis has a pixel on both poles.
    ny = utils.nint(abs(180 / wcs.wcs.cdelt[1]) + yo)

    phi0 = enmap.pix2sky(shape, wcs, [0,0])[1]

    # Create full sky minfo.
    stride_lon = np.sign(wcs.wcs.cdelt[0])
    stride_lat = np.abs(np.sign(wcs.wcs.cdelt[1])) * nx
    if mtype == 'CC':
        minfo = MapInfo.map_info_clenshaw_curtis(
            ny, nphi=nx, phi0=phi0, stride_lon=stride_lon, stride_lat=stride_lat)
    else:
        minfo = MapInfo.map_info_fejer1(
            ny, nphi=nx, phi0=phi0, stride_lon=stride_lon, stride_lat=stride_lat)
        
    # Now find indices of first and last rings.
    # This is also where you crash if you don't find a match.
    ntheta = shape[-2]
    theta = enmap.pix2sky(shape, wcs, [np.arange(ntheta),np.zeros(ntheta)])[0]
    # Convert to colatitude, which is what ducc uses.
    theta = np.pi / 2 - theta

    atol = 1e-6
    idx_start = np.flatnonzero(np.abs(theta[0] - minfo.theta) < atol)
    if idx_start.size != 1:
        raise ValueError(f'Input geometry is not {mtype} at atol = {atol}')

    idx_end = np.flatnonzero(np.abs(theta[-1] - minfo.theta) < atol)
    if idx_end.size != 1:
        raise ValueError(f'Input geometry is not {mtype} at atol = {atol}')

    idx_start, idx_end = idx_start[0], idx_end[0]
    slice2keep = np.s_[min(idx_start, idx_end):max(idx_start, idx_end)+1]
    
    if wcs.wcs.cdelt[0] > 0:
        offsets2keep = minfo.offsets[slice2keep] - minfo.offsets[slice2keep][0] 
    else:
        offsets2keep = minfo.offsets[slice2keep] - minfo.offsets[slice2keep][0] + nx - 1
        phi0 -= np.radians(wcs.wcs.cdelt[0])

    theta2keep = minfo.theta[slice2keep]
    weight2keep = minfo.weight[slice2keep]
    stride = np.full(theta2keep.size, stride_lon)

    if wcs.wcs.cdelt[1] > 0:
        theta2keep = np.ascontiguousarray(theta2keep[::-1])
        weight2keep = np.ascontiguousarray(weight2keep[::-1])

    minfo_cut = MapInfo(theta2keep, nphi=nx, phi0=phi0,
                        offsets=offsets2keep,
                        stride=stride.astype(np.int32), weight=weight2keep)

    return minfo_cut    

def select_mask_edge(mask, minfo):
    '''
    Return boolean mask that is True for all pixels that are on the edge of
    the mask.

    Parameters
    ----------
    mask : (..., npix) bool array
        False underneath the mask, True otherwise.
    minfo : map_utils.MapInfo object
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

    Parameters
    ----------    
    imap : (..., npix)
        Input map.
    mask : (..., npix) or (npix) bool array
        Mask, False for bad areas.
    minfo : map_utils.MapInfo object
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
    pix = pix.reshape(2, minfo.nrow * int(minfo.nphi[0]))
    
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

    Parameters
    ----------
    imap : (npol, npix)
        Input map.
    lmat : (npol, npol, nell) or (npol, nell) array
        Matrix in multipole domain, if diagonal only the diaganal suffices.
    minfo : map_utils.MapInfo object
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
    ainfo = curvedsky.alm_info(lmax)
    alm = np.zeros((npol, ainfo.nelem), dtype=type_utils.to_complex(imap.dtype))

    sht.map2alm(imap, alm, minfo, ainfo, spin, adjoint=False)
    alm_c_utils.lmul(alm, lmat, ainfo, inplace=True)

    if inplace:
        omap = imap
    else:
        omap = np.zeros_like(imap)

    sht.alm2map(alm, omap, ainfo, minfo, spin, adjoint=False)

    return omap

def fmul_pix(imap, minfo, fmat2d=None, fmat1d=None, ells=None, modlmap=None,
             inplace=False):
    '''
    Convert map to 2D Fourier domain, apply matrix and convert back.

    Parameters
    ----------
    imap : (npol, npix)
        Input map.
    fmat2d : (npol, npol, nly, nlx) or (npol, nly, nlx) array, optional
        Matrix, if diagonal only the diagal suffices.
    fmat1d : (npol, npol, nell) or (npol, nly, nell) array, optional
        Matrix, if diagonal only the diagal suffices.
    ells : (nell) array, optional
        Array with multipoles, can be non-integer, needed for `fmat1d`.
    modlmap : (nly, nlx) array
        Map of absolute wavenumbers, needed for `fmat1d`.
    inplace : bool, optional
        If set, use input map array for output.

    Returns
    -------
    omap : (npol, npix)
        Output map.
    '''

    imap = mat_utils.atleast_nd(imap, 2)
    npol = imap.shape[0]

    imap2d = view_2d(imap, minfo)
    fmap = dft.allocate_fmap(imap2d.shape, imap2d.dtype)

    dft.rfft(imap2d, fmap)
    dft.fmul(fmap, fmat2d=fmat2d, fmat1d=fmat1d, ells=ells, modlmap=modlmap,
             out=fmap)    

    if inplace:
        omap2d = imap2d
    else:
        omap2d = np.zeros_like(imap2d)
        
    dft.irfft(fmap, omap2d)

    return view_1d(omap2d, minfo)

def minfo2lmax(minfo):
    '''
    Determine lmax from map_info based on the max number of phi samples
    on the map's rings (usually the equator).

    Parameters
    ----------
    minfo : map_utils.MapInfo object
        Metainfo of map.

    Returns
    -------
    lmax : int
        Max multipole supported by pixelization.
    '''
    
    return int(0.5 * (np.max(minfo.nphi) - 1))

def minfo2wcs(minfo):
    '''
    Determine CAR WCS object from map_info.

    Parameters
    ----------
    minfo : map_utils.MapInfo object
        Metainfo of map.

    Returns
    -------
    wcs : astropy.wcs.WCS object
        The wcs object of the enmap geometry

    Raises
    ------
    ValueError
        If no WCS can be determined (i.e. for Gauss Legendre minfo).
    '''

    diff = np.diff(minfo.theta)
    # CAR needs uniformly spaced theta coordinates.
    if not np.allclose(diff, np.full(diff.size, diff[0])):
        raise ValueError(f'Cannot determine CAR WCS for input minfo')
        
    dtheta = np.abs(minfo.theta[0] - minfo.theta[-1]) / minfo.nrow
    dphi = 2 * np.pi / minfo.nphi[0]
    _, phis = _get_gauss_coords(minfo)

    theta_idx_mid = minfo.nrow // 2
    phi_idx_mid = phis.size // 2

    wcs   = wcsutils.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
    wcs.wcs.cdelt = [-np.degrees(dphi), -np.degrees(dtheta)]
    wcs.wcs.crval = np.degrees([phis[phi_idx_mid], np.pi / 2 - minfo.theta[theta_idx_mid]])
    wcs.wcs.crpix = [phi_idx_mid, theta_idx_mid]

    return wcs

def copy_minfo(minfo):
    '''
    Return copy (new instance) of map info object.

    Parameters
    ----------
    minfo : map_utils.MapInfo object
        Object to be copied.

    Returns
    -------
    minfo_copy : map_utils.MapInfo object
        Copy.
    '''

    return MapInfo(theta=minfo.theta, nphi=minfo.nphi,
                   phi0=minfo.phi0, offsets=minfo.offsets,
                   stride=minfo.stride, weight=minfo.weight)

def minfo_is_equiv(minfo_1, minfo_2):
    '''
    Test whether two map info objects are equivalent.

    Parameters
    ----------
    minfo_1 : map_utils.MapInfo object
        First map info object.
    minfo_2 : map_utils.MapInfo object
        Second map info object.

    Returns
    -------
    is_equiv : bool
        True if equivalent, False if not.
    '''

    is_equiv = True
    attributes = ['theta', 'nphi', 'phi0', 'offsets', 'stride', 'weight']

    for attr in attributes:
        try:
            np.testing.assert_allclose(getattr(minfo_1, attr), getattr(minfo_2, attr))
        except AssertionError:
            is_equiv = False
            break

    return is_equiv

def write_minfo(fname, minfo):
    '''
    Write map info object to an hdf file.

    Parameters
    ----------
    fname : str
        Path to file. Will append .hdf5 if no file extension is found.
    minfo : map_utils.MapInfo object
        Map info object to be stored.    
    '''

    if not os.path.splitext(fname)[1]:
        fname = fname + '.hdf5'

    with h5py.File(fname, 'w') as hfile:
        
        append_minfo_to_hdf(hfile, minfo)
    
def append_minfo_to_hdf(hfile, minfo):
    '''
    Add map info object datasets to provided hdf file.

    Parameters
    ----------
    hfile : HDF5 file or group
        Writeable hdf file or group.
    minfo : map_utils.MapInfo object
        Map info object to be stored.        
    '''
    
    hfile.create_dataset('theta', data=minfo.theta)
    hfile.create_dataset('nphi', data=minfo.nphi)
    hfile.create_dataset('phi0', data=minfo.phi0)
    hfile.create_dataset('offsets', data=minfo.offsets)
    hfile.create_dataset('stride', data=minfo.stride)
    hfile.create_dataset('weight', data=minfo.weight)

def read_minfo(fname):
    '''
    Read map info object from hdf file.

    Parameters
    ----------
    fname : str
        Path to file.

    Returns
    -------
    minfo : map_utils.MapInfo object
        Map info object.
    '''

    with h5py.File(fname, 'r') as hfile:
        
        minfo = minfo_from_hdf(hfile)

    return minfo

def minfo_from_hdf(hfile):
    '''
    Extract map info object from hdf datasets.

    Parameters
    ----------
    hfile : HDF5 file or group
        Hdf file or group.
    
    Returns
    -------
    minfo : map_utils.MapInfo object
        Map info object.
    '''
    
    theta = hfile['theta'][()]
    nphi = hfile['nphi'][()]
    phi0 = hfile['phi0'][()]
    offsets = hfile['offsets'][()]
    stride = hfile['stride'][()]
    weight = hfile['weight'][()]

    return MapInfo(theta=theta, nphi=nphi,
                   phi0=phi0, offsets=offsets,
                   stride=stride, weight=weight)

def write_map(fname, imap, minfo, symm_axes=None):
    '''
    Write map and accompanying map info object to an hdf file.

    Parameters
    ----------
    fname : str
        Path to file.  Will append .hdf5 if no file extension is found.
    imap : (..., npix) array
        Map to be stored.
    minfo : map_utils.MapInfo object
        Map info object to be stored.
    symm_axes : array-like, optional
        Map is symmetric in these adjacent axes, only store upper 
        triangular part. If sequence of sequences, first flatten 
        axes to get to symmetric matrix, see `mat_utils.flattened_view`.
    '''

    if not os.path.splitext(fname)[1]:
        fname = fname + '.hdf5'

    with h5py.File(fname, 'w') as hfile:
        
        append_map_to_hdf(hfile, imap, minfo, symm_axes=symm_axes)

def append_map_to_hdf(hfile, imap, minfo, symm_axes=None):
    '''
    Add map dataset to provided hdf file.

    Parameters
    ----------
    hfile : HDF5 file or group
        Writeable hdf file or group.
    imap : (..., npix) array
        Map to be stored.        
    minfo : map_utils.MapInfo object
        Map info object to be stored.
    symm_axes : array-like, or array-like of array-like, optional
        Map is symmetric in these adjacent axes, only store upper 
        triangular part. If sequence of sequences, first flatten 
        axes to get to symmetric matrix, see `mat_utils.flattened_view`.
    '''

    if symm_axes is not None:

        shape_orig = imap.shape

        if type_utils.is_seq_of_seq(symm_axes):
            imap, flat_axes = mat_utils.flattened_view(
                imap, symm_axes, return_flat_axes=True)
            symm_axes = flat_axes

        imap = mat_utils.symm2triu(imap, axes=symm_axes)
        hfile.attrs['symm_axis'] = symm_axes[0]
        hfile.attrs['shape_orig'] = shape_orig

    hfile.create_dataset('map', data=imap)

    minfo_group = hfile.create_group('minfo')
    append_minfo_to_hdf(minfo_group, minfo)        

def read_map(fname):
    '''
    Read map and map info object from hdf file.

    Parameters
    ----------
    fname : str
        Path to file.

    Returns
    -------
    omap : (..., npix) array
        Map array.        
    minfo : map_utils.MapInfo object
        Map info object.
    '''

    with h5py.File(fname, 'r') as hfile:
        
        omap, minfo = map_from_hdf(hfile)

    return omap, minfo

def map_from_hdf(hfile):
    '''
    Extract map array from hdf datasets. If needed, expands 
    upper-triangular elemets to full symmetric matrix.

    Parameters
    ----------
    hfile : HDF5 file or group
        Hdf file or group.
    
    Returns
    -------
    omap : (..., npix) array
        Map array.        
    minfo : map_utils.MapInfo object
        Map info object.
    '''
    
    omap = hfile['map'][()]

    symm_axis = hfile.attrs.get('symm_axis', None)
    if symm_axis is not None:
        omap = mat_utils.triu2symm(omap, symm_axis)

    shape_orig = hfile.attrs.get('shape_orig', None)
    if shape_orig is not None:
        omap = omap.reshape(shape_orig)

    minfo = minfo_from_hdf(hfile['minfo'])

    return omap, minfo
