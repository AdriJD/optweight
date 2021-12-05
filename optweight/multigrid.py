'''
Multigrid solver meant as preconditioner for the masked pixels, heavily inspired by 
https://arxiv.org/pdf/1710.00621.pdf and https://github.com/dagss/cmbcr.
'''
import numpy as np

from pixell import sharp

from optweight import (map_utils, operators, noise_utils, alm_utils, alm_c_utils,
                       type_utils, sht)

class Level():
    '''
    A level represents the linear system given by M Y d_ell Yt M x = b,
    where M is a boolean mask that is True for unobserved pixels and
    d_ell is a highpasss filtered version of icov_ell.

    Parameters
    ----------
    mask : (npol, npix) or (npix) bool array
        Mask, True for observed pixels.
    minfo : sharp.map_info object
        Meta info mask.
    d_ell : (npol, npol, nell) array
        Lowpass filtered inverse signal power spectrum.
    dense_smoothing : bool, optional
        If set, initialize the dense version of the smoother, i.e. the
        pseudo-inverse of M Y d_ell Yt M.

    Attributes
    ----------
    mask_unobs : (npol, npix) bool array
        Mask, False for observed pixels.
    minfo : sharp.map_info object
        Meta info mask.
    npol : int
        Number of polarizations
    lmax : int
        Multipole supported by grid described by minfo.
    d_ell :  (npol, npol, nell) array
        Lowpass filtered inverse signal power spectrum.
    g_op : callable
        The G = M Y d_ell Yt M function, taking and returning (npol, npix) maps.
    '''

    def __init__(self, mask, minfo, d_ell, dense_smoothing=False):

        assert mask.dtype == bool, f'mask must be bool, got {mask.dtype}'

        if mask.ndim == 1:
            mask = mask[np.newaxis,:]
        self.mask_unobs = ~mask
        self.minfo = minfo

        assert d_ell.shape == (self.npol, self.npol, self.lmax + 1), (
            f'Invalid shape d_ell expected {(self.npol,self.npol,self.lmax+1)} '
            f'got {d_ell.shape}')
        self.d_ell = d_ell

        self.g_op = operators.PixEllPixMatVecMap(
            self.mask_unobs, self.d_ell, self.minfo, [0, 2])

        if dense_smoothing:
            self.smoother, self.pinv_g = self._init_dense_smoother()
        else:
            self.smoother, self.pinv_g = self._init_sparse_smoother(), None

    @property
    def lmax(self):
        return map_utils.minfo2lmax(self.minfo)

    @property
    def npol(self):
        return self.mask_unobs.shape[0]

    def _init_sparse_smoother(self):
        '''
        Initialize the omega diag(Gh)^-1 solver, i.e. an approximate solver
        that should be used for every level except the most coarse level.

        Returns
        -------
        smoother : callable
            Function that takes and returns (npol, npix) map.
        '''
        m_prec = np.zeros(self.npol)

        # Sample 1 pixel in I and 1 pixel in Q and U to get diagonal.
        # Note that Q and U should give the same value.
        test_map = np.zeros((self.npol, self.minfo.npix), dtype=np.float32)

        # Find pixel in unobserved part. Doesn't matter which one.
        sample_idx = np.argmax(self.mask_unobs, axis=-1)

        for pidx in range(self.npol):

            test_map[pidx,sample_idx[pidx]] = 1.
            omap = self.g_op(test_map)
            m_prec[pidx] = 1 / omap[pidx,sample_idx[pidx]]
            test_map[pidx,sample_idx[pidx]] = 0

        # Multiply by damping factor.
        m_prec *= 0.2

        return lambda x: m_prec[:,np.newaxis] * x

    def _init_dense_smoother(self):
        '''
        Initialize the exact solver by computing the (pseudo) inverse of the
        G matrix. Should be used the most coarse level.

        Returns
        -------
        smoother : callable
            Function that takes and returns (npol, npix) map.
        '''

        mat = operators.op2mat(
            self._g_op_unobs, np.sum(self.mask_unobs), np.float64)
        pinv_mat = np.linalg.pinv(mat)

        def smoother(imap):
            omap = np.zeros(imap.shape, imap.dtype)
            omap[self.mask_unobs] = np.dot(pinv_mat, imap[self.mask_unobs])
            return omap

        return smoother, pinv_mat

    def _g_op_unobs(self, imap_unobs):
        '''
        Wrapper around g_op for input that only consists of the
        masked pixels.

        Parameters
        ----------
        imap_unobs : (n_unobs) array
            1D array with all imap values as unobserved pixels.

        Returns
        -------
        omap_unobs : (n_unobs) array
            1D array after applying G.
        '''

        imap = np.zeros(self.mask_unobs.shape, dtype=imap_unobs.dtype)
        imap[self.mask_unobs] = imap_unobs
        omap = self.g_op(imap)
        return omap[self.mask_unobs]

def get_levels(mask, minfo, icov_ell, min_pix=1000):
    '''
    Create a list of level objects each half of the previous level's lmax.

    Parameters
    ----------
    mask : (npol, npix) or (npix) bool array
        Mask, True for observed pixels.
    minfo : sharp.map_info object
        Meta info mask.
    icov_ell : (npol, npol, nell) or (npol, nell) array
        Inverse signal power spectrum.
    min_pix : int, optional
        Once this number of masked pixels is reached or exceeded in any
        of the `npol` masks, stop making levels.

    Returns
    -------
    levels : list of multigrid.Level objects
        Levels going from input resolution to resolution at which
        the minimum of masked pixels is reached.

    Raises
    ------
    ValueError
        If mask is not bool type.
        If shape icov_ell is not (npol, npol, nell) or (npol, nell)
    '''

    if icov_ell.ndim == 2:
        npol, nell = icov_ell.shape
        # Upgrade to dense matrix.
        icov_ell = icov_ell * np.eye(npol)[:,:,np.newaxis]
    elif icov_ell.ndim == 3:
        npol, _, nell = icov_ell.shape
    else:
        raise ValueError(f'shape icov_ell : {icov_ell.shape} not supported')

    if mask.ndim == 1:
        mask = np.ones((npol, mask.size), dtype=mask.dtype) * mask
    else:
        assert mask.shape[0] == npol, (
            f'shape[0] mask does not match npol {mask.shape[0]} != {npol}')

    if mask.dtype != bool:
        raise ValueError(f'Input mask must be bool, got {mask.dtype}')

    lmax = map_utils.minfo2lmax(minfo)
    levels = []
    last_level = False
    for idx in range(16):

        downgrade = 2 ** idx
        lmax_level = lmax // downgrade

        # We always want full-sky minfos, we're interested in masked regions.
        minfo_level = map_utils.get_equal_area_gauss_minfo(
            2 * lmax_level, ratio_pow=1, gl_band=0)

        mask_level = map_utils.gauss2map(
            mask.astype(np.float32), minfo, minfo_level, order=1)
        # Mask includes pixels made out of 100% masked pixels in fine grid.
        mask_level[(mask_level > 0) & (mask_level < 1)] = 1.
        mask_level = mask_level.astype(bool)

        assert lmax_level <= nell - 1, (
            f'lmax_level {lmax_level} exceeds lmax of icov_ell {nell - 1}')

        nmasked_per_pol = np.sum(~mask_level, axis=1)
        if np.any(nmasked_per_pol < min_pix):
            last_level = True

        if np.any(nmasked_per_pol == 0):
            raise ValueError(f'No masked pixels left for min_pix : {min_pix} '
                             f'in one or more of the masks : {nmasked_per_pol}')

        if idx == 0:
            r_ell = lowpass_filter(lmax_level)
            d_ell = icov_ell[:,:,:lmax_level+1] * r_ell ** 2
        else:
            r_ell = noise_utils._band_limit_gauss_beam(
                lmax_level, fwhm=2 * np.pi / lmax_level)
            d_ell = levels[idx - 1].d_ell[:,:,:lmax_level+1] * r_ell ** 2

        levels.append(Level(mask_level, minfo_level, d_ell,
                            dense_smoothing=last_level))
        if last_level:
            break

    return levels

def restrict(imap, level_in, level_out, adjoint=False):
    '''
    Interpolate/restrict map from one grid to another.

    Parameters
    ----------
    imap : (npol, npix_in) array
        Input map with npix matching the geometry of `level_in`.
    level_in : multigrid.Level instance
        Input level.
    level_out : multigrid.Level instance
        Output level.
    adjoint : bool, optional
        If set, perform adjoint operation, i.e. W Y R Yt instead of Y R Yt W.

    Returns
    -------
    omap : (npol, npix_out) array
        Output map with npix matching the geometry of `level_out`.
    '''

    minfo_in = level_in.minfo
    minfo_out = level_out.minfo

    lmax_in = map_utils.minfo2lmax(minfo_in)
    lmax_out = map_utils.minfo2lmax(minfo_out)

    ainfo_in = sharp.alm_info(lmax_in)
    ainfo_out = sharp.alm_info(lmax_out)

    omap = np.zeros((imap.shape[0], minfo_out.npix), dtype=imap.dtype)
    imap = imap * level_in.mask_unobs

    if lmax_out < lmax_in:
        alm = np.zeros((imap.shape[0], ainfo_out.nelem),
                       dtype=type_utils.to_complex(imap.dtype))
        sht.map2alm(imap, alm, level_in.minfo, ainfo_out, [0, 2], adjoint=adjoint)
    else:
        alm = np.zeros((imap.shape[0], ainfo_in.nelem),
                       dtype=type_utils.to_complex(imap.dtype))
        sht.map2alm(imap, alm, level_in.minfo, ainfo_in, [0, 2], adjoint=adjoint)
        ainfo_out = ainfo_in

    if not adjoint:
       r_ell = noise_utils._band_limit_gauss_beam(
           lmax_out, fwhm=2 * np.pi / lmax_out)
    else:
       r_ell = noise_utils._band_limit_gauss_beam(
           lmax_in, fwhm=2 * np.pi / lmax_in)

    alm_c_utils.lmul(alm, r_ell, ainfo_out, inplace=True)
    sht.alm2map(alm, omap, ainfo_out, minfo_out, [0,2], adjoint=adjoint)
    omap *= level_out.mask_unobs

    return omap

def coarsen(imap, level_in, level_out):
    '''
    Restrict a map from a high resolution level to low resolution level.

    Parameters
    ----------
    imap : (npol, npix_in) array
        Input map with npix matching the geometry of `level_in`.
    level_in : multigrid.Level instance
        Input level.
    level_out : multigrid.Level instance
        Output level.
    
    Returns
    -------
    omap : (npol, npix_out) array
        Output map with npix matching the geometry of `level_out`.
    '''

    return restrict(imap, level_in, level_out, adjoint=False)

def interpolate(imap, level_in, level_out):
    '''
    Interpolate a map from a low resolution level to a high resolution level.

    Parameters
    ----------
    imap : (npol, npix_in) array
        Input map with npix matching the geometry of `level_in`.
    level_in : multigrid.Level instance
        Input level.
    level_out : multigrid.Level instance
        Output level.
    
    Returns
    -------
    omap : (npol, npix_out) array
        Output map with npix matching the geometry of `level_out`.
    '''

    return restrict(imap, level_in, level_out, adjoint=True)

def v_cycle(levels, imap, idx=0, n_jacobi=1):
    '''
    The solver, finds an approximation of G^-1 applied to the input map.

    Parameters
    ----------
    levels : list of multigrid.Level instances
        See `get_levels`.
    imap : (npol, npix) array
        Input map.
    idx : int, optional
        Current level index.
    n_jacobi : int, optional
        Number of Jacobi iterations for the diagonal smoothers.

    Returns
    -------
    omap : (npol, npix) array
        Output map that is approximately G^-1 imap.
    '''
    
    if idx == len(levels) - 1:
        return levels[idx].smoother(imap)

    else:
        x_vec = np.zeros_like(imap)

        x_vec += levels[idx].smoother(imap)
        for _ in range(n_jacobi - 1):
            x_vec += levels[idx].smoother(imap - levels[idx].g_op(x_vec))

        r_h = imap - levels[idx].g_op(x_vec)

        r_H = coarsen(r_h, levels[idx], levels[idx+1])

        c_H = v_cycle(levels, r_H, idx=idx + 1, n_jacobi=n_jacobi)

        c_h = interpolate(c_H, levels[idx + 1], levels[idx])

        x_vec += c_h

        for _ in range(n_jacobi):
            x_vec += levels[idx].smoother(imap - levels[idx].g_op(x_vec))

        return x_vec

def lowpass_filter(lmax):
    '''
    Return lowpass filter r_ell.

    Parameters
    ----------
    lmax : int
        Bandlimit of filter, filter is defined such that r_(lmax / 2) = sqrt(0.05).

    Returns
    -------
    r_ell : (nell) array
        Lowpass filter.
    '''

    beta = - np.log(np.sqrt(0.05)) / ((lmax / 2) * (lmax / 2 + 1)) ** 2
    ells = np.arange(lmax + 1)
    return np.exp(-beta * (ells * (ells + 1)) ** 2)
