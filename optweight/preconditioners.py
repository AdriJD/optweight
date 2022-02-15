import numpy as np

from pixell import utils, sharp

from optweight import (operators, mat_utils, multigrid, type_utils, sht,
                       alm_utils, map_utils, alm_c_utils)

class HarmonicPreconditioner(operators.MatVecAlm):
    '''
    Harmonic preconditioner: M = (C^-1 + B itau B)^-1.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse covariance matrix, either symmetric but dense in first two axes
        or diagonal, in which case only the diagonal elements are needed.
    itau : (npol, npol) or (npol, npol, nell) array, optional
        Isotropic noise (co)variance.
    icov_pix : (npol, npol, npix) or (npol, npix) array, optional
        Inverse noise covariance. If diagonal, only the diagonal suffices. Only
        needed when `itau` or `icov_wav` are not given.
    minfo : sharp.map_info object, optional
        Metainfo for inverse noise covariance. Needed if `icov_pix` and/or 
        mask_pix are provided.
    icov_wav : wavtrans.Wav object, optional
        Wavelet block matrix representing the inverse noise covariance. Only
        needed when `itau` or `icov_pix` are not given.
    w_ell : (nwav, nell) array, optional
        Wavelet kernels. Needed when `icov_wav` is given.
    mask_pix : (npol, npix) array, optional
        Pixel mask used in `itau` computation. Needed when the provided 
        icov_pix/wav are not masked like the data are. If provided, also 
        requires minfo.
    b_ell : (npol, nell) array, optional
        Beam window function.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''

    def __init__(self, ainfo, icov_ell, itau=None, icov_pix=None, minfo=None, 
                 icov_wav=None, w_ell=None, mask_pix=None, b_ell=None):

        npol, nell = icov_ell.shape[-2:]

        if itau is None:
            if icov_pix is None and icov_wav is None:
                raise ValueError('itau, icov_pix and icov_wav cannot all be None')                
            if icov_wav is None:
                itau = map_utils.get_isotropic_ivar(icov_pix, minfo, mask=mask_pix)
            elif icov_pix is None:
                itau = map_utils.get_ivar_ell(icov_wav, w_ell, mask=mask_pix, 
                                              minfo_mask=minfo)
            else:
                raise ValueError('icov_pix and icov_wav cannot both be given.')
        elif icov_pix is not None or icov_wav is not None:
            raise ValueError('if itau is provided, icov_pix/wav should be None.')

        if itau.ndim == 1:
            itau = itau * np.eye(npol)

        if itau.ndim == 2:
            itau = itau[:,:,np.newaxis]

        if b_ell is None:
            b_ell = np.ones((npol, nell))

        b_ell = b_ell * np.eye(npol)[:,:,np.newaxis]
        op = icov_ell + np.einsum('ijl, jkl, kol -> iol', b_ell, itau, b_ell)        

        self.preconditioner = operators.EllMatVecAlm(ainfo, op, -1, inplace=False)
            
    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        return self.preconditioner(alm)

class PseudoInvPreconditioner(operators.MatVecAlm):
    '''
    Implementation of the pseudo-inverse preconditioner from Seljebotn et al.,
    2017 (1710.00621).

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse signal covariance, If diagonal, only the diagonal suffices.
    icov_pix : (npol, npol, npix) or (npol, npix) array
        Inverse noise covariance. If diagonal, only the diagonal suffices.
    minfo : sharp.map_info object
        Metainfo for inverse noise covariance.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    itau : (npol, npol) or (npol) array or float, optional
        Isotropic noise (co)variance. Inferred from `icov_pix` if not provided.
    b_ell : (npol, nell) array, optional
        Beam window function.
    cov_pix : (npol, npol, npix) or (npol, npix) array, optional
        Noise covariance. If diagonal, only the diagonal suffices.
    mask_pix : (npol, pix) array, optional
        Pixel mask used in `itau` computation. Needed when the provided 
        icov_pix is not masked like the data are.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''

    def __init__(self, ainfo, icov_ell, icov_pix, minfo, spin,
                 itau=None, b_ell=None, cov_pix=None):

        npol, nell = icov_ell.shape[-2:]

        if itau is None:
            itau = map_utils.get_isotropic_ivar(icov_pix, minfo, mask=mask_pix)

        if itau.ndim == 2:
            itau = itau[:,:,np.newaxis]

        if np.count_nonzero(itau - itau * np.eye(npol)[:,:,np.newaxis]):
            raise NotImplementedError(
                'Pinv precondition cannot handle off-diagonal itau elements for now.')

        if b_ell is None:
            b_ell = np.ones((npol, nell))

        b_ell = b_ell * np.eye(npol)[:,:,np.newaxis]

        if cov_pix is None:
            cov_pix = mat_utils.matpow(icov_pix, -1)

        op = icov_ell + np.einsum('ijl, jkl, kol -> iol', b_ell, itau, b_ell)        
        self.harmonic_prec = operators.EllMatVecAlm(
            ainfo, op, -1, inplace=True)
        
        self.icov_signal = operators.EllMatVecAlm(
            ainfo, icov_ell, inplace=True)

        self.ivar_noise_iso = operators.EllMatVecAlm(
            ainfo, itau * b_ell, inplace=True)

        self.pcov_noise = operators.PixMatVecAlm(
            ainfo, cov_pix, minfo, spin, inplace=True, adjoint=True)

    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        alm = alm.copy()

        alm = self.harmonic_prec(alm)

        alm_signal = alm.copy()
        alm_noise = alm

        alm_signal = self.icov_signal(alm_signal)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.pcov_noise(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)

        alm = alm_noise
        alm += alm_signal

        alm = self.harmonic_prec(alm)

        return alm

class PseudoInvPreconditionerWav(operators.MatVecAlm):
    '''
    Adaptation of the pseudo-inverse preconditioner from Seljebotn et al.,
    2017 (1710.00621) to a wavelet-based noise model.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse signal covariance, If diagonal, only the diagonal suffices.
    icov_wav : wavtrans.Wav object
        Wavelet block matrix representing the inverse noise covariance.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    mask_pix = (npol, npix) array, optional
        Pixel mask.
    minfo_mask : sharp.map_info object, optional
        Metainfo for pixel mask covariance.
    itau_ell : (npol, npol, nell) array
        Isotropic noise (co)variance. Inferred from icov_wav if not provided.
    b_ell : (npol, nell) array, optional
        Beam window function.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''
    
    def __init__(self, ainfo, icov_ell, icov_wav, w_ell, spin,
                 mask_pix=None, minfo_mask=None, itau_ell=None,
                 b_ell=None):

        npol, nell = icov_ell.shape[-2:]

        if itau_ell is None:
            itau_ell = map_utils.get_ivar_ell(icov_wav, w_ell, mask=mask_pix, 
                                              minfo_mask=minfo_mask)

        if itau_ell.ndim != 3:
            raise ValueError(
                'Wrong dimensions itau_ell : expected 3, got {}'.
                format(itau_ell.ndim))

        if b_ell is None:
            b_ell = np.ones((npol, nell))

        b_ell = b_ell * np.eye(npol)[:,:,np.newaxis]

        if mask_pix is None:
            self.imask = lambda alm: alm
        else:            
            self.imask = operators.PixMatVecAlm(
                ainfo, mask_pix, minfo_mask, spin, power=-1,
                use_weights=True)

        op = icov_ell + np.einsum('ijl, jkl, kol -> iol', b_ell, itau_ell, b_ell)        
        self.harmonic_prec = operators.EllMatVecAlm(ainfo, op, -1, inplace=True)
        
        self.icov_signal = operators.EllMatVecAlm(
            ainfo, icov_ell, inplace=True)

        self.beam = operators.EllMatVecAlm(
            ainfo, b_ell, inplace=True)

        self.ivar_noise_iso = operators.EllMatVecAlm(
            ainfo, itau_ell, inplace=True)

        self.pcov_noise = operators.WavMatVecAlm(
            ainfo, icov_wav, w_ell, spin, power=-1, adjoint=True)

    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''

        alm = alm.copy()

        alm = self.harmonic_prec(alm)

        alm_signal = alm.copy()
        alm_noise = alm

        alm_signal = self.icov_signal(alm_signal)
        alm_noise = self.beam(alm_noise)
        alm_noise = self.imask(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.pcov_noise(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.imask(alm_noise)
        alm_noise = self.beam(alm_noise)

        alm = alm_noise
        alm += alm_signal

        alm = self.harmonic_prec(alm)

        return alm

class MaskedPreconditioner(operators.MatVecAlm):
    '''
    Adaptation of the masked preconditioner from Seljebotn et al.,
    2017 (1710.00621).

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse signal covariance, If diagonal, only the diagonal suffices.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    mask_bool = (npol, npix) array
        Pixel mask, True for observed pixels.
    minfo : sharp.map_info object
        Metainfo for pixel mask covariance.
    min_pix : int, optional
        Once this number of masked pixels is reached or exceeded in any
        of the `npol` masks, stop making levels.
    n_jacobi : int, optional
        Number of Jacobi iterations for the diagonal smoothers.
    lmax_r_ell : int, optional
        Lmax parameter for r_ell filter that is applied to first (finest) 
        level. See `multigrid.lowpass_filter`.
    nsteps : int, optional
        The amount of v-cycles used. Usually requires `n_jacobi` > 1.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''
    
    def __init__(self, ainfo, icov_ell, spin, mask_bool, minfo, min_pix=1000,
                 n_jacobi=1, lmax_r_ell=6000, nsteps=1):

        if icov_ell.ndim == 1:
            icov_ell = icov_ell[np.newaxis,:]

        self.npol = icov_ell.shape[0]        
        self.spin = spin
        self.ainfo = ainfo
        self.n_jacobi = n_jacobi
        self.lmax = self.ainfo.lmax
        self.nsteps = nsteps
        self.levels = multigrid.get_levels(mask_bool, minfo, icov_ell,
                                           self.spin, min_pix=min_pix,
                                           lmax_r_ell=lmax_r_ell)
        self.r_ell = multigrid.lowpass_filter(lmax_r_ell)[:self.lmax+1]
        self.mask_unobs = self.levels[0].mask_unobs
        self.minfo = self.levels[0].minfo

    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''
        
        alm = alm.copy()
        imap = np.zeros((self.npol, self.minfo.npix),
                        dtype=type_utils.to_real(alm.dtype))

        alm_c_utils.lmul(alm, self.r_ell, self.ainfo, inplace=True)
        sht.alm2map(alm, imap, self.ainfo, self.minfo, self.spin)

        imap *= self.mask_unobs
        omap = None
        for _ in range(self.nsteps):            
            omap = multigrid.v_cycle(self.levels, imap, self.spin, 
                                     n_jacobi=self.n_jacobi, x0=omap)
        omap *= self.mask_unobs
        sht.map2alm(omap, alm, self.minfo, self.ainfo, self.spin,
                    adjoint=True)
        alm_c_utils.lmul(alm, self.r_ell, self.ainfo, inplace=True)

        return alm

class MaskedPreconditionerCG(operators.MatVecAlm):
    '''
    Masked preconditioner using conjugate gradient solver. 

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse signal covariance, If diagonal, only the diagonal suffices.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    mask_bool = (npol, npix) array
        Pixel mask, True for observed pixels.
    minfo : sharp.map_info object
        Metainfo for pixel mask covariance.
    lmax_r_ell : int, optional
        Lmax parameter for r_ell filter that is applied to the masked equation 
        system. See `multigrid.lowpass_filter`.
    lmax : int, optional
        Only apply precondtioner to multipoles up to lmax.
    nsteps : int, optional
        Number of CG steps used.
    verbose : bool, optional
        Print info about CG convergence.

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.

    Notes
    -----
    Useful for LCDM polarization, for which the multigrid preconditioner works
    less well because icov_ell does not resemble ell^2 (as it does for TT).
    '''
    
    def __init__(self, ainfo, icov_ell, spin, mask_bool, minfo, lmax_r_ell=None, 
                 lmax=None, nsteps=15, verbose=False):

        self.spin = spin
        self.npol = icov_ell.shape[0]        
        self.ainfo = ainfo
        self.nsteps = nsteps
        self.verbose = verbose

        if lmax is not None:
            icov_ell = icov_ell[...,:lmax+1].copy()
            if lmax > ainfo.lmax:
                raise ValueError(f'lmax {lmax} has be be <= ainfo.lmax {ainfo.lmax}')
        else:
            lmax = self.ainfo.lmax

        self.lmax = lmax
        self.ainfo_lowres = sharp.alm_info(self.lmax)

        if lmax_r_ell is not None:
            self.r_ell = multigrid.lowpass_filter(lmax_r_ell, lmax=ainfo.lmax)
            d_ell = icov_ell * self.r_ell[:self.lmax+1] ** 2
        else:
            self.r_ell = None
            d_ell = icov_ell.copy()

        if mask_bool.ndim == 1:
            mask_bool = np.ones(
                (self.npol, mask_bool.size), dtype=mask_bool.dtype) * mask_bool
        else:
            assert mask_bool.shape[0] == self.npol, (
                f'shape[0] mask != npol {mask_bool.shape[0]} != {self.npol}')

        if mask_bool.dtype != bool:
            raise ValueError(f'Input mask must be bool, got {mask_bool.dtype}')

        mask_bool, minfo = multigrid.get_equal_area_mask_bool(
            mask_bool, minfo, lmax=lmax)
        self.mask_unobs = ~mask_bool
        self.minfo = minfo
        self.g_op = operators.PixEllPixMatVecMap(
            self.mask_unobs, d_ell, self.minfo, self.spin, lmax=self.lmax)
        self.dot = lambda x, y : np.dot(x.reshape(-1), y.reshape(-1))
        self.prec = operators.PixEllPixMatVecMap(
            self.mask_unobs, icov_ell, self.minfo, self.spin, power_x=-1, lmax=self.lmax,
            adjoint=True)

    def call(self, alm):
        '''
        Apply the preconditioner to a set of alms.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Output from matrix-vector operation.
        '''
        
        alm = alm.copy()        
        imap = np.zeros((self.npol, self.minfo.npix),
                        dtype=type_utils.to_real(alm.dtype))

        alm_lowres, _ = alm_utils.trunc_alm(alm, self.ainfo, self.lmax)

        if self.r_ell is not None:
            alm_c_utils.lmul(alm_lowres, self.r_ell[:self.lmax+1],
                             self.ainfo_lowres, inplace=True)

        sht.alm2map(alm_lowres, imap, self.ainfo_lowres, self.minfo, self.spin)
        imap *= self.mask_unobs
        cg = utils.CG(self.g_op, imap, dot=self.dot, M=self.prec)

        for idx in range(self.nsteps):
            cg.step()
            if self.verbose:
                print(idx, cg.err)
        imap = cg.x
        imap *= self.mask_unobs

        sht.map2alm(imap, alm_lowres, self.minfo, self.ainfo_lowres, self.spin,
                    adjoint=True)

        if self.r_ell is not None:
            alm_c_utils.lmul(alm_lowres, self.r_ell[:self.lmax+1],
                             self.ainfo_lowres, inplace=True)

        alm *= 0
        alm_utils.add_to_alm(alm, alm_lowres, self.ainfo, self.ainfo_lowres)

        return alm
