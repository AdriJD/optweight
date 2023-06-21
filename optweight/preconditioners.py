import numpy as np

from pixell import utils, sharp

from optweight import (operators, mat_utils, multigrid, type_utils, sht,
                       alm_utils, map_utils, alm_c_utils, dft)

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
    icov_noise_ell : (npol, npol, nell) or (npol, nell) array, optional
        Inverse noise covariance (with fsky correction). If diagonal, only the 
        diagonal suffices. Used for iN_ell^0.5 iN_wav iN_ell^0.5 noise model. 
        If provided, `mask_pix` will be ignored, w_ell needs to be given.
    sfilt : (npol, nell) or (npol, npol, nell) array, optional
        Symmetric positive definite scaling matrix, if diagonal only the diagonal
        suffices.    

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.

    Notes
    -----
    This class is starting to become a bit complicated. There are four modes:
    1) itau is provided
    2) itau not provided, but icov_pix + minfo + (optionally mask_pix) are
    3) itau not provided, but icov_wav + w_ell + (optionally mask_pix and minfo) are
    4) itau not provided, but icov_noise_ell + w_ell are.
    '''

    def __init__(self, ainfo, icov_ell, itau=None, icov_pix=None, minfo=None, 
                 icov_wav=None, w_ell=None, mask_pix=None, b_ell=None,
                 icov_noise_ell=None, sfilt=None):

        npol, nell = icov_ell.shape[-2:]

        if itau is None:

            if icov_pix is None and icov_wav is None and icov_noise_ell is None:
                raise ValueError('provide atleast itau or icov_{pix/wav/noise_ell}')

            if icov_wav is not None:
                assert icov_pix is None, 'icov_wav given, icov_pix is not None'
                assert icov_noise_ell is None, 'icov_wav given, icov_noise_ell is not None'
                itau = map_utils.get_ivar_ell(icov_wav, w_ell, mask=mask_pix, 
                                              minfo_mask=minfo)
            if icov_pix is not None:
                assert icov_wav is None, 'icov_pix given, icov_wav is not None'
                assert icov_noise_ell is None, 'icov_pix given, icov_noise_ell is not None'
                itau = map_utils.get_isotropic_ivar(icov_pix, minfo, mask=mask_pix)
            if icov_noise_ell is not None:
                assert icov_wav is None, 'icov_noise_ell given, icov_wav is not None'
                assert icov_pix is None, 'icov_noise_ell given, icov_pix is not None'
                itau = get_itau_ell_harm(icov_noise_ell, w_ell)

        elif icov_pix is not None or icov_wav is not None or icov_noise_ell is not None:
            raise ValueError('itau is provided but icov_{pix/wav/noise_ell} are not None.')

        if itau.ndim == 1:
            itau = itau * np.eye(npol)

        if itau.ndim == 2:
            itau = itau[:,:,np.newaxis]

        if b_ell is None:
            b_ell = np.ones((npol, nell))

        b_ell = b_ell * np.eye(npol)[:,:,np.newaxis]

        if sfilt is not None:
            sfilt = mat_utils.full_matrix(sfilt)
            op = np.einsum('ijl, jkl, kol -> iol', sfilt, icov_ell, sfilt)
            op2 = np.einsum('ijl, jkl, kol -> iol', b_ell, itau, b_ell)
            op2 = np.einsum('ijl, jkl, kol -> iol', sfilt, op2, sfilt)
            op += op2
        else:            
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
    sfilt : (npol, nell) or (npol, npol, nell) array, optional
        Symmetric positive definite scaling matrix, if diagonal only the diagonal
        suffices.    

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''

    def __init__(self, ainfo, icov_ell, icov_pix, minfo, spin,
                 itau=None, b_ell=None, cov_pix=None, mask_pix=None,
                 sfilt=None):

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

        if sfilt is not None:
            sftil = mat_utils.full_matrix(sfilt)

        self.harmonic_prec = HarmonicPreconditioner(ainfo, icov_ell,
                                    itau=itau, b_ell=b_ell, sfilt=sfilt)
        
        if sfilt is not None:
            self.icov_signal = operators.EllMatVecAlm(
                ainfo,
                np.einsum('ijl, jkl, kol -> iol', sfilt, icov_ell, sfilt),
                inplace=True)
        else:
            self.icov_signal = operators.EllMatVecAlm(
                ainfo, icov_ell, inplace=True)

        if sfilt is not None:
            self.ivar_noise_iso = operators.EllMatVecAlm(
                ainfo,
                np.einsum('ijl, jkl, kol -> iol', itau, b_ell, sfilt),
                inplace=True)
            self.ivar_noise_iso_adj = operators.EllMatVecAlm(
                ainfo,
                np.einsum('ijl, jkl, kol -> iol', sfilt, b_ell, itau),
                inplace=True)
        else:
            self.ivar_noise_iso = operators.EllMatVecAlm(
                ainfo, itau * b_ell, inplace=True)
            self.ivar_noise_iso_adj = self.ivar_noise_iso
            
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
        alm_noise = self.ivar_noise_iso_adj(alm_noise)

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
        Isotropic noise (co)variance. Inferred from icov_wav or icov_noise_ell 
        if not provided.
    b_ell : (npol, nell) array, optional
        Beam window function.
    icov_noise_ell : (npol, npol, nell) or (npol, nell) array, optional
        Inverse noise covariance (with fsky correction). If diagonal, only the 
        diagonal suffices. Will update noise model to iN_ell^0.5 iN_wav iN_ell^0.5,
        so make sure iN_wav corresponds to the inverse noise covariance after
        flattening by iN_ell^0.5.
    sfilt : (npol, nell) or (npol, npol, nell) array, optional
        Symmetric positive definite scaling matrix, if diagonal only the diagonal
        suffices.    

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''
    
    def __init__(self, ainfo, icov_ell, icov_wav, w_ell, spin,
                 mask_pix=None, minfo_mask=None, itau_ell=None,
                 b_ell=None, icov_noise_ell=None, sfilt=None):

        npol, nell = icov_ell.shape[-2:]

        if itau_ell is None:

            if icov_noise_ell is None:
                itau_ell = map_utils.get_ivar_ell(icov_wav, w_ell, mask=mask_pix, 
                                                  minfo_mask=minfo_mask)
            if icov_noise_ell is not None:
                itau_ell = get_itau_ell_harm(icov_noise_ell, w_ell)

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

        if sfilt is not None:
            sftil = mat_utils.full_matrix(sfilt)

        self.harmonic_prec = HarmonicPreconditioner(ainfo, icov_ell,
                                    itau=itau_ell, b_ell=b_ell, sfilt=sfilt)
        
        if sfilt is not None:
            self.icov_signal = operators.EllMatVecAlm(
                ainfo,
                np.einsum('ijl, jkl, kol -> iol', sfilt, icov_ell, sfilt),
                inplace=True)
        else:
            self.icov_signal = operators.EllMatVecAlm(
                ainfo, icov_ell, inplace=True)

        self.beam = operators.EllMatVecAlm(
            ainfo, b_ell, inplace=True)

        if sfilt is not None:
            self.sfilt = operators.EllMatVecAlm(
            ainfo, sfilt, inplace=True)
        else:
            sfilt = lambda x: x

        self.ivar_noise_iso = operators.EllMatVecAlm(
            ainfo, itau_ell, inplace=True)

        if icov_noise_ell is None:
            self.pcov_noise = operators.WavMatVecAlm(
                ainfo, icov_wav, w_ell, spin, power=-1, adjoint=True)
        else:
            self.pcov_noise = operators.EllWavEllMatVecAlm(
                ainfo, icov_wav, w_ell, icov_noise_ell, spin, power_m=-1,
                power_x=-0.5, adjoint=True)

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
        alm_noise = self.sfilt(alm_noise)
        alm_noise = self.beam(alm_noise)
        alm_noise = self.imask(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.pcov_noise(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.imask(alm_noise)
        alm_noise = self.beam(alm_noise)
        alm_noise = self.sfilt(alm_noise)

        alm = alm_noise
        alm += alm_signal

        alm = self.harmonic_prec(alm)

        return alm

class PseudoInvPreconditionerFWav(operators.MatVecAlm):
    '''
    Adaptation of the pseudo-inverse preconditioner from Seljebotn et al.,
    2017 (1710.00621) to a 2D Fourier wavelet-based noise model.

    Parameters
    ----------
    ainfo : sharp.alm_info object
        Metainfo for input alms.
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse signal covariance, If diagonal, only the diagonal suffices.
    cov_wav : wavtrans.Wav object
        Wavelet block matrix representing the noise covariance.
    fkernelset : fkernel.FKernelSet object
        Wavelet kernels.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    itau_ell : (npol, npol, nell) array
        Isotropic noise (co)variance. 
    cov_noise_2d : (npol, npol, nly, nlx) array or (npol, nly, nlx) array
        Noise covariance in 2D Fourier domain. If diagonal, only the 
        diagonal suffices.
    mask_pix = (npol, npix) array
        Pixel mask.
    minfo_mask : sharp.map_info object
        Metainfo for pixel mask covariance.
    b_ell : (npol, nell) array, optional
        Beam window function.
    sfilt : (npol, nell) or (npol, npol, nell) array, optional
        Symmetric positive definite scaling matrix, if diagonal only the diagonal
        suffices.    

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''
    
    def __init__(self, ainfo, icov_ell, cov_wav, fkernelset, spin,
                 itau_ell, cov_noise_2d, mask_pix, minfo_mask,
                 b_ell=None, sfilt=None):

        self.npol, self.nell = icov_ell.shape[-2:]

        if itau_ell.ndim != 3:
            raise ValueError(
                'Wrong dimensions itau_ell : expected 3, got {}'.
                format(itau_ell.ndim))

        if b_ell is None:
            b_ell = np.ones((self.npol, self.nell))
        b_ell = b_ell * np.eye(self.npol)[:,:,np.newaxis]

        self.ainfo = ainfo
        self.mask_pix = mask_pix
        self.minfo_mask = minfo_mask
        self.spin = spin
        
        if sfilt is not None:
            sftil = mat_utils.full_matrix(sfilt)

        self.harmonic_prec = HarmonicPreconditioner(ainfo, icov_ell,
                                    itau=itau_ell, b_ell=b_ell, sfilt=sfilt)

        if sfilt is not None:
            self.icov_signal = operators.EllMatVecAlm(
                ainfo,
                np.einsum('ijl, jkl, kol -> iol', sfilt, icov_ell, sfilt),
                inplace=True)
        else:
            self.icov_signal = operators.EllMatVecAlm(
                ainfo, icov_ell, inplace=True)

        self.beam = operators.EllMatVecAlm(
            ainfo, b_ell, inplace=True)
        if sfilt is not None:
            self.sfilt = operators.EllMatVecAlm(
            ainfo, sfilt, inplace=True)
        else:
            sfilt = lambda x: x

        self.ivar_noise_iso = operators.EllMatVecAlm(
            ainfo, itau_ell, inplace=True)
        self.cov_wav_op = operators.FWavMatVecF(cov_wav, fkernelset)
        self.x_op = operators.FMatVecF(cov_noise_2d, power=0.5, inplace=True)

    def pcov_noise(self, alm):
        '''
        Apply the pseudo-inverse noise covariance to a set of alm inplace.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Input alms.

        Returns
        -------
        out : (npol, nelem) complex array
            Input vector modified inplace.
        '''

        map_tmp = np.zeros(
            (self.npol, self.minfo_mask.npix), type_utils.to_real(alm.dtype))
        sht.alm2map(
            alm, map_tmp, self.ainfo, self.minfo_mask, self.spin, adjoint=True)

        map_tmp *= self.mask_pix

        map_tmp = map_utils.view_2d(map_tmp, self.minfo_mask)
        fmap = dft.allocate_fmap(map_tmp.shape, map_tmp.dtype)
        dft.rfft(map_tmp, fmap)
        self.x_op(fmap)
        self.cov_wav_op(fmap)
        self.x_op(fmap)
        dft.irfft(fmap, map_tmp)
        map_tmp = map_utils.view_1d(map_tmp, self.minfo_mask)

        map_tmp *= self.mask_pix

        sht.map2alm(
            map_tmp, alm, self.minfo_mask, self.ainfo, self.spin, adjoint=True)

        return alm

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
        alm_noise = self.sfilt(alm_noise)
        alm_noise = self.beam(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.pcov_noise(alm_noise)
        alm_noise = self.ivar_noise_iso(alm_noise)
        alm_noise = self.beam(alm_noise)
        alm_noise = self.sfilt(alm_noise)

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
    sfilt : (npol, nell) or (npol, npol, nell) array, optional
        Symmetric positive definite scaling matrix, if diagonal only the diagonal
        suffices.    

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.
    '''
    
    def __init__(self, ainfo, icov_ell, spin, mask_bool, minfo, min_pix=1000,
                 n_jacobi=1, lmax_r_ell=6000, nsteps=1, sfilt=None):

        if icov_ell.ndim == 1:
            icov_ell = icov_ell[np.newaxis,:]

        if sfilt is not None:
            sfilt = mat_utils.full_matrix(sfilt)
            icov_ell = mat_utils.full_matrix(icov_ell)
            icov_ell = np.einsum('ijl, jkl, kol -> iol', sfilt, icov_ell, sfilt)

        self.npol = icov_ell.shape[0]        
        self.spin = spin
        self.ainfo = ainfo
        self.n_jacobi = n_jacobi
        self.lmax = self.ainfo.lmax
        self.nsteps = nsteps
        self.levels = multigrid.get_levels(mask_bool, minfo, icov_ell,
                                           self.spin, min_pix=min_pix,
                                           lmax_r_ell=lmax_r_ell)
        self.r_ell = multigrid.lowpass_filter(lmax_r_ell, lmax=self.lmax)
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
    sfilt : (npol, nell) or (npol, npol, nell) array, optional
        Symmetric positive definite scaling matrix, if diagonal only the diagonal
        suffices.    

    Methods
    -------
    call(alm) : Apply the preconditioner to a set of alms.

    Notes
    -----
    Useful for LCDM polarization, for which the multigrid preconditioner works
    less well because icov_ell does not resemble ell^2 (as it does for TT).
    '''
    
    def __init__(self, ainfo, icov_ell, spin, mask_bool, minfo, lmax_r_ell=None, 
                 lmax=None, nsteps=15, verbose=False, sfilt=None):

        self.spin = spin
        self.npol = icov_ell.shape[0]        
        self.ainfo = ainfo
        self.nsteps = nsteps
        self.verbose = verbose

        if sfilt is not None:
            sfilt = mat_utils.full_matrix(sfilt)
            icov_ell = mat_utils.full_matrix(icov_ell)
            icov_ell = np.einsum('ijl, jkl, kol -> iol', sfilt, icov_ell, sfilt)

        if lmax is not None:
            if lmax > self.ainfo.lmax:    
                lmax = self.ainfo.lmax
            icov_ell = icov_ell[...,:lmax+1].copy()
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

def get_itau_ell_harm(icov_ell, w_ell):
    '''
    Compute inverse variance per wavelet band.

    Parameters
    ----------
    icov_ell : (npol, npol, nell) array or (npol, nell) array
        Inverse covariance, If diagonal, only the diagonal suffices.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    
    Returns
    -------
    itau_ell : (npol, npol, nell) array
        Inverse variance spectrum. Note that only the diagonal 
        is calculated.
    '''

    icov_ell = mat_utils.full_matrix(mat_utils.atleast_nd(icov_ell, 2))    
    itau_ell = np.zeros_like(icov_ell)
    npol = itau_ell.shape[0]

    for widx in range(w_ell.shape[0]):
        
        w_ell_sq = w_ell[widx] ** 2
        sum_w_ell_sq = np.sum(w_ell_sq)
        
        for pidx in range(npol):

            a_j = np.sum(icov_ell[pidx,pidx] * w_ell_sq) / sum_w_ell_sq
            itau_ell[pidx,pidx] += a_j * w_ell_sq

    return itau_ell
