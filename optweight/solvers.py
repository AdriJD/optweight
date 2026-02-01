import numpy as np

from pixell import curvedsky, utils

from optweight import (operators, alm_utils, map_utils, mat_utils,
                       noise_utils, dft, sht as sht_tools)

class CGWienerMap(utils.CG):
    '''
    Given the data model d = P s + n, where s is input signal in SH
    coefficients and the projection matrix P is given by Mf M Y B L,
    construct a CG solver for x in the equation system A x = b where:

        A : H S^-1 H + H Lt B Yt M Mft N^-1 Mf M Y B L H,
        x : H^-1 x^wf, where x^wf is the Wiener filtered version of the input
            map as SH coefficients.
        b : H Lt B Yt Mft M N^-1 d,

    and

        d    : input map,
        L    : Lensing operator,
        Lt   : Adjoint lensing operator,
        B    : Beam convolution operator,
        Y    : Spherical harmonic transformation (alm2map),
        Yt   : adjoint Spherical harmonic transformation,
        Mf   : Filter operator,
        Mft  : Adjoint filter operator,
        M    : Mask operator,
        N^-1 : Inverse noise covariance in pixel space,
        S^-1 : Inverse signal covariance in harmonic space,
        H    : Scaling filter that scales the linear system to help
               preconditioning but has no physical meaning.

    When the class instance is provided with random draws, the solver will
    instead solve A x = b + b_rand where:

        x      : H^{-1} x^cr, where x^cr is a constrained signal realisation,
        b      : H Lt B Yt Ht M N^-1 d,
        b_rand : H Lt B Yt Mft M N^-0.5 w_s + H S^-0.5 w_n,

    where:

        w_s : Unit-variance Gaussian noise in pixel space,
        w_n : Unit-variance Gaussian noise in spherical harmonic space.

    Parameters
    ----------
    imap : array
        Input data map(s) (masked and beam convolved).
    icov_signal : callable
        Callable that takes alm_data-shaped alm array as input and applies the
        inverse signal covariance matrix.
    icov_noise : callable
        Callable that takes imap-shaped map array as input and applies the
        inverse noise covariance matrix.
    sht : tuple of callable, optional
        Tuple containing the spherical harmonic Y operator and the adjoint Yt operator.
        Y transforms from alm_data-shaped alm array to imap-shape map array and Yt does
        the opposite.
    beam : callable, optional
        Callable that takes alm_data-shaped alm array as input and applies the
        beam window function.
    lens : tuple of callable
        Tuple containing the lensing L operator and the adjoint Lt operator.
        Both L and Lt transform an alm_data-shaped array to another alm array.
    filt : tuple of callable, optional
        Tuple containing filter and filter_adjoint operators.
    mask : callable, optional
        Callable that takes imap-shaped map array as input and applies the
        pixel mask.
    rand_isignal : array, optional
         Draw from inverse signal covariance.
    rand_inoise : array, optional
         Draw from inverse noise covariance matrix.
    swap_bm : bool, optional
        If set, swap the order of the beam/filter and mask operations. Helps convergence
        with large beams and high SNR data.
    scale_filter : callable, optional
        Callable that takes alm_data-shaped alm array as input and applies a symmetric
        positive definite matrix that scales the linear system. May be used to improve
        the separation between masked and unmasked pixels and improve convergence.
    '''

    def __init__(self, imap, icov_signal, icov_noise, sht, beam=None, lens=None,
                 filt=None, mask=None, rand_isignal=None, rand_inoise=None,
                 swap_bm=False, scale_filter=None):

        self.imap = imap
        self.icov_signal = icov_signal
        self.icov_noise = icov_noise

        if beam is None:
            beam = lambda alm: alm
        self.beam = beam

        self.sht = sht[0]
        self.sht_adjoint = sht[1]

        if lens is None:
            self.lens = lambda alm : alm
            self.lens_adjoint = lambda alm : alm
        else:
            self.set_lens(lens)

        if filt is None:
            self.filt = lambda alm : alm
            self.filt_adjoint = lambda alm : alm
        else:
            self.filt = filt[0]
            self.filt_adjoint = filt[1]

        if mask is None:
            mask = lambda imap: imap
        self.mask = mask

        self.swap_bm = swap_bm
        if scale_filter is None:
            scale_filter = lambda alm : alm
        self.scale_filter = scale_filter

        self.set_b_vec(
            self.imap, rand_isignal=rand_isignal, rand_inoise=rand_inoise)

        self.preconditioner = None

    def init_solver(self, **kwargs):
        '''
        Finalize the initalization of the CG solver. Should
        be called after any preconditioner(s) is/are set.

        Parameters
        ----------
        **kwargs
            Keyword arguments for pixell.utils.CG.
        '''

        kwargs.setdefault('dot', alm_utils.contract_almxblm)
        if self.preconditioner is not None:
            kwargs.setdefault('M', self.preconditioner)
        super().__init__(self.a_matrix, self.b_vec, **kwargs)

    def add_preconditioner(self, prec, sel=None):
        '''
        Add a precondioner to the CG solver. Can be called multiple times.

        Parameters
        ----------
        prec : callable
            Callable that takes alm_data-shaped alm array as input and applies
            preconditioner.
        sel : slice, optional
            Slice into alm input array in case preconditioner is only to be
            applied to a slice.
        '''

        if self.preconditioner is None:
            if sel is None:
                self.preconditioner = prec
            else:
                def sliced_prec(alm):
                    alm[sel] = prec(alm[sel])
                    return alm
                self.preconditioner = sliced_prec
        else:
            self.preconditioner = operators.add_operators(
                self.preconditioner, prec, slice_2=sel)

    def reset_preconditioner(self):
        '''
        Reset preconditioner to default (identity matrix). Useful when
        restarting solver.
        '''

        self.preconditioner = None

    def proj(self, alm):
        ''' Apply the projection matrix.'''

        if not self.swap_bm:
            # F M Y B L
            alm = self.lens(alm)            
            alm = self.beam(alm)
            omap = self.sht(alm)
            omap = self.mask(omap)
            omap = self.filt(omap)
        else:
            # F Y B L (Yt W M Y)
            alm = self.mask(alm)
            alm = self.lens(alm)
            alm = self.beam(alm)
            omap = self.sht(alm)
            omap = self.filt(omap)

        return omap

    def proj_adjoint(self, imap):
        ''' Apply the adjoint projection matrix.'''

        if not self.swap_bm:
            # Lt B Yt M Mft
            omap = self.filt_adjoint(imap)
            omap = self.mask(omap)
            alm = self.sht_adjoint(omap)
            alm = self.beam(alm)
            alm = self.lens_adjoint(alm)
        else:
            # (Yt W M Y) Lt B Yt Mft
            omap = self.filt_adjoint(imap)
            alm = self.sht_adjoint(omap)
            alm = self.beam(alm)
            alm = self.lens_adjoint(alm)
            alm = self.mask(alm)            

        return alm

    def a_matrix(self, alm):
        '''
        Apply the A matrix to input alm.

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to A(alm).
        '''

        alm = self.scale_filter(alm)

        omap = self.proj(alm)
        omap = self.icov_noise(omap)
        oalm = self.proj_adjoint(omap)
        oalm = self.scale_filter(oalm)

        # Note that alm already has scale_filter applied.
        oalm += self.scale_filter(self.icov_signal(alm))

        return oalm

    def get_b_vec(self, imap):
        '''
        Convert input map to the b vector (not in place).

        Parameters
        ----------
        imap : array
            Input data map(s) (masked and beam convolved).

        Returns
        -------
        out : array
            Output alm array, corresponding to b.
        '''
        
        omap = self.icov_noise(imap)
        oalm = self.proj_adjoint(omap)
        oalm = self.scale_filter(oalm)

        return oalm

    def get_b_vec_constr(self, rand_isignal, rand_inoise):
        '''
        Compute the random component of the b_rand vector used for
        drawing constrained realizations.

        Parameters
        ----------
        rand_isignal : array, optional
            Draw from inverse signal covariance.
        rand_inoise : array, optional
            Draw from inverse noise covariance matrix.

        Returns
        -------
        out : array
            Output alm array, corresponding to b_rand.
        '''

        oalm = self.scale_filter(self.proj_adjoint(rand_inoise))
        oalm += self.scale_filter(rand_isignal)

        return oalm

    def set_b_vec(self, imap, rand_isignal=None, rand_inoise=None):
        '''
        Set (or reset) the RHS of the equation system. This corresponds
        to setting RHS = b + b_rand.

        Parameters
        ----------
        imap : array
            Input data map(s) (masked and beam convolved).
        rand_isignal : array, optional
            Draw from inverse signal covariance.
        rand_inoise : array, optional
            Draw from inverse noise covariance matrix.
        '''

        self.b_vec = self.get_b_vec(imap)

        if rand_isignal is not None and rand_inoise is not None:
            self.b_vec += self.get_b_vec_constr(rand_isignal, rand_inoise)

        # Copy of input RHS because the solver overwrites this vector.
        self.b0 = self.b_vec.copy()

    def set_lens(self, lens):
        '''
        lens : tuple of callable
            Tuple containing the lensing L operator and the adjoint Lt operator.
            Both L and Lt transform an alm_data-shaped array to another alm array.
        '''
        
        self.lens = lens[0]
        self.lens_adjoint = lens[1]
        
    def get_wiener(self):
        '''Return copy of Wiener-filtered input at current state.'''

        return self.scale_filter(self.x.copy())

    def get_icov(self):
        '''Return copy of inverse covariance filtered input at current state.'''

        return self.icov_signal(self.get_wiener())

    def get_chisq(self):
        '''Return x^dagger S^-1 x + (d - P x)^dagger N^-1 (d - P x) at current state.'''

        x_w = self.get_wiener()
        out = self.dot(x_w, self.icov_signal(x_w))
        res = (self.imap - self.proj(x_w))
        out += np.sum(res * self.icov_noise(res))
        return out

    def get_residual(self):
        '''Return sqrt[(A x - b)^dagger (A x - b)] at current state'''

        r = self.A(self.x) - self.b0
        return np.sqrt(self.dot(r, r))

    def get_qform(self):
        '''Return 0.5 * x^dagger A x - x^dagger b at current state.'''

        return 0.5 * self.dot(self.x, self.A(self.x)) - self.dot(self.x, self.b0)

    @classmethod
    def from_arrays(cls, imap, minfo, ainfo, icov_ell, icov_pix, *extra_args,
                    b_ell=None, mask_pix=None, minfo_mask=None, draw_constr=False,
                    spin=None, swap_bm=False, sfilt=None, lensop=None, seed=None):
        '''
        Initialize solver with arrays instead of callables.

        Parameters
        ----------
        imap : (npol, npix) array
            Input map
        minfo : map_utils.MapInfo object
            Metainfo for input map.
        ainfo : pixell.curvedsky.alm_info object
            Metainfo for output alms.
        icov_ell : (npol, npol, nell) or (npol, nell) array
            Inverse signal covariance. If diagonal, only the diagonal suffices.
        icov_pix : (npol, npol, npix) or (npol, npix) array
            Inverse noise covariance. If diagonal, only the diagonal suffices.
        *extra_args
            Possible extra arguments to init, used for inherited classes.
        b_ell : (npol, nell) array, optional
            Beam window functions.
        mask_pix = (npol, npix) array, optional
            Pixel mask.
        minfo_mask : map_utils.MapInfo object
            Metainfo for pixel mask.
        draw_constr : bool, optional
            If set, initialize for constrained realization instead of Wiener.
        spin : int, array-like, optional
            Spin values for transform, should be compatible with npol. If not provided,
            value will be derived from npol: 1->0, 2->2, 3->[0, 2].
        swap_bm : bool, optional
            If set, swap the order of the beam and mask operations. Helps convergence
            with large beams and high SNR data.
        sfilt : (npol, nell) or (npol, npol, nell) array, optional
            Symmetric positive definite scaling matrix, if diagonal only the diagonal
            suffices.
        lensop : lensing.LensAlm object
            Lensing instance used to compute lensing and adjoint lensing.
        seed : int or np.random._generator.Generator object, optional
            Seed for np.random.seed.

        Raises
        ------
        ValueErorr
            If input shape do not match.
        '''

        if spin is None:
            spin = sht_tools.default_spin(imap.shape)        

        imap = mat_utils.atleast_nd(imap, 2)
        icov_pix = mat_utils.atleast_nd(icov_pix, 2)
        icov_ell = mat_utils.atleast_nd(icov_ell, 2)
        
        if icov_pix.shape[-2:] != imap.shape:
            raise ValueError(f'Mismatch {icov_pix.shape=} and {imap.shape=}')
        if icov_ell.shape[0] != imap.shape[0]:
            raise ValueError(f'Mismatch {icov_ell.shape=} and {imap.shape=}')
            
        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)
        icov_noise = operators.PixMatVecMap(icov_pix, 1, inplace=False)

        if b_ell is not None:            
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if sfilt is not None:
            sfilt = operators.EllMatVecAlm(ainfo, sfilt)

        # NOTE, FIXME.
        filt = None

        if mask_pix is not None:
            if minfo_mask is None:
                minfo_mask = minfo
            if swap_bm:
                mask = operators.PixMatVecAlm(
                    ainfo, mask_pix, minfo_mask, spin, use_weights=True)
            else:
                mask = operators.PixMatVecMap(mask_pix, 1, inplace=False)
        else:
            mask = None

        if draw_constr:
            rand_isignal = alm_utils.rand_alm(icov_ell, ainfo, seed)
            rand_inoise = map_utils.rand_map_pix(icov_pix, seed)
        else:
            rand_isignal = None
            rand_inoise = None

        sht = (operators.YMatVecAlm(ainfo, minfo, spin, qweight=False),
               operators.YTWMatVecMap(minfo, ainfo, spin, qweight=False))

        if lensop:
            if not alm_utils.ainfo_is_equiv(ainfo, lensop.ainfo):
                raise ValueError(f'f{ainfo=} != {lensop.ainfo=}')
            lens = (lensop.lens, lensop.lens_adjoint)
        else:
            lens = None

        return cls(imap, icov_signal, icov_noise, sht, beam=beam, lens=lens,
                   filt=filt, mask=mask, rand_isignal=rand_isignal,
                   rand_inoise=rand_inoise, swap_bm=swap_bm, scale_filter=sfilt)

    @classmethod
    def from_arrays_const_cor(cls, imap, minfo, ainfo, icov_ell, icov_pix, icov_noise_ell,
                              *extra_args, nsteps=5, b_ell=None, mask_pix=None, minfo_mask=None,
                              no_masked_noise=False, draw_constr=False, spin=None, sfilt=None,
                              lensop=None, seed=None, verbose=False):
        '''
        Initialize solver with arrays instead of callables for a constant-correlation noise model:
        
        N = M N^0.5_pix Y C_ell Y^H N^0.5_pix M.

        Parameters
        ----------
        imap : (npol, npix) array
            Input map
        minfo : map_utils.MapInfo object
            Metainfo for input map.
        ainfo : pixell.curvedsky.alm_info object
            Metainfo for output alms.
        icov_ell : (npol, npol, nell) or (npol, nell) array
            Inverse signal covariance. If diagonal, only the diagonal suffices.
        icov_pix : (npol, npol, npix) or (npol, npix) array
            Inverse noise covariance. If diagonal, only the diagonal suffices.
        icov_noise_ell :  (npol, npol, nell) or (npol, nell) array
            Inverse noise covariance. If diagonal, only the diagonal suffices.
        *extra_args
            Possible extra arguments to init, used for inherited classes.
        nsteps : int, optional
            Number of CG steps used to invert N.
        b_ell : (npol, nell) array, optional
            Beam window functions.
        mask_pix = (npol, npix) array, optional
            Pixel mask.
        minfo_mask : map_utils.MapInfo object
            Metainfo for pixel mask.
        no_masked_noise : bool, optional
            If set, assume that the noise has not been masked, i.e. M in the the noise
            model is set to 1.
        draw_constr : bool, optional
            If set, initialize for constrained realization instead of Wiener.
        spin : int, array-like, optional
            Spin values for transform, should be compatible with npol. If not provided,
            value will be derived from npol: 1->0, 2->2, 3->[0, 2].
        sfilt : (npol, nell) or (npol, npol, nell) array, optional
            Symmetric positive definite scaling matrix, if diagonal only the diagonal
            suffices.
        lensop : lensing.LensAlm object
            Lensing instance used to compute lensing and adjoint lensing.
        seed : int or np.random._generator.Generator object, optional
            Seed for np.random.seed.
        verbose : bool, optional
            If set, print information about convergence of internal CG solver.

        Raises
        ------
        ValueErorr
            If input shape do not match.
        '''

        if spin is None:
            spin = sht_tools.default_spin(imap.shape)        

        imap = mat_utils.atleast_nd(imap, 2)
        icov_pix = mat_utils.atleast_nd(icov_pix, 2)
        icov_ell = mat_utils.atleast_nd(icov_ell, 2)
        icov_noise_ell = mat_utils.atleast_nd(icov_noise_ell, 2)
        
        if icov_pix.shape[-2:] != imap.shape:
            raise ValueError(f'Mismatch {icov_pix.shape=} and {imap.shape=}')
        if icov_ell.shape[0] != imap.shape[0]:
            raise ValueError(f'Mismatch {icov_ell.shape=} and {imap.shape=}')
            
        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)
        icov_noise = operators.InvPixEllPixMatVecMap(
            icov_pix, icov_noise_ell, minfo, spin, power_m=-0.5, power_x=-1,
            mask=None if no_masked_noise else mask_pix, nsteps=nsteps, verbose=verbose)

        if b_ell is not None:            
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if sfilt is not None:
            sfilt = operators.EllMatVecAlm(ainfo, sfilt)

        # NOTE, FIXME.
        filt = None

        if mask_pix is not None:
            if minfo_mask is None:
                minfo_mask = minfo
            mask = operators.PixMatVecMap(mask_pix, 1, inplace=False)
        else:
            mask = None

        if draw_constr:
            rand_isignal = alm_utils.rand_alm(icov_ell, ainfo, seed)
            rand_inoise = alm_utils.unit_var_alm(ainfo, imap.shape[:-1], seed)
            sqrt_icov_noise_op = operators.InvSqrtPixEllPixMatVecMap(
                icov_pix, icov_noise_ell, minfo, spin, power_m=-0.5, power_x=-1,
                mask=None if no_masked_noise else mask_pix, inv_op=icov_noise, verbose=verbose)
            rand_inoise = sqrt_icov_noise_op(rand_inoise)
        else:
            rand_isignal = None
            rand_inoise = None

        sht = (operators.YMatVecAlm(ainfo, minfo, spin, qweight=False),
               operators.YTWMatVecMap(minfo, ainfo, spin, qweight=False))

        if lensop:
            if not alm_utils.ainfo_is_equiv(ainfo, lensop.ainfo):
                raise ValueError(f'f{ainfo=} != {lensop.ainfo=}')
            lens = (lensop.lens, lensop.lens_adjoint)
        else:
            lens = None

        return cls(imap, icov_signal, icov_noise, sht, beam=beam, lens=lens,
                   filt=filt, mask=mask, rand_isignal=rand_isignal,
                   rand_inoise=rand_inoise, scale_filter=sfilt)
    
    @classmethod
    def from_arrays_fwav(cls, imap, minfo, ainfo, icov_ell, cov_wav, fkernelset,
                         *extra_args, b_ell=None, mask_pix=None, minfo_mask=None,
                         draw_constr=False, spin=None, swap_bm=False, sfilt=None,
                         cov_noise_2d=None, lensop=None, seed=None):
        '''
        Initialize solver with Fourier-wavelet-based noise model from arrays
        instead of callables.

        Parameters
        ----------
        imap : (npol, npix) array
            Input map
        minfo : map_utils.MapInfo object
            Metainfo for input map.
        ainfo : pixell.curvedsky.alm_info object
            Metainfo for output alms.
        icov_ell : (npol, npol, nell) or (npol, nell) array
            Inverse signal covariance. If diagonal, only the diagonal suffices.
        cov_wav : wavtrans.Wav object
            Wavelet block matrix representing the noise covariance.
        fkernelset : fkernel.FKernelSet object
            Wavelet kernels.

        *extra_args
            Possible extra arguments to init, used for inherited classes.

        b_ell : (npol, nell) array, optional
            Beam window functions.
        mask_pix = (npol, npix) array, optional
            Pixel mask.
        minfo_mask : map_utils.MapInfo object
            Metainfo for pixel mask.
        draw_constr : bool, optional
            If set, initialize for constrained realization instead of Wiener.
        spin : int, array-like, optional
            Spin values for transform, should be compatible with npol. If not provided,
            value will be derived from npol: 1->0, 2->2, 3->[0, 2].
        swap_bm : bool, optional
            If set, swap the order of the beam and mask operations. Helps convergence
            with large beams and high SNR data.
        sfilt : (npol, nell) or (npol, npol, nell) array, optional
            Symmetric positive definite scaling matrix, if diagonal only the diagonal
            suffices.
        cov_noise_2d : (npol, npol, nly, nlx) array or (npol, nly, nlx) array, optional
            Noise covariance in 2D Fourier domain. If diagonal, only the
            diagonal suffices. Will update noise model to iN^0.5 iN_wav iN^0.5,
            so make sure iN_wav corresponds to the inverse noise covariance after
            flattening by iN^0.5.
        lensop : lensing.LensAlm object
            Lensing instance used to compute lensing and adjoint lensing.        
        seed : int or np.random._generator.Generator object, optional
            Seed for np.random.seed.
        '''

        if spin is None:
            spin = sht_tools.default_spin(imap.shape)

        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)
        if cov_noise_2d is not None:
            power_x = -0.5
        else:
            power_x = 0
        icov_noise = operators.FInvFWavFMatVecMap(
            minfo, cov_wav, fkernelset, cov_noise_2d, power_x=power_x, nsteps=3)

        if b_ell is not None:
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if sfilt is not None:
            sfilt = operators.EllMatVecAlm(ainfo, sfilt)

        # NOTE, FIXME.
        filt = None

        if mask_pix is not None:
            if minfo_mask is None:
                minfo_mask = minfo
            if swap_bm:
                mask = operators.PixMatVecAlm(
                    ainfo, mask_pix, minfo_mask, spin, use_weights=True)
            else:
                mask = operators.PixMatVecMap(mask_pix, 1, inplace=False)
        else:
            mask = None

        if draw_constr:
            unit_var_alm = alm_utils.unit_var_alm(
                ainfo, (icov_ell.shape[0],), seed)
            rand_isignal = operators.EllMatVecAlm(
                ainfo, icov_ell, power=0.5, inplace=True)(unit_var_alm)
            wav_draw = noise_utils.unit_var_wav(cov_wav.get_minfos_diag(),
                         imap.shape[:-1], cov_wav.dtype, seed=None)
            sqrt_icov_cov_wav_op = operators.InvSqrtFWavMatVecWav(
                cov_wav, fkernelset, inv_op=icov_noise.inv_op)
            fmap_draw = sqrt_icov_cov_wav_op(wav_draw)
            fmap_draw = icov_noise.x_op(fmap_draw)
            rand_inoise = imap * 0
            dft.irfft(fmap_draw, map_utils.view_2d(rand_inoise, minfo))

        else:
            rand_isignal = None
            rand_inoise = None
            
        sht = (operators.YMatVecAlm(ainfo, minfo, spin, qweight=False),
               operators.YTWMatVecMap(minfo, ainfo, spin, qweight=False))

        if lensop:
            if not alm_utils.ainfo_is_equiv(ainfo, lensop.ainfo):
                raise ValueError(f'f{ainfo=} != {lensop.ainfo=}')
            lens = (lensop.lens, lensop.lens_adjoint)
        else:
            lens = None
        
        return cls(imap, icov_signal, icov_noise, sht, beam=beam, lens=lens,
                   filt=filt, mask=mask, rand_isignal=rand_isignal,
                   rand_inoise=rand_inoise, swap_bm=swap_bm, scale_filter=sfilt)

class CGWiener(utils.CG):
    '''
    Construct a CG solver for x in the equation system A x = b where:

        A : S^-1 + B Ft M N^-1 M F B,
        x : the Wiener filtered version of the input,
        b : B Ft M N^-1 a,

    and

        a    : input vector (masked and beam-convolved sky + noise),
        B    : beam convolution operator,
        F    : filter operator,
        Ft   : adjoint filter operator,
        M    : mask operator,
        N^-1 : inverse noise covariance,
        S^-1 : inverse signal covariance.

    When the class instance is provided with random draws from the inverse noise
    and signal covariance, the solver will instead solve A x = b where:

        x : constrained realisation drawn from P(s |a, (B Ft M N^-1 M F B)^-1, S),
        b : B Ft M N^-1 a + w_s + B w_n,

    where:

        w_s : draw from inverse signal covariance,
        w_n : draw from inverse noise covariance.

    Parameters
    ----------
    alm_data : array
        Input data alms (already masked and beam convolved, so M B a).
    icov_signal : callable
        Callable that takes alm_data-shaped alm array as input and applies the
        inverse signal covariance matrix.
    icov_noise : callable
        Callable that takes alm_data-shaped alm array as input and applies the
        inverse noise covariance matrix.
    beam : callable, optional
        Callable that takes alm_data-shaped alm array as input and applies the
        beam window function.
    filt : tuple of callable, optional
        Tuple containing filter and filter_adjoint operators.
    mask : callable, optional
        Callable that takes alm_data-shaped alm array as input and applies the
        pixel mask.
    rand_isignal : array, optional
        Draw from inverse signal covariance as SH coefficients in alm_data shape.
    rand_inoise : array, optional
        Draw from inverse noise covariance as SH coefficients in alm_data shape.
    swap_bm : bool, optional
        If set, swap the order of the beam/filter and mask operations. Helps convergence
        with large beams and high SNR data.
    '''

    def __init__(self, alm_data, icov_signal, icov_noise, beam=None, filt=None,
                 mask=None, rand_isignal=None, rand_inoise=None, swap_bm=False):

        self.alm_data = alm_data
        self.icov_signal = icov_signal
        self.icov_noise = icov_noise

        if beam is None:
            beam = lambda alm: alm
        self.beam = beam

        if filt is None:
            self.filt = lambda alm : alm
            self.filt_adjoint = lambda alm : alm
        else:
            self.filt = filt[0]
            self.filt_adjoint = filt[1]

        if mask is None:
            mask = lambda alm: alm
        self.mask = mask

        self.rand_isignal = rand_isignal
        self.rand_inoise = rand_inoise
        self.swap_bm = swap_bm

        if self.rand_isignal is not None and self.rand_inoise is not None:
            self.b_vec = self.get_b_vec_constr(self.alm_data)
        else:
            self.b_vec = self.get_b_vec(self.alm_data)
        self.b0 = self.b_vec.copy()

        self.preconditioner = None

    def init_solver(self, **kwargs):
        '''
        Finalize the initalization of the CG solver. Should
        be called after any preconditioner(s) is/are set.

        Parameters
        ----------
        **kwargs
            Keyword arguments for pixell.utils.CG.
        '''

        kwargs.setdefault('dot', alm_utils.contract_almxblm)
        if self.preconditioner is not None:
            kwargs.setdefault('M', self.preconditioner)
        super().__init__(self.a_matrix, self.b_vec, **kwargs)

    def add_preconditioner(self, prec, sel=None):
        '''
        Add a precondioner to the CG solver. Can be called multiple times.

        Parameters
        ----------
        prec : callable
            Callable that takes alm_data-shaped alm array as input and applies
            preconditioner.
        sel : slice, optional
            Slice into alm input array in case preconditioner is only to be
            applied to a slice.
        '''

        if self.preconditioner is None:
            if sel is None:
                self.preconditioner = prec
            else:
                def sliced_prec(alm):
                    alm[sel] = prec(alm[sel])
                    return alm
                self.preconditioner = sliced_prec
        else:
            self.preconditioner = operators.add_operators(
                self.preconditioner, prec, slice_2=sel)

    def reset_preconditioner(self):
        '''
        Reset preconditioner to default (identity matrix). Useful when
        restarting solver.
        '''

        self.preconditioner = None

    def a_matrix(self, alm):
        '''
        Apply the A (= S^-1 + B Ft M N^-1 M F B) matrix to input alm.

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to A(alm).
        '''

        if self.swap_bm:
            alm_noise = self.mask(alm.copy())
            alm_noise = self.beam(alm_noise)
            alm_noise = self.filt(alm_noise)
        else:
            alm_noise = self.beam(alm.copy())
            alm_noise = self.filt(alm_noise)
            alm_noise = self.mask(alm_noise)

        alm_noise = self.icov_noise(alm_noise)

        if self.swap_bm:
            alm_noise = self.filt_adjoint(alm_noise)
            alm_noise = self.beam(alm_noise)
            alm_noise = self.mask(alm_noise)
        else:
            alm_noise = self.mask(alm_noise)
            alm_noise = self.filt_adjoint(alm_noise)
            alm_noise = self.beam(alm_noise)

        alm_signal = self.icov_signal(alm.copy())

        return alm_signal + alm_noise

    def get_b_vec(self, alm):
        '''
        Convert input alm to the b (= B N^-1 a) vector (not in place).

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to b.
        '''
        
        alm = self.icov_noise(alm.copy())
        if self.swap_bm:
            alm = self.filt_adjoint(alm)
            alm = self.beam(alm)
            alm = self.mask(alm)
        else:
            alm = self.mask(alm)
            alm = self.filt_adjoint(alm)
            alm = self.beam(alm)

        return alm

    def get_b_vec_constr(self, alm):
        '''
        Convert input alm to the b vector used for drawing constrained
        realizations (not in place).

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to b.
        '''

        alm = self.get_b_vec(alm)
        if self.swap_bm:
            alm += self.mask(self.beam(self.filt_adjoint(self.rand_inoise.copy())))
        else:
            alm += self.beam(self.filt_adjoint(self.mask(self.rand_inoise.copy())))
        alm += self.rand_isignal

        return alm

    def get_wiener(self):
        '''Return copy of Wiener-filtered input at current state.'''

        return self.x.copy()

    def get_icov(self):
        '''Return copy of (S + B^-1 N B^-1)^-1 B^-1 filtered input at current state.'''

        return self.icov_signal(self.x.copy())

    def get_chisq(self):
        '''Return x^dagger S^-1 x + (a - x)^dagger B M N^-1 M B (a - x) at current state.'''

        x_w = self.get_wiener()
        out = self.dot(x_w, self.icov_signal(x_w))
        out += self.dot(self.beam(self.filt_adjoint(self.mask(self.alm_data - x_w))),
                self.icov_noise(self.beam(self.filt(self.mask(self.alm_data - x_w)))))
        return out

    def get_residual(self):
        '''Return sqrt[(A(x) - b)^dagger (A(x) - b)] at current state'''

        r = self.A(self.x) - self.b0
        return np.sqrt(self.dot(r, r))

    def get_qform(self):
        '''Return 0.5 * x^dagger A x - x^dagger b at current state.'''

        return 0.5 * self.dot(self.x, self.A(self.x)) - self.dot(self.x, self.b0)

    @classmethod
    def from_arrays(cls, alm_data, ainfo, icov_ell, icov_pix, minfo, *extra_args,
                    b_ell=None, mask_pix=None, draw_constr=False, spin=None,
                    icov_noise_flat_ell=None, swap_bm=False, kfilt=None, minfo_kfilt=None):
        '''
        Initialize solver with arrays instead of callables.

        Parameters
        ----------
        alm_data : (npol, nelem) complex array
            SH coefficients of data.
        ainfo : pixell.curvedsky.alm_info object
            Metainfo of data alms.
        icov_ell : (npol, npol, nell) or (npol, nell) array
            Inverse signal covariance. If diagonal, only the diagonal suffices.
        icov_pix : (npol, npol, npix) or (npol, npix) array
            Inverse noise covariance. If diagonal, only the diagonal suffices.
        minfo : map_utils.MapInfo object
            Metainfo for inverse noise covariance.
        *extra_args
            Possible extra arguments to init, used for inherited classes.
        b_ell : (npol, nell) array, optional
            Beam window functions.
        mask_pix = (npol, npix) array, optional
            Pixel mask.
        draw_constr : bool, optional
            If set, initialize for constrained realization instead of Wiener.
        spin : int, array-like, optional
            Spin values for transform, should be compatible with npol. If not provided,
            value will be derived from npol: 1->0, 2->2, 3->[0, 2].
        icov_noise_flat_ell (npol, npol, nell) or (npol, nell) array, optional.
            Inverse noise covariance of flattened (icov_pix weighted) data.
            If diagonal, only the diagonal suffices. If provided, updates noise model to
            icnf_ell^0.5 icov_pix icnf_ell^0.5.
        swap_bm : bool, optional
            If set, swap the order of the beam and mask operations. Helps convergence
            with large beams and high SNR data.
        kfilt : (npol, npol, ky, kx) or (npol, ky, kx) ndmap, optional
            Fourier filter, WCS should correspond to Fourier space and fftshift should
            have been applied to Y axis such that (ly, lx) = 0 lies at (lny // 2 + 1, 0).
        minfo_kfilt : map_utils.MapInfo object, optional
            Metainfo for Clenshaw Curtis map used for Fourier filter.
        '''

        if spin is None:
            spin = sht_tools.default_spin(alm_data.shape)

        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)

        if icov_noise_flat_ell is None:
            icov_noise = operators.PixMatVecAlm(
                ainfo, icov_pix, minfo, spin)
        else:
            icov_noise = operators.EllPixEllMatVecAlm(
                ainfo, icov_pix, icov_noise_flat_ell, minfo, spin, power_x=0.5)

        if b_ell is not None:
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if kfilt is not None:
            filt = (operators.FMatVecAlm(ainfo, kfilt, minfo_kfilt, spin),
                    operators.FMatVecAlm(ainfo, kfilt, minfo_kfilt, spin, adjoint=True))
        else:
            filt = None

        if mask_pix is not None:
            mask = operators.PixMatVecAlm(
                ainfo, mask_pix, minfo, spin, use_weights=True)
        else:
            mask = None

        if draw_constr:
            if icov_noise_flat_ell is not None:
                raise NotImplementedError('icov_noise_flat_ell not implemented for '
                                          'constrained realizations for now.')

            rand_isignal = curvedsky.rand_alm(icov_ell, return_ainfo=False)
            rand_inoise = alm_utils.rand_alm_pix(
                icov_pix, ainfo, minfo, spin, adjoint=True)
        else:
            rand_isignal = None
            rand_inoise = None

        return cls(alm_data, icov_signal, icov_noise, *extra_args, beam=beam, filt=filt,
                   mask=mask, rand_isignal=rand_isignal, rand_inoise=rand_inoise,
                   swap_bm=swap_bm)

    @classmethod
    def from_arrays_wav(cls, alm_data, ainfo, icov_ell, icov_wav, w_ell,
                        *extra_args, b_ell=None, mask_pix=None, minfo_mask=None,
                        draw_constr=False, spin=None, swap_bm=False, icov_noise_ell=None,
                        kfilt=None, minfo_kfilt=None):
        '''
        Initialize solver with wavelet-based noise model with arrays
        instead of callables.

        Parameters
        ----------
        alm_data : (npol, nelem) complex array
            SH coefficients of data.
        ainfo : pixell.curvedsky.alm_info object
            Metainfo of data alms.
        icov_ell : (npol, npol, nell) or (npol, nell) array
            Inverse signal covariance. If diagonal, only the diagonal suffices.
        icov_wav : wavtrans.Wav object
            Wavelet block matrix representing the inverse noise covariance.
        w_ell : (nwav, nell) array
            Wavelet kernels.
        *extra_args
            Possible extra arguments to init, used for inherited classes.
        b_ell : (npol, nell) array, optional
            Beam window functions.
        mask_pix = (npol, npix) array, optional
            Pixel mask.
        minfo_mask : map_utils.MapInfo object
            Metainfo for pixel mask.
        draw_constr : bool, optional
            If set, initialize for constrained realization instead of Wiener.
        spin : int, array-like, optional
            Spin values for transform, should be compatible with npol. If not provided,
            value will be derived from npol: 1->0, 2->2, 3->[0, 2].
        swap_bm : bool, optional
            If set, swap the order of the beam and mask operations. Helps convergence
            with large beams and high SNR data.
        icov_noise_ell : (npol, npol, nell) or (npol, nell) array, optional
            Inverse noise covariance (with fsky correction). If diagonal, only the
            diagonal suffices. Will update noise model to iN_ell^0.5 iN_wav iN_ell^0.5,
            so make sure iN_wav corresponds to the inverse noise covariance after
            flattening by iN_ell^0.5.
        kfilt : (npol, npol, ky, kx) or (npol, ky, kx) ndmap, optional
            Fourier filter, WCS should correspond to Fourier space and fftshift should
            have been applied to Y axis such that (ly, lx) = 0 lies at (lny // 2 + 1, 0).
        minfo_kfilt : map_utils.MapInfo object, optional
            Metainfo for Clenshaw Curtis map used for Fourier filter.
        '''

        if spin is None:
            spin = sht_tools.default_spin(alm_data.shape)

        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)

        if icov_noise_ell is None:
            icov_noise = operators.WavMatVecAlm(
                ainfo, icov_wav, w_ell, spin)
        else:
            icov_noise = operators.EllWavEllMatVecAlm(
                ainfo, icov_wav, w_ell, icov_noise_ell, spin, power_x=0.5)

        if b_ell is not None:
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if kfilt is not None:
            filt = (operators.FMatVecAlm(ainfo, kfilt, minfo_kfilt, spin),
                    operators.FMatVecAlm(ainfo, kfilt, minfo_kfilt, spin, adjoint=True))
        else:
            filt = None

        if mask_pix is not None:
            mask = operators.PixMatVecAlm(
                ainfo, mask_pix, minfo_mask, spin, use_weights=True)
        else:
            mask = None

        if draw_constr:
            rand_isignal = curvedsky.rand_alm(icov_ell, return_ainfo=False)
            rand_inoise = alm_utils.rand_alm_wav(icov_wav, ainfo, w_ell,
                                                 spin, adjoint=True)
            if icov_noise_ell is not None:
                sqrt_icov_noise = operators.EllMatVecAlm(ainfo, icov_noise_ell,
                                                         power=0.5, inplace=True)
                sqrt_icov_noise(rand_inoise)
        else:
            rand_isignal = None
            rand_inoise = None

        return cls(alm_data, icov_signal, icov_noise, *extra_args, beam=beam, filt=filt,
                   mask=mask, rand_isignal=rand_isignal, rand_inoise=rand_inoise,
                   swap_bm=swap_bm)

class CGWienerScaled(CGWiener):
    '''
    Construct a CG solver for x in the equation system A x = b where:

        A : 1 + S^1/2 B Ft M N^-1 M F B S^1/2,
        x : the Wiener filtered version of the input scaled by S^-1/2,
        b : S^1/2 B Ft M N^-1 a,

    and

        a     : input vector (beam-convolved sky + noise),
        B     : beam convolution operator,
        F    : filter operator,
        Ft   : adjoint filter operator,
        M    : mask operator,
        N^-1  : inverse noise covariance,
        S^1/2 : square root of signal covariance.

    When the class instance is provided with random draws from the inverse noise
    and signal covariance, the solver will instead solve A x = b where:

        x : constrained realisation drawn from P(s |a, B^-1 N B^-1, S) scaled by S^-1/2,
        b : B Ft M N^-1 a + w_s + S^1/2 B Ft M w_n,

    where:

        w_s : draw from unit-variance gaussian distribution,
        w_n : draw from inverse noise covariance.

    Parameters
    ----------
    alm_data : array
        Input data alms (already beam convolved).
    icov_signal : callable
        Callable that takes alm_data-shaped alm array as input and applies the
        inverse signal covariance matrix.
    icov_noise : callable
        Callable that takes alm_data-shaped alm array as input and applies the
        inverse noise covariance matrix.
    sqrt_cov_signal : callable
        Callable that takes alm_data-shaped alm array as input and applies the
        square root of the signal covariance matrix.
    beam : callable, optional
        Callable that takes alm_data-shaped alm array as input and applies the
        beam window function.
    filt : tuple of callable, optional
        Tuple containing filter and filter_adjoint operators.
    rand_isignal : array, optional
        Draw from inverse signal covariance as SH coefficients in alm_data shape.
    rand_inoise : array, optional
        Draw from inverse noise covariance as SH coefficients in alm_data shape.
    swap_bm : bool, optional
        If set, swap the order of the beam and mask operations. Helps convergence
        with large beams and high SNR data.
    '''

    def __init__(self, alm_data, icov_signal, icov_noise, sqrt_cov_signal, beam=None,
                 filt=None, mask=None, rand_isignal=None, rand_inoise=None, swap_bm=False):

        self.sqrt_cov_signal = sqrt_cov_signal

        CGWiener.__init__(self, alm_data, icov_signal, icov_noise, beam=beam, filt=filt,
                          mask=mask, rand_isignal=rand_isignal, rand_inoise=rand_inoise,
                          swap_bm=swap_bm)

    def a_matrix(self, alm):
        '''
        Apply the A (= (1 + S^1/2 B Ft M N^-1 M F B S^1/2)) matrix to input alm.

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to A(alm).
        '''

        alm_noise = self.sqrt_cov_signal(alm.copy())
        if self.swap_bm:
            alm_noise = self.mask(alm_noise)
            alm_noise = self.beam(alm_noise)
            alm_noise = self.filt(alm_noise)
        else:
            alm_noise = self.beam(alm_noise)
            alm_noise = self.filt(alm_noise)
            alm_noise = self.mask(alm_noise)
        alm_noise = self.icov_noise(alm_noise)
        if self.swap_bm:
            alm_noise = self.filt_adjoint(alm_noise)
            alm_noise = self.beam(alm_noise)
            alm_noise = self.mask(alm_noise)
        else:
            alm_noise = self.mask(alm_noise)
            alm_noise = self.filt_adjoint(alm_noise)
            alm_noise = self.beam(alm_noise)
        alm_noise = self.sqrt_cov_signal(alm_noise)

        alm_signal = alm

        return alm_signal + alm_noise

    def get_b_vec(self, alm):
        '''
        Convert input alm to the b (= S^1/2 B Ft M N^-1 a) vector (not in place).

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to b.
        '''

        alm = self.icov_noise(alm.copy())
        if self.swap_bm:
            alm = self.filt_adjoint(alm)
            alm = self.beam(alm)
            alm = self.mask(alm)
        else:
            alm = self.mask(alm)
            alm = self.filt_adjoint(alm)
            alm = self.beam(alm)
        alm = self.sqrt_cov_signal(alm)

        return alm

    def get_b_vec_constr(self, alm):
        '''
        Convert input alm to the b vector used for drawing constrained
        realizations (not in place).

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to b.
        '''

        alm = self.get_b_vec(alm)
        if self.swap_bm:
            alm += self.sqrt_cov_signal(self.mask(self.beam(self.filt_adjoint(
                self.rand_inoise.copy()))))
        else:
            alm += self.sqrt_cov_signal(self.beam(self.filt_adjoint(self.mask(
                self.rand_inoise.copy()))))
        alm += self.sqrt_cov_signal(self.rand_isignal.copy())

        return alm

    def get_wiener(self):
        '''Return copy of Wiener-filtered input at current state.'''
        return self.sqrt_cov_signal(self.x.copy())

    def get_icov(self):
        '''Return copy of (S + B^-1 N B^-1)^-1 B^-1 filtered input at current state.'''
        return self.icov_signal(self.get_wiener())

    @classmethod
    def from_arrays(cls, alm_data, ainfo, icov_ell, icov_pix, minfo,
                    b_ell=None, draw_constr=False, spin=None, swap_bm=False):
        '''Initialize solver with arrays instead of callables.'''

        sqrt_cov_signal = operators.EllMatVecAlm(ainfo, icov_ell, power=-0.5)

        return super(CGWienerScaled, cls).from_arrays(
            alm_data, ainfo, icov_ell, icov_pix, minfo, sqrt_cov_signal, b_ell=b_ell,
            draw_constr=draw_constr, spin=spin, swap_bm=swap_bm)
