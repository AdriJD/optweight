import numpy as np

from pixell import curvedsky, sharp, utils

from optweight import operators, alm_utils, map_utils, preconditioners, sht, mat_utils

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
        sel : slice
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
        ainfo : sharp.alm_info object
            Metainfo of data alms.
        icov_ell : (npol, npol, nell) or (npol, nell) array
            Inverse signal covariance. If diagonal, only the diagonal suffices.
        icov_pix : (npol, npol, npix) or (npol, npix) array
            Inverse noise covariance. If diagonal, only the diagonal suffices.
        minfo : sharp.map_info object
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
        minfo_kfilt : sharp.map_info object, optional
            Metainfo for Clenshaw Curtis map used for Fourier filter.
        '''

        if spin is None:
            spin = sht.default_spin(alm_data.shape)

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
        ainfo : sharp.alm_info object
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
        minfo_mask : sharp.map_info object
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
        minfo_kfilt : sharp.map_info object, optional
            Metainfo for Clenshaw Curtis map used for Fourier filter.
        '''

        if spin is None:
            spin = sht.default_spin(alm_data.shape)

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
