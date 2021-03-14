import numpy as np

from enlib import cg
from pixell import curvedsky,sharp

from optweight import operators, alm_utils, map_utils, preconditioners, sht

class CGWiener(cg.CG):
    '''
    Construct a CG solver for x in the equation system A x = b where:

        A : S^-1 + B M N^-1 M B,
        x : the Wiener filtered version of the input,
        b : B M N^-1 a,

    and

        a    : input vector (masked and beam-convolved sky + noise),
        B    : beam convolution operator,
        M    : mask operator,
        N^-1 : inverse noise covariance,
        S^-1 : inverse signal covariance.

    When the class instance is provided with random draws from the inverse noise
    and signal covariance, the solver will instead solve A x = b where:

        x : constrained realisation drawn from P(s |a, (B M N^-1 M B)^-1, S),
        b : B M N^-1 a + w_s + B w_n,

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
    mask : callable, optional
        Callable that takes alm_data-shaped alm array as input and applies the
        pixel mask.
    rand_isignal : array, optional
        Draw from inverse signal covariance as SH coefficients in alm_data shape.
    rand_inoise : array, optional
        Draw from inverse noise covariance as SH coefficients in alm_data shape.
    **kwargs
        Keyword arguments for enlib.cg.CG.
    '''

    def __init__(self, alm_data, icov_signal, icov_noise, beam=None,
                 mask=None, rand_isignal=None, rand_inoise=None, **kwargs):

        self.alm_data = alm_data
        self.icov_signal = icov_signal
        self.icov_noise = icov_noise
        if beam is None:
            beam = lambda alm: alm
        self.beam = beam
        if mask is None:
            mask = lambda alm: alm
        self.mask = mask
        self.rand_isignal = rand_isignal
        self.rand_inoise = rand_inoise

        if self.rand_isignal is not None and self.rand_inoise is not None:
            b = self.b_vec_constr(self.alm_data)
        else:
            b = self.b_vec(self.alm_data)
        self.b0 = b.copy()

        kwargs.setdefault('dot', alm_utils.contract_almxblm)
        cg.CG.__init__(self, self.a_matrix, b, **kwargs)

    def a_matrix(self, alm):
        '''
        Apply the A (= S^-1 + B M N^-1 M B) matrix to input alm.

        Parameters
        ----------
        alm : array
            Input alm array.

        Returns
        -------
        out : array
            Output alm array, corresponding to A(alm).
        '''

        alm_noise = self.beam(alm.copy())
        alm_noise = self.mask(alm_noise)
        alm_noise = self.icov_noise(alm_noise)
        alm_noise = self.mask(alm_noise)
        alm_noise = self.beam(alm_noise)

        alm_signal = self.icov_signal(alm.copy())

        return alm_signal + alm_noise

    def b_vec(self, alm):
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
        alm = self.mask(alm)
        alm = self.beam(alm)

        return alm

    def b_vec_constr(self, alm):
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

        alm = self.b_vec(alm)
        alm += self.beam(self.mask(self.rand_inoise.copy()))
        alm += self.rand_isignal

        return alm

    def get_wiener(self):
        '''Return copy of Wiener-filtered input at current state.'''

        return self.x.copy()

    def get_icov(self):
        '''Return copy of (S + B^-1 N B^-1)^-1 B^-1 filtered input at current state.'''

        return self.icov_signal(self.x.copy())

    def get_chisq(self):
        '''Return x^dagger S^-1 x + (a - x)^dagger B M N^-1 M B (a - x).'''

        x_w = self.get_wiener()
        out = self.dot(x_w, self.icov_signal(x_w))
        out += self.dot(self.beam(self.mask(self.alm_data - x_w)),
                        self.icov_noise(self.beam(self.mask(self.alm_data - x_w))))
        return out

    def get_residual(self):
        '''Return sqrt[(A(x) - b)^dagger (A(x) - b)]'''

        r = self.A(self.x) - self.b0
        return np.sqrt(self.dot(r, r))

    @classmethod
    def from_arrays(cls, alm_data, ainfo, icov_ell, icov_pix, minfo, *extra_args,
                    b_ell=None, mask_pix=None, draw_constr=False, prec=None, spin=None,
                    **kwargs):
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
        prec : {'harmonic', 'pinv'}, optional
            Select type of preconditioner, one of:

            harmonic
                Use (S^-1 + itau * 1)^-1, where itau is isotropic inverse variance.
            pinv
                Use Pseudo-inverse method from Seljebotn et al.

        spin : int, array-like, optional
            Spin values for transform, should be compatible with npol. If not provided,
            value will be derived from npol: 1->0, 2->2, 3->[0, 2].
        **kwargs
            Keyword arguments for enlib.cg.CG.
        '''

        if kwargs.get('M') and prec:
            raise ValueError('Pick only one preconditioner')

        if spin is None:
            spin = sht.default_spin(alm_data.shape)

        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)
        icov_noise = operators.PixMatVecAlm(
            ainfo, icov_pix, minfo, spin)

        if b_ell is not None:
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if mask_pix is not None:
            mask =  operators.PixMatVecAlm(
                ainfo, mask_pix, minfo, spin, use_weights=True)
        else:
            mask=None

        if draw_constr:
            rand_isignal = curvedsky.rand_alm(icov_ell, return_ainfo=False)
            rand_inoise = alm_utils.rand_alm_pix(
                icov_pix, ainfo, minfo, spin)

        else:
            rand_isignal = None
            rand_inoise = None

        if prec == 'harmonic':

            itau = map_utils.get_isotropic_ivar(icov_pix, minfo)
            preconditioner = preconditioners.HarmonicPreconditioner(
                ainfo, icov_ell, itau, b_ell=b_ell)

        elif prec == 'pinv':
         
            itau = map_utils.get_isotropic_ivar(icov_pix, minfo)
            preconditioner = preconditioners.PseudoInvPreconditioner(
                ainfo, icov_ell, itau, icov_pix, minfo, spin, b_ell=b_ell)

        elif prec is None:
            preconditioner = None

        else:
            raise ValueError('Preconditioner: {} not understood'.format(prec))

        if preconditioner:
            kwargs.setdefault('M', preconditioner)

        return cls(alm_data, icov_signal, icov_noise, *extra_args, beam=beam,
                   mask=mask, rand_isignal=rand_isignal, rand_inoise=rand_inoise,
                   **kwargs)

    @classmethod
    def from_arrays_wav(cls, alm_data, ainfo, icov_ell, icov_wav, w_ell,
                        *extra_args, b_ell=None, mask_pix=None, minfo_mask=None,
                        icov_pix=None, minfo_icov_pix=None, prec=None, spin=None, 
                        **kwargs):
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
        icov_pix : (npol, npol, npix) or (npol, npix) array, optional
            Inverse noise covariance. If diagonal, only the diagonal suffices.
        minfo_icov_pix : sharp.map_info object
            Metainfo for pixel mask.
        prec : {'harmonic', 'pinv_wav', 'pinv'}, optional
            Select type of preconditioner, one of:

            harmonic
                Use (S^-1 + itau * 1)^-1, where itau is an approximate 
                inverse noise variance spectrum.
            pinv_wav
                Use Pseudo-inverse method from Seljebotn et al adapted
                to a wavelet-based noise model.
            pinv
                Use Pseudo-inverse method from Seljebotn et al. Requires 
                a covariance matrix to be passed though the `icov_pix` kwarg.

        spin : int, array-like, optional
            Spin values for transform, should be compatible with npol. If not provided,
            value will be derived from npol: 1->0, 2->2, 3->[0, 2].
        **kwargs
            Keyword arguments for enlib.cg.CG.
        '''

        if kwargs.get('M') and prec:
            raise ValueError('Pick only one preconditioner')

        if spin is None:
            spin = sht.default_spin(alm_data.shape)

        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)
        icov_noise = operators.WavMatVecAlm(
            ainfo, icov_wav, w_ell, spin)

        if b_ell is not None:
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if mask_pix is not None:
            mask = operators.PixMatVecAlm(
                ainfo, mask_pix, minfo_mask, spin, use_weights=True)
        else:
            mask = None

        if prec == 'harmonic':

            itau_ell = map_utils.get_ivar_ell(icov_wav, w_ell)
            preconditioner = preconditioners.HarmonicPreconditioner(
                ainfo, icov_ell, itau_ell, b_ell=b_ell)

        elif prec == 'pinv':
         
            if icov_pix is None:
                raise ValueError('pinv preconditioner requires icov_pix.')
            if minfo_icov_pix is None:
                raise ValueError('pinv preconditioner requires minfo_icov_pix.')

            itau = map_utils.get_isotropic_ivar(icov_pix, minfo_icov_pix)
            preconditioner = preconditioners.PseudoInvPreconditioner(
                ainfo, icov_ell, itau, icov_pix, minfo_icov_pix, spin, b_ell=b_ell)

        elif prec == 'pinv_wav':
         
            itau_ell = map_utils.get_ivar_ell(icov_wav, w_ell)
            preconditioner = preconditioners.PseudoInvPreconditionerWav(
                ainfo, icov_ell, itau_ell, icov_wav, w_ell, spin, mask_pix=mask_pix, 
                minfo_mask=minfo_mask, b_ell=b_ell)

        elif prec is None:
            preconditioner = None

        else:
            raise ValueError('Preconditioner: {} not understood'.format(prec))

        if preconditioner:
            kwargs.setdefault('M', preconditioner)

        return cls(alm_data, icov_signal, icov_noise, *extra_args, beam=beam,
                   mask=mask, rand_isignal=None, rand_inoise=None,
                   **kwargs)

class CGWienerScaled(CGWiener):
    '''
    Construct a CG solver for x in the equation system A x = b where:

        A : 1 + S^1/2 B N^-1 B S^1/2,
        x : the Wiener filtered version of the input scaled by S^-1/2,
        b : S^1/2 B N^-1 a,

    and

        a     : input vector (beam-convolved sky + noise),
        B     : beam convolution operator,
        N^-1  : inverse noise covariance,
        S^1/2 : square root of signal covariance.

    When the class instance is provided with random draws from the inverse noise
    and signal covariance, the solver will instead solve A x = b where:

        x : constrained realisation drawn from P(s |a, B^-1 N B^-1, S) scaled by S^-1/2,
        b : B N^-1 a + w_s + S^1/2 B w_n,

    where:

        w_s : draw from univariate distribution,
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
    rand_isignal : array, optional
        Draw from inverse signal covariance as SH coefficients in alm_data shape.
    rand_inoise : array, optional
        Draw from inverse noise covariance as SH coefficients in alm_data shape.
    **kwargs
        Keyword arguments for enlib.cg.CG.
    '''

    def __init__(self, alm_data, icov_signal, icov_noise, sqrt_cov_signal, beam=None,
                 rand_isignal=None, rand_inoise=None, **kwargs):

        self.sqrt_cov_signal = sqrt_cov_signal

        CGWiener.__init__(self, alm_data, icov_signal, icov_noise, beam=beam,
                          rand_isignal=rand_isignal, rand_inoise=rand_inoise, **kwargs)

    def a_matrix(self, alm):
        '''
        Apply the A (= (1 + S^1/2 B N^-1 B S^1/2)) matrix to input alm.

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
        alm_noise = self.beam(alm_noise)
        alm_noise = self.icov_noise(alm_noise)
        alm_noise = self.beam(alm_noise)
        alm_noise = self.sqrt_cov_signal(alm_noise)

        alm_signal = alm

        return alm_signal + alm_noise

    def b_vec(self, alm):
        '''
        Convert input alm to the b (= S^1/2 B N^-1 a) vector (not in place).

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
        alm = self.beam(alm)
        alm = self.sqrt_cov_signal(alm)

        return alm

    def b_vec_constr(self, alm):
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

        alm = self.b_vec(alm)
        alm += self.sqrt_cov_signal(self.beam(self.rand_inoise.copy()))
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
                    b_ell=None, draw_constr=False, prec=None, spin=None, **kwargs):
        '''Initialize solver with arrays instead of callables.'''

        sqrt_cov_signal = operators.EllMatVecAlm(ainfo, icov_ell, power=-0.5)

        return super(CGWienerScaled, cls).from_arrays(
            alm_data, ainfo, icov_ell, icov_pix, minfo, sqrt_cov_signal, b_ell=b_ell,
            draw_constr=draw_constr, prec=prec, spin=spin, **kwargs)
