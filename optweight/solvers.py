import numpy as np

from enlib import cg
from pixell import curvedsky,sharp

from optweight import operators
from optweight import alm_utils
from optweight import map_utils
from optweight import preconditioners
from optweight import sht

import warnings


class CGPixFilter(object):
    def __init__(self,ncomp,theory_cls,b_ell,lmax,
                 icov=None,icov_pix=None,minfo=None,
                 include_te=False,
                 rtol_icov=1e-2):

        """
        Prepare to filter maps using a pixel-space instrument noise model
        and a harmonic space signal model. The initialization does a slow
        reprojection of the icov map to Gauss-Legendre pixelization.

        Parameters
        ----------
        ncomp : int
            The number of map components. Use ncomp=1 for T-only and
            ncomp=3 for T,E,B filtering.
        theory_cls : dict
            A dictionary mapping the keys TT and optionally TE, EE and
            BB to 1d numpy arrays containing CMB C_ell power spectra 
            (no 2pi, ell^2 or beam factors), starting at ell=0 and up to at
            least lmax. Should have units (e.g. uK^2) consistent with alm 
            and icov inputs.
        b_ell : (nells,) or (ncomp,nells) array
            A numpy array containing the map-space beam transfer function
            (starting at ell=0) to assume in the noise model. Separate
            beams can be specified for T,E,B if the array is 2d.
        lmax : int
            The maximum multipole for the filtering, used to determine the
            resolution of the Gauss-Legendre pixelization geometry. If icov_pix
            is provided instead of icov, this lmax must correspond to the lmax
            of the icov_pix map.
        icov : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) or (Ny,Nx) ndmap
            An enmap containing the inverse (co-)variance per pixel (zeros in
            unobserved region), in units (e.g. 1/uK^2) consistent with
            the alms and theory_cls. If the icov_pix map in Gauss-Legendre
            pixelization is provided, that will be used instead. IQ, IU, QU
            elements can also be specified if icov is 4-dimensional.
            Within ACT and SO, these are sometimes referred to as 'ivar' or
            'div'
        icov_pix : (ncomp,npix) or (npix,) array
            An array containing the inverse variance per pixel (zeros in
            unobserved region), in units (e.g. 1/uK^2) consistent with
            the alms and theory_cls in Gauss-Legendre
            pixelization. If this is not provided, the provided icov
            map will be reprojected.
        minfo : sharp.map_info object
            Metainfo for inverse noise covariance icov_pix.
        include_te : bool
            Whether or not to jointly filter T,E,B maps by accounting for the
            signal TE correlation. If True, the returned alms will be optimally
            filtered, but the "T" and "E" maps will not be pure-T and pure-E.
        rtol_icov: float, optional
            Elements below rtol_icov times the median of nonzero elements 
            are set to zero in the reprojected icov_pix map.
        """

        if np.any(np.logical_not(np.isfinite(b_ell))): raise Exception

        if ncomp!=1 and ncomp!=3: raise Exception


        if icov_pix is None:
            if icov.ndim==2:
                # If only T icov is provided, assume Q,U icov is 0.5x
                icov = icov[None] * np.asarray([1.,0.5,0.5][:ncomp])[:,None,None]
            elif icov.ndim==3:
                if icov.shape[0]!=ncomp: raise Exception
            elif icov.ndim==4:
                if icov.shape[0]!=ncomp or icov.shape[1]!=ncomp: raise Exception
            else:
                raise ValueError
            if np.any(np.logical_not(np.isfinite(icov))): raise Exception
            icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest')
        else:
            if minfo is None:
                warnings.warn(f"optfilt: No minfo for icov_guess specified. Using 2x {lmax}.")
                minfo = map_utils.get_enmap_minfo(icov.shape,icov.wcs,2 * lmax)

        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception

        icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=rtol_icov)

        tlmax = theory_cls['TT'].size - 1
        if not(tlmax>=lmax): raise Exception
        cov_ell = np.zeros((ncomp, ncomp, tlmax + 1))
        cov_ell[0,0] = theory_cls['TT']
        if ncomp>1:
            if include_te:
                cov_ell[0,1] = theory_cls['TE']
                cov_ell[1,0] = theory_cls['TE']
            cov_ell[1,1] = theory_cls['EE']
            cov_ell[2,2] = theory_cls['BB']

        # Invert to get inverse signal cov.
        icov_ell = np.ones((ncomp, ncomp, tlmax + 1))
        for lidx in range(icov_ell.shape[-1]):
            if lidx < 2:
                # Set monopole and dipole to zero.
                icov_ell[:,:,lidx] = 0
            else:
                icov_ell[:,:,lidx] = np.linalg.inv(cov_ell[:,:,lidx])

        if b_ell.ndim==1:
            b_ell = b_ell[None] * np.asarray((1,1,1)[:ncomp])[:,None]
        elif b_ell.ndim==2:
            if b_ell.shape[0]!=ncomp: raise Exception
        else:
            raise ValueError

        self.icov_ell = icov_ell
        self.icov_pix = icov_pix
        self.minfo = minfo
        self.b_ell = b_ell
        self.ncomp = ncomp

    def filter(self,alm,
               benchmark=False,verbose=True,
               niter=None,ainfo=None,
               stype='pcg_pinv',err_tol=1e-15):

        """
        Filter a map using a pixel-space instrument noise model
        and a harmonic space signal model.

        The returned map is the beam-deconvolved Wiener filtered
        map. For example, for isotropic inputs
        d = M*b + n
        The returned Wiener filtered map is
        C b^-1  (C + N b^-2) d
        and the inverse variance filtered map is
        b^-1  (C + N b^-2) d

        Parameters
        ----------
        alm : (ncomp,nalm) or (nalm,) array
            Input data alms (already masked and beam convolved, so M B a).
        niter : int
            The number of Conjugate Gradient iterations to be performed. The default
            is 35, but this may be too small (unconverged filtering) or too large 
            (wasted iterations) for your application. Test before deciding on 
            this parameter.
        stype : str
            The type of pre-conditioner to use. 
            Choose from 'cg', 'cg_scaled', 'pcg_harm' and 'pcg_pinv'.
            The default is 'pcg_pinv'.
        ainfo : sharp.alm_info object
            Metainfo for internally used alms. Will be determined from the alm
            size if not specified.
        verbose : bool
            Whether or not to print information and progress.
        benchmark: int
            Provide benchmarks every 'benchmark' iterations. This includes 
            chi_squared, residuals and power spectra during iteration.
            This can considerably slow down filtering, especially if done
            every step of the iteration. Set to None to not get any
            benchmarks other than the inexpensive error calculation.
        err_tol: float
            If the CG error is below this number, stop iterating even if niter
            has not been reached.

        Returns
        -------

        output : dict
            A dictionary that maps the following keys to the corresponding products.
            - 'walm': (ncomp,nalm) array containing the Wiener filtered alms.
            - 'ialm': (ncomp,nalm) array containing the inverse variance filtered alms.
            - 'solver': The CG solver object instance
            - Convergence statistics 'errors', and if benchmark is True,
              'residuals', 'chisqs', 'ps' calculated at iteration numbers
              'itnums'    

        """

        x0 = None
        if np.any(np.logical_not(np.isfinite(alm))): raise Exception
        if alm.ndim==1:
            alm = alm[None]
            if self.ncomp!=1: raise Exception
        elif alm.ndim==2:
            if self.ncomp!=alm.shape[0]: raise Exception
        else:
            raise ValueError
        ainfo = sharp.alm_info(nalm=alm.shape[-1]) if ainfo is None else ainfo
        if stype == 'cg' or stype == 'cg_scaled':
            prec = None
        elif stype == 'pcg_harm':
            prec = 'harmonic'
        elif stype == 'pcg_pinv':
            prec = 'pinv'
        if stype == 'cg_scaled':
            solver = CGWienerScaled.from_arrays(alm, ainfo, self.icov_ell, 
                                                        self.icov_pix, self.minfo, 
                                                        b_ell=self.b_ell,
                                                        draw_constr=False, 
                                                        prec=None, x0=x0)
        else:
            solver = CGWiener.from_arrays(alm, ainfo, self.icov_ell, 
                                                  self.icov_pix, self.minfo, b_ell=self.b_ell,
                                                  draw_constr=False, 
                                                  prec=prec, x0=x0)
        errors = []
        errors.append( np.nan)
        if benchmark:
            warnings.warn("optfilt: Benchmarking is turned on. This can significantly slow down the filtering.")
            b_copy = solver.b.copy()
            chisqs = []
            residuals = []
            ps_c_ells = []
            itnums = []
            chisqs.append( solver.get_chisq() )
            residuals.append( solver.get_residual() )
            itnums.append(0)
            if verbose: print('|b| :', np.sqrt(solver.dot(b_copy, b_copy)))


        if niter is None:
            niter = 35
            warnings.warn(f"optfilt: Using the default number of iterations {niter}.")

        for idx in range(niter):
            solver.step()
            errors.append( solver.err )
            if benchmark:
                if (idx+1)%benchmark==0:
                    chisqs.append( solver.get_chisq() )
                    residuals.append( solver.get_residual() )
                    ps_c_ells.append( ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:]) )
                    itnums.append(idx)
                    print(f"optfilt benchmark: /t {chisq/ alm.size / 2 if benchmark else ''} /t {residual if benchmark else ''}")
            if verbose: print(f"optfilt step {solver.i} / {niter},  error {errors[-1]:.2e}")
            if solver.err < err_tol: 
                warnings.warn(f"Stopping early because the error {solver.err} is below err_tol {err_tol}")
                break

        output = {}
        output['walm'] = solver.get_wiener()
        output['ialm'] = solver.get_icov()
        output['solver'] = solver
        output['errors'] = errors
        if benchmark:
            output['chisqs'] = chisqs
            output['residuals'] = residuals
            output['ps'] = ps_c_ells
            output['itnums'] = itnums
        return output

def cg_pix_filter(alm,theory_cls,b_ell,lmax,
                  icov=None,icov_pix=None,minfo=None,
                  include_te=False,
                  niter=None,stype='pcg_pinv',ainfo=None,
                  benchmark=None,verbose=True,
                  err_tol=1e-15,
                  rtol_icov=1e-2):

    """
    Filter a map using a pixel-space instrument noise model
    and a harmonic space signal model.

    The returned map is the beam-deconvolved Wiener filtered
    map. For example, for isotropic inputs
    d = M*b + n
    The returned Wiener filtered map is
    C b^-1  (C + N b^-2) d
    and the inverse variance filtered map is
    b^-1  (C + N b^-2) d

    Parameters
    ----------
    alm : (ncomp,nalm) or (nalm,) array
        Input data alms (already masked and beam convolved, so M B a).
    theory_cls : dict
        A dictionary mapping the keys TT and optionally TE, EE and
        BB to 1d numpy arrays containing CMB C_ell power spectra 
        (no 2pi, ell^2 or beam factors), starting at ell=0 and up to at
        least lmax. Should have units (e.g. uK^2) consistent with alm 
        and icov inputs.
    b_ell : (nells,) or (ncomp,nells) array
        A numpy array containing the map-space beam transfer function
        (starting at ell=0) to assume in the noise model. Separate
        beams can be specified for T,E,B if the array is 2d.
    lmax : int
        The maximum multipole for the filtering, used to determine the
        resolution of the Gauss-Legendre pixelization geometry. If icov_pix
        is provided instead of icov, this lmax must correspond to the lmax
        of the icov_pix map.
    icov : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) or (Ny,Nx) ndmap
        An enmap containing the inverse (co-)variance per pixel (zeros in
        unobserved region), in units (e.g. 1/uK^2) consistent with
        the alms and theory_cls. If the icov_pix map in Gauss-Legendre
        pixelization is provided, that will be used instead. IQ, IU, QU
        elements can also be specified if icov is 4-dimensional.
        Within ACT and SO, these are sometimes referred to as 'ivar' or
        'div'
    icov_pix : (ncomp,npix) or (npix,) array
        An array containing the inverse variance per pixel (zeros in
        unobserved region), in units (e.g. 1/uK^2) consistent with
        the alms and theory_cls in Gauss-Legendre
        pixelization. If this is not provided, the provided icov
        map will be reprojected.
    minfo : sharp.map_info object
        Metainfo for inverse noise covariance icov_pix.
    rtol_icov: float, optional
        Elements below rtol_icov times the median of nonzero elements 
        are set to zero in the reprojected icov_pix map.
    include_te : bool
        Whether or not to jointly filter T,E,B maps by accounting for the
        signal TE correlation. If True, the returned alms will be optimally
        filtered, but the "T" and "E" maps will not be pure-T and pure-E.
    niter : int
        The number of Conjugate Gradient iterations to be performed. The default
        is 35, but this may be too small (unconverged filtering) or too large 
        (wasted iterations) for your application. Test before deciding on 
        this parameter.
    stype : str
        The type of pre-conditioner to use. 
        Choose from 'cg', 'cg_scaled', 'pcg_harm' and 'pcg_pinv'.
        The default is 'pcg_pinv'.
    ainfo : sharp.alm_info object
        Metainfo for internally used alms. Will be determined from the alm
        size if not specified.
    verbose : bool
        Whether or not to print information and progress.
    benchmark: int
        Provide benchmarks every 'benchmark' iterations. This includes 
        chi_squared, residuals and power spectra during iteration.
        This can considerably slow down filtering, especially if done
        every step of the iteration. Set to None to not get any
        benchmarks other than the inexpensive error calculation.
    err_tol: float
        If the CG error is below this number, stop iterating even if niter
        has not been reached.

    Returns
    -------

    output : dict
        A dictionary that maps the following keys to the corresponding products.
        - 'walm': (ncomp,nalm) array containing the Wiener filtered alms.
        - 'ialm': (ncomp,nalm) array containing the inverse variance filtered alms.
        - 'solver': The CG solver object instance
        - Convergence statistics 'errors', and if benchmark is True,
          'residuals', 'chisqs', 'ps' calculated at iteration numbers
          'itnums'    
        
    """

    if alm.ndim==1:
        alm = alm[None]
        ncomp = 1
    elif alm.ndim==2:
        ncomp = alm.shape[0]
    else:
        raise ValueError

    cgobj = CGPixFilter(ncomp,theory_cls=theory_cls,b_ell=b_ell,lmax=lmax,
                        icov=icov,icov_pix=icov_pix,minfo=minfo,
                        include_te=include_te,
                        rtol_icov=rtol_icov)
    return cgobj.filter(alm,benchmark=benchmark,
                        verbose=verbose,ainfo=ainfo,
                        niter=niter,stype=stype,
                        err_tol=err_tol)


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
        Apply the A (= S^-1 + B M N^-1 M B)) matrix to input alm.

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
        '''Return copy of (S + N)^-1 filtered input at current state.'''

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
                    b_ell=None, mask_pix=None, draw_constr=False, prec=None, **kwargs):
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

        **kwargs
            Keyword arguments for enlib.cg.CG.
        '''

        if kwargs.get('M') and prec:
            raise ValueError('Pick only one preconditioner')

        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)

        icov_noise = operators.PixMatVecAlm(
            ainfo, icov_pix, minfo, [0, 2])

        if b_ell is not None:
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if mask_pix is not None:
            mask =  operators.PixMatVecAlm(
                ainfo, mask_pix, minfo, [0, 2], use_weights=True)
        else:
            mask=None

        if draw_constr:
            rand_isignal = curvedsky.rand_alm(icov_ell, return_ainfo=False)
            rand_inoise = alm_utils.rand_alm_pix(
                icov_pix, ainfo, minfo, dtype=alm_data.dtype)

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
                ainfo, icov_ell, itau, icov_pix, minfo, b_ell=b_ell)

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
                        icov_pix=None, minfo_icov_pix=None, prec=None, **kwargs):
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

        **kwargs
            Keyword arguments for enlib.cg.CG.
        '''

        icov_signal = operators.EllMatVecAlm(ainfo, icov_ell)

        icov_noise = operators.WavMatVecAlm(
            ainfo, icov_wav, w_ell, [0, 2])        

        if b_ell is not None:
            beam = operators.EllMatVecAlm(ainfo, b_ell)
        else:
            beam = None

        if mask_pix is not None:
            mask = operators.PixMatVecAlm(
                ainfo, mask_pix, minfo_mask, [0, 2], use_weights=True)
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
                ainfo, icov_ell, itau, icov_pix, minfo_icov_pix, b_ell=b_ell)

        elif prec == 'pinv_wav':
         
            itau_ell = map_utils.get_ivar_ell(icov_wav, w_ell)
            preconditioner = preconditioners.PseudoInvPreconditionerWav(
                ainfo, icov_ell, itau_ell, icov_wav, w_ell, mask_pix=mask_pix, 
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
        '''Return copy of (S + N)^-1 filtered input at current state.'''
        return self.icov_signal(self.get_wiener(self.x.copy()))

    @classmethod
    def from_arrays(cls, alm_data, ainfo, icov_ell, icov_pix, minfo,
                    b_ell=None, draw_constr=False, prec=None, **kwargs):
        '''Initialize solver with arrays instead of callables.'''

        sqrt_cov_signal = operators.EllMatVecAlm(ainfo, icov_ell, power=-0.5)

        return super(CGWienerScaled, cls).from_arrays(
            alm_data, ainfo, icov_ell, icov_pix, minfo, sqrt_cov_signal, b_ell=b_ell,
            draw_constr=draw_constr, prec=prec, **kwargs)
