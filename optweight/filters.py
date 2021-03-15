import numpy as np
import warnings

from enlib import cg
from pixell import curvedsky,sharp

from optweight import map_utils, mat_utils

class CGPixFilter(object):
    def __init__(self, ncomp, theory_cls, b_ell, lmax, icov=None,
                 icov_pix=None, cov_noise_ell=None, minfo=None,
                 include_te=False, rtol_icov=1e-2, order=1):
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
        cov_noise_ell : (ncomp,ncomp,nells) or (ncomp,nells) array
            Power spectrum of flattened (ivar_pix-weighted) noise in units
            (e.g. 1/uK^2) consistent with the alms and theory_cls. Used to model
            noise that deviates from the white noise described by the icov_pix maps
            as function of multipole.
        minfo : sharp.map_info object
            Metainfo for inverse noise covariance icov_pix.
        include_te : bool
            Whether or not to jointly filter T,E,B maps by accounting for the
            signal TE correlation. If True, the returned alms will be optimally
            filtered, but the "T" and "E" maps will not be pure-T and pure-E.
        rtol_icov: float, optional
            Elements below rtol_icov times the median of nonzero elements 
            are set to zero in the reprojected icov_pix map.
        order: int, optional
            The order of spline interpolation when transforming icov to GL
            pixelization.
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
            icov_pix, minfo = map_utils.enmap2gauss(icov, 2 * lmax, area_pow=1, mode='nearest',order=order)
        else:
            if minfo is None:
                warnings.warn(f"optfilt: No minfo for icov_guess specified. Using 2x {lmax}.")
                minfo = map_utils.get_enmap_minfo(icov.shape,icov.wcs,2 * lmax)

        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception

        icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=rtol_icov)
        if np.any(icov_pix<0): raise Exception

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

        if cov_noise_ell is not None:
            icov_noise_ell = mat_utils.matpow(cov_noise_ell, -1)
        else:
            icov_noise_ell is None
                
        if b_ell.ndim==1:
            b_ell = b_ell[None] * np.asarray((1,1,1)[:ncomp])[:,None]
        elif b_ell.ndim==2:
            if b_ell.shape[0]!=ncomp: raise Exception
        else:
            raise ValueError

        self.icov_ell = icov_ell
        self.icov_pix = icov_pix
        self.icov_noise_ell = icov_noise_ell
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
                                          icov_noise_flat_ell=self.icov_noise_ell,
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
                    print(f"optfilt benchmark: /t {chisq/ alm.size / 2 if benchmark else ''} /t "
                          f"{residual if benchmark else ''}")
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

def cg_pix_filter(alm,theory_cls, b_ell, lmax,
                  icov=None,icov_pix=None, cov_noise_ell=None, minfo=None,
                  include_te=False, niter=None, stype='pcg_pinv', ainfo=None,
                  benchmark=None, verbose=True, err_tol=1e-15, rtol_icov=1e-2,
                  order=1):
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
    cov_noise_ell : (ncomp,ncomp,nells) or (ncomp,nells) array
        Power spectrum of flattened (ivar_pix-weighted) noise in units
        (e.g. 1/uK^2) consistent with the alms and theory_cls. Used to model
        noise that deviates from the white noise described by the icov_pix maps
        as function of multipole.
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
    order: int, optional
        The order of spline interpolation when transforming icov to GL
        pixelization.

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

    cgobj = CGPixFilter(ncomp,theory_cls=theory_cls, b_ell=b_ell, lmax=lmax,
                        icov=icov, icov_pix=icov_pix, cov_noise_ell=cov_noise_ell,
                        minfo=minfo, include_te=include_te, rtol_icov=rtol_icov,order=order)
    return cgobj.filter(alm,benchmark=benchmark, verbose=verbose, ainfo=ainfo,
                        niter=niter,stype=stype, err_tol=err_tol)
