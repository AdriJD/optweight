import numpy as np
import warnings
from timeit import default_timer as timer

from pixell import curvedsky, utils, enmap, curvedsky

from optweight import map_utils, mat_utils, solvers, preconditioners

class CGPixFilter(object):
    def __init__(self, ncomp, theory_cls, b_ell, lmax,
                 icov_pix=None, mask_bool=None, cov_noise_ell=None, minfo=None,
                 include_te=False, rtol_icov=1e-2, order=1, swap_bm=True,
                 scale_a=False):
        """
        Prepare to filter maps using a pixel-space instrument noise model
        and a harmonic space signal model. 

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
            resolution of the Gauss-Legendre pixelization geometry.
        icov_pix : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) or (Ny,Nx) ndmap
            An enmap containing the inverse (co-)variance per pixel (zeros in
            unobserved region), in units (e.g. 1/uK^2) consistent with
            the alms and theory_cls. IQ, IU, QU elements can also be specified
            if icov_pix is 4-dimensional. Within ACT and SO, these are sometimes 
            referred to as 'ivar' or 'div'. Can also be a map in Gauss-Legendre
            or HEALPix pixelization. In those cases last dimensions should be 
            'npix' instead of 'Ny,Nx'.
        mask_bool : (ncomp,Ny,Nx) or (Ny,Nx) ndmap
            Boolean mask (True for observed pixels. Geometry must match that of
            'icov_pix'. If not provided, will be determined from 'ivar'.
        cov_noise_ell : (ncomp,ncomp,nells) or (ncomp,nells) array
            Power spectrum of flattened (ivar_pix-weighted) noise in units
            (e.g. 1/uK^2) consistent with the alms and theory_cls. Used to model
            noise that deviates from the white noise described by the icov maps
            as function of multipole.
        minfo : map_utils.MapInfo object
            Metainfo for inverse noise covariance icov_pix in case a Gauss-Legendre
            map is provided.
        include_te : bool
            Whether or not to jointly filter T,E,B maps by accounting for the
            signal TE correlation. If True, the returned alms will be optimally
            filtered, but the "T" and "E" maps will not be pure-T and pure-E.
        rtol_icov: float, optional
            Elements below rtol_icov times the median of nonzero elements 
            are set to zero in the reprojected icov map.
        order: int, optional
            The order of spline interpolation when transforming icov_pix to GL
            pixelization.
        swap_bm : bool, optional
            Swap the order of the beam and mask operations. Helps convergence
            with large beams and high SNR data.
        scale_a : bool, optional
            If set, scale the A matrix to localization of N^-1 term. This may
            help convergence with small beams and high SNR data.
        """

        if np.any(np.logical_not(np.isfinite(b_ell))): raise Exception

        if ncomp != 1 and ncomp != 3: raise Exception

        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception


        # REPLACE WITH CALLS TO MATCH_ENMAP_MINFO.

        if isinstance(icov_pix, enmap.ndmap):
            #icov_pix, minfo = map_utils.enmap2gauss(icov_pix, 2 * lmax, area_pow=1,
            #                                    mode='nearest', order=order)
            minfo = map_utils.match_enmap_minfo(icov_pix.shape, icov_pix.wcs)            
            #if mask_bool is not None:
                #mask_bool, _ = map_utils.enmap2gauss(mask_bool.astype(np.float32),
                #                                     2 * lmax, area_pow=0,
                #                                     mode='nearest', order=1)                
                
        
        elif minfo is None:
            # Assume map is HEALPix, convert to GL.
            icov_pix, minfo = map_utils.healpix2gauss(icov_pix, 2 * lmax, area_pow=1)
            if mask_bool is not None:
                mask_bool, _ = map_utils.healpix2gauss(mask_bool.astype(np.float32),
                                                       2 * lmax, area_pow=1)
        if icov_pix.ndim == 1:
            # If only T icov is provided, assume Q,U icov is 0.5x
            icov_pix = icov_pix[np.newaxis] * np.asarray([1.,0.5,0.5][:ncomp])[:,np.newaxis]
        elif icov_pix.ndim == 2:
            if icov_pix.shape[0] != ncomp: raise Exception
        elif icov_pix.ndim == 3:
            if icov_pix.shape[0] != ncomp or icov_pix.shape[1] != ncomp: raise Exception
        else:
            raise ValueError
        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception
        #icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=rtol_icov, threshold=True) # NOTE Threshold

        if mask_bool is None:
            mask_bool = np.zeros((ncomp, icov_pix.shape[-1]), dtype=bool)
            if icov_pix.ndim == 2:
                mask_bool[:] = icov_pix.astype(bool)
            else:
                for cidx in range(ncomp):
                    mask_bool[cidx] = icov_pix[cidx,cidx].astype(bool)
        elif mask_bool.ndim == 2 and mask_bool.shape[0] != 1:
            if mask_bool.shape[0] != ncomp: raise Exception
        elif mask_bool.ndim == 1:
            mask_bool = mask_bool[np.newaxis,:]
        mask_bool = mask_bool.astype(bool, copy=False)

        # if not swap_bm:            
        #     if mask_bool.ndim == 1:
        #        icov_pix *= mask_bool
        #     elif mask_bool.ndim == 2:
        #        if icov_pix.ndim == 2:
        #            icov_pix *= mask_bool
        #        else:
        #            icov_pix *= np.einsum('ak, bk -> abk', mask_bool, mask_bool)
            
        tlmax = theory_cls['TT'].size - 1
        if not(tlmax >= lmax): raise Exception
        cov_ell = np.zeros((ncomp, ncomp, lmax + 1))
        cov_ell[0,0] = theory_cls['TT'][:lmax+1]
        if ncomp > 1:
            if include_te:
                cov_ell[0,1] = theory_cls['TE'][:lmax+1]
                cov_ell[1,0] = theory_cls['TE'][:lmax+1]
            cov_ell[1,1] = theory_cls['EE'][:lmax+1]
            cov_ell[2,2] = theory_cls['BB'][:lmax+1]

        # Invert to get inverse signal cov.
        icov_ell = np.zeros_like(cov_ell)
        for lidx in range(icov_ell.shape[-1]):
            icov_ell[:,:,lidx] = np.linalg.pinv(cov_ell[:,:,lidx])

        if cov_noise_ell is not None:
            icov_noise_ell = mat_utils.matpow(cov_noise_ell, -1)
        else:
            icov_noise_ell = None
                
        if b_ell.ndim==1:
            b_ell = b_ell[None] * np.asarray((1, 1, 1)[:ncomp])[:,None]
        elif b_ell.ndim == 2:
            if b_ell.shape[0] != ncomp: raise Exception
        else:
            raise ValueError

        if scale_a:
            sfilt = mat_utils.matpow(b_ell, -0.5)
        else:
            sfilt = None

        self.icov_ell = icov_ell
        self.icov_pix = icov_pix
        self.mask_bool = mask_bool
        self.icov_noise_ell = icov_noise_ell
        self.minfo = minfo
        self.b_ell = b_ell
        self.sfilt = sfilt
        self.ncomp = ncomp
        self.swap_bm = swap_bm
        self.lmax = lmax

    def filter(self, imap, benchmark=False, verbose=True, niter=None, niter_masked_cg=5, 
               lmax_masked_cg=3000, ainfo=None, stype='pcg_pinv', err_tol=1e-15):
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
        imap : (ncomp, Ny, Nx) or (Ny, Nx) ndmap array
            Input data.
        niter : int
            The number of Conjugate Gradient iterations to be performed. The default
            is 15, but this may be too small (unconverged filtering) or too large 
            (wasted iterations) for your application. Test before deciding on 
            this parameter.
        stype : str
            The type of pre-conditioner to use. 
            Choose from 'cg', 'pcg_harm' and 'pcg_pinv'.
            The default is 'pcg_pinv'.
        niter_masked_cg : int
            Number of initial iterations using (an expensive) preconditioner for the
            masked pixels.
        lmax_masked_cg : int
            Band-limit for preconditioner for masked solver. Should be set to multipole
            where SNR becomes smaller 1.
        ainfo : pixell.curvedsky.alm_info object
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

        #if np.any(np.logical_not(np.isfinite(alm))): raise Exception
        #if alm.ndim==1:
        #    alm = alm[None]
        #    if self.ncomp!=1: raise Exception
        #if imap.ndim == 2:
        #    imap = imap[np.newaxis,:,:]
        #elif imap.ndim == 3:
        #    if self.ncomp != imap.shape[0]: raise Exception
        ##else:
        #    raise ValueError
            
        #ainfo = curvedsky.alm_info(nalm=alm.shape[-1]) if ainfo is None else ainfo
        ainfo = curvedsky.alm_info(self.lmax)
        mask_pix = self.mask_bool.astype(np.float32)# if self.swap_bm else None
        #solver = solvers.CGWiener.from_arrays(alm, ainfo, self.icov_ell, 
        #                                      self.icov_pix, self.minfo, b_ell=self.b_ell,
        #                                      draw_constr=False, mask_pix=mask_pix,
        #                                      icov_noise_flat_ell=self.icov_noise_ell,
        #                                      swap_bm=self.swap_bm)
        print('imap', imap.shape)
        print('icov_pix', self.icov_pix.shape)
        print('mask_pix', mask_pix.shape)
        print(f'{self.swap_bm =}')
        solver = solvers.CGWienerMap.from_arrays(imap, self.minfo, ainfo, self.icov_ell, 
                                                 self.icov_pix, b_ell=self.b_ell,
                                                 draw_constr=False, mask_pix=mask_pix,
                                                 swap_bm=self.swap_bm, spin=[0, 2],
                                                 sfilt=self.sfilt)
        
        if stype == 'pcg_harm':
            itau = map_utils.get_isotropic_ivar(self.icov_pix, self.minfo)
            if self.icov_noise_ell is not None:
                nell = self.icov_noise_ell.shape[-1]
                sqrt_icnf = mat_utils.matpow(self.icov_noise_ell, 0.5)
                itau = itau[:,:,np.newaxis] * np.ones(itau.shape + (nell,))
                itau = np.einsum('ijl, jkl, kol -> iol', sqrt_icnf, itau, sqrt_icnf)

            prec_main = preconditioners.HarmonicPreconditioner(
                ainfo, self.icov_ell, b_ell=self.b_ell, itau=itau, sfilt=self.sfilt)

        elif stype == 'pcg_pinv':
            if self.icov_noise_ell is not None:
                raise NotImplementedError('icov_noise_ell not implemented for '
                                          'pinv, use harmonic preconditioner for now.')
            prec_main = preconditioners.PseudoInvPreconditioner(
                ainfo, self.icov_ell, self.icov_pix, self.minfo, [0, 2], b_ell=self.b_ell,
                sfilt=self.sfilt)


        # THIS SHOULD BE DONE IN PRECOMPUTING STEP!!!!
        # CAN I NOT SPEEDUP THE MG DENSE COMPUTATION?
        prec_masked_cg = preconditioners.MaskedPreconditionerCG(
            ainfo, self.icov_ell, [0, 2], self.mask_bool, self.minfo,
            #lmax=lmax_masked_cg, nsteps=15)
            lmax=lmax_masked_cg, nsteps=5, sfilt=self.sfilt)

        prec_masked_mg = preconditioners.MaskedPreconditioner(
            ainfo, self.icov_ell[0:1,0:1], 0, self.mask_bool[0], self.minfo,
            #min_pix=10000, n_jacobi=1, lmax_r_ell=6000)
            min_pix=1000, n_jacobi=1, lmax_r_ell=6000,
            #min_pix=1000, n_jacobi=1, lmax_r_ell=4000, # NOTE for sfilt
            sfilt=self.sfilt[0:1,0:1] if self.sfilt is not None else None)

        solver.add_preconditioner(prec_main)
        solver.add_preconditioner(prec_masked_cg)
        solver.init_solver()
        
        errors = []
        errors.append(np.nan)
        if benchmark:
            warnings.warn("optweight: Benchmarking is turned on. "\
                          "This significantly slows down the filtering.")
            chisqs = []
            residuals = []
            qforms = []
            ps_c_ells = []
            itnums = []
            chisqs.append(solver.get_chisq())
            residuals.append(solver.get_residual())
            itnums.append(0)
            if verbose: print('|b| :', np.sqrt(solver.dot(solver.b0, solver.b0)))

        if niter is None:
            niter = 15
            warnings.warn(f"optweight: Using the default number of iterations {niter}.")

        for idx in range(niter_masked_cg + niter):
            if idx == niter_masked_cg:
                solver.reset_preconditioner()
                solver.add_preconditioner(prec_main)
                solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
                solver.b_vec = solver.b0
                solver.init_solver(x0=solver.x)

            t_start = timer()
            solver.step()
            t_eval = timer() - t_start

            if idx >= niter_masked_cg:
                errors.append(solver.err * errors[niter_masked_cg-1])
            else:
                errors.append(solver.err)

            if benchmark:
                if (idx+1)%benchmark==0:
                    chisq = solver.get_chisq()
                    residual = solver.get_residual()
                    qform = solver.get_qform()
                    chisqs.append(chisq)
                    residuals.append(residual)
                    qforms.append(qform)                    
                    ps_c_ells.append(ainfo.alm2cl(
                        solver.get_wiener()[:,None,:], solver.get_wiener()[None,:,:]))
                    itnums.append(idx)
                    print(f"optweight benchmark: \t chisq : {chisq:.2f} \t "
                          f"residual : {residual:.2f} \t qform : {qform:.2f}")
            if verbose:
                print(f"optweight step {solver.i} / {niter}, error {errors[-1]:.2e}, time {t_eval:.3f} s")
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
            output['qforms'] = qforms
            output['ps'] = ps_c_ells
            output['itnums'] = itnums
        return output

def cg_pix_filter(imap, theory_cls, b_ell, lmax,
                  icov_pix=None, mask_bool=None, cov_noise_ell=None, minfo=None,
                  include_te=False, niter=None, stype='pcg_pinv', ainfo=None,
                  niter_masked_cg=7, lmax_masked_cg=3000, benchmark=None, 
                  verbose=True, err_tol=1e-15, rtol_icov=1e-2, order=1,
                  swap_bm=True, scale_a=False):
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
    imap : (ncomp, Ny, Nx) or (Ny, Nx) ndmap array
        Input data.
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
    icov_pix : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) or (Ny,Nx) ndmap
        An enmap containing the inverse (co-)variance per pixel (zeros in
        unobserved region), in units (e.g. 1/uK^2) consistent with
        the alms and theory_cls. IQ, IU, QU elements can also be specified
        if icov_pix is 4-dimensional. Within ACT and SO, these are sometimes 
        referred to as 'ivar' or 'div'. Can also be a map in Gauss-Legendre
        or HEALPix pixelization. In those cases last dimensions should be 
        'npix' instead of 'Ny,Nx'.
    mask_bool : (ncomp,Ny,Nx) or (Ny,Nx) ndmap
        Boolean mask (True for observed pixels. Geometry must match that of
        'icov_pix'. If not provided, will be determined from 'ivar'.
    cov_noise_ell : (ncomp,ncomp,nells) or (ncomp,nells) array
        Power spectrum of flattened (ivar_pix-weighted) noise in units
        (e.g. 1/uK^2) consistent with the alms and theory_cls. Used to model
        noise that deviates from the white noise described by the icov_pix maps
        as function of multipole.
    minfo : map_utils.MapInfo object
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
        is 15, but this may be too small (unconverged filtering) or too large 
        (wasted iterations) for your application. Test before deciding on 
        this parameter.
    stype : str
        The type of pre-conditioner to use. 
        Choose from 'cg', 'cg_scaled', 'pcg_harm' and 'pcg_pinv'.
        The default is 'pcg_pinv'.
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for internally used alms. Will be determined from the alm
        size if not specified.
    niter_masked_cg : int
        Number of initial iterations using (an expensive) preconditioner for the
        masked pixels.
    lmax_masked_cg : int
        Band-limit for preconditioner for masked solver. Should be set to multipole
        where SNR becomes smaller 1.
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
    swap_bm : bool, optional
        Swap the order of the beam and mask operations. Helps convergence
        with large beams and high SNR data.
    scale_a : bool, optional
        If set, scale the A matrix to localization of N^-1 term. This may
        help convergence with small beams and high SNR data.

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

    #if imap.ndim == 2:
    #    imap = imap[np.newaxis,:,:]
    ncomp = 3
    #elif imap.ndim == 3:
    #    ncomp = imap.shape[0]
    #else:
    #    raise ValueError

    cgobj = CGPixFilter(ncomp, theory_cls=theory_cls, b_ell=b_ell, lmax=lmax,
                        icov_pix=icov_pix, mask_bool=mask_bool, cov_noise_ell=cov_noise_ell,
                        minfo=minfo, include_te=include_te, rtol_icov=rtol_icov,order=order,
                        swap_bm=swap_bm, scale_a=scale_a)
    return cgobj.filter(imap, benchmark=benchmark, verbose=verbose, ainfo=ainfo,
                        niter_masked_cg=niter_masked_cg, lmax_masked_cg=lmax_masked_cg,
                        niter=niter, stype=stype, err_tol=err_tol)

####

class CGPixFilterOld(object):
    def __init__(self, ncomp, theory_cls, b_ell, lmax,
                 icov_pix=None, mask_bool=None, cov_noise_ell=None, minfo=None,
                 include_te=False, rtol_icov=1e-2, order=1, swap_bm=True):
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
            resolution of the Gauss-Legendre pixelization geometry.
        icov_pix : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) or (Ny,Nx) ndmap
            An enmap containing the inverse (co-)variance per pixel (zeros in
            unobserved region), in units (e.g. 1/uK^2) consistent with
            the alms and theory_cls. IQ, IU, QU elements can also be specified
            if icov_pix is 4-dimensional. Within ACT and SO, these are sometimes 
            referred to as 'ivar' or 'div'. Can also be a map in Gauss-Legendre
            or HEALPix pixelization. In those cases last dimensions should be 
            'npix' instead of 'Ny,Nx'.
        mask_bool : (ncomp,Ny,Nx) or (Ny,Nx) ndmap
            Boolean mask (True for observed pixels. Geometry must match that of
            'icov_pix'. If not provided, will be determined from 'ivar'.
        cov_noise_ell : (ncomp,ncomp,nells) or (ncomp,nells) array
            Power spectrum of flattened (ivar_pix-weighted) noise in units
            (e.g. 1/uK^2) consistent with the alms and theory_cls. Used to model
            noise that deviates from the white noise described by the icov maps
            as function of multipole.
        minfo : map_utils.MapInfo object
            Metainfo for inverse noise covariance icov_pix in case a Gauss-Legendre
            is provided.
        include_te : bool
            Whether or not to jointly filter T,E,B maps by accounting for the
            signal TE correlation. If True, the returned alms will be optimally
            filtered, but the "T" and "E" maps will not be pure-T and pure-E.
        rtol_icov: float, optional
            Elements below rtol_icov times the median of nonzero elements 
            are set to zero in the reprojected icov map.
        order: int, optional
            The order of spline interpolation when transforming icov_pix to GL
            pixelization.
        swap_bm : bool, optional
           Swap the order of the beam and mask operations. Helps convergence
           with large beams and high SNR data.
        """

        if np.any(np.logical_not(np.isfinite(b_ell))): raise Exception

        if ncomp!=1 and ncomp!=3: raise Exception

        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception

        if isinstance(icov_pix, enmap.ndmap):
            icov_pix, minfo = map_utils.enmap2gauss(icov_pix, 2 * lmax, area_pow=1,
                                                mode='nearest', order=order)
            if mask_bool is not None:
                mask_bool, _ = map_utils.enmap2gauss(mask_bool.astype(np.float32),
                                                     2 * lmax, area_pow=0,
                                                     mode='nearest', order=1)                
        elif minfo is None:
            # Assume map is HEALPix.
            icov_pix, minfo = map_utils.healpix2gauss(icov_pix, 2 * lmax, area_pow=1)
            if mask_bool is not None:
                mask_bool, _ = map_utils.healpix2gauss(mask_bool.astype(np.float32),
                                                       2 * lmax, area_pow=1)
        if icov_pix.ndim == 1:
            # If only T icov is provided, assume Q,U icov is 0.5x
            icov_pix = icov_pix[np.newaxis] * np.asarray([1.,0.5,0.5][:ncomp])[:,np.newaxis]
        elif icov_pix.ndim == 2:
            if icov_pix.shape[0] != ncomp: raise Exception
        elif icov_pix.ndim == 3:
            if icov_pix.shape[0] != ncomp or icov_pix.shape[1] != ncomp: raise Exception
        else:
            raise ValueError
        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception
        icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=rtol_icov)

        if mask_bool is None:
            mask_bool = np.zeros((ncomp, icov_pix.shape[-1]), dtype=bool)
            if icov_pix.ndim == 2:
                mask_bool[:] = icov_pix.astype(bool)
            else:
                for cidx in range(ncomp):
                    mask_bool[cidx] = icov_pix[cidx,cidx].astype(bool)
        elif mask_bool.ndim == 2 and mask_bool.shape[0] != 1:
            if mask_bool.shape[0] != ncomp: raise Exception
        elif mask_bool.ndim == 1:
            mask_bool = mask_bool[np.newaxis,:]
        mask_bool = mask_bool.astype(bool, copy=False)

        if not swap_bm:
            if mask_bool.ndim == 1:
                icov_pix *= mask_bool
            elif mask_bool.ndim == 2:
                if icov_pix.ndim == 2:
                    icov_pix *= mask_bool
                else:
                    icov_pix *= np.einsum('ak, bk -> abk', mask_bool, mask_bool)
            
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
            icov_noise_ell = None
                
        if b_ell.ndim==1:
            b_ell = b_ell[None] * np.asarray((1,1,1)[:ncomp])[:,None]
        elif b_ell.ndim==2:
            if b_ell.shape[0]!=ncomp: raise Exception
        else:
            raise ValueError

        self.icov_ell = icov_ell
        self.icov_pix = icov_pix
        self.mask_bool = mask_bool
        self.icov_noise_ell = icov_noise_ell
        self.minfo = minfo
        self.b_ell = b_ell
        self.ncomp = ncomp
        self.swap_bm = swap_bm

    def filter(self,alm, benchmark=False, verbose=True, niter=None, niter_masked_cg=5, 
               lmax_masked_cg=3000, ainfo=None, stype='pcg_pinv', err_tol=1e-15):
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
            is 15, but this may be too small (unconverged filtering) or too large 
            (wasted iterations) for your application. Test before deciding on 
            this parameter.
        stype : str
            The type of pre-conditioner to use. 
            Choose from 'cg', 'pcg_harm' and 'pcg_pinv'.
            The default is 'pcg_pinv'.
        niter_masked_cg : int
            Number of initial iterations using (an expensive) preconditioner for the
            masked pixels.
        lmax_masked_cg : int
            Band-limit for preconditioner for masked solver. Should be set to multipole
            where SNR becomes smaller 1.
        ainfo : pixell.curvedsky.alm_info object
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

        if np.any(np.logical_not(np.isfinite(alm))): raise Exception
        if alm.ndim==1:
            alm = alm[None]
            if self.ncomp!=1: raise Exception
        elif alm.ndim==2:
            if self.ncomp!=alm.shape[0]: raise Exception
        else:
            raise ValueError
            
        ainfo = curvedsky.alm_info(nalm=alm.shape[-1]) if ainfo is None else ainfo
        mask_pix = self.mask_bool.astype(np.float32) if self.swap_bm else None
        solver = solvers.CGWiener.from_arrays(alm, ainfo, self.icov_ell, 
                                              self.icov_pix, self.minfo, b_ell=self.b_ell,
                                              draw_constr=False, mask_pix=mask_pix,
                                              icov_noise_flat_ell=self.icov_noise_ell,
                                              swap_bm=self.swap_bm)
        
        if stype == 'pcg_harm':
            itau = map_utils.get_isotropic_ivar(self.icov_pix, self.minfo)
            if self.icov_noise_ell is not None:
                nell = self.icov_noise_ell.shape[-1]
                sqrt_icnf = mat_utils.matpow(self.icov_noise_ell, 0.5)
                itau = itau[:,:,np.newaxis] * np.ones(itau.shape + (nell,))
                itau = np.einsum('ijl, jkl, kol -> iol', sqrt_icnf, itau, sqrt_icnf)

            prec_main = preconditioners.HarmonicPreconditioner(
                ainfo, self.icov_ell, b_ell=self.b_ell, itau=itau)

        elif stype == 'pcg_pinv':
            if self.icov_noise_ell is not None:
                raise NotImplementedError('icov_noise_ell not implemented for '
                                          'pinv, use harmonic preconditioner for now.')
            prec_main = preconditioners.PseudoInvPreconditioner(
                ainfo, self.icov_ell, self.icov_pix, self.minfo, [0, 2], b_ell=self.b_ell)

        prec_masked_cg = preconditioners.MaskedPreconditionerCG(
            ainfo, self.icov_ell, [0, 2], self.mask_bool, self.minfo,
            lmax=lmax_masked_cg, nsteps=15)

        prec_masked_mg = preconditioners.MaskedPreconditioner(
            ainfo, self.icov_ell[0:1,0:1], 0, self.mask_bool[0], self.minfo,
            min_pix=10000, n_jacobi=1, lmax_r_ell=6000)

        solver.add_preconditioner(prec_main)
        solver.add_preconditioner(prec_masked_cg)
        solver.init_solver()
        
        errors = []
        errors.append(np.nan)
        if benchmark:
            warnings.warn("optfilt: Benchmarking is turned on. This can significantly slow down the filtering.")
            chisqs = []
            residuals = []
            ps_c_ells = []
            itnums = []
            chisqs.append(solver.get_chisq())
            residuals.append(solver.get_residual())
            itnums.append(0)
            if verbose: print('|b| :', np.sqrt(solver.dot(solver.b0, solver.b0)))

        if niter is None:
            niter = 15
            warnings.warn(f"optfilt: Using the default number of iterations {niter}.")

        for idx in range(niter_masked_cg + niter):
            if idx == niter_masked_cg:
                solver.reset_preconditioner()
                solver.add_preconditioner(prec_main)
                solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
                solver.b_vec = solver.b0
                solver.init_solver(x0=solver.x)

            solver.step()
            if idx >= niter_masked_cg:
                errors.append(solver.err * errors[niter_masked_cg-1])
            else:
                errors.append(solver.err)
            if benchmark:
                if (idx+1)%benchmark==0:
                    chisq = solver.get_chisq()
                    residual = solver.get_residual()
                    chisqs.append(chisq)
                    residuals.append(residual)
                    ps_c_ells.append(ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:]))
                    itnums.append(idx)
                    print(f"optfilt benchmark: \t {chisq if benchmark else ''} \t "
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

def cg_pix_filter_old(alm, theory_cls, b_ell, lmax,
                      icov_pix=None, mask_bool=None, cov_noise_ell=None, minfo=None,
                      include_te=False, niter=None, stype='pcg_pinv', ainfo=None,
                      niter_masked_cg=7, lmax_masked_cg=3000, benchmark=None, 
                      verbose=True, err_tol=1e-15, rtol_icov=1e-2, order=1,
                      swap_bm=True):
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
    icov_pix : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) or (Ny,Nx) ndmap
        An enmap containing the inverse (co-)variance per pixel (zeros in
        unobserved region), in units (e.g. 1/uK^2) consistent with
        the alms and theory_cls. IQ, IU, QU elements can also be specified
        if icov_pix is 4-dimensional. Within ACT and SO, these are sometimes 
        referred to as 'ivar' or 'div'. Can also be a map in Gauss-Legendre
        or HEALPix pixelization. In those cases last dimensions should be 
        'npix' instead of 'Ny,Nx'.
    mask_bool : (ncomp,Ny,Nx) or (Ny,Nx) ndmap
        Boolean mask (True for observed pixels. Geometry must match that of
        'icov_pix'. If not provided, will be determined from 'ivar'.
    cov_noise_ell : (ncomp,ncomp,nells) or (ncomp,nells) array
        Power spectrum of flattened (ivar_pix-weighted) noise in units
        (e.g. 1/uK^2) consistent with the alms and theory_cls. Used to model
        noise that deviates from the white noise described by the icov_pix maps
        as function of multipole.
    minfo : map_utils.MapInfo object
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
        is 15, but this may be too small (unconverged filtering) or too large 
        (wasted iterations) for your application. Test before deciding on 
        this parameter.
    stype : str
        The type of pre-conditioner to use. 
        Choose from 'cg', 'cg_scaled', 'pcg_harm' and 'pcg_pinv'.
        The default is 'pcg_pinv'.
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for internally used alms. Will be determined from the alm
        size if not specified.
    niter_masked_cg : int
        Number of initial iterations using (an expensive) preconditioner for the
        masked pixels.
    lmax_masked_cg : int
        Band-limit for preconditioner for masked solver. Should be set to multipole
        where SNR becomes smaller 1.
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
    swap_bm : bool, optional
        Swap the order of the beam and mask operations. Helps convergence
        with large beams and high SNR data.

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

    cgobj = CGPixFilterOld(ncomp,theory_cls=theory_cls, b_ell=b_ell, lmax=lmax,
                        icov_pix=icov_pix, mask_bool=mask_bool, cov_noise_ell=cov_noise_ell,
                        minfo=minfo, include_te=include_te, rtol_icov=rtol_icov,order=order)
    return cgobj.filter(alm,benchmark=benchmark, verbose=verbose, ainfo=ainfo,
                        niter_masked_cg=niter_masked_cg, lmax_masked_cg=lmax_masked_cg,
                        niter=niter, stype=stype, err_tol=err_tol)
