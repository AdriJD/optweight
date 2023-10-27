import numpy as np
import warnings
from timeit import default_timer as timer

from pixell import curvedsky, utils, enmap, curvedsky

from optweight import map_utils, mat_utils, solvers, preconditioners

class CGPixFilter(object):
    def __init__(self, theory_cls, b_ell, icov_pix, mask_bool,
                 include_te=True, q_low=0, q_high=1, swap_bm=False,
                 scale_a=False):
        """
        Prepare to filter maps using a pixel-space instrument noise model
        and a harmonic space signal model. 

        Parameters
        ----------
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
        icov_pix : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) ndmap
            An enmap containing the inverse (co-)variance per pixel in units
            (e.g. 1/uK^2) consistent with the alms and theory_cls. IQ, IU, QU
            elements can also be specified if icov_pix is 4-dimensional.
        mask_bool : (ncomp,Ny,Nx) or (Ny,Nx) ndmap
            Boolean mask (True for observed pixels). Geometry must match that of
            'icov_pix'.
        include_te : bool, optional
            Whether or not to jointly filter T,E,B maps by accounting for the
            signal TE correlation. If True, the returned alms will be optimally
            filtered, but the "T" and "E" maps will not be pure-T and pure-E.
        q_low : float or (ncomp) array, optional
            Pixels in icov map with values below this quantile are thresholded.
            May be set per polarization.
        q_high : float or (ncomp) array, optional
            Pixels in icov map with values above this quantile are thresholded.
            May be set  per polarization.                
        swap_bm : bool, optional
            Swap the order of the beam and mask operations. Helps convergence
            with large beams and high SNR data.
        scale_a : bool, optional
            If set, scale the A matrix to localization of N^-1 term. This may
            help convergence with small beams and high SNR data.
        """

        if np.any(np.logical_not(np.isfinite(b_ell))): raise Exception

        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception

        shape_in = icov_pix.shape[-2:]
        
        icov_pix = mat_utils.atleast_nd(icov_pix, 3)
        mask_bool = mat_utils.atleast_nd(mask_bool.astype(bool, copy=False), 3)        
                            
        ncomp = icov_pix.shape[0]
        if mask_bool.shape[0] == 1:
            mask_bool = (np.ones(3)[:,np.newaxis,np.newaxis] * mask_bool).astype(bool)

        for mtype in ['CC', 'fejer1',]:
            try:
                minfo = map_utils.match_enmap_minfo(
                    icov_pix.shape, icov_pix.wcs, mtype=mtype)
            except ValueError:
                continue
            else:
                break
            
        lmax = map_utils.minfo2lmax(minfo)
        icov_pix = map_utils.view_1d(icov_pix, minfo)
        mask_bool = map_utils.view_1d(mask_bool, minfo)        

        if q_low != 0 or q_high != 1:
            icov_pix = map_utils.threshold_icov(icov_pix, q_low=q_low, q_high=q_high)
                                                
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
                
        if b_ell.ndim == 1:
            b_ell = b_ell[np.newaxis] * np.asarray((1, 1, 1)[:ncomp])[:,np.newaxis]
        elif b_ell.ndim == 2:
            if b_ell.shape[0] != ncomp: raise Exception
        else:
            raise ValueError

        b_ell = np.ascontiguousarray(b_ell[:,:lmax+1])
        
        if scale_a:
            sfilt = mat_utils.matpow(b_ell, -0.5)
            lmax_mg = 3000
        else:
            sfilt = None
            lmax_mg = 6000

        ainfo = curvedsky.alm_info(lmax)

        if ncomp == 1:
            spin = 0
        elif ncomp == 3:
            spin = [0, 2]

        prec_pinv = preconditioners.PseudoInvPreconditioner(
            ainfo, icov_ell, icov_pix, minfo, spin, b_ell=b_ell, sfilt=sfilt)

        prec_masked_cg = preconditioners.MaskedPreconditionerCG(
            ainfo, icov_ell, spin, mask_bool[0].astype(bool), minfo, lmax=lmax,
            nsteps=15, lmax_r_ell=None, sfilt=sfilt)

        prec_masked_mg = preconditioners.MaskedPreconditioner(
            ainfo, icov_ell[0:1,0:1], 0, mask_bool[0], minfo,
            min_pix=1000, n_jacobi=1, lmax_r_ell=lmax_mg,
            sfilt=None if sfilt is None else sfilt[0:1,0:1])

        self.shape_in = shape_in
        self.icov_ell = icov_ell
        self.icov_pix = icov_pix
        self.mask_bool = mask_bool
        self.minfo = minfo
        self.b_ell = b_ell
        self.sfilt = sfilt
        self.ncomp = ncomp
        self.swap_bm = swap_bm
        self.lmax = lmax
        self.ainfo = ainfo
        self.spin = spin
        self.prec_pinv = prec_pinv
        self.prec_masked_cg = prec_masked_cg
        self.prec_masked_mg = prec_masked_mg   

    def filter(self, imap, niter=None, niter_masked_cg=5, 
               benchmark=False, verbose=True, err_tol=1e-15):
        """
        Filter a map using a pixel-space instrument noise model
        and a harmonic space signal model.

        Parameters
        ----------
        imap : (ncomp, Ny, Nx) ndmap array
            Input data.
        niter : int
            The number of Conjugate Gradient iterations to be performed. The default
            is 15, but this may be too small (unconverged filtering) or too large 
            (wasted iterations) for your application. Test before deciding on 
            this parameter.
        niter_masked_cg : int
            Number of initial iterations using (an expensive) preconditioner for the
            masked pixels.
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

        assert imap.shape == (self.ncomp,) + self.shape_in 

        imap = map_utils.view_1d(imap.astype(self.icov_pix.dtype, copy=False), self.minfo)

        solver = solvers.CGWienerMap.from_arrays(imap, self.minfo, self.ainfo, self.icov_ell, 
                                                 self.icov_pix, b_ell=self.b_ell,
                                                 draw_constr=False, mask_pix=self.mask_bool,
                                                 swap_bm=self.swap_bm, spin=self.spin,
                                                 sfilt=self.sfilt)
                
        solver.add_preconditioner(self.prec_pinv)
        solver.add_preconditioner(self.prec_masked_cg)
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
            warnings.warn(f"optweight: Using the default number of iterations :"\
                          f"{niter_cg=} + {niter=}.")

        for idx in range(niter_masked_cg + niter):
            if idx == niter_masked_cg:
                solver.reset_preconditioner()
                solver.add_preconditioner(self.prec_pinv)
                solver.add_preconditioner(self.prec_masked_mg, sel=np.s_[0])
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
                print(f"optweight step {solver.i} / {niter_cg + niter}, error {errors[-1]:.2e}, time {t_eval:.3f} s")
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
