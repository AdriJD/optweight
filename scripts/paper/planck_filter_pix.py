import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import argparse

import healpy as hp
from astropy.io import fits
from pixell import curvedsky, enplot, utils

from optweight import sht, map_utils, mat_utils, solvers, operators, preconditioners

opj = os.path.join
np.random.seed(39)

def get_planck_b_ell(rimo_file, lmax):
    '''
    Return b_ell.

    Parameters
    ----------
    rimo_file : str
        Path to RIMO beam file.
    lmax : int
        Truncate to this lmax.
    
    Returns
    -------
    b_ell : (npol, nell)
    '''
    
    with fits.open(rimo_file) as hdul:        
        b_ell_T = hdul[1].data['T']
        b_ell_E = hdul[1].data['E']
        b_ell_B = hdul[1].data['B']
    
    b_ell = np.zeros((3, lmax+1))
    b_ell[0] = b_ell_T[:lmax+1]
    b_ell[1] = b_ell_E[:lmax+1]
    b_ell[2] = b_ell_B[:lmax+1]

    return b_ell

def main(basedir, draw_constr=False, test_conv=False, niter_cg=20, niter_mg=40,
         no_masked_prec=False, pol_mg=False, use_prec_harm=False, noise_scaling=None,
         no_beam=False, lmax_masked_cg=1500, write_steps=False):
    '''
    
    Parameters
    ----------
    basedir : str
        Output directory.
    draw_constr : bool, optional
        Draw constrained realization.
    test_conv : bool, optional
        Replace b with A(x) for know LCDM x.
    niter_cg : int, optional
        Number of CG steps with nested CG as masked prec.
    niter_mg : int, optional
        Number of CG steps with multigrid as masked prec.
    no_masked_prec : bool, optional
        Do not use the preconditioners for masked pixels.
    pol_mg : bool, optional
        Use the multigrid precondioner for polarization as well
    use_prec_harm : bool, optional
        Use the harmonic preconditioner instead of the pseudo inverse.
    noise_scaling : float, optional
        Scale noise covariance by this number.
    no_beam : bool, optional
        Turn off the beam.
    lmax_masked_cg : int, optional
        Lmax used for nested masked cg preconditioner.
    write_steps : bool, optional
        Write CG steps to disk as alms.
    '''
    
    if test_conv:
        draw_constr = True

    lmax = 2500

    #maskdir = '/home/adriaand/project/actpol/20201009_pcg_planck/meta'
    maskdir = '/mnt/home/aduivenvoorden/project/actpol/20201009_pcg_planck/meta'
    imgdir = opj(basedir, 'img')
    odir = opj(basedir, 'out')

    utils.mkdir(imgdir)
    utils.mkdir(odir)

    # Load II, IQ, IU, QQ, QU, UU cov.
    cov = hp.read_map(opj(maskdir, 'HFI_SkyMap_100_2048_R3.01_full.fits'), field=(4, 5, 6, 7, 8, 9))
    cov *= 1e12 # Convert from K^2 to uK^2.

    if noise_scaling is not None:
        cov *= noise_scaling

    cov, minfo = map_utils.healpix2gauss(cov, 2*lmax, area_pow=-1)
    cov_pix = np.zeros((3, 3, cov.shape[-1]))
    cov_pix[0,0] = cov[0]
    cov_pix[0,1] = cov[1]
    cov_pix[0,2] = cov[2]
    cov_pix[1,0] = cov[1]
    cov_pix[1,1] = cov[3]
    cov_pix[1,2] = cov[4]
    cov_pix[2,0] = cov[2]
    cov_pix[2,1] = cov[4]
    cov_pix[2,2] = cov[5]

    # NOTE try thresholding low *cov* values.
    #cov_pix = map_utils.round_icov_matrix(cov_pix, rtol=1e-1, threshold=True)

    for idx in range(3):
        for jdx in range(3):
            fig, ax = plt.subplots(dpi=300)
            im = ax.imshow(cov_pix[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]), interpolation='none')
            fig.colorbar(im, ax=ax)
            fig.savefig(opj(imgdir, 'cov_real_{}_{}'.format(idx, jdx)))
            plt.close(fig)

    for idx in range(3):
        for jdx in range(3):
            fig, ax = plt.subplots(dpi=300)
            im = ax.imshow(np.log10(np.abs(cov_pix[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]))), interpolation='none')
            fig.colorbar(im, ax=ax)
            fig.savefig(opj(imgdir, 'cov_real_log_{}_{}'.format(idx, jdx)))
            plt.close(fig)

    icov_pix = mat_utils.matpow(cov_pix, -1)
    icov_pix = map_utils.round_icov_matrix(icov_pix, rtol=1e-2)

    for idx in range(3):
        for jdx in range(3):
            fig, ax = plt.subplots(dpi=300)
            im = ax.imshow(np.log10(np.abs(icov_pix[idx,jdx].reshape(minfo.nrow, minfo.nphi[0]))),
                           interpolation='none')
            fig.colorbar(im, ax=ax)
            fig.savefig(opj(imgdir, 'icov_{}_{}'.format(idx, jdx)))
            plt.close(fig)

    mask_I = hp.read_map(opj(maskdir, 'COM_Mask_Likelihood-temperature-100-hm2_2048_R3.00.fits'), field=0)
    mask_I, _ = map_utils.healpix2gauss(mask_I[np.newaxis,:], 2*lmax, area_pow=0)
    mask_I[mask_I>=0.1] = 1
    mask_I[mask_I<0.1] = 0

    mask_P = hp.read_map(opj(maskdir, 'COM_Mask_Likelihood-polarization-100-hm2_2048_R3.00.fits'), field=0)
    mask_P, _ = map_utils.healpix2gauss(mask_P[np.newaxis,:], 2*lmax, area_pow=0)
    mask_P[mask_P>=0.1] = 1
    mask_P[mask_P<0.1] = 0

    print('fsky T', np.sum(map_utils.inv_qweight_map(mask_I, minfo, qweight=True)) / 4 / np.pi)
    print('fsky P', np.sum(map_utils.inv_qweight_map(mask_P, minfo, qweight=True)) / 4 / np.pi)

    mask_gl = np.zeros((3, mask_I.shape[-1]))
    mask_gl[0] = mask_I
    mask_gl[1] = mask_P
    mask_gl[2] = mask_P

    # Load beam.
    b_ell = get_planck_b_ell(opj(maskdir, 'BeamWf_HFI_R3.01', 'Bl_TEB_R3.01_fullsky_100x100.fits'), lmax)
    if no_beam:
        b_ell = np.ones_like(b_ell)

    # Preprare spectrum. Input file is Dls in uk^2.
    c_ell = np.loadtxt(
        opj(maskdir, 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'),
        skiprows=1, usecols=[1, 2, 3, 4]) #  TT, TE, EE, BB.
    c_ell = c_ell.T
    ells = np.arange(lmax + 1)
    dells = ells * (ells + 1)  / 2 / np.pi
    cov_ell = np.zeros((3, 3, lmax + 1))
    cov_ell[0,0,2:] = c_ell[0,:lmax-1] 
    cov_ell[0,1,2:] = c_ell[1,:lmax-1] 
    cov_ell[1,0,2:] = c_ell[1,:lmax-1] 
    cov_ell[1,1,2:] = c_ell[2,:lmax-1] 
    cov_ell[2,2,2:] = c_ell[3,:lmax-1] 

    fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
    for idxs, ax in np.ndenumerate(axs):
        axs[idxs].plot(cov_ell[idxs])
    fig.savefig(opj(imgdir, 'cov_ell'))
    plt.close(fig)

    cov_ell[...,1:] /= dells[1:]

    icov_ell = np.ones((3, 3, lmax + 1))
    for lidx in range(icov_ell.shape[-1]):
        if lidx < 2:
            # Set monopole and dipole to zero.
            icov_ell[:,:,lidx] = 0
        else:
            icov_ell[:,:,lidx] = np.linalg.inv(cov_ell[:,:,lidx])

    # Draw alms.
    alm, ainfo = curvedsky.rand_alm(cov_ell, return_ainfo=True)

    if not test_conv:
        for pidx in range(alm.shape[0]):
            hp.almxfl(alm[pidx], b_ell[pidx], inplace=True)
    # Draw map-based noise and add to alm.
    noise = map_utils.rand_map_pix(cov_pix)
    alm_noise = alm.copy()
    sht.map2alm(noise, alm_noise, minfo, ainfo, [0,2], adjoint=False)
    nl = ainfo.alm2cl(alm_noise[:,None,:], alm_noise[None,:,:])

    fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
    for idxs, ax in np.ndenumerate(axs):
        axs[idxs].plot(nl[idxs])
    fig.savefig(opj(imgdir, 'n_ell'))
    plt.close(fig)

    if not test_conv:
        
        imap = np.zeros((3, minfo.npix))
        sht.alm2map(alm, imap, ainfo, minfo, [0, 2])
        imap += noise 
        imap *= mask_gl

    niter = niter_cg + niter_mg

    ps_c_ell = np.zeros((niter, 3, 3, lmax + 1))
    cg_errors = np.zeros(niter + 1)
    chisqs = np.zeros(niter + 1)
    errors = np.zeros((4, niter + 1)) # Total, I, E, B.
    residuals = np.zeros(niter + 1)
    times = np.zeros(niter)
    qforms = np.zeros(niter)

    solver = solvers.CGWienerMap.from_arrays(imap, minfo, ainfo, icov_ell, icov_pix, minfo, b_ell=b_ell,
                                             draw_constr=draw_constr, mask_pix=mask_gl, spin=[0, 2],
                                             swap_bm=True)
    
    prec_pinv = preconditioners.PseudoInvPreconditioner(
        ainfo, icov_ell, icov_pix, minfo, [0, 2], b_ell=b_ell)

    prec_harm = preconditioners.HarmonicPreconditioner(
        ainfo, icov_ell, b_ell=b_ell, icov_pix=icov_pix, minfo=minfo)

    prec_masked_cg = preconditioners.MaskedPreconditionerCG(
        ainfo, icov_ell, [0, 2], mask_gl.astype(bool), minfo, lmax=lmax_masked_cg, 
        nsteps=15, lmax_r_ell=None)

    if pol_mg:
        prec_masked_mg = preconditioners.MaskedPreconditioner(
            ainfo, icov_ell, [0, 2], mask_gl[0].astype(bool), minfo,
            min_pix=10000, n_jacobi=1, lmax_r_ell=6000)
    else:
        print('normal mg')
        prec_masked_mg = preconditioners.MaskedPreconditioner(
            ainfo, icov_ell[0:1,0:1], 0, mask_gl[0].astype(bool), minfo,
            min_pix=10000, n_jacobi=1, lmax_r_ell=6000)

    if use_prec_harm:
        solver.add_preconditioner(prec_harm)
    else:
        print('normal pinv')
        solver.add_preconditioner(prec_pinv)

    if not no_masked_prec:
        print('normal cg')
        solver.add_preconditioner(prec_masked_cg)
    solver.init_solver()

    if test_conv:
        print('test_conv')
        # Replace b with A(x) such that we can compute error.
        solver.b_vec = solver.A(alm)
        solver.b0 = solver.b_vec.copy()
        solver.init_solver()

    cg_errors[0] = 1
    #chisqs[0] = solver.get_chisq()
    errors[0,0] = np.sqrt(solver.dot(alm, alm))
    errors[1,0] = np.sqrt(solver.dot(alm[0], alm[0]))
    errors[2,0] = np.sqrt(solver.dot(alm[1], alm[1]))
    errors[3,0] = np.sqrt(solver.dot(alm[2], alm[2]))
    residuals[0] = solver.get_residual()
                
    for idx in range(niter_cg):
        t0 = time.time()
        solver.step()
        dt = time.time() - t0                
        if write_steps:
            hp.write_alm(opj(odir, f'alm_x_{idx}.fits'), solver.get_wiener(),
                         overwrite=True)
        residual = solver.get_residual()
        #chisq = solver.get_chisq()        
        diff = solver.x - alm
        
        cg_errors[idx+1] = solver.err
        errors[0,idx+1] = np.sqrt(solver.dot(diff, diff))
        errors[1,idx+1] = np.sqrt(solver.dot(diff[0], diff[0]))
        errors[2,idx+1] = np.sqrt(solver.dot(diff[1], diff[1]))
        errors[3,idx+1] = np.sqrt(solver.dot(diff[2], diff[2]))
        residuals[idx+1] = residual
        #chisqs[idx+1] = chisq
        times[idx] = dt
        qforms[idx] = solver.get_qform()
        ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])
        
        #print(f'{solver.i}, cg_err : {solver.err}, chisq : {chisq}, residual : {residual}, '
        #      f'err[0] : {errors[1,idx+1]}, err[1] : {errors[2,idx+1]}, '
        #      f'err[2] : {errors[3,idx+1]}, qform = {qforms[idx]}, dt : {dt}')
        print(f'{solver.i}, cg_err : {solver.err}, residual : {residual}, '
              f'err[0] : {errors[1,idx+1]}, err[1] : {errors[2,idx+1]}, '
              f'err[2] : {errors[3,idx+1]}, qform = {qforms[idx]}, dt : {dt}')

    solver.reset_preconditioner()
    if use_prec_harm:
        solver.add_preconditioner(prec_harm)
    else:
        print('normal pinv 2')
        solver.add_preconditioner(prec_pinv)

    if not no_masked_prec:
        if pol_mg:
            solver.add_preconditioner(prec_masked_mg)
        else:
            print('normal mg 2')
            solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])
    solver.b_vec = solver.b0
    solver.init_solver(x0=solver.x)

    for idx in range(niter_cg, niter_cg + niter_mg):

        t0 = time.time()
        solver.step()
        dt = time.time() - t0                
        if write_steps:
            hp.write_alm(opj(odir, f'alm_x_{idx}.fits'), solver.get_wiener(),
                         overwrite=True)
        residual = solver.get_residual()
        #chisq = solver.get_chisq()        
        diff = solver.x - alm
        
        cg_errors[idx+1] = solver.err * cg_errors[niter_cg]
        errors[0,idx+1] = np.sqrt(solver.dot(diff, diff))
        errors[1,idx+1] = np.sqrt(solver.dot(diff[0], diff[0]))
        errors[2,idx+1] = np.sqrt(solver.dot(diff[1], diff[1]))
        errors[3,idx+1] = np.sqrt(solver.dot(diff[2], diff[2]))
        residuals[idx+1] = residual
        #chisqs[idx+1] = chisq
        times[idx] = dt
        qforms[idx] = solver.get_qform()
        ps_c_ell[idx,...] = ainfo.alm2cl(solver.x[:,None,:], solver.x[None,:,:])
        
        #print(f'{solver.i}, cg_err : {solver.err}, chisq : {chisq}, residual : {residual}, '
        #      f'err[0] : {errors[1,idx+1]}, err[1] : {errors[2,idx+1]}, '
        #      f'err[2] : {errors[3,idx+1]}, qform = {qforms[idx]}, dt : {dt}')
        print(f'{solver.i}, cg_err : {solver.err}, residual : {residual}, '
              f'err[0] : {errors[1,idx+1]}, err[1] : {errors[2,idx+1]}, '
              f'err[2] : {errors[3,idx+1]}, qform = {qforms[idx]}, dt : {dt}')

    np.save(opj(odir, 'ps_c_ell'), ps_c_ell)
    np.save(opj(odir, 'n_ell'), nl)
    np.save(opj(odir, 'cov_ell'), cov_ell)
    np.save(opj(odir, 'b_ell'), b_ell)
    np.save(opj(odir, 'cg_errors'), cg_errors)
    np.save(opj(odir, 'errors'), errors)
    np.save(opj(odir, 'residuals'), residuals)
    np.save(opj(odir, 'times'), times)
    #np.save(opj(odir, 'chisqs'), chisqs)

    fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True, squeeze=False)
    for ax in axs.ravel():
        ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, niter)])
    for idx in range(niter):
        for aidxs, ax in np.ndenumerate(axs):
            axs[aidxs].plot(ells, dells * ps_c_ell[idx,aidxs[0],aidxs[1]],
                            lw=0.5)
    for aidxs, ax in np.ndenumerate(axs):
        axs[aidxs].plot(ells, dells * cov_ell[aidxs[0],aidxs[1]], lw=0.5, color='black', ls=':')
    axs[0,0].set_ylim(0, 1e4)
    fig.savefig(opj(imgdir, 'ps_c_ell'))
    plt.close(fig)

    for idxs, ax in np.ndenumerate(axs):
        if idxs[0] == idxs[1]:
            ax.set_yscale('log')
            ax.set_ylim(ax.get_ylim())
            ax.plot(ells, dells * nl[idxs[0],idxs[1]] / b_ell[0] ** 2, lw=1, color='black')
    axs[0,0].set_ylim(0.1, 1e4)
    fig.savefig(opj(imgdir, 'ps_c_ell_log'))
    plt.close(fig)

    # Plot input sky signal.
    omap = curvedsky.make_projectable_map_by_pos(
        [[np.pi/2, -np.pi/2],[-np.pi, np.pi]], lmax, dims=(alm.shape[0],))
    omap = curvedsky.alm2map(alm, omap)
    plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=4)
    enplot.write(opj(imgdir, 'alm_in'), plot)

    if not test_conv:
        # Plot input data.
        alm_data = alm.copy()
        sht.map2alm(imap, alm, minfo, ainfo, [0, 2])
        curvedsky.alm2map(alm, omap, ainfo=ainfo)
        plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=4)
        enplot.write(opj(imgdir, 'imap'), plot)
    
    # Plot result
    omap = curvedsky.alm2map(solver.get_wiener(), omap)
    plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, range='250:5', downgrade=4)
    if draw_constr:
        enplot.write(opj(imgdir, 'alm_constr'), plot)
    else:
        enplot.write(opj(imgdir, 'alm_out'), plot)

    if not draw_constr:
        omap = curvedsky.alm2map(solver.get_icov(), omap)
        plot = enplot.plot(omap, colorbar=True, font_size=50, grid=False, downgrade=4)
        enplot.write(opj(imgdir, 'alm_icov'), plot)

    if test_conv:
        omap = curvedsky.alm2map(solver.get_wiener() - alm, omap)
        for pidx in range(3):
            plot = enplot.plot(omap[pidx], colorbar=True, font_size=50, grid=False, downgrade=4)
            enplot.write(opj(imgdir, f'alm_diff_{pidx}'), plot)

    b_out = solver.A(solver.x)
    omap_b_out = curvedsky.alm2map(b_out, omap.copy())
    omap_b_in = curvedsky.alm2map(solver.b0, omap.copy())
    omap_b_diff = curvedsky.alm2map(b_out - solver.b0, omap.copy())

    if not draw_constr:

        for pidx in range(alm.shape[0]):

            plot = enplot.plot(omap_b_out[pidx], colorbar=True, grid=False, font_size=50, downgrade=4)
            enplot.write(opj(imgdir, 'b_out_{}'.format(pidx)), plot)

            plot = enplot.plot(omap_b_in[pidx], colorbar=True, grid=False, font_size=50, downgrade=4)
            enplot.write(opj(imgdir, 'b_{}'.format(pidx)), plot)

            plot = enplot.plot(omap_b_diff[pidx], colorbar=True, grid=False, font_size=50, downgrade=4)
            enplot.write(opj(imgdir, 'b_diff_{}'.format(pidx)), plot)

    # Save the output.
    if not test_conv:
        map_utils.write_map(opj(odir, 'imap'), imap, minfo)
    hp.write_alm(opj(odir, 'alm_in.fits'), alm, overwrite=True)
    if not draw_constr:
        hp.write_alm(opj(odir, 'alm_icov.fits'), solver.get_icov(), overwrite=True)
        hp.write_alm(opj(odir, 'alm_out.fits'), solver.get_wiener(), overwrite=True)
    if draw_constr:
        hp.write_alm(opj(odir, 'alm_constr.fits'), solver.get_wiener(), overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=str,
                        help='Output directory')
    parser.add_argument('--draw-constr', action='store_true',
                        help='Draw a constrained realization')
    parser.add_argument('--test-conv', action='store_true',
                        help='Replace b with A(x) for known LCDM x.')
    parser.add_argument('--niter-cg', type=int, default=20, 
                        help='Number of CG steps with nested CG as masked prec.')
    parser.add_argument('--niter-mg', type=int, default=40, 
                        help='Number of CG steps with multigrid as masked prec.')
    parser.add_argument('--no-masked-prec', action='store_true', 
                        help='Do not use the preconditioners for masked pixels.')
    parser.add_argument('--pol-mg', action='store_true', 
                        help='Use the multigrid precondioner for polarization as well.')
    parser.add_argument('--use-prec-harm', action='store_true', 
                        help='Use the harmonic preconditioner instead of the pseudo inverse.')
    parser.add_argument('--noise-scaling', type=float, 
                        help='Scale noise covariance by this number')
    parser.add_argument('--no-beam', action='store_true', 
                        help='Turn off beam.')
    parser.add_argument('--lmax-masked-cg', type=int, default=1500,
                        help="lmax_masked_cg")
    parser.add_argument('--write-steps', action='store_true',
                        help="Write x to disk each step.")
    args = parser.parse_args()

    print(args)

    main(args.basedir, draw_constr=args.draw_constr, test_conv=args.test_conv,
         niter_cg=args.niter_cg, niter_mg=args.niter_mg, no_masked_prec=args.no_masked_prec,
         pol_mg=args.pol_mg, use_prec_harm=args.use_prec_harm, noise_scaling=args.noise_scaling,
         no_beam=args.no_beam, lmax_masked_cg=args.lmax_masked_cg, write_steps=args.write_steps)
