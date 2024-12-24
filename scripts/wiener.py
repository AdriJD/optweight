import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

from pixell import enmap, curvedsky, sharp, enplot
from enlib import cg
from nawrapper import maptools

from optweight import solvers
from optweight import sht
from optweight import map_utils

opj = os.path.join
#np.random.seed(39)
np.random.seed(50)

lmax = 2500

basedir = '/home/adriaand/project/actpol/20200916_pcg_icov'
maskdir = opj(basedir, 'mask')
imgdir = opj(basedir, 'img')
plot_opts = {'colorbar' : True, 'mask' : np.nan, 'autocrop' : False}

#mask = enmap.read_map(opj(maskdir, 'deep1_mr3f_20190502_190904_master_apo_w0.fits'))
#mask = enmap.read_map(opj(maskdir, 'boss_mr3f_20190502_190904_master_apo_w0.fits'))
mask = enmap.read_map(opj(maskdir, 'deep56_mr3f_20190502_190904_master_apo_w0.fits'))
# Add some apodized masked point sources.
npoint = 100

mask_ps = enmap.ones(mask.shape, wcs=mask.wcs)
for i in range(npoint):
    idx_y = np.random.randint(low=0, high=mask.shape[-2])
    idx_x = np.random.randint(low=0, high=mask.shape[-1])
    mask_ps[idx_y,idx_x] = 0
mask *= maptools.apod_C2(mask_ps, 1)
mask_ps[:] = 1
npoint = 5
for i in range(npoint):
    idx_y = np.random.randint(low=0, high=mask.shape[-2])
    idx_x = np.random.randint(low=0, high=mask.shape[-1])
    mask_ps[idx_y,idx_x] = 0
mask *= maptools.apod_C2(mask_ps, 5)

ny, nx = mask.shape
mask[ny//2-int(0.05*ny):ny//2+int(0.05*ny),:] = 0
mask_p = mask.copy()
mask_p[:,nx//4-int(0.05*nx):nx//4+int(0.05*nx)] = 0

#alm_mask = curvedsky.map2alm_cyl(mask, lmax=lmax)
#alm_mask_p = curvedsky.map2alm_cyl(mask_p, lmax=lmax)
#omap = enmap.zeros(mask.shape, mask.wcs)
#omap = curvedsky.alm2map(alm_mask, omap)
#plot = enplot.plot(omap, **plot_opts)
#enplot.write(opj(imgdir, 'alm_mask'), plot)
#omap = curvedsky.alm2map(alm_mask_p, omap)
#plot = enplot.plot(omap, **plot_opts)
#enplot.write(opj(imgdir, 'alm_mask_p'), plot)

plot = enplot.plot(mask, **plot_opts)
enplot.write(opj(imgdir, 'mask'), plot)
plot = enplot.plot(mask_p, **plot_opts)
enplot.write(opj(imgdir, 'mask_p'), plot)


#dec_range, ra_range = enmap.pix2sky(mask.shape, mask.wcs, [[0,ny], [0,nx]])
#theta_range = np.pi / 2 - dec_range

# Factor 2 in lmax is because of pixel cov.
mask_gauss, map_info = map_utils.enmap2gauss(mask, lmax=2*lmax)
mask_p_gauss, _ = map_utils.enmap2gauss(mask_p, lmax=2*lmax)

nrings = map_info.nrow
nphi = map_info.nphi[0]

#nrings = 2 * int(np.floor(lmax / 2)) + 1
#nphi = 2 * lmax + 1

#map_info = sharp.map_info_gauss_legendre(nrings, nphi)
# Truncate to theta in mask.
#theta_mask = np.logical_and(map_info.theta >= theta_range.min(), map_info.theta <= theta_range.max())
#theta = map_info.theta[theta_mask]
#weight = map_info.weight[theta_mask]
#stride = map_info.stride[theta_mask]
#offsets = map_info.offsets[theta_mask]
#offsets -= offsets[0]
#nrings = theta.size
# Create new map_info corresponding to truncated theta range.
#map_info = sharp.map_info(theta, nphi=nphi, phi0=0, offsets=offsets, stride=stride, weight=weight)

# Create random alms.
# alm_info = sharp.alm_info(lmax=lmax)
# ells = np.arange(lmax + 1)
# ps = np.zeros((3, 3, lmax + 1))
# ps[0,0,:2] = 1
# ps[1,1,:2] = 1
# ps[2,2,:2] = 1
# ps[0,0,2:] = ells[2:] ** (-2.)
# ps[1,1,2:] = ells[2:] ** (-2.)
# ps[2,2,2:] = ells[2:] ** (-2.)
# #ps[2,2,1:300] = 1e-5 # i.e. Lensing
# #ps[2,2,300:] = 1e-5 * (ells[300:] / ells[300]) ** (-2.) # i.e. Lensing
# ps[0,1,2:] = np.sin(0.01 * ells[2:]) * ells[2:] ** (-2.) * 0.1
# ps[1,0,2:] = np.sin(0.01 * ells[2:]) * ells[2:] ** (-2.) * 0.1
# #ps[1:,1:,:100] *= 0.1
# alm, alm_info = curvedsky.rand_alm(ps, ainfo=alm_info, return_ainfo=True)
# alm[:,:2] = 0
# alm[:,alm_info.lmax+1] = 0

#####
# Preprare spectrum. Input file is Dls in uk^2.
basedir_pl = '/home/adriaand/project/actpol/20201009_pcg_planck'
maskdir_pl = opj(basedir_pl, 'meta')
c_ell = np.loadtxt(
    opj(maskdir_pl, 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'),
    skiprows=1, usecols=[1, 2, 3, 4]) #  TT, TE, EE, BB.
c_ell = c_ell.T
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi
cov_ell = np.zeros((3, 3, lmax + 1))
cov_ell[0,0,2:] = c_ell[0,:lmax-1] 
#cov_ell[0,1,2:] = c_ell[1,:lmax-1] 
#cov_ell[1,0,2:] = c_ell[1,:lmax-1] 
#cov_ell[1,1,2:] = c_ell[2,:lmax-1] 
#cov_ell[2,2,2:] = c_ell[3,:lmax-1] 
cov_ell[1,1,2:] = c_ell[0,:lmax-1] 
cov_ell[2,2,2:] = c_ell[0,:lmax-1] 
cov_ell[...,1:] /= dells[1:]



icov_ell = np.ones((3, 3, lmax + 1))
for lidx in range(icov_ell.shape[-1]):
    if lidx < 2:
        # Set monopole and dipole to zero.
        icov_ell[:,:,lidx] = 0
    else:
        icov_ell[:,:,lidx] = np.linalg.inv(cov_ell[:,:,lidx])

# Draw alms.
alm, alm_info = curvedsky.rand_alm(cov_ell, return_ainfo=True)
###

icov_pix = np.zeros((3, 3, map_info.npix)) 
#sht.alm2map(alm_mask, icov_pix[0,0], alm_info, map_info, 0)
#sht.alm2map(alm_mask_p, icov_pix[1,1], alm_info, map_info, 0)
icov_pix[0,0] = mask_p_gauss
icov_pix[1,1] = mask_p_gauss
icov_pix[2,2] = mask_p_gauss
#icov_pix[2,2] = icov_pix[1,1]
icov_pix[icov_pix > 1] = 1
icov_pix = np.abs(icov_pix) * 0.01
#icov_pix[np.abs(icov_pix) < 0] = 0
icov_pix[np.abs(icov_pix) < 0.0001] = 0

# Plot icov.
icov_2d = icov_pix[0,0].reshape((nrings, nphi))
# Find first and last nonzero phi.
start = None
end = None
#for pidx in range(icov_2d.shape[1]):
#    if np.any(icov_2d[:,pidx]):
#        start = pidx
#        break
#for pidx in range(icov_2d.shape[1]):
#    if np.any(icov_2d[:,-(pidx+1)]):
#        end = -(pidx + 1)
#        break
#print(start, end)
fig, ax = plt.subplots(dpi=300)
im = ax.imshow(np.log10(np.abs(icov_2d[:,end:start:-1])), interpolation='none')
fig.colorbar(im, ax=ax)
fig.savefig(opj(imgdir, 'icov'))
plt.close(fig)

# Plot alm.
omap_gauss = np.zeros((3, map_info.npix))
sht.alm2map(alm, omap_gauss, alm_info, map_info, [0,2])
for pidx in range(3):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(omap_gauss.reshape((3,nrings,nphi))[pidx,:,end:start:-1], interpolation='none')
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(imgdir, 'alm_in_gauss_{}'.format(pidx)))
    plt.close(fig)

# icov_ell = np.ones((3, 3, lmax + 1))
# for lidx in range(icov_ell.shape[-1]):
#     if lidx < 2:
#         # Set monopole and dipole to zero.
#         icov_ell[:,:,lidx] = 0
#     else:
#         icov_ell[:,:,lidx] = np.linalg.inv(ps[:,:,lidx])

#for aidx, alpha in enumerate([1e10, 1e8, 1e6, 1e4, 1e2, 1]):
for sidx in range(1):
    for aidx, alpha in enumerate([1]):
        if alpha == 1:
            iters = 150
        else:
            iters = 10

        if aidx == 0:
            x0 = None
        else:
            x0 = solver.x    

        #def A(alm): return pcg.a_matrix(alm, icov_pix / alpha, icov_ell, map_info, alm_info, [0,2])
        #ps_pre = ps.copy()
        #ps_pre[:,:,100:] = ps_pre[0,0,100]
        #def M(alm): return alm_info.lmul(alm, ps_pre)
        #b = pcg.b_vec(alm, icov_pix, map_info, alm_info, [0,2])
        #b = pcg.b_vec_constr(alm, icov_pix / alpha, icov_ell, map_info, alm_info, [0,2])

        solver = solvers.CGWiener.from_arrays(
            alm, alm_info, icov_ell, icov_pix / alpha, map_info, b_ell=None,
            draw_constr=False, prec='harmonic', x0=x0)


        b_copy = solver.b.copy()
        #dot = pcg.contract_almxblm
        #solver = cg.CG(A, b, dot=dot)
        #solver = cg.CG(A, b, dot=dot, x0=x0)
        #solver = cg.CG(A, b, dot=dot, M=M)

        while solver.i < iters:
            solver.step()
            print(solver.i, solver.err)

    omap = enmap.zeros((3,) + mask.shape, wcs=mask.wcs)

    print('alm_out')
    omap = curvedsky.alm2map(solver.x, omap)
    plot = enplot.plot(omap, **plot_opts)
    enplot.write(opj(imgdir, 'alm_out_{}'.format(sidx)), plot)


print('alm_in')
omap = curvedsky.alm2map(alm, omap)
omap *= mask_p
plot = enplot.plot(omap, **plot_opts)
enplot.write(opj(imgdir, 'alm_in'), plot)

print('alm_out_icov')
x_icov = alm_info.lmul(solver.x, icov_ell)
omap = curvedsky.alm2map(x_icov, omap)
plot = enplot.plot(omap, **plot_opts)
enplot.write(opj(imgdir, 'alm_out_icov'), plot)

print('b')
omap = curvedsky.alm2map(b_copy, omap)
plot = enplot.plot(omap, **plot_opts)
enplot.write(opj(imgdir, 'b'), plot)

print('b_solved')
omap = curvedsky.alm2map(solver.A(solver.x), omap.copy())
#omap = curvedsky.alm2map(pcg.a_matrix(solver.x, icov_pix / alpha, icov_ell, map_info, alm_info, [0,2]), omap)
plot = enplot.plot(omap, **plot_opts)
enplot.write(opj(imgdir, 'b_solved'), plot)
