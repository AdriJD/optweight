import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

opj = os.path.join

basedir = '/home/adriaand/project/actpol/20201208_pcg_act_wavelet_pinv'
imgdir = opj(basedir, 'img')

errors = np.load(opj(imgdir, 'errors.npy'))
residuals = np.load(opj(imgdir, 'residuals.npy'))
residuals /= residuals[:,0][:,np.newaxis]
chisqs = np.load(opj(imgdir, 'chisqs.npy'))
times = np.load(opj(imgdir, 'times.npy'))
#cov_ell = np.load(opj(imgdir, 'cov_ell.npy'))
#n_ell = np.load(opj(imgdir, 'n_ell.npy'))
#ps_c_ell = np.load(opj(imgdir, 'ps_c_ell.npy'))
cumtimes = np.cumsum(times, axis=1)

print(times)
print(cumtimes)

fig, axs = plt.subplots(nrows=2, dpi=300, sharex=True, figsize=(3, 4), 
                        constrained_layout=True)
#for sidx, stype in enumerate(['pcg_harm', 'pcg_pinv']):
for sidx, stype in enumerate(['pcg_pinv', 'pcg_harm']):
    color = 'C{}'.format(sidx+2)
#for sidx, stype in enumerate(['pcg_pinv', 'pcg_harm']):
#    color = 'C{}'.format(sidx+2)

    axs[0].plot(chisqs[sidx], label=stype, color=color)
    axs[1].plot(residuals[sidx], label=stype, color=color)
axs[0].set_ylabel(r'$\chi^2$')
#axs[1].set_ylabel(r'$|\mathbf{A} \mathbf{x} - \mathbf{b}| / |\mathbf{b}|$')
axs[1].set_ylabel(r'$|A x - b| / |b|$')
axs[0].legend(frameon=False)
for ax in axs:
    ax.set_yscale('log')
axs[1].set_xlabel('CG steps')
fig.savefig(opj(imgdir, 'stats_nice'))
plt.close(fig)

fig, axs = plt.subplots(nrows=2, dpi=300, sharex=True, figsize=(3, 4), 
                        constrained_layout=True)
#for sidx, stype in enumerate(['pcg_harm', 'pcg_pinv']):
for sidx, stype in enumerate(['pcg_pinv', 'pcg_harm']):
    color = 'C{}'.format(sidx+2)
    axs[0].plot(cumtimes[sidx], chisqs[sidx], label=stype, color=color)
    axs[1].plot(cumtimes[sidx], residuals[sidx], label=stype, color=color)
axs[0].set_ylabel(r'$\chi^2$')
axs[1].set_ylabel(r'$|A x - b| / |b|$')
axs[0].legend(frameon=False)
for ax in axs:
    ax.set_yscale('log')
axs[1].set_xlabel('Wall time [s]')
fig.savefig(opj(imgdir, 'stats_time'))
plt.close(fig)

fig, axs = plt.subplots(nrows=3, dpi=300, sharex=True, figsize=(3, 5), 
                        constrained_layout=True)
#for sidx, stype in enumerate(['pcg_harm', 'pcg_pinv']):
for sidx, stype in enumerate(['pcg_pinv', 'pcg_harm']):
    color = 'C{}'.format(sidx+2)
    axs[0].plot(cumtimes[sidx], errors[sidx], label=stype, color=color)
    axs[1].plot(cumtimes[sidx], chisqs[sidx], label=stype, color=color)
    axs[2].plot(cumtimes[sidx], residuals[sidx], label=stype, color=color)
axs[0].set_ylabel(r'$\chi^2$')
axs[1].set_ylabel(r'$|A x - b| / |b|$')
axs[0].legend(frameon=False)
for ax in axs:
    ax.set_yscale('log')
axs[1].set_xlabel('Wall time [s]')
fig.savefig(opj(imgdir, 'stats_ext_time'))
plt.close(fig)


exit()

lmax = cov_ell.shape[-1] - 1
ells = np.arange(lmax + 1)
dells = ells * (ells + 1) / 2 / np.pi

fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for idxs, ax in np.ndenumerate(axs):
    #axs[idxs].plot(ells, dells * (cov_ell[idxs] + n_ell[idxs]), label=r'$C_{\ell}$')
    axs[idxs].plot(ells, dells * n_ell[idxs], label=r'$N_{\ell}$')
    axs[idxs].plot(ells, dells * cov_ell[idxs], label=r'$S_{\ell}$')

axs[0,0].legend(frameon=False)
axs[0,0].set_xlabel(r'$\ell$')
axs[1,1].set_xlabel(r'$\ell$')
axs[2,2].set_xlabel(r'$\ell$')
axs[0,0].set_ylabel(r'$D_{\ell}^{TT}$ [$\mu K^2$]')
axs[0,1].set_ylabel(r'$D_{\ell}^{TE}$ [$\mu K^2$]')
axs[0,2].set_ylabel(r'$D_{\ell}^{TB}$ [$\mu K^2$]')
axs[1,1].set_ylabel(r'$D_{\ell}^{EE}$ [$\mu K^2$]')
axs[1,2].set_ylabel(r'$D_{\ell}^{EB}$ [$\mu K^2$]')
axs[2,2].set_ylabel(r'$D_{\ell}^{BB}$ [$\mu K^2$]')

axs[1,0].set_visible(False)
axs[2,0].set_visible(False)
axs[2,1].set_visible(False)
fig.savefig(opj(imgdir, 'tot_ell_nice'))
plt.close(fig)

niter = ps_c_ell.shape[0]
fig, axs = plt.subplots(ncols=3, nrows=3, dpi=300, constrained_layout=True)
for ax in axs.ravel():
    ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, niter)])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
for idx in range(niter):
    for aidxs, ax in np.ndenumerate(axs):
        axs[aidxs].plot(ells, dells * ps_c_ell[idx,aidxs[0],aidxs[1]],
                        lw=0.5)

axs[0,0].set_xlabel(r'$\ell$')
axs[1,1].set_xlabel(r'$\ell$')
axs[2,2].set_xlabel(r'$\ell$')
axs[0,0].set_ylabel(r'$D_{\ell}^{TT}$ [$\mu K^2$]')
axs[0,1].set_ylabel(r'$D_{\ell}^{TE}$ [$\mu K^2$]')
axs[0,2].set_ylabel(r'$D_{\ell}^{TB}$ [$\mu K^2$]')
axs[1,1].set_ylabel(r'$D_{\ell}^{EE}$ [$\mu K^2$]')
axs[1,2].set_ylabel(r'$D_{\ell}^{EB}$ [$\mu K^2$]')
axs[2,2].set_ylabel(r'$D_{\ell}^{BB}$ [$\mu K^2$]')

axs[1,0].set_visible(False)
axs[2,0].set_visible(False)
axs[2,1].set_visible(False)

fig.savefig(opj(imgdir, 'ps_c_ell_nice'))
plt.close(fig)


fig, axs = plt.subplots(ncols=1, nrows=3, dpi=300, constrained_layout=True, sharex=True)
for ax in axs.ravel():
    ax.set_prop_cycle('color',[plt.cm.gnuplot(i) for i in np.linspace(0, 1, niter)])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
for idx in range(niter):
    for aidxs, ax in enumerate(range(3)):
        axs[aidxs].plot(ells, dells * ps_c_ell[idx,aidxs,aidxs],
                        lw=0.5)

axs[2].set_xlabel(r'$\ell$')
axs[0].set_ylabel(r'$D_{\ell}^{TT}$ [$\mu K^2$]')
axs[1].set_ylabel(r'$D_{\ell}^{EE}$ [$\mu K^2$]')
axs[2].set_ylabel(r'$D_{\ell}^{BB}$ [$\mu K^2$]')

cmap = plt.get_cmap("gnuplot", niter)
norm = matplotlib.colors.BoundaryNorm(np.arange(niter + 1) + 0.5, niter)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axs, ticks=[1, 10, 20, 30])
cbar.set_label('CG step')
fig.savefig(opj(imgdir, 'ps_c_ell_nice_diag'))
plt.close(fig)



fig, axs = plt.subplots(ncols=1, nrows=3, dpi=300, constrained_layout=True, sharex=True)
for ax in axs.ravel():
    ax.set_prop_cycle('color',[plt.cm.gnuplot(i) for i in np.linspace(0, 1, niter)])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
for idx in range(niter):
    for aidxs, ax in enumerate(range(3)):
        axs[aidxs].plot(ells, np.abs(ps_c_ell[idx,aidxs,aidxs]) / np.abs(ps_c_ell[-1,aidxs,aidxs]),
                        lw=0.5)

        
axs[2].set_xlabel(r'$\ell$')
axs[0].set_ylabel(r'$C_{\ell}^{TT, i}$ / $C_{\ell}^{TT, 35}$ ')
axs[1].set_ylabel(r'$C_{\ell}^{EE, i}$ / $C_{\ell}^{EE, 35}$ ')
axs[2].set_ylabel(r'$C_{\ell}^{BB, i}$ / $C_{\ell}^{BB, 35}$ ')

axs[0].set_ylim(0.9)
axs[1].set_ylim(0.5, 4)
axs[2].set_ylim(0.5, 4)

cmap = plt.get_cmap("gnuplot", niter)
norm = matplotlib.colors.BoundaryNorm(np.arange(niter + 1) + 0.5, niter)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axs, ticks=[1, 10, 20, 30])
cbar.set_label('CG step')
fig.savefig(opj(imgdir, 'ps_c_ell_nice_diag_rel'))
plt.close(fig)
