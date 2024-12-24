import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from pixell import enmap

opj = os.path.join

idir = '/home/adriaand/project/actpol/20220516_pcg_planck/planck07/out'
imgdir = '/home/adriaand/project/actpol/20220516_pcg_planck/planck07/img'

ps_c_ell = np.load(opj(idir, 'ps_c_ell.npy'))
niter = ps_c_ell.shape[0]
lmax = ps_c_ell.shape[-1] - 1
ells = np.arange(lmax + 1)
dells = ells * (ells + 1) / 2 / np.pi

n_ell = np.load(opj(idir, 'n_ell.npy'))
cov_ell = np.load(opj(idir, 'cov_ell.npy'))
b_ell = np.load(opj(idir, 'b_ell.npy'))

fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(4, 3))
ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, niter)])

ax.plot(ells, dells * enmap.smooth_spectrum(n_ell[1,1], width=50) / b_ell[1] ** 2,
        lw=2, color='black', label=r'noise, $N_{\ell} / B^2_{\ell}$')

for idx in range(niter):
    ax.plot(ells, dells * enmap.smooth_spectrum(ps_c_ell[idx,1,1], width=5))

ax.plot(ells, dells * enmap.smooth_spectrum(cov_ell[1,1], width=5),
        lw=1, color='black', ls=':', label=r'signal, $S_{\ell}$')
ax.set_ylim(0, 1e2)
fig.savefig(opj(imgdir, 'ps_c_ell_EE'))
plt.close(fig)
ax.set_ylim(1e-2, 1e2)
ax.set_yscale('log')
ax.set_ylabel(r'$C^{TT}_{\ell}$ [$\mathrm{\mu K^2_{CMB}}$]')
ax.set_xlabel(r'$\ell$')
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'ps_c_ell_EE_log'))
plt.close(fig)
