import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import numpy as np
from scipy.linalg import eigh

from optweight import sht, map_utils, mat_utils, solvers, operators, preconditioners

def colorbar(mappable, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label)
    plt.sca(last_axes)
    return cbar

opj = os.path.join
#np.random.seed(39)
np.random.seed(41)
#np.random.seed(44)

imgdir = '/home/adriaand/project/actpol/20230305_optweight_1d'

os.makedirs(imgdir, exist_ok=True)

#sample_rate = 10
sample_rate = 10
nsec = int(2 * np.pi)
nsamp = sample_rate * nsec
xcoords = np.linspace(0, nsec, nsamp)

global_amp = 12

freqs = np.fft.rfftfreq(xcoords.size, d=1 / sample_rate)
ks = 2 * np.pi * freqs
nk = ks.size
#print(freqs)
print(ks)
#print(freqs.size)
print(np.exp(ks / 20))

icov_signal = np.ones((1, ks.size)) * ks ** 2 * np.exp(ks / 20)
#icov_signal = np.ones((1, ks.size)) * ks ** 3 * np.exp(ks / 20)
print('icov_signal', icov_signal)
#icov_signal += 1
icov_signal /= global_amp ** 2
#icov_signal[0,0:4] = 0
#icov_signal[0,0] = 10

sqrt_icov_signal = mat_utils.matpow(icov_signal, 0.5, return_diag=True)
sqrt_cov_signal = mat_utils.matpow(icov_signal, -0.5, return_diag=True)
cov_signal = mat_utils.matpow(icov_signal, -1, return_diag=True)

#bk = np.exp(-0.5 * ks ** 2 * np.radians(5) ** 2)
bk = np.exp(-0.5 * ks ** 2 * np.radians(3) ** 2)
#bk = np.exp(-0.5 * ks ** 2 * np.radians(1) ** 2)
#bk = np.ones(ks.size)

icov_noise = np.ones((1, nsamp))
icov_noise[0,nsamp//6:2*nsamp//6] *= 0.05
#icov_noise *= 5e5
icov_noise *= 1e5
#icov_noise *= 3e4
#icov_noise *= 2e4
#icov_noise *= 1e4
#icov_noise *= 5e3
icov_noise /= global_amp ** 2


#mask_low = 3 * nsamp // 5
#mask_low = 2 * nsamp // 5
mask_low = 1 * nsamp // 2
mask_high = 4 * nsamp // 5
#mask_high = 3 * nsamp // 5
mask = np.ones((1, nsamp))
mask[0,mask_low:mask_high] = 0
#mask[0, mask_low+1] = 0.5
#mask[0, mask_high-2] = 0.5
#mask[0,mask_low:mask_high] = 1
#print(mask)

#icov_noise[0,mask_low:mask_high] *= 0.001



sqrt_icov_noise = mat_utils.matpow(icov_noise, 0.5, return_diag=True)
sqrt_cov_noise = mat_utils.matpow(icov_noise, -0.5, return_diag=True)
cov_noise = mat_utils.matpow(icov_noise, -1, return_diag=True)


#print(icov_signal)
#print(cov_signal)

fig, ax = plt.subplots(dpi=300, figsize=(4, 3), constrained_layout=True)
ax.plot(ks, bk, color='black')
ax.set_ylabel(r'Beam, $B(k)$')
ax.set_xlabel(r'wavenumber, $k$')
fig.savefig(opj(imgdir, 'beam'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
ax.plot(ks, icov_signal[0], color='black')
fig.savefig(opj(imgdir, 'icov_signal'))
plt.close(fig)

cov2plot = cov_signal.copy() 
cov2plot[0,0] = np.nan
fig, ax = plt.subplots(dpi=300, figsize=(4, 3), constrained_layout=True)
ax.plot(ks, cov2plot[0], color='black')
ax.set_yscale('log')
ax.set_ylabel(r'Signal power spectrum, $S(k)$')
ax.set_xlabel(r'wavenumber, $k$')
fig.savefig(opj(imgdir, 'cov_signal'))
plt.close(fig)

def draw_complex(nk):
    out = np.zeros((1, nk), dtype=np.complex128)
    out[:] += np.random.randn(1, nk) + 1j * np.random.randn(1, nk)
    return out

def draw_real(nsamp):
    return np.random.randn(1, nsamp)

print(draw_complex(nk).shape)

fsignal = sqrt_cov_signal * draw_complex(nk)
signal = np.fft.irfft(fsignal)
noise = sqrt_cov_noise * draw_real(nsamp)
data = np.fft.irfft(bk ** 2 * fsignal) + noise

data2plot = data.copy()
data2plot[mask == 0] = np.nan

data[mask == 0] = 0

icov_signal_lamb = lambda fx : icov_signal * fx
icov_noise_lamb = lambda x : icov_noise * x
#sht_lamb = (lambda fx : np.fft.irfft(fx), lambda x : np.fft.rfft(x) * nsamp)
#sht_lamb = (lambda fx : np.fft.irfft(fx), lambda x : np.fft.rfft(x))
sht_lamb = (lambda fx : np.fft.irfft(fx), lambda x : np.fft.rfft(x) / nsamp)
beam_lamb = lambda fx : bk * fx
mask_lamb = lambda x : mask * x

alpha2 = np.sum(icov_noise ** 2) / np.sum(icov_noise)
print('alpha2', alpha2)
#alpha2 *= 0.01
prec = mat_utils.matpow(icov_signal + alpha2 * bk ** 2, -1, return_diag=True)
prec_lamb = lambda fx : prec * fx
#prec_lamb = lambda fx : fx


ndraws = 10
draws = np.zeros((ndraws, nsamp))

for didx in range(ndraws):

    if didx < ndraws - 1:
        rand_isignal = sqrt_icov_signal * draw_complex(nk)
        rand_inoise = sqrt_icov_noise * draw_real(nsamp)
    else:
        rand_isignal = None
        rand_inoise = None

    solver = solvers.CGWienerMap(data, icov_signal_lamb, icov_noise_lamb, sht_lamb,
                             beam=beam_lamb, mask=mask_lamb, rand_isignal=rand_isignal,
                             rand_inoise=rand_inoise)

    solver.add_preconditioner(prec_lamb)
    solver.init_solver()

    #for idx in range(400):
    for idx in range(1000):
        solver.step()
        if solver.err < 1e-20:
            break
        if idx % 100 == 0:
            print(solver.i, solver.err)
        fwiener = solver.x
        wiener = np.fft.irfft(fwiener)
    draws[didx] = wiener[0]

ma_lamb = lambda fx : solver.M(solver.a_matrix(fx))
ma_map_lamb = lambda x : sht_lamb[0](solver.M(solver.a_matrix(sht_lamb[1](x))))

#a_mat = operators.op2mat(solver.a_matrix, nk, np.complex128, input_shape=(1, nk))
#ma_mat = operators.op2mat(ma_lamb, nk, np.complex128, input_shape=(1, nk))
ma_mat = operators.op2mat(ma_map_lamb, nsamp, np.float64, input_shape=(1, nsamp))

print('cond', np.linalg.cond(ma_mat))
eigvals, eigvecs = eigh(ma_mat)
#error_modes = np.sqrt(np.abs(eigvals)) * eigvecs
error_modes = 1 / np.sqrt(np.abs(eigvals)) * eigvecs

#print(error_modes.shape)
#exit()
ma_mat_trunc = error_modes.dot(error_modes.T) 

#print(eigvecs)

fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(4, 3))
ax.plot(eigvals, color='black')
ax.set_yscale('log')
ax.set_ylabel('Eigenvalue')
ax.set_xlabel('Index')
fig.savefig(opj(imgdir, 'eigvals'))
plt.close(fig)

nplot = 10
fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(5, 3.5))
#for idx in range(0, eigvals.size, 10):
ax.set_prop_cycle('color', [plt.cm.plasma(i) for i in np.linspace(0, 1, nplot)])
for idx in range(nplot):
    ax.plot(np.degrees(xcoords), error_modes[:,idx])
ax.set_ylabel(r'amplitude [A.U.]')
ax.set_xlabel(r'$\phi$ [deg]')
ax.axvline(np.degrees(xcoords[mask_low]), color='black', ls=':', lw=1)
ax.axvline(np.degrees(xcoords[mask_high - 1]), color='black', ls=':', lw=1)
ax.set_title(f'Eigenvectors with lowest eigenvalues')
fig.savefig(opj(imgdir, 'error_modes'))
plt.close(fig)

#fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(3.5, 3.5))
#im = ax.imshow(np.abs(a_mat))
#fig.savefig(opj(imgdir, 'a_mat'))
#plt.close(fig)

fig, ax = plt.subplots(dpi=300, figsize=(5, 4))
im = ax.imshow(np.log10(np.abs(ma_mat)))
colorbar(im, label=r'$\log M^{-1} A$')
ax.set_xlabel('pixel index')
ax.set_ylabel('pixel index')
fig.savefig(opj(imgdir, 'ma_mat'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(5, 3.5))
ax.plot(np.degrees(xcoords), signal[0], label='input signal')
ax.legend(frameon=False, loc='upper right')
ax.set_ylim(-1.3, 1.3)
ax.set_ylabel(r'amplitude [$\mathrm{\mu K_{CMB}}$]')
ax.set_xlabel(r'$\phi$ [deg]')
fig.savefig(opj(imgdir, 'signal'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(5, 3.5))
ax.plot(np.degrees(xcoords), signal[0], label='input signal')
ax.errorbar(np.degrees(xcoords), data2plot[0], yerr=sqrt_cov_noise[0], label='data',
            ls='none', color='black', marker='o', ms=3, elinewidth=1, capsize=2)

ax.axvline(np.degrees(xcoords[mask_low]), color='black', ls=':', lw=1)
ax.axvline(np.degrees(xcoords[mask_high - 1]), color='black', ls=':', lw=1)
ax.legend(frameon=False, loc='upper right')
ax.set_ylim(-1.3, 1.3)
ax.set_ylabel(r'amplitude [$\mathrm{\mu K_{CMB}}$]')
ax.set_xlabel(r'$\phi$ [deg]')
fig.savefig(opj(imgdir, 'data'))
plt.close(fig)


fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(5, 3.5))
ax.plot(np.degrees(xcoords), signal[0], label='input signal')
#ax.plot(np.degrees(xcoords), np.mean(draws, axis=0), color='C2')
ax.plot(np.degrees(xcoords), draws[-1], color='C2', label='wiener')
ax.errorbar(np.degrees(xcoords), data2plot[0], yerr=sqrt_cov_noise[0], label='data',
            ls='none', color='black', marker='o', ms=3, elinewidth=1, capsize=2)

ax.axvline(np.degrees(xcoords[mask_low]), color='black', ls=':', lw=1)
ax.axvline(np.degrees(xcoords[mask_high - 1]), color='black', ls=':', lw=1)
ax.legend(frameon=False, loc='upper right')
ax.set_ylim(-1.3, 1.3)
ax.set_ylabel(r'amplitude [$\mathrm{\mu K_{CMB}}$]')
ax.set_xlabel(r'$\phi$ [deg]')
fig.savefig(opj(imgdir, 'wiener'))
plt.close(fig)
    
fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(5, 3.5))
ax.plot(np.degrees(xcoords), signal[0], label='input signal')
#ax.plot(np.degrees(xcoords), np.mean(draws, axis=0), color='C2')
for didx in range(ndraws - 1):
    label = 'signal draw' if didx == 0 else None
    ax.plot(np.degrees(xcoords), draws[didx], color='C1', alpha=0.4, lw=1, label=label)
ax.plot(np.degrees(xcoords), draws[-1], color='C2', label='wiener')
ax.errorbar(np.degrees(xcoords), data2plot[0], yerr=sqrt_cov_noise[0], label='data',
            ls='none', color='black', marker='o', ms=3, elinewidth=1, capsize=2)

ax.axvline(np.degrees(xcoords[mask_low]), color='black', ls=':', lw=1)
ax.axvline(np.degrees(xcoords[mask_high - 1]), color='black', ls=':', lw=1)
ax.legend(frameon=False, loc='upper right')
ax.set_ylim(-1.3, 1.3)
ax.set_ylabel(r'amplitude [$\mathrm{\mu K_{CMB}}$]')
ax.set_xlabel(r'$\phi$ [deg]')
fig.savefig(opj(imgdir, 'result'))
plt.close(fig)





