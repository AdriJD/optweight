import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy
import os
import time

import healpy as hp
from pixell import curvedsky, enplot, utils, enmap, sharp

from optweight import sht, map_utils, solvers, operators, preconditioners, wlm_utils
from optweight import noise_utils, alm_utils, mat_utils, alm_c_utils, wavtrans

opj = os.path.join
np.random.seed(39)

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

lmax = 1000
ells = np.arange(lmax + 1)
dells = ells * (ells + 1)  / 2 / np.pi

imapdir = '/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019'
maskdir = '/projects/ACT/zatkins/sync/20201207/masks/masks_20200723'
odir = '/home/adriaand/project/actpol/20210416_data_noise'
imgdir = opj(odir, 'img')

os.makedirs(imgdir, exist_ok=True)

# Load map and mask
imap = enmap.read_fits(opj(imapdir, 's19_cmb_pa5_f150_nohwp_day_1pass_2way_set0_map.fits'))
imap -= enmap.read_fits(opj(imapdir, 's19_cmb_pa5_f150_nohwp_day_1pass_2way_set1_map.fits'))

imap = enmap.downgrade(imap, 4, op=np.mean)

for pidx in range(imap.shape[0]):

    plot = enplot.plot(imap[pidx], colorbar=True, grid=False)
    enplot.write(opj(imgdir, f's19_cmb_pa5_f150_{pidx}'), plot)
