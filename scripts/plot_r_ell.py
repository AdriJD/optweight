import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from optweight import multigrid

opj = os.path.join

odir = '/home/adriaand/project/actpol/20220121_r_ell'

fig, ax = plt.subplots(dpi=300)
for lmax in [1000, 2000, 4000, 6000, 10000, 100000]:
    ax.plot(multigrid.lowpass_filter(lmax), label=lmax)
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(100)
fig.savefig(opj(odir, 'r_ell'))
plt.close(fig)
    
