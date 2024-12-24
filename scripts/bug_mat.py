import numpy as np
from pixell import enmap,utils as u,curvedsky as cs
import optweight

lmax = 4000
ells = np.arange(lmax+1)
theory_cls = {'TT': ells*0. + 1.}
b_ell = ells*0. + 1.
shape,wcs = enmap.fullsky_geometry(res=2*u.arcmin)
ivar = enmap.ones(shape,wcs)
ivar[:100]=0
omap = ivar.copy()
print('starting')
cg = optweight.CGPixFilter(1, theory_cls, b_ell, lmax,
                           icov_pix=ivar)
print('done setup')
alm = cs.map2alm(omap,lmax=lmax,tweak=True)
print('done alm2map')
output = cg.filter(alm,benchmark=True,verbose=True)
