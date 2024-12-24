import numpy as np
import time

from pixell import sharp
from optweight.alm_c_utils import lmul

lmax = 5000

ainfo = sharp.alm_info(lmax)
alm = np.ones((3, ainfo.nelem), dtype=np.complex128)
alm_out = np.zeros_like(alm)
lmat = np.ones((3, 3, lmax + 1))
lmat_diag = np.ones((3, lmax + 1))

t0 = time.time()
lmul(alm, lmat, ainfo, alm_out=alm_out)
print(time.time() - t0)

t0 = time.time()
ainfo.lmul(alm, lmat)
print(time.time() - t0)

t0 = time.time()
lmul(alm, lmat_diag, ainfo, alm_out=alm_out)
print('diag me', time.time() - t0)

alm_sp = np.ones((3, ainfo.nelem), dtype=np.complex64)
alm_out_sp = np.zeros_like(alm_sp)
lmat_sp = np.ones((3, 3, lmax + 1), dtype=np.float32)
lmat_diag_sp = np.ones((3, lmax + 1), dtype=np.float32)

t0 = time.time()
lmul(alm_sp, lmat_sp, ainfo, alm_out=alm_out_sp)
print(time.time() - t0)

t0 = time.time()
ainfo.lmul(alm_sp, lmat_sp)
print(time.time() - t0)

t0 = time.time()
lmul(alm_sp, lmat_diag_sp, ainfo, alm_out=alm_out_sp)
print('diag me', time.time() - t0)

