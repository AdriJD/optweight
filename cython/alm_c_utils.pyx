cimport calm_c_utils
import numpy as np

def lmul(alm, lmat, ainfo, alm_out=None, inplace=False):
    '''       
    Compute the a'[i,lm] = m[i,j,ell] a[j,lm] matrix matrix multiplication.

    Arguments
    ---------
    alm : (ncomp, nelem) complex array
        Input alms.
    lmat : (ncomp, ncomp, nell) or (ncomp, nell) array
        Matrix, if diagonal only the diaganal suffices.
    ainfo : sharp.alm_info object
        Metainfo alms.
    alm_out : (ncomp, nelem) complex array, optional
        Output alms. Will be overwritten!
    inplace : bool
        In-place computation, output stored in alm.

    Returns
    -------
    alm_out : (ncomp, nelem) complex array
        Output alms.

    Raises
    ------
    ValueError
        If input/output shapes/dtypes do not match.
        If alm array does not have triangular layout.
	if both alm_out and inplace are set.

    Notes
    -----
    Similar to pixell's lmul but a bit faster serially and with openmp support.
    '''

    if alm.ndim == 1:
        ncomp = 1
    else:
        ncomp = alm.shape[0]

    if ncomp * ainfo.nelem != alm.size:
        raise ValueError(
          'ncomp * nelem != alm.size; only triangular alm storage supported')

    lmax = ainfo.lmax
    nelem = ainfo.nelem
    dtype = alm.dtype

    if alm.size != ncomp * nelem:
        raise ValueError('got alm.size {}, expected {} x {}'.
                         format(alm.size, ncomp, nelem))

    if alm_out is not None and inplace:
        raise ValueError('Cannot set both inplace and alm_out')
    
    if inplace is False:
        if alm_out is None:             
            alm_out = np.empty(alm.shape, dtype=dtype, order='C')
        else:
            if alm_out.size != alm.size:
                raise ValueError('alm.size {} != alm_out.size {}'.
                                 format(alm.size, alm_out.size))
            if alm_out.dtype != dtype:
                raise ValueError('alm.dtype {} != alm_out.dtype {}'.
                                 format(alm.size, alm_out.size))
    
    if lmat.ndim == 1:
        lmat = lmat * np.ones((ncomp, lmat.size), dtype=lmat.dtype)

    if dtype == np.complex128:

        if lmat.dtype != np.float64:
            raise ValueError('Expected float64 got lmat.dtype : {}'.format(
                lmat.dtype))

        if inplace:
            _lmul_inplace_dp(lmat, alm, lmax, ncomp)
        else:
            _lmul_dp(lmat, alm, alm_out, lmax, ncomp)
    
    elif dtype == np.complex64:

        if lmat.dtype != np.float32:
            raise ValueError('Expected float32 got lmat.dtype : {}'.format(
                lmat.dtype))

        if inplace:
            _lmul_inplace_sp(lmat, alm, lmax, ncomp)
        else:
            _lmul_sp(lmat, alm, alm_out, lmax, ncomp)

    if inplace:
        return alm
    else:
        return alm_out
    
def _lmul_dp(lmat, alm, alm_out, lmax, ncomp):
    '''Double precision variant.'''

    cdef double complex [::1] alm_ = alm.reshape(-1)
    cdef double [::1] lmat_ = lmat.reshape(-1)
    cdef double complex [::1] alm_out_ = alm_out.reshape(-1)
    
    if lmat.ndim == 3:  
        if lmat.size != ncomp * ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x {} x ({} + 1)'.format(
                lmat.size, ncomp, ncomp, lmax))
    
        calm_c_utils.lmul_dp(&lmat_[0], &alm_[0], &alm_out_[0], lmax, ncomp)

    elif lmat.ndim == 2:

        if lmat.size != ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x ({} + 1)'.format(
                 lmat.size, ncomp, lmax))

        calm_c_utils.lmul_diag_dp(&lmat_[0], &alm_[0], &alm_out_[0], lmax, ncomp)

    else:
        raise ValueError('Shape lmat : {} not suported'.format(lmat.shape))

def _lmul_inplace_dp(lmat, alm, lmax, ncomp):
    '''Double precision inplace variant.'''

    cdef double complex [::1] alm_ = alm.reshape(-1)
    cdef double [::1] lmat_ = lmat.reshape(-1)

    if lmat.ndim == 3:  
        if lmat.size != ncomp * ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x {} x ({} + 1)'.format(
                lmat.size, ncomp, ncomp, lmax))
    
        calm_c_utils.lmul_inplace_dp(&lmat_[0], &alm_[0], lmax, ncomp)

    elif lmat.ndim == 2:

        if lmat.size != ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x ({} + 1)'.format(
                 lmat.size, ncomp, lmax))

        calm_c_utils.lmul_diag_inplace_dp(&lmat_[0], &alm_[0], lmax, ncomp)

    else:
        raise ValueError('Shape lmat : {} not suported'.format(lmat.shape))

def _lmul_sp(lmat, alm, alm_out, lmax, ncomp):
    '''Single precision variant.'''

    cdef float complex [::1] alm_ = alm.reshape(-1)
    cdef float [::1] lmat_ = lmat.reshape(-1)
    cdef float complex [::1] alm_out_ = alm_out.reshape(-1)
    
    if lmat.ndim == 3:  
        if lmat.size != ncomp * ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x {} x ({} + 1)'.format(
                lmat.size, ncomp, ncomp, lmax))
    
        calm_c_utils.lmul_sp(&lmat_[0], &alm_[0], &alm_out_[0], lmax, ncomp)

    elif lmat.ndim == 2:

        if lmat.size != ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x ({} + 1)'.format(
                 lmat.size, ncomp, lmax))

        calm_c_utils.lmul_diag_sp(&lmat_[0], &alm_[0], &alm_out_[0], lmax, ncomp)

    else:
        raise ValueError('Shape lmat : {} not suported'.format(lmat.shape))

def _lmul_inplace_sp(lmat, alm, lmax, ncomp):
    '''Single precision inplace variant.'''

    cdef float complex [::1] alm_ = alm.reshape(-1)
    cdef float [::1] lmat_ = lmat.reshape(-1)
    
    if lmat.ndim == 3:  
        if lmat.size != ncomp * ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x {} x ({} + 1)'.format(
                lmat.size, ncomp, ncomp, lmax))
    
        calm_c_utils.lmul_inplace_sp(&lmat_[0], &alm_[0], lmax, ncomp)

    elif lmat.ndim == 2:

        if lmat.size != ncomp * (lmax + 1):
            raise ValueError('lmat.size {} != {} x ({} + 1)'.format(
                 lmat.size, ncomp, lmax))

        calm_c_utils.lmul_diag_inplace_sp(&lmat_[0], &alm_[0], lmax, ncomp)

    else:
        raise ValueError('Shape lmat : {} not suported'.format(lmat.shape))
