cimport cmap_c_utils
import numpy as np
cimport numpy as np
np.import_array()

def apply_ringweight(imap, minfo, custom_weight=None):
    '''
    Apply quadrature weight per ring.

    Parameters
    ----------
    imap : (ncomp, npix) array
        Input map. Will be overwritten!
    minfo : map_utils.MapInfo object.
        Metainfo describing map geometry.
    custom_weight : (ntheta) array, optional
        Custom weight per ring.

    Raises
    ------
    ValueError
        If input shape is wrong.
	If shape of custom_weight is wrong.
    '''

    # Perform some checks to make sure minfo lines up with imap.
    if imap.shape[-1] != minfo.npix:
        raise ValueError(f'input {imap.shape=} disagrees with {minfo.npix=}')

    if custom_weight is not None:
        if custom_weight.shape != minfo.weight.shape:
            raise ValueError(
              f'{custom_weight.shape=}, expected {minfo.weight.shape=}')          
        weight = custom_weight.astype(np.float64)
    else:
        weight = minfo.weight

    if not (0 < imap.ndim < 3):
        raise ValueError(f'input map has to be 1 or 2d, got {imap.ndim=}')
    if imap.ndim == 1:
        imap = imap[np.newaxis,:]
    ncomp = imap.shape[0]

    if imap.dtype == np.float32:
        for pidx in range(ncomp):
            _apply_ringweight_sp(imap[pidx], weight, minfo.nphi, minfo.offsets,
	                         minfo.stride, minfo.nrow)
    elif imap.dtype == np.float64:
        for pidx in range(ncomp):
            _apply_ringweight_dp(imap[pidx], weight, minfo.nphi, minfo.offsets,
	                         minfo.stride, minfo.nrow)

def _apply_ringweight_sp(imap, weight, nphi, offsets, stride, nrow):

    cdef float [::1] imap_ = imap.reshape(-1)
    cdef const double [::1] weight_ = weight
    cdef const long [::1] nphi_ = nphi.astype(np.int64)
    cdef const long [::1] offsets_ = offsets.astype(np.int64)
    cdef const long [::1] stride_ = stride.astype(np.int64)
    cmap_c_utils._apply_ringweight_core_sp(&imap_[0], &weight_[0], &nphi_[0],
                                           &offsets_[0], &stride_[0], nrow)

def _apply_ringweight_dp(imap, weight, nphi, offsets, stride, nrow):

    cdef double [::1] imap_ = imap.reshape(-1)
    cdef const double [::1] weight_ = weight
    cdef const long [::1] nphi_ = nphi.astype(np.int64)
    cdef const long [::1] offsets_ = offsets.astype(np.int64)
    cdef const long [::1] stride_ = stride.astype(np.int64)
    cmap_c_utils._apply_ringweight_core_dp(&imap_[0], &weight_[0], &nphi_[0],
                                           &offsets_[0], &stride_[0], nrow)


