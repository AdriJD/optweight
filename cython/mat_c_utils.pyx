cimport cmat_c_utils
import numpy as np
cimport numpy as np
np.import_array()

def eigpow(imat, power, lim=None, lim0=None):
    '''
    Port of Enlib's eigpow code. Raises a positive (semi)definite 
    matrix to an arbitrairy real power.

    Parameters
    ----------
    imat : (ncomp, ncomp, nsamp) array
        Input matrix.
    power : float
        Raise input matrix to this power.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.

    returns
    -------
    omat : (ncomp, ncomp, nsamp) array
        Output matrix, does not share memory with input.

    Raises
    ------
    ValueError
        If lim, lim0 are < 0.
        If input shape is wrong.
    '''

    ishape = imat.shape
    if imat.ndim not in (2, 3):
        raise ValueError(f'Input ndim : {imat.ndim} != 2 or 3.')
    if imat.ndim == 2:
        imat = imat[:,:,np.newaxis]

    ncomp = imat.shape[0]
    nsamp = imat.shape[2]

    if imat.shape[1] != ncomp:
        raise ValueError(
        f'Input shape : {ishape} does not match (ncomp, ncomp, nsamp) shape')

    if lim0 is None:
        lim0 = np.finfo(imat.dtype).tiny ** 0.5
    if lim is None:
        lim = 1e-6
	
    # Transpose and copy matrix. Assume worth it to reduce cache misses later.
    imat = np.ascontiguousarray(np.transpose(imat, (2, 0, 1)))
    
    cdef float [::1] imat_ = imat.reshape(-1)
    cmat_c_utils._eigpow_core_rsp_c(&imat_[0], power, lim, lim0, nsamp, ncomp)

    omat = np.ascontiguousarray(np.transpose(imat, (1, 2, 0))).view()
    omat.shape = ishape # Will crash if this needs copy (should not happen).

    return omat

def eigpow3(imat, power, lim, lim0):
    '''
    Port of Enlib's eigpow code. Raises a positive (semi)definite 
    matrix to an arbitrairy real power.

    Parameters
    ----------
    imat : (nsamp, ncomp, ncomp) array
            
        
    '''

    ncomp = imat.shape[1]
    nsamp = imat.shape[0]

    cdef float [::1] imat_ = imat.reshape(-1)
    cmat_c_utils._eigpow_core_rsp_c(&imat_[0], power, lim, lim0, nsamp, ncomp)

    return imat