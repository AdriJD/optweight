import numpy as np

from pixell import utils

from optweight import wavtrans

def full_matrix(mat):
    '''
    If needed, expand matrix diagonal to full-sized matrix.

    Parameters
    ----------
    mat : (npol, npol, N) array or (npol, N) array
        Matrix, dense or diagonal in first two axes.

    Returns
    -------
    mat_full : (npol, npol, N) array
        Full-sized matrix.

    Raises
    ------
    ValueError
        If input array dimensions are not understood.
    '''

    if mat.ndim not in (2, 3):
        raise ValueError(
            'Matrix dimension : {} not supported. Requires dim = 2 or 3.'.
            format(mat.ndim))

    if mat.ndim == 2:
        # Diagonal of matrix, expand to full matrix.
        npol = mat.shape[0]
        mat = np.eye(npol, dtype=mat.dtype)[:,:,np.newaxis] * mat

    return mat

def matpow(mat, power):
    '''
    Raise matrix to a given power.

    Parameters
    ----------
    mat : (npol, npol, N) array or (npol, N) array
        Matrix, either symmetric but dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    power : int, float
        Power of matrix.

    Returns
    -------
    mat_out : (npol, npol, N) array
        Output matrix.
    '''

    mat = full_matrix(mat)

    if power == 1:
        return mat
        
    # 64 bit to avoid truncation of small values in eigpow.
    dtype_in = mat.dtype
    if dtype_in == np.float32:
        dtype = np.float64
    elif dtype_in == np.complex64:
        dtype = np.complex128
    else:
        dtype = dtype_in

    mat = np.ascontiguousarray(np.transpose(mat, (2, 0, 1)), dtype=dtype)
    mat = utils.eigpow(mat, power)
    mat = np.ascontiguousarray(np.transpose(mat, (1, 2, 0)), dtype=dtype_in)

    return mat

def wavmatpow(m_wav, power):
    '''
    Raise wavelet block matrix to a given power.

    Parameters
    ----------
    m_wav : wavtrans.Wav object
        Wavelet block matrix.
    power : int, float
        Power of matrix.
    
    Returns
    -------
    m_wav_power : wavtrans.Wav object
        Wavelet block matrix raised to power.

    Raises
    ------
    NotImplementedErrror
        If matrix power is computed for non-diagonal block matrix.
    '''

    if not np.array_equal(m_wav.indices[:,0], m_wav.indices[:,1]):
        raise NotImplementedError(
            'Can only raise diagonal block matrix to a power')

    m_wav_power = wavtrans.Wav(2)

    for jidx in range(m_wav.shape[0]):

        minfo = m_wav.minfos[jidx,jidx]
        map_mat = m_wav.maps[jidx,jidx]

        map_mat = matpow(map_mat, power)
        m_wav_power.add((jidx,jidx), map_mat, minfo)

    return m_wav_power

def get_near_psd(mat):
    '''
    Get close positive semi-definite matrix by zeroing negative eigenvalues.

    Arguments
    ---------
    mat : (npol, npol, N) array or (npol, N) array
        Matrix, either symmetric but dense in first two axes or diagonal,
    
    Returns
    -------
    out : (npol, npol, N)
        Positive semi-definite matrix that is "close" to input matrix.
    '''

    mat = full_matrix(mat)

    # 64 bit for more accuracy.
    dtype_in = mat.dtype
    if dtype_in == np.float32:
        dtype = np.float64
    else:
        dtype = dtype_in

    eigvals, eigvecs = np.linalg.eigh(np.transpose(mat, (2, 0, 1)).astype(dtype=dtype))

    mask = eigvals < 0
    if np.sum(mask) == 0:
        return mat.astype(dtype_in)

    eigvals[mask] = 0
    out = np.einsum('ijk, ikm -> ijm',
            eigvecs, eigvals[:,:,np.newaxis] * np.transpose(eigvecs, (0, 2, 1)))
    
    return np.ascontiguousarray(np.transpose(out, (1, 2, 0)), dtype=dtype_in)
