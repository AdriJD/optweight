import numpy as np

from enlib import array_ops

from optweight import wavtrans

def symm2triu(mat, axes, return_axis=False):
    '''
    Return copy with only the upper-triangular elements of matrix.

    Parameters
    ----------
    mat : (..., N, N, ...) array
        Matrix that is symmetric in two of it's axes.
    axes : array-like
        Matrix is symmetric in these adjacent axes, e.g. [0, 1].
    return_axis : bool, optional
        If set, return the axis index that contains the 
        upper-tringular elements in output matrix.

    Returns
    -------
    mat_triu : (..., N * (N + 1) / 2, ...) array
        Matrix with upper triangular elements (row-major order).
    triu_axis : int
        Only if return_axis is set. The axis index that contains the 
        upper-tringular elements in output matrix.
        

    Raises
    ------
    ValueError
        If axes are not adjacent.
        If input matrix is not N x N in specified axes.
    '''
    
    if len(axes) != 2:
        raise ValueError(f'Need two axes, got {len(axes)} ({axes})')

    shape_in = mat.shape
    ndim_in = mat.ndim

    # Convert negative axes to positive and sort them.
    axes = np.asarray(axes)
    axes[axes < 0] = axes[axes < 0] + ndim_in
    axes = np.sort(axes)

    if shape_in[axes[0]] != shape_in[axes[1]]:
        raise ValueError(
        f'Matrix (shape : {shape_in}) is not N x N in specified axes : {axes}.')
    
    if np.abs(axes[0] - axes[1]) != 1:
        raise ValueError(f"Axes are not adjacent: {axes}")

    # Create boolean index array.
    n_side = shape_in[axes[0]]
    mask = np.triu(np.ones((n_side, n_side), dtype=bool))
    
    # Create tuple of slices into upper triangular part.
    idx_tup = (np.s_[:],) * axes[0] + (mask,) + (np.s_[:],) * (ndim_in - axes[1] - 1)
    mat_triu = mat[idx_tup]

    if return_axis:        
        return np.ascontiguousarray(mat_triu), axes[0]
    else:
        return np.ascontiguousarray(mat_triu)

def triu2symm(mat, axis):
    '''
    Return copy with upper-triangular elements expanded into symmmetric matrix.

    Parameters
    ----------
    mat: (..., N * (N + 1) / 2, ...) array
        Matrix with upper triangular elements (row-major order).
    axis : array-like
        Upper=triangular elements are stored in this axis.

    Returns
    -------
    mat_symm : (..., N, N, ...) array
        Matrix that is symmetric in two of it's axes.

    Raises
    ------
    ValueError
        If N * (N + 1) / 2 is not an integer.
    '''
    
    shape_in = mat.shape
    ndim_in = mat.ndim

    # Convert negative axes to positive.
    if axis < 0:
        axis = axis + ndim_in

    # Determine output shape with quadratic formula.
    n_side = (-1 + np.sqrt(1 + 8 * shape_in[axis])) / 2
    if np.floor(n_side) != n_side:
        raise ValueError(f'Cannot intepret axis : {axis} of shape {shape_in} '
                         f'as upper triangular elements')
    else:
        n_side = int(n_side)
    
    mat_symm = np.zeros(shape_in[:axis] + (n_side, n_side) + shape_in[axis+1:],
                        dtype=mat.dtype)

    lead_tup = (np.s_[:],) * axis
    trail_tup = (np.s_[:],) * (ndim_in - axis - 1)

    for idx_in, (idx_out, jdx_out) in enumerate(zip(*np.triu_indices(n_side))):

        # Create index tuples for input and output array.
        idx_tup_in = lead_tup + (np.s_[idx_in],) + trail_tup
        idx_tup_out = lead_tup + np.s_[idx_out,jdx_out] + trail_tup

        mat_symm[idx_tup_out] = mat[idx_tup_in]

        # Also fill lower triangular part by flipping i and j.
        if idx_out != jdx_out:
            idx_tup_out = lead_tup + np.s_[jdx_out,idx_out] + trail_tup            
            mat_symm[idx_tup_out] = mat[idx_tup_in]
        
    return mat_symm

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

def matpow(mat, power, return_diag=False, skip_unit_pow=True):
    '''
    Raise matrix to a given power.

    Parameters
    ----------
    mat : (npol, npol, N) array or (npol, N) array
        Matrix, either symmetric but dense in first two axes or diagonal,
        in which case only the diagonal elements are needed.
    power : int, float
        Power of matrix.
    return_diag : bool, optional
        If set, only return diagonal if input matrix was diagonal.
    skip_unit_pow : bool, optional
        If False, evalute even if power = 1, useful for non-psd matrices.

    Returns
    -------
    mat_out : (npol, npol, N) array or (npol, N) array
        Output matrix, dense by default, but (npol, N) if return_diag is 
        set and input was diagonal.
    '''

    ndim_in = mat.ndim

    if ndim_in == 1:
        mat = mat[np.newaxis,:]

    if power != 1 or not skip_unit_pow:

        if ndim_in == 2:
            mat = mat.copy()
            mat **= power 

        else:
            # 64 bit to avoid truncation of small values in eigpow.
            dtype_in = mat.dtype
            if dtype_in == np.float32:
                dtype = np.float64
            elif dtype_in == np.complex64:
                dtype = np.complex128
            else:
                dtype = dtype_in

            mat = np.ascontiguousarray(np.transpose(mat, (2, 0, 1)), dtype=dtype)
            mat = array_ops.eigpow(mat, power)
            mat = np.ascontiguousarray(np.transpose(mat, (1, 2, 0)), dtype=dtype_in)

    if not return_diag:
        mat = full_matrix(mat)

    if not mat.flags['OWNDATA']:
        # Copy so that output always points to new array regardless of power.
        mat = mat.copy()

    return mat

def wavmatpow(m_wav, power, **matpow_kwargs):
    '''
    Raise wavelet block matrix to a given power.

    Parameters
    ----------
    m_wav : wavtrans.Wav object
        Wavelet block matrix.
    power : int, float
        Power of matrix.
    matpow_kwargs : dict, optional
        Keyword arguments to matpow.
    
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

    m_wav_power = wavtrans.Wav(2, dtype=m_wav.dtype)

    for jidx in range(m_wav.shape[0]):

        minfo = m_wav.minfos[jidx,jidx]
        map_mat = m_wav.maps[jidx,jidx]

        map_mat = matpow(map_mat, power, **matpow_kwargs)
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
    out : (npol, npol, N) or (npol, N) array
        Positive semi-definite matrix that is "close" to input matrix.
    '''

    if mat.ndim == 1:
        mat = mat[np.newaxis,:]

    if mat.ndim == 2:
        out = mat.copy()
        out[out<0] = 0
        return out

    # Handle 3d case.
    return matpow(mat, 1, skip_unit_pow=False)
