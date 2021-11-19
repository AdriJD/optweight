import numpy as np

from pixell import utils
from enlib import array_ops

from optweight import wavtrans, type_utils

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

def flattened_view(mat, axes, return_flat_axes=False):
    '''
    Return view of input that has provided axes combined.

    Parameters
    ----------
    mat : array
        N-dimensional array.
    axes : array-like of array-like
        Axis indices to be flattened. E.g. if mat.shape = (3, 2, 5, 10)
        and axes = [[0, 1], [3]], the output shape is (3 * 2, 5, 10)
    return_flat_axes : bool
        If set, also return list of axes in final shape that were constructed
        by flattening input axes.

    Returns
    -------
    mat_view : array
        View of input array with new shape.

    Raises
    ------
    ValueError
        If axes are repeated.
    '''

    shape_in = np.asarray(mat.shape)
    shape_out = shape_in.copy()
    flat_axes = np.zeros_like(shape_out)
    mat_view = mat.view()

    # Turn negative axes into positive axes.
    for idx in range(len(axes)):
        for jidx in range(len(axes[idx])):
            ax = axes[idx][jidx]
            if ax < 0: axes[idx][jidx] = mat.ndim + ax

    # Check for repeats.
    axes_flat = []
    for sublist in axes:
        for idx in sublist:
            axes_flat.append(idx)            
    if not np.unique(axes_flat).size == len(axes_flat):
        raise ValueError(f'Axes cannot contain duplicates, got : {axes}')

    for sublist in axes:
        sublist = np.asarray(sublist)
        prod = np.prod(shape_in[sublist])
        shape_out[sublist[0]] = prod
        shape_out[sublist[1:]] = 0

        flat_axes[sublist[0]] = 1

    mask = shape_out != 0
    shape_out = shape_out[mask]
    flat_axes = list(np.nonzero(flat_axes[mask])[0])
    mat_view.shape = shape_out

    if return_flat_axes:
        return mat_view, flat_axes
    else:
        return mat_view
            
def matpow(mat, power, return_diag=False, skip_unit_pow=True, axes=None,
           inplace=False, chunksize=10000):
    '''
    Raise positive semidefinite matrix to a given power.

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
    axes : sequence or suequece of sequences, optional
        If sequence: list of 2 dimensions that form the symmetric matrix, use e.g. 
        [1, 2] in case  matrix is not (npol, npol, N) but (M, npol, npol, N). If 
        sequence of sequences: axes that should be flattened and then be used as
        symmetric part of matrix, use e.g. [[0, 1], [2, 3]] in case of e.g. 
        (ncomp, npol, ncomp, npol, npix) matrix.
    inplace : bool, optional
        If set, do operation in-place.
    chunksize : int, optional
        Do operations over non-PSD axes of input in chunks of this size.

    Returns
    -------
    mat_out : (npol, npol, N) array or (npol, N) array
        Output matrix, dense by default, but (npol, N) if return_diag is 
        set and input was diagonal.
    '''

    if mat.ndim == 1:
        mat = mat[np.newaxis,:]
    shape_in = mat.shape

    if type_utils.is_seq_of_seq(axes):
        mat, flat_ax = flattened_view(mat, axes, return_flat_axes=True)
    else:
        flat_ax = None
            
    if not 2 <= mat.ndim <= 3:
        raise ValueError(
            f'Matrix is not (npol, npol, N) or (npol, N), got : {mat.shape} '
            f'even after possible reshaping based on axes : {axes}')    

    if power != 1 or not skip_unit_pow:
        
        if mat.ndim == 2 and axes is None:
            # Setting negative eigenvalues to zero.
            if not inplace:
                mat = mat.copy()
            mask = mat <= 0
            mat[mask] = 0
            np.power(mat, power, out=mat, where=~mask)

        else:
            # 64 bit to avoid truncation of small values in eigpow.
            dtype_in = mat.dtype
            if dtype_in == np.float32:
                dtype = np.float64
            elif dtype_in == np.complex64:
                dtype = np.complex128
            else:
                dtype = dtype_in

            if axes is None:
                axes = [0, 1]

            axes_mat = flat_ax if flat_ax else axes
            mat = _eigpow(mat, power, axes_mat, dtype, inplace=inplace)

    mat = mat.reshape(shape_in)

    # If axes were specified, there is not need for full matrix.
    # When doing inplace, also don't expand into full matrix.
    if not return_diag and axes is None and not inplace:
        mat = full_matrix(mat)
    
    if not mat.flags['OWNDATA'] and not inplace:
        # Copy so that output always points to new array regardless of power.
        mat = mat.copy()
    
    return mat

def _eigpow(mat, power, axes, dtype_internal, inplace=False, chunksize=10000):
    '''
    Wrapper around eigpow that allows to divide up the problem in chunks
    to reduce memory usage.

    Parameters
    ----------
    mat : array
        Array that is a PSD matrix in two of its axes.
    power : int, float
        Power of matrix.
    axes : sequence
        List of 2 dimensions that form the PSD matrix.
    dtype_internal : type
        Dtype used for internal calculations.
    inplace : bool, optional
        If set, do operation in-place.
    chunksize : int, optional
        Do eigpow over non-PSD axes of input in chunks of this size.
    
    Returns
    -------
    mat : array
        Input array raised to power in the 2 PSD axes.
    '''

    dtype_in = mat.dtype

    if not inplace:
        mat = mat.copy()

    # Always chunk-up problem, so that internal copies don't cost too much memory.
    # Flatview could result in copy, but for our usual case with (ncomp, ncomp, npix)
    # it won't.
    with utils.flatview(mat, axes, pos=-1) as matf:

        for start in range(0, matf.shape[-1], chunksize):

            matslice = slice(start, start+chunksize)
            # If dtype is same, we do not have to make copy.
            matf_tmp = matf[:,:,matslice].astype(dtype_internal, copy=False)

            # You know that axes are [0,1] in this case.
            array_ops.eigpow(matf_tmp, power, axes=[0,1], copy=False)
            matf[:,:,matslice] = matf_tmp.astype(dtype_in, copy=False)

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
        
    inplace = matpow_kwargs.pop('inplace', False)
    if not inplace:
        m_wav_power = wavtrans.Wav(2, dtype=m_wav.dtype)

    for jidx in range(m_wav.shape[0]):

        minfo = m_wav.minfos[jidx,jidx]
        map_mat = m_wav.maps[jidx,jidx]
        map_mat = matpow(map_mat, power, inplace=inplace, **matpow_kwargs)

        if not inplace:
            m_wav_power.add((jidx,jidx), map_mat, minfo)

    if inplace:
        return m_wav
    else:
        return m_wav_power

def get_near_psd(mat, axes=None, inplace=False):
    '''
    Get close positive semi-definite matrix by zeroing negative eigenvalues.

    Arguments
    ---------
    mat : (npol, npol, N) array or (npol, N) array
        Matrix, either symmetric but dense in first two axes or diagonal,
    axes : array-like, optional
        List of 2 dimensions that form the symmetric matrix, use in case
        matrix is not (npol, npol, N) but e.g. (M, npol, npol, N).
    inplace : bool, optional
        If set, do computations in-place.
    
    Returns
    -------
    out : (npol, npol, N) or (npol, N) array
        Positive semi-definite matrix that is "close" to input matrix.
    '''

    if mat.ndim == 1:
        mat = mat[np.newaxis,:]

    if mat.ndim == 2 and axes is None:
        if not inplace:
            out = mat.copy()
        out[out<0] = 0
        return out

    # Handle 3d case.
    return matpow(mat, 1, skip_unit_pow=False, axes=axes, inplace=inplace)

def atleast_nd(mat, ndim, append=False): 
    '''
    View input as array with at least N dimensions by prepending axes.

    Parameters
    ----------
    mat : array-like
        Input array.
    ndim : int
        Number of dimensions.
    append : bool, optional
        If set, append extra axes instead of prepending.

    Returns
    -------
    out : array
        Array with ndim >= N.
    '''

    mat = np.asanyarray(mat) 
    ndim_add = ndim - mat.ndim
    
    # Note, negative arg to range results in empty tuple.
    axes = np.arange(ndim_add)

    if append:
        axes -= axes.size

    return np.expand_dims(mat, tuple(axes))
