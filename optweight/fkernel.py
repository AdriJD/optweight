import numpy as np
from scipy.interpolate import interp1d

from optweight import wlm_utils, dft

class FKernelSet():
    '''
    A container for 2D Fourier wavelet kernels with different shapes.

    Attributes
    ----------
    kernel_dict : dict
        Dictionary containing `FKernel` instances.
    dtype : type
        The dtype of the kernels.
    '''

    def __init__(self):

        self.fkernel_dict = {}
        self.dtype = np.float64

    def __getitem__(self, key):
        return self.fkernel_dict[key]

    def __setitem__(self, key, fkernel):
        ''' 
        Add a kernel to to set

        Parameters
        ----------
        key : int, string
            Key to internal dictionary.
        fkernel : fkernel.Fkernel object
            Kernel to add.

        Raises
        ------
        TypeError
            If dtype of input kernel does not match the dtype of this set.
        '''

        if not isinstance(fkernel, FKernel):
            raise ValueError(
                f'Only FKernel instances allowed, got {type(fkernel)}')

        if not self.fkernel_dict:
            # If set is still empty, take dtype from this new kernel.
            self.dtype = fkernel.dtype
        elif not self.dtype is fkernel.dtype:
            raise TypeError(f'Input is {fkernel.dtype} while set dtype is {self.dtype}')
            
        self.fkernel_dict[key] = fkernel

    def __len__(self):

        return len(self.fkernel_dict)

    def astype(self, dtype, copy=True):
        '''
        Update dtype of kernels.

        Parameters
        ----------
        dtype : type
            New type.
        copy : bool, optional
            If False, do not copy kernels if numpy is able to cast without copy.
        '''

        for key in self.fkernel_dict:
            
            fkernel = self.fkernel_dict[key].fkernel
            self.fkernel_dict[key].fkernel = fkernel.astype(dtype, copy=copy)
            self.dtype = dtype

        return self

    def to_full_array(self):
        '''
        Return full-sized set of kernels. Useful for debugging.

        Returns
        -------
        full : (nwav, ny, nx) array
            Full-sized kernel array.
        '''

        if not self.fkernel_dict:
            raise ValueError('Calling `to_full_array` on empty set.')
        
        first_key = list(self.fkernel_dict.keys())[0]     
        full = np.zeros((len(self),) + self.fkernel_dict[first_key].full_shape,
                        dtype=self.dtype)

        for key in self.fkernel_dict:
            kernel = self.fkernel_dict[key]
            dft.add_to_fmap(full[key], kernel.fkernel, kernel.slices_y, kernel.slice_x)

        return full

class FKernel():
    '''
    Class describing a single 2D Fourier wavelet kernel.

    Parameters
    ----------
    fkernel : (nly, nlx) array
        Kernel in 2D Fourier space.
    ly : (nly) array
        Wavenumbers in y direction.
    lx : (nlx) array
        Wavenumbers in x direction.
    slices_y : tuple of slices, optional
        Two slices that select nonzero positive frequences and nonzero
        negaive frequencies. Determines where this kernel fits into full
        res fmap.
    slice_x : slice, optional
        Slice with nonzero elements in the x direction
    full_shape : tuple, optional
        Full resolution shape (ny, nx) from which this kernel was sliced.

    Attributes
    ----------
    fkernel : (nly, nlx) array
        Kernel in 2D Fourier space.
    ly : (nly) array
        Wavenumbers in y direction.
    lx : (nlx) array
        Wavenumbers in x direction.
    lmax : int
        Maximum wavenumber
    slices_y : tuple of slices or None
        Two slices that select nonzero positive frequences and nonzero
        negaive frequencies. Determines where this kernel fits into full
        res fmap.
    slice_x : slice or None
        Slice with nonzero elements in the x direction
    full_shape : tuple or None
        Full resolution shape (ny, nx) from which this kernel was sliced.


    Raises
    ------
    ValueError
        If input shapes do not match.
    '''

    def __init__(self, fkernel, ly, lx, slices_y=None, slice_x=None, full_shape=None):

        if fkernel.shape[-2:] != (ly.size, lx.size):
            raise ValueError(
                f'shape mismatch, fkernel : {fkernel.shape[-2:]}'\
                f', ly : {ly.size}, lx : {lx.size}')

        self.fkernel = fkernel
        self.ly = ly
        self.lx = lx
        self.slices_y = slices_y
        self.slice_x = slice_x
        self.full_shape = full_shape
        
        self.lmax = int(np.ceil(np.sqrt(
            np.max(np.abs(ly)) ** 2 + np.max(np.abs(lx)) ** 2)))

    def modlmap(self, dtype=np.float64):
        '''Return map of  absolute wavenumbers, see `dft.modlmap_real`.'''
        return dft.laxes2modlmap(ly, lx, dtype=dtype)

    @property
    def dtype(self):
        ''' Return dtype of fkernel array.'''
        return self.fkernel.dtype

    def to_full_array(self):
        '''
        Return kernel inserted into the full-sized array from which it was 
        orginally sliced. Useful for debugging.
        '''
        
        if None in (self.slices_y, self.slice_x, self.full_shape):
            return ValueError('`to_full_array` requires `slices_y`, `slice_x` and `full_shape`')
        full = np.zeros(self.shape_full, dtype=self.dtype)
        dft.add_to_fmap(full, self.fkernel, self.slices_y, self.slice_x)
        return full

def digitize_1d(arr):
    '''
    Turn a smooth array with values between 0 and 1 into an on/off array
    that approximates it.

    Parameters
    ----------
    arr : (N) array
        1D input array containing smooth input.

    Returns
    -------
    out : (N) bool array
        Digital approximation of input.

    Notes
    -----
    Adapted from pixell.wavelets. Integral of normalized input is
    conserved in limit of large array.
    '''

    if arr.min() < 0 or arr.max() > 1:
        raise ValueError('Input values do not lie between 0 and 1.')

    out = np.round(np.cumsum(arr))
    out = np.concatenate([out[0:1], out[1:] != out[:-1]])
    return out.astype(bool)

def digitize_kernels(w_ell, upsamp_fact=10):
    '''
    Approximate smooth kernels as on/off array and possibly upsample in
    ell.

    Parameters
    ----------
    w_ell : (nwav, nell) array
        Wavelet kernels. Non-neighbouring kernels should not overlap.
    upsamp_fact : int
        Upsample the kernels by this factor in ell.

    Returns
    -------
    d_ell : (nwav, nell * upsamp_fact) bool array
        Digital wavelet kernels.
    ells_upsamp : (nell * upsamp_fact) array
        Upsampled multipoles.

    Raises
    ------
    ValueError
        If non-neighbours input wavelet kernels overlap.
    '''

    nwav = w_ell.shape[0]

    err = ValueError('Input `w_ell` has overlapping non-neighbouring kernels')
    EPS = 1e-8
    if nwav > 2 and not np.all(np.prod(w_ell[::2], axis=0) < EPS):
        raise err
    if nwav > 3 and not np.all(np.prod(w_ell[1::2], axis=0) < EPS):
        raise err

    lmax = w_ell.shape[-1] - 1
    ells = np.arange(0, lmax + 1, 1)
    ells_fine = np.linspace(0, lmax + 1, upsamp_fact * ells.size)

    # Interpolate smooth input.
    w_ell_fine = np.zeros((nwav, ells_fine.size))
    d_ell_fine = np.ones_like(w_ell_fine, dtype=bool)
    for widx in range(nwav):
        w_ell_fine[widx] = interp1d(ells, w_ell[widx], fill_value='extrapolate',
                                    kind='linear')(ells_fine)
    # Digitize even kernels.
    for idx in range(0, nwav, 2):
        d_ell_fine[idx] = digitize_1d(w_ell_fine[idx])

    # Populate odd kernels.
    d_ell_fine[1::2] *= np.logical_xor(
        True, np.sum(d_ell_fine[::2,:], axis=0)[np.newaxis,:] > 0.5)

    for idx in range(1, nwav, 2):
        d_ell_fine[idx,w_ell_fine[idx] < 1e-5] = 0

    return d_ell_fine, ells_fine

def _find_last_nonzero_idx(arr):
    '''
    Find index of last nonzero element in 1D array.

    Parameters
    ----------
    arr : (narr) array
        Input array.

    Returns
    -------
    idx_end : int
        Index of last nonzero element.

    Raises
    ------
    ValueError
        If input array has no nonzero elements
        If input array is not 1D.
    '''

    if arr.ndim != 1:
        raise ValueError(f'Input array is not 1D, got ndim : {arr.ndim}')

    arr = arr.astype(bool, copy=False)
    if not np.any(arr):
        raise ValueError('Input array is completely zero.')

    narr = arr.size
    last_nonzero_idx = np.nonzero(arr[::-1])[0][0]
    idx_end = narr - 1 - last_nonzero_idx

    return idx_end

def find_kernel_slice(fkernel_arr, minval=1e-6):
    '''
    Return the slices that capture the nonzero elements of input kernel.

    Parameters
    ----------
    fkernel_arr : (ny, nx) array
        Input kernel.
    minval : float, optional
        Consider values below this value as zero.

    Returns
    -------
    slices_y : tuple of slices
        Two slices that select nonzero positive frequences and nonzero
        negaive frequencies.
    slice_x : slice
        Slice with nonzero elements in the x direction

    Raises
    ------
    ValueError
        If input array is completely zero.
    '''

    if fkernel_arr.ndim != 2:
        raise ValueError(f'Input array must be 2D, got {fkernel_arr.ndim}')

    # Find x slice
    # We assume the x-axis starts with the lowest frequencies.
    nonzero = np.any(fkernel_arr >= minval, axis=0)

    if not np.any(nonzero):
        raise ValueError('Input array is completely zero.')

    nx = nonzero.size
    if nonzero[-1] == True:
        # Last element is still nonzero, so we must include full x slice.
        idx_x_end = nx - 1
    else:
        # Find last non-zero element
        idx_x_end = _find_last_nonzero_idx(nonzero)

    slice_x = slice(0, idx_x_end + 1)

    # Find y slices.
    nonzero = np.any(fkernel_arr >= minval, axis=1)

    ny = nonzero.size
    # Note that idx=0 contains monopole, which we always want to include.
    nonzero_ypos = nonzero[1:int(np.ceil(ny / 2))]
    try:
        idx_ypos_end = _find_last_nonzero_idx(nonzero_ypos)
    except ValueError:
        # Zero input array. -1 gives idx_ypos_end = -1 + 2 = 1.
        idx_ypos_end = -1

    # Note, reversed.
    nonzero_yneg = nonzero[int(np.ceil(ny / 2)):][::-1]
    try:
        idx_yneg_end = _find_last_nonzero_idx(nonzero_yneg)
    except ValueError:
        idx_yneg_end = -1

    idx_y_end = max(idx_ypos_end, idx_yneg_end)
    if idx_y_end == -1:
        # Happens when both are empty.
        slice_ypos = slice(0, 1)
        slice_yneg = slice(0, 0)
    else:
        n_pos = nonzero_ypos.size + 1
        n_neg = nonzero_yneg.size
        slice_ypos = slice(0, min(idx_y_end + 2, n_pos)) # +1 for monopole.
        slice_yneg = slice(-min(idx_y_end + 1, n_neg), None)

    return (slice_ypos, slice_yneg), slice_x

def w_ell2fkernels(w_ell, ells, ly, lx, interp_kind):
    '''
    Turn set of 1d wavelet kernels into maps of 2D fourier coefficients.

    Parameters
    ----------
    w_ell : (nwav, nell) array
        Wavelet kernels. Non-neighbouring kernels should not overlap.
    ells : (nell) array
        Multiploles corresponding to `w_ell`, allowed to be non-integer.
    ly : (nly) array
        Wavenumbers in y direction.
    lx : (nlx) array
        Wavenumbers in x direction.
    interp_kind : str
        Interpolation method, see `scipy.interpolate.interp1d`.

    Returns
    -------
    fkernels : (nwav, nly, nlx) array
        Output kernels.
    '''

    nwav = w_ell.shape[0]

    modlmap = dft.laxes2modlmap(ly, lx)

    fkernels = FKernelSet()

    for idx in range(nwav):

        if idx == nwav - 1:
            fill_value = (0., 1.)
        else:
            fill_value = (0., 0.)

        cs = interp1d(ells, w_ell[idx], kind=interp_kind,
                      bounds_error=False, fill_value=fill_value)
        fkernel_full = cs(modlmap)

        slices_y, slice_x = find_kernel_slice(fkernel_full)        
        fkernel_sliced, laxes_sliced = dft.slice_fmap(
            fkernel_full, slices_y, slice_x, laxes=(ly, lx))                
        fkernels[idx] = FKernel(fkernel_sliced, *laxes_sliced,
                                slices_y=slices_y, slice_x=slice_x,
                                full_shape=(ly.size, lx.size))

    return fkernels

def get_sd_kernels_fourier(ly, lx, lamb, j0=None, lmin=None, jmax=None,
                           lmax_j=None, digital=True, oversample=10):
    '''
    2D representation of the Scale-Discrete wavelet kernels for 2D Fourier wavelet
    transforms.

    Parameters
    ----------
    ly : (nly) array
        Wavenumbers in y direction.
    lx : (nlx) array
        Wavenumbers in x direction.
    lamb : float
        Lambda parameter specifying the width of the kernels.
    j0 : int, optional
        Minimum J scale used, i.e. the phi wavelet.
    lmin : int, optional
        Multipole after which the first kernel (phi) ends, alternative to j0.
    jmax : int, optional
        Maximum J scale used, i.e.g the omega wavelet.
    lmax_j : int, optional
        Multipole after which the second to last multipole ends, alternative
        to jmax.
    digital : bool
        Produce digital (on/off) version of the kernels.
    oversample : int
        Oversample the SD kernels by this factor with respect to the Fourier ell
        spacing. Only used when `digital` is set.

    Returns
    -------
    fkernels : (nwav, nly, nlx) array
        Output kernels. If `digital` was set, this will be a boolean array.
    '''

    lmax = int(np.ceil(np.sqrt(np.max(np.abs(ly)) ** 2 + np.max(np.abs(lx)) ** 2)))

    w_ell, _ = wlm_utils.get_sd_kernels(
        lamb, lmax, j0=j0, lmin=lmin, jmax=jmax, lmax_j=lmax_j)

    if digital:
        delta_ell = np.mean([np.abs(lx[0] - lx[1]),
                             np.abs(ly[0] - ly[1])])

        upsamp_fact = max(1, int(np.round(oversample / delta_ell)))
        w_ell, ells = digitize_kernels(w_ell, upsamp_fact=upsamp_fact)
    else:
        ells = np.arange(w_ell.shape[-1])

    if digital:
        dtype = bool
    else:
        dtype = w_ell.dtype

    interp_kind = 'nearest'
    return w_ell2fkernels(
        w_ell, ells, ly, lx, interp_kind).astype(dtype, copy=False)
