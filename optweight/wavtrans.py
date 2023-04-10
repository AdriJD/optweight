import numpy as np
import os

from pixell import sharp
import h5py

from optweight import (alm_utils, sht, type_utils, type_utils, map_utils, 
                       wlm_utils, dft)

class Wav():
    '''
    A container for vectors or matrices of wavelet maps with
    varying pixelization schemes.

    Parameters
    ----------
    ndim : int
        Dimensions of block matrix/vector. Either 1 or 2.
    indices : (nwav, ndim) array, optional
        Indices to wavelet maps. ndim is one for vector indices, two
        for matrix indices.
    minfos : (nwav) object array of sharp.mapinfo objects, optional
        Map_info metadata for each entry of indices array.
    preshape : tuple, optional
        First dimensions of the maps, i.e. map.shape = preshape + (npix,)
    dtype : type, optional
        dtype of map arrays.

    Attributes
    ----------
    maps : dict of arrays
        Wavelet maps.
    minfos : dict of sharp.mapinfo objects
        Metainfo for each wavelet map.
    shape : tuple
        Shape of corresponding dense vector or matrix.
    ndim : int
    preshape : tuple
    dtype : type
    indices : (nwav, ndim) array

    Raises
    ------
    ValueError
        If shapes of input arrays do not match.
        If negative indices are provided
    NotImplementedError
        If ndim is not 1 or 2.

    Examples
    --------
    Create upper-triangular part of 2x2 matrix.

    >>> indices = np.asarray([(0,0), (0,1), (1,1)])
    >>> minfos = np.asarray([sharp.map_info_gauss_legendre(lmax) for
    ...                      lmax in [100, 100, 400]])
    >>> preshape = (3,)
    >>> wavmat = Wav(2, indices=indices, minfos=minfos, preshape=preshape)

    Shape of maps is combination of preshape and npix.

    >>> wavmat.maps[0,1]
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])

    Number of pixels can vary between indices in matrix.
    >>> wavmat.maps[0,1].shape
    (3, 20000)
    >>> wavmat.maps[1,1].shape
    (3, 320000)

    Metainfo can differ per map.
    >>> assert wavmat.maps[0,0].shape == preshape + (wavmat.minfos[0,0].npix,)
    True
    >>> assert wavmat.maps[1,1].shape == preshape + (wavmat.minfos[0,0].npix,)
    False

    Shape tells you size of corresponding dense matrix.
    >>> wavmat.shape
    (2, 2)
    '''

    def __init__(self, ndim, indices=None, minfos=None, preshape=None,
                 dtype=np.float64):

        if not (ndim == 1 or ndim == 2):
            raise NotImplementedError('{}-D matrices not supported'.format(ndim))

        if indices is None:
            indices = np.zeros((0, ndim), dtype=int)
        elif indices.ndim == 1:
            indices = indices[:,np.newaxis]

        self.ndim = ndim
        self.preshape = () if preshape is None else preshape
        self.dtype = dtype
        self.maps = {}
        self.minfos = {}
        self.indices = indices

        for idx in range(indices.shape[0]):

            minfo = minfos[idx]
            m_arr = np.zeros(self.preshape + (minfo.npix,), dtype=dtype)
            index = indices[idx]
        
            self.add(index, m_arr, minfo)

    @property
    def shape(self):
        '''Return shape of block vector/matrix.'''

        shape = ()
        
        for idx in range(self.indices.shape[1]):
            try:
                shape += (1 + np.max(self.indices[:,idx]),)
            except ValueError as e:
                if self.indices.shape[0] == 0:
                    shape += (0,)
                else:
                    raise e

        return shape
        
    def add(self, index, m_arr, minfo):
        '''
        Add map to the block vector/matrix.

        Parameters
        ----------
        index : (ndim) array
            Index to block vector/matrix.
        m_arr : (preshape) + (npix) array
            Map array to be placed at this index.
        minfo : sharp.map_info object
            Metainfo describing map array.

        Raises
        ------
        ValueError
            If index has wrong length.
            If index contains negative numbers.
            If index does not contain integers.
            If array has wrong dtype.
        '''
        
        index = np.asarray(index)
        if index.ndim == 0:
            index = index[np.newaxis]
        if index.size != self.ndim:
            raise ValueError('Wrong dimension for index, got : {} expected : {}'.
                             format(index.size, self.ndim))        

        if not issubclass(index.dtype.type, np.integer):
            raise ValueError('Index should be integer, instead got : {}'.
                             format(index.dtype))

        if np.any(index < 0):
            raise ValueError('Index cannot be negative, got {}'.format(index))

        index2dict = tuple(index)
        if len(index) == 1:
            # We use integers to index vectors, tuples for matrices.
            index2dict = index[0]

        if m_arr.dtype != self.dtype:
            raise ValueError('Adding array of wrong dtype : {}, expected {}'.
                             format(m_arr.dtype, self.dtype))

        # Check if preshape matches existing maps, if none exist allow updating.
        preshape = m_arr.shape[:-1]
        if self.maps:
            if preshape != self.preshape:
                raise ValueError(
                    'Wrong leading dimensions of map array: expected : {}, got {}'.
                    format(self.preshape, preshape))
        else:
            self.preshape = preshape
        
        self.maps[index2dict] = m_arr
        self.minfos[index2dict] = minfo

        # Add index to indices or, if index already exist, overwrite that entry.
        indices = np.append(self.indices, index[np.newaxis,:], axis=0)
        self.indices = np.unique(indices, axis=0)

    def diag(self):
        ''' 
        Return new instance containing diagonal of matrix if block matrix
        or just a copy if block vector.
        '''

        indices = self.get_indices_diag()
        wav_new = Wav(1, preshape=self.preshape, dtype=self.dtype)

        for index in indices:

            minfo = map_utils.copy_minfo(self.minfos[index])
            m_arr = self.maps[index].copy()
        
            if self.ndim == 1:
                wav_new.add(index, m_arr, minfo)
            elif self.ndim == 2:
                wav_new.add(index[0], m_arr, minfo)

        return wav_new

    def get_minfos_diag(self, copy=True):
        '''
        Get map_info objects corresponding to diagonal if block matrix.

        Parameters
        ----------
        copy : bool, optional
            If set, return copies of map info objects.

        Returns
        -------
        minfos : (ndiag) list of sharp.map_info objects
            Map_info objects of the diagonal.
        '''

        indices = self.get_indices_diag()
        minfos = []

        for index in indices:
            minfo = self.minfos[index]
            if copy:
                minfos.append(map_utils.copy_minfo(minfo))
            else:
                minfos.append(minfo)

        return minfos

    def get_indices_diag(self):
        '''Return list of indices of diagonal if block matrix.'''

        if self.ndim == 1:
            indices = list(self.indices[:,0])

        elif self.ndim == 2:
            indices = [tuple(idx) for idx in self.indices if idx[0] == idx[1]]
        
        return indices

    @classmethod
    def from_enmap(cls, shape, wcs, w_ell, *args, pad_factor=10, **kwargs):
        '''
        Initialize from the geometry of an enmap, supporting cut-sky maps.

        Parameters
        ----------
        shape : tuple
            Shape of enmap.
        wcs : astropy.wcs.WCS object
            The wcs object of the enmap geometry
        w_ell : (nwav, nell) array
            Wavelet kernels.
        pad_factor : float
            Add extra space in theta (colattitute) above and below region 
            spanned by enmap. See `get_enmap_minfos`.
        
        Returns
        -------
        wav : Wav object
            Zero-initialized wavelet object with minfos determined by enmap.

        Raises
        ------
        ValueError
            If shape of w_ell does not match provided wav matrix indices.
        '''

        # Overwrite minfos if provided.
        kwargs['minfos'] = get_enmap_minfos(shape, wcs, w_ell, pad_factor=pad_factor)

        # If no indices are given, create them here assuming vector or diag matrix
        # (based on "ndim").
        nwav = w_ell.shape[0]
        ndim = args[0]        
        kwargs.setdefault('indices', 
                (np.ones((nwav, ndim)) * np.arange(nwav)[:,np.newaxis]).astype(int))

        if kwargs['indices'].shape[0] != nwav:
            raise ValueError(f'Mismatch shape indices : {indices.shape} and '
                             f'w_ell : {w_ell.shape}')

        return cls(*args, **kwargs)

    @classmethod
    def from_enmap_cc(cls, shape, wcs, nwav, *args, **kwargs):
        '''
        Initialize vector from the geometry of CC enmap, supporting cut-sky maps.

        Parameters
        ----------
        shape : tuple
            Shape of enmap.
        wcs : astropy.wcs.WCS object
            The wcs object of the enmap geometry
        
        Returns
        -------
        wav : Wav object
            Zero-initialized wavelet object with minfos determined by enmap.
        '''

        minfo = map_utils.match_enmap_minfo(shape, wcs)
        
        # Overwrite minfos if provided.
        kwargs['minfos'] = [minfo] * nwav

        # If no indices are given, create them here assuming vector or diag matrix
        # (based on "ndim").
        ndim = args[0]  
        kwargs.setdefault('indices', 
                (np.ones((nwav, ndim)) * np.arange(nwav)[:,np.newaxis]).astype(int))

        if kwargs['indices'].shape[0] != nwav:
            raise ValueError(f'Mismatch shape indices : {indices.shape} and '
                             f'w_ell : {w_ell.shape}')

        return cls(*args, **kwargs)

def dot_wav(wav_a, wav_b):
    '''
    Compute dot product between two wavelet objects.

    Parameters
    ----------
    wav_a : wavtrans.Wav object
        Set of wavelet maps
    wav_b : wavtrans.Wav object
        Set of wavelet maps

    Returns
    -------
    dot : float
        Inner product of the two sets of map.

    Raises
    ------
    ValueError
        If indices of input wavelet object do not match.
        If preshape of input wavelet object do not match.
    
    Notes
    -----
    Simply computes element-wise products of all maps.
    '''

    if not np.all(wav_a.indices == wav_b.indices):
        raise ValueError('Mismatch indices between wav_a and wav_b')

    dot = 0.
    for key in wav_a.maps:
        dot += np.tensordot(wav_a.maps[key], wav_b.maps[key],
                            axes=wav_a.maps[key].ndim)
    return dot

def wav2alm(wav, alm, ainfo, spin, w_ell, adjoint=False):
    '''
    Convert wavelet maps to SH coefficients.

    Parameters
    ----------
    wav : wavtrans.Wav object
        Vector of wavelet maps.
    alm : (npol, nelem) array
        Output SH coefficients. Will be overwritten!
    ainfo : sharp.alm_info object
        Metainfo output alms.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    adjoint : bool, optional
        If set, compute adjoint synthesis: Yt, so map2alm without
        theta integration weights.

    Returns
    -------
    alm : (npol, nelem) array
        Output SH coefficients. 
    ainfo : sharp.alm_info object
        Metainfo output alms.
    
    Raises
    ------
    ValueError
        If input wav is not a vector.
    '''

    if len(wav.shape) != 1:
        raise ValueError('Input wavelet container is not a vector')
        
    indices = np.arange(w_ell.shape[0])
    lmax = w_ell.shape[-1] - 1

    alm *= 0

    for idx in indices:
        
        lmax_w = lmax - np.argmax(w_ell[idx,::-1] > 0)
        imap = wav.maps[idx]

        if imap.ndim == 1:
            imap = imap[np.newaxis,:]

        dtype = type_utils.to_complex(imap.dtype)
        winfo = sharp.alm_info(lmax=lmax_w)
        wlm = np.zeros((alm.shape[:-1]) + (winfo.nelem,), dtype=dtype)

        sht.map2alm(imap, wlm, wav.minfos[idx], winfo, spin,
                    adjoint=adjoint)

        alm_utils.wlm2alm_axisym([wlm], [winfo], w_ell[idx:idx+1,:],
                                 alm=alm, ainfo=ainfo)  

    return alm, ainfo

def alm2wav(alm, ainfo, spin, w_ell, wav=None, adjoint=False, lmaxs=None):
    '''
    Convert SH coefficients to wavelet maps. Defaults to Gauss-Legendre maps.

    Parameters
    ----------
    alm : (..., npol, nelem) array
        SH coefficients.
    ainfo : sharp.alm_info object
        Metainfo input alms.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    wav : wavtrans.Wav object, optional
        Wavelet container for output wavelet maps.
    adjoint : bool
        If set, compute adjoint analysis: WY, so alm2map with integration weights.
    lmaxs : (nwav) array lmax values, optional
        Max multipole for each wavelet.

    Returns
    -------
    wav : wavtrans.Wav object
        Vector of wavelet maps.
    '''

    lmax = ainfo.lmax
    wlms, winfos = alm_utils.alm2wlm_axisym(alm, ainfo, w_ell, lmaxs=lmaxs)
    indices = np.arange(len(wlms))

    if wav is None:
        preshape = alm.shape[:-1]

        minfos = []
        for widx in indices:

            lmax_w = winfos[widx].lmax
            minfos.append(sharp.map_info_gauss_legendre(
                lmax_w + 1, nphi=2 * lmax_w + 1))

        minfos = np.asarray(minfos)
        dtype = type_utils.to_real(alm.dtype)
        wav = Wav(1, indices=indices, minfos=minfos, preshape=preshape, dtype=dtype)

    for widx in indices:
        sht.alm2map(wlms[widx], wav.maps[widx], winfos[widx], wav.minfos[widx],
                    spin, adjoint=adjoint)

    return wav

def preshape2npol(preshape):
    '''
    Try to determine npol from the leading dimensions of a 
    wavelet object.

    Parameters
    ----------
    preshape : tuple
        Leading dimensions of array.

    Returns
    -------
    npol : int
    
    Raises
    ------
    ValueError
        If npol cannot be determined, i.e. wavelet is not block
        vector or square matrix.
    '''

    if len(preshape) > 2 or len(set(preshape)) > 1:
        raise ValueError('Could not determine npol from preshape : {}'
                         .format(preshape))
    if len(preshape) == 0:
        npol = 1
    else:
        npol = preshape[0]

    return npol

def write_wav(fname, wav, extra=None, **kwargs):
    '''
    Write wavelet object to an hdf file.

    Parameters
    ----------
    fname : str
        Path to file. Will append .hdf5 if no file extension is found.
    wav : wavtrans.Wav object
        Wavelet object to be stored.
    extra : dict, optional
        Extra key-object pairs to store as datasets in the hdf file.
    kwargs : dict, optonal
        Extra keyword arguments to map_utils.append_map_to_hdf.
    '''

    if not os.path.splitext(fname)[1]:
        fname = fname + '.hdf5'

    with h5py.File(fname, 'w') as hfile:
        
        hfile.attrs['ndim'] = wav.ndim
        hfile.attrs['indices'] = wav.indices
        hfile.attrs['preshape'] = wav.preshape
        hfile.attrs['dtype'] = np.dtype(wav.dtype).char

        if extra is not None:
            for key in extra.keys():
                hfile.create_dataset(key, data=extra[key])                

        for idx in wav.indices:
            
            idx_arr = np.asarray(idx)
            if idx_arr.size == 1:
                dname = f'm_{idx_arr[0]}'
                idx2dict = idx_arr[0]
            else:
                dname = f'm_{idx_arr[0]}_{idx_arr[1]}'
                idx2dict = tuple(idx_arr)

            mgroup = hfile.create_group(dname)
            map_utils.append_map_to_hdf(mgroup, wav.maps[idx2dict],
                                        wav.minfos[idx2dict], **kwargs)

def get_enmap_minfos(shape, wcs, w_ell, pad_factor=10):
    '''
    Find map_info objects that sufficiently describe enmap geometry
    for a set of wavelet kernels.

    Parameters
    ----------
    shape : tuple
        Shape of enmap.
    wcs : astropy.wcs.WCS object
        The wcs object of the enmap geometry
    w_ell : (nwav, nell) array
        Wavelet kernels.
    pad_factor : float
        Add extra space in theta (colattitute) above and below region 
        spanned by enmap. Extra space (in degrees) per wavelet kernel lmax
        is determined as `pad_factor` * 180 / lmax.

    Returns
    -------
    minfos : (nwav) object array of sharp.mapinfo objects
        Map_info metadata for each wavelet kernel described by `w_ell`.
    '''    

    lmaxs = wlm_utils.get_lmax_array(w_ell)
    minfos = []

    for widx in range(w_ell.shape[0]):

        pad = np.radians(pad_factor * 180 / lmaxs[widx])
        minfos.append(map_utils.get_enmap_minfo(
            shape, wcs, 2 * lmaxs[widx], pad=pad))

    return np.asarray(minfos)

def read_wav(fname, extra=None):
    '''
    Read wavelet object from an hdf file.

    Parameters
    ----------
    fname : str
        Path to file.
    extra : list of strings
        Look for these extra datasets in hdf file.

    Returns
    -------
    wav : : wavtrans.Wav object
        Wavelet object.
    extra_dict : dict
        Dictionary with extra objects extracted. Only
        if extra is given.

    Raises
    ------
    KeyError
        If extra keys requested but no extra datasets found.
    '''

    with h5py.File(fname, 'r') as hfile:

        if extra is not None:
            extra_dict = {}
            for key in extra:
                
                extra_dict[key] = hfile[key][()]
        
        ndim = hfile.attrs['ndim']
        indices = hfile.attrs['indices']
        preshape = tuple(hfile.attrs['preshape'])
        dtype = np.dtype(hfile.attrs['dtype'])

        wav = Wav(ndim, preshape=preshape, dtype=dtype)
        
        for idx in indices:
            
            idx_arr = np.asarray(idx)
            if idx_arr.size == 1:
                dname = f'm_{idx_arr[0]}'
                idx2dict = idx_arr[0]
            else:
                dname = f'm_{idx_arr[0]}_{idx_arr[1]}'
                idx2dict = tuple(idx_arr)

            mgroup = hfile[dname]
            m_arr, minfo = map_utils.map_from_hdf(mgroup)

            wav.add(idx2dict, m_arr, minfo)
        
    if extra is not None:
        return wav, extra_dict
    else:
        return wav

def f2wav(fmap, wav, fkernelset):
    '''
    Convert 2D fourier coefficients to wavelet maps.
        
    Parameters
    ----------
    fmap : (..., ny, nx//2+1) complex ndmap
        Input fourier coefficients.
    wav : wavtrans.Wav object
        Wavelet container for output wavelet maps.
    fkernelset : fkernel.FKernelSet object
        Wavelet kernels.

    Returns
    -------
    wav : wavtrans.Wav object
        Vector of wavelet maps.    
    '''
    
    for widx, fkern in fkernelset:

        map_2d = map_utils.view_2d(wav.maps[widx], wav.minfos[widx])
        fmap_slice = dft.slice_fmap(fmap, fkern.slices_y, fkern.slice_x)
        dft.irfft(fmap_slice * fkern.fkernel, map_2d)

    return wav

def wav2f(wav, fmap, fkernelset):
    '''
    Convert vector of wavelet maps to 2D fourier coefficients.
        
    Parameters
    ----------
    wav : wavtrans.Wav object
        Wavelet container for input wavelet maps.
    fmap : (..., ny, nx//2+1) complex ndmap
        Output buffer, will be overwritten!
    fkernelset : fkernel.FKernelSet object
        Wavelet kernels.

    Returns
    -------
    fmap : (..., ny, nx//2+1) complex ndmap
        Output Fourier coefficients.
    '''

    fmap *= 0 

    for widx, fkern in fkernelset:
        
        tmp = np.zeros(fmap.shape[:-2] + fkern.fkernel.shape, dtype=fmap.dtype)
        map_2d = map_utils.view_2d(wav.maps[widx], wav.minfos[widx])

        dft.rfft(map_2d, tmp)

        tmp = tmp * fkern.fkernel
        dft.add_to_fmap(fmap, tmp, fkern.slices_y, fkern.slice_x)
    
    return fmap
