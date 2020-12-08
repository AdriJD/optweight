import numpy as np

def to_complex(dtype):
    '''
    Return complex numpy dtype with appropriate number of bits 
    given input real dtype.

    Parameters
    ----------
    dtype : numpy.dtype
        Input real dtype.
    
    Returns
    -------
    dtype_complex : numpy dtype
        Complex dtype

    Raises
    ------
    ValueError
        If input dtype is not suported.
    '''

    if dtype == np.float32:
        return np.complex64
    elif dtype == np.float64:
        return np.complex128
    elif dtype == np.float128:
        return np.complex256
    else:
        raise ValueError('dtype" {} is not a supported complex type'.
                         format(dtype))

def to_real(dtype):
    '''
    Return real numpy dtype with appropriate number of bits
    given input complex dtype.

    Parameters
    ----------
    dtype : numpy.dtype
        Input complex dtype.

    Returns
    -------
    dtype_real : numpy dtype
        Real dtype

    Raises
    ------
    ValueError
        If input dtype is not suported.
    '''

    if dtype == np.complex64:
        return np.float32
    elif dtype == np.complex128:
        return np.float64
    elif dtype == np.complex256:
        return np.float128
    else:
        raise ValueError('dtype" {} is not a supported real type'.
                         format(dtype))
