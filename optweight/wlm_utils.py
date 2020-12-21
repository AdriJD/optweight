'''
Simple python implementation of the Scale-Discrete wavelet kernels
described in Leistedt et al., 2013 (1211.1680).
'''
import numpy as np

def get_sd_kernels(lamb, lmax, j0=None, lmin=None, jmax=None, lmax_j=None,
                   return_j=False):
    '''
    Compute Scale-Discrete wavelet kernels.

    Parameters
    ----------
    lamb : float
        Lambda parameter specifying the width of the kernels.
    lmax : int
        Maximum multipole.
    j0 : int, optional
        Minimum J scale used, i.e. the phi wavelet.
    lmin : int, optional
        Multipole after which the first kernel (phi) ends, alternative to j0.
    jmax : int, optional
        Maximum J scale used, i.e.g the omega wavelet.
    lmax_j : int, optional
        Multipole after which the second to last multipole ends, alternative
        to jmax.
    return_j : bool, optional
        Return J scales.

    Returns
    -------
    w_ell : (nwav, nell) array
        Wavelet kernels.
    lmaxs : (nwav) array
        Maximum multipole of each wavelet.
    j_scales : (nwav) array
        Wavelet scales.

    Raises
    ------
    ValueError
        if both j0 and lmin are provided.
        If j0 exceeds jmax (which is set by lmax).
        If j0 is smaller than 0.
        if both jmax and lmax_j are provided.

    Notes
    -----
    Wavelet numbers J run from 0 to jmax if j0 or lmin is not specified.
    The lowest wavelet number corresponds to Phi (i.e. the scaling kernel that
    that starts as a constant of 1 at ell = 0. This differs from the s2let code.
    Optionally, the last wavelet scale can be "omega", i.e. a kernel that is 1
    for all ell above lambda.
    '''

    if j0 and lmin:
        raise ValueError('Cannot have both j0 and lmin.')

    if jmax and lmax_j:
        raise ValueError('Cannot have both jmax and lmax_j.')

    jmax_lmax = lmax_to_j_scale(lmax, lamb)
    if lmax_j:
        jmax = lmax_to_j_scale(lmax_j, lamb)
    if jmax is None:
        jmax = jmax_lmax
    else:
        jmax = min(jmax, jmax_lmax)

    if j0 is None:
        if lmin is None:
            j0 = 0
        else:
            if lmin < 0:
                raise ValueError('lmin : {} cannot be negative'.format(lmin))
            j0 = max(0, lmax_to_j_scale(lmin, lamb) - 1)

    if j0 < 0:
        raise ValueError('j0 : {} cannot be negative.'.format(j0))

    if j0 > jmax:
        raise ValueError('j0 exceeds jmax, pick lower j0, lower lmin or higher lmax.')

    nwav = jmax + 1 - j0
    js = np.arange(j0, jmax + 1)
    ells = np.arange(lmax + 1)

    w_ell = np.zeros((nwav, lmax + 1))
    lmaxs = np.zeros(nwav, dtype=int)

    for jidx, j_scale in enumerate(js):

        if jidx == 0:
            w_ell[jidx,:] = phi_ell(ells, lamb, j_scale + 1)
        elif j_scale == jmax:
            w_ell[jidx,:] = omega_ell(ells, lamb, j_scale)
        else:
            w_ell[jidx,:] = psi_ell(ells, lamb, j_scale)

        if j_scale == jmax:
            lmaxs[jidx] = lmax
        else:
            lmaxs[jidx] = min(lmax, j_scale_to_lmax(j_scale, lamb))

    if return_j:
        return w_ell, lmaxs, js

    return w_ell, lmaxs

def j_scale_to_lmax(j_scale, lamb):
    '''
    Return lmax for given wavelet number.

    Parameters
    ----------
    j_scale : int
        Wavelet number.
    lamb : float
        Lambda parameter specifying the kernels.

    Returns
    -------
    lmax : int
        Band-limit of wavelet.
    '''

    # We round up to be conservative.
    return int(np.ceil(lamb ** (j_scale + 1)))

def j_scale_to_lmin(j_scale, lamb):
    '''
    Return lmin for given wavelet number.

    Parameters
    ----------
    j_scale : int
        Wavelet number.
    lamb : float
        Lambda parameter specifying the kernels.

    Returns
    -------
    lmin : int
        Lower band-limit of wavelet.
    '''

    return max(0, int(lamb ** (j_scale - 1)))

def lmax_to_j_scale(lmax, lamb):
    '''
    Return largest wavelet number J that is needed to reach lmax.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    lamb : float
        Lambda parameter specifying the kernels.

    Returns
    -------
    j_scale : int
        Wavelet number.
    '''

    return int(np.ceil(np.log10(lmax) / np.log10(lamb)))

def psi_ell(ells, lamb, j_scale):
    '''Calculate Eq. 15 in Leisteid et al. Note no prefactor.'''
    return kappa_lambda(ells / (lamb ** j_scale), lamb)

def phi_ell(ells, lamb, j0_scale):
    '''Calculate Eq. 16 in Leisteid et al. Note no prefactor.'''
    return eta_lambda(ells / (lamb ** j0_scale), lamb)

def omega_ell(ells, lamb, j_scale):
    '''Calculate zeta_lambda (ell / lambda^J).'''
    return zeta_lambda(ells / (lamb ** j_scale), lamb)

def zeta_lambda(ts, lamb):
    ''' Calculate sqrt(1 - k_lambda(t)).'''

    # To avoid floating point errors in difference.
    arg = 1 - k_lambda(ts, lamb)
    mask = arg >= 0
    out = np.zeros_like(arg)

    np.sqrt(arg, where=mask, out=out)
    return out

def eta_lambda(ts, lamb):
    '''Calculate Eq. 14 in Leisteid et al.'''
    return np.sqrt(k_lambda(ts, lamb))

def kappa_lambda(ts, lamb):
    '''Calculate Eq. 13 in Leisteid et al.'''

    arg = k_lambda(ts / lamb, lamb) - k_lambda(ts, lamb)
    # In order to avoid floating point errors that become nans.
    arg[arg < 0] = 0

    return np.sqrt(arg)

def k_lambda(ts, lamb):
    '''Calculate Eq. 12 in Leisteid et al.'''
    ts = np.asarray(ts)
    out = np.zeros(ts.size)
    nint = 1000

    out = np.zeros(ts.size)

    for tidx, tee in enumerate(ts):

        if tee < 1 / lamb:
            out[tidx] = 1
            continue
        if tee > 1:
            out[tidx] = 0
            continue

        ts_prime_num = np.linspace(tee, 1, num=nint, endpoint=True)
        ts_prime_den = np.linspace(1 / lamb, 1, num=nint, endpoint=True)

        out[tidx] = np.trapz(
            s_lambda(ts_prime_num, lamb) ** 2 / ts_prime_num, x=ts_prime_num)
        out[tidx] /= np.trapz(
            s_lambda(ts_prime_den, lamb) ** 2 / ts_prime_den, x=ts_prime_den)

    return out

def s_lambda(ts, lamb):
    '''Calculate Eq. 11 in Leisteid et al.'''
    if lamb <= 0:
        raise ValueError('lambda parameter needs to be positive.')

    return schwarz(2 * lamb * (ts - 1 / lamb) / (lamb - 1) - 1)

def schwarz(ts):
    '''Calculate Eq. 10 in Leisteid et al.'''
    mask = (ts > -1) & (ts < 1)
    out = np.zeros(ts.size)
    np.divide(-1, 1 - ts ** 2, where=mask, out=out)
    np.exp(out, where=mask, out=out)

    return out
