/*
 * Port of Enlib's eigpow code. Raises a positive (semi)definite 
 *   matrix to an arbitrairy real power.
 *
 * Arguments
 * ---------
 * imat     : (ncomp * ncomp * nsamp) input matrix, will be overwritten!
 * power    : Raise input matrix to this power.
 * lim      : Set eigenvalues smaller than lim * max(eigenvalues) to zero.
 * lim0     : If max(eigenvalues) < lim0, set whole matrix to zero.
 * nsamp    : Size of last dimension of input matrix.
 * ncomp    : Size of first two dimensions of input matrix.
 */

void _eigpow_core_rsp(float *imat, const float power, const float lim,
		      const float lim0, const int nsamp, const int ncomp);
void _eigpow_core_rdp(double *imat, const double power, const double lim,
		      const double lim0, const int nsamp, const int ncomp);
