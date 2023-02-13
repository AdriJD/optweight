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
		      const float lim0, const int nsamp,
		      const long long int ncomp);
void _eigpow_core_rdp(double *imat, const double power, const double lim,
		      const double lim0, const int nsamp,
		      const long long int ncomp);

/*
 * Find maximum value of input array.
 *
 * Arguments
 * ---------
 * arr        : (n_elements) input array.
 * n_elements : Number of elements in array.
 *
 * Returns
 * -------
 * max_value  : Maximum value of input array.
 */

float find_max_sp(const float *arr, const int n_elements);
float find_max_dp(const double *arr, const int n_elements);
