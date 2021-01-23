#include <stdlib.h>
#include <complex.h>
#include <omp.h>

/*
 * Compute matrix multiplication out:[i,nelem] = mat_ell[i,j,nell] alm[j,nelem].
 *
 * Arguments
 * ---------
 * lmat     : (ncomp * ncomp * nell) array with input matrix.
 * alm_in   : (ncomp * nell) array with input alm vector.
 * alm_out  : (ncomp * nell) array for output alm vector.
 * lmax     : Maximum multipole of both matrix and alm arrays.
 * ncomp    : Number of components of alm vector.
 */

void lmul_dp(const double *restrict lmat,
	     const double _Complex *restrict alm_in,
	     double _Complex *restrict alm_out,
	     int lmax,
	     int ncomp);

/*
 * Compute multiplication out:[i,nelem] = mat_ell[i,nell] alm[i,nelem].
 *
 * Arguments
 * ---------
 * lmat     : (ncomp * nell) array with input diagonal of matrix.
 * alm_in   : (ncomp * nell) array with input alm vector.
 * alm_out  : (ncomp * nell) array for output alm vector.
 * lmax     : Maximum multipole of both matrix and alm arrays.
 * ncomp    : Number of components of alm vector.
 */

void lmul_diag_dp(const double *restrict lmat,
	     const double _Complex *restrict alm_in,
	     double _Complex *restrict alm_out,
	     int lmax,
	     int ncomp);

/* 
 * Single precision versions.
 */

void lmul_sp(const float *restrict lmat,
	     const float _Complex *restrict alm_in,
	     float _Complex *restrict alm_out,
	     int lmax,
	     int ncomp);

void lmul_diag_sp(const float *restrict lmat,
	     const float _Complex *restrict alm_in,
	     float _Complex *restrict alm_out,
	     int lmax,
	     int ncomp);

