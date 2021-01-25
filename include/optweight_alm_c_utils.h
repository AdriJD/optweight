#include <stdlib.h>
#include <complex.h>
#include <omp.h>

/*
 * Compute matrix multiplication out[i,nelem] = mat_ell[i,j,nell] alm[j,nelem].
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
 * Compute multiplication out[i,nelem] = mat_ell[i,nell] alm[i,nelem].
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
 * Compute alm[i,nelem] += w_ell[nell] wlm[i,nelem].
 *
 * Arguments
 * ---------
 * w_ell    : (nell) f_ell array.
 * wlm      : (ncomp * nell) array with input alm vector.
 * alm      : (ncomp * nell) array for output alm vector.
 * lmax_w   : Maximum multipole of wlm array.
 * lmax_a   : Maximum multipole of alm array.
 * ncomp    : Number of components of alm vector.
 */

void wlm2alm_dp(const double *restrict w_ell,
             const double _Complex *restrict wlm,
             double _Complex *restrict alm,
             int lmax_w,
             int lmax_a,
             int ncomp);

/*
 * Compute inplace matrix multiplication alm[i,nelem] = mat_ell[i,j,nell] alm[j,nelem].
 *
 * Arguments
 * ---------
 * lmat     : (ncomp * ncomp * nell) array with input matrix.
 * alm      : (ncomp * nell) array with alm vector.
 * lmax     : Maximum multipole of both matrix and alm arrays.
 * ncomp    : Number of components of alm vector.
 */

void lmul_inplace_dp(const double *restrict lmat,
             double _Complex *restrict alm,
             int lmax,
             int ncomp);

/*
 * Compute multiplication out[i,nelem] = mat_ell[i,nell] alm[i,nelem].
 *
 * Arguments
 * ---------
 * lmat     : (ncomp * nell) array with input diagonal of matrix.
 * alm      : (ncomp * nell) array with alm vector.
 * lmax     : Maximum multipole of both matrix and alm arrays.
 * ncomp    : Number of components of alm vector.
 */

void lmul_diag_inplace_dp(const double *restrict lmat,
             double _Complex *restrict alm,
             int lmax,
             int ncomp);

/*
 * Truncate alm to a smaller lmax.
 *
 * Arguments
 * ---------
 * alm      : (ncomp * nell) array with input alm vector.
 * alm_out  : (ncomp * nell_out) array with output alm vector.
 * lmax     : Maximum multipole of input alm array.
 * lmax_out : Maximum multipole of output alm array.
 * ncomp    : Number of components of alm vector.
 */

void trunc_alm_dp(const double _Complex *restrict alm,
             double _Complex *restrict alm_out,
             int lmax,
             int lmax_out,
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

void wlm2alm_sp(const float *restrict w_ell,
             const float _Complex *restrict wlm,
             float _Complex *restrict alm,
             int lmax_w,
             int lmax_a,
             int ncomp);

void lmul_inplace_sp(const float *restrict lmat,
             float _Complex *restrict alm,
             int lmax,
             int ncomp);

void lmul_diag_inplace_sp(const float *restrict lmat,
             float _Complex *restrict alm,
             int lmax,
             int ncomp);

void trunc_alm_sp(const float _Complex *restrict alm,
             float _Complex *restrict alm_out,
             int lmax,
             int lmax_out,
             int ncomp);
