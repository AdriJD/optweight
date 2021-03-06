cdef extern from "optweight_alm_c_utils.h":
    void lmul_dp(const double *lmat,
                 const double complex *alm_in,
                 double complex *alm_out,
                 int lmax,
                 int ncomp);

    void lmul_diag_dp(const double *lmat,
                 const double complex *alm_in,
                 double complex *alm_out,
                 int lmax,
                 int ncomp);

    void wlm2alm_dp(const double *w_ell,
                 const double complex *wlm,
                 double complex *alm,
                 int lmax_w,
                 int lmax_a,
                 int ncomp);

    void lmul_inplace_dp(const double *lmat,
                 const double complex *alm_in,
                 int lmax,
                 int ncomp);

    void lmul_diag_inplace_dp(const double *lmat,
                 double complex *alm,
                 int lmax,
                 int ncomp);

    void trunc_alm_dp(const double complex *alm,
                 double complex *alm_out,
                 int lmax,
                 int lmax_out,
                 int ncomp)


    void lmul_sp(const float *lmat,
                 const float complex *alm_in,
                 float complex *alm_out,
                 int lmax,
                 int ncomp);

    void lmul_diag_sp(const float *lmat,
                 const float complex *alm_in,
                 float complex *alm_out,
                 int lmax,
                 int ncomp);

    void wlm2alm_sp(const float *w_ell,
                 const float complex *wlm,
                 float complex *alm,
                 int lmax_w,
                 int lmax_a,
                 int ncomp);

    void lmul_inplace_sp(const float *lmat,
                 float complex *alm,
                 int lmax,
                 int ncomp);

    void lmul_diag_inplace_sp(const float *lmat,
                 float complex *alm,
                 int lmax,
                 int ncomp);

    void trunc_alm_sp(const float complex *alm,
                 float complex *alm_out,
                 int lmax,
                 int lmax_out,
                 int ncomp)
