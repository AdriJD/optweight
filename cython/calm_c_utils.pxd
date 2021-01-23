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
