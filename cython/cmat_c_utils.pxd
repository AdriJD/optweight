cdef extern from "optweight_mat_c_utils.h":
    void _eigpow_core_rsp(float *imat,
                          const float power,
                          const float lim,
                          const float lim0,
                          const int nsamp,
                          const int ncomp);
    void _eigpow_core_rdp(double *imat,
                          const double power,
                          const double lim,
                          const double lim0,
                          const int nsamp,
                          const int ncomp);