cdef extern from "optweight_mat_c_utils.h":
    void _eigpow_core_rsp_c(float *imat,
                            float power,
                            float lim,
                            float lim0,
                            int nsamp,
                            int ncomp);