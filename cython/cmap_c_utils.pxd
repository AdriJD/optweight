cdef extern from "optweight_map_c_utils.h":
    void _apply_ringweight_core_sp(float *imap,
    	                           const double *weight,
				   const long *nphi,
				   const long *offsets,
				   const long *stride,
				   const long nrow);
    void _apply_ringweight_core_dp(double *imap,
    	                           const double *weight,
				   const long *nphi,
				   const long *offsets,
				   const long *stride,
				   const long nrow);