#include "optweight_alm_c_utils_test.h"

// _trunc_alm

// _wlm2alm

static inline int get_mstart(int m, int lmax){
  return m * (2 * lmax + 1 - m) / 2;
}

void lmul_dp(const double *restrict lmat,
	     const double _Complex *restrict alm_in,
	     double _Complex *restrict alm_out,
	     int lmax,
	     int ncomp){

  // Assuming triangular layout with mmax = lmax.
  int nelem = (lmax * lmax + 3 * lmax) / 2 + 1;
  int nell = lmax + 1;

  // Set output array to zero.
  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart = idx * nelem;
	  alm_out[astart + mstart + ell] = 0;
	}
      }
    }

    // Actual matrix multiplication.
    // Weirdly enough, this access pattern gives less cache misses 
    // and thus a bit better performance. Perhaps easier prefetching?
    // OMP False sharing issues with lmat matrix seem to be not significant.
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem;

	  for (int jdx=0; jdx<ncomp; jdx++){

	    int astart_j = jdx * nelem;
	    int matstart = (idx * ncomp + jdx) * nell;       
	    
	    alm_out[astart_i + mstart + ell] += lmat[matstart + ell] 
	      * alm_in[astart_j + mstart + ell];
	  }
	}
      }
    }
  }
}

void lmul_diag_dp(const double *restrict lmat,
		  const double _Complex *restrict alm_in,
		  double _Complex *restrict alm_out,
		  int lmax,
		  int ncomp){

  // Assuming triangular layout with mmax = lmax.
  int nelem = (lmax * lmax + 3 * lmax) / 2 + 1;
  int nell = lmax + 1;

  // Set output array to zero.
  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      //int mstart = m * (2 * lmax + 1 - m) / 2;
      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart = idx * nelem;
	  alm_out[astart + mstart + ell] = 0;
	}
      }
    }

    // Diagonal matrix multiplication.
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem;
	  int matstart = idx * nell;       
	    
	  alm_out[astart_i + mstart + ell] += lmat[matstart + ell] 
	    * alm_in[astart_i + mstart + ell];  
        }
      }
    }
  } 
}
      
// Single precision versions.

void lmul_sp(const float *restrict lmat,
	     const float _Complex *restrict alm_in,
	     float _Complex *restrict alm_out,
	     int lmax,
	     int ncomp){

  // Assuming triangular layout with mmax = lmax.
  int nelem = (lmax * lmax + 3 * lmax) / 2 + 1;
  int nell = lmax + 1;

  // Set output array to zero.
  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart = idx * nelem;
	  alm_out[astart + mstart + ell] = 0;
	}
      }
    }

    // Actual matrix multiplication.
    // Weirdly enough, this access pattern gives less cache misses 
    // and thus a bit better performance. Perhaps easier prefetching?
    // OMP False sharing issues with lmat matrix seem to be not significant.
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem;

	  for (int jdx=0; jdx<ncomp; jdx++){

	    int astart_j = jdx * nelem;
	    int matstart = (idx * ncomp + jdx) * nell;       
	    
	    alm_out[astart_i + mstart + ell] += lmat[matstart + ell] 
	      * alm_in[astart_j + mstart + ell];
	  }
	}
      }
    }
  }
}

void lmul_diag_sp(const float *restrict lmat,
		  const float _Complex *restrict alm_in,
		  float _Complex *restrict alm_out,
		  int lmax,
		  int ncomp){

  // Assuming triangular layout with mmax = lmax.
  int nelem = (lmax * lmax + 3 * lmax) / 2 + 1;
  int nell = lmax + 1;

  // Set output array to zero.
  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      //int mstart = m * (2 * lmax + 1 - m) / 2;
      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart = idx * nelem;
	  alm_out[astart + mstart + ell] = 0;
	}
      }
    }

    // Diagonal matrix multiplication.
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem;
	  int matstart = idx * nell;       
	    
	  alm_out[astart_i + mstart + ell] += lmat[matstart + ell] 
	    * alm_in[astart_i + mstart + ell];  
        }
      }
    }
  } 
}
