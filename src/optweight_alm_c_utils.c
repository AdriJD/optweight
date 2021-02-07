#include "optweight_alm_c_utils.h"

static inline int get_mstart(int m, int lmax){
  return m * (2 * lmax + 1 - m) / 2;
}

static int get_nelem(int lmax){
  // Assuming triangular layout with mmax = lmax.
  return (lmax * lmax + 3 * lmax) / 2 + 1;
}

void lmul_dp(const double *restrict lmat,
	     const double _Complex *restrict alm_in,
	     double _Complex *restrict alm_out,
	     int lmax,
	     int ncomp){

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    // Set output array to zero.
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

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    // Set output array to zero.
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

void wlm2alm_dp(const double *restrict w_ell,
		const double _Complex *restrict wlm,
		double _Complex *restrict alm,
		int lmax_w,
		int lmax_a,
		int ncomp){

  int nelem_a = get_nelem(lmax_a);
  int nelem_w = get_nelem(lmax_w);

  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax_w; m++){

      int mstart_w = get_mstart(m, lmax_w);
      int mstart_a = get_mstart(m, lmax_a);

      for (int ell=m; ell<=lmax_w; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem_a;
	  int wstart_i = idx * nelem_w;

	  alm[astart_i + mstart_a + ell] += w_ell[ell] 
	    * wlm[wstart_i + mstart_w + ell];  
        }
      }
    }
  } 
}

void lmul_inplace_dp(const double *restrict lmat,
         	     double _Complex *restrict alm,
	             int lmax,
	             int ncomp){

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    double _Complex *tmp = malloc(sizeof *tmp * ncomp);  

    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){
        
	for (int idx=0; idx<ncomp; idx++){
	  int astart_i = idx * nelem;
	  tmp[idx] = alm[astart_i + mstart + ell];
        }
	for (int idx=0; idx<ncomp; idx++){	

	  int astart_i = idx * nelem;
	  alm[astart_i + mstart + ell] = 0;

	  for (int jdx=0; jdx<ncomp; jdx++){

	    int astart_j = jdx * nelem;
	    int matstart = (idx * ncomp + jdx) * nell;       
	    alm[astart_i + mstart + ell] += lmat[matstart + ell] 
               * tmp[jdx];
	  }
	}
      }
    }
    free(tmp);  
  }
}

void lmul_diag_inplace_dp(const double *restrict lmat,
                	  double _Complex *restrict alm,
	                  int lmax,
	                  int ncomp){

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    double _Complex tmp;;  

    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){
        
	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem;
	  tmp = alm[astart_i + mstart + ell];
	  alm[astart_i + mstart + ell] = 0;
	  int matstart = idx * nell;       	    
	  alm[astart_i + mstart + ell] += lmat[matstart + ell] * tmp;	  
	}
      }
    }
  }
}

void trunc_alm_dp(const double _Complex *restrict alm,
		  double _Complex *restrict alm_out,
		  int lmax,
		  int lmax_out,
		  int ncomp){

  int nelem = get_nelem(lmax);
  int nelem_out = get_nelem(lmax_out);
  int nell_out = lmax_out + 1;

  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax_out; m++){

      int mstart = get_mstart(m, lmax);
      int mstart_out = get_mstart(m, lmax_out);

      for (int ell=m; ell<=lmax_out; ell++){
        
	for (int idx=0; idx<ncomp; idx++){

	  int astart = idx * nelem;
	  int astart_out = idx * nelem_out;      	    
	  alm_out[astart_out + mstart_out + ell] = alm[astart + mstart + ell];
	}
      }
    }
  }
}

/********* Single precision versions. **********/

void lmul_sp(const float *restrict lmat,
	     const float _Complex *restrict alm_in,
	     float _Complex *restrict alm_out,
	     int lmax,
	     int ncomp){

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    // Set output array to zero.
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

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    // Set output array to zero.
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

void wlm2alm_sp(const float *restrict w_ell,
		const float _Complex *restrict wlm,
		float _Complex *restrict alm,
		int lmax_w,
		int lmax_a,
		int ncomp){

  int nelem_a = get_nelem(lmax_a);

  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax_w; m++){

      int mstart_w = get_mstart(m, lmax_w);
      int mstart_a = get_mstart(m, lmax_a);

      for (int ell=m; ell<=lmax_w; ell++){

	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem_a;

	  alm[astart_i + mstart_a + ell] += w_ell[ell] 
	    * wlm[mstart_w + ell];  
        }
      }
    }
  } 
}

void lmul_inplace_sp(const float *restrict lmat,
	             float _Complex *restrict alm,
	             int lmax,
	             int ncomp){

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    float _Complex *tmp = malloc(sizeof *tmp * ncomp);  

    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){
        
	for (int idx=0; idx<ncomp; idx++){
	  int astart_i = idx * nelem;
	  tmp[idx] = alm[astart_i + mstart + ell];
        }
	for (int idx=0; idx<ncomp; idx++){	

	  int astart_i = idx * nelem;
	  alm[astart_i + mstart + ell] = 0;

	  for (int jdx=0; jdx<ncomp; jdx++){

	    int astart_j = jdx * nelem;
	    int matstart = (idx * ncomp + jdx) * nell;       
	    alm[astart_i + mstart + ell] += lmat[matstart + ell] 
               * tmp[jdx];
	  }
	}
      }
    }
    free(tmp);  
  }
}

void lmul_diag_inplace_sp(const float *restrict lmat,
	                  float _Complex *restrict alm,
	                  int lmax,
	                  int ncomp){

  int nelem = get_nelem(lmax);
  int nell = lmax + 1;

  #pragma omp parallel
  {
    float _Complex tmp;;  

    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax; m++){

      int mstart = get_mstart(m, lmax);

      for (int ell=m; ell<=lmax; ell++){
        
	for (int idx=0; idx<ncomp; idx++){

	  int astart_i = idx * nelem;
	  tmp = alm[astart_i + mstart + ell];
	  alm[astart_i + mstart + ell] = 0;
	  int matstart = idx * nell;       	    
	  alm[astart_i + mstart + ell] += lmat[matstart + ell] * tmp;	  
	}
      }
    }
  }
}

void trunc_alm_sp(const float _Complex *restrict alm,
		  float _Complex *restrict alm_out,
		  int lmax,
		  int lmax_out,
		  int ncomp){

  int nelem = get_nelem(lmax);
  int nelem_out = get_nelem(lmax_out);
  int nell_out = lmax_out + 1;

  #pragma omp parallel
  {
    #pragma omp for schedule(guided)
    for (int m=0; m<=lmax_out; m++){

      int mstart = get_mstart(m, lmax);
      int mstart_out = get_mstart(m, lmax_out);

      for (int ell=m; ell<=lmax_out; ell++){
        
	for (int idx=0; idx<ncomp; idx++){

	  int astart = idx * nelem;
	  int astart_out = idx * nelem_out;      	    
	  alm_out[astart_out + mstart_out + ell] = alm[astart + mstart + ell];
	}
      }
    }
  }
}
