#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "optweight_map_c_utils.h"

void _apply_ringweight_core_sp(float *imap, const double *weight, const long *nphi,
			       const long *offsets, const long *stride, const long nrow){

#pragma omp parallel
    {
    #pragma omp for schedule(dynamic, 4)	
    for (int tidx=0; tidx<nrow; tidx++){

	for (int phidx=0; phidx<nphi[tidx]; phidx++){
	
	    imap[offsets[tidx] + phidx * stride[tidx]] *= (float)weight[tidx];
	}
    }
    }
}

void _apply_ringweight_core_dp(double *imap, const double *weight, const long *nphi,
			       const long *offsets, const long *stride, const long nrow){

#pragma omp parallel
    {
    #pragma omp for schedule(dynamic, 4)    
    for (int tidx=0; tidx<nrow; tidx++){

	for (int phidx=0; phidx<nphi[tidx]; phidx++){
	
	    imap[offsets[tidx] + phidx * stride[tidx]] *= weight[tidx];
	}
    }
    }
}
