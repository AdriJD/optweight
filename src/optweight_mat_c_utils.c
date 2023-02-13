#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <mkl.h>
#include "optweight_mat_c_utils.h"

void _eigpow_core_rsp(float *imat, const float power, const float lim,
		      const float lim0, const int nsamp,
		      const long long int ncomp){
    
#pragma omp parallel 
    {
		
        float *vecs = malloc(ncomp * ncomp * sizeof(float));
        float *tmp = malloc(ncomp * ncomp * sizeof(float));
        float *eigs = malloc(ncomp * sizeof(float));

	int eigval_step = 1;
	int matsize = ncomp * ncomp;	
	long long int info;
        long long int lwork = -1;
	float maxval;
	float meig;
	float worksize;

	// Initial call with lwork=-1 to determine size of work array.
        ssyev("V", "U", &ncomp, vecs, &ncomp, eigs, &worksize,
              &lwork, &info);
        lwork = (int) worksize;
        float *work = malloc(lwork * sizeof(float));	

	#pragma omp for schedule(dynamic, 256)
	for (int idx=0; idx<nsamp; idx++){

             cblas_scopy(matsize, &imat[idx * ncomp * ncomp], eigval_step,
                         vecs, eigval_step);
             ssyev("V", "U", &ncomp, vecs, &ncomp, eigs, work, &lwork, &info);	     

             // Find max eigenvalue.
	     maxval = find_max_sp(eigs, ncomp);
	     
             if (maxval < lim0){
                 // Set input matrix to zero.
                 for (int jdx=0; jdx<matsize; jdx++){
                     imat[idx * ncomp * ncomp + jdx] = 0.;
		 }
	     }
	     else {

                 meig = maxval * lim;

                 for (int jdx=0; jdx<ncomp; jdx++){

                     if (eigs[jdx] < meig){
                         // Set corresponding row in E V to zero.
                         for (int kdx=0; kdx<ncomp; kdx++){
                             tmp[jdx * ncomp + kdx] = 0.;
			 }
		     }
		     else {
			 // Compute D = E^power V for this eigenvalue.
			 for (int kdx=0; kdx<ncomp; kdx++){
                             tmp[jdx * ncomp + kdx] = pow(eigs[jdx], power) \
				 * vecs[jdx * ncomp + kdx];
			 }
		     }
		 }

		 // Compute V^T D (= V^T E^power V = A^T = A).
                 cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ncomp, ncomp,
			     ncomp, 1., vecs, ncomp, tmp, ncomp, 0.,
		 	     &imat[idx * ncomp * ncomp], ncomp);		         
	     }
	}	     
        free(tmp);
        free(vecs);
        free(eigs);
        free(work);
    }
}

void _eigpow_core_rdp(double *imat, const double power, const double lim,
		      const double lim0, const int nsamp,
		      const long long int ncomp){
    
#pragma omp parallel 
    {
		
        double *vecs = malloc(ncomp * ncomp * sizeof(double));
        double *tmp = malloc(ncomp * ncomp * sizeof(double));
        double *eigs = malloc(ncomp * sizeof(double));

	int eigval_step = 1;
	int matsize = ncomp * ncomp;	
	long long int info;
        long long int lwork = -1;
	double maxval;
	double meig;
	double worksize;

	// Initial call with lwork=-1 to determine size of work array.
        dsyev("V", "U", &ncomp, vecs, &ncomp, eigs, &worksize,
              &lwork, &info);
        lwork = (int) worksize;
        double *work = malloc(lwork * sizeof(double));	

	#pragma omp for schedule(dynamic, 256)
	for (int idx=0; idx<nsamp; idx++){

             cblas_dcopy(matsize, &imat[idx * ncomp * ncomp], eigval_step,
                         vecs, eigval_step);
             dsyev("V", "U", &ncomp, vecs, &ncomp, eigs, work, &lwork, &info);	     

             // Find max eigenvalue.
	     maxval = find_max_dp(eigs, ncomp);
	     
             if (maxval < lim0){
                 // Set input matrix to zero.
                 for (int jdx=0; jdx<matsize; jdx++){
                     imat[idx * ncomp * ncomp + jdx] = 0.;
		 }
	     }
	     else {

                 meig = maxval * lim;

                 for (int jdx=0; jdx<ncomp; jdx++){

                     if (eigs[jdx] < meig){
                         // Set corresponding row in E V to zero.
                         for (int kdx=0; kdx<ncomp; kdx++){
                             tmp[jdx * ncomp + kdx] = 0.;
			 }
		     }
		     else {
			 // Compute D = E^power V for this eigenvalue.
			 for (int kdx=0; kdx<ncomp; kdx++){
                             tmp[jdx * ncomp + kdx] = pow(eigs[jdx], power) \
				 * vecs[jdx * ncomp + kdx];
			 }
		     }
		 }

		 // Compute V^T D (= V^T E^power V = A^T = A).
                 cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ncomp, ncomp,
			     ncomp, 1., vecs, ncomp, tmp, ncomp, 0.,
		 	     &imat[idx * ncomp * ncomp], ncomp);		         
	     }
	}	     
        free(tmp);
        free(vecs);
        free(eigs);
        free(work);
    }
}

float find_max_sp(const float *arr, const int n_elements){

    float maxval = 0;
    
    for (int idx=0; idx<n_elements; idx++){
	if (arr[idx] > maxval){
	    maxval = arr[idx];
	}
    }
    return maxval;
}

float find_max_dp(const double *arr, const int n_elements){

    double maxval = 0;
    
    for (int idx=0; idx<n_elements; idx++){
	if (arr[idx] > maxval){
	    maxval = arr[idx];
	}
    }
    return maxval;
}
