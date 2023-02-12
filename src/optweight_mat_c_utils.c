#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
//#include <mkl_cblas.h>
#include <mkl.h>

void _eigpow_core_rsp_c(float *imat, float power, float lim, float lim0,
                        int nsamp, int ncomp){
    int info;
    float worksize;
    int lwork = -1;
    int max_idx;
    int eigval_step = 1;

    int matsize = ncomp * ncomp;
	
    float maxval;
    float meig;

    //char* jobz = 'V';
    //char* uplo = 'U';
    
    //char* transa = 'T';
    //char* transb = 'N';
    CBLAS_LAYOUT transa = 'T';
    CBLAS_LAYOUT transb = 'N';

    //float alpha = 1.;
    //float beta = 0.;

    int idx;
    int jdx;
    int kdx;
    int thread_id = -1;

//#pragma omp parallel firstprivate(power, lim, lim0, nsamp, ncomp, matsize)
#pragma omp parallel private(maxval, meig, max_idx, worksize, info) firstprivate(lwork)
    {
		
        float *vecs = malloc(ncomp * ncomp * sizeof(float));
        float *tmp = malloc(ncomp * ncomp * sizeof(float));
        float *eigs = malloc(ncomp * sizeof(float));

        lwork = -1;

        //cblas_ssyev(jobz, uplo, &ncomp, vecs, &ncomp, eigs, &worksize,
        //            &lwork,  &info);
        //ssyev(jobz, uplo, ncomp, vecs, ncomp, eigs, worksize,
        //      lwork, info);

	
        ssyev("V", "U", &ncomp, vecs, &ncomp, eigs, &worksize,
              &lwork, &info);

        lwork = (int) worksize;
        float *work = malloc(lwork * sizeof(float));	

	//printf("%d\n", lwork);

	//printf("%d \n", omp_get_thread_num());
	
        //for idx in prange(nsamp, schedule='static', chunksize=40):
        //for (int idx=0; idx<nsamp; idx++){
	#pragma omp for schedule(dynamic, 256)
	for (int idx=0; idx<nsamp; idx++){

	    //printf("%d %d \n", omp_get_thread_num(), idx);
	     // Copy current slice of input matrix into vecs.
             //cblas_scopy(&matsize, &imat[idx * ncomp * ncomp], &eigval_step,
             //            vecs, &eigval_step)
	    
	     //for (int jdx=0; jdx<matsize; jdx++){		
	    // printf("imat[%d] : %f\n", jdx, *(imat + (idx * ncomp * ncomp) + jdx));
	    //}
	    
             //cblas_scopy(&matsize, &imat[idx * ncomp * ncomp], &eigval_step,
             //            vecs, &eigval_step);

	    
             cblas_scopy(matsize, &imat[idx * ncomp * ncomp], eigval_step,
                         vecs, eigval_step);



	     
	     //for (int jdx=0; jdx<matsize; jdx++){		
	     //	 printf("vecs[%d] : %f\n", jdx, vecs[jdx]);
	     //}


             ssyev("V", "U", &ncomp, vecs, &ncomp, eigs, work, &lwork, &info);	     

	     //for (int jdx=0; jdx<ncomp; jdx++){
//		 printf("eigs[%d] : %f\n", jdx, eigs[jdx]);
//	     }
//	     for (int jdx=0; jdx<matsize; jdx++){
//		 printf("vecs[%d] : %f\n", jdx, vecs[jdx]);
//	     }
	     
             // Find max eigenvalue.
             //max_idx = cblas_isamax(&ncomp, eigs, &eigval_step);
	     max_idx = cblas_isamax(ncomp, eigs, eigval_step);
             maxval = eigs[max_idx];
	     //printf("maxval %f \n", maxval);

	     
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
			 //printf("bla \n");
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
                 cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ncomp, ncomp, ncomp, 1., vecs, ncomp, tmp, ncomp, 0.,
		 	     &imat[idx * ncomp * ncomp], ncomp);
		 
		 //for (int jdx=0; jdx<matsize; jdx++){		
		 //    printf("imat[%d] : %f\n", jdx, *(imat + (idx * ncomp * ncomp) + jdx));
		 //}

        
    }


}
	     

        free(tmp);
        free(vecs);
        free(eigs);
        free(work);
    }
}
