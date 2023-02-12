cimport cmat_c_utils

from cython.parallel import parallel, prange, threadid
from cython import boundscheck, wraparound
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport pow
import numpy as np
cimport numpy as np
np.import_array()
from scipy.linalg.cython_lapack cimport ssyev
from scipy.linalg.cython_blas cimport isamax, idamax, scopy, sgemm

#from cython.cimports.openmp cimport omp_set_max_active_levels

def call():

    imat = np.ones((3, 3, 2)) * np.eye(3)[:,:,None] * np.asarray([1, 3, 20])[:,None,None]
    #imat[:] = np.asarray([[2, 1, 0.5], [0, 4, 0.1], [0, 0.2, 20]])[:,:,None]
    print(imat)
    #print(eigpow(imat.astype(np.float32), 1, 1, 1))
    print(eigpow(imat.astype(np.float32), 0.5, 1e-6, 1e-10))

def eigpow(imat, power, lim, lim0):
    '''
    Port of Enlib's eigpow code. Raises a positive (semi)definite 
    matrix to an arbitrairy real power.

    Parameters
    ----------
    imat : (ncomp, ncomp, nsamp) array
            
        
    '''

    ncomp = imat.shape[0]
    nsamp = imat.shape[2]

    # Transpose and copy matrix. Assume worth it to reduce cache misses later.
    imat = np.ascontiguousarray(np.transpose(imat, (2, 0, 1)))

    cdef float [::1] imat_ = imat.reshape(-1)
    _eigpow_core_rsp(&imat_[0], power, lim, lim0, nsamp, ncomp)

    return np.ascontiguousarray(np.transpose(imat, (1, 2, 0)))

def eigpow2(imat, power, lim, lim0):
    '''
    Port of Enlib's eigpow code. Raises a positive (semi)definite 
    matrix to an arbitrairy real power.

    Parameters
    ----------
    imat : (nsamp, ncomp, ncomp) array
            
        
    '''

    ncomp = imat.shape[1]
    nsamp = imat.shape[0]

    cdef float [::1] imat_ = imat.reshape(-1)
    _eigpow_core_rsp(&imat_[0], power, lim, lim0, nsamp, ncomp)

    return imat

def eigpow3(imat, power, lim, lim0):
    '''
    Port of Enlib's eigpow code. Raises a positive (semi)definite 
    matrix to an arbitrairy real power.

    Parameters
    ----------
    imat : (nsamp, ncomp, ncomp) array
            
        
    '''

    ncomp = imat.shape[1]
    nsamp = imat.shape[0]

    cdef float [::1] imat_ = imat.reshape(-1)
    cmat_c_utils._eigpow_core_rsp_c(&imat_[0], power, lim, lim0, nsamp, ncomp)

    return imat

@boundscheck(False)
@wraparound(False)
cdef void _eigpow_core_rsp(float *imat, float power, float lim, float lim0,
                           int nsamp, int ncomp) nogil:
    '''
    imat : (nsamp, ncomp, ncomp) array
        Input array, modified in-place.
    power : float
        Raise matrix to this power.
    ''' 
     
    cdef int info
    cdef float worksize
    cdef int lwork = -1
    cdef int max_idx
    cdef int eigval_step = 1

    cdef int matsize = ncomp * ncomp

    cdef float maxval 
    cdef float meig

    cdef char* jobz = 'V'
    cdef char* uplo = 'U'
    
    cdef char* transa = 'T'
    cdef char* transb = 'N'
    cdef float alpha = 1.
    cdef float beta = 0.

    cdef int idx
    cdef int jdx
    cdef int kdx
    cdef int thread_id = -1

    #cdef int num_levels = 1
    #omp_set_max_active_levels(num_levels)

    with parallel():

        vecs = <float*>malloc(ncomp * ncomp * sizeof(float))
        tmp = <float*>malloc(ncomp * ncomp * sizeof(float))
        eigs = <float*>malloc(ncomp * sizeof(float))

        lwork = -1

        ssyev(jobz, uplo, &ncomp, vecs, &ncomp, eigs, &worksize,
              &lwork,  &info)

        lwork = <int>worksize
        work = <float*>malloc(lwork * sizeof(float))

        for idx in prange(nsamp, schedule='static', chunksize=40):

             # Copy current slice of input matrix into vecs.
             scopy(&matsize, &imat[idx * ncomp * ncomp], &eigval_step,
                   vecs, &eigval_step)

             ssyev(jobz, uplo, &ncomp, vecs, &ncomp, eigs, work, &lwork, &info)  

             #for jdx in range(ncomp):
             #    printf("eigs[%d] : %f\n", jdx, eigs[jdx])

             #for jdx in range(matsize):
             #    printf("vecs[%d] : %f\n", jdx, vecs[jdx])

             # Find max eigenvalue.
             max_idx = isamax(&ncomp, eigs, &eigval_step)
             maxval = eigs[max_idx - 1]

             thread_id = threadid()   
             #printf("thread %d %d\n", thread_id, idx)

             #printf("lim0 %f\n", lim0)

             if maxval < lim0:     
    
                 # Set input matrix to zero.
                 for jdx in range(matsize):
                     imat[idx * ncomp * ncomp + jdx] = 0.

                 #for jdx in range(matsize):
                 #    printf("%d : %f\n", jdx, imat[idx * ncomp * ncomp + jdx])

             else:

                 meig = maxval * lim

                 for jdx in range(ncomp):

                     if eigs[jdx] < meig:
                         # Set corresponding row in E V to zero.
                         for kdx in range(ncomp):
                             tmp[jdx * ncomp + kdx] = 0.
                     else:
                         # Compute D = E^power V for this eigenvalue.
                         for kdx in range(ncomp):
                             tmp[jdx * ncomp + kdx] = pow(eigs[jdx], power) \
                                * vecs[jdx * ncomp + kdx]

                 # Compute V^T D (= V^T E^power V = A^T = A).
                 sgemm(transa, transb, &ncomp, &ncomp, &ncomp, &alpha, vecs, &ncomp, tmp, &ncomp, &beta,
                       &imat[idx * ncomp * ncomp], &ncomp)
             

        free(tmp)
        free(vecs)
        free(eigs)
        free(work)

