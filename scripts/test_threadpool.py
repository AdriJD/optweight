import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
import time
from optweight import mat_c_utils

mat = (np.ones((10_000_000, 3, 3), dtype=np.float32) * np.eye(3)[None, :, :]).astype(np.float32)
#mat = (np.ones((1, 5000, 5000), dtype=np.float32) * np.eye(5000)[None, :, :]).astype(np.float32)

with threadpool_limits(limits=1, user_api='blas') as threadpoolctx:
#with threadpool_limits(limits=1, user_api='openmp') as threadpoolctx:
#with threadpool_limits(limits="sequential_blas_under_openmp", user_api='blas') as threadpoolctx:
    #max_threads = threadpoolctx.get_original_num_threads()["openmp"]
    t0 = time.time()
    result = mat_c_utils.eigpow2(mat, 1, 0.1, 0.1)
    print(time.time() - t0)

t0 = time.time()
result = mat_c_utils.eigpow2(mat, 1, 0.1, 0.1)
print(time.time() - t0)
