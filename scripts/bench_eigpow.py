import numpy as np
from pixell import utils

mat = np.ones((3, 3, 1000000))

utils.eigpow(mat, 0.5, axes=[0, 1])
