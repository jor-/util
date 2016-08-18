

import numpy as np
import scipy.io

def save(file, A, precision=None):
    if precision is not None:
        precision_min = int(np.ceil(np.log10(max(A.shape))))
        if precision < precision_min:
            warnings.warn('Precision must be at least {1} to represent all indices. Precision changed from {2} to {1}.'.format(precision_min, precision))
            precision = precision_min

    scipy.io.mmwrite(file, A, precision=precision)

def load(file):
    return scipy.io.mmread(file)