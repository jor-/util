import numpy as np
import scipy.sparse
import warnings

import util.logging
logger = util.logging.logger


def squared(A):
    assert A.ndim == 2 and A.shape[0] == A.shape[1]


def sorted_squared_csr(A):
    squared(A)

    if not scipy.sparse.isspmatrix_csr(A):
        warnings.warn('CSR matrix format is required. Converting to CSR matrix.', scipy.sparse.SparseEfficiencyWarning)
        A = scipy.sparse.csr_matrix(A)

    A.sort_indices()
    A.eliminate_zeros()
    return A


def sorted_squared_csc(A):
    squared(A)

    if not scipy.sparse.isspmatrix_csc(A):
        warnings.warn('CSC matrix format is required. Converting to CSC matrix.', scipy.sparse.SparseEfficiencyWarning)
        A = scipy.sparse.csc_matrix(A)

    A.sort_indices()
    A.eliminate_zeros()
    return A


def permutation_matrix(A):
    squared(A)
    assert A.nnz == A.shape[0]


def min_dtype(A, min_dtype):
    min_dtype = np.dtype(min_dtype)
    if A.dtype < min_dtype:
        logger.debug('Converting matrix {!r} to type {}.'.format(A, dtype))
        A = A.astype(dtype)
    return A


def index_dtype(A, dtype):
    if not (scipy.sparse.isspmatrix_csc(A) or scipy.sparse.isspmatrix_csr(A)):
        raise NotImplementedError("Only CSR and CSC are supported yet.")
    A.indices = np.asarray(A.indices, dtype=dtype)
    A.indptr = np.asarray(A.indptr, dtype=dtype)
    return A