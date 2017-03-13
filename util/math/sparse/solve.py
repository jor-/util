import warnings

import scipy.sparse
import scipy.sparse.linalg

import util.math.sparse.check

import util.logging
logger = util.logging.logger



def LU(L, U, b, P=None):
    logger.debug('Solving system of dim {} with LU factors'.format(len(b)))

    if P is not None:
        util.math.sparse.check.permutation_matrix(P)
        b = P * b

    x = scipy.sparse.linalg.spsolve_triangular(L, b, lower=True)
    x = scipy.sparse.linalg.spsolve_triangular(U, x, lower=False)

    if P is not None:
        x = P.transpose() * x

    return x



def cholesky(L, b, P=None):
    '''
    P A P' = L L'
    '''

    logger.debug('Solving system of dim {} with cholesky factors'.format(len(b)))

    ## convert L and U to csr format
    is_csr = scipy.sparse.isspmatrix_csr(L)
    is_csc = scipy.sparse.isspmatrix_csc(L)

    if not is_csr and not is_csc:
        warnings.warn('cholesky requires L be in CSR or CSC matrix format. Converting matrix.', scipy.sparse.SparseEfficiencyWarning)

    if is_csc:
        U = L.transpose()
    if not is_csr:
        L = L.tocsr()
    if not is_csc:
        U = L.transpose().tocsr()

    assert scipy.sparse.isspmatrix_csr(L)
    assert scipy.sparse.isspmatrix_csr(U)

    ## compute
    return LU(L, U, b, P=P)
