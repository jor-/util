import warnings

import numpy as np
import scipy.sparse

import util.math.matrix
import util.math.sparse.check

import util.logging
logger = util.logging.logger



def forward_substitution(L, b):
    logger.debug('Starting forward substitution for system of shape {}'.format(b.shape))
    
    ## check input
    L = util.math.sparse.check.sorted_squared_csr(L)
    if b.ndim not in [1, 2]:
        raise ValueError('b must have 1 or 2 dims but its shape is {}.'.format(b.shape))
    if L.shape[1] != b.shape[0]:
        raise ValueError('The size of the second dim of L must be equal to the size of the first dim of b but the shape of L is {} and the shape of b is {}.'.format(L.shape, b.shape))
    
    # ## make b two dim
    # b_ndim = b.ndim
    # if b_ndim == 1:
    #     b = b[:, np.newaxis]
    # n, m = b.shape
    
    ## init
    x = np.zeros(b.shape)
    column_start = L.indptr[0]
    
    ## fill x (forward)
    for i in range(len(b)):
        column_stop = L.indptr[i+1]
        
        ## check regularity and triangularity 
        if column_stop <= column_start:
            raise util.math.matrix.SingularMatrixError(L, 'The {}th row is zero!'.format(i))
        if L.indices[column_stop-1] > i:
            raise util.math.matrix.NoLeftTriangularMatrixError(L, 'The entry at ({},{}) is not zero!'.format(i, L.indices[column_stop-1]))
        if L.indices[column_stop-1] < i:
            raise util.math.matrix.SingularMatrixError(L, 'The {}th diagonal entry of the tridiagonal matrix is zero!'.format(i))
        
        ## compute value
        column_indices = L.indices[column_start:column_stop-1]    # skip diagonal entry
        assert np.all(column_indices[:-1] < column_indices[1:])
        data = L.data[column_start:column_stop-1]
        
        for j, Lij in zip(column_indices, data):
            x[i] -= Lij * x[j]
            assert j < i
        x[i] += b[i]
        x[i] /= L.data[column_stop-1]       # divide by ith diagonal entry
        
        ## next row
        column_start = column_stop
    
    ## return
    return x




def backward_substitution(R, b):
    logger.debug('Starting backward substitution for system of shape {}'.format(b.shape))
    
    ## check input
    R = util.math.sparse.check.sorted_squared_csr(R)
    if b.ndim not in [1, 2]:
        raise ValueError('b must have 1 or 2 dims but its shape is {}.'.format(b.shape))
    if R.shape[1] != b.shape[0]:
        raise ValueError('The size of the second dim of R must be equal to the size of the first dim of b but the shape of R is {} and the shape of b is {}.'.format(R.shape, b.shape))
    
    ## init
    x = np.zeros(b.shape)
    column_stop = R.indptr[n]
    
    ## fill x (backward)
    for i in range(len(b)-1, -1, -1):
        column_start = R.indptr[i]
        
        ## check regularity and triangularity 
        if column_stop <= column_start:
            raise util.math.matrix.SingularMatrixError(R, 'The {}th row is zero!'.format(i))
        if R.indices[column_start] < i:
            raise util.math.matrix.NoRightTriangularMatrixError(R, 'The entry at ({},{}) is not zero!'.format(i, R.indices[column_start]))
        if R.indices[column_start] > i:
            raise util.math.matrix.SingularMatrixError(R, 'The {}th diagonal entry of the tridiagonal matrix is zero!'.format(i))
        
        ## compute value
        column_indices = R.indices[column_start+1:column_stop]    # skip diagonal entry
        assert np.all(column_indices[:-1] < column_indices[1:])
        data = R.data[column_start+1:column_stop]
        
        for j, Rij in zip(column_indices, data):
            x[i] -= Rij * x[j]
            assert j > i
        x[i] += b[i]
        x[i] /= R.data[column_start]       # divide by ith diagonal entry
        
        ## next row
        column_stop = column_start
    
    ## return
    return x




def LR(L, R, b, P=None):
    logger.debug('Solving system of dim {} with LR factors'.format(len(b)))
    
    if P is not None:
        util.math.sparse.check.permutation_matrix(P)
        b = P * b
    
    x = forward_substitution(L, b)
    x = backward_substitution(R, x)
    
    if P is not None:
        x = P.transpose() * x
    
    return x


def cholesky(L, b, P=None):
    logger.debug('Solving system of dim {} with cholesky factors'.format(len(b)))
    
    ## convert L and R to csr format
    is_csr = scipy.sparse.isspmatrix_csr(L)
    is_csc = scipy.sparse.isspmatrix_csc(L)
    
    if not is_csr and not is_csc:
        warnings.warn('cholesky requires L be in CSR or CSC matrix format. Converting matrix.', scipy.sparse.SparseEfficiencyWarning)
    
    if is_csc:
        R = L.transpose()
    if not is_csr:
        L = L.tocsr()
    if not is_csc:
        R = L.transpose().tocsr()
    
    assert scipy.sparse.isspmatrix_csr(L)
    assert scipy.sparse.isspmatrix_csr(R)
    
    ## compute
    return LR(L, R, b, P=P)
    
    