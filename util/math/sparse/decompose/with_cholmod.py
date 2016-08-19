import warnings
import os.path

import numpy as np
import scipy.sparse
import scikits.sparse.cholmod

import util.math.matrix
import util.math.sparse.check

import util.logging
logger = util.logging.logger


RETURN_L = 'L'
RETURN_L_D = 'L_D'
RETURN_P_L = 'P_L'
RETURN_P_L_D = 'P_L_D'
CHOLMOD_ORDERING_METHODS = ('natural', 'default', 'best')



def approximate_positive_definite(A, min_abs_value=0, ordering_method='natural', min_diag_value=10**(-4), reorder_after_each_step=True, use_long=False, reduction_factors_file=None):
    logger.debug('Calculating positive definite approximation of matrix {!r} with min diagonal entry {}, min absolute value {}, ordering_method {}, reorder_after_each_step {}, use_long {} and reduction_factors_file {}.'.format(A, min_diag_value, min_abs_value, ordering_method, reorder_after_each_step, use_long, reduction_factors_file))
    
    assert min_diag_value > 0

    max_reduction_factor = 1 - 10**-6
    def multiply_off_diagonal_entries_with_factor(i, reduction_factor):
        # logger.debug('Multiplying off diagonal entries of row and column {} with {}.'.format(i, reduction_factor))
        
        ## check reduction_factor
        if not np.isfinite(reduction_factor):
            raise ValueError('Reduction factor {} with index {} has to be finite!'.format(reduction_factor, i))
        if not reduction_factor < 1:
            raise ValueError('Reduction factor {} with index {} has to be < 1!'.format(reduction_factor, i))
        if not reduction_factor >= 0:
            raise ValueError('Reduction factor {} with index {} has to be >= 0!'.format(reduction_factor, i))
        
        ## check max reduction factor
        if reduction_factor > max_reduction_factor:
            logger.warning('Current reduction factor {} for row and column {} is greater then max reduction factor {}. Past total reducing reduction factor is {}. Changing reduction factor to max reduction factor.'.format(reduction_factor, i, max_reduction_factor, reduction_factors[i]))
            reduction_factor = max_reduction_factor
        
        ## save reduction factor
        if reduction_factors_file is not None:
            reduction_factors[i] *= reduction_factor
            np.save(reduction_factors_file, reduction_factors)

        ## get indices
        A_ii = A[i,i]
        A_i_start_index = A.indptr[i]
        A_i_stop_index = A.indptr[i+1]
        assert A_i_stop_index - A_i_start_index > 1

        ## set column
        A.data[A_i_start_index:A_i_stop_index] *= reduction_factor

        ## set 0 for low values
        for k in range(A_i_start_index, A_i_stop_index):
            if np.abs(A.data[k]) < min_abs_value:
                A.data[k] = 0

        ## set row
        A_i_data = A.data[A_i_start_index:A_i_stop_index]
        A_i_rows = A.indices[A_i_start_index:A_i_stop_index]
        for j, A_ji in zip(A_i_rows, A_i_data):
            if i != j:
                A[i, j] = A_ji

        ## set diagonal entry
        A[i,i] = A_ii

        ## eliminate zeros
        if reorder_after_each_step or ordering_method == 'natural':
            A.eliminate_zeros()
    
    
    def get_p_i(i, f):        
        if ordering_method == 'natural':
            p_i = i
        else:
            p_i = f.P()[i]
        return p_i
        
    

    ## check input
    #TODO symmetry check
    A = util.math.sparse.check.sorted_squared_csc(A)
    A = util.math.sparse.check.min_dtype(A, np.float32)
    if use_long:
        A = util.math.sparse.check.index_dtype(A, np.int64)

    ## init
    n = A.shape[0]
    resolution = np.finfo(A.dtype).resolution
    if min_abs_value < resolution:
        logger.warning('Setting min abs value to resolution {} of data type.'.format(resolution))
        min_abs_value = resolution
    if min_diag_value < resolution:
        logger.warning('Setting min diag value to resolution {} of data type.'.format(resolution))
        min_diag_value = resolution

    ## apply reduction file values
    if reduction_factors_file is not None and os.path.exists(reduction_factors_file):
        reduction_factors = np.load(reduction_factors_file)
        for i in np.where((reduction_factors != 1))[0]:
            try:
                multiply_off_diagonal_entries_with_factor(i, reduction_factors[i])
            except ValueError as e:
                raise ValueError('Reduction factors file is wrong.') from e
    else:
        reduction_factors = np.ones(n, dtype=A.dtype)

    ## remove values below min abs
    mask = np.abs(A.data) < min_abs_value
    A.data[mask] = 0
    del mask
    A.eliminate_zeros()

    ## calculate positive definite approximation of A
    logger.debug('Checking if matrix is positive definite.')
    finished = False
    while not finished:
        
        ## compute cholesky factor
        try:
            try:
                f = scikits.sparse.cholmod.cholesky(A, ordering_method=ordering_method, use_long=use_long)
            except scikits.sparse.cholmod.CholmodTooLargeError as e:
                if not use_long:
                    warnings.warn('Problem to large for int, switching to long.')
                    return approximate_positive_definite(A, min_diag_value=min_diag_value, min_abs_value=min_abs_value, use_long=True)
                else:
                    raise
        except scikits.sparse.cholmod.CholmodNotPositiveDefiniteError as e:
            f = e.factor
        
        ## calculate LD and d
        LD = f.LD()     # Do not use f.L_D() -> higher memory consumption
        assert scipy.sparse.isspmatrix_csc(LD)
        d = LD.diagonal()
        
        ## check if all diagonal entries >= min_diag_value
        i_array = np.where(np.logical_not(np.logical_or(d >= min_diag_value, np.isclose(d, min_diag_value, atol=0, rtol=10**-4))))[0]
        finished = len(i_array) == 0
        
        ## if not, reduce off diagonal entries at row and column
        if not finished:
            ## get i and p_i
            i = i_array[0]
            p_i = get_p_i(i, f)
            del f, i_array
            
            ## check diagonal entry
            A_ii = A[p_i, p_i]
            if A_ii < min_diag_value:
                raise util.math.matrix.NoPositiveDefiniteMatrixError(A, 'Diagonal values of matrix must be at least {} to be able to ensure the minimum value, but the {}th entry is {}.'.format(min_diag_value, p_i, A_ii))
    
            ## get values for calulation of reduction factors
            assert scipy.sparse.isspmatrix_csc(LD)
            L_i = LD[i].tocsr()
            L_i_columns = L_i.indices
            L_i_data = L_i.data
            assert len(L_i_data) == len(L_i_columns) >= 2
            assert L_i_columns[-1] == i
    
            ## calculate reduction factor
            s = 0
            for j, L_ij in zip(L_i_columns[:-1], L_i_data[:-1]):
                assert j < i
                s += L_ij**2 * d[j]
            reduction_factor_i = ((A_ii - min_diag_value) / s)**(1/2)
        
            ## multiply off diagonal entries with reduction factor
            logger.debug('Row {} of cholesky decomposition, corresponding to row/column {} of input matrix, has diagonal value {} which is less than the min value {}. Multiplying off diagonal entries with {}.'.format(i, p_i, d[i], min_diag_value, reduction_factor_i))
            del LD, d, L_i, L_i_columns, L_i_data
            multiply_off_diagonal_entries_with_factor(p_i, reduction_factor_i)
            
    
    ## return
    A.eliminate_zeros()
    logger.debug('Returning reduction factors with average reduction factor {} and positive definite matrix {!r}.'.format(reduction_factors.mean(), A))
    return (A, reduction_factors)



def cholesky(A, ordering_method='default', return_type=RETURN_P_L, use_long=False):
    '''
    P A P' = L L'
    '''
    logger.debug('Calculating cholesky decomposition for matrix {!r} with ordering method {}, return type {} and use_long {}.'.format(A, ordering_method, return_type, use_long))

    ## check input
    return_types = (RETURN_L, RETURN_L_D, RETURN_P_L, RETURN_P_L_D)
    if ordering_method not in CHOLMOD_ORDERING_METHODS:
        raise ValueError('Unknown ordering method {}. Only values in {} are supported.'.format(ordering_method, CHOLMOD_ORDERING_METHODS))
    if return_type not in return_types:
        raise ValueError('Unknown return type {}. Only values in {} are supported.'.format(return_type, return_types))
        if ordering_method != 'natural' and return_type in (RETURN_L, RETURN_L_D):
            raise ValueError('Return type {} is only supported for "natural" ordering method.'.format(return_type))

    #TODO symmetry check
    A = util.math.sparse.check.sorted_squared_csc(A)

    ## calculate cholesky decomposition
    try:
        try:
            f = scikits.sparse.cholmod.cholesky(A, ordering_method=ordering_method, use_long=use_long)
        except scikits.sparse.cholmod.CholmodTooLargeError as e:
            if not use_long:
                warnings.warn('Problem to large for int, switching to long.')
                return cholesky(A, ordering_method=ordering_method, return_type=return_type, use_long=True)
            else:
                raise
    except scikits.sparse.cholmod.CholmodNotPositiveDefiniteError as e:
        raise util.math.matrix.NoPositiveDefiniteMatrixError(A, 'Row/column {} makes matrix not positive definite.'.format(e.column))
    del A

    ## calculate permutation matrix
    p = f.P()
    n = len(p)
    if return_type in (RETURN_P_L, RETURN_P_L_D):
        P = scipy.sparse.dok_matrix((n,n), dtype=np.int8)
        for i in range(n):
            P[i,p[i]] = 1
        P = P.tocsr()
        P.astype(np.int8)

    ## return P, L
    if return_type in (RETURN_L, RETURN_P_L):
        L = f.L().tocsr()
        if return_type == RETURN_L:
            assert np.all(p == np.arange(n))
            logger.debug('Returning lower triangular matrix {!r}.'.format(L))
            return (L,)
        else:
            logger.debug('Returning permutation matrix {!r} and lower triangular matrix {!r}.'.format(P, L))
            return (P, L)

    ## return P, L, D
    if return_type in (RETURN_L_D, RETURN_P_L_D):
        L, D = f.L_D()
        # Do not use f.L_D() -> higher memory consumption
        # LD = f.LD()
        if return_type == RETURN_L_D:
            logger.debug('Returning lower triangular matrix {!r} and diagonal matrix {!r}.'.format(P, L, D))
            return (L, D)
        else:
            logger.debug('Returning permutation matrix {!r}, lower triangular matrix {!r} and diagonal matrix {!r}.'.format(P, L, D))
            return (P, L, D)



