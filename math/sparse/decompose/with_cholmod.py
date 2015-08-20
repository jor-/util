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



def approximate_positive_definite(A, min_diag_entry=10**(-4), min_abs_value=0, ordering_method='natural', reorder_after_each_step=True, use_long=False, reduction_factors_file=None):
    logger.debug('Calculating positive definite approximation of matrix {!r} with min diagonal entry {}, min absolute value {}, ordering_method {}, reorder_after_each_step {}, use_long {} and reduction_factors_file {}.'.format(A, min_diag_entry, min_abs_value, ordering_method, reorder_after_each_step, use_long, reduction_factors_file))

    def multiply_off_diagonal_entries_with_factor(i, factor):
        assert factor >= 0 and factor < 1

        ## get indices
        A_ii = A[i,i]
        A_i_start_index = A.indptr[i]
        A_i_stop_index = A.indptr[i+1]
        assert A_i_stop_index - A_i_start_index > 1

        ## set column
        A.data[A_i_start_index:A_i_stop_index] *= factor

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
    finished = False

    if reduction_factors_file is not None and os.path.exists(reduction_factors_file):
        reduction_factors = np.load(reduction_factors_file)
        for i in np.where((reduction_factors != 1))[0]:
            multiply_off_diagonal_entries_with_factor(i, reduction_factors[i])
        # for i in range(len(reduction_factors)):
        #     factor = reduction_factors[i]
        #     if factor != 1:
        #         multiply_off_diagonal_entries_with_factor(i, factor)
    else:
        reduction_factors = np.ones(n, dtype=A.dtype)

    ## remove values below min abs
    mask = np.abs(A.data) < min_abs_value
    A.data[mask] = 0
    del mask
    A.eliminate_zeros()

    ## calculate positive definite approximation of A
    while not finished:

        ## try cholesky decomposition
        try:
            try:
                f = scikits.sparse.cholmod.cholesky(A, ordering_method=ordering_method, use_long=use_long)
            except scikits.sparse.cholmod.CholmodTooLargeError as e:
                if not use_long:
                    warnings.warn('Problem to large for int, switching to long.')
                    return approximate_positive_definite(A, min_diag_entry=min_diag_entry, min_abs_value=min_abs_value, use_long=True)
                else:
                    raise
            finished = True
        except scikits.sparse.cholmod.CholmodNotPositiveDefiniteError as e:
            i = e.column
            f = e.factor

        ## if not positive definite change row and column
        if not finished:
            if ordering_method == 'natural':
                p_i = i
            else:
                p_i = f.P()[i]

            ## check diagonal entry
            A_ii = A[p_i,p_i]
            if A_ii <= 0:
                raise util.math.matrix.NoPositiveDefiniteMatrixError(A, 'Diagonal entries of matrix must be positiv but {}th entry is {}.'.format(p_i, A_ii))

            ## calculate reduction factor
            # (L, D) = f.L_D()
            # assert scipy.sparse.isspmatrix_csr(L)
            # D_diag = D.diagonal()
            # L_i_start_index = L.indptr[i]
            # L_i_stop_index = L.indptr[i+1]
            # L_i_columns = L.indices[L_i_start_index:L_i_stop_index]
            # L_i_data = L.data[L_i_start_index:L_i_stop_index]
            LD = f.LD()     # Do not use f.L_D() -> higher memory consumption
            del f
            assert scipy.sparse.isspmatrix_csc(LD)
            D_diag = LD.diagonal()
            L_i = LD[i].tocsr()
            # assert scipy.sparse.isspmatrix_csr(L_i)
            L_i_columns = L_i.indices
            L_i_data = L_i.data

            s = 0
            for j, L_ij in zip(L_i_columns, L_i_data):
                s += L_ij**2 * D_diag[j]
            del LD, D_diag, L_i, L_i_columns, L_i_data

            reduction_factor_i = ((A_ii - min_diag_entry) / s)**(1/2)
            reduction_factors[p_i] *= reduction_factor_i
            if reduction_factors_file is not None:
                np.save(reduction_factors_file, reduction_factors)
            logger.debug('Row {} of cholesky decomposition not constructable. Row/column {} makes matrix not positive definite. Multiplying off diagonal entries with {}.'.format(i, p_i, reduction_factor_i))

            ## multiply off diagonal entries with reduction factor
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
        P = scipy.sparse.dok_matrix((n,n))
        for i in range(n):
            P[i,p[i]] = 1
        P = P.tocsr()

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



# def cholesky(A, approximate_if_not_positive_semidefinite=False, reduction_factor=0.9, min_abs_value=0, ordering_method='default', return_type=RETURN_P_L_D):
#     logger.debug('Calculating cholesky decomposition for matrix {!r}.'.format(A))
#
#     ## check input
#     return_types = (RETURN_A, RETURN_P_L, RETURN_P_L_D)
#     if return_type not in return_types:
#         raise ValueError('Unknown return type {}. Only values in {} are supported.'.format(return_type, return_types))
#     if ordering_method not in CHOLMOD_ORDERING_METHODS:
#         raise ValueError('Unknown ordering method {}. Only values in {} are supported.'.format(ordering_method, CHOLMOD_ORDERING_METHODS))
#
#     #TODO symmetry check
#     A = util.math.sparse.check.sorted_squared_csc(A)
#
#     ## calculate cholesky decomposition
#     if not approximate_if_not_positive_semidefinite:
#         try:
#             f = scikits.sparse.cholmod.cholesky(A, ordering_method=ordering_method)
#         except scikits.sparse.cholmod.CholmodNotPositiveDefiniteError as e:
#             raise util.math.matrix.NoPositiveDefiniteMatrixError(A, 'Row/column {} makes matrix not positive definite.'.format(e.column))
#
#     ## calculate positive definite approximation of A
#     else:
#         old_column = -1
#         finished = False
#         while not finished:
#             ## try cholesky decomposition
#             try:
#                 f = scikits.sparse.cholmod.cholesky(A, ordering_method='natural')
#                 finished = True
#             except scikits.sparse.cholmod.CholmodNotPositiveDefiniteError as e:
#                 column = e.column
#
#             ## if not positive definite change row and column
#             if not finished:
#                 ## warn
#                 if old_column == column:
#                     total_reduction_factor *= reduction_factor
#                 else:
#                     if old_column != -1:
#                         logger.warning('Multiplying column/row {} off diagonal entries with {}.'.format(old_column, total_reduction_factor))
#                     logger.warning('Column/row {} makes matrix not positive definite.'.format(column))
#                     old_column = column
#                     total_reduction_factor = reduction_factor
#
#                 ## save diagonal entry
#                 A_cc = A[column,column]
#                 if A_cc <= 0:
#                     raise util.math.matrix.NoPositiveDefiniteMatrixError(A, 'Diagonal entries of matrix must be positiv but {}th entry is {}.'.format(column, A_cc))
#
#                 ## check
#                 start_index = A.indptr[column]
#                 stop_index = A.indptr[column+1]
#                 assert stop_index - start_index > 1
#
#                 ## set column
#                 A.data[start_index:stop_index] *= reduction_factor
#
#                 ## set 0 for low values
#                 row_indices = A.indices[start_index:stop_index]
#                 zero_row_indices = row_indices[np.abs(A.data[start_index:stop_index]) < min_abs_value]
#                 for row in zero_row_indices:
#                     A[column, row] = 0
#
#                 ## set row
#                 data = A.data[start_index:stop_index]
#                 for row, data_ij in zip(row_indices, data):
#                     if row != column:
#                         A[column, row] = data_ij
#
#                 ## set diagonal entry
#                 A[column,column] = A_cc
#
#                 ## eliminate zeros
#                 if len(zero_row_indices) > 0:
#                     A.eliminate_zeros()
#
#         ## calculate cholesky decomposition
#         if return_types != RETURN_A:
#             f = scikits.sparse.cholmod.cholesky(A, ordering_method=ordering_method)
#
#
#     ## calculate permuation matrix
#     if return_types in (RETURN_P_L, RETURN_P_L_D):
#         p = f.P()
#         n = len(p)
#         P = scipy.sparse.dok_matrix((n,n))
#         for i in range(n):
#             P[i,p[i]] = 1
#         P = P.tocsr()
#
#     ## return A
#     if return_types == RETURN_A:
#         logger.debug('Returning positive definite matrix {!r}.'.format(A))
#         return A
#
#     ## return P, L
#     if return_types == RETURN_P_L:
#         L = f.L().tocsr()
#         logger.debug('Returning permutation matrix {!r} and lower triangular matrix {!r}.'.format(P, L))
#         return (P, L)
#
#     ## return P, L, D
#     if return_types == RETURN_P_L_D:
#         LD = f.LD()
#         d = LD.diagonal()
#         D = scipy.sparse.dia_matrix((d[np.newaxis,:], [0]), shape=(n,n))
#         L = LD.tocsr()
#         L.setdiag(1)
#         logger.debug('Returning permutation matrix {!r}, lower triangular matrix {!r} and diagonal matrix {!r}.'.format(P, L, D))
#         return (P, L, D)


