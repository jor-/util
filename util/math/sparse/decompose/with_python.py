import numpy as np
import scipy.sparse

import util.math.matrix
import util.math.sparse.check

import util.logging
logger = util.logging.logger


RETURN_L = 'L'
RETURN_L_D = 'L_D'


def cholesky(A, approximate_if_not_positive_semidefinite=False, return_type=RETURN_L_D):
    logger.debug('Calculating cholesky decomposition for matrix {!r}.'.format(A))

    ## check input
    return_types = (RETURN_L, RETURN_L_D)
    if return_type not in return_types:
        raise ValueError('Unknown return type {}. Only values in {} are supported.'.format(return_type, return_types))

    #TODO check symmetry of A
    # A = util.math.sparse.check.sorted_squared_csr(A)
    A = util.math.sparse.check.squared(A)
    A = scipy.sparse.tril(A).tocsr()

    ## init
    n = A.shape[0]

    L = scipy.sparse.lil_matrix((n,n))
    d = np.zeros(n)

    ## fill matrices
    for i in range(n):
        logger.debug('Calculating row {} of decomposition.'.format(i))

        positive_semi_definite = False
        reduction_factor = None
        while not positive_semi_definite:
            ## get A_i
            A_i_column_start = A.indptr[i]
            A_i_column_stop = A.indptr[i+1]
            A_i_columns = A.indices[A_i_column_start:A_i_column_stop]
            if reduction_factor is not None:
                A.data[A_i_column_start:A_i_column_stop] *= reduction_factor
            A_i_data = A.data[A_i_column_start:A_i_column_stop]
            A_i_pos = 0

            ## get L_i
            L_i_columns = L.rows[i]
            L_i_data = L.data[i]


            ## calculate row i of L
            for j in range(i):

                d_j = d[j]
                if d_j > 0:

                    # set L_ij = A[i,j]
                    while A_i_pos < len(A_i_columns) and A_i_columns[A_i_pos] < j:
                        A_i_pos += 1

                    if A_i_pos < len(A_i_columns) and A_i_columns[A_i_pos] == j:
                        L_ij = A_i_data[A_i_pos]
                    else:
                        L_ij = 0

                    # set L_ij -= sum(L_i * L_j * d)
                    L_j_columns = L.rows[j]
                    L_j_data = L.data[j]

                    L_i_pos = 0
                    L_j_pos = 0

                    while L_i_pos < len(L_i_columns) and L_j_pos < len(L_j_columns):
                        if L_i_columns[L_i_pos] == L_j_columns[L_j_pos]:
                            L_ij -= L_i_data[L_i_pos] * L_j_data[L_j_pos] *  d[L_i_columns[L_i_pos]]
                            L_i_pos += 1
                            L_j_pos += 1
                        elif L_i_columns[L_i_pos] < L_j_columns[L_j_pos]:
                            L_i_pos += 1
                        else:
                            L_j_pos += 1

                    # set L_ij /= d[j]
                    L_ij /= d_j

                    # set L[i,j] = L_ij
                    if L_ij != 0:
                        L[i,j] = L_ij


            ## calculate diagonal entry i

            # set d_i = A[i,i]
            while A_i_pos < len(A_i_columns) and A_i_columns[A_i_pos] < i:
                A_i_pos += 1
            A_ii = A_i_data[A_i_pos]

            if A_i_pos < len(A_i_columns) and A_i_columns[A_i_pos] == i:
                d_i = A_ii
            else:
                d_i = 0

            # set d_i -= sum(L_i**2 * d)
            for k in range(len(L_i_columns)):
                column_k = L_i_columns[k]
                d_i -= L_i_data[k]**2 * d[column_k]

            # set d[i] = d_i
            if d_i > 0:
                d[i] = d_i
                positive_semi_definite = True
                reduction_factor = None
            elif d_i < 0:
                if approximate_if_not_positive_semidefinite:
                    if A_ii > 0:
                        reduction_factor = (A_ii / (A_ii - d_i))**(1/2)
                    else:
                        reduction_factor = 0
                    logger.warn('Column/row {} makes matrix not positive definite. Diagonal entry would be {}. Multiplying row/column with {}.'.format(i, d_i, reduction_factor))
                else:
                    raise util.math.matrix.NoPositiveDefiniteMatrixError(A, 'Column/row {} of matrix is not positive definite. Diagonal entry would be {}.'.format(i, d_i))


    ## set 1 diagonal of L
    L.setdiag(1)

    ## convert to suitable sparse matrix formats
    logger.debug('Converting matrices to suitable formats.')
    L = L.tocsr()
    D = scipy.sparse.dia_matrix((d[np.newaxis,:], [0]), shape=(n,n))

    # ## return A
    # if return_types == RETURN_A:
    #     A = L * D * L.transpose()
    #     logger.debug('Returning positive definite matrix {!r}.'.format(A))
    #     return A

    ## return L
    if return_type == RETURN_L:
        D.data = D.data**(1/2)
        L = L*D
        logger.debug('Returning lower triangulat matrix {!r}.'.format(L))
        return L

    ## return L, D
    if return_type == RETURN_L_D:
        logger.debug('Returning lower triangulat matrix {!r} and diagonal matrix {!r}.'.format(L, D))
        return (L, D)



