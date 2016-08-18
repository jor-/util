import numpy as np
import scipy.sparse
import scipy.sparse.sputils

import util.math.util
import util.logging
logger = util.logging.logger



def csr_matrix(value_function, shape, value_range, dtype=None, number_of_processes=None, chunksize=None):
    import multiprocessing

    assert callable(value_function)

    ## prepare values for pool
    if number_of_processes is None:
        number_of_processes = multiprocessing.cpu_count()
    if chunksize is None:
        chunksize = max([int(len(value_range) / (number_of_processes*10**3)), 1])

    ## get data
    logger.debug('Creating crs matrix of shape {} and type {} with {} processes and chunksize {}.'.format(shape, dtype, number_of_processes, chunksize))

    ## add values to matrix
    def sum_values_to_csr_matrix(results):
        i = 0

        ## init matrix
        m = scipy.sparse.csr_matrix(shape, dtype=dtype)

        ## add results
        for m_i in results:
            logger.debug('Adding values for index {} to total matrix.'.format(i))
            m = m + m_i
            i = i+1

        return m

    ## parallel
    if number_of_processes > 1:
        with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
            results = pool.imap(value_function, value_range, chunksize=chunksize)
            m = sum_values_to_csr_matrix(results)
    ## serial
    else:
        results = map(value_function, value_range)
        m = sum_values_to_csr_matrix(results)


    return m



def diag(d):
    n = len(d)
    D = scipy.sparse.dia_matrix((d[np.newaxis,:], [0]), shape=(n,n))
    return D



def list_to_array(values, dtype=None):
    # a = np.asarray(values, dtype=dtype) #-> numpy bug
    n = len(values)
    a = np.empty(n, dtype=dtype)
    for i in range(n):
        a[i] = values[i]
    return a




class InsertableMatrix():

    def __init__(self, shape, dtype=np.float64):
        self.shape = shape
        self.data_dtype = dtype
        self.row_indices = []
        self.colum_indices = []
        self.data = []
        self.nnz = 0
        logger.debug('Initiating insertable matrix with shape {} and data dtype {}.'.format(self.shape, self.data_dtype))


    def insert(self, i, j, v):
        self.row_indices.append(i)
        self.colum_indices.append(j)
        self.data.append(v)
        assert len(self.row_indices) == len(self.colum_indices) == len(self.data)
        assert len(self.data) == self.nnz + 1
        self.nnz = self.nnz + 1
    
    
    def __setitem__(self, index, value):
        i, j = index
        self.insert(i, j, value)


    def tocoo(self):
        logger.debug('Prepare coo matrix with {} entries and data dtype {}.'.format(len(self.data), self.data_dtype))
        assert len(self.row_indices) == len(self.colum_indices) == len(self.data)

        indices_dtype = util.math.util.min_int_dtype(*self.shape)
        row = list_to_array(self.row_indices, dtype=indices_dtype)
        col = list_to_array(self.colum_indices, dtype=indices_dtype)
        data = list_to_array(self.data, dtype=self.data_dtype)

        matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=self.shape)
        if matrix.dtype != self.data_dtype:
            matrix = matrix.astype(self.data_dtype)

        logger.debug('Returning matrix {!r}.'.format(matrix))
        assert matrix.dtype == self.data_dtype
        return matrix


    def asformat(self, format='csc'):
        logger.debug('Prepare {} matrix with {} entries and dtype {}.'.format(format, len(self.data), self.data_dtype))
        matrix = self.coo_matrix().asformat(format)
        if matrix.dtype != self.data_dtype:
            matrix = matrix.astype(self.data_dtype)

        logger.debug('Returning matrix {!r}.'.format(matrix))
        assert matrix.dtype == self.data_dtype
        return matrix


