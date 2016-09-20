import numpy as np

import util.io.fs


FILE_EXT = '.{matrix_format}.npz'
SPARSE_FORMATS= ('csc', 'csr', 'bsr')


def is_spmatrix(obj):
    try:
        import scipy.sparse
    except ImportError:
        return False
    else:
        return scipy.sparse.isspmatrix(obj)

def is_file(file):
    return np.any([util.io.fs.has_file_ext(file, FILE_EXT.format(matrix_format=matrix_format)) for matrix_format in SPARSE_FORMATS])

def add_file_ext(file, sparse_format):
    file_ext = FILE_EXT.format(matrix_format=sparse_format)
    return util.io.fs.add_file_ext_if_needed(file, file_ext)


def save(file, matrix):
    import scipy.sparse
    file = add_file_ext(file, sparse_format=matrix.format)
    scipy.sparse.save_npz(file, matrix)

def load(file):
    import scipy.sparse
    return scipy.sparse.load_npz(file)

