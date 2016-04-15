import numpy as np

import logging
logger = logging.getLogger(__name__)


PETSC_VEC_HEADER = 1211214

def load_petsc_vec_to_numpy_array(file):
    logger.debug('Loading petsc vector from {} to numpy vector.'.format(file))

    with open(file, "rb") as file_object:
        ## omit header
        np.fromfile(file_object, dtype='>i4', count=1)
        ## read length
        nvec = np.fromfile(file_object, dtype='>i4', count=1)
        assert nvec.ndim == 1 and len(nvec) == 1
        nvec = nvec[0]
        ## read values
        v = np.fromfile(file_object, dtype='>f8', count=nvec)
        assert v.ndim == 1 and len(v) == nvec

    return v


def save_numpy_array_to_petsc_vec(file, vec):
    logger.debug('Saving numpy vector to petsc vector in {}.'.format(file))

    with open(file, mode='xb') as file_object:
        ## write header
        header = np.array(PETSC_VEC_HEADER, dtype='>i4')
        header.tofile(file_object)

        ## write length (32 bit int)
        length = np.array(len(vec), dtype='>i4')
        length.tofile(file_object)

        ## write values
        vec = vec.astype('>f8')
        vec.tofile(file_object)




def load_petsc_mat_to_array(file, dtype=float):
    logger.debug('Loading petsc matrix from %s.', file)

    ## open file
    f = open(file, "rb")

    ## omit header
    np.fromfile(f, dtype=">i4", count=1)
    ## read dims
    nx     = np.fromfile(f, dtype=">i4", count=1)
    ny     = np.fromfile(f, dtype=">i4", count=1)
    nnz    = np.fromfile(f, dtype=">i4", count=1)
    nrow   = np.fromfile(f, dtype=">i4", count=nx)
    colidx = np.fromfile(f, dtype=">i4", count=nnz)
    val    = np.fromfile(f, dtype=">f8", count=nnz)

    ## close file
    f.close()

    ## create full matrix
    array = np.zeros(shape=(nx, ny), dtype=dtype)
    offset = 0
    for i in range(nx):
        if not nrow[i] == 0.0:
            for j in range(nrow[i]):
                array[i, colidx[offset]] = dtype(val[offset])
                offset = offset + 1

    ## return array
    return array