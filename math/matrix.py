import numpy as np


class MatrixError(Exception):

    def __init__(self, matrix, message=None, additional_message=None):
        self.matrix = matrix
        if message is None:
            message = 'Error with matrix {!r}!'.format(self.matrix)
        if additional_message is not None:
            message = message + ' ' + additional_message
        self.message = message

    def __str__(self):
        return self.message


class SingularMatrixError(MatrixError):

    def __init__(self, matrix, additional_message=None):
        message = 'Matrix {!r} is singular.'.format(matrix)
        super().__init__(matrix, message, additional_message)


class NoTriangularMatrixError(MatrixError):

    def __init__(self, matrix, additional_message=None):
        message = 'Matrix {!r} is no triangular matrix.'.format(matrix)
        super().__init__(matrix, message, additional_message)


class NoLeftTriangularMatrixError(MatrixError):

    def __init__(self, matrix, additional_message=None):
        message = 'Matrix {!r} is no left triangular matrix.'.format(matrix)
        super().__init__(matrix, message, additional_message)


class NoRightTriangularMatrixError(MatrixError):

    def __init__(self, matrix, additional_message=None):
        message = 'Matrix {!r} is no right triangular matrix.'.format(matrix)
        super().__init__(matrix, message, additional_message)


class NoPositiveDefiniteMatrixError(MatrixError):

    def __init__(self, matrix, additional_message=None):
        message = 'Matrix {!r} is not positive definite.'.format(matrix)
        super().__init__(matrix, message, additional_message)





def convert_to_matrix(array, dtype=None):
    array = np.asanyarray(array)
    matrix = np.asmatrix(array, dtype=dtype)
    if array.ndim == 1:
        matrix = matrix.T
    return matrix


def convert_matrix_to_array(matrix, dtype=None):
    array = np.asarray(matrix, dtype=dtype)
    if array.size == 1:
        array = array[0, 0]
    elif array.shape[1] == 1:
        array = array[:, 0]
    return array



