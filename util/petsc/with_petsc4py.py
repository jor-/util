import numpy as np
from petsc4py import PETSc as petsc

import logging
logger = logging.getLogger(__name__)


## save petsc

def save_petsc(file, petsc_object):
    logger.debug('Saving petsc object to %s.', file)

    viewer = petsc.Viewer().createBinary(file, petsc.Viewer.Mode.WRITE)
    petsc_object.view(viewer)
    viewer.destroy()



## load petsc

def load_petsc_vec(file):
    logger.debug('Loading petsc vector from %s.', file)

    viewer = petsc.Viewer().createBinary(file, petsc.Viewer.Mode.READ)
    vec = petsc.Vec().load(viewer)
    viewer.destroy()

    return vec


def load_petsc_mat(file):
    logger.debug('Loading petsc matrix from %s.', file)

    viewer = petsc.Viewer().createBinary(file, petsc.Viewer.Mode.READ)
    mat = petsc.Mat().load(viewer)
    viewer.destroy()

    return mat



def petsc_vec_to_array(vec):
    logger.debug('Converting petsc vector to array.')
    array = np.array(vec, copy=True)

#     ownership_range = vec.getOwnershipRange()
#     size_local = ownership_range[1] - ownership_range[0]
#     size_global = vec.getSize()
#
#     array = np.empty(size_global, dtype=dtype)
#
#     for i_local in range(size_local):
#         i_global = y_ownership_range[0] + i_local


    return array


def petsc_mat_to_array(mat, dtype=float):
    logger.debug('Converting petsc matrix to array.')
    shape = mat.getSize()
    array = np.zeros(shape, dtype=dtype)

    #get values row by row since petsc matrix are sparce
    for i in range(shape[0]):
        row = mat.getRow(i)
        indices = row[0]
        values = row[1]
        array[i, indices] = values;

    return array



def load_petsc_vec_to_numpy_array(file):
    vec = load_petsc_vec(file)
    array = petsc_vec_to_array(vec, dtype=dtype)
    vec.destroy()

    return array


def load_petsc_mat_to_array(file, dtype=float):
    mat = load_petsc_mat(file)
    array = petsc_mat_to_array(mat, dtype=dtype)
    mat.destroy()

    return array



## create functions

def create_petsc_vec_by_function(function, n, finish_assembly=True):
    ## create vec
    logger.debug('Creating petsc vec of length %d from function.', n)

    vec = petsc.Vec()
    vec.createMPI(n, comm=petsc.COMM_WORLD)

    for i in range(n):
        vec.setValue(i, function(i))

    ## assembly
    logger.debug('Beginning assembly.')
    vec.assemblyBegin()

    if finish_assembly:
        logger.debug('Ending assembly.')
        vec.assemblyEnd()

    return vec


def create_petsc_mat_by_function(function, size, finish_assembly=True):
    ## convert size
    try:
        n = m = int(size)
    except TypeError:
        if len(size) == 1:
            n = m = size[0]
        elif len(size) == 2:
            n = size[0]
            m = size[1]
        else:
            raise TypeError('Size has to be an int or a tuple of length one or two, but its zype is %s.' % type(size))

    ## create mat
    logger.debug('Converting array to petsc mat.')

    mat = petsc.Mat()
    mat.createDense([n,m], comm=petsc.COMM_WORLD)

    for i, j in np.ndindex(n, m):
        mat.setValue(i, j, function(i,j))

    ## assembly
    logger.debug('Beginning assembly.')
    mat.assemblyBegin(petsc.Mat.AssemblyType.FINAL)

    if finish_assembly:
        logger.debug('Ending assembly.')
        mat.assemblyEnd(petsc.Mat.AssemblyType.FINAL)

    return mat



## array to petsc

def array_to_petsc_vec(array, finish_assembly=True):
    ## check array
    if len(array.shape) == 1:
        n = array.shape[0]
    else:
        raise ValueError('array has to be a vector, but it´s shape is %s.' % array.shape)

    ## create vec
    logger.debug('Converting array to petsc vec.')

    function = lambda i: array[i]
    vec = create_petsc_vec_by_function(function, n, finish_assembly=finish_assembly)

    return vec


def array_to_petsc_mat(array, finish_assembly=True):
    ## check array
    if len(array.shape) == 2:
        (n, m) = array.shape
    else:
        raise ValueError('array has to be a matrix, but it´s shape is %s.' % array.shape)

    ## create mat
    logger.debug('Converting array to petsc mat.')

    function = lambda i,j: array[i,j]
    mat = create_petsc_mat_by_function(function, (n, m), finish_assembly=finish_assembly)

    return mat



## other

def print_petsc(petsc_object):
    petsc_object.view(petsc.Viewer.STDOUT())




def solve_linear_equations(A, b, solver_type=petsc.KSP.Type.CG, monitor=True):
    n = b.getSize()

    logger.debug('Prepare for solving linear equation system of size %d with solver %s.' % (n, solver_type))

    if monitor:
        petsc_opts = petsc.Options()
        petsc_opts.setValue('ksp_monitor', True)

    x = petsc.Vec()
    x.createMPI(n, comm=petsc.COMM_WORLD)

    ksp = petsc.KSP()
    ksp.create(petsc.COMM_WORLD)
    ksp.setFromOptions()
    ksp.setOperators(A)
    ksp.setType(solver_type)

    pc = ksp.getPC()
    pc.setType(petsc.PC.Type.NONE)

    logger.debug('Solving linear equation system.')

    if monitor:
        ksp.view(petsc.Viewer.STDOUT())

    ksp.solve(b, x)

    x.assemblyBegin()
    x.assemblyEnd()

    pc.destroy()
    ksp.destroy()

    logger.debug('Linear equation system solved.')

    return x



class Matrix_Shell_Petsc:

    def __init__(self, entry_function):
        self.entry_function = entry_function

    @property
    def n(self):
        return self.covariance_model.n

    def mult(self, context, x, y):
        logger.debug('Multiplying petsc matrix with vector without explicit matrix.')

        ## copy x to local vec
        scatter, x_local = petsc.Scatter.toAll(x)
        scatter.scatterBegin(x, x_local)
        scatter.scatterEnd(x, x_local)
        scatter.destroy()


        ## set y values
        y_ownership_range = y.getOwnershipRange()
        y_size_local = y_ownership_range[1] - y_ownership_range[0]
        y_size_global = y.getSize()

        for i_local in range(y_size_local):
            i_global = y_ownership_range[0] + i_local

            ## compute value
            value = 0
            for j_global in range(y_size_global):
                value += self.entry_function(i_global, j_global) * x_local.getValue(j_global)

            y.setValue(i_global, value)
        y.assemblyBegin()
        y.assemblyEnd()

        ## destroy local copy
        x_local.destroy()