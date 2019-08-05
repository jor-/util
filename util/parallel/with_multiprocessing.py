import numpy as np
import ctypes

import multiprocessing
import multiprocessing.pool

import util.parallel.universal

import util.logging


# sharred array

def shared_array_generic(size_or_initializer, shape, dtype=np.float64):
    util.logging.debug('Creating shared array with shape {} and dtype {}.'.format(shape, dtype))

    # convert numpy type to C type
    if dtype.type is np.int64:
        ctype = ctypes.c_int64
    elif dtype.type is np.int32:
        ctype = ctypes.c_int32
    elif dtype.type is np.int16:
        ctype = ctypes.c_int16
    elif dtype.type is np.float64:
        ctype = ctypes.c_double
    elif dtype.type is np.float32:
        ctype = ctypes.c_float
    elif dtype.type is np.float128:
        ctype = ctypes.c_longdouble
    elif dtype.type is np.bool_:
        ctype = ctypes.c_bool
    else:
        raise ValueError('Data type {} of array is not supported.'.format(dtype.type))
    util.logging.debug('Using ctype {}'.format(ctype))

    # make shared array
    shared_array_base = multiprocessing.Array(ctype, size_or_initializer, lock=False)
    shared_array = np.frombuffer(shared_array_base, dtype)
    shared_array.shape = shape   # prevent copy

    # return
    util.logging.debug('Shared array created.')
    return shared_array


def shared_array(array):
    if array is not None:
        util.logging.debug('Creating shared array from array.')
    #     shared_array = shared_array_generic(array.flat, array.shape, array.dtype, lock=lock)
        shared_array = shared_array_generic(array.size, array.shape, array.dtype)  # do not use initializer ->  it needs additional memory
        shared_array[:] = array[:]
    #     np.testing.assert_array_equal(shared_array, array)
        return shared_array
    else:
        return None


def shared_zeros(shape, dtype=np.float64):
    util.logging.debug('Creating shared zeros array.')
    return shared_array_generic(np.array(shape).prod(), shape, dtype=dtype)


def share_all_arrays(args):
    # make list
    if args is not None:
        args_list = list(args)
    else:
        args_list = []

    # share arrays
    for i in range(len(args_list)):
        arg = args_list[i]

        if type(arg) in (np.ndarray, np.core.memmap):
            util.logging.debug('Sharing array at index {}.'.format(i))
            args_list[i] = shared_array(arg)

    # return args
    args = type(args)(args_list)
    return args


# map functions

def map_parallel(function, values, number_of_processes=None, chunksize=1):
    assert callable(function)

    util.logging.debug(f'Using parallel map with multiprocessing pool with {number_of_processes} processes and chunksize {chunksize}.')

    with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
        results = pool.map(function, values, chunksize=chunksize)
        results = tuple(results)

    util.logging.debug('Parallel calculation with {} results completed.'.format(len(results)))

    return results


def starmap_parallel(function, values, number_of_processes=None, chunksize=1):
    assert callable(function)

    util.logging.debug(f'Using parallel starmap with multiprocessing pool with {number_of_processes} processes and chunksize {chunksize}.')

    with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
        results = pool.starmap(function, values, chunksize=chunksize)
        results = tuple(results)

    util.logging.debug('Parallel calculation with {} results completed.'.format(len(results)))

    return results


# shared arguments

def create_array_with_shared_kargs(shape, function, number_of_processes=None, chunksize=1, **kargs):
    util.logging.debug(f'Creating array with shape {shape} with multiprocessing pool with {number_of_processes} processes and chunksize {chunksize} and {len(kargs)} shared kargs.')

    # prepare indices
    indices = np.ndindex(*shape)

    # execute in parallel
    with GlobalKargs(**kargs):
        results = map_parallel(eval_with_global_kargs, indices, number_of_processes=number_of_processes, chunksize=chunksize)
        array = np.array(tuple(results))

    # create array
    util.logging.debug('Calculation completed. Got {} results.'.format(len(array)))
    array = array.reshape(shape)
    return array


def map_parallel_with_args(function, indices, *args, number_of_processes=None, chunksize=1, share_args=True):
    util.logging.debug(f'Mapping function with {len(args)} args of types {tuple(map(type, args))} and share {share_args} to values with multiprocessing pool with {number_of_processes} processes and chunksize {chunksize}.')

    # execute in parallel
    if share_args:
        with GlobalArgs(function, *args):
            results = map_parallel(eval_with_global_args, indices, chunksize=chunksize)
    else:
        values = util.parallel.universal.args_generator_with_indices(indices, args)
        results = starmap_parallel(function, values, chunksize=chunksize)

    util.logging.debug('Parallel multiprocessing calculation with {} results completed.'.format(len(results)))

    return results


def create_array_with_args(shape, function, *args, number_of_processes=None, chunksize=1, share_args=True):
    util.logging.debug(f'Creating array with shape {shape} with multiprocessing pool with {number_of_processes} processes and chunksize {chunksize}')
    results = map_parallel_with_args(function, np.ndindex(*shape), *args, number_of_processes=number_of_processes, chunksize=chunksize, share_args=share_args)
    array = np.array(tuple(results))
    array = array.reshape(shape)
    return array


# global args

class MultipleUseError(Exception):

    def __init__(self, global_variable):
        self.global_variable = global_variable

    def __str__(self):
        return 'The global variable is already used for multiprocessing computation. Its content is {}'.format(self.global_variable)


class GlobalKargs:
    def __init__(self, f, **kargs):
        # store kargs
        self.f = f
        self.kargs = kargs

        # init global variable
        global _global_kargs
        try:
            _global_kargs
        except NameError:
            _global_kargs = None

    def __enter__(self):
        util.logging.debug('Storing {} global kargs of types {}.'.format(len(self.kargs), tuple(map(type, self.kargs))))

        # store global variable
        global _global_kargs
        global _global_kargs_f
        if _global_kargs is None:
            _global_kargs = self.kargs
            _global_kargs_f = self.f
        else:
            raise MultipleUseError(_global_kargs)

    def __exit__(self, exc_type, exc_value, traceback):
        util.logging.debug('Deleting global kargs.')

        # del global variable
        if exc_type is not MultipleUseError:
            global _global_kargs
            global _global_kargs_f
            _global_kargs = None
            _global_kargs_f = None


class GlobalArgs:
    def __init__(self, f, *args):
        # store args
        self.f = f
        self.args = args

        # init global variable
        global _global_args
        try:
            _global_args
        except NameError:
            _global_args = None

    def __enter__(self):
        util.logging.debug('Storing {} global args of types {}.'.format(len(self.args), tuple(map(type, self.args))))

        # store global vari_global_args_fable
        global _global_args
        global _global_args_f
        if _global_args is None:
            _global_args = self.args
            _global_args_f = self.f
        else:
            raise MultipleUseError(_global_args)

    def __exit__(self, exc_type, exc_value, traceback):
        util.logging.debug('Deleting global args.')

        # del global variable
        if exc_type is not MultipleUseError:
            global _global_args
            global _global_args_f
            _global_args = None
            _global_args_f = None


def eval_with_global_kargs(i, f):
    global _global_kargs
    return f(i, **_global_kargs)


def eval_with_global_args(i):
    global _global_args
    global _global_args_f
    return _global_args_f(i, *_global_args)
