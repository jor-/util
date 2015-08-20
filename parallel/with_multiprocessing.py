import numpy as np
import ctypes

import multiprocessing
import multiprocessing.pool

import util.parallel.universal

import util.logging
logger = util.logging.logger



## sharred array

def shared_array_generic(size_or_initializer, shape, dtype=np.float64):
    logger.debug('Creating shared array with shape {} and dtype {}.'.format(shape, dtype))

    ## convert numpy type to C type
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
    logger.debug('Using ctype {}'.format(ctype))

    ## make shared array
    shared_array_base = multiprocessing.Array(ctype, size_or_initializer, lock=False)
    shared_array = np.frombuffer(shared_array_base, dtype)
    shared_array.shape = shape   # prevent copy

    ## return
    logger.debug('Shared array created.')
    return shared_array


def shared_array(array):
    if array is not None:
        logger.debug('Creating shared array from array.')
    #     shared_array = shared_array_generic(array.flat, array.shape, array.dtype, lock=lock)
        shared_array = shared_array_generic(array.size, array.shape, array.dtype) ## do not use initializer ->  it needs additional memory
        shared_array[:] = array[:]
    #     np.testing.assert_array_equal(shared_array, array)
        return shared_array
    else:
        return None


def shared_zeros(shape, dtype=np.float64):
    logger.debug('Creating shared zeros array.')
    return shared_array_generic(np.array(shape).prod(), shape, dtype=dtype)


def share_all_arrays(args):
    ## make list
    if args is not None:
        args_list = list(args)
    else:
        args_list = []

    ## share arrays
    for i in range(len(args_list)):
        arg = args_list[i]

        if type(arg) in (np.ndarray, np.core.memmap):
            logger.debug('Sharing array at index {}.'.format(i))
            args_list[i] = shared_array(arg)

    ## return args
    args = type(args)(args_list)
    return args



## map functions

def map_parallel(function, values, number_of_processes=None, chunksize=1):
    assert callable(function)

    logger.debug('Creating multiprocessing pool with {} processes and chunksize {}.'.format(number_of_processes, chunksize))

    with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
        results = pool.map(function, values, chunksize=chunksize)
        results = tuple(results)

    logger.debug('Parallel calculation with {} results completed.'.format(len(results)))

    return results


def starmap_parallel(function, values, number_of_processes=None, chunksize=1):
    assert callable(function)

    logger.debug('Creating multiprocessing pool with {} processes and chunksize {}.'.format(number_of_processes, chunksize))

    with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
        results = pool.starmap(function, values, chunksize=chunksize)
        results = tuple(results)

    logger.debug('Parallel calculation with {} results completed.'.format(len(results)))

    return results



## shared arguments

def create_array_with_shared_kargs(shape, function, **kargs):
    logger.debug('Creating array with shape {} with multiprocessing and {} shared kargs.'.format(shape, len(kargs)))

    ## prepare indices
    indices = np.ndindex(*shape)

    ## execute in parallel
    with GlobalKargs(**kargs):
        results = map_parallel(eval_with_global_kargs, indices)
        array = np.array(tuple(results))

    ## create array
    logger.debug('Calculation completed. Got {} results.'.format(len(array)))
    array = array.reshape(shape)
    return array



def map_parallel_with_args(function, indices, *args, number_of_processes=None, chunksize=1, share_args=True):
    logger.debug('Mapping function with {} args of types {} and share {} to values with multiprocessing.'.format(len(args), tuple(map(type, args)), share_args))

    ## execute in parallel
    if share_args:
#         args = share_all_arrays(args, lock=False)
#         values = util.parallel.universal.args_generator_with_indices(indices, (function,))
        with GlobalArgs(function, *args):
            results = map_parallel(eval_with_global_args, indices)
    else:
        values = util.parallel.universal.args_generator_with_indices(indices, args)
        results = starmap_parallel(function, values)

    logger.debug('Parallel multiprocessing calculation with {} results completed.'.format(len(results)))

    return results




## global args

class MultipleUseError(Exception):

    def __init__(self, global_variable):
        self.global_variable = global_variable

    def __str__(self):
        return 'The global variable is already used for multiprocessing computation. Its content is {}'.format(self.global_variable)


class GlobalKargs:
    def __init__(self, f, **kargs):
        ## store kargs
        self.f = f
        self.kargs = kargs

        ## init global variable
        global _global_kargs
        try:
            _global_kargs
        except NameError:
            _global_kargs = None

    def __enter__(self):
        logger.debug('Storing {} global kargs of types {}.'.format(len(self.kargs), tuple(map(type, self.kargs))))

        ## store global variable
        global _global_kargs
        global _global_kargs_f
        if _global_kargs is None:
            _global_kargs = self.kargs
            _global_kargs_f = self.f
        else:
            raise MultipleUseError(_global_kargs)

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug('Deleting global kargs.')

        ## del global variable
        if exc_type is not MultipleUseError:
            global _global_kargs
            global _global_kargs_f
            _global_kargs = None
            _global_kargs_f = None


class GlobalArgs:
    def __init__(self, f, *args):
        ## store args
        self.f = f
        self.args = args

        ## init global variable
        global _global_args
        try:
            _global_args
        except NameError:
            _global_args = None

    def __enter__(self):
        logger.debug('Storing {} global args of types {}.'.format(len(self.args), tuple(map(type, self.args))))

        ## store global vari_global_args_fable
        global _global_args
        global _global_args_f
        if _global_args is None:
            _global_args = self.args
            _global_args_f = self.f
        else:
            raise MultipleUseError(_global_args)

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug('Deleting global args.')

        ## del global variable
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












##############################
# import numpy as np
# import ctypes
#
# import multiprocessing
# import multiprocessing.pool
#
# import util.parallel.universal
#
# import util.logging
# logger = util.logging.logger
#
#
#
# ## sharred array
#
# def shared_array_generic(size_or_initializer, shape, dtype=np.float64, lock=True):
#     logger.debug('Creating shared array with shape {}, dtype {} and lock {}.'.format(shape, dtype, lock))
#
#     ## convert numpy type to C type
#     if dtype.type is np.int64:
#         ctype = ctypes.c_int64
#     elif dtype.type is np.int32:
#         ctype = ctypes.c_int32
#     elif dtype.type is np.int16:
#         ctype = ctypes.c_int16
#     elif dtype.type is np.float64:
#         ctype = ctypes.c_double
#     elif dtype.type is np.float32:
#         ctype = ctypes.c_float
#     elif dtype.type is np.float128:
#         ctype = ctypes.c_longdouble
#     elif dtype.type is np.bool_:
#         ctype = ctypes.c_bool
#     else:
#         raise ValueError('Data type {} of array is not supported.'.format(dtype.type))
#     logger.debug('Using ctype {}'.format(ctype))
#
#     ## make shared array
#     shared_array_base = multiprocessing.Array(ctype, size_or_initializer, lock=lock)
#     shared_array = np.frombuffer(shared_array_base, dtype)
#     shared_array.shape = shape   # prevent copy
#
#     ## return
#     logger.debug('Shared array created.')
#     return shared_array
#
#
# def shared_array(array, lock=True):
#     if array is not None:
#         logger.debug('Creating shared array from array.')
#     #     shared_array = shared_array_generic(array.flat, array.shape, array.dtype, lock=lock)
#         shared_array = shared_array_generic(array.size, array.shape, array.dtype, lock=lock) ## do not use initializer ->  it needs additional memory
#         shared_array[:] = array[:]
#     #     np.testing.assert_array_equal(shared_array, array)
#         return shared_array
#     else:
#         return None
#
#
# def shared_zeros(shape, dtype=np.float64, lock=True):
#     logger.debug('Creating shared zeros array.')
#     return shared_array_generic(np.array(shape).prod(), shape, dtype=dtype, lock=lock)
#
#
# def share_all_arrays(args, lock=True):
#     ## make list
#     if args is not None:
#         args_list = list(args)
#     else:
#         args_list = []
#
#     ## share arrays
#     for i in range(len(args_list)):
#         arg = args_list[i]
#
#         if type(arg) in (np.ndarray, np.core.memmap):
#             logger.debug('Sharing array at index {}.'.format(i))
#             args_list[i] = shared_array(arg, lock=lock)
#
#     ## return args
#     args = type(args)(args_list)
#     return args
#
#
#
# ## map functions
#
# def map_parallel(function, values, number_of_processes=None, chunksize=1):
#     assert callable(function)
#
#     logger.debug('Creating multiprocessing pool with {} processes and chunksize {}.'.format(number_of_processes, chunksize))
#
#     with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
#         results = pool.map(function, values, chunksize=chunksize)
#         results = tuple(results)
#
#     logger.debug('Parallel calculation with {} results completed.'.format(len(results)))
#
#     return results
#
#
# def starmap_parallel(function, values, number_of_processes=None, chunksize=1):
#     assert callable(function)
#
#     logger.debug('Creating multiprocessing pool with {} processes and chunksize {}.'.format(number_of_processes, chunksize))
#
#     with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
#         results = pool.starmap(function, values, chunksize=chunksize)
#         results = tuple(results)
#
#     logger.debug('Parallel calculation with {} results completed.'.format(len(results)))
#
#     return results
#
#
# ## shared arguments
#
# class MultipleUseError(Exception):
#
#     def __init__(self, global_variable):
#         self.global_variable = global_variable
#
#     def __str__(self):
#         return 'The global variable is already used for multiprocessing computation. Its content is {}'.format(self.global_variable)
#
#
# class GlobalKargs:
#     def __init__(self, **kargs):
#         ## store kargs
#         self.kargs = kargs
#
#         ## init global variable
#         global _global_kargs
#         try:
#             _global_kargs
#         except NameError:
#             _global_kargs = None
#
#     def __enter__(self):
#         logger.debug('Storing {} global kargs of types {}.'.format(len(self.kargs), tuple(map(type, self.kargs))))
#
#         ## store global variable
#         global _global_kargs
#         if _global_kargs is None:
#             _global_kargs = self.kargs
#         else:
#             raise MultipleUseError(_global_kargs)
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         logger.debug('Deleting global kargs.')
#
#         ## del global variable
#         if exc_type is not MultipleUseError:
#             global _global_kargs
#             _global_kargs = None
#
#
# class GlobalArgs:
#     def __init__(self, *args):
#         ## store args
#         self.args = args
#
#         ## init global variable
#         global _global_args
#         try:
#             _global_args
#         except NameError:
#             _global_args = None
#
#     def __enter__(self):
#         logger.debug('Storing {} global args of types {}.'.format(len(self.args), tuple(map(type, self.args))))
#
#         ## store global variable
#         global _global_args
#         if _global_args is None:
#             _global_args = self.args
#         else:
#             raise MultipleUseError(_global_args)
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         logger.debug('Deleting global args.')
#
#         ## del global variable
#         if exc_type is not MultipleUseError:
#             global _global_args
#             _global_args = None
#
#
# def eval_with_global_kargs(i, f):
#     global _global_kargs
#     return f(i, **_global_kargs)
#
# # def eval_with_global_kargs_zipped(args):
# #     return eval_with_global_kargs(*args)
#
# def eval_with_global_args(i, f):
#     global _global_args
#     return f(i, *_global_args)
#
# # def eval_with_global_args_zipped(args):
# #     return eval_with_global_args(*args)
#
#
#
#
#
# def create_array_with_shared_kargs(shape, function, **kargs):
#     logger.debug('Creating array with shape {} with multiprocessing and {} shared kargs.'.format(shape, len(kargs)))
#
# #     ## prepare input args
# #     values = util.parallel.universal.args_generator_with_shape(shape, (eval_with_global_kargs, function))
# #
# #     ## execute in parallel
# #     with GlobalKargs(**kargs):
# #         results = map_parallel(util.parallel.universal.eval_function_with_index_and_args, values)
#
#     ## prepare input args
#     values = util.parallel.universal.args_generator_with_shape(shape, (function,))
#
#     ## execute in parallel
#     with GlobalKargs(**kargs):
#         results = starmap_parallel(eval_with_global_args, values)
#
#     ## create array
#     array = np.array(list(results))
#     logger.debug('Calculation completed. Got {} results.'.format(len(array)))
#     array = array.reshape(shape)
#     return array
#
#
# # def create_array_with_shared_args(shape, function, *args):
# #     logger.debug('Creating array with shape {} with multiprocessing and shared args.'.format(shape))
# #
# #     ## prepare input args
# #     values = args_generator(shape, (function,))
# #
# #     ## execute in parallel
# #     with GlobalArgs(*args):
# #         results = map_parallel(eval_with_global_args_zipped, values)
# #
# #     ## create array
# #     array = np.array(list(results))
# #     logger.debug('Calculation completed. Got {} results.'.format(len(array)))
# #     array = array.reshape(shape)
# #     return array
#
#
#
# # def map_parallel_with_args(function, values, *args, number_of_processes=None, chunksize=1, share_args=True):
# #     logger.debug('Mapping function with {} args and share {} to values.'.format(len(args), share_args))
# #
# #     ## execute in parallel
# #     if share_args:
# #         values = util.parallel.universal.args_generator_with_indices(values, (function))
# #         with GlobalArgs(*args):
# #             results = map_parallel(eval_with_global_args_zipped, values)
# #     else:
# #         values = util.parallel.universal.args_generator_with_indices(values, args)
# #         results = map_parallel(function, values)
# #
# #     return results
#
# # def map_parallel_with_args(function, values, *args, number_of_processes=None, chunksize=1, share_args=True):
# #     logger.debug('Mapping function with {} args of types {} and share {} to values.'.format(len(args), map(type, args), share_args))
# #
# #     ## execute in parallel
# #     if share_args:
# #         values = util.parallel.universal.args_generator_with_indices(values, (eval_with_global_args, function))
# #         with GlobalArgs(*args):
# #             results = map_parallel(util.parallel.universal.eval_function_with_index_and_args, values)
# #     else:
# #         values = util.parallel.universal.args_generator_with_indices(values, (function,) + args)
# #         results = map_parallel(util.parallel.universal.eval_function_with_index_and_args, values)
# #
# #     return results
#
# def map_parallel_with_args(function, indices, *args, number_of_processes=None, chunksize=1, share_args=True):
#     logger.debug('Mapping function with {} args of types {} and share {} to values with multiprocessing.'.format(len(args), tuple(map(type, args)), share_args))
#
#     ## execute in parallel
#     if share_args:
# #         args = share_all_arrays(args, lock=False)
#         values = util.parallel.universal.args_generator_with_indices(indices, (function,))
#         with GlobalArgs(*args):
#             results = starmap_parallel(eval_with_global_args, values)
#     else:
#         values = util.parallel.universal.args_generator_with_indices(indices, args)
#         results = starmap_parallel(function, values)
#
#     return results
#
#
#
# # def create_array_with_args(shape, function, *args, share_args=True):
# #     logger.debug('Creating array with shape {} with multiprocessing, {} args and share {}.'.format(shape, len(args), share_args))
# #
# #
# #     ## create array
# #     array = np.array(list(results))
# #     logger.debug('Calculation completed. Got {} results.'.format(len(array)))
# #     array = array.reshape(shape)
# #     return array
#
#
#
#
#
#
#
#
# # def create_array(shape, function, function_args=None, function_args_first=True, number_of_processes=None, chunksize=1):
# #     logger.debug('Creating array with shape {} in parallel with multiprocessing.'.format(shape))
# #
# #     ## create indices
# #     indices = np.ndindex(*shape)
# #     if function_args is None:
# #         function_args = ()
# #     if function_args_first:
# #         indices = [function_args + (i,) for i in indices]
# #     else:
# #         indices = [(i,) + function_args for i in indices]
# #
# #     ## execute in parallel
# #     results = starmap(function, indices, number_of_processes=number_of_processes, chunksize=chunksize)
# #
# #     ## create array
# #     array = np.array(results).reshape(shape)
# #     return array
#
#
#
# #
# #
# #
# # #import multiprocessing.pool
# # #import logging
# # #
# # # # def map(function, values, number_of_processes=None, chunksize=1):
# # # #     process_pool = multiprocessing.pool.Pool(processes=number_of_processes)
# # # #     result = process_pool.map(function, values, chunksize=chunksize)
# # # #     process_pool.close()
# # # #     process_pool.join()
# # # #     return result
# # #
# # def map(function, values, number_of_processes=None, chunksize=1):
# #     with multiprocessing.pool.Pool(processes=number_of_processes) as pool:
# #         result = pool.map(function, values, chunksize=chunksize)
# #     pool.join()
# #     return result
# #
# #
# # import numpy as np
# # import multiprocessing
# # import multiprocessing.pool
# # import ctypes
# #
# #
# # logger = multiprocessing.get_logger()
# #
# # def shared_zeros(shape, lock=True):
# #     ## create a shared numpy array
# #     shape = np.asarray(shape)
# #     size = int(shape.prod())
# #
# #     shared_array_base = multiprocessing.Array(ctypes.c_double, size, lock=lock)
# #     if lock:
# #         shared_array_base_obj = shared_array_base.get_obj()
# #     else:
# #         shared_array_base_obj = shared_array_base
# #     shared_array = np.frombuffer(shared_array_base_obj)
# #     shared_array = shared_array.reshape(shape)
# #
# #     logger.debug('Shared array with shape {} and lock {} created.'.format(shared_array, lock))
# #
# #     return shared_array
# #
# #
# #
# #
# # def create_array_in_parallel(shape, function, iterable, number_of_processes=None, chunksize=1):
# #     assert callable(function)
# #
# #     global shared_array
# #     shared_array = shared_zeros(shape, lock=False)
# #
# #     logger.debug('Creating pool with {} processes and chunksize.'.format(number_of_processes, chunksize))
# #
# #     pool = multiprocessing.pool.Pool(processes=number_of_processes)
# #     pool.map_async(function, iterable, chunksize=chunksize)
# #     pool.close()
# #     pool.join()
# #
# #     logger.debug('Parallel calculation completed.')
# #
# #     array = np.array(shared_array, copy=True)
# #     return array
# #
# #
# #
# # def compute(index):
# #     logger.debug('Calculating for index {}.'.format(index))
# #     shared_array[:,:, index] = index
# #
# #
# # def new_main():
# #     shape = (2,3,4)
# #     iterable = range(4)
# #     number_of_processes = 2
# #     chunksize = 2
# #     array = create_array_in_parallel(shape, compute, iterable, number_of_processes=number_of_processes, chunksize=chunksize)
# #     return array
# #
# #
# # if __name__ == '__main__':
# #     print(new_main())
# #
# # # def compute_2(array, index):
# # #     print('Setting index {} at array with shape {}'.format(index, array.shape))
# # #     array[:,:, index] = index
# # #
# # # def new_main_2():
# # #     shape = (2,3,4)
# # #     iterable = range(4)
# # #     number_of_processes = 2
# # #     chunksize = 2
# # #     function = compute_2
# # #
# # #     shared_array = shared_zeros(shape, lock=False)
# # #
# # #     pool = multiprocessing.pool.Pool(processes=number_of_processes)
# # #     for i in iterable:
# # #         pool.apply_async(function, args=(shared_array, i))
# # #     pool.close()
# # #     pool.join()
# # #
# # #     array = np.array(shared_array, copy=True)
# # #     shared_array = None
# # #     return array
# #
# #
# #
# #
# #
# #
# # if __name__=='__main__':
# #     arr = new_main()
# #     print(arr)
# # #     arr = new_main_2()
# # #     print(arr)