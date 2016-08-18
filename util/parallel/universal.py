import warnings
import numpy as np
# import itertools

import util.logging
logger = util.logging.logger

from .constants import MODES

CURRENT_MODE = MODES['scoop']


def max_parallel_mode():
    if util.parallel.is_running.scoop_module():
        max_parallel_mode = MODES['scoop']
    else:
        max_parallel_mode = MODES['multiprocessing']
    logger.debug('Maximal parallel mode is {}.'.format(max_parallel_mode))
    return max_parallel_mode




## map functions

# def map_serial_with_args(function, values, *args):
#     logger.debug('Mapping function with {} args of types {} to values in serial.'.format(len(args), tuple(map(type, args))))
#
#     values = args_generator_with_indices(values, args)
#     results = itertools.starmap(function, values)
#     results = tuple(results)
#
#     logger.debug('Serial calculation with {} results completed.'.format(len(results)))
#
#     return results


# def map_parallel(function, values, parallel_mode=MODES['scoop']):
#     assert callable(function)
#
#     if parallel_mode == MODES['scoop']:
#         CURRENT_MODE = MODES['multiprocessing']
#         import util.parallel.with_scoop
#         results = util.parallel.with_scoop.map_parallel(function, values)
#     elif parallel_mode == MODES['multiprocessing']:
#         CURRENT_MODE = MODES['serial']
#         import util.parallel.with_multiprocessing
#         results = util.parallel.with_multiprocessing.map_parallel(function, values)
#     else:
#         logger.debug('Calculating in serial.')
#         results = map(function, values)
#
#     return results


def map_parallel_with_args(function, values, *args, parallel_mode=MODES['scoop'], number_of_processes=None, chunksize=1, share_args=True):
    assert callable(function)

    if parallel_mode == MODES['scoop']:
        CURRENT_MODE = MODES['multiprocessing']
        import util.parallel.with_scoop
        results = util.parallel.with_scoop.map_parallel_with_args(function, values, *args)
    elif parallel_mode == MODES['multiprocessing']:
        CURRENT_MODE = MODES['serial']
        import util.parallel.with_multiprocessing
        results = util.parallel.with_multiprocessing.map_parallel_with_args(function, values, *args, number_of_processes=number_of_processes, chunksize=chunksize, share_args=share_args)
    else:
        import util.parallel.with_serial
        results = util.parallel.with_serial.map_serial_with_args(function, values, *args)

    return results




# def create_array(shape, function, args=None, parallel_mode=MODES['scoop']):
#     logger.debug('Creating array with shape {} with parallel mode {}.'.format(shape, parallel_mode))
#
#     ## check that mode is not in use
#     if parallel_mode > CURRENT_MODE:
#         logger.debug('Parallel mode {} used before. Switching to parallel mode {}.'.format(parallel_mode, CURRENT_MODE))
#         parallel_mode = CURRENT_MODE
#
#     ## check if scoop running otherwise fallback
#     if parallel_mode == MODES['scoop'] and not util.parallel.is_running.scoop_module():
#         parallel_mode = MODES['multiprocessing']
#         logger.debug('Scoop is not running falling back to parallel mode {}.'.format(parallel_mode))
#
# #     ## if multiprocessing share arrays
# #     if parallel_mode == MODES['multiprocessing']:
# #         args = util.parallel.with_multiprocessing.share_all_arrays(args, lock=False)
#
#     ## create args generator
#     values = args_generator(shape, args)
#
#     ## execute in parallel
#     results = map_parallel(function, values, parallel_mode=parallel_mode)
#
#     ## create array
#     array = np.array(list(results))
#     logger.debug('Calculation completed. Got {} results.'.format(len(array)))
#     array = array.reshape(shape)
#     return array


## create array

def create_array(shape, function, *args, parallel_mode=MODES['scoop'], number_of_processes=None, chunksize=1, share_args=True):
    logger.debug('Creating array with shape {} with parallel mode {}.'.format(shape, parallel_mode))

    ## check that mode is not in use
    if parallel_mode > CURRENT_MODE:
        logger.debug('Parallel mode {} used before. Switching to parallel mode {}.'.format(parallel_mode, CURRENT_MODE))
        parallel_mode = CURRENT_MODE

    ## check if scoop running otherwise fallback
    if parallel_mode == MODES['scoop'] and not util.parallel.is_running.scoop_module():
        parallel_mode = MODES['multiprocessing']
        logger.debug('Scoop is not running falling back to parallel mode {}.'.format(parallel_mode))

    ## execute in parallel
    results = map_parallel_with_args(function, np.ndindex(*shape), *args, parallel_mode=parallel_mode, number_of_processes=number_of_processes, chunksize=chunksize, share_args=share_args)

    ## create array
    array = np.array(tuple(results))
#     logger.debug('Calculation completed. Got {} results.'.format(len(array)))
    array = array.reshape(shape)
    return array



## args generators

def args_generator_with_shape(shape, args, index_position=0):
    logger.debug('Creating arg generator for shape {} and {} args with index position {}.'.format(shape, len(args), index_position))
    indices = np.ndindex(*shape)
    args_generator_with_indices(indices, args, index_position=index_position)


def args_generator_with_indices(indices, args, index_position=0):
    logger.debug('Creating arg generator for indices and {} args of types {} with index position {}.'.format(len(args), tuple(map(type, args)), index_position))

    ## make list of function args
    if args is not None:
        args_list = list(args)
    else:
        args_list = []

    ## insert index
    args_list.insert(index_position, None)

    ## return function to compute function args with current index
    for index in indices:
        args_list[index_position] = index
#         logger.debug('Returning index {} together with {} args.'.format(index, len(args)))
        yield tuple(args_list)




## zipped function
def eval_function_with_index_and_args(args):
    assert len(args) >= 2
    index = args[0]
    function = args[1]
    args = args[2:]
    return function(index, *args)


#
#
# def map_parallel(function, values, parallel=True):
#     assert callable(function)
#
# #     if parallel:
# #         try:
# #             results = util.parallel.with_scoop.map_parallel(function, values)
# #             scoop_error = False
# #         except RuntimeWarning:
# #             logger.debug('Scoop is not loaded as module.')
# #             scoop_error = True
# #
# #     if not parallel or scoop_error:
# #         logger.debug('Calculating in serial.')
# #         results = map(function, values)
# #         logger.debug('Serial calculation completed.')
#
#     if parallel:
#         results = util.parallel.with_scoop.map_parallel(function, values)
#     else:
#         logger.debug('Calculating in serial.')
#         results = map(function, values)
#
#     return results
#
#
#
# def create_array(shape, function, args=None, index_position=0, parallel=True):
#     logger.debug('Creating array with shape {} in parallel {}.'.format(shape, parallel))
#
#     ## check if scoop running
#     if parallel and not util.parallel.is_running.scoop_module():
#         logger.debug('Scoop is not running falling back to serial mode.')
#         parallel = False
#
#     ## execute in parallel
#     results = map_parallel(function, args_function_generator(shape, args, index_position), parallel=parallel)
#
#     ## create array
#     array = np.array(list(results))
#     logger.debug('Calculation completed. Got {} results.'.format(len(array)))
#     array = array.reshape(shape)
#     return array
#
#
# # def create_array(shape, function, args=None, constants=True, index_position=0, parallel=True):
# #     logger.debug('Creating array with shape {} in parallel {}.'.format(shape, parallel))
# #
# #     ## check if scoop running
# #     if parallel and not util.parallel.is_running.scoop_module():
# #         logger.debug('Scoop is not running falling back to serial mode.')
# #         parallel = False
# #
# #
# #     def args_function_generator(shape, args, constants, index_position, parallel):
# #         logger.debug('Create arg function generator.')
# #
# #         ## make list of function args
# #         if args is not None:
# #             args_list = list(args)
# #         else:
# #             args_list = []
# #
# #         ## create compute function arg function
# #         if parallel and constants:
# #             ## set constants
# #             constants_uuids = [util.parallel.with_scoop.set_shared_object(arg) for arg in args]
# #
# #             def args_function(index):
# #                 args_list = [util.parallel.with_scoop.get_shared_object(constants_uuid) for constants_uuid in constants_uuids]
# #                 args_list.insert(index_position, index)
# #                 return tuple(args_list)
# #         else:
# #             args_list.insert(index_position, None)
# #             def args_function(index):
# #                 args_list[index_position] = index
# #                 return tuple(args_list)
# #
# #         logger.debug('Arg function generator created.')
# #
# #         ## return function to compute function args with current index
# #         indices = np.ndindex(*shape)
# #         for index in indices:
# #             yield lambda :args_function(index)
# #
# #     ## execute in parallel
# #     results = map_parallel(function, args_function_generator(shape, args, constants, index_position, parallel), parallel=parallel)
# #
# # #         ## add index to function args
# # #         list_function_args.insert(index_position, None)
# # #
# # #         ## return function args with current index
# # #         indices = np.ndindex(*shape)
# # #         for index in indices:
# # #             list_function_args[index_position] = index
# # #             yield tuple(list_function_args)
# # #
# # #
# # #     ## execute in parallel
# # #     results = map_parallel(function, function_arg_generator(shape, function_args, index_position), parallel=parallel)
# #
# #     ## create array
# #     array = np.array(list(results))
# #     logger.debug('Got {} results in parallel.'.format(len(array)))
# #     array = array.reshape(shape)
# #     return array