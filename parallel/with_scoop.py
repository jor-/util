import warnings
import uuid

import numpy as np

import scoop
import scoop.futures

import util.parallel.universal

import util.logging
logger = util.logging.logger



def set_shared_object(object, name=None):
    if name is None:
        name = str(uuid.uuid4())

    logger.debug('Setting shared scoop object with name "{}".'.format(name))
    d = {name:object}
    scoop.shared.setConst(**d)

    return name


def get_shared_object(name):
    logger.debug('Getting shared scoop object "{}".'.format(name))
    return scoop.shared.getConst(name)





def map_parallel(function, values):
    assert callable(function)

    logger.debug('Calculating in parallel with scoop.')
#     results = scoop.futures.map(function, values) # Do not use map_as_completed, as it does not garanties the order.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", module='scoop', category=RuntimeWarning, message='SCOOP was not started properly.')
        results = scoop.futures.map(function, values) # Do not use map_as_completed, as it does not garanties the order.
        results = tuple(results)

    logger.debug('Parallel calculation with {} results completed.'.format(len(results)))

    return results



def map_parallel_with_args(function, values, *args):
    logger.debug('Mapping function with {} args of types {} to values with scoop.'.format(len(args), tuple(map(type, args))))

    values = util.parallel.universal.args_generator_with_indices(values, (function,) + args)
    results = map_parallel(util.parallel.universal.eval_function_with_index_and_args, values)

    return reults





# def create_array(shape, function, function_args=None, index_position=1):
#     logger.debug('Creating array with shape {} in parallel with scoop.'.format(shape))
#
#     def function_arg_generator(shape, function_args, index_position):
#         index_position -= 1
#         if function_args is not None:
#             list_function_args = list(function_args)
#         else:
#             list_function_args = []
#         list_function_args.insert(index_position, None)
#
#         indices = np.ndindex(*shape)
#         for index in indices:
#             list_function_args[index_position] = index
#             yield tuple(list_function_args)
#
#
#     ## execute in parallel
#     results = map_parallel(function, function_arg_generator(shape, function_args, index_position))
#
#     ## create array
#     array = np.array(list(results))
#     logger.debug('Got {} results in parallel.'.format(len(array)))
#     array = array.reshape(shape)
#     return array
#
#
# # def create_array(shape, function, function_args=None, function_args_first=True):
# #     logger.debug('Creating array with shape {} in parallel with scoop.'.format(shape))
# #
# #     ## create indices
# #     indices = np.ndindex(*shape)
# #     if function_args is None:
# #         function_args = ()
# #     if function_args_first:
# #         indices = [(function_args, i) for i in indices]
# #     else:
# #         indices = [(i, function_args) for i in indices]
# #
# #     ## execute in parallel
# #     results = map(function, indices)
# #
# #     ## create array
# #     array = np.array(list(results))
# #     logger.debug('Got {} results in parallel.'.format(len(array)))
# #     array = array.reshape(shape)
# #     return array



# def is_scoop_loaded():
#     ## check if root process is running with scoop
#     try:
#         globals()['scoop']
#         return True
#     ## check if worker process
#     except KeyError:
#         return globals()['__name__']=='SCOOP_WORKER'
