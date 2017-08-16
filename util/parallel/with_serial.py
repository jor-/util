import itertools

import util.parallel.universal

import util.logging



def map_serial_with_args(function, indices, *args):
    util.logging.debug('Mapping function with {} args of types {} to values in serial.'.format(len(args), tuple(map(type, args))))

    values = util.parallel.universal.args_generator_with_indices(indices, args)
    results = itertools.starmap(function, values)
    results = tuple(results)

    util.logging.debug('Serial calculation with {} results completed.'.format(len(results)))

    return results
