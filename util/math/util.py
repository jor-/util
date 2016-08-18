import numpy as np

import util.logging
logger = util.logging.logger


def max_dtype(a, b):
    logger.debug('Calculating max dtype of {} and {}.'.format(a,b))

    if isinstance(a, np.floating):
        if isinstance(b, np.integer):
            return a
        if isinstance(b, np.floating):
            if np.ffinfo(a).resolution > np.ffinfo(b).resolution:
                return a
            else:
                return b


    if isinstance(a, np.integer):
        if isinstance(b, np.integer):
            if np.iinfo(a).max > np.iinfo(b).max:
                return a
            else:
                return b
        if isinstance(b, np.floating):
            return b

    raise ValueError('Dtype {} and {} are not comparable.'.format(a,b))



def min_int_dtype(*values, unsigned=False):    
    ## get max value
    if len(values) > 0:
        max_value = max(values)
    else:
        max_value = 0

    ## get dtype
    if unsigned:
        int_dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)
    else:
        int_dtypes = (np.int8, np.int16, np.int32, np.int64)
    i = 0
    while np.iinfo(int_dtypes[i]).max < max_value:
        i = i+1
    dtype = int_dtypes[i]
    
    ## return 
    logger.debug('Minimal int dtype (unsigned={}) for values {} is {}.'.format(unsigned, values, dtype))
    return dtype

