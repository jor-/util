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