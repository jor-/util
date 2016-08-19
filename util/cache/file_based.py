import util.io.fs
import util.io.universal

import util.logging
logger = util.logging.logger



## file cache

def decorator(cache_file_function=None):
    
    def decorate_function(function, cache_file_function=None):
        ## if no cache file function is passed used passed cache file function name or default cache file function name 
        cache_file_function_defined = not (cache_file_function is None or isinstance(cache_file_function, str))
        if not cache_file_function_defined:
            ## passed cache file function name
            if isinstance(cache_file_function, str):
                cache_file_function_name = cache_file_function
            ## default cache file function name
            else:
                function_name = function.__name__
                cache_file_function_name = '{function_name}_cache_file'.format(function_name=function_name)
        
        def function_wrapper(*args, **kargs):            
            ## calculate cache file
            if cache_file_function_defined:
                cache_file = cache_file_function(*args, **kargs)
            else:
                cache_file = None
                try:
                    self = args[0]
                except IndexError:
                    logger.warn('This is not a method call! Cache file {} is not defined. Using no cache!'.format(cache_file_function_name))
                else:
                    try:
                        cache_file_function_by_name = getattr(self, cache_file_function_name)
                    except AttributeError:
                        logger.warn('Cache file {} is not defined in {}. Using no cache!'.format(cache_file_function_name, self))
                    else:
                        cache_file = cache_file_function_by_name(*args[1:], **kargs)
            
            ## if cache file is defined, use cache
            if cache_file is not None:                
                ## if cache file is available, load cached value
                try:
                    value = util.io.universal.load(cache_file)
                    logger.debug('Calculated value loaded from cache file {}.'.format(cache_file))
                ## if cache file is not available, calculate value and save as cached value
                except OSError:
                    logger.debug('No cached value found for {}.'.format(cache_file))
                    value = function(*args, **kargs)
                    logger.debug('Saving calculated value to cache file {}.'.format(cache_file))
                    util.io.universal.save(cache_file, value)
                    util.io.fs.make_read_only(cache_file)
            ## if cache file not defined, calculate value without cache
            else:
                value = function(*args, **kargs)
            
            return value
        
        return function_wrapper
    
    return lambda function: decorate_function(function, cache_file_function=cache_file_function)

