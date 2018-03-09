import os.path

import util.cache.auxiliary
import util.io.fs
import util.io.universal

import util.logging


# file cache

def save(file, value, save_function):
    if save_function is not None:
        util.logging.debug('Saving calculated value to cache file {}.'.format(file))
        try:
            save_function(file, value)
        except PermissionError as e:
            if os.path.exists(file):
                util.logging.debug('Cache file {} was already created while calculating the value.'.format(file))
            else:
                util.logging.warn('File permissions are not sufficient to save calculated value to cache file {}: {}.'.format(file, e))
        except OSError as e:
            util.logging.warn('The calculated value could not be saved to cache file {}: {}.'.format(file, e))
        else:
            util.logging.debug('Calculated value saved to cache file {}.'.format(file))
            util.io.fs.apply_recursively(file, util.io.fs.remove_write_permission)
    else:
        util.logging.debug('Not saving calculated value to cache file {} because no save function is provided.'.format(file))


def decorator(cache_file_function=None, load_function=None, save_function=None):
    if load_function is None:
        load_function = util.io.universal.load
    if save_function is None:
        save_function = util.io.universal.save

    def decorate(function, cache_file_function=None, load_function=None, save_function=None):
        # if no cache file function is passed used passed cache file function name or default cache file function name
        cache_file_function_defined = not (cache_file_function is None or isinstance(cache_file_function, str))
        if not cache_file_function_defined:
            # passed cache file function name
            if isinstance(cache_file_function, str):
                cache_file_function_name = cache_file_function
            # default cache file function name
            else:
                function_name = function.__name__
                cache_file_function_name = '{function_name}_cache_file'.format(function_name=function_name)

        def wrapper(*args, **kargs):
            # calculate cache file
            if cache_file_function_defined:
                cache_file = cache_file_function(*args, **kargs)
            else:
                cache_file = None
                try:
                    self = args[0]
                except IndexError:
                    util.logging.warn('Can not use default cache file function name, because cache file {} is not defined and this is not a method call! Using no cache!'.format(cache_file_function_name))
                else:
                    try:
                        cache_file_function_by_name = getattr(self, cache_file_function_name)
                    except AttributeError:
                        util.logging.warn('Cache file {} is not defined in {}. Using no cache!'.format(cache_file_function_name, self))
                    else:
                        cache_file = cache_file_function_by_name(*args[1:], **kargs)

            # if cache file is defined, use cache
            if cache_file is not None and load_function is not None:
                # try to load cache file
                try:
                    value = load_function(cache_file)
                except FileNotFoundError as e:
                    util.logging.debug('No cached value found at {}.'.format(cache_file))
                    calculate_value = True
                except OSError as e:
                    util.logging.warn('Cached value at {} could not be loaded: {}.'.format(cache_file, e))
                    calculate_value = True
                else:
                    util.logging.debug('Calculated value loaded from cache file {}.'.format(cache_file))
                    calculate_value = False

                # if cache file is not loadable, calculate value and save as cached value
                if calculate_value:
                    value = function(*args, **kargs)
                    save(cache_file, value, save_function)

            # if cache file not defined, calculate value without cache
            else:
                value = function(*args, **kargs)

            return value

        wrapper = util.cache.auxiliary.set_wrapper_attributes(wrapper, function)
        return wrapper

    return lambda function: decorate(function, cache_file_function=cache_file_function, load_function=load_function, save_function=save_function)
