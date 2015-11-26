import os
import stat

import numpy as np

import util.io.fs
import util.io.object
import util.parallel.with_multiprocessing

import util.logging
logger = util.logging.logger


class CacheMissError(LookupError):

    def __init__(self, value_name=None):
        if value_name is not None:
            err_str = 'Cache miss for value {}.'.format(value_name)
        else:
            err_str = 'Cache miss.'
        super().__init__(err_str)


class Cache():

    def load_value(self, value_name):
        raise NotImplementedError("Please implement this method.")

    def save_value(self, value_name, value):
        raise NotImplementedError("Please implement this method.")


    def get_value(self, value_name, calculate_function, as_shared_array=False):
        try:
            value = self.load_value(value_name)
            logger.debug('Loaded value "{}" from {}.'.format(value_name, self))
        except CacheMissError:
            logger.debug('Value "{}" not found in {}. Calculating value and saving.'.format(value_name, self))
            value = calculate_function()
            if as_shared_array:
                value = util.parallel.with_multiprocessing.shared_array(value)
            self.save_value(value_name, value)

        return value

    def __getitem__(self, key):
        return self.get_value(*key)


    def has_value(self, value_name):
        try:
            self.load_value(value_name)
            logger.debug('Value "{}" is available in {}.'.format(value_name, self))
            return True
        except CacheMissError:
            logger.debug('Value "{}" is not available in {}.'.format(value_name, self))
            return False

    def __contains__(self, key):
        return self.has_value(key)


    def del_value(self, value_name):
        raise NotImplementedError("Please implement this method.")

    def __delitem__(self, key):
        self.del_value(key)


    def __str__(self):
        return self.__class__.__name__



class MemoryCache(Cache):

    def __init__(self):
        self.memory_cache = {}

    def load_value(self, value_name):
        try:
            return self.memory_cache[value_name]
        except KeyError:
            pass
        raise CacheMissError(value_name)

    def save_value(self, value_name, value):
        self.memory_cache[value_name] = value

    def del_value(self, value_name):
        del self.memory_cache[value_name]



class MemoryCacheDeactivatable(MemoryCache):

    def __init__(self, enabled=True):
        super().__init__()
        self.memory_cache_enabled = enabled


    def switch(self, enabled):
        self.memory_cache_enabled = enabled

    def is_enabled(self):
        return self.memory_cache_enabled


    def load_value(self, value_name):
        if self.is_enabled():
            return super().load_value(value_name)
        else:
            raise CacheMissError(value_name)

    def save_value(self, value_name, value):
        if self.is_enabled():
            self.memory_cache[value_name] = value



class HDD_Cache(Cache):

    def __init__(self, cache_dir, load_function, save_function, use_memory_cache=False):
        self.use_memory_cache = use_memory_cache
        self.memory_cache = MemoryCacheDeactivatable(enabled=use_memory_cache)

        self.cache_dir = cache_dir

        assert callable(load_function) and callable(save_function)
        self.load_function = load_function
        self.save_function = save_function
    

    def memory_cache_switch(self, enabled):
        self.use_memory_cache  = enabled


    def get_file(self, value_name):
        return os.path.join(self.cache_dir, value_name)


    def load_value(self, value_name):
        try:
            return self.memory_cache.load_value(value_name)
        except CacheMissError:
            file = self.get_file(value_name)
            try:
                value = self.load_function(file)
            except OSError:
                raise CacheMissError(value_name)
            self.memory_cache.save_value(value_name, value)
            return value


    def save_value(self, value_name, value):
        self.memory_cache.save_value(value_name, value)
        file = self.get_file(value_name)
        util.io.fs.makedirs(file, exist_ok=True)
        self.save_function(file, value)
        util.io.fs.make_read_only(file)


    def get_value(self, value_name, calculate_function, as_shared_array=False, use_memory_cache=True):
        self.memory_cache.switch(self.use_memory_cache and use_memory_cache)
        return super().get_value(value_name, calculate_function, as_shared_array=as_shared_array)


    def has_value(self, value_name):
        file = self.get_file(value_name)
        exists = os.path.exists(file)
        logger.debug('Value "{}" existing {} in {}.'.format(value_name, exists, self))
        return exists


    def __str__(self):
        return self.__class__.__name__ + ': ' + self.cache_dir






class HDD_NPY_Cache(HDD_Cache):

    def __init__(self, cache_dir, use_memory_cache=False):
        load_function = np.load
        save_function = np.save
        super().__init__(cache_dir, load_function, save_function, use_memory_cache=use_memory_cache)



class HDD_ObjectCache(HDD_Cache):

    def __init__(self, cache_dir, use_memory_cache=False):
        super().__init__(cache_dir, util.io.object.load, util.io.object.save, use_memory_cache=use_memory_cache)



class HDD_ObjectWithSaveCache(HDD_Cache):

    def __init__(self, cache_dir, load_function, use_memory_cache=False):
        def save_function(file, value):
            value.save(file)
        super().__init__(cache_dir, load_function, save_function, use_memory_cache=use_memory_cache)



