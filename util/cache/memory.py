import itertools

import cachetools
import cachetools.keys

import util.cache.auxiliary


## hash

_kwargs_mark = (object(),)

def hash_key(*args, **kwargs):
    """Return a cache key for the specified hashable arguments."""

    if kwargs:
        return cachetools.keys._HashedTuple(args + _kwargs_mark + tuple(itertools.chain(sorted(kwargs.items()))))
    else:
        return cachetools.keys._HashedTuple(args)


def typed_hash_key(*args, **kwargs):
    """Return a typed cache key for the specified hashable arguments."""

    key = hash_key(*args, **kwargs)
    key += tuple(type(v) for v in args)
    key += tuple(type(v) for _, v in sorted(kwargs.items()))
    return key


## dependency

_dependency_mark = (object(),)

def attribute_dependend_key(key_function, *dependencies):
    """Return a cache key for the specified hashable arguments with additional dependent arguments."""

    def dependend_key_function(self, *args, **kwargs):
        key = hash_key(*args, **kwargs)
        
        if len(dependencies) > 0:
            dependecies_dict = {}
            for dependency in dependencies:
                value = eval(dependency)
                dependecies_dict[dependency] = value
            
            key = key + cachetools.keys._HashedTuple(_dependency_mark + tuple(itertools.chain(sorted(dependecies_dict.items()))))

        return key
    
    return dependend_key_function


def attribute_dependend_hash_key(*dependencies):
    return attribute_dependend_key(hash_key, *dependencies)


def attribute_dependend_typed_hash_key(*dependencies):
    return attribute_dependend_key(typed_hash_key, *dependencies)


def add_dependency_to_key(key=hash_key, dependency=None):
    if dependency is not None:
        if isinstance(dependency, str):
            dependency = (dependency,)
        key = attribute_dependend_key(key, *dependency)
    return key
    

## decorator

def decorator(maxsize=1, key=hash_key, lock=None, dependency=None):
    key = add_dependency_to_key(key, dependency=dependency)
    cache = cachetools.LRUCache(maxsize=maxsize)
    return cachetools.cached(cache, key=key, lock=lock)


def method_decorator(maxsize=1, key=hash_key, lock=None, dependency=None, shared=False):
    if shared:
        return decorator(maxsize=maxsize, key=key, lock=lock, dependency=dependency)
    else:
        key = add_dependency_to_key(key, dependency=dependency)
        
        def decorate(method):
            method_name = method.__name__
        
            def get_cache(self):
                ## cache dict
                try:
                    cache_dict = self.__cache_dict__
                except AttributeError:
                    cache_dict = {}
                    self.__cache_dict__ = cache_dict
                ## cache for method
                try:
                    cache = cache_dict[method_name]
                except KeyError:
                    cache = cachetools.LRUCache(maxsize=maxsize)
                    cache_dict[method_name] = cache
                ## return cache
                return cache
            
            wrapper = cachetools.cachedmethod(get_cache, key=key, lock=lock)(method)
            wrapper = util.cache.auxiliary.set_wrapper_attributes(wrapper, method)
            return wrapper
        
        return decorate

