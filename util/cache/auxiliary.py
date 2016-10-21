def set_wrapper_attributes(wrapper, base_function):
    wrapper.__doc__ = base_function.__doc__
    wrapper.__name__ = base_function.__name__
    wrapper.__qualname__ = base_function.__qualname__
    wrapper.__func__ = base_function
    return wrapper
    