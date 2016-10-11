import os.path
import stat
import collections
import copy

import numpy as np
import h5py

import util.io.fs
import util.io.universal
import util.logging
logger = util.logging.logger


## Options File

class OptionsFile():

    def __init__(self, file, mode='a', replace_environment_vars_at_set=False, replace_environment_vars_at_get=False):
        ## prepare file name
        if os.path.isdir(file):
            file = os.path.join(file, 'options.hdf5')
        else:
            (root, ext) = os.path.splitext(file)
            if ext == '':
                file += '.hdf5'
        ## open
        self.open(file, mode)
        ## save replace variable
        self.replace_environment_vars_at_set = replace_environment_vars_at_set
        self.replace_environment_vars_at_get = replace_environment_vars_at_get


    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


    def __setitem__(self, key, value):
        ## check value
        if value is None:
            raise ValueError('Value None is not allowed (for key {}).'.format(key))
        
        ## replace env
        if self.replace_environment_vars_at_set:
            value = self._replace_environment_vars(value)

        ## check if writable
        if not self.is_writable():
            raise OSError('Option file {} is not writable.'.format(self.filename))
        
        ## insert supported types
        try:
            f = self.__hdf5_file
            ## set if key exists
            try:
                f[key][()] = value
            ## set if key not exists
            except KeyError:
                f[key] = value
        
        ## try to insert unsupported types
        except TypeError as e:
            successfully_inserted = False
            
            ## if dict, insert each item since dict is not supported in HDF5
            if isinstance(value, dict):
                for (key_i, value_i) in value.items():
                    self['{}/{}'.format(key, key_i)] = value_i
                successfully_inserted = True
            
            ## if iterable of unicode strings
            if not successfully_inserted:
                value_array = np.asanyarray(value)
                if value_array.dtype.kind == 'U':                
                    h5_dtype = h5py.special_dtype(vlen=str)
                    dataset = f.create_dataset(key, value_array.shape, dtype=h5_dtype)
                    dataset[:] = value_array
                    successfully_inserted = True
                    
            ## if tuple or list with different types, insert each item since generic object arrays are not supported in HDF5
            if not successfully_inserted and (isinstance(value, tuple) or isinstance(value, list)):
                for i in range(len(value)):
                    self['{}/{}'.format(key, i)] = value[i]
                successfully_inserted = True
            
            ## if not insertable, raise error
            if not successfully_inserted:
                raise TypeError('The value {} for key {} could not be stored, because this type of value is not supported in hdf5.'.format(value, key)) from e
    

    def __getitem__(self, name):
        ## get value
        try:
            item = self.__hdf5_file[name]
        except KeyError as e:
            raise KeyError('The key {} is not in the option file {}.'.format(name, self.filename)) from e
        value = item.value
        ## replace env
        if self.replace_environment_vars_at_get:
            value = self._replace_environment_vars(value)
        ## return
        return value


    def __delitem__(self, name):
        try:
            del self.__hdf5_file[name]
        except KeyError as e:
            raise KeyError('The key {} is not in the option file {}.'.format(name, self.filename)) from e
        #TODO delete empty groups
    

    @property
    def __hdf5_file(self):
        hdf5_file_object = self.__hdf5_file_object
        if hdf5_file_object is not None:
            return hdf5_file_object
        else:
            raise ValueError('File is closed.')


    def __str__(self):
        return 'Option file {}'.format(self.filename)

    @property
    def filename(self):
        hdf5_file = self.__hdf5_file
        return hdf5_file.filename


    def open(self, file, mode='a'):
        self.close()

        logger.debug('Opening option file {} with mode {}.'.format(file, mode))

        try:
            f = h5py.File(file, mode=mode)
        except OSError as e:
            if mode != 'r':
                logger.debug('File {} could not been open. Trying read_only mode.'.format(file))
                self.open(file, mode='r')
            else:
                raise OSError('Option file {} could not been open. Error: {}. Error code: {}.'.format(file, e.strerror, e.errno)) from e
        else:
            logger.debug('File {} opened.'.format(file))
            self.__hdf5_file_object = f


    def close(self):
        try:
            file = self.__hdf5_file_object
        except AttributeError:
            file = None

        if file is not None:
            logger.debug('Closing option file {}.'.format(file.filename))
            file.close()
            self.__hdf5_file_object = None


    @staticmethod
    def _replace_environment_vars(value):
        def is_str(value):
            return isinstance(value, str) or isinstance(value, bytes)
        
        def expand(value):
            if is_str(value):
                value = os.path.expanduser(value)
                value = os.path.expandvars(value)
            return value

        if is_str(value):
            value = expand(value)
        elif isinstance(value, collections.Iterable) and any(map(is_str, value)):
            value = tuple(map(expand, value))
        return value


    ## permissions

    def is_writable(self):
        f = self.__hdf5_file
        return f.mode == 'r+'

    def is_read_only(self):
        f = self.__hdf5_file
        return f.mode == 'r'

    def make_writable(self):
        if not self.is_writable():
            logger.debug('Opening {} writable.'.format(self.filename))
            file = self.filename
            self.close()
            util.io.fs.make_writable(file)
            self.open(file)
        else:
            logger.debug('File {} is writable.'.format(self.filename))
        assert self.is_writable

    def make_read_only(self):
        if not self.is_read_only():
            logger.debug('Opening {} read_only.'.format(self.filename))
            file = self.filename
            self.close()
            util.io.fs.make_read_only(file)
            self.open(file, 'r')
        else:
            logger.debug('File {} is read_only.'.format(self.filename))
        assert self.is_read_only


    ## print

    def print_all_options(self):
        f = self.__hdf5_file

        def print_option(name, object):
            ## check if dataset
            try:
                value = object.value
            except AttributeError:
                value = None

            ## check type
            if value is not None:
                print('{}: {}'.format(name, value))

        f.visititems(print_option)


    ## replace str

    def get_all_str_options(self):
        string_object_list = []

        def append_if_string_option(name, object):
            ## check if dataset
            try:
                value = object.value
            except AttributeError:
                pass
            ## check type
            else:
                if isinstance(value, str) or (isinstance(value, np.ndarray) and any(tuple(map(lambda v: isinstance(v[np.newaxis][0], str), np.nditer(value, flags=['refs_ok']))))):
                    string_object_list.append(name)

        f = self.__hdf5_file
        f.visititems(append_if_string_option)

        return string_object_list
    

    def replace_all_str_options(self, old_str, new_str):
        def replace_string_in_option(option, object):
            ## check if dataset
            try:
                value = object.value
            except AttributeError:
                pass
            ## check type
            else:
                ## replace in (scalar) string
                if isinstance(value, str):
                    new_value = value.replace(old_str, new_str)
                ## replace in string array
                elif isinstance(value, np.ndarray) and all(map(lambda v: isinstance(v[np.newaxis][0], str), np.nditer(value, flags=['refs_ok']))):
                    new_value = value.copy()
                    for v in np.nditer(new_value, flags=['refs_ok'], op_flags=['readwrite']):
                        v[...] = v[np.newaxis][0].replace(old_str, new_str)
                    new_value = new_value.astype('U')
                ## otherwise skip
                else:
                    new_value = None
                ## set replaced option value
                if new_value is not None and np.any(value != new_value):
                    self[option] = new_value
                    logger.info('Option {} updated from {} to {}.'.format(option, value, new_value))
                else:
                    logger.debug('Option {} with value {} does not have to be updated.'.format(option, value))

        f = self.__hdf5_file
        f.visititems(replace_string_in_option)




## Options Dict

class OptionError(KeyError):
    
    def __init__(self, option, option_object, message):
        self.option = option
        self.option_object = option_object
        message = '{}: {}'.format(type(option_object).__name__, message)
        super().__init__(message)


class UnknownOptionError(OptionError):
    
    def __init__(self, option, option_object):
        message = 'Option {} is unknown.'.format(option)
        super().__init__(option, option_object, message)


class NoneSetOptionError(OptionError):
    
    def __init__(self, option, option_object):
        message = 'Option {} is not set.'.format(option)
        super().__init__(option, option_object, message)


class ImmutableOptionError(OptionError):
    
    def __init__(self, option, option_object):
        message = 'Option {} can not be changed.'.format(option)
        super().__init__(option, option_object, message)


class IncalculableOptionError(OptionError):
    
    def __init__(self, option, option_object):
        message = 'Option {} is incalculable.'.format(option)
        super().__init__(option, option_object, message)


class UnknownListenerError(KeyError):
    
    def __init__(self, listener):
        message = 'Listener {} is unknown.'.format(listener)
        super().__init__(message)



class OptionsBase():
    
    NON_OPTION_NAMES = ('_options', '_option_names')
    
    def __init__(self, options=None, default_options=None, option_names=None):
        logger.debug('Initiating {} with options {}, default_options {} and option_names {}.'.format(type(self).__name__, options, default_options, option_names))
        self._option_names = option_names
        self._options = {}

        ## set passed and default options
        if default_options is not None:
            for option in default_options:
                if options is None or option not in options:
                    self[option] = default_options[option]

        ## set options
        if options is not None:
            for option in options:
                self[option] = options[option]
    
    
    ## dict methods
    
    def __getattr__(self, option):
        if self._has_option(option):
            return self._get_option(option)
        else:
            return super().__getattribute__(option)    
    
    
    def __setattr__(self, option, value):
        if self._has_option(option):
            self._set_option(option, value)
        else:
            object.__setattr__(self, option, value)

    
    def __delattr__(self, option):
        if self._has_option(option):
            self._del_option(option)
        else:
            object.__delattr__(self, option)

    
    def __getitem__(self, option):        
        return eval('self.'+option)

    
    def __setitem__(self, option, value):
        self.__setattr__(option, value)
    
    
    def __delitem__(self, option):
        self.__delattr__(option)
    

    def __len__(self):
        return len(self._options)    
    
    
    def __contains__(self, option):
        return self._has_value(option)
    
    
    def __iter__(self):
        return self._options.keys()
    
    
    def __str__(self):
        return str(self._options)
    
    
    def __repr__(self):
        return '{module_name}.{class_name}({options!r})'.format(module_name=self.__module__, class_name=self.__class__.__name__,  options=self._options)
    
    
    ## get and set options method
    
    def _get_option(self, option):
        logger.debug('Getting option {}.'.format(option))
        try:
            return self._options[option]
        except KeyError:
            logger.debug('Option {} is not set.'.format(option))
            raise NoneSetOptionError(option, self)
    
    
    def _set_option(self, option, value):
        logger.debug('Setting option {}.'.format(option))
        self._options[option] = value
    
    
    def _del_option(self, option, not_exist_okay=False):
        logger.debug('Deleting option {} with not_exist_okay {}.'.format(option, not_exist_okay))
        try:
            del self._options[option]
        except KeyError:
            logger.debug('Option {} is not set.'.format(option))
            if not not_exist_okay:
                raise NoneSetOptionError(option, self)
    
    
    def _clear_all_options(self):
        for option in tuple(self._options.keys()):
            self._del_option(option)
    
    
    def _has_option(self, option):
        return (not option in type(self).NON_OPTION_NAMES) and (self._option_names is None or option in self._option_names)
        
    
    def _has_value(self, option):
        return option in self._options
    
    ## copy
    def copy(self):
        return copy.deepcopy(self)



class OptionsWithValueCheck(OptionsBase):
    
    VALUE_CHECK_METHOD_NAME = '{}_check'
    
    
    def _set_option(self, option, value):
        value_check_method_name = self.VALUE_CHECK_METHOD_NAME.format(option)

        try:
            value_check_method = getattr(self, value_check_method_name)
        except AttributeError:
            pass
        else:
            checked_value = value_check_method(value)
            if checked_value is not None:
                value = checked_value
        
        super(). _set_option(option, value)
    


class OptionsWithListeners(OptionsBase):
    
    NON_OPTION_NAMES = OptionsBase.NON_OPTION_NAMES + ('_listeners',)

    def __init__(self, **kargs):
        self._listeners = {}
        super().__init__(**kargs)
    
    
    ## get and set options method
    
    def _set_option(self, option, new_value):
        
        ## check old value
        if self._has_value(option):
            old_value = self._get_option(option)
            must_set = np.any(new_value != old_value)
        else:
            must_set = True
            
        ## set new option value and call listener
        if must_set:
            super()._set_option(option, new_value)
            self._call_listeners(option)
    
    
    def _del_option(self, option, not_exist_okay=False):
        super()._del_option(option, not_exist_okay=not_exist_okay)
        self._call_listeners(option)
    
    
    ## listener methods
    
    def add_listener(self, option, listener):
        ## check input
        if not self._has_option(option):
            raise UnknownOptionError(option, self)
        if not callable(listener):
            raise ValueError('The listener must be callable, but it is {}.'.format(listener))
        
        ## add listener
        try:
            listener_list = self._listeners[option]
        except KeyError:
            self._listeners[option] = [listener]
        else:
            listener_list.append(listener)
    
    
    def remove_listener(self, option, listener):
        if not self._has_option(option):
            raise UnknownOptionError(option, self)
        if not callable(listener):
            raise ValueError('The listener must be callable, but it is {}.'.format(listener))
            
        try:
            listener_list = self._listeners[option]
        except KeyError:
            raise UnknownOptionError(listener, self)
        else:
            listener_list.remove(listener)
    
    
    def get_listeners(self, option):
        try:
            return self._listeners[option]
        except KeyError:
            return []
    
    
    def _call_listeners(self, option):
        try:
            listener_list = self._listeners[option]
        except KeyError:
            pass
        else:
            if self._has_value(option):
                value =  self._get_option(option)
            else:
                value = None
            
            for listener in listener_list:
                listener(option, value)



class OptionsWithListenersAndCopy(OptionsWithListeners):
    
    def add_copy_listener(self, independent_option, dependent_option, dependent_option_object=None, copy_now=True):
        if dependent_option_object is None:
            dependent_option_object = self
        
        def listener(independent_option, new_value):
            dependent_option_object[dependent_option] = new_value
        
        if copy_now:
            try:
                value = self[independent_option]
            except NoneSetOptionError:
                pass
            else:
                listener(independent_option, value)
        
        self.add_listener(independent_option, listener)



class OptionsWithListenersAndDependencies(OptionsWithListeners):
    
    DEPENDENT_VALUE_CHANGED_METHOD_NAME = '{}_depending_value_changed'

    def __init__(self, dependencies=None, **kargs):
        super().__init__(**kargs)
        
        if dependencies is not None:
            for dependent_option, independent_option in dependencies.items():
                self.add_dependency(independent_option, dependent_option)
    
    
    def add_dependency(self, independent_option, dependent_option, dependent_option_object=None):
        if dependent_option_object is None:
            dependent_option_object = self

        depending_value_changed_method_name = self.DEPENDENT_VALUE_CHANGED_METHOD_NAME.format(dependent_option)
        
        def listener(independent_option, new_value):
            try:
                depending_value_changed_method = getattr(dependent_option_object, depending_value_changed_method_name)
            except AttributeError:
                remove_value = True
            else:
                remove_value = depending_value_changed_method(independent_option, new_value)
            
            if remove_value:
                dependent_option_object._del_option(dependent_option, not_exist_okay=True)
        
        self.add_listener(independent_option, listener)
    


class OptionsWithCache(OptionsBase):
    
    CALCULATION_METHOD_NAME= '{}_calculated'
    
    def _get_option(self, option):
        try:
            return super()._get_option(option)
        except NoneSetOptionError as e:
            try:
                return self._recalculated_option_value(option)
            except IncalculableOptionError:
                raise e
        
    
    def _recalculated_option_value(self, option):
        logger.debug('Recalculating option {}.'.format(option))
        calculation_method_name = self.CALCULATION_METHOD_NAME.format(option)
        try:
            value = getattr(self, calculation_method_name)
        except (AttributeError, NoneSetOptionError):
            raise IncalculableOptionError(option, self)
        else:
            self._set_option(option, value)
            return value



class Options(OptionsWithCache, OptionsWithValueCheck, OptionsWithListenersAndCopy, OptionsWithListenersAndDependencies):
    pass
    
    

def as_options(option_object, option_class=Options):
    if not isinstance(option_object, option_class):
        option_object = option_class(option_object)
    return option_object

