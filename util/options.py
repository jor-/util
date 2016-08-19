import os.path
import stat

import numpy as np
import h5py

import util.io.fs
import util.logging
logger = util.logging.logger

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
        value = self._replace_environment_vars(value, self.replace_environment_vars_at_set)

        ## check if writable
        if not self.is_writable():
            raise OSError('Option file is not writable.')
        
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
        value = self._replace_environment_vars(value, self.replace_environment_vars_at_get)
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


    @property
    def filename(self):
        hdf5_file = self.__hdf5_file
        return hdf5_file.filename


    def open(self, file, mode='a'):
        self.close()

        logger.debug('Opening option file {} with mode {}.'.format(file, mode))

        try:
            f = h5py.File(file, mode=mode)
        except OSError:
            logger.debug('File {} could not been open. Trying read_only mode.'.format(file))
            f = h5py.File(file, mode='r')

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
    def _replace_environment_vars(value, replace=True):
        if replace and (isinstance(value, str) or isinstance(value, bytes)):
            value = os.path.expanduser(value)
            value = os.path.expandvars(value)
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
        f = self.__hdf5_file
        string_object_list = []

        def append_if_string_option(name, object):
            ## check if dataset
            try:
                value = object.value
            except AttributeError:
                value = None

            ## check type
            if type(value) is str:
                string_object_list.append(name)


        f.visititems(append_if_string_option)

        return string_object_list


    def replace_all_str_options(self, old_str, new_str):
        for option in self.get_all_str_options():
            old_option = self[option]
            new_option = old_option.replace(old_str, new_str)
            if old_option != new_option:
                self[option] = new_option
                logger.info('Option {} updated from {} to {}.'.format(option, old_option, new_option))
            else:
                logger.debug('Option {} with value {} does not have to be updated.'.format(option, old_option))
