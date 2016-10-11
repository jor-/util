import abc
import os
import re

import numpy as np

import util.io.fs
import util.index_database.general

import util.logging
logger = util.logging.logger



class Database(util.index_database.general.Database):
    
    def __init__(self, value_dir, value_filenames, value_reliable_decimal_places=15, tolerance_options=None):
        ## call super constructor
        super().__init__(value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
        
        ## set value dir
        self.value_dir = value_dir
        
        ## save filenames in list
        if isinstance(value_filenames, str):
            value_filenames = [value_filenames]
        self.value_filenames = value_filenames
        assert len(self.value_filenames) > 0
        
    
    def __str__(self):
        return 'Index file system database {}'.format(self.value_dir)
    
    
    ## setter and getter for files and dirs
    
    @property
    def value_dir(self):
        return self._value_dir
    
    @value_dir.setter
    def value_dir(self, value_dir):               
        ## check value dir
        for s in ['{', '}']:
            if len(value_dir.split(s)) != 2:
                raise ValueError('The value dirname must contain exactly one "{}". But the filename is {}.'.format(s, value_file))
        
        ## calculate dirs
        value_dir = os.path.realpath(value_dir)
        
        base_dir_including_index = value_dir
        while re.search('\{.*\}', os.path.dirname(base_dir_including_index)):
            base_dir_including_index = os.path.dirname(base_dir_including_index)
        
        base_dir = os.path.dirname(base_dir_including_index)
        assert len(base_dir) > 0

        ## make base dir
        os.makedirs(base_dir, exist_ok=True)
        
        ## store needed dirs
        self._value_dir = value_dir
        self._base_dir_including_index = base_dir_including_index
        self._base_dir = base_dir

        ## save needed regular expressions        
        index_dir = os.path.basename(base_dir_including_index)
        self._index_dir_regular_expression_search = re.sub('\{.*\}', '[0-9]*', index_dir)
        self._index_dir_regular_expression_split = re.sub('\{.*\}', '', index_dir)
    
    
    @property
    def value_filenames(self):
        return self._value_filenames
    
    @value_filenames.setter
    def value_filenames(self, value_filenames):        
        ## save filenames in list
        if isinstance(value_filenames, str):
            value_filenames = [value_filenames]
        assert len(value_filenames) > 0
        
        self._value_filenames = value_filenames
    
    
    ## load and save

    @abc.abstractmethod
    def _load_file(self, value_file):
        raise NotImplementedError()

    @abc.abstractmethod
    def _save_file(self, value_file, value):
        raise NotImplementedError()

    
    ## access
    
    def value_files(self, index):
        ## make absolute filenames
        value_dir = self.value_dir.format(index)
        value_files = [os.path.join(value_dir, value_filename) for value_filename in self.value_filenames]
        return value_files
        
    
    def get_value(self, index):
        value_files = self.value_files(index)

        ## load each file
        value_array = []
        for value_file in value_files:        
            try:
                current_value_array = self._load_file(value_file)
            except FileNotFoundError:
                raise util.index_database.general.DatabaseIndexError(self, index)
            value_array.append(current_value_array)

        ## return as one array
        if len(value_array) > 1:
            value_array = np.asanyarray(value_array)
        else:
            value_array = value_array[0]
        return value_array


    def set_value(self, index, value, overwrite=False):
        logger.debug('{}: Setting value at index {} to {} with overwrite {}.'.format(self, index, value, overwrite))
        
        ## check value and make it as two dim array
        value = np.asanyarray(value)        
        assert (len(self.value_filenames) == 1 and (value.ndim == 1 or (value.ndim == 2 and len(value) == len(self.value_filenames)))) or (len(self.value_filenames) > 1 and len(value) == len(self.value_filenames))
        
        if value.ndim == 1:
            value = [value]

        ## make value dir
        value_dir = self.value_dir.format(index)
        os.makedirs(value_dir, exist_ok=True)

        ## save each value to file
        for i in range(len(self.value_filenames)):
            value_file = os.path.join(value_dir, self.value_filenames[i])
            value_file_exists = os.path.exists(value_file)
            if value_file_exists and overwrite:
                util.io.fs.make_writable(value_file)
            if overwrite or not value_file_exists:
                self._save_file(value_file, value[i])
                util.io.fs.make_read_only(value_file)
            else:
                logger.debug('{}: Overwritting value at index {} is not allowed.'.format(self, index))
                raise util.index_database.general.DatabaseOverwriteError(self, index)
    
    
    def used_indices(self):
        ## get all value files
        all_index_dirs = util.io.fs.find_with_regular_expression(self._base_dir, self._index_dir_regular_expression_search, exclude_files=True, use_absolute_filenames=False, recursive=False)
        
        ## get all used indices
        used_indices = []
        for index_dir in all_index_dirs:
            # get name starting at index
            extracted = re.split(self._index_dir_regular_expression_split, index_dir)
            assert len(extracted) in [2, 3]
            extracted = extracted[1]
            # get index only
            extracted = re.search(r'\d+', extracted).group()
            # append int index
            index = int(extracted)
            used_indices.append(index)
        
        logger.debug('{}: Got {} used indices.'.format(self, len(used_indices)))
        return used_indices
    
    
    def remove_index(self, index, force=False):
        logger.debug('{}: Removing index {}.'.format(self, index))
        
        ## get topmost dir with index
        base_dir_including_index = self._base_dir_including_index.format(index)

        ## remove directory
        logger.debug('{}: Removing value dir {}.'.format(self, base_dir_including_index))
        assert base_dir_including_index.startswith(self._base_dir) and len(base_dir_including_index) > len(self._base_dir)
        try:
            util.io.fs.remove_recursively(base_dir_including_index, force=force, exclude_dir=False)
        except FileNotFoundError:
            raise util.index_database.general.DatabaseIndexError(self, index)
            
