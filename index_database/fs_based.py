import os.path
import re

import numpy as np

import util.io.fs
import util.index_database.general

import util.logging
logger = util.logging.logger



class Database(util.index_database.general.Database):
    
    def __init__(self, value_file, value_reliable_decimal_places=15, tolerance_options=None):
        ## call super constructor
        super().__init__(value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)

        ## set value file informations
        for s in ['{', '}']:
            if len(value_file.split(s)) != 2:
                raise ValueError('The value filename must contain exactly one "{}". But the filename is {}.'.format(s, value_file))
        
        value_file = os.path.realpath(value_file)
        self.value_file = value_file
        self.base_dir = os.path.dirname(value_file.split('{')[0])
        assert len(self.base_dir) > 0
            
        self.value_file_regular_expression_search = re.sub('\{.*\}','[0-9]*', self.value_file)
        self.value_file_regular_expression_split = re.sub('\{.*\}','|', self.value_file)
        
    
    def __str__(self):
        return 'Index file system database {}'.format(self.base_dir)
    
    
    ## value dir
    
    def value_dir(self, index):
        value_file = self.value_file.format(index)
        value_dir = os.path.dirname(value_file)
        return value_dir


    
    ## access
    
    def get_value(self, index):
        value_file = self.value_file.format(index)
        try:
            return np.loadtxt(value_file)
        except FileNotFoundError:
            raise util.index_database.general.DatabaseIndexError(index)


    def set_value(self, index, value, overwrite=False):
        logger.debug('{}: Setting value at index {} to {} with overwrite {}.'.format(self, index, value, overwrite))

        ## make value dir
        value_dir = self.value_dir(index)
        os.makedirs(value_dir, exist_ok=True)

        ## make value file
        value_file = self.value_file.format(index)
        value_file_exists = os.path.exists(value_file)
        if value_file_exists and overwrite:
            util.io.fs.make_writable(value_file)
        if overwrite or not value_file_exists:
            np.savetxt(value_file, value, fmt=self.value_fmt)
            util.io.fs.make_read_only(value_file)
        else:
            logger.debug('{}: Overwritting value at index {} is not allowed.'.format(self, index))
            raise util.index_database.general.DatabaseOverwriteError(index)
    
    
    # def add_value(self, value):
    #     logger.debug('Adding value {} to {}.'.format(value, self))
    #     
    #     ## get used indices
    #     used_indices = self.used_indices()
    #     
    #     ## create value dir
    #     index = 0
    #     created = False
    #     while not created:
    #         if not index in used_indices:
    #             value_file = self.value_file.format(index)
    #             value_dir = self.value_dir(index)
    #             try:
    #                 os.makedirs(value_dir, exist_ok=False)
    #                 created = True
    #             except FileExistsError:
    #                 index += 1
    #         else:
    #             index += 1

    #       ## create value
    #     np.savetxt(value_file, value, fmt=self.value_fmt)
    #     util.io.fs.make_read_only(value_file)
    #     
    #     ## return index
    #     logger.debug('Value {} added with index {}.'.format(value, index))
    #     return index
    

    def used_indices(self):
        ## get all value files
        def condition(file):
            return re.match(self.value_file_regular_expression_search, file) is not None
        all_value_files = util.io.fs.filter_files(self.base_dir, condition, recursive=True)
        
        ## get all used indices
        used_indices = []
        for value_file in all_value_files:
            split_result = re.split(self.value_file_regular_expression_split, value_file)
            assert len(split_result) == 3
            index = int(split_result[1])
            used_indices.append(index)
        
        logger.debug('{}: Got {} used indices.'.format(self, len(used_indices)))
        return used_indices
    
    
    def remove_index(self, index, force=False):
        logger.debug('{}: Removing index {}.'.format(self, index))
        value_dir = self.value_file
        while re.search('\{.*\}', os.path.dirname(value_dir)):
            value_dir = os.path.dirname(value_dir)
        value_dir = value_dir.format(index)

        logger.debug('{}: Removing value dir {}.'.format(self, value_dir))
        assert value_dir.startswith(self.base_dir) and len(value_dir) > len(self.base_dir)
        try:
            util.io.fs.remove_recursively(value_dir, force=force, exclude_dir=False)
        except FileNotFoundError:
            raise util.index_database.general.DatabaseIndexError(index)
            


