import util.index_database.general
import util.index_database.array_based
import util.index_database.fs_based

import util.logging
logger = util.logging.logger



class Database(util.index_database.general.Database):
    
    def __init__(self, array_file, value_file, value_reliable_decimal_places=18, tolerance_options=None):
        ## call super constructor
        super().__init__(value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
        ## create array and file db
        self.array_db = util.index_database.array_based.Database(array_file, value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
        self.file_db = util.index_database.fs_based.Database(value_file, value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
    
    
    def __str__(self):
        return 'Index array and file system database'

    
    ## access
    def get_value(self, index):
        return self.array_db.get_value(index)

    def has_value(self, index):
        return self.array_db.has_value(index)

    def set_value(self, index, value, overwrite=False):
        logger.debug('{}: Setting value at index {} to {} with overwrite {}.'.format(self, index, value, overwrite))
        self.file_db.set_value(index, value, overwrite=overwrite)
        self.array_db.set_value(index, value, overwrite=overwrite)

    def add_value(self, value):
        logger.debug('{}: Adding value {}'.format(self, value))
        with self.array_db.locked_file.lock_object(exclusive=True):
            index = self.array_db.add_value(value)
            try:
                self.file_db.set_value(index, value, overwrite=False)
            except util.index_database.general.DatabaseIndexError as e:
                self.array_db.remove_index(index)
                raise e
        return index
        
    

    def used_indices(self):
        return self.array_db.used_indices()
    
    def remove_index(self, index, force=False):
        logger.debug('{}: Removing index {}.'.format(self, index))
        self.file_db.remove_index(index, force=force)
        self.array_db.remove_index(index)
    
    def closest_indices(self, value):
        return self.array_db.closest_indices(value)

    def closest_index(self, value):
        return self.array_db.closest_index(value)

    def index(self, value):
        return self.array_db.index(value)


    ## merge
    
    def merge_file_db_to_array_db(self):
        logger.debug('{}: Merging file db to array db.'.format(self))
        for index in self.file_db.used_indices():
            file_db_value = self.file_db.get_value(index)
            if not self.array_db.has_value(index) or not self.are_values_equal(self.array_db.get_value(index), file_db_value):
                self.array_db.set_value(index, file_db_value, overwrite=False)
        
