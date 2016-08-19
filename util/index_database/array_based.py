import numpy as np

import util.index_database.general
import util.io.filelock.np

import util.logging
logger = util.logging.logger



class Database(util.index_database.general.Database):
    
    def __init__(self, array_file, value_reliable_decimal_places=15, tolerance_options=None):
        ## call super constructor
        super().__init__(value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)

        ## set array file
        self.locked_file = util.io.filelock.np.LockedFile(array_file)
    
    
    def __str__(self):
        return 'Index array database {}'.format(self.locked_file.file)
    
    
    ## access
    
    def get_value(self, index):
        ## load database array
        try:
            db = self.locked_file.load()
        except FileNotFoundError:
            raise util.index_database.general.DatabaseIndexError(index)
        ## get value at index
        try:
            value = db[index]
        except IndexError:
            raise util.index_database.general.DatabaseIndexError(index)
        ## check if valid value is at index 
        if np.all(np.isnan(value)):
            raise util.index_database.general.DatabaseIndexError(index)
        else:
            return value
    
    
    def set_value(self, index, value, overwrite=False):
        logger.debug('{}: Setting value at index {} to {} with overwrite {}.'.format(self, index, value, overwrite))
        value = np.asarray(value)
        
        with self.locked_file.lock_object(exclusive=True):
            try:
                db = self.locked_file.load()
            except FileNotFoundError:
                db = value[np.newaxis]
            else:
                if index < len(db):
                    has_value = self.has_value(index)
                    if overwrite or not has_value:
                        db[index] = value
                    else:
                        logger.debug('{}: Overwritting value at index {} is not allowed.'.format(self, index))
                        raise util.index_database.general.DatabaseOverwriteError(index)
                else:
                    db_extension_len = index - len(db) + 1
                    db_extension = np.empty([db_extension_len, db.shape[1]]) * np.nan
                    db = np.concatenate([db, db_extension])
                    db[index] = value
            
            self.locked_file.save(db)


    def add_value(self, value):
        with self.locked_file.lock_object(exclusive=True):
            return super().add_value(value)


    def used_indices(self):
        try:
            db = self.locked_file.load()
        except FileNotFoundError:
            return ()
        else:
            used_mask = np.all(np.logical_not(np.isnan(db)), axis=1)
            used_indices = np.where(used_mask)[0]
            
            logger.debug('{}: Got {} used indices.'.format(self, len(used_indices)))
            return used_indices.astype(np.int32)
    
    
    def remove_index(self, index):
        logger.debug('{}: Removing index {}.'.format(self, index))
        
        with self.locked_file.lock_object(exclusive=True):
            if not self.has_value(index):
                raise util.index_database.general.DatabaseIndexError(index)
            
            db = self.locked_file.load()
            db[index] = db[index] * np.nan
            
            while np.all(np.isnan(db[-1])):
                db = db[:-1]

            self.locked_file.save(db)
    

    def closest_indices(self, value):
        with self.locked_file.lock_object(exclusive=False):
            return super().closest_indices(value)


    def index(self, value):
        with self.locked_file.lock_object(exclusive=False):
            return super().index(value)



    # def add_value(self, value):
    #     logger.debug('Adding value {} to {}.'.format(value, self))
    #     
    #     with self.locked_file.lock_object(exclusive=True):
    #         db = self.locked_file.load()
    #     
    #         ## get used indices
    #         used_indices = self.used_indices()
    #         
    #         ## create value dir
    #         index = 0
    #         created = False
    #         while not created:
    #             if not index in used_indices:
    #                 if index < len(db):
    #                     db[index] = value
    #                 else:
    #                     assert index == len(db)
    #                     db = np.concatenate(db, value[np.newaxis])
    #             else:
    #                 index += 1
    # 
    #         ## create value
    #         self.locked_file.save(db)
    #     
    #     ## return index
    #     logger.debug('Value {} added with index {}.'.format(value, index))
    #     return index

