import os

import numpy as np

import util.index_database.general
import util.io.filelock.np
import util.logging


class Database(util.index_database.general.Database):

    def __init__(self, array_file, value_reliable_decimal_places=None, tolerance_options=None):
        super().__init__(value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
        self.array_file = array_file

    def __str__(self):
        return 'Index array database {}'.format(self.locked_file.file)

    # *** setter and getter for file *** #

    @property
    def array_file(self):
        return self.locked_file.file

    @array_file.setter
    def array_file(self, array_file):
        # create dir if not existing
        os.makedirs(os.path.dirname(array_file), exist_ok=True)

        # set array file
        self.locked_file = util.io.filelock.np.LockedArray(array_file)

    # *** access to values *** #

    def get_value(self, index):
        # load database array
        try:
            db = self.locked_file.load()
        except FileNotFoundError:
            raise util.index_database.general.DatabaseIndexError(self, index)
        # get value at index
        try:
            value = db[index]
        except IndexError:
            raise util.index_database.general.DatabaseIndexError(self, index)
        # check if valid value is at index
        if np.all(np.isnan(value)):
            raise util.index_database.general.DatabaseIndexError(self, index)
        else:
            return value

    def set_value(self, index, value, overwrite=False):
        util.logging.debug('{}: Setting value at index {} to {} with overwrite {}.'.format(self, index, value, overwrite))
        value = np.asanyarray(value)

        with self.locked_file.lock(exclusive=True):
            try:
                db = self.locked_file.load()
            except FileNotFoundError:
                db = value[np.newaxis]
            else:
                db_extension_len = index - len(db) + 1
                # index already in array
                if db_extension_len <= 0:
                    if overwrite or not self.has_value(index):
                        db[index] = value
                    elif not np.all(self.get_value(index) == value):
                        util.logging.debug('{}: Overwritting value at index {} is not allowed.'.format(self, index))
                        raise util.index_database.general.DatabaseOverwriteError(self, index)
                # index not in array
                else:
                    db_extension = np.empty([db_extension_len, db.shape[1]]) * np.nan
                    db = np.concatenate([db, db_extension])
                    db[index] = value
            # save changed array
            self.locked_file.save(db)

    def add_value(self, value):
        with self.locked_file.lock(exclusive=True):
            return super().add_value(value)

    # *** access to indices *** #

    def remove_index(self, index):
        util.logging.debug('{}: Removing index {}.'.format(self, index))

        with self.locked_file.lock(exclusive=True):
            if not self.has_value(index):
                raise util.index_database.general.DatabaseIndexError(self, index)

            db = self.locked_file.load()
            db[index] = db[index] * np.nan

            while len(db) > 0 and np.all(np.isnan(db[-1])):
                db = db[:-1]

            self.locked_file.save(db)

    def closest_index(self, value):
        with self.locked_file.lock(exclusive=False):
            return super().closest_index(value)

    def index(self, value):
        with self.locked_file.lock(exclusive=False):
            return super().index(value)

    # *** all values and indices *** #

    def all_indices(self):
        all_indices = self.all_indices_and_values()[0]
        util.logging.debug('{}: Got {} used indices.'.format(self, len(all_indices)))
        return all_indices

    def all_values(self):
        all_values = self.all_indices_and_values()[1]
        util.logging.debug('{}: Got {} used values.'.format(self, len(all_values)))
        return all_values

    def all_indices_and_values(self):
        try:
            db = self.locked_file.load()
        except FileNotFoundError:
            return ((), ())
        else:
            used_mask = np.all(np.logical_not(np.isnan(db)), axis=1)
            used_indices = np.where(used_mask)[0]
            used_indices = used_indices.astype(np.int32)
            used_values = db[used_mask]
            util.logging.debug('{}: Got {} used indices and values.'.format(self, len(used_indices)))
            return (used_indices, used_values)
