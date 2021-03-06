import os.path

import numpy as np

import util.index_database.general
import util.index_database.array_based
import util.index_database.txt_file_based
import util.logging


class Database(util.index_database.general.Database):

    def __init__(self, array_file, value_file, value_reliable_decimal_places=None, tolerance_options=None):
        super().__init__(value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
        self.array_db = util.index_database.array_based.Database(array_file, value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
        value_dir, value_filename = os.path.split(value_file)
        self.txt_file_db = util.index_database.txt_file_based.Database(value_dir, value_filename, value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)

    def __str__(self):
        return '{} and {}'.format(self.array_db, self.txt_file_db)

    # *** setter and getter for files *** #

    @property
    def array_file(self):
        return self.array_db.array_file

    @array_file.setter
    def array_file(self, array_file):
        self.array_db.array_file = array_file

    @property
    def value_file(self):
        return os.path.join(self.txt_file_db.value_dir, self.txt_file_db.value_filenames[0])

    @value_file.setter
    def value_file(self, value_file):
        value_dir, value_filename = os.path.split(value_file)
        if value_dir != self.txt_file_db.value_dir:
            self.txt_file_db.value_dir = value_dir
        if value_filename != self.txt_file_db.value_filenames[0]:
            self.txt_file_db.value_filenames[0] = value_filename

    # *** access to values *** #

    def get_value(self, index):
        return self.array_db.get_value(index)

    def has_value(self, index):
        return self.array_db.has_value(index)

    def set_value(self, index, value, overwrite=False):
        util.logging.debug('{}: Setting value at index {} to {} with overwrite {}.'.format(self, index, value, overwrite))
        self.txt_file_db.set_value(index, value, overwrite=overwrite)
        self.array_db.set_value(index, value, overwrite=overwrite)

    def add_value(self, value):
        util.logging.debug('{}: Adding value {}'.format(self, value))
        with self.array_db.locked_file.lock(exclusive=True):
            index = self.array_db.add_value(value)
            try:
                self.txt_file_db.set_value(index, value, overwrite=False)
            except util.index_database.general.DatabaseIndexError as e:
                self.array_db.remove_index(index)
                raise e
        return index

    # *** access to indices *** #

    def remove_index(self, index, force=False):
        util.logging.debug('{}: Removing index {}.'.format(self, index))
        self.txt_file_db.remove_index(index, force=force)
        self.array_db.remove_index(index)

    def closest_index(self, value):
        return self.array_db.closest_index(value)

    def index(self, value):
        return self.array_db.index(value)

    # *** all values and indices *** #

    def all_indices(self):
        return self.array_db.all_indices()

    def all_values(self):
        return self.array_db.all_values()

    # *** merge *** #

    def merge_txt_file_db_to_array_db(self):
        util.logging.debug('{}: Merging file db to array db.'.format(self))
        for index in self.txt_file_db.all_indices():
            txt_file_db_value = self.txt_file_db.get_value(index)
            self.array_db.set_value(index, txt_file_db_value, overwrite=False)

    # *** check integrity *** #

    def check_integrity(self):
        # check if differenet indices
        util.logging.debug('{}: Checking for missing values in array and file database.'.format(self))

        array_db = self.array_db
        file_db = self.txt_file_db

        file_all_indices = file_db.all_indices()
        array_all_indices = array_db.all_indices()
        file_all_indices = set(file_all_indices)
        array_all_indices = set(array_all_indices)

        diff = array_all_indices - file_all_indices
        if len(diff) > 0:
            raise util.index_database.general.DatabaseError(self, 'Array db has values at indices {} and file db has no values their!'.format(diff))
        diff = file_all_indices - array_all_indices
        if len(diff) > 0:
            raise util.index_database.general.DatabaseError(self, 'File db has values at indices {} and array db has no values their!'.format(diff))

        # check if different values in array and text database
        util.logging.debug('{}: Checking for different values in array and file database.'.format(self))

        for index in array_all_indices:
            v_a = array_db.get_value(index)
            v_f = file_db.get_value(index)
            diff = np.max(np.abs(v_a - v_f))
            if diff > self._min_absolute_tolerance:
                raise util.index_database.general.DatabaseError(self, 'Array db and file db value at index {} are not equal: {} != {}!'.format(index, v_a, v_f))

        # check if same values multiple times in (array) database
        array_db.check_integrity()
