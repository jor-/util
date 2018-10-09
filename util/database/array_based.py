import os

import numpy as np
import overrides

import util.database.general
import util.io.filelock.np
import util.logging


class Database(util.database.general.Database):

    def __init__(self, array_file, key_length=None, value_length=None, tolerance_options=None):
        super().__init__(tolerance_options=tolerance_options)
        self.array_file = array_file
        self._key_length = key_length
        self._value_length = value_length
        if key_length is None and value_length is None:
            raise ValueError('key_length or value_length has to be specified but both are None.')

    def __str__(self):
        return 'Array database {}'.format(self.locked_file.file)

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

    # *** lengths values *** #

    def key_length(self, n):
        if self._key_length is not None:
            return self._key_length
        else:
            return n - self._value_length

    def value_length(self, n):
        if self._value_length is not None:
            return self._value_length
        else:
            return n - self._key_length

    # *** access to keys *** #

    @overrides.overrides
    def get_keys(self):
        util.logging.debug(f'{self}: Getting all keys.')
        try:
            db_array = self.locked_file.load()
        except FileNotFoundError:
            keys = ()
        else:
            key_length = self.key_length(db_array.shape[1])
            keys = db_array[:, :key_length]

        util.logging.debug(f'{self}: Got {len(keys)} keys.')
        return keys

    @overrides.overrides
    def get_key_with_index(self, index):
        with self.locked_file.lock(exclusive=False):
            return super().get_key_with_index(index)

    @overrides.overrides
    def set_key_with_index(self, index, key, overwrite=False):
        util.logging.debug(f'{self}: Setting key at index {index} to {key} with overwrite {overwrite}.')
        key = np.asanyarray(key)

        with self.locked_file.lock(exclusive=True):
            try:
                db_array = self.locked_file.load()
            except FileNotFoundError as e:
                raise util.database.general.DatabaseIndexError(self, index) from e
            else:
                if len(db_array) <= index:
                    raise util.database.general.DatabaseIndexError(self, index)
                elif overwrite:
                    key_length = self.key_length(db_array.shape[1])
                    db_array[index, :key_length] = key
                    self.locked_file.save(db_array)
                elif not np.all(self.get_key_with_index(index) == key):
                    raise util.database.general.DatabaseOverwriteIndexError(self, index)

    @overrides.overrides
    def contains_key(self, key, use_tolerances=True):
        with self.locked_file.lock(exclusive=False):
            return super().contains_key(key, use_tolerances=use_tolerances)

    # *** access to indices *** #

    @overrides.overrides
    def _get_closest_index_and_matches_with_key(self, key):
        with self.locked_file.lock(exclusive=False):
            return super()._get_closest_index_and_matches_with_key(key)

    @overrides.overrides
    def get_closest_index_with_key(self, key):
        with self.locked_file.lock(exclusive=False):
            return super().get_closest_index_with_key(key)

    @overrides.overrides
    def get_index_with_key(self, key, use_tolerances=True):
        with self.locked_file.lock(exclusive=False):
            return super().get_index_with_key(key, use_tolerances=use_tolerances)

    @overrides.overrides
    def contains_index(self, index):
        with self.locked_file.lock(exclusive=False):
            return super().contains_index(index)

    # *** get values *** #

    @overrides.overrides
    def get_values(self):
        util.logging.debug(f'{self}: Getting all values.')
        try:
            db_array = self.locked_file.load()
        except FileNotFoundError:
            values = ()
        else:
            key_length = self.key_length(db_array.shape[1])
            values = db_array[:, key_length:]

        util.logging.debug(f'{self}: Got {len(values)} values.')
        return values

    @overrides.overrides
    def get_value_with_index(self, index):
        with self.locked_file.lock(exclusive=False):
            return super().get_value_with_index(index)

    @overrides.overrides
    def get_value_with_key(self, key, use_tolerances=True):
        with self.locked_file.lock(exclusive=False):
            return super().get_value_with_key(key, use_tolerances=use_tolerances)

    # *** set values *** #

    @overrides.overrides
    def set_value_with_index(self, index, value, overwrite=False):
        util.logging.debug(f'{self}: Setting value at index {index} to {value} with overwrite {overwrite}.')
        value = np.asanyarray(value)

        with self.locked_file.lock(exclusive=True):
            try:
                db_array = self.locked_file.load()
            except FileNotFoundError as e:
                raise util.database.general.DatabaseIndexError(self, index) from e
            else:
                if len(db_array) <= index:
                    raise util.database.general.DatabaseIndexError(self, index)
                elif overwrite:
                    key_length = self.key_length(db_array.shape[1])
                    db_array[index, key_length:] = value
                    self.locked_file.save(db_array)
                elif not np.all(self.get_value_with_index(index) == value):
                    raise util.database.general.DatabaseOverwriteIndexError(self, index)

    @overrides.overrides
    def set_value_with_key(self, key, value, use_tolerances=False, overwrite=False):
        with self.locked_file.lock(exclusive=True):
            return super().set_value_with_key(key, value, use_tolerances=use_tolerances, overwrite=overwrite)

    @overrides.overrides
    def get_or_set_value(self, key, value, use_tolerances=True):
        with self.locked_file.lock(exclusive=True):
            return super().get_or_set_value(key, value, use_tolerances=use_tolerances)

    @overrides.overrides
    def append_key_and_value(self, key, value):
        util.logging.debug(f'{self}: Appending value {value} with key {key}.')

        key = np.asanyarray(key)
        value = np.asanyarray(value)
        item = np.concatenate([key.flat, value.flat])[np.newaxis]

        with self.locked_file.lock(exclusive=True):
            try:
                db_array = self.locked_file.load()
            except FileNotFoundError:
                db_array = item
            else:
                db_array = np.concatenate([db_array, item])
            self.locked_file.save(db_array)

    # *** del values *** #

    @overrides.overrides
    def del_key_and_value_with_index(self, index):
        util.logging.debug(f'{self}: Deleting key and value at index {index}.')

        with self.locked_file.lock(exclusive=True):
            try:
                db_array = self.locked_file.load()
            except FileNotFoundError as e:
                raise util.database.general.DatabaseIndexError(self, index) from e
            else:
                if len(db_array) <= index:
                    raise util.database.general.DatabaseIndexError(self, index)
                elif len(db_array) == index + 1:
                    db_array = db_array[:index]
                else:
                    db_array = np.concatenate([db_array[:index], db_array[index + 1:]])
                self.locked_file.save(db_array)

    @overrides.overrides
    def del_key_and_value_with_key(self, key, use_tolerances=True):
        with self.locked_file.lock(exclusive=True):
            return super().del_value_with_key(key, use_tolerances=use_tolerances)
