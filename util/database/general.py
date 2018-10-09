import abc
import collections.abc

import numpy as np

import util.logging


class Database(collections.abc.MutableMapping):

    def __init__(self, tolerance_options=None):
        # set tolerance options
        if tolerance_options is None:
            tolerance_options = {}
        self.tolerance_options = {}

        # set relative option
        try:
            relative = tolerance_options['relative']
        except KeyError:
            relative = None

        if relative is None:
            relative = np.array([0])
        else:
            relative = np.asanyarray(relative).reshape(-1)
            if np.any(relative < 0):
                raise ValueError(f'The relative tolerance {relative} has to be positive.')

        self.tolerance_options['relative'] = relative

        # set absolute option
        try:
            absolute = tolerance_options['absolute']
        except KeyError:
            absolute = None

        tolerance_dtype = np.float64
        min_absolute_tolerance = np.finfo(tolerance_dtype).eps
        if absolute is None:
            absolute = np.array([min_absolute_tolerance])
        else:
            absolute = np.asanyarray(absolute).reshape(-1)
            if np.any(absolute < 0):
                raise ValueError('The absolute tolerance {absolute} has to be positive.')
            elif np.any(absolute < min_absolute_tolerance):
                util.logging.warn('The absolute tolerance {absolute} is not support. Using smallest supported absolute tolerance {min_absolute_tolerance}.')
                absolute = np.asanyarray(absolute, dtype=tolerance_dtype)
                absolute[absolute < min_absolute_tolerance] = min_absolute_tolerance

        self.tolerance_options['absolute'] = absolute

        # check both options
        if not (len(self.tolerance_options['absolute']) == 1 or len(self.tolerance_options['relative']) == 1 or len(self.tolerance_options['relative']) == len(self.tolerance_options['absolute'])):
            raise ValueError('The relative and absolute tolerances habe to be scalaras or arrays of equal length, but the relative tolerance is {self.tolerance_options["relative"]} and the absolute is {self.tolerance_options["absolute"]}.')

        util.logging.debug('Index database initiated with tolerance options {self.tolerance_options}.')

    # *** tolerances *** #

    @property
    def relative_tolerance(self):
        return self.tolerance_options['relative']

    @property
    def absolute_tolerance(self):
        return self.tolerance_options['absolute']

    # *** access to keys *** #

    @abc.abstractmethod
    def get_keys(self):
        raise NotImplementedError()

    def get_key_with_index(self, index):
        keys = self.get_keys()
        if len(keys) == 0:
            raise DatabaseIndexError(self, index)
        try:
            keys = keys[index]
        except IndexError as e:
            raise DatabaseIndexError(self, index) from e
        else:
            return keys

    @abc.abstractmethod
    def set_key_with_index(self, index, key, overwrite=False):
        raise NotImplementedError()

    def contains_key(self, key, use_tolerances=True):
        try:
            self.get_index_with_key(key, use_tolerances=use_tolerances)
        except DatabaseKeyError:
            return False
        else:
            return True

    # *** access to indices *** #

    def _get_closest_index_and_matches_with_key(self, key):
        key = np.asanyarray(key)
        assert key.ndim == 1

        # get keys
        keys = self.get_keys()
        keys = np.asanyarray(keys)
        if len(keys) == 0:
            raise DatabaseEmptyError(self)
        assert keys.ndim == 2
        assert len(key) == keys.shape[1]

        # calculate value weights
        relative_weights = np.minimum(np.abs(keys), np.abs(key))
        assert len(self.relative_tolerance) in (1, len(key))
        assert len(self.absolute_tolerance) in (1, len(key))
        total_weights = np.maximum(relative_weights * self.relative_tolerance, self.absolute_tolerance)
        assert np.all(total_weights > 0)

        # calculate max difference
        value_differences = np.abs(keys - key) / total_weights
        value_differences = value_differences.max(axis=1)

        # get closest index
        closest_index = np.argmin(value_differences)
        matches = value_differences[closest_index] <= 1

        return closest_index, matches

    def get_closest_index_with_key(self, key):
        util.logging.debug(f'{self}: Calculating closest index for key {key}.')
        closest_index = self._get_closest_index_and_matches_with_key(key)[0]
        util.logging.debug(f'{self}: Closest index is {closest_index}.')
        return closest_index

    def get_index_with_key(self, key, use_tolerances=True):
        util.logging.debug(f'{self}: Getting index for key {key} with use_tolerances {use_tolerances}.')

        try:
            closest_index, matches = self._get_closest_index_and_matches_with_key(key)
        except DatabaseEmptyError as e:
            raise DatabaseKeyError(self, key) from e
        else:
            if not matches:
                raise DatabaseKeyError(self, key)
            if not use_tolerances:
                closest_key = self.get_key_with_index(closest_index)
                if not np.all(key == closest_key):
                    raise DatabaseKeyError(self, key)

        util.logging.debug(f'{self}: Index for key {key} with use_tolerances {use_tolerances} is {closest_index}.')
        return closest_index

    def contains_index(self, index):
        try:
            self.get_key_with_index(index)
        except DatabaseIndexError:
            return False
        else:
            return True

    # *** get values *** #

    @abc.abstractmethod
    def get_values(self):
        raise NotImplementedError()

    def get_value_with_index(self, index):
        values = self.get_values()
        if len(values) == 0:
            raise DatabaseIndexError(self, index)
        try:
            value = values[index]
        except IndexError as e:
            raise DatabaseIndexError(self, index) from e
        else:
            return value

    def get_value_with_key(self, key, use_tolerances=True):
        index = self.get_index_with_key(key, use_tolerances=use_tolerances)
        return self.get_value_with_index(index)

    # *** set values *** #

    @abc.abstractmethod
    def set_value_with_index(self, index, value, overwrite=False):
        raise NotImplementedError()

    def set_value_with_key(self, key, value, use_tolerances=False, overwrite=False):
        try:
            index = self.get_index_with_key(key, use_tolerances=use_tolerances)
        except DatabaseKeyError:
            self.append_key_and_value(key, value)
        else:
            self.set_value_with_index(index, value, overwrite=overwrite)

    def get_or_set_value(self, key, value, use_tolerances=True):
        try:
            value = self.get_value_with_key(key, use_tolerances=use_tolerances)
        except DatabaseKeyError:
            self.append_key_and_value(key, value)
        return value

    @abc.abstractmethod
    def append_key_and_value(self, key, value):
        raise NotImplementedError()

    # *** del values *** #

    @abc.abstractmethod
    def del_key_and_value_with_index(self, index):
        raise NotImplementedError()

    def del_key_and_value_with_key(self, key, use_tolerances=True):
        index = self.get_index_with_key(key, use_tolerances=use_tolerances)
        return self.del_key_and_value_with_index(index)

    # *** MutableMapping special methods *** #

    def keys(self):
        return self.get_keys()

    def values(self):
        return self.get_values()

    def setdefault(self, key, value):
        return self.get_or_set_value(key, value)

    def __contains__(self, key):
        return self.contains_key(key)

    def __len__(self):
        return len(self.get_keys())

    def __iter__(self):
        return iter(self.get_keys())

    def __getitem__(self, key):
        return self.get_value_with_key(key)

    def __delitem__(self, key):
        return self.del_key_and_value_with_key(key)

    def __setitem__(self, key, value):
        return self.set_value_with_key(key, value)

    # *** check integrity *** #

    def check_integrity(self):
        util.logging.debug(f'{self}: Checking for same keys mutiple times in database.')
        keys = self.get_keys()
        unique_keys, inverse_indices, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        for bad_unique_index in np.where(counts > 1)[0]:
            bad_indices = np.where(inverse_indices == bad_unique_index)[0]
            assert len(bad_indices) > 1
            util.logging.error('{}: Indices {} have same key {}.'.format(self, bad_indices, keys[bad_indices[0]]))


# *** exceptions *** #

class DatabaseError(Exception):
    def __init__(self, database, message):
        self.database = database
        message = f'{database}: {message}'
        super().__init__(message)


class DatabaseKeyError(DatabaseError, KeyError):
    def __init__(self, database, key):
        message = f'Database has no value with key {key}.'
        super().__init__(database, message)


class DatabaseIndexError(DatabaseError, IndexError):
    def __init__(self, database, index):
        message = f'Database has no value at index {index}.'
        super().__init__(database, message)


class DatabaseOverwriteError(DatabaseError):
    def __init__(self, database):
        message = f'Overwrite in database is not permitted.'
        super().__init__(database, message)


class DatabaseOverwriteKeyError(DatabaseError):
    def __init__(self, database, key):
        message = f'Database has value at key {key}. Overwrite is not permitted.'
        super().__init__(database, message)


class DatabaseOverwriteIndexError(DatabaseError):
    def __init__(self, database, index):
        message = f'Database has value at index {index}. Overwrite is not permitted.'
        super().__init__(database, message)


class DatabaseEmptyError(DatabaseError):
    def __init__(self, database):
        message = f'Database is empty.'
        super().__init__(database, message)
