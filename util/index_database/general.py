import abc

import numpy as np

import util.logging


class Database:

    def __init__(self, value_reliable_decimal_places=None, tolerance_options=None):
        # set value format
        if value_reliable_decimal_places is None:
            value_reliable_decimal_places = np.finfo(np.float64).precision
        value_reliable_decimal_places = int(value_reliable_decimal_places)
        assert value_reliable_decimal_places >= 0
        self.value_fmt = '%.{}f'.format(value_reliable_decimal_places)

        # set tolerance options
        if tolerance_options is None:
            tolerance_options = {}
        self._tolerance_options = {}

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
                raise ValueError('The relative tolerance {} has to be positive.'.format(relative))

        self._tolerance_options['relative'] = relative

        # min absolute tolerance
        min_absolute_tolerance = 10**(-value_reliable_decimal_places)
        self._min_absolute_tolerance = min_absolute_tolerance

        # set absolute option
        try:
            absolute = tolerance_options['absolute']
        except KeyError:
            absolute = None

        if absolute is None:
            absolute = np.asarray([min_absolute_tolerance])
        else:
            absolute = np.asanyarray(absolute).reshape(-1)
            if np.any(absolute < 0):
                raise ValueError('The absolute tolerance {} has to be positive.'.format(absolute))
            elif np.any(absolute < min_absolute_tolerance):
                util.logging.warn('The absolute tolerance {} is not support. Using smallest supported absolute tolerance {}.'.format(absolute, min_absolute_tolerance))
                absolute = np.asanyarray(absolute, dtype=np.float64)
                absolute[absolute < min_absolute_tolerance] = min_absolute_tolerance

        self._tolerance_options['absolute'] = absolute

        # check both options
        if not (len(self._tolerance_options['absolute']) == 1 or len(self._tolerance_options['relative']) == 1 or len(self._tolerance_options['relative']) == len(self._tolerance_options['absolute'])):
            raise ValueError('The relative and absolute tolerances habe to be scalaras or arrays of equal length, but the relative tolerance is {} and the absolute is {}.'.format(self._tolerance_options['relative'], self._tolerance_options['absolute']))

        util.logging.debug('Index database initiated with {} value format and tolerance options {}.'.format(self.value_fmt, self._tolerance_options))

    # *** tolerances *** #

    @property
    def relative_tolerance(self):
        return self._tolerance_options['relative']

    @property
    def absolute_tolerance(self):
        return self._tolerance_options['absolute']

    # *** access to values *** #

    @abc.abstractmethod
    def get_value(self, index):
        raise NotImplementedError()

    def has_value(self, index):
        try:
            self.get_value(index)
        except DatabaseIndexError:
            has_value = False
        else:
            has_value = True

        util.logging.debug('{}: Has value at index {}: {}.'.format(self, index, has_value))
        return has_value

    @abc.abstractmethod
    def set_value(self, index, value, overwrite=True):
        raise NotImplementedError()

    def add_value(self, value):
        util.logging.debug('{}: Adding value {}'.format(self, value))

        # checking value
        value = np.asanyarray(value)
        if not np.all(np.isfinite(value)):
            raise ValueError('Value must be finite. But it is {}.'.format(value))

        # get used indices
        all_indices = self.all_indices()

        # create value
        index = 0
        created = False
        while not created:
            if index not in all_indices:
                try:
                    self.set_value(index, value, overwrite=False)
                except DatabaseOverwriteError:
                    index += 1
                else:
                    created = True
            else:
                index += 1

        # return index
        util.logging.debug('{}: Value {} added with index {}.'.format(self, value, index))
        return index

    # *** access to indices *** #

    def number_of_indices(self):
        return len(self.all_indices())

    @abc.abstractmethod
    def remove_index(self, index):
        raise NotImplementedError()

    def _get_closest_index_and_matches(self, value):

        def value_difference(self, v1, v2):
            # check input
            if len(v1) != len(v2):
                raise ValueError('Both values must have equal lengths, but length of {} is {} and length of {} is {}.'.format(v1, len(v1), v2, len(v2)))
            if not len(self.relative_tolerance) in (1, len(v1)):
                raise ValueError('The relative tolerances must be a scalar or of equal length as the values, but the relative tolerance is {} with length {} and the values have length {}.'.format(self.relative_tolerance, len(self.relative_tolerance), len(v1)))
            if not len(self.relative_tolerance) in (1, len(v1)):
                raise ValueError('The absolute tolerances must be a scalar or of equal length as the values, but the absolute tolerance is {} with length {} and the values have length {}.'.format(self.absolute_tolerance, len(self.absolute_tolerance), len(v1)))

            # calculate value weights
            relative_weights = np.minimum(np.abs(v1), np.abs(v2))

            assert len(self.relative_tolerance) in (1, len(v1))
            assert len(self.absolute_tolerance) in (1, len(v1))
            total_weights = np.maximum(relative_weights * self.relative_tolerance, self.absolute_tolerance)
            assert np.all(total_weights > 0)

            # calculate max difference
            value_differences = np.abs(v1 - v2) / total_weights
            value_difference = value_differences.max()

            return value_difference

        util.logging.debug('{}: Searching for index of value as close as possible to {}.'.format(self, value))
        value = np.asanyarray(value)

        # get all used indices
        all_indices = self.all_indices()
        all_indices = np.asarray(all_indices)

        # init value differences
        n = len(all_indices)
        value_differences = np.empty(n)

        # calculate value differences
        for i in range(n):
            current_index = all_indices[i]
            try:
                current_value = self.get_value(current_index)
            except DatabaseIndexError as e:
                util.logging.warnig('{}: Could not read the value file for index {}: {}'.format(self, current_index, e.with_traceback(None)))
                value_differences[i] = float('inf')
            else:
                value_differences[i] = value_difference(self, value, current_value)

        # get closest index
        if n > 0:
            i = np.argmin(value_differences)
            closest_index = all_indices[i]
            value_difference = value_differences[i]
            matches = value_difference <= 1
            util.logging.debug('{}: Closest index is {}.'.format(self, closest_index))
        else:
            closest_index = None
            matches = None
            util.logging.debug('{}: No closest index found.'.format(self))

        return closest_index, matches

    def closest_index(self, value):
        return self._get_closest_index_and_matches(value)[0]

    def index(self, value):
        # search for directories with matching parameters
        util.logging.debug('{}: Searching for index of value {}.'.format(self, value))

        closest_index, matches = self._get_closest_index_and_matches(value)
        if closest_index is not None and matches:
            util.logging.debug('{}: Index for value {} is {}.'.format(self, value, closest_index))
            return closest_index
        else:
            util.logging.debug('{}: No index found for value {}.'.format(self, value))
            return None

    def get_or_add_index(self, value, add=True):
        index = self.index(value)
        if index is None and add:
            index = self.add_value(value)
        return index

    # *** all values and indices *** #

    @abc.abstractmethod
    def all_indices(self):
        raise NotImplementedError()

    def all_values(self):
        np.vstack(map(self.get_value, self.all_indices))

    def all_indices_and_values(self):
        return (self.all_indices(), self.all_values())

    # *** check integrity *** #

    def check_integrity(self):
        util.logging.debug('{}: Checking for same values mutiple times in database.'.format(self))
        values = self.all_values()
        unique_values, inverse_indices, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
        for bad_unique_index in np.where(counts > 1)[0]:
            bad_indices = np.where(inverse_indices == bad_unique_index)[0]
            assert len(bad_indices) > 1
            util.logging.error('{}: Indices {} have same value {}.'.format(self, bad_indices, values[bad_indices[0]]))


# *** exceptions *** #

class DatabaseError(Exception):
    def __init__(self, database, message):
        self.database = database
        message = '{}: {}'.format(database, message)
        super().__init__(message)


class DatabaseIndexError(DatabaseError, IndexError):
    def __init__(self, database, index):
        message = 'Database has no value at index {}.'.format(index)
        super().__init__(database, message)


class DatabaseOverwriteError(DatabaseError):
    def __init__(self, database, index):
        message = 'Database has value at index {}. Overwrite is not permitted.'.format(index)
        super().__init__(database, message)
