import abc

import numpy as np

import util.logging
logger = util.logging.logger



class Database:
    
    def __init__(self, value_reliable_decimal_places=18, tolerance_options=None):
        
        ## set value format
        value_reliable_decimal_places = int(value_reliable_decimal_places)
        assert value_reliable_decimal_places >= 0
        self.value_fmt = '%.{}f'.format(value_reliable_decimal_places)

        ## set tolerance options
        if tolerance_options is None:
            tolerance_options = {}
        self._tolerance_options = {}
        
        ## set relative option
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

        ## min absolute tolerance
        min_absolute_tolerance = 10**(-value_reliable_decimal_places)

        ## set absolute option
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
                logger.warn('The absolute tolerance {} is not support. Using smallest supported absolute tolerance {}.'.format(absolute, min_absolute_tolerance))
                absolute = np.asanyarray(absolute, dtype=np.float64)
                absolute[absolute < min_absolute_tolerance] = min_absolute_tolerance
        
        self._tolerance_options['absolute'] = absolute
        
        ## check both options
        if not (len(self._tolerance_options['absolute']) == 1 or len(self._tolerance_options['relative']) == 1 or len(self._tolerance_options['relative']) == len(self._tolerance_options['absolute'])):
            raise ValueError('The relative and absolute tolerances habe to be scalaras or arrays of equal length, but the relative tolerance is {} and the absolute is {}.'.format(self._tolerance_options['relative'], self._tolerance_options['absolute']))
        
        logger.debug('Index database initiated with {} value format and tolerance options {}.'.format(self.value_fmt, self._tolerance_options))
        
    
    ## tolerances
    
    @property
    def relative_tolerance(self):
        return self._tolerance_options['relative']  
    
    @property
    def absolute_tolerance(self):
        return self._tolerance_options['absolute']
    

    ## value comparison
    
    def value_difference(self, v1, v2):
        ## check input
        if len(v1) != len(v2):
            raise ValueError('Both values must have equal lengths, but length of {} is {} and length of {} is {}.'.format(v1, len(v1), v2, len(v2)))
        if not len(self.relative_tolerance) in (1, len(v1)):
            raise ValueError('The relative tolerances must be a scalar or of equal length as the values, but the relative tolerance is {} with length {} and the values have length {}.'.format(self.relative_tolerance, len(self.relative_tolerance), len(v1)))
        if not len(self.relative_tolerance) in (1, len(v1)):
            raise ValueError('The absolute tolerances must be a scalar or of equal length as the values, but the absolute tolerance is {} with length {} and the values have length {}.'.format(self.absolute_tolerance, len(self.absolute_tolerance), len(v1)))
            
        
        ## calculate value weights
        relative_weights = np.minimum(np.abs(v1), np.abs(v2))

        assert len(self.relative_tolerance) in (1, len(v1))
        assert len(self.absolute_tolerance) in (1, len(v1))
        total_weights = np.maximum(relative_weights * self.relative_tolerance, self.absolute_tolerance)
        assert np.all(total_weights > 0)
        
        ## calculate max difference
        value_differences = np.abs(v1 - v2) / total_weights
        value_difference = value_differences.max()

        return value_difference

    
    def are_values_equal(self, v1, v2):
        return self.value_difference(v1, v2) <= 1
    
    
    ## access to values

    @abc.abstractmethod
    def get_value(self, index):
        raise NotImplementedError()

    def has_value(self, index):
        try:
            value = self.get_value(index)
        except DatabaseIndexError:
            has_value = False
        else:
            has_value = True

        logger.debug('{}: Has value at index {}: {}.'.format(self, index, has_value))
        return has_value

    @abc.abstractmethod
    def set_value(self, index, value, overwrite=True):
        raise NotImplementedError()

    def add_value(self, value):
        logger.debug('{}: Adding value {}'.format(self, value))
        
        ## get used indices
        used_indices = self.used_indices()
        
        ## create value
        index = 0
        created = False
        while not created:
            if not index in used_indices:
                try:
                    self.set_value(index, value, overwrite=False)
                except DatabaseOverwriteError:
                    index += 1
                else:
                    created = True
            else:
                index += 1
        
        ## return index
        logger.debug('{}: Value {} added with index {}.'.format(self, value, index))
        return index

    def all_values(self):
        np.vstack(map(self.get_value, self.used_indices))

    
    ## access to indices
    
    @abc.abstractmethod
    def used_indices(self):
        raise NotImplementedError()

    def number_of_used_indices(self):
        return len(self.used_indices())  

    @abc.abstractmethod
    def remove_index(self, index):
        raise NotImplementedError()


    def closest_indices(self, value):
        logger.debug('{}: Calculating closest indices for value {}.'.format(self, value))
        value = np.asanyarray(value)
        
        ## get all used indices
        used_indices = self.used_indices()
        used_indices = np.asarray(used_indices)
        
        ## init value differences
        n = len(used_indices)
        value_differences = np.empty(n)

        ## calculate value differences
        for i in range(n):
            current_index = used_indices[i]
            try:
                current_value = self.get_value(current_index)
            except DatabaseIndexError as e:
                logger.warnig('{}: Could not read the value file for index {}: {}'.format(self, current_index, e.with_traceback(None)))
                value_differences[i] = float('inf')
            else:
                value_differences[i] = self.value_difference(value, current_value)
        
        ## return sorted indices
        sort = np.argsort(value_differences)
        return used_indices[sort]


    def closest_index(self, value):
        logger.debug('{}: Searching for index of value as close as possible to {}.'.format(self, value))
        
        ## get closest indices
        closest_indices = self.closest_indices(value)
        
        ## return
        if len(closest_indices) > 0:
            logger.debug('{}: Closest index is {}.'.format(self, closest_indices[0]))
            return closest_indices[0]
        else:
            logger.debug('{}: No closest index found.'.format(self))
            return None


    def index(self, value):
        ## search for directories with matching parameters
        logger.debug('{}: Searching for index of value {}.'.format(self, value))

        closest_index = self.closest_index(value)
        if closest_index is not None and self.are_values_equal(value, self.get_value(closest_index)):
            logger.debug('{}: Index for value {} is {}.'.format(self, value, closest_index))
            return closest_index
        else:
            logger.debug('{}: No index found for value {}.'.format(self, value))
            return None


    def get_or_add_index(self, value, add=True):
        index = self.index(value)
        if index is None and add:
            index = self.add_value(value)
        return index




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
    
        