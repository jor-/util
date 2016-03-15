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
        
        try:
            tolerance_options['relative']
        except KeyError:
            tolerance_options['relative'] = np.asarray([0])
        else:
            if tolerance_options['relative'] is None:
                tolerance_options['relative'] = np.asarray([0])
            else:
                tolerance_options['relative'] = np.asarray(tolerance_options['relative']).reshape(-1)
                if np.any(tolerance_options['relative'] < 0):
                    raise ValueError('The relative tolerance {} has to be positive.'.format(tolerance_options['relative']))

        min_absolute_tolerance = 10**(-value_reliable_decimal_places)
        try:
            tolerance_options['absolute']
        except KeyError:
            tolerance_options['absolute'] = np.asarray([min_absolute_tolerance])
        else:
            if tolerance_options['absolute'] is None:
                tolerance_options['absolute'] = np.asarray([min_absolute_tolerance])
            else:
                tolerance_options['absolute'] = np.asarray(tolerance_options['absolute']).reshape(-1)
                if np.any(tolerance_options['absolute'] < 0):
                    raise ValueError('The absolute tolerance {} has to be positive.'.format(tolerance_options['absolute']))
                elif np.any(tolerance_options['absolute'] < min_absolute_tolerance):
                    util.logging.warn('The absolute tolerance {} is not support. Using smallest supported absolute tolerance {}.'.format(tolerance_options['absolute'], min_absolute_tolerance))
                    tolerance_options['absolute'][tolerance_options['absolute'] < min_absolute_tolerance] = min_absolute_tolerance
        
        self._tolerance_options = tolerance_options

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
    
    
    
    ## access

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


    @abc.abstractmethod
    def used_indices(self):
        raise NotImplementedError()    

    @abc.abstractmethod
    def remove_index(self, index):
        raise NotImplementedError()


    def closest_indices(self, value):
        logger.debug('{}: Calculating closest indices for value {}.'.format(self, value))
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



class DatabaseIndexError(IndexError):
    
    def __init__(self, index):
        message = 'Database has no value at index {}.'.format(index)
        super().__init__(message)


class DatabaseOverwriteError(Exception):    

    def __init__(self, index):
        message = 'Database has value at index {}. Overwrite is not permitted.'.format(index)
        super().__init__(message)
    
        