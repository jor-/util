import collections.abc

import numpy as np
import scipy.interpolate

import util.io.object
import util.parallel.with_multiprocessing
import util.logging
logger = util.logging.logger



def get_float_index_for_equidistant_values(value, value_range, dim):
    ## check input
    if value < value_range[0] or value > value_range[1]:
        raise ValueError('Value {} has to be in range {}.'.format(value, value_range))

    ## compute index
    index = (value - value_range[0]) / (value_range[1] - value_range[0]) * (dim)
    if index == dim:
        index = index - 1

    return index


def get_nearest_value_in_array(array, value):
    logger.debug('Getting nearest value in array for {}.'.format(value))

    distance = np.sum((array - value)**2,axis=-1)
    index = np.argmin(distance)
    nearest_value = array[index]

    logger.debug('Nearest value in array for {} is {}.'.format(value, nearest_value))

    return nearest_value


## interpolation

def data_with_float_index(data, index):
    ## get data at float index (linear) interpolated
    if len(index) == 0:
        return data
    else:
        index = np.asanyarray(index)
        lower = np.floor(index[0])
        upper = np.ceil(index[0])
        fraction = index[0] % 1
        return (1 - fraction) * data_with_float_index(data[lower], index[1:]) + fraction * data_with_float_index(data[upper], index[1:])


def data_with_regular_grid(data, points, point_ranges):
    ## make arrays
    data = np.asanyarray(data)
    points = np.asanyarray(points)

    ## check input
    assert data.ndim >= points.shape[1]

    ## make result array
    result_shape = (points.shape[0],) + data.shape[points.shape[1]:]
    results = np.empty(result_shape, dtype=data.dtype)

    ## convert points to indices
    point_ranges = np.asanyarray(point_ranges)
    normalized_points = (points - point_ranges[:,0]) / (point_ranges[:,1] - point_ranges[:,0])
    indice_shape = np.array(data.shape[:points.shape[1]])
    indices = normalized_points * (indice_shape - 1)

    ## interpolate
    for i in range(len(indices)):
        results[i] = data_with_float_index(data, indices[i])

    return results



def wrap_around(values, index, value_range_len, amount=1, return_also_indices=False):
    logger.debug('Wrapping around index {} with an amount of {} and value range len {}.'.format(index, amount, value_range_len))

    if amount < 0 or amount > 1:
        raise ValueError('Amount has to be between 0 and 1, but its {}.'.format(amount))

    ## prepare values
    values_len = len(values)
    values_len_amount = int(np.round(values_len * amount))

    logger.debug('Adding two times {} data values.'.format(values_len_amount-1))

    if amount != 0 and amount != 1:
        value_indices = np.argsort(values[:,index])
    else:
        value_indices = np.arange(values_len)

    ## concatenate
    if amount != 0:
        value_indices = np.concatenate((value_indices[- values_len_amount:], value_indices, value_indices[:values_len_amount]), axis=0)

        values = values[value_indices]
        values[:values_len_amount, index] -= value_range_len
        values[- values_len_amount:, index] += value_range_len

    ## return
    assert values.ndim >= 1
    assert values.shape[0] == values_len + 2 * values_len_amount
    assert value_indices.ndim == 1
    assert len(value_indices) == len(values)

    if return_also_indices:
        return values, value_indices
    else:
        return values


##

def change_dim(data, dim_index, new_dim):
    old_dim = data.shape[dim_index]
    if not (old_dim % new_dim == 0 or new_dim % old_dim == 0):
        raise ValueError('The dim {} of the data at index {} has to be a factor of the new_dim {} or vice versa.'.format(old_dim, dim_index, new_dim))

    new_shape = data.shape[:dim_index] + (new_dim,) + data.shape[dim_index+1:]
    assert data.ndim == len(new_shape)

    new_data = np.empty(new_shape, dtype=data.dtype)
    if new_dim < old_dim:
        step = int(old_dim / new_dim)
        logger.debug('Averaging dim {} to new dim {} with step {}.'.format(old_dim, new_dim, step))

        index_list = [Ellipsis,] * data.ndim
        for i in range(new_dim):
            index_list[dim_index] = slice(i * step, (i+1) * step)
            old_index = tuple(index_list)
            index_list[dim_index] = i
            new_index = tuple(index_list)
            new_data[new_index] = data[old_index].mean(axis=dim_index)

    elif new_dim > old_dim:
        step = int(new_dim / old_dim)
        logger.debug('Linear interpolating dim {} to new dim {} with step {}.'.format(old_dim, new_dim, step))

        index_list = [Ellipsis,] * data.ndim
        for i in range(new_dim):
            old_i_left = np.floor(i / step)
            old_i_right = old_i_left + 1
            if old_i_right == old_dim:
                old_i_right = 0
            old_i_fract = i / step - old_i_left

            index_list[dim_index] = old_i_left
            old_index_left = tuple(index_list)
            index_list[dim_index] = old_i_right
            old_index_right = tuple(index_list)
            index_list[dim_index] = i
            new_index = tuple(index_list)

            new_data[new_index] = (1 - old_i_fract) * data[old_index_left] + old_i_fract * data[old_index_right]

    else:
        new_data = data

    return new_data



## Base Interpolator

class Interpolator_Base():

    def __init__(self, data_points, data_values, method, possible_methods, scaling_values=None, copy_arrays=True):
        logger.debug('Initiating base interpolator with {} data points with scaling values {}, method {} and possible methods {}.'.format(len(data_points), scaling_values, method, possible_methods))

        if scaling_values is not None:
            if isinstance(scaling_values, collections.abc.Iterable):
                scaling_values = tuple(v if v is not None else 1 for v in scaling_values)
                scaling_values = np.array(scaling_values)
            else:
                scaling_values = np.array([scaling_values])
            assert scaling_values.ndim == 1

        self._scaling_values = scaling_values
        self._copy_arrays = np.asanyarray(copy_arrays)
        self._data_points = self._prepare_data_points(data_points)
        self._possible_methods = possible_methods
        self.method = method
        self.data_values = data_values

        assert self._scaling_values is None or len(self._scaling_values) in (1, self._data_points.shape[1])


    @property
    def data_points(self):
        return self._data_points

    @property
    def data_values(self):
        return self._data_values

    @data_values.setter
    def data_values(self, data_values):
        self._set_data_values(data_values)

    def _set_data_values(self, data_values):
        self._data_values = self._prepare_data_values(data_values)
        logger.debug('{} data values set.'.format(len(self._data_values)))

    @property
    def possible_methods(self):
        return self._possible_methods

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        if method not in self.possible_methods:
            raise ValueError('Method has to be in {}, but its value is {}.'.format(self.possible_methods, method))
        self._method = method

    @property
    def scaling_values(self):
        return self._scaling_values

    def _copy(self, index):
        copy_arrays = self._copy_arrays
        if copy_arrays.ndim == 0:
            return copy_arrays[()]
        elif len(copy_arrays) == 1:
            return copy_arrays[0]
        else:
            return copy_arrays[index]


    ## prepare values

    def _prepare_data_values(self, data_values):
        logger.debug('Preparing data values.')

        data_values = np.ascontiguousarray(data_values, dtype=np.double)

        if len(self.data_points) != len(data_values):
            raise ValueError('The data points and values must have the same first dimension, but its {} and {}.'.format(self.data_points.shape[0], data_values.shape[0]))
        if data_values.ndim != 1:
            raise ValueError('The data values must have one dimension, but its dimension is {}.'.format(data_values.ndim))

        assert data_values.ndim == 1
        assert data_values.shape[0] == self.data_points.shape[0]

        return data_values


    def _modify_data_points(self, points):
        points = self._scale(points)
        return points


    def _prepare_data_points(self, data_points):
        if self._copy(0):
            logger.debug('Preparing data points with a copy.')
            data_points = np.array(data_points, copy=True)
        else:
            logger.debug('Preparing data points.')
            data_points = np.asanyarray(data_points)

        if data_points.ndim == 1:
            data_points = data_points[None,:]
        elif data_points.ndim >= 3:
            raise ValueError('The data points must have two dimension, but its dimension is {}.'.format(data_points.ndim))
        if data_points.shape[0] == 0:
            raise ValueError('No data points passed.')
        if data_points.shape[1] < 2:
            raise ValueError('The data points must be at least 2 dimensional.')
        if self.scaling_values is not None and not len(self.scaling_values) in (1, data_points.shape[1]):
            raise ValueError('The length of the scaling values ({}) must be the number of dimensions of the data points ({}).'.format(len(self.scaling_values), data_points.shape[1]))

        data_points = self._modify_data_points(data_points)

        assert data_points.ndim == 2

        return data_points


    def _modify_interpolation_points(self, points):
        points = self._scale(points)
        return points


    def _prepare_interpolation_points(self, interpolation_points):
        if self._copy(1):
            logger.debug('Preparing interpolation points with a copy.')
            interpolation_points = np.array(interpolation_points, copy=True)
        else:
            logger.debug('Preparing interpolation points.')
            interpolation_points = np.asanyarray(interpolation_points)

        if interpolation_points.ndim == 1:
            interpolation_points = interpolation_points[None,:]
        elif interpolation_points.ndim >= 3:
            raise ValueError('Interpolation points have to be a vector or matrix array, but its shape is {}.'.format(points.shape))
        if self.data_points.shape[1] != interpolation_points.shape[1]:
            raise ValueError('The data and interpolation points must have the same second dimension, but its {} and {}.'.format(self.data_points.shape[1], interpolation_points.shape[1]))

        interpolation_points = self._modify_interpolation_points(interpolation_points)

        assert interpolation_points.ndim == 2
        assert self.data_points.shape[1] == interpolation_points.shape[1]

        return interpolation_points


    ## scale points
    
    def _scale(self, points):
        scaling_values = self.scaling_values

        if scaling_values is not None and np.any(scaling_values != 1):
            logger.debug('Scaling {} points with values {}.'.format(len(points), scaling_values))
            assert len(scaling_values) in (1, points.shape[1])
            points = points * scaling_values
        else:
            logger.debug('Do not scaling {} points'.format(len(points)))

        return points


    ## interpolate
    
    def _calculate_interpolation(self, interpolation_points):
        raise NotImplementedError("Please implement this method.")


    def interpolate(self, interpolation_points):
        interpolation_points_dim = len(interpolation_points)

        if interpolation_points_dim > 0:
            ## check input
            interpolation_points = self._prepare_interpolation_points(interpolation_points)

            ## interpolate
            logger.debug('Interpolating values with method {} at {} points from {} data points.'.format(self.method, interpolation_points_dim, len(self.data_values)))

            interpolated_values = self._calculate_interpolation(interpolation_points)

            number_of_interpolated_values = np.logical_not(np.isnan(interpolated_values)).sum()
            logger.debug('Values interpolated for {} points.'.format(number_of_interpolated_values))
        else:
            interpolated_values = np.array(())

        assert interpolated_values.ndim == 1
        assert interpolated_values.shape[0] == interpolation_points.shape[0]

        return interpolated_values


    ## save and load
    
    def save(self, file):
        util.io.object.save(file, self)
        logger.debug('Interpolator saved to {}.'.format(file))

    @staticmethod
    def load(file):
        interpolator = util.io.object.load(file)
        logger.debug('Interpolator loaded from {}.'.format(file))
        return interpolator



## Changeable interpolator

class Interpolator_Values_Changeable(Interpolator_Base):

    def __init__(self, data_points, data_values, method, scaling_values=None, copy_arrays=True):
        logger.debug('Initiating values changable interpolator with {} data points, scaling values {} and method {}.'.format(len(data_points), scaling_values, method))

        self._interpolator = None
        super().__init__(data_points, data_values, method, possible_methods=('nearest', 'linear'), scaling_values=scaling_values, copy_arrays=copy_arrays)


    def _set_data_values(self, data_values):
        super()._set_data_values(data_values)

        if self._interpolator is not None:
            if self.method == 'nearest':
                self._interpolator.values = data_values
            else:
                self._interpolator.values = data_values[:,None]
            logger.debug('Data values in {} interpolator updated.'.format(self.method))
        else:
            logger.debug('Data values in {} interpolator must not be updated, since interpolated is not constructed.'.format(self.method))


    def _get_interpolator(self):
        interpolator = self._interpolator

        if interpolator is None:
            logger.debug('Constructing {} interpolator.'.format(self.method))
            if self.method == 'nearest':
                tree_options = {'compact_nodes': False, 'balanced_tree': False}
                interpolator = scipy.interpolate.ndgriddata.NearestNDInterpolator(self.data_points, self.data_values, tree_options=tree_options)
            else:
                interpolator = scipy.interpolate.interpnd.LinearNDInterpolator(self.data_points, self.data_values)
            self._interpolator = interpolator
        else:
            logger.debug('Returning cached {} interpolator.'.format(self.method))

        assert callable(interpolator)

        return interpolator


    def _calculate_interpolation(self, interpolation_points):
        interpolator = self._get_interpolator()
        interpolated_values = interpolator(interpolation_points)
        return interpolated_values



## Changable and partitionable interpolator

class Interpolator_Values_Changeable_Partitionable(Interpolator_Base):

    def __init__(self, data_points, data_values, method='linear', number_of_interpolators=1, single_overlapping_amount=0.5, scaling_values=None, copy_arrays=True, parallel=False):
        logger.debug('Initiating partitionable interpolator with {} data points, scaling values {} and {} interpolators with single_overlapping_amount of {}.'.format(len(data_points), scaling_values, number_of_interpolators, single_overlapping_amount))

        self.parallel = parallel

        ## sort data by first index
        indices_sorted_by_first_dim = np.argsort(data_points[:,0])
        self._indices_sorted_by_first_dim = indices_sorted_by_first_dim
        data_points_sorted = data_points[indices_sorted_by_first_dim]
        data_values_sorted = data_values[indices_sorted_by_first_dim]

        self._interpolators = [None,] * number_of_interpolators
        copy_arrays = np.asanyarray(copy_arrays)
        if copy_arrays.ndim == 0:
            copy_interpolation_points = copy_arrays[()]
        elif len(copy_arrays) == 1:
            copy_interpolation_points = copy_arrays[0]
        else:
            copy_interpolation_points = copy_arrays[1]
        super().__init__(data_points_sorted, data_values, method, possible_methods=('nearest', 'linear'), scaling_values=scaling_values, copy_arrays=(False, copy_interpolation_points))


        ## compute interpolation bound values

        def get_last_index_of_same_value(values, index, step=1):
            value_len= len(values)
            if index < 0:
                index = 0
            elif index >= value_len:
                index = value_len - 1
            last_index = index
            next_index = last_index + step

            while next_index < value_len and next_index >= 0 and values[next_index] == values[last_index]:
                last_index = next_index
                next_index = last_index + step

            return last_index

        def get_last_index_of_same_value_vectorize(values, indices, step=1):
            last_indices = np.empty_like(indices, dtype=np.int)
            for i in range(len(indices)):
                last_indices[i] = get_last_index_of_same_value(values, indices[i], step=step)
            return last_indices


        interpolation_bound_indices = np.floor(np.arange(0, number_of_interpolators + 1) * data_points_sorted.shape[0] / (number_of_interpolators))
        interpolation_bound_indices = interpolation_bound_indices.astype(np.int)
        interpolation_bound_indices = get_last_index_of_same_value_vectorize(data_points_sorted[:,0], interpolation_bound_indices, step=-1)
        interpolation_bound_indices[-1] = data_points_sorted.shape[0] - 1

        logger.debug('The interpolation bounds indices for the partitioned interpolator are {}.'.format(interpolation_bound_indices))


        interpolation_bound_values = data_points_sorted[:,0][interpolation_bound_indices]
        interpolation_bound_values = np.asarray(interpolation_bound_values, dtype=np.float)
        interpolation_bound_values[0] = -np.inf
        interpolation_bound_values[-1] = np.inf

        self.interpolation_bound_values = interpolation_bound_values

        logger.debug('The interpolation bounds for the partitioned interpolator are {}.'.format(interpolation_bound_values))


        ## compute value range indices
        def get_interpolator_data_values_ranges(data_values_sorted_first_dim, interpolation_bound_indices, single_overlapping_amount):

            overlapping_len = len(data_values_sorted_first_dim) * single_overlapping_amount
            overlapping_interpolation_bound_indices_lower = np.floor(interpolation_bound_indices - overlapping_len)
            overlapping_interpolation_bound_indices_lower = overlapping_interpolation_bound_indices_lower.astype(np.int)
            overlapping_interpolation_bound_indices_upper = np.ceil(interpolation_bound_indices + overlapping_len)
            overlapping_interpolation_bound_indices_upper = overlapping_interpolation_bound_indices_upper.astype(np.int)

            number_of_interpolators = len(interpolation_bound_indices) - 1
            interpolator_data_value_ranges = np.empty((number_of_interpolators, 2), dtype=np.int)
            
            interpolator_data_value_ranges[:,0] = get_last_index_of_same_value_vectorize(data_values_sorted_first_dim, overlapping_interpolation_bound_indices_lower[:-1], step=-1)
            interpolator_data_value_ranges[:,1] = get_last_index_of_same_value_vectorize(data_values_sorted_first_dim, overlapping_interpolation_bound_indices_upper[1:], step=1)

            return interpolator_data_value_ranges

        data_value_range_indices = get_interpolator_data_values_ranges(data_points_sorted[:,0], interpolation_bound_indices, single_overlapping_amount)
        self.data_value_range_indices = data_value_range_indices

        logger.debug('The data value range indices for the partitioned interpolator are {}.'.format(data_value_range_indices))


    def _set_data_values(self, data_values):
        super()._set_data_values(data_values[self._indices_sorted_by_first_dim])

        for interpolator_index in range(self.number_of_interpolators):
            interpolator = self._interpolators[interpolator_index]
            if interpolator is not None:
                (index_start, index_end) = self.data_value_range_indices[interpolator_index]
                interpolator.data_values = self.data_values[index_start:index_end]
                logger.debug('Data values in interpolator with index {} with data from index {} to {} updated.'.format(interpolator_index, index_start, index_end))
            else:
                logger.debug('Data values in interpolator with index {} must not be updated, since interpolated is not constructed.'.format(interpolator_index))


    @property
    def number_of_interpolators(self):
        return len(self._interpolators)


    def _get_interpolator(self, interpolator_index):
        interpolator = self._interpolators[interpolator_index]

        if interpolator is None:
            logger.debug('Constructing interpolator with index {}.'.format(interpolator_index))

            (index_start, index_end) = self.data_value_range_indices[interpolator_index]
            interpolator = Interpolator_Values_Changeable(self.data_points[index_start:index_end], self.data_values[index_start:index_end], method=self.method, copy_arrays=False)
            self._interpolators[interpolator_index] = interpolator
        else:
            logger.debug('Returning cached interpolator with index {}.'.format(interpolator_index))

        return interpolator


    def _calculate_interpolation(self, interpolation_points):
        ## sort points by first dim
        indices_sorted_by_first_dim = np.argsort(interpolation_points[:,0])
        interpolation_points = interpolation_points[np.argsort(interpolation_points[:,0])]

        ## assign interpolation points to interpolators
        interpolation_points_dim = len(interpolation_points)
        interpolation_bound_values = self.interpolation_bound_values

        interpolator_index = 0
        start_index = 0
        end_index = 0

        end_indices = [0]

        number_of_interpolators = self.number_of_interpolators
        assert number_of_interpolators == len(interpolation_bound_values) - 1

        for interpolator_index in range(number_of_interpolators):
            while end_index < interpolation_points_dim and interpolation_points[end_index, 0] < interpolation_bound_values[interpolator_index + 1]:
                end_index += 1

            end_indices.append(end_index)
            start_index = end_index

        logger.debug('Interpolation points ranges {} assigned to interpolators {} with {} interpolator in use.'.format(end_indices, len(end_indices), number_of_interpolators))

        ## interpolate
        interpolated_values_sorted = np.empty(interpolation_points_dim)

        if not self.parallel or number_of_interpolators <= 1:
            logger.debug('Starting serial interpolation.')
            for interpolator_index in range(number_of_interpolators):
                interpolated_values_sorted[end_indices[interpolator_index]:end_indices[interpolator_index+1]] = self._calculate_interpolation_for_index(interpolator_index, interpolation_points[end_indices[interpolator_index]:end_indices[interpolator_index+1]])
        else:
            logger.debug('Starting interpolation with {} processes.'.format(number_of_interpolators))
            result = util.parallel.with_multiprocessing.map_parallel(self._calculate_interpolation_for_index_zipped, [(interpolator_index, interpolation_points[end_indices[interpolator_index]:end_indices[interpolator_index+1]]) for interpolator_index in range(number_of_interpolators)], number_of_processes=number_of_interpolators, chunksize=1)
            for interpolator_index in range(number_of_interpolators):
                interpolated_values_sorted[end_indices[interpolator_index]:end_indices[interpolator_index+1]] = result[interpolator_index]

        ## revert sort
        indices_sorted_by_first_dim_rev = np.empty(interpolation_points_dim, dtype=int)
        indices_sorted_by_first_dim_rev[indices_sorted_by_first_dim] = np.arange(interpolation_points_dim)

        assert np.all(indices_sorted_by_first_dim[indices_sorted_by_first_dim_rev] == np.arange(interpolation_points_dim))

        interpolated_values = interpolated_values_sorted[indices_sorted_by_first_dim_rev]
        return interpolated_values


    def _calculate_interpolation_for_index(self, interpolator_index, interpolation_points):
        interpolator = self._get_interpolator(interpolator_index)
        return interpolator.interpolate(interpolation_points)

    def _calculate_interpolation_for_index_zipped(self, zipped):
        interpolator_index, interpolation_points = zipped
        return self._calculate_interpolation_for_index(interpolator_index, interpolation_points)



## Nearest and linear Interpolator


class Interpolator_Nearest(Interpolator_Values_Changeable):
    def __init__(self, data_points, data_values, scaling_values=None, copy_arrays=True):
        super().__init__(data_points, data_values, method='nearest', scaling_values=scaling_values, copy_arrays=copy_arrays)

class Interpolator_Linear(Interpolator_Values_Changeable):
    def __init__(self, data_points, data_values, scaling_values=None, copy_arrays=True):
        super().__init__(data_points, data_values, method='linear', scaling_values=scaling_values, copy_arrays=copy_arrays)

class Interpolator_Linear_Partitionable(Interpolator_Values_Changeable_Partitionable):
    def __init__(self, data_points, data_values, scaling_values=None, copy_arrays=True, number_of_interpolators=1, single_overlapping_amount=0, parallel=True):
        super().__init__(data_points, data_values, method='linear', scaling_values=scaling_values, copy_arrays=copy_arrays, number_of_interpolators=number_of_interpolators, single_overlapping_amount=single_overlapping_amount, parallel=parallel)



## Universal interpolator

class Interpolator(Interpolator_Base):

    def __init__(self, data_points, data_values, method='linear_then_nearest_new_data', number_of_linear_interpolators=1, single_overlapping_amount_linear_interpolators=0, scaling_values=None, copy_arrays=True, parallel=False):

        logger.debug('Initiating interpolator {} with {} data points, scaling values {}, number of linear interpolators {} and single overlapping amount of linear interpolators {}.'.format(method, len(data_points), scaling_values, number_of_linear_interpolators, single_overlapping_amount_linear_interpolators))

        self.interpolators = []

        super().__init__(data_points, data_values, method=method, possible_methods=('linear_then_nearest_new_data', 'linear_then_nearest_same_data'), scaling_values=scaling_values, copy_arrays=copy_arrays)

        if number_of_linear_interpolators > 1:
            interpolator = Interpolator_Linear_Partitionable(self.data_points, self.data_values, number_of_interpolators=number_of_linear_interpolators, single_overlapping_amount=single_overlapping_amount_linear_interpolators, copy_arrays=False, parallel=parallel)
            self.interpolators.append(interpolator)
        elif number_of_linear_interpolators == 1:
            interpolator = Interpolator_Linear(self.data_points, self.data_values, copy_arrays=False)
            self.interpolators.append(interpolator)
        if number_of_linear_interpolators == 0 or method == 'linear_then_nearest_same_data':
            interpolator = Interpolator_Nearest(self.data_points, self.data_values, copy_arrays=False)
            self.interpolators.append(interpolator)

        assert len(self.interpolators) >= 1


    def _set_data_values(self, data_values):
        super()._set_data_values(data_values)

        for interpolator in self.interpolators:
            interpolator.data_values = self.data_values


    def _calculate_interpolation(self, interpolation_points):

        interpolation_values = self.interpolators[0].interpolate(interpolation_points)
        no_value_mask = np.isnan(interpolation_values)

        if np.any(no_value_mask):
            if self.method == 'linear_then_nearest_new_data':
                new_data_points = np.concatenate([self.data_points, interpolation_points[~ no_value_mask]])
                new_data_values = np.concatenate([self.data_values, interpolation_values[~ no_value_mask]])
                interpolator_nearest = Interpolator_Values_Changeable(self.data_points, self.data_values, method='nearest', copy_arrays=False)
            elif self.method == 'linear_then_nearest_same_data':
                interpolator_nearest = self.interpolators[1]

            interpolation_values[no_value_mask] = interpolator_nearest.interpolate(interpolation_points[no_value_mask])

        assert np.all(np.logical_not(np.isnan(interpolation_values)))

        return interpolation_values



class Periodic_Interpolator(Interpolator):

    def __init__(self, data_points, data_values, point_range_size, wrap_around_amount=None, method='linear_then_nearest_new_data', number_of_linear_interpolators=1, single_overlapping_amount_linear_interpolators=0, scaling_values=None, copy_arrays=True, parallel=False):

        logger.debug('Initiating periodic interpolator with point_range_size {} and wrap around amount {}.'.format(point_range_size, wrap_around_amount))

        point_range_size = np.asanyarray(point_range_size)
        wrap_around_amount = np.asanyarray(wrap_around_amount)

        if not (data_points.shape[1] == len(point_range_size) == len(wrap_around_amount)):
            raise ValueError('Second dim of data_points ({}) and len of point_range_size ({}) and len of wrap_around_amount ({}) have to be equal.'.format(data_points.shape[1], len(point_range_size), len(wrap_around_amount)))
        assert np.all(np.max(data_points, axis=0) - np.min(data_points, axis=0) <= point_range_size)

        self._point_range_size = point_range_size
        self._wrap_around_amount = wrap_around_amount
        self._data_indices = np.arange(len(data_points))

        super().__init__(data_points, data_values, method=method, number_of_linear_interpolators=number_of_linear_interpolators, single_overlapping_amount_linear_interpolators=single_overlapping_amount_linear_interpolators, scaling_values=scaling_values, copy_arrays=copy_arrays, parallel=parallel)

        assert len(self._data_points) == len(self._data_values) == len(self._data_indices)


    def _prepare_data_values(self, data_values):
        logger.debug('Wrapping around data values.')

        data_values = data_values[self._data_indices]
        data_values = super()._prepare_data_values(data_values)

        return data_values


    def _modify_points(self, points, is_data_points):
        for i in range(points.shape[1]):
            if self._wrap_around_amount is not None and self._wrap_around_amount[i] > 0:
                ## modulo for periodicity
                logger.debug('Calculate modulo {} of index {} of points.'.format(self._point_range_size[i], i))
                points[:, i] = points[:, i] % self._point_range_size[i]

                ## if data points, wrap around
                if is_data_points:
                    points, indices = util.math.interpolate.wrap_around(points, i, self._point_range_size[i], amount=self._wrap_around_amount[i], return_also_indices=True)
                    self._data_indices = self._data_indices[indices]

        return points


    def _modify_interpolation_points(self, points):
        points = self._modify_points(points, False)
        points = super()._modify_interpolation_points(points)
        return points


    def _modify_data_points(self, points):
        points = self._modify_points(points, True)
        points = super()._modify_data_points(points)
        return points



## universial interpolation method

def interpolate(data_points, data_values, interpolation_points, number_of_linear_interpolators=1, single_overlapping_amount_linear_interpolators=0, scaling_values=None):
    interpolator = Interpolator(data_points, data_values, number_of_linear_interpolators=number_of_linear_interpolators, single_overlapping_amount_linear_interpolators=single_overlapping_amount_linear_interpolators, scaling_values=scaling_values)
    return interpolator.interpolate(interpolation_points)
