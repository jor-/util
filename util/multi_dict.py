import numpy as np

import util.math.sort
import util.io.object
import util.logging

logger = util.logging.logger


def _isdict(d):
    if isinstance(d, dict):
        return True
    else:
        from blist import sorteddict
        return isinstance(d, sorteddict)


class MultiDict():

    def __init__(self, sorted=False):
        if sorted:
            from blist import sorteddict
            self._value_dict = sorteddict()
        else:
            self._value_dict = dict()

    ## properties

    @property
    def value_dict(self):
        return self._value_dict

    @property
    def sorted(self):
        return not isinstance(self.value_dict, dict)

    @property
    def SUPPORTED_RETURN_TYPES(self):
        return ('array', 'self', 'self_type', 'self_type_unsorted', 'self_type_sorted', 'multi_dict_unsorted', 'multi_dict_sorted')


    ## str

    def __str__(self):
        return str(self.value_dict)


    ## mapping methods

    def get_value_list(self, key):
        value = self.value_dict
        for i in range(len(key)):
            if not _isdict(value):
                raise KeyError('Key {} not applicable. Under key {} is a value of type {}.'.format(key, key[:i], type(value)))
            try:
                value = value[key[i]]
            except TypeError as e:
                raise KeyError('Key {} not applicable. The {}th value of the key is of type {}.'.format(key, i, type(key[i]))) from e
        return value

    def __getitem__(self, key):
        return self.get_value_list(key)


    def _get_last_dict(self, key):
        value_dict = self.value_dict
        value_dict_type = type(value_dict)

        for i in range(len(key)-1):
            value_dict = value_dict.setdefault(key[i], value_dict_type())

        return value_dict


    def set_value_list(self, key, value_list):
        value_list = list(value_list)
        last_dict = self._get_last_dict(key)
        last_dict[key[len(key)-1]] = value_list

    def __setitem__(self, key, value):
        self.set_value_list(key, value)


    def has_values(self, key):
        try:
            value_list = self.get_value_list(key)
        except KeyError:
            return False
        return len(value_list) > 0

    def __contains__(self, key):
        return self.has_values(key)


    @property
    def len(self):
        total_len = 0
        for (key, value_list) in self.iterator_keys_and_value_lists():
            total_len += len(value_list)

        return total_len

    def __len__(self):
        return self.len



    ## add

    def _get_or_init_value_list(self, key):
        last_dict = self._get_last_dict(key)
        value_list = last_dict.setdefault(key[len(key)-1], [])
        return value_list

    def extend_value_list(self, key, value_list):
        if len(value_list) > 0:
            self._get_or_init_value_list(key).extend(value_list)

    def append_value(self, key, value):
        self._get_or_init_value_list(key).append(value)

    def _add_value_lists(self, keys, value_lists, add_function):
        assert callable(add_function)
        if keys is None:
            keys = []
        if value_lists is None:
            value_lists = []

        if len(keys) != len(value_lists):
            raise ValueError('Len of keys {} and len of values {} have to be the same!'.format(len(keys), len(value_lists)))

        values_len = len(value_lists)
        logger.debug('Adding {} values.'.format(values_len))
        for i in range(values_len):
            add_function(keys[i], value_lists[i])

    def extend_value_lists(self, keys, value_lists):
        add_function = lambda key, value_list: self.extend_value_list(key, value_list)
        self._add_value_lists(keys, value_lists, add_function)

    def append_values(self, keys, values):
        add_function = lambda key, value: self.append_value(key, value)
        self._add_value_lists(keys, values, add_function)


    ## remove
    
    def clear(self):
        self._value_dict = type(self.value_dict)()

    def remove_value(self, key, value):
        logger.debug('Removing value {} for key {}.'.format(value, key))
        value_list = self.get_value_list(key)
        n = len(value_list)
        value_list[:] = [v for v in value_list if not np.all(np.isclose(v, value))]
        if len(value_list) == n:
            raise KeyError('Value {} was not deposited for key {}.'.format(value, key))
        #TODO remove dict entries if list is empty
    
    def remove_values(self, multi_dict):
        logger.debug('Removing {} values.'.format(len(multi_dict)))
        for key, value_list in multi_dict.iterator_keys_and_value_lists():
            for value in value_list:
                self.remove_value(key, value)


    ## access

    def keys(self):
        all_keys = []
        for (key, values) in self.iterator_keys_and_value_lists():
            for value in values:
                all_keys.append(key)

        all_keys = np.array(all_keys)
        # assert all_keys.ndim == 2
        return all_keys


    def values(self):
        all_values = []
        for (key, values) in self.iterator_keys_and_value_lists():
            all_values.extend(values)

        all_values = np.array(all_values)
        return all_values


    def items(self):
        all_keys = self.keys()
        all_keys = all_keys.reshape((all_keys.shape[0], -1))
        all_values = self.values()
        if all_values.ndim == 1:
            all_values = all_values[:, np.newaxis]
        return np.concatenate([all_keys, all_values], axis=1)

    def toarray(self):
        return self.items()


    ## io
    def save(self, file, only_dict=True):
        logger.debug('Saving {} to {}.'.format(self, file))
        if only_dict:
            util.io.object.save(file, self.value_dict)
        else:
            util.io.object.save(file, self)


    # def load(self, file):
    #     logger.debug('Loading {} from {}.'.format(self, file))
    #     self._value_dict = util.io.io.load_object(file)
    #     return self

    @classmethod
    def load(cls, file):
        logger.debug('Loading {} from {}.'.format(cls.__name__, file))

        ## load object
        loaded_object = util.io.object.load(file)

        ## check if dict
        is_dict = isinstance(loaded_object, dict)
        if not is_dict:
            from blist import sorteddict
            is_dict = isinstance(loaded_object, sorteddict)

        ## if dict make new object with dict
        if is_dict:
            obj = cls()
            obj._value_dict = loaded_object
        ## otherwise return loaded object
        else:
            obj = loaded_object

        return obj


    ## properties

    @property
    def value_dict(self):
        return self._value_dict

    @property
    def sorted(self):
        return not isinstance(self.value_dict, dict)


        obj = cls()
        obj._value_dict = util.io.object.load(file)
        return obj



    ## create

    def new_like(self, sorted=None):
        if sorted is None:
            sorted = self.sorted
        new = type(self)(sorted=sorted)
        return new
    
    
    def copy(self):
        import copy
        return copy.deepcopy(self)


    def _return_items_as_type(self, keys, value_lists, return_type=None):
        logger.debug('Returning {} values as type {}.'.format(len(keys), return_type))

        ## chose default
        if return_type is None:
            return_type = 'array'
        if return_type == 'multi_dict':
            return_type = 'multi_dict_unsorted'


        ## check input
        if return_type not in self.SUPPORTED_RETURN_TYPES:
            raise ValueError('Unknown return_type "{}". Only {} are supported.'.format(return_type, self.SUPPORTED_RETURN_TYPES))

        n = len(keys)
        if n != len(value_lists):
            raise ValueError('Len of keys {} and len of value lists {} have to be the same!'.format(len(keys), len(value_lists)))


        ## make list of value lists
        value_lists = list(value_lists)
        for i in range(n):
            try:
                value_lists[i] = list(value_lists[i])
            except TypeError:
                value_lists[i] = [value_lists[i]]


        ## return multi_dict type
        if return_type in ('self', 'self_type', 'self_type_unsorted', 'self_type_sorted', 'multi_dict_unsorted', 'multi_dict_sorted'):
            if return_type == 'self':
                m = self
                m.clear()
            if return_type in ('self_type', 'self_type_unsorted', 'self_type_sorted'):
                if return_type == 'self_type':
                    sorted = None
                if return_type == 'self_type_unsorted':
                    sorted = False
                if return_type == 'self_type_sorted':
                    sorted = True
                m = self.new_like(sorted=sorted)
            if return_type == 'multi_dict_unsorted':
                m = MultiDict(sorted=False)
            if return_type == 'multi_dict_sorted':
                m = MultiDict(sorted=True)

            m.extend_value_lists(keys, value_lists)
#             try:
#                 m.extend_value_lists(keys, value_lists)
#             except TypeError:
#                 m.append_values(keys, value_lists)
# #             m.append_values(keys, values)

            obj = m

        ## return array
        if return_type == 'array':
            if len(value_lists) > 0:
                def get_value_len(value):
                    try:
                        return len(value)
                    except TypeError:
                        return 1
                value_ref_len = get_value_len(value_lists[0][0])
                for value_list in value_lists:
                    for value in value_list:
                        if get_value_len(value) != value_ref_len:
                            raise ValueError('Len of each value has to be the same, but a len is {}!'.format(get_value_len(value)))

                n = len(keys)
                m = len(keys[0]) + value_ref_len

                array = np.empty((n, m))
                for i in range(n):
                    array[i, :-value_ref_len] = keys[i]
                    for value in value_lists[i]:
                        array[i, -value_ref_len:] = value

                obj = array
            else:
                obj = np.empty((0, 0))

        return obj


    ## iterate

    def _iterate_generator_value_dict(self, value_dict, value_dict_type=None, key_prefix=()):
        if value_dict_type is None:
            value_dict_type = type(value_dict)
        for (key, value) in value_dict.items():
            total_key = key_prefix + (key,)
            if isinstance(value, value_dict_type):
                yield from self._iterate_generator_value_dict(value, value_dict_type=value_dict_type, key_prefix=total_key)
            else:
                yield (total_key, value)


    def iterator_keys_and_value_lists(self):
        value_dict = self.value_dict
        yield from self._iterate_generator_value_dict(value_dict)


    def iterate_items(self, fun, min_number_of_values=1, return_type='array'):
        assert callable(fun)

        ## init
        new_keys = []
        new_values = []

        ## iterate
        for (key, values) in self.iterator_keys_and_value_lists():
            if len(values) >= min_number_of_values:
                key = np.asarray(key)
                values = np.asarray(values)
                new_value = fun(key, values)

                ## insert
                new_keys.append(key)
                new_values.append(new_value)

        ## finishing
        return self._return_items_as_type(new_keys, new_values, return_type=return_type)


    def iterate_values(self, fun, min_number_of_values=1, return_type='array'):
        fun_wrapper = lambda key, values: fun(values)
        return self.iterate_items(fun_wrapper, min_number_of_values=min_number_of_values, return_type=return_type)



    ## transform keys

    def transform_keys(self, transform_function):
        logger.debug('Transforming keys of {}.'.format(self))
        assert callable(transform_function)

        value_dict = self._value_dict
        self.clear()

        for (key, value_list) in self._iterate_generator_value_dict(value_dict):
            key_transformed = transform_function(key)
            self.extend_value_list(key_transformed, value_list)


    def keys_to_int_keys(self, dtype=np.int):
        logger.debug('Converting keys to type {}.'.format(dtype))
        transform_function = lambda key: tuple(np.array(np.round(key), dtype=dtype))
        self.transform_keys(transform_function)


    def dicard_key_dim(self, key_dim):
        logger.debug('Discarding key dim {}.'.format(key_dim))

        def transform_function(current_key):
            current_key = list(current_key)
            current_key[key_dim] = 0
            current_key = tuple(current_key)
            return current_key

        self.transform_keys(transform_function)


    def dicard_key_dims(self, key_dims):
        for key_dim in key_dims:
            self.dicard_key_dim(key_dim)



    ## transform values

    def transform_value_lists(self, transform_function):
        logger.debug('Transforming value lists of {}.'.format(self))
        assert callable(transform_function)

        for (key, value_list) in self.iterator_keys_and_value_lists():
            transformed_value_list = transform_function(key, value_list)
            self.set_value_list(key, transformed_value_list)

    def transform_values(self, transform_function):
        logger.debug('Transforming values of {}.'.format(self))
        assert callable(transform_function)

        def transform_function_wapper(key, value_list):
            transformed_value_list = []
            for value in value_list:
                transformed_value_list.append(transform_function(key, value))
            return transformed_value_list

        self.transform_value_lists(transform_function_wapper)

        # for (key, value_list) in self.iterator_keys_and_value_lists():
        #     transformed_value_list = []
        #     for value in value_list:
        #         transformed_value_list.append(transform_function(key, value))
        #     self.set_value_list(key, transformed_value_list)


    def set_min_value(self, min_value):
        logger.debug('Applying min value {} to values.'.format(min_value))

        transform_function = lambda key, value: max([value, min_value])
        self.transform_value(transform_function)


#     def is_at_least_value(self, value_threshold):
#         logger.debug('Applying is at least value {} to values.'.format(value_threshold))
#
#         transform_function = lambda key, value: value >= value_threshold)
#         self.transform_value(transform_function)


    def log_values(self):
        logger.debug('Applying logarithm to values.')

        transform_function = lambda key, value: np.log(value)
        self.transform_value(transform_function)



    ## filter

    def filter_with_boolean_function(self, boolean_filter_function, return_type='self'):
        assert callable(boolean_filter_function)

        filtered_keys = []
        filtered_value_lists = []

        for (key, value_list) in self.iterator_keys_and_value_lists():
            if boolean_filter_function(key, value_list):
                filtered_keys.append(key)
                filtered_value_lists.append(value_list)

        logger.debug('Filtered {} value lists.'.format(len(filtered_value_lists)))
        return self._return_items_as_type(filtered_keys, filtered_value_lists, return_type=return_type)


    def filter_min_number_of_values(self, min_number_of_values=1, return_type='self'):
        def boolean_filter_function(key, value_list):
            return len(value_list) >= min_number_of_values

        return self.filter_with_boolean_function(boolean_filter_function, return_type=return_type)


    def filter_key_range(self, key_index, bounds, return_type='self'):
        def boolean_filter_function(key, value_list):
            return bounds[0] <= key[key_index] <= bounds[1]

        return self.filter_with_boolean_function(boolean_filter_function, return_type=return_type)



    def filter_with_list_function(self, list_filter_function, return_type='self'):
        assert callable(list_filter_function)

        filtered_keys = []
        filtered_value_lists = []

        for (key, value_list) in self.iterator_keys_and_value_lists():
            filter_value_list = list_filter_function(key, value_list)
            if len(filter_value_list) > 0:
                filtered_keys.append(key)
                filtered_value_lists.append(filter_value_list)

        logger.debug('Filtered {} value lists.'.format(len(filtered_value_lists)))
        return self._return_items_as_type(filtered_keys, filtered_value_lists, return_type=return_type)


    def filter_finite(self, return_type='self'):
        def list_filter_function(key, value_list):
            filtered_value_list = []
            for value in value_list:
                if np.all(np.isfinite(value)):
                    filtered_value_list.append(value)
            return filtered_value_list

        return self.filter_with_list_function(list_filter_function, return_type=return_type)





    ## compute values

    def numbers(self, min_number_of_values=1, return_type='array'):
        logger.debug('Calculate numbers of values with at least {} values.'.format(min_number_of_values))

        return self.iterate_values(len, min_number_of_values, return_type=return_type)


    def means(self, min_number_of_values=1, min_value=0, return_type='array'):
        logger.debug('Calculate means of values with at least {} values with minimal mean {}.'.format(min_number_of_values, min_value))
        if min_value is None:
            min_value = - np.inf

        def calculate_function(values):
            mean = np.average(values)
            mean = max([mean, min_value])
            return mean

        return self.iterate_values(calculate_function, min_number_of_values, return_type=return_type)


    def variances(self, min_number_of_values=3, min_value=0, return_type='array'):
        logger.debug('Calculate variances of values with at least {} values with minimal variance {}.'.format(min_number_of_values, min_value))
        if min_value is None:
            min_value = 0

        def calculate_function(values):
            mean = np.average(values)
            number_of_values = values.size
            variance = np.sum((values - mean)**2) / (number_of_values - 1)
            variance = max([variance, min_value])
            return variance

        return self.iterate_values(calculate_function, min_number_of_values, return_type=return_type)


    def standard_deviations(self, min_number_of_values=3, min_value=0, return_type='array'):
        logger.debug('Calculate standard deviations of values with at least {} values with minimal deviation {}.'.format(min_number_of_values, min_value))
        if min_value is None:
            min_value = 0

        def calculate_function(values):
            mean = np.average(values)
            number_of_values = values.size
            deviation = (np.sum((values - mean)**2) / (number_of_values - 1))**(1/2)
            deviation = max([deviation, min_value])
            return deviation

        return self.iterate_values(calculate_function, min_number_of_values, return_type=return_type)




    ## tests for normality

    def dagostino_pearson_test(self, min_number_of_values=50, alpha=0.05, return_type='array'):
        logger.debug('Calculate D´Agostino-Person-test for normality of values with minimal {} values with alpha {}.'.format(min_number_of_values, alpha))
        import scipy.stats

        test_values = self.iterate_values(lambda x: scipy.stats.normaltest(x)[1], min_number_of_values, return_type=return_type)

        if alpha is not None:
            if return_type == 'array':
                test_values[:,-1] = (test_values[:,-1] >= alpha).astype(np.float)
            else:
                transform_function = lambda key, value: (value >= alpha).astype(np.float)
                self.transform_value(transform_function)

        return test_values


    def shapiro_wilk_test(self, min_number_of_values=50, alpha=0.05, return_type='array'):
        logger.debug('Calculate Shapiro-Wilk-test for normality of values with minimal {} values with alpha {}.'.format(min_number_of_values, alpha))
        import scipy.stats

        test_values = self.iterate_values(lambda x: scipy.stats.shapiro(x)[1], min_number_of_values, return_type=return_type)

        if alpha is not None:
            if return_type == 'array':
                test_values[:,-1] = (test_values[:,-1] >= alpha).astype(np.float)
            else:
                transform_function = lambda key, value: (value >= alpha).astype(np.float)
                self.transform_value(transform_function)

        return test_values


    def anderson_darling_test(self, min_number_of_values=50, alpha=0.05, return_type='array'):
        logger.debug('Calculate Anderson-Darling-test for normality of values with minimal {} values with alpha {}.'.format(min_number_of_values, alpha))
        import scipy.stats

        def test(x, alpha):
            ## get test values
            t = scipy.stats.anderson(x)
            test_value = t[0]
            test_bounds = t[1]
            test_alphas = t[2] / 100

            ## get bound for alpha
            index = np.where(test_alphas == alpha)[0]
            if len(index) == 0:
                raise ValueError('The alpha value {} is not supported for this test. Only to values {} are supported.'.format(alpha, test_alphas))
            index = index[0]
            bound = test_bounds[index]

            ## check if test passed
            return test_value <= bound

        test_values = self.iterate_values(lambda x:test(x, alpha), min_number_of_values, return_type=return_type)
        return test_values



class MultiDictPreprocessKey(MultiDict):

    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)


    @staticmethod
    def _check_keys(keys):
        pass

    @staticmethod
    def _preprocess_keys(keys):
        raise NotImplementedError("Please implement this method.")


    ## override for key pairs

    def get_value_list(self, key):
        self._check_keys(key)
        key = self._preprocess_keys(key)
        return super().get_value_list(key)


    def set_value_list(self, key, value_list):
        self._check_keys(key)
        key = self._preprocess_keys(key)
        super().set_value_list(key, value_list)


    def _get_or_init_value_list(self, key):
        self._check_keys(key)
        key = self._preprocess_keys(key)
        return super()._get_or_init_value_list(key)




class MultiDictPointPairs(MultiDictPreprocessKey):

    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)


    @staticmethod
    def _check_keys(keys):
        try:
            are_two_keys = len(keys) == 2
        except TypeError:
            raise ValueError('keys {} have to be a pair of keys.'.format(keys))

        try:
            are_two_keys = len(keys[0]) == len(keys[1])
        except (TypeError, KeyError):
            raise ValueError('keys {} have to be two keys of equal length.'.format(keys))


    def _iterate_generator_value_dict(self, value_dict, value_dict_type=None, key_prefix=()):
        if value_dict_type is None:
            value_dict_type = type(value_dict)
        for (key, value) in value_dict.items():
            total_key = key_prefix + (key,)
            if isinstance(value, value_dict_type):
                yield from self._iterate_generator_value_dict(value, value_dict_type, key_prefix=total_key)
            else:
                split_index = int(len(total_key)/2)
                yield ((total_key[0:split_index], total_key[split_index:]), value)


    def _return_items_as_type(self, keys, value_lists, return_type=None):
        if return_type is None or return_type == 'array':
            flattened_keys = []
            for key in keys:
                flattened_key = tuple(np.array(key).flatten())
                flattened_keys.append(flattened_key)
            keys = flattened_keys

        return super()._return_items_as_type(keys, value_lists, return_type=return_type)





class MultiDictPermutablePointPairs(MultiDictPointPairs):

    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)
    #     self._pair_order = 1
    #
    #
    # @property
    # def pair_order(self):
    #     return self._pair_order

    # @staticmethod
    # def _check_keys(keys):
    #     try:
    #         are_two_keys = len(keys) == 2
    #     except TypeError:
    #         raise ValueError('keys {} have to be a pair of keys.'.format(keys))
    #
    #     try:
    #         are_two_keys = len(keys[0]) == len(keys[1])
    #     except (TypeError, KeyError):
    #         raise ValueError('keys {} have to be two keys of equal length.'.format(keys))


    @staticmethod
    def _preprocess_keys(keys):
        ## if two keys, make them to one sorted key
        key_array = np.array(keys)
        sorted_indices = util.math.sort.lex_sorted_indices(key_array, order=1)
        key_array = key_array[sorted_indices]
        keys = tuple(key_array.flat)
        return keys


    # def _iterate_generator_value_dict(self, value_dict, value_dict_type=None, key_prefix=()):
    #     if value_dict_type is None:
    #         value_dict_type = type(value_dict)
    #     for (key, value) in value_dict.items():
    #         total_key = key_prefix + (key,)
    #         if isinstance(value, value_dict_type):
    #             yield from self._iterate_generator_value_dict(value, value_dict_type, key_prefix=total_key)
    #         else:
    #             split_index = int(len(total_key)/2)
    #             yield ((total_key[0:split_index], total_key[split_index:]), value)
    #
    #
    # def _return_items_as_type(self, keys, value_lists, return_type=None):
    #     if return_type is None or return_type == 'array':
    #         flattened_keys = []
    #         for key in keys:
    #             flattened_key = tuple(np.array(key).flatten())
    #             flattened_keys.append(flattened_key)
    #         keys = flattened_keys
    #
    #     return super()._return_items_as_type(keys, value_lists, return_type=return_type)





class MultiDictDiffPointPairs(MultiDictPreprocessKey):

    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)


    @staticmethod
    def _preprocess_keys(keys):
        ## if two keys, use diff as key
        key_array = np.array(keys)
        if key_array.ndim > 1:
            key = tuple(np.abs(key_array[1] - key_array[0]))
        else:
            key = tuple(np.abs(key_array))
        return key


