import numpy as np


def _step_sizes(x, eps, typical_x=None, use_always_typical_x=True, bounds=None, both_directions=True, dtype=np.float64):
    # init values
    if typical_x is None:
        typical_x = np.ones_like(x)
    elif not len(x) == len(typical_x):
        raise ValueError(f'x and typical_x must have the same length but their length are {len(x)} and {len(typical_x)}.')
    n = len(x)
    if bounds is None:
        bounds = ((-np.inf, np.inf),) * len(x)
    else:
        bounds = np.asanyarray(bounds)
    if both_directions:
        h = np.empty((n, 2), dtype=dtype)
        x_h = np.empty((n, 2), dtype=x.dtype)
        x_h[:, 0] = x
        x_h[:, 1] = x
    else:
        h = np.empty((n, 1), dtype=dtype)
        x_h = x.copy()

    # calculate f for changes in a single component
    for i in range(n):
        # calculate h
        if use_always_typical_x:
            typical_x_i = np.abs(typical_x[i])
        else:
            typical_x_i = np.max([np.abs(typical_x[i]), np.abs(x[i])])
        h_i = eps * typical_x_i
        if not both_directions and x[i] < 0:  # h_i and x[i] should have same np.sign
            h_i *= -1

        # calculate x_h
        if both_directions:
            x_h[i, 0] += h_i
            x_h[i, 1] -= h_i
        else:
            x_h[i] += h_i

        # consider bounds
        lower_bound = bounds[i][0]
        upper_bound = bounds[i][1]
        if both_directions:
            for j, x_h_i_j in enumerate(x_h[i]):
                x_h[i, j] = min(max(x_h_i_j, lower_bound), upper_bound)
        elif (x_h[i] < lower_bound or x_h[i] > upper_bound):
            h_i *= -1
            x_h[i] = x[i] + h_i

        # recalculate h to improvement of accuracy of h
        if both_directions:
            for j, x_h_i_j in enumerate(x_h[i]):
                h[i, j] = x_h[i, j] - x[i]
        else:
            h[i] = x_h[i] - x[i]

    # return
    assert not both_directions or np.all(h[:, 0] >= h[:, 1])
    return h


def _f_x_h(f, x, h, *h_indices, dtype=np.float64):
    # calculate x
    x_h = x.copy()
    for h_index in h_indices:
        x_h[h_index[0]] += h[h_index]
    # eval f
    f_x_h = f(x_h)
    f_x_h = np.asanyarray(f_x_h, dtype=dtype)
    # return
    return f_x_h


def first_derivative(f, x, f_x=None, typical_x=None, bounds=None, eps=None, use_always_typical_x=True, accuracy_order=2):
    assert accuracy_order in (1, 2)

    # convert x
    x = np.asanyarray(x)

    # calculate step size
    if eps is None:
        if accuracy_order == 1:
            eps = np.finfo(np.float64).eps**(1 / 2)
        elif accuracy_order == 2:
            eps = np.finfo(np.float64).eps**(1 / 3)
    dtype = np.float128
    h = _step_sizes(x, eps, typical_x=typical_x, use_always_typical_x=use_always_typical_x, bounds=bounds, both_directions=accuracy_order >= 2, dtype=dtype)

    # define function for evaluation
    def f_x_h(*h_indices):
        return _f_x_h(f, x, h, *h_indices, dtype=dtype)

    # init values
    n = len(x)

    # calculate f(x)
    if f_x is None and accuracy_order == 1:
        f_x = f(x)
    if f_x is not None:
        f_x = np.asanyarray(f_x, dtype=dtype)
        df = np.zeros((n,) + f_x.shape, dtype=dtype)
    else:
        df = None

    # calculate df
    for i in range(n):
        if accuracy_order == 1:
            df[i] = f_x_h((i,)) - f_x
            df[i] /= h[i]
        elif accuracy_order == 2:
            f_x_h_0 = f_x_h((i, 0))
            f_x_h_1 = f_x_h((i, 1))
            if df is None:
                df = np.zeros((n,) + f_x_h_0.shape, dtype=dtype)
            df[i] = f_x_h_0 - f_x_h_1
            df[i] /= np.sum(np.abs(h[i]), dtype=dtype)
        else:
            assert False

    return df


def second_derivative(f, x, f_x=None, typical_x=None, bounds=None, eps=None, use_always_typical_x=True, accuracy_order=2):
    assert accuracy_order in (1, 2, 4)

    # convert x
    x = np.asanyarray(x)

    # calculate step size
    if eps is None:
        if accuracy_order == 1:
            eps = np.finfo(np.float64).eps**(1 / 3)
        elif accuracy_order == 2:
            eps = np.finfo(np.float64).eps**(1 / 4)
        elif accuracy_order == 4:
            eps = np.finfo(np.float64).eps**(1 / 6)
        else:
            assert False
    dtype = np.float128
    h = _step_sizes(x, eps, typical_x=typical_x, use_always_typical_x=use_always_typical_x, bounds=bounds, both_directions=accuracy_order >= 2, dtype=dtype)
    if accuracy_order > 2:
        h2 = _step_sizes(x, 2 * eps, typical_x=typical_x, use_always_typical_x=use_always_typical_x, bounds=bounds, both_directions=True, dtype=dtype)

    # define function for evaluation
    def f_x_h(h, *h_indices):
        return _f_x_h(f, x, h, *h_indices, dtype=dtype)

    # init values
    n = len(x)

    # calculate f(x)
    if f_x is None:
        f_x = f(x)
    f_x = np.asanyarray(f_x, dtype=dtype)

    # create df array
    df_shape = (n, n) + f_x.shape
    df = np.empty(df_shape, dtype=dtype)

    # calculate f for changes in a single component
    f_single_h_shape = h.shape + f_x.shape
    f_single_h = np.empty(f_single_h_shape, dtype=dtype)
    for i, h_i in np.ndenumerate(h):
        f_single_h[i] = f_x_h(h, i)

    # calculate df
    if accuracy_order == 1:
        for i in range(n):
            for j in range(i + 1):
                df[i, j] = (f_x_h(h, (i,), (j,)) + f_x) - (f_single_h[i] + f_single_h[j])
                df[i, j] /= np.abs(h[i]).mean(dtype=dtype) * np.abs(h[j]).mean(dtype=dtype)
    elif accuracy_order == 2:
        for i in range(n):
            # diagonal values
            df[i, i] = (f_single_h[i, 0] + f_single_h[i, 1]) - 2 * f_x
            df[i, i] /= np.abs(h[i]).mean(dtype=dtype)**2
            # off diagonal values
            for j in range(i):
                df[i, j] = (2 * f_x + f_x_h(h, (i, 0), (j, 0)) + f_x_h(h, (i, 1), (j, 1))) - (f_single_h[i, 0] + f_single_h[i, 1] + f_single_h[j, 0] + f_single_h[j, 1])
                df[i, j] /= 2 * np.abs(h[i]).mean(dtype=dtype) * np.abs(h[j]).mean(dtype=dtype)
    elif accuracy_order == 4:
        for i in range(n):
            # diagonal values
            df[i, i] = (16 * (f_x_h(h, (i, 0)) + f_x_h(h, (i, 1)))) - (30 * f_x + f_x_h(h2, (i, 0)) + f_x_h(h2, (i, 1)))
            df[i, i] /= 12 * np.abs(h[i]).mean(dtype=dtype)**2
            # off diagonal values
            for j in range(i):
                df[i, j] = np.nan
    else:
        assert False

    # make symmetric
    for i in range(n):
        for j in range(i):
            df[j, i] = df[i, j]

    # return
    return df
