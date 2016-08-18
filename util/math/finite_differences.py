import numpy as np



def calculate(f, x, f_x=None, typical_x=None, bounds=None, accuracy_order=2, eps=np.spacing(1)):
    ## init unpassed values
    if f_x is None:
        f_x =  f(x)
    if typical_x is None:
        typical_x = np.ones_like(x)
    elif not len(x) == len(typical_x):
        raise ValueError('x and typical_x must have the same length but their length are {} and {}.'.format(len(x), len(typical_x)))
    if bounds is None:
        bounds = ((-np.inf, np.inf),) * len(x)

    ## set h factors according to accuracy
    if accuracy_order == 1:
        h_factors = (1,)
        eta = eps**(1/2)
    elif accuracy_order == 2:
        h_factors = (1, -1)
        eta = eps**(1/3)
    else:
        raise ValueError('Accuracy order {} not supported.'.format(accuracy_order))

    ## init values
    f_x = np.asarray(f_x)
    try:
        l = len(f_x)
    except TypeError:
        l = 1
    n = len(x)
    m = len(h_factors)
    df = np.zeros([l, n])

    ## for each x dim
    for i in range(n):
        h = np.empty(m)
        ## for each h factor
        for j in range(m):
            ## calculate h
            h[j] = eta * np.max([np.abs(x[i]), np.abs(typical_x[i])]) * h_factors[j]
            if x[i] < 0:
                h[j] = - h[j]
            x_h = np.copy(x)
            x_h[i] += h[j]

            ## consider bounds
            lower_bound = bounds[i][0]
            upper_bound = bounds[i][1]
            violates_lower_bound = x_h[i] < lower_bound
            violates_upper_bound = x_h[i] > upper_bound

            if accuracy_order == 1:
                if violates_lower_bound or violates_upper_bound:
                    h[j] *= -1
                    x_h[i] = x[i] + h[j]
            else:
                if violates_lower_bound or violates_upper_bound:
                    if violates_lower_bound:
                        x_h[i] = lower_bound
                    else:
                        x_h[i] = upper_bound

            ## eval f and add to df
            h[j] = x_h[i] - x[i]
            f_x_h = f(x_h)

            df[:, i] += (-1)**j * f_x_h

        ## calculate df_i
        if accuracy_order == 1:
            df[:, i] -= f_x
            df[:, i] /= h
        else:
            df[:, i] /= np.sum(np.abs(h))

    return df
