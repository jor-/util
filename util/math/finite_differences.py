import numpy as np



def calculate(f, x, f_x=None, typical_x=None, bounds=None, accuracy_order=2, eps=None, use_always_typical_x=True):
    x = np.asanyarray(x)
    
    ## init unpassed values
    if typical_x is None:
        typical_x = np.ones_like(x)
    elif not len(x) == len(typical_x):
        raise ValueError('x and typical_x must have the same length but their length are {} and {}.'.format(len(x), len(typical_x)))
    if bounds is None:
        bounds = ((-np.inf, np.inf),) * len(x)
    else:
        bounds = np.asanyarray(bounds)

    ## set h factors according to accuracy
    if accuracy_order == 1:
        h_factors = (1,)
        if eps is None:
            eps = np.spacing(1)**(1/2)
    elif accuracy_order == 2:
        h_factors = (1, -1)
        if eps is None:
            eps = np.spacing(1)**(1/3)
    else:
        raise ValueError('Accuracy order {} not supported.'.format(accuracy_order))
    
    ## calculate f(x) if needed
    if f_x is None and accuracy_order == 1:
        f_x =  f(x)

    ## init values
    n = len(x)
    m = len(h_factors)
    df = None

    ## for each x dim
    for i in range(n):
        h = np.empty(m, dtype=np.float64)
        
        ## for each h factor
        for j in range(m):
            ## calculate h
            if use_always_typical_x:
                typical_x_i = np.abs(typical_x[i])
            else:
                typical_x_i = np.max([np.abs(typical_x[i]), np.abs(x[i])])
            h[j] = h_factors[j] * eps * typical_x_i
            if accuracy_order == 1 and x[i] < 0:
                h[j] = - h[j]
            x_h = np.array(x, dtype=np.float64, copy=True)
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
            
            ## recalculate h   (improvement of accuracy of h)
            h[j] = x_h[i] - x[i]

            ## eval f and add to df
            f_x_h = f(x_h)
            f_x_h = np.asanyarray(f_x_h)

            if df is None:
                df_shape = (n,) + f_x_h.shape
                df = np.zeros(df_shape)
            
            df[i] += (-1)**j * f_x_h

        ## calculate df_i
        if accuracy_order == 1:
            df[i] -= f_x
            df[i] /= h
        else:
            df[i] /= np.sum(np.abs(h))

    return df
