import scipy.optimize
import numpy as np

import util.math.optimize.universal
import util.math.finite_differences

import util.logging
logger = util.logging.logger




def minimize(f, x0, jac=None, bounds=None, ineq_constraints=None, ineq_constraints_jac=None, global_method='random_start', global_iterations=1, global_stepsize=0.5, global_stepsize_update_interval=10, local_max_iterations=10**6, local_max_fun_evals=10**6):
    ## prepare input
    if jac is None:
        jac = lambda x: util.math.finite_differences.calculate(f, x, bounds=bounds, accuracy_order=2)[0]

    x0 = np.asarray(x0)
    if x0.ndim == 1:
        n = x0.shape[0]
        x0 = x0.reshape(1, n)
        m = 1
    else:
        n = x0.shape[1]
        m = x0.shape[0]

    if bounds is not None and len(bounds) != n:
        raise ValueError('x0 and bounds must have the same length, but their length are {} and {}.'.format(n, len(bounds)))

    if not global_method in ['random_start', 'basin_hopping']:
        raise ValueError('global_method {} has to be in {}'.format(global_method, 'random_start', 'basin_hopping'))

    ## chose local methods
    if ineq_constraints is not None:
        local_methods = ('SLSQP',)
    elif bounds is not None:
#         local_methods = ('SLSQP', 'L-BFGS-B', 'TNC')
        local_methods = ('L-BFGS-B', 'TNC')
    else:
        local_methods = ('SLSQP', 'BFGS', 'TNC')

    logger.debug('Glocal method {} chosen with local methods {}.'.format(global_method, local_methods))
    if bounds is not None:
        logger.debug('Using bounds {} for the optimization.'.format(bounds))

    ## prepare local options
    if ineq_constraints is None:
        constraints = ()
    else:
        constraints_dict = ({'type': 'ineq', 'fun': ineq_constraints})
        if ineq_constraints_jac is not None:
            constraints_dict['jac'] = ineq_constraints_jac
        constraints = [constraints_dict]

# #     #
#     if bounds is not None:
#         constraints = []
#         def lower_bound_contraint_dict(i):
#             def fun(x):
#                 return x[i] - bounds[i][0]
#             def jac(x):
#                 df = np.zeros(x.shape)
#                 df[i] = x[i]
#                 return df
#             return {'type': 'ineq', 'fun': fun, 'jac': jac}
#         def upper_bound_contraint_dict(i):
#             def fun(x):
#                 return bounds[i][1] - x[i]
#             def jac(x):
#                 df = np.zeros(x.shape)
#                 df[i] = - x[i]
#                 return df
#             return {'type': 'ineq', 'fun': fun, 'jac': jac}
#         for i in range(n):
#             constraints.append(lower_bound_contraint_dict(i))
#             constraints.append(upper_bound_contraint_dict(i))
# #     #

    disp = util.logging.is_debug()
    if disp:
        logger.debug('Debug output in optimization enabled.')
    else:
        logger.debug('Debug output in optimization disabled.')

    local_minimizer_options = {'method':'', 'jac':jac, 'bounds':bounds, 'constraints':constraints, 'options':{'maxiter': local_max_iterations, 'maxfun': local_max_fun_evals, 'disp':disp}}
#     local_minimizer_options = {'method':'', 'jac':jac, 'constraints':constraints, 'options':{'maxiter': local_max_iterations, 'maxfun': local_max_fun_evals, 'disp':disp}}
#     local_minimizer_options = {'method':'', 'jac':jac, 'bounds':bounds, 'options':{'maxiter': local_max_iterations, 'maxfun': local_max_fun_evals, 'disp':disp}}

    ## run optimization
    x_min = None
    f_min = np.inf

    ## random start point method
    if global_method == 'random_start':

        ## prepare start points
        if m < global_iterations and (bounds is None or not np.all(np.isfinite(bounds))):
            raise ValueError('Bounds must be finite if random start point method is chosen and the global iterations is greater then the number of starting points, but they are {}.'.format(bounds))

        x0s = np.empty([global_iterations, n])
        for i in range(global_iterations):
            if i < m:
                x0s[i] = x0[i]
            else:
                accept_x0 = False
                while not accept_x0:
                    for j in range(n):
                        x0s[i, j] = np.random.uniform(low=bounds[j][0], high=bounds[j][1], size=1)
                    accept_x0 = ineq_constraints(x0s[i]) >= 0

        ## run all local methods and from all start points
        for local_method in local_methods:
            local_minimizer_options['method'] = local_method
            for x0 in x0s:
                result = scipy.optimize.minimize(f, x0, **local_minimizer_options)
                logger.debug('Minimization with local method {} started from {} stopped with the following results: {}'.format(local_method, x0, result))

                if result.success and result.fun < f_min:
                    x_min = result.x
                    f_min = result.fun

    ## basin hopping method
    elif global_method == 'basin_hopping':
        x0 = x0[0]

        ## incorporate bounds and constarints
        if bounds is not None:
            global_bounds = np.empty([2, n])
            for i in range(n):
                if np.isreal(bounds[i][0]):
                    global_bounds[0, i] = bounds[i][0]
                else:
                    global_bounds[0, i] = - np.inf
                if np.isreal(bounds[i][1]):
                    global_bounds[1, i] = bounds[i][1]
                else:
                    global_bounds[1, i] = np.inf

            def bounds_test(**kwargs):
                x = kwargs['x_new']
                okay = bool(np.all(global_bounds[0] <= x) and np.all(x <= global_bounds[1]))
                return okay
        else:
            def bounds_test(**kwargs):
                return True

        if ineq_constraints is not None:
            def constraints_test(**kwargs):
                x = kwargs['x_new']
                okay = bounds_test(**kwargs) and bool(ineq_constraints(x) >= 0)
                return okay
        else:
            constraints_test = bounds_test

        ## run all local methods
        for local_method in local_methods:
            local_minimizer_options['method'] = local_method
            result = scipy.optimize.basinhopping(f, x0, minimizer_kwargs=local_minimizer_options, niter=global_iterations, disp=disp, accept_test=constraints_test, stepsize=global_stepsize, interval=global_stepsize_update_interval)
            logger.debug('Minimization basin hopping with local method {} started from {} stopped with the following results: {}'.format(local_method, x0, result))
            if result.fun < f_min:
                x_min = result.x
                f_min = result.fun


    ## return result
    if x_min is not None:
        return (x_min, f_min)
    else:
        raise util.math.optimize.universal.OptimizationError(result)

