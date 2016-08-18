import numpy as np
import scipy.sparse.linalg

import util.logging
logger = util.logging.logger


class CGMethodError(Exception):

    def __init__(self, exit_code):
        self.exit_code = exit_code

    def __str__(self):
        return 'CG method error. Illegal input or breakdown. Exit code {}.'.format(self.exit_code)


# k = 1
# xk_prev = None

def cg(A, b, x0=None, tol=10**-5, maxiter=None):
    logger.debug('CG method starting with tolerance {} and max iterations {}.'.format(tol, maxiter))

    ## chose x0
    if x0 is None:
        x0 = b

    ## callback -> output
    k = 1
    A_b = A * b
    # xk_prev = np.copy(x0)
    def callback(xk):
        nonlocal k
        # nonlocal xk_prev
        # diff_abs = np.abs(xk - xk_prev)
        # diff_rel = diff_abs / np.abs(xk_prev)
        #
        # logger.debug('Iteration {}: abs: avg diff = {:e}, averaged rel diff = {:e}, 2-norm abs diff = {:e}, 2-norm rel diff = {:e}'.format(k, diff_abs.mean(), diff_rel.mean(), np.linalg.norm(diff_abs, ord=2), np.linalg.norm(diff_rel, ord=2)))
        # xk_prev = np.copy(xk)

        A_xk = A * xk
        f = np.sum((xk- b) * (A_xk - A_b))
        residuum = np.abs(A_xk - b)
        logger.debug('Iteration {}: f = {:e}, avg residuum = {:e}, max residuum = {:e}, 2-norm residuum = {:e}'.format(k, f, residuum.mean(), residuum.max(), np.linalg.norm(residuum, ord=2)))
        k += 1

    ## run
    (x, exit_code) = scipy.sparse.linalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)

    ## check exit code
    if exit_code == 0:
        logger.debug('CG method converged after {} iterations.'.format(k))
        return x
    elif exit_code > 0:
        logger.warning('CG method exited. Tolerance {} not achieved after {} iterations. Exit code {}'.format(tol, k, exit_code))
        return x
    else:
        e = CGMethodError(exit_code)
        logger.error(str(e))
        raise e
