
def ndim(a, n):
    if a.ndim != n:
        raise ValueError('The passed array must have {} dims, but it has {}.'.format(n, a.ndim))