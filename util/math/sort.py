import numpy as np

import util.math.check


def lex_sorted_indices(a, order=1):
    util.math.check.ndim(a, 2)

    n = a.shape[1]
    if order > 0:
        axes = range(n)
    else:
        axes = range(n-1, -1, -1)

    return lex_sorted_indices_by_axes(a, axes)


def lex_sorted_indices_by_axes(a, axes):
    util.math.check.ndim(a, 2)

    lex_list = []
    for i in range(len(axes)-1, -1, -1):
        lex_list.append(a[:, axes[i]])
    sorted_indices = np.lexsort(lex_list)

    return sorted_indices
