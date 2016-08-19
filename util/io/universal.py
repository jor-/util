import os

import util.io.np
import util.io.object
import util.io.matrix.sparse


def save(file, o, make_dirs=True):
    if make_dirs:
        dir = os.path.dirname(file)
        if len(dir) > 0:
            os.makedirs(dir, exist_ok=True)
    
    if util.io.matrix.sparse.is_spmatrix(o):
        util.io.matrix.sparse.save(file, o)
    elif util.io.np.is_file(file):
        util.io.np.save(file, o)
    else:
        util.io.object.save(file, o)


def load(file):
    if util.io.matrix.sparse.is_file(file):
        return util.io.matrix.sparse.load(file)
    elif util.io.np.is_file(file):
        return util.io.np.load(file)
    else:
        return util.io.object.load(file)