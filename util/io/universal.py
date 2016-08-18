import util.io.np
import util.io.object


def save(file, o):
    if util.io.np.is_file(file):
        util.io.np.save(file, o)
    else:
        util.io.object.save(file, o)

def load(file):
    if util.io.np.is_file(file):
        return util.io.np.load(file)
    else:
        return util.io.object.load(file)