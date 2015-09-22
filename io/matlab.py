import scipy.io

def save(file, array, value_name, oned_as='column'):
    scipy.io.savemat(file, {value_name: array}, oned_as=oned_as)

def load(file, value_name):
    return scipy.io.loadmat(file, squeeze_me=True)[value_name]