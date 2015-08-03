try:
    import util.petsc.with_petsc4py as petsc_module
except ImportError:
    import util.petsc.without_petsc4py as petsc_module



def load_petsc_vec_to_numpy_array(file):
    return petsc_module.load_petsc_vec_to_numpy_array(file)

def save_numpy_array_to_petsc_vec(vec, file):
    import util.petsc.without_petsc4py
    util.petsc.without_petsc4py.save_numpy_array_to_petsc_vec(vec, file)


def load_petsc_mat_to_array(file, dtype=float):
    return petsc_module.load_petsc_mat_to_array(file, dtype=dtype)