import os

import numpy as np

import util.io.fs


FILE_EXT = 'npy'

def is_file(file):
    return util.io.fs.has_file_ext(file, FILE_EXT)

def make_file(file):
    return util.io.fs.add_file_ext_if_needed(file, FILE_EXT)



def save(file, values, make_read_only=False, overwrite=False, create_path_if_not_exists=True):
    file = make_file(file)
    values = np.asarray(values)

    if create_path_if_not_exists:
        (dir, filename) = os.path.split(file)
        os.makedirs(dir, exist_ok=True)
    if overwrite:
        util.io.fs.make_writable(file, not_exist_ok=True)
    np.save(file, values)
    if make_read_only:
        util.io.fs.make_read_only(file)


def load(file):
    return np.load(file)



def save_txt(values, file, format_string=None, make_read_only=False, overwrite=False, create_path_if_not_exists=True):
    values = np.asarray(values)

    if len(values.shape) == 0:
        values = values.reshape(1)

    ## chose format string if not passed
    if format_string is None:
        if values.dtype == np.int:
            format_string = '%d'
        else:
            format_string = '%.18e'

    if create_path_if_not_exists:
        (dir, filename) = os.path.split(file)
        os.makedirs(dir, exist_ok=True)
    if overwrite:
        util.io.fs.make_writable(file, not_exist_ok=True)
    np.savetxt(file, values, fmt=format_string)
    if make_read_only:
        util.io.fs.make_read_only(file)



def load_txt(file):
    ## load values from file
    values = np.loadtxt(file)

    ## cast to int if possible
    values_int = values.astype(np.int)
    if (values_int == values).all():
        values = values_int

    ## if only one value return pure value
    if values.size == 1:
        values = values[0]

    return values



def save_npy_and_txt(array, file, make_read_only=False, overwrite=False, create_path_if_not_exists=True, format_string='%.6g'):
    (file_without_ext, file_ext) = os.path.splitext(file)
    (dir, filename_without_ext) = os.path.split(file_without_ext)
    os.makedirs(dir, exist_ok=True)

    file_npy = file_without_ext + '.npy'
    file_txt = file_without_ext + '.txt'

    array = np.asarray(array)

    save(file_npy, array, make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)
    save_txt(array, file_txt, format_string=format_string, make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)
