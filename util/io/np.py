import os

import numpy as np

import util.io.fs


FILE_EXT = '.npy'
COMPRESSED_FILE_EXT = '.npz'


def get_ext(compressed=False):
    if compressed:
        return COMPRESSED_FILE_EXT
    else:
        return FILE_EXT


def is_file(file, compressed=None):
    if compressed is None:
        return is_file(file, compressed=False) or is_file(file, compressed=True)
    else:
        ext = get_ext(compressed=compressed)
        return util.io.fs.has_file_ext(file, ext)


def add_file_ext(file, compressed=False):
    ext = get_ext(compressed=compressed)
    return util.io.fs.add_file_ext_if_needed(file, ext)


def save(file, values, compressed=None, make_read_only=False, overwrite=False, create_path_if_not_exists=True):
    ## check input values
    is_values_dict = isinstance(values, dict)
    is_values_tuple = (isinstance(values, tuple) or isinstance(values, list)) and all(map(lambda a: isinstance(a, np.ndarray), values))
    
    if not is_values_dict and not is_values_tuple:
        values = np.asanyarray(values)
    
    if is_file(file, compressed=False):
        if compressed:
            raise ValueError('Compressed values can only be stored in "npz" file format, but the file {} has ending "npy".'.format(file))
        if is_values_dict or is_values_tuple:
            raise ValueError('Multiple values {} can only be stored in "npz" file format, but the file {} has ending "npy".'.format(file))
    
    ## set file ext
    use_npz = is_values_dict or is_values_tuple or (compressed is not None and compressed) or is_file(file, compressed=True)    
    file = add_file_ext(file, compressed=use_npz)
    
    ## set compressed if not passed
    if compressed is None:
        compressed = use_npz
    
    ## create dir
    if create_path_if_not_exists:
        (dir, filename) = os.path.split(file)
        os.makedirs(dir, exist_ok=True)
    
    ## remove if overwrite
    if overwrite:
        util.io.fs.remove_file(file, force=True, not_exist_okay=True)
    
    ## save
    if use_npz:
        if compressed:
            np.savez_compressed(file, values)
        else:
            np.savez(file, values)
    else:
        np.save(file, values)
    
    ## make read only
    if make_read_only:
        util.io.fs.make_read_only(file)
    

def load(file, mmap_mode=None):
    ## load value
    value = np.load(file,mmap_mode=mmap_mode)
    ## unpack value if npz file with one variable
    try:
        value.keys
    except AttributeError:
        pass
    else:
        keys = value.keys()
        if len(keys) == 1:
            value = value[keys[0]]
    ## return
    return value


def save_txt(file, values, format_string=None, make_read_only=False, overwrite=False, create_path_if_not_exists=True):
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
        util.io.fs.remove_file(file, force=True, not_exist_okay=True)
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


def save_np_and_txt(file, array, compressed=None, make_read_only=False, overwrite=False, create_path_if_not_exists=True, format_string='%.6g'):
    (file_without_ext, file_ext) = os.path.splitext(file)
    (dir, filename_without_ext) = os.path.split(file_without_ext)
    os.makedirs(dir, exist_ok=True)

    file_np = add_file_ext(file_without_ext, compressed=compressed)
    file_txt = file_without_ext + '.txt'

    array = np.asarray(array)

    save(file_np, array, compressed=compressed, make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)
    save_txt(file_txt, array, format_string=format_string, make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)
