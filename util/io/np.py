import os

import numpy as np

import util.io.fs


FILE_EXT = '.npy'
COMPRESSED_FILE_EXT = '.npz'
TEXT_FILE_EXT = '.txt'


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


def _save_generic(file, values, save_function, make_read_only=False, overwrite=False, create_path_if_not_exists=True):

    # create dir
    if create_path_if_not_exists:
        (dir, filename) = os.path.split(file)
        os.makedirs(dir, exist_ok=True)

    # check if file already exists
    saved = False
    if os.path.exists(file):
        file_values = load_np_or_txt(file)
        if np.all(values == file_values):
            util.logging.warn('The value was already saved in {}'.format(file))
            saved = True
        elif overwrite:
            util.io.fs.remove_file(file, force=True, not_exist_okay=True)

    # save
    if not saved:
        save_function(file, values)

    # make read only
    if make_read_only:
        util.io.fs.make_read_only(file)


def save(file, values, compressed=None, make_read_only=False, overwrite=False, create_path_if_not_exists=True):
    # check input values
    is_values_dict = isinstance(values, dict)
    is_values_tuple = (isinstance(values, tuple) or isinstance(values, list)) and all(map(lambda a: isinstance(a, np.ndarray), values))

    if not is_values_dict and not is_values_tuple:
        values = np.asanyarray(values)

    # check if have to use npz
    use_npz = is_values_dict or is_values_tuple or compressed or is_file(file, compressed=True)

    # set compressed if not passed
    if compressed is None:
        compressed = use_npz

    # check file format
    if not is_file(file, compressed=compressed):
        if compressed:
            raise ValueError('Compressed values can only be stored files with "npz" ending, but the file is {}.'.format(file))
        if not compressed:
            raise ValueError('Uncompressed values can only be stored files with "np" ending, but the file is {}.'.format(file))
    if (is_values_dict or is_values_tuple) and is_file(file, compressed=False):
        raise ValueError('Multiple values {} can only be stored in files with "npz" ending, but the file is {}.'.format(file))

    # save
    def save_function(file, values):
        if use_npz:
            if compressed:
                np.savez_compressed(file, values)
            else:
                np.savez(file, values)
        else:
            np.save(file, values)

    _save_generic(file, values, save_function,
                  make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)


def save_txt(file, values, format_string=None, make_read_only=False, overwrite=False, create_path_if_not_exists=True):
    # append file extension if needed
    if not file.endswith(TEXT_FILE_EXT):
        file = file + TEXT_FILE_EXT

    # cast value to array
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)

    # chose format string if not passed
    if format_string is None:
        if values.dtype.kind == 'i':
            format_string = '%d'
        else:
            assert values.dtype.kind == 'f'
            format_string = '%.{}e'.format(np.finfo(values.dtype).precision)

    # save
    def save_function(file, values):
        np.savetxt(file, values, fmt=format_string)

    _save_generic(file, values, save_function,
                  make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)


def load(file, mmap_mode=None):
    # load values
    values = np.load(file, mmap_mode=mmap_mode)

    # unpack values if npz file with one array
    try:
        values.keys
    except AttributeError:
        pass
    else:
        keys = values.keys()
        if len(keys) == 1:
            values = values[keys[0]]

    # if zero dimensional array, unpack value
    if values.ndim == 0:
        values = values.reshape(1)

    # return
    return values


def load_txt(file):
    # load values from file
    values = np.loadtxt(file)

    # cast to int if possible
    values_int = values.astype(np.int)
    if (values_int == values).all():
        values = values_int

    # if zero dimensional array, unpack value
    if values.ndim == 0:
        values = values.reshape(1)

    return values


def save_np_or_txt(file, values, make_read_only=False, overwrite=False, create_path_if_not_exists=True, save_as_np=True, save_as_txt=False):
    file_without_ext, ext = os.path.splitext(file)
    if save_as_np:
        if ext == FILE_EXT or ext == COMPRESSED_FILE_EXT:
            file_np = file
        else:
            file_np = file_without_ext
        save(file_np, values, make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)
    if save_as_txt:
        if ext == TEXT_FILE_EXT:
            file_txt = file
        else:
            file_txt = file_without_ext
        save_txt(file_txt, values, make_read_only=make_read_only, overwrite=overwrite, create_path_if_not_exists=create_path_if_not_exists)


def load_np_or_txt(file, mmap_mode=None):
    if file.endswith(TEXT_FILE_EXT):
        return load_txt(file)
    elif file.endswith(FILE_EXT) or file.endswith(COMPRESSED_FILE_EXT):
        return load(file, mmap_mode=mmap_mode)
    else:
        raise ValueError('File {} has unknown file extension.'.format(file))
