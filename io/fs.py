import os
import re
import stat
import time

import util.logging
logger = util.logging.logger


## file extension

def has_file_ext(file, ext):
    actual_ext = os.path.splitext(file)[1]
    if not ext.startswith('.'):
        actual_ext = actual_ext[1:]
    return actual_ext == ext


def add_file_ext_if_needed(file, ext):
    if has_file_ext(file, ext):
        return file
    else:
        if not ext.startswith('.'):
            ext = '.' + ext
        return file + ext


## get files

def filter_files(path, condition, recursive=False):
    ## get all filenames
    if os.path.exists(path):
        if recursive:
            path_filenames = []
            walk_path(path, path_filenames.append, path_filenames.append, topdown=True, exclude_dir=True)
        else:
            path_filenames = os.listdir(path)
    else:
        path_filenames = []
    
    ## filter filenames
    filtered_files = []
    for path_filename in path_filenames:
        path_file = os.path.join(path, path_filename)
        if condition(path_file):
            filtered_files += [path_file]

    return filtered_files

def get_dirs(path=os.getcwd(), with_links=True):
    if with_links:
        dirs = filter_files(path, os.path.isdir, recursive=False)
    else:
        fun = lambda file: os.path.isdir(file) and not os.path.islink(file)
        dirs = filter_files(path, fun, recursive=False)

    return dirs

def get_files(path=os.getcwd(), regular_expression=None):
    if regular_expression is None:
        condition = os.path.isfile
    else:
        regular_expression_object = re.compile(regular_expression)
        def condition(file):
            filename = os.path.split(file)[1]
            return os.path.isfile(file) and regular_expression_object.match(filename) is not None

    return filter_files(path, condition, recursive=False)


## permissions

def file_mode(file):
    return os.stat(file)[stat.ST_MODE]

def file_permission(file):
    permission_octal_string = oct(file_mode(file))[-3:]
    permission_int = int(permission_octal_string, 8)
    return permission_int

def remove_write_permission(file):
    permission_old = file_mode(file)
    permission_new = permission_old
    for write_permission in (stat.S_IWUSR, stat.S_IWGRP, stat.S_IWOTH):
        permission_new = permission_new & ~ write_permission
    os.chmod(file, permission_new)
    logger.debug('Removing write permission of file {}. Mode changed from {} to {}.'.format(file, oct(permission_old)[-3:],oct( permission_new)[-3:]))

def add_write_permission(file):
    permission_old = file_mode(file)
    permission_new = permission_old
    for read_permission, write_permission in ((stat.S_IRUSR, stat.S_IWUSR), (stat.S_IRGRP, stat.S_IWGRP), (stat.S_IROTH, stat.S_IWOTH)):
        if permission_new & read_permission:
            permission_new = permission_new | write_permission
    os.chmod(file, permission_new)
    logger.debug('Adding write permission to file {}. Mode changed from {} to {}.'.format(file, oct(permission_old)[-3:], oct(permission_new)[-3:]))
    

def make_read_only(*files, not_exist_ok=False):
    for file in files:
        if not_exist_ok:
            try:
                remove_write_permission(file)
            except FileNotFoundError:
                logger.debug('File {} not existing, but this is okay.'.format(file))
                pass
        else:
            remove_write_permission(file)

def make_read_only_recursively(path, exclude_dir=True):
    logger.debug('Making recursively all files in {} read-only.'.format(path))
    file_function = lambda file: make_read_only(file)
    dir_function = lambda file: make_read_only(file)
    walk_path(path, file_function, dir_function, exclude_dir=exclude_dir, topdown=False)


def make_writable(file, not_exist_ok=False):
    if not_exist_ok:
        try:
            add_write_permission(file)
        except FileNotFoundError:
            logger.debug('File {} not existing, but this is okay.'.format(file))
            pass
    else:
        add_write_permission(file)


def make_writable_recursively(path, exclude_dir=True):
    logger.debug('Making recursively all files in {} writeable.'.format(path))
    file_function = lambda file: make_writable(file)
    dir_function = lambda file: make_writable(file)
    walk_path(path, file_function, dir_function, exclude_dir=exclude_dir, topdown=False)


## remove

def remove_dir(path, force=False):
    if force:
        make_writable(path)
    os.rmdir(path)

def remove_file(file, force=False, not_exist_okay=False):
    if not not_exist_okay or os.path.exists(file):
        if force:
            make_writable(file)
        os.remove(file)

def remove_recursively(path, force=False, exclude_dir=False):
    if force:
        make_writable_recursively(path, exclude_dir=exclude_dir)
    walk_path(path, os.remove, os.rmdir, exclude_dir=exclude_dir, topdown=False)


## make

def makedirs(file, exist_ok=True):
    dir = os.path.dirname(file)
    if dir is not None:
        if not os.path.exists(dir):
            logger.debug('Creating recrusive directories {}.'.format(dir))
            os.makedirs(dir, exist_ok=exist_ok)
    else:
        raise ValueError('Directories of file {} is not creatable.'.format(file))


##

def flush_and_close(file):
    file.flush()
    os.fsync(file.fileno())
    file.close()
    while not os.path.exists(file.name):
        logger.warning('File {} is not available after flush and fsync. Waiting.'.format(file.name))
        time.sleep(1)


def walk_path(path, file_function, dir_function, topdown=True, exclude_dir=True):
    for (dirpath, dirnames, filenames) in os.walk(path, topdown=topdown):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                file_function(file)
            for dirname in dirnames:
                dir = os.path.join(dirpath, dirname)
                dir_function(dir)
    if not exclude_dir:
        dir_function(path)