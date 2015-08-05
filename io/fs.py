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

def filter_files(path, condition):
    filtered_files = []
    
    try:
        path_filenames = os.listdir(path)
    except OSError:
        return []
    for path_filename in path_filenames:
        path_file = os.path.join(path, path_filename)
        if condition(path_file):
            filtered_files += [path_file]
    
    return filtered_files

def get_dirs(path=os.getcwd(), with_links=True):
    if with_links:
        dirs = filter_files(path, os.path.isdir)
    else:
        fun = lambda file: os.path.isdir(file) and not os.path.islink(file)
        dirs = filter_files(path, fun)
    
    return dirs

def get_files(path=os.getcwd(), regular_expression=None):
    if regular_expression is None:
        condition = os.path.isfile
    else:
        regular_expression_object = re.compile(regular_expression)
        def condition(file):
            filename = os.path.split(file)[1]
            return os.path.isfile(file) and regular_expression_object.match(filename) is not None
    
    return filter_files(path, condition)


## permissions

def make_read_only(*files, not_exist_ok=False):
    for file in files:
        logger.debug('Making {} read-only.'.format(file))
        if not_exist_ok:
            try:
                os.chmod(file, stat.S_IRUSR)
            except FileNotFoundError:
                logger.debug('File {} not existing, but this is okay.'.format(file))
                pass
        else:
            os.chmod(file, stat.S_IRUSR)

def make_read_only_recursively(path, exclude_dir=True):
    logger.debug('Making recursively all files in {} read-only.'.format(path))
    
    file_function = lambda file: os.chmod(file, stat.S_IRUSR)
    dir_function = lambda file: os.chmod(file, stat.S_IRUSR | stat.S_IXUSR)
    walk_path_bottom_up(path, file_function, dir_function, exclude_dir=exclude_dir)


def make_writable(file, not_exist_ok=False):
    logger.debug('Making {} writable.'.format(file))
    if not_exist_ok:
        try:
            os.chmod(file, get_file_permission(file) | stat.S_IWUSR)
        except FileNotFoundError:
            logger.debug('File {} not existing, but this is okay.'.format(file))
            pass
    else:
        os.chmod(file, get_file_permission(file) | stat.S_IWUSR)
        

def make_writable_recursively(path, exclude_dir=True):
    logger.debug('Making recursively all files in {} writeable.'.format(path))
    
    file_function = lambda file: os.chmod(file, stat.S_IRUSR | stat.S_IWUSR)
    dir_function = lambda file: os.chmod(file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    walk_path_bottom_up(path, file_function, dir_function, exclude_dir=exclude_dir)


def get_file_permission(file):
    permission_octal_string = oct(os.stat(file).st_mode)[-3:]
    permission_int = int(permission_octal_string, 8)
    return permission_int


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
    walk_path_bottom_up(path, os.remove, os.rmdir, exclude_dir=exclude_dir)


# ## extend
# def extend_filename(file):
#     file = os.path.expanduser(file)
#     file = os.path.expandvars(file)
#     if 


##

def flush_and_close(file):
    file.flush()
    os.fsync(file.fileno())
    file.close()
    while not os.path.exists(file.name):
        logger.warning('File {} is not available after flush and fsync. Waiting.'.format(file.name))
        time.sleep(1)


def makedirs_if_not_exists(file):
    dir = os.path.dirname(file)
    if not os.path.exists(dir):
        os.makedirs(dir)
        logger.debug('directory {} created'.format(dir))


def walk_path_bottom_up(path, file_function, dir_function, exclude_dir=True):
    for (dirpath, dirnames, filenames) in os.walk(path, topdown=False):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                file_function(file)
            for dirname in dirnames:
                dir = os.path.join(dirpath, dirname)
                dir_function(dir)
    if not exclude_dir:
        dir_function(path)