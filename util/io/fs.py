import errno
import fnmatch
import os
import re
import stat
import time

import util.logging
logger = util.logging.logger


## file extension

def has_file_ext(file, ext):
    if not ext.startswith('.'):
        ext = '.' + ext
    return file.endswith(ext)


def add_file_ext_if_needed(file, ext):
    if has_file_ext(file, ext):
        return file
    else:
        if not ext.startswith('.'):
            ext = '.' + ext
        return file + ext


## walk

def walk_all_in_dir(dir, file_function=None, dir_function=None, topdown=True, exclude_dir=True):
    for (dirpath, dirnames, filenames) in os.walk(dir, topdown=topdown):
            if file_function is not None:
                for filename in filenames:
                    current_file = os.path.join(dirpath, filename)
                    file_function(current_file)
            if dir_function is not None:
                for dirname in dirnames:
                    current_dir = os.path.join(dirpath, dirname)
                    dir_function(current_dir)
    if not exclude_dir and dir_function is not None:
        dir_function(dir)


def walk_all_files_in_dir(dir, function, topdown=True):
    walk_all_in_dir(dir, file_function=function, dir_function=None, topdown=topdown, exclude_dir=True)


def walk_all_dirs_in_dir(dir, function, topdown=True, exclude_dir=True):
    walk_all_in_dir(dir, file_function=None, dir_function=function, topdown=topdown, exclude_dir=exclude_dir)


## get files

def find_with_condition_function(path, condition_function, exclude_files=False, exclude_dirs=False, use_absolute_filenames=False, recursive=False):
    ## use absolute path
    path = os.path.abspath(path)
    
    ## definde append filtered result function
    filtered_results = []
    
    def append_filtered(filename):
        if use_absolute_filenames:
            filename = os.path.join(path, filename)
        if condition_function(filename):
            filtered_results.append(filename)
    
    ## filter recursive
    if os.path.exists(path):
        if recursive:
            if not exclude_files:
                file_filter_function = append_filtered
            else:
                file_filter_function = None
            if not exclude_dirs:
                dir_filter_function = append_filtered
            else:
                dir_filter_function = None
            
            walk_all_in_dir(path, file_function=file_filter_function, dir_function=dir_filter_function, topdown=True, exclude_dir=True)
    
    ## filter not recursive
        else:
            for filename in os.listdir(path):
                append_filtered(filename)
            
            if exclude_files:
                if use_absolute_filenames:
                    filtered_results = [file for file in filtered_results if not os.path.isfile(file)]
                else:
                    filtered_results = [file for file in filtered_results if not os.path.isfile(os.path.join(path, file))]            
            if exclude_dirs:
                if use_absolute_filenames:
                    filtered_results = [file for file in filtered_results if not os.path.isdir(file)]
                else:
                    filtered_results = [file for file in filtered_results if not os.path.isdir(os.path.join(path, file))]     
    
    ## path is not existing
    else:
        filtered_results = []
    
    ## return 
    return filtered_results


def find_with_filename_pattern(path, filename_pattern, exclude_files=False, exclude_dirs=False, use_absolute_filenames=False, recursive=False):
    condition_function = lambda filename: fnmatch.fnmatch(filename, filename_pattern)
    filtered_results = find_with_condition_function(path, condition_function, exclude_files=exclude_files, exclude_dirs=exclude_dirs, use_absolute_filenames=use_absolute_filenames, recursive=recursive)
    return filtered_results


def find_with_regular_expression(path, regular_expression, exclude_files=False, exclude_dirs=False, use_absolute_filenames=False, recursive=False):
    regular_expression_object = re.compile(regular_expression)
    condition_function = lambda filename: regular_expression_object.fullmatch(filename)
    filtered_results = find_with_condition_function(path, condition_function, exclude_files=exclude_files, exclude_dirs=exclude_dirs, use_absolute_filenames=use_absolute_filenames, recursive=recursive)
    return filtered_results


def get_dirs(path=os.getcwd(), filename_pattern=None, use_absolute_filenames=False, recursive=False):
    if filename_pattern is None:
        filename_pattern = '*'
    dirs = find_with_filename_pattern(path, filename_pattern, exclude_files=True, use_absolute_filenames=use_absolute_filenames, recursive=recursive)
    return dirs

def get_files(path=os.getcwd(), filename_pattern=None, use_absolute_filenames=False, recursive=False):
    if filename_pattern is None:
        filename_pattern = '*'
    files = find_with_filename_pattern(path, filename_pattern, exclude_dirs=True, use_absolute_filenames=use_absolute_filenames, recursive=recursive)
    return files


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


def add_group_permissions(file, read=True, write=True, execute=True):
    permission_old = file_mode(file)
    permission_new = permission_old
    if read:
        permission_new = permission_new | stat.S_IRGRP
    if write:
        permission_new = permission_new | stat.S_IWGRP
    if execute:
        permission_new = permission_new | stat.S_IXGRP
    os.chmod(file, permission_new)
    logger.debug('Adding group permission (read={read}, write={write}, execute={execute}) to {file}. Mode changed from {permission_old} to {permission_new}.'.format(file=file, permission_old=oct(permission_old)[-3:], permission_new=oct(permission_new)[-3:], read=read, write=write, execute=execute))
    

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
    walk_all_in_dir(path, file_function, dir_function, exclude_dir=exclude_dir, topdown=False)


def make_writable(file, not_exist_ok=False):
    if not_exist_ok:
        try:
            add_write_permission(file)
        except FileNotFoundError:
            logger.debug('File {} does not exist, but this is okay.'.format(file))
            pass
    else:
        add_write_permission(file)


def make_writable_recursively(path, exclude_dir=True):
    logger.debug('Making recursively all files in {} writeable.'.format(path))
    file_function = lambda file: make_writable(file)
    dir_function = lambda file: make_writable(file)
    walk_all_in_dir(path, file_function, dir_function, exclude_dir=exclude_dir, topdown=False)


## remove

def _remove_general(remove_function, file, force=False, not_exist_okay=False):
    try:
        remove_function(file)
    except FileNotFoundError:
        if not not_exist_okay:
            raise
    except PermissionError:
        if force:
            (dir, filename) = os.path.split(file)
            make_writable(dir)
            remove_function(file)
        else:
            raise
    

def remove_dir(dir, force=False, not_exist_okay=False):
    _remove_general(os.rmdir, dir, force=force, not_exist_okay=not_exist_okay)


def remove_file(file, force=False, not_exist_okay=False):
    _remove_general(os.remove, file, force=force, not_exist_okay=not_exist_okay)


def remove_universal(file, force=False, not_exist_okay=False):
    try:
        remove_file(file, force=force, not_exist_okay=not_exist_okay)
    except IsADirectoryError:
        remove_dir(file, force=force, not_exist_okay=not_exist_okay)


def remove_recursively(dir, force=False, not_exist_okay=False, exclude_dir=False):
    try:
        remove_file(dir, force=force, not_exist_okay=not_exist_okay)
    except IsADirectoryError:
        walk_all_in_dir(dir, lambda file: remove_file(file, force=force, not_exist_okay=not_exist_okay), lambda dir: remove_dir(dir, force=force, not_exist_okay=not_exist_okay), exclude_dir=exclude_dir, topdown=False)


## utility functions

def flush_and_close(file):
    file.flush()
    os.fsync(file.fileno())
    file.close()
    while not os.path.exists(file.name):
        logger.warning('File {} is not available after flush and fsync. Waiting.'.format(file.name))
        time.sleep(1)


## fd functions

def fd_is_file(fd, file, not_exist_okay=False):
    ## get file stats
    try:
        file_stat = os.stat(file)
    except FileNotFoundError as e:
        if not_exist_okay:
            logger.debug('File {} does not exist, but this is okay.'.format(file))
            return False
        else:
            raise
    except OSError as e:
        if e.errno == errno.ESTALE and not_exist_okay:
            logger.debug('File {} does not exist, but this is okay.'.format(file))
            return False
        else:
            raise

    ## get fd stats
    try:
        fd_stat = os.fstat(fd)
    except OSError as e:
        if e.errno == errno.ESTALE and not_exist_okay:
            logger.debug('File referenced by file desciptor {} was removed, but this is okay.'.format(fd))
            return False
        else:
            raise
    
    ## check if same device and inode
    return fd_stat.st_dev == file_stat.st_dev and fd_stat.st_ino == file_stat.st_ino
     
