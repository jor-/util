import fcntl
import errno
import os
import stat
import time

import util.io.fs
import util.io.filelock.general
import util.logging


class FileLock:

    def __init__(self, file, exclusive=True, timeout=None, sleep=0.5):
        self._lock_count = 0
        self._file = file
        self._exclusive = exclusive
        self._timeout = timeout
        self._sleep = sleep
        self._fd = None
        util.logging.debug('{}: Initiating file lock with timeout {} and sleep {}.'.format(self, timeout, sleep))

    def __str__(self):
        if self._exclusive:
            return 'File lock (exclusive) for {}'.format(self.file)
        else:
            return 'File lock (shared) for {}'.format(self.file)

    # *** lock file and state *** #

    @property
    def file(self):
        return self._file

    @property
    def lockfile(self):
        return self._file + '.lock'

    @property
    def exclusive(self):
        return self._exclusive

    @exclusive.setter
    def exclusive(self, exclusive):
        if self.is_locked() and self.exclusive != exclusive:
            raise ValueError('You have to relase the lock first to change its exclusive property.')
        self._exclusive = exclusive

    def is_locked(self, exclusive_is_okay=True, shared_is_okay=True):
        return self._lock_count > 0 and ((self._exclusive and exclusive_is_okay) or (not self._exclusive and shared_is_okay))

    def lock_object(self, exclusive=True):
        if not (not exclusive and self.exclusive and self.is_locked()):
            self.exclusive = exclusive
        return self

    # *** acquire and release *** #

    def _open_lockfile(self):
        if self._fd is None:
            # prepare flags and mode
            open_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_SYNC
            open_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
            # open
            try:
                self._fd = os.open(self.lockfile, open_flags, mode=open_mode)
            except OSError as e:
                if e.errno == errno.ESTALE:
                    util.logging.warning('{}: {}. Retrying to open lock file'.format(self, e))
                    self._open_lockfile()
                else:
                    raise
            else:
                util.logging.debug('{}: Lock file {} opened.'.format(self, self.lockfile))

        assert self._fd is not None

    def _close_lockfile(self):
        # close
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            util.logging.debug('{}: Lock file {} closed.'.format(self, self.lockfile))

        assert self._fd is None

    def _lock_lockfile(self, exclusive=True, blocking=True):
        assert self._fd is not None
        assert self._lock_count == 0

        # prepare lock flags
        if exclusive:
            lock_flags = fcntl.LOCK_EX
        else:
            lock_flags = fcntl.LOCK_SH
        if not blocking:
            lock_flags = lock_flags | fcntl.LOCK_NB
        # lock
        fcntl.lockf(self._fd, lock_flags)
        self._exclusive = exclusive
        self._lock_count = 1
        util.logging.debug('{}: Lock file {} locked with exclusive={}.'.format(self, self.lockfile, exclusive))

        assert self.is_locked(exclusive_is_okay=exclusive, shared_is_okay=not exclusive)

    def _unlock_lockfile(self):
        # unlock
        if self._lock_count > 0:
            assert self._fd is not None

            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._lock_count = 0
            util.logging.debug('{}: Lock file {} unlocked.'.format(self, self.lockfile))

        assert not self.is_locked()

    def _acquire(self, exclusive=True, timeout=None):
        # save start time for timeout
        if timeout is not None:
            start_time = time.time()

        util.logging.debug('{}: Acquiring with timeout {}.'.format(self, timeout))

        has_lock = False
        try:
            while not has_lock:
                # open file
                self._open_lockfile()

                # try to get lock
                try:
                    self._lock_lockfile(exclusive=exclusive, blocking=timeout is None)
                except BlockingIOError as e:
                    # check if regular timeout
                    if timeout is None and e.errno not in (errno.EAGAIN, errno.EACCES):
                        util.logging.warning('{}: Retrying to get lock because an BlockingIOError occured: {}'.format(self, e))
                else:
                    # if file was removed in between, open new file
                    if not util.io.fs.fd_is_file(self._fd, self.lockfile, not_exist_okay=True):
                        util.logging.debug('{}: Lock file {} was removed in beetween. Opening new lock file.'.format(self, self.lockfile))
                        self._unlock_lockfile()
                        self._close_lockfile()
                    # lock successfull
                    else:
                        util.logging.debug('{}: Fresh acquired.'.format(self))
                        has_lock = True

                # handle timout
                if not has_lock:
                    # if timeout reached, raise FileLockTimeoutError
                    if timeout is not None and time.time() >= (start_time + timeout):
                        util.logging.debug('{}: Could not be acquired. Timeout {} reached.'.format(self, timeout))
                        raise util.io.filelock.general.FileLockTimeoutError(self.lockfile, timeout)
                    # else wait
                    else:
                        time.sleep(self._sleep)
        except Exception:
            self._unlock_lockfile()
            self._close_lockfile()
            raise

        assert self.is_locked(exclusive_is_okay=exclusive, shared_is_okay=not exclusive)
        assert util.io.fs.fd_is_file(self._fd, self.lockfile, not_exist_okay=False)
        util.logging.debug('{}: Acquired.'.format(self))

    def acquire(self):
        if self._lock_count == 0:
            self._acquire(exclusive=self._exclusive, timeout=self._timeout)
        else:
            self._lock_count = self._lock_count + 1
        util.logging.debug('{}: Now aquired {} times. (One time added).'.format(self, self._lock_count))

    def _release(self):
        try:

            # try to get exclusive lock
            if not self.is_locked(exclusive_is_okay=True, shared_is_okay=False):
                self._unlock_lockfile()
                try:
                    self._acquire(exclusive=True, timeout=0)
                except util.io.filelock.general.FileLockTimeoutError:
                    util.logging.debug('{}: Could not remove lock file {}. It is locked by another process.'.format(self, self.lockfile))
                    assert not self.is_locked(exclusive_is_okay=True, shared_is_okay=False)
                else:
                    assert self.is_locked(exclusive_is_okay=True, shared_is_okay=False)

            # if exclusive lock, remove lock file
            if self.is_locked(exclusive_is_okay=True, shared_is_okay=False):
                assert util.io.fs.fd_is_file(self._fd, self.lockfile, not_exist_okay=False)
                try:
                    os.remove(self.lockfile)
                except OSError as e:
                    if e.errno == errno.EBUSY:
                        util.logging.debug('{}: Could not remove lock file {}. It is used by another process.'.format(self, self.lockfile))
                    else:
                        raise
                else:
                    util.logging.debug('{}: Lock file {} removed.'.format(self, self.lockfile))

        # cleanup
        finally:
            self._unlock_lockfile()
            self._close_lockfile()

        assert not self.is_locked()
        util.logging.debug('{}: Entirely released.'.format(self))

    def release(self):
        if self._lock_count == 1:
            self._release()
        elif self._lock_count > 1:
            self._lock_count = self._lock_count - 1
            util.logging.debug('{}: Now aquired {} times. (One time removed).'.format(self, self._lock_count))
        else:
            util.logging.debug('{}: Must not be released since it is not aquired.'.format(self))

    # *** contex manager *** #

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False

    def __del__(self):
        if self._lock_count > 1:
            self._release()


class LockedFile(FileLock):

    def __init__(self, file, save_function, load_function, cache_beyond_lock=True, timeout=None, sleep=0.5):
        util.logging.debug('Locked file {}: Creating lock file with cache_beyond_lock {}.'.format(file, cache_beyond_lock))
        super().__init__(file, timeout=timeout, sleep=sleep)

        self.cache_beyond_lock = cache_beyond_lock
        self.file_value = None
        self.cached_file_modified_time = None

        if not callable(save_function):
            raise ValueError('save_function must be a function.')
        self._save_function = save_function
        if not callable(load_function):
            raise ValueError('load_function must be a function.')
        self._load_function = load_function

    def _release(self):
        if not self.cache_beyond_lock:
            self.file_value = None
        super()._release()

    # *** modified time *** #

    def modified_time(self):
        return os.stat(self.file).st_mtime_ns

    # *** cache functions *** #

    def _cache_set_value(self, value):
        self.file_value = value
        if self.cache_beyond_lock:
            self.cached_file_modified_time = self.modified_time()

    def _cache_is_valid(self):
        return self.file_value is not None and (not self.cache_beyond_lock or self.cached_file_modified_time == self.modified_time())

    # *** load *** #

    def load(self):
        if not self._cache_is_valid():
            util.logging.debug('Locked file {}: Loading value.'.format(self.file))
            with self.lock_object(exclusive=False):
                value = self._load_function(self.file)
                self._cache_set_value(value)
            util.logging.debug('Locked file {}: Value loaded.'.format(self.file))
        else:
            value = self.file_value

        assert value is not None
        return value

    # *** save *** #

    def save(self, value):
        util.logging.debug('Locked file {}: Saving content.'.format(self.file))
        with self.lock_object(exclusive=True):
            self._save_function(self.file, value)
            self._cache_set_value(value)
        util.logging.debug('Locked file {}: Content saved.'.format(self.file))
