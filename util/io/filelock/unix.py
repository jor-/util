import abc
import fcntl
import errno
import os
import stat
import time

import util.io.fs
import util.io.filelock.general

import util.logging
logger = util.logging.logger



class FileLock:

    def __init__(self, file, exclusive=True, timeout=None, sleep=0.5):
        self._lock_count = 0
        self._file = file
        self._want_exclusive = exclusive
        self._is_exclusive = exclusive
        self._timeout = timeout
        self._sleep = sleep
        self._fd = None
        logger.debug('{}: Initiating file lock with timeout {} and sleep {}.'.format(self, timeout, sleep))
    
    
    def __str__(self):
        if self._is_exclusive:
            return 'File lock (exclusive) for {}'.format(self.file)
        else:
            return 'File lock (shared) for {}'.format(self.file)


    @property
    def file(self):
        return self._file

    @property
    def lockfile(self):
        return self._file + '.lock'


    @property
    def exclusive(self):
        return self._is_exclusive
    
    @exclusive.setter
    def exclusive(self, want_exclusive):
        if self._want_exclusive != want_exclusive:
            if want_exclusive and self._lock_count > 0 and not self._is_exclusive:
                raise ValueError('Lock is acquired as shared lock. To change to exclusive lock, first release this lock.')
            self._want_exclusive = want_exclusive
            logger.debug('{}: exclusive changed to {}.'.format(self, self._want_exclusive))
    
    
    def is_locked(self, exclusive_is_okay=True, shared_is_okay=True):
        return self._lock_count > 0 and ((self._is_exclusive and exclusive_is_okay) or (not self._is_exclusive and shared_is_okay))
    
    def lock_object(self, exclusive=True):
        self.want_exclusive = exclusive
        return self


    ## acquire and release
    
    def _open_lockfile(self):
        if self._fd is None:
            ## prepare flags and mode
            open_flags = os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC
            open_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
            ## open
            try:
                self._fd = os.open(self.lockfile, open_flags, mode=open_mode)
            except OSError as e:
                if e.errno == errno.ESTALE:
                    logger.warning('{}: {}. Retrying to open lock file'.format(self, e))
                    self._open_lockfile()
                else:
                    raise
            else:
                logger.debug('{}: Lock file {} opened.'.format(self, self.lockfile))
    
    
    def _close_lockfile(self):
        ## close
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            logger.debug('{}: Lock file {} closed.'.format(self, self.lockfile))
    
    
    def _lock_lockfile(self, exclusive=True, blocking=True):  
        assert self._fd is not None   
        assert self._lock_count == 0
           
        ## prepare lock flags
        if exclusive:
            lock_flags = fcntl.LOCK_EX
        else:
            lock_flags = fcntl.LOCK_SH
        if not blocking:
            lock_flags = lock_flags | fcntl.LOCK_NB
        ## lock
        fcntl.flock(self._fd, lock_flags)
        self._is_exclusive = exclusive
        self._lock_count = 1
        logger.debug('{}: Lock file {} locked with exclusive={}.'.format(self, self.lockfile, exclusive))
    
    
    def _unlock_lockfile(self):
        assert self._fd is not None
        ## unlock
        if self._lock_count > 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._lock_count = 0
            logger.debug('{}: Lock file {} unlocked.'.format(self, self.lockfile))
    

    def _acquire(self, exclusive=True, timeout=None):
        ## save start time for timeout
        if timeout is not None:
            start_time = time.time()
        
        ## save lock type
        self._is_exclusive = exclusive
        logger.debug('{}: Acquiring with timeout {}.'.format(self, timeout))
        
        has_lock = False
        try:
            while not has_lock:        
                ## open file
                self._open_lockfile()
                
                ## try to get lock
                try:
                    self._lock_lockfile(exclusive=exclusive, blocking=timeout is None)
                except BlockingIOError as e:
                    ## check if regular timeout
                    if e.errno != errno.EAGAIN or timeout is None:
                        logger.warning('{}: Retrying to get lock because an BlockingIOError occured: {}'.format(self, e))
                else:
                    ## if file was removed in between, open new file
                    if not util.io.fs.fd_is_file(self._fd, self.lockfile, not_exist_okay=True):
                        logger.debug('{}: Lock file {} was removed in beetween. Opening new lock file.'.format(self, self.lockfile))
                        self._unlock_lockfile()
                        self._close_lockfile()
                    ## lock successfull
                    else:
                        logger.debug('{}: Fresh acquired.'.format(self))
                        has_lock = True
                
                ## handle timout
                if not has_lock:
                    ## if timeout reached, raise FileLockTimeoutError
                    if timeout is not None and time.time() > (start_time + timeout):
                        logger.debug('{}: Could not be acquired. Timeout {} reached.'.format(self, timeout))
                        raise util.io.filelock.general.FileLockTimeoutError(self.lockfile, timeout) from e
                    ## else wait
                    else:
                        time.sleep(self._sleep)
        except:
            self._unlock_lockfile()
            self._close_lockfile()
            raise
        
        assert self._lock_count > 0
        assert self._fd is not None
        assert self.is_locked(exclusive_is_okay=exclusive, shared_is_okay=not exclusive)
    
    
    def acquire(self):
        if self._lock_count == 0:
            self._acquire(exclusive=self._want_exclusive, timeout=self._timeout)
        else:
            self._lock_count = self._lock_count + 1
        logger.debug('{}: Now aquired {} times. (One time added).'.format(self, self._lock_count))


    def _release(self):
        try:
        
            ## try to get exclusive lock
            if not self.is_locked(exclusive_is_okay=True, shared_is_okay=False):
                self._unlock_lockfile()
                try:
                    self._acquire(exclusive=True, timeout=0)
                    assert self.is_locked(exclusive_is_okay=True, shared_is_okay=False)
                except util.io.filelock.general.FileLockTimeoutError:
                    logger.debug('{}: Could not remove lock file {}. It is locked by another process.'.format(self, self.lockfile))
            
            ## if exclusive lock, remove lock file
            if self.is_locked(exclusive_is_okay=True, shared_is_okay=False):
                assert self._fd is not None
                assert self.lockfile is not None
                assert util.io.fs.fd_is_file(self._fd, self.lockfile, not_exist_okay=False)
                try:
                    os.remove(self.lockfile)
                except OSError as e:
                    if e.errno == errno.EBUSY:
                        logger.debug('{}: Could not remove lock file {}. It is used by another process.'.format(self, self.lockfile))
                    else:
                        raise
                else:
                    logger.debug('{}: Lock file {} removed.'.format(self, self.lockfile))
        
        ## cleanup
        finally:
            self._unlock_lockfile()
            self._close_lockfile()
        
        assert self._lock_count == 0
        assert self._fd is None
        assert not self.is_locked()
        logger.debug('{}: Entirely released.'.format(self))


    def release(self):
        if self._lock_count == 1:
            self._release()
        elif self._lock_count > 1:
            self._lock_count = self._lock_count - 1        
            logger.debug('{}: Now aquired {} times. (One time removed).'.format(self, self._lock_count))
        else:
            logger.debug('{}: Must not be released since it is not aquired.'.format(self))
            

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
    
    def __del__(self):
        if self._lock_count > 1:
            self._release()
        



class LockedFile(FileLock):

    def __init__(self, file, cache_beyond_lock=True, timeout=None, sleep=0.5):
        logger.debug('Locked file {}: Creating lock file with cache_beyond_lock {}.'.format(file, cache_beyond_lock))
        super().__init__(file, timeout=timeout, sleep=sleep)
        self.cache_beyond_lock = cache_beyond_lock
        self.file_value = None
        self.cached_file_modified_time = None
    
    
    def _release(self):
        if not self.cache_beyond_lock:
            self.file_value = None
        super()._release()
    
    
    ## modified time
    
    def modified_time(self):
        return os.stat(self.file).st_mtime_ns
    
    ## cache functions
    
    def _cache_set_value(self, value):
        self.file_value = value
        if self.cache_beyond_lock:
            self.cached_file_modified_time = self.modified_time()
    
    def _cache_is_valid(self):
        return self.file_value is not None and (not self.cache_beyond_lock or self.cached_file_modified_time == self.modified_time())
        
    
    ## load
    
    @abc.abstractmethod
    def _load(self, file):
        pass
        
    def load(self):
        if not self._cache_is_valid():
            logger.debug('Locked file {}: Loading value.'.format(self.file))
            with self.lock_object(exclusive=False):
                value = self._load(self.file)
                self._cache_set_value(value)
            logger.debug('Locked file {}: Value loaded.'.format(self.file))
        else:
            value = self.file_value
        
        assert value is not None
        return value  


    ## save
    @abc.abstractmethod
    def _save(self, file, value):
        pass
    
    def save(self, value):
        logger.debug('Locked file {}: Saving content.'.format(self.file))
        with self.lock_object(exclusive=True):
            self._save(self.file, value)
            self._cache_set_value(value)
        logger.debug('Locked file {}: Content saved.'.format(self.file))
    
