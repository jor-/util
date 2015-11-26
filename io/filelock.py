"""
A file lock implementation that tries to avoid platform specific
issues. It is inspired by a whole bunch of different implementations
listed below.

 - https://bitbucket.org/jaraco/yg.lockfile/src/6c448dcbf6e5/yg/lockfile/__init__.py
 - http://svn.zope.org/zc.lockfile/trunk/src/zc/lockfile/__init__.py?rev=121133&view=markup
 - http://stackoverflow.com/questions/489861/locking-a-file-in-python
 - http://www.evanfosmark.com/2009/01/cross-platform-file-locking-support-in-python/
 - http://packages.python.org/lockfile/lockfile.html

There are some tests below and a blog posting conceptually the
problems I wanted to try and solve. The tests reflect these ideas.

 - http://ionrock.wordpress.com/2012/06/28/file-locking-in-python/

I'm not advocating using this package. But if you do happen to try it
out and have suggestions please let me know.
"""

import fcntl
import os
import time

import util.logging
logger = util.logging.get_logger()


class FileLockTimeoutError(TimeoutError):
    
    def __init__(self, lock_filename, timeout, pid):
        self.lock_filename = lock_filename
        self.timeout = timeout
        self.pid = pid
        error_message = 'The lock {} could not be aquired by process {} within timeout {}.'.format(lock_filename, pid, timeout)
        super().__init__(error_message)


class FileLock(object):

    def __init__(self, filename, timeout=None, sleep_time=0.5):
        self.filename = filename
        self.lock_filename = '{}.lock'.format(self.filename)
        self.timeout = timeout
        self.sleep_time = sleep_time
        self.pid = os.getpid()
        self.fd = None
    
    
    def lock_pid(self):
        """
        Get the pid of the lock.
        """
        
        if os.path.exists(self.lock_filename):
            return int(open(self.lock_filename).read())
        else:
            return None


    def is_locked_by_me(self):
        """
        See if the lock exists and is belongs to this process.
        """
        
        ## get pid of lock
        lock_pid = self.lock_pid()

        ## not locked
        if lock_pid is None:
            logger.debug('Lock {} is not aquired.'.format(self.lock_filename))
            return False
        
        ## locked by me
        if self.pid == lock_pid:
            logger.debug('Lock {} is aquired by me (pid: {}).'.format(self.lock_filename, self.pid))
            return True

        ## locked by alive process
        try:
            os.kill(lock_pid, 0)
            logger.debug('Lock {} is aquired by another (alive) process (pid: {}).'.format(self.lock_filename, lock_pid))
        
        ## locked by dead process
        except OSError:
            logger.debug('Lock {} is aquired by a dead process (pid: {}).'.format(self.lock_filename, lock_pid))
            try:
                lock_fd = os.open(self.lock_filename, os.O_RDWR)
            except FileNotFoundError:
                logger.debug('The lock is now removed by another process.')
            else:
                try:
                    fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    logger.debug('The lock will be removed by another process.')
                else:
                    if self.lock_pid() == lock_pid:
                        os.remove(self.lock_filename)
                        fcntl.lockf(lock_fd, fcntl.LOCK_UN)
                        logger.debug('The lock is removed by me.')
                    else:
                        logger.debug('The lock is substituted by another lock.')
                os.close(lock_fd)
        return False


    def acquire_try_once(self):
        """
        Try to aquire the lock once.
        """
        
        if self.is_locked_by_me():
            return True
        else:
            try:
                self.fd = os.open(self.lock_filename, os.O_CREAT | os.O_RDWR | os.O_EXCL)
            except FileExistsError:
                logger.debug('I tried to get the lock {}, but another process was faster.'.format(self.lock_filename))
                return False
            else:
                os.write(self.fd, str(self.pid).encode('utf-8'))
                logger.debug('I got the lock {}.'.format(self.lock_filename))
                return True


    def acquire(self):
        """
        Try to aquire the lock.
        """
        
        if self.timeout is not None:
            sleep_intervals = int(self.timeout / self.sleep_time)
        else:
            sleep_intervals = float('inf')
        
        while not self.acquire_try_once() and sleep_intervals > 0:
            time.sleep(self.sleep_time)
            sleep_intervals -= 1
        
        if not self.is_locked_by_me():
            raise FileLockTimeoutError()
    

    def release(self):
        """
        Release the lock.
        """
        
        if self.is_locked_by_me():
            os.remove(self.lock_filename)
            logger.debug('The lock {} is released by me (pid: {}).'.format(self.lock_filename, self.pid))
        if self.fd:
            os.close(self.fd)
            self.fd = None


    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def __del__(self):
        self.release()



class TestFileLock(object):
    
    def __init__(self):
        self.filename = 'test.txt'
        self.lock_filename = '{}.lock'.format(self.filename)

    def test_remove_dead_lock(self):
        with open(self.lockfile, 'w+') as f:
            f.write('9999999')
        assert os.path.exists(self.lockfile)
        with FileLock(self.filename):
            assert True

    def test_resume_lock(self):
        with open(self.lockfile, 'w+') as f:
            f.write(str(os.getpid()))
        with FileLock(self.filename, timeout=1):
            assert True

    def test_double_lock(self):
        with FileLock(self.filename):
            with FileLock(self.filename):
                assert True

    def test_exception_after_timeout(self):
        with open(self.lockfile, 'w+') as f:
            f.write('9999999')
        try:
            with FileLock(self.filename, timeout=1):
                assert False
        except FileLockTimeoutError:
            assert True


