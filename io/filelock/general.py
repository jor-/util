
class FileLockTimeoutError(TimeoutError):
    
    def __init__(self, lock_filename, timeout):
        self.lock_filename = lock_filename
        self.timeout = timeout
        error_message = 'The lock for {} could not be aquired within timeout {}.'.format(lock_filename, timeout)
        super().__init__(error_message)



class FileLockNotAvailableError(Exception):
    
    def __init__(self, lock_filename, exclusive):
        self.lock_filename = lock_filename
        self.exclusive = exclusive
        if exclusive:
            lock_str = 'exclusive lock'
        else:
            lock_str = 'shared lock'
        error_message = 'The {} for {} was not available.'.format(lock_str, lock_filename)
        super().__init__(error_message)

