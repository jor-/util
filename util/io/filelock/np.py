import numpy as np

import util.io.filelock.unix


class LockedFile(util.io.filelock.unix.LockedFile):

    def __init__(self, file, cache_beyond_lock=True, timeout=None, sleep=0.5):
        save_function = np.save
        load_function = np.load
        super().__init__(file, save_function, load_function, timeout=timeout, sleep=sleep)
