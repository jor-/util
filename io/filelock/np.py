import numpy as np

import util.io.filelock.unix


class LockedFile(util.io.filelock.unix.LockedFile):

    def _load(self, file):
        return np.load(file)

    def _save(self, file, value):
        np.save(file, value)
    
