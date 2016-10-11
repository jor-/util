import numpy as np

import util.index_database.fs_based


class Database(util.index_database.fs_based.Database):
    
    def __init__(self, value_dir, value_filenames, value_reliable_decimal_places=15, tolerance_options=None):
        ## call super constructor
        super().__init__(value_dir, value_filenames, value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
    
    
    ## load and save
    def _load_file(self, value_file):
        value = np.loadtxt(value_file)
        if value.ndim == 0:
            value = value.reshape(-1)
        assert value.ndim == 1
        return value
    
    def _save_file(self, value_file, value):
        np.savetxt(value_file, value, fmt=self.value_fmt)

