import util.index_database.fs_based
import util.petsc.universal


class Database(util.index_database.fs_based.Database):
    
    def __init__(self, value_dir, value_filenames, value_reliable_decimal_places=15, tolerance_options=None):
        ## call super constructor
        super().__init__(value_dir, value_filenames, value_reliable_decimal_places=value_reliable_decimal_places, tolerance_options=tolerance_options)
    
    
    ## load and save
    def _load_file(self, value_file):
        return util.petsc.universal.load_petsc_vec_to_numpy_array(value_file)
    
    def _save_file(self, value_file, value):
        util.petsc.universal.save_numpy_array_to_petsc_vec(value_file, value)

