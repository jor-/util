import argparse

import util.math.sparse.decompose.with_cholmod
import util.io.object

import util.logging

if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('load_file', type=argparse.FileType('r'))
    parser.add_argument('save_file', type=argparse.FileType('w'),)
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    args = parser.parse_args()

    with util.logging.Logger(disp_stdout=args.debug):
        A = util.io.object.load(args.load_file.name)
        B = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(A, min_abs_value=0.01)
        util.io.object.save(args.save_file.name, B)