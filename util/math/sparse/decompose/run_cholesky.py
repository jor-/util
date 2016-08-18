import argparse

import util.math.sparse.decompose.with_cholmod
import util.io.object

import util.logging

if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('load_file', type=argparse.FileType('r'))
    parser.add_argument('save_files', type=argparse.FileType('w'), nargs='+')
    parser.add_argument('-o', '--ordering_method', default='default')
    parser.add_argument('-r', '--return_type', default='P_L')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    args = parser.parse_args()

    n = len(args.return_type.split('_'))
    if len(args.save_files) != n:
        raise ValueError('For return type {} are {} save files needed.'.format(return_type, n))

    with util.logging.Logger(disp_stdout=args.debug):
        A = util.io.object.load(args.load_file.name)
        decomposition = util.math.sparse.decompose.with_cholmod.cholesky(A, ordering_method=args.ordering_method, return_type=args.return_type)
        for i in range(n):
            util.io.object.save(args.save_files[i].name, decomposition[i])
