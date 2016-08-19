import pickle
import pickletools

import util.io.fs



FILE_EXT = '.ppy'

def is_file(file):
    return util.io.fs.has_file_ext(file, FILE_EXT)

def add_file_ext(file):
    return util.io.fs.add_file_ext_if_needed(file, FILE_EXT)



def save(file, o, protocol=-1):
    file = add_file_ext(file)
    with open(file, 'wb') as file_object:
        pickle.dump(o, file_object, protocol=protocol)

def load(file):
    with open(file, 'rb') as file_object:
        o = pickle.load(file_object)
    return o


def protocol_version(file):
    maxproto = -1
    with open(file, 'rb') as file_object:
        for opcode, arg, pos in pickletools.genops(file_object):
            maxproto = max(maxproto, opcode.proto)
    return maxproto