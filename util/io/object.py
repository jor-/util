import pickle
import pickletools

import util.io.fs
import util.logging


FILE_EXT = '.ppy'
MAX_PROTOCOL = -1


def is_file(file):
    return util.io.fs.has_file_ext(file, FILE_EXT)


def add_file_ext(file):
    return util.io.fs.add_file_ext_if_needed(file, FILE_EXT)


def save(file, obj, protocol=MAX_PROTOCOL):
    file = add_file_ext(file)
    util.logging.debug('Saving {} to {}.'.format(obj, file))
    with open(file, 'wb') as file_object:
        pickle.dump(obj, file_object, protocol=protocol)


def load(file):
    util.logging.debug('Loading from {}.'.format(file))
    with open(file, 'rb') as file_object:
        obj = pickle.load(file_object)
    return obj


def protocol_version(file):
    with open(file, 'rb') as file_object:
        for opcode, arg, pos in pickletools.genops(file_object):
            maxproto = max(MAX_PROTOCOL, opcode.proto)
    return maxproto
