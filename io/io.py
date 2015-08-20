import os
import sys
import stat
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)





# def get_sequence_from_values_or_file(value):
#     # check if float
#     try:
#         value = (float(value),)
#     # check if file
#     except (TypeError, ValueError):
#         try:
#             os.path.isfile(value)
#             try:
#                 value = np.loadtxt(value)
#             except (OSError, IOError) as e:
#                 raise ValueError('The value {} seams to be a file, but it could not been read.'.format(value)) from e
#     # check if sequence
#         except TypeError:
#             try:
#                 len(value)
#             except TypeError:
#                 raise ValueError('Value has to be a float, a sequence or a file containing a sequence.')
#
#     return value




## std out and std err

class String_List_Stream:
    def __init__(self, string_list, callback_stream):
        self.string_list = string_list
        self.callback_stream = callback_stream

    def write(self, message):
        self.string_list.append(message)
        self.callback_stream.write(message)


class Copy_Std_Streams:
    def __init__(self, string_list_out, string_list_err):
        self.string_list_out = string_list_out
        self.string_list_err = string_list_err

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.string_list_stream_out = String_List_Stream(self.string_list_out, self.old_stdout)
        self.string_list_stream_err = String_List_Stream(self.string_list_err, self.old_stderr)
        sys.stdout, sys.stderr = self.string_list_stream_out, self.string_list_stream_err
        self.old_stdout.flush(); self.old_stderr.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr




