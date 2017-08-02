import os.path

import util.io.env

HOME_DIR = util.io.env.load('HOME')
try:
    WORK_DIR = util.io.env.load('WORK')
except util.io.env.EnvironmentLookupError:
    WORK_DIR = HOME_DIR
try:
    TMP_DIR = util.io.env.load('TMP')
except util.io.env.EnvironmentLookupError:
    TMP_DIR = os.path.join(WORK_DIR, 'tmp')
