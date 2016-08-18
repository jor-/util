import os.path

import util.io.env

WORK_DIR = util.io.env.load('WORK')
HOME_DIR = util.io.env.load('HOME')
try:
    TMP_DIR = util.io.env.load('TMP')
except util.io.env.EnvironmentLookupError:
    TMP_DIR = os.path.join(WORK_DIR, 'tmp')