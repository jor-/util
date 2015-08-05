import os.path

import util.io.env

WORK_DIR = util.io.env.load('WORK')
HOME_DIR = util.io.env.load('HOME')
TMP_DIR = os.path.join(WORK_DIR, 'tmp')