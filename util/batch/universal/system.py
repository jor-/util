import util.io.env

import util.logging


BATCH_SYSTEM_ENV_NAME = 'BATCH_SYSTEM'
try:
    BATCH_SYSTEM_STR = util.io.env.load(BATCH_SYSTEM_ENV_NAME)
except util.io.env.EnvironmentLookupError:
    IS_RZ = False
    IS_NEC_2 = False
    IS_NEC_5 = False
    IS_NONE = True
else:
    IS_RZ = BATCH_SYSTEM_STR == 'RZ-PBS'
    IS_NEC_2 = BATCH_SYSTEM_STR == 'NEC-NQSII'
    IS_NEC_5 = BATCH_SYSTEM_STR == 'NEC-NQSV'
    IS_NONE = False
IS_NEC = IS_NEC_2 or IS_NEC_5

if IS_RZ:
    util.logging.debug('Choosing batch system {}.'.format(BATCH_SYSTEM_STR))
    from util.batch.rz.system import *
elif IS_NEC:
    util.logging.debug('Choosing batch system {}.'.format(BATCH_SYSTEM_STR))
    from util.batch.nec.system import *
elif IS_NONE:
    util.logging.warn('Environmental variable {} is not set. Choosing general batch system.'.format(BATCH_SYSTEM_ENV_NAME))
    from util.batch.general.system import *
else:
    util.logging.warn('Batch system {} is unknown. Choosing general batch system.'.format(BATCH_SYSTEM_STR))
    from util.batch.general.system import *
