import util.io.env

import util.logging
logger = util.logging.logger


BATCH_SYSTEM_ENV_NAME = 'BATCH_SYSTEM'
try:
    BATCH_SYSTEM_STR = util.io.env.load(BATCH_SYSTEM_ENV_NAME)
    IS_RZ = BATCH_SYSTEM_STR == 'RZ-PBS'
    IS_NEC = BATCH_SYSTEM_STR == 'NEC-NQSII'
    IS_NONE = False
except util.io.env.EnvironmentLookupError:
    IS_RZ = False
    IS_NEC = False
    IS_NONE = True

if IS_RZ:
    logger.debug('Choosing batch system {}.'.format(BATCH_SYSTEM_STR))
    from util.batch.rz.system import *
elif IS_NEC:
    logger.debug('Choosing batch system {}.'.format(BATCH_SYSTEM_STR))
    from util.batch.nec.system import *
elif IS_NONE:
    logger.warn('Environmental variable {} is not set. Chosing general batch system.'.format(BATCH_SYSTEM_ENV_NAME))
    from util.batch.general.system import *
else:
    raise ValueError('Batch system {} is unknown.'.format(BATCH_SYSTEM_STR))
