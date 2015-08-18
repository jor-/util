import util.io.env

import util.logging
logger = util.logging.logger


BATCH_SYSTEM_STR = util.io.env.load('BATCH_SYSTEM')
logger.debug('Choosing batch system {}.'.format(BATCH_SYSTEM_STR))
IS_RZ = BATCH_SYSTEM_STR == 'RZ-PBS'
IS_NEC = BATCH_SYSTEM_STR == 'NEC-NQSII'

if IS_RZ:
    from util.batch.rz.system import *
elif IS_NEC:
    from util.batch.nec.system import *
else:
    raise ValueError('Batch system {} is unknown.'.format(BATCH_SYSTEM_STR))
    