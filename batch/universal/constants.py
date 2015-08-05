# import platform
# _HOSTNAME = platform.node()
# IS_RZ = _HOSTNAME == 'rzcl00b'
# IS_NEC = _HOSTNAME.startswith('nesh-')
# 
# if IS_RZ:
#     from util.batch.rz.constants import *
# elif IS_NEC:
#     from util.batch.nec.constants import *
# else:
#     raise ValueError('Universal batch system is not supported for host {}.'.format(_HOSTNAME))
from util.batch.rz.constants import *