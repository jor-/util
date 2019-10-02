import os

from util.batch.universal.system import IS_NEC_2, IS_NEC_5

PRE_COMMANDS = {
    'mpirun': os.linesep.join(['module load intel17.0.4', 'module load intelmpi17.0.4', 'module list'])
}
PRE_COMMANDS['metos3d'] = PRE_COMMANDS['mpirun']

COMMANDS = {
    'mpirun': 'mpirun $NQSII_MPIOPTS -np {cpus:d} {command}',
    'time': 'TIME_FMT="\nStatistics for %C:\nElapsed time: %Es, Exit code: %x\nCPU: %Us user mode, %Ss kernel mode, %P workload\nMemory: %Mkb max, %W swap outs\nContex switches: %c involuntarily, %w voluntarily\nPage faults: %F major, %R minor\nFile system I/O: %I inputs, %O outputs"\ncommand time -f "$TIME_FMT" {command}',
    'nodes': '/usr/local/bin/qcl',
    'python': 'python3'
}
if IS_NEC_2:
    COMMANDS['sub'] = '/usr/bin/nqsII/qsub'
    COMMANDS['stat'] = '/usr/local/bin/qstatall'
else:
    COMMANDS['sub'] = '/opt/nec/nqsv/bin/qsub'
    COMMANDS['stat'] = '/opt/nec/nqsv/bin/qstat'
    COMMANDS['stat_args'] = ['-l', '-J']

NODE_INFOS = {
    'clexpress': {'nodes': 2, 'speed': 2.1, 'cpus': 32, 'memory': 192, 'max_walltime': 2},
    'clmedium': {'nodes': 120, 'speed': 2.1, 'cpus': 32, 'memory': 192, 'max_walltime': 48},
    'cllong': {'nodes': 50, 'speed': 2.1, 'cpus': 32, 'memory': 192, 'max_walltime': 100},
    'clbigmem': {'nodes': 8, 'speed': 2.1, 'cpus': 32, 'memory': 384, 'max_walltime': 200},
    'clfo2': {'nodes': 18, 'speed': 2.5, 'cpus': 24, 'memory': 128, 'max_walltime': 200},
    'feque': {'nodes': 1, 'speed': 0, 'cpus': 16, 'memory': 768, 'leave_free': 1, 'max_walltime': 1}
}

QUEUES = tuple(NODE_INFOS.keys())
MAX_WALLTIME = {queue: NODE_INFOS[queue]['max_walltime'] for queue in QUEUES}

EXCEEDED_WALLTIME_ERROR_MESSAGE = "Batch job received signal SIGKILL. (Exceeded per-req elapse time limit)"
