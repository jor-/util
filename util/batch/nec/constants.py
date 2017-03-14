import os

QUEUES = ('clexpress', 'clmedium', 'cllong', 'long', 'clbigmem', 'clfocean', 'clfo2', 'feque')
MAX_WALLTIME = {'clexpress':2, 'clmedium':48, 'cllong':100, 'clbigmem':100, 'clfocean':100, 'clfo2':100, 'feque':10}

PRE_COMMANDS = {
    'mpirun': os.linesep.join(['module load intel16.0.3', 'module load intelmpi16.0.3', 'module list'])
}
PRE_COMMANDS['metos3d'] = PRE_COMMANDS['mpirun']

COMMANDS = {
    'mpirun': 'mpirun $NQSII_MPIOPTS -np {cpus:d} {command}',
    'time': 'TIME_FMT="\nStatistics for %C:\nElapsed time: %Es, Exit code: %x\nCPU: %Us user mode, %Ss kernel mode, %P workload\nMemory: %Mkb max, %W swap outs\nContex switches: %c involuntarily, %w voluntarily\nPage faults: %F major, %R minor\nFile system I/O: %I inputs, %O outputs"\n/sfs/fs3/sw/tools/time1.7/bin/time -f "$TIME_FMT" {command}',
    'sub': '/usr/bin/nqsII/qsub',
    'stat': '/usr/local/bin/qstatall',
    'nodes': '/sfs/fs3/sw/tools/qcl/qcl',
    'python': 'python3'
}

NODE_INFOS = {
    'clexpress': {'nodes': 2, 'speed': 2.6, 'cpus': 16, 'memory': 128, 'max_walltime': MAX_WALLTIME['clexpress']},
    'clmedium': {'nodes': 60, 'speed': 2.6, 'cpus': 16, 'memory': 128, 'max_walltime': MAX_WALLTIME['clmedium']},
    'cllong': {'nodes': 30, 'speed': 2.6, 'cpus': 16, 'memory': 128, 'max_walltime': MAX_WALLTIME['cllong']},
    'clbigmem': {'nodes': 4, 'speed': 2.6, 'cpus': 16, 'memory': 256, 'max_walltime': MAX_WALLTIME['clbigmem']},
    'clfocean': {'nodes': 4, 'speed': 2.6, 'cpus': 16, 'memory': 128, 'max_walltime': MAX_WALLTIME['clfocean']},
    'clfo2': {'nodes': 18, 'speed': 2.5, 'cpus': 24, 'memory': 128, 'max_walltime': MAX_WALLTIME['clfo2']},
    'feque': {'nodes': 1, 'speed': 0, 'cpus': 16, 'memory': 128, 'leave_free': 1, 'max_walltime': MAX_WALLTIME['feque']}
}


EXCEEDED_WALLTIME_ERROR_MESSAGE = "Batch job received signal SIGKILL. (Exceeded per-req elapse time limit)"