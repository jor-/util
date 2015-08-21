QUEUES = ('clfocean', 'clfo2', 'clexpress', 'clmedium', 'cllong', 'long', 'clbigmem', 'feque')
MAX_WALLTIME = {'clfocean':100, 'clfo2':100, 'clexpress':2, 'clmedium':48, 'cllong':100, 'clbigmem':100, 'feque':10}

MODEL_RENAMING = {'python':'Python3.3.6', 'python3':'Python3.3.6', 'hdf5':'hdf5_intel', 'matlab':'matlab2014a', 'petsc':'petsc-3.3-p4-intel'}

COMMANDS = {'mpirun': 'mpirun $NQSII_MPIOPTS -np {cpus:d} {command}',
            'time': 'TIME_FMT="\nStatistics for %C:\nElapsed time: %Es, Exit code: %x\nCPU: %Us user mode, %Ss kernel mode, %P workload\nMemory: %Mkb max, %W swap outs\nContex switches: %c involuntarily, %w voluntarily\nPage faults: %F major, %R minor\nFile system I/O: %I inputs, %O outputs"\n/sfs/fs3/sw/tools/time1.7/bin/time -f "$TIME_FMT" {command}',
            'sub': '/usr/bin/nqsII/qsub',
            'stat': '/usr/bin/nqsII/qstat',
            'nodes': '/usr/local/bin/qcl'}

NODE_INFOS = {'clexpress': {'nodes': 2, 'speed': 2.6, 'cpus': 16, 'memory': 128},
              'clmedium': {'nodes': 76, 'speed': 2.6, 'cpus': 16, 'memory': 128},
              'cllong': {'nodes': 30, 'speed': 2.6, 'cpus': 16, 'memory': 128},
              'clbigmem': {'nodes': 4, 'speed': 2.6, 'cpus': 16, 'memory': 256},
              'clfocean': {'nodes': 4, 'speed': 2.6, 'cpus': 16, 'memory': 128},
              'clfo2': {'nodes': 18, 'speed': 2.5, 'cpus': 24, 'memory': 128},
              'feque': {'nodes': 1, 'speed': 0, 'cpus': 16, 'memory': 128, 'leave_free': 1}}