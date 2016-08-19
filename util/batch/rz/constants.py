QUEUES = ('f_ocean', 'f_ocean2', 'express', 'small', 'medium', 'long', 'para_low')
MAX_WALLTIME = {'express': 3, 'small': 24, 'medium': 240, 'long': 480, 'para_low': 1000}

MODEL_RENAMING = {'python':'Python-3.3.2', 'python3':'Python-3.3.2', 'hdf5':'hdf5-1.8.16', 'matlab':'matlab2015a', 'petsc':'', 'intel14': 'intel1402', 'intelmpi14': 'intelmpi1402', 'intel16': 'intel16.0.0', 'intelmpi16': 'intelmpi16.0.0'}

COMMANDS = {'mpirun': 'mpirun -n {cpus:d} -machinefile $PBS_NODEFILE -r rsh {command}',
            'time': 'TIME_FMT="\nStatistics for %C:\nElapsed time: %Es, Exit code: %x\nCPU: %Us user mode, %Ss kernel mode, %P workload\nMemory: %Mkb max, %W swap outs\nContex switches: %c involuntarily, %w voluntarily\nPage faults: %F major, %R minor\nFile system I/O: %I inputs, %O outputs"\ncommand time -f "$TIME_FMT" {command}',
            'sub': '/opt/pbs/default/bin/qsub',
            'stat': '/opt/pbs/default/bin/qstat',
            'nodes': '/usr/local/bin/qnodes',
            'python': '/home/sunip229/local/bin/python3'}

NODE_INFOS = {'f_ocean2': {'nodes': 12, 'speed': 2.6, 'cpus': 16, 'memory': 128},
              'westmere': {'nodes': 18, 'speed': 2.67, 'cpus': 12, 'memory': 48, 'max_walltime': MAX_WALLTIME['para_low']},
              'shanghai': {'nodes': 26, 'speed': 2.4, 'cpus': 8, 'memory': 32, 'leave_free': 26, 'max_walltime': MAX_WALLTIME['para_low']},
              'amd128': {'nodes': 1, 'speed': 2.4, 'cpus': 16, 'memory': 128, 'leave_free': 1, 'max_walltime': MAX_WALLTIME['para_low']},
              'amd256': {'nodes': 1, 'speed': 2.1, 'cpus': 48, 'memory': 256, 'leave_free': 1, 'max_walltime': MAX_WALLTIME['para_low']},
              'fobigmem': {'nodes': 2, 'speed': 2.6, 'cpus': 24, 'memory': 1024, 'leave_free': 2}}