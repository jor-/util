QUEUES = ('clfocean', 'clfo2', 'clexpress', 'clmedium', 'cllong', 'long', 'clbigmem', 'feque')
MAX_WALLTIME = {'clfocean':100, 'clfo2':100, 'clexpress':2, 'clmedium':48, 'cllong':100, 'clbigmem':100, 'feque':1}

MODEL_RENAMING = {'python':'Python3.3.6', 'python3':'Python3.3.6', 'hdf5':'hdf5_intel', 'matlab':'matlab2014a', 'petsc':'petsc-3.3-p4-intel'}

COMMANDS = {'mpirun': 'mpirun $NQSII_MPIOPTS -np {cpus:d} {command}',
            'time': '/sfs/fs3/sw/tools/time1.7/bin/time -f "Statistics for %C:\nexit code: %x, elapsed time: %Es\nCPU: %Us user mode, %Ss kernel mode, %P workload\nMemory: %Mkb max, %W swap outs\nContex switches: %c involuntarily, %w voluntarily\nPage faults: %F major, %R minor\nFile system I/O: %I inputs, %O outputs" {command}',
            # 'time': 'time {command}',
            'sub': '/usr/bin/nqsII/qsub',
            'stat': '/usr/bin/nqsII/qstat'}

# MPI_COMMAND = 'mpirun $NQSII_MPIOPTS -np {cpus:d} {command}'
# QSUB_COMMAND = '/usr/bin/nqsII/qsub'
# QSTAT_COMMAND = '/usr/bin/nqsII/qstat'
