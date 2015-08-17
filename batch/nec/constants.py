QUEUES = ('clfocean', 'clfo2', 'clexpress', 'clmedium', 'cllong', 'long', 'clbigmem', 'feque')
MAX_WALLTIME = {'clfocean':100, 'clfo2':100, 'clexpress':2, 'clmedium':48, 'cllong':100, 'clbigmem':100, 'feque':1}

MODEL_RENAMING = {'python':'Python3.3.6', 'python3':'Python3.3.6', 'hdf5':'hdf5_intel', 'matlab':'matlab2014a', 'petsc':'petsc-3.3-p4-intel'}

QSUB_COMMAND = '/usr/bin/nqsII/qsub'
MPI_COMMAND = 'mpirun $NQSII_MPIOPTS -np {cpus:d} {command}'