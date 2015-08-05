import os

import util.batch.general.system

import util.logging
logger = util.logging.logger



## batch setup

class BatchSystem(util.batch.general.system.BatchSystem):
    def __init__(self):
        from util.batch.nec.constants import QSUB_COMMAND, QUEUES, MAX_WALLTIME, MODEL_RENAMING
        super().__init__(QSUB_COMMAND, QUEUES, max_walltime=MAX_WALLTIME, module_renaming=MODEL_RENAMING)

BATCH_SYSTEM = BatchSystem()


## job 

class Job(util.batch.general.system.Job):
    
    def __init__(self, output_dir, force_load=False):
        super().__init__(BATCH_SYSTEM, output_dir, force_load=force_load)
    

    def init_job_file(self, job_name, nodes_setup, walltime_hours=None, write_output_file=True):
        ## set queue if missing
        queue = nodes_setup.node_kind
        
        ## super
        super().init_job_file(job_name, nodes_setup, queue=queue, cpu_kind=None, walltime_hours=walltime_hours, write_output_file=write_output_file)
    
    
    def _make_job_file_header(self, use_mpi=False):
        content = []
        ## shell
        content.append('#!/bin/bash')
        content.append('')
        ## name
        content.append('#PBS -N {}'.format(self.options['/job/name']))
        ## output file
        if self.output_file is not None:
            content.append('#PBS -j o')
            content.append('#PBS -o {}'.format(self.output_file))
        ## queue
        content.append('#PBS -q {}'.format(self.options['/job/queue']))
        ## walltime
        if self.walltime_hours is not None:
            content.append('#PBS -l elapstim_req={:02d}:00:00'.format(self.walltime_hours))
        ## select
        content.append('#PBS -b {:d}'.format(self.options['/job/nodes']))
        content.append('#PBS -l cpunum_job={:d}'.format(self.options['/job/cpus']))
        content.append('#PBS -l memsz_job={:d}gb'.format(self.options['/job/memory_gb']))
        ## MPI
        if use_mpi:
            content.append('#PBS -T intmpi')
        ## return
        content.append('')
        content.append('')
        return os.linesep.join(content)
    

    def _make_job_file_modules(self, modules):
        content = []
        if len(modules) > 0:
            ## init module system
            content.append('. /usr/share/Modules/init/bash')
            ## system modules
            for module in modules:
                content.append('module load {}'.format(module))
            content.append('module list')
            content.append('')
            content.append('')
        return os.linesep.join(content)

    
    def write_job_file(self, run_command, modules=()):
        with open(self.opt['/job/option_file'], mode='w') as file:
            file.write(self._make_job_file_header(use_mpi=intelmpi in modules))
            file.write(self._make_job_file_modules(modules))
            file.write(self._make_job_file_command(run_command))




## node setups

from util.batch.general.system import NodeSetup
# class NodeSetup(util.batch.general.system.NodeSetup):
#     def __init__(self, *args, **kargs):
#         super().__init__()
    