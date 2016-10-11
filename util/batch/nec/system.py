import os
import subprocess

import numpy as np

import util.batch.general.system

import util.logging
logger = util.logging.logger

from util.batch.general.system import *



## batch setup

class BatchSystem(util.batch.general.system.BatchSystem):

    def __init__(self):
        from util.batch.nec.constants import COMMANDS, QUEUES, MAX_WALLTIME, MODEL_RENAMING, NODE_INFOS
        super().__init__(COMMANDS, QUEUES, max_walltime=MAX_WALLTIME, module_renaming=MODEL_RENAMING, node_infos=NODE_INFOS)


    def __str__(self):
        return 'NEC batch system'


    def _get_job_id_from_submit_output(self, submit_output):
        # Output form: "Request 130530.ace-ssiox submitted to queue: clmedium."
        submit_output_splitted = submit_output.split(' ')
        assert len(submit_output_splitted) == 6
        assert submit_output_splitted[5][:-1] in self.queues
        job_id = submit_output_splitted[1]
        return job_id


    def job_state(self, job_id):
        ## get state of job
        output = subprocess.check_output((self.status_command, job_id)).decode("utf-8")
        logger.debug('qstat result:\n{}'.format(output.strip()))
        return output


    def is_job_running(self, job_id):
        output = self.job_state(job_id)
        return 'RequestID' in output


    def _nodes_state(self):
        output = subprocess.check_output(self.nodes_command).decode('utf8')
        lines = output.splitlines()
        
        state = {}
        
        for line in lines:
            line = line.strip()
            for node_kind in self.node_infos.kinds():
                if line.startswith(node_kind):
                    line_splitted = line.split(' ')
                    number_of_free_nodes = int(line_splitted[-1])
                    
                    if number_of_free_nodes < 0:
                        logger.warn('Number of free nodes in the following line is negative, setting free nodes to zero.\n{}'.format(line))
                        number_of_free_nodes = 0
                    
                    logger.debug('Extracting nodes states from line "{}": node kind {} with {} free nodes.'.format(line, node_kind, number_of_free_nodes))
                    free_cpus = np.ones(number_of_free_nodes, dtype=np.uint32) * self.node_infos.cpus(node_kind)
                    free_memory = np.ones(number_of_free_nodes, dtype=np.uint32) * self.node_infos.memory(node_kind)
                    
                    state[node_kind] = (free_cpus, free_memory)
        
        return util.batch.general.system.NodesState(state)

        

BATCH_SYSTEM = BatchSystem()


## job

class Job(util.batch.general.system.Job):

    def __init__(self, output_dir, force_load=False):
        from util.batch.nec.constants import EXCEEDED_WALLTIME_ERROR_MESSAGE
        super().__init__(BATCH_SYSTEM, output_dir, force_load=force_load, exceeded_walltime_error_message=EXCEEDED_WALLTIME_ERROR_MESSAGE)


    def init_job_file(self, job_name, nodes_setup, queue=None):
        ## set queue if missing
        if queue is not None and queue != nodes_setup.node_kind:
            logger.warn('Queue {} and cpu kind {} have to be the same. Setting Queue to cpu kind.'.format(queue, nodes_setup.node_kind))
        queue = nodes_setup.node_kind

        ## super
        super().init_job_file(job_name, nodes_setup, queue=queue, cpu_kind=None)


    def _make_job_file_header(self, use_mpi):
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
            ## system modules
            for module in modules:
                content.append('module load {}'.format(module))
            content.append('module list')
            content.append('')
            content.append('')
        return os.linesep.join(content)

