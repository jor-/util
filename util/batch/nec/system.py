import os
import subprocess

import numpy as np

import util.batch.general.system

import util.logging

from util.batch.general.system import *



## batch setup

class BatchSystem(util.batch.general.system.BatchSystem):

    def __init__(self):
        from util.batch.nec.constants import COMMANDS, QUEUES, PRE_COMMANDS, MAX_WALLTIME, NODE_INFOS
        super().__init__(COMMANDS, QUEUES, pre_commands=PRE_COMMANDS, max_walltime=MAX_WALLTIME, node_infos=NODE_INFOS)


    def __str__(self):
        return 'NEC batch system'


    def _get_job_id_from_submit_output(self, submit_output):
        # Output form: "Request 130530.ace-ssiox submitted to queue: clmedium."
        submit_output_splitted = submit_output.split(' ')
        assert len(submit_output_splitted) == 6
        assert submit_output_splitted[5][:-1] in self.queues
        job_id = submit_output_splitted[1]
        return job_id


    def is_job_running(self, job_id):
        output = self.job_state(job_id, return_output=True)
        return 'RequestID' in output


    def _nodes_state(self):
        nodes_command = self.nodes_command
        try:
            output = subprocess.check_output(nodes_command)
        except (OSError, subprocess.CalledProcessError) as e:
            raise util.batch.general.system.CommandError(nodes_command, cause=e) from e
        else:
            output = output.decode('utf8')
            lines = output.splitlines()
            state = {}
            for line in lines:
                line = line.strip()
                for node_kind in self.node_infos.kinds():
                    if line.startswith(node_kind):
                        line_splitted = line.split(' ')
                        line_splitted = [line_part for line_part in line_splitted if len(line_part) > 0]
                        # line format: Batch class  Walltime [h]  Cores/node  RAM [gb]  Total [*]  Used [*]  Avail [*]  Run.jobs/user
                        assert line_splitted[0] == node_kind
                        assert len(line_splitted) == 8
                        number_of_free_nodes = int(line_splitted[6])

                        if number_of_free_nodes < 0:
                            util.logging.warn('Number of free nodes in the following line is negative, setting free nodes to zero.\n{}'.format(line))
                            number_of_free_nodes = 0

                        util.logging.debug('Extracting nodes states from line "{}": node kind {} with {} free nodes.'.format(line, node_kind, number_of_free_nodes))
                        free_cpus = np.ones(number_of_free_nodes, dtype=np.uint32) * self.node_infos.cpus(node_kind)
                        free_memory = np.ones(number_of_free_nodes, dtype=np.uint32) * self.node_infos.memory(node_kind)

                        state[node_kind] = (free_cpus, free_memory)
            return util.batch.general.system.NodesState(state)



BATCH_SYSTEM = BatchSystem()


## job

class Job(util.batch.general.system.Job):

    def __init__(self, output_dir, force_load=False):
        from util.batch.nec.constants import EXCEEDED_WALLTIME_ERROR_MESSAGE
        super().__init__(output_dir, batch_system=BATCH_SYSTEM, force_load=force_load, exceeded_walltime_error_message=EXCEEDED_WALLTIME_ERROR_MESSAGE)


    def set_job_options(self, job_name, nodes_setup, queue=None):
        ## set queue if missing
        if queue is not None and queue != nodes_setup.node_kind:
            util.logging.warn('Queue {} and cpu kind {} have to be the same. Setting Queue to cpu kind.'.format(queue, nodes_setup.node_kind))
        queue = nodes_setup.node_kind

        ## super
        super().set_job_options(job_name, nodes_setup, queue=queue, cpu_kind=None)


    def _job_file_header(self, use_mpi=True):
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

