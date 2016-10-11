import os
import re
import subprocess

import numpy as np

import util.batch.general.system

import util.logging
logger = util.logging.logger

from util.batch.general.system import *



## batch setup

class BatchSystem(util.batch.general.system.BatchSystem):

    def __init__(self):
        from util.batch.rz.constants import COMMANDS, QUEUES, MAX_WALLTIME, MODEL_RENAMING, NODE_INFOS
        super().__init__(COMMANDS, QUEUES, max_walltime=MAX_WALLTIME, module_renaming=MODEL_RENAMING, node_infos=NODE_INFOS)


    def __str__(self):
        return 'RZ batch system'


    def _get_job_id_from_submit_output(self, submit_output):
        return submit_output


    def job_state(self, job_id):
        ## remove suffix from job id
        SUFFIX = '.rz.uni-kiel.de'
        if job_id.endswith(SUFFIX):
            job_id = job_id[:-len(SUFFIX)]

        ## get state of job
        process = subprocess.Popen((self.status_command, '-a', job_id), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        qstat_result = process.communicate()[0].decode("utf-8")
        qstat_returncode = process.returncode

        logger.debug('qstat result: {} with exit code: {}'.format(qstat_result, qstat_returncode))

        ## 255 => cannot connect to server
        if qstat_returncode == 255:
            raise ConnectionError(qstat_result)

        return qstat_returncode


    def is_job_running(self, job_id):
        qstat_returncode = self.job_state(job_id)
        return qstat_returncode == 0


    def is_job_finished(self, job_id):
        qstat_returncode = self.job_state(job_id)
        return qstat_returncode == 35 or qstat_returncode == 153


    ## node setups
    
    def _nodes_state_one_kind(self, kind):
        logger.debug('Getting nodes state for kind {}.'.format(kind))
    
        ## grep free nodes
        def grep_qnodes(expression):
            command = '{} | grep -E {}'.format(self.nodes_command, expression)
            try:
                grep_result = subprocess.check_output(command, shell=True).decode("utf-8")
            except subprocess.CalledProcessError as e:
                logger.warning('Command {} returns with exit code {} and output "{}"'.format(command, e.returncode, e.output.decode("utf-8")))
                grep_result = 'offline'
    
            return grep_result
    
        # 24 f_ocean Barcelona nodes (8 CPUs per node, 2.1 GHz) (f_ocean queue)
        if kind == 'f_ocean' or  kind == 'barcelona':
            grep_result = grep_qnodes('"rzcl05[1-9]|rzcl06[0-9]|rzcl07[0-4]"')
        # 12 f_ocean2 nodes (16 CPUs per node, 2.6 GHz) (f_ocean2 queue)
        elif kind == 'f_ocean2':
            grep_result = grep_qnodes('"rzcl26[2-9]|rzcl27[0-3]"')
        # 2 fobigmem nodes (32 CPUs per node, 2.6 GHz) (fobigmem queue)
        elif kind == 'fobigmem':
            grep_result = grep_qnodes('"rzcl28[7-8]"')
        # 18 Westmere-nodes (12 CPUs per node, 2.67 GHz)
        elif kind == 'westmere':
            grep_result = grep_qnodes('"rzcl17[8-9]|rzcl18[0-9]|rzcl19[0-5]"')
        # 26 AMD-Shanghai nodes (8 CPUs per node, 2.4 GHz)
        elif kind == 'shanghai':
            grep_result = grep_qnodes('"rzcl11[8-9]|rzcl1[2-3][0-9]|rzcl14[0-3]"')
        # 1 AMD-Shanghai nodes (16 CPUs per node, 2.4 GHz)
        elif kind == 'amd128':
            grep_result = grep_qnodes('"rzcl116"')
        # 1 AMD-Magny node (48 CPUs per node, 2.1 GHz)
        elif kind == 'amd256':
            grep_result = grep_qnodes('"rzcl200"')
        # Shanghai Ethernet nodes (8 CPUs per node, 2.4 GHz) (bio_ocean queue)
        elif kind == 'bio_ocean' or kind == 'shanghai-ethernet':
            grep_result = grep_qnodes('"rzcl07[5-9]|rzcl0[8-9][0-9]|rzcl10[0-9]|rzcl11[0-4]"')
        # Shanghai Infiniband nodes (8 CPUs per node, 2.4 GHz) (math queue)
        elif kind == 'math' or kind == 'shanghai-infiniband':
            grep_result = grep_qnodes('"rzcl11[8-9]|rzcl1[2-3][0-9]|rzcl14[0-3]"')
        else:
            raise ValueError('Unknown CPU kind: ' + kind)
    
        logger.debug(grep_result)
    
        ## extract free cpus and memory from grep result
        grep_result_lines = grep_result.splitlines()
    
        number_of_nodes = len(grep_result_lines)
        free_cpus = np.empty(number_of_nodes, dtype=np.uint32)
        free_memory = np.empty(number_of_nodes, dtype=np.uint32)
    
        # format: "rzcl179 (7/12) (41943040kb/49449316kb) (free) (1285234.rzcluster/6)"
        for i in range(number_of_nodes):
            grep_result_line = grep_result_lines[i]
    
            # check if node down
            if 'down' in grep_result_line or 'state-unknown' in grep_result_line or 'offline' in grep_result_line:
                free_cpus[i] = 0
                free_memory[i] = 0
            # calculte free cpus and memory
            else:
                grep_result_line_split = grep_result_line.split()
    
                grep_cpus = [int(int_str) for int_str in re.findall('\d+', grep_result_line_split[1])]
                grep_memory = [int(int_str) for int_str in re.findall('\d+', grep_result_line_split[2])]
    
                free_cpus[i] = grep_cpus[1] - grep_cpus[0]
                free_memory[i] = int(np.floor((grep_memory[1] - grep_memory[0]) / (1024**2)))
    
        return (free_cpus, free_memory)


    def _nodes_state(self):
        state = {}
        for kind in self.node_infos.kinds():
            state[kind] = self._nodes_state_one_kind(kind)
        return util.batch.general.system.NodesState(state)
    


BATCH_SYSTEM = BatchSystem()


## job

class Job(util.batch.general.system.Job):

    def __init__(self, output_dir, force_load=False):
        super().__init__(BATCH_SYSTEM, output_dir, force_load=force_load, max_job_name_len=15)


    def init_job_file(self, job_name, nodes_setup, queue=None):
        ## set queue if missing
        cpu_kind = nodes_setup.node_kind

        if cpu_kind in ('f_ocean', 'f_ocean2'):
            if queue is not None and queue != cpu_kind:
                logger.warning('Queue {1} not supported for CPU kind {2}. CPU kind changed to {2}'.format(queue, cpu_kind))
            queue = cpu_kind
        else:
            if queue is None:
                walltime_hours = nodes_setup.walltime
                if walltime_hours is None:
                    queue = 'medium'
                elif walltime_hours <= self.batch_system.max_walltime['express']:
                    queue = 'express'
                elif walltime_hours <= self.batch_system.max_walltime['small']:
                    queue = 'small'
                elif walltime_hours <= self.batch_system.max_walltime['medium']:
                    queue = 'medium'
                elif walltime_hours <= self.batch_system.max_walltime['long']:
                    queue = 'long'
                elif walltime_hours <= self.batch_system.max_walltime['para_low']:
                    queue = 'para_low'
                else:
                    raise ValueError('Walltime hours > {} are not supported.'.format(self.batch_system.max_walltime['para_low']))

        ## set cpu kind
        if cpu_kind == 'f_ocean2':
            cpu_kind = None
        if cpu_kind == 'shanghai':
            cpu_kind = 'all'

        ## super
        super().init_job_file(job_name, nodes_setup, queue=queue, cpu_kind=cpu_kind)


    def _make_job_file_header(self, use_mpi):
        content = []
        ## shell
        content.append('#!/bin/bash')
        content.append('')
        ## name
        content.append('#PBS -N {}'.format(self.options['/job/name']))
        ## output file
        if self.output_file is not None:
            content.append('#PBS -j oe')
            content.append('#PBS -o {}'.format(self.output_file))
        ## queue
        content.append('#PBS -q {}'.format(self.options['/job/queue']))
        ## walltime
        if self.walltime_hours is not None:
            content.append('#PBS -l walltime={:02d}:00:00'.format(self.walltime_hours))
        ## select
        if self.cpu_kind is not None:
            cpu_kind_select = '{}=true:'.format(self.cpu_kind)
        else:
            cpu_kind_select = ''
        content.append('#PBS -l select={:d}:{}ncpus={:d}:mem={:d}gb'.format(self.options['/job/nodes'], cpu_kind_select, self.options['/job/cpus'], self.options['/job/memory_gb']))
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

