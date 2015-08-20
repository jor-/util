import os
import re
import subprocess
import time

import numpy as np

import util.batch.general.system

import util.logging
logger = util.logging.logger



## batch setup

class BatchSystem(util.batch.general.system.BatchSystem):

    def __init__(self):
        # from util.batch.rz.constants import QSUB_COMMAND, MPI_COMMAND, QSTAT_COMMAND, QUEUES, MAX_WALLTIME, MODEL_RENAMING
        # super().__init__(QSUB_COMMAND, MPI_COMMAND, QUEUES, max_walltime=MAX_WALLTIME, module_renaming=MODEL_RENAMING)
        # self.status_command = QSTAT_COMMAND
        from util.batch.nec.constants import COMMANDS, QUEUES, MAX_WALLTIME, MODEL_RENAMING
        super().__init__(COMMANDS, QUEUES, max_walltime=MAX_WALLTIME, module_renaming=MODEL_RENAMING)


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





BATCH_SYSTEM = BatchSystem()


## job

class Job(util.batch.general.system.Job):

    def __init__(self, output_dir, force_load=False):
        super().__init__(BATCH_SYSTEM, output_dir, force_load=force_load, max_job_name_len=15)


    def init_job_file(self, job_name, nodes_setup, queue=None, walltime_hours=None, write_output_file=True):
        from util.batch.universal.constants import MAX_WALLTIME, QUEUES

        ## set queue if missing
        cpu_kind = nodes_setup.node_kind

        if cpu_kind in ('f_ocean', 'f_ocean2'):
            if queue is not None and queue != cpu_kind:
                logger.warning('Queue {1} not supported for CPU kind {2}. CPU kind changed to {2}'.format(queue, cpu_kind))
            queue = cpu_kind
        else:
            if queue is None:
                if walltime_hours is None:
                    queue = 'medium'
                elif walltime_hours <= MAX_WALLTIME['express']:
                    queue = 'express'
                elif walltime_hours <= MAX_WALLTIME['small']:
                    queue = 'small'
                elif walltime_hours <= MAX_WALLTIME['medium']:
                    queue = 'medium'
                elif walltime_hours <= MAX_WALLTIME['long']:
                    queue = 'long'
                elif walltime_hours <= MAX_WALLTIME['para_low']:
                    queue = 'para_low'
                else:
                    raise ValueError('Walltime hours > {} are not supported.'.format(MAX_WALLTIME['para_low']))

        ## set cpu kind
        if cpu_kind == 'f_ocean2':
            cpu_kind = None
        if cpu_kind == 'shanghai':
            cpu_kind = 'all'

        ## super
        super().init_job_file(job_name, nodes_setup, queue=queue, cpu_kind=cpu_kind, walltime_hours=walltime_hours, write_output_file=write_output_file)



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
            ## intel compiler and mpi
            if 'intel' in modules:
                content.append('. /cluster/Software/intel14/composer_xe_2013_sp1/bin/compilervars.sh intel64')
                while'intel' in modules:
                    modules.remove('intel')
            if 'intelmpi' in modules or 'intelmpi' in modules:
                content.append('. /cluster/Software/intel14/impi/4.1.1.036/intel64/bin/mpivars.sh')
                while'intelmpi' in modules:
                    modules.remove('intelmpi')
            ## system modules
            for module in modules:
                content.append('module load {}'.format(module))
            content.append('module list')
            content.append('')
            content.append('')
        return os.linesep.join(content)




## node setups

def get_best_cpu_configurations_for_kind(kind, memory_required, nodes=None, cpus=None, nodes_max=float('inf'), nodes_left_free=0, total_cpus_max=float('inf')):
    from util.batch.rz.constants import QNODES_COMMAND
    logger.debug('Getting best cpu configuration for kind {} with mermory {}, nodes {}, cpus {}, nodes max {} and nodes left free {}.'.format(kind, memory_required, nodes, cpus, nodes_max, nodes_left_free))

    ## check input
    if nodes_max <= 0:
        raise ValueError('nodes_max {} has to be greater 0.'.format(nodes_max))
    if total_cpus_max <= 0:
        raise ValueError('total_cpus_max {} has to be greater 0.'.format(total_cpus_max))
    if nodes_left_free < 0:
        raise ValueError('nodes_left_free {} has to be greater or equal to 0.'.format(nodes_left_free))
    if nodes is not None:
        if nodes <= 0:
            raise ValueError('nodes {} has to be greater 0.'.format(nodes))
        if nodes > nodes_max:
            raise ValueError('nodes_max {} has to be greater or equal to nodes {}.'.format(nodes_max, nodes))
    if cpus is not None:
        if cpus <= 0:
            raise ValueError('cpus {} has to be greater 0.'.format(cpus))
    if nodes is not None and cpus is not None:
        if nodes * cpus > total_cpus_max:
            raise ValueError('total_cpus_max {} has to be greater or equal to nodes {} multiplied with cpus {}.'.format(total_cpus_max, nodes, cpus))


    ## grep free nodes
    def grep_qnodes(expression):
        command = '{} | grep -E {}'.format(QNODES_COMMAND, expression)
        # command = '/usr/local/bin/qnodes | grep -E ' + expression
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
    free_cpus = np.empty(number_of_nodes, dtype=np.int)
    free_memory = np.empty(number_of_nodes, dtype=np.int)

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

    ## get only nodes with required memory
    free_cpus = free_cpus[free_memory >= memory_required]

    ## calculate best configuration
    best_nodes = 0
    best_cpus = 0

    if len(free_cpus) > 0:
        ## chose numbers of cpus to check
        if cpus is not None:
            cpus_to_check = (cpus,)
        else:
            cpus_to_check = range(max(free_cpus), 0, -1)

        ## get number of nodes for each number of cpus
        for cpus_to_check_i in cpus_to_check:
            ## calculate useable nodes (respect max nodes and left free nodes)
            free_nodes = free_cpus[free_cpus >= cpus_to_check_i].size
            free_nodes = free_nodes - nodes_left_free
            free_nodes = min(free_nodes, nodes_max)

            ## respect fix number of nodes if passed
            if nodes is not None:
                if free_nodes >= nodes:
                    free_nodes = nodes
                else:
                    free_nodes = 0

            ## respect total max cpus
            while free_nodes * cpus_to_check_i > total_cpus_max:
                if free_nodes > 1:
                    free_nodes -=1
                else:
                    cpus -= 1

            ## check if best configuration
            if free_nodes * cpus_to_check_i > best_nodes * best_cpus:
                best_nodes = free_nodes
                best_cpus = cpus_to_check_i

    logger.debug('Best CPU configuration is for this kind: {}'.format((best_nodes, best_cpus)))

    assert best_nodes <= nodes_max
    assert best_nodes * best_cpus <= total_cpus_max
    assert nodes is None or best_nodes == nodes or best_nodes == 0
    assert cpus is None or best_cpus == cpus or best_cpus == 0
    return (best_nodes, best_cpus)


def get_best_cpu_configurations(memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_left_free=0, total_cpus_max=float('inf')):
    from util.batch.rz.constants import NODE_INFOS

    ## chose node kinds if not passed
    if node_kind is None:
        node_kind = NODE_INFOS.keys()
    elif isinstance(node_kind, str):
        node_kind = (node_kind,)

    ## init
    logger.debug('Calculating best CPU configurations for {}GB memory with node kinds {}, nodes {}, cpus {}, nodes_max {}, nodes_left_free {} and total_cpus_max {}'.format(memory_required, node_kind, nodes, cpus, nodes_max, nodes_left_free, total_cpus_max))

    best_kind = node_kind[0]
    best_nodes = 0
    best_cpus = 0
    best_cpu_power = 0

    ## calculate best CPU configuration
    for node_kind_i in node_kind:
        node_info_values_i = NODE_INFOS[node_kind_i]
        nodes_cpu_power_i, nodes_max_i, nodes_left_free_i = node_info_values_i
        nodes_max_i = min(nodes_max, nodes_max_i)
        nodes_left_free_i = max(nodes_left_free, nodes_left_free_i)
        (best_nodes_i, best_cpus_i) = get_best_cpu_configurations_for_kind(node_kind_i, memory_required, nodes=nodes, cpus=cpus, nodes_max=nodes_max_i, nodes_left_free=nodes_left_free_i)

        if nodes_cpu_power_i * best_cpus_i * best_nodes_i > best_cpu_power * best_cpus * best_nodes:
            best_kind = node_kind_i
            best_nodes = best_nodes_i
            best_cpus = best_cpus_i
            best_cpu_power = nodes_cpu_power_i

    best_configuration = (best_kind, best_nodes, best_cpus)

    logger.debug('Best CPU configuration is: {}.'.format(best_configuration))

    assert best_kind in node_kind
    assert best_nodes <= nodes_max
    assert best_nodes * best_cpus <= total_cpus_max
    return best_configuration



def wait_for_needed_resources(memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_left_free=0, total_cpus_min=1, total_cpus_max=float('inf')):
    logger.debug('Waiting for at least {} CPUs with {}GB memory, with node_kind {}, nodes {}, cpus {}, nodes_max {}, nodes_left_free {}, total_cpus_min{} and total_cpus_max {}.'.format(total_cpus_min, memory_required, node_kind, nodes, cpus, nodes_max, nodes_left_free, total_cpus_min, total_cpus_max))

    ## check input
    if total_cpus_min > total_cpus_max:
        raise ValueError('total_cpus_max has to be greater or equal to total_cpus_min, but {} < {}.'.format(total_cpus_max, total_cpus_min))

    ## calculate
    best_nodes = 0
    best_cpus = 0
    resources_free = False
    while not resources_free:
        (best_cpu_kind, best_nodes, best_cpus) = get_best_cpu_configurations(memory_required, node_kind=node_kind, nodes=nodes, cpus=cpus, nodes_max=nodes_max, nodes_left_free=nodes_left_free, total_cpus_max=total_cpus_max)
        cpus_avail = best_nodes * best_cpus
        resources_free = (cpus_avail >= total_cpus_min)
        if not resources_free:
            logger.debug('No enough resources free. {} CPUs available, but {} CPUs needed. Waiting ...'.format(cpus_avail, total_cpus_min))
            time.sleep(60)

    return (best_cpu_kind, best_nodes, best_cpus)


class NodeSetup(util.batch.general.system.NodeSetup):

    def configuration_missing(self):
        if not self.configuration_is_complete():
            (node_kind, nodes, cpus) = wait_for_needed_resources(self['memory'], node_kind=self['node_kind'], nodes=self['nodes'], cpus=self['cpus'], nodes_max=self['nodes_max'], nodes_left_free=self['nodes_left_free'], total_cpus_min=self['total_cpus_min'], total_cpus_max=self['total_cpus_max'])
            self['node_kind'] = node_kind
            self['nodes'] = nodes
            self['cpus'] = cpus