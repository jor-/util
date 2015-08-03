import subprocess
import time
import math
import re
import os

import numpy as np

import util.logging
logger = util.logging.logger

import util.rzcluster.constants



def start_job(job_file):
    logger.debug('Starting job with option file {}.'.format(job_file))
    
    if not os.path.exists(job_file):
        raise FileNotFoundError(job_file)

    job_id = subprocess.check_output((util.rzcluster.constants.QSUB_COMMAND, job_file)).decode("utf-8")
    job_id = job_id.rstrip()
    
    logger.debug('Job has ID {}.'.format(job_id))
    
    return job_id



def get_qstat_job_state(job_id):
    ## remove suffix from job id
    SUFFIX = '.rz.uni-kiel.de'
    if job_id.endswith(SUFFIX):
        job_id = job_id[:-len(SUFFIX)]
    
    ## get state of job
    process = subprocess.Popen((util.rzcluster.constants.QSTAT_COMMAND, '-a', job_id), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    qstat_result = process.communicate()[0].decode("utf-8")
    qstat_returncode = process.returncode
    
    logger.debug('qstat result: {} exit code: {}'.format(qstat_result, qstat_returncode))
    
    ## 255 => cannot connect to server
    if qstat_returncode == 255:
        raise ConnectionError(qstat_result)
    
    return qstat_returncode



def is_job_running(job_id):
    qstat_returncode = get_qstat_job_state(job_id)
    return qstat_returncode == 0

    
    
def is_job_finished(job_id):
    qstat_returncode = get_qstat_job_state(job_id)
    return qstat_returncode == 35 or qstat_returncode == 153
    


def wait_until_job_finished(job_id, wait_pause_seconds=30):
    logger.debug('Waiting for job {} to finish.'.format(job_id))
    
    while not is_job_finished(job_id):
        time.sleep(wait_pause_seconds)
    
    logger.debug('Job {} finished.'.format(job_id))



## node setups

def get_best_cpu_configurations_for_kind(kind, memory_required, nodes=None, cpus=None, nodes_max=float('inf'), nodes_left_free=0, total_cpus_max=float('inf')):
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
        command = '{} | grep -E {}'.format(util.rzcluster.constants.QNODES_COMMAND, expression)
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
    # # Opteron nodes (4 CPUs per node, 2.8 GHz)
    # elif kind == 'quad' or kind == 'opteron':
    #     grep_result = grep_qnodes('"rzcl03[6-9]|rzcl04[0-7]"')
    # # f_ocean2 express nodes (16 CPUs per node, 2.6 GHz) (foexpress queue)
    # elif kind == 'foexpress':
    #     grep_result = grep_qnodes('"rzcl27[2-3]"')
    # Shanghai Ethernet nodes (8 CPUs per node, 2.4 GHz) (bio_ocean queue)
    elif kind == 'bio_ocean' or kind == 'shanghai-ethernet':
        grep_result = grep_qnodes('"rzcl07[5-9]|rzcl0[8-9][0-9]|rzcl10[0-9]|rzcl11[0-4]"')
    # Shanghai Infiniband nodes (8 CPUs per node, 2.4 GHz) (math queue)
    elif kind == 'math' or kind == 'shanghai-infiniband':
        grep_result = grep_qnodes('"rzcl11[8-9]|rzcl1[2-3][0-9]|rzcl14[0-3]"')
#         # 3 AMD-Shanghai nodes (16 CPUs per node, 2.4 GHz)
#         elif kind == 'amd128':
#             grep_result = grep_qnodes('"rzcl11[5-7]"')
    else:
        raise ValueError('Unknown CPU kind: ' + kind)
    
    logger.debug(grep_result)
#     else:
#         grep_result = ''
#         logger.debug('Not using CPU kind {}.'.format(kind))
    
    
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
            free_memory[i] = math.floor((grep_memory[1] - grep_memory[0]) / (1024**2))
    
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
        for cpus in cpus_to_check:
            ## calculate useable nodes (respect max nodes and left free nodes)
            free_nodes = free_cpus[free_cpus >= cpus].size
            free_nodes = free_nodes - nodes_left_free
            free_nodes = min(free_nodes, nodes_max)
            
            ## respect fix number of nodes if passed
            if nodes is not None:
                if free_nodes >= nodes:
                    free_nodes = nodes
                else:
                    free_nodes = 0
            
            ## respect total max cpus
            while free_nodes * cpus > total_cpus_max:
                if free_nodes > 1:
                    free_nodes -=1
                else:
                    cpus -= 1
            
            ## check if best configuration
            if free_nodes * cpus > best_nodes * best_cpus:
                best_nodes = free_nodes
                best_cpus = cpus
    
    logger.debug('Best CPU configuration is for this kind: {}'.format((best_nodes, best_cpus)))
    
    assert best_nodes <= nodes_max
    assert best_nodes * best_cpus <= total_cpus_max
    return (best_nodes, best_cpus)


def get_best_cpu_configurations(memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_left_free=0, total_cpus_max=float('inf')):
    from util.rzcluster.constants import NODE_INFOS
    
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



def get_tmp_dir():
    return os.environ['TMPDIR']



class NodeSetup:
    
    def __init__(self, memory=1, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_left_free=0, total_cpus_min=1, total_cpus_max=float('inf')):

        assert total_cpus_max is None or total_cpus_min is None or total_cpus_max >= total_cpus_min
        assert nodes_max is None or nodes is None or nodes_max >= nodes
        assert total_cpus_min is None or nodes is None or cpus is None or total_cpus_min <= nodes * cpus
        assert total_cpus_max is None or nodes is None or cpus is None or total_cpus_max >= nodes * cpus
        assert total_cpus_max is None or nodes is None or total_cpus_max >= nodes
        assert total_cpus_max is None or cpus is None or total_cpus_max >= cpus
        
        ## prepare input
        if node_kind is not None and not isinstance(node_kind, str) and len(node_kind) == 1:
            node_kind = node_kind[0]
        if nodes_max == 1 and nodes is None:
            nodes = 1
        if nodes is not None and total_cpus_max == nodes:
            cpus = 1
        
        ## save setup
        setup = {'memory':memory, 'node_kind':node_kind, 'nodes':nodes, 'cpus':cpus, 'total_cpus_min':total_cpus_min, 'nodes_max':nodes_max, 'nodes_left_free':nodes_left_free, 'total_cpus_max':total_cpus_max}
        self.setup = setup
    
    def __getitem__(self, key):
        return self.setup[key]
    
    def __setitem__(self, key, value):
        self.setup[key] = value
    
    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, self.setup)
    
    def __copy__(self):
        copy = type(self)()
        copy.setup = self.setup.copy()   
        return copy
    
    def copy(self):
        return self.__copy__()
    
    # def __deepcopy__(self):
    #     return self.copy()
    
    
    def is_complete(self):
        return self['memory'] is not None and self['node_kind'] is not None and isinstance(self['node_kind'], str) and self['nodes'] is not None and self['cpus'] is not None
    
    
    def complete_missing_configuration(self):
        if not self.is_complete():
            (node_kind, nodes, cpus) = wait_for_needed_resources(self['memory'], node_kind=self['node_kind'], nodes=self['nodes'], cpus=self['cpus'], nodes_max=self['nodes_max'], nodes_left_free=self['nodes_left_free'], total_cpus_min=self['total_cpus_min'], total_cpus_max=self['total_cpus_max'])
            self['node_kind'] = node_kind
            self['nodes'] = nodes
            self['cpus'] = cpus
    
    
    def get_configuration_value(self, key, test=None):
        assert test is None or callable(test)
        
        value = self.setup[key]
        if value is None or (test is not None and not test(value)):
            self.complete_missing_configuration()
            value = self.setup[key]
        
        assert value is not None
        return value
        
    
    @property
    def memory(self):
        return self.setup['memory']
    
    @property
    def node_kind(self):
        return self.get_configuration_value('node_kind', test=lambda v : isinstance(v, str))   
    
    @property
    def nodes(self):
        return self.get_configuration_value('nodes')  
    
    @property
    def cpus(self):
        return self.get_configuration_value('cpus')    
    




