import abc
import math
import os
import subprocess
import time

import numpy as np

import util.io.fs
import util.options

import util.logging
logger = util.logging.logger


class NodeInfos():
    
    def __init__(self, node_infos):
        self.node_infos = node_infos
    
    def kinds(self):
        return tuple(self.node_infos.keys())

    def nodes(self, kind):
        return self.node_infos[kind]['nodes']

    def cpus(self, kind):
        return self.node_infos[kind]['cpus']

    def speed(self, kind):
        return self.node_infos[kind]['speed']

    def memory(self, kind):
        return self.node_infos[kind]['memory']

    def leave_free(self, kind):
        node_info_kind = self.node_infos[kind]
        try:
            return node_info_kind['leave_free']
        except KeyError:
            return 0

    def max_walltime(self, kind):
        node_info_kind = self.node_infos[kind]
        try:
            return node_info_kind['max_walltime']
        except KeyError:
            return float('inf')



class NodesState():
    
    def __init__(self, nodes_state):
        self.nodes_state = nodes_state
    
    def nodes_state_for_kind(self, kind):
        nodes_state_values_for_kind = self.nodes_state_values_for_kind(kind)
        return NodesState({kind: nodes_state_values_for_kind})
    
    def nodes_state_values_for_kind(self, kind):
        try:
            nodes_state_values_for_kind = self.nodes_state[kind]
        except KeyError as e:
            logger.warning('Node kind {} not found in nodes state {}.'.format(kind, self.nodes_state))
            nodes_state_values_for_kind = [np.array([]), np.array([])]
        return nodes_state_values_for_kind
    
    def free_cpus(self, kind, required_memory=0):
        if required_memory == 0:
            free_cpus = self.nodes_state_values_for_kind(kind)[0]
        else:
            free_memory = self.free_memory(kind)
            free_cpus = self.free_cpus(kind, required_memory=0)
            free_cpus = free_cpus[free_memory >= required_memory]
        return free_cpus

    def free_memory(self, kind):
        return self.nodes_state_values_for_kind(kind)[1]



class NodeSetup:

    def __init__(self, memory=None, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_min=1, total_cpus_max=float('inf'), check_for_better=False, walltime=None):
        
        ## set batch system
        from util.batch.universal.system import BATCH_SYSTEM
        self.batch_system = BATCH_SYSTEM
        
        ## check input
        assert nodes is None or nodes >= 1
        assert cpus is None or cpus >= 1
        assert total_cpus_max is None or total_cpus_min is None or total_cpus_max >= total_cpus_min
        assert nodes_max is None or nodes is None or nodes_max >= nodes
        assert total_cpus_min is None or nodes is None or cpus is None or total_cpus_min <= nodes * cpus
        assert total_cpus_max is None or nodes is None or cpus is None or total_cpus_max >= nodes * cpus
        assert total_cpus_max is None or nodes is None or total_cpus_max >= nodes
        assert total_cpus_max is None or cpus is None or total_cpus_max >= cpus
        
        if node_kind is not None and cpus is not None:
            max_cpus = self.batch_system.node_infos.cpus(node_kind)
            if cpus > max_cpus:
                raise ValueError('For node kind {} are maximal {} cpus per node available but {} are reqeusted'.format(node_kind, max_cpus, cpus))      
        
        if node_kind is not None and nodes is not None:
            max_nodes = self.batch_system.node_infos.nodes(node_kind)
            if nodes > max_nodes:
                raise ValueError('For node kind {} are maximal {} nodes available but {} are reqeusted'.format(node_kind, max_nodes, nodes))

        ## prepare input
        if node_kind is not None and not isinstance(node_kind, str) and len(node_kind) == 1:
            node_kind = node_kind[0]
        if nodes_max == 1 and nodes is None:
            nodes = 1
        if nodes is not None and total_cpus_max == nodes:
            cpus = 1

        ## save setup
        setup = {'memory': memory, 'node_kind': node_kind, 'nodes': nodes, 'cpus': cpus, 'nodes_max': nodes_max, 'nodes_leave_free': nodes_leave_free, 'total_cpus_min': total_cpus_min, 'total_cpus_max': total_cpus_max, 'check_for_better': check_for_better, 'walltime': walltime}
        self.setup = setup
        

    def __getitem__(self, key):
        return self.setup[key]

    def __setitem__(self, key, value):
        self.setup[key] = value

    def __str__(self):
        dict_str = str(self.setup).replace(': inf', ': float("inf")')
        return '{}(**{})'.format(self.__class__.__name__, dict_str)
    
    def __repr__(self):
        dict_str = str(self.setup).replace(': inf', ': float("inf")')
        return '{}.{}(**{})'.format(self.__class__.__module__, self.__class__.__name__, dict_str)
        

    def __copy__(self):
        copy = type(self)(**self.setup)
        return copy

    def copy(self):
        return self.__copy__()


    def configuration_is_complete(self):
        return self['memory'] is not None and self['node_kind'] is not None and isinstance(self['node_kind'], str) and self['nodes'] is not None and self['cpus'] is not None


    def complete_configuration(self):
        if not self.configuration_is_complete():
            logger.debug('Node setup incomplete. Try to complete it.')
            if self['memory'] is None:
                raise ValueError('Memory has to be set.')
            try:
                (node_kind, nodes, cpus) = self.batch_system.wait_for_needed_resources(self['memory'], node_kind=self['node_kind'], nodes=self['nodes'], cpus=self['cpus'], nodes_max=self['nodes_max'], nodes_leave_free=self['nodes_leave_free'], total_cpus_min=self['total_cpus_min'], total_cpus_max=self['total_cpus_max'])
            except NotImplementedError:
                logger.error('Batch system does not support completion of node setup.')
                raise NodeSetupIncompleteError(self)
            self['node_kind'] = node_kind
            self['nodes'] = nodes
            self['cpus'] = cpus


    def configuration_value(self, key, test=None):
        assert test is None or callable(test)

        value = self.setup[key]
        if value is None or (test is not None and not test(value)):
            self.complete_configuration()
            value = self.setup[key]

        assert value is not None
        return value


    def update_with_best_configuration(self, check_for_better=True, not_free_speed_factor=0.7):
        if check_for_better:
            self['check_for_better'] = False
            setup_triple = (self.node_kind, self.nodes, self.cpus)
            logger.debug('Try to find better node setup configuration than {}.'.format(setup_triple))
            speed = self.batch_system.speed(*setup_triple)
            best_setup_triple = self.batch_system.best_cpu_configurations(self.memory, nodes_max=self['nodes_max'], total_cpus_max=self['total_cpus_max'], walltime=self.walltime)
            best_speed = self.batch_system.speed(*best_setup_triple)
            if best_speed > speed:
                logger.debug('Using better node setup configuration {}.'.format(best_setup_triple))
                self['node_kind'], self['nodes'], self['cpus'] = best_setup_triple
            elif not self.batch_system.is_free(self.memory, self.node_kind, self.nodes, self.cpus):
                logger.debug('Node setup configuration {} is not free.'.format(setup_triple))
                if best_speed >= speed * not_free_speed_factor:
                    logger.debug('Using node setup configuration {}.'.format(best_setup_triple))
                    self['node_kind'], self['nodes'], self['cpus'] = best_setup_triple
                else:
                    logger.debug('Not using best node setup configuration {} since it is to slow.'.format(best_setup_triple))


    @property
    def memory(self):
        return self.setup['memory']
    
    @memory.setter
    def memory(self, memory):
        self.setup['memory'] = memory
    

    @property
    def node_kind(self):
        self.update_with_best_configuration(self['check_for_better'])
        return self.configuration_value('node_kind', test=lambda v: isinstance(v, str))

    @property
    def nodes(self):
        self.update_with_best_configuration(self['check_for_better'])
        return self.configuration_value('nodes')

    @property
    def cpus(self):
        self.update_with_best_configuration(self['check_for_better'])
        return self.configuration_value('cpus')


    @property
    def walltime(self):
        return self.setup['walltime']
    
    @walltime.setter
    def walltime(self, walltime):
        self.setup['walltime'] = walltime


    @property
    def total_cpus_min(self):
        return self.setup['total_cpus_min']
    
    @total_cpus_min.setter
    def total_cpus_min(self, total_cpus_min):
        self.setup['total_cpus_min'] = total_cpus_min


    @property
    def nodes_max(self):
        return self.setup['nodes_max']
    
    @nodes_max.setter
    def nodes_max(self, nodes_max):
        self.setup['nodes_max'] = nodes_max



class NodeSetupIncompleteError(Exception):

    def __init__(self, nodes_setup):
        error_message = 'The node setup is incomplete: node_kind={}, nodes={} and cpus={}.'.format(nodes_setup.node_kind, nodes_setup.nodes, nodes_setup.cpus)
        super().__init__(error_message)




class BatchSystem():

    def __init__(self, commands, queues, max_walltime={}, module_renaming={}, node_infos={}):
        self.commands = commands
        logger.debug('{} initiating with commands {}, queues {}, max_walltime {} and module_renaming {}.'.format(self, commands, queues, max_walltime, module_renaming))
        self.queues = queues
        self.max_walltime = max_walltime
        self.module_renaming = module_renaming
        
        if not isinstance(node_infos, NodeInfos):
            node_infos = NodeInfos(node_infos)
        self.node_infos = node_infos


    @property
    def mpi_command(self):
        return self.commands['mpirun']

    @property
    def time_command(self):
        return self.commands['time']

    @property
    def submit_command(self):
        return self.commands['sub']

    @property
    def status_command(self):
        return self.commands['stat']

    @property
    def nodes_command(self):
        return self.commands['nodes']


    def __str__(self):
        return 'General batch system'

    ## check methods

    def check_queue(self, queue):
        if queue is not None and queue not in self.queues:
            raise ValueError('Unknown queue {}.'.format(queue))
        return queue


    def check_walltime(self, queue, walltime_hours):
        ## get max walltime
        try:
            max_walltime_for_queue = self.max_walltime[queue]
        except KeyError:
            max_walltime_for_queue = float('inf')
        ## check walltime
        if walltime_hours is not None:
            if walltime_hours <= max_walltime_for_queue:
                walltime_hours = math.ceil(walltime_hours)
            else:
                raise ValueError('Max walltime {} is greater than max walltime for queue {}.'.format(walltime_hours, max_walltime_for_queue))
        else:
            if max_walltime_for_queue < float('inf'):
                walltime_hours = max_walltime_for_queue
        ## return
        assert (walltime_hours is None and max_walltime_for_queue == float('inf')) or walltime_hours <= max_walltime_for_queue
        return walltime_hours


    def check_modules(self, modules):
        if len(modules) > 0:
            modules = list(modules)

            ## add required modules
            for module_requested, modules_required in [('petsc', ('intelmpi', 'intel')), ('intelmpi', ('intel',)), ('hdf5', ('intel',))]:
                if module_requested in modules:
                    for module_required in modules_required:
                        if module_required not in modules:
                            logger.warning('Module "{}" needs module "{}" to be loaded first.'.format(module_requested, module_required))
                            modules = [module_required] + modules

            ## check positions
            first_index = 0
            for module in  ('intel', 'intelmpi'):
                if module in modules and modules[first_index] != module:
                    logger.warning('Module "{}" has to be at position {}.'.format(module, first_index))
                    modules.remove(module)
                    modules = [module] + modules
                first_index += 1

            ## rename modules
            for i in range(len(modules)):
                module = modules[i]
                try:
                    module_new = self.module_renaming[module]
                except KeyError:
                    module_new = module
                modules[i] = module_new
        return modules


    ## other methods

    def start_job(self, job_file):
        logger.debug('Starting job with option file {}.'.format(job_file))

        if not os.path.exists(job_file):
            raise FileNotFoundError(job_file)

        submit_output = subprocess.check_output((self.submit_command, job_file)).decode('utf-8')
        submit_output = submit_output.strip()
        job_id = self._get_job_id_from_submit_output(submit_output)

        logger.debug('Started job has ID {}.'.format(job_id))

        return job_id


    def is_mpi_used(self, modules):
        for module in modules:
            if 'intelmpi' in module:
                return True
        return False


    def add_mpi_to_command(self, command, cpus, use_mpi=True):
        if use_mpi:
            command = self.mpi_command.format(command=command, cpus=cpus)
        return command
    
    
    ## best node setups
    
    def speed(self, node_kind, nodes, cpus):
        return self.node_infos.speed(node_kind) * nodes * cpus
    
    
    def is_free(self, memory, node_kind, nodes, cpus):
        ## get nodes with required memory
        nodes_state = self._nodes_state()
        free_cpus = nodes_state.free_cpus(node_kind, required_memory=memory)
        
        ## calculate useable nodes
        free_nodes = free_cpus[free_cpus >= cpus].size
        free_nodes = free_nodes - self.node_infos.leave_free(node_kind)
        
        return free_nodes >= nodes
    
    
    @staticmethod
    def _best_cpu_configurations_for_state(nodes_state, node_kind, memory_required, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_max=float('inf')):
        logger.debug('Getting best cpu configuration for node state {} with memory {}, nodes {}, cpus {}, nodes max {} and nodes left free {}.'.format(nodes_state, memory_required, nodes, cpus, nodes_max, nodes_leave_free))
    
        ## check input
        if nodes_max <= 0:
            raise ValueError('nodes_max {} has to be greater 0.'.format(nodes_max))
        if total_cpus_max <= 0:
            raise ValueError('total_cpus_max {} has to be greater 0.'.format(total_cpus_max))
        if nodes_leave_free < 0:
            raise ValueError('nodes_leave_free {} has to be greater or equal to 0.'.format(nodes_leave_free))
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
    
        ## get only nodes with required memory
        free_cpus = nodes_state.free_cpus(node_kind, required_memory=memory_required)
    
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
                free_nodes = free_nodes - nodes_leave_free
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
                        free_nodes -= 1
                    else:
                        cpus_to_check_i = total_cpus_max
    
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

    
    def best_cpu_configurations(self, memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_max=float('inf'), walltime=None):

        logger.debug('Calculating best CPU configurations for {}GB memory with node kinds {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {}, total_cpus_max {} and walltime {}'.format(memory_required, node_kind, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_max, walltime))
    
        ## chose node kinds if not passed
        if node_kind is None:
            if walltime is None:
                walltime = 0
            node_kind = []
            for node_kind_i in self.node_infos.kinds():
                if self.node_infos.nodes(node_kind_i) > self.node_infos.leave_free(node_kind_i) and self.node_infos.max_walltime(node_kind_i) >= walltime:
                    node_kind.append(node_kind_i)
        elif isinstance(node_kind, str):
            node_kind = (node_kind,)
        nodes_state = self._nodes_state()
    
        ## init
        best_kind = node_kind[0]
        best_nodes = 0
        best_cpus = 0
        best_cpu_power = 0
    
        ## calculate best CPU configuration
        for node_kind_i in node_kind:
            nodes_cpu_power_i = self.node_infos.speed(node_kind_i)
            nodes_max_i = self.node_infos.nodes(node_kind_i)
            nodes_max_i = min(nodes_max, nodes_max_i)
            nodes_leave_free_i = self.node_infos.leave_free(node_kind_i)
            nodes_leave_free_i = max(nodes_leave_free, nodes_leave_free_i)
            
            (best_nodes_i, best_cpus_i) = self._best_cpu_configurations_for_state(nodes_state, node_kind_i, memory_required, nodes=nodes, cpus=cpus, nodes_max=nodes_max_i, nodes_leave_free=nodes_leave_free_i, total_cpus_max=total_cpus_max)

            logger.debug('Best CPU configurations for {}GB memory with node kind {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {} and total_cpus_max {} is {}.'.format(memory_required, node_kind_i, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_max, (best_nodes_i, best_cpus_i)))
    
            if nodes_cpu_power_i * best_cpus_i * best_nodes_i > best_cpu_power * best_cpus * best_nodes:
                best_kind = node_kind_i
                best_nodes = best_nodes_i
                best_cpus = best_cpus_i
                best_cpu_power = nodes_cpu_power_i
        
        ## return
        best_configuration = (best_kind, best_nodes, best_cpus)
    
        logger.debug('Best CPU configuration is: {}.'.format(best_configuration))
    
        assert best_kind in node_kind
        assert best_nodes <= nodes_max
        assert best_nodes * best_cpus <= total_cpus_max
        return best_configuration

    
    def wait_for_needed_resources(self, memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_min=1, total_cpus_max=float('inf')):
        logger.debug('Waiting for at least {} CPUs with {}GB memory, with node_kind {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {}, total_cpus_min {} and total_cpus_max {}.'.format(total_cpus_min, memory_required, node_kind, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_min, total_cpus_max))
    
        ## check input
        if total_cpus_min > total_cpus_max:
            raise ValueError('total_cpus_max has to be greater or equal to total_cpus_min, but {} < {}.'.format(total_cpus_max, total_cpus_min))
    
        ## calculate
        best_nodes = 0
        best_cpus = 0
        resources_free = False
        while not resources_free:
            (best_cpu_kind, best_nodes, best_cpus) = self.best_cpu_configurations(memory_required, node_kind=node_kind, nodes=nodes, cpus=cpus, nodes_max=nodes_max, nodes_leave_free=nodes_leave_free, total_cpus_max=total_cpus_max)
            cpus_avail = best_nodes * best_cpus
            resources_free = (cpus_avail >= total_cpus_min)
            if not resources_free:
                logger.debug('No enough resources free. {} CPUs available, but {} CPUs needed. Waiting ...'.format(cpus_avail, total_cpus_min))
                time.sleep(60)
    
        return (best_cpu_kind, best_nodes, best_cpus)


    ## abstract methods

    @abc.abstractmethod
    def _get_job_id_from_submit_output(self, submit_output):
        raise NotImplementedError()

    @abc.abstractmethod
    def _nodes_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def is_job_running(self, job_id):
        raise NotImplementedError()



class Job():

    def __init__(self, batch_system, output_dir, force_load=False, max_job_name_len=80, exceeded_walltime_error_message=None):
        ## batch system
        self.batch_system = batch_system
        self.max_job_name_len = max_job_name_len
        self.exceeded_walltime_error_message = exceeded_walltime_error_message

        ## check input
        if output_dir is None:
            raise ValueError('The output dir is not allowed to be None.')
        output_dir_expanded = os.path.expandvars(output_dir)
        
        ## get option file
        try:
            option_file_expanded = os.path.join(output_dir_expanded, 'job_options.hdf5')
        except Exception as e:
            raise ValueError('The output dir {} is not allowed.'.format(output_dir)) from e

        ## load option file if existing or forced
        if force_load or os.path.exists(option_file_expanded):
            self.__options = util.options.OptionsFile(option_file_expanded, mode='r+', replace_environment_vars_at_get=True)
            logger.debug('Job {} loaded.'.format(option_file_expanded))

        ## make new job options file otherwise
        else:
            os.makedirs(output_dir_expanded, exist_ok=True)

            self.__options = util.options.OptionsFile(option_file_expanded, mode='w-', replace_environment_vars_at_get=True)

            self.options['/job/output_file'] = os.path.join(output_dir, 'job_output.txt')
            self.options['/job/option_file'] = os.path.join(output_dir, 'job_options.txt')
            self.options['/job/id_file'] = os.path.join(output_dir, 'job_id.txt')
            self.options['/job/unfinished_file'] = os.path.join(output_dir, 'unfinished.txt')
            self.options['/job/finished_file'] = os.path.join(output_dir, 'finished.txt')

            logger.debug('Job {} initialized.'.format(option_file_expanded))


    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def __str__(self):
        output_dir = self.output_dir
        try:
            job_id = self.id
        except KeyError:
            job_id = None
        if job_id is not None:
            job_str = 'job {} with output path {}'.format(job_id, output_dir)
        else:
            job_str = 'not started job with output path {}'.format(job_id, output_dir)
        return job_str


    @property
    def options(self):
        return self.__options


    ## option properties

    def option_value(self, name, not_exist_okay=False, replace_environment_vars=True):
        replace_environment_vars_old = self.options.replace_environment_vars_at_get
        self.options.replace_environment_vars_at_get = replace_environment_vars
        try:
            try:
                return self.options[name]
            except KeyError:
                if not_exist_okay:
                    return None
                else:
                    raise
        finally:
            self.options.replace_environment_vars_at_get = replace_environment_vars_old


    @property
    def id(self):
        try:
            return self.options['/job/id']
        except KeyError:
            raise JobNotStartedError(self)

    @property
    def output_dir(self):
        return os.path.dirname(self.option_value('/job/output_file', not_exist_okay=False))

    @property
    def output_dir_not_expanded(self):
        return os.path.dirname(self.option_value('/job/output_file', not_exist_okay=False, replace_environment_vars=False))

    @property
    def output_file(self):
        return self.option_value('/job/output_file', not_exist_okay=False)

    @property
    def output(self):
        output_file = self.output_file
        with open(output_file, 'r') as file:
            output = file.read()
        return output

    @property
    def option_file(self):
        return self.option_value('/job/option_file', not_exist_okay=False)

    @property
    def unfinished_file(self):
        return self.option_value('/job/unfinished_file', not_exist_okay=False)

    @property
    def finished_file(self):
        return self.option_value('/job/finished_file', not_exist_okay=False)

    @property
    def id_file(self):
        return self.option_value('/job/id_file', not_exist_okay=False)
    
    @property
    def exit_code(self):
        ## check if finished file exists
        if not os.path.exists(self.finished_file):
            ValueError('Finished file {} does not exist. The job is not finished'.format(self.finished_file))
        ## read exit code
        with open(self.finished_file, mode='r') as finished_file:
            exit_code = finished_file.read()
        ## check exit code
        if len(exit_code) > 0:
            try:
                exit_code = int(exit_code)
                return exit_code
            except ValueError:
                raise ValueError('Finished file {} does not contain an exit code but rather {}.'.format(self.finished_file, exit_code))
        else:
            raise ValueError('Finished file {} is empty.'.format(self.finished_file))

    @property
    def cpu_kind(self):
        return self.option_value('/job/cpu_kind', not_exist_okay=True)

    @property
    def nodes(self):
        return self.option_value('/job/nodes', not_exist_okay=True)

    @property
    def cpus(self):
        return self.option_value('/job/cpus', not_exist_okay=True)

    @property
    def queue(self):
        return self.option_value('/job/queue', not_exist_okay=True)

    @property
    def walltime_hours(self):
        return self.option_value('/job/walltime_hours', not_exist_okay=True)


    ## init methods

    def init_job_file(self, job_name, nodes_setup, queue=None, cpu_kind=None):
        ## check qeue and walltime
        queue = self.batch_system.check_queue(queue)
        walltime_hours = nodes_setup.walltime
        walltime_hours = self.batch_system.check_walltime(queue, walltime_hours)

        ## set job options
        self.options['/job/memory_gb'] = nodes_setup.memory
        self.options['/job/nodes'] = nodes_setup.nodes
        self.options['/job/cpus'] = nodes_setup.cpus
        self.options['/job/queue'] = queue
        self.options['/job/name'] = job_name[:self.max_job_name_len]
        if cpu_kind is not None:
            self.options['/job/cpu_kind'] = cpu_kind
        if walltime_hours is not None:
            self.options['/job/walltime_hours'] = walltime_hours


    @abc.abstractmethod
    def _make_job_file_header(self, use_mpi):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_job_file_modules(self, modules):
        raise NotImplementedError()

    def _make_job_file_command(self, run_command, pre_run_command=None, add_timing=True):
        if add_timing:
            run_command = self.batch_system.time_command.format(command=run_command)
        content = []
        content.append('touch {}'.format(self.options['/job/unfinished_file']))
        content.append('echo "Job started."')
        content.append('')
        if pre_run_command is not None:
            content.append(pre_run_command)
        content.append(run_command)
        content.append('')
        content.append('EXIT_CODE=$?')
        content.append('echo "Job finished with exit code $EXIT_CODE."')
        content.append('rm {}'.format(self.options['/job/unfinished_file']))
        content.append('echo $EXIT_CODE > {}'.format(self.options['/job/finished_file']))
        content.append('exit')
        content.append('')
        return os.linesep.join(content)


    def write_job_file(self, run_command, pre_run_command=None, modules=()):
        modules = self.batch_system.check_modules(modules)
        use_mpi = self.batch_system.is_mpi_used(modules)
        cpus = self.options['/job/nodes'] * self.options['/job/cpus']
        run_command = self.batch_system.add_mpi_to_command(run_command, cpus, use_mpi=use_mpi)
        with open(self.option_file, mode='w') as file:
            file.write(self._make_job_file_header(use_mpi=use_mpi))
            file.write(self._make_job_file_modules(modules))
            file.write(self._make_job_file_command(run_command, pre_run_command=pre_run_command))


    ## other methods

    def start(self):
        job_id = self.batch_system.start_job(self.options['/job/option_file'])
        self.options['/job/id'] = job_id

        id_file = self.id_file
        if id_file is not None:
            with open(self.options['/job/id_file'], 'w') as id_file:
                id_file.write(job_id)


    def is_started(self):
        try:
            self.options['/job/id']
            return True
        except KeyError:
            return False


    def is_finished(self, check_exit_code=True):
        ## if finished file exists, check exit code and output file
        if os.path.exists(self.finished_file):
            if check_exit_code:
                exit_code = self.exit_code
                if exit_code != 0:
                    raise JobExitCodeError(self)
            return self.output_file is None or os.path.exists(self.output_file)
        
        ## if finished file odes not exist, check if running
        elif self.is_started() and not self.batch_system.is_job_running(self.id):
            time.sleep(60)
            if os.path.exists(self.finished_file):
                return self.is_finished(check_exit_code=check_exit_code)
            else:
                output = self.output
                if self.exceeded_walltime_error_message is not None and self.exceeded_walltime_error_message in output:
                    raise JobExceededWalltimeError(self)
                else:
                    raise JobError(self, 'The job is not finished but it is not running! The finished file {} is missing'.format(self.finished_file), output)
        
        ## if not not started or running, return false
        else:
            return False


    def is_running(self):
        return self.is_started() and not self.is_finished(check_exit_code=False)


    def wait_until_finished(self, check_exit_code=True, pause_seconds=None, pause_seconds_min=5, pause_seconds_max=60, pause_seconds_increment_cycle=50):
        adaptive = pause_seconds is None
        if adaptive:
            logger.debug('Waiting for job {} to finish with adaptive sleep period with min {} and max {} seconds and increment cycle {}.'.format(self.id, pause_seconds_min, pause_seconds_max, pause_seconds_increment_cycle))
            pause_seconds = pause_seconds_min
        else:
            logger.debug('Waiting for job {} to finish with {}s sleep period.'.format(self.id, pause_seconds))

        cycle = 0
        while not self.is_finished(check_exit_code=check_exit_code):
            time.sleep(pause_seconds)

            if adaptive:
                cycle += 1
                if cycle == pause_seconds_increment_cycle:
                    pause_seconds += 1
                    cycle = 0

        logger.debug('Job {} finished with exit code {}.'.format(self.id, self.exit_code))


    def make_read_only_input(self, read_only=True):
        if read_only:
            self.options.make_read_only()
            util.io.fs.make_read_only(self.option_file)
            util.io.fs.make_read_only(self.id_file)

    def make_read_only_output(self, read_only=True):
        if read_only:
            if self.output_file is not None:
                util.io.fs.make_read_only(self.output_file)
            util.io.fs.make_read_only(self.finished_file)

    def make_read_only(self, read_only=True):
        self.make_read_only_input(read_only=read_only)
        self.make_read_only_output(read_only=read_only)


    def close(self):
        try:
            options = self.__options
        except AttributeError:
            options = None

        if options is not None:
            options.close()
    
    
    ## check integrity

    def check_integrity(self, should_be_started=False, should_be_readonly=False):
        
        ## check if options entires exist
        self.option_file
        self.output_file
        self.unfinished_file
        self.finished_file
        self.id_file
        
        ## check if started, running and finished state
        is_started = self.is_started()
        is_running = self.is_running()
        
        if is_started or should_be_started:
            job_id = self.id
            try:
                is_running_batch_system = self.batch_system.is_job_running(job_id)
            except:
                pass
            else:
                if is_running != is_running_batch_system:
                    raise JobError(self, 'Its is not clear if the job is running!')
        
        if is_started and not is_running:
            is_finished = self.is_finished(check_exit_code=True)
            if not is_finished:
                raise JobError(self, 'The job should finished but it is not!')
        else:
            is_finished = self.is_finished(check_exit_code=False)
        
        ## check errors in output file
        if is_started and os.path.exists(self.output_file) or is_finished:
            output = self.output
            for line in output.splitlines():
                line_lower = line.lower()
                if ('error' in line_lower and not 'error_path' in line_lower) or 'warning' in line_lower or 'fatal' in line_lower or 'permission denied' in line_lower:
                    raise JobError(self, 'There are errors in the job output file: {}!'.format(line))
        
        ## check read only
        if should_be_readonly and not self.options.is_read_only():
            raise JobError(self, 'Job option file is writeable!')
        
        ## check files
        output_dir = self.output_dir
        def check_if_file_exists(file, should_exists=True):
            if not file.startswith(output_dir):
                raise JobError(self, 'Job option {} should start with {} but its is {}.'.format(file_key, output_dir, file))
            exists =  os.path.exists(file)
            if should_exists and not exists:
                raise JobError(self, 'File {} does not exist.'.format(file))
            if not should_exists and exists:
                raise JobError(self, 'File {} should not exist.'.format(file))
        
        if is_started:
            check_if_file_exists(self.option_file)
            check_if_file_exists(self.id_file)        
        if is_finished:
            check_if_file_exists(self.output_file)
            check_if_file_exists(self.finished_file)
            check_if_file_exists(self.unfinished_file, should_exists=False)
        if is_running:
            check_if_file_exists(self.finished_file, should_exists=False)
            check_if_file_exists(self.unfinished_file)
            


class JobError(Exception):
    def __init__(self, job, error_message, include_output=False):
        ## store job
        self.job = job
        
        ## construct error message
        output_dir = job.output_dir
        if job.is_started:
            job_id = job.id
            error_message = 'An error accured in job {} stored at {}: {}'.format(job_id, output_dir, error_message)
        else:        
            error_message = 'An error accured in job stored at {}: {}'.format(output_dir, error_message)

        ## add output
        if include_output:
            try:
                output = job.output
            except OSError:
                pass
            else:
                error_message = error_message + '\nThe job output was:\n{}'.format(output)
        
        ## super call
        super().__init__(error_message)

class JobNotStartedError(JobError):
    def __init__(self, job):
        error_message = 'The job is not started!'
        super().__init__(job)

class JobExitCodeError(JobError):
    def __init__(self, job):
        self.exit_code = job.exit_code
        error_message = 'The command of the job exited with code {}.'.format(self.exit_code)
        super().__init__(job, error_message, include_output=True)

class JobExceededWalltimeError(JobError):
    def __init__(self, job):
        self.walltime = job.walltime_hours
        error_message = 'The job exceeded walltime {}.'.format(self.walltime)
        super().__init__(job, error_message)

class JobMissingOptionError(JobError):
    def __init__(self, job, option):
        error_message = 'Job option {} is missing!'.format(option)
        super().__init__(job, error_message)
