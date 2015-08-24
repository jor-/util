import abc
import math
import os
import subprocess
import time

import util.io.fs
import util.options
import util.batch.universal.system

import util.logging
logger = util.logging.logger



class BatchSystem():

    def __init__(self, commands, queues, max_walltime={}, module_renaming={}, node_infos={}):
        self.commands = commands
        # self.submit_command = submit_command
        # self.mpi_command = mpi_command
        # self.time_command = 'command time -f "Statistics for %C:\nexit code: %x, elapsed time: %Es\nCPU: %Us user mode, %Ss kernel mode, %P workload\nMemory: %Mkb max, %W swap outs\nContex switches: %c involuntarily, %w voluntarily\nPage faults: %F major, %R minor\nFile system I/O: %I inputs, %O outputs" {command}'
        # logger.debug('{} initiated with submit_command {}, queues {}, max_walltime {} and module_renaming {}.'.format(self, submit_command, queues, max_walltime, module_renaming))
        logger.debug('{} initiating with commands {}, queues {}, max_walltime {} and module_renaming {}.'.format(self, commands, queues, max_walltime, module_renaming))
        self.queues = queues
        self.max_walltime = max_walltime
        self.module_renaming = module_renaming
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

        submit_output = subprocess.check_output((self.submit_command, job_file)).decode("utf-8")
        submit_output = submit_output.strip()
        job_id = self._get_job_id_from_submit_output(submit_output)

        logger.debug('Started job has ID {}.'.format(job_id))

        return job_id


    def is_mpi_used(self, modules):
        return 'intelmpi' in modules


    def add_mpi_to_command(self, command, cpus, use_mpi=True):
        if use_mpi:
            command = self.mpi_command.format(command=command, cpus=cpus)
        return command
    
    ## best node setups
    
    def speed(self, nodes_kind, nodes, cpus):
        return self.node_infos[nodes_kind]['speed'] * nodes * cpus
    
    
    @staticmethod
    def _best_cpu_configurations_for_state(nodes_state, memory_required, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_max=float('inf')):
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
        free_cpus, free_memory = nodes_state
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


    
    def best_cpu_configurations(self, memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_max=float('inf'), walltime=1):

        logger.debug('Calculating best CPU configurations for {}GB memory with node kinds {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {} and total_cpus_max {}'.format(memory_required, node_kind, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_max))
    
        ## chose node kinds if not passed
        if node_kind is None:
            node_kind = []
            for node_kind_i, node_infos_i in self.node_infos.items():
                try:
                    nodes_leave_free_i = node_infos_i['leave_free']
                except KeyError:
                    nodes_leave_free_i = 0
                try:
                    max_walltime_i = node_infos_i['max_walltime']
                except KeyError:
                    max_walltime_i = float('inf')
                if node_infos_i['nodes'] > nodes_leave_free_i and max_walltime_i >= walltime:
                    node_kind.append(node_kind_i)
            # node_kind = tuple(self.node_infos.keys())
        elif isinstance(node_kind, str):
            node_kind = (node_kind,)
        nodes_state = self._nodes_state(node_kind)
    
        ## init
        best_kind = node_kind[0]
        best_nodes = 0
        best_cpus = 0
        best_cpu_power = 0
    
        ## calculate best CPU configuration
        for node_kind_i in node_kind:
            nodes_state_i = nodes_state[node_kind_i]
            node_info_values_i = self.node_infos[node_kind_i]
            nodes_cpu_power_i = node_info_values_i['speed']
            nodes_max_i = node_info_values_i['nodes']
            try:
                nodes_leave_free_i = node_info_values_i['leave_free']
            except KeyError:
                nodes_leave_free_i = 0
            nodes_max_i = min(nodes_max, nodes_max_i)
            nodes_leave_free_i = max(nodes_leave_free, nodes_leave_free_i)
            
            (best_nodes_i, best_cpus_i) = self._best_cpu_configurations_for_state(nodes_state_i, memory_required, nodes=nodes, cpus=cpus, nodes_max=nodes_max_i, nodes_leave_free=nodes_leave_free_i, total_cpus_max=total_cpus_max)
    
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
        logger.debug('Waiting for at least {} CPUs with {}GB memory, with node_kind {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {}, total_cpus_min{} and total_cpus_max {}.'.format(total_cpus_min, memory_required, node_kind, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_min, total_cpus_max))
    
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
    def _nodes_state(self, kinds):
        raise NotImplementedError()


class Job():

    def __init__(self, batch_system, output_dir, force_load=False, max_job_name_len=80):
        ## batch system
        self.batch_system = batch_system
        self.max_job_name_len = max_job_name_len

        ## check input
        if output_dir is None:
            raise ValueError('The output dir is not allowed to be None.')

        ## get option file
        try:
            option_file = os.path.join(output_dir, 'job_options.hdf5')
        except Exception as e:
            raise ValueError('The output dir {} is not allowed.'.format(output_dir)) from e

        ## load option file if existing or forced
        if force_load or os.path.exists(option_file):
            self.__options = util.options.Options(option_file, mode='r+', replace_environment_vars_at_get=True)
            logger.debug('Job {} loaded.'.format(option_file))

        ## make new job options file otherwise
        else:
            os.makedirs(output_dir, exist_ok=True)

            self.__options = util.options.Options(option_file, mode='w-', replace_environment_vars_at_get=True)

            self.__options['/job/output_dir'] = output_dir
            self.__options['/job/option_file'] = os.path.join(output_dir, 'job_options.txt')
            self.__options['/job/id_file'] = os.path.join(output_dir, 'job_id.txt')
            self.__options['/job/unfinished_file'] = os.path.join(output_dir, 'unfinished.txt')
            self.__options['/job/finished_file'] = os.path.join(output_dir, 'finished.txt')

            logger.debug('Job {} initialized.'.format(option_file))


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

    def option_value(self, name, not_exist_okay=True):
        if not_exist_okay:
            try:
                return self.options[name]
            except KeyError:
                return None
        else:
            return self.options[name]


    @property
    def id(self):
        try:
            return self.options['/job/id']
        except KeyError:
            raise KeyError('Job with option file ' + self.options.filename + ' is not started!')

    @property
    def output_dir(self):
        return self.option_value('/job/output_dir', False)


    @property
    def option_file(self):
        return self.option_value('/job/option_file', False)

    @property
    def unfinished_file(self):
        return self.option_value('/job/unfinished_file', False)

    @property
    def finished_file(self):
        return self.option_value('/job/finished_file', False)

    @property
    def id_file(self):
        return self.option_value('/job/id_file', True)

    @property
    def output_file(self):
        return self.option_value('/job/output_file', True)

    @property
    def exit_code(self):
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
        return self.option_value('/job/cpu_kind', True)

    @property
    def nodes(self):
        return self.option_value('/job/nodes', True)

    @property
    def cpus(self):
        return self.option_value('/job/cpus', True)

    @property
    def queue(self):
        return self.option_value('/job/queue', True)

    @property
    def walltime_hours(self):
        return self.option_value('/job/walltime_hours', True)




    ## init methods

    def init_job_file(self, job_name, nodes_setup, queue=None, cpu_kind=None, walltime_hours=None, write_output_file=True):
        ## check qeue and walltime
        queue = self.batch_system.check_queue(queue)
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

        if write_output_file:
            self.options['/job/output_file'] = os.path.join(self.output_dir, 'job_output.txt')



    @abc.abstractmethod
    def _make_job_file_header(self, use_mpi):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_job_file_modules(self, modules):
        raise NotImplementedError()


    def _make_job_file_command(self, run_command, add_timing=True):
        if add_timing:
            # run_command = 'time {}'.format(run_command)
            run_command = self.batch_system.time_command.format(command=run_command)
        content = []
        content.append('touch {}'.format(self.options['/job/unfinished_file']))
        content.append('echo "Job started."')
        content.append('')
        content.append(run_command)
        content.append('')
        content.append('EXIT_CODE=$?')
        content.append('echo "Job finished with exit code $EXIT_CODE."')
        content.append('rm {}'.format(self.options['/job/unfinished_file']))
        content.append('echo $EXIT_CODE > {}'.format(self.options['/job/finished_file']))
        # content.append('')
        # content.append('qstat -f $PBS_JOBID')
        content.append('exit')
        content.append('')
        return os.linesep.join(content)


    def write_job_file(self, run_command, modules=()):
        modules = self.batch_system.check_modules(modules)
        use_mpi = self.batch_system.is_mpi_used(modules)
        cpus = self.options['/job/nodes'] * self.options['/job/cpus']
        run_command = self.batch_system.add_mpi_to_command(run_command, cpus, use_mpi=use_mpi)
        with open(self.option_file, mode='w') as file:
            file.write(self._make_job_file_header(use_mpi=use_mpi))
            file.write(self._make_job_file_modules(modules))
            file.write(self._make_job_file_command(run_command))



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
        if os.path.exists(self.finished_file) and (self.output_file is None or os.path.exists(self.output_file)):
            if check_exit_code:
                exit_code = self.exit_code
                if exit_code != 0:
                    raise JobError(self.id, self.output_dir, exit_code)
            return True
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



class JobError(Exception):

    def __init__(self, job_id, exit_code, output_path):
        error_message = 'The command of job {} at {} exited with code {}.'.format(job_id, output_path, exit_code)
        super().__init__(error_message)




class NodeSetup:

    def __init__(self, memory=1, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_left_free=0, total_cpus_min=1, total_cpus_max=float('inf'), check_for_better=False, walltime=1):

        assert nodes is None or nodes >= 1
        assert cpus is None or cpus >= 1
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
        setup = {'memory': memory, 'node_kind': node_kind, 'nodes': nodes, 'cpus': cpus, 'total_cpus_min': total_cpus_min, 'nodes_max': nodes_max, 'nodes_left_free': nodes_left_free, 'total_cpus_max': total_cpus_max, 'check_for_better': check_for_better, 'walltime': walltime}
        self.setup = setup
        self.batch_system = util.batch.universal.system.BATCH_SYSTEM

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


    def configuration_is_complete(self):
        return self['memory'] is not None and self['node_kind'] is not None and isinstance(self['node_kind'], str) and self['nodes'] is not None and self['cpus'] is not None


    # def configuration_incomplete(self):
    #     if not self.configuration_is_complete():
    #         raise NodeSetupIncompleteError(self)

    def configuration_incomplete(self):
        if not self.configuration_is_complete():
            logger.debug('Node setup incomplete. Try to complete it.')
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
            self.configuration_incomplete()
            value = self.setup[key]

        assert value is not None
        return value


    def update_with_best_configuration(self, check_for_better=True):
        # self.check_for_better = False #TODO: fix for qcl command missing
        if check_for_better:
            self['check_for_better'] = False
            setup_triple = (self.node_kind, self.nodes, self.cpus)
            logger.debug('Try to find better node setup configuration than {}.'.format(setup_triple))
            speed = self.batch_system.speed(*setup_triple)
            best_setup_triple = self.batch_system.best_cpu_configurations(self.memory, nodes_max=self['nodes_max'], total_cpus_max=self['total_cpus_max'], walltime=self['walltime'])
            best_speed = self.batch_system.speed(*best_setup_triple)
            if best_speed > speed:
                logger.debug('Using better node setup configuration {}.'.format(best_setup_triple))
                self['node_kind'], self['nodes'], self['cpus'] = best_setup_triple


    @property
    def memory(self):
        return self.setup['memory']

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



class NodeSetupIncompleteError(Exception):

    def __init__(self, nodes_setup):
        error_message = 'The node setup is incomplete: node_kind={}, nodes={} and cpus={}.'.format(nodes_setup.node_kind, nodes_setup.nodes, nodes_setup.cpus)
        super().__init__(error_message)
