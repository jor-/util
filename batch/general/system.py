import abc
import math
import os
import subprocess
import time

import util.io.fs
import util.options

import util.logging
logger = util.logging.logger


class Job():
    
    def __init__(self, batch_system, output_dir, force_load=False):
        ## batch system
        self.batch_system = batch_system
        
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
    
    def __exit__(self, type, value, traceback):
        self.close()
    
    
    def __str__(self):
        output_dir = self.output_dir
        try:
            id = self.id
        except KeyError:
            id = None
        if id is not None:
            job_str = 'job {} with output path {}'.format(id, output_dir)
        else:
            job_str = 'not started job with output path {}'.format(id, output_dir)
        return job_str
    
    
    
    @property
    def options(self):
        return self.__options
    
    
    ## option properties
    
    @property
    def id(self):
        try:
            return self.options['/job/id']
        except KeyError:
            raise KeyError('Job with option file ' + self.options.filename + ' is not started!')
    
    @property
    def output_dir(self):
        return self.options['/job/output_dir']
    
    @property
    def id_file(self):
        try:
            return self.options['/job/id_file']
        except KeyError:
            return None
    
    @property
    def option_file(self):
        return self.options['/job/option_file']
    
    @property
    def output_file(self):
        try:
            return self.options['/job/output_file']
        except KeyError:
            return None
    
    @property
    def unfinished_file(self):
        return self.options['/job/unfinished_file']
    
    @property
    def finished_file(self):
        return self.options['/job/finished_file']
    
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
        try:
            cpu_kind = self.options['/job/cpu_kind']
        except KeyError:
            cpu_kind = None
        
        return cpu_kind  
    
    @property
    def queue(self):
        try:
            queue = self.options['/job/queue']
        except KeyError:
            queue = None
        
        return queue   
    
    @property
    def walltime_hours(self):
        try:
            walltime_hours = self.options['/job/walltime_hours']
        except KeyError:
            walltime_hours = None
        
        return walltime_hours
    
    
    
    
    ## init methods

    def init_job_file(self, job_name, nodes_setup, queue=None, cpu_kind=None, walltime_hours=None, write_output_file=True):
        ## check qeue and walltime
        
        queue = self.batch_system.check_queue(queue)
        walltime_hours = self.batch_system.check_walltime(queue, walltime_hours)
        
        # from util.batch.universal.constants import MAX_WALLTIME, QUEUES
        # 
        # ## check queue
        # if queue is not None and queue not in QUEUES:
        #     raise ValueError('Unknown queue {}.'.format(queue))
        # 
        # ## check walltime
        # try:
        #     max_walltime_for_queue = MAX_WALLTIME[queue]
        # except KeyError:
        #     max_walltime_for_queue = float('inf')
        # if walltime_hours is not None:
        #     walltime_hours = math.ceil(walltime_hours)
        #     if max_walltime_for_queue < walltime_hours:
        #         raise ValueError('Max walltime {} is greater than max walltime for queue {}.'.format(walltime_hours, max_walltime_for_queue))
        # else:
        #     if max_walltime_for_queue < float('inf'):
        #         walltime_hours = max_walltime_for_queue
        
        
        ## set job options
        self.options['/job/memory_gb'] = nodes_setup.memory
        self.options['/job/nodes'] = nodes_setup.nodes
        self.options['/job/cpus'] = nodes_setup.cpus
        self.options['/job/queue'] = queue
        self.options['/job/name'] = job_name[:15]
        if cpu_kind is not None:
            self.options['/job/cpu_kind'] = cpu_kind
        if walltime_hours is not None:
            self.options['/job/walltime_hours'] = walltime_hours
        
        if write_output_file:
            self.options['/job/output_file'] = os.path.join(self.output_dir, 'job_output.txt')
    
    
    
    @abc.abstractmethod
    def _make_job_file_header(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def _make_job_file_modules(self, modules):
        raise NotImplementedError()

    def _make_job_file_command(self, run_command):
        content = []
        content.append('touch {}'.format(self.options['/job/unfinished_file']))
        content.append('echo "Job started."')
        content.append('')
        content.append(run_command)
        content.append('')
        content.append('EXIT_CODE=$?')
        content.append('echo "Job finished."')
        content.append('rm {}'.format(self.options['/job/unfinished_file']))
        content.append('echo $EXIT_CODE > {}'.format(self.options['/job/finished_file']))
        content.append('')
        content.append('qstat -f $PBS_JOBID')
        content.append('exit')
        content.append('')
        return os.linesep.join(content)
    
    
    # def _check_modules(self, modules):
    #     if len(modules) > 0:
    #         modules = list(modules)
    #         
    #         ## add required modules
    #         # for module_required, modules_requested in [('intelmpi', ('petsc',)), ('intel', ('intelmpi', 'petsc', 'hdf5'))]:
    #         for module_requested, modules_required in [('petsc', ('intelmpi', 'intel')), ('intelmpi', ('intel',)), ('hdf5', ('intel',))]:
    #             if module_requested in modules:
    #                 for module_required in modules_required:
    #                     if module_required not in modules:
    #                         logger.warning('Module "{}" needs module "{}" to be loaded first.'.format(module_requested, module_required))
    #                         modules = [module_required] + modules
    #         
    #         ## check positions
    #         first_index = 0
    #         for module in  ('intel', 'intelmpi'):
    #             if module in modules and modules[first_index] != module:
    #                 logger.warning('Module "{}" has to be at position {}.'.format(module, first_index))
    #                 modules.remove(module)
    #                 modules = [module] + modules
    #                 first_index += 1
    #     
    #         ## rename modules
    #         for i in range(len(modules)):
    #             module = modules[i]
    #             try:
    #                 module_new = self.module_rename_dict[module]
    #             except KeyError:
    #                 module_new = module
    #             modules[i] = module_new
    #     return modules
    
    
    def write_job_file(self, run_command, modules=()):
        modules = self.batch_system.check_modules(modules)
        with open(self.options['/job/option_file'], mode='w') as file:
            file.write(self._make_job_file_header())
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
    
    def __init__(self, id, exit_code, output_path):
        error_message = 'The command of job {} at {} exited with code {}.'.format(id, output_path, exit_code)
        super().__init__(error_message)


class BatchSystem():
    
    def __init__(self, submit_command, queues, max_walltime={}, module_renaming={}):
        self.submit_command = submit_command
        self.queues = queues
        self.max_walltime = max_walltime
        self.module_renaming = module_renaming
    
    
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
        assert walltime_hours <= max_walltime_for_queue
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
    

    def start_job(self, job_file):
        logger.debug('Starting job with option file {}.'.format(job_file))
        
        if not os.path.exists(job_file):
            raise FileNotFoundError(job_file)
    
        job_id = subprocess.check_output((self.submit_command, job_file)).decode("utf-8")
        job_id = job_id.rstrip()
        
        logger.debug('Started job has ID {}.'.format(job_id))
        
        return job_id
    




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
    
    
    def configuration_is_complete(self):
        return self['memory'] is not None and self['node_kind'] is not None and isinstance(self['node_kind'], str) and self['nodes'] is not None and self['cpus'] is not None
    
    
    def configuration_missing(self):
        if not self.configuration_is_complete():
            raise NodeSetupIncompleteError(self)
    
    
    def configuration_value(self, key, test=None):
        assert test is None or callable(test)
        
        value = self.setup[key]
        if value is None or (test is not None and not test(value)):
            self.configuration_missing()
            value = self.setup[key]
        
        assert value is not None
        return value
        
    
    @property
    def memory(self):
        return self.setup['memory']
    
    @property
    def node_kind(self):
        return self.configuration_value('node_kind', test=lambda v : isinstance(v, str))   
    
    @property
    def nodes(self):
        return self.configuration_value('nodes')  
    
    @property
    def cpus(self):
        return self.configuration_value('cpus')    
    


class NodeSetupIncompleteError(Exception):
    
    def __init__(self, node_setup):
        error_message = 'The node setup is incomplete: node_kind={}, nodes={} and cpus={}.'.format(node_setup.node_kind, node_setup.nodes, node_setup.cpus)
        super().__init__(error_message)
