import os
import time

import util.logging
logger = util.logging.logger

import util.rzcluster.interact
import util.io.fs
import util.options


class Job():
    
    def __init__(self, output_dir, force_load=False):
        
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
            opt = util.options.Options(option_file, mode='r+')
            
            logger.debug('Job {} loaded.'.format(option_file))
        
        ## make new job options file otherwise
        else:
            os.makedirs(output_dir, exist_ok=True)
            
            opt = util.options.Options(option_file, mode='w-')
            
            opt['/job/output_dir'] = output_dir
            opt['/job/option_file'] = os.path.join(output_dir, 'job_options.txt')
            opt['/job/id_file'] = os.path.join(output_dir, 'job_id.txt')
            opt['/job/unfinished_file'] = os.path.join(output_dir, 'unfinished.txt')
            opt['/job/finished_file'] = os.path.join(output_dir, 'finished.txt')
            
            logger.debug('Job {} initialized.'.format(option_file))
        
        self.__options = opt
    
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
    def finished_file(self):
        return self.options['/job/finished_file']
    
    @property
    def unfinished_file(self):
        return self.options['/job/unfinished_file']
    
    
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
    
    def init_job_file(self, job_name, nodes_setup, queue=None, walltime_hours=None, write_output_file=True):
        from .constants import MAX_WALLTIME
        
        ## set queue if missing
        from util.rzcluster.constants import QUEUES
        
        cpu_kind = nodes_setup.node_kind
        
        if cpu_kind in ('f_ocean', 'f_ocean2', 'foexpress'):
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
                    
            elif queue not in QUEUES:
                raise ValueError('Unknown queue {}.'.format(queue))
        
        
        ## set cpu kind
        if cpu_kind in ('f_ocean2', 'foexpress'):
            cpu_kind = None
        if cpu_kind == 'shanghai':
            cpu_kind = 'all'
        
        ## max walltime
        try:
            max_walltime_for_queue = MAX_WALLTIME[queue]
        except KeyError:
            max_walltime_for_queue = float('inf')
        ## set walltime
        if walltime_hours is None:
            if max_walltime_for_queue < float('inf'):
                walltime_hours = max_walltime_for_queue
        ## check walltime
        else:
            if max_walltime_for_queue < walltime_hours:
                raise ValueError('Max walltime {} is greater than max walltime for queue {}.'.format(walltime_hours, max_walltime_for_queue))
        
        
        ## set job options
        opt = self.options

        opt['/job/memory_gb'] = nodes_setup.memory
        opt['/job/nodes'] = nodes_setup.nodes
        opt['/job/cpus'] = nodes_setup.cpus
        opt['/job/queue'] = queue
        opt['/job/name'] = job_name[:15]
        if cpu_kind is not None:
            opt['/job/cpu_kind'] = cpu_kind
        if walltime_hours is not None:
            opt['/job/walltime_hours'] = walltime_hours
        
        if write_output_file:
            opt['/job/output_file'] = os.path.join(self.output_dir, 'job_output.txt')
    
    
    
    def init_job_file_best(self, job_name, memory_gb, cpus_min=1, nodes_max=float('inf'), nodes_left_free=0, walltime_hours=None):
        nodes_setup = util.rzcluster.interact.NodeSetup(memory=memory_gb, node_kind=None, nodes=None, cpus=None, total_cpus_min=cpus_min, nodes_max=nodes_max, nodes_left_free=nodes_left_free)
        self.init_job_file(job_name, nodes_setup, walltime_hours=walltime_hours)
    
    
    
    def write_job_file(self, run_command, modules=()):
        opt = self.options
        
        f = open(opt['/job/option_file'], mode='w')
        
        f.write('#!/bin/bash \n\n')
        
        f.write('#PBS -N %s \n' % opt['/job/name'])
        
        ## output file
        output_file = self.output_file
        if output_file is not None:
            f.write('#PBS -j oe \n')
            f.write('#PBS -o %s \n' % output_file)
        
        ## walltime
        walltime_hours = self.walltime_hours
        if walltime_hours is not None:
            f.write('#PBS -l walltime=%02i:00:00 \n' % walltime_hours)
        
        ## select
        cpu_kind = self.cpu_kind
        if cpu_kind is not None:
            f.write('#PBS -l select=%i:%s=true:ncpus=%i:mem=%igb \n' % (opt['/job/nodes'], cpu_kind, opt['/job/cpus'], opt['/job/memory_gb']))
        else:
            f.write('#PBS -l select=%i:ncpus=%i:mem=%igb \n' % (opt['/job/nodes'], opt['/job/cpus'], opt['/job/memory_gb']))
        
#         ## disable hyper threading
#         f.write('#PBS -l place=scatter \n')
        
        ## queue
        f.write('#PBS -q %s \n\n' % opt['/job/queue'])
        
        ## load models
        if len(modules) > 0:
            f.write('. /usr/share/Modules/init/bash \n')
            if 'petsc' in modules:
                f.write('. /cluster/Software/intel14/composer_xe_2013_sp1/bin/compilervars.sh intel64 \n')
                f.write('. /cluster/Software/intel14/impi/4.1.1.036/intel64/bin/mpivars.sh \n')
            
            ## load modules
            for module in modules:
                if module in ('python', 'python3'):
                    module = 'python3.3'
                elif module == 'hdf5':
                    module = 'hdf5_1.8.11'
                elif module in ('matlab', 'matlab2014a'):
                    module = 'matlab2014a'
                elif module == 'matlab2011b':
                    module = 'matlab2011b'
                elif module == 'petsc':
                    module = 'petsc-intel14'
                f.write('module load ' + module + '\n')
            f.write('module list \n\n')
        
        
        ## run command
        f.write('touch {} \n'.format(opt['/job/unfinished_file']))
        f.write('echo "Job started."\n\n')
        
        f.write(run_command)
        f.write('\n\n')
        
        f.write('EXIT_CODE=$?\n')
        f.write('echo "Job finished."\n')
        f.write('rm {} \n'.format(opt['/job/unfinished_file']))
        f.write('echo $EXIT_CODE > {} \n\n'.format(opt['/job/finished_file']))
        
        f.write('qstat -f $PBS_JOBID \n')
        f.write('exit \n')
        
        util.io.fs.flush_and_close(f)
    
    
    
    ## other methods    
    
    def start(self):
        opt = self.options
        
        job_id = util.rzcluster.interact.start_job(opt['/job/option_file'])
        logger.debug('Job startet with id {}.'.format(job_id))
        
        opt['/job/id'] = job_id
        
        id_file = self.id_file
        
        if id_file is not None:
            with open(opt['/job/id_file'], "w") as id_file:
                id_file.write(job_id)
    

    def is_started(self):
        opt = self.options
        
        try:
            opt['/job/id']
            return True
        except KeyError:
            return False
    
    def is_finished(self):
        if os.path.exists(self.finished_file) and (self.output_file is None or os.path.exists(self.output_file)):
            ## read exit code
            with open(self.finished_file, mode='r') as finished_file:
                exit_code = finished_file.read()
            ## check exit code
            if len(exit_code) > 0:
                try:
                    exit_code = int(exit_code)
                except ValueError:
                    raise ValueError('Finished file {} does not contain an exit code but rather {}.'.format(self.finished_file, exit_code))
                if exit_code != 0:
                    raise JobError(self.id, exit_code)
            else:
                logger.warning('There is not exit code in finished file {}.'.format(self.finished_file))
            
            return True
        return False
    
    def is_running(self):
        return self.is_started() and not self.is_finished()


    def wait_until_finished(self, pause_seconds=None, pause_seconds_min=5, pause_seconds_max=60, pause_seconds_increment_cycle=50):
        adaptive = pause_seconds is None
        if adaptive:
            logger.debug('Waiting for job {} to finish with adaptive sleep period with min {} and max {} seconds and increment cycle {}.'.format(self.id, pause_seconds_min, pause_seconds_max, pause_seconds_increment_cycle))
            pause_seconds = pause_seconds_min
        else:
            logger.debug('Waiting for job {} to finish with {}s sleep period.'.format(self.id, pause_seconds))

        cycle = 0
        while not self.is_finished():
            time.sleep(pause_seconds)
            
            if adaptive:
                cycle += 1
                if cycle == pause_seconds_increment_cycle:
                    pause_seconds += 1
                    cycle = 0
        
        logger.debug('Job {} finished.'.format(self.id))
    
    
    
    
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
    
    def __init__(self, id, exit_code):
        error_message = 'The command of job {} exited with code {}.'.format(id, exit_code)
        super(error_message)
