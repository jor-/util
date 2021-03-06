import abc
import math
import os
import pathlib
import shutil
import subprocess
import tempfile
import time

import numpy as np

import util.constants
import util.batch.general.constants
import util.io.conda
import util.io.env
import util.io.fs
import util.logging
import util.options


# *** nodes *** #

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
        except KeyError:
            util.logging.warning('Node kind {} not found in nodes state {}.'.format(kind, self.nodes_state))
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

        # set batch system
        from util.batch.universal.system import BATCH_SYSTEM
        self.batch_system = BATCH_SYSTEM

        # check input
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
                raise ValueError('For node kind {} are maximal {} cpus per node available but {} are requested'.format(node_kind, max_cpus, cpus))

        if node_kind is not None and nodes is not None:
            max_nodes = self.batch_system.node_infos.nodes(node_kind)
            if nodes > max_nodes:
                raise ValueError('For node kind {} are maximal {} nodes available but {} are requested'.format(node_kind, max_nodes, nodes))

        # prepare input
        if node_kind is not None and not isinstance(node_kind, str) and len(node_kind) == 1:
            node_kind = node_kind[0]
        if nodes_max == 1 and nodes is None:
            nodes = 1
        if nodes is not None and total_cpus_max == nodes:
            cpus = 1

        # save setup
        setup = {
            'memory': memory,
            'node_kind': node_kind,
            'nodes': nodes,
            'cpus': cpus,
            'nodes_max': nodes_max,
            'nodes_leave_free': nodes_leave_free,
            'total_cpus_min': total_cpus_min,
            'total_cpus_max': total_cpus_max,
            'check_for_better': check_for_better,
            'walltime': walltime}
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
            util.logging.debug('Node setup incomplete. Try to complete it.')
            if self['memory'] is None:
                raise ValueError('Memory has to be set.')
            try:
                (node_kind, nodes, cpus) = self.batch_system.wait_for_needed_resources(self['memory'], node_kind=self['node_kind'], nodes=self['nodes'], cpus=self['cpus'], nodes_max=self['nodes_max'], nodes_leave_free=self['nodes_leave_free'], total_cpus_min=self['total_cpus_min'], total_cpus_max=self['total_cpus_max'])
            except NotImplementedError:
                util.logging.error('Batch system does not support completion of node setup.')
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
            util.logging.debug('Try to find better node setup configuration.')
            try:
                best_setup_triple = self.batch_system.best_cpu_configurations(self.memory, nodes_max=self['nodes_max'], total_cpus_max=self['total_cpus_max'], walltime=self.walltime)
            except CommandError as e:
                util.logging.exception(e.message)
                util.logging.warn('Could not update node setup with best configuration.')
            else:
                best_speed = self.batch_system.speed(*best_setup_triple)
                setup_triple = (self.node_kind, self.nodes, self.cpus)
                speed = self.batch_system.speed(*setup_triple)
                if best_speed > speed:
                    util.logging.debug('Using better node setup configuration {}.'.format(best_setup_triple))
                    self['node_kind'], self['nodes'], self['cpus'] = best_setup_triple
                elif not self.batch_system.is_free(self.memory, self.node_kind, self.nodes, self.cpus):
                    util.logging.debug('Node setup configuration {} is not free.'.format(setup_triple))
                    if best_speed >= speed * not_free_speed_factor:
                        util.logging.debug('Using node setup configuration {}.'.format(best_setup_triple))
                        self['node_kind'], self['nodes'], self['cpus'] = best_setup_triple
                    else:
                        util.logging.debug('Not using best node setup configuration {} since it is to slow.'.format(best_setup_triple))

    # *** properties *** #

    @property
    def memory(self):
        value = self.setup['memory']
        if value is None:
            raise AttributeError('"memory" is not set.')
        return value

    @memory.setter
    def memory(self, value):
        value = int(value)
        self.setup['memory'] = value

    @property
    def node_kind(self):
        self.update_with_best_configuration(self['check_for_better'])
        value = self.configuration_value('node_kind', test=lambda v: isinstance(v, str))
        assert value is not None
        return value

    @node_kind.setter
    def node_kind(self, value):
        value = str(value)
        self.setup['node_kind'] = value

    @property
    def nodes(self):
        self.update_with_best_configuration(self['check_for_better'])
        value = self.configuration_value('nodes')
        assert value is not None
        return value

    @nodes.setter
    def nodes(self, value):
        value = int(value)
        self.setup['nodes'] = value

    @property
    def cpus(self):
        self.update_with_best_configuration(self['check_for_better'])
        value = self.configuration_value('cpus')
        assert value is not None
        return value

    @cpus.setter
    def cpus(self, value):
        value = int(value)
        self.setup['cpus'] = value

    @property
    def walltime(self):
        return self.setup['walltime']

    @walltime.setter
    def walltime(self, value):
        value = int(value)
        self.setup['walltime'] = value

    @property
    def total_cpus_min(self):
        return self.setup['total_cpus_min']

    @total_cpus_min.setter
    def total_cpus_min(self, value):
        value = int(value)
        self.setup['total_cpus_min'] = value

    @property
    def total_cpus_max(self):
        return self.setup['total_cpus_max']

    @total_cpus_max.setter
    def total_cpus_max(self, value):
        value = int(value)
        self.setup['total_cpus_max'] = value

    @property
    def nodes_max(self):
        return self.setup['nodes_max']

    @nodes_max.setter
    def nodes_max(self, value):
        value = int(value)
        self.setup['nodes_max'] = value

    @property
    def nodes_leave_free(self):
        return self.setup['nodes_leave_free']

    @nodes_leave_free.setter
    def nodes_leave_free(self, value):
        value = int(value)
        self.setup['nodes_leave_free'] = value

    @property
    def check_for_better(self):
        return self.setup['check_for_better']

    @check_for_better.setter
    def check_for_better(self, value):
        value = bool(value)
        self.setup['check_for_better'] = value


class NodeSetupIncompleteError(Exception):

    def __init__(self, nodes_setup):
        error_message = 'The node setup is incomplete: node_kind={}, nodes={} and cpus={}.'.format(nodes_setup.node_kind, nodes_setup.nodes, nodes_setup.cpus)
        super().__init__(error_message)


# *** batch system *** #

class CommandError(Exception):

    def __init__(self, command, cause=None, return_code=None, output=None, error_output=None, message=None):

        # restore non-passed parameters from original exception
        if cause is not None:

            if return_code is None:
                try:
                    return_code = cause.returncode
                except AttributeError:
                    pass

            if output is None:
                try:
                    output = cause.stdout
                except AttributeError:
                    pass
                else:
                    if output is not None:
                        output = output.decode('utf8')

            if error_output is None:
                try:
                    error_output = cause.stderr
                except AttributeError:
                    pass
                else:
                    if error_output is not None:
                        error_output = error_output.decode('utf8')

        # create message
        if message is None:
            message = 'Command {} could not be executed successfully.'.format(command)
            if return_code is not None:
                message = message + ' The return code was: {}.'.format(return_code)
            if output is not None:
                message = message + ' The output was: "{}".'.format(output)
            if error_output is not None:
                message = message + ' The error output was: "{}".'.format(error_output)

        # store parameters
        self.command = command
        self.cause = cause
        self.return_code = return_code
        self.output = output
        self.error_output = error_output
        self.message = message

        super().__init__(message)


class CommandInvalidOutputError(CommandError):
    def __init__(self, command, output=None, message=None):

        # create message
        if message is None:
            message = 'Command {} has invalid output.'.format(command)
            if output is not None:
                message = message + ' The output was: "{}".'.format(output)

        super().__init__(command, output=output, message=message)


class BatchSystem():

    def __init__(self, commands, queues, pre_commands={}, max_walltime={}, node_infos={}):
        util.logging.debug('{} initiating with commands {}, queues {}, pre_commands {} and max_walltime {}.'.format(self, commands, queues, pre_commands, max_walltime))
        self.commands = commands
        self.pre_commands = pre_commands
        self.queues = queues
        self.max_walltime = max_walltime

        if not isinstance(node_infos, NodeInfos):
            node_infos = NodeInfos(node_infos)
        self.node_infos = node_infos

    @property
    def mpi_command(self):
        return self.command('mpirun')

    @property
    def time_command(self):
        return self.command('time')

    @property
    def submit_command(self):
        return self.command('sub')

    @property
    def status_command(self):
        return self.command('stat')

    @property
    def status_command_args(self):
        try:
            return self.command('stat_args')
        except KeyError:
            return []

    @property
    def nodes_command(self):
        return self.command('nodes')

    def command(self, name):
        return self.commands[name]

    def pre_command(self, name):
        try:
            return self.pre_commands[name]
        except KeyError:
            return ''

    def __str__(self):
        return 'General batch system'

    # check methods

    def check_queue(self, queue):
        if queue is not None and queue not in self.queues:
            raise ValueError('Unknown queue {}.'.format(queue))
        return queue

    def check_walltime(self, queue, walltime_hours):
        # get max walltime
        try:
            max_walltime_for_queue = self.max_walltime[queue]
        except KeyError:
            max_walltime_for_queue = float('inf')
        # check walltime
        if walltime_hours is not None:
            if walltime_hours <= max_walltime_for_queue:
                walltime_hours = math.ceil(walltime_hours)
            else:
                raise ValueError('Max walltime {} is greater than max walltime for queue {}.'.format(walltime_hours, max_walltime_for_queue))
        else:
            if max_walltime_for_queue < float('inf'):
                walltime_hours = max_walltime_for_queue
        # return
        assert (walltime_hours is None and max_walltime_for_queue == float('inf')) or walltime_hours <= max_walltime_for_queue
        return walltime_hours

    # other methods

    def start_job(self, job_file):
        util.logging.debug('Starting job with option file {}.'.format(job_file))

        if not os.path.exists(job_file):
            raise FileNotFoundError(job_file)

        command = (self.submit_command, job_file)
        try:
            output = subprocess.check_output(command, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, OSError) as e:
            raise util.batch.general.system.CommandError(command, cause=e) from e
        else:
            util.logging.debug('Job submit result is {}.'.format(output))
            output = output.decode('utf-8').strip()
            job_id = self._get_job_id_from_submit_output(output)
            util.logging.debug('Started job has ID {}.'.format(job_id))
            return job_id

    def job_state(self, job_id, return_output=True):
        # make command list
        command_list = [self.status_command] + self.status_command_args + [job_id]

        # run status command
        try:
            output = subprocess.check_output(command_list, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, OSError) as e:
            raise util.batch.general.system.CommandError(command_list, cause=e) from e
        else:
            util.logging.debug('Status command result: {}'.format(output))
            if return_output:
                output = output.decode("utf-8")
                return output

    # best node setups

    def speed(self, node_kind, nodes, cpus):
        return self.node_infos.speed(node_kind) * nodes * cpus

    def is_free(self, memory, node_kind, nodes, cpus):
        # get nodes with required memory
        nodes_state = self._nodes_state()
        free_cpus = nodes_state.free_cpus(node_kind, required_memory=memory)

        # calculate useable nodes
        free_nodes = free_cpus[free_cpus >= cpus].size
        free_nodes = free_nodes - self.node_infos.leave_free(node_kind)

        return free_nodes >= nodes

    @staticmethod
    def _best_cpu_configurations_for_state(nodes_state, node_kind, memory_required, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_max=float('inf')):
        util.logging.debug('Getting best cpu configuration for node state {} with memory {}, nodes {}, cpus {}, nodes max {} and nodes left free {}.'.format(nodes_state, memory_required, nodes, cpus, nodes_max, nodes_leave_free))

        # check input
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

        # get only nodes with required memory
        free_cpus = nodes_state.free_cpus(node_kind, required_memory=memory_required)

        # calculate best configuration
        best_nodes = 0
        best_cpus = 0

        if len(free_cpus) > 0:
            # chose numbers of cpus to check
            if cpus is not None:
                cpus_to_check = (cpus,)
            else:
                cpus_to_check = range(max(free_cpus), 0, -1)

            # get number of nodes for each number of cpus
            for cpus_to_check_i in cpus_to_check:
                # calculate useable nodes (respect max nodes and left free nodes)
                free_nodes = free_cpus[free_cpus >= cpus_to_check_i].size
                free_nodes = free_nodes - nodes_leave_free
                free_nodes = min(free_nodes, nodes_max)

                # respect fix number of nodes if passed
                if nodes is not None:
                    if free_nodes >= nodes:
                        free_nodes = nodes
                    else:
                        free_nodes = 0

                # respect total max cpus
                while free_nodes * cpus_to_check_i > total_cpus_max:
                    if free_nodes > 1:
                        free_nodes -= 1
                    else:
                        cpus_to_check_i = total_cpus_max

                # check if best configuration
                if free_nodes * cpus_to_check_i > best_nodes * best_cpus:
                    best_nodes = free_nodes
                    best_cpus = cpus_to_check_i

        util.logging.debug('Best CPU configuration is for this kind: {}'.format((best_nodes, best_cpus)))

        assert best_nodes <= nodes_max
        assert best_nodes * best_cpus <= total_cpus_max
        assert nodes is None or best_nodes == nodes or best_nodes == 0
        assert cpus is None or best_cpus == cpus or best_cpus == 0
        return (best_nodes, best_cpus)

    def best_cpu_configurations(self, memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_max=float('inf'), walltime=None):

        util.logging.debug('Calculating best CPU configurations for {}GB memory with node kinds {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {}, total_cpus_max {} and walltime {}'.format(memory_required, node_kind, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_max, walltime))

        # chose node kinds if not passed
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

        # init
        best_kind = node_kind[0]
        best_nodes = 0
        best_cpus = 0
        best_cpu_power = 0

        # calculate best CPU configuration
        for node_kind_i in node_kind:
            nodes_cpu_power_i = self.node_infos.speed(node_kind_i)
            nodes_max_i = self.node_infos.nodes(node_kind_i)
            nodes_max_i = min(nodes_max, nodes_max_i)
            nodes_leave_free_i = self.node_infos.leave_free(node_kind_i)
            nodes_leave_free_i = max(nodes_leave_free, nodes_leave_free_i)

            (best_nodes_i, best_cpus_i) = self._best_cpu_configurations_for_state(nodes_state, node_kind_i, memory_required, nodes=nodes, cpus=cpus, nodes_max=nodes_max_i, nodes_leave_free=nodes_leave_free_i, total_cpus_max=total_cpus_max)

            util.logging.debug('Best CPU configurations for {}GB memory with node kind {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {} and total_cpus_max {} is {}.'.format(memory_required, node_kind_i, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_max, (best_nodes_i, best_cpus_i)))

            if nodes_cpu_power_i * best_cpus_i * best_nodes_i > best_cpu_power * best_cpus * best_nodes:
                best_kind = node_kind_i
                best_nodes = best_nodes_i
                best_cpus = best_cpus_i
                best_cpu_power = nodes_cpu_power_i

        # return
        best_configuration = (best_kind, best_nodes, best_cpus)

        util.logging.debug('Best CPU configuration is: {}.'.format(best_configuration))

        assert best_kind in node_kind
        assert best_nodes <= nodes_max
        assert best_nodes * best_cpus <= total_cpus_max
        return best_configuration

    def wait_for_needed_resources(self, memory_required, node_kind=None, nodes=None, cpus=None, nodes_max=float('inf'), nodes_leave_free=0, total_cpus_min=1, total_cpus_max=float('inf')):
        util.logging.debug('Waiting for at least {} CPUs with {}GB memory, with node_kind {}, nodes {}, cpus {}, nodes_max {}, nodes_leave_free {}, total_cpus_min {} and total_cpus_max {}.'.format(total_cpus_min, memory_required, node_kind, nodes, cpus, nodes_max, nodes_leave_free, total_cpus_min, total_cpus_max))

        # check input
        if total_cpus_min > total_cpus_max:
            raise ValueError('total_cpus_max has to be greater or equal to total_cpus_min, but {} < {}.'.format(total_cpus_max, total_cpus_min))

        # calculate
        best_nodes = 0
        best_cpus = 0
        resources_free = False
        while not resources_free:
            (best_cpu_kind, best_nodes, best_cpus) = self.best_cpu_configurations(memory_required, node_kind=node_kind, nodes=nodes, cpus=cpus, nodes_max=nodes_max, nodes_leave_free=nodes_leave_free, total_cpus_max=total_cpus_max)
            cpus_avail = best_nodes * best_cpus
            resources_free = (cpus_avail >= total_cpus_min)
            if not resources_free:
                util.logging.debug('No enough resources free. {} CPUs available, but {} CPUs needed. Waiting ...'.format(cpus_avail, total_cpus_min))
                time.sleep(60)

        return (best_cpu_kind, best_nodes, best_cpus)

    # abstract methods

    @abc.abstractmethod
    def is_job_running(self, job_id):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_job_id_from_submit_output(self, submit_output):
        raise NotImplementedError()

    @abc.abstractmethod
    def _nodes_state(self):
        raise NotImplementedError()


BATCH_SYSTEM = BatchSystem({}, ())


# *** job *** #

class Job():

    ERROR_KEYWORDS = ('error', 'warning', 'fatal', 'permission denied')
    IGNORE_ERROR_KEYWORDS = ()

    def __init__(self, output_dir=None, batch_system=None, force_load=False, max_job_name_len=80, exceeded_walltime_error_message=None, remove_output_dir_on_close=False):
        # remove_output_dir_on_close
        self.remove_output_dir_on_close = remove_output_dir_on_close

        # if no output dir, use tmp output dir
        if output_dir is None:
            output_dir = util.constants.TMP_DIR
            os.makedirs(output_dir, exist_ok=True)
            output_dir = tempfile.mkdtemp(dir=output_dir, prefix='job_')

        # batch system
        if batch_system is None:
            batch_system = BATCH_SYSTEM
        self.batch_system = batch_system
        self.max_job_name_len = max_job_name_len
        self.exceeded_walltime_error_message = exceeded_walltime_error_message

        # check input
        if output_dir is None:
            raise ValueError('The output dir is not allowed to be None.')
        output_dir_expanded = os.path.expandvars(output_dir)
        option_file_expanded = os.path.join(output_dir_expanded, 'job_options.hdf5')

        # load option file if existing or forced
        if force_load or os.path.exists(option_file_expanded):
            try:
                self.__options = util.options.OptionsFile(option_file_expanded, mode='r+', replace_environment_vars_at_get=True)
            except OSError as e:
                raise JobOptionFileError(option_file_expanded) from e
            util.logging.debug('Job {} loaded.'.format(option_file_expanded))

        # make new job options file otherwise
        else:
            os.makedirs(output_dir_expanded, exist_ok=True)

            self.__options = util.options.OptionsFile(option_file_expanded, mode='w-', replace_environment_vars_at_get=True)

            self.options['/job/output_file'] = os.path.join(output_dir, 'job_output.txt')
            self.options['/job/option_file'] = os.path.join(output_dir, 'job_options.txt')
            self.options['/job/id_file'] = os.path.join(output_dir, 'job_id.txt')
            self.options['/job/unfinished_file'] = os.path.join(output_dir, 'unfinished.txt')
            self.options['/job/finished_file'] = os.path.join(output_dir, 'finished.txt')

            util.logging.debug('Job {} initialized.'.format(option_file_expanded))

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __str__(self):
        try:
            output_dir = self.output_dir
        except util.options.ClosedOptionsFileError:
            job_str = 'closed job'
        else:
            try:
                job_id = self.id
            except JobNotStartedError:
                job_id = '(not started)'
            job_str = 'job {} with output path {}'.format(job_id, output_dir)
        return job_str

    @property
    def options(self):
        return self.__options

    # option properties

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
        # check if finished file exists
        if not os.path.exists(self.finished_file):
            JobError(self, f'Finished file {self.finished_file} does not exist. The job is not finished.')
        # read exit code
        with open(self.finished_file, mode='r') as finished_file:
            exit_code = finished_file.read()
        # check exit code
        if len(exit_code) > 0:
            try:
                exit_code = int(exit_code)
                return exit_code
            except ValueError:
                raise JobError(self, f'Finished file {self.finished_file} does not contain an exit code but rather {exit_code}.')
        else:
            raise JobError(self, f'Finished file {self.finished_file} is empty.')

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

    # write job file methods

    def set_job_options(self, job_name, nodes_setup, queue=None, cpu_kind=None):
        # check qeue and walltime
        queue = self.batch_system.check_queue(queue)
        walltime_hours = nodes_setup.walltime
        walltime_hours = self.batch_system.check_walltime(queue, walltime_hours)

        # set job options
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
    def _job_file_header(self, use_mpi=True):
        raise NotImplementedError()

    def _job_file_command(self, command, pre_command=None, add_timing=True, use_mpi=True, use_conda=True):
        # add mpi
        if use_mpi:
            cpus = self.options['/job/nodes'] * self.options['/job/cpus']
            if cpus > 1:
                command = self.batch_system.mpi_command.format(command=command, cpus=cpus)
        # add timing
        if add_timing:
            command = self.batch_system.time_command.format(command=command)
        # add start
        content = []
        content.append('touch {}'.format(self.options['/job/unfinished_file']))
        content.append('echo "Job started."')
        content.append('')
        # add conda
        if use_conda:
            try:
                conda_activate_file = util.io.conda.conda_activate_file()
            except util.io.conda.CondaNotInstalledError:
                pass
            else:
                content.append(f'. {conda_activate_file}')
                try:
                    conda_env = util.io.conda.conda_env()
                except util.io.conda.CondaNotInstalledError:
                    conda_env = ''  # base env
                content.append(f'conda activate {conda_env}')
        # add commands
        if pre_command is not None:
            content.append(pre_command)
        content.append(command)
        # add exit
        content.append('')
        content.append('EXIT_CODE=$?')
        content.append('echo "Job finished with exit code $EXIT_CODE."')
        content.append('rm {}'.format(self.options['/job/unfinished_file']))
        content.append('echo $EXIT_CODE > {}'.format(self.options['/job/finished_file']))
        content.append('exit')
        content.append('')
        return os.linesep.join(content)

    def write_job_file(self, command, pre_command=None, use_mpi=True, use_conda=True, add_timing=True):
        job_file_command = (self._job_file_header(use_mpi=use_mpi) + os.linesep +
                            self._job_file_command(command, pre_command=pre_command,
                                                   use_mpi=use_mpi, use_conda=use_conda,
                                                   add_timing=add_timing))
        with open(self.option_file, mode='w') as f:
            f.write(job_file_command)
            f.flush()
            os.fsync(f.fileno())

    # other methods

    def start(self):
        job_id = self.batch_system.start_job(self.options['/job/option_file'])
        self.options['/job/id'] = job_id

        id_file = self.id_file
        if id_file is not None:
            with open(self.options['/job/id_file'], 'w') as id_file:
                id_file.write(job_id)

    def is_submitted(self):
        try:
            self.options['/job/id']
        except KeyError:
            return False
        else:
            return True

    def is_running(self):
        return self.batch_system.is_job_running(self.id)

    def is_finished(self, check_exit_code=True):
        finished_file = pathlib.Path(self.finished_file)
        output_file = pathlib.Path(self.output_file)

        # job finished
        if finished_file.exists():
            # check exit code
            if check_exit_code:
                exit_code = self.exit_code
                if exit_code != 0:
                    raise JobExitCodeError(self)
            # check if output file exists
            if output_file.exists():
                return True
            else:
                finished_file_time_since_creation_in_seconds = time.time() - finished_file.stat().st_mtime
                if finished_file_time_since_creation_in_seconds > util.batch.general.constants.MAX_WAIT_FOR_OUTPUT_FILE_SECONDS:
                    raise JobError(self, f'Output file {output_file} is missing!')
                else:
                    return False

        # job cancelled
        elif output_file.exists() and not self.is_running():
            raise JobCancelledError(self)

        # job queued or running
        else:
            return False

    def wait_until_finished(self, check_exit_code=True, pause_seconds=None, pause_seconds_min=5, pause_seconds_max=60, pause_seconds_increment_cycle=50):
        adaptive = pause_seconds is None
        if adaptive:
            util.logging.debug('Waiting for job {} to finish with adaptive sleep period with min {} and max {} seconds and increment cycle {}.'.format(self.id, pause_seconds_min, pause_seconds_max, pause_seconds_increment_cycle))
            pause_seconds = pause_seconds_min
        else:
            util.logging.debug('Waiting for job {} to finish with {}s sleep period.'.format(self.id, pause_seconds))

        cycle = 0
        while not self.is_finished(check_exit_code=check_exit_code):
            time.sleep(pause_seconds)

            if adaptive:
                cycle += 1
                if cycle == pause_seconds_increment_cycle:
                    pause_seconds += 1
                    cycle = 0

        util.logging.debug('Job {} finished with exit code {}.'.format(self.id, self.exit_code))

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

    def remove(self):
        output_dir = self.output_dir
        self.options.close()
        shutil.rmtree(output_dir)

    def open(self):
        if not self.is_opened:
            assert not self.options.is_opened
            self.options.open()
        return self

    def close(self):
        if not self.is_closed:
            assert not self.options.is_closed
            # remove if output dir if desired
            if self.remove_output_dir_on_close:
                try:
                    is_finished = self.is_finished(check_exit_code=True)
                except JobError as e:
                    util.logging.warning('{} stopped with an error {} and thus is not removed.'.format(self, e))
                else:
                    if is_finished:
                        try:
                            self.remove()
                        except OSError as e:
                            util.logging.warning('{} could not be removed: {}'.format(self, e))
            # close options file
            self.options.close()
        assert self.is_closed

    @property
    def is_opened(self):
        try:
            options = self.options
        except AttributeError:
            return False
        else:
            return options.is_opened

    @property
    def is_closed(self):
        try:
            options = self.options
        except AttributeError:
            return True
        else:
            return options.is_closed

    # check integrity

    def check_output_file(self, error_keywords=None, ignore_error_keywords=None):
        # prepare keyword lists
        if error_keywords is None:
            error_keywords = self.ERROR_KEYWORDS
        error_keywords = tuple(error_keyword.lower() for error_keyword in error_keywords)
        if ignore_error_keywords is None:
            ignore_error_keywords = self.IGNORE_ERROR_KEYWORDS
        ignore_error_keywords = tuple(ignore_error_keyword.lower() for ignore_error_keyword in ignore_error_keywords)

        # check each line in output
        for line in self.output.splitlines():
            line = line.lower()
            for error_keyword in error_keywords:
                if error_keyword in line:
                    ignore = False
                    for ignore_error_keyword in ignore_error_keywords:
                        if ignore_error_keyword in line:
                            ignore = True
                    if not ignore:
                        raise util.batch.universal.system.JobError(self, f'There was a line in the job output machting keyword {error_keyword}: {line}', include_output=True)

    def check_integrity(self, force_to_be_submitted=False, force_to_be_readonly=False):
        # check if options entires exist
        self.option_file
        self.output_file
        self.unfinished_file
        self.finished_file
        self.id_file

        # check for missing output file
        output_file = pathlib.Path(self.output_file)
        finished_file = pathlib.Path(self.finished_file)
        if finished_file.exists() and not output_file.exists():
            finished_file_time_since_creation_in_seconds = time.time() - finished_file.stat().st_mtime
            if finished_file_time_since_creation_in_seconds >= util.batch.general.constants.MAX_WAIT_FOR_OUTPUT_FILE_SECONDS:
                raise JobError(self, 'Output file is missing!')

        # check submitted state
        is_submitted = self.is_submitted()
        if force_to_be_submitted and not is_submitted:
            raise JobError(self, 'Job should be started!')
        if is_submitted:
            job_id = self.id

        # check finished state
        is_finished = self.is_finished(check_exit_code=True)
        if is_finished and not is_submitted:
            raise JobError(self, 'Job is finished but was not started!')

        # check errors in output file
        if is_submitted and os.path.exists(self.output_file):
            self.check_output_file()

        # check read only
        if force_to_be_readonly and not self.options.is_read_only():
            raise JobError(self, 'Job option file is writeable!')

        # check files
        output_dir = self.output_dir

        def check_if_file_exists(file, should_exists=True):
            if not file.startswith(output_dir):
                raise JobError(self, 'Job option should start with {} but its is {}.'.format(output_dir, file))
            exists = os.path.exists(file)
            if should_exists and not exists:
                raise JobError(self, 'File {} does not exist.'.format(file))
            if not should_exists and exists:
                raise JobError(self, 'File {} should not exist.'.format(file))

        if is_submitted:
            check_if_file_exists(self.option_file)
            check_if_file_exists(self.id_file)
        if is_finished:
            check_if_file_exists(self.output_file)
            check_if_file_exists(self.finished_file)
            check_if_file_exists(self.unfinished_file, should_exists=False)


class JobError(Exception):
    def __init__(self, job=None, error_message=None, include_output=False):
        if job is not None:
            # store job
            self.job = job

            # construct error message
            output_dir = job.output_dir
            if job.is_submitted():
                job_id = job.id
                error_message = 'An error accured in job {} stored at {}: {}'.format(job_id, output_dir, error_message)
            else:
                error_message = 'An error accured in job stored at {}: {}'.format(output_dir, error_message)

            # add output
            if include_output:
                try:
                    output = job.output
                except OSError:
                    pass
                else:
                    error_message = error_message + '\nThe job output was:\n{}'.format(output)

        # call super init
        if error_message is not None:
            super().__init__(error_message)
        else:
            super().__init__()


class JobNotStartedError(JobError):
    def __init__(self, job):
        error_message = 'The job is not started!'
        super().__init__(error_message=error_message, job=job)


class JobExitCodeError(JobError):
    def __init__(self, job):
        self.exit_code = job.exit_code
        error_message = f'The command of the job exited with code {self.exit_code}.'
        super().__init__(error_message=error_message, job=job, include_output=True)


class JobCancelledError(JobError):
    def __init__(self, job):
        error_message = f'The job was cancelled.'
        super().__init__(error_message=error_message, job=job)


class JobExceededWalltimeError(JobCancelledError):
    def __init__(self, job):
        self.walltime = job.walltime_hours
        error_message = f'The job exceeded walltime {self.walltime}.'
        JobError.__init__(error_message=error_message, job=job)


class JobMissingOptionError(JobError):
    def __init__(self, job, option):
        error_message = f'Job option {option} is missing!'
        super().__init__(error_message=error_message, job=job)


class JobOptionFileError(JobError):
    def __init__(self, option_file):
        error_message = f'Job option file {option_file} can not be red!'
        super().__init__(error_message=error_message)
