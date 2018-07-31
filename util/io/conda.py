import pathlib

import util.io.env


class CondaNotInstalledError(Exception):
    def __init__(self):
        message = 'Conda is not installed.'
        self.message = message
        super().__init__(message)


CONDA_EXE_ENV_NAME = 'CONDA_EXE'


def conda_activate_file():
    try:
        conda_exe = util.io.env.load(CONDA_EXE_ENV_NAME)
    except util.io.env.EnvironmentLookupError:
        raise CondaNotInstalledError()
    else:
        conda_activate_file = pathlib.Path(conda_exe)
        assert conda_activate_file.name == 'conda'
        conda_activate_file = conda_activate_file.parent
        assert conda_activate_file.name == 'bin'
        conda_activate_file = conda_activate_file.parent
        conda_activate_file = conda_activate_file.joinpath('etc', 'profile.d', 'conda.sh')
        assert conda_activate_file.exists()
        return conda_activate_file


CONDA_ENV_NAME = 'CONDA_DEFAULT_ENV'


def conda_env():
    try:
        return util.io.env.load(CONDA_ENV_NAME)
    except util.io.env.EnvironmentLookupError:
        raise CondaNotInstalledError()
