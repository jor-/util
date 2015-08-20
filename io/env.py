import os

def load(variable_name):
    try:
        return os.environ[variable_name]
    except KeyError:
        raise EnvironmentLookupError(variable_name)


class EnvironmentLookupError(LookupError):

    def __init__(self, variable_name):
        message = 'Environment variable {} is not set!'.format(variable_name)
        super().__init__(message)
