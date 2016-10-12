import logging
import os
import sys

import util.parallel.is_running

LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')


def get_logger_logging():
    return logging.getLogger()

def get_logger_multiprocessing():
    import multiprocessing
    return multiprocessing.get_logger()

def get_logger_scoop():
    import scoop
    return scoop.logger


def get_logger():
    if util.parallel.is_running.scoop_module():
        logger = get_logger_scoop()
        logger.debug('Returning logger from scoop module.')
    elif util.parallel.is_running.multiprocessing_module():
        logger = get_logger_multiprocessing()
        logger.debug('Returning logger from multiprocessing module.')
    else:
        logger = get_logger_logging()
        logger.debug('Returning logger from logging module.')

    return logger


logger = get_logger()


def is_debug():
    return logger.level <= logging.DEBUG


class Logger():

    def __init__(self, disp_stdout=True, log_file=None, level='DEBUG'):
        disp_stdout = disp_stdout or (disp_stdout is None and log_file is None)
        self.enabled = disp_stdout or log_file is not None

        if self.enabled:
            self.log_file = log_file
            self.handlers = []

            ## set logger
            logger.setLevel(level)
            self.logger = logger

            ## create formatter
            formatter_normal = logging.Formatter(
                "%(levelname)-8s %(module)s:%(lineno)d -> %(message)s",
                datefmt=None,
                style='%')
            try:
                import colorlog
            except ImportError:
                colorlog = None
                logger.warn('The package colorlog is not installed. Using uncolored logging.')
            if colorlog is not None:
                formatter_stdout = colorlog.ColoredFormatter(
                    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(module)s:%(lineno)d -> %(message)s",
                    datefmt=None,
                    reset=True,
                    log_colors={'DEBUG':    'cyan',
                                'INFO':     'green',
                                'WARNING':  'yellow',
                                'ERROR':    'red',
                                'CRITICAL': 'red,bg_white'},
                    secondary_log_colors={},
                    style='%')
            else:
                formatter_stdout = formatter_normal

            ## add stdout and stderr handler
            if disp_stdout:
                if not self.has_stream_handler(sys.stdout) and not self.has_stream_handler(sys.stderr):
                    handler = logging.StreamHandler(sys.stdout)
                    handler.setFormatter(formatter_stdout)
                    self.add_handler(handler)

            ## add log file handler
            if log_file is not None and log_file != '':
                try:
                    os.remove(log_file)
                except (OSError, IOError):
                    pass
                handler = logging.FileHandler(log_file)
                handler.setFormatter(formatter_normal)
                self.add_handler(handler)

            ## add null handler if no other handler
            if not logger.hasHandlers():
                self.add_handler(logging.NullHandler())

            ## debug infos
            if disp_stdout:
                logger.debug('Logger {} configured with output to stdout.'.format(logger.name))
            if log_file is not None:
                logger.debug('Logger {} configured with output to file {}.'.format(logger.name, log_file))

    def __del__(self):
        self.close()

    def __enter__(self):
        if self.enabled:
            self.logger.info('Starting logging.')
        return self

    def __exit__(self, eType, eValue, eTrace):
        if self.enabled:
            if eValue is not None and eType is not SystemExit:
                self.logger.exception(eValue)
            self.logger.info('Logging stopped.')
        self.close()


    def close(self):
        if self.enabled:
            self.remove_all_handler()

    ## handler

    def add_handler(self, handler):
        self.handlers.append(handler)
        self.logger.addHandler(handler)

    def remove_handler(self, handler):
        handler.flush()
        handler.close()
        self.handlers.remove(handler)
        self.logger.removeHandler(handler)

    def remove_all_handler(self):
        for handler in self.handlers:
            self.remove_handler(handler)

    def has_stream_handler(self, stream):
        for handler in self.logger.handlers:
            try:
                if handler.stream == stream:
                    return True
            except AttributeError:
                pass
        return False

