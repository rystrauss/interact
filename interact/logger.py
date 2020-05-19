"""Utilities for logging information during agent training.

Author: Ryan Strauss
"""

import os
from enum import Enum

import tensorflow as tf


class Colors(Enum):
    """Colors that can be used to colorize text in the console.

    These colors should be passed to `Logger.log`.
    """
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'


class Logger:
    """A utility for logging messages and data.

    Strings that are logged with this class are simultaneously sent to the console and saved to a log file.
    This class is also used to save summaries to TensorBoard.

    Args:
        dir: The directory to which logs will be saved.
    """

    def __init__(self, dir):
        self._dir = os.path.expanduser(dir)

        if os.path.exists(self._dir):
            raise ValueError('Logger directory already exists.')
        os.makedirs(self._dir, exist_ok=True)

        self._file = open(os.path.join(self._dir, 'log.txt'), 'w')
        self._summary_writer = tf.summary.create_file_writer(os.path.join(self._dir, 'tb'))

    @property
    def directory(self):
        return self._dir

    @property
    def file(self):
        return self._file

    @property
    def writer(self):
        return self._summary_writer

    def log_scalars(self, step, **kwargs):
        with self._summary_writer.as_default():
            for k, v in kwargs.items():
                tf.summary.scalar(k, v, step)

    def log(self, message, color=None):
        self._file.write(message)
        self._file.write('\n')

        if color is None:
            print(message)
        else:
            if not isinstance(color, Colors):
                raise ValueError(f'{color} is not a valid color.')

            print(f'{color.value}{message}\033[0m')

    def info(self, message):
        self.log(message, color=Colors.YELLOW)

    def warn(self, message):
        self.log(message, color=Colors.RED)

    def close(self):
        self._file.close()
        self._summary_writer.close()


def printc(color, *args, **kwargs):
    assert isinstance(color, Colors)
    args = [*args]
    args[0] = color.value + args[0]
    args[-1] = args[-1] + '\033[0m'
    print(*args, **kwargs)
