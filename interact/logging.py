import os
from enum import Enum

import tensorflow as tf


class Colors(Enum):
    """Colors that can be used to colorize text in the console.

    These colors should be passed to `Logger.log`.
    """

    PINK = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"


class Logger:
    """A utility for logging messages and data.

    Args:
        dir: The directory to which logs will be saved.
    """

    def __init__(self, dir):
        self._dir = os.path.expanduser(dir)

        if os.path.exists(self._dir):
            raise ValueError("Logger directory already exists.")
        os.makedirs(self._dir, exist_ok=True)

        self._summary_writer = None

    @property
    def directory(self):
        """The directory associated with this logger."""
        return self._dir

    @property
    def writer(self):
        """The writer that Tensorboard summaries are saved to."""
        if self._summary_writer is None:
            self._summary_writer = tf.summary.create_file_writer(
                os.path.join(self._dir, "tb")
            )

        return self._summary_writer

    def log_scalars(self, step, prefix=None, **kwargs):
        """Logs scalar values to Tensorboard.

        Args:
            step: The step associated with this summary.
            kwargs: Key-value pairs to be logged.

        Returns:
            None.
        """
        with self.writer.as_default():
            for key, value in kwargs.items():
                if prefix is not None:
                    key = os.path.join(prefix, key)
                tf.summary.scalar(key, value, step)

    def log_histograms(self, step, prefix=None, **kwargs):
        """Logs histograms to Tensorboard.

        Args:
            step: The step associated with this summary.
            kwargs: Key-value pairs to be logged.

        Returns:
            None.
        """
        with self.writer.as_default():
            for key, value in kwargs.items():
                if prefix is not None:
                    key = os.path.join(prefix, key)
                tf.summary.histogram(key, value, step)

    def close(self):
        """Closes this logger object.

        Returns:
            None.
        """
        if self._summary_writer is not None:
            self._summary_writer.close()


def printc(color, *args, **kwargs):
    """Prints a colorized message to the console.

    Args:
        color: The color of the message. Should be a member of the `Colors` enum.
        *args: The arguments to be passed to the built-in print function.
        **kwargs: The keyword arguments to be passed to the built-in print function.

    Returns:
        None.
    """
    assert isinstance(color, Colors)
    args = [*args]
    args[0] = color.value + args[0]
    args[-1] = args[-1] + "\033[0m"
    print(*args, **kwargs)
