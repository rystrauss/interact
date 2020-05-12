"""This module contains Schedule objects which define values that follow some schedule over time.

This module is adapted from OpenAI Baselines.

Author: Ryan Strauss
"""
from abc import ABC, abstractmethod


class Schedule(ABC):
    """An abstract schedule object."""

    @abstractmethod
    def value(self, time):
        """Gets the value of this schedule at a given time.

        Args:
            time (int): The step in time that the schedule should be queried.

        Returns:
            The value of this schedule at the requested point in time.
        """
        pass


class LinearSchedule(Schedule):
    """Schedule that follows a linear decay throughout time.

    After a specified number of timesteps, the final value is always returned.
    """

    def __init__(self, initial_value, final_value, num_steps):
        """Initializes the schedule.

        Args:
            initial_value (float): The initial value.
            final_value (float): The final value.
            num_steps (int): The length of the decay, in time steps.
                After this number of steps, the final value will always
                be returned.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps

    def value(self, time):
        decay = min(float(time) / self.num_steps, 1.)
        return self.initial_value + decay * (self.final_value - self.initial_value)
