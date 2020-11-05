import tensorflow as tf


class LinearDecay(tf.keras.optimizers.schedules.PolynomialDecay):
    """A schedule that linearly decays the learning rate over the course of training.

    The learning rate at time `step_t` is given by:
    `initial_learning_rate * (1. - step_t / decay_steps)`

    Args:
        initial_learning_rate: The learning rate at the beginning of training.
        decay_steps: The total number of updates that will be performed.
        end_learning_rate: The learning rate after `decay_steps` steps.
    """

    def __init__(self, initial_learning_rate, decay_steps, end_learning_rate=0):
        super().__init__(initial_learning_rate, decay_steps, end_learning_rate=end_learning_rate)
