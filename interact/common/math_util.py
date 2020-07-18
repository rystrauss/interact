"""Utilities for mathematical operations.

Author: Ryan Strauss
"""

import numpy as np


def safe_mean(arr):
    """A safe mean operation that returns NaN if the array being operated on is empty.

    Args:
        arr: The array to be operated on.

    Returns:
        The mean of the array, or NaN if the array is empty.
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def discount_with_dones(rewards, dones, gamma):
    """Given a list of rewards, calculates the corresponding list of returns.

    Args:
        rewards: A list of rewards, where each element is the reward received at that time step.
        dones: A list of booleans that indicate whether or not an episode finished at that time step.
        gamma: The discount factor.

    Returns:
        A list of the corresponding returns.
    """
    discounted = []
    ret = 0

    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)
        discounted.append(ret)

    return discounted[::-1]


def explained_variance(y_pred, y):
    """Computes fraction of variance that predictions explain about target.

    Args:
        y_pred: The predicted values.
        y: THe target values.

    Returns:
        The fraction of variance that `y_pred` explain about `y`.
    """
    assert y.ndim == 1 and y_pred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - y_pred) / vary
