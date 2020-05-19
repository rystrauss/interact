import numpy as np


def safe_mean(arr):
    return np.nan if len(arr) == 0 else np.mean(arr)
