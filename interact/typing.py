from typing import Union, Tuple, List

from numpy import ndarray
from tensorflow import Tensor

# Represents a generic tensor type.
TensorType = Union[Tensor, ndarray]

# A shape of a tensor.
TensorShape = Union[Tuple[int], List[int]]
