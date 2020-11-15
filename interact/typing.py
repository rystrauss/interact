from typing import Any, Union, Tuple, List

# Represents a generic tensor type.
# This could be an np.ndarray, tf.Tensor, or a torch.Tensor.
TensorType = Any

# A shape of a tensor.
TensorShape = Union[Tuple[int], List[int]]
