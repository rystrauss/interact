from typing import List

import tensorflow as tf

from interact.typing import TensorType


def explained_variance(targets: TensorType, preds: TensorType) -> tf.Tensor:
    """Computes the explained variance between predictions and targets.

    Values closer to 1.0 mean that the targets and predictions are highly correlated.

    Args:
        targets: The target values.
        preds: The predicted values.

    Returns:
        The scalar percentage of variance in targets that is explained by preds.
    """
    _, y_var = tf.nn.moments(targets, axes=[0])
    _, diff_var = tf.nn.moments(targets - preds, axes=[0])
    return tf.maximum(-1.0, 1 - (diff_var / y_var))


def polyak_update(
    online_variables: List[tf.Tensor], target_variables: List[tf.Tensor], tau: float
):
    """Performs a Polyak averaging update.

    Generally used for doing soft updates of a target q-function.

    Args:
        online_variables: The variables of the online network.
        target_variables: The variables of the target network.
        tau: The Polyak averaging parameter.

    Returns:
        None.
    """
    for online_var, target_var in zip(online_variables, target_variables):
        target_var.assign(tau * online_var + (1 - tau) * target_var)
