import tensorflow as tf

from interact.typing import TensorType


def explained_variance(targets: TensorType, preds: TensorType) -> tf.Tensor:
    """Computes the percentage of the targets' variance that is explained by the predictions.

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
