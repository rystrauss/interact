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


class NormcInitializer(tf.keras.initializers.Initializer):

    def __init__(self, stddev=1.0):
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        out = tf.random.normal(shape, stddev=self.stddev, dtype=dtype or tf.float32)
        out *= tf.sqrt(tf.reduce_sum(out ** 2, axis=0, keepdims=True))
        return out
