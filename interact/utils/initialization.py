import tensorflow as tf


class NormcInitializer(tf.keras.initializers.Initializer):
    def __init__(self, stddev=1.0):
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        out = tf.random.normal(shape, stddev=self.stddev, dtype=dtype or tf.float32)
        out *= tf.sqrt(tf.reduce_sum(out ** 2, axis=0, keepdims=True))
        return out
