import tensorflow as tf


class DuelingAggregator(tf.keras.layers.Layer):
    """Implements the aggregation module of the dueling network architecture.

    This layer accepts two inputs, the value stream and the advantage stream.
    This layer expects the input as a list that looks like
    [value_stream, advantage_stream].
    """

    def __init__(self, **kwargs):
        super(DuelingAggregator, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        value_stream, advantage_stream = inputs
        output_dim = advantage_stream.shape[1]
        value_stream = tf.tile(value_stream, [1, output_dim])
        # This line corresponds to Equation 9 from Wang et. al.
        output = value_stream + (
            advantage_stream - tf.reduce_mean(advantage_stream, axis=-1, keepdims=True)
        )
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[1])
