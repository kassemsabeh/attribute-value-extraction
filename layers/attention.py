from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.keras import layers

class AttentionLayer(layers.Layer):
    
    def __init__(self, units=32, return_attention=False):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.return_attention = return_attention
    
    def build(self, input_shape):
        feature_dim = int(input_shape[2])
        
        self.Wt = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_normal',
        )
        self.Wx = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_normal'
        )
        self.bg = self.add_weight(
            shape=(self.units,),
            initializer='zeros'
        )
        self.Wa = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_normal'
        )
        self.ba = self.add_weight(
            shape=(1,),
            initializer='zeros'
        )

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        q = tf.expand_dims(tf.matmul(inputs, self.Wt), 2)
        k = tf.expand_dims(tf.matmul(inputs, self.Wx), 1)
        h = tf.tanh(q + k + self.bg)
        g = tf.reshape(tf.matmul(h, self.Wa), (batch_size, input_len, input_len))

        e = tf.exp(tf.math.reduce_max(g, axis=-1, keepdims=True))
        a = e / tf.math.reduce_sum(e, axis=-1, keepdims=True)

        v = tf.multiply(a, inputs)

        if self.return_attention:
            return [v, a]
        return v