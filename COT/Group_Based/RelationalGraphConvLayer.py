from tensorflow import keras
import tensorflow as tf


class RelationalGraphConvLayer(keras.layers.Layer):
    def __init__(self, units=128, activation='relu', use_bias=False, kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]
        atom_dim = 172 + 1
        self.kernel = self.add_weight(shape=(atom_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name="W",
                                      dtype=tf.float32,
                                      )

        if self.use_bias:
            self.bias = self.add_weight(shape=(1, self.units),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        trainable=True,
                                        name="b",
                                        dtype=tf.float32)
            self.built = True

    def call(self, inputs, training=False):
        adjacency_0, features_0, adjacency_1, features_1, imine_feature, conditional_feature_1, conditional_feature_2 = inputs

        x1 = keras.layers.Concatenate(axis=-1)([adjacency_0, features_0])  # tf.matmul(adjacency_0,features_0)
        # conditional_features_transposed = tf.transpose(conditional_feature_1, (0, 2, 1))
        x = x1 = keras.layers.Concatenate(axis=-1)(
            [x1, conditional_feature_1])  # tf.add(x1,conditional_features_transposed)#
        # x=tf.matmul(x1,self.kernel)

        x2 = keras.layers.Concatenate(axis=-1)([adjacency_1, features_1])  # tf.matmul(adjacency_0,features_0)
        # conditional_features_transposed_2 = tf.transpose(conditional_feature_2, (0, 2, 1))
        y = x2 = keras.layers.Concatenate(axis=-1)(
            [x2, conditional_feature_2])  # tf.add(x2,conditional_features_transposed_2)#
        # y= tf.matmul(x2, self.kernel)

        z = keras.layers.Concatenate(axis=-1)([x, y, imine_feature])

        if self.use_bias:
            x += self.bias
            y += self.bias
        return x, x + y  # x+y,x*y#x,y#self.activation(z), self.activation(z)