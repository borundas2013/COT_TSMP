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
        print("Input shapes:", input_shape)
        edge_index_dim = input_shape[0][1]  # edge_index shape
        node_dim = input_shape[1][2]  # node features dimension
        edge_dim = input_shape[2][2]  # edge features dimension
        
        total_dim = node_dim + edge_dim + edge_index_dim+2
        self.kernel = self.add_weight(
            shape=(total_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="W",
            dtype=tf.float32,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="b",
                dtype=tf.float32
            )
        self.built = True

    def call(self, inputs, training=False):
        edge_index_0, node_features_0, edge_features_0, \
        edge_index_1, node_features_1, edge_features_1, \
        property_feature = inputs

       
        # Simply transpose edge_index to match dimensions
        edge_index_0 = tf.transpose(edge_index_0, [0, 2, 1])  # [batch, num_edges, 2]
        edge_index_1 = tf.transpose(edge_index_1, [0, 2, 1])
 

        # Combine features
        x = tf.concat([node_features_0, edge_features_0,edge_index_0], axis=-1)
        y = tf.concat([node_features_1, edge_features_1,edge_index_1], axis=-1)

        

        # Add property information
        property_expanded = tf.expand_dims(property_feature, axis=1)  # [batch, 1, property_dim]
        property_tiled = tf.tile(property_expanded, [1, x.shape[1], 1])  # [batch, num_nodes, property_dim]

        # Add property information through addition
        x = tf.concat([x, property_tiled], axis=-1)
        y = tf.concat([y, property_tiled], axis=-1)

        # Transform combined features
        x = tf.matmul(x, self.kernel)
        y = tf.matmul(y, self.kernel)

        if self.use_bias:
            x += self.bias
            y += self.bias

        if self.activation is not None:
            x = self.activation(x)
            y = self.activation(y)

        return x, y