import tensorflow as tf
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from datetime import datetime

@keras.saving.register_keras_serializable()
class GroupAwareLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_dense = None
        self.combine_dense = None
    
    def build(self, input_shape):
        features_shape, group_vec_shape = input_shape
        
        # Initialize layers in build
        self.group_dense = tf.keras.layers.Dense(
            128, 
            activation='relu',
            name='group_dense'
        )
        self.combine_dense = tf.keras.layers.Dense(
            256, 
            activation='relu',
            name='combine_dense'
        )
        
        super(GroupAwareLayer, self).build(input_shape)
    
    def call(self, inputs):
        features, group_vec = inputs
        
        # Process group vector
        group_dense = self.group_dense(group_vec)
        
        # Expand dimensions and tile
        group_dense_expanded = tf.expand_dims(group_dense, axis=1)
        group_dense_tiled = tf.tile(
            group_dense_expanded,
            [1, tf.shape(features)[1], 1]
        )
        
        # Combine features
        combined = tf.keras.layers.Concatenate(axis=-1)([features, group_dense_tiled])
        return self.combine_dense(combined)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)