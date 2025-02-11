import numpy as np
import pandas as pd
import ast
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Model.Constants import ATOM_DIM, NUM_ATOMS, LATENT_DIM
from Model.Features import prepare_graph_features, get_imine_feature_list, get_condtions
from tensorflow.keras.models import load_model


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

        # tf.matmul(adjacency_0,features_0)
        x1 = keras.layers.Concatenate(axis=-1)([adjacency_0, features_0])
        conditional_features_transposed = tf.transpose(
            conditional_feature_1, (0, 2, 1))
        x = x1 = keras.layers.Concatenate(axis=-1)(
            [x1, conditional_feature_1])  # tf.add(x1,conditional_features_transposed)#
        # x=tf.matmul(x1,self.kernel)

        # tf.matmul(adjacency_0,features_0)
        x2 = keras.layers.Concatenate(axis=-1)([adjacency_1, features_1])
        conditional_features_transposed_2 = tf.transpose(
            conditional_feature_2, (0, 2, 1))
        y = x2 = keras.layers.Concatenate(axis=-1)(
            [x2, conditional_feature_2])  # tf.add(x2,conditional_features_transposed_2)#
        # y= tf.matmul(x2, self.kernel)

        z = keras.layers.Concatenate(axis=-1)([x, y, imine_feature])

        if self.use_bias:
            x += self.bias
            y += self.bias
        # x+y,x*y#x,y#self.activation(z), self.activation(z)
        return x + y, x + y


def get_encoder(gconv_units, latent_dim, adjacency_shape, feature_shape, imine_shape, condition_shape,
                dense_units, dropout_rate):
    adjacency_0 = keras.layers.Input(shape=adjacency_shape)
    features_0 = keras.layers.Input(shape=feature_shape)
    adjacency_1 = keras.layers.Input(shape=adjacency_shape)
    features_1 = keras.layers.Input(shape=feature_shape)
    imine_feature = keras.layers.Input(shape=imine_shape)
    conditional_features_1 = keras.layers.Input(shape=condition_shape)
    conditional_features_2 = keras.layers.Input(shape=condition_shape)

    features_transformed_0 = features_0
    features_transformed_1 = features_1

    for units in gconv_units:
        features_transformed_0, features_transformed_1 = RelationalGraphConvLayer(units)(
            [adjacency_0, features_transformed_0, adjacency_1, features_transformed_1, imine_feature, conditional_features_1, conditional_features_2])


    x = layers.Dense(512, activation='relu')(features_transformed_0)
    y = layers.Dense(512, activation='relu')(features_transformed_1)
    x = keras.layers.LayerNormalization()(x)
    y = keras.layers.LayerNormalization()(y)

    x = x + keras.layers.Attention(use_scale=True)([x, x])
    x = keras.layers.Attention(use_scale=True)([x, x])
    y = y + keras.layers.Attention(use_scale=True)([y, y])
    y = keras.layers.Attention(use_scale=True)([y, y])
    x = keras.layers.GlobalAveragePooling1D()(x)
    y = keras.layers.GlobalAveragePooling1D()(y)

    # conditional_features_transposed_1 = tf.transpose(conditional_features_1, (0, 2,1))
    # conditional_features_transposed_2 = tf.transpose(conditional_features_2, (0, 2,1))

    z_mean = layers.Dense(latent_dim, dtype='float32', name='z_mean')(x)
    log_var = layers.Dense(latent_dim, dtype='float32', name='log_var')(x)
    z_mean_1 = layers.Dense(latent_dim, dtype='float32', name='z_mean_1')(y)
    log_var_1 = layers.Dense(latent_dim, dtype='float32', name='log_var_1')(y)
    encoder = keras.Model([adjacency_0, features_0, adjacency_1, features_1,
                           imine_feature, conditional_features_1, conditional_features_2],
                          [z_mean, log_var, z_mean_1, log_var_1,
                              conditional_features_1, conditional_features_2],
                          name='encoder')

    return encoder


def predict_property(generated_smiles):
    encoder = get_encoder(
        gconv_units=[9],
        adjacency_shape=(NUM_ATOMS, NUM_ATOMS),
        feature_shape=(NUM_ATOMS, ATOM_DIM),
        imine_shape=(NUM_ATOMS, 1),
        condition_shape=(NUM_ATOMS, NUM_ATOMS),
        latent_dim=LATENT_DIM,
        dense_units=[128, 256, 512],
        dropout_rate=0.2,
    )

    encoder.load_weights('pred/encoder_weights_lt.h5')

    adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor = prepare_graph_features(
        generated_smiles)

    condition_1_array, condition_2_array = get_condtions(generated_smiles)
    imine_features_array = get_imine_feature_list(generated_smiles)
    z_mean, log_var, z_mean_1, log_var_1, conditional_features_1, conditional_features_2 = encoder.predict(
        [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_features_array,
         condition_1_array, condition_2_array])
    er_model = load_model('pred/ER_MODEL_ly.h5')
    tg_model = load_model('pred/TG_MODEL_ly.h5')
    er_prediction = []
    tg_prediction = []
    for num in np.arange(0.1, 1.0, 0.1):
        random_number = round(num, 2)
        random_number_2 = round(1 - random_number, 2)

        molar_ratio_1 = random_number+np.zeros((1, NUM_ATOMS))
        molar_ratio_2 = random_number_2 + np.zeros((1, NUM_ATOMS))
        X_new = np.concatenate(
            (z_mean, z_mean_1, molar_ratio_1, molar_ratio_2), axis=-1)
        er_prediction.append(er_model.predict(X_new)[0][0])
        tg_prediction.append(tg_model.predict(X_new)[0][0])
    return max(er_prediction), max(tg_prediction)



# monomer1_smiles = 'c3cc(N(CC1CC1)CC2CO2)ccc3OCC4CO4'
# monomer2_smiles = 'NCCNCCN(CCNCCC(CN)CCN)CCN(CCNCCN)CCN(CCN)CCN'  # NCCN'
# generated_smiles = [[monomer1_smiles, monomer2_smiles]]
# print(predict_property(generated_smiles))
