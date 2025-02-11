

import ast

import pandas as pd
import numpy as np

import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers
import math
import os
import Constants
from Features import convert_smiles_to_graph, convert_graph_to_mol,get_condtions,get_imine_feature_list, prepare_graph_features,data_load,prepare_condition,convert_graph_to_mol_2
from rdkit import Chem
from Reward_Function import calculate_composite_reward
import random
tf.config.run_functions_eagerly(True)


_,df_smiles=data_load(Constants.DATA_FILE_PATH)
adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor = prepare_graph_features(df_smiles)
print(adjacency_0_tensor.shape,feature_0_tensor.shape)
print('-------------')
condition_1_array, condition_2_array = get_condtions(df_smiles)
imine_features_array = get_imine_feature_list(df_smiles)

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


def get_encoder(gconv_units, latent_dim, adjacency_shape, feature_shape, imine_shape, condition_shape,
                dense_units, dropout_rate):
    print(adjacency_shape,feature_shape)
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
            [adjacency_0, features_transformed_0, adjacency_1, features_transformed_1, imine_feature
                , conditional_features_1, conditional_features_2])

    # for units in dense_units:
    #   y=x+y
    #    x = layers.Dense(units, activation='relu')(x)
    #   x = layers.Dropout(dropout_rate)(x)
    #    y = layers.Dense(units, activation='relu')(y)
    #   y = layers.Dropout(dropout_rate)(y)
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
                          [z_mean, log_var, z_mean_1, log_var_1, conditional_features_1, conditional_features_2],
                          name='encoder')

    return encoder


def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape, condition_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))
    latent_inputs_1 = keras.Input(shape=(latent_dim,))
    conditional_features_1 = keras.Input(shape=condition_shape)
    conditional_features_2 = keras.Input(shape=condition_shape)

    x = latent_inputs
    y = latent_inputs_1
    cond1 = keras.layers.Flatten()(conditional_features_1)
    cond1 = keras.layers.Dense(160, activation='relu')(cond1)

    cond2 = keras.layers.Flatten()(conditional_features_2)
    cond2 = keras.layers.Dense(160, activation='relu')(cond2)

    x = keras.layers.Concatenate(axis=-1)([x, cond1])
    y = keras.layers.Concatenate(axis=-1)([y, cond2])

    x = layers.Dense(512, activation='relu')(x)
    y = layers.Dense(512, activation='relu')(y)

    x = keras.layers.LayerNormalization()(x)
    y = keras.layers.LayerNormalization()(y)
    x = x + keras.layers.Attention(use_scale=True)([x, x])
    y = y + keras.layers.Attention(use_scale=True)([y, y])
    x = layers.Dense(512, activation='relu')(x)
    y = layers.Dense(512, activation='relu')(y)

    x_0_adjacency = keras.layers.Dense(25600)(x)
    x_0_adjacency = keras.layers.Reshape((160,160))(x_0_adjacency)

    # Replace the TensorFlow transpose operation with Keras Permute
    x_0_adjacency_transposed = keras.layers.Permute((2, 1))(x_0_adjacency)
    #_0_adjacency = (x_0_adjacency + tf.transpose(x_0_adjacency, (0, 2, 1)))
    #print(x_0_adjacency.shape,x_0_adjacency_transposed.shape)

    # Symmetrify tensors in the last two dimensions
    x_0_adjacency = (x_0_adjacency + x_0_adjacency_transposed)

    # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
    x_0_features = keras.layers.Dense(1920)(x)
    x_0_features = keras.layers.Reshape((160,12))(x_0_features)
    x_0_features = keras.layers.Softmax(axis=2)(x_0_features)

    # Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
    x_1_adjacency = keras.layers.Dense(25600)(y)
    x_1_adjacency = keras.layers.Reshape((160,160))(x_1_adjacency)

    # Replace the TensorFlow transpose operation with Keras Permute
    x_1_adjacency_transposed = keras.layers.Permute((2, 1))(x_1_adjacency)
    #x_1_adjacency = (x_1_adjacency + tf.transpose(x_1_adjacency, (0, 2, 1)))

    # Symmetrify tensors in the last two dimensions
    x_1_adjacency = (x_1_adjacency + x_1_adjacency_transposed)

    # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
    x_1_features = keras.layers.Dense(1920)(y)
    x_1_features = keras.layers.Reshape((160,12))(x_1_features)
    x_1_features = keras.layers.Softmax(axis=2)(x_1_features)

    decoder = keras.Model(
        [latent_inputs, latent_inputs_1, conditional_features_1, conditional_features_2],
        outputs=[x_0_adjacency, x_0_features, x_1_adjacency, x_1_features],
        name="decoder"
    )

    return decoder

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var, z_mean_1, z_log_var_1 = inputs
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon, z_mean_1 + tf.exp(0.5 * z_log_var_1) * epsilon


class MoleculeGenerator(keras.Model):
    def __init__(self, encoder, decoder, max_len, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.imine_prediction_layer = layers.Dense(1)
        self.max_len = max_len

        self.train_total_loss_tracker = keras.metrics.Mean(name="train_total_loss")
        self.train_loss1_tracker = keras.metrics.Mean(name="train_total_loss1")
        self.train_loss2_tracker = keras.metrics.Mean(name="train_total_loss2")
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")

    def _gradient_penalty(self, graph_real, graph_generated, imine_tensor, condition_1, condition_2):
        # Unpack graphs
        adjacency_0_real, features_0_real, adjacency_1_real, features_1_real = graph_real
        adjacency_0_generated, features_0_generated, adjacency_1_generated, features_1_generated = graph_generated

        # Generate interpolated graphs (adjacency_interp and features_interp)
        alpha = tf.random.uniform([32])
        alpha = tf.reshape(alpha, (32, 1, 1))
        adjacency_0_interp = (adjacency_0_real * alpha) + (1 - alpha) * adjacency_0_generated
        adjacency_1_interp = (adjacency_1_real * alpha) + (1 - alpha) * adjacency_1_generated
        alpha = tf.reshape(alpha, (32, 1, 1))
        features_0_interp = (features_0_real * alpha) + (1 - alpha) * features_0_generated
        features_1_interp = (features_1_real * alpha) + (1 - alpha) * features_1_generated

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(adjacency_0_interp)
            tape.watch(features_0_interp)
            tape.watch(adjacency_1_interp)
            tape.watch(features_1_interp)
            _, _, _, _, _, _, _, _, logits = self(
                [adjacency_0_interp, features_0_interp, adjacency_1_interp, features_1_interp, imine_tensor,
                 condition_1, condition_2], training=True
            )

        # Compute the gradients with respect to the interpolated graphs
        grads = tape.gradient(logits, [adjacency_0_interp, features_0_interp, adjacency_1_interp, features_1_interp])

        # Compute the gradient penalty
        grads_adjacency_0_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
        grads_features_0_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
        grads_adjacency_1_penalty = (1 - tf.norm(grads[2], axis=1)) ** 2
        grads_features_1_penalty = (1 - tf.norm(grads[3], axis=2)) ** 2
        return tf.reduce_mean(
            tf.reduce_mean(grads_adjacency_0_penalty, axis=(-2, -1))
            + tf.reduce_mean(grads_features_0_penalty, axis=(-1))
            + tf.reduce_mean(grads_adjacency_1_penalty, axis=(-2, -1))  # need to check
            + tf.reduce_mean(grads_features_1_penalty, axis=(-1))  # need to check
        )

    # def train_step(self, data):
    #     adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1, condition_2 = \
    #         data[0]
    #     graph_real = [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor]
    #     # self.batch_size = tf.shape(adjacency_0_tensor)[0]
    #
    #     with tf.GradientTape() as tape:
    #         z_mean, z_log_var, z_mean_1, z_log_var_1, gen_0_adjacency, gen_0_features, gen_1_adjacency, gen_1_features, imine_pred = self(
    #             [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1,
    #              condition_2], training=True)
    #         graph_generated = [gen_0_adjacency, gen_0_features, gen_1_adjacency, gen_1_features]
    #         total_loss = self._compute_loss(
    #             z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated, imine_tensor, imine_pred,
    #             condition_1, condition_2
    #         )
    #     grads = tape.gradient([total_loss], self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #
    #     self.train_total_loss_tracker.update_state(total_loss)
    #
    #     return {'loss': self.train_total_loss_tracker.result()}
    def train_step(self, data):
        adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1, condition_2 = \
            data[0]
        graph_real = [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z_mean_1, z_log_var_1, gen_0_adjacency, gen_0_features, gen_1_adjacency, gen_1_features, imine_pred = self(
                [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1,
                 condition_2], training=True)
            graph_generated = [gen_0_adjacency, gen_0_features, gen_1_adjacency, gen_1_features]
            total_loss = self._compute_loss(
                z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated, imine_tensor, imine_pred,
                condition_1, condition_2
            )
        grads = tape.gradient([total_loss], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.train_total_loss_tracker.update_state(total_loss)

        return {'loss': self.train_total_loss_tracker.result()}

    def _compute_loss(self, z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated, imine_tensor,
                      imine_pred, condition_1, condition_2):
        adjacency_0_real, features_0_real, adjacency_1_real, features_1_real = graph_real
        adjacency_0_gen, features_0_gen, adjacency_1_gen, features_1_gen = graph_generated

        adjacency_0_loss = tf.reduce_mean(
            tf.keras.losses.mean_squared_error(adjacency_0_real, adjacency_0_gen)
        )

        adjacency_1_loss = tf.reduce_mean(
            tf.keras.losses.mean_squared_error(adjacency_1_real, adjacency_1_gen)
        )

        features_0_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(features_0_real, features_0_gen),
                axis=(1),
            )
        )

        features_1_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(features_1_real, features_1_gen),
                axis=(1),
            )
        )

        kl_loss_0 = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1
        )
        kl_loss_0 = tf.reduce_mean(kl_loss_0)
        kl_loss_1 = -0.5 * tf.reduce_sum(
            1 + z_log_var_1 - tf.square(z_mean_1) - tf.exp(z_log_var_1), 1
        )
        kl_loss_1 = tf.reduce_mean(kl_loss_1)
        total_loss = adjacency_0_loss + features_1_loss + kl_loss_0 + adjacency_1_loss + features_0_loss + kl_loss_1

        mols= convert_graph_to_mol_2([[adjacency_0_gen], [features_0_gen],[adjacency_1_gen], [features_1_gen]])
        monomer_smiles = []
        for mol1, mol2 in mols:
            if mol1 is not None and mol2 is not None:
                monomer1_smiles = Chem.MolFromSmiles(mol1)
                monomer2_smiles = Chem.MolFromSmiles(mol2)
                monomer_smiles.append([monomer1_smiles, monomer2_smiles])
        print(monomer_smiles)
        # Calculate rewards in a batch
        if monomer_smiles:
            reward_score = calculate_composite_reward(monomer_smiles)
        else:
            reward_score = 0.0
        print("Reward score: ",reward_score)


        reward_weight = 0.5  # Weight to control the influence of the reward
        total_loss -= reward_weight * reward_score
        print("Total Loss: ",total_loss)

        return total_loss

    def inference(self, batch_size, sample):
        # np.random.seed(121)
        # tf.random.set_seed(42)
        ad_0, feat_0, ad_1, feat_1 = convert_smiles_to_graph(sample)
        print("inference",sample)
        g = [[ad_0], [feat_0], [ad_1], [feat_1]]
        all_mols = []
        imine_list = []

        # temp1 = round(np.random.uniform(0, 4),3)
        # feat_0 = feat_0 + temp1
        # feat_1 = feat_1 + temp1
        random_imine_value = np.random.uniform(0, 2)  # random.randint(0, 1)
        array_160_1 = np.zeros((Constants.NUM_ATOMS, 1))
        imine_list.append(array_160_1 + random_imine_value)

        cond1, cond2 = prepare_condition(sample)
        print(random_imine_value)
        z_mean, log_var, z_mean_1, log_var_1, cond1, cond2 = model.encoder([np.array([ad_0]), np.array([feat_0]),
                                                                            np.array([ad_1]), np.array([feat_1]),
                                                                            np.array(imine_list),
                                                                            np.array([cond1]), np.array([cond2])])
        z1, z2 = Sampling()([z_mean, log_var, z_mean_1, log_var_1])

        min_val = np.mean(z1)
        max_val = np.std(z1)

        min_val_1 = np.mean(z2)
        max_val_1 = np.std(z2)

        for i in range(batch_size):
            z = tf.random.normal((1, Constants.LATENT_DIM))
            z_p = tf.random.normal((1, Constants.LATENT_DIM))
            # z = np.random.normal(min_val,max_val,(1, LATENT_DIM))
            # z_p = np.random.normal(min_val_1,max_val_1,(1, LATENT_DIM))
            z3 = z1 + z  # np.multiply(z1, z) #
            z4 = z2 + z_p  # np.multiply(z1, z_p) #
            reconstruction_adjacnency_0, recontstruction_features_0, reconstruction_adjacnency_1, recontstruction_features_1 = model.decoder.predict(
                [z3, z4, cond1, cond2])

            adjacency_0 = tf.linalg.set_diag(reconstruction_adjacnency_0,
                                             tf.zeros(tf.shape(reconstruction_adjacnency_0)[:-1]))
            adjacency_0 = abs(reconstruction_adjacnency_0[0].astype(int))
            features_0 = tf.argmax(recontstruction_features_0, axis=2)
            features_0 = tf.one_hot(features_0, depth=Constants.ATOM_DIM, axis=2)

            adjacency_1 = tf.linalg.set_diag(reconstruction_adjacnency_1,
                                             tf.zeros(tf.shape(reconstruction_adjacnency_1)[:-1]))
            features_1 = tf.argmax(recontstruction_features_1, axis=2)
            features_1 = tf.one_hot(features_1, depth=Constants.ATOM_DIM, axis=2)
            adjacency_1 = abs(reconstruction_adjacnency_1[0].astype(int))
            # graph2=[[adjacency_0[0].numpy()],[features_0[0].numpy()],[adjacency_1[0].numpy()],[features_1[0].numpy()]]
            graph2 = [[adjacency_0], [features_0[0].numpy()],
                      [adjacency_1], [features_0[0].numpy()]]
            all_mols.append(convert_graph_to_mol(graph2))
        return all_mols

    def call(self, inputs):
        z_mean, log_var, z_mean_1, log_var_1, cond1, cond2 = self.encoder(inputs)
        z, z1 = Sampling()([z_mean, log_var, z_mean_1, log_var_1])
        gen_adjacency_0, gen_features_0, gen_adjacency_1, gen_features_1 = self.decoder([z, z1, cond1, cond2])
        imine_pred = self.imine_prediction_layer(z_mean + z_mean_1)

        return z_mean, log_var, z_mean_1, log_var_1, gen_adjacency_0, gen_features_0, gen_adjacency_1, gen_features_1, imine_pred


vae_optimizer = tf.keras.optimizers.Adam(learning_rate=Constants.VAE_LR)

encoder = get_encoder(
    gconv_units=[9],
    adjacency_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS),
    feature_shape=(Constants.NUM_ATOMS, Constants.ATOM_DIM),
    imine_shape=(Constants.NUM_ATOMS, 1),
    condition_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS),
    latent_dim=Constants.LATENT_DIM,
    dense_units=[128, 256, 512],
    dropout_rate=0.2,
)
decoder = get_decoder(
    dense_units=[128, 256, 512, 1024],
    dropout_rate=0.2,
    latent_dim=Constants.LATENT_DIM,
    adjacency_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS),
    feature_shape=(Constants.NUM_ATOMS, Constants.ATOM_DIM),
    condition_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS)
)

model = MoleculeGenerator(encoder, decoder, 160)

checkpoint_path = "/ddnB/work/borun22/TSMPLLM/Reward_Model/Model/check_points/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
n_batches = len(adjacency_0_tensor) / Constants.BATCH_SIZE
n_batches = math.ceil(n_batches)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(
#    filepath=checkpoint_path,
#    verbose=1,
#    save_weights_only=True,
#    save_freq=5 * n_batches)

stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, verbose=1, restore_best_weights=True)
# model.save_weights(checkpoint_path.format(epoch=0))

model.compile(vae_optimizer)

history = model.fit([adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor,
                     imine_features_array, condition_1_array, condition_2_array],
                    epochs=Constants.EPOCHS, callbacks=[stopping_callback])

#encoder.save_weights('/home/C00521897/Fall 22/New_Monomer_generation/saved_model/encoder_weights.h5')
#decoder.save_weights('/home/C00521897/Fall 22/New_Monomer_generation/saved_model/decoder_weights.h5')
#model.save('/home/C00521897/Fall 22/New_Monomer_generation/saved_model/vae_model')


def write_samples_in_file():
    i = 0
    with open(Constants.OUTPUT_SAVE_PATH, 'a') as the_file:
        # random.seed(121)
        smiles = random.choices(df_smiles, k=25)
        for sample in smiles:
            i = i + 1
            text = '\n-----------------' + str(i) + '---------------------------------\n'
            mols = model.inference(5, sample)
            the_file.write(text)
            the_file.write(str(sample))
            the_file.write('\n--------------------------------------------------\n')
            for index, m in enumerate(mols):
                if m[0][0] is not None:
                    smiles = Chem.MolToSmiles(m[0][0])
                else:
                    smiles = "None"
                smiles = smiles + ","
                if m[0][1] is not None:
                    smiles += Chem.MolToSmiles(m[0][1]) + "\n"
                else:
                    smiles += "None\n"
                the_file.write(smiles)


write_samples_in_file()
