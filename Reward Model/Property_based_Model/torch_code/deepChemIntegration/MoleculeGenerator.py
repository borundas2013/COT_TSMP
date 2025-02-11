from tensorflow.keras import layers
from tensorflow.keras import backend as K
import keras
import tensorflow as tf
from Sampling import Sampling
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from feature_extraction import features_to_smiles

class MoleculeGenerator(keras.Model):
    def __init__(self, encoder, decoder, max_len, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.imine_prediction_layer = layers.Dense(1)
        self.max_len = max_len
        self.train_total_loss_tracker = keras.metrics.Mean(name="train_total_loss")
        self.kl_weight = 0.0  # Start with a low KL weight for annealing

    def _compute_loss(self, z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated):
        node_features_0_real, edge_features_0_real, edge_index_0_real, \
        node_features_1_real, edge_features_1_real, edge_index_1_real = graph_real
        
        node_features_0_gen, edge_features_0_gen, edge_index_0_gen, \
        node_features_1_gen, edge_features_1_gen, edge_index_1_gen = graph_generated

        print("Edge index shapes:")
        print("Real:", K.int_shape(graph_real[2]), K.int_shape(graph_real[5]))
        print("Generated:", K.int_shape(graph_generated[2]), K.int_shape(graph_generated[5]))

        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        # Node feature losses
        node_features_0_loss = K.mean(K.sum(cce(node_features_0_real, node_features_0_gen), axis=1))
        node_features_1_loss = K.mean(K.sum(cce(node_features_1_real, node_features_1_gen), axis=1))

        # Edge feature losses
        edge_features_0_loss = K.mean(K.sum(cce(edge_features_0_real, edge_features_0_gen), axis=1))
        edge_features_1_loss = K.mean(K.sum(cce(edge_features_1_real, edge_features_1_gen), axis=1))

        # Edge index losses (using MSE for continuous values)
        edge_index_0_loss = K.mean(K.sum(cce(edge_index_0_real, edge_index_0_gen), axis=1))
        edge_index_1_loss = K.mean(K.sum(cce(edge_index_1_real, edge_index_1_gen), axis=1))

        # KL divergence loss
        kl_loss_0 = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        kl_loss_1 = -0.5 * K.sum(1 + z_log_var_1 - K.square(z_mean_1) - K.exp(z_log_var_1), axis=1)

        # Total loss
        total_loss = (node_features_0_loss + node_features_1_loss + 
                     edge_features_0_loss + edge_features_1_loss +
                     edge_index_0_loss + edge_index_1_loss +
                     kl_loss_0 + kl_loss_1)
        
    

        # Convert generated graph features to SMILES strings for each batch item
        batch_size = tf.shape(node_features_0_gen)[0]
        rewards = []
        lambda_weight = 0.4

        for i in range(batch_size): 
            sm1 = features_to_smiles(
                node_features_0_gen[i], edge_features_0_gen[i], edge_index_0_gen[i]
            )
            sm2 = features_to_smiles(
                node_features_1_gen[i], edge_features_1_gen[i], edge_index_1_gen[i]
            )
            print(sm1,sm2)

            # Calculate rewards if valid SMILES were generated
            if sm1 is not None and sm2 is not None:
                smiles = [sm1, sm2]
                reward = self._calculate_complexity_reward(smiles)
                rewards.append(reward)
            else:
                rewards.append(0.0)
        
        avg_reward = K.mean(tf.convert_to_tensor(rewards, dtype=tf.float32)) if rewards else 0.0
        weighted_loss = tf.abs((lambda_weight * avg_reward) - (1 - lambda_weight) * total_loss)

        return weighted_loss, avg_reward

    def _calculate_complexity_reward(self, smiles):
        reward = 0.0      
        sm1 = smiles[0]
        sm2 = smiles[1]
        reward += Descriptors.MolWt(Chem.MolFromSmiles(sm1)) * 0.01
        reward += Descriptors.FractionCSP3(Chem.MolFromSmiles(sm2)) * 0.09
        return reward

    @tf.function
    def train_step(self, data):
        node_features_0 = data['node_features_0']
        edge_features_0 = data['edge_features_0']
        edge_index_0 = data['edge_index_0']
        node_features_1 = data['node_features_1']
        edge_features_1 = data['edge_features_1']
        edge_index_1 = data['edge_index_1']
        property_matrix = data['property_matrix']
        # condition_1 = data['condition_1']
        # condition_2 = data['condition_2']

        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z_mean_1, z_log_var_1, gen_node_0, gen_edge_0, gen_edge_idx_0, \
            gen_node_1, gen_edge_1, gen_edge_idx_1 = self(
                [node_features_0, edge_features_0, edge_index_0,
                 node_features_1, edge_features_1, edge_index_1,
                 property_matrix],
                training=True
            )

            graph_generated = [gen_node_0, gen_edge_0, gen_edge_idx_0,
                             gen_node_1, gen_edge_1, gen_edge_idx_1]
            graph_real = [node_features_0, edge_features_0, edge_index_0,
                         node_features_1, edge_features_1, edge_index_1]

            total_loss, avg_reward = self._compute_loss(
                z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated)

        # Apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.train_total_loss_tracker.update_state(total_loss)

        return {'loss': self.train_total_loss_tracker.result(), 'reward': avg_reward}

    def call(self, inputs, training=False):
        node_features_0, edge_features_0, edge_index_0, \
        node_features_1, edge_features_1, edge_index_1, \
        property_matrix = inputs

        # Encoder forward pass
        z_mean, log_var, z_mean_1, log_var_1 = self.encoder(
            [node_features_0, edge_features_0, edge_index_0,
             node_features_1, edge_features_1, edge_index_1,
             property_matrix],
            training=training
        )

        # Sample from the latent space
        z, z1 = Sampling()([z_mean, log_var, z_mean_1, log_var_1])

        # Decoder forward pass
        gen_node_0, gen_edge_0, gen_edge_idx_0, \
        gen_node_1, gen_edge_1, gen_edge_idx_1 = self.decoder(
            [z, z1, property_matrix],
            training=training
        )

        return z_mean, log_var, z_mean_1, log_var_1, \
               gen_node_0, gen_edge_0, gen_edge_idx_0, \
               gen_node_1, gen_edge_1, gen_edge_idx_1

def train_model_with_reward(model, dataset, epochs, lambda_weight, save_path):
    best_score = float('-inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = []
        epoch_rewards = []
        for batch in dataset:
            result = model.train_step(batch)
            epoch_losses.append(result['loss'])
            epoch_rewards.append(result['reward'])
        
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        # Weighted combination of reward and loss
        score = lambda_weight * avg_reward - (1 - lambda_weight) * avg_loss

        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.2f}, Avg Reward = {avg_reward:.2f}, Score = {score:.2f}")
        model.save_weights(save_path)
        if score > best_score:
            best_score = score
            model.save_weights(save_path)
            print(f"New best model saved with score: {best_score:.4f}")

    print("Training complete.")

