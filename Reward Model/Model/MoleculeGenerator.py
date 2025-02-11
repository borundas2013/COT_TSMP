from tensorflow.keras import layers
from tensorflow.keras import backend as K
import keras
import tensorflow as tf
from Features import convert_graph_to_mol_2
from Reward_Function import calculate_composite_reward
from Sampling import Sampling
import numpy as np

class MoleculeGenerator(keras.Model):
    def __init__(self, encoder, decoder, max_len, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.imine_prediction_layer = layers.Dense(1)
        self.max_len = max_len
        self.train_total_loss_tracker = keras.metrics.Mean(name="train_total_loss")
        self.kl_weight = 0.0  # Start with a low KL weight for annealing

    def _compute_loss(self, z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated, imine_tensor,
                      imine_pred, condition_1, condition_2):
        adjacency_0_real, features_0_real, adjacency_1_real, features_1_real = graph_real
        adjacency_0_gen, features_0_gen, adjacency_1_gen, features_1_gen = graph_generated
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


        adjacency_0_loss = K.mean(
                K.sum(
                    mse(adjacency_0_real, adjacency_0_gen),
                    axis=1  # Reduce across the desired axis (e.g., summing along the last axis)
                )
        )

        adjacency_1_loss = K.mean(
                K.sum(
                    mse(adjacency_1_real, adjacency_1_gen),
                    axis=1
                )
        )

        features_0_loss = K.mean(
                K.sum(
                    cce(features_0_real, features_0_gen),
                    axis=1  # Sum across the feature axis
                )
        )

        features_1_loss = K.mean(
                K.sum(
                    cce(features_1_real, features_1_gen),
                    axis=1
                )
        )

        # KL divergence loss (with annealing)
        kl_loss_0 = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        kl_loss_1 = -0.5 * K.sum(1 + z_log_var_1 - K.square(z_mean_1) - K.exp(z_log_var_1), axis=1)

        # Anneal KL weight over time
        self.kl_weight = min(1.0, self.kl_weight + 0.001)

        # Composite loss
        total_loss = (adjacency_0_loss + features_0_loss + adjacency_1_loss + features_1_loss) + \
                     self.kl_weight * (kl_loss_0 + kl_loss_1)

        # (Optional) Molecular validity and complexity reward
        generated_molecules = convert_graph_to_mol_2(graph_generated)
        reward = self._calculate_complexity_reward(generated_molecules)

        # Final weighted loss: reconstruction + KL + reward
        final_loss = total_loss - 0.1 * reward  # Adjust reward scaling factor

        return final_loss

    def _calculate_complexity_reward(self, generated_molecules):
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        reward = 0.0
        valid_molecule_count = 0
        for mols in generated_molecules:
            if mols[0] is not None and mols[1] is not None:
                sm1 = Chem.MolToSmiles(mols[0])
                sm2 = Chem.MolToSmiles(mols[1])
                # Reward based on molecular weight and complexity
                reward += Descriptors.MolWt(Chem.MolFromSmiles(sm1)) * 0.01
                reward += Descriptors.FractionCSP3(Chem.MolFromSmiles(sm1)) * 10
                valid_molecule_count += 1

        if valid_molecule_count > 0:
            reward = reward / valid_molecule_count  # Average reward

        return reward

    def train_step(self, data):
        adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1, condition_2 = data
        graph_real = [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor]

        lambda_weight = 0.8  # Control balance between reward and loss

        with tf.GradientTape() as tape:
            # Forward pass using the `call()` method
            z_mean, z_log_var, z_mean_1, z_log_var_1, gen_0_adjacency, gen_0_features, gen_1_adjacency, gen_1_features, imine_pred = self(
                [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1,
                 condition_2],
                training=True
            )

            graph_generated = [gen_0_adjacency, gen_0_features, gen_1_adjacency, gen_1_features]

            # Compute VAE reconstruction loss (ensure it's a TensorFlow tensor)
            total_loss = self._compute_loss(
                z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated, imine_tensor, imine_pred,
                condition_1, condition_2
            )
            print('TOTAL LOSS', total_loss)


            generated_molecules = convert_graph_to_mol_2(graph_generated)
            print(generated_molecules)
            rewards=0.0# Implement this function
            for mols in generated_molecules:
                if mols[0] is not None and mols[1] is not None:
                    sm1= Chem.MolToSmiles(mols[0])
                    sm2= Chem.MolToSmiles(mols[1])
                    smiles=[sm1,sm2]
                    rewards = tf.convert_to_tensor(calculate_composite_reward(smiles),dtype=tf.float32)
            avg_reward =tf.reduce_mean(rewards)
            #
            # # Weighted combination of loss and reward using TensorFlow ops

            weighted_loss = abs(lambda_weight * avg_reward - (1 - lambda_weight) * total_loss)


        # Apply gradients
        grads = tape.gradient([weighted_loss], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.train_total_loss_tracker.update_state(total_loss)

        return {'loss': self.train_total_loss_tracker.result(), 'reward': avg_reward}

    # def _compute_loss(self, z_log_var, z_mean, z_mean_1, z_log_var_1, graph_real, graph_generated, imine_tensor, imine_pred, condition_1, condition_2):
    #     adjacency_0_real, features_0_real, adjacency_1_real, features_1_real = graph_real
    #     adjacency_0_gen, features_0_gen, adjacency_1_gen, features_1_gen = graph_generated
    #     mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    #     cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    #
    #     # Use the instantiated objects to calculate the loss and then reduce with sum or mean
    #     adjacency_0_loss = K.mean(
    #         K.sum(
    #             mse(adjacency_0_real, adjacency_0_gen),
    #             axis=1  # Reduce across the desired axis (e.g., summing along the last axis)
    #         )
    #     )
    #
    #     adjacency_1_loss = K.mean(
    #         K.sum(
    #             mse(adjacency_1_real, adjacency_1_gen),
    #             axis=1
    #         )
    #     )
    #
    #     features_0_loss = K.mean(
    #         K.sum(
    #             cce(features_0_real, features_0_gen),
    #             axis=1  # Sum across the feature axis
    #         )
    #     )
    #
    #     features_1_loss = K.mean(
    #         K.sum(
    #             cce(features_1_real, features_1_gen),
    #             axis=1
    #         )
    #     )
    #
    #     kl_loss_0 = -0.5 * K.sum(
    #         1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1
    #     )
    #     kl_loss_0 = K.mean(kl_loss_0)
    #
    #     kl_loss_1 = -0.5 * K.sum(
    #         1 + z_log_var_1 - K.square(z_mean_1) - K.exp(z_log_var_1), axis=1
    #     )
    #     kl_loss_1 = K.mean(kl_loss_1)
    #     total_loss = adjacency_0_loss + features_1_loss + kl_loss_0 + adjacency_1_loss + features_0_loss + kl_loss_1
    #
    #     return total_loss

    def call(self, inputs, training=False):
        adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1, condition_2 = inputs

        # Pass inputs through the encoder
        z_mean, log_var, z_mean_1, log_var_1, cond1, cond2 = self.encoder(
            [adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor, imine_tensor, condition_1,
             condition_2],
            training=training
        )

        # Sample from the latent space using the Sampling layer
        z, z1 = Sampling()([z_mean, log_var, z_mean_1, log_var_1])

        # Decode the latent representation to reconstruct the graphs
        gen_adjacency_0, gen_features_0, gen_adjacency_1, gen_features_1 = self.decoder([z, z1, cond1, cond2],
                                                                                        training=training)

        # Predict imine-related values (optional, adjust as needed)
        imine_pred = self.imine_prediction_layer(z_mean + z_mean_1)

        return z_mean, log_var, z_mean_1, log_var_1, gen_adjacency_0, gen_features_0, gen_adjacency_1, gen_features_1, imine_pred


# Custom Training Loop for Subclassed Model
def train_model_with_reward(model, dataset, epochs, lambda_weight, save_path):
    best_score = -float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = []
        epoch_reward = []

        for batch_data in dataset:
            result = model.train_step(batch_data)

            epoch_loss.append(result['loss'].numpy())
            epoch_reward.append(result['reward'])

        avg_loss = np.mean(epoch_loss)
        avg_reward = np.mean(epoch_reward)

        # Weighted combination of reward and loss
        score = lambda_weight * avg_reward - (1 - lambda_weight) * avg_loss

        if score > best_score:
            best_score = score
            model.save_weights(save_path)
            print(f"New best model saved with score: {best_score} (Reward: {avg_reward}, Loss: {avg_loss})")

    print("Training complete.")
