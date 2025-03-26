import tensorflow as tf
import numpy as np
from rdkit import Chem
from Data_Process_with_prevocab_gen import decode_smiles, extract_group_smarts
import Constants
from Model import *
from Data_Process_with_prevocab_gen import *
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import json
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from CustomLoss import *
from Sample_Predictor import *
from sklearn.model_selection import train_test_split
#from PPOTrainer import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Input, Dense, LSTM, Dropout, Embedding, Lambda
from tensorflow.keras.models import Model
import wandb
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

T = 1000  # Number of diffusion steps
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

NUM_SAMPLES = 100
EPOCHS = 100
BATCH_SIZE = 64

@keras.saving.register_keras_serializable()  # Add this decorator
class TimeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(TimeEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
    
    def call(self, inputs):
        return tf.reshape(inputs, [-1, 1, self.embedding_dim])
    
    # Add get_config method for serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim
        })
        return config
    
@keras.saving.register_keras_serializable()
class PositionalEncoding2(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embedding_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        
    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            "position_embeddings",
            shape=(self.sequence_length, self.embedding_dim),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False
        )

        position = tf.range(self.sequence_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.embedding_dim, dtype=tf.float32)[tf.newaxis, :]
        div_term = tf.math.exp(-tf.math.log(10000.0) * (2 * (i // 2) / self.embedding_dim))
        
        self.position_embeddings.assign(
            tf.concat([
                tf.math.sin(position * div_term[:, ::2]),
                tf.math.cos(position * div_term[:, 1::2])
            ], axis=-1)
        )
        
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)
            return inputs + self.position_embeddings[tf.newaxis, :, :] * mask
        return inputs + self.position_embeddings[tf.newaxis, :, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim
        })
        return config


@keras.saving.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embedding_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        
    def build(self, input_shape):
        position_embeddings = np.zeros((self.sequence_length, self.embedding_dim))
        
        for pos in range(self.sequence_length):
            for i in range(0, self.embedding_dim, 2):
                position_embeddings[pos, i] = np.sin(pos / (10000 ** (2 * i / self.embedding_dim)))
                if i + 1 < self.embedding_dim:
                    position_embeddings[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / self.embedding_dim)))
                    
        self.position_embeddings = tf.convert_to_tensor(position_embeddings, dtype=tf.float32)
        
    def call(self, inputs):
        # Add the positional encoding to the input embeddings
        return inputs + self.position_embeddings[tf.newaxis, :, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim
        })
        return config



class DiffusionScheduler:
    def __init__(self, T, schedule_type='cosine'):
        self.T = T
        self.schedule_type = schedule_type
        self.betas = self._get_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
    def _get_schedule(self):
        if self.schedule_type == 'linear':
            return np.linspace(beta_start, beta_end, self.T)
        elif self.schedule_type == 'cosine':
            # Cosine schedule as proposed in "Improved Denoising Diffusion"
            steps = self.T + 1
            x = np.linspace(0, self.T, steps)
            alphas_cumprod = np.cos(((x / self.T) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0001, 0.9999)




time_embed = Embedding(input_dim=T, output_dim=128)

def build_improved_diffusion_model(vocab_size, seq_length, embedding_dim=128):
    input_seq = Input(shape=(seq_length,), name="input_seq")
    t_input = Input(shape=(1,), name="t_input")

    # Embedding layers
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_seq)

    # Time embedding
    t_embed = time_embed(t_input)
    t_embed = TimeEmbeddingLayer(embedding_dim)(t_embed)
    x = x + t_embed * tf.sqrt(tf.cast(embedding_dim, tf.float32))

    # Positional encoding
    pos_encoding = PositionalEncoding(seq_length, embedding_dim)(x)
    x = x + pos_encoding

    # Transformer blocks
    for _ in range(2):  # Reduced to 2 for better stability
        # Self-attention block
        attention = MultiHeadAttention(num_heads=8, key_dim=embedding_dim)(x, x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attention)

        # Feed-forward block
        ff = Dense(embedding_dim * 4, activation="relu")(x)
        ff = Dropout(0.2)(ff)  # Increased dropout
        ff = Dense(embedding_dim)(ff)
        x = LayerNormalization(epsilon=1e-6)(x + ff)

    # LSTM block (combine attention with sequential learning)
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # Output projection
    out = Dense(vocab_size, activation="softmax")(x)

    # Compile model
    model = Model(inputs=[input_seq, t_input], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# def build_improved_diffusion_model(vocab_size, seq_length, embedding_dim=128):
#     input_seq = Input(shape=(seq_length,), name="input_seq")
#     t_input = Input(shape=(1,), name="t_input")
    
#     # Embedding layers
#     x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_seq)
    
#     # Time embedding
#     t_embed = time_embed(t_input)
#     t_embed = Lambda(lambda x: tf.reshape(x, [-1, 1, embedding_dim]))(t_embed)
#     x = x + t_embed
    
#     # Add positional encoding
#     pos_encoding = PositionalEncoding(seq_length, embedding_dim)(x)
#     x = x + pos_encoding
    
#     # Main processing blocks
#     for _ in range(3):
#         # Self-attention block
#         attention = MultiHeadAttention(num_heads=8, key_dim=embedding_dim)(x, x, x)
#         x = LayerNormalization(epsilon=1e-6)(x + attention)
        
#         # Feed-forward block
#         ff = Dense(embedding_dim * 4, activation="relu")(x)
#         ff = Dropout(0.1)(ff)
#         ff = Dense(embedding_dim)(ff)
#         x = LayerNormalization(epsilon=1e-6)(x + ff)
    
#     # Output projection
#     x = LSTM(256, return_sequences=True)(x)
#     x = Dropout(0.1)(x)
#     out = Dense(vocab_size, activation="softmax")(x)
    
#     model = Model(inputs=[input_seq, t_input], outputs=out)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#     return model


class MolecularMetrics:
   
    def calculate_validity(self,smiles_list):
        valid_count = 0
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
        return valid_count / len(smiles_list)
    
    def calculate_novelty(self,generated_smiles, training_smiles):
        # Convert tensor to numpy array if it's a tensor
        if tf.is_tensor(training_smiles):
            training_smiles = training_smiles.numpy()
        
        # Convert training smiles to actual SMILES strings if they're still token IDs
        if isinstance(training_smiles[0], (np.ndarray, list)):
            training_smiles = [decode_smiles(smile, Constants.TOKENIZER_PATH) for smile in training_smiles]
        
        # Now create the set
        training_set = set(training_smiles)
        novel_count = sum(1 for smiles in generated_smiles if smiles not in training_set)
        return novel_count / len(generated_smiles) if generated_smiles else 0
    
   
    def calculate_diversity(self,smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]
        
        diversity = 0
        n = len(fps)
        for i in range(n):
            for j in range(i + 1, n):
                diversity += 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        return 2 * diversity / (n * (n - 1)) if n > 1 else 0


class DiffusionTrainer:
    def __init__(self, model, scheduler, training_data, validation_data, max_length, vocab_size):
        self.model = model
        self.scheduler = scheduler
        self.training_data = training_data.numpy() if tf.is_tensor(training_data) else training_data
        self.validation_data = validation_data.numpy() if tf.is_tensor(validation_data) else validation_data
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.metrics = MolecularMetrics()
    
    def sample(self, model, t_steps, max_length, vocab_size):
        # Start with random noise of shape [1, max_length]
        x = tf.random.uniform(shape=(1, max_length), minval=0, maxval=vocab_size, dtype=tf.int32)
        x = tf.cast(x, tf.float32)  # Convert x to float32 for operations
        
        for t in reversed(range(t_steps)):
            # Predict the noise (as softmax over vocab)
            predicted_probs = model.predict([x, tf.convert_to_tensor([t])])

            # Convert predicted probabilities to token predictions using argmax
            predicted_tokens = tf.argmax(predicted_probs, axis=-1)

            # Align shapes by converting predicted tokens to float
            predicted_tokens = tf.cast(predicted_tokens, tf.float32)

            # Scale by beta and alphas to reverse the diffusion step
            beta_t = tf.cast(betas[t], tf.float32)
            alpha_t = tf.cast(alphas[t], tf.float32)

            # Reverse process to remove noise
            x = (x - beta_t * predicted_tokens) / tf.sqrt(alpha_t)

            # Clip to valid token range
            x = tf.clip_by_value(x, 0, vocab_size - 1)

        # Convert back to integer IDs
        x = tf.cast(x, tf.int32)
        return x

    def forward_diffusion(self,x0, t, vocab_size):
        x0 = tf.cast(x0, tf.float32)
        noise = tf.random.normal(shape=x0.shape, dtype=tf.float32)

        sqrt_alpha_cumprod = tf.reshape(tf.cast(tf.sqrt(tf.gather(alphas_cumprod, t)), tf.float32), [-1, 1])
        sqrt_one_minus_alpha_cumprod = tf.reshape(tf.cast(tf.sqrt(tf.gather(1.0 - alphas_cumprod, t)), tf.float32), [-1, 1])

        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

        # Clip output values to valid token range
        xt = tf.clip_by_value(xt, 0, vocab_size - 1)
        return xt, noise
        
    def train(self, epochs, batch_size, num_samples_eval=100):
        for epoch in range(epochs):
            # Training loop
            epoch_losses = []
            for i in range(len(self.training_data) // batch_size):
                x0 = self.training_data[i * batch_size:(i + 1) * batch_size]
                t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.scheduler.T, dtype=tf.int32)
                
                xt, noise = self.forward_diffusion(x0, t, self.vocab_size)
                loss = self.model.train_on_batch([xt, t], x0)
                epoch_losses.append(loss[0])
                
            # Evaluation
            generated_samples = self.generate_samples(num_samples_eval)
            metrics = self.evaluate_samples(generated_samples,self.training_data)
            
            # Logging
            #self.log_metrics(epoch, np.mean(epoch_losses), metrics)
            
    def generate_samples(self, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = self.sample(self.model, self.scheduler.T, self.max_length, self.vocab_size)
            smiles = decode_smiles(sample.numpy()[0], Constants.TOKENIZER_PATH)
            print(smiles)
            samples.append(smiles)
        return samples
    
    def evaluate_samples2(self, generated_samples):
        return {
            'validity': self.metrics.calculate_validity(generated_samples),
            'novelty': self.metrics.calculate_novelty(generated_samples, self.training_data),
            'diversity': self.metrics.calculate_diversity(generated_samples)
        }
    def evaluate_samples(self, generated_smiles, training_smiles):
        """
        Calculate and print all metrics for generated molecules
        """
        print("\nEvaluation Metrics:")
        print("=" * 50)
        
        # Convert training_smiles from numpy array to list of SMILES strings if needed
        if isinstance(training_smiles, np.ndarray):
            training_smiles = [decode_smiles(smile, Constants.TOKENIZER_PATH) for smile in training_smiles]
        
        # Validity Rate
        valid_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]
        valid_mols = [m for m in valid_mols if m is not None]
        validity_rate = len(valid_mols) / len(generated_smiles) if generated_smiles else 0
        print(f"Validity Rate: {validity_rate:.4f} ({len(valid_mols)} / {len(generated_smiles)} valid molecules)")
        
        # Only calculate other metrics if we have valid molecules
        if valid_mols:
            # Chemical Diversity
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in valid_mols]
            diversity_sum = 0
            pair_count = 0
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    diversity_sum += 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    pair_count += 1
            diversity = diversity_sum / pair_count if pair_count > 0 else 0
            print(f"Chemical Diversity: {diversity:.4f}")
            
            # Novelty (compared to training set)
            training_set = set(training_smiles)  # Now training_smiles are strings
            novel_count = sum(1 for s in generated_smiles if s not in training_set)
            novelty_rate = novel_count / len(generated_smiles)
            print(f"Novelty Rate: {novelty_rate:.4f} ({novel_count} / {len(generated_smiles)} novel molecules)")
            
            # # Additional Properties Distribution
            # mw_list = [Descriptors.ExactMolWt(mol) for mol in valid_mols]
            # logp_list = [Descriptors.MolLogP(mol) for mol in valid_mols]
            # hbd_list = [Descriptors.NumHDonors(mol) for mol in valid_mols]
            # hba_list = [Descriptors.NumHAcceptors(mol) for mol in valid_mols]
            
            # print("\nMolecular Properties (Mean ± Std):")
            # print(f"Molecular Weight: {np.mean(mw_list):.2f} ± {np.std(mw_list):.2f}")
            # print(f"LogP: {np.mean(logp_list):.2f} ± {np.std(logp_list):.2f}")
            # print(f"H-Bond Donors: {np.mean(hbd_list):.2f} ± {np.std(hbd_list):.2f}")
            # print(f"H-Bond Acceptors: {np.mean(hba_list):.2f} ± {np.std(hba_list):.2f}")
        
        print("=" * 50)
        
        # Return metrics as dictionary
        metrics_dict = {
            'validity_rate': validity_rate,
            'diversity': diversity if valid_mols else 0,
            'novelty_rate': novelty_rate if valid_mols else 0
        }
        
        return metrics_dict

# Modify the training loop to use these metrics
def train(self, epochs, batch_size, num_samples_eval=1000):
    """
    Training loop with detailed metrics
    """
    metrics_history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training loop
        epoch_losses = []
        for i in range(len(self.training_data) // batch_size):
            x0 = self.training_data[i * batch_size:(i + 1) * batch_size]
            t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.scheduler.T, dtype=tf.int32)
            
            xt, noise = self.forward_diffusion(x0, t, self.vocab_size)
            loss = self.model.train_on_batch([xt, t], x0)
            epoch_losses.append(loss[0])
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{len(self.training_data) // batch_size}, Loss: {loss[0]:.4f}")
        
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\nAverage Epoch Loss: {avg_epoch_loss:.4f}")
        
        # Generate and evaluate samples
        print("\nGenerating samples for evaluation...")
        generated_samples = self.generate_samples(num_samples_eval)
        metrics = self.evaluate_samples(generated_samples, self.training_data)
        metrics['epoch'] = epoch + 1
        metrics['avg_loss'] = avg_epoch_loss
        metrics_history.append(metrics)
        
        # Save metrics to file
        with open(f'training_metrics_epoch_{epoch + 1}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    return metrics_history
    
    def log_metrics(self, epoch, loss, metrics):
        # Convert numpy types to Python native types
        log_data = {
            'epoch': int(epoch),
            'loss': float(loss),  # Convert numpy.float32 to Python float
        }
        
        # Convert metrics values to Python native types
        metrics_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (np.float32, np.float64)):
                metrics_dict[key] = float(value)
            elif isinstance(value, np.integer):
                metrics_dict[key] = int(value)
            else:
                metrics_dict[key] = value
        
        log_data.update(metrics_dict)
        
        # Save to file
        with open(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(log_data, f, indent=4)
        
        # Print to console
        print(f"Epoch {epoch}: {log_data}")
        # wandb.log(log_data)
        # logging.info(f"Epoch {epoch}: {log_data}")
        
        # Visualize sample molecules
        # if epoch % 10 == 0:
        #     self.visualize_molecules(epoch)
    
    # def visualize_molecules(self, epoch):
    #     samples = self.generate_samples(4)
    #     fig = plt.figure(figsize=(10, 10))
    #     for idx, smiles in enumerate(samples):
    #         mol = Chem.MolFromSmiles(smiles)
    #         if mol:
    #             ax = fig.add_subplot(2, 2, idx + 1)
    #             img = Draw.MolToImage(mol)
    #             ax.imshow(img)
    #             ax.axis('off')
    #     plt.savefig(f'molecules_epoch_{epoch}.png')
    #     wandb.log({'molecule_samples': wandb.Image(plt)})
    #     plt.close()

from rdkit.Chem import Descriptors

def sample_and_save_smiles(model, t_steps, max_length, vocab_size, num_samples=100):
    """
    Generates SMILES strings and saves them to a file with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_smiles_{timestamp}.txt"
    
    # Start with random noise
    x = tf.random.uniform(shape=(num_samples, max_length), minval=0, maxval=vocab_size, dtype=tf.int32)
    x = tf.cast(x, tf.float32)
    
    # Denoising process
    for t in reversed(range(t_steps)):
        # Create time tensor with same batch size as x
        t_batch = tf.ones((num_samples,), dtype=tf.int32) * t  # Fix: Create time tensor with correct batch size
        
        predicted_probs = model.predict([x, t_batch])  # Fix: Use t_batch instead of [t]
        predicted_tokens = tf.argmax(predicted_probs, axis=-1)
        predicted_tokens = tf.cast(predicted_tokens, tf.float32)
        
        beta_t = tf.cast(betas[t], tf.float32)
        alpha_t = tf.cast(alphas[t], tf.float32)
        x = (x - beta_t * predicted_tokens) / tf.sqrt(alpha_t)
        x = tf.clip_by_value(x, 0, vocab_size - 1)
    
    # Get final results
    final_x = tf.cast(x, tf.int32)
    final_smiles = []
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(f"Generated SMILES - {timestamp}\n")
        f.write("-" * 50 + "\n")
        
        for i in range(num_samples):
            smiles = decode_smiles(final_x[i].numpy(), Constants.TOKENIZER_PATH)
            final_smiles.append(smiles)
            
            # Write to file
            f.write(f"Sample {i+1}: {smiles}\n")
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} molecules")
    
    print(f"\nGenerated SMILES have been saved to: {filename}")
    return final_smiles, filename

def save_model_and_parameters(model, vocab_size, max_length, save_dir="saved_model"):
    """
    Save the model and its parameters
    """

    save_dir = f"{save_dir}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the model
    model.save(f"{save_dir}/diffusion_model.keras")
    
    # Save parameters
    params = {
        'vocab_size': vocab_size,
        'max_length': max_length,
        'T': T,  # number of diffusion steps
        'beta_start': beta_start,
        'beta_end': beta_end
    }
    
    with open(f"{save_dir}/parameters.json", 'w') as f:
        json.dump(params, f)
    
    print(f"Model and parameters saved in {save_dir}")
    return save_dir

def load_model_and_generate(model_dir, num_samples=5):
    """
    Load the saved model and generate SMILES
    """
    # Load parameters
    with open(f"{model_dir}/parameters.json", 'r') as f:
        params = json.load(f)
    
    # Load model
    model = tf.keras.models.load_model(f"{model_dir}/diffusion_model.keras")
    
   
    
    return  model, params


def sample_and_save_smiles_with_initial(model, t_steps, max_length, vocab_size, num_samples=100):
    """
    Generates SMILES strings and saves initial and final versions with arrows
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_smiles_comparison_{timestamp}.txt"
    
    # Start with random noise
    x = tf.random.uniform(shape=(num_samples, max_length), minval=0, maxval=vocab_size, dtype=tf.int32)
    x = tf.cast(x, tf.float32)
    
    # Get initial SMILES
    initial_x = tf.cast(x, tf.int32)
    initial_smiles = [decode_smiles(sample.numpy(), Constants.TOKENIZER_PATH) for sample in initial_x]
    
    # Denoising process
    for t in reversed(range(t_steps)):
        t_batch = tf.ones((num_samples,), dtype=tf.int32) * t
        predicted_probs = model.predict([x, t_batch])
        predicted_tokens = tf.argmax(predicted_probs, axis=-1)
        predicted_tokens = tf.cast(predicted_tokens, tf.float32)
        
        beta_t = tf.cast(betas[t], tf.float32)
        alpha_t = tf.cast(alphas[t], tf.float32)
        x = (x - beta_t * predicted_tokens) / tf.sqrt(alpha_t)
        x = tf.clip_by_value(x, 0, vocab_size - 1)
    
    # Get final SMILES
    final_x = tf.cast(x, tf.int32)
    final_smiles = [decode_smiles(sample.numpy(), Constants.TOKENIZER_PATH) for sample in final_x]
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(f"Generated SMILES Comparisons - {timestamp}\n")
        f.write("Initial SMILES -> Final SMILES\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (initial, final) in enumerate(zip(initial_smiles, final_smiles)):
            # Check if final molecule is valid
            mol = Chem.MolFromSmiles(final) if final else None
            validity = "[VALID]" if mol is not None else "[INVALID]"
            
            f.write(f"Sample {i+1}:\n")
            f.write(f"{initial} -> {final} {validity}\n")
            if mol is not None:
                f.write(f"MW: {Descriptors.ExactMolWt(mol):.2f}, ")
                f.write(f"LogP: {Descriptors.MolLogP(mol):.2f}\n")
            f.write("-" * 80 + "\n")
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_samples} molecules")
    
    print(f"\nComparisons have been saved to: {filename}")
    return initial_smiles, final_smiles, filename


    
    



# ==================== Training and Saving ====================
if __name__ == "__main__":
    # Load and prepare training data
    smiles_list = read_smiles_from_file(Constants.TRAINING_FILE)[:10]
    train_smiles, val_smiles = train_test_split(smiles_list, test_size=0.1)

    x_smiles, x_groups, y, vocab_size, max_length, smiles_vocab = make_diffusion_data(
        smiles_list, Constants.VOCAB_PATH, Constants.TOKENIZER_PATH
    )


    x_smiles_train, x_groups_train, y_train, vocab_size_train, max_length_train, smiles_vocab_train = make_diffusion_data(
        smiles_list, Constants.VOCAB_PATH, Constants.TOKENIZER_PATH
    )

    x_smiles_val, x_groups_val, y_val, vocab_size_val, max_length_val, smiles_vocab_val = make_diffusion_data(
        val_smiles, Constants.VOCAB_PATH, Constants.TOKENIZER_PATH
    )


    #model = build_diffusion_model(vocab_size=vocab_size, seq_length=max_length)
    x_smiles_train = tf.convert_to_tensor(x_smiles_train)
    x_groups_train = tf.convert_to_tensor(x_groups_train)
    y_train = tf.convert_to_tensor(y_train)

    x_smiles_val = tf.convert_to_tensor(x_smiles_val)
    x_groups_val = tf.convert_to_tensor(x_groups_val)
    y_val = tf.convert_to_tensor(y_val)

    scheduler = DiffusionScheduler(T=100, schedule_type='cosine')
    model = build_improved_diffusion_model(vocab_size=vocab_size, seq_length=max_length)
    
    # Initialize trainer and train
    trainer = DiffusionTrainer(
        model=model,
        scheduler=scheduler,
        training_data=x_smiles_train,
        validation_data=x_smiles_val,
        max_length=max_length,
        vocab_size=vocab_size
    )
    
    trainer.train(
        epochs=1,
        batch_size=16,
        num_samples_eval=2
    )

    save_dir = save_model_and_parameters(model, vocab_size, max_length)
    
    # Test loading and generation
    print("\nTesting loaded model...")
    model, params = load_model_and_generate(save_dir, num_samples=5)
     # Generate SMILES
    # generated_smiles, filename = sample_and_save_smiles(
    #     model, 
    #     t_steps=params['T'],
    #     max_length=params['max_length'],
    #     vocab_size=params['vocab_size'],
    #     num_samples=10
    # )

    initial_smiles, final_smiles, output_file = sample_and_save_smiles_with_initial(
        model, params['T'], params['max_length'], params['vocab_size'], num_samples=10
    )

    



   