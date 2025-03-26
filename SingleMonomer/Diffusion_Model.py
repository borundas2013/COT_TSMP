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
#from PPOTrainer import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Embedding, Lambda
from tensorflow.keras.models import Model

T = 1000  # Number of diffusion steps
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)



def forward_diffusion(x0, t):
    x0 = tf.cast(x0, tf.float32)
    noise = tf.random.normal(shape=x0.shape, dtype=tf.float32)

    sqrt_alpha_cumprod = tf.reshape(tf.cast(tf.sqrt(tf.gather(alphas_cumprod, t)), tf.float32), [-1, 1])
    sqrt_one_minus_alpha_cumprod = tf.reshape(tf.cast(tf.sqrt(tf.gather(1.0 - alphas_cumprod, t)), tf.float32), [-1, 1])

    xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    # Clip output values to valid token range
    xt = tf.clip_by_value(xt, 0, vocab_size - 1)
    return xt, noise





time_embed = Embedding(input_dim=T, output_dim=128)

def build_diffusion_model(vocab_size, seq_length, embedding_dim=128):
    input_seq=Input(shape=(seq_length,),name="input_seq")
    t_input=Input(shape=(1,),name="t_input")

    x=Embedding(input_dim=vocab_size,output_dim=embedding_dim)(input_seq)

    t_embed=time_embed(t_input)
    t_embed=Lambda(lambda x: tf.reshape(x, [-1, 1, embedding_dim]))(t_embed)
    
    x=x+t_embed

    x=LSTM(256,return_sequences=True)(x)
    x=LSTM(256,return_sequences=True)(x)

    out=Dense(vocab_size,activation="softmax")(x)

    model=Model(inputs=[input_seq,t_input],outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy")
    return model

def sample(model, t_steps, max_length):
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



    
    



# ==================== Training and Saving ====================
if __name__ == "__main__":
    # Load and prepare training data
    smiles_list = read_smiles_from_file(Constants.TRAINING_FILE)[:128]
    x_smiles, x_groups, y, vocab_size, max_length, smiles_vocab = make_diffusion_data(
        smiles_list, Constants.VOCAB_PATH, Constants.TOKENIZER_PATH
    )

    model = build_diffusion_model(vocab_size=vocab_size, seq_length=max_length)
    x_smiles = tf.convert_to_tensor(x_smiles)
    x_groups = tf.convert_to_tensor(x_groups)
    y = tf.convert_to_tensor(y)

    BATCH_SIZE = 32
    EPOCHS = 5

    for epoch in range(EPOCHS):
        for i in range(len(x_smiles) // BATCH_SIZE):
            x0 = x_smiles[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            t = tf.random.uniform(shape=(BATCH_SIZE,), minval=0, maxval=T, dtype=tf.int32)
            
            # Forward process
            xt, noise = forward_diffusion(x0, t)
            target = x0
            print(target.shape)

            # Replace invalid tokens with PAD token (0)
            target = tf.where(target < 0, tf.zeros_like(target), target)

            # Clip values to vocab range (0 to vocab_size - 1)
            target = tf.clip_by_value(target, 0, vocab_size - 1)

            # Train using target token IDs
            loss = model.train_on_batch([xt, t], target)

            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    generated_samples = sample(model, T, max_length)
    print(generated_samples.shape)
    print(generated_samples)
    smiles = decode_smiles(generated_samples.numpy()[0], Constants.TOKENIZER_PATH)
    print(smiles)



   