

import ast

import pandas as pd
import numpy as np

import tensorflow as tf


import Constants
from Features import get_condtions,get_imine_feature_list, prepare_graph_features,data_load,prepare_condition,convert_graph_to_mol_2
from rdkit import Chem
import random
from encoder_decoder import get_encoder,get_decoder
from MoleculeGenerator import MoleculeGenerator,train_model_with_reward
from Sample_Generator import generate_new_samples
tf.config.run_functions_eagerly(True)

_,df_smiles=data_load(Constants.DATA_FILE_PATH)
adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor = prepare_graph_features(df_smiles)
print(adjacency_0_tensor.shape,feature_0_tensor.shape)
print('-------------')
condition_1_array, condition_2_array = get_condtions(df_smiles)
imine_features_array = get_imine_feature_list(df_smiles)


### Usage Example:
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


vae_optimizer = tf.keras.optimizers.Adam(learning_rate=Constants.VAE_LR)

model = MoleculeGenerator(encoder, decoder, 160)
model.compile(vae_optimizer)

data=[adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor,
                     imine_features_array, condition_1_array, condition_2_array]
dataset = tf.data.Dataset.from_tensor_slices((adjacency_0_tensor, feature_0_tensor, adjacency_1_tensor, feature_1_tensor,
                     imine_features_array, condition_1_array, condition_2_array)).batch(25)
# Train the model with reward-based optimization
train_model_with_reward(model, dataset, epochs=Constants.EPOCHS, lambda_weight=0.9, save_path="saved/best_vae_model")

sv_model= MoleculeGenerator(encoder, decoder, 160)
# Generate new SMILES using the best model
sv_model.load_weights('saved/best_vae_model')
def write_samples_in_file():
    mols = []
    i=0
    with open('test_reward_2.txt', 'a') as the_file:
        # random.seed(121)
        smiles = random.choices(df_smiles, k=10)#df_smiles#
        for sample in smiles:
            i = i+1
            text = '\n-----------------'+str(i)+'---------------------------------\n'
            mols = generate_new_samples(sv_model,5, sample)
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