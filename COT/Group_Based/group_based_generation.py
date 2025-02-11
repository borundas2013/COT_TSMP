from Group_Based.MoleculeGenerator import MoleculeGenerator
from Group_Based.encoder_decoder import get_decoder,get_encoder
import Group_Based.Constants as Constants
from Group_Based.Features import convert_graph_to_mol,data_load,convert_smiles_to_graph
from Property_Based.Sampling import Sampling
import tensorflow as tf
import random
import numpy as np
from rdkit import Chem
import os
import sys
from contextlib import contextmanager
import numpy as np

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
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

molecule_generator = MoleculeGenerator(encoder, decoder, Constants.NUM_ATOMS)  
try:
    molecule_generator.load_weights(Constants.MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model weights file not found at {Constants.MODEL_PATH}")
    print("Please ensure the weights file exists in the correct location.")
    exit(1)


def generate_molecules_group_based (group):
   

    cond_1 = np.zeros((Constants.NUM_ATOMS, Constants.NUM_ATOMS), 'float32')
    cond_2 = np.zeros((Constants.NUM_ATOMS, Constants.NUM_ATOMS), 'float32')

    if group == 'epoxy':
        # Repeat the epoxy pattern 3 times
        for _ in range(3):

            idx1 = np.random.randint(0, Constants.NUM_ATOMS-2)
            idx2 = idx1 + 1 
            idx3 = idx1 + 2       
            # Set 1s for the three adjacent positions
            cond_1[idx1, idx2] = cond_1[idx2, idx1] = Constants.EPOXY_GROUP
            cond_1[idx2, idx3] = cond_1[idx3, idx2] = Constants.EPOXY_GROUP
            cond_1[idx1, idx3] = cond_1[idx3, idx1] = Constants.EPOXY_GROUP

            cond_2[idx1, idx2] = cond_2[idx2, idx1] = Constants.IMINE_GROUP
    smiles_list = call_molecule_generator( cond_1, cond_2)
    return smiles_list


def get_smiles_list():
    _,df_smiles = data_load('Group_Based/unique_smiles_er.csv')
    smiles_list = df_smiles['Smiles'].tolist()
    print(len(smiles_list))
    smiles_list = random.sample(smiles_list, 10)
    return smiles_list


def call_molecule_generator(cond_1, cond_2):
    smiles_list=[]
   
    random_smiles_list=get_smiles_list()
   
    for s in random_smiles_list:
        row_smiles=s[1:-1]  
        row_smiles=row_smiles.replace("'","")
        sample=row_smiles.split(',')
        ad_0, feat_0, ad_1, feat_1 = convert_smiles_to_graph(sample)
        all_mols = []
        imine_list = []

        random_imine_value = np.random.uniform(0, 2)
        array_160_1 = np.zeros((Constants.NUM_ATOMS, 1))
        imine_list.append(array_160_1 + random_imine_value)

        # Encode the input graph to get the latent representation
        z_mean, log_var, z_mean_1, log_var_1, cond1, cond2 = molecule_generator.encoder.predict(
            [np.array([ad_0]), np.array([feat_0]), np.array([ad_1]), np.array([feat_1]),
            np.array(imine_list), np.array([cond_1]), np.array([cond_2])]
    )

    # Perform sampling from latent space
        z1, z2 = Sampling()([z_mean, log_var, z_mean_1, log_var_1])

        for i in range(5):
            z = tf.random.normal((1, Constants.LATENT_DIM))
            z_p = tf.random.normal((1, Constants.LATENT_DIM))

            new_z1 = z1 + z
            new_z2 = z2 + z_p

            reconstruction_adjacency_0, recontstruction_features_0, reconstruction_adjacency_1, recontstruction_features_1 = molecule_generator.decoder.predict(
            [new_z1, new_z2, np.array([cond_1]), np.array([cond_2])]
            )
            adjacency_0 = tf.linalg.set_diag(reconstruction_adjacency_0, tf.zeros(tf.shape(reconstruction_adjacency_0)[:-1]))
            adjacency_0 = abs(reconstruction_adjacency_0[0].astype(int))
            features_0 = tf.argmax(recontstruction_features_0, axis=2)
            features_0 = tf.one_hot(features_0, depth=Constants.ATOM_DIM, axis=2)

            adjacency_1 = tf.linalg.set_diag(reconstruction_adjacency_1, tf.zeros(tf.shape(reconstruction_adjacency_1)[:-1]))
            adjacency_1 = abs(reconstruction_adjacency_1[0].astype(int))
            features_1 = tf.argmax(recontstruction_features_1, axis=2)
            features_1 = tf.one_hot(features_1, depth=Constants.ATOM_DIM, axis=2)

        # Construct the graph for conversion to molecule
            graph2 = [[adjacency_0], [features_0[0].numpy()], [adjacency_1], [features_1[0].numpy()]]
            mols=convert_graph_to_mol(graph2)
            for mol in mols:
                if mol[0]is not None and mol[1]is not None:
                    s1=Chem.MolToSmiles(mol[0])
                    s2=Chem.MolToSmiles(mol[1])
                    s1=s1.replace(".","")   
                    s2=s2.replace(".","")
                    smiles_list.append([s1,s2])
                

    
   
    return random.sample(smiles_list, 1)

if __name__ == "__main__":

    generate_molecules_group_based( "epoxy")








