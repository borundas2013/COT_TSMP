from Property_Based.MoleculeGenerator import MoleculeGenerator
from Property_Based.encoder_decoder import get_decoder,get_encoder
from Property_Based.Sampling import Sampling
import Property_Based.Constants as Constants
from Property_Based.Features import convert_graph_to_mol,convert_smiles_to_graph,data_load
import tensorflow as tf
import numpy as np
from rdkit import Chem
import random
import warnings
encoder = get_encoder(
    gconv_units=[9],
    latent_dim=Constants.LATENT_DIM,
    node_feature_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS),
    edge_feature_shape=(Constants.NUM_ATOMS, Constants.ATOM_DIM),
    property_shape=(Constants.NO_PROPERTY,),
    condition_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS)
)

# Initialize decoder
decoder = get_decoder(
    latent_dim=Constants.LATENT_DIM,
    node_feature_dim=Constants.NUM_ATOMS,
    edge_feature_dim=Constants.ATOM_DIM,
    condition_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS),
    property_shape=(Constants.NO_PROPERTY,)
)

molecule_generator = MoleculeGenerator(encoder, decoder, Constants.NUM_ATOMS)  
molecule_generator.load_weights(Constants.MODEL_PATH)

def get_smiles_list():
    _,df_smiles = data_load('Property_based/unique_smiles_er.csv')
    smiles_list = df_smiles['Smiles'].tolist()
    smiles_list = random.sample(smiles_list, 5)
    return smiles_list

def generate_molecules_property_based(glass_transition_temperature, stress_recovery):
    

    property_matrix = np.array([glass_transition_temperature, stress_recovery])

    smiles_list = call_molecule_generator( property_matrix)
    return smiles_list


def call_molecule_generator(property_matrix):
    smiles_list = []
    random_smiles_list=get_smiles_list()
    for s in random_smiles_list:
        row_smiles=s[1:-1]  
        row_smiles=row_smiles.replace("'","")
        sample=row_smiles.split(',')
        ad_0, feat_0, ad_1, feat_1 = convert_smiles_to_graph(sample)
        z_mean, log_var, z_mean_1, log_var_1 = molecule_generator.encoder.predict(
        [np.array([ad_0]), np.array([feat_0]), np.array([ad_1]), np.array([feat_1]),
         np.array([property_matrix])]
    )
        z1, z2 = Sampling(temperature=1.0)([z_mean, log_var, z_mean_1, log_var_1])
        for i in range(5):
            z = tf.random.normal((1, Constants.LATENT_DIM))
            z_p = tf.random.normal((1, Constants.LATENT_DIM))

            new_z1 = z1 + z
            new_z2 = z2 + z_p

            reconstruction_adjacency_0, recontstruction_features_0, reconstruction_adjacency_1, recontstruction_features_1 = molecule_generator.decoder.predict(
            [new_z1, new_z2, np.array([property_matrix])]
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
                if mol[0]is not None:  
                    s1=Chem.MolToSmiles(mol[0])
                    s1=s1.replace(".","")
                    smiles_list.append(s1)
                if mol[1]is not None:
                    s2=Chem.MolToSmiles(mol[1])
                    s2=s2.replace(".","")
                    smiles_list.append(s2)
           
    smiles_list=np.unique(smiles_list)
    return smiles_list[:2]










