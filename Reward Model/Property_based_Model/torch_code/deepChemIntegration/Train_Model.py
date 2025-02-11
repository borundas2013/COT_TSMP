import tensorflow as tf
import numpy as np
import Constants
from rdkit import Chem
from encoder_decoder import get_encoder,get_decoder
from MoleculeGenerator import MoleculeGenerator,train_model_with_reward
from Sample_Generator import generate_new_samples
from feature_extraction import extract_and_pad_features,get_property_matrix,features_to_smiles
import random


tf.config.run_functions_eagerly(True)


# Load and prepare data
smiles_file = 'unique_smiles_Er.csv'
features = extract_and_pad_features(smiles_file)
padded_nodes_1, padded_edges_1, edge_index_1, \
padded_nodes_2, padded_edges_2, edge_index_2 = features

# Get property matrix
property_matrix = get_property_matrix(smiles_file)

#imine_features_array = get_imine_feature_list(df_smiles)


# ### Usage Example:
encoder = get_encoder(
    gconv_units=[9],
    latent_dim=Constants.LATENT_DIM,
    node_feature_shape=(Constants.NUM_ATOMS, Constants.ATOM_DIM),
    edge_feature_shape=(Constants.MAX_EDGES, Constants.EDGE_DIM),
    edge_index_shape=(2, Constants.MAX_EDGES),
    property_shape=(Constants.NO_PROPERTY,),
    condition_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS)
)

# Initialize decoder
decoder = get_decoder(
    latent_dim=Constants.LATENT_DIM,
    node_feature_dim=Constants.ATOM_DIM,
    edge_feature_dim=Constants.EDGE_DIM,
    max_nodes=Constants.NUM_ATOMS,
    max_edges=Constants.MAX_EDGES,
    condition_shape=(Constants.NUM_ATOMS, Constants.NUM_ATOMS),
    property_shape=(Constants.NO_PROPERTY,)
)

# Initialize optimizer and model
vae_optimizer = tf.keras.optimizers.Adam(learning_rate=Constants.VAE_LR)
model = MoleculeGenerator(encoder, decoder, Constants.NUM_ATOMS)
model.compile(vae_optimizer)


print("Shapes:")
print("padded_nodes_1:", padded_nodes_1.shape)
print("padded_edges_1:", padded_edges_1.shape)
print("edge_index_1:", edge_index_1.shape)
print("padded_nodes_2:", padded_nodes_2.shape)
print("padded_edges_2:", padded_edges_2.shape)
print("edge_index_2:", edge_index_2.shape)
print("property_matrix:", property_matrix.shape)
# Create dataset
dataset = tf.data.Dataset.from_tensor_slices({
    'node_features_0': padded_nodes_1,
    'edge_features_0': padded_edges_1,
    'edge_index_0': edge_index_1,
    'node_features_1': padded_nodes_2,
    'edge_features_1': padded_edges_2,
    'edge_index_1': edge_index_2,
    'property_matrix': property_matrix
}).batch(25)

# Train the model
train_model_with_reward(
    model, 
    dataset, 
    epochs=Constants.EPOCHS, 
    lambda_weight=0.5, 
    save_path="best_vae_model.weights.h5"
)

# # Load saved model for generation
saved_model = MoleculeGenerator(encoder, decoder, Constants.NUM_ATOMS)
saved_model.load_weights('best_vae_model.weights.h5')

def write_samples_in_file():
    """Generate and write samples to file."""
    with open('test_reward_2.txt', 'a') as the_file:
        # Get random samples from the dataset
        sample_indices = random.sample(range(len(padded_nodes_1)),1)
        print(sample_indices)
        
        for i, idx in enumerate(sample_indices, 1):
            # Get features for the selected sample
            idx=0
            sample_features = {
                'node_features_0': padded_nodes_1[idx],
                'edge_features_0': padded_edges_1[idx],
                'edge_index_0': edge_index_1[idx],
                'node_features_1': padded_nodes_2[idx],
                'edge_features_1': padded_edges_2[idx],
                'edge_index_1': edge_index_2[idx],
                'property_matrix': property_matrix[idx]
            }
            
            # Generate new molecules
            sm1=features_to_smiles(sample_features['node_features_0'],sample_features['edge_features_0'],sample_features['edge_index_0'])
            sm2=features_to_smiles(sample_features['node_features_1'],sample_features['edge_features_1'],sample_features['edge_index_1'])
            print(sample_features['node_features_0'].shape,sample_features['edge_features_0'].shape,sample_features['edge_index_0'].shape)
            print(sm1,sm2)  

            mols = generate_new_samples(model, 2, sample_features)
            
            # # Write to file
            # the_file.write(f'\n-----------------{i}---------------------------------\n')
            # for mol_pair in mols:
            #     if mol_pair[0][0] is not None and mol_pair[0][1] is not None:
            #         smiles_1 = Chem.MolToSmiles(mol_pair[0][0])
            #         smiles_2 = Chem.MolToSmiles(mol_pair[0][1])
            #         the_file.write(f"{smiles_1},{smiles_2}\n")
            #     else:
            #         the_file.write("None,None\n")
            # the_file.write('--------------------------------------------------\n')

# Generate samples
write_samples_in_file()