import Constants
from Sampling import Sampling
import numpy as np
import tensorflow as tf
from feature_extraction import features_to_smiles



def generate_new_samples(model, batch_size, sample_features, sampling_temperature=1.0):
    """Generate new molecule samples from input features.
    
    Args:
        model: The trained VAE model
        batch_size: Number of samples to generate
        sample_features: Dictionary containing input molecular features
        sampling_temperature: Temperature parameter for sampling (default=1.0)
    
    Returns:
        List of generated molecule pairs
    """
    all_mols = []

    # Get input features
    node_features_0 = sample_features['node_features_0'] 
    edge_features_0 = sample_features['edge_features_0']
    edge_index_0 = sample_features['edge_index_0']
    node_features_1 = sample_features['node_features_1']
    edge_features_1 = sample_features['edge_features_1'] 
    edge_index_1 = sample_features['edge_index_1']
    property_matrix = sample_features['property_matrix']

    # Encode input features to get latent representation
    z_mean, log_var, z_mean_1, log_var_1 = model.encoder.predict([
        np.array([node_features_0]), 
        np.array([edge_features_0]),
        np.array([edge_index_0]),
        np.array([node_features_1]),
        np.array([edge_features_1]), 
        np.array([edge_index_1]),
        np.array([property_matrix])
    ])

    # Sample from latent space
    z1, z2 = Sampling(temperature=sampling_temperature)([z_mean, log_var, z_mean_1, log_var_1])

    # Generate molecules
    for i in range(batch_size):
        # Add random noise to latent vectors
        z3 = tf.random.normal((1, Constants.LATENT_DIM))
        z4 = tf.random.normal((1, Constants.LATENT_DIM))
        new_z1 = z1 + z3
        new_z2 = z2 + z4

        # Generate new property values
        new_property = np.random.uniform(size=(1, Constants.NO_PROPERTY))

        # Decode to get molecule features
        gen_node_0, gen_edge_0, gen_edge_idx_0, \
        gen_node_1, gen_edge_1, gen_edge_idx_1 = model.decoder.predict([
            new_z1, new_z2, new_property
        ])

        # Process generated features
        gen_node_0 = tf.argmax(gen_node_0, axis=2)
        gen_node_0 = tf.one_hot(gen_node_0, depth=Constants.ATOM_DIM)
        gen_node_1 = tf.argmax(gen_node_1, axis=2)
        gen_node_1 = tf.one_hot(gen_node_1, depth=Constants.ATOM_DIM)

        # Construct graph for molecule conversion
        graph = [
            [gen_node_0[0].numpy()], [gen_edge_0[0]], [gen_edge_idx_0[0]],
            [gen_node_1[0].numpy()], [gen_edge_1[0]], [gen_edge_idx_1[0]]
        ]
        
        print(graph[0][0].shape,graph[1][0].shape,graph[2][0].shape)
        print(graph[3][0])
        print(graph[4][0])
        print(graph[5][0])
        mol_pair = [features_to_smiles(
            graph[0][0], graph[1][0], graph[2][0]),
            features_to_smiles(
            graph[3][0], graph[4][0], graph[5][0])]
        all_mols.append([mol_pair])
        print(mol_pair)

    return all_mols
