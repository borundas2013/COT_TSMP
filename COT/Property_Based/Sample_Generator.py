


from Features import convert_smiles_to_graph,prepare_condition, convert_graph_to_mol
import Constants
from Sampling import Sampling
import numpy as np
import tensorflow as tf


def generate_new_samples(sv_model, batch_size, sample, sampling_temperature=1.0):
    ad_0, feat_0, ad_1, feat_1 = convert_smiles_to_graph(sample)
    all_mols = []

 
    
    cond1, cond2 = prepare_condition(sample)
    property_matrix = np.random.randint(100, 301, size=2)
    print(property_matrix)

    # Encode the input graph to get the latent representation
    z_mean, log_var, z_mean_1, log_var_1 = sv_model.encoder.predict(
        [np.array([ad_0]), np.array([feat_0]), np.array([ad_1]), np.array([feat_1]),
         np.array([property_matrix])]
    )

    # Perform sampling from latent space with temperature control
    z1, z2 = Sampling(temperature=sampling_temperature)([z_mean, log_var, z_mean_1, log_var_1])

   

    # Squeeze cond1 and cond2 to ensure they have the correct shape
    cond1 = np.squeeze(cond1)
    cond2 = np.squeeze(cond2)

    # Generate molecules for each point in the grid
    for i in range(batch_size):
        z3 = tf.random.normal((1, Constants.LATENT_DIM))
        z4 = tf.random.normal((1, Constants.LATENT_DIM))
        property_matrix = property_matrix #np.random.randint(100, 301, size=2)
        
        # Add z3 to z1 and z4 to z2
        new_z1 = z1 + z3
        new_z2 = z2 + z3

        reconstruction_adjacency_0, recontstruction_features_0, reconstruction_adjacency_1, recontstruction_features_1 = sv_model.decoder.predict(
            [new_z1, new_z2, np.array([property_matrix])]
        )

        # Process the decoded adjacency and feature matrices
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
        all_mols.append(convert_graph_to_mol(graph2))

    return all_mols

