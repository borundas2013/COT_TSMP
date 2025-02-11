from Features import convert_smiles_to_graph,prepare_condition, convert_graph_to_mol
import Constants
from Sampling import Sampling
import numpy as np
import tensorflow as tf

# def grid_search_latent_space(z0, grid_size=5, step_size=0.1):
#     grid = []
#     for dx in np.linspace(-step_size, step_size, grid_size):
#         for dy in np.linspace(-step_size, step_size, grid_size):
#             new_z = z0 + np.array([dx, dy])
#             grid.append(new_z)
#     return grid
# def generate_new_samples( sv_model,batch_size, sample):
#     # np.random.seed(121)
#     # tf.random.set_seed(42)
#     ad_0, feat_0, ad_1, feat_1 = convert_smiles_to_graph(sample)
#
#     all_mols = []
#     imine_list = []
#
#     # temp1 = round(np.random.uniform(0, 4),3)
#     # feat_0 = feat_0 + temp1
#     # feat_1 = feat_1 + temp1
#     random_imine_value = np.random.uniform(0, 2)  # random.randint(0, 1)
#     array_160_1 = np.zeros((Constants.NUM_ATOMS, 1))
#     imine_list.append(array_160_1 + random_imine_value)
#
#     cond1, cond2 = prepare_condition(sample)
#     print(random_imine_value)
#     z_mean, log_var, z_mean_1, log_var_1, cond1, cond2 = sv_model.encoder.predict([np.array([ad_0]), np.array([feat_0]),
#                                                                         np.array([ad_1]), np.array([feat_1]),
#                                                                         np.array(imine_list),
#                                                                         np.array([cond1]), np.array([cond2])])
#     z1, z2 = Sampling()([z_mean, log_var, z_mean_1, log_var_1])
#
#     for i in range(batch_size):
#         # z = tf.random.normal((1, Constants.LATENT_DIM))
#         # z_p = tf.random.normal((1, Constants.LATENT_DIM))
#         # # z = np.random.normal(min_val,max_val,(1, LATENT_DIM))
#         # # z_p = np.random.normal(min_val_1,max_val_1,(1, LATENT_DIM))
#         # z3 = z1 + z  # np.multiply(z1, z) #
#         # z4 = z2 + z_p  # np.multiply(z1, z_p) #
#         z3 = grid_search_latent_space(z1, grid_size=5, step_size=0.05)
#         z4 = grid_search_latent_space(z2, grid_size=5, step_size=0.05)
#         reconstruction_adjacnency_0, recontstruction_features_0, reconstruction_adjacnency_1, recontstruction_features_1 = sv_model.decoder.predict(
#             [z3, z4, cond1, cond2])
#
#         adjacency_0 = tf.linalg.set_diag(reconstruction_adjacnency_0,
#                                          tf.zeros(tf.shape(reconstruction_adjacnency_0)[:-1]))
#         adjacency_0 = abs(reconstruction_adjacnency_0[0].astype(int))
#         features_0 = tf.argmax(recontstruction_features_0, axis=2)
#         features_0 = tf.one_hot(features_0, depth=Constants.ATOM_DIM, axis=2)
#
#         adjacency_1 = tf.linalg.set_diag(reconstruction_adjacnency_1,
#                                          tf.zeros(tf.shape(reconstruction_adjacnency_1)[:-1]))
#         features_1 = tf.argmax(recontstruction_features_1, axis=2)
#         features_1 = tf.one_hot(features_1, depth=Constants.ATOM_DIM, axis=2)
#         adjacency_1 = abs(reconstruction_adjacnency_1[0].astype(int))
#         # graph2=[[adjacency_0[0].numpy()],[features_0[0].numpy()],[adjacency_1[0].numpy()],[features_1[0].numpy()]]
#         graph2 = [[adjacency_0], [features_0[0].numpy()],
#                   [adjacency_1], [features_0[0].numpy()]]
#         all_mols.append(convert_graph_to_mol(graph2))
#     return all_mols

def grid_search_latent_space(z0, grid_size=5, step_size=0.1):
    """
    Generate a grid of latent vectors around a central point z0.
    Perturb each dimension of the latent space by step_size.
    """
    print(z0.shape)
    z0_shape = z0.shape[-1]  # Get the dimensionality of z0
    grid = []

    # Generate perturbations for each dimension
    for i in range(grid_size):
        for j in range(grid_size):
            # Create perturbations matching the shape of z0
            perturbation = np.random.normal(0, step_size, z0_shape)
            new_z = z0 + perturbation  # Add perturbation to z0
            grid.append(new_z)
    print(np.array(grid).shape)
    return np.array(grid)

def generate_new_samples(sv_model, batch_size, sample):
    ad_0, feat_0, ad_1, feat_1 = convert_smiles_to_graph(sample)
    all_mols = []
    imine_list = []

    random_imine_value = np.random.uniform(0, 2)
    array_160_1 = np.zeros((Constants.NUM_ATOMS, 1))
    imine_list.append(array_160_1 + random_imine_value)

    cond1, cond2 = prepare_condition(sample)

    # Encode the input graph to get the latent representation
    z_mean, log_var, z_mean_1, log_var_1, cond1, cond2 = sv_model.encoder.predict(
        [np.array([ad_0]), np.array([feat_0]), np.array([ad_1]), np.array([feat_1]),
         np.array(imine_list), np.array([cond1]), np.array([cond2])]
    )

    # Perform sampling from latent space
    z1, z2 = Sampling()([z_mean, log_var, z_mean_1, log_var_1])

    # Generate latent vectors using grid search
    z3_grid = grid_search_latent_space(z1, grid_size=5, step_size=0.05)
    z4_grid = grid_search_latent_space(z2, grid_size=5, step_size=0.05)

    # Squeeze cond1 and cond2 to ensure they have the correct shape
    cond1 = np.squeeze(cond1)  # Make sure shape is (1, 160, 160)
    cond2 = np.squeeze(cond2)  # Make sure shape is (1, 160, 160)

    # Generate molecules for each point in the grid
    for z3, z4 in zip(z3_grid, z4_grid):
        # Reshape latent vectors to match the expected input shape (1, 160)
        z3 = np.reshape(z3, (1, -1))  # Reshape to (1, 160)
        z4 = np.reshape(z4, (1, -1))  # Reshape to (1, 160)

        # Ensure cond1 and cond2 are passed with the correct shape
        reconstruction_adjacency_0, recontstruction_features_0, reconstruction_adjacency_1, recontstruction_features_1 = sv_model.decoder.predict(
            [z3, z4, np.array([cond1]), np.array([cond2])]
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
