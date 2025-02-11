from tensorflow import keras
from tensorflow.keras import layers
from RelationalGraphConvLayer import RelationalGraphConvLayer

def get_encoder(gconv_units, latent_dim, node_feature_shape, edge_feature_shape, edge_index_shape, property_shape, condition_shape):
    # Input layers
    node_features_0 = keras.layers.Input(shape=node_feature_shape)
    edge_features_0 = keras.layers.Input(shape=edge_feature_shape)
    edge_index_0 = keras.layers.Input(shape=edge_index_shape)
    
    node_features_1 = keras.layers.Input(shape=node_feature_shape)
    edge_features_1 = keras.layers.Input(shape=edge_feature_shape)
    edge_index_1 = keras.layers.Input(shape=edge_index_shape)
    
    # conditional_features_1 = keras.layers.Input(shape=condition_shape)
    # conditional_features_2 = keras.layers.Input(shape=condition_shape)
    property_feature = keras.layers.Input(shape=property_shape)

    features_transformed_0 = node_features_0
    features_transformed_1 = node_features_1

    # Feature transformation with graph convolution
    for units in gconv_units:
        # Apply RelationalGraphConvLayer and transform features
        features_transformed_0, features_transformed_1 = RelationalGraphConvLayer(units)(
            [edge_index_0, features_transformed_0, edge_features_0,
             edge_index_1, features_transformed_1, edge_features_1,
             property_feature])

    # Adding attention layers and normalizing
    x = layers.Dense(512, activation='relu')(features_transformed_0)
    y = layers.Dense(512, activation='relu')(features_transformed_1)
    x = keras.layers.LayerNormalization()(x)
    y = keras.layers.LayerNormalization()(y)

    x = x + keras.layers.Attention(use_scale=True)([x, x])
    y = y + keras.layers.Attention(use_scale=True)([y, y])

    # Global pooling
    x = keras.layers.GlobalAveragePooling1D()(x)
    y = keras.layers.GlobalAveragePooling1D()(y)

    # Latent space projection
    z_mean = layers.Dense(latent_dim, dtype='float32', name='z_mean')(x)
    log_var = layers.Dense(latent_dim, dtype='float32', name='log_var')(x)
    z_mean_1 = layers.Dense(latent_dim, dtype='float32', name='z_mean_1')(y)
    log_var_1 = layers.Dense(latent_dim, dtype='float32', name='log_var_1')(y)

    encoder = keras.Model(
        inputs=[
            node_features_0, edge_features_0, edge_index_0,
            node_features_1, edge_features_1, edge_index_1,
            property_feature
        ],
        outputs=[z_mean, log_var, z_mean_1, log_var_1],
        name='encoder'
    )

    return encoder

def get_decoder(latent_dim, node_feature_dim, edge_feature_dim, max_nodes, max_edges, condition_shape, property_shape):
    # Input layers
    latent_inputs = keras.Input(shape=(latent_dim,))
    latent_inputs_1 = keras.Input(shape=(latent_dim,))
    # conditional_features_1 = keras.Input(shape=condition_shape)
    # conditional_features_2 = keras.Input(shape=condition_shape)
    property_feature = keras.layers.Input(shape=property_shape)

    x = latent_inputs
    y = latent_inputs_1

    # Process conditional features
    # cond1 = keras.layers.Flatten()(conditional_features_1)
    # cond1 = keras.layers.Dense(160, activation='relu')(cond1)
    
    # cond2 = keras.layers.Flatten()(conditional_features_2)
    # cond2 = keras.layers.Dense(160, activation='relu')(cond2)
    
    # Process property feature
    prop_feature = keras.layers.Dense(512, activation='relu')(property_feature)
    
    # Concatenate features
    # x = keras.layers.Concatenate(axis=-1)([x, cond1, prop_feature])
    # y = keras.layers.Concatenate(axis=-1)([y, cond2, prop_feature])
    x = keras.layers.Concatenate(axis=-1)([x, prop_feature])
    y = keras.layers.Concatenate(axis=-1)([y, prop_feature])

    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    y = layers.Dense(512, activation='relu')(y)
    
    x = keras.layers.LayerNormalization()(x)
    y = keras.layers.LayerNormalization()(y)   

    # Attention mechanism
    x = keras.layers.Reshape((1, 512))(x)
    y = keras.layers.Reshape((1, 512))(y)

    x = x + keras.layers.Attention(use_scale=True)([x, x])
    y = y + keras.layers.Attention(use_scale=True)([y, y])

    x = layers.Flatten()(x)
    y = layers.Flatten()(y)

    x = layers.Dense(512, activation='relu')(x)
    y = layers.Dense(512, activation='relu')(y)

    # Generate node features
    node_features_0 = keras.layers.Dense(max_nodes * node_feature_dim)(x)
    node_features_0 = keras.layers.Reshape((max_nodes, node_feature_dim))(node_features_0)
    node_features_0 = keras.layers.Activation('sigmoid')(node_features_0)

    node_features_1 = keras.layers.Dense(max_nodes * node_feature_dim)(y)
    node_features_1 = keras.layers.Reshape((max_nodes, node_feature_dim))(node_features_1)
    node_features_1 = keras.layers.Activation('sigmoid')(node_features_1)

    # Generate edge features
    edge_features_0 = keras.layers.Dense(max_edges * edge_feature_dim)(x)
    edge_features_0 = keras.layers.Reshape((max_edges, edge_feature_dim))(edge_features_0)
    edge_features_0 = keras.layers.Activation('sigmoid')(edge_features_0)

    edge_features_1 = keras.layers.Dense(max_edges * edge_feature_dim)(y)
    edge_features_1 = keras.layers.Reshape((max_edges, edge_feature_dim))(edge_features_1)
    edge_features_1 = keras.layers.Activation('sigmoid')(edge_features_1)

    # Generate edge indices
    edge_index_0 = keras.layers.Dense(2 * max_edges)(x)
    edge_index_0 = keras.layers.Reshape((2, max_edges))(edge_index_0)
    
    edge_index_1 = keras.layers.Dense(2 * max_edges)(y)
    edge_index_1 = keras.layers.Reshape((2, max_edges))(edge_index_1)

    decoder = keras.Model(
        inputs=[latent_inputs, latent_inputs_1, property_feature],
        outputs=[
            node_features_0, edge_features_0, edge_index_0,
            node_features_1, edge_features_1, edge_index_1
        ],
        name="decoder"
    )

    return decoder

