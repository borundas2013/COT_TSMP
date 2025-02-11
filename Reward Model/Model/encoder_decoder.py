from tensorflow import keras
from tensorflow.keras import layers
from RelationalGraphConvLayer import RelationalGraphConvLayer


# def get_encoder(gconv_units, latent_dim, adjacency_shape, feature_shape, imine_shape, condition_shape,
#                 dense_units, dropout_rate):
#     print(adjacency_shape, feature_shape)
#     adjacency_0 = keras.layers.Input(shape=adjacency_shape)
#     features_0 = keras.layers.Input(shape=feature_shape)
#     adjacency_1 = keras.layers.Input(shape=adjacency_shape)
#     features_1 = keras.layers.Input(shape=feature_shape)
#     imine_feature = keras.layers.Input(shape=imine_shape)
#     conditional_features_1 = keras.layers.Input(shape=condition_shape)
#     conditional_features_2 = keras.layers.Input(shape=condition_shape)
#
#     features_transformed_0 = features_0
#     features_transformed_1 = features_1
#
#     for units in gconv_units:
#         features_transformed_0, features_transformed_1 = RelationalGraphConvLayer(units)(
#             [adjacency_0, features_transformed_0, adjacency_1, features_transformed_1, imine_feature
#                 , conditional_features_1, conditional_features_2])
#
#     # for units in dense_units:
#     #   y=x+y
#     #    x = layers.Dense(units, activation='relu')(x)
#     #   x = layers.Dropout(dropout_rate)(x)
#     #    y = layers.Dense(units, activation='relu')(y)
#     #   y = layers.Dropout(dropout_rate)(y)
#     x = layers.Dense(512, activation='relu')(features_transformed_0)
#     y = layers.Dense(512, activation='relu')(features_transformed_1)
#     x = keras.layers.LayerNormalization()(x)
#     y = keras.layers.LayerNormalization()(y)
#
#     x = x + keras.layers.Attention(use_scale=True)([x, x])
#     x = keras.layers.Attention(use_scale=True)([x, x])
#     y = y + keras.layers.Attention(use_scale=True)([y, y])
#     y = keras.layers.Attention(use_scale=True)([y, y])
#     x = keras.layers.GlobalAveragePooling1D()(x)
#     y = keras.layers.GlobalAveragePooling1D()(y)
#
#     # conditional_features_transposed_1 = tf.transpose(conditional_features_1, (0, 2,1))
#     # conditional_features_transposed_2 = tf.transpose(conditional_features_2, (0, 2,1))
#
#     z_mean = layers.Dense(latent_dim, dtype='float32', name='z_mean')(x)
#     log_var = layers.Dense(latent_dim, dtype='float32', name='log_var')(x)
#     z_mean_1 = layers.Dense(latent_dim, dtype='float32', name='z_mean_1')(y)
#     log_var_1 = layers.Dense(latent_dim, dtype='float32', name='log_var_1')(y)
#     encoder = keras.Model([adjacency_0, features_0, adjacency_1, features_1,
#                            imine_feature, conditional_features_1, conditional_features_2],
#                           [z_mean, log_var, z_mean_1, log_var_1, conditional_features_1, conditional_features_2],
#                           name='encoder')
#
#     return encoder
#
#
# def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape, condition_shape):
#     latent_inputs = keras.Input(shape=(latent_dim,))
#     latent_inputs_1 = keras.Input(shape=(latent_dim,))
#     conditional_features_1 = keras.Input(shape=condition_shape)
#     conditional_features_2 = keras.Input(shape=condition_shape)
#
#     x = latent_inputs
#     y = latent_inputs_1
#     cond1 = keras.layers.Flatten()(conditional_features_1)
#     cond1 = keras.layers.Dense(160, activation='relu')(cond1)
#
#     cond2 = keras.layers.Flatten()(conditional_features_2)
#     cond2 = keras.layers.Dense(160, activation='relu')(cond2)
#
#     x = keras.layers.Concatenate(axis=-1)([x, cond1])
#     y = keras.layers.Concatenate(axis=-1)([y, cond2])
#
#     x = layers.Dense(512, activation='relu')(x)
#     y = layers.Dense(512, activation='relu')(y)
#
#     x = keras.layers.LayerNormalization()(x)
#     y = keras.layers.LayerNormalization()(y)
#
#     x = keras.layers.Reshape((1, 512))(x)
#     y = keras.layers.Reshape((1, 512))(y)
#
#     x = x + keras.layers.Attention(use_scale=True)([x, x])
#     y = y + keras.layers.Attention(use_scale=True)([y, y])
#
#     # Using Reshape to remove the sequence length dimension (1) after Attention
#     x = keras.layers.Reshape((512,))(x)
#     y = keras.layers.Reshape((512,))(y)
#
#     x = layers.Dense(512, activation='relu')(x)
#     y = layers.Dense(512, activation='relu')(y)
#
#     x_0_adjacency = keras.layers.Dense(25600)(x)
#     x_0_adjacency = keras.layers.Reshape((160, 160))(x_0_adjacency)
#
#     # Replace the TensorFlow transpose operation with Keras Permute
#     x_0_adjacency_transposed = keras.layers.Permute((2, 1))(x_0_adjacency)
#     # _0_adjacency = (x_0_adjacency + tf.transpose(x_0_adjacency, (0, 2, 1)))
#     # print(x_0_adjacency.shape,x_0_adjacency_transposed.shape)
#
#     # Symmetrify tensors in the last two dimensions
#     x_0_adjacency = (x_0_adjacency + x_0_adjacency_transposed)
#
#     # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
#     x_0_features = keras.layers.Dense(1920)(x)
#     x_0_features = keras.layers.Reshape((160, 12))(x_0_features)
#     x_0_features = keras.layers.Softmax(axis=2)(x_0_features)
#
#     # Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
#     x_1_adjacency = keras.layers.Dense(25600)(y)
#     x_1_adjacency = keras.layers.Reshape((160, 160))(x_1_adjacency)
#
#     # Replace the TensorFlow transpose operation with Keras Permute
#     x_1_adjacency_transposed = keras.layers.Permute((2, 1))(x_1_adjacency)
#     # x_1_adjacency = (x_1_adjacency + tf.transpose(x_1_adjacency, (0, 2, 1)))
#
#     # Symmetrify tensors in the last two dimensions
#     x_1_adjacency = (x_1_adjacency + x_1_adjacency_transposed)
#
#     # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
#     x_1_features = keras.layers.Dense(1920)(y)
#     x_1_features = keras.layers.Reshape((160, 12))(x_1_features)
#     x_1_features = keras.layers.Softmax(axis=2)(x_1_features)
#
#     decoder = keras.Model(
#         [latent_inputs, latent_inputs_1, conditional_features_1, conditional_features_2],
#         outputs=[x_0_adjacency, x_0_features, x_1_adjacency, x_1_features],
#         name="decoder"
#     )
#
#     return decoder


def get_encoder(gconv_units, latent_dim, adjacency_shape, feature_shape, imine_shape, condition_shape,
                dense_units, dropout_rate):
    print(adjacency_shape, feature_shape)
    adjacency_0 = keras.layers.Input(shape=adjacency_shape)
    features_0 = keras.layers.Input(shape=feature_shape)
    adjacency_1 = keras.layers.Input(shape=adjacency_shape)
    features_1 = keras.layers.Input(shape=feature_shape)
    imine_feature = keras.layers.Input(shape=imine_shape)
    conditional_features_1 = keras.layers.Input(shape=condition_shape)
    conditional_features_2 = keras.layers.Input(shape=condition_shape)

    features_transformed_0 = features_0
    features_transformed_1 = features_1

    # Feature transformation with residual connections and additional dense layers
    for units in gconv_units:
        residual_0 = features_transformed_0
        residual_1 = features_transformed_1

        # Apply RelationalGraphConvLayer and transform features
        features_transformed_0, features_transformed_1 = RelationalGraphConvLayer(units)(
            [adjacency_0, features_transformed_0, adjacency_1, features_transformed_1, imine_feature,
             conditional_features_1, conditional_features_2])

        # Add Dense layer to make sure residual and transformed features have the same shape
        if features_transformed_0.shape[-1] != residual_0.shape[-1]:
            residual_0 = layers.Dense(features_transformed_0.shape[-1])(residual_0)  # Ensure shape match
        if features_transformed_1.shape[-1] != residual_1.shape[-1]:
            residual_1 = layers.Dense(features_transformed_1.shape[-1])(residual_1)  # Ensure shape match

        # Now safely add the residual connections
        features_transformed_0 += residual_0
        features_transformed_1 += residual_1

    # Adding attention layers and normalizing
    x = layers.Dense(512, activation='relu')(features_transformed_0)
    y = layers.Dense(512, activation='relu')(features_transformed_1)
    x = keras.layers.LayerNormalization()(x)
    y = keras.layers.LayerNormalization()(y)

    # Using multi-head attention
    x = x + keras.layers.MultiHeadAttention(num_heads=8, key_dim=512)(x, x)
    y = y + keras.layers.MultiHeadAttention(num_heads=8, key_dim=512)(y, y)

    # Global pooling
    x = keras.layers.GlobalAveragePooling1D()(x)
    y = keras.layers.GlobalAveragePooling1D()(y)

    # Latent space projection
    z_mean = layers.Dense(latent_dim, dtype='float32', name='z_mean')(x)
    log_var = layers.Dense(latent_dim, dtype='float32', name='log_var')(x)
    z_mean_1 = layers.Dense(latent_dim, dtype='float32', name='z_mean_1')(y)
    log_var_1 = layers.Dense(latent_dim, dtype='float32', name='log_var_1')(y)

    encoder = keras.Model([adjacency_0, features_0, adjacency_1, features_1,
                           imine_feature, conditional_features_1, conditional_features_2],
                          [z_mean, log_var, z_mean_1, log_var_1, conditional_features_1, conditional_features_2],
                          name='encoder')

    return encoder

def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape, condition_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))
    latent_inputs_1 = keras.Input(shape=(latent_dim,))
    conditional_features_1 = keras.Input(shape=condition_shape)
    conditional_features_2 = keras.Input(shape=condition_shape)

    x = latent_inputs
    y = latent_inputs_1

    # Processing the conditional features
    cond1 = keras.layers.Flatten()(conditional_features_1)
    cond1 = keras.layers.Dense(160, activation='relu')(cond1)
    cond2 = keras.layers.Flatten()(conditional_features_2)
    cond2 = keras.layers.Dense(160, activation='relu')(cond2)

    # Concatenate latent space and conditional features
    x = keras.layers.Concatenate(axis=-1)([x, cond1])
    y = keras.layers.Concatenate(axis=-1)([y, cond2])

    # Adding more dense layers to the decoder
    x = layers.Dense(1024, activation='relu')(x)
    y = layers.Dense(1024, activation='relu')(y)

    # Normalization
    x = keras.layers.LayerNormalization()(x)
    y = keras.layers.LayerNormalization()(y)

    # Reshaping and applying multi-head attention
    x = keras.layers.Reshape((1, 1024))(x)
    y = keras.layers.Reshape((1, 1024))(y)

    x = x + keras.layers.MultiHeadAttention(num_heads=8, key_dim=1024)(x, x)
    y = y + keras.layers.MultiHeadAttention(num_heads=8, key_dim=1024)(y, y)

    # Remove the sequence length dimension and reshape
    x = keras.layers.Reshape((1024,))(x)
    y = keras.layers.Reshape((1024,))(y)

    # Further dense layers
    x = layers.Dense(512, activation='relu')(x)
    y = layers.Dense(512, activation='relu')(y)

    # Mapping to adjacency and feature tensors
    x_0_adjacency = keras.layers.Dense(25600)(x)
    x_0_adjacency = keras.layers.Reshape((160, 160))(x_0_adjacency)
    x_0_adjacency_transposed = keras.layers.Permute((2, 1))(x_0_adjacency)
    x_0_adjacency = (x_0_adjacency + x_0_adjacency_transposed) / 2  # Symmetrize adjacency matrix

    x_0_features = keras.layers.Dense(1920)(x)
    x_0_features = keras.layers.Reshape((160, 12))(x_0_features)
    x_0_features = keras.layers.Softmax(axis=2)(x_0_features)

    x_1_adjacency = keras.layers.Dense(25600)(y)
    x_1_adjacency = keras.layers.Reshape((160, 160))(x_1_adjacency)
    x_1_adjacency_transposed = keras.layers.Permute((2, 1))(x_1_adjacency)
    x_1_adjacency = (x_1_adjacency + x_1_adjacency_transposed) / 2  # Symmetrize adjacency matrix

    x_1_features = keras.layers.Dense(1920)(y)
    x_1_features = keras.layers.Reshape((160, 12))(x_1_features)
    x_1_features = keras.layers.Softmax(axis=2)(x_1_features)

    decoder = keras.Model(
        [latent_inputs, latent_inputs_1, conditional_features_1, conditional_features_2],
        outputs=[x_0_adjacency, x_0_features, x_1_adjacency, x_1_features],
        name="decoder"
    )

    return decoder
# def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape, condition_shape):
#     latent_inputs = keras.Input(shape=(latent_dim,))
#     latent_inputs_1 = keras.Input(shape=(latent_dim,))
#     conditional_features_1 = keras.Input(shape=condition_shape)
#     conditional_features_2 = keras.Input(shape=condition_shape)
#
#     x = latent_inputs
#     y = latent_inputs_1
#
#     # Processing the conditional features
#     cond1 = keras.layers.Flatten()(conditional_features_1)
#     cond1 = keras.layers.Dense(160, activation='relu')(cond1)
#     cond2 = keras.layers.Flatten()(conditional_features_2)
#     cond2 = keras.layers.Dense(160, activation='relu')(cond2)
#
#     # Concatenate latent space and conditional features
#     x = keras.layers.Concatenate(axis=-1)([x, cond1])
#     y = keras.layers.Concatenate(axis=-1)([y, cond2])
#
#     # Adding more dense layers to the decoder
#     x = layers.Dense(1024, activation='relu')(x)
#     y = layers.Dense(1024, activation='relu')(y)
#
#     # Normalization
#     x = keras.layers.LayerNormalization()(x)
#     y = keras.layers.LayerNormalization()(y)
#
#     # Reshaping and applying multi-head attention
#     x = keras.layers.Reshape((1, 1024))(x)
#     y = keras.layers.Reshape((1, 1024))(y)
#
#     x = x + keras.layers.MultiHeadAttention(num_heads=8, key_dim=1024)(x, x)
#     y = y + keras.layers.MultiHeadAttention(num_heads=8, key_dim=1024)(y, y)
#
#     # Remove the sequence length dimension and reshape
#     x = keras.layers.Reshape((1024,))(x)
#     y = keras.layers.Reshape((1024,))(y)
#
#     # Further dense layers
#     x = layers.Dense(512, activation='relu')(x)
#     y = layers.Dense(512, activation='relu')(y)
#
#     # Mapping to adjacency and feature tensors
#     x_0_adjacency = keras.layers.Dense(25600)(x)
#     x_0_adjacency = keras.layers.Reshape((160, 160))(x_0_adjacency)
#
#     # Apply softmax to adjacency matrix to make it a probability distribution
#     x_0_adjacency = keras.layers.Softmax(axis=-1)(x_0_adjacency)
#
#     x_1_adjacency = keras.layers.Dense(25600)(y)
#     x_1_adjacency = keras.layers.Reshape((160, 160))(x_1_adjacency)
#
#     # Apply softmax to the second adjacency matrix as well
#     x_1_adjacency = keras.layers.Softmax(axis=-1)(x_1_adjacency)
#
#     # Apply softmax for categorical feature output
#     x_0_features = keras.layers.Dense(1920)(x)
#     x_0_features = keras.layers.Reshape((160, 12))(x_0_features)
#     x_0_features = keras.layers.Softmax(axis=2)(x_0_features)
#
#     # Repeat for the second feature tensors
#     x_1_features = keras.layers.Dense(1920)(y)
#     x_1_features = keras.layers.Reshape((160, 12))(x_1_features)
#     x_1_features = keras.layers.Softmax(axis=2)(x_1_features)
#
#     decoder = keras.Model(
#         [latent_inputs, latent_inputs_1, conditional_features_1, conditional_features_2],
#         outputs=[x_0_adjacency, x_0_features, x_1_adjacency, x_1_features],
#         name="decoder"
#     )
#
#     return decoder
