from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Reshape, Embedding, LSTM, GRU, Dense, Concatenate, TimeDistributed, Multiply, Add, Layer
import tensorflow as tf

class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

class TileLayer(Layer):
    def __init__(self, max_length, **kwargs):
        super(TileLayer, self).__init__(**kwargs)
        self.max_length = max_length

    def call(self, inputs):
        return tf.tile(inputs, [1, self.max_length, 1])

def build_model(max_length, vocab_size, embedding_dim, latent_dim,group_size):
    # Encoder for SMILES
    smiles_input = Input(shape=(max_length,), name="SMILES_Input")
    smiles_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name="SMILES_Embedding")(smiles_input)
    encoder_lstm = LSTM(latent_dim, return_state=True, name="Encoder_LSTM")
    _, state_h, state_c = encoder_lstm(smiles_embedding)

    # Group Input
    group_input = Input(shape=(group_size,), name="Group_Input")
    group_dense = Dense(latent_dim, activation="relu", name="Group_Dense")(group_input)

    # Combine Group Input with Encoder States
    state_h_cond = Concatenate(name="State_H_Cond")([state_h, group_dense])
    state_c_cond = Concatenate(name="State_C_Cond")([state_c, group_dense])

    # Decoder for SMILES
    decoder_input = Input(shape=(None,), name="Decoder_Input")
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name="Decoder_Embedding")(decoder_input)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, name="Decoder_LSTM")
    decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h_cond, state_c_cond])

    # Output Layer for Generated SMILES
    decoder_dense = TimeDistributed(Dense(vocab_size, activation="softmax"), name="Output_Layer")
    decoder_output = decoder_dense(decoder_output)

    # Full Model
    model = Model([smiles_input, group_input, decoder_input], decoder_output)
    #model.summary()
    return model


# def build_gru_model(max_length, vocab_size, embedding_dim, latent_dim, group_size):
#     # Encoder for SMILES
#     smiles_input = Input(shape=(max_length,), name="SMILES_Input")
#     smiles_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name="SMILES_Embedding")(smiles_input)
#     encoder_gru = GRU(latent_dim, return_state=True, name="Encoder_GRU")
#     _, state_h = encoder_gru(smiles_embedding)

#     # Group Input
#     group_input = Input(shape=(group_size,), name="Group_Input")
#     group_dense = Dense(latent_dim, activation="relu", name="Group_Dense")(group_input)

#     # Combine Group Input with Encoder State
#     state_h_cond = Concatenate(name="State_H_Cond")([state_h, group_dense])

#     # Decoder for SMILES
#     decoder_input = Input(shape=(None,), name="Decoder_Input")
#     decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name="Decoder_Embedding")(decoder_input)
#     decoder_gru = GRU(latent_dim * 2, return_sequences=True, return_state=True, name="Decoder_GRU")
#     decoder_output, _ = decoder_gru(decoder_embedding, initial_state=state_h_cond)

#     # Output Layer for Generated SMILES
#     decoder_dense = TimeDistributed(Dense(vocab_size, activation="softmax"), name="Output_Layer")
#     decoder_output = decoder_dense(decoder_output)

#     # Full Model
#     model = Model([smiles_input, group_input, decoder_input], decoder_output)
#     return model

def build_gru_model(max_length, vocab_size, embedding_dim=64, latent_dim=256, group_size=512):
    # SMILES input
    smiles_input = Input(shape=(max_length,), name='smiles_input')
    
    # Group input - ensure it's properly shaped
    group_input = Input(shape=(group_size,), name='group_input')
    # Use custom layers for expand_dims and tile
    group_input_reshaped = ExpandDimsLayer()(group_input)
    group_input_tiled = TileLayer(max_length)(group_input_reshaped)
    
    # Embedding layer for SMILES
    embedding = Embedding(vocab_size, embedding_dim)(smiles_input)
    
    # Concatenate embedded SMILES with group input
    merged = Concatenate(axis=-1)([embedding, group_input_tiled])
    
    # Rest of your model architecture
    encoder = GRU(latent_dim, return_sequences=True)(merged)
    
    # Decoder input
    decoder_input = Input(shape=(max_length,), name='decoder_input')
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_input)
    
    # Decoder GRU
    decoder_gru = GRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_embedding, initial_state=encoder[:, -1, :])
    
    # Output layer
    decoder_dense = Dense(vocab_size, activation='softmax')
    outputs = decoder_dense(decoder_outputs)
    
    # Create model
    model = Model([smiles_input, group_input, decoder_input], outputs)
    return model
def build_gru_model2(max_length, vocab_size, embedding_dim=128, latent_dim=512, group_size=8):
    # SMILES input
    smiles_input = Input(shape=(max_length,), name='smiles_input')
    
    # Group input - ensure it's properly shaped
    group_input = Input(shape=(group_size,), name='group_input')
    group_input_reshaped = ExpandDimsLayer()(group_input)
    group_input_tiled = TileLayer(max_length)(group_input_reshaped)
    
    # Embedding layer for SMILES
    embedding = Embedding(vocab_size, embedding_dim)(smiles_input)
    
    # Concatenate embedded SMILES with group input
    merged = Concatenate(axis=-1)([embedding, group_input_tiled])
    
    # Encoder GRU with increased latent dimension
    encoder = GRU(latent_dim, return_sequences=True, dropout=0.2)(merged)
    
    # Additional GRU layer for deeper learning
    encoder = GRU(latent_dim, return_sequences=True, dropout=0.2)(encoder)
    
    # Decoder input
    decoder_input = Input(shape=(max_length,), name='decoder_input')
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_input)
    
    # Decoder GRU
    decoder_gru = GRU(latent_dim, return_sequences=True, dropout=0.2)
    decoder_outputs = decoder_gru(decoder_embedding, initial_state=encoder[:, -1, :])
    
    # Output layer
    decoder_dense = Dense(vocab_size, activation='softmax')
    outputs = decoder_dense(decoder_outputs)
    
    # Create model
    model = Model([smiles_input, group_input, decoder_input], outputs)
    return model