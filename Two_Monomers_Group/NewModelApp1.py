from LoadPreTrainedModel import load_and_retrain
import os
from pathlib import Path
from dual_smile_process import prepare_training_data
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,  Dense, Add, Dropout
import Constants
from Data_Process_with_prevocab import *
import numpy as np  
import random
import heapq

def get_project_root():
    """Get the path to the project root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent

root_dir = get_project_root()
    

def create_model(max_length, vocab_size, pretrained_model):
    inputs = [
        Input(shape=(max_length,), name='monomer1_input'),
        Input(shape=(max_length,), name='monomer2_input'),
        Input(shape=(len(Constants.GROUP_VOCAB),), name='group_input'),
        Input(shape=(max_length,), name='decoder_input1'),
        Input(shape=(max_length,), name='decoder_input2')
    ]
    
    monomer1_input, monomer2_input, group_input, decoder_input1, decoder_input2 = inputs

    # Load pretrained components
    pretrained_embedding = pretrained_model.get_layer("embedding")
    pretrained_encoder = pretrained_model.get_layer("gru")
    pretrained_decoder_gru = pretrained_model.get_layer("gru_2")
    pretrained_dense = pretrained_model.get_layer("dense")
    
      # Freeze pretrained layers
    #pretrained_embedding.trainable = False
    # for layer in pretrained_encoder:
    #     layer.trainable = False
    #pretrained_decoder_gru.trainable = False
    #pretrained_dense.trainable = False
    #pretrained_encoder.trainable = False
    

    # Embed inputs
    monomer1_embedded = pretrained_embedding(monomer1_input)  # Shape: (batch, max_length, 128)
    monomer2_embedded = pretrained_embedding(monomer2_input)  # Shape: (batch, max_length, 128)

    monomer1_dense_embedding = Dense(136, activation="relu", name="monomer1_dense_embedding")(monomer1_embedded)
    monomer2_dense_embedding = Dense(136, activation="relu", name="monomer2_dense_embedding")(monomer2_embedded)

    # Use pretrained encoder directly
    encoder_output1 = pretrained_encoder(monomer1_dense_embedding)  # Shape: (batch, max_length, 512)
    encoder_output2 = pretrained_encoder(monomer2_dense_embedding)  # Shape: (batch, max_length, 512)

    # ✅ Extract last state manually if return_state=False
    encoder_state1 = encoder_output1[:, -1, :]  # Shape: (batch, 512)
    encoder_state2 = encoder_output2[:, -1, :]  # Shape: (batch, 512)

    # ✅ Project encoder state to match decoder size (128)
    encoder_state1_projected = Dense(128, activation="relu", name="encoder_projection1")(encoder_state1)
    encoder_state2_projected = Dense(128, activation="relu", name="encoder_projection2")(encoder_state2)

    # ✅ Project group input to size 128
    group_projected = Dense(128, activation="relu", name="group_projection")(group_input)  # Shape: (batch, 128)

    # ✅ Combine encoder state + group state using addition
    modified_state1 = Add(name="modified_state1")([encoder_state1_projected, group_projected])
    modified_state2 = Add(name="modified_state2")([encoder_state2_projected, group_projected])

    modified_state1 = Dense(512, activation="relu", name="state_projection1")(modified_state1)
    modified_state2 = Dense(512, activation="relu", name="state_projection2")(modified_state2)

    # ✅ Decoder embeddings
    decoder_embedded1 = pretrained_embedding(decoder_input1)  # Shape: (batch, max_length, 128)
    decoder_embedded2 = pretrained_embedding(decoder_input2)  # Shape: (batch, max_length, 128)

    decoder_dense_embedded1 = Dense(136, activation="relu", name="decoder_dense_embedding1")(decoder_embedded1)
    decoder_dense_embedded2 = Dense(136, activation="relu", name="decoder_dense_embedding2")(decoder_embedded2)

    # ✅ Project to correct size (128) before passing to GRU
    decoder_input1_combined = Dense(128, activation="relu", name="decoder_projection1")(decoder_dense_embedded1)
    decoder_input2_combined = Dense(128, activation="relu", name="decoder_projection2")(decoder_dense_embedded2)

    use_teacher_forcing = True
    if use_teacher_forcing:
        decoder_output1 = pretrained_decoder_gru(decoder_input1_combined, initial_state=modified_state1)
        decoder_output2 = pretrained_decoder_gru(decoder_input2_combined, initial_state=modified_state2)
    else:
        for t in range(max_length):
            decoder_output1, _ = pretrained_decoder_gru(
                tf.expand_dims(decoder_input1[:, t], axis=1), 
                initial_state=modified_state1
            )
            decoder_output2, _ = pretrained_decoder_gru(
                tf.expand_dims(decoder_input2[:, t], axis=1), 
                initial_state=modified_state2
            )

    # ✅ Pass projected state directly into GRU
    #decoder_output1 = pretrained_decoder_gru(decoder_input1_combined, initial_state=modified_state1)
    #decoder_output2 = pretrained_decoder_gru(decoder_input2_combined, initial_state=modified_state2)

     # Add attention layer here
    attention_layer1 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
    attention_layer2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
    
    # Apply attention between decoder output and encoder output
    context_vector1 = attention_layer1(query=decoder_output1, value=encoder_output1,key=encoder_output1)  # Using encoder_output1 from monomer1
    context_vector2 = attention_layer2(query=decoder_output2, value=encoder_output2,key=encoder_output2)  # Using encoder_output2 from monomer2

    # Combine context vectors with decoder outputs
    decoder_output1 = tf.keras.layers.Concatenate()([decoder_output1, context_vector1])
    decoder_output2 = tf.keras.layers.Concatenate()([decoder_output2, context_vector2])

    decoder_output1 = tf.keras.layers.Dense(512, activation="relu", name="decoder_output1")(decoder_output1)
    decoder_output2 = tf.keras.layers.Dense(512, activation="relu", name="decoder_output2")(decoder_output2)

    # ✅ Add dropout
    decoder_output1 = Dropout(0.1, name='dropout1')(decoder_output1)
    decoder_output2 = Dropout(0.1, name='dropout2')(decoder_output2)

    # ✅ Create two separate final dense layers with same weights as pretrained_dense
    final_dense1 = Dense(units=pretrained_dense.units, activation=pretrained_dense.activation, name='final_dense1')
    final_dense2 = Dense(units=pretrained_dense.units, activation=pretrained_dense.activation, name='final_dense2')

    final_dense1.build(decoder_output1.shape)
    final_dense2.build(decoder_output2.shape)
    final_dense1.set_weights(pretrained_dense.get_weights())
    final_dense2.set_weights(pretrained_dense.get_weights())

    # ✅ Final outputs
    monomer1_output = final_dense1(decoder_output1)
    monomer2_output = final_dense2(decoder_output2)

    # ✅ Model definition
    two_monomer_model = Model(
        inputs=[monomer1_input, monomer2_input, group_input, decoder_input1, decoder_input2],
        outputs=[monomer1_output, monomer2_output]
    )

    # # # ✅ Compile the model
    # two_monomer_model.compile(
    #     optimizer='adam',
    #     loss=["categorical_crossentropy", "categorical_crossentropy"],
    #     loss_weights=[1.0, 1.0]
    # )

    two_monomer_model.summary()
    
    return two_monomer_model
    
    
    



 
    
    
    
    
    
# def custom_loss(y_true, y_pred):
#     # Add custom weighting if needed
#     monomer_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
#     return monomer_loss
    


# if __name__ == "__main__":
   
#     root_dir = get_project_root()
    
#     # Setup paths
#     save_dir_abs = os.path.join(root_dir, "pretrained_model", "saved_models_rl_gpu_3")
#     file_path = os.path.join(root_dir, 'Data', "smiles.xlsx")
#     saved_new_model = os.path.join(root_dir, "group_based_rl_model")
#     pretrained_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)

#     # Prepare and split training data
#     all_data = prepare_training_data(
#         max_length=model_params['max_length'],
#         vocab=smiles_vocab,
#         file_path=file_path
#     )
    
#     # Split data into train and validation sets
#     train_size = int(0.8 * len(all_data[0]['monomer1_input']))
#     train_data = (
#         {k: v[:train_size] for k, v in all_data[0].items()},
#         {k: v[:train_size] for k, v in all_data[1].items()}
#     )
#     val_data = (
#         {k: v[train_size:] for k, v in all_data[0].items()},
#         {k: v[train_size:] for k, v in all_data[1].items()}
#     )
#     X,y = train_data

#     train_dataset = tf.data.Dataset.from_tensor_slices((
#         {
#             'monomer1_input': X['monomer1_input'],
#             'monomer2_input': X['monomer2_input'],
#             'group_input': X['group_input'],
#             'decoder_input1': X['decoder_input1'],
#             'decoder_input2': X['decoder_input2'],
#         },
#         {
#             'decoder1': y['monomer1_output'],
#             'decoder2': y['monomer2_output']
#         }
#     )).batch(Constants.BATCH_SIZE)

#     model=create_model(model_params['max_length'], len(smiles_vocab),pretrained_model)
#     print("\nModel summary:")
#     model.summary()
#     model.fit(train_dataset, validation_data=val_data, epochs=Constants.EPOCHS)

#     # Make predictions
#     test_data = {
#         'monomer1_input': X['monomer1_input'][:5],  # Take first 5 samples for testing
#         'monomer2_input': X['monomer2_input'][:5],
#         'group_input': X['group_input'][:5],
#         'decoder_input1': X['decoder_input1'][:5],
#         'decoder_input2': X['decoder_input2'][:5],
#     }

#     generated_monomer = beam_search(model, test_data, smiles_vocab, k=5, max_length=100)
#     print(generated_monomer)

    # predictions = model.predict(test_data)

    # # Since the model has two outputs, predictions will be a list of two arrays
    # monomer1_predictions = predictions[0]  # First output (final_dense1)
    # monomer2_predictions = predictions[1]  # Second output (final_dense2)

    # # Now you can look at the first 5 predictions
    # # Convert predictions back to SMILES
    # def decode_smiles_str(pred, vocab):
    #     # Get indices of highest probability tokens
    #     indices = np.argmax(pred, axis=-1)
    #     smiles = decode_smiles(indices)
    #     return smiles

    # print("\nPredicted SMILES:")
    # print("Monomer 1 predictions:")
    # for i, pred in enumerate(monomer1_predictions[:5]):
    #     smiles = decode_smiles_str(pred, smiles_vocab)
    #     print(f"Sample {i+1}: {smiles}")
    
    # print("\nMonomer 2 predictions:")
    # for i, pred in enumerate(monomer2_predictions[:5]):
    #     smiles = decode_smiles_str(pred, smiles_vocab)
    #     print(f"Sample {i+1}: {smiles}")
   




