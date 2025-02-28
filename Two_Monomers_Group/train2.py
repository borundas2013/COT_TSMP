import os
import tensorflow as tf
from Constants import *
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from pretrained_weights import *
from saveandload import *
from dual_smile_process import *
from validation_prediction import *
from group_based_model import create_group_relationship_model
from pathlib import Path
from sample_generator import *
from losswithReward import CombinedLoss
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(tf.config.list_physical_devices('GPU'))

def get_project_root():
    """Get the path to the project root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent


# def calculate_reaction_reward(y_true1, y_true2, pred1, pred2):
#     """Calculate reaction reward for batch predictions"""
#     # Get predicted and true tokens
#     scores = []
#     for i in range(len(pred1)):
#         pred_tokens1 = tf.argmax(pred1[i], axis=-1)
#         pred_tokens2 = tf.argmax(pred2[i], axis=-1)
#         true_tokens1 = tf.argmax(y_true1[i], axis=-1)
#         true_tokens2 = tf.argmax(y_true2[i], axis=-1)
    
#         try:
#             true_smiles1 = decode_smiles(true_tokens1)
#             true_smiles2 = decode_smiles(true_tokens2)
#             pred_smiles1 = decode_smiles(pred_tokens1)
#             pred_smiles2 = decode_smiles(pred_tokens2)
        
#             score = get_reaction_score(true_smiles1, true_smiles2, pred_smiles1, pred_smiles2)
#             scores.append(score)     
#         except Exception as e:
#             print("\nError in reaction reward:", e)
#             return 0.0
#     print("Scores: ", scores)   
#     print("Reaction Reward: ", np.mean(scores))
#     return np.mean(scores)

def calculate_reaction_reward(y_true1, y_true2, pred1, pred2):
    """Calculate reaction reward for batch predictions."""

    def process_single_sample(inputs):
        """Processes a single batch element and computes reaction reward."""
        y_t1, y_t2, p1, p2 = inputs

        # Get predicted and true tokens
        pred_tokens1 = tf.argmax(p1, axis=-1)
        pred_tokens2 = tf.argmax(p2, axis=-1)
        true_tokens1 = tf.argmax(y_t1, axis=-1)
        true_tokens2 = tf.argmax(y_t2, axis=-1)

        try:
            # Convert tokens to SMILES
            true_smiles1 = decode_smiles(true_tokens1.numpy())
            true_smiles2 = decode_smiles(true_tokens2.numpy())
            pred_smiles1 = decode_smiles(pred_tokens1.numpy())
            pred_smiles2 = decode_smiles(pred_tokens2.numpy())
            print("True Smiles 1: ", true_smiles1)
            print("True Smiles 2: ", true_smiles2)
            print("Pred Smiles 1: ", pred_smiles1)
            print("Pred Smiles 2: ", pred_smiles2)

            # Compute reaction score
            score = get_reaction_score(true_smiles1, true_smiles2, pred_smiles1, pred_smiles2)
            return tf.convert_to_tensor(score, dtype=tf.float32)
        except Exception as e:
            tf.print("\nError in reaction reward calculation:", e)
            return tf.convert_to_tensor(0.0, dtype=tf.float32)  # Default to zero reward on error

    # ✅ Vectorized computation for batch-wise processing
    scores = tf.map_fn(
        process_single_sample, 
        (y_true1, y_true2, pred1, pred2), 
        dtype=tf.float32
    )

    # Compute mean reaction reward
    reaction_reward = tf.reduce_mean(scores)

    # ✅ Print debug info without breaking the computation graph
    tf.print("Reaction Scores:", scores)
    tf.print("Mean Reaction Reward:", reaction_reward)

    return reaction_reward

# def generate_complete_smiles(model, x_batch, max_length, vocab):
#     """Generate complete SMILES sequences token by token"""
    
#     decoder1_input = x_batch['decoder_input1']  # Keep full length (245)
#     decoder2_input = x_batch['decoder_input2']
#     generated_tokens1=[]  # Keep full length (245)
#     generated_tokens2=[]  # Keep full length (245)

#     try:
#         for i in range(5):
#             predictions = model(
#                 {
#                     'monomer1_input': x_batch['monomer1_input'],
#                     'monomer2_input': x_batch['monomer2_input'],
#                     'group_input': x_batch['group_input'],
#                     'decoder_input1': decoder1_input,
#                     'decoder_input2': decoder2_input
#                 }, 
#                 training=True
#             )
#             print("Predictions: ", predictions.shape)
            
#             # Get next token predictions
#             next_token1 = np.argmax(predictions[0][:, -1:, :], axis=-1)
#             next_token2 = np.argmax(predictions[1][:, -1:, :], axis=-1) 
#             token1 = next_token1[0][0]  
#             token2 = next_token2[0][0]

#             generated_tokens1.append(token1)
#             generated_tokens2.append(token2)
            

#             # Check for END token in each sequence
#             is_end1 = token1 == vocab['<end>']
#             is_end2 = token2 == vocab['<end>']
#             if is_end1:
#                 print("Sequence 1 completed at step ", i+1)
#             if is_end2:
#                 print("Sequence 2 completed at step ", i+1)
            
#             # Maintain 245 length by shifting window
#             decoder1_input = tf.concat([decoder1_input[:, 1:], next_token1], axis=1)  # Remove first, add new at end
#             decoder2_input = tf.concat([decoder2_input[:, 1:], next_token2], axis=1)  # Remove first, add new at end
            
#             if is_end1 and is_end2:
#                 print(f"Sequences completed at step {i+1}")
#                 break
        
#         return generated_tokens1, generated_tokens1
        
#     except Exception as e:
#         print(f"Error in sequence generation: {str(e)}")
#         return None, None


import tensorflow as tf
import numpy as np



def generate_complete_smiles(model, x_batch, max_length, vocab_size):
    """Generate complete SMILES sequences token by token during training with logits output."""
    
    # decoder1_input = x_batch['decoder_input1']
    # decoder2_input = x_batch['decoder_input2']

    max_length = 246

    batch_size = tf.shape(x_batch['decoder_input1'])[0]
    seq_length = tf.shape(x_batch['decoder_input1'])[1]
    
    # Create zero tensors with same shape
    decoder1_input = tf.zeros((batch_size, seq_length), dtype=tf.int32)
    decoder2_input = tf.zeros((batch_size, seq_length), dtype=tf.int32)
    
    # Store full vocab distributions (batch_size, max_length, vocab_size)
    generated_logits1 = tf.TensorArray(dtype=tf.float32, size=max_length, dynamic_size=False, clear_after_read=False)
    generated_logits2 = tf.TensorArray(dtype=tf.float32, size=max_length, dynamic_size=False, clear_after_read=False)

    for i in range(3):
        predictions = model(
            {
                'monomer1_input': x_batch['monomer1_input'],
                'monomer2_input': x_batch['monomer2_input'],
                'group_input': x_batch['group_input'],
                'decoder_input1': decoder1_input,
                'decoder_input2': decoder2_input
            }, 
            training=True  # Ensure training mode
        )
        print("Predictions: ", predictions[0][0])
        
        logits1 = predictions[0][:, -1:, :]  # Shape: (batch_size, 1, vocab_size)
        logits2 = predictions[1][:, -1:, :]  # Shape: (batch_size, 1, vocab_size)

        generated_logits1 = generated_logits1.write(i, tf.squeeze(logits1, axis=1))  
        generated_logits2 = generated_logits2.write(i, tf.squeeze(logits2, axis=1))  

        next_tokens1 = tf.argmax(logits1, axis=-1, output_type=tf.int32)  # (batch_size, 1)
        next_tokens2 = tf.argmax(logits2, axis=-1, output_type=tf.int32)  # (batch_size, 1)
        print("Next Tokens 1: ", next_tokens1)
        print("Next Tokens 2: ", next_tokens2)

        # No manual tensor detachment here to keep gradients!
       
        decoder1_input = tf.concat([decoder1_input[:, :i], next_tokens1, decoder1_input[:, i+1:]], axis=1)
        decoder2_input = tf.concat([decoder2_input[:, :i], next_tokens2, decoder2_input[:, i+1:]], axis=1)
        print("Decoder 1 Input: ", decoder1_input)
        print("Decoder 2 Input: ", decoder2_input)

        

        

    generated_logits1 = tf.transpose(generated_logits1.stack(), perm=[1, 0, 2])  # (batch_size, max_length, vocab_size)
    generated_logits2 = tf.transpose(generated_logits2.stack(), perm=[1, 0, 2])  # (batch_size, max_length, vocab_size)

    return generated_logits1, generated_logits2

import tensorflow as tf

def generate_complete_smiles2(model, x_batch, max_length, vocab_size):
    """Generate complete SMILES sequences token by token during training with logits output, stopping at EOS."""
    
    batch_size = tf.shape(x_batch['decoder_input1'])[0]
    seq_length = tf.shape(x_batch['decoder_input1'])[1]
    
    # Use provided decoder input from x_batch
    decoder1_input = x_batch['decoder_input1']
    decoder2_input = x_batch['decoder_input2']
    
    # Store full vocab distributions (batch_size, max_length, vocab_size)
    generated_logits1 = tf.TensorArray(dtype=tf.float32, size=max_length, dynamic_size=False, clear_after_read=False)
    generated_logits2 = tf.TensorArray(dtype=tf.float32, size=max_length, dynamic_size=False, clear_after_read=False)

    # Keep track of finished sequences
    finished_sequences = tf.zeros((batch_size,), dtype=tf.bool)  # (batch_size,)

    for i in range(3):
        predictions = model(
            {
                'monomer1_input': x_batch['monomer1_input'],
                'monomer2_input': x_batch['monomer2_input'],
                'group_input': x_batch['group_input'],
                'decoder_input1': decoder1_input,
                'decoder_input2': decoder2_input
            }, 
            training=True  # Training-time generation
        )

        logits1 = predictions[0][:, i:i+1, :]  # (batch_size, 1, vocab_size)
        logits2 = predictions[1][:, i:i+1, :]  # (batch_size, 1, vocab_size)

        generated_logits1 = generated_logits1.write(i, tf.squeeze(logits1, axis=1))  
        generated_logits2 = generated_logits2.write(i, tf.squeeze(logits2, axis=1))  

        next_tokens1 = tf.argmax(logits1, axis=-1, output_type=tf.int32)  # (batch_size, 1)
        next_tokens2 = tf.argmax(logits2, axis=-1, output_type=tf.int32)  # (batch_size, 1)

        # Check if EOS token is generated
        eos_mask1 = tf.equal(next_tokens1, 1)  # (batch_size, 1) → True where EOS is found
        eos_mask2 = tf.equal(next_tokens2, 1)  # (batch_size, 1)
        finished_sequences = tf.logical_or(finished_sequences, tf.squeeze(eos_mask1) | tf.squeeze(eos_mask2))

        # Stop generation if all sequences are finished
        if tf.reduce_all(finished_sequences):
            break

        # ✅ Fix: Create index tensor with [batch_idx, seq_pos] format
        indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], i)], axis=1)  # (batch_size, 2)
        
        # ✅ Fix: Update decoder input at time step `i`
        decoder1_input = tf.tensor_scatter_nd_update(decoder1_input, indices, tf.squeeze(next_tokens1, axis=1))
        decoder2_input = tf.tensor_scatter_nd_update(decoder2_input, indices, tf.squeeze(next_tokens2, axis=1))

    generated_logits1 = tf.transpose(generated_logits1.stack(), perm=[1, 0, 2])  # (batch_size, max_length, vocab_size)
    generated_logits2 = tf.transpose(generated_logits2.stack(), perm=[1, 0, 2])  # (batch_size, max_length, vocab_size)

    return generated_logits1, generated_logits2





def custom_training_loop(model, train_data, val_data=None, epochs=1, max_length=245, vocab_size=None):
    """
    Custom training loop with custom loss function
    """
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Unpack training data
    X, y = train_data
    
    # Setup batch size and steps
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    base_batch_size = Constants.BATCH_SIZE
    global_batch_size = base_batch_size * max(1, num_gpus)
    steps_per_epoch = len(X['monomer1_input']) // global_batch_size
    
    print(f"\nTraining Configuration:")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Base batch size per GPU: {base_batch_size}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Total training samples: {len(X['monomer1_input'])}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'monomer1_input': X['monomer1_input'],
            'monomer2_input': X['monomer2_input'],
            'group_input': X['group_input'],
            'decoder_input1': X['decoder_input1'],
            'decoder_input2': X['decoder_input2']
        },
        {
            'decoder1': y['monomer1_output'],
            'decoder2': y['monomer2_output']
        }
    )).shuffle(buffer_size=1000).batch(global_batch_size)

    # Initialize custom loss
    custom_loss = CombinedLoss()
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [] if val_data is not None else None
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        progress_bar = tf.keras.utils.Progbar(steps_per_epoch)
        
        # Initialize epoch metrics
        epoch_loss = tf.keras.metrics.Mean()
        
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # Forward pass
                generated_smiles1, generated_smiles2 = generate_complete_smiles(model, x_batch, max_length=max_length,vocab_size=vocab_size)
               
                

                # Calculate losses for each decoder
                loss_decoder1 = custom_loss(y_batch['decoder1'],   generated_smiles1)
                loss_decoder2 = custom_loss(y_batch['decoder2'],    generated_smiles2)

                reaction_reward = calculate_reaction_reward(y_batch['decoder1'],y_batch['decoder2'],generated_smiles1,generated_smiles2)
                print("Reaction Reward: ", reaction_reward)
                print("Loss Decoder 1: ", loss_decoder1)
                print("Loss Decoder 2: ", loss_decoder2)    
                print("Total Loss: ", loss_decoder1+loss_decoder2)
                
                # # Combined loss
                total_loss = loss_decoder1 + loss_decoder2 #- 0.8*reaction_reward
                print("After reward Total Loss: ", total_loss)
            
            # Calculate and apply gradients
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update metricsis t
            epoch_loss.update_state(total_loss)
            
            # Update progress bar
            progress_bar.update(
                step + 1, 
                [('loss', total_loss.numpy())]
            )
        
        # End of epoch
        avg_loss = epoch_loss.result()
        print(f"\nEpoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        history['loss'].append(avg_loss.numpy())
        
        # Validation
        if val_data is not None:
            val_loss = validate_model(model, val_data, global_batch_size, custom_loss)
            history['val_loss'].append(val_loss.numpy())
    
    return history

def validate_model(model, val_data, batch_size, custom_loss):
    """
    Validation function
    """
    X_val, y_val = val_data
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'monomer1_input': X_val['monomer1_input'],
            'monomer2_input': X_val['monomer2_input'],
            'group_input': X_val['group_input'],
            'decoder_input1': X_val['decoder_input1'],
            'decoder_input2': X_val['decoder_input2']
        },
        {
            'decoder1': y_val['monomer1_output'],
            'decoder2': y_val['monomer2_output']
        }
    )).batch(batch_size)
    
    val_loss = tf.keras.metrics.Mean()
    
    for x_batch, y_batch in val_dataset:
        predictions = model(x_batch, training=False)
        loss1 = custom_loss(y_batch['decoder1'], predictions[0])
        loss2 = custom_loss(y_batch['decoder2'], predictions[1])
        total_loss = loss1 + loss2
        val_loss.update_state(total_loss)
    
    print(f"\nValidation Loss: {val_loss.result():.4f}")
    return val_loss.result()

if __name__ == "__main__":
    root_dir = get_project_root()
    
    # Setup paths
    save_dir_abs = os.path.join(root_dir, "pretrained_model", "saved_models_rl_gpu_3")
    file_path = os.path.join(root_dir, 'Data', "smiles.xlsx")
    saved_new_model = os.path.join(root_dir, "group_based_rl_model")
    
    try:
        pretrained_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)
            
            # Prepare and split training data
        all_data = prepare_training_data(
                max_length=model_params['max_length'],
                vocab=smiles_vocab,
                file_path=file_path
            )
            
            # Split data into train and validation
        train_size = int(0.8 * len(all_data[0]['monomer1_input']))
        train_data = (
                {k: v[:train_size] for k, v in all_data[0].items()},
                {k: v[:train_size] for k, v in all_data[1].items()}
            )
        val_data = (
                {k: v[train_size:] for k, v in all_data[0].items()},
                {k: v[train_size:] for k, v in all_data[1].items()}
            )
            
            # Create model
        new_model = create_group_relationship_model(
                pretrained_model=pretrained_model,
                max_length=model_params['max_length'],
                vocab_size=len(smiles_vocab)
            )
            
        print("\nModel summary:")
        new_model.summary()
            
            # Train using custom training loop
        history = custom_training_loop(
                model=new_model,
                train_data=train_data,
                val_data=val_data,
                epochs=Constants.EPOCHS,
                max_length=model_params['max_length'],
                vocab_size=len(smiles_vocab)
            )
            
        print('Training completed successfully')
            
        #     # Save the model
        weights_path, params_path = save_model(
            model=new_model,
            model_params={
                'max_length': model_params['max_length'],
                'vocab_size': len(smiles_vocab),
            },
            save_dir=saved_new_model
        )
        
        # Create prediction model and generate examples
        prediction_model = create_group_relationship_model(
                pretrained_model=pretrained_model,
                max_length=model_params['max_length'],
                vocab_size=len(smiles_vocab),
            )
        #weights_path = os.path.join(root_dir, "group_based_rl_model", "weights_model.weights.h5")
        prediction_model.load_weights(weights_path)
            
            # Generate example predictions
        monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
        group_combination = [["C=C", "C=C(C=O)"], ["C=C", "CCS"], ["C1OC1", "NC"], ["C=C", "OH"]]
        
        for smiles1, smiles2 in zip(monomer1_list[:1], monomer2_list[:1]):
            group_smarts1 = extract_group_smarts2(smiles1)
            group_smarts2 = extract_group_smarts2(smiles2)
            for group in group_combination:
                    print(f"\nGenerating with groups: {group}")
                    generated, valid = generate_monomer_pair_with_temperature(
                        model=prediction_model,
                        input_smiles=[smiles1, smiles2],
                        desired_groups=group,
                        vocab=smiles_vocab,
                        max_length=model_params['max_length'],
                        temperatures=[0.2,0.4],
                        group_smarts1=group_smarts1,
                        group_smarts2=group_smarts2
                    )
                    
                    print("\nValid pairs generated:")
                    for pair in valid:
                        print(f"Monomer 1: {pair[0]}")
                        print(f"Monomer 2: {pair[1]}")
                        print("-" * 40)
    
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise