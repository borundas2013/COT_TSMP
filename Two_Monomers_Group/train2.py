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
from NewModelApp1 import *
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


def calculate_reaction_reward(y_true1, y_true2, pred1, pred2):
    """Calculate reaction reward for batch predictions."""

    def process_single_sample(inputs):
        """Processes a single batch element and computes reaction reward."""
        y_t1, y_t2, p1, p2 = inputs

        # Get predicted and true tokens
        pred_tokens1 = tf.argmax(p1, axis=-1)
        pred_tokens2 = tf.argmax(p2, axis=-1)
        true_tokens1 = tf.argmax(y_t1, axis=-1)#y_t1#
        true_tokens2 = tf.argmax(y_t2, axis=-1)#y_t2#

        try:
            # Convert tokens to SMILES
            true_smiles1 = decode_smiles(true_tokens1.numpy().astype(np.int32))
            true_smiles2 = decode_smiles(true_tokens2.numpy().astype(np.int32))
            pred_smiles1 = decode_smiles(pred_tokens1.numpy())
            pred_smiles2 = decode_smiles(pred_tokens2.numpy())
            print("-------------Reaction Reward Calculation-------------------")
            print("True Smiles 1: ", true_smiles1)
            print("True Smiles 2: ", true_smiles2)
            print("Pred Smiles 1: ", pred_smiles1)
            print("Pred Smiles 2: ", pred_smiles2)
            # Save predicted valid SMILES to file if they form valid molecules
            if not pred_smiles1 or not pred_smiles2:
                print("Warning: Empty predicted SMILES")
                return tf.convert_to_tensor(0.0, dtype=tf.float32)
            pred_mol1 = Chem.MolFromSmiles(pred_smiles1)
            pred_mol2 = Chem.MolFromSmiles(pred_smiles2)
            score = 0
            
            if pred_mol1 is not None and pred_mol2 is not None:
                pair_data = {
                    "true_smiles": {
                        "monomer1": true_smiles1,
                        "monomer2": true_smiles2
                    },
                    "predicted_smiles": {
                        "monomer1": pred_smiles1,
                        "monomer2": pred_smiles2
                    }
                }
                with open(Constants.VALID_PAIRS_FILE, 'a') as f:
                    json.dump(pair_data, f)
                    f.write('\n')

                valid, chemical_groups = check_reaction_validity(true_smiles1, true_smiles2)
                print("Valid: ", valid)
                print("Chemical Groups: ", chemical_groups)


            # Compute reaction score
                score = get_reaction_score(pred_smiles1, pred_smiles2, chemical_groups)
                print("Score: ", score)
            return tf.convert_to_tensor(score, dtype=tf.float32)
        except Exception as e:
            print("\nError in reaction reward calculation:", e)
            return tf.convert_to_tensor(0.0, dtype=tf.float32)  # Default to zero reward on error

    #  Vectorized computation for batch-wise processing
    scores = tf.map_fn(
        process_single_sample, 
        (y_true1, y_true2, pred1, pred2), 
        dtype=tf.float32
    )

    # Compute mean reaction reward
    reaction_reward = tf.reduce_mean(scores)

    # Print debug info without breaking the computation graph
    print("Reaction Scores:", scores)
    print("Mean Reaction Reward:", reaction_reward)

    return reaction_reward

def generate_complete_smiles(model, x_batch):
    """Generate complete SMILES sequences token by token during training with logits output."""
    
    decoder1_input = x_batch['decoder_input1']
    decoder2_input = x_batch['decoder_input2']

    print("Input batch shapes:")
    print(f"monomer1_input shape: {x_batch['monomer1_input'].shape}")
    print(f"monomer2_input shape: {x_batch['monomer2_input'].shape}")
    print(f"group_input shape: {x_batch['group_input'].shape}")
    print(f"decoder_input1 shape: {x_batch['decoder_input1'].shape}")
    print(f"decoder_input2 shape: {x_batch['decoder_input2'].shape}")

        

   
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
          
    return predictions[0], predictions[1]


def custom_training_loop(model, train_data, val_data=None, epochs=1, max_length=245, vocab_size=None, vocab=None, save_dir=None):
   
    
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
            'decoder_input2': X['decoder_input2'],
            'original_monomer1': X['original_monomer1'],
            'original_monomer2': X['original_monomer2']
        },
        {
            'decoder1': y['monomer1_output'],
            'decoder2': y['monomer2_output']
        }
    )).shuffle(buffer_size=1000).batch(global_batch_size)

    # Initialize custom loss
    custom_loss = CombinedLoss()
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss=custom_loss)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [] if val_data is not None else None
    }
    best_val_loss=float('inf')
    valid_data_loss = []


    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_steps = steps_per_epoch
        print("Total Steps this epoch: ", total_steps)
       
        
        # Initialize epoch metrics
        epoch_loss = tf.keras.metrics.Mean()
        
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                original_monomer1 = x_batch['original_monomer1']
                original_monomer2 = x_batch['original_monomer2']
                #print("Original Monomer 1: ", decode_smiles(original_monomer1[0]))
                #print("Original Monomer 2: ", decode_smiles(original_monomer2[0]))
                # Forward pass
                generated_tokens1, generated_tokens2 = generate_complete_smiles(model, x_batch)
               
                

                # Calculate losses for each decoder
                loss_decoder1 = custom_loss(original_monomer1,   generated_tokens1)
                loss_decoder2 = custom_loss(original_monomer2,    generated_tokens2)

                #reaction_reward = calculate_reaction_reward(original_monomer1,original_monomer2,generated_tokens1,generated_tokens2)
                #print("Reaction Reward: ", reaction_reward)
                print("Loss Decoder 1: ", loss_decoder1)
                print("Loss Decoder 2: ", loss_decoder2)    
                print("Total Loss: ", loss_decoder1+loss_decoder2)
                
                # # Combined loss
                total_loss = loss_decoder1 + loss_decoder2 #- 0.8*reaction_reward
                print("After reward Total Loss: ", total_loss)
            
            # Calculate and apply gradients
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update metricsis t
            epoch_loss.update_state(total_loss)
            
            # Update progress bar
            if (step+1) % 10 == 0:
                print(f"Step {step+1} of {total_steps} - Loss: {total_loss.numpy():.4f}")
        
        # End of epoch
        avg_loss = epoch_loss.result()
        print(f"\nEpoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        history['loss'].append(avg_loss.numpy())

        print("-----------Validation Start-----------")
        
        # Validation
        if val_data is not None:
            val_loss = validate_model(model, val_data, global_batch_size,
                                       custom_loss,max_length,vocab_size,vocab)
            print("Validation Loss: ", val_loss)
            valid_data_loss.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New Best Validation Loss: ", best_val_loss)
                model_path, params_path = save_model(model, model_params, save_dir=save_dir)
                patience_counter = 0
                
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{Constants.PATIENCE}")
        
                if patience_counter >= Constants.PATIENCE:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break

            history['val_loss'].append(val_loss.numpy())
    
    return history, model_path, params_path


def validate_model(model, val_data, batch_size, custom_loss, max_length, vocab_size, vocab):
    """
    Validation function with comprehensive metrics and error handling
    """
    X, y = val_data
    
    # Initialize metrics
    val_loss = tf.keras.metrics.Mean()
    reconstruction_loss = tf.keras.metrics.Mean()
    #reward_metric = tf.keras.metrics.Mean()
    valid_predictions = 0
    total_samples = 0
    
    try:
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'monomer1_input': X['monomer1_input'],
                'monomer2_input': X['monomer2_input'],
                'group_input': X['group_input'],
                'decoder_input1': X['decoder_input1'],
                'decoder_input2': X['decoder_input2'],
                'original_monomer1': X['original_monomer1'],
                'original_monomer2': X['original_monomer2']
            },
            {
                'decoder1': y['monomer1_output'],
                'decoder2': y['monomer2_output']
            }
        )).batch(batch_size)

        for x_batch, y_batch in val_dataset:
            try:
                # Generate predictions
                generated_smiles1, generated_smiles2 = generate_complete_smiles(model, x_batch)
                original_monomer1 = x_batch['original_monomer1']
                original_monomer2 = x_batch['original_monomer2']
                print("Original Monomer 1: ", original_monomer1.shape)
                print("Original Monomer 2: ", original_monomer2.shape)
                # Calculate reconstruction loss
                loss1 = custom_loss(original_monomer1, generated_smiles1)
                loss2 = custom_loss(original_monomer2, generated_smiles2)
                recon_loss = loss1 + loss2
                reconstruction_loss.update_state(recon_loss)
                
                # Calculate reaction reward (optional during validation)
                # reward = calculate_reaction_reward(
                #     original_monomer1, original_monomer2,
                #     generated_smiles1, generated_smiles2
                # )
                #reward_metric.update_state(reward)
                
                # Total loss (you might want to use different weights for validation)
                total_loss = recon_loss #- 0.8 * reward
                val_loss.update_state(total_loss)
                
                # Count valid predictions
                pred_tokens1 = tf.argmax(generated_smiles1, axis=-1)
                pred_tokens2 = tf.argmax(generated_smiles2, axis=-1)
    
                for i in range(len(pred_tokens1)):
                    decoded_smiles1 = decode_smiles(pred_tokens1[i])    
                    decoded_smiles2 = decode_smiles(pred_tokens2[i])
                    if Chem.MolFromSmiles(decoded_smiles1) is not None and Chem.MolFromSmiles(decoded_smiles2) is not None:
                        valid_predictions += 1
                    print(f"Sample {i}:")
                    print(f"Monomer 1: {decoded_smiles1}")
                    print(f"Monomer 2: {decoded_smiles2}")
                    print(f"Valid: {valid_predictions}")
                
                
            except Exception as batch_error:
                print(f"Error processing validation batch: {str(batch_error)}")
                continue
        
        # Calculate validation metrics
        validity_rate = valid_predictions / total_samples if total_samples > 0 else 0
        
        # Print comprehensive validation results
        print("\nValidation Results:")
        print(f"Total Loss: {val_loss.result():.4f}")
        print(f"Reconstruction Loss: {reconstruction_loss.result():.4f}")
        #print(f"Average Reward: {reward_metric.result():.4f}")
        print(f"Validity Rate: {validity_rate:.2%}")
        
        return val_loss.result()
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return float('inf')  # Return worst possible loss on failure

if __name__ == "__main__":
    root_dir = get_project_root()
    
    # Setup paths
    save_dir_abs = os.path.join(root_dir, "pretrained_model", "saved_models_rl_gpu_4")
    file_path = os.path.join(root_dir, 'Data', "smiles_orginal.xlsx")
    saved_new_model = os.path.join(root_dir, "group_based_rl_model_noise_s_trainable")
    
    try:
        pretrained_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)
        
        # Prepare and split training data
        all_data = prepare_training_data(
            max_length=model_params['max_length'],
            vocab=smiles_vocab,
            file_path=file_path
        )

        # all_data = prepare_training_data_with_noise(
        #     max_length=model_params['max_length'],
        #     vocab=smiles_vocab,
        #     file_path=file_path,
        #     noise_config=Constants.NOISE_CONFIG
        # )
        
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
        # new_model = create_group_relationship_model(
        #     pretrained_model=pretrained_model,
        #     max_length=model_params['max_length'],
        #     vocab_size=len(smiles_vocab)
        # )
        new_model = create_model(model_params['max_length'], len(smiles_vocab), pretrained_model)
        
        print("\nModel summary:")
        new_model.summary()
        
        # Train using custom training loop
        history, weights_path, params_path = custom_training_loop(
            model=new_model,
            train_data=train_data,
            val_data=val_data,
            epochs=Constants.EPOCHS,
            max_length=model_params['max_length'],
            vocab_size=len(smiles_vocab),
            vocab=smiles_vocab,
            save_dir=saved_new_model
        )
        
        print('Training completed successfully')
        
        # Save the model
        # weights_path, params_path = save_model(
        #     model=new_model,
        #     model_params={
        #         'max_length': model_params['max_length'],
        #         'vocab_size': len(smiles_vocab),
        #     },
        #     save_dir=saved_new_model
        # )
        
        # Create prediction model and generate examples
        # prediction_model = create_group_relationship_model(
        #     pretrained_model=pretrained_model,
        #     max_length=model_params['max_length'],
        #     vocab_size=len(smiles_vocab),
        # )
        
        # weights_path='/ddnB/work/borun22/Transfer_learning/NewCOT/Group_based/updated_group_based/group_based_rl_model_noise/weights_model.weights.h5'

        prediction_model = create_model(model_params['max_length'], len(smiles_vocab), pretrained_model)
        # weights_path = os.path.join(root_dir, "group_based_rl_model", "weights_model.weights.h5")
        prediction_model.load_weights(weights_path)
        
        # Generate example predictions
        monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
        # group_combination = [["C=C", "C=C(C=O)"], ["C=C", "CCS"], ["C1OC1", "NC"], ["C=C", "OH"]]
        
        group_combinations = [["C=C", "C=C(C=O)"], ["C=C", "CCS"], ["[OX2]1[CX3][CX3]1", "[NX2]=[CX3]"], ["C=C", "[OH]"]]
        selected_groups = random.choice(group_combinations)

        smiles_1 = []
        smiles_2 = []

        for index in range(len(monomer1_list[:1])):
            selected_groups = random.choice(group_combinations)
            mol = Chem.MolFromSmiles(monomer1_list[index])
            mol2 = Chem.MolFromSmiles(monomer2_list[index])
            pattern = Chem.MolFromSmarts(selected_groups[0])
            pattern2 = Chem.MolFromSmarts(selected_groups[1])
            smiles_1.append(monomer1_list[index])
            smiles_2.append(monomer2_list[index])
            # if mol is not None and mol2 is not None:
            #     if len(mol.GetSubstructMatches(pattern)) >= 2 and len(mol2.GetSubstructMatches(pattern2)) >= 2:
            #         smiles_1.append(monomer1_list[index])
            #         smiles_2.append(monomer2_list[index])

        print('Selected groups: ', selected_groups)
        print('Number of smiles 1: ', len(smiles_1))
        print('Number of smiles 2: ', len(smiles_2))

        # smiles1 = random.choice(smiles_1)
        # smiles2 = random.choice(smiles_2)

        for i in range(len(smiles_1)):
            smiles1 = smiles_1[i]
            smiles2 = smiles_2[i]
            gen, valid_pairs = generate_monomer_pair_with_temperature(
                model=prediction_model,
                input_smiles=[smiles1, smiles2],
                desired_groups=selected_groups,
                vocab=smiles_vocab,
                max_length=model_params['max_length'],
                temperatures=[1.5, 1.2, 1.0, 0.8, 0.6, 0.4],
                group_smarts1=selected_groups[0],
                group_smarts2=selected_groups[1],
                add_noise=False
            )

            gen2, valid_pairs_noise = generate_monomer_pair_with_temperature(
                model=prediction_model,
                input_smiles=[smiles1, smiles2],
                desired_groups=selected_groups,
                vocab=smiles_vocab,
                max_length=model_params['max_length'],
                temperatures=[1.5, 1.2, 1.0, 0.8, 0.6, 0.4],
                group_smarts1=selected_groups[0],
                group_smarts2=selected_groups[1],
                add_noise=True
            )

            for pair in valid_pairs:
                print(f"Monomer 1: {pair[0]}")
                print(f"Monomer 2: {pair[1]}")
                print("-" * 40)
            print("-" * 40)
            for pair in valid_pairs_noise:
                print(f"Monomer 1: {pair[0]}")
                print(f"Monomer 2: {pair[1]}")
                print("-" * 40)

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise


