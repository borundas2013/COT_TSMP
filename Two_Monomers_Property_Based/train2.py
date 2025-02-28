import os
import tensorflow as tf
from Constants import *
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from pretrained_weights import *
from saveandload import *
from dual_smile_process import *
from validation_prediction import *
from property_based_model import create_group_relationship_model
from pathlib import Path
from sample_generator import *
from losswithReward import CombinedLoss
#from Reward_score_property import *
from Property_Prediction.predict import reward_score
from rewardnormalizer import RewardNormalizer
import numpy as np
import sys
import random
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def get_project_root():
    """Get the path to the project root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent


def calculate_reaction_reward(y_true1, y_true2, pred1, pred2):
    """Calculate reaction reward for batch predictions"""
    # Get predicted and true tokens
    scores = []
    for i in range(len(pred1)):
        pred_tokens1 = tf.argmax(pred1[i], axis=-1)
        pred_tokens2 = tf.argmax(pred2[i], axis=-1)
        true_tokens1 = tf.argmax(y_true1[i], axis=-1)
        true_tokens2 = tf.argmax(y_true2[i], axis=-1)
    
        try:
            true_smiles1 = decode_smiles(true_tokens1)
            true_smiles2 = decode_smiles(true_tokens2)
            pred_smiles1 = decode_smiles(pred_tokens1)
            pred_smiles2 = decode_smiles(pred_tokens2)
        
            score = get_reaction_score(true_smiles1, true_smiles2, pred_smiles1, pred_smiles2)
            scores.append(score)     
        except Exception as e:
            print("\nError in reaction reward:", e)
            return 0.0
    print("Scores: ", scores)   
    print("Reaction Reward: ", np.mean(scores))
    return np.mean(scores)

def calculate_property_reward(pred1, pred2, actual_tg, actual_er):
    scores = []
    for i in range(len(pred1)):
        pred_tokens1 = tf.argmax(pred1[i], axis=-1)
        pred_tokens2 = tf.argmax(pred2[i], axis=-1)

    
        try:
            pred_smiles1 = decode_smiles(pred_tokens1)
            pred_smiles2 = decode_smiles(pred_tokens2)
        
            final_score, tg_score, er_score = reward_score(pred_smiles1, pred_smiles2, actual_tg[i].numpy(), actual_er[i].numpy())
            scores.append(final_score)     
        except Exception as e:
            print("\nTrain 2 Error in reaction reward:", e)
            return 0.0
    print("Scores: ", scores)   
    print("Property Reward: ", np.mean(scores))
    return np.mean(scores)

def custom_training_loop(model, train_data, val_data=None, epochs=1):
    """
    Custom training loop with custom loss function
    """
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Unpack training data
    X, y = train_data
    reaction_normalizer = RewardNormalizer()
    property_normalizer = RewardNormalizer()
    
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
            'er_list': X['er_list'],
            'tg_list': X['tg_list']
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
        alpha = min((epoch + 1) / epochs, 0.8)
        progress_bar = tf.keras.utils.Progbar(steps_per_epoch)
        
        # Initialize epoch metrics
        epoch_loss = tf.keras.metrics.Mean()
        
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(x_batch, training=True)
                
                # Calculate losses for each decoder
                loss_decoder1 = custom_loss(y_batch['decoder1'], predictions[0])
                loss_decoder2 = custom_loss(y_batch['decoder2'], predictions[1])

                reaction_reward = calculate_reaction_reward(y_batch['decoder1'],y_batch['decoder2'],predictions[0],predictions[1])
                print("Reaction Reward: ", reaction_reward)
                print("Loss Decoder 1: ", loss_decoder1)
                print("Loss Decoder 2: ", loss_decoder2)    
                print("Total Loss: ", loss_decoder1+loss_decoder2)

                property_reward = calculate_property_reward(predictions[0],predictions[1],x_batch['tg_list'],x_batch['er_list'])
                print("Property Reward: ", property_reward)
                
                # Combined loss
                # total_loss = loss_decoder1 + loss_decoder2 - 0.8*reaction_reward - 0.8*property_reward
                # print("After reward Total Loss: ", total_loss)
                norm_reaction_reward = reaction_normalizer.normalize(reaction_reward)
                norm_property_reward = property_normalizer.normalize(property_reward)
                
                print("Raw Reaction Reward:", reaction_reward)
                print("Normalized Reaction Reward:", norm_reaction_reward)
                print("Raw Property Reward:", property_reward)
                print("Normalized Property Reward:", norm_property_reward)
                print("Current alpha:", alpha)
                
                # Combined loss with normalized rewards and adaptive weighting
                total_loss = loss_decoder1 + loss_decoder2 - alpha * (norm_reaction_reward + norm_property_reward)
                print("After reward Total Loss:", total_loss)
            
            # Calculate and apply gradients
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update metrics
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
            'decoder_input2': X_val['decoder_input2'],
            'er_list': X_val['er_list'],
            'tg_list': X_val['tg_list']
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
    file_path = os.path.join(root_dir, 'Dataset', "unique_smiles_Er.xlsx")
    saved_new_model = os.path.join(root_dir, "property_based_rl_model")
    
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
                epochs=Constants.EPOCHS
            )
            
        print('Training completed successfully')
            
            # Save the model
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
        prediction_model.load_weights(weights_path)
            
            # Generate example predictions
        monomer1_list, monomer2_list, er_list, tg_list = process_dual_monomer_data(file_path)
        group_combination = [["C=C", "C=C(C=O)"]] #[["C=C", "C=C(C=O)"], ["C=C", "CCS"], ["C1OC1", "NC"], ["C=C", "OH"]]
        monomer1_list,monomer2_list = random.sample(monomer1_list,1), random.sample(monomer2_list,1)
        for smiles1, smiles2 in zip(monomer1_list, monomer2_list):
            tg = np.random.randint(50, 200)
            er = np.random.randint(tg + 50, 200)  
            print(f"ER: {er}, TG: {tg}")# Ensure er is at least 50 higher than tg
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
                        group_smarts2=group_smarts2,
                        er=er,
                        tg=tg
                    )
                    
                    print("\nValid pairs generated:")
                    for pair in valid:
                        print(f"Monomer 1: {pair[0]}")
                        print(f"Monomer 2: {pair[1]}")
                        print("-" * 40)
    
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise