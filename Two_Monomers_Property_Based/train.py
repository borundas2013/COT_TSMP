import os

os.environ["KERAS_BACKEND"] = "tensorflow"
from Constants import *
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from pretrained_weights import *
from saveandload import *
from dual_smile_process import *
from validation_prediction import *
import tensorflow as tf
import os
from pathlib import Path
from sample_generator import *


def get_project_root():
    """Get the path to the project root directory"""
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    #while root_dir.name != 'Code' and root_dir.parent != root_dir:
    #    root_dir = root_dir.parent
    return root_dir


def train_with_relationships(model, train_data, val_data=None, epochs=1):
    X, y = train_data
    print(len(X['monomer1_input']))
    
    # Get number of GPUs
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    base_batch_size = 64
    global_batch_size = 1##base_batch_size * num_gpus
    print(f"\nTraining Configuration:")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Base batch size per GPU: {base_batch_size}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Total training samples: {len(X['monomer1_input'])}")
    print(f"Steps per epoch: {len(X['monomer1_input']) // global_batch_size}")
    
    # Create tf.data.Dataset for training
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
    ))
    
    # Batch and prefetch for better performance
    train_dataset = train_dataset.batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
    
    # If validation data is provided
    validation_dataset = None
    if val_data is not None:
        X_val, y_val = val_data
        validation_dataset = tf.data.Dataset.from_tensor_slices((
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
        )).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )
    return history

if __name__ == "__main__":
    root_dir = get_project_root()
    
    # Setup paths
    save_dir_abs = os.path.join(root_dir, "pretrained_model", "saved_models_new")
    file_path = os.path.join(root_dir, 'Data', "smiles.xlsx")
    saved_new_model = os.path.join(root_dir, "saved_models_new")
    
    # Initialize distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    try:
        # Training part with strategy scope
        with strategy.scope():
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
            
            new_model = create_group_relationship_model(
                pretrained_model=pretrained_model,
                max_length=model_params['max_length'],
                vocab_size=len(smiles_vocab),
            )
            
            print("\nModel summary:")
            new_model.summary()
            
            history = train_with_relationships(
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

        # Create a non-distributed model for prediction
        prediction_model = create_group_relationship_model(
            pretrained_model=pretrained_model,
            max_length=model_params['max_length'],
            vocab_size=len(smiles_vocab),
        )
        # Load the trained weights
        prediction_model.load_weights(weights_path)

        # Generation part outside strategy scope
        monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
        sampled_smiles = monomer1_list#random.sample(monomer1_list, 1)
        sampled_smiles2 = monomer2_list#random.sample(monomer2_list, 1)
        print(len(sampled_smiles))
        print(len(sampled_smiles2))
        group_combination = [["C=C", "C=C(C=O)"],["C=C", "CCS"],["C1OC1", "NC"],["C=C", "OH"]]
        generated_data = []

        for index, smiles in enumerate(sampled_smiles):
            for group in group_combination:
                desired_groups = group
                temperatures = [0.2,0.4,0.6,0.8,1.0,1.2,1.4]
                for temp in temperatures:
                    print(f"\nGenerating with temperature: {temp}")
                    smiles_list = [sampled_smiles[index],sampled_smiles2[index]]
                    generated, valid = generate_monomer_pair_with_temperature(
                        model=prediction_model,  # Use the non-distributed model
                        input_smiles=smiles_list,
                        desired_groups=desired_groups,
                        vocab=smiles_vocab,
                        max_length=model_params['max_length'],
                        temperature=temp,
                        num_samples=10
                    )
                    generated_data.append({
                        'input_smiles': smiles,
                        'desired_groups': desired_groups,
                        'temperature': temp,
                        'valid_pairs': valid
                    })
                    print(f"\nValid pairs generated at temperature {temp}:")
                    for pair in valid:
                        print(f"Monomer 1: {pair[0]}")
                        print(f"Monomer 2: {pair[1]}")
                        print("-" * 40)

        print("\nGeneration Summary:")
        for data in generated_data:
            print(f"\nInput SMILES: {data['input_smiles']}")
            print(f"Desired Groups: {data['desired_groups']}")
            print(f"Temperature: {data['temperature']}")
            print(f"Number of valid pairs: {len(data['valid_pairs'])}")

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise