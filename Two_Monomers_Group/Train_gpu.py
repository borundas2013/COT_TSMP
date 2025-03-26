import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import random
import json
from rdkit import Chem

from Constants import *
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from pretrained_weights import *
from saveandload import *
from dual_smile_process import *
from validation_prediction import *
from sample_generator import *
from losswithReward import CombinedLoss
from NewModelApp1 import *

class MonomerTrainer:
    def __init__(self):
        # Setup GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
        # Initialize strategy
        self.strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {self.strategy.num_replicas_in_sync}")
        
        # Setup paths
        self.root_dir = self.get_project_root()
        self.save_dir_abs = os.path.join(self.root_dir, "pretrained_model", "saved_models_rl_gpu_4")
        self.file_path = os.path.join(self.root_dir, 'Data', "smiles_orginal.xlsx")
        self.saved_new_model = os.path.join(self.root_dir, "group_based_rl_model_noise_s_trainable")

    @staticmethod
    def get_project_root():
        """Get the path to the project root directory"""
        current_file = Path(__file__).resolve()
        return current_file.parent

    def load_and_prepare_data(self):
        """Load pretrained model and prepare training data"""
        with self.strategy.scope():
            self.pretrained_model, self.smiles_vocab, self.model_params = load_and_retrain(save_dir=self.save_dir_abs)
            
            # Prepare training data
            all_data = prepare_training_data(
                max_length=self.model_params['max_length'],
                vocab=self.smiles_vocab,
                file_path=self.file_path
            )
            
            # Split data
            train_size = int(0.8 * len(all_data[0]['monomer1_input']))
            self.train_data = (
                {k: v[:train_size] for k, v in all_data[0].items()},
                {k: v[:train_size] for k, v in all_data[1].items()}
            )
            self.val_data = (
                {k: v[train_size:] for k, v in all_data[0].items()},
                {k: v[train_size:] for k, v in all_data[1].items()}
            )

    def create_and_compile_model(self):
        """Create and compile the model"""
        with self.strategy.scope():
            self.model = create_model(
                self.model_params['max_length'], 
                len(self.smiles_vocab),
                self.pretrained_model
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                loss=CombinedLoss()
            )
            print("\nModel summary:")
            self.model.summary()

    def generate_complete_smiles(self, x_batch):
        """Generate complete SMILES sequences"""
        if isinstance(x_batch, tuple):
            x_batch = x_batch[0]
        predictions = self.model(
            {
                'monomer1_input': x_batch['monomer1_input'],
                'monomer2_input': x_batch['monomer2_input'],
                'group_input': x_batch['group_input'],
                'decoder_input1': x_batch['decoder_input1'],
                'decoder_input2': x_batch['decoder_input2']
            }, 
            training=True
        )
        return predictions[0], predictions[1]

    def train(self, epochs=Constants.EPOCHS):
        """Main training loop"""
        # Setup training parameters
        num_gpus = self.strategy.num_replicas_in_sync
        base_batch_size = Constants.BATCH_SIZE
        global_batch_size = base_batch_size * num_gpus
        
        # Create distributed dataset
        train_dataset = self.create_dataset(self.train_data, global_batch_size)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'loss': [], 'val_loss': []}
        best_model_path = None
        best_params_path = None
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            avg_loss = self.train_epoch(train_dist_dataset, epoch)
            history['loss'].append(avg_loss)
            
            # Validation
            if self.val_data is not None:
                val_loss = self.validate_epoch()
                history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    patience_counter = 0
                    best_model_path, best_params_path = save_model(
                        self.model,
                        self.model_params,
                        save_dir=self.saved_new_model
                    )
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve from {best_val_loss:.4f}. Patience counter: {patience_counter}")
                    if patience_counter >= Constants.PATIENCE:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return history, best_model_path, best_params_path
                    

    def train_epoch(self, dataset, epoch):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for x_batch in dataset:
            # Run the distributed training step
            per_replica_losses = self.strategy.run(self.train_step, args=(x_batch,))
            # Reduce the loss across all replicas
            loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            
            total_loss += loss
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Batch {num_batches}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        return avg_loss

    def train_step(self, inputs):
        """Single training step within the strategy.run context"""
        x_batch, y_batch = inputs
        
        with tf.GradientTape() as tape:
            # Forward pass
            generated_tokens1, generated_tokens2 = self.generate_complete_smiles(x_batch)
            
            # Calculate loss
            loss = self.calculate_loss(x_batch, generated_tokens1, generated_tokens2)
            # Scale the loss by number of replicas for proper gradient calculation
            scaled_loss = loss / self.strategy.num_replicas_in_sync
        
        # Get gradients and apply them in a distributed way
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    
    def calculate_reaction_reward(self, y_true1, y_true2, pred1, pred2):
        """Calculate reaction reward for batch predictions."""

        def process_single_sample(inputs):
            """Processes a single batch element and computes reaction reward."""
            y_t1, y_t2, p1, p2 = inputs

            # Get predicted and true tokens
            pred_tokens1 = tf.argmax(p1, axis=-1)
            pred_tokens2 = tf.argmax(p2, axis=-1)
            true_tokens1 = y_t1#tf.argmax(y_t1, axis=-1)
            true_tokens2 = y_t2#tf.argmax(y_t2, axis=-1)

            try:
                # Convert tokens to SMILES
                true_smiles1 = decode_smiles(true_tokens1.numpy().astype(np.int32))
                true_smiles2 = decode_smiles(true_tokens2.numpy().astype(np.int32))
                pred_smiles1 = decode_smiles(pred_tokens1.numpy())
                pred_smiles2 = decode_smiles(pred_tokens2.numpy())

                print("True Smiles 1: ", true_smiles1)
                print("True Smiles 2: ", true_smiles2)
                print("Pred Smiles 1: ", pred_smiles1)
                print("Pred Smiles 2: ", pred_smiles2)
                
                # Save predicted valid SMILES to file if they form valid molecules
                if not pred_smiles1 or not pred_smiles2:
                    return tf.convert_to_tensor(0.0, dtype=tf.float32)
                    
                pred_mol1 = Chem.MolFromSmiles(pred_smiles1)
                pred_mol2 = Chem.MolFromSmiles(pred_smiles2)
                score = 0
                
                if pred_mol1 is not None and pred_mol2 is not None:
                    valid, chemical_groups = check_reaction_validity(true_smiles1, true_smiles2)
                    if valid:
                        score = get_reaction_score(pred_smiles1, pred_smiles2, chemical_groups)
                        
                return tf.convert_to_tensor(score, dtype=tf.float32)
            except Exception as e:
                return tf.convert_to_tensor(0.0, dtype=tf.float32)

        # Vectorized computation for batch-wise processing
        scores = tf.map_fn(
            process_single_sample, 
            (y_true1, y_true2, pred1, pred2), 
            dtype=tf.float32
        )

        # Compute mean reaction reward
        return tf.reduce_mean(scores)
    
    def validate_epoch(self):
        """Run validation on the current model"""
        X, y = self.val_data
        batch_size = Constants.BATCH_SIZE * self.strategy.num_replicas_in_sync
        
        # Initialize metrics
        val_loss = tf.keras.metrics.Mean()
        reconstruction_loss = tf.keras.metrics.Mean()
        reward_metric = tf.keras.metrics.Mean()
        valid_predictions = 0
        total_samples = 0
        
        try:
            # Create validation dataset
            val_dataset = val_dataset = self.create_dataset(self.val_data, batch_size)
            val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)

            # Process each batch
            for x_batch, y_batch in val_dataset:
                try:
                    # Generate predictions
                    generated_smiles1, generated_smiles2 = self.generate_complete_smiles(x_batch)
                    original_monomer1 = x_batch['original_monomer1']
                    original_monomer2 = x_batch['original_monomer2']
                    
                    # Calculate reconstruction loss
                    loss1 = self.model.loss(original_monomer1, generated_smiles1)
                    loss2 = self.model.loss(original_monomer2, generated_smiles2)
                    recon_loss = loss1 + loss2
                    reconstruction_loss.update_state(recon_loss)
                    
                    # Calculate reaction reward
                    reward = self.calculate_reaction_reward(
                        original_monomer1, original_monomer2,
                        generated_smiles1, generated_smiles2
                    )
                    reward_metric.update_state(reward)
                    
                    # Total loss with reward
                    total_loss = recon_loss - 0.8 * reward
                    val_loss.update_state(total_loss)
                    
                    # Count valid predictions
                    pred_tokens1 = tf.argmax(generated_smiles1, axis=-1)
                    pred_tokens2 = tf.argmax(generated_smiles2, axis=-1)
                    
                    # Process valid predictions
                    for i in range(len(pred_tokens1)):
                        total_samples += 1
                        try:
                            decoded_smiles1 = decode_smiles(pred_tokens1[i])
                            decoded_smiles2 = decode_smiles(pred_tokens2[i])
                            if Chem.MolFromSmiles(decoded_smiles1) is not None and Chem.MolFromSmiles(decoded_smiles2) is not None:
                                valid_predictions += 1
                        except Exception as e:
                            print(f"Error decoding SMILES: {e}")
                            continue
                    
                except Exception as batch_error:
                    print(f"Error processing validation batch: {str(batch_error)}")
                    continue
            
            # Calculate validation metrics
            validity_rate = valid_predictions / total_samples if total_samples > 0 else 0
            
            # Print comprehensive validation results
            print("\nValidation Results:")
            print(f"Total Loss: {val_loss.result():.4f}")
            print(f"Reconstruction Loss: {reconstruction_loss.result():.4f}")
            print(f"Average Reward: {reward_metric.result():.4f}")
            print(f"Validity Rate: {validity_rate:.2%}")
            
            return val_loss.result()
            
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return float('inf')  # Return worst possible loss on failure
    
    def create_dataset(self, data, batch_size):
        """Create a TensorFlow dataset from the input data"""
        X, y = data
        
        return tf.data.Dataset.from_tensor_slices((
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
        )).shuffle(buffer_size=1000).batch(batch_size,drop_remainder=False)
    
    def calculate_loss(self, x_batch, generated_tokens1, generated_tokens2):
        """Calculate the total loss including reaction reward"""
        original_monomer1 = x_batch['original_monomer1']
        original_monomer2 = x_batch['original_monomer2']
        
        # Calculate reconstruction losses
        loss_decoder1 = self.model.loss(original_monomer1, generated_tokens1)
        loss_decoder2 = self.model.loss(original_monomer2, generated_tokens2)
        print("loss_decoder1: ", loss_decoder1)
        print("loss_decoder2: ", loss_decoder2)
        
        # Calculate reaction reward
        reaction_reward = self.calculate_reaction_reward(
            original_monomer1, 
            original_monomer2,
            generated_tokens1, 
            generated_tokens2
        )
        
        # Combined loss with reward
        total_loss = loss_decoder1 + loss_decoder2 - 0.8 * reaction_reward
        
        return total_loss

    def run_training(self):
        """Main method to run the entire training process"""
        try:
            print("Loading data...")
            self.load_and_prepare_data()
            
            print("Creating model...")
            self.create_and_compile_model()
            
            print("Starting training...")
            history, best_model_path, best_params_path = self.train()
            
            print("Training completed successfully")
            print(f"Best model saved at: {best_model_path}")
            
            self.best_model_path = best_model_path
            self.best_params_path = best_params_path
            
            return history, best_model_path, best_params_path
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            raise
    def run_predictions(self, weights_path=None):
        """Run predictions with the trained model"""
        try:
            # Use the saved best model path if none provided
            if weights_path is None:
                if hasattr(self, 'best_model_path') and self.best_model_path:
                    weights_path = self.best_model_path
                    print(f"Using best model from training: {weights_path}")
                else:
                    # Look for model weights in the saved model directory
                    weights_path = os.path.join(self.saved_new_model, "weights_model.weights.h5")
                    if os.path.exists(weights_path):
                        print(f"Using existing model: {weights_path}")
                    else:
                        raise ValueError("No model weights found. Please train a model first or provide a weights path.")
            
            with tf.device('/GPU:0'):
                prediction_model = create_model(
                    self.model_params['max_length'], 
                    len(self.smiles_vocab),
                    self.pretrained_model
                )
                
                # Load weights
                print(f"Loading model weights from: {weights_path}")
                prediction_model.load_weights(weights_path)
                
                # Process data for predictions
                monomer1_list, monomer2_list = process_dual_monomer_data(self.file_path)
                
                # Define group combinations
                group_combinations = [
                    ["C=C", "C=C(C=O)"], 
                    ["C=C", "CCS"], 
                    ["[OX2]1[CX3][CX3]1", "[NX2]=[CX3]"], 
                    ["C=C", "[OH]"]
                ]
                
                # Select samples for prediction
                selected_groups = random.choice(group_combinations)
                print(f'Selected groups: {selected_groups}')
                
                # Use a simple approach - direct manual generation instead of using sample_generator
                # to avoid the empty batch issue
                pairs = self.generate_monomer_pairs_directly(
                    prediction_model,
                    monomer1_list[0:1],  # Use just the first monomer to ensure we have data
                    monomer2_list[0:1],
                    selected_groups
                )
                
                # Print results
                print("\nGenerated Monomer Pairs:")
                for i, pair in enumerate(pairs):
                    print(f"Pair {i+1}:")
                    print(f"Monomer 1: {pair[0]}")
                    print(f"Monomer 2: {pair[1]}")
                    print("-" * 40)
                
                return pairs
                
        except Exception as e:
            print(f"Error in predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def sample_with_temperature(self,predictions, temperature):
        """Sample from predictions with temperature scaling"""
        if temperature == 0:
            return np.argmax(predictions)
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)

    def generate_monomer_pairs_directly(self, model, monomer1_list, monomer2_list, desired_groups):
        """Generate monomer pairs directly, bypassing the sample_generator to avoid empty batch issues"""
        pairs = []
        
        # Encode input SMILES
        monomer1 = monomer1_list[0]
        monomer2 = monomer2_list[0]
        
        print(f"Generating predictions for:")
        print(f"Monomer 1: {monomer1}")
        print(f"Monomer 2: {monomer2}")
        print(f"Desired groups: {desired_groups}")
        
        try:
            # Prepare inputs manually
            max_length = self.model_params['max_length']
            
            # Tokenize input SMILES
            tokens,tokens2  = tokenize_smiles([monomer1]),tokenize_smiles([monomer2])
            padded_tokens = pad_token(tokens, max_length, self.smiles_vocab)
            padded_tokens2 = pad_token(tokens2, max_length, self.smiles_vocab)
            input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
            input_seq2 = np.array(padded_tokens2)
            
            # Create group input (this will need adjustment based on your group encoding)
            group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
            group_features = np.array([group_features])
            
            decoder_seq = np.zeros((1, max_length))
            decoder_seq[0, 0] = self.smiles_vocab['<start>']
            decoder_seq2 = np.zeros((1, max_length))
            decoder_seq2[0, 0] = self.smiles_vocab['<start>']

            all_tokens = []
            for temperature in [0.8, 1.0, 1.2]:
                print(f"\nGenerating monomer pairs with temperature {temperature}:")
                print("=" * 80)
                print(f"Input SMILES: {monomer1}, {monomer2}")
                print(f"Desired Groups: {desired_groups}")
                print("-" * 80)
                generated_tokens = []
                generated_tokens2 = []
                for i in range(max_length):
                    output = model.predict({
                        'monomer1_input': input_seq,
                        'monomer2_input': input_seq2,   
                        'group_input': group_features,
                        'decoder_input1': decoder_seq,
                        'decoder_input2': decoder_seq2
                    },verbose=0)    
                    
                    next_token_probs = output[0][0, i]
                    next_token = self.sample_with_temperature(next_token_probs, temperature)
                    
                    next_token_probs2 = output[1][0, i]
                    next_token2 = self.sample_with_temperature(next_token_probs2, temperature)
                    
                    generated_tokens.append(next_token)
                    generated_tokens2.append(next_token2)
                    
                    if next_token == self.smiles_vocab['<end>']:
                        if next_token2 == self.smiles_vocab['<end>']:
                            break   
                    if next_token2 == self.smiles_vocab['<end>']:
                        if next_token == self.smiles_vocab['<end>']:
                            break
                    
                    if i < max_length - 2:
                        decoder_seq[0, i + 1] = next_token
                        decoder_seq2[0, i + 1] = next_token2        
                    
                    all_tokens.append([generated_tokens,generated_tokens2,temperature])
            
            generated_pairs = []
            valid_pairs = []
            
            generated_output_file = "generated_pairs_new_data_noise.json"
            valid_output_file = "valid_pairs_new_data_noise.json"
            all_pairs = []
            valid_pairs_j = []
            try:
                with open(generated_output_file, 'r') as f:
                    all_pairs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_pairs = []

            try:
                with open(valid_output_file, 'r') as f:
                    valid_pairs_j = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                valid_pairs_j = []

            for tokens in all_tokens:
                smiles1,smiles2 = decode_smiles(tokens1[i][0]), decode_smiles(tokens1[i][1])
                temp = tokens[i][2]
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                valid = mol1 is not None and smiles1 != ""
                valid &= mol2 is not None and smiles2 != ""
                generated_pairs.append((smiles1,smiles2,temp))
                if valid:
                    pair_info = {
                        "input_smiles": [monomer1+", "+monomer2],
                        "temperature": temperature,
                        "desired_groups": desired_groups,
                        "monomer1": {
                            "smiles": smiles1,
                        },
                        "monomer2": {
                            "smiles": smiles2, 
                        },
                    }
                    valid_pairs_j.append(pair_info)
                    valid_pairs.append((smiles1, smiles2))
                    with open(valid_output_file, 'w') as f:
                        json.dump(valid_pairs_j, f, indent=4)

                else:
                    pair_info = {
                        "input_smiles": [monomer1+", "+monomer2],
                        "temperature": temperature,
                        "desired_groups": desired_groups,
                        "monomer1": {
                            "smiles": smiles1,
                        },
                        "monomer2": {
                            "smiles": smiles2, 
                        },
                    }
                    all_pairs.append(pair_info)
                    with open(generated_output_file, 'w') as f:
                        json.dump(all_pairs, f, indent=4)
        except Exception as e:
            print(f"Error in generating pairs: {e}")
            raise


            #save_smiles_pair_as_image(pair_info)

            

          
    
    

if __name__ == "__main__":
    trainer = MonomerTrainer()
    history, best_model_path, best_params_path = trainer.run_training()
    trainer.run_predictions(best_model_path)