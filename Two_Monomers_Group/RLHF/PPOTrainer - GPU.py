import os
import sys


from Generator.generator import GeneratorModel
from GroupRewardModel.reward_model import RewardModel
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
import tensorflow as tf
from dual_smile_process import *
import RLHFConstants as RLHFConstants
from DiversityRewardModel.diversity_reward import DiversityReward
from FeedbackCollector.feedbackCollector import HumanFeedbackCollector
from ppo_validation import PPOValidator


import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from Data_Process_with_prevocab import decode_smiles
import json
from datetime import datetime
import Constants
from collections import defaultdict
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
#strategy = tf.distribute.MirroredStrategy(gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



def configure_tensorflow():
    """Configure TensorFlow settings before initialization"""
    try:
        # Set threading configurations
        #tf.config.threading.set_inter_op_parallelism_threads(
        ##    len(tf.config.list_physical_devices('CPU'))
        #)
        tf.config.threading.set_intra_op_parallelism_threads(
            len(tf.config.list_physical_devices('CPU'))
        )
        
        # Enable device logging
        tf.debugging.set_log_device_placement(True)
        
        # Configure GPU memory growth if GPUs are available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU memory growth setting failed: {e}")
    except Exception as e:
        print(f"TensorFlow configuration error: {e}")

def setup_strategy():
    """Setup training strategy based on available hardware"""
    try:
        # Check for GPU availability
        gpus = tf.config.list_logical_devices('GPU')
        if len(gpus) > 0:
            print(f"Found {len(gpus)} GPUs: {gpus}")
            strategy = tf.distribute.MirroredStrategy(gpus)
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
        else:
            print("No GPUs found. Using default strategy (CPU)")
            strategy = tf.distribute.get_strategy()
    except Exception as e:
        print(f"Error setting up distributed strategy: {e}")
        print("Falling back to default strategy (CPU)")
        strategy = tf.distribute.get_strategy()

    if tf.config.list_physical_devices('GPU'):
        # GPU-specific optimizations
        tf.config.experimental.enable_tensor_float_32_execution(True)

    return strategy

def print_device_info():
    """Print information about available devices"""
    print("\nDevice Information:")
    print("-" * 50)
    
    # CPU information
    cpu_devices = tf.config.list_physical_devices('CPU')
    print(f"CPUs available: {len(cpu_devices)}")
    
    # GPU information
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpu_devices)}")
    
    if len(gpu_devices) > 0:
        for gpu in gpu_devices:
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"  - {gpu.device_type}: {gpu_details.get('device_name', 'Unknown')}")
            except:
                print(f"  - {gpu.device_type}: Details not available")
    
    # Memory growth settings for GPUs
    if len(gpu_devices) > 0:
        print("\nGPU Memory Growth Settings:")
        for gpu in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  - Memory growth enabled for {gpu.device_type}")
            except RuntimeError as e:
                print(f"  - Warning: {e}")


# Create the strategy


class PPOTrainer:
    def __init__(
        self,
        generator,
        reward_model,
        diversity_reward,
        learning_rate=1e-5,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=1,
        save_dir=RLHFConstants.WEIGHT_PATH,
        human_feedback_collector=None,
        strategy=None
    ):
        
        #self.strategy = tf.distribute.get_strategy()
        #strategy = setup_strategy()
        #self.use_strategy = strategy is not None
        self.strategy = strategy if strategy is not None else tf.distribute.get_strategy()
        print(f"Training using strategy: {self.strategy.__class__.__name__}")
        print(f"Number of replicas: {self.strategy.num_replicas_in_sync}")
        
        # Adjust batch size based on strategy
        self.global_batch_size = batch_size
        if isinstance(self.strategy, tf.distribute.MirroredStrategy):
            self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync
            print(f"Adjusted batch size for distributed training: {self.global_batch_size}")
        else:
            print(f"Using single device batch size: {self.global_batch_size}")
        self.generator = generator
        self.reward_model = reward_model
        self.diversity_reward = diversity_reward
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = self.global_batch_size
        self.human_feedback_collector = human_feedback_collector
        # Setup saving directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir_config = Path(RLHFConstants.CONFIG_PATH)
        self.save_dir_config.mkdir(parents=True, exist_ok=True)

       
        
        # Training history
        self.history = {
            'mean_rewards': [],
            'total_losses': [],
            'policy_losses': [],

        }
        
        with self.strategy.scope():
            # Create optimizer
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
        # Create old generator for PPO
            self.old_generator = GeneratorModel(
                generator.base_model,
                generator.vocab,
                generator.max_length
            )
            self.old_generator.prediction_model.set_weights(generator.prediction_model.get_weights())
        # Save initial configuration
        self.save_config()

    def save_config(self):
        """Save trainer configuration"""
        config = {
            'learning_rate': float(self.optimizer.learning_rate.numpy()),
            'clip_epsilon': self.clip_epsilon,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'batch_size': self.batch_size,
            'creation_date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        with open(self.save_dir_config / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        # Determine filename based on whether this is the best model
        filename = 'best_model.weights.h5' if is_best else f'model_epoch_{epoch}.weights.h5'
        
        # Save model weights with correct file extension
        self.generator.prediction_model.save_weights(
            str(self.save_dir / filename)
        )
        
        # Convert metrics to simple Python types for JSON serialization
        serializable_metrics = {
            'loss': float(metrics['total_loss']),
            'reward': float(metrics['mean_reward'])
        }
        
        # Save training state with only serializable data
        state = {
            'epoch': int(epoch),
            'metrics': serializable_metrics,
            'is_best': is_best
        }
        
        # Save state to JSON
        state_filename = 'best_training_state.json' if is_best else f'training_state_epoch_{epoch}.json'
        with open(self.save_dir / state_filename, 'w') as f:
            json.dump(state, f, indent=4)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        # Load model weights
        self.generator.prediction_model.load_weights(checkpoint_path)
        self.old_generator.prediction_model.set_weights(self.generator.prediction_model.get_weights())
        
        # Load training state
        state_path = checkpoint_path.replace('.weights.h5', '.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.history = state['history']
                return state['epoch']
        return 0

    
    def is_valid_smiles(self, smiles):
        """Check if a SMILES string is valid"""
        try:
            mol=Chem.MolFromSmiles(smiles)
            if mol is None and smiles != "":
                return False
            return True
        except Exception:
            return False
        
    def collect_samples(self, input_data_batch):
        """Collect samples from a batch of input data"""
        samples = []
        
        for i, example in enumerate(input_data_batch):
            smiles1 = example['smiles1']
            smiles2 = example['smiles2']
            group1 = example['group1']
            group2 = example['group2']
            attempts = 0
            while attempts < RLHFConstants.PPO_MAX_ATTEMPT:
                try:
                    result = self.generator.generate(
                        temperatures=RLHFConstants.TEMPERATURES,
                        smiles1=smiles1,
                        smiles2=smiles2,
                        group1=group1,
                        group2=group2,
                        trainging=True
                    )
                    tokens = result['tokens']
                    if tokens:
                        for token in tokens:
                            generated_smiles1 = decode_smiles(token[0])
                            generated_smiles2 = decode_smiles(token[1])
                            samples.append({
                                'smiles1': generated_smiles1,
                                'smiles2': generated_smiles2,
                                'group1': group1,
                                'group2': group2,
                                'input_smiles1': smiles1,
                                'input_smiles2': smiles2
                            })
                       
                         # Successfully generated sample, move to next example
                    
                except Exception as e:
                    print(f"Generation error: {e}")
                attempts += 1
           
            #samples.extend(RLHFConstants.TEST_SAMPLES[:2])
            print("samples:",len(samples))
        
        return samples
    
    def rank_samples_by_reward(self, samples, preferred_ratio=0.3):
        """Rank samples and split into preferred and rejected with consistent ratio"""
        scored_samples = self.reward_model.get_batch_rewards(samples)
        sorted_indices = tf.argsort(scored_samples, direction='DESCENDING')
        
        total_samples = len(samples)
        n_preferred = int(total_samples * preferred_ratio)  # e.g., 30 * 0.3 = 9 preferred samples
        
        # Convert to numpy array for integer indexing
        samples_array = np.array(samples)
        preferred_samples = samples_array[sorted_indices[:n_preferred]]
        rejected_samples = samples_array[sorted_indices[n_preferred:]]
        
        print(f"Total samples: {total_samples}, Preferred: {len(preferred_samples)}, Rejected: {len(rejected_samples)}")
        return preferred_samples, rejected_samples

    def compute_preference_loss(self, samples):
        """Compute preference loss with shape checking"""
        preferred_samples, rejected_samples = self.rank_samples_by_reward(samples)
        
        # Get scores
        preferred_scores = self.reward_model.get_batch_rewards(preferred_samples)
        rejected_scores = self.reward_model.get_batch_rewards(rejected_samples)
        
        # Make sure we have equal number of comparisons
        min_size = min(len(preferred_scores), len(rejected_scores))
        preferred_scores = preferred_scores[:min_size]
        rejected_scores = rejected_scores[:min_size]
        
        print(f"Preference loss shapes - Preferred: {preferred_scores.shape}, Rejected: {rejected_scores.shape}")
        return self.preference_loss(preferred_scores, rejected_scores)

    def preference_loss(self, preferred_scores, rejected_scores):
        """Compute preference loss with shape validation"""
        # Ensure inputs are tensors and same shape
        preferred_scores = tf.convert_to_tensor(preferred_scores, dtype=tf.float32)
        rejected_scores = tf.convert_to_tensor(rejected_scores, dtype=tf.float32)
        print("Preferred scores:", preferred_scores)
        print("Rejected scores:", rejected_scores)
        
        # Add shape assertions
        tf.debugging.assert_equal(tf.shape(preferred_scores), tf.shape(rejected_scores), 
                                message="Preferred and rejected scores must have same shape")
        
        diff = tf.abs(preferred_scores - rejected_scores)
        loss = -tf.math.log_sigmoid(diff)
        print("Preference loss:", tf.reduce_mean(loss),loss)
        return tf.reduce_mean(loss)
    
    def compute_policy_diff(self, old_action_distribution, new_action_distribution, rewards):
        """Compute PPO policy ratio"""
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        
        # Calculate probability ratio
        ratio = new_action_distribution / (old_action_distribution + eps)
        
        # PPO clip
        ratio_clipped = tf.clip_by_value(
            ratio, 
            1 - self.clip_epsilon, 
            1 + self.clip_epsilon
        )

        policy_diff = tf.reduce_mean(tf.minimum(ratio, ratio_clipped), axis=-1)  # shape: [batch_size, 2*max_length]
    
        return tf.reduce_mean(policy_diff, axis=-1)
        
    def train_step_single_device(self, input_data,epoch,num_devices):
        """Perform one training step on a single device"""
        metrics = {
            'total_loss': tf.constant(0.0, dtype=tf.float32),
            #'policy_loss': tf.constant(0.0, dtype=tf.float32),
            'preference_loss': tf.constant(0.0, dtype=tf.float32),   
            'mean_reward': tf.constant(0.0, dtype=tf.float32)
        }

        with tf.GradientTape() as tape:
            samples = self.collect_samples(input_data)

            if not samples:
                print("Warning: No valid samples generated in this step")
                return metrics
            
            
            rewards = self.reward_model.get_batch_rewards(samples)  
            diversity_rewards,_ = self.diversity_reward.calculate_reward(samples)
            
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            diversity_rewards = tf.convert_to_tensor(diversity_rewards, dtype=tf.float32) 
            # print("diversity_rewards:",diversity_rewards)
            # print("rewards:",rewards)
            
            # #values = self.generator.get_values(samples)
            
            # std_rewards = tf.math.reduce_std(rewards)
            # std_diversity_rewards = tf.math.reduce_std(diversity_rewards)
           
            
            # if std_rewards > 0:
            #     rewards = (rewards - tf.reduce_mean(rewards)) / (std_rewards + 1e-8)

            
            # if std_diversity_rewards > 0:
            #     diversity_rewards = (diversity_rewards - tf.reduce_mean(diversity_rewards)) / (std_diversity_rewards + 1e-8)
                
            total_rewards = 0.6 * rewards + 0.4 * diversity_rewards
            total_rewards = (total_rewards - tf.reduce_mean(total_rewards)) / (tf.math.reduce_std(total_rewards) + 1e-8)
            print("Total rewards:", total_rewards)

            preference_loss = self.compute_preference_loss(samples)

           
            #advantages = total_rewards[:, None] - values
            
            old_action_distribution = self.old_generator.get_action_distribution(samples)
            new_action_distribution = self.generator.get_action_distribution(samples)   

            # policy_diff = tf.reduce_mean(
            #     tf.square(new_action_distribution - old_action_distribution),
            #     axis=-1  # Reduce over vocab dimension
            # )
            mini_batch_size = 2
            old_dists = []
            new_dists = []

            for i in range(0, len(samples), mini_batch_size):
                mini_batch = samples[i:i + mini_batch_size]
                old_dists.append(self.old_generator.get_action_distribution(mini_batch))
                new_dists.append(self.generator.get_action_distribution(mini_batch))

            old_action_distribution = tf.concat(old_dists, axis=0)
            new_action_distribution = tf.concat(new_dists, axis=0)

            #policy_diff = tf.reduce_mean(tf.square(new_action_distribution - old_action_distribution), axis=-1)
            #policy_loss = tf.reduce_mean(policy_diff)
            policy_diff = self.compute_policy_diff(old_action_distribution, new_action_distribution, total_rewards)
            #expanded_rewards = tf.expand_dims(total_rewards, -1) 
            policy_loss = -tf.reduce_mean(total_rewards * policy_diff)
            total_loss = preference_loss#policy_loss + 0.5 * preference_loss 
            total_loss = total_loss / num_devices
            
            #print("Policy Loss:", policy_loss.numpy())
            print("Preference Loss:", preference_loss.numpy())
            #print("Mean Policy Diff:", tf.reduce_mean(policy_diff).numpy())#/ self.strategy.num_replicas_in_sync  

            
            
            
                
            metrics = {
                'total_loss': tf.cast(total_loss, tf.float32),
                #'policy_loss': tf.cast(policy_loss, tf.float32),
                'preference_loss': tf.cast(preference_loss, tf.float32),
                'mean_reward': tf.cast(tf.reduce_mean(total_rewards), tf.float32)
            }
        if not tf.math.is_nan(total_loss):
            trainable_vars = self.generator.prediction_model.trainable_variables
            grads = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))   

        return metrics, samples
                    
    
    def train_step(self, input_data,epoch):
        """Perform one training step"""
        # Initialize metrics with default values

        #return self.train_step_single_device(input_data,epoch,1)

        def replica_train_step(input_data):
            metrics = {
            'total_loss': tf.constant(0.0, dtype=tf.float32),
            #'policy_loss': tf.constant(0.0, dtype=tf.float32),
            'value_loss': tf.constant(0.0, dtype=tf.float32),
            'mean_reward': tf.constant(0.0, dtype=tf.float32)
            }

            with tf.GradientTape() as tape:
                metrics, samples = self.train_step_single_device(input_data,epoch,self.strategy.num_replicas_in_sync)

                
            return metrics,samples

        per_replica_metrics, samples = self.strategy.run(replica_train_step, args=(input_data,))
        if epoch % RLHFConstants.FEEDBACK_COLLECT_EPOCH == 0:
                new_feedback = self.human_feedback_collector.collect_feedback(samples,batch_size=len(samples))
                self.reward_model.update_with_feedback(new_feedback)
        if isinstance(self.strategy, tf.distribute.MirroredStrategy):
            reduced_metrics = {}
            for key in per_replica_metrics:
                # Ensure the values are proper tensors before reduction
                values = tf.distribute.get_replica_context().all_gather(
                    per_replica_metrics[key], axis=0
                )
                reduced_metrics[key] = tf.reduce_mean(values).numpy()
        else:
            # For single device, convert tensors to numpy
            reduced_metrics = {
                key: value.numpy() 
                for key, value in per_replica_metrics.items()
            }
        return reduced_metrics
    
    def train(self, input_data, num_epochs=100, 
              save_freq=5, start_epoch=0, batch_size=1):
        """Full training loop with batching"""
        best_reward = float('-inf')
        total_samples = len(input_data)
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_rewards = []
            epoch_losses = []
            
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(total_samples)
            
            # Process data in batches
            for step in tqdm(range(0, total_samples, batch_size)):
                # Get batch indices
                batch_indices = indices[step:min(step + batch_size, total_samples)]
                # Create batch
                batch_data = [input_data[i] for i in batch_indices]
                
                # Train on batch
                metrics = self.train_step(batch_data, epoch+1)
                epoch_rewards.append(metrics['mean_reward'])
                epoch_losses.append(metrics['total_loss'])
                print(f"Step {step//batch_size + 1} metrics:", metrics)
            
            # Compute epoch metrics
            mean_reward = np.mean(epoch_rewards)
            mean_loss = np.mean(epoch_losses)
            
            # Update history
            self.history['mean_rewards'].append(mean_reward)
            self.history['total_losses'].append(mean_loss)
            
            print(f"Mean Reward: {mean_reward:.4f}")
            print(f"Mean Loss: {mean_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1, {
                    'total_loss': mean_loss,
                    'mean_reward': mean_reward
                })
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                self.save_checkpoint(epoch + 1, {
                    'total_loss': mean_loss,
                    'mean_reward': mean_reward
                }, is_best=True)
            
            # Update old generator
            self.old_generator.prediction_model.set_weights(
                self.generator.prediction_model.get_weights()
            )

        # Save final model
        self.save_checkpoint(num_epochs, {
            'total_loss': mean_loss,
            'mean_reward': mean_reward
        })
        
        return self.history

    

def load_training_data(data_path):
    monomer_1,monomer_2 = process_dual_monomer_data(data_path)
    valid_groups = []
    invalid_groups = []

    for i in range(len(monomer_1)):
        if filter_valid_groups(monomer_1[i],monomer_2[i]):
            valid_groups.append([monomer_1[i],monomer_2[i]])
        else:
            invalid_groups.append([monomer_1[i],monomer_2[i]])
    print("valid_groups:",len(valid_groups))
    print("invalid_groups:",len(invalid_groups))
    random_invalid_groups = random.sample(invalid_groups,271)
    valid_groups.extend(random_invalid_groups)
    random.shuffle(valid_groups)
    print("valid_groups:",len(valid_groups))
    input_data = []     
    for i in range(len(valid_groups)):
        groups_1 = extract_group_smarts2(valid_groups[i][0])[0]
        groups_2 = extract_group_smarts2(valid_groups[i][1])[0]
        input_data.append({
            'smiles1': valid_groups[i][0],
            'smiles2': valid_groups[i][1],
            'group1': groups_1,
            'group2': groups_2
        })
    return input_data

# Usage example
if __name__ == "__main__":
    configure_tensorflow()
    print_device_info()
    strategy = setup_strategy()
    save_dir_abs = os.path.join(RLHFConstants.PRETRAINED_MODEL_PATH, RLHFConstants.PRETRAINED_MODEL_NAME)
    file_path = os.path.join(RLHFConstants.DATA_PATH, RLHFConstants.DATA_FILE_NAME)
    
    # Put all model creation and training code inside strategy.scope()
    with strategy.scope():
        # Load base model and create generator
        based_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)
        generator = GeneratorModel(based_model, smiles_vocab, model_params['max_length'])
        generator.prediction_model.load_weights(
            os.path.join(RLHFConstants.GENERATOR_MODEL_PATH, 
                        RLHFConstants.GENERATOR_MODEL_NAME, 
                        "weights_model.weights.h5"),
            skip_mismatch=True
        )
        
        # Create other models
        reward_model = RewardModel()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        diversity_reward = DiversityReward(min_length=25)
        human_feedback_collector = HumanFeedbackCollector(save_path=RLHFConstants.FEEDBACK_COLLECTOR_PATH)
        
        # Create trainer
        trainer = PPOTrainer(
            generator=generator,
            reward_model=reward_model,
            diversity_reward=diversity_reward,
            human_feedback_collector=human_feedback_collector,
            batch_size=RLHFConstants.PPO_BATCH_SIZE,
            strategy=strategy
        )
        
        
        # Load training data
        input_data = load_training_data(file_path)
        
        # Train
        history = trainer.train(
            input_data=input_data,
            num_epochs=RLHFConstants.PPO_EPOCHS,
            save_freq=RLHFConstants.PPO_SAVE_FREQ,
            batch_size=RLHFConstants.PPO_BATCH_SIZE
        )
        
        # Validation
        validator = PPOValidator()
        for data in input_data:
            custom_data = {
                'smiles1': data['smiles1'],
                'smiles2': data['smiles2'],
                'group1': data['group1'],
                'group2': data['group2']
            }
            results = validator.generate_molecules(custom_data)
            print(results)
