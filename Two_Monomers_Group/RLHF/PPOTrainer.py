import os
import sys

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Add RLHF directory to Python path
rlhf_dir = os.path.dirname(current_dir)
sys.path.append(rlhf_dir)

# Add Generator directory to Python path 
generator_dir = current_dir
sys.path.append(generator_dir)

from Generator.generator import GeneratorModel
from GroupRewardModel.reward_model import RewardModel
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
import tensorflow as tf
from dual_smile_process import *
from DiversityRewardModel.diversity_reward import DiversityReward


import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from Data_Process_with_prevocab import decode_smiles
import json
from datetime import datetime

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
        save_dir="models/ppo_training"
    ):
        self.generator = generator
        self.reward_model = reward_model
        self.diversity_reward = diversity_reward
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # Setup saving directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'mean_rewards': [],
            'total_losses': [],
            'policy_losses': [],
            'entropy_losses': []
        }
        
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
        
        with open(self.save_dir / 'config.json', 'w') as f:
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
        state_filename = 'best_model_state.json' if is_best else f'training_state_epoch_{epoch}.json'
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

    @tf.function
    def compute_loss(self, old_logprobs, new_logprobs, rewards, entropy):
        """Compute PPO losses"""
        ratio = tf.exp(new_logprobs - old_logprobs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * rewards, clipped_ratio * rewards)
        )
        
        entropy_loss = -tf.reduce_mean(entropy)
        
        total_loss = policy_loss + self.entropy_coef * entropy_loss
        print(total_loss,policy_loss,entropy_loss)
        
        return total_loss, policy_loss, entropy_loss
    
    def is_valid_smiles(self, smiles):
        """Check if a SMILES string is valid"""
        try:
            mol=Chem.MolFromSmiles(smiles)
            if mol is None and smiles != "":
                return False
            return True
        except Exception:
            return False
    
    def train_step(self, input_data):
        """Perform one training step"""
        # Initialize metrics with default values
        metrics = {
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'entropy_loss': 0.0,
            'mean_reward': 0.0
        }

        with tf.GradientTape() as tape:
            # Generate samples using input data
            samples = []
            max_attempts = 1  # Try a few times to get valid SMILES
            
            while len(samples) == 0 and max_attempts > 0:
                # Randomly select an example from input data
                idx = np.random.randint(0, len(input_data))
                example = input_data[idx]
                
                # Get input data for generation
                smiles1 = example['smiles1']
                smiles2 = example['smiles2']
                group1 = example['group1']
                group2 = example['group2']
                
                try:
                    # Generate new molecules
                    tokens, probs, logprobs = self.generator.generate(
                        temperatures=[0.8],
                        smiles1=smiles1,
                        smiles2=smiles2,
                        group1=group1,
                        group2=group2
                    )
                    
                    # Store generated samples
                    generated_smiles1 = 'CC1OC1CCCC'#decode_smiles(tokens[0][0])
                    generated_smiles2 = 'CCOCCNCC'#decode_smiles(tokens[0][1])
                    
                    print("Generated smiles1:", generated_smiles1)
                    print("Generated smiles2:", generated_smiles2)
                    
                    # Validate SMILES strings
                    if generated_smiles1 and generated_smiles2 and self.is_valid_smiles(generated_smiles1) and self.is_valid_smiles(generated_smiles2):  
                        # Additional validation could be added here
                        samples.append({
                            'smiles1': generated_smiles1,
                            'smiles2': generated_smiles2,
                            'group1': group1,
                            'group2': group2,
                            'logprobs': probs
                        })
                except Exception as e:
                    print(f"Generation error: {e}")
                
                max_attempts -= 1

            if not samples:  # If no valid samples were generated
                print("Warning: No valid samples generated in this step")
                return metrics

            try:
                # Get rewards for valid samples
                rewards = []
                diversity_rewards = []
                for sample in samples:
                    try:
                        reward = self.reward_model.get_reward(
                            sample['smiles1'],
                            sample['smiles2'],
                            sample['group1'],
                            sample['group2']
                        )
                        reward = tf.reshape(reward, [-1])
                       
                        rewards.append(float(reward) if reward is not None else 0.0)
                        _,diversity_reward = self.diversity_reward.calculate_reward(
                            samples,
                            input_data
                        )
                        diversity_reward = tf.reshape(diversity_reward, [-1])
                        
                        diversity_rewards.append(float(diversity_reward) if diversity_reward is not None else 0.0)
                    except Exception as e:
                        print(f"Reward calculation error: {e}")
                        rewards.append(0.0)
                        diversity_rewards.append(0.0)

                if not rewards:
                    return metrics
                
                print("rewards:",rewards)
                print("diversity_rewards:",diversity_rewards)
                    
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                diversity_rewards = tf.convert_to_tensor(diversity_rewards, dtype=tf.float32)
                values = self.generator.get_values(samples)
                std_rewards = tf.math.reduce_std(rewards)
                std_diversity_rewards = tf.math.reduce_std(diversity_rewards)
                print("std_rewards:",std_rewards)
                print("std_diversity_rewards:",std_diversity_rewards)

                if std_rewards > 0:
                    rewards = (rewards - tf.reduce_mean(rewards)) / (std_rewards + 1e-8)
                
                if std_diversity_rewards > 0:
                    diversity_rewards = (diversity_rewards - tf.reduce_mean(diversity_rewards)) / (std_diversity_rewards + 1e-8)
                total_rewards = 0.6 * rewards + 0.4 * diversity_rewards
                advantages = total_rewards - values
                print("total_rewards:",total_rewards)
                print("values:",values)
                print("advantages:",advantages)
                print("Rewards:",rewards)
                print("Diversity_rewards:",diversity_rewards)
                

                # Compute losses
                old_logprobs = self.old_generator.get_logprobs_batch(samples)
                new_logprobs = self.generator.get_logprobs_batch(samples)
                entropy = self.generator.get_entropy(samples)
                
                total_loss, policy_loss, entropy_loss = self.compute_loss(
                    old_logprobs, new_logprobs, advantages, entropy
                )

                # Update metrics
                metrics.update({
                    'total_loss': float(total_loss.numpy()),
                    'policy_loss': float(policy_loss.numpy()),
                    'entropy_loss': float(entropy_loss.numpy()),
                    'mean_reward': float(tf.reduce_mean(total_rewards).numpy())
                })

                # Apply gradients if loss is valid
                if not tf.math.is_nan(total_loss):
                    trainable_vars = self.generator.prediction_model.trainable_variables
                    grads = tape.gradient(total_loss, trainable_vars)
                    self.optimizer.apply_gradients(zip(grads, trainable_vars))

                print("Advantage Mean:", tf.reduce_mean(advantages).numpy())
                print("Logprobs (old vs new):", old_logprobs[:3].numpy(), new_logprobs[:3].numpy())

            except Exception as e:
                print(f"Training error: {e}")
                
        return metrics

    def train(self, input_data, num_epochs=100, steps_per_epoch=10, 
              save_freq=5, start_epoch=0):
        """Full training loop"""
        best_reward = float('-inf')
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_rewards = []
            epoch_losses = []
            
            # Training steps
            for step in tqdm(range(steps_per_epoch)):
                metrics = self.train_step(input_data)
                epoch_rewards.append(metrics['mean_reward'])
                epoch_losses.append(metrics['total_loss'])
                print("metrics:",metrics)
            
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
                self.save_checkpoint(epoch + 1, metrics)
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                self.save_checkpoint(epoch + 1, metrics, is_best=True)
            
            # Update old generator
            self.old_generator.prediction_model.set_weights(self.generator.prediction_model.get_weights())

        # Save final model
        self.save_checkpoint(num_epochs, metrics)
        
        return self.history

def load_training_data(data_path):
    monomer_1,monomer_2 = process_dual_monomer_data(file_path)
    input_data = []
    for i in range(len(monomer_1[:2])):
        groups_1 = "C=C"
        groups_2 = "C=C"
        input_data.append({
            'smiles1': monomer_1[i],
            'smiles2': monomer_2[i],
            'group1': groups_1,
            'group2': groups_2
        })
    return input_data

# Usage example
if __name__ == "__main__":
    # Initialize your models

    save_dir_abs = os.path.join("Two_Monomers_Group", "pretrained_model", "saved_models_rl_gpu_3")
    file_path = os.path.join('Two_Monomers_Group', 'Data', "smiles_orginal.xlsx")
    based_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)

    generator = GeneratorModel(based_model, smiles_vocab, model_params['max_length'])
    generator.prediction_model.load_weights(os.path.join("Two_Monomers_Group", "RLHF","Generator","models","group_based_rl_model_n2", "weights_model.weights.h5"),
                                            skip_mismatch=True)
    reward_model = RewardModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    diversity_reward = DiversityReward(min_length=20)
  
    
    # Create trainer
    trainer = PPOTrainer(
        generator=generator,
        reward_model=reward_model,
        diversity_reward=diversity_reward,
        save_dir="models/ppo_training"
    )
    
    # Load training data
    #input_data = load_training_data(file_path)
    input_data = load_training_data(file_path)
    
    # Optional: Load checkpoint
    # start_epoch = trainer.load_checkpoint("models/ppo_training/checkpoints/model_epoch_X.weights.h5")
    
    # Train
    history = trainer.train(
        input_data=input_data,
        num_epochs=1,
        steps_per_epoch=1,
        save_freq=1
        # start_epoch=start_epoch  # If loading from checkpoint
    )

