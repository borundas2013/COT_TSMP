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
from RLHFConstants import *

import os
import sys
from Generator.generator import GeneratorModel
from LoadPreTrainedModel import load_and_retrain
from Data_Process_with_prevocab import decode_smiles

import os
import sys
import json
from rdkit import Chem
from Generator.generator import GeneratorModel
from LoadPreTrainedModel import load_and_retrain
from Data_Process_with_prevocab import decode_smiles
from RLHFConstants import *

class PPOValidator:
    def __init__(self):
        """Initialize the PPO validator with models and paths"""
        # Load configuration first
        self.config = self._load_config()
        print("Loaded configuration:", self.config)
        
        # Load base model and vocabulary
        self.save_dir_abs = os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME)
        self.based_model, self.smiles_vocab, self.model_params = load_and_retrain(save_dir=self.save_dir_abs)
        
        # Initialize generator
        self.generator = GeneratorModel(
            self.based_model, 
            self.smiles_vocab, 
            self.model_params['max_length']
        )
        
        # Load PPO weights
        self.ppo_weights_path = os.path.join(WEIGHT_PATH, WEIGHT_PATH_NAME)
        self.load_ppo_weights()

    def _load_config(self):
        """Load configuration from config.json"""
        config_path = os.path.join(CONFIG_PATH, 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Config file not found at: {config_path}")
            return self._get_default_config()
        except json.JSONDecodeError:
            print(f"Error parsing config file at: {config_path}")
            return self._get_default_config()

    def _get_default_config(self):
        """Return default configuration if config file is not found"""
        return {
            'learning_rate': 1e-5,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'batch_size': 1
        }

    def load_ppo_weights(self):
        """Load the trained PPO weights"""
        try:
            if not os.path.exists(self.ppo_weights_path):
                raise FileNotFoundError(f"Weight file not found at: {self.ppo_weights_path}")
                
            self.generator.prediction_model.load_weights(
                self.ppo_weights_path, 
                skip_mismatch=True
            )
            print(f"Successfully loaded PPO weights")
        except Exception as e:
            print(f"Error loading PPO weights: {e}")
            raise

    def generate_molecules(self, val_data, temperatures=None):
        """Generate molecules using the trained model"""
            
        if temperatures is None:
            temperatures = TEMPERATURES
            
        # Use batch size from config
        batch_size = self.config.get('batch_size', 1)
        print(f"Using batch size from config: {batch_size}")
        
        results = []
        
     
            
        for i in range(NUM_SAMPLES):
            try:   
                result = self.generator.generate(
                    val_data, 
                    training=False
                )
                
                if result:
                    tokens1 = result['monomer1_output']
                    tokens2 = result['monomer2_output']
                    for i in range(len(tokens1)):
                        token1=tf.argmax(tokens1[i], axis=-1)
                        token2=tf.argmax(tokens2[i], axis=-1)  
                        smiles1 = decode_smiles(token1.numpy())
                        smiles2 = decode_smiles(token2.numpy())
                    
                        if smiles1 != "" and smiles2 != "":
                            # Check if both SMILES are valid molecules
                            mol1 = Chem.MolFromSmiles(smiles1)
                            mol2 = Chem.MolFromSmiles(smiles2)
                            
                            if mol1 is not None and mol2 is not None:
                                generation_data = {
                                    'input': {
                                        'template_smiles1': val_data['original_monomer1'],
                                        'template_smiles2': val_data['original_monomer2'],
                                        'group1': val_data['group_input'][0],
                                        'group2': val_data['group_input'][1],
                                    },
                                    'output': {
                                        'generated_smiles1': smiles1,
                                        'generated_smiles2': smiles2,
                                    }
                                }
                                
                                # Save to JSON file
                                save_path = os.path.join(GENERATED_MOLECULES_PATH, GENERATED_MOLECULES_FILE_NAME)
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                
                                # Load existing data if file exists
                                existing_data = []
                                if os.path.exists(save_path):
                                    with open(save_path, 'r') as f:
                                        existing_data = json.load(f)
                                
                                existing_data.append(generation_data)
                                
                                with open(save_path, 'w') as f:
                                    json.dump(existing_data, f, indent=4)
                                
                                results.append(generation_data)
                                
                                print(f"SMILES 1: {smiles1}")
                                print(f"SMILES 2: {smiles2}")
                                print(f"Saved valid molecules to {save_path}")
                   
                        
            except Exception as e:
                print(f"Error during generation: {e}")
                
        return results

def main():
    validator = PPOValidator()
    results = validator.generate_molecules()
    print(f"\nTotal successful generations: {len(results)}")

if __name__ == "__main__":
    main()