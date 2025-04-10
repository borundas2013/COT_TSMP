
from pathlib import Path
from rdkit import Chem

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

import tensorflow as tf
from GroupRewardModel.Group_Reward import GroupRewardScorePredictor
from constants import Constants
from pathlib import Path
from rdkit import Chem
import numpy as np
from data_processor import DataProcessor
import json
import RLHF.RLHFConstants as RLHFConstants



class RewardModel(tf.keras.Model):
    def __init__(self):
        super(RewardModel, self).__init__()
        root_dir = Path(__file__).parent
        model_dir = root_dir / Constants.MODEL_DIR
        self.predictor = GroupRewardScorePredictor(model_path=str(model_dir))

    def call(self, inputs):
        """Forward pass for batch prediction"""
        smiles1, smiles2, group1, group2 = inputs
        scores = []
        
        # Batch prediction
        for s1, s2, g1, g2 in zip(smiles1, smiles2, group1, group2):
            score = self.predictor.predict(s1, s2, g1, g2)
            scores.append(score)
            
        return tf.convert_to_tensor(scores)
    
    def update_with_feedback(self, feedback):
        """Update the reward model with new feedback"""
        data_path = RLHFConstants.SCORE_PATH
        data = DataProcessor.load_data(data_path)
        train_data, test_data = DataProcessor.split_data(data)
        print(len(train_data['smiles1']))
        X = []
        y = []
        for entry in feedback:
           train_data['smiles1'].append(entry['smiles1'])
           train_data['smiles2'].append(entry['smiles2'])
           train_data['group1'].append(entry['group1'])
           train_data['group2'].append(entry['group2'])
           train_data['score'].append(entry['score'])

        print(len(train_data['smiles1']))
       
        self.predictor.train(
            train_data,
            validation_split=Constants.VALIDATION_SPLIT,
            epochs=Constants.DEFAULT_EPOCHS
        )
        self.predictor.save_models()
        print(f"Updated reward model with {len(feedback)} new feedback samples")

    def get_reward(self, smiles1, smiles2, group1, group2):
        """Get reward score for a single pair"""
        return self.predictor.predict(smiles1, smiles2, group1, group2)

    def get_batch_rewards(self, samples):
        """Get reward scores for a batch of samples"""
        rewards = []
        for sample in samples:
            reward = self.get_reward(
                sample['smiles1'],
                sample['smiles2'],
                sample['group1'],
                sample['group2']
            )
            # Save valid sample with reward score to json
            if reward is not None:
                sample_with_reward = {
                    'smiles1': sample['smiles1'],
                    'smiles2': sample['smiles2'], 
                    'group1': sample['group1'],
                    'group2': sample['group2'],
                    'reward': float(reward)
                }
                
                # Create directory if it doesn't exist
                os.makedirs(RLHFConstants.SAMPLES_WITH_REWARD_PATH, exist_ok=True)
                
                # Load existing data if file exists
                json_path = os.path.join(RLHFConstants.SAMPLES_WITH_REWARD_PATH, 
                                       RLHFConstants.SAMPLES_WITH_REWARD_FILE_NAME)
                try:
                    with open(json_path, 'r') as f:
                        saved_samples = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    saved_samples = []
                    
                # Append new sample and save
                saved_samples.append(sample_with_reward)
                with open(json_path, 'w') as f:
                    json.dump(saved_samples, f, indent=4)
            if reward is None:
                rewards.append(0.0)
            else:
                rewards.append(reward)
        return tf.convert_to_tensor(rewards)

# Usage example:
if __name__ == "__main__":
    reward_model = RewardModel()
    
    # Single prediction
    smiles1 = 'c3cc(N(CC1CO1)CC2CO2)ccc3Cc6ccc(N(CC4CO4)CC5CO5)cc6'
    smiles2 = 'CC(N)COCC(C)COCC(C)COCC(C)COCC(C)COCC(C)COCC(C)N'
    group1 = "C1OC1"
    group2 = "NC"
    
    score = reward_model.get_reward(smiles1, smiles2, group1, group2)
    print(f"\nPredictions for test pair:")
    print(f"Monomer 1: {smiles1}")
    print(f"Monomer 2: {smiles2}")
    print(f"Reward score: {score:.2f}")
    
    