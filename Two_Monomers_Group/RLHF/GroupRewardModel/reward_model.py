
from pathlib import Path
from rdkit import Chem

# def load_predictor():
#     """Load the trained predictor model"""
#     root_dir = Path(__file__).parent
#     model_dir = root_dir / Constants.MODEL_DIR
#     return GroupRewardScorePredictor(model_path=str(model_dir))

# def predict_score(smiles1: str, smiles2: str, group1: str, group2: str) -> tuple:
#     """Predict Er and Tg for a given SMILES pair"""
#     predictor = load_predictor()
#     return predictor.predict(smiles1, smiles2, group1, group2)



# if __name__ == "__main__":
#     # Example usage
#     smiles1 = 'c3cc(N(CC1CO1)CC2CO2)ccc3Cc6ccc(N(CC4CO4)CC5CO5)cc6'
#     smiles2 = 'CC(N)COCC(C)COCC(C)COCC(C)COCC(C)COCC(C)COCC(C)N'
#     group1 = "C1OC1" # example value
#     group2 = "NC"
    
#     score = predict_score(smiles1, smiles2, group1, group2)
#     print(f"\nPredictions for test pair:")
#     print(f"Monomer 1: {smiles1}")
#     print(f"Monomer 2: {smiles2}")
#     print(f"ER reward score: {score:.2f}")

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
            if reward is None:
                reward = 0
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
    