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
from DiversityRewardModel.diversity_reward import DiversityReward


import json
from datetime import datetime
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw


import json
from pathlib import Path
from datetime import datetime
from RLHFConstants import *
 # adjust this import if needed

class HumanFeedbackCollector:
    def __init__(self, save_path=FEEDBACK_COLLECTOR_PATH):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.feedback_data = self.load_existing_feedback()
        self.reward_model = RewardModel()

    def load_existing_feedback(self):
        """Load existing feedback from file (safe version)"""
        feedback_file = self.save_path / FEEDBACK_COLLECTOR_FILE_NAME
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[Warning] Could not load feedback JSON: {e}")
                print("Resetting file to empty array.")
                # Overwrite the corrupted file safely
                with open(feedback_file, 'w') as f:
                    json.dump([], f)
                return []
        return []


    def make_serializable(self, entry):
        """Ensure all fields in entry are JSON-serializable"""
        return {
            'smiles1': str(entry['smiles1']),
            'smiles2': str(entry['smiles2']),
            'score': float(entry['score']),
            'timestamp': str(entry['timestamp'])
        }

    def save_feedback(self, feedback):
        """Save feedback to file in JSON-safe format"""
        serializable_feedback = [self.make_serializable(entry) for entry in feedback]
        with open(self.save_path /FEEDBACK_COLLECTOR_FILE_NAME, 'w') as f:
            json.dump(serializable_feedback, f, indent=2)

    def get_human_score(self, smiles1, smiles2, group1, group2):
        """Get score from human or reward model"""
        print("\nEvaluating generated molecule pair:")
        print(f"Monomer 1: {smiles1}")
        print(f"Monomer 2: {smiles2}")
       
        
        try:
            # score = float(input("Enter score (1-5): "))
            # if 1 <= score <= 5:
            #     return score
            # else:
            #     print("Invalid score. Please enter a score between 1 and 5.")
            #     return 0.0  # fallback
            
            score = self.reward_model.get_reward(smiles1, smiles2, group1, group2)
            if score is None:
                return 0.0
            return float(score)
        except Exception as e:
            print(f"Error getting reward score: {e}")
            return 0.0  # fallback

    def collect_feedback(self, generated_samples, batch_size=5):
        """Collect human/automated feedback on generated samples"""
        feedback = []
        print(f"\nCollecting feedback for {batch_size} molecule pairs...")

        for i, sample in enumerate(generated_samples[:batch_size]):
            print(f"\nMolecule pair {i+1}/{batch_size}")
            
            score = self.get_human_score(
                sample['smiles1'],
                sample['smiles2'],
                sample['group1'],
                sample['group2']
            )
            
            feedback_entry = {
                'smiles1': sample['smiles1'],
                'smiles2': sample['smiles2'],
                'group1': sample['group1'],
                'group2': sample['group2'],
                'score': score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            feedback.append(feedback_entry)
            self.feedback_data.append(feedback_entry)
            self.save_feedback(self.feedback_data)

        return feedback

