import tensorflow as tf
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

class DiversityReward:
    def __init__(self, min_length=20, diversity_weight=0.5, 
                 batch_diversity_weight=0.3, novelty_weight=0.2, 
                 length_penalty_weight=0.5):
        self.min_length = min_length
        self.diversity_weight = diversity_weight
        self.batch_diversity_weight = batch_diversity_weight
        self.novelty_weight = novelty_weight
        self.length_penalty_weight = length_penalty_weight

    def calculate_tanimoto_similarity(self, smiles1, smiles2, input_smiles1, input_smiles2):
        """Calculate Tanimoto similarity between generated and input SMILES pairs"""
        try:
            # Convert SMILES to molecules
            m1 = Chem.MolFromSmiles(smiles1)
            m2 = Chem.MolFromSmiles(smiles2)
            m1_input = Chem.MolFromSmiles(input_smiles1)
            m2_input = Chem.MolFromSmiles(input_smiles2)
            
            # Check for valid molecules
            if not all([m1, m2, m1_input, m2_input]):
                return 0.0, 0.0
            
            # Generate Morgan fingerprints
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 1024)
            fp1_input = AllChem.GetMorganFingerprintAsBitVect(m1_input, 2, 1024)
            fp2_input = AllChem.GetMorganFingerprintAsBitVect(m2_input, 2, 1024)
            
            # Calculate similarities
            sim1 = DataStructs.TanimotoSimilarity(fp1, fp1_input)
            sim2 = DataStructs.TanimotoSimilarity(fp2, fp2_input)
            
            return sim1, sim2
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0, 0.0

    def calculate_length_penalty(self, smiles1, smiles2):
        """Calculate penalty for short SMILES strings"""
        len1 = len(smiles1)
        len2 = len(smiles2)
        
        # Calculate how much shorter than min_length each SMILES is
        penalty1 = max(0.0, 1.0 - (len1 / self.min_length))
        penalty2 = max(0.0, 1.0 - (len2 / self.min_length))
        
        # Average penalty
        length_penalty = (penalty1 + penalty2) / 2.0
        
        print(f"SMILES lengths: {len1}, {len2}")
        print(f"Length penalty: {length_penalty}")
        
        return length_penalty

    def calculate_batch_diversity(self, sample, other_samples):
        """Calculate diversity between a sample and other samples in batch"""
        if not other_samples:
            return 1.0
            
        pairwise_diversities = []
        gen_smiles1, gen_smiles2 = sample['smiles1'], sample['smiles2']
        
        for other in other_samples:
            other_sim1, other_sim2 = self.calculate_tanimoto_similarity(
                gen_smiles1, gen_smiles2,
                other['smiles1'], other['smiles2']
            )
            pairwise_div = 1.0 - (other_sim1 + other_sim2) / 2.0
            pairwise_diversities.append(pairwise_div)
            
        return np.mean(pairwise_diversities)

    def calculate_reward(self, generated_samples, input_data, verbose=True):
        diversity_rewards = []
        
        for i, sample in enumerate(generated_samples):
            gen_smiles1 = sample['smiles1']
            gen_smiles2 = sample['smiles2']
            input_smiles1 = input_data[i]['smiles1']
            input_smiles2 = input_data[i]['smiles2']
            
            # Check for invalid or empty SMILES
            if not gen_smiles1 or not gen_smiles2:
                diversity_rewards.append(0.0)
                continue
                
            # Calculate length penalty
            length_penalty = self.calculate_length_penalty(gen_smiles1, gen_smiles2)
            
            # Calculate similarity with input
            sim1, sim2 = self.calculate_tanimoto_similarity(
                gen_smiles1, gen_smiles2,
                input_smiles1, input_smiles2
            )
            
            # Calculate diversity score
            diversity_score = 1.0 - (sim1 + sim2) / 2.0
            
            # Add novelty bonus if significantly different
            novelty_bonus = 0.2 if diversity_score > 0.4 else 0.0
            
            # Calculate diversity with other generated samples
            other_samples = [s for j, s in enumerate(generated_samples) if j != i]
            batch_diversity = self.calculate_batch_diversity(sample, other_samples)
            
            # Calculate final reward
            base_reward = (
                self.diversity_weight * diversity_score +
                self.batch_diversity_weight * batch_diversity +
                self.novelty_weight * novelty_bonus
            )
            
            final_reward = base_reward - (self.length_penalty_weight * length_penalty)
            final_reward = max(0.0, final_reward)
            
            if verbose:
                self._print_reward_details(
                    i, gen_smiles1, gen_smiles2,
                    diversity_score, batch_diversity,
                    length_penalty, base_reward, final_reward
                )
            
            diversity_rewards.append(final_reward)
        
        return diversity_rewards, tf.convert_to_tensor(final_reward, dtype=tf.float32)

    def _print_reward_details(self, index, smiles1, smiles2, diversity_score,
                            batch_diversity, length_penalty, base_reward, final_reward):
        """Print detailed information about reward calculation"""
        print(f"\nSample {index}:")
        print(f"Generated SMILES: {smiles1}, {smiles2}")
        print(f"Diversity score: {diversity_score:.3f}")
        print(f"Batch diversity: {batch_diversity:.3f}")
        print(f"Length penalty: {length_penalty:.3f}")
        print(f"Base reward: {base_reward:.3f}")
        print(f"Final reward: {final_reward:.3f}")
        print("-" * 50)

# Example usage:
if __name__ == "__main__":
    # Create reward calculator
    reward_calculator = DiversityReward(
        min_length=8,
        diversity_weight=0.5,
        batch_diversity_weight=0.3,
        novelty_weight=0.2,
        length_penalty_weight=0.5
    )
    
    # Test data
    generated_samples = [
        {'smiles1': "CCO", 'smiles2': "CCNC1OC1"},  # Short SMILES
        {'smiles1': "CC1OC1CCCC", 'smiles2': "CCOCCNCC"}  # Longer SMILES
    ]
    
    input_data = [
        {'smiles1': "CCOCC1OC1CCC1OC1C", 'smiles2': "CCOCCCNC1OC1CC1OC1CCC1OC1C"},
        {'smiles1': "CC1OC1C", 'smiles2': "CCOC"}
    ]
    
    # Calculate rewards
    rewards = reward_calculator.calculate_reward(generated_samples, input_data)
    print("\nFinal rewards:", rewards.numpy())


