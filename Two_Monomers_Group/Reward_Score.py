from Data_Process_with_prevocab import *
from rdkit import Chem
from dual_smile_process import *


# def get_reward_score(smile, chemical_groups):
#     if not smile or not chemical_groups:
#         print("Error: Must provide a SMILES string and chemical groups")
#         return 0.0
        
#     mol = Chem.MolFromSmiles(smile)
#     if mol is None:
#         print(f"Invalid SMILES: {smile}")
#         return 0.0
        
#     groups = extract_group_smarts(smile)
#     # Calculate similarity between target and generated functional groups
#     target_set = set(chemical_groups)
#     print("Target set:", target_set)
#     generated_set = set(groups)
#     print("Generated set:", generated_set)
#     intersection = len(target_set.intersection(generated_set))
#     print("Intersection:", intersection)
#     union = len(target_set.union(generated_set))
#     print("Union:", union)
#     score = intersection / max(union, 1)
    
#     print(f"Score: {score}")
#     return score

# def get_reaction_score(smiles1, smiles2, pred_smiles1, pred_smiles2):
#     # Check reaction validity
#     valid, chemical_groups = check_reaction_validity(smiles1, smiles2)
#     if not valid or len(chemical_groups) == 0:
#         print("Invalid reaction or no chemical groups found")
#         return 0.0
#     try:
#         # Calculate scores for original pairs
#         score1 = get_reward_score(smiles1, chemical_groups[0])
#         score2 = get_reward_score(smiles2, chemical_groups[1])
#         original_score = (score1 + score2) / 2
        
#         # Calculate scores for predicted pairs
#         score1_pred = get_reward_score(pred_smiles1, chemical_groups[0])
#         score2_pred = get_reward_score(pred_smiles2, chemical_groups[1])
#         predicted_score = (score1_pred + score2_pred) / 2
        
#         # Calculate penalties
#         def calculate_penalty(pred_smiles, required_groups):
#             """Calculate penalty for missing required groups"""
#             if not pred_smiles:
#                 return 1.0  # Maximum penalty for invalid SMILES
                
#             mol = Chem.MolFromSmiles(pred_smiles)
#             if mol is None:
#                 return 1.0  # Maximum penalty for invalid SMILES
                
#             penalty = 0.0
#             for group in required_groups:
#                 pattern = Chem.MolFromSmarts(group)
#                 if not mol.HasSubstructMatch(pattern):
#                     penalty += 0.5  # Add 0.5 penalty for each missing group
            
#             return min(penalty, 1.0)  # Cap penalty at 1.0
        
#         # Apply penalties
#         penalty1 = calculate_penalty(pred_smiles1, chemical_groups[0])
#         penalty2 = calculate_penalty(pred_smiles2, chemical_groups[1])
#         total_penalty = (penalty1 + penalty2) / 2
        
#         # Calculate final scores with penalties
#         final_score1 = max(0, score1_pred - penalty1)
#         final_score2 = max(0, score2_pred - penalty2)
        
#         # Calculate total score with weights and penalties
#         total_score = (original_score + (final_score1 + final_score2) / 2) / 2
        
#         # Return detailed scores and penalties
#         # return total_score, {
#         #     'valid': True,
#         #     'original_scores': (score1, score2),
#         #     'predicted_scores': (score1_pred, score2_pred),
#         #     'final_scores': (final_score1, final_score2),
#         #     'penalties': (penalty1, penalty2),
#         #     'total_penalty': total_penalty,
#         #     'original_avg': original_score,
#         #     'predicted_avg': predicted_score,
#         #     'chemical_groups': chemical_groups,
#         #     'total_score': total_score
#         # }
#         return total_score
        
#     except Exception as e:
#         print("Error: ", str(e))
#         return 0.0


# if __name__ == "__main__":
#     # Test the reward score calculation with some example SMILES
#     test_smiles = 'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'

    
#     target_chemical_groups = ["C=C", "C1OC1"]
    
#     score=get_reward_score(test_smiles,target_chemical_groups)
#     print(score)

import tensorflow as tf
from rdkit import Chem
from dual_smile_process import *

def get_reward_score(smile, chemical_groups):
    """Calculate reward score based on functional group similarity between target and predicted SMILES."""
    
    if not smile or not chemical_groups:
        tf.print("Error: Must provide a SMILES string and chemical groups")
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
        
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        tf.print(f"Invalid SMILES: {smile}")
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
        
    groups = extract_group_smarts(smile)
    
    if not groups:
        tf.print("Warning: No functional groups found in predicted SMILES.")
        return tf.convert_to_tensor(0.0, dtype=tf.float32)

    # Compute Jaccard similarity between target and generated functional groups
    target_set = set(chemical_groups)
    generated_set = set(groups)

    intersection = len(target_set.intersection(generated_set))
    union = max(len(target_set.union(generated_set)), 1)  # Avoid division by zero

    score = tf.convert_to_tensor(intersection / union, dtype=tf.float32)

    return score

def get_reaction_score(smiles1, smiles2, pred_smiles1, pred_smiles2):
    """Calculate the reaction score considering similarity, penalties, and validity."""
    
    # Validate reaction and extract chemical groups
    valid, chemical_groups = check_reaction_validity(smiles1, smiles2)
    
    if not valid or len(chemical_groups) == 0:
        tf.print("Invalid reaction or no chemical groups found")
        return tf.convert_to_tensor(0.0, dtype=tf.float32)

    try:
        # Compute reward scores for original and predicted SMILES
        score1 = get_reward_score(smiles1, chemical_groups[0])
        score2 = get_reward_score(smiles2, chemical_groups[1])
        original_score = (score1 + score2) / 2.0

        score1_pred = get_reward_score(pred_smiles1, chemical_groups[0])
        score2_pred = get_reward_score(pred_smiles2, chemical_groups[1])
        predicted_score = (score1_pred + score2_pred) / 2.0

        def calculate_penalty(pred_smiles, required_groups):
            """Calculate penalty for missing required groups."""
            if not pred_smiles:
                return tf.convert_to_tensor(1.0, dtype=tf.float32)  # Max penalty for invalid SMILES
                
            mol = Chem.MolFromSmiles(pred_smiles)
            if mol is None:
                return tf.convert_to_tensor(1.0, dtype=tf.float32)  # Max penalty for invalid SMILES
                
            penalty = 0.0
            for group in required_groups:
                pattern = Chem.MolFromSmarts(group)
                if not mol.HasSubstructMatch(pattern):
                    penalty += 0.5  # Add 0.5 penalty for each missing group
            
            return tf.convert_to_tensor(min(penalty, 1.0), dtype=tf.float32)  # Cap penalty at 1.0
        
        # Compute penalties for missing required functional groups
        penalty1 = calculate_penalty(pred_smiles1, chemical_groups[0])
        penalty2 = calculate_penalty(pred_smiles2, chemical_groups[1])
        total_penalty = (penalty1 + penalty2) / 2.0

        # Adjust scores with penalties
        final_score1 = tf.maximum(0.0, score1_pred - penalty1)
        final_score2 = tf.maximum(0.0, score2_pred - penalty2)

        # Compute total weighted score
        total_score = (original_score + (final_score1 + final_score2) / 2.0) / 2.0

        return total_score

    except Exception as e:
        tf.print("Error in get_reaction_score:", str(e))
        return tf.convert_to_tensor(0.0, dtype=tf.float32)

    

