from Data_Process_with_prevocab import *
from rdkit import Chem
from dual_smile_process import *

import tensorflow as tf
from rdkit import Chem
from dual_smile_process import *

def get_reward_score(smile, chemical_groups):
    """Calculate reward score based on functional group similarity between target and predicted SMILES."""
    
    if not smile or not chemical_groups:
        print("Error: Must provide a SMILES string and chemical groups")
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
        
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Invalid SMILES: {smile}")
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
        
    groups = extract_group_smarts(smile)
    
    if not groups:
        print("Warning: No functional groups found in predicted SMILES.")
        return tf.convert_to_tensor(0.0, dtype=tf.float32)

    # Compute Jaccard similarity between target and generated functional groups
    target_set = set(chemical_groups)
    generated_set = set(groups)

    intersection = len(target_set.intersection(generated_set))
    union = max(len(target_set.union(generated_set)), 1)  # Avoid division by zero

    score = tf.convert_to_tensor(intersection / union, dtype=tf.float32)

    return score

def get_reaction_score(pred_smiles1, pred_smiles2, chemical_groups):
    """Calculate the reaction score considering similarity, penalties, and validity."""
    try:
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
        total_score = (final_score1 + final_score2)/2

        return total_score

    except Exception as e:
        print("Error in get_reaction_score:", str(e))
        return tf.convert_to_tensor(0.0, dtype=tf.float32)

    

