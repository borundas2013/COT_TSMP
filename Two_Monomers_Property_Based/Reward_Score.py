from Data_Process_with_prevocab import *
from rdkit import Chem
from dual_smile_process import *

# def get_reward_score(smiles_list, chemical_groups):
#     if len(smiles_list) != 2 or len(chemical_groups) != 2:
#         print("Error: Must provide exactly 2 SMILES strings and 2 sets of chemical groups")
#         return 0.0
        
#     total_score = 0
#     for i, smile in enumerate(smiles_list):
#         mol = Chem.MolFromSmiles(smile)
#         if mol is None:
#             print(f"Invalid SMILES: {smile}")
#             return 0.0
            
#         groups = extract_group_smarts(smile)
#         # Calculate similarity between target and generated functional groups
#         target_set = set(chemical_groups[i])  # Use corresponding group info for each SMILES
#         print(f"Molecule {i+1} Target set:", target_set)
#         generated_set = set(groups)
#         print(f"Molecule {i+1} Generated set:", generated_set)
#         intersection = len(target_set.intersection(generated_set))
#         print(f"Molecule {i+1} Intersection:", intersection)
#         union = len(target_set.union(generated_set))
#         print(f"Molecule {i+1} Union:", union)
#         score = intersection / max(union, 1)
#         total_score += score
        
#     print(f"Total score: {total_score}")
#     return total_score / 2  # Return average score across both molecules

def get_reward_score(smile, chemical_groups):
    if not smile or not chemical_groups:
        print("Error: Must provide a SMILES string and chemical groups")
        return 0.0
        
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Invalid SMILES: {smile}")
        return 0.0
        
    groups = extract_group_smarts(smile)
    # Calculate similarity between target and generated functional groups
    target_set = set(chemical_groups)
    print("Target set:", target_set)
    generated_set = set(groups)
    print("Generated set:", generated_set)
    intersection = len(target_set.intersection(generated_set))
    print("Intersection:", intersection)
    union = len(target_set.union(generated_set))
    print("Union:", union)
    score = intersection / max(union, 1)
    
    print(f"Score: {score}")
    return score

def get_reaction_score(smiles1, smiles2, pred_smiles1, pred_smiles2):
    """
    Calculate reaction score with penalties for missing or incorrect functional groups.
    
    Args:
        smiles1 (str): First original SMILES
        smiles2 (str): Second original SMILES
        pred_smiles1 (str): First predicted SMILES
        pred_smiles2 (str): Second predicted SMILES
    
    Returns:
        tuple: (total_score, dict of detailed scores)
    """
    # Check reaction validity
    valid, chemical_groups = check_reaction_validity(smiles1, smiles2)
    if not valid or len(chemical_groups) == 0:
        print("Invalid reaction or no chemical groups found")
        return 0.0
    try:
        # Calculate scores for original pairs
        score1 = get_reward_score(smiles1, chemical_groups[0])
        score2 = get_reward_score(smiles2, chemical_groups[1])
        original_score = (score1 + score2) / 2
        
        # Calculate scores for predicted pairs
        score1_pred = get_reward_score(pred_smiles1, chemical_groups[0])
        score2_pred = get_reward_score(pred_smiles2, chemical_groups[1])
        predicted_score = (score1_pred + score2_pred) / 2
        
        # Calculate penalties
        def calculate_penalty(pred_smiles, required_groups):
            """Calculate penalty for missing required groups"""
            if not pred_smiles:
                return 1.0  # Maximum penalty for invalid SMILES
                
            mol = Chem.MolFromSmiles(pred_smiles)
            if mol is None:
                return 1.0  # Maximum penalty for invalid SMILES
                
            penalty = 0.0
            for group in required_groups:
                pattern = Chem.MolFromSmarts(group)
                if not mol.HasSubstructMatch(pattern):
                    penalty += 0.5  # Add 0.5 penalty for each missing group
            
            return min(penalty, 1.0)  # Cap penalty at 1.0
        
        # Apply penalties
        penalty1 = calculate_penalty(pred_smiles1, chemical_groups[0])
        penalty2 = calculate_penalty(pred_smiles2, chemical_groups[1])
        total_penalty = (penalty1 + penalty2) / 2
        
        # Calculate final scores with penalties
        final_score1 = max(0, score1_pred - penalty1)
        final_score2 = max(0, score2_pred - penalty2)
        
        # Calculate total score with weights and penalties
        total_score = (original_score + (final_score1 + final_score2) / 2) / 2
        
        # Return detailed scores and penalties
        # return total_score, {
        #     'valid': True,
        #     'original_scores': (score1, score2),
        #     'predicted_scores': (score1_pred, score2_pred),
        #     'final_scores': (final_score1, final_score2),
        #     'penalties': (penalty1, penalty2),
        #     'total_penalty': total_penalty,
        #     'original_avg': original_score,
        #     'predicted_avg': predicted_score,
        #     'chemical_groups': chemical_groups,
        #     'total_score': total_score
        # }
        return total_score
        
    except Exception as e:
        print("Error: ", str(e))
        return 0.0
        # return 0.0, {
        #     'valid': False,
        #     'original_scores': (0.0, 0.0),
        #     'predicted_scores': (0.0, 0.0),
        #     'penalties': (1.0, 1.0),  # Maximum penalty for errors
        #     'error': str(e)
        # }

# Example usage:
def print_reaction_score_details(smiles1, smiles2, pred_smiles1, pred_smiles2):
    """Print detailed reaction score information"""
    score, details = get_reaction_score(smiles1, smiles2, pred_smiles1, pred_smiles2)
    
    print("\nReaction Score Details:")
    print("-" * 50)
    
    if details['valid']:
        print(f"Original SMILES 1: {smiles1}")
        print(f"Original SMILES 2: {smiles2}")
        print(f"Predicted SMILES 1: {pred_smiles1}")
        print(f"Predicted SMILES 2: {pred_smiles2}")
        print("\nScores:")
        print(f"Original Pair: {details['original_scores']}")
        print(f"Predicted Pair (before penalties): {details['predicted_scores']}")
        print(f"Penalties: {details['penalties']}")
        print(f"Final Scores (after penalties): {details['final_scores']}")
        print(f"\nRequired Chemical Groups:")
        for i, groups in enumerate(details['chemical_groups']):
            print(f"Monomer {i+1}: {groups}")
        print(f"\nTotal Score: {details['total_score']:.3f}")
    else:
        print(f"Error: {details['error']}")

if __name__ == "__main__":
    # Test the reward score calculation with some example SMILES
    test_smiles = 'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'

    
    target_chemical_groups = ["C=C", "C1OC1"]
    
    score=get_reward_score(test_smiles,target_chemical_groups)
    print(score)
    

