from Data_Process_with_prevocab_gen import *
from Sample_Predictor import *


def calculate_reward_score(smiles_list,chemical_groups):
    functional_score = 0
    for smile in smiles_list:
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
        functional_score += intersection / max(union, 1)
        return functional_score
   
if __name__ == "__main__":
    # Test the reward score calculation with some example SMILES
    test_smiles = [
        'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'
    ]
    target_chemical_groups = ["C=C"]
    
    score=calculate_reward_score(test_smiles,target_chemical_groups)
    print(score)
    

