from Data_Process_with_prevocab_gen import *
from Sample_Predictor import *
from rdkit.Chem import AllChem, DataStructs


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
    
def calculate_diversity_reward(smiles1, smiles2):
    """
    Calculate diversity reward based on Tanimoto similarity between two molecules.
    Returns a score between 0 and 1, where higher values indicate more diversity.
    """
    if not smiles1 or not smiles2:
        return 0.0
        
    try:
        # Convert SMILES to molecules
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
            
        # Calculate Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        if similarity == 0.0 and similarity < 0.5:
            return 1.0
        elif similarity >= 0.5:
            return 1-similarity
        
        
    except:
        return 0.0


   
if __name__ == "__main__":
    # Test the reward score calculation with some example SMILES
    test_smiles = [
        'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'
    ]
    target_chemical_groups = ["C=C"]
    
    score=calculate_reward_score(test_smiles,target_chemical_groups)
    print(score)
    

