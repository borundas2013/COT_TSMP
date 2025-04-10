import json
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import sys
import os
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# Add parent directory to Python path to import dual_smile_process
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dual_smile_process import *

morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)



def calculate_tanimoto(smiles1, smiles2):
    """Calculate Tanimoto similarity between two SMILES strings"""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 1.0  # Return max similarity for invalid SMILES
        
        #fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        #fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        fp1 =morgan_gen.GetFingerprint(mol1)
        fp2 = morgan_gen.GetFingerprint(mol2)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        print("Error in calculate_tanimoto:", str(e))
        return 1.0
    
def check_complexity(true_smiles1, true_smiles2, pred_smiles1, pred_smiles2):
    """
    Compare molecular complexity between true and predicted SMILES pairs.
    Returns True if predicted molecules have higher complexity than true molecules.
    """
    try:
        # Convert SMILES to molecules
        true_mol1 = Chem.MolFromSmiles(true_smiles1)
        true_mol2 = Chem.MolFromSmiles(true_smiles2)
        pred_mol1 = Chem.MolFromSmiles(pred_smiles1) 
        pred_mol2 = Chem.MolFromSmiles(pred_smiles2)

        if any(mol is None for mol in [true_mol1, true_mol2, pred_mol1, pred_mol2]):
            return False,0

        # Calculate complexity metrics for each molecule
        def get_complexity(mol):
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds() 
            num_rings = len(Chem.GetSymmSSSR(mol))
            return num_atoms + num_bonds + (num_rings * 2)  # Weighted sum

        true_complexity = get_complexity(true_mol1) + get_complexity(true_mol2)
        pred_complexity = get_complexity(pred_mol1) + get_complexity(pred_mol2)
        ratio = pred_complexity / true_complexity

        return ratio > 1.0, ratio

    except:
        return False, 0


def check_group_match(true_smiles1, true_smiles2, pred_smiles1, pred_smiles2):
    """
    Check if predicted SMILES have the same functional groups as input SMILES
    Returns True if groups match for both pairs, False otherwise
    """
    # Get functional groups for all SMILES
    true_mol1 = Chem.MolFromSmiles(true_smiles1)
    true_mol2 = Chem.MolFromSmiles(true_smiles2)
    pred_mol1 = Chem.MolFromSmiles(pred_smiles1)
    pred_mol2 = Chem.MolFromSmiles(pred_smiles2)
    
    if true_mol1 is None or true_mol2 is None or pred_mol1 is None or pred_mol2 is None:
        return False
        
    true_groups1 = []
    true_groups2 = []
    pred_groups1 = []
    pred_groups2 = []
    
    # Check each functional group type for true_smiles1
    if count_functional_groups(true_smiles1, Constants.EPOXY_SMARTS) >= 2:
        true_groups1.append("C1OC1")
    if count_functional_groups(true_smiles1, Constants.IMINE_SMARTS) >= 2:
        true_groups1.append("NC")
    if count_functional_groups(true_smiles1, Constants.THIOL_SMARTS) >= 2:
        true_groups1.append("CCS") 
    if count_functional_groups(true_smiles1, Constants.ACRYL_SMARTS) >= 2:
        true_groups1.append("C=C(C=O)")
    if count_functional_groups(true_smiles1, Constants.VINYL_SMARTS) >= 2:
        true_groups1.append("C=C")

    # Check each functional group type for true_smiles2        
    if count_functional_groups(true_smiles2, Constants.EPOXY_SMARTS) >= 2:
        true_groups2.append("C1OC1")
    if count_functional_groups(true_smiles2, Constants.IMINE_SMARTS) >= 2:
        true_groups2.append("NC")
    if count_functional_groups(true_smiles2, Constants.THIOL_SMARTS) >= 2:
        true_groups2.append("CCS")
    if count_functional_groups(true_smiles2, Constants.ACRYL_SMARTS) >= 2:
        true_groups2.append("C=C(C=O)")
    if count_functional_groups(true_smiles2, Constants.VINYL_SMARTS) >= 2:
        true_groups2.append("C=C")

    # Check each functional group type for pred_smiles1
    if count_functional_groups(pred_smiles1, Constants.EPOXY_SMARTS) >= 2:
        pred_groups1.append("C1OC1")
    if count_functional_groups(pred_smiles1, Constants.IMINE_SMARTS) >= 2:
        pred_groups1.append("NC")
    if count_functional_groups(pred_smiles1, Constants.THIOL_SMARTS) >= 2:
        pred_groups1.append("CCS")
    if count_functional_groups(pred_smiles1, Constants.ACRYL_SMARTS) >= 2:
        pred_groups1.append("C=C(C=O)")
    if count_functional_groups(pred_smiles1, Constants.VINYL_SMARTS) >= 2:
        pred_groups1.append("C=C")

    # Check each functional group type for pred_smiles2
    if count_functional_groups(pred_smiles2, Constants.EPOXY_SMARTS) >= 2:
        pred_groups2.append("C1OC1")
    if count_functional_groups(pred_smiles2, Constants.IMINE_SMARTS) >= 2:
        pred_groups2.append("NC")
    if count_functional_groups(pred_smiles2, Constants.THIOL_SMARTS) >= 2:
        pred_groups2.append("CCS")
    if count_functional_groups(pred_smiles2, Constants.ACRYL_SMARTS) >= 2:
        pred_groups2.append("C=C(C=O)")
    if count_functional_groups(pred_smiles2, Constants.VINYL_SMARTS) >= 2:
        pred_groups2.append("C=C")

    # Compare groups for both pairs
    value_1 =set(true_groups1) == set(pred_groups1) and set(true_groups2) == set(pred_groups2)
    value_2 = set(true_groups1) == set(pred_groups2) and set(true_groups2) == set(pred_groups1)
    return value_1 or value_2

def check_monomer_uniqueness(pred_m1, pred_m2, monomer_list1, monomer_list2):
    mol1 = Chem.MolFromSmiles(pred_m1)
    mol2 = Chem.MolFromSmiles(pred_m2)
    result = False
    if mol1 is None or mol2 is None:
        return False    
    if (pred_m1 not in monomer_list1 and pred_m2 not in monomer_list2) and (pred_m1 not in monomer_list2 and pred_m2 not in monomer_list1):
        return True
    
    return False
    
    





def analyze_training_outputs(json_file,min_tanimoto_threshold=0.2,max_tanimoto_threshold=0.6):
    """Analyze training outputs by comparing true vs predicted SMILES"""
    results_good = []
    results_bad = []

    monomer_list1,monomer_list2=process_dual_monomer_data('Two_Monomers_Group/Data/smiles.xlsx')
    print(len(monomer_list1),len(monomer_list2))
    
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract true and predicted SMILES
            true_m1 = data['true_smiles']['monomer1']
            true_m2 = data['true_smiles']['monomer2'] 
            pred_m1 = data['predicted_smiles']['monomer1']
            pred_m2 = data['predicted_smiles']['monomer2']
            tanimoto_m1 = calculate_tanimoto(true_m1, pred_m1)
            tanimoto_m2 = calculate_tanimoto(true_m2, pred_m2)
            tanimoto_avg = (tanimoto_m1 + tanimoto_m2) / 2
            mol1 = Chem.MolFromSmiles(pred_m1)
            mol2 = Chem.MolFromSmiles(pred_m2)
            if mol1 is None or mol2 is None:
                continue

            reaction_valid = check_reaction_validity(pred_m1,pred_m2)
            
            tanimoto_pred = calculate_tanimoto( pred_m1,pred_m2)

            group_match = check_group_match(true_m1, true_m2, pred_m1, pred_m2)
            complexity_match, complexity_ratio = check_complexity(true_m1, true_m2, pred_m1, pred_m2)

            if tanimoto_avg >= min_tanimoto_threshold and tanimoto_avg <= max_tanimoto_threshold:
                if tanimoto_pred >= min_tanimoto_threshold and tanimoto_pred <= max_tanimoto_threshold:
                    if  group_match and complexity_match and check_monomer_uniqueness(pred_m1, pred_m2, monomer_list1, monomer_list2):
                        results_good.append({
                            'true_monomer1': true_m1,
                            'true_monomer2': true_m2,
                            'pred_monomer1': pred_m1,
                            'pred_monomer2': pred_m2,
                            'tanimoto_avg': tanimoto_avg,
                            'complexity_ratio': complexity_ratio
                    })
            else:
                results_bad.append({
                    'true_monomer1': true_m1,
                    'true_monomer2': true_m2,
                    'pred_monomer1': pred_m1,
                    'pred_monomer2': pred_m2,
                    'tanimoto_avg': tanimoto_avg,
                    'complexity_ratio': complexity_ratio
                })
                
           

    return results_good, results_bad

if __name__ == "__main__":

    
    # Example usage
    json_file = "Two_Monomers_Group/output_analyzer/latest2/valid_pairs_during_training_1.json"
    try:
        results_good, results_bad = analyze_training_outputs(json_file)
         # Save results to CSV files
        good_df = pd.DataFrame(results_good)
        bad_df = pd.DataFrame(results_bad)
        good_df.to_csv('Two_Monomers_Group/output_analyzer/latest2/good_results_1.csv', index=False)
        bad_df.to_csv('Two_Monomers_Group/output_analyzer/latest2/bad_results_1.csv', index=False)

        # Print statistics
        print("\nAnalysis Results:")
        print(f"Total samples analyzed: {len(results_good) + len(results_bad)}")
        print(f"Good results: {len(results_good)}")
        print(f"Bad results: {len(results_bad)}")
        
        # Calculate and print average Tanimoto scores
        if results_good:
            avg_good_tanimoto = sum(r['tanimoto_avg'] for r in results_good) / len(results_good)
            print(f"\nGood results statistics:")
            print(f"Average Tanimoto similarity (true vs pred): {avg_good_tanimoto:.3f}")

            
        if results_bad:
            avg_bad_tanimoto = sum(r['tanimoto_avg'] for r in results_bad) / len(results_bad)
            print(f"\nBad results statistics:")
            print(f"Average Tanimoto similarity (true vs pred): {avg_bad_tanimoto:.3f}")
        

            
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Please ensure the training output file exists in the correct location")
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")

