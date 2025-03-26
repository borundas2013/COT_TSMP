import json
import pandas as pd
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

def process_json_to_csv(json_file_path, output_csv_path):
    # Read JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract required fields into a list of dictionaries
    processed_data = []
    for entry in data:
        row = {
            'input_smiles_1': entry['input_smiles'][0],
            'input_smiles_2': entry['input_smiles'][1],
            'monomer1_smiles': entry['monomer1']['smiles'],
            'monomer2_smiles': entry['monomer2']['smiles'],
            'From_noise': entry['add_noise']
        }
        processed_data.append(row)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(processed_data)
    df.to_csv(output_csv_path, index=False)


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
    all_monomers = monomer_list1 + monomer_list2
    result = False
    if mol1 is None or mol2 is None:
        return False    
    if (pred_m1 not in monomer_list1 and pred_m2 not in monomer_list2) and (pred_m1 not in monomer_list2 and pred_m2 not in monomer_list1):
        return True
    
    return False

def analyze_training_outputs(file_path,min_tanimoto_threshold=0.2,max_tanimoto_threshold=0.9):
    """Analyze training outputs by comparing true vs predicted SMILES"""
    results_good = []
    results_bad = []

    monomer_list1,monomer_list2=process_dual_monomer_data('Two_Monomers_Group/Data/smiles_orginal.xlsx')
    print(len(monomer_list1),len(monomer_list2))
    
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        # Extract true and predicted SMILES from CSV columns
        true_m1 = row['input_smiles_1']
        true_m2 = row['input_smiles_2']
        pred_m1 = row['monomer1_smiles'] 
        pred_m2 = row['monomer2_smiles']
        from_noise = row['From_noise']
        tanimoto_m1 = calculate_tanimoto(true_m1, pred_m1)
        tanimoto_m2 = calculate_tanimoto(true_m2, pred_m2)
        tanimoto_avg = (tanimoto_m1 + tanimoto_m2) / 2
        mol1 = Chem.MolFromSmiles(pred_m1)
        mol2 = Chem.MolFromSmiles(pred_m2)
        if mol1 is None or mol2 is None:
            continue

        #reaction_valid,reaction_type = check_reaction_validity(pred_m1,pred_m2)
        reaction_valid,reaction_type = check_reaction_validity_new(pred_m1,pred_m2)
        
        tanimoto_pred = calculate_tanimoto(pred_m1,pred_m2)

        group_match = check_group_match(true_m1, true_m2, pred_m1, pred_m2)
        complexity_match, complexity_ratio = check_complexity(true_m1, true_m2, pred_m1, pred_m2)
        size_match = filter_two_monomers(pred_m1, pred_m2)

        check_tanimoto_1 = calculate_tanimoto(true_m1,pred_m1)
        check_tanimoto_2 = calculate_tanimoto(true_m2,pred_m2)
        check_tanimoto_11 = calculate_tanimoto(true_m1,pred_m2)
        check_tanimoto_22 = calculate_tanimoto(true_m2,pred_m1)
        uniuqe_mn= check_tanimoto_1 < 1
        uniuqe_mn1= check_tanimoto_2 < 1 
        uniuqe_mn2= check_tanimoto_11 < 1
        uniuqe_mn3= check_tanimoto_22 < 1

        

        if tanimoto_avg >= min_tanimoto_threshold and tanimoto_avg <= max_tanimoto_threshold:
            if tanimoto_pred >= min_tanimoto_threshold and tanimoto_pred <= max_tanimoto_threshold:
                if  uniuqe_mn and uniuqe_mn1 and uniuqe_mn2 and uniuqe_mn3 and reaction_valid and size_match and check_monomer_uniqueness(pred_m1, pred_m2, monomer_list1, monomer_list2):
                    new_result = {
                        'true_monomer1': true_m1,
                        'true_monomer2': true_m2,
                        'pred_monomer1': pred_m1,
                        'pred_monomer2': pred_m2,
                        'tanimoto_avg': tanimoto_avg,
                        'complexity_ratio': complexity_ratio,
                        'From_noise': from_noise,
                        'reaction_type': reaction_type,
                        'check_tanimoto_1': check_tanimoto_1,
                        'check_tanimoto_2': check_tanimoto_2,
                        'check_tanimoto_11': check_tanimoto_11,
                        'check_tanimoto_22': check_tanimoto_22
                            }

                    # Check if this result is already in results_good
                    is_duplicate = False
                    for existing_result in results_good:
                        if (existing_result['true_monomer1'] == true_m1 and 
                             existing_result['true_monomer2'] == true_m2 and 
                                existing_result['pred_monomer1'] == pred_m1 and 
                                existing_result['pred_monomer2'] == pred_m2):
                                is_duplicate = True
                                break
                    if not is_duplicate:
                            results_good.append(new_result)
        else:
            results_bad.append({
                'true_monomer1': true_m1,
                'true_monomer2': true_m2,
                'pred_monomer1': pred_m1,
                'pred_monomer2': pred_m2,
                'tanimoto_avg': tanimoto_avg,
                'complexity_ratio': complexity_ratio,
                'From_noise': from_noise
            })
           

    return results_good, results_bad


from rdkit import Chem
from rdkit.Chem import AllChem

# Dictionary of reactions
reactions = {
    "Ester + Amine → Amide": '[O:1]=[C:2]-[O:3].[N:4]>>[O:1]=[C:2]-[N:4]',
    "Carboxyl + Alcohol → Ester": '[O:1]=[C:2]-[OH:3].[O:4][C:5]>>[O:1]=[C:2]-[O:4][C:5]',
    "Amine + Epoxy → Hydroxyamine": '[O:1][C:2][C:3].[N:4]>>[O:1][C:2][C:3][N:4]',
    "Isocyanate + Alcohol → Urethane": '[N:1]=[C:2]=[O:3].[O:4][C:5]>>[N:1][C:2]([O:4][C:5])=[O:3]',
    "Thiol + Isocyanate → Thiourethane": '[N:1]=[C:2]=[O:3].[S:4][C:5]>>[N:1][C:2]([S:4][C:5])=[O:3]',
    "Thiol + Epoxy → Thioether + Hydroxyl": '[O:1][C:2][C:3].[S:4]>>[O:1][C:2][C:3][S:4]',
    "Michael Addition (Enone + Amine/Thiolate)": '[C:1]=[C:2][C:3]=[O:4].[N:5]>>[C:1]-[C:2]-[C:3]([N:5])=[O:4]',
    "Radical Polymerization (Vinyl + Vinyl)": '[C:1]=[C:2].[C:3]=[C:4]>>[C:1]-[C:2]-[C:3]-[C:4]',
    "Diels-Alder Reaction (Diene + Dienophile)": '[C:1]=[C:2][C:3]=[C:4].[C:5]=[C:6]>>[C:1][C:2][C:3][C:4][C:5][C:6]',
    "Thiol + Vinyl → Thioether": '[S:1][H].[C:2]=[C:3]>>[S:1][C:2]-[C:3]',
    "Thiol + Hydroxyl → Thioester": '[S:1][H].[O:2][C:3]>>[S:1][C:3]',
    "Aldehyde + Amine → Imine": '[C:1]=[O:2].[N:3]>>[C:1]=[N:3]',
    "Aldehyde + Alcohol → Acetal": '[C:1]=[O:2].[O:3][C:4]>>[C:1]([O:3][C:4])[O:3][C:4]',
    "Hydrolysis of Ester → Carboxyl + Alcohol": '[O:1]=[C:2][O:3][C:4]>>[O:1]=[C:2][O:3].[O:4][C:4]',
    "Amine + Carboxyl → Amide": '[O:1]=[C:2][O:3].[N:4]>>[O:1]=[C:2][N:4]',
    "Lactone Formation (Hydroxy Acid → Lactone)": '[O:1]=[C:2][O:3][C:4]>>[O:1]=[C:2][O:3]',
    "Phenol + Epoxy → Ether": '[O:1][c:2].[O:3][C:4]>>[O:1][c:2][C:4][O:3]',
    "Ring-Opening Metathesis Polymerization (ROMP)": '[C:1]=[C:2]-[C:3]=[C:4]>>[C:1]-[C:2]-[C:3]-[C:4]',
    "Aza-Michael Reaction (Amine + α,β-Unsaturated Ketone)": '[N:1][H].[C:2]=[C:3][C:4]=[O:5]>>[N:1][C:2]-[C:3][C:4]=[O:5]',
}

# Compile the reactions into RDKit objects
compiled_reactions = {name: AllChem.ReactionFromSmarts(smarts) for name, smarts in reactions.items()}

def check_reaction_validity_new(pred_m1, pred_m2):
    """
    Check if the predicted SMILES can undergo a valid reaction.
    Returns True if the reaction is valid, False otherwise.
    """
    try:
        # Convert SMILES to molecules
        mol1 = Chem.MolFromSmiles(pred_m1)
        mol2 = Chem.MolFromSmiles(pred_m2)  
        for reaction_name, reaction in compiled_reactions.items():
            products = reaction.RunReactants((mol1, mol2))
            if products:
                return True, reaction_name
        return False, None
    except Exception as e:
        print(f"Error checking reaction validity: {e}")
        return False, None
    

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

def filter_two_monomers(smiles1, smiles2, min_heavy_atoms=5, min_mol_weight=100):
    """
    Filter out a pair of monomers based on size.

    Args:
        smiles1 (str): First monomer SMILES.
        smiles2 (str): Second monomer SMILES.
        min_heavy_atoms (int): Minimum number of heavy atoms.
        min_mol_weight (float): Minimum molecular weight.

    Returns:
        bool: True if both monomers pass the filter, False otherwise.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 and mol2:
        # First monomer criteria
        num_heavy_atoms_1 = mol1.GetNumHeavyAtoms()
        mol_weight_1 = Descriptors.MolWt(mol1)

        # Second monomer criteria
        num_heavy_atoms_2 = mol2.GetNumHeavyAtoms()
        mol_weight_2 = Descriptors.MolWt(mol2)

        # Pass filter if both monomers are big enough
        if (num_heavy_atoms_1 >= min_heavy_atoms and mol_weight_1 >= min_mol_weight and
            num_heavy_atoms_2 >= min_heavy_atoms and mol_weight_2 >= min_mol_weight):
            if rdMolDescriptors.CalcNumRings(mol1) > 0 or rdMolDescriptors.CalcNumRings(mol2) > 0:
                return True
    return False




if __name__ == "__main__":
    json_file_path = "Two_Monomers_Group/output_analyzer/latest2/generated_valid_predict_smiles.json"  # Replace with your JSON file path
    output_csv_path = "Two_Monomers_Group/output_analyzer/latest2/output_generated_valid_predict_smiles.csv"  # Replace with desired output path
    process_json_to_csv(json_file_path, output_csv_path)
    results_good, results_bad = analyze_training_outputs(output_csv_path)
    print(f"Number of good results: {len(results_good)}")
    print(f"Number of bad results: {len(results_bad)}")

    # Save results to CSV files
    good_df = pd.DataFrame(results_good)
    bad_df = pd.DataFrame(results_bad)
    good_df.to_csv('Two_Monomers_Group/output_analyzer/latest2/good_results_generated_valid_predict_smiles.csv', index=False)
    bad_df.to_csv('Two_Monomers_Group/output_analyzer/latest2/bad_results_generated_valid_predict_smiles.csv', index=False)
