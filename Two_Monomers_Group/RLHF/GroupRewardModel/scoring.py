import os
import sys
from pathlib import Path

def setup_path():
    """Add parent directory to Python path to enable imports"""
    # Get the directory containing this file
    current_dir = Path(__file__).parent
    
    # Add parent directories to path
    parent_dir = current_dir.parent  # Two_Monomers_Group
    root_dir = parent_dir.parent     # COT_TSMP
    
    # Add to Python path if not already there
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

setup_path()



from Data_Process_with_prevocab import *
import pandas as pd
from dual_smile_process import *
from rdkit import Chem
from rdkit.Chem import QED


def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None


# def score_group_reactivity(smiles1, smiles2):
#     # Define valid pairings (order matters)
#     pairs = [
#         (Constants.EPOXY_SMARTS, Constants.IMINE_SMARTS, ['C1OC1', 'NC']),
#         (Constants.IMINE_SMARTS, Constants.EPOXY_SMARTS, ['NC', 'C1OC1']),
#         (Constants.VINYL_SMARTS, Constants.THIOL_SMARTS, ['C=C', 'CCS']),
#         (Constants.THIOL_SMARTS, Constants.VINYL_SMARTS, ['CCS', 'C=C']),
#         (Constants.VINYL_SMARTS, Constants.ACRYL_SMARTS, ['C=C', 'C=C(C=O)']),
#         (Constants.ACRYL_SMARTS, Constants.VINYL_SMARTS, ['C=C(C=O)', 'C=C']),
#     ]

#     for smarts1, smarts2, labels in pairs:
#         count1 = count_functional_groups(smiles1, smarts1)
#         count2 = count_functional_groups(smiles2, smarts2)
#         total = count1 + count2

#         # ✅ Valid if both monomers contain ≥ 2 of the required groups
#         if count1 >= 2 and count2 >= 2:
#             score = round(min(1.0, total / 6.0) * 5.0, 2)  # scale to 5
#             return {
#                 "reaction_pair": labels,
#                 "valid": True,
#                 "group_count": total,
#                 "score": score
#             }
#         # ❌ Invalid if both monomers contain < 2 of the required groups
#         if count1 + count2 > 0:
#             score = round(min(1.0, total / 4.0) * 2.5, 2)  # partial score up to ~2.5
#             return {
#                 "reaction_pair": labels,
#                 "valid": False,
#                 "group_count": total,
#                 "score": score
#             }

#     # ❌ No functional groups present
#     return {
#         "reaction_pair": None,
#         "valid": False,
#         "group_count": 0,
#         "score": 0.5  # lowest default score
#     }


def calculate_score(monomer1, monomer2):
    # Check basic SMILES validity
    m1_valid = is_valid_smiles(monomer1)
    m2_valid = is_valid_smiles(monomer2)

    if not (m1_valid and m2_valid):
        return 1.0  # Minimum score for invalid SMILES

    # Get reaction groups and their counts
    reaction_groups, total_count = check_reaction_validity_with_invalid_groups(monomer1, monomer2)
    
    # Base score calculation based on functional group counts
    def calculate_group_score(count):
        if count >= 5:  # Both monomers have 2 or more groups each
            return 5.0
        elif count == 4:  # One monomer has 2, other has 1
            return 4.0
        elif count == 3:  # Both monomers have 1 each
            return 3.0
        elif count == 2:  # Only one monomer has the group
            return 2.0
        else:
            return 1.0

    # Identify the reaction type and score accordingly
    if reaction_groups[0] == 'C1OC1' and reaction_groups[1] == 'NC':  # Epoxy-Imine
        score = calculate_group_score(total_count)
    elif reaction_groups[0] == 'C=C' and reaction_groups[1] == 'CCS':  # Vinyl-Thiol
        score = calculate_group_score(total_count)
    elif reaction_groups[0] == 'C=C' and reaction_groups[1] == 'C=C(C=O)':  # Vinyl-Acryl
        score = calculate_group_score(total_count)
    elif reaction_groups[0] == 'NC' and reaction_groups[1] == 'C1OC1':  # Imine-Epoxy
        score = calculate_group_score(total_count)
    elif reaction_groups[0] == 'CCS' and reaction_groups[1] == 'C=C':  # Thiol-Vinyl
        score = calculate_group_score(total_count)
    elif reaction_groups[0] == 'C=C(C=O)' and reaction_groups[1] == 'C=C':  # Acryl-Vinyl
        score = calculate_group_score(total_count)
    else:
        score = 1.0  # No valid reaction pair found

    # Calculate QED score as a small factor
    mol1 = Chem.MolFromSmiles(monomer1)
    mol2 = Chem.MolFromSmiles(monomer2)
    qed_score = (QED.qed(mol1) + QED.qed(mol2)) / 2

    # Final weighted score (90% functional groups, 10% QED)
    final_score = 0.9 * score + 0.1 * (qed_score * 5)
    return round(final_score, 2), total_count, reaction_groups

def score_two_monomer_sample(monomer1, monomer2):
    count_valid = 0
    count_invalid = 0
    results = []
    for i in range(len(monomer1)):
        score, total_count, reaction_groups = calculate_score(monomer1[i], monomer2[i])
        result = {
            'monomer1': monomer1[i],
            'monomer2': monomer2[i],
            'total_count': total_count,
            'group1': reaction_groups[0],
            'group2': reaction_groups[1],
            'score': score
        }
        results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('Two_Monomers_Group/Data/data_with_score.csv', index=False)
    

if __name__ == "__main__":
    # Load the data
    data_path = "Two_Monomers_Group/Data/smiles_orginal.xlsx"
    monomer_1,monomer_2 = process_dual_monomer_data(data_path)
    print("LENGTH:-> ",len(monomer_1),len(monomer_2))
    score_two_monomer_sample(monomer_1,monomer_2)
    #score_two_monomer_sample(['CCCC2OC2C1OC1CCC3OC3'],['CCNCCCCNCNCC'])


