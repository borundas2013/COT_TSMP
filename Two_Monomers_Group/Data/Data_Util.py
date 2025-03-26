from rdkit import Chem
from rdkit.Chem import AllChem
import os
import sys

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (Two_Monomers_Group)
parent_dir = os.path.dirname(current_dir)

# Get the grandparent directory (COT_TSMP)
grandparent_dir = os.path.dirname(parent_dir)

# Add both parent and grandparent to Python path
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from Data_Process_with_prevocab import *
from dual_smile_process import *


import pandas as pd

read_file = pd.read_csv('Two_Monomers_Group/Data/zinc_dataset.csv')
print(read_file.head())
print(read_file.columns)
print(read_file.shape)
monomer1_list = []
monomer2_list = []
reaction_type_list = []
monomer_list = []

for index, row in read_file.iterrows():
    
    if index ==1000:
        break
    for j in range(index+1, 1001):
       
        polymer1 = read_file.iloc[index]['SMILES']
        polymer2 = read_file.iloc[j]['SMILES']
        mol1 = Chem.MolFromSmiles(polymer1)
        mol2 = Chem.MolFromSmiles(polymer2)
        if mol1 is None or mol2 is None:
            continue
        reaction_valid,reaction_type = check_reaction_validity(polymer1,polymer2)
        

        if reaction_valid:
            # Create a dictionary with the data
            monomer_list.append(polymer1+","+polymer2)
            monomer1_list.append(polymer1)
            monomer2_list.append(polymer2)
            reaction_type_list.append(reaction_type)
            
       

    data = {
        'monomer1': monomer1_list,
        'monomer2': monomer2_list,
        'monomer': monomer_list,
        'reaction_type': reaction_type_list,
    }   
    df = pd.DataFrame(data)
    df.to_csv('Two_Monomers_Group/Data/valid_polymer_pairs_updated.csv', 
             mode='a', 
             header=not os.path.exists('Two_Monomers_Group/Data/valid_polymer_pairs_updated.csv'),
             index=False)
    
    

  



# # Define example polymers (in SMILES format)
# polymer1 = "C=CC(=O)OCCCC"  # Example: Acrylic acid
# polymer2 = "C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(C)=O"           # Example: Ethylamine

# # Convert to RDKit molecules
# mol1 = Chem.MolFromSmiles(polymer1)
# mol2 = Chem.MolFromSmiles(polymer2)

# # Check if they have functional groups that could react
# rxn = AllChem.ReactionFromSmarts('[O:1]=[C:2]-[O:3].[N:4]>>[O:1]=[C:2]-[N:4]')  # Ester + Amine → Amide

# # Test reaction
# products = rxn.RunReactants((mol1, mol2))

# if products:
#     print("Reaction is possible. Possible products:")
#     for product in products:
#         for mol in product:
#             print(Chem.MolToSmiles(mol))
# else:
#     print("No reaction possible between these polymers.")
