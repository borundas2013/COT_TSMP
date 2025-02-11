
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sascorer import calculateScore

from rdkit import Chem

# Expanded compatible chemical groups with SMARTS patterns and minimum required counts
compatible_groups = {
    # Epoxy groups react with amines (primary and secondary)
    'epoxy': {'smarts': Chem.MolFromSmarts('C1OC1'), 'min_count': 2, 'compatible_with': ['primary_amine', 'secondary_amine', 'hydroxyl']},

    # Primary amines react with epoxides, carboxylic acids, isocyanates, and acid chlorides
    'primary_amine': {'smarts': Chem.MolFromSmarts('[NX3;H2]'), 'min_count': 2, 'compatible_with': ['epoxy', 'carboxyl', 'isocyanate', 'acid_chloride']},

    # Secondary amines react with epoxides, isocyanates, and acid chlorides
    'secondary_amine': {'smarts': Chem.MolFromSmarts('[NX3;H1]([C])'), 'min_count': 2, 'compatible_with': ['epoxy', 'isocyanate', 'acid_chloride']},

    # Hydroxyl groups react with carboxylic acids, isocyanates, and epoxides
    'hydroxyl': {'smarts': Chem.MolFromSmarts('[OH]'), 'min_count': 2, 'compatible_with': ['carboxyl', 'isocyanate', 'epoxy']},

    # Carboxyl groups react with amines, alcohols, and epoxides
    'carboxyl': {'smarts': Chem.MolFromSmarts('C(=O)[OH]'), 'min_count': 2, 'compatible_with': ['primary_amine', 'secondary_amine', 'hydroxyl', 'epoxy']},

    # Isocyanates react with amines and alcohols (formation of urethanes or ureas)
    'isocyanate': {'smarts': Chem.MolFromSmarts('N=C=O'), 'min_count': 2, 'compatible_with': ['primary_amine', 'secondary_amine', 'hydroxyl']},

    # Acid chlorides react with amines and alcohols
    'acid_chloride': {'smarts': Chem.MolFromSmarts('C(=O)Cl'), 'min_count': 2, 'compatible_with': ['primary_amine', 'secondary_amine', 'hydroxyl']},

    # Thiols react with epoxides, isocyanates, and maleimides
    'thiol': {'smarts': Chem.MolFromSmarts('[SH]'), 'min_count': 2, 'compatible_with': ['epoxy', 'isocyanate', 'maleimide']},

    # Maleimides react with thiols in thiol-Michael reactions
    'maleimide': {'smarts': Chem.MolFromSmarts('C1=CC(=O)NC(=O)C=C1'), 'min_count': 2, 'compatible_with': ['thiol']},

    # Alkenes can undergo radical or ionic polymerization (e.g., with peroxides or Lewis acids)
    'alkene': {'smarts': Chem.MolFromSmarts('C=C'), 'min_count': 2, 'compatible_with': ['radical_initiator']},

    # Radical initiators react with alkenes to initiate polymerization
    'radical_initiator': {'smarts': Chem.MolFromSmiles('[O-][O+]'), 'min_count': 1, 'compatible_with': ['alkene', 'vinyl', 'acrylate']},

    # Vinyl groups can undergo radical polymerization initiated by radical initiators
    'vinyl': {'smarts': Chem.MolFromSmarts('[CH]=C'), 'min_count': 2, 'compatible_with': ['radical_initiator']},

    # Acrylate groups can polymerize under radical conditions with initiators like peroxides
    'acrylate': {'smarts': Chem.MolFromSmarts('C=CC(=O)O'), 'min_count': 2, 'compatible_with': ['radical_initiator']},
}

def check_and_count_group_presence(monomer_smiles, functional_group_smarts):
    """
    Function to check the presence of a functional group in a monomer and count its occurrences.
    """
    monomer_molecule = Chem.MolFromSmiles(monomer_smiles)
    if monomer_molecule:
        return len(monomer_molecule.GetSubstructMatches(functional_group_smarts))
    return 0

def check_compatibility_and_counts(monomer1_smiles, monomer2_smiles):
    """
    Function to check if two monomers have compatible chemical groups with the required minimum count.
    """
    monomer1_compatible = False
    monomer2_compatible = False


    # Check monomer 1 for its groups and their compatibility with monomer 2
    for group_name1, group_info1 in compatible_groups.items():
        group_smarts1 = group_info1['smarts']
        min_count1 = group_info1['min_count']

        # Check if monomer 1 has at least the minimum count of group 1
        monomer1_count = check_and_count_group_presence(monomer1_smiles, group_smarts1)
        if monomer1_count >= min_count1:
            monomer1_compatible =True
            compatible_with_list = group_info1['compatible_with']
            break
        else:
            monomer1_compatible = False
    for group_name2, group_info2 in compatible_groups.items():
        if group_name2 in compatible_with_list:  # Check if group in monomer 2 is compatible
            group_smarts2 = group_info2['smarts']
            min_count2 = group_info2['min_count']

            # Check if monomer 2 has at least the minimum count of the compatible group
            monomer2_count = check_and_count_group_presence(monomer2_smiles, group_smarts2)
            #print(monomer2_smiles,group_smarts2,monomer2_count)
            if monomer2_count >= min_count2:
                monomer2_compatible = True
                break
            else:
                monomer2_compatible=False


    return monomer1_compatible, monomer2_compatible
def compatibility_based_reward_score(generated_smiles):
    monomer1_smiles, monomer2_smiles = generated_smiles[0],generated_smiles[1]
    monomer1_compatible, monomer2_compatible = check_compatibility_and_counts(monomer1_smiles, monomer2_smiles)
    #print(monomer1_compatible,monomer2_compatible)
    # Reward score calculation
    if monomer1_compatible and monomer2_compatible:
        reward_score = 1.0  # Maximum reward if both monomers are compatible and meet the criteria
    elif monomer1_compatible or monomer2_compatible:
        reward_score = 0.5  # Partial reward if only one monomer meets the criteria
    else:
        reward_score = 0.0  # No reward if neither monomer meets the criteria

    return reward_score

# Example generated monomer SMILES
monomer1_smiles = 'CC1OC1CC3OC3'  # Example monomer 1 with epoxy groups
monomer2_smiles = 'NCCN'          # Example monomer 2 with primary amine groups

# Calculate the compatibility-based reward score
#reward = compatibility_based_reward_score([monomer1_smiles, monomer2_smiles])

#nt(f"Compatibility-based reward score for the two monomers: {reward}")
