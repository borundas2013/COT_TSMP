
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sascorer import calculateScore

epoxy_smarts = Chem.MolFromSmarts('C1OC1')
primary_amine_smarts = Chem.MolFromSmarts('[NX3;H2]')
secondary_amine_smarts = Chem.MolFromSmarts('[NX3;H1]([C])')


def check_and_count_group_presence(monomer_smiles, functional_group_smarts):
    monomer_molecule = Chem.MolFromSmiles(monomer_smiles)
    if monomer_molecule:
        return len(monomer_molecule.GetSubstructMatches(functional_group_smarts))
    return 0


def functional_group_reward_score(generated_smiles):
    monomer1_smiles = generated_smiles[0]
    monomer2_smiles = generated_smiles[1]
    # monomer1_smiles = 'C1OC1CC3OC3'  # extract from generated smiles
    # monomer2_smiles = 'NCCN'  # extract from geneated smiles
    monomer1_epoxy_count = check_and_count_group_presence(monomer1_smiles, epoxy_smarts)
    monomer2_primary_amine_count = check_and_count_group_presence(monomer2_smiles, primary_amine_smarts)
    monomer2_secondary_amine_count = check_and_count_group_presence(monomer2_smiles, secondary_amine_smarts)
    monomer2_total_amine_count = monomer2_primary_amine_count + monomer2_secondary_amine_count

    # Determine if the criteria are met
    monomer1_meets_criteria = monomer1_epoxy_count >= 2
    monomer2_meets_criteria = monomer2_total_amine_count >= 2

    # Reward score calculation
    if monomer1_meets_criteria and monomer2_meets_criteria:
        reward_score = 1.0  # Maximum reward if both conditions are met
    elif monomer1_meets_criteria or monomer2_meets_criteria:
        reward_score = 0.5  # Partial reward if only one condition is met
    else:
        reward_score = 0.0  # No reward if neither condition is met

    return reward_score


#reward = functional_group_reward_score("")

#print(f"Reward score for the monomers: {reward}")
