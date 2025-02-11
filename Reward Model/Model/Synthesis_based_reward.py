

# def synthesis_reward_scoring(geneated_smiles):
#     polymer_smiles = "CCOCCOC=C" #geneated_smile
#     polymer_molecule = Chem.MolFromSmiles(polymer_smiles)
#     sa_score_polymer = calculateScore(polymer_molecule)
#     return round((10-sa_score_polymer)/10, 2)


from rdkit import Chem
from sascorer import calculateScore


def synthesis_reward_scoring(generated_smiles):
    monomer1_smiles=generated_smiles[0]
    monomer2_smiles=generated_smiles[1]
    combined_smiles = monomer1_smiles + '.' + monomer2_smiles  # Use '.' for a mixture or concatenate directly
    combined_molecule = Chem.MolFromSmiles(combined_smiles)
    if not combined_molecule:
        return 0.0  # Return 0 if the molecule is invalid
    sa_score_polymer = calculateScore(combined_molecule)
    reward_score = round((10 - sa_score_polymer) / 10, 2)

    return reward_score


# Example monomers
# monomer1_smiles = 'c3cc(N(CC1CC1)CC2CO2)ccc3OCC4CO4'#'C1COC1Cl'  # Epichlorohydrin (epoxy monomer)
# monomer2_smiles = 'NCCNCCN(CCNCCC(CN)CCN)CCN(CCNCCN)CCN(CCN)CCN'#NCCN'  # Ethylene diamine (amine monomer)
#
# # Calculate the synthesis reward score for the two-monomer-based TSMP
# synthesis_reward = synthesis_reward_scoring_for_combined(monomer1_smiles, monomer2_smiles)
#
# print(f"Synthesis reward score for the two-monomer-based TSMP: {synthesis_reward}")
