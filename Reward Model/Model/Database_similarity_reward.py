
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
# Example known polymers database (SMILES strings)
known_polymer_smiles = [
    'C1COC1CCN',       # Example known polymer
    'CC(C)COC(C)CNCC', # Example known polymer with amine and ether groups
    'O=C(NCCN)N1CCOCC1',
    'C1OC1CCC2OC2NCCN',
    'C1COC1CNCCN',  # Epoxy-amine cured polymer (similar to a network polymer formed by epichlorohydrin and ethylene diamine)
    'CC(C)COC(C)CNCC',  # Another possible polymer
    'NCCOCCNCCOCCN'
    # Add more known polymers here...
]

# Convert known polymer SMILES to RDKit molecules and compute fingerprints
known_polymers = [Chem.MolFromSmiles(smiles) for smiles in known_polymer_smiles]
known_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in known_polymers]


def compute_combined_fingerprint(monomer1_smiles, monomer2_smiles):
    """
    Function to compute the combined fingerprint of two monomers forming a TSMP.
    """
    # Combine the monomers' SMILES into a single string
    combined_smiles = monomer1_smiles + '.' + monomer2_smiles  # Combine using '.' for mixtures

    # Convert to RDKit molecule
    combined_molecule = Chem.MolFromSmiles(combined_smiles)
    if not combined_molecule:
        return None  # Return None if the molecule is invalid

    # Compute fingerprint for the combined molecule
    combined_fingerprint = AllChem.GetMorganFingerprintAsBitVect(combined_molecule, 2, nBits=2048)
    return combined_fingerprint


def compute_similarity_score_for_combined(monomer1_smiles, monomer2_smiles):
    """
    Function to compute the similarity score of a generated two-monomer-based TSMP
    with respect to known polymers in the database.
    """
    # Compute the combined fingerprint of the two monomers
    combined_fingerprint = compute_combined_fingerprint(monomer1_smiles, monomer2_smiles)
    if not combined_fingerprint:
        return 0.0  # Return a similarity score of 0 if the combined molecule is invalid

    # Compute the maximum similarity score against the known polymer database
    max_similarity = 0.0
    for known_fp in known_fingerprints:
        similarity = DataStructs.TanimotoSimilarity(combined_fingerprint, known_fp)
        max_similarity = max(max_similarity, similarity)

    return max_similarity


def database_comparison_reward_for_combined(monomer1_smiles, monomer2_smiles, threshold=0.7):
    """
    Reward function based on the Tanimoto similarity of a generated two-monomer-based TSMP
    to known structures in the database.
    """
    similarity_score = compute_similarity_score_for_combined(monomer1_smiles, monomer2_smiles)
    print(similarity_score)

    # Define reward based on similarity score
    if similarity_score >= threshold:
        reward = 1.0  # High reward for high similarity
    elif similarity_score >= 0.5:
        reward = 0.5  # Moderate reward for medium similarity
    else:
        reward = 0.1  # Low reward for low similarity

    return reward



def similarity_reward_scoring(generated_smiles):
    monomer1_smiles = generated_smiles[0]
    monomer2_smiles = generated_smiles[1]
    reward = database_comparison_reward_for_combined(monomer1_smiles, monomer2_smiles)
    return 0.8#reward
    # monomer1_smiles = 'C1COC1Cl'  # Epoxy-containing monomer
    # monomer2_smiles = 'NCCN'

# Example two monomers for a generated TSMP
# Amine-containing monomer

# Calculate the reward based on database comparison


#print(f"Reward score for the two-monomer-based TSMP: {reward}")