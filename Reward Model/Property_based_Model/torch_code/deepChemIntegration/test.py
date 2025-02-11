# Install RDKit if necessary
# !pip install rdkit-pypi

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import math

# Sample data: Replace with your actual list of SMILES
smiles_list = ["CCO", "CCN", "CCC", "CCOCC", "CCNCC"]

# Convert SMILES to RDKit molecules and generate fingerprints
def get_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
    return fingerprints

fingerprints = get_fingerprints(smiles_list)

### Shannon Entropy Calculation ###
def shannon_entropy(smiles_list):
    # Count occurrences of each SMILES string
    counts = Counter(smiles_list)
    total = len(smiles_list)
    # Calculate Shannon entropy
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy

entropy = shannon_entropy(smiles_list)
print(f"Shannon Entropy: {entropy}")

### Pairwise Dissimilarity Calculation ###
def pairwise_dissimilarity(fingerprints):
    n = len(fingerprints)
    # Initialize an empty matrix
    dissimilarities = np.zeros((n, n))
    
    # Calculate pairwise dissimilarities
    for i in range(n):
        for j in range(i+1, n):
            # Calculate 1 - Tanimoto similarity
            dissimilarity = 1 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            # Fill both upper and lower triangles of the matrix
            dissimilarities[i,j] = dissimilarity
            dissimilarities[j,i] = dissimilarity
    
    mean_dissimilarity = np.mean(dissimilarities)
    return dissimilarities, mean_dissimilarity

dissimilarity_matrix, mean_dissimilarity = pairwise_dissimilarity(fingerprints)
print(f"Pairwise Dissimilarity Matrix:\n{dissimilarity_matrix}")
print(f"Mean Pairwise Dissimilarity: {mean_dissimilarity}")

### Tanimoto Score Calculation ###
def tanimoto_score(fingerprint1, fingerprint2):
    return DataStructs.TanimotoSimilarity(fingerprint1, fingerprint2)

# Example usage for Tanimoto score between first two samples
tanimoto = tanimoto_score(fingerprints[0], fingerprints[1])
print(f"Tanimoto Score between first two samples: {tanimoto}")
