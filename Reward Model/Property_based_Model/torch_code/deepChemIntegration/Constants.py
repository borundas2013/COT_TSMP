from rdkit.Chem import BondType
import ast
BATCH_SIZE = 16
EPOCHS = 2
VAE_LR = 5e-4
NUM_ATOMS = 280  # Maximum number of atoms
DATA_FILE_PATH='unique_smiles_er.csv'#'../Data/smiles.xlsx'
OUTPUT_SAVE_PATH='../Data/output.txt'

LATENT_DIM = 160
ATOM_DIM = 30
MAX_EDGES=280
EDGE_DIM=11
NO_PROPERTY=2


EPOXY_GROUP = 100
IMINE_GROUP = 101
VINYL_GROUP = 102
ACRYLATE_GROUP = 104
THIOL_GROUP = 103