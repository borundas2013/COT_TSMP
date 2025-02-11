from rdkit.Chem import BondType
import ast
BATCH_SIZE = 16
EPOCHS = 10
VAE_LR = 5e-4
NUM_ATOMS = 160  # Maximum number of atoms
DATA_FILE_PATH='smiles.xlsx'
OUTPUT_SAVE_PATH='output.txt'
MODEL_PATH='Group_Based/weights/group_based_generative_model.weights.h5'

 # Number of atom types
BOND_DIM = 1  # 5  # Number of bond types
LATENT_DIM = 160
#SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br","Si","n","c","o"]'
SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br","Si"]'
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)
ATOM_DIM = len(SMILE_CHARSET)
BOND_MAPPING = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
BOND_MAPPING.update(
    {0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC}
)



SMILE_TO_INDEX = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
INDEX_TO_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
ATOM_MAPPING = dict(SMILE_TO_INDEX)
ATOM_MAPPING.update(INDEX_TO_SMILE)

EPOXY_GROUP = 100
IMINE_GROUP = 101
VINYL_GROUP = 102
ACRYLATE_GROUP = 104
THIOL_GROUP = 103