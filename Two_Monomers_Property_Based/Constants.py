GROUP_VOCAB = {"C1OC1": 0, "NC": 1,"Nc":2, "C=C":3, "CCS":4,"C=C(C=O)":5, "c1ccccc1":6,"C1=CC=CC=C1":7}
GROUP_SIZE = len(GROUP_VOCAB)
EMBEDDING_DIM = 64
LATENT_DIM = 128
#VOCAB_SIZE = 21
EPOCHS = 1
BATCH_SIZE = 16

EPOXY_SMARTS = "[OX2]1[CX3][CX3]1"    # Epoxy group
IMINE_SMARTS = "[NX2]=[CX3]"          # Imine group
VINYL_SMARTS = "C=C"                  # Vinyl group
THIOL_SMARTS = "CCS"                  # Thiol group
ACRYL_SMARTS = "C=C(C=O)"             # Acrylic group
BEZEN_SMARTS = "c1ccccc1"


NOISE_CONFIG = {
    'gaussian': {'enabled': True, 'level': 0.1},
    'dropout': {'enabled': False, 'rate': 0.1},
    'swap': {'enabled': True, 'rate': 0.05},
    'mask': {'enabled': False, 'rate': 0.05}
}
    


VALID_PAIRS_FILE = "valid_pairs_during_training.json"
GENERATED_ALL_PREDICT_FILE = "generated_predict_smiles.json"
GENERATED_VALID_PREDICTED_FILE = "generated_training_smiles.json"

MODEL_PATH = "property_based_rl_model"
PRETRAINED_MODEL_PATH = "saved_models_rl_gpu_4"
DATA_PATH = "unique_smiles_Er.xlsx"


VOCAB_PATH = 'Two_Monomers_Group/tokenizers/vocab_1000/vocab.txt'#'tokenizers/vocab_1000/vocab.txt'
TOKENIZER_PATH = 'Two_Monomers_Group/tokenizers/vocab_1000/tokenizer'#tokenizers/vocab_1000/tokenizer'

VOCAB_PATH = 'Two_Monomers_Group/tokenizers_updated/vocab_1000/vocab.txt'#'tokenizers/vocab_1000/vocab.txt'
TOKENIZER_PATH = 'Two_Monomers_Group/tokenizers_updated/vocab_1000/tokenizer'#tokenizers/vocab_1000/tokenizer'



