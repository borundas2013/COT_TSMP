GROUP_VOCAB = {"C1OC1": 0, "NC": 1,"Nc":2, "C=C":3, "CCS":4,"C=C(C=O)":5, "c1ccccc1":6,"C1=CC=CC=C1":7}
GROUP_SIZE = len(GROUP_VOCAB)
EMBEDDING_DIM = 64
LATENT_DIM = 128
#VOCAB_SIZE = 21
EPOCHS = 2#200
BATCH_SIZE = 64

MODEL_SAVED_DIR_CURRENT_WEIGHT = "saved_models_current"
MODEL_SAVED_DIR_HIGH_WEIGHT = "saved_models" #[1.0,1.0,1.0,1.0] #[1.,0.5,0.5]  
MODEL_SAVED_DIR_NEW_MODEL = "saved_models_new_2000" #[1.0,1.0,0.8]

GENERATED_TRAINING_SMILES_DIR_1 = "generated_training_smiles_2000.txt"


TRAINING_FILE = 'CombinedData_K_CH_DE_unique.csv'
VOCAB_PATH = 'tokenizers_updated/vocab_2000/vocab.txt'
TOKENIZER_PATH = 'tokenizers_updated/vocab_2000/tokenizer'


