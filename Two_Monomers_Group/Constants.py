GROUP_VOCAB = {"C1OC1": 0, "NC": 1,"Nc":2, "C=C":3, "CCS":4,"C=C(C=O)":5, "c1ccccc1":6,"C1=CC=CC=C1":7}
GROUP_SIZE = len(GROUP_VOCAB)
EMBEDDING_DIM = 64
LATENT_DIM = 128
#VOCAB_SIZE = 21
#EPOCHS = 2
BATCH_SIZE = 32

MODEL_SAVED_DIR_CURRENT_WEIGHT = "saved_models_new"

GENERATED_TRAINING_SMILES_DIR = "generated_training_smiles"

VOCAB_PATH = 'vocab/word_vocab.txt'#'tokenizers/vocab_1000/vocab.txt'
TOKENIZER_PATH = 'vocab/smiles_tokenizer'#tokenizers/vocab_1000/tokenizer'

group_patterns = {
                    "C=C": "C=C",
                    "NC": "[NX2]=[CX3]",
                    "C1OC1": "[OX2]1[CX3][CX3]1",
                    "CCS": "CCS",
                    "C=C(C=O)": "C=C(C=O)"
                }



