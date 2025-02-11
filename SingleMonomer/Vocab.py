from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from Data_Process import *
from rdkit import Chem
import re
from Data_Process_with_prevocab_gen import read_smiles_from_file

# Initialize a tokenizer with BPE or WordPiece model
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Define pre-tokenizer (regex-based for SMILES)
#tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# Load your dataset
smiles_data = read_smiles_from_file('CombinedData_K_CH_DE_unique.csv')
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])

# Train on your dataset
tokenizer.train(smiles_data, trainer)

# Save tokenizer to a directory
tokenizer.save("tokenizer_bpe.json")

# Load trained tokenizer
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_bpe.json"
)

# Save in Hugging Face format
fast_tokenizer.save_pretrained("tokenizer_dir")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer_dir")

