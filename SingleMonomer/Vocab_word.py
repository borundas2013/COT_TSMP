from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import PreTrainedTokenizerFast
from Data_Process_with_prevocab_gen import read_smiles_from_file

# Initialize WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece(
    unk_token="[UNK]",
    max_input_chars_per_word=100  # Increased for long SMILES
))

# Define chemical patterns for tokenization
chemical_patterns = (
    r"(\[|\]|"                # Brackets
    r"Br|Cl|"                 # Common two-letter elements
    r"OH|NH|"                 # Common groups
    r"[CNOPSFIHBr]|"         # Single-letter elements
    r"[0-9]|"                # Numbers
    r"=|#|-|\(|\)|\+|"       # Operators and parentheses
    r"\\|\/|@|\.|:|"         # Special characters
    r"c1|n1|o1|s1)"          # Common aromatic markers
)

# Configure pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(pattern=chemical_patterns, behavior="isolated"),
    pre_tokenizers.Whitespace()
])

# Configure post-processor
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

# Configure trainer
trainer = trainers.WordPieceTrainer(
    vocab_size=3000,              # Adjust based on your needs
    min_frequency=2,              # Minimum frequency for a token
    special_tokens=[
        "[PAD]",                  # Padding token
        "[CLS]",                  # Start of sequence
        "[SEP]",                  # End of sequence
        "[MASK]",                 # For masked language modeling
        "[UNK]"                   # Unknown token
    ],
    continuing_subword_prefix=""  # No prefix needed for SMILES
)

# Load and prepare training data
smiles_data = read_smiles_from_file('CombinedData_K_CH_DE_unique.csv')

# Train the tokenizer
tokenizer.train_from_iterator(smiles_data, trainer)

# Save the base tokenizer
tokenizer.save("smiles_wordpiece_u.json")

# Convert to PreTrainedTokenizerFast for easier use with transformers
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    padding_side="right"
)

# Save the fast tokenizer
fast_tokenizer.save_pretrained("smiles_tokenizer_u")
vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # Sort by token ID
with open("word_vocab_u.txt", "w", encoding="utf-8") as f:
    for token, id in sorted_vocab:
        f.write(f"{token}\t{id}\n")

# # Test the tokenizer
# test_smiles = [
#     "CCO",                    # Ethanol
#     "CC(=O)OH",              # Acetic acid
#     "c1ccccc1",              # Benzene
#     "CC(=O)CCc1ccccc1",      # Phenylacetone
#     "N[C@@H](C)C(=O)O"       # Alanine
# ]

# print("\nTokenization Examples:")
# for smiles in test_smiles:
#     # Encode
#     encoded = fast_tokenizer.encode(smiles, add_special_tokens=True)
#     tokens = fast_tokenizer.convert_ids_to_tokens(encoded)
    
#     # Decode
#     decoded = fast_tokenizer.decode(encoded, skip_special_tokens=True)
    
#     print(f"\nOriginal SMILES: {smiles}")
#     print(f"Tokens: {tokens}")
#     print(f"Decoded SMILES: {decoded.replace(' ', '')}")
#     print(f"Token IDs: {encoded}")

# Example of how to use for batch processing
def process_smiles_batch(smiles_list, max_length=512):
    """Process a batch of SMILES strings."""
    return fast_tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# # Example of batch processing
# #batch_encoding = process_smiles_batch(test_smiles)
# print("\nBatch Processing Example:")
# print(f"Input shape: {batch_encoding['input_ids'].shape}")
# print(f"Attention mask shape: {batch_encoding['attention_mask'].shape}")