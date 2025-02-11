from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import PreTrainedTokenizerFast
from Data_Process_with_prevocab_gen import read_smiles_from_file
import matplotlib.pyplot as plt
import os

def analyze_vocab_stats(smiles_data):
    """
    Analyze vocabulary statistics for different vocab sizes
    Returns: Dictionary with stats for each vocab size
    """
    # Try different vocab sizes
    vocab_sizes = [1000, 2000, 3000, 4000, 5000]
    stats = {}
    
    for vocab_size in vocab_sizes:
        print(f"\nAnalyzing vocab size: {vocab_size}")
        
        # Initialize WordPiece tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        
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
        
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
            continuing_subword_prefix=""
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(smiles_data, trainer)
        vocab = tokenizer.get_vocab()
        
        # Count unknown tokens
        unk_count = 0
        total_tokens = 0
        for smiles in smiles_data:
            encoding = tokenizer.encode(smiles)
            unk_count += encoding.ids.count(vocab["[UNK]"])
            total_tokens += len(encoding.ids)
        
        coverage = 1 - (unk_count / total_tokens)
        stats[vocab_size] = {
            "vocab_size": len(vocab),
            "unk_ratio": unk_count / total_tokens,
            "coverage": coverage
        }
        
        print(f"Coverage: {coverage:.3f}")
        print(f"Unknown token ratio: {unk_count/total_tokens:.3f}")
    
    # Plot results
    plot_vocab_stats(stats)
    
    return stats

def plot_vocab_stats(stats):
    """Plot vocabulary statistics"""
    sizes = list(stats.keys())
    coverage = [stat["coverage"] for stat in stats.values()]
    unk_ratio = [stat["unk_ratio"] for stat in stats.values()]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, coverage, 'b-', label='Coverage')
    plt.plot(sizes, unk_ratio, 'r-', label='Unknown Token Ratio')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Ratio')
    plt.title('Vocabulary Size vs Coverage/Unknown Tokens')
    plt.legend()
    plt.grid(True)
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plt.savefig(os.path.join(plots_dir, 'vocab_stats.png'))
    plt.close()

def optimize_vocab_size(smiles_data):
    """
    Find optimal vocabulary size based on coverage and dataset statistics
    Returns: Optimal vocabulary size
    """
    # Get dataset statistics
    total_smiles = len(smiles_data)
    avg_length = sum(len(s) for s in smiles_data) / total_smiles
    unique_chars = len(set(''.join(smiles_data)))
    
    print(f"\nDataset Statistics:")
    print(f"Total SMILES: {total_smiles}")
    print(f"Average length: {avg_length:.1f}")
    print(f"Unique characters: {unique_chars}")
    
    # Analyze different vocab sizes
    stats = analyze_vocab_stats(smiles_data)
    
    # Find optimal size (>95% coverage with smallest vocab)
    optimal_size = min(
        (size for size, stat in stats.items() if stat["coverage"] > 0.95),
        default=5000  # Default if no size achieves 95% coverage
    )
    
    print(f"\nOptimal vocabulary size: {optimal_size}")
    return optimal_size

def save_tokenizer_for_size(smiles_data, vocab_size, output_dir):
    """
    Save tokenizer for a specific vocabulary size
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    
    # Define chemical patterns
    chemical_patterns = (
        r"(\[|\]|Br|Cl|OH|NH|[CNOPSFIHBr]|[0-9]|=|#|-|\(|\)|\+|\\|\/|@|\.|:|c1|n1|o1|s1)"
    )
    
    # Configure tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=chemical_patterns, behavior="isolated"),
        pre_tokenizers.Whitespace()
    ])
    
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
        continuing_subword_prefix=""
    )
    
    # Train tokenizer
    tokenizer.train_from_iterator(smiles_data, trainer)
    
    # Save the raw tokenizer
    tokenizer.save(os.path.join(output_dir, f"smiles_wordpiece_{vocab_size}.json"))
    
    # Save vocabulary to text file
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    with open(os.path.join(output_dir, f"vocab_{vocab_size}.txt"), "w") as f:
        for token, id in vocab_sorted:
            f.write(f"{token}\t{id}\n")
    
    # Save as PreTrainedTokenizerFast
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    pretrained_tokenizer.save_pretrained(os.path.join(output_dir, f"pretrained_{vocab_size}"))
    
    print(f"\nSaved tokenizer with vocab size {vocab_size} in {output_dir}:")
    print(f"- Raw tokenizer: smiles_wordpiece_{vocab_size}.json")
    print(f"- Vocabulary: vocab_{vocab_size}.txt")
    print(f"- PreTrainedTokenizer: pretrained_{vocab_size}/")
    
    return tokenizer, pretrained_tokenizer

if __name__ == "__main__":
    # Load SMILES data
    smiles_data = read_smiles_from_file('CombinedData_K_CH_DE_unique.csv')
    
    # Find optimal vocabulary size
    optimal_vocab_size = optimize_vocab_size(smiles_data)
    
    # Base output directory
    base_output_dir = "tokenizers"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Save tokenizers for different vocab sizes
    vocab_sizes = [2000, 3000, optimal_vocab_size]
    
    for vocab_size in vocab_sizes:
        output_dir = os.path.join(base_output_dir, f"vocab_{vocab_size}")
        save_tokenizer_for_size(smiles_data, vocab_size, output_dir) 