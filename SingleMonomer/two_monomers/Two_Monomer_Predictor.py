from LoadPreTrainedModel import *
from Two_Monomer_Model import *
from Constants import *
from Data_Process_with_prevocab import *
import pandas as pd
import random



def count_functional_groups(smiles, smarts_pattern):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern)))
def generate_similar_monomer_pair(model, input_monomer1, input_monomer2, desired_combination, 
                                smiles_vocab, max_length, temperature=0.8, num_pairs=5):
    """
    Generate new monomer pairs similar to input monomers with desired functional groups
    """
    def validate_groups(smiles, required_group):
       mol = Chem.MolFromSmiles(smiles)
       
       if mol is None:
           return False, None
       
       for group in required_group:
           if mol.HasSubstructMatch(Chem.MolFromSmarts(group)):
               return True, group
       return False, None

    # Convert desired_combination from strings to encoded groups
    group1, group2 = desired_combination  # Assuming input like ["epoxy", "imine"]
    encoded_groups = np.zeros((1, len(Constants.GROUP_VOCAB)))
    
    # Process input monomers
    tokens1 = tokenize_smiles([input_monomer1])[0]
    tokens2 = tokenize_smiles([input_monomer2])[0]
    input1_padded = pad_token([tokens1], max_length, smiles_vocab)[0]
    input2_padded = pad_token([tokens2], max_length, smiles_vocab)[0]
    
    valid_pairs = []
    attempts = 0
    max_attempts = num_pairs * 20
    
    while len(valid_pairs) < num_pairs and attempts < max_attempts:
        # Generate first monomer using input_monomer1 as reference
        target_seq1 = np.zeros((1, max_length))
        target_seq1[0, 0] = smiles_vocab['<start>']
        generated_tokens1 = []
        
        # Generate sequence for first monomer
        for i in range(1, max_length):
            predictions = model.predict(
                {
                    'smiles_input': np.array([input1_padded]),
                    'decoder_input': target_seq1,
                    'group_input': encoded_groups
                },
                verbose=0
            )
            
            probs = predictions[0, i-1]
            if temperature == 0:
                next_token = np.argmax(probs)
            else:
                probs = np.log(probs) / temperature
                exp_probs = np.exp(probs)
                probs = exp_probs / np.sum(exp_probs)
                next_token = np.random.choice(len(probs), p=probs)
            
            generated_tokens1.append(next_token)
            target_seq1[0, i] = next_token
        
        monomer1 = decode_smiles(generated_tokens1)
        
        # Validate first monomer
        valid1, group_id1 = validate_groups(monomer1, group1)
        if not valid1:
            attempts += 1
            continue
            
        # Generate second monomer using input_monomer2 and generated first monomer
        tokens_new1 = tokenize_smiles([monomer1])[0]
        encoder_input = pad_token([tokens_new1], max_length, smiles_vocab)[0]
        group_features = encode_groups([group_id1], Constants.GROUP_VOCAB)
        
        target_seq2 = np.zeros((1, max_length))
        target_seq2[0, 0] = smiles_vocab['<start>']
        generated_tokens2 = []
        
        # Generate sequence for second monomer
        for i in range(1, max_length):
            predictions = model.predict(
                {
                    'smiles_input': np.array([input2_padded]),  # Use input_monomer2 as reference
                    'decoder_input': target_seq2,
                    'group_input': group_features
                },
                verbose=0
            )
            
            probs = predictions[0, i-1]
            if temperature == 0:
                next_token = np.argmax(probs)
            else:
                probs = np.log(probs) / temperature
                exp_probs = np.exp(probs)
                probs = exp_probs / np.sum(exp_probs)
                next_token = np.random.choice(len(probs), p=probs)
            
            generated_tokens2.append(next_token)
            target_seq2[0, i] = next_token
            
            if next_token == smiles_vocab['<end>']:
                break
        
        monomer2 = decode_smiles(generated_tokens2)
        
        # Validate second monomer
        valid2, group_id2 = validate_groups(monomer2, group2)
        if valid2:
            valid_pairs.append({
                'input_monomer1': input_monomer1,
                'input_monomer2': input_monomer2,
                'generated_monomer1': monomer1,
                'generated_monomer2': monomer2,
                'groups1': [group_id1],
                'groups2': [group_id2]
            })
            print(f"Generated valid pair {len(valid_pairs)}/{num_pairs}")
        
        attempts += 1
    
    if len(valid_pairs) < num_pairs:
        print(f"Warning: Only generated {len(valid_pairs)} valid pairs out of {num_pairs} requested")
    
    return valid_pairs

def load_monomer_pairs(file_path):
    """
    Load and validate monomer pairs from Excel file
    Returns only valid pairs where:
    1. SMILES string splits into exactly 2 monomers
    2. Both monomers are valid SMILES strings
    """
    df = pd.read_excel(file_path)
    monomer_pairs = df['SMILES'].str.split(',', expand=True)
    
    valid_pairs = []
    total_pairs = len(monomer_pairs)
    invalid_count = 0
    wrong_format_count = 0
    
    for idx, row in monomer_pairs.iterrows():
        # Check if we have exactly 2 monomers
        row_values = [val for val in row if isinstance(val, str)]
        if len(row_values) != 2:
            wrong_format_count += 1
            print(f"Wrong number of monomers at row {idx+1}: {df['SMILES'][idx]}")
            continue
            
        m1, m2 = row_values[0].strip(), row_values[1].strip()
        
        # Check if both monomers are valid SMILES
        mol1 = Chem.MolFromSmiles(m1)
        mol2 = Chem.MolFromSmiles(m2)
        
        if mol1 is not None and mol2 is not None:
            valid_pairs.append((m1, m2))
        else:
            invalid_count += 1
            if mol1 is None:
                print(f"Invalid first monomer SMILES at row {idx+1}: {m1}")
            if mol2 is None:
                print(f"Invalid second monomer SMILES at row {idx+1}: {m2}")
    
    print(f"\nMonomer Pairs Summary:")
    print(f"Total entries: {total_pairs}")
    print(f"Wrong format (not 2 monomers): {wrong_format_count}")
    print(f"Invalid SMILES: {invalid_count}")
    print(f"Valid pairs: {len(valid_pairs)}")
    
    if not valid_pairs:
        raise ValueError("No valid monomer pairs found in the input file!")
        
    return valid_pairs

def get_random_group_combination():
    """Get a random valid group combination"""
    valid_combinations = [
        ['C1OC1', 'NC'],     # epoxy-imine
        ['C=C', 'CCS'],      # vinyl-thiol
        ['C=C', 'C=C'],      # vinyl-vinyl
        ['C=C', 'C=C(C=O)']  # vinyl-acryl
    ]
    return random.choice(valid_combinations)

def generate_multiple_pairs(model, smiles_vocab, model_params, input_file, 
                          num_input_pairs=5, samples_per_pair=3):

    # Load all monomer pairs
    all_pairs = load_monomer_pairs('Small_Data/smiles.xlsx')
    
    # Randomly select input pairs
    selected_pairs = random.sample(all_pairs, min(num_input_pairs, len(all_pairs)))
    print(selected_pairs)
    
    all_generated_pairs = []
    
    for input_m1, input_m2 in selected_pairs:
        # Get random desired group combination
        desired_group = get_random_group_combination()
        
        print(f"\nGenerating from input pair:")
        print(f"Monomer 1: {input_m1}")
        print(f"Monomer 2: {input_m2}")
        print(f"Desired groups: {desired_group}")
        
        # Generate new pairs
        pairs = generate_similar_monomer_pair(
            model=model,
            input_monomer1=input_m1,
            input_monomer2=input_m2,
            desired_combination=desired_group,
            smiles_vocab=smiles_vocab,
            max_length=model_params['max_length'],
            temperature=0.8,
            num_pairs=samples_per_pair
        )
        
        all_generated_pairs.extend(pairs)
        
    return all_generated_pairs

# # Example usage
# if __name__ == "__main__":
#     # Load model and vocabulary
#     loaded_model, loaded_vocab, loaded_params = load_dual_monomer_model(
#         save_dir="saved_dual_monomer_model"
#     )
    
#     # Generate multiple pairs
#     generated_pairs = generate_multiple_pairs(
#         model=loaded_model,
#         smiles_vocab=loaded_vocab,
#         model_params=loaded_params,
#         input_file="path/to/your/input_file.xlsx",
#         num_input_pairs=5,    # Number of input pairs to randomly select
#         samples_per_pair=3    # Number of new samples per input pair
#     )
    
#     # Print results
#     print("\nAll Generated Pairs:")
#     for i, pair in enumerate(generated_pairs, 1):
#         print(f"\nPair {i}:")
#         print(f"Input Monomer 1: {pair['input_monomer1']}")
#         print(f"Generated Monomer 1 (with {pair['groups1']}): {pair['generated_monomer1']}")
#         print(f"Input Monomer 2: {pair['input_monomer2']}")
#         print(f"Generated Monomer 2 (with {pair['groups2']}): {pair['generated_monomer2']}")
    
#     # Optionally save results to file
#     results_df = pd.DataFrame({
#         'Input_Monomer1': [p['input_monomer1'] for p in generated_pairs],
#         'Generated_Monomer1': [p['generated_monomer1'] for p in generated_pairs],
#         'Groups1': [p['groups1'] for p in generated_pairs],
#         'Input_Monomer2': [p['input_monomer2'] for p in generated_pairs],
#         'Generated_Monomer2': [p['generated_monomer2'] for p in generated_pairs],
#         'Groups2': [p['groups2'] for p in generated_pairs]
#     })
    
#     results_df.to_excel('generated_monomer_pairs.xlsx', index=False)