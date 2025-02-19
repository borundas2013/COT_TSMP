from Constants import *
from Data_Process_with_prevocab import *
from rdkit import Chem
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
from datetime import datetime
import json


from rdkit import Chem
from rdkit.Chem import Draw
import os
from datetime import datetime

def save_smiles_pair_as_image(pair_info, save_dir="generated_molecules_images_New"):
    """
    Save a pair of SMILES strings as a side-by-side image with their information.
    
    Args:
        pair_info (dict): Dictionary containing SMILES pair information
        save_dir (str): Directory to save the images
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create molecules from SMILES
        mol1 = Chem.MolFromSmiles(pair_info["monomer1"]["smiles"])
        mol2 = Chem.MolFromSmiles(pair_info["monomer2"]["smiles"])
        
        if mol1 is None or mol2 is None:
            print(f"Error: Invalid SMILES string(s)")
            return
        
        # Generate 2D depictions
        img1 = Draw.MolToImage(mol1)
        img2 = Draw.MolToImage(mol2)
        
        # Create a side-by-side image
        combined_img = Draw.MolsToGridImage(
            [mol1, mol2],
            legends=[
                f"Monomer 1\nPresent Groups: {', '.join(pair_info['monomer1']['present_groups'])}\nMissing Groups: {', '.join(pair_info['monomer1']['missing_groups'])}",
                f"Monomer 2\nPresent Groups: {', '.join(pair_info['monomer2']['present_groups'])}\nMissing Groups: {', '.join(pair_info['monomer2']['missing_groups'])}"
            ],
            subImgSize=(400, 400),
            returnPNG=False
        )
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pair_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Save the image
        combined_img.save(filepath)
        
        # Save additional information in a text file
        info_filename = f"pair_{timestamp}_info.txt"
        info_filepath = os.path.join(save_dir, info_filename)
        
        with open(info_filepath, 'w') as f:
            f.write(f"Generated on: {timestamp}\n")
            f.write(f"Input SMILES: {pair_info['input_smiles']}\n")
            f.write(f"Temperature: {pair_info['temperature']}\n")
            f.write(f"Desired Groups: {', '.join(pair_info['desired_groups'])}\n\n")
            f.write("Monomer 1:\n")
            f.write(f"SMILES: {pair_info['monomer1']['smiles']}\n")
            f.write(f"Present Groups: {', '.join(pair_info['monomer1']['present_groups'])}\n")
            f.write(f"Missing Groups: {', '.join(pair_info['monomer1']['missing_groups'])}\n\n")
            f.write("Monomer 2:\n")
            f.write(f"SMILES: {pair_info['monomer2']['smiles']}\n")
            f.write(f"Present Groups: {', '.join(pair_info['monomer2']['present_groups'])}\n")
            f.write(f"Missing Groups: {', '.join(pair_info['monomer2']['missing_groups'])}\n")
        
        print(f"Saved pair image and info to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None

def generate_monomer_pair_with_temperature(model, input_smiles, desired_groups, vocab, max_length, temperature=1.0, num_samples=5):
    # Prepare input SMILES
    tokens,tokens2  = tokenize_smiles([input_smiles[0]]),tokenize_smiles([input_smiles[1]])
    padded_tokens = pad_token(tokens, max_length, vocab)
    padded_tokens2 = pad_token(tokens2, max_length, vocab)
    input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
    input_seq2 = np.array(padded_tokens2)  # Shape: (1, max_length)
    
    # Prepare group features
    group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
    group_features = np.array([group_features])
    
    def sample_with_temperature(predictions, temperature):
        """Sample from predictions with temperature scaling"""
        if temperature == 0:
            return np.argmax(predictions)
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)
    
    def generate_monomer(decoder_index):
        """Generate single monomer sequence"""
        # Initialize decoder sequence with proper shape
        decoder_seq = np.zeros((1, max_length))
        decoder_seq[0, 0] = vocab['<start>']
        decoder_seq2 = np.zeros((1, max_length))
        decoder_seq2[0, 0] = vocab['<start>']
        

        generated_tokens = []
        for i in range(max_length-1):
            # Ensure all inputs have correct shapes
            # output = model.predict([
            #     input_seq,  # Shape: (1, max_length)
            #     input_seq2,  # Shape: (1, max_length)
            #     group_features,  # Shape: (1, num_groups)
            #     decoder_seq,  # Shape: (1, max_length)
            #     decoder_seq2,  # Shape: (1, max_length)
            # ], verbose=0)
            

            output = model.predict({
            'monomer1_input': input_seq,
            'monomer2_input': input_seq2,
            'group_input': group_features,
            'decoder_input1': decoder_seq,
            'decoder_input2': decoder_seq2
            },verbose=0)
            
            next_token_probs = output[decoder_index][0, i]
            next_token = sample_with_temperature(next_token_probs, temperature)
            
            generated_tokens.append(next_token)
            if next_token == vocab['<end>']:
                break
            if i < max_length - 2:
                decoder_seq[0, i + 1] = next_token
        
        return generated_tokens
    
    def check_groups(smiles, desired_groups):
        """Check presence of desired groups"""
        if smiles == "":
            return [], desired_groups   
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], desired_groups
    
        present_groups = []
        not_present_groups = []
        for group in desired_groups:
            pattern = Constants.group_patterns.get(group)
            if pattern and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                present_groups.append(group)
            else:
                not_present_groups.append(group)
        return present_groups, not_present_groups
    
    # Generate multiple pairs
    generated_pairs = []
    valid_pairs = []
    
    print(f"\nGenerating {num_samples} monomer pairs with temperature {temperature}:")
    print("=" * 80)
    print(f"Input SMILES: {input_smiles}")
    print(f"Desired Groups: {', '.join(desired_groups)}")
    print("-" * 80)
    generated_output_file = "generated_pairs_new.json"
    valid_output_file = "valid_pairs_new.json"
    all_pairs = []
    valid_pairs_j = []
    try:
        with open(generated_output_file, 'r') as f:
            all_pairs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_pairs = []

    try:
        with open(valid_output_file, 'r') as f:
            valid_pairs_j = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        valid_pairs_j = []
    
    for i in range(num_samples):
        # Generate monomers using each decoder
        print(generate_monomer)
        tokens1 = generate_monomer(0)  # decoder1
        tokens2 = generate_monomer(1)  # decoder2
        
        # Convert to SMILES
        smiles1 = decode_smiles(tokens1)
        smiles2 = decode_smiles(tokens2)
        
        # Check validity
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        valid1 = mol1 is not None and smiles1 != ""
        valid2 = mol2 is not None and smiles2 != ""
        
        print(f"\nPair {i+1}:")
        print(f"Monomer 1: {smiles1}")
        print(f"Valid: {valid1}")
        present1 = []
        missing1 = []
        present2 = []
        missing2 = []
        if valid1:
            present1, missing1 = check_groups(smiles1, desired_groups)
            print(f"Present groups: {present1}")
            print(f"Missing groups: {missing1}")
        
        print(f"Monomer 2: {smiles2}")
        print(f"Valid: {valid2}")
        if valid2:
            present2, missing2 = check_groups(smiles2, desired_groups)
            print(f"Present groups: {present2}")
            print(f"Missing groups: {missing2}")
        
        generated_pairs.append((smiles1, smiles2))
        if valid1 and valid2:
            # Create dictionary with pair info
            pair_info = {
                "input_smiles": input_smiles,
                "temperature": temperature,
                "desired_groups": desired_groups,
                "monomer1": {
                    "smiles": smiles1,
                    "present_groups": present1,
                    "missing_groups": missing1
                },
                "monomer2": {
                    "smiles": smiles2, 
                    "present_groups": present2,
                    "missing_groups": missing2
                }
            }
                        
            valid_pairs.append((smiles1, smiles2))
            valid_pairs_j.append(pair_info)
            save_smiles_pair_as_image(pair_info)

            

            
         
            with open(valid_output_file, 'w') as f:
                json.dump(valid_pairs_j, f, indent=4)
        else:
            # Create dictionary with pair info
            pair_info = {
                "input_smiles": input_smiles,
                "temperature": temperature,
                "desired_groups": desired_groups,
                "monomer1": {
                    "smiles": smiles1,
                    "present_groups": present1,
                    "missing_groups": missing1
                },
                "monomer2": {
                    "smiles": smiles2, 
                    "present_groups": present2,
                    "missing_groups": missing2
                }
            }
            all_pairs.append(pair_info)
            
            with open(generated_output_file, 'w') as f:
                json.dump(all_pairs, f, indent=4)
    
    
    return generated_pairs, valid_pairs

