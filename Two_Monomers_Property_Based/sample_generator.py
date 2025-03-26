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

def add_swap_noise(tokens, swap_rate=0.1):
    """Randomly swap adjacent tokens"""
    noisy_tokens = tokens.copy()
    for i in range(len(tokens)):
        for j in range(1, len(tokens[i])-1):  # Avoid swapping start/end tokens
            # if np.random.random() < swap_rate:
            #     noisy_tokens[i][j], noisy_tokens[i][j+1] = \
            #     noisy_tokens[i][j+1], noisy_tokens[i][j]
            noisy_tokens[i][j], noisy_tokens[i][j+1] = noisy_tokens[i][j+1], noisy_tokens[i][j]
    return noisy_tokens

def add_gaussian_noise(tokens, noise_level=1):
    """Add Gaussian noise to token embeddings"""
    noise = np.random.normal(0, noise_level, tokens.shape)
    noisy_tokens = tokens + noise
    return noisy_tokens

def generate_monomer_pair_with_temperature(model, input_smiles, desired_groups, vocab, 
                                           max_length, temperatures=[0.2,0.4,0.6,0.8,1.0,1.2], group_smarts1=None, 
                                           group_smarts2=None,er=None,tg=None, add_noise=False):
    # Prepare input SMILES
    tokens,tokens2  = tokenize_smiles([input_smiles[0]]),tokenize_smiles([input_smiles[1]])
    padded_tokens = pad_token(tokens, max_length, vocab)
    padded_tokens2 = pad_token(tokens2, max_length, vocab)
    input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
    input_seq2 = np.array(padded_tokens2)  # Shape: (1, max_length)
    er_list = np.array([er])
    tg_list = np.array([tg])

    if add_noise:
        noisy_tokens1 = add_swap_noise(input_seq, 0.1)
        noisy_tokens2 = add_swap_noise(input_seq2, 0.1) 

        noisy_tokens1 = add_gaussian_noise(noisy_tokens1, 0.1)
        noisy_tokens2 = add_gaussian_noise(noisy_tokens2, 0.1)
        input_seq =noisy_tokens1
        input_seq2 =noisy_tokens2
    
    # Prepare group features
    group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
    group_features = np.array([group_features])
    
   
    
    def generate_monomer():
        """Generate single monomer sequence"""
        # Initialize decoder sequence with proper shape
        
        decoder_seq = np.zeros((1, max_length))
        decoder_seq[0, 0] = vocab['<start>']
        decoder_seq2 = np.zeros((1, max_length))
        decoder_seq2[0, 0] = vocab['<start>']
        

        all_tokens = []
        for temperature in temperatures:
            print(f"\nGenerating monomer pairs with temperature {temperature}:")
            print("=" * 80)
            print(f"Input SMILES: {input_smiles}")
            print(f"Desired Property: ER: {er}, TG: {tg}")
            print("-" * 80)
            generated_tokens = []
            generated_tokens2 = []
            last_token = None
            last_token2 = None
            #it should be 245
            for i in range(max_length):  
                output = model.predict({
                    'monomer1_input': input_seq,
                    'monomer2_input': input_seq2,
                    'group_input': group_features,
                    'decoder_input1': decoder_seq,
                    'decoder_input2': decoder_seq2,
                    'er_list': er_list,
                    'tg_list': tg_list
                    },verbose=0)
            
                next_token_probs = output['monomer1_output'][0, i]
                next_token_probs2 = output['monomer2_output'][0, i]

                if last_token != vocab['<end>']:
                   next_token = sample_with_temperature(next_token_probs, temperature)
                   generated_tokens.append(next_token)
                   last_token = next_token
                if last_token2 != vocab['<end>']:
                   next_token2 = sample_with_temperature(next_token_probs2, temperature)
                   generated_tokens2.append(next_token2)
                   last_token2 = next_token2

                
                if last_token == vocab['<end>'] and last_token2 == vocab['<end>']:
                    break

                if i < max_length - 2:
                    decoder_seq[0, i + 1] = next_token
                    decoder_seq2[0, i + 1] = next_token
            all_tokens.append([generated_tokens,generated_tokens2,temperature])
        return all_tokens
    
    
    
    # Generate multiple pairs
    generated_pairs = []
    valid_pairs = []
    
    
    generated_output_file = Constants.GENERATED_ALL_PREDICT_FILE
    valid_output_file = Constants.GENERATED_VALID_PREDICTED_FILE
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
    
   
    tokens1 = generate_monomer()  # decoder1
    
    for i in range(len(tokens1)):
        # Convert to SMILES
        smiles1,smiles2 = decode_smiles(tokens1[i][0]), decode_smiles(tokens1[i][1])#decode_smiles(tokens1[i][0])
        #smiles2 = decode_smiles(tokens2[i][0])
        temperature = tokens1[i][2]
        
        # Check validity
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        valid1 = mol1 is not None and smiles1 != ""
        valid2 = mol2 is not None and smiles2 != ""
        
        print(f"\nPair {i+1}:")
        print(f"Monomer 1: {smiles1}")
        print(f"Valid: {valid1}")
    
        
        generated_pairs.append((smiles1, smiles2))
        if valid1 and valid2:
            # Create dictionary with pair info
            pair_info = {
                "input_smiles": input_smiles,
                "temperature": temperature,
                "desired_groups": desired_groups,
                "Input Group SMARTS 1": group_smarts1,
                "Input Group SMARTS 2": group_smarts2,
                "ER": er,
                "TG": tg,
                "monomer1": {
                    "smiles": smiles1,
                },
                "monomer2": {
                    "smiles": smiles2, 
                },
                "add_noise": add_noise
            }
                        
            valid_pairs.append((smiles1, smiles2))
            valid_pairs_j.append(pair_info)
            #save_smiles_pair_as_image(pair_info)

            

            
         
            with open(valid_output_file, 'w') as f:
                json.dump(valid_pairs_j, f, indent=4)
        else:
            # Create dictionary with pair info
            pair_info = {
                "input_smiles": input_smiles,
                "temperature": temperature,
                "desired_groups": desired_groups,
                "ER": er,
                "TG": tg,
                "monomer1": {
                     "smiles": smiles1,
                    
                },
                "monomer2": {
                    "smiles": smiles2, 
                },
                "add_noise": add_noise
            }
            all_pairs.append(pair_info)
            
            with open(generated_output_file, 'w') as f:
                json.dump(all_pairs, f, indent=4)
    
    
    return generated_pairs, valid_pairs

