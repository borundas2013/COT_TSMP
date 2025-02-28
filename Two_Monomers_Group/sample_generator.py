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


def generate_monomer_pair_with_temperature(model, input_smiles, desired_groups, vocab, max_length, temperatures=[0.2,0.4,0.6,0.8,1.0,1.2], group_smarts1=None, group_smarts2=None):
    # Prepare input SMILES
    tokens,tokens2  = tokenize_smiles([input_smiles[0]]),tokenize_smiles([input_smiles[1]])
    padded_tokens = pad_token(tokens, max_length, vocab)
    padded_tokens2 = pad_token(tokens2, max_length, vocab)
    input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
    input_seq2 = np.array(padded_tokens2)  # Shape: (1, max_length)
    
    # Prepare group features
    group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
    group_features = np.array([group_features])
    
   
    
    def generate_monomer(decoder_index):
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
            print(f"Desired Groups: {', '.join(desired_groups)}")
            print("-" * 80)
            generated_tokens = []
            generated_tokens2 = []
            #it should be 245
            for i in range(10):  
                output = model.predict({
                    'monomer1_input': input_seq,
                    'monomer2_input': input_seq2,
                    'group_input': group_features,
                    'decoder_input1': decoder_seq,
                    'decoder_input2': decoder_seq2
                    },verbose=0)
            
                next_token_probs = output[0][0, i]
                next_token = sample_with_temperature(next_token_probs, temperature)

                next_token_probs2 = output[1][0, i]
                next_token2 = sample_with_temperature(next_token_probs2, temperature)
            
                generated_tokens.append(next_token)
                generated_tokens2.append(next_token2)
                if next_token == vocab['<end>']:
                    if next_token2 == vocab['<end>']:
                        break
                if next_token2 == vocab['<end>']:
                    if next_token == vocab['<end>']:
                        break

                if i < max_length - 2:
                    decoder_seq[0, i + 1] = next_token
                    decoder_seq2[0, i + 1] = next_token
            all_tokens.append([generated_tokens,generated_tokens2,temperature])
        return all_tokens
    
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
            pattern =group #Constants.group_patterns.get(group)
            if pattern and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                present_groups.append(group)
            else:
                not_present_groups.append(group)
        return present_groups, not_present_groups
    
    # Generate multiple pairs
    generated_pairs = []
    valid_pairs = []
    
    
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
    
   
    tokens1 = generate_monomer(0)  # decoder1
    #tokens2 = generate_monomer(1)  # decoder2
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
                "Input Group SMARTS 1": group_smarts1,
                "Input Group SMARTS 2": group_smarts2,
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

